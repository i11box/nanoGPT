"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Diffusion utilities (ported and simplified from diffusion-forcing)
# -----------------------------------------------------------------------------


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal positional embedding used for diffusion time-step encodings."""

    def __init__(self, dim, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        # x: (...,) integer or float time-steps
        device = x.device
        half_dim = self.dim // 2
        emb_factor = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)  # (D/2,)
        x = x.float().unsqueeze(-1)                                           # (..., 1)
        emb = x * emb                                                         # (..., D/2)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)                       # (..., D)
        return emb


def linear_beta_schedule(timesteps: int):
    """Linear schedule, as in the original DDPM paper."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """Cosine schedule (https://openreview.net/forum?id=-NEXDKk8gZ)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    timesteps: int,
    start: float = -3,
    end: float = 3,
    tau: float = 1,
    clamp_min: float = 1e-5,
):
    """Sigmoid schedule (https://arxiv.org/abs/2212.11972, Figure 8)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, clamp_min, 0.999)
    return betas

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # Diffusion Forcing hyperparameters (used only in DFGPT)
    df_timesteps: int = 32
    df_sampling_timesteps: int = 32
    df_snr_clip: float = 5.0
    df_cum_snr_decay: float = 0.95
    df_objective: str = "pred_noise"  # "pred_noise", "pred_x0", or "pred_v"
    df_beta_schedule: str = "cosine"  # "linear", "cosine", "sigmoid"
    df_clip_noise: float = 5.0
    df_ddim_eta: float = 0.0
    df_stabilization_level: int = 0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class DFGPT(GPT):
    """
    Diffusion Forcing variant of GPT.
    Trains in continuous embedding space with per-token diffusion and fused SNR reweighting.
    """

    def __init__(self, config):
        super().__init__(config)

        # diffusion hyperparameters
        self.df_timesteps = getattr(config, "df_timesteps", 32)
        self.df_sampling_timesteps = getattr(config, "df_sampling_timesteps", 32)
        self.df_snr_clip = getattr(config, "df_snr_clip", 5.0)
        self.df_cum_snr_decay = getattr(config, "df_cum_snr_decay", 0.95)
        self.df_objective = getattr(config, "df_objective", "pred_noise")
        self.df_beta_schedule = getattr(config, "df_beta_schedule", "cosine")
        self.df_clip_noise = getattr(config, "df_clip_noise", 5.0)
        self.df_ddim_eta = getattr(config, "df_ddim_eta", 0.0)
        self.df_stabilization_level = getattr(config, "df_stabilization_level", 0)

        assert self.df_sampling_timesteps <= self.df_timesteps
        self.df_is_ddim_sampling = self.df_sampling_timesteps < self.df_timesteps

        # time-step embedding and prediction head (back to embedding space)
        self.df_t_embed = SinusoidalPosEmb(self.config.n_embd)
        self.df_head = nn.Linear(self.config.n_embd, self.config.n_embd, bias=True)

        # precompute diffusion buffers (betas, alphas, SNR, etc.)
        self._build_diffusion_buffers()

    # --- diffusion utilities -------------------------------------------------

    def _build_diffusion_buffers(self):
        T = self.df_timesteps
        # beta schedule (matches diffusion-forcing choices)
        if self.df_beta_schedule == "linear":
            betas = linear_beta_schedule(T)
        elif self.df_beta_schedule == "cosine":
            betas = cosine_beta_schedule(T)
        elif self.df_beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(T)
        else:
            raise ValueError(f"unknown df_beta_schedule {self.df_beta_schedule}")

        betas = betas.to(torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("df_betas", betas)
        self.register_buffer("df_alphas_cumprod", alphas_cumprod)
        self.register_buffer("df_alphas_cumprod_prev", alphas_cumprod_prev)

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        self.register_buffer("df_posterior_variance", posterior_variance)
        self.register_buffer("df_posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("df_posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("df_posterior_mean_coef2", posterior_mean_coef2)

        # SNR and clipped SNR (for reweighting)
        snr = alphas_cumprod / (1.0 - alphas_cumprod)
        clipped_snr = snr.clone()
        clipped_snr.clamp_(max=self.df_snr_clip)

        self.register_buffer("df_snr", snr)
        self.register_buffer("df_clipped_snr", clipped_snr)

    def _extract(self, buf, t, x_shape):
        # buf: (T,), t: (B, T_seq), return shape broadcastable to x_shape
        out = buf[t]  # (B, T_seq)
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def _q_sample(self, x0, t, noise):
        # x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise
        sqrt_ac = torch.sqrt(self._extract(self.df_alphas_cumprod, t, x0.shape))
        sqrt_om = torch.sqrt(1.0 - self._extract(self.df_alphas_cumprod, t, x0.shape))
        return sqrt_ac * x0 + sqrt_om * noise

    def _predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(torch.sqrt(1.0 / self.df_alphas_cumprod), t, x_t.shape) * x_t
            - self._extract(torch.sqrt(1.0 / self.df_alphas_cumprod - 1.0), t, x_t.shape) * noise
        )

    def _predict_noise_from_start(self, x_t, t, x0):
        return (
            self._extract(torch.sqrt(1.0 / self.df_alphas_cumprod), t, x_t.shape) * x_t - x0
        ) / self._extract(torch.sqrt(1.0 / self.df_alphas_cumprod - 1.0), t, x_t.shape)

    def _predict_v(self, x_start, t, noise):
        sqrt_ac = self._extract(torch.sqrt(self.df_alphas_cumprod), t, x_start.shape)
        sqrt_om = self._extract(torch.sqrt(1.0 - self.df_alphas_cumprod), t, x_start.shape)
        return sqrt_ac * noise - sqrt_om * x_start

    def _predict_start_from_v(self, x_t, t, v):
        sqrt_ac = self._extract(torch.sqrt(self.df_alphas_cumprod), t, x_t.shape)
        sqrt_om = self._extract(torch.sqrt(1.0 - self.df_alphas_cumprod), t, x_t.shape)
        return sqrt_ac * x_t - sqrt_om * v

    def _compute_loss_weight(self, noise_levels):
        """
        Fused SNR loss reweighting, adapted from Diffusion Forcing implementation.
        noise_levels: (B, T_seq)
        """
        snr = self.df_snr[noise_levels]              # (B, T)
        clipped_snr = self.df_clipped_snr[noise_levels]
        normalized_clipped_snr = clipped_snr / self.df_snr_clip
        normalized_snr = snr / self.df_snr_clip

        # if not fused, fall back to min-SNR weighting
        if self.df_cum_snr_decay <= 0.0:
            if self.df_objective == "pred_noise":
                return clipped_snr / snr
            elif self.df_objective == "pred_x0":
                return clipped_snr
            elif self.df_objective == "pred_v":
                return clipped_snr / (snr + 1.0)

        # compute fused SNR along time dimension (tokens)
        # work in (T, B) layout to mirror original implementation
        normalized_clipped_tb = normalized_clipped_snr.transpose(0, 1)  # (T, B)
        normalized_tb = normalized_snr.transpose(0, 1)                  # (T, B)
        T, B = normalized_tb.shape

        cum_snr_tb = torch.zeros_like(normalized_tb)
        for t in range(T):
            if t == 0:
                cum_snr_tb[t] = normalized_clipped_tb[t]
            else:
                cum_snr_tb[t] = (
                    self.df_cum_snr_decay * cum_snr_tb[t - 1]
                    + (1.0 - self.df_cum_snr_decay) * normalized_clipped_tb[t]
                )

        cum_snr_tb = F.pad(cum_snr_tb[:-1], (0, 0, 1, 0), value=0.0)
        clipped_fused_snr_tb = 1.0 - (1.0 - cum_snr_tb * self.df_cum_snr_decay) * (
            1.0 - normalized_clipped_tb
        )
        fused_snr_tb = 1.0 - (1.0 - cum_snr_tb * self.df_cum_snr_decay) * (1.0 - normalized_tb)

        if self.df_objective == "pred_noise":
            weight_tb = clipped_fused_snr_tb / fused_snr_tb
        elif self.df_objective == "pred_x0":
            weight_tb = clipped_fused_snr_tb * self.df_snr_clip
        elif self.df_objective == "pred_v":
            weight_tb = clipped_fused_snr_tb * self.df_snr_clip / (fused_snr_tb * self.df_snr_clip + 1.0)
        else:
            raise ValueError(f"unknown df_objective {self.df_objective}")

        return weight_tb.transpose(0, 1)  # back to (B, T)

    def _add_shape_channels_seq(self, x_scalar, x_like):
        """
        Helper to broadcast scalar/tensor of shape (B, T) to match x_like shape (B, T, C)
        by adding trailing singleton dimensions.
        """
        while x_scalar.dim() < x_like.dim():
            x_scalar = x_scalar.unsqueeze(-1)
        return x_scalar

    # --- sampling (DDPM / DDIM) ----------------------------------------------

    def _df_model_predictions_seq(self, x_seq, noise_levels_seq):
        """
        Wrapper for model predictions on sequences in (T, B, C) layout.
        x_seq: (T, B, C), noise_levels_seq: (T, B)
        """
        x = x_seq.transpose(0, 1)                  # (B, T, C)
        k = noise_levels_seq.transpose(0, 1)       # (B, T)
        pred_noise, x_start, model_out = self._df_model_predictions(x, k)
        return (
            pred_noise.transpose(0, 1),
            x_start.transpose(0, 1),
            model_out.transpose(0, 1),
        )

    def _ddpm_sample_step_seq(self, x, curr_noise_level):
        """
        DDPM sampling step on sequence x with per-token noise level.
        x: (T, B, C)
        curr_noise_level: (T, B) in real steps (-1 .. timesteps-1)
        """
        clipped_curr = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.df_stabilization_level - 1, dtype=torch.long),
            curr_noise_level,
        )

        # ==========================================
        # 【关键修复】创建 >=0 的安全索引，防止 CUDA 崩溃
        # ==========================================
        # 如果索引是 -1，我们在查表时强行视为 0。
        # 虽然查出来的值是错的，但稍后会被 torch.where 丢弃，所以是安全的。
        safe_curr = torch.clamp(clipped_curr, min=0)

        orig_x = x.clone().detach()
        
        # 使用 safe_curr 而不是 clipped_curr 去查表
        scaled_context = self._q_sample(
            x.transpose(0, 1),                     # (B, T, C)
            safe_curr.transpose(0, 1),             # (B, T) <--- 使用安全索引
            noise=torch.randn_like(x.transpose(0, 1)),
        ).transpose(0, 1)
        
        x = torch.where(
            self._add_shape_channels_seq(curr_noise_level < 0, x),
            scaled_context,
            orig_x,
        )

        # 使用 safe_curr 预测
        pred_noise, x_start, _ = self._df_model_predictions_seq(x, safe_curr) # <--- 使用安全索引

        x_start_bt = x_start.transpose(0, 1)
        x_bt = x.transpose(0, 1)
        t_bt = safe_curr.transpose(0, 1) # <--- 使用安全索引

        posterior_mean = (
            self._extract(self.df_posterior_mean_coef1, t_bt, x_start_bt.shape) * x_start_bt
            + self._extract(self.df_posterior_mean_coef2, t_bt, x_bt.shape) * x_bt
        ).transpose(0, 1)
        
        model_log_variance = self._extract(
            self.df_posterior_log_variance_clipped, t_bt, x_bt.shape
        ).transpose(0, 1)

        noise = torch.where(
            self._add_shape_channels_seq(clipped_curr > 0, x),
            torch.randn_like(x),
            torch.zeros_like(x),
        )
        noise = torch.clamp(noise, -self.df_clip_noise, self.df_clip_noise)
        x_pred = posterior_mean + torch.exp(0.5 * model_log_variance) * noise

        # 最后把那些原本是 -1 的位置还原回 orig_x (Clean)
        return torch.where(
            self._add_shape_channels_seq(curr_noise_level == -1, x),
            orig_x,
            x_pred,
        )

    def _ddim_sample_step_seq(self, x, curr_noise_level, next_noise_level):
        """
        DDIM sampling step on sequence x with per-token noise level.
        """
        clipped_curr = torch.where(
            curr_noise_level < 0,
            torch.full_like(curr_noise_level, self.df_stabilization_level - 1, dtype=torch.long),
            curr_noise_level,
        )

        # ==========================================
        # 【关键修复】安全索引
        # ==========================================
        safe_curr = torch.clamp(clipped_curr, min=0)
        safe_next = torch.clamp(next_noise_level, min=0) # next 也要防 -1

        orig_x = x.clone().detach()
        
        # 使用 safe_curr 查表
        x_bt = x.transpose(0, 1)
        scaled_context_bt = self._q_sample(
            x_bt,
            safe_curr.transpose(0, 1),
            noise=torch.zeros_like(x_bt),
        )
        scaled_context = scaled_context_bt.transpose(0, 1)
        x = torch.where(
            self._add_shape_channels_seq(curr_noise_level < 0, x),
            scaled_context,
            orig_x,
        )

        # 使用 safe_curr / safe_next 查表
        alpha = self.df_alphas_cumprod[safe_curr]                # (T, B)
        
        # 处理 alpha_next：如果 next < 0，不论查表结果如何，直接设为 1.0
        alpha_next_temp = self.df_alphas_cumprod[safe_next]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level, dtype=alpha_next_temp.dtype), # 修正：<0 时 alpha=1.0
            alpha_next_temp,
        )
        
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level, dtype=alpha.dtype),
            self.df_ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha_next_shaped = self._add_shape_channels_seq(alpha_next, x)
        c_shaped = self._add_shape_channels_seq(c, x)
        sigma_shaped = self._add_shape_channels_seq(sigma, x)

        # 使用 safe_curr 预测
        pred_noise, x_start, _ = self._df_model_predictions_seq(x, safe_curr)

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.df_clip_noise, self.df_clip_noise)
        x_pred = x_start * alpha_next_shaped.sqrt() + pred_noise * c_shaped + sigma_shaped * noise

        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self._add_shape_channels_seq(mask, x),
            orig_x,
            x_pred,
        )

        return x_pred

    def _sample_step_seq(self, x, curr_noise_level, next_noise_level):
        """
        Unified sampling step (DDPM or DDIM) on sequence x.
        x: (T, B, C)
        curr_noise_level, next_noise_level: (T, B) in scheduling index space [0, df_sampling_timesteps]
        """
        # map scheduling indices (0..sampling_timesteps) to real diffusion steps (-1..timesteps-1)
        real_steps = torch.linspace(
            -1, self.df_timesteps - 1, steps=self.df_sampling_timesteps + 1, device=x.device
        ).long()
        curr_real = real_steps[curr_noise_level]
        next_real = real_steps[next_noise_level]

        if self.df_is_ddim_sampling:
            return self._ddim_sample_step_seq(x, curr_real, next_real)

        # DDPM sanity checks (parity with diffusion-forcing)            
        assert torch.all(
            (curr_real - 1 == next_real) | ((curr_real == -1) & (next_real == -1))
        ), "Wrong noise level given for ddpm sampling."
        assert (
            self.df_sampling_timesteps == self.df_timesteps
        ), "sampling_timesteps should equal timesteps for ddpm sampling."

        return self._ddpm_sample_step_seq(x, curr_real)

    # --- scheduling matrix & DF-style generation -----------------------------

    def _generate_scheduling_matrix(self, horizon: int, mode: str = "pyramid", uncertainty_scale: float = 1.0):
        """
        Generate scheduling matrix over a horizon, matching diffusion-forcing logic.
        Returns an int64 numpy array of shape (height, horizon) with entries in [0, df_sampling_timesteps].
        """
        if mode == "pyramid":
            if not self.df_is_ddim_sampling:
                # DDPM mode: ensure strict "step-by-step" constraint
                # Each token must have noise levels that decrease by exactly 1 per row
                # Start from the maximum noise level for each token position
                max_noise_per_token = np.array([
                    self.df_sampling_timesteps + int(t * uncertainty_scale)
                    for t in range(horizon)
                ], dtype=np.int64)
                # Clip to valid range
                max_noise_per_token = np.clip(max_noise_per_token, 0, self.df_sampling_timesteps)
                
                # Calculate height: need enough rows to go from max noise to 0 for each token
                max_start_noise = max_noise_per_token.max()
                height = max_start_noise + 2  # +2 to include 0 and ensure we can finish
                
                scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
                for t in range(horizon):
                    start_noise = max_noise_per_token[t]
                    # For each row m, noise level = max(0, start_noise - m)
                    # This ensures strict -1 per row constraint
                    for m in range(height):
                        scheduling_matrix[m, t] = max(0, start_noise - m)
                
                return scheduling_matrix
            else:
                # DDIM mode: more flexible scheduling allowed
                height = self.df_sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
                scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
                for m in range(height):
                    for t in range(horizon):
                        scheduling_matrix[m, t] = self.df_sampling_timesteps + int(t * uncertainty_scale) - m
                return np.clip(scheduling_matrix, 0, self.df_sampling_timesteps)
        elif mode == "full_sequence":
            return np.arange(self.df_sampling_timesteps, -1, -1, dtype=np.int64)[:, None].repeat(horizon, axis=1)
        else:
            raise ValueError(f"unknown scheduling mode {mode}")

    @torch.no_grad()
    def generate_df(
        self,
        idx,
        max_new_tokens: int,
        chunk_size: int = 0,
        scheduling_mode: str = "pyramid",
        uncertainty_scale: float = 1.0,
    ):
        """
        Diffusion Forcing style generation with scheduling matrix + sliding window.
        idx: (B, T_ctx) conditioning tokens
        Returns: (B, T_ctx + max_new_tokens) generated token indices
        """
        self.eval()
        device = idx.device
        b, t_ctx = idx.size()
        n_tokens = self.config.block_size

        # initial context embeddings as clean x0
        pos_ctx = torch.arange(0, t_ctx, dtype=torch.long, device=device)
        tok_emb_ctx = self.transformer.wte(idx)
        pos_emb_ctx = self.transformer.wpe(pos_ctx)
        scale = math.sqrt(self.config.n_embd)
        x0_ctx = (tok_emb_ctx + pos_emb_ctx.unsqueeze(0)) * scale
        # x0_ctx = tok_emb_ctx + pos_emb_ctx.unsqueeze(0)           # (B, T_ctx, C)
        xs_pred = x0_ctx.transpose(0, 1).contiguous()             # (T_ctx, B, C)

        curr_pos = t_ctx
        total_len = t_ctx + max_new_tokens

        while curr_pos < total_len:
            # determine horizon for this chunk
            effective_chunk_size = chunk_size if chunk_size > 0 else n_tokens
            horizon = min(total_len - curr_pos, effective_chunk_size)

            assert horizon <= n_tokens, (
                f"horizon={horizon}, total_len={total_len},curr_pos={curr_pos}, "
                f"chunk_size={chunk_size}, n_tokens={n_tokens} , horizon exceeds model block_size."
            )
            scheduling_matrix = self._generate_scheduling_matrix(
                horizon, mode=scheduling_mode, uncertainty_scale=uncertainty_scale
            )

            # initialize new chunk as noise
            chunk = torch.randn((horizon, b, self.config.n_embd), device=device)
            chunk = torch.clamp(chunk, -self.df_clip_noise, self.df_clip_noise)
            xs_pred = torch.cat([xs_pred, chunk], dim=0)          # (T_ctx + generated, B, C)

            # sliding window: only input the last n_tokens positions to the model
            start_pos = max(0, curr_pos + horizon - n_tokens)

            for m in range(scheduling_matrix.shape[0] - 1):
                # build from/to noise levels (prefix zeros for context)
                from_noise_levels = np.concatenate(
                    (np.zeros((curr_pos,), dtype=np.int64), scheduling_matrix[m])
                )[:, None].repeat(b, axis=1)                       # (curr_pos + horizon, B)
                to_noise_levels = np.concatenate(
                    (np.zeros((curr_pos,), dtype=np.int64), scheduling_matrix[m + 1])
                )[:, None].repeat(b, axis=1)

                from_noise_levels_t = torch.from_numpy(from_noise_levels).to(device=device, dtype=torch.long)
                to_noise_levels_t = torch.from_numpy(to_noise_levels).to(device=device, dtype=torch.long)

                xs_pred[start_pos:] = self._sample_step_seq(
                    xs_pred[start_pos:],
                    from_noise_levels_t[start_pos:],
                    to_noise_levels_t[start_pos:],
                )

            curr_pos += horizon

        # decode embeddings to tokens for the newly generated segment
        full_emb = xs_pred.transpose(0, 1)                         # (B, T_total, C)
        new_emb = full_emb[:, t_ctx : t_ctx + max_new_tokens, :]   # (B, max_new_tokens, C)
        
        # generate position embedding
        pos_new = torch.arange(t_ctx, t_ctx + max_new_tokens, dtype=torch.long, device=device)
        scale = math.sqrt(self.config.n_embd)
        pos_emb_new = self.transformer.wpe(pos_new)
        pos_emb_new_scaled = pos_emb_new.unsqueeze(0) * scale
        pure_new_emb = new_emb - pos_emb_new_scaled
        
        # logits = self.lm_head(new_emb)                             # (B, max_new_tokens, V)
        new_ids = self._decode_tokens(pure_new_emb)
        # new_ids = torch.argmax(logits, dim=-1)                     # (B, max_new_tokens)

        return torch.cat([idx, new_ids], dim=1)

    def _df_model_predictions(self, x, noise_levels):
        """
        Core Diffusion Forcing model: predicts noise/x0/v from noised input x and noise_levels.
        x: (B, T, C), noise_levels: (B, T)
        """
        b, t, _ = x.shape
        # time-step embedding (for diffusion step)
        t_embed = self.df_t_embed(noise_levels.view(-1))          # (B*T, C)
        t_embed = t_embed.view(b, t, -1)                          # (B, T, C)
        h = x + t_embed

        # GPT backbone
        h = self.transformer.drop(h)
        for block in self.transformer.h:
            h = block(h)
        h = self.transformer.ln_f(h)                              # (B, T, C)

        model_out = self.df_head(h)                               # (B, T, C)

        if self.df_objective == "pred_noise":
            pred_noise = torch.clamp(model_out, -self.df_clip_noise, self.df_clip_noise)
            x_start = self._predict_start_from_noise(x, noise_levels, pred_noise)
        elif self.df_objective == "pred_x0":
            x_start = model_out
            pred_noise = self._predict_noise_from_start(x, noise_levels, x_start)
        elif self.df_objective == "pred_v":
            v = model_out
            x_start = self._predict_start_from_v(x, noise_levels, v)
            pred_noise = self._predict_noise_from_start(x, noise_levels, x_start)
        else:
            raise ValueError(f"unknown df_objective {self.df_objective}")

        return pred_noise, x_start, model_out

    # --- training forward ----------------------------------------------------

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: unused (for API compatibility with GPT)
        Returns:
            logits: None
            loss: scalar diffusion-forcing loss
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        # clean token embeddings x0
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # (T,)
        tok_emb = self.transformer.wte(idx)                        # (B, T, C)
        scale = math.sqrt(self.config.n_embd)
        pos_emb = self.transformer.wpe(pos)                        # (T, C)
        x0 = (tok_emb + pos_emb.unsqueeze(0)) * scale
        # x0 = tok_emb + pos_emb.unsqueeze(0)                        # (B, T, C)

        # per-token noise levels and Gaussian noise
        noise_levels = torch.randint(0, self.df_timesteps, (b, t), device=device)  # (B, T)
        noise = torch.randn_like(x0)
        noise = torch.clamp(noise, -self.df_clip_noise, self.df_clip_noise)

        # diffuse x0 to x_t
        x_t = self._q_sample(x0, noise_levels, noise)                                # (B, T, C)

        # core DF model
        pred_noise, x_start_pred, model_out = self._df_model_predictions(x_t, noise_levels)

        if self.df_objective == "pred_noise":
            target = noise
        elif self.df_objective == "pred_x0":
            target = x0
        elif self.df_objective == "pred_v":
            target = self._predict_v(x0, noise_levels, noise)
        else:
            raise ValueError(f"unknown df_objective {self.df_objective}")

        mse = F.mse_loss(model_out, target.detach(), reduction="none")               # (B, T, C)
        loss_weight = self._compute_loss_weight(noise_levels)                        # (B, T)
        loss_weight = loss_weight.unsqueeze(-1)                                      # (B, T, 1)
        loss = (mse * loss_weight).mean()

        # TODO: 辅助交叉熵损失
        scale = math.sqrt(self.config.n_embd)
        pred_logits = self.lm_head(x_start_pred / scale) 

        # 计算标准的 GPT 分类 Loss
        ce_loss = F.cross_entropy(pred_logits.view(-1, pred_logits.size(-1)), idx.view(-1))

        # 3. 混合 Loss
        # 给 CE Loss 一个权重 (lambda)，通常 0.1 或 1.0 都可以
        # 这个 Loss 会提供极其鲜明的梯度，强迫模型把字"对准"
        loss += 0.5 * ce_loss

        # keep API identical to GPT
        return None, loss

    @torch.no_grad()
    def teacher_forcing_loss(self, idx, targets):
        """
        Return the standard GPT cross-entropy loss for logging / comparison.
        """
        _, loss = super().forward(idx, targets)
        return loss
    
    #---------test methods--------------
    @torch.no_grad()
    def test_copy_ability(self, idx):
        self.eval()
        device = idx.device
        b, t = idx.size()

        # --- 修正开始 ---
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        
        # 必须乘 scale！
        scale = math.sqrt(self.config.n_embd)
        x0 = (tok_emb + pos_emb.unsqueeze(0)) * scale
        # === DEBUG PRINT ===
        print(f"DEBUG: x0 mean: {x0.mean().item():.4f}, std: {x0.std().item():.4f}")
        print(f"DEBUG: x0 norm (magnitude): {x0.norm(dim=-1).mean().item():.4f}")
        # ===================
        # --- 修正结束 ---

        noise_levels = torch.zeros((b, t), dtype=torch.long, device=device)
        x_t = x0 # noise=0, so input is clean

        pred_noise, x_start_pred, model_out = self._df_model_predictions(x_t, noise_levels)
        
        mse_loss = F.mse_loss(x_start_pred, x0).item()

        # Decode
        if self.df_objective == "pred_x0":
            predicted_emb = x_start_pred
        else:
            # 如果是预测 noise，我们需要从 x_t (即 x0) 减去预测的 noise (应该是0) 恢复 x_start
            # 在 test_copy_ability 且 k=0 时，逻辑上 x_start_pred 应该就是结果
            predicted_emb = x_start_pred
        pos_emb_scaled = self.transformer.wpe(pos).unsqueeze(0) * scale
        
        # 1. 还原纯 Token 向量
        pure_token_pred = x_start_pred - pos_emb_scaled
        
        # 2. 用这个纯向量去解码
        pred_tokens = self._decode_tokens(pure_token_pred)
        # pred_tokens = self._decode_tokens(predicted_emb)

        return pred_tokens, mse_loss, x0, x_start_pred
    
    def _decode_tokens(self, embedding):
        """
        使用欧氏距离 (L2 Distance) 解码
        embedding: (B, T, C)
        """
        # 1. 获取 Embedding 权重并缩放 (如果你在 forward 里乘了 scale，这里也要乘)
        scale = math.sqrt(self.config.n_embd)
        w = self.transformer.wte.weight * scale # (V, C)
        
        # 2. 计算距离
        # embedding: (B, T, C)
        # w: (V, C) -> unsqueeze -> (1, V, C)
        # cdist 支持广播: (B, T, C) vs (1, V, C) -> (B, T, V)
        dists = torch.cdist(embedding, w.unsqueeze(0)) 
        
        # 3. 找最近邻
        # 此时 dists 形状是 (B, T, V)，argmin 后是 (B, T)
        return torch.argmin(dists, dim=-1)