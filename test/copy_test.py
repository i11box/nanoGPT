"""
Test if Diffusion Forcing model can "copy" input when noise_level = 0
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from nanoGPT.model import GPTConfig, GPT, DFGPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = '/inspire/ssd/project/video-generation/public/hyr/out-webnovel' # ignored if init_from is not 'resume'
test_text = "叶秋深吸了一口气"  # Test input
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
use_diffusion_forcing = True
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = DFGPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode test input
test_ids = encode(test_text)
x = torch.tensor(test_ids, dtype=torch.long, device=device)[None, ...]  # (1, T)

print(f"Input text: '{test_text}'")
print(f"Input tokens: {test_ids}")
print(f"Input tensor shape: {x.shape}")

# Test copy ability
with torch.no_grad():
    with ctx:
        if hasattr(model, 'test_copy_ability'):  # DFGPT has this method
            print("\n=== Testing Diffusion Forcing Copy Ability ===")
            pred_tokens, mse_loss, x0, x_start_pred = model.test_copy_ability(x)

            print(f"MSE between input embedding and predicted embedding: {mse_loss:.6f}")
            print(f"Predicted tokens: {pred_tokens[0].tolist()}")
            print(f"Decoded prediction: '{decode(pred_tokens[0].tolist())}'")

            # Check if tokens match
            input_tokens = x[0].tolist()
            pred_tokens_list = pred_tokens[0].tolist()
            token_match = input_tokens == pred_tokens_list
            print(f"Token-level match: {token_match}")

            if token_match:
                print("✅ SUCCESS: Model can perfectly copy input when noise_level=0")
            else:
                print("❌ FAILURE: Model cannot copy input correctly")
                print("This suggests issues with:")
                print("  1. Output layer (df_head) dimensions")
                print("  2. Missing ReLU that clips negative values")
                print("  3. Missing normalization (e.g., divide by sqrt(d_model))")
                print("  4. Wrong objective setting")

        else:
            print("Model does not have test_copy_ability method (probably standard GPT)")
