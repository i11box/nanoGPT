# Diffusion Forcing Language Model - 项目文档

## 📋 项目目标

将 **Diffusion Forcing** 训练方法迁移到 **nanoGPT** 代码库，实现一个可以与标准 **Teacher Forcing** GPT 进行公平对比的语言模型。

**核心目标**：
- 在 nanoGPT 框架内实现完整的 Diffusion Forcing 训练和采样流程
- 保持 nanoGPT 的代码风格和简洁性
- 实现与原版 Diffusion Forcing 仓库（`diffusion-forcing/algorithms/diffusion_forcing`）**完全一致**的训练/采样逻辑
- 通过一个简单的开关（`use_diffusion_forcing`）切换 Teacher Forcing 和 Diffusion Forcing

---

## 🎯 技术背景：Diffusion Forcing 核心概念

### 什么是 Diffusion Forcing？

Diffusion Forcing 是一种结合了 **next-token prediction** 和 **full-sequence diffusion** 的训练方法：

1. **训练阶段**：
   - 每个 token 独立采样一个 noise level（例如 `[5, 12, 3, 20, ...]`）
   - 在 **embedding 空间**对每个 token 独立加噪：`x_t[i] = q_sample(x0[i], noise_level[i], noise[i])`
   - 模型**一次性并行预测整个序列**，但每个位置看到不同的 noise level（通过时间嵌入传入）
   - 使用 **Fused SNR Reweighting** 计算损失

2. **采样阶段**：
   - 使用**调度矩阵**（scheduling matrix）控制不同 token 的去噪进度（pyramid/trapezoid 调度）
   - **滑动窗口**机制：每次只对最后 `n_tokens` 个位置进行去噪
   - 多步迭代：通过调度矩阵的多个 row，逐步去噪整个序列

### 关键设计原则

⚠️ **重要理解**：Diffusion Forcing **不是**单个 token 逐个去噪，而是：
- **并行多 token 去噪**：模型一次性处理整个序列 `(B, T, C)`
- **独立 noise level**：每个 token 有独立的噪声级别，体现"因果不确定性"
- **调度矩阵**：采样时，越远的 token 噪声越大，需要更多去噪步数

---

## 🔑 关键实现细节

### 1. 模型架构：`DFGPT` 类

**位置**：`nanoGPT/model.py`，继承自 `GPT`

**核心组件**：
- **复用 GPT 组件**：
  - `transformer.wte`：token embedding
  - `transformer.wpe`：position embedding
  - `transformer.h`：GPT transformer blocks（保持因果注意力）
  - `transformer.ln_f`：final layer norm

- **新增 Diffusion 组件**：
  - `df_t_embed = SinusoidalPosEmb(n_embd)`：扩散时间步嵌入
  - `df_head = Linear(n_embd, n_embd)`：预测头，将 transformer 输出映射回 embedding 空间
  - Diffusion 缓冲区：`df_betas`, `df_alphas_cumprod`, `df_snr`, `df_clipped_snr` 等

### 2. 训练流程：`DFGPT.forward()`

**输入**：`idx: (B, T)` token indices

**流程**：
```python
1. x0 = wte(idx) + wpe(pos)                    # 干净 embedding (B, T, C)
2. noise_levels ~ Uniform[0, df_timesteps]    # 每个 token 独立采样 (B, T)
3. noise ~ N(0, I)                             # 高斯噪声 (B, T, C)
4. x_t = q_sample(x0, noise_levels, noise)     # 每个位置独立加噪
5. pred = _df_model_predictions(x_t, noise_levels)  # 一次性预测整个序列
6. loss = MSE(pred, target) * FusedSNR(noise_levels)
```

**关键点**：
- ✅ **一次性预测整个序列**：`_df_model_predictions` 接收 `(B, T, C)`，输出 `(B, T, C)`
- ✅ **每个 token 独立的 noise level**：通过 `df_t_embed(noise_levels)` 传入
- ✅ **Fused SNR Reweighting**：沿时间维度（token）做指数滑动平均

### 3. Fused SNR Reweighting

**位置**：`DFGPT._compute_loss_weight()`

**数学形式**（完全对应原版 `Diffusion.compute_loss_weights`）：
```python
# 1. 归一化 SNR
normalized_snr = snr / snr_clip
normalized_clipped_snr = clipped_snr / snr_clip

# 2. 沿时间维度累积（指数滑动平均）
cum_snr[t] = cum_snr_decay * cum_snr[t-1] + (1 - cum_snr_decay) * normalized_clipped_snr[t]

# 3. Fused SNR
fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_clipped_snr)

# 4. 权重（根据 objective）
if pred_noise:
    weight = clipped_fused_snr / fused_snr
elif pred_x0:
    weight = clipped_fused_snr * snr_clip
elif pred_v:
    weight = clipped_fused_snr * snr_clip / (fused_snr * snr_clip + 1)
```

### 4. 采样流程：`DFGPT.generate_df()`

**核心逻辑**（对应原版 `DiffusionForcingBase.validation_step`）：

```python
1. 初始化 context embeddings 为干净 x0
2. while curr_pos < total_len:
     a. 生成调度矩阵（pyramid/full_sequence）
     b. 初始化新 chunk 为高斯噪声
     c. 滑动窗口：start_pos = max(0, curr_pos + horizon - block_size)
     d. 对调度矩阵的每一行：
        - 构建 from_noise_levels / to_noise_levels
        - xs_pred[start_pos:] = sample_step(xs_pred[start_pos:], from, to)
     e. curr_pos += horizon
3. 解码 embedding → token ids
```

**调度矩阵**（`_generate_scheduling_matrix`）：
- **pyramid**：`scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m`
- **full_sequence**：所有 token 同步去噪

### 5. DDPM/DDIM 采样：`_sample_step_seq()`

**关键特性**（完全对应原版 `Diffusion.sample_step`）：
- **Stabilization**：`noise_level == -1` 时，用 `stabilization_level-1` 重新 `q_sample` context
- **只更新 noise level 下降的位置**：`torch.where(curr == next, orig_x, x_pred)`
- **调度索引映射**：`[0, sampling_timesteps]` → `[-1, timesteps-1]`

---

## ⚠️ 关键注意点

### 1. 与原版 Diffusion Forcing 的对应关系

| 原版（diffusion-forcing） | 当前实现（nanoGPT） |
|---------------------------|---------------------|
| `Diffusion.forward()` | `DFGPT.forward()` |
| `Diffusion.compute_loss_weights()` | `DFGPT._compute_loss_weight()` |
| `Diffusion.model_predictions()` | `DFGPT._df_model_predictions()` |
| `Diffusion.sample_step()` | `DFGPT._sample_step_seq()` |
| `Diffusion.ddim_sample_step()` | `DFGPT._ddim_sample_step_seq()` |
| `DiffusionForcingBase.validation_step()` | `DFGPT.generate_df()` |
| `DiffusionForcingBase._generate_scheduling_matrix()` | `DFGPT._generate_scheduling_matrix()` |

---

## 🚀 使用方法

### 1. Teacher Forcing（标准 GPT）

```bash
python train.py \
    --config=config/train_shakespeare_char.py \
    --use_diffusion_forcing=False
```

### 2. Diffusion Forcing

```bash
python train.py \
    --config=config/train_shakespeare_char.py \
    --use_diffusion_forcing=True
```

### 3. 配置 Diffusion Forcing 超参

在 `GPTConfig` 或配置文件中设置：

```python
df_timesteps = 32              # 扩散时间步数
df_sampling_timesteps = 32     # 采样时间步数（≤ timesteps）
df_snr_clip = 5.0              # SNR 剪裁值
df_cum_snr_decay = 0.95        # Fused SNR 衰减系数
df_objective = "pred_noise"    # "pred_noise" / "pred_x0" / "pred_v"
df_beta_schedule = "cosine"    # "linear" / "cosine" / "sigmoid"
df_clip_noise = 5.0            # 噪声剪裁
df_ddim_eta = 0.0              # DDIM eta（0=确定性，>0=随机）
df_stabilization_level = 0     # Stabilization level
```

---

## 📝 待办事项 / 可选改进

1. **时间嵌入融合方式**：如需完全一致，可改为拼接 + MLP（当前是简单相加）
2. **Guidance 机制**：可添加类似 `df_planning.goal_guidance()` 的条件引导
3. **更多调度模式**：trapezoid、autoregressive 等
4. **评估指标**：添加困惑度、BLEU 等，便于对比 Teacher Forcing vs Diffusion Forcing

---

## 📚 参考

- **原版 Diffusion Forcing 仓库**：`D:\05_Project\03_Python\toys\diffusion-forcing\algorithms\diffusion_forcing`
- **关键文件**：
  - `models/diffusion.py`：核心 Diffusion 类
  - `models/transformer.py`：序列 Transformer
  - `df_base.py`：基础训练/采样逻辑
  - `df_planning.py`：规划任务（含 Guidance）

---

## 💡 快速理解要点

1. **Diffusion Forcing = 在 embedding 空间做扩散，而不是 token 空间**
2. **训练时：并行多 token，每个 token 独立 noise level**
3. **采样时：调度矩阵 + 滑动窗口 + 多步迭代**
4. **保持 nanoGPT 风格：单文件、简洁 API、最小改动**
5. **完全对应原版逻辑：数学公式、采样流程、调度矩阵都一致**
```