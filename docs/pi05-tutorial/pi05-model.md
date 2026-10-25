This page is how `Pi0` implements π0.5: discrete state, adaRMS timestep injection, the attention mask, and the action chunk. The class is still `Pi0`.

文件是 `src/openpi/models/pi0.py` 和 `src/openpi/models/pi0_config.py`。`Pi0Config.model_type` 在 `pi05=True` 时返回 `ModelType.PI05`，否则 `PI0`。`create` 始终返回 `Pi0`。

`Pi0Config` 自己的注释写了和 π0 的两点差别：状态进离散语言 token，而不是 suffix 里的连续输入；action expert 用 adaRMS 注入 flow 的时间。代码还多一个默认：`max_token_len` 从 48 变成 200。`pi05_libero` 把第一点关掉了（`discrete_state_input=False`），第二点仍然开着。

## 构造时多出来的层

`pi05=True`：

- `time_mlp_in`、`time_mlp_out`：都是 `Linear(expert_width, expert_width)`，expert width 是 1024。
- 没有 `state_proj`，没有 `action_time_mlp_*`。

`pi05=False`：

- `state_proj`：`Linear(action_dim, expert_width)`。
- `action_time_mlp_in`：`Linear(2 * expert_width, expert_width)`。
- `action_time_mlp_out`：`Linear(expert_width, expert_width)`。

两边都有 `action_in_proj`（`action_dim → expert_width`）和 `action_out_proj`（`expert_width → action_dim`）。

## Prefix

`embed_prefix` 对 π0 和 π0.5 相同。每张图的 SigLIP token 拼上语言嵌入。每个图像 token 和每个语言 token 的 `ar_mask` 都是 `False`。按 `make_attn_mask` 的约定，`False` 表示和前一个 token 共享注意力块，所以图像和语言彼此全连接。

`make_attn_mask` 用 `cumsum(mask_ar)`。token i 能看见所有 cumsum 小于等于自己的有效 token。prefix 全是 `False` 时 cumsum 不变，块内全可见。

## Suffix 和注意力

`embed_suffix(obs, noisy_actions, timestep)`：

π0 先放一个状态 token：`state_proj(obs.state)`，形状 `[B, 1, 1024]`，`ar_mask` 是 `True`。这个 `True` 打开新的注意力块，prefix 不能看见状态和后面的动作。

π0.5 跳过这一步。状态如果要进模型，只能是 tokenizer 已经写进 prefix 的那些离散数字。

动作 token 是 `action_in_proj(noisy_actions)`，形状 `[B, action_horizon, 1024]`。时间是标量，`posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)`，正弦余弦各一半，对 `[0, 1]` 敏感。

π0.5 把这个时间向量送进 `time_mlp_in → swish → time_mlp_out → swish`，得到 `adarms_cond`，形状 `[B, 1024]`。动作 token 本身不加时间。`adarms_cond` 传给 Gemma 的第二个专家。

π0 把时间向量在 horizon 上复制，和动作 token 在最后一维拼接，再过 `action_time_mlp_*`。`adarms_cond` 是 `None`。

动作段的 `ar_mask` 是 `[True] + [False] * (action_horizon - 1)`。第一个动作 token 新开一块，后面的动作 token 和它全连接。prefix（以及 π0 的状态 token）看不见动作；动作能看见 prefix、状态和全部动作 token。

这就是动作 chunk：一次前向产出整个 horizon，而不是自回归地逐拍生成。`action_out_proj` 只取 suffix 输出的最后 `action_horizon` 个位置。π0 的 suffix 比 π0.5 多一个状态 token，所以必须用 `-action_horizon:` 切掉它。

## 和 π0-FAST 的差别

π0-FAST 在 `pi0_fast.py`。一个 Gemma 专家，输入是图像 token 加 FAST 文本 token，损失是移位后的 token 交叉熵，采样是 `decode=True` 的自回归。没有 `action_in_proj`，没有 flow 时间，没有 adaRMS。图像键在 DROID 上也不一样：FAST 用 `base_0_rgb`、`base_1_rgb`、`wrist_0_rgb`，并且不把 padding 图像 mask 掉。

π0.5 没有 FAST 头。知识隔离那套「VLM 预测离散动作、梯度不进 VLM」不在 `compute_loss` 里。全量 `pi05_*` 配置的 `freeze_filter` 是默认的 `nnx.Nothing`，VLM 和 action expert 都会收到 flow 损失的梯度。

## 和论文时间约定的差别

采样函数里的注释写明：这里用扩散文献里更常见的约定，`t=1` 是噪声，`t=0` 是数据，和 π0 论文相反。细节在 [flow matching](flow-matching.md)。

## 要改的地方

改注意力：`embed_prefix` / `embed_suffix` 里追加到 `ar_mask` 的布尔值，以及 `make_attn_mask`。把动作改成因果而不是一块双向，就是把后面的 `False` 改成 `True`。这会让已有 checkpoint 的注意力模式对不上，只能当新实验。

让 `pi05_libero` 也吃本体感觉：把该配置的 `discrete_state_input` 设为 `True`。不要在 `pi05=True` 时再加回 `state_proj`，那条分支只在 `not self.pi05` 时存在，权重里也没有 `state_proj`。
