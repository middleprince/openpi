This page is the flow-matching loss and the Euler sampler inside `Pi0.compute_loss` and `Pi0.sample_actions`. The time convention is the opposite of the π0 paper.

JAX 实现在 `src/openpi/models/pi0.py`。PyTorch 在 `src/openpi/models_pytorch/pi0_pytorch.py` 的 `forward` 和 `sample_actions`，公式相同。

## 训练

`compute_loss(rng, observation, actions, train=False)` 返回 `[batch, action_horizon]`，每个动作步是动作维上的均方误差。`train_step` 再对这个张量做 `jnp.mean`。

步骤：

1. `preprocess_observation(..., train=train)`。
2. `noise ~ Normal(0, 1)`，形状与 `actions` 相同，即 `[B, ah, ad]`。
3. `time = Beta(1.5, 1) * 0.999 + 0.001`，形状 `[B]`。时间落在 `(0.001, 1)`。
4. `x_t = time * noise + (1 - time) * actions`。
5. `u_t = noise - actions`。这是 `x_t` 对时间的速度：`dx/dt = noise - actions`。
6. 一次前向把 prefix 和 suffix 拼在一起，`adarms_cond` 是 `[None, time_emb]`（π0.5）或 `[None, None]`（π0）。
7. `v_t = action_out_proj(suffix_out[:, -action_horizon:])`。
8. 损失是 `mean((v_t - u_t)^2, axis=-1)`。

`t` 靠近 1 时 `x_t` 接近噪声；靠近 0 时接近真实动作。网络预测的是指向噪声、离开数据的那个向量，采样时再乘上负的 `dt` 走回去。

PyTorch 的 `sample_time` 同样是 `sample_beta(1.5, 1.0) * 0.999 + 0.001`。`forward` 返回 `mse_loss(..., reduction="none")`，形状注释写的是 `batch × num_steps × num_motors`，这里的 num_steps 指动作 horizon，不是积分步数。

## 采样

`sample_actions(rng, observation, num_steps=10, noise=None)`：

1. `preprocess_observation(..., train=False)`。
2. `dt = -1.0 / num_steps`。
3. `noise` 缺省时是标准正态，形状 `[B, action_horizon, action_dim]`。
4. 先只跑 prefix，把 KV cache 填上。prefix 在积分过程中不变。
5. `while_loop` 从 `time=1` 开始。每步用当前 `x_t` 跑 suffix（cache 不更新），得到 `v_t`，然后 `x_t = x_t + dt * v_t`，`time = time + dt`。
6. 停止条件是 `time >= -dt / 2`，用来躲开浮点误差。`num_steps=10` 时走 10 步，停在大约 0。

返回的是积分结束的 `x_0`，形状 `[B, action_horizon, action_dim]`。`Policy.infer` 去掉 batch 维，再做 `Unnormalize` 和 `*Outputs`。

PyTorch 的 `sample_actions(device, observation, noise=None, num_steps=10)` 同样 `dt = -1 / num_steps`，用 `torch.no_grad`。第一个参数是 device 字符串，不是 JAX 的 rng。`Policy` 在 PyTorch 分支把 device 传进去。

## 和 π0 论文相反的时间

`sample_actions` 的注释原文是：这里用扩散文献里更常见的约定，`t=1` 是噪声，`t=0` 是目标分布，和 π0 论文相反。改损失时如果按论文把 `t=0` 当成噪声，就要同时改 `x_t` 的插值、`u_t` 的符号和 `dt` 的符号。只改一边，采样会朝噪声走。

## 步数和 horizon 是两件东西

`num_steps` 是积分步数，默认 10。它不是权重的一部分。`Policy` 把 `sample_kwargs` 原样传给 `sample_actions`，所以 `create_trained_policy(..., sample_kwargs={"num_steps": 5})` 是真实参数。`serve_policy.py` 没有这个命令行开关。

`action_horizon` 是 chunk 长度，来自 `Pi0Config`，决定 `noisy_actions` 的时间维。它也不是 `Linear` 的形状。公开 checkpoint 按自己的 horizon 训练：LIBERO 10，DROID 推理 15，DROID 微调 16，ALOHA 默认 50。

## 要改的地方

时间分布：JAX 的 `jax.random.beta(time_rng, 1.5, 1, batch_shape)` 和 PyTorch 的 `sample_beta(1.5, 1.0, ...)`。两边要一起改。

积分器：JAX 的 `step` 闭包和 PyTorch `sample_actions` 里的循环。现在是欧拉法，一步 `x + dt * v`。没有二阶求解器。

损失只在动作维上平均，horizon 维留到 `train_step` 再和 batch 一起平均。若要给靠近当前时刻的步更高权重，改 `compute_loss` 的 `jnp.mean(..., axis=-1)` 之前。
