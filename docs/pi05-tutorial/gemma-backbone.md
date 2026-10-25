This page is the two-expert Gemma used as the VLM backbone and the action expert, including adaRMS and which parameters LoRA freezes.

实现在 `src/openpi/models/gemma.py`。π0 和 π0.5 共用这一份。π0-FAST 用 `gemma_fast.py`，只有一个专家，并带解码用的 KV cache。

## 宽度

`get_config` 的几种 `Variant`：

| variant | width | depth | mlp | heads | kv heads | head dim | LoRA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dummy` | 64 | 4 | 128 | 8 | 1 | 16 | 无 |
| `gemma_2b` | 2048 | 18 | 16384 | 8 | 1 | 256 | 无 |
| `gemma_2b_lora` | 2048 | 18 | 16384 | 8 | 1 | 256 | attn 和 ffn，rank 16，alpha 16 |
| `gemma_300m` | 1024 | 18 | 4096 | 8 | 1 | 256 | 无。注释写 311M |
| `gemma_300m_lora` | 1024 | 18 | 4096 | 8 | 1 | 256 | rank 32，alpha 32 |

`Pi0Config` 默认 `paligemma_variant="gemma_2b"`、`action_expert_variant="gemma_300m"`。两个专家的 depth、head 数、kv 头数、head dim 必须相同，因为 `Attention` 会把它们的 QKV 在序列维拼起来。宽度可以不同：每个专家有自己的投影，投到共享的 head dim。

`debug` 和 `debug_pi05` 把两个 variant 都设成 `dummy`，用来在没有大 GPU 的时候把训练循环跑通。

## 双专家怎么接

`Module` 的 `configs` 是一个序列。第 0 个是 PaliGemma，第 1 个是 action expert。词嵌入只建在第 0 个专家上，`PALIGEMMA_VOCAB_SIZE = 257_152`。`embed` 把 token 乘上 `sqrt(width)`。

`__call__` 接收 `embedded: Sequence`，某一段可以是 `None`（推理时先只跑 prefix，再只跑 suffix）。`Attention` 跳过 `None`，把剩下的 Q、K、V 在时间维拼接，做一次注意力，再按原来的长度切回各专家，用各自的 `attn_vec_einsum` 投回自己的宽度。

参数名：`_name` 对专家 0 不加后缀，这样能直接加载 PaliGemma checkpoint；专家 1 加 `_1`，例如 `attn_1`。`get_freeze_filter` 用 `.*llm.*_1.*` 识别 action expert。

RoPE 在 `_apply_rope`，`max_wavelength=10_000`。位置是调用方算好的整数向量，不是 block 内部的 `arange`。π0 和 π0.5 的位置是 `cumsum(input_mask) - 1`，padding 不占位置。

## adaRMS

`RMSNorm.__call__(x, cond)`：

- `cond is None`：普通 RMSNorm，可学习的 `scale` 初始化为 0，输出 `(1 + scale) * x / rms`。
- `cond` 不是 `None`：一个 `Dense` 把 cond 映到 `3 * width`，拆成 scale、shift、gate。归一化结果是 `normed * (1 + scale) + shift`。gate 交给 `_gated_residual`：`x + y * gate`。cond 为 `None` 时残差就是 `x + y`。

方差在 float32 里算。`Dense` 的 kernel 初始化为 0，所以训练开始时 adaRMS 接近恒等，gate 接近 0。这会让动作专家在第一步几乎不吃残差分支，直到这些 Dense 离开 0。

`Pi0` 构造 Gemma 时 `adarms=config.pi05`，并且 `lazy_init(..., use_adarms=[False, True] if config.pi05 else [False, False])`。π0.5 只在 action expert 上建 adaRMS 的 Dense。VLM 专家的 `adarms_cond` 始终是 `None`。时间向量不进 VLM。

## 冻结

`get_freeze_filter`：

- 名字里没有 `lora`：返回 `nnx.Nothing`，全部可训练。
- 只有 PaliGemma 是 LoRA：冻结 `.*llm.*`，但排除 `.*llm.*_1.*`，再排除 `.*lora.*`。action expert 全训练，VLM 只训练 LoRA。
- 只有 action expert 是 LoRA：只冻结 action expert 的非 LoRA 参数。
- 两边都是 LoRA：两个专家的基座都冻住，LoRA 可训练。

SigLIP（`PaliGemma.img`）和动作投影不在 `.*llm.*` 下，LoRA 配置里它们仍然更新。

`scripts/train.py` 的 `init` 把 `freeze_filter` 命中的参数 cast 成 bfloat16。`train_step` 的梯度只对 `trainable_filter`（`Param` 且不在 freeze 里）计算。

现成的 LoRA `TrainConfig` 是 `pi0_libero_low_mem_finetune` 和 `pi0_fast_libero_low_mem_finetune`。没有名为 π0.5 的 LoRA 配置。要冻住 π0.5 的 VLM，沿用 `get_freeze_filter` 的写法，把 variant 改成带 `_lora` 的名字，并令 `freeze_filter` 和 `model` 用同一份 config。LoRA 例子把 `ema_decay` 设为 `None`。

## 要改的地方

换骨干宽度：改 `get_config`，并保证两个专家的 depth 和 head dim 一致。已发布的 `pi05_base` 对得上 `gemma_2b` + `gemma_300m`，改宽度就不能加载那份权重。

adaRMS 的注入点在 `Block.__call__` 的两处 `RMSNorm`（attention 前和 FFN 前）以及 `final_norms`。π0.5 的时间 MLP 输出作为 `adarms_cond[1]` 传进来，见 `Pi0.embed_suffix`。
