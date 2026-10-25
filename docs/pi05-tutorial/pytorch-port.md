This page is the PyTorch port of π0 and π0.5: what it shares with JAX, which training features are absent, and how checkpoints are converted.

代码在 `src/openpi/models_pytorch/pi0_pytorch.py`，类名 `PI0Pytorch`。Gemma 和 SigLIP 用的是 `transformers_replace/` 里的补丁，不是上游 `transformers` 原文件。仓库 README（PyTorch Support，2025 年 9 月）写了已在 LIBERO 上验证推理和微调。同一节列出当前没有的功能：π0-FAST、混合精度训练、FSDP、LoRA、训练时的 EMA。

## 和 JAX 对齐的部分

`PI0Pytorch.__init__` 读同一个 `Pi0Config`。`pi05=True` 时 `use_adarms=[False, True]`，并建 `time_mlp_in` / `time_mlp_out`。`pi05=False` 时建 `state_proj` 和 `action_time_mlp_*`。

`embed_prefix` 仍是图像 token 加语言 token，图像段的 attention mask 用 0，表示块内可见。`embed_suffix` 在非 π0.5 时加状态 token。时间编码和 JAX 一样进 adaRMS 或进 action-time MLP。`forward` 的插值是 `x_t = t * noise + (1 - t) * actions`，`u_t = noise - actions`。`sample_actions` 从 `t=1` 以 `dt = -1/num_steps` 做欧拉积分，默认 10 步。

`preprocessing_pytorch.py` 复现非 wrist 的 95% 裁剪、转回原分辨率、约 ±5 度旋转，以及颜色抖动。几何增强用 `torch.randint` 和 `grid_sample`，为了能过 `torch.compile`，避免 `.item()`。它和 JAX 的 augmax 不是同一次随机数，数值不会逐元素一致。

## 补丁和精度

README 要求 `transformers` 4.53.2，然后：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

补丁做三件事：AdaRMS、激活精度、以及 KV cache 可以只读不写。README 警告：uv 默认 hardlink 时，这会改到 uv cache 里的 transformers，别的项目也会看到。撤销要 `uv cache clean transformers`。

JAX 推理主要是 bfloat16，少数计算用 float32。JAX 训练默认混合精度：权重和梯度 float32，大部分激活 bfloat16；把 config 的 `dtype` 设成 float32 可以全精度训练。

PyTorch 推理把大部分权重放在 bfloat16，少数层转回 float32。`create_trained_policy` 在加载后调用 `model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")`。PyTorch 训练是全 bfloat16（默认）或全 float32，由 `pytorch_training_precision` 控制。没有混合精度。README 写 bfloat16 更省显存，loss 比 float32 高。

`Pi0Config.pytorch_compile_mode` 默认 `"max-autotune"`，允许的值还有 `default`、`reduce-overhead`、`max-autotune-no-cudagraphs`。

## 权重怎么进来

JAX checkpoint 是 Orbax 目录，推理文件在 `params`。PyTorch 是 `model.safetensors`。`BaseModelConfig.load_pytorch` 用 `safetensors.torch.load_model`。`create_trained_policy` 看到 checkpoint 目录里有 `model.safetensors` 就走 PyTorch，否则走 JAX。同一套 `Policy.infer`。

转换命令在 README：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

`pi05_libero` 的 `pytorch_weight_path` 在配置里是占位字符串 `"/path/to/your/pytorch_weight_path"`。JAX 训练不读它。PyTorch 训练才会用它加载基座。

## 训练入口

README 里的命令，注意 JAX 用 `--exp-name`，PyTorch 示例用 `--exp_name`：

```bash
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume

uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
```

单机多卡和多机命令都在 README 的 Finetuning with PyTorch。JAX 的 `scripts/train.py` 不支持多机，README Requirements 写了这一点。JAX 的多卡是 `fsdp_devices`。

这些命令在写本指南时没有运行，未验证。

## 要改的地方

改 flow 或 adaRMS 时，JAX 的 `pi0.py` 和 PyTorch 的 `pi0_pytorch.py` 是两份代码。只改一边，转换后的 checkpoint 行为会对不上。

PyTorch 没有 LoRA 和 EMA。想做低显存 π0.5 微调，用 JAX 的 `get_freeze_filter`，不要在 `train_pytorch.py` 里找一个不存在的 LoRA 开关。
