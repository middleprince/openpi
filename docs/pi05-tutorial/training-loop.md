This page is one JAX training step: who is frozen, what is logged, and where checkpoints are written. It is not the 10k-hour pretraining run.

入口是 `scripts/train.py`。`main(_config.cli())` 用 tyro 从 `_CONFIGS` 选配置，并允许覆盖字段。仓库 README 的写法：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

`compute_norm_stats.py` 的参数名是 `config_name`，tyro 会把它变成 `--config-name`。`max_frames` 可选。写指南时没有执行这两条命令，未验证。

## 启动时做的事

`batch_size` 必须能被 `jax.device_count()` 整除，否则 `main` 直接 `ValueError`。

`init_train_state`：

1. 用 `config.model.create` 建 `Pi0`。
2. `weight_loader.load` 把 checkpoint 里对得上的叶子填进去。`CheckpointWeightLoader` 的路径类似 `gs://openpi-assets/checkpoints/pi05_base/params`。对不上的叶子保持初始化。
3. `freeze_filter` 命中的参数转成 bfloat16。
4. 优化器只 `init` 在 `trainable_filter` 上。
5. `ema_decay` 不是 `None` 时，`ema_params` 从当前参数复制一份。

默认优化器是 `AdamW`：`b1=0.9`，`b2=0.95`，`eps=1e-8`，`weight_decay=1e-10`，`clip_gradient_norm=1.0`。注释说 weight decay 设成 0 会在某些情况下 OOM，所以用一个很小的数。默认学习率是 `CosineDecaySchedule`：`warmup_steps=1000`，`peak_lr=2.5e-5`，`decay_steps=30000`，`decay_lr=2.5e-6`。`pi05_libero` 盖掉了这份默认：warmup 10000，peak 和 decay 都是 `5e-5`，`decay_steps=1_000_000`，`ema_decay=0.999`，`batch_size=256`。

`fsdp_devices > 1` 时按 README 的说明在设备上切分，用显存换速度。`--fsdp-devices` 是 tyro 对 `fsdp_devices` 的覆盖。README 写训练脚本还不支持多机。

## 一个 step

`train_step` 把 `model.train()` 打开，然后 `nnx.value_and_grad`，`argnums` 是 `DiffState(0, config.trainable_filter)`。冻结参数没有梯度。

更新之后如果 `ema_decay` 有值：`ema = decay * ema + (1 - decay) * new`。`pi05_libero` 的 0.999 意味着 EMA 走得很慢。

日志字典只有三个数：`loss`、`grad_norm`、`param_norm`。`param_norm` 排除了 bias、scale、pos_embedding、input_embedding，以及 ndim ≤ 1 的张量。每 `log_interval` 步（默认 100）对这段 step 取平均，打印并 `wandb.log`。`wandb_enabled=False` 时 `wandb.init(mode="disabled")`。

第一个 batch 还会把相机图横向拼起来，以 `camera_views` 记到 wandb 的 step 0。这是检查图像键有没有接错的现成信号。

每 `save_interval` 步（默认 1000），以及最后一步，调用 `checkpoints.save_state`。`keep_period` 控制哪些 step 不被删。`resume` 和 `overwrite` 不能同时为真。

## 存下去的是哪份权重

`_split_params`：如果 `ema_params` 不是 `None`，写进 checkpoint 项 `params` 的是 EMA，训练状态里的 `ema_params` 被清空后另存。没有 EMA 时，`params` 就是在线权重。`create_trained_policy` 加载的是这份 `params`。所以开着 EMA 的 π0.5 微调，推理用的是 EMA，不是最后一步的在线权重。

`save_assets` 把数据配置里的 norm stats 拷进 checkpoint 的 `assets/`。推理从这里读，不从 `./assets` 读。

## 谁在被训练

全量 π0.5 配置没有设置 `freeze_filter`，因此是 `nnx.Nothing`。flow 损失的梯度进入：

- SigLIP（`PaliGemma.img`）
- VLM 专家（`llm` 里不带 `_1` 的 Gemma）
- action expert（`llm` 里带 `_1` 的 Gemma）
- `action_in_proj`、`time_mlp_in`、`time_mlp_out`、`action_out_proj`

这和 π0.5 博客里的 knowledge insulation 不是一回事。博客里的离散动作头和 stop-gradient 不在 `compute_loss` 里。若要让 VLM 基座不更新，需要自己设 `freeze_filter`。现成的写法在 `pi0_libero_low_mem_finetune`：variant 带 `lora`，`freeze_filter` 用 `get_freeze_filter()`，`ema_decay=None`。那份配置加载的是 `pi0_base`，不是 `pi05_base`。

## 一个下午和完整配方

`debug_pi05` 用 `dummy` 宽度、`FakeDataConfig`、`batch_size=2`、`num_train_steps=10`、`wandb_enabled=False`、`overwrite=True`，并且已经填了 `exp_name`。它不下载机器人数据，也不加载 `pi05_base`。命令形式与 README 相同：`uv run scripts/train.py debug_pi05`。未验证。

`pi05_libero` 是 30000 step、batch 256、从 `pi05_base` 全量微调。README 的显存表把全量微调放在 > 70 GB。`pi05_full_droid_finetune` 是 100000 step、batch 256、RLDS、`num_workers=0`。旁边的 `pi0_fast_full_droid_finetune` 注释写 100k step 在 8 张 H100 上大约 2 天。π0.5 那份配置没有重复这句话，不要把 2 天直接安到 `pi05_full_droid_finetune` 上。两份配置的数据目录还是占位或内网路径（`/mnt/pi-data/kevin` 和 `<path_to_droid_rlds_dataset>`），本机不能直接跑。

预训练 `pi05_base` 所需的 1 万小时以上混合数据不在仓库里。
