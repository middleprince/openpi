This page lists the π0.5 `TrainConfig` entries, the public checkpoints they load, and what a checkpoint directory contains.

配置都在 `src/openpi/training/config.py` 的 `_CONFIGS`。按名字取用 `get_config`。命令行用 `tyro`，第一个参数是配置名。

## 公开 checkpoint

仓库 `README.md` 的表。基座用于微调，专家用于直接推理。下载由 `openpi.shared.download.maybe_download` 完成，缓存默认 `~/.cache/openpi`，可用环境变量 `OPENPI_DATA_HOME` 改位置。

| 名字 | 路径 | 代码里谁用它 |
| --- | --- | --- |
| π0 base | `gs://openpi-assets/checkpoints/pi0_base` | `pi0_libero`、`pi0_aloha_pen_uncap`、`pi0_aloha_sim` 加载 `.../params` |
| π0-FAST base | `gs://openpi-assets/checkpoints/pi0_fast_base` | `pi0_fast_libero`、`pi0_fast_full_droid_finetune` |
| π0.5 base | `gs://openpi-assets/checkpoints/pi05_base` | `pi05_libero`、`pi05_aloha_pen_uncap`、`pi05_full_droid_finetune` 加载 `.../params`。`serve_policy` 的 ALOHA 默认直接把目录指到 `pi05_base` |
| π0.5 LIBERO | `gs://openpi-assets/checkpoints/pi05_libero` | `serve_policy` 的 `EnvMode.LIBERO`。README 写这是用 `pi05_libero` 训出来的 |
| π0.5 DROID | `gs://openpi-assets/checkpoints/pi05_droid` | `serve_policy` 的 `EnvMode.DROID`，以及 `pi05_droid_finetune` 的权重和 assets |

π0 的 ALOHA 毛巾、保鲜盒、开笔，以及 `pi0_droid`、`pi0_fast_droid`、`pi0_aloha_sim` 也在同一张表里。`pi05_aloha` 这个配置名存在，默认 serve 用它配 `pi05_base`，表里没有单独的 `pi05_aloha` 专家 checkpoint。

## π0.5 配置对照

下面只列和实验有关的字段。没写的字段用 `TrainConfig` 的默认：`num_train_steps=30000`，`batch_size=32`，`ema_decay=0.99`，`save_interval=1000`，`AdamW`，余弦学习率 peak `2.5e-5`。

| name | 模型 | 权重 | 数据 | 值得注意的覆盖 |
| --- | --- | --- | --- | --- |
| `pi05_aloha` | `pi05=True`，horizon 50 | 无 loader | asset id `trossen` | 推理用。`policy_metadata` 里有 `reset_pose` |
| `pi05_droid` | horizon 15 | 无 loader | asset id `droid` | 推理用。`prompt_from_task=True` |
| `pi05_libero` | horizon 10，`discrete_state_input=False` | `pi05_base/params` | LeRobot `physical-intelligence/libero` | batch 256，lr `5e-5`，warmup 10000，ema 0.999，`extra_delta_transform=False`。`pytorch_weight_path` 是占位路径 |
| `pi05_aloha_pen_uncap` | `pi05=True` | `pi05_base/params` | `physical-intelligence/aloha_pen_uncap_diverse`，复用 `pi05_base/assets` 的 `trossen` | 20000 step，batch 64，默认 prompt `uncap the pen` |
| `pi05_full_droid_finetune` | horizon 16，`action_dim=32` | `pi05_base/params` | RLDS，`rlds_data_dir="/mnt/pi-data/kevin"` | 100000 step，batch 256，`num_workers=0`，复用 `pi05_base` 的 droid stats |
| `pi05_droid_finetune` | horizon 16 | `pi05_droid/params` | 你自己的 LeRobot repo id | 20000 step，batch 32，必须复用 `pi05_droid/assets` 的 `droid` |
| `debug_pi05` | dummy 宽度，`pi05=True` | 无 | `FakeDataConfig` | 10 step，batch 2，不启 wandb |

`pi0_libero` 与 `pi05_libero` 的结构差别就是实验页要打印的那些字段：`pi05`、`discrete_state_input`、`action_horizon`、`extra_delta_transform`、weight loader 的路径、学习率和 EMA。`pi0_fast_libero` 的 `model` 类型是 `Pi0FASTConfig`，`action_dim=7`，没有 `pi05` 字段。

## Checkpoint 目录

训练写到 `checkpoint_base_dir / name / exp_name`，默认 `./checkpoints/<config>/<exp_name>`。Orbax 管理 step 子目录。推理 README 的例子是：

```text
checkpoints/pi05_libero/my_experiment/20000
```

这个目录里应有 `params`（JAX）或转换后的 `model.safetensors`（PyTorch），以及 `assets/<asset_id>/norm_stats.json`。`create_trained_policy` 用 `params` 或 `model.safetensors` 判断后端。

`resume=True` 时从该目录的最新 step 恢复，包括优化器状态。`overwrite=True` 会清掉同名目录。二者不能同时开。

## 要改的地方

新实验先复制一份最接近的 `TrainConfig`，改 `name`。`name` 必须唯一，文件末尾有检查。

微调 π0.5 LIBERO 式的 LeRobot 数据：复制 `pi05_libero`，换 `repo_id`，决定要不要 `extra_delta_transform`，先跑 `compute_norm_stats.py`。若机器人在预训练混合里出现过，并且动作定义一致，改为 `AssetsConfig` 指向 `pi05_base/assets` 的对应 asset id，跳过重新统计。`docs/norm_stats.md` 讨论了这两种选择。

DROID 小数据集：用 `pi05_droid_finetune`，不要改掉 `assets_dir`。注释写了必须复用原来的 DROID norm stats。转换脚本是 `examples/droid/convert_droid_data_to_lerobot.py`。
