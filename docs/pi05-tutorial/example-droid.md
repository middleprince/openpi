This page is the DROID π0.5 checkpoint: the inference keys, the 15-step action chunk, and the two fine-tuning configs.

`examples/droid/README.md` 把 `pi05_droid` 写成当前最强的通用 DROID 策略，并指向 RoboArena。推理显存按仓库总表是 > 8 GB，文档自己写「大约一张 4090」。写指南时没有跑，未验证。

## 推理配置

`pi05_droid` 的模型是 `Pi0Config(action_horizon=15, pi05=True)`。`discrete_state_input` 没有写，因此等于 `True`：7 个关节加 1 个夹爪会离散化进 prompt。`action_dim` 保持 32。`DroidOutputs` 只把前 8 维还给机器人，不改时间维。因此 `pi05_droid` 的 `infer` 返回形状是 `(15, 8)`。

`examples/droid/main.py` 和这个配置不一致。注释写返回的是 `[10, 8]` 的关节速度加夹爪，并且 `assert pred_action_chunk.shape == (10, 8)`。`open_loop_horizon` 默认 8，注释说大约 0.5 秒，因为同文件把 `DROID_CONTROL_FREQUENCY` 设为 15。夹爪按 0.5 二值化，然后整条动作 `clip` 到 `[-1, 1]`。按今天的 `pi05_droid` 配置启动服务器，这个断言会失败。`pi0_droid` 和 `pi0_fast_droid` 的 horizon 才是 10。要用这份客户端跑 π0.5，需要把断言改成配置里的 horizon，或者把 `pi05_droid` 的 `action_horizon` 改成 10 再加载。改 horizon 能加载（权重形状不依赖它），但是否等于发布时的训练长度，仓库没有写明。未验证。

`docs/norm_stats.md` 写 DROID 的动作是 7 维关节速度加 1 维夹爪。`pi05_full_droid_finetune` 却把 `action_space` 设为 `DroidActionSpace.JOINT_POSITION`。`droid_rlds_dataset.py` 在 `JOINT_POSITION` 时读 `joint_position`，否则读 `joint_velocity`，再接上 `gripper_position`。推理客户端的注释说的是速度。这三处要一起看，不要假定它们是同一种动作。

`DroidInputs` 在 `PI0` 和 `PI05` 上使用同一套图像名：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。第三张是零，mask 为 `False`。π0-FAST 用另一套名字（`base_0_rgb`、`base_1_rgb`、`wrist_0_rgb`）并且不 mask 掉填充图。把 FAST 的观测字典送给 `pi05_droid` 会在 `DroidInputs` 里缺键。

`make_droid_example()` 的键：

| 键 | 形状 |
| --- | --- |
| `observation/exterior_image_1_left` | uint8，224×224×3 |
| `observation/wrist_image_left` | 同上 |
| `observation/joint_position` | 7 |
| `observation/gripper_position` | 1，标量会在代码里被扩成一维 |
| `prompt` | 字符串。bytes 会按 utf-8 解码 |

服务器：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

或 `uv run scripts/serve_policy.py --env=DROID`。未验证。

机器人侧不在这个仓库里跑控制环。`examples/droid/README.md` 要求把 `examples/droid/main.py` 拷到 DROID 仓库的 `scripts/`，改相机 ID，安装 `packages/openpi-client`，并指定服务器 IP 和用左还是右外部相机。文档写明策略只吃一个外部相机。

## 微调整个 DROID

`pi05_full_droid_finetune`：`action_horizon=16`，`action_dim=32`，权重 `pi05_base/params`，norm stats 用 `pi05_base/assets` 的 `droid`。数据是 RLDS，`RLDSDroidDataConfig`，动作空间 `DroidActionSpace.JOINT_POSITION`。配置里的 `rlds_data_dir` 是 `/mnt/pi-data/kevin`。这是作者机器上的路径，跑之前要改成你的 DROID RLDS 父目录。`num_workers=0` 是注释里强调的，RLDS loader 自己管进程。100000 step，batch 256。

`examples/droid/README_train.md` 是这条管线的说明，包括 idle filter。那一页还覆盖 π0-FAST 的全量 DROID 训练。π0-FAST 那份配置的注释写了 8 张 H100 大约 2 天。π0.5 这份没有写墙钟时间。

## 微调你自己的小数据集

`pi05_droid_finetune` 从 `pi05_droid/params` 出发，而不是 `pi05_base`。horizon 16。LeRobot repo id 默认是占位的 `your_hf_username/my_droid_dataset`。assets 必须是 `gs://openpi-assets/checkpoints/pi05_droid/assets` 的 `droid`。20000 step，batch 32。转换脚本 `examples/droid/convert_droid_data_to_lerobot.py`。注释说它面向几十小时以内的数据。

`examples/droid/compute_droid_nonidle_ranges.py` 给过滤用。全量训练的 filter 字典通过 `RLDSDroidDataConfig` 的字段接进去，见该配置类上的注释。

## 要改的地方

只换语言指令：观测里的 `prompt`。`pi05_droid` 的 `prompt_from_task=True` 影响的是训练数据，推理时 prompt 来自你发给服务器的字典。

要改 chunk 长度，同时改 `pi05_droid` 的 `action_horizon` 和客户端每次执行的步数。15 是这份已发布权重的训练长度。
