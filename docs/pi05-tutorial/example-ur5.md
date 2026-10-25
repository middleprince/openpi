This page is the UR5 sketch. It is a README, not a module under `src/openpi/policies/`.

`examples/ur5/README.md` 示范怎样接一只 README 总述里说的新机器人。它给出 `UR5Inputs` 和 `UR5Outputs` 的草稿，并说对照 `libero_policy.py` 的注释。仓库里没有 `ur5_policy.py`，也没有名为 `pi05_ur5` 的 `TrainConfig`。

## 草稿在做什么

`UR5Inputs` 把 `joints` 和 `gripper` 拼成 `state`，把 `base_rgb` 和 `wrist_rgb` 放进 `base_0_rgb` 和 `left_wrist_0_rgb`，右腕填零。`image_mask` 的右腕在 `model_type == PI0_FAST` 时为 `True`，否则为 `False`。这和 `LiberoInputs` 的约定相同。默认 `model_type` 是 `PI0`。接到 π0.5 时要传 `ModelType.PI05`，否则 mask 行为碰巧一样（两者都不是 FAST），但你如果复制了 FAST 分支的判断，就会和 `pi05` 配置不一致。

后面的章节让你仿 `LeRobotLiberoDataConfig` 写数据配置，再仿 `pi0_libero` 写 `TrainConfig`。norm stats 的 asset id，`docs/norm_stats.md` 给 UR5e 的是 `ur5e`，双臂是 `ur5e_dual`。那张表指向 `pi0_base` 和 `pi0_fast_base` 的 assets。π0.5 基座在代码里使用的目录是 `gs://openpi-assets/checkpoints/pi05_base/assets`。是否含有 `ur5e` 这份统计，要以你下载下来的目录为准，文档没有为 `pi05_base` 重新列这张表。

动作定义仍是 `docs/norm_stats.md` 那套：6 维关节加 1 维夹爪，文档写 UR5e 的控制频率是 20 Hz，夹爪 0 是开、1 是合。

## 接到 π0.5 时要写的三处

1. 把草稿收成真正的 `DataTransformFn`，放进 `src/openpi/policies/`，`model_type` 用 `ModelType.PI05`。
2. 在 `training/config.py` 增加一个 `DataConfigFactory`，`ModelTransformFactory` 不用改。`use_quantile_norm` 会因为 `model_type != PI0` 自动打开。
3. 增加 `TrainConfig`：`model=Pi0Config(pi05=True, ...)`，`weight_loader` 指向 `gs://openpi-assets/checkpoints/pi05_base/params`。horizon 按你的控制频率和想执行的秒数选。LIBERO 用 10，DROID 推理用 15，都是可以抄的起点，不是 UR5 的测量值。

然后：

```bash
uv run scripts/compute_norm_stats.py --config-name <your_config>
uv run scripts/train.py <your_config> --exp-name=<run> --overwrite
```

未验证。单卡请把 `batch_size` 从 `pi05_libero` 的 256 降下来。tyro 可以覆盖 `TrainConfig` 的字段，README 已经用 `--exp-name` 和 `--overwrite` 这样做。

## 要改的地方

UR5 的关节维数是 6 加夹爪，不是 LIBERO 的 8，也不是 DROID 的 7 加夹爪。`*Outputs` 切回去的维数必须和 `state` 的有效维一致。多出来的到 32 的填充在 `PadStatesAndActions`，输出端不返回它们。
