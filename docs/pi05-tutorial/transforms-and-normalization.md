This page is the transform pipeline that runs before `Observation.from_dict`: repack, robot frames, normalization, tokenization, and padding.

变换都是 `DataTransformFn`，实现在 `src/openpi/transforms.py`。训练和推理共用 robot 变换和归一化；repack 只挂在数据集上。`policy_config.create_trained_policy` 把推理链写成：

```
repack.inputs
InjectDefaultPrompt
data_transforms.inputs
Normalize
model_transforms.inputs
→ model
model_transforms.outputs
Unnormalize
data_transforms.outputs
repack.outputs
```

训练的 `transform_dataset` 只有输入半边，顺序与上面的前半相同。

## 每一层做什么

`RepackTransform` 按扁平路径改键。LIBERO 的 LeRobot 数据把 `image` 映到 `observation/image`，这样后面的 `LiberoInputs` 在训练和仿真里看到同一套键。注释写在 `LeRobotLiberoDataConfig.create`。

`InjectDefaultPrompt` 只在数据里没有 `prompt` 时写入默认句子。ALOHA 的毛巾、保鲜盒、开笔配置靠它。LIBERO 和 DROID 微调把 `prompt_from_task=True`，prompt 来自 `PromptFromLeRobotTask` 读到的 `task_index`。

Robot `*Inputs` 产出 `state`、`image`、`image_mask`、可选的 `actions` 和 `prompt`。这一步还在归一化之前，所以 delta 用的是原始单位。

`DeltaActions` 对 mask 为真的维度做 `actions -= state`。`AbsoluteActions` 在输出端加回去。ALOHA 默认 `use_delta_joint_actions=True`，mask 是 `make_bool_mask(6, -1, 6, -1)`：两条臂各 6 个关节做增量，夹爪保持绝对。LIBERO 的原始动作已经是增量。`pi0_libero` 仍设 `extra_delta_transform=True`，注释说这是为了旧的 π0 checkpoint。`pi05_libero` 设 `extra_delta_transform=False`，不再套一层 delta。

`Normalize` / `Unnormalize`：

- `use_quantiles=False`：`(x - mean) / (std + 1e-6)`。
- `use_quantiles=True`：`(x - q01) / (q99 - q01 + 1e-6) * 2 - 1`，大约落到 `[-1, 1]`。反变换只对有统计的前几维做，多出来的填充维原样返回。

`DataConfigFactory.create_base_config` 里：

```python
use_quantile_norm=model_config.model_type != ModelType.PI0
```

π0 用 z-score。π0.5 和 π0-FAST 用分位数。离散状态的 256 个 bin 假定归一化之后落在 `[-1, 1]`，所以 π0.5 和 FAST 必须走分位数这条，不能沿用 π0 的 mean/std 公式。

`norm_stats is None` 时 `Normalize` 直接返回原数据。训练不允许走到这里：`transform_dataset` 在 `repo_id != "fake"` 且没有 stats 时抛出 `ValueError`，提示先跑 `scripts/compute_norm_stats.py`。

`ResizeImages(224, 224)` 和 `TokenizePrompt` 在归一化之后。`PadStatesAndActions` 把最后一维补到 `action_dim`（默认 32）。

## `norm_stats.json`

`scripts/compute_norm_stats.py` 的 `main(config_name, max_frames=None)` 只统计键 `state` 和 `actions`。它先做 repack 和 robot 变换，不做 `Normalize`，也不做 tokenize。结果写到 `config.assets_dirs / data_config.repo_id`，文件名是 `norm_stats.json`（`normalize.save`）。`assets_dirs` 是 `./assets/<config name>`。

`NormStats` 有 `mean`、`std`、`q01`、`q99`。`RunningStats` 用直方图估分位数。

推理时 `create_trained_policy` 不读训练配置里的 assets 目录，而读 checkpoint 里的 `assets/<asset_id>/norm_stats.json`，避免和训练时的统计不一致。文件不存在时 `normalize.load` 抛出 `FileNotFoundError`，消息是 `Norm stats file not found at: ...`。

复用预训练统计的写法在 `docs/norm_stats.md`，代码例子是 `pi05_aloha_pen_uncap`：

```python
assets=AssetsConfig(
    assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
    asset_id="trossen",
)
```

`docs/norm_stats.md` 的表是按 `pi0_base` 和 `pi0_fast_base` 写的。π0.5 的配置在代码里指向 `pi05_base/assets` 和 `pi05_droid/assets`，asset id 仍是 `trossen` 或 `droid`。那份文档没有把 `pi05_base` 列进表里。

仓库 README 的故障表写过：某些很少用到的维度 `q01`、`q99` 或 `std` 会非常小，归一化后的状态和动作会爆，loss 发散。可以手工改 `norm_stats.json`。

## 要改的地方

新数据的键只在 `RepackTransform` 和 `*Inputs` 里改。`ModelTransformFactory` 的注释说换数据集时不用改 tokenize 和 pad。

动作是绝对关节角时，在 data transforms 上 `push` 一对 `DeltaActions` / `AbsoluteActions`，mask 用 `make_bool_mask`：正数表示增量维的个数，负数表示跳过（夹爪）。LIBERO 的 π0.5 不要再打开 `extra_delta_transform`，除非你的数据和 `pi05_libero` 那个 checkpoint 的约定不同。
