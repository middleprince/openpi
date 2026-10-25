This page defines the tensor contract: three image slots, a state vector, and an action chunk of shape `[batch, action_horizon, action_dim]`.

模型不接受各机器人自己的键名。数据变换负责把它们收成 `Observation`。定义在 `src/openpi/models/model.py`。

## `Observation`

`Observation` 是 Flax struct。字段：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `images` | 每个值 `[*b, h, w, 3]` | float32，范围 `[-1, 1]`，或进入 `from_dict` 之前的 uint8 |
| `image_masks` | 每个值 `[*b]` | `True` 表示这个相机有效 |
| `state` | `[*b, s]` | 低维本体感觉。`s` 在 pad 之后等于 `action_dim` |
| `tokenized_prompt` | `[*b, l]` | int32 token，可空 |
| `tokenized_prompt_mask` | `[*b, l]` | 非 padding 为 `True` |
| `token_ar_mask` | `[*b, l]` | 只有 π0-FAST 用 |
| `token_loss_mask` | `[*b, l]` | 只有 π0-FAST 用 |

`Observation.from_dict` 读的字典键是 `image`、`image_mask`、`state`，不是 struct 上的 `images`。uint8 图像会变成 `x / 255 * 2 - 1`。PyTorch uint8 还会从 NHWC 转成 NCHW，这是 `from_dict` 里单独的分支。

`Actions` 的别名是 `Float[*b, ah, ad]`，字典键是 `actions`。

## 三个相机槽

`IMAGE_KEYS` 固定为 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。`IMAGE_RESOLUTION` 是 `(224, 224)`。注释写着如果以后发布小模型，分辨率可能改。当前 SigLIP 用 patch 14，224 / 14 = 16，每张图 256 个 token。

缺相机时不要删键。`LiberoInputs` 和 `DroidInputs` 把不存在的右腕填成零图像，并把 `image_mask` 设为 `False`（π0-FAST 例外，padding 图像的 mask 仍是 `True`）。`preprocess_observation` 里，mask 缺省时填全 `True`。掩码进 `make_attn_mask` 之后，无效相机的 token 既不能被看，也不能去看别人。

`AlohaInputs.EXPECTED_CAMERAS` 含 `cam_low`，但映射进模型的只有 `cam_high` → `base_0_rgb`、`cam_left_wrist`、`cam_right_wrist`。`cam_low` 会被接受，然后丢掉。

## 动作维度

`Pi0Config` 的默认值是 `action_dim=32`、`action_horizon=50`。公开配置改 horizon，很少改 dim：

| config | `pi05` | `action_horizon` | `action_dim` | `discrete_state_input` |
| --- | --- | --- | --- | --- |
| `Pi0Config()`（`pi0_libero` 等） | `False` | 50 | 32 | `False` |
| `pi05_libero` | `True` | 10 | 32 | `False`（显式） |
| `pi05_droid` | `True` | 15 | 32 | 默认跟着 `pi05`，即 `True` |
| `pi05_full_droid_finetune` | `True` | 16 | 32 | `True` |
| `pi05_droid_finetune` | `True` | 16 | 32 | `True` |
| `pi05_aloha` / `pi05_aloha_pen_uncap` | `True` | 50 | 32 | `True` |
| `pi0_fast_libero` | FAST 配置 | 10 | 7 | FAST 自己离散化状态 |
| `pi0_fast_droid` | FAST 配置 | 10 | 8 | 同上 |

`PadStatesAndActions` 把状态和动作的最后一维零填充到 `action_dim`。输出端再切回来：`LiberoOutputs` 取前 7 维，`DroidOutputs` 取前 8 维，`AlohaOutputs` 取前 14 维。

`action_horizon` 不出现在 `Linear` 的权重形状里。`action_in_proj` 是 `Linear(action_dim, expert_width)`。因此换 horizon 不会让 checkpoint 的 shape check 失败。模型仍是按训练时的 horizon 学的，换短 chunk 是行为变化，不是仓库文档里的开关。实验页写了怎么改。

## `preprocess_observation`

`src/openpi/models/model.py` 的 `preprocess_observation`：

- 图像不是 224×224 就 `resize_with_pad`。
- `train=True` 时，非 wrist 相机做 95% 随机裁剪、resize 回去、`Rotate((-5, 5))`，然后所有相机做 `ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)`。数值在 `[-1, 1]` 和 `[0, 1]` 之间转一圈，因为 augmax 吃 `[0, 1]`。
- wrist 相机只有颜色抖动，没有几何增强。

`Pi0.embed_prefix` 调用 SigLIP 时写死 `train=False`。训练时的图像随机性来自上面的增强，不来自 SigLIP dropout。

## 要改的地方

新相机布局：改对应的 `*Inputs.__call__`，保持三个槽的键名。不要改 `IMAGE_KEYS`，除非你准备同时改 `Pi0Config.inputs_spec` 和 PyTorch 的 `IMAGE_KEYS`。

本体感觉的语义在 `docs/norm_stats.md`。双臂关节在前，夹爪单独一维；DROID 的动作是 7 维关节速度加 1 维夹爪，控制频率文档写的是 15 Hz。π0.5 微调如果复用 `pi05_base` 或 `pi05_droid` 的 norm stats，动作定义必须和那份统计一致。
