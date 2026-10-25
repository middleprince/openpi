This page is the LIBERO example: the published π0.5 checkpoint, the data keys, and why this config drops proprioception.

公开 checkpoint 是 `gs://openpi-assets/checkpoints/pi05_libero`。`examples/libero/README.md` 写它由 `pi05_libero` 配置训出，并给出一张表：spatial 98.8、object 98.2、goal 98.0、libero 10 92.4，平均 96.85。复现这张表要评测这个 checkpoint。写指南时没有跑评测，未验证。

## 配置在代码里的样子

`pi05_libero` 的模型是 `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`。数据是 `LeRobotLiberoDataConfig`，`repo_id="physical-intelligence/libero"`，`prompt_from_task=True`，`extra_delta_transform=False`。权重从 `pi05_base/params` 来。batch 256，30000 step，学习率峰值 `5e-5`，EMA 0.999。

`discrete_state_input=False` 加上 `pi05=True`：没有连续状态 token，文本里也没有状态 bin。8 维 LIBERO 状态仍然出现在数据字典里，供输出端使用，但 `Pi0.embed_prefix` 和 `embed_suffix` 都不读它。

`extra_delta_transform=False` 是相对 `pi0_libero` 的另一处差别。LIBERO 动作本身已是增量。旧 π0 checkpoint 多做了一次 `DeltaActions(make_bool_mask(6, -1))`。π0.5 这份不做。

## 键

仿真客户端 `examples/libero/main.py` 不使用 `ActionChunkBroker`。`replan_steps` 默认 5：每次 `infer` 后只执行 `actions` 的前 5 步，然后用新观测再请求。断言要求返回的 chunk 长度不小于 `replan_steps`。`pi05_libero` 的 horizon 是 10，所以默认能过。状态在客户端里由 `robot0_eef_pos`、四元数转成的轴角、`robot0_gripper_qpos` 拼起来。转换脚本把 RLDS 的 `observation.state` 存成形状 `(8,)`，`make_libero_example` 也用长度 8。图像在客户端先 `resize_with_pad` 到 `resize_size`（默认 224）。

`LiberoInputs` 读取：

| 键 | 形状 |
| --- | --- |
| `observation/state` | 8 |
| `observation/image` | 第三人称，HWC 或 CHW |
| `observation/wrist_image` | 腕部 |
| `prompt` | 字符串 |

右腕槽填零，`image_mask["right_wrist_0_rgb"]` 为 `False`。`LiberoOutputs` 返回 `actions` 的前 7 维。`make_libero_example()` 造一条随机输入，键和上面相同。

LeRobot 转换脚本是 `examples/libero/convert_libero_data_to_lerobot.py`。README 说只想微调 LIBERO 时可以不跑它，因为配置指向已经转好的 `physical-intelligence/libero`。自己的数据才需要改这个脚本。命令：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

未验证。原始数据说明在仓库 README，指向 Hugging Face 上的 `openvla/modified_libero_rlds`。

## 训练和服务器

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

这三条都在仓库 README，未验证。batch 256 的全量微调对应 README 里 > 70 GB 那一档，不是单卡 4090 的配置。

已经训好的公开权重不用自己训：

```bash
uv run scripts/serve_policy.py --env LIBERO
```

## 仿真客户端

`examples/libero/README.md` 推荐 Docker：

```bash
sudo xhost +local:docker
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

换任务套件：`CLIENT_ARGS="--args.task-suite-name libero_10"`。换自己的 checkpoint：`SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"`。

不用 Docker 时，客户端是 Python 3.8 的独立 venv，要装 `examples/libero/requirements.txt`、`third_party/libero`，并把 `third_party/libero` 放进 `PYTHONPATH`。这需要子模块。服务器仍用仓库根的 `uv run scripts/serve_policy.py --env LIBERO`。EGL 出问题时文档写 `MUJOCO_GL=glx`。

## 要改的地方

LIBERO 上做结构实验，改 `pi05_libero` 这一份，不要改 `pi0_libero`。最容易误开的两个开关是 `discrete_state_input` 和 `extra_delta_transform`。打开它们等于换了一套输入约定，不能拿 `pi05_libero` 的已发布权重继续评。
