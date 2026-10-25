These labs use the public π0.5 checkpoints and the config fields that actually exist. Commands are copied from the README or scripts and were not run while writing this guide.

每一节都是：假设、要看的文件、命令、要量的东西、可能会坏在哪里。需要 GPU 的命令都未验证。推理按 README 需要 > 8 GB。`pi05_libero` 的全量训练配置不是单卡实验。

## 1. 加载公开 π0.5 并走最短推理

假设：`pi05_libero` 的已发布 checkpoint 能在没有仿真器的情况下返回动作块。

文件：`scripts/serve_policy.py` 的 `DEFAULT_CHECKPOINT`，`examples/simple_client/main.py`。

```bash
uv run scripts/serve_policy.py --env LIBERO
uv run examples/simple_client/main.py --env LIBERO --num-steps 20
```

先按仓库 README 装好环境（`GIT_LFS_SKIP_SMUDGE=1 uv sync` 和 `uv pip install -e .`）。未验证。

量：脚本打出来的表，列是 `policy_infer_ms` 和 `server_infer_ms`。脚本不打印动作。LIBERO 服务器返回的 `actions` 在 `LiberoOutputs` 之后是 `(10, 7)`。要看形状，在 `examples/simple_client/main.py` 的循环里读 `action["actions"].shape`。

会坏在：访问不了 `gs://openpi-assets`；显存不够；两个进程的 `--env` 不一致导致键错误。

DROID 的等价命令是 `--env DROID`，checkpoint `gs://openpi-assets/checkpoints/pi05_droid`，horizon 15，输出 8 维。

## 2. 对照 π0 和 π0.5 的配置字段

假设：`pi05_libero` 和 `pi0_libero` 的差别集中在少数字段，而不是另一个类。

文件：`src/openpi/training/config.py` 里这三个 `TrainConfig`，以及 `Pi0Config.__post_init__`。

在仓库根目录、已经 `uv sync` 的环境里：

```bash
uv run python -c '
from openpi.training import config as c
for name in ["pi0_libero", "pi05_libero", "pi0_fast_libero", "pi05_droid", "pi0_droid"]:
    cfg = c.get_config(name)
    m = cfg.model
    print(name, type(m).__name__, getattr(m, "pi05", None), m.action_dim, m.action_horizon, m.max_token_len, getattr(m, "discrete_state_input", None), m.model_type)
'
```

未验证。`get_config` 是 `config.py` 里的真实函数。

预期你会看到的代码事实：

| name | 类型 | `pi05` | dim | horizon | `max_token_len` | `discrete_state_input` |
| --- | --- | --- | --- | --- | --- | --- |
| `pi0_libero` | `Pi0Config` | `False` | 32 | 50 | 48 | `False` |
| `pi05_libero` | `Pi0Config` | `True` | 32 | 10 | 200 | `False` |
| `pi0_fast_libero` | `Pi0FASTConfig` | 没有该字段 | 7 | 10 | 180 | 没有该字段 |
| `pi05_droid` | `Pi0Config` | `True` | 32 | 15 | 200 | `True` |
| `pi0_droid` | `Pi0Config` | `False` | 32 | 10 | 48 | `False` |

同时看数据配置：`pi0_libero` 的 `extra_delta_transform` 是 `True`，`pi05_libero` 是 `False`。权重路径分别是 `pi0_base` 和 `pi05_base`。

`pi05_libero` 的 `discrete_state_input=False` 意味着这份 LIBERO π0.5 不把状态写进文本，也没有连续状态 token。`pi05_droid` 会写进文本。

## 3. 改 flow 的积分步数

假设：`num_steps` 只影响采样循环的次数，不改变权重大小。步数少，`policy_infer_ms` 下降。

文件：`Pi0.sample_actions` 的参数 `num_steps=10`，`Policy.__init__` 的 `sample_kwargs`。

`serve_policy.py` 没有这个旗标。进程内调用：

```python
policy = policy_config.create_trained_policy(
    config,
    checkpoint_dir,
    sample_kwargs={"num_steps": 5},
)
```

`config` 和 `checkpoint_dir` 用上一节的 `pi05_droid` 或 `pi05_libero`。用同一个 `noise=` 传给 `policy.infer`，形状 `[action_horizon, action_dim]`，否则两次采样的随机噪声会盖住步数的影响。未验证。

量：`policy_timing["infer_ms"]`，以及两段 `actions` 的差的范数。

会坏在：把 `num_steps` 传给 `serve_policy.py` 的 tyro（没有这个字段）；把步数设成 0（`dt = -1/num_steps` 会除零）。

## 4. 改 action horizon

假设：`action_in_proj` 的权重不依赖 horizon，所以已有 checkpoint 仍能 `load`。输出的时间维会变成新的 horizon。质量会变，因为权重是在原 horizon 上训的。

文件：`Pi0Config.action_horizon`，`LiberoOutputs` 只切特征维、不切时间维。

复制一份配置或在调用前替换模型配置里的 `action_horizon`，再 `create_trained_policy`。不要改 `gs://` 上的原文件。未验证。

量：`actions.shape[0]`。LIBERO 公开配置是 10，改成 5 或 50 后第一维应跟着变，第二维仍是 7。

会坏在：LIBERO 客户端的 `replan_steps`（默认 5）大于新的 horizon，`examples/libero/main.py` 的断言会失败。DROID 客户端写死 `assert pred_action_chunk.shape == (10, 8)`，而 `pi05_droid` 的配置 horizon 已经是 15，不改断言就会在这条已发布配置上失败。ALOHA 仿真 broker 默认只执行 10 步，实机默认 25 步，模型可以更长。FAST 的 `max_token_len` 和 horizon 绑在一起，这条实验只对 `Pi0`（含 π0.5）有意义。

## 5. 没有 norm stats 时会怎样

假设：训练在统计缺失时拒绝启动；推理在 `norm_stats.json` 缺失时在进网络之前失败。

文件：`data_loader.transform_dataset`，`normalize.load`，`create_trained_policy`。

训练：对一个还没有 `./assets/<config>/<repo_id>/norm_stats.json` 的 LeRobot 配置直接跑 `scripts/train.py`。`repo_id` 不是 `fake` 时，`transform_dataset` 抛出 `ValueError`，文本里有 `scripts/compute_norm_stats.py`。`debug_pi05` 的 `repo_id` 是 `fake`，故意跳过这件事，不能用来做这个实验。

推理：公开 checkpoint 自带 stats。把本地缓存里该 checkpoint 的 `assets/<asset_id>/norm_stats.json` 临时改名后再 `create_trained_policy`。`normalize.load` 抛出 `FileNotFoundError`，消息是 `Norm stats file not found at: ...`。做完把文件名改回去。未验证。

若绕过加载、让 `Normalize` 拿到 `None`，它会原样放行。π0.5 的分位数把动作约映射到 `[-1, 1]`，tokenizer 的 256 个 bin 也假定这个范围。原始关节角会堆在两端的 bin 上，flow 的目标尺度也会和权重不一致。现象是损失很大或动作饱和，而不是一个专门的异常。

π0 用 z-score，π0.5 用分位数。把 `pi0_base` 的 `norm_stats.json` 配到 `pi05_*` 上，键在的话不会报错，尺度是错的。

## 6. VLM 和 action expert 谁在更新

假设：默认 π0.5 全量配置两边都训练；LoRA 冻结的是 `.*llm.*` 里非 LoRA 的权重，不包含 SigLIP。

文件：`Pi0Config.get_freeze_filter`，`scripts/train.py` 的 `train_step`，`pi0_libero_low_mem_finetune`。

`pi05_libero` 没有设置 `freeze_filter`，就是 `nnx.Nothing`。SigLIP、VLM、action expert、`time_mlp_*` 都在 `trainable_filter` 里。EMA 0.999 会另存一份缓慢平均，推理加载的是 EMA。

`pi0_libero_low_mem_finetune` 把两个 variant 设成 `gemma_2b_lora` 和 `gemma_300m_lora`，`freeze_filter` 用同一份 config 的 `get_freeze_filter()`，`ema_decay=None`，权重是 `pi0_base`。这是仓库里低显存微调的真实例子，README 写 > 22.5 GB。没有 π0.5 的对应名字。要做 π0.5 LoRA，复制这份结构，把 `Pi0Config` 设上 `pi05=True`，weight loader 换成 `pi05_base/params`，并确认 `freeze_filter` 用的是同一份带 `pi05=True` 的 config。未验证能不能和 `pi05_base` 的参数树对上，因为发布的 LoRA 例子只挂在 π0 上。

知识隔离不会因为你冻住 VLM 就出现。离散动作损失不在 `Pi0.compute_loss` 里。冻住 VLM 只是让 flow 梯度不更新那些参数。

量：训练日志里的 `grad_norm`。把 `freeze_filter` 设成 `nnx.All(nnx.Param)` 会让可训练集合为空，优化器那一步会坏，这能反过来确认过滤器是接上的。

## 7. JAX 和 PyTorch 的入口

假设：同一套 `Policy.infer`，后端由 checkpoint 目录里有没有 `model.safetensors` 决定。

文件：`scripts/train.py`，`scripts/train_pytorch.py`，`examples/convert_jax_model_to_pytorch.py`，README 的 PyTorch 一节。

JAX 最短训练循环是 dummy 配置：

```bash
uv run scripts/train.py debug_pi05
```

配置里已经写了 `exp_name` 和 `overwrite=True`。不下载 `pi05_base`。未验证。

PyTorch 的 README 例子是：

```bash
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
```

注意 JAX 文档用 `--exp-name`，PyTorch 例子用 `--exp_name`。PyTorch 在训练前要按 README 把 `transformers_replace` 拷进环境，并确认 `transformers` 是 4.53.2。π0-FAST、LoRA、FSDP、EMA、混合精度不在 PyTorch 训练里。

把 JAX 的 `pi05_libero` 或 `pi05_droid` 转成 PyTorch 再用原来的 `serve_policy.py`，只要 `--policy.dir` 指向含 `model.safetensors` 的目录。转换命令在 README，未验证。

## 8. 新机器人接在哪里

假设：模型槽位不用改，改的是字典键和 `TrainConfig`。

文件：`src/openpi/policies/libero_policy.py`，`LeRobotLiberoDataConfig`，`examples/ur5/README.md`，`docs/norm_stats.md`。

步骤：

1. 写 `Inputs` / `Outputs`。三个图像键保持 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。缺的相机填零，π0.5 把 mask 设为 `False`。
2. 写 `DataConfigFactory`，repack 把数据集键变成推理时的键。绝对关节角加上 `DeltaActions` / `AbsoluteActions`。
3. `TrainConfig.model` 用 `Pi0Config(pi05=True, action_horizon=...)`。`weight_loader` 用 `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`。
4. 动作空间和某个已有 asset 一致时，用 `AssetsConfig(assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets", asset_id=...)`。否则先跑 `compute_norm_stats.py`。π0.5 会走分位数。
5. 服务器：`serve_policy.py policy:checkpoint --policy.config=<name> --policy.dir=<ckpt>`。

UR5 的草稿还不是可导入的类。LIBERO 的 `pi05_libero` 是最完整的 π0.5 微调样板，DROID 的 `pi05_droid_finetune` 是「从专家 checkpoint 接着训、并强制复用原 norm stats」的样板。

会坏在：输出维数切错；`batch_size` 不能被 GPU 数整除；RLDS 配置把 `num_workers` 设成非 0；`pi05_full_droid_finetune` 的 `rlds_data_dir` 仍指向 `/mnt/pi-data/kevin`。
