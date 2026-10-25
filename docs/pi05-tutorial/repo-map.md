This page is the directory map and the call graph from `scripts/train.py` and `scripts/serve_policy.py` to the websocket client.

仓库根目录的工作可以按调用方向记。子模块 `third_party/aloha` 和 `third_party/libero` 是上游机器人代码，π0.5 的模型不在里面。只有跑 ALOHA 实机或 LIBERO 仿真时才需要 `git submodule update --init --recursive`（仓库 `README.md` 和 `examples/libero/README.md`）。

## 目录

| 路径 | 职责 |
| --- | --- |
| `src/openpi/models/pi0.py` | π0 与 π0.5 的前向、flow 损失、采样。类名是 `Pi0`。 |
| `src/openpi/models/pi0_config.py` | `Pi0Config`。`pi05` 和 `discrete_state_input` 在这里。 |
| `src/openpi/models/pi0_fast.py` | π0-FAST。自回归，不用 flow。 |
| `src/openpi/models/model.py` | `Observation`、`Actions`、`preprocess_observation`、`restore_params`。 |
| `src/openpi/models/gemma.py` | 双专家 Gemma，含 adaRMS。 |
| `src/openpi/models/siglip.py` | SigLIP 视觉编码器。 |
| `src/openpi/models/tokenizer.py` | PaliGemma / FAST / binning / FSQ tokenizer。 |
| `src/openpi/models/vit.py` | 从 vision_transformer 改来的 ViT。π0 / π0.5 的前向不引用它。 |
| `src/openpi/models_pytorch/` | π0 与 π0.5 的 PyTorch 端口，以及要覆盖进 `transformers` 的补丁。 |
| `src/openpi/transforms.py` | 归一化、delta、tokenize、pad。 |
| `src/openpi/training/config.py` | 全部 `TrainConfig`，包括 `pi05_libero`、`pi05_droid`。 |
| `src/openpi/training/data_loader.py` | LeRobot 与 RLDS，以及变换顺序。 |
| `src/openpi/training/checkpoints.py` | Orbax checkpoint。推理用的 `params` 在开了 EMA 时是 EMA 权重。 |
| `src/openpi/policies/` | `Policy`、`create_trained_policy`，以及 LIBERO / ALOHA / DROID 的输入输出。 |
| `src/openpi/serving/websocket_policy_server.py` | websocket 服务。 |
| `packages/openpi-client/` | 机器人侧的轻依赖客户端。 |
| `scripts/train.py` | JAX 训练入口。 |
| `scripts/train_pytorch.py` | PyTorch 训练入口。 |
| `scripts/serve_policy.py` | 策略服务器。 |
| `scripts/compute_norm_stats.py` | 写 `norm_stats.json`。 |
| `examples/` | LIBERO、ALOHA、DROID、UR5 草稿、`simple_client`、JAX→PyTorch 转换。 |

`src/openpi/training/misc/roboarena_config.py` 和 `polaris_config.py` 被 `config.py` 展开进 `_CONFIGS`。它们是基准配置，不是 π0.5 主路径。

## 训练调用

`scripts/train.py` 的 `main` 调用 `_config.cli()`，也就是 tyro 在 `_CONFIGS` 里选一个 `TrainConfig`。

```
main
  create_data_loader
    repack → robot inputs → Normalize → model inputs
    Observation.from_dict
  init_train_state
    Pi0Config.create → Pi0
    CheckpointWeightLoader
    freeze_filter 把冻结参数转成 bfloat16
  train_step
    Pi0.compute_loss
    只对 trainable_filter 里的参数做 value_and_grad
  checkpoints.save_state
```

数据键在 `data_loader.transform_dataset` 里按这个顺序变：`repack_transforms`、`data_transforms`、`Normalize`、`model_transforms`。`ModelTransformFactory` 按 `model_type` 分支到 π0、π0.5 或 π0-FAST。

## 推理调用

```
scripts/serve_policy.py
  create_trained_policy
    目录里有 model.safetensors 则走 PyTorch，否则 restore_params
    同一套 input / output transforms
  WebsocketPolicyServer
    先发 metadata，再循环 infer
packages/openpi-client WebsocketClientPolicy
  msgpack 观测进去，动作字典出来
examples/*/main.py 或 simple_client
```

`serve_policy.py` 的默认 checkpoint 已经是 π0.5，除了仿真 ALOHA：

| `EnvMode` | config | checkpoint |
| --- | --- | --- |
| `ALOHA` | `pi05_aloha` | `gs://openpi-assets/checkpoints/pi05_base` |
| `ALOHA_SIM` | `pi0_aloha_sim` | `gs://openpi-assets/checkpoints/pi0_aloha_sim` |
| `DROID` | `pi05_droid` | `gs://openpi-assets/checkpoints/pi05_droid` |
| `LIBERO` | `pi05_libero` | `gs://openpi-assets/checkpoints/pi05_libero` |

## 你会改的少数位置

新机器人：仿 `LiberoInputs` / `LiberoOutputs`，仿 `LeRobotLiberoDataConfig`，再在 `_CONFIGS` 里加一个 `TrainConfig`。UR5 的草稿在 `examples/ur5/README.md`，没有对应的 `ur5_policy.py`。

改 π0.5 结构：`Pi0.embed_suffix`、`Pi0.compute_loss`、`Pi0.sample_actions`，以及 `gemma.RMSNorm` 的 adaRMS 分支。PyTorch 侧是 `src/openpi/models_pytorch/pi0_pytorch.py` 里的同名方法。改 JAX 时要一起看 PyTorch，两边是分别实现的。
