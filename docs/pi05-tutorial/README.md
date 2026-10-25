This guide walks through the π0.5 code path in this repository: data, preprocessing, the VLM, the action expert, flow matching, training, the policy server, and the robot client. The prose is Chinese. Paths, symbols, and commands stay in English.

本指南按仓库里的真实代码写，目标是让你能改实验，而不是复述论文。每一页先用一段英文说明这一页讲什么，后面的中文对着文件和符号。仓库在 2026-08 之后仍以这套结构为准；若记忆和文件不一致，以文件为准。论文或博客和代码不一致的地方会直接写出来。

这里的命令都来自仓库的 `README.md` 或脚本。写这份指南时没有在 GPU 上跑过训练或推理，所以凡是要启动进程的命令都标了「未验证」。

## 先分清两件事

公开的 `pi05_base` 是在 1 万小时以上的机器人数据上预训练的基座（见仓库 `README.md` 的 Model Checkpoints）。这个仓库没有预训练数据混合，也没有一条命令能把 `pi05_base` 从头训出来。能在这里做的是：加载已发布的 checkpoint、在单卡上做推理，以及按 `TrainConfig` 做微调。

单卡能做的事和完整微调不是同一档：

| 用途 | 仓库给出的显存 | 例子 |
| --- | --- | --- |
| 推理 | > 8 GB | RTX 4090 |
| LoRA 微调 | > 22.5 GB | RTX 4090 |
| 全量微调 | > 70 GB | A100 80GB / H100 |

`pi05_libero` 的 `batch_size` 是 256、`num_train_steps` 是 30000。这不是一个下午、一张消费级卡的配置。一个下午能做的是 `debug_pi05`（假数据、dummy 宽度）、`simple_client` 打到已发布的 π0.5 checkpoint，以及把 `batch_size` 改小之后的短程微调。LoRA 的现成例子是 `pi0_libero_low_mem_finetune` 和 `pi0_fast_libero_low_mem_finetune`，`_CONFIGS` 里没有 π0.5 的 LoRA 配置。

仓库 `README.md` 写明：π0.5 在这个仓库里只支持 flow matching 头，训练和推理都是。π0.5 博客里的 knowledge insulation（VLM 上的离散动作目标，加上动作专家的 flow 损失，并且把 flow 梯度挡住）没有实现在 `Pi0.compute_loss` 里。

## 自上而下

一次推理或一个训练 step 的数据是这样走的：

1. 机器人或 LeRobot / RLDS 样本给出图像、关节状态、语言。键名因平台而异。
2. `RepackTransform` 只在训练时把数据集键改成推理时用的键。`LiberoInputs`、`AlohaInputs`、`DroidInputs` 把它们收成三个相机槽：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。
3. `Normalize` 按 `norm_stats.json` 处理 `state` 和 `actions`。π0 用 z-score；π0.5 和 π0-FAST 用分位数，映射到大约 `[-1, 1]`。
4. `PaligemmaTokenizer` 把 prompt 变成 token。`pi05=True` 且 `discrete_state_input=True` 时，状态被离散化后写进同一段文本。`pi05_libero` 显式把 `discrete_state_input` 设为 `False`，这段文本里没有状态。
5. SigLIP（`So400m/14`，`pool_type="none"`）把每张 224×224 图像变成 256 个视觉 token，再线性投到 Gemma 2B 的宽度 2048。
6. 两个 Gemma 专家共享注意力：PaliGemma（宽度 2048）读图像和语言；action expert（宽度 1024）读动作 chunk。π0.5 用 adaRMS 注入 flow 的时间；π0 把时间向量和动作 token 拼在一起过 MLP，并多一个连续状态 token。
7. 训练时 `compute_loss` 回归 flow 的目标速度。推理时 `sample_actions` 从噪声积分 `num_steps` 步（默认 10），得到形状 `[action_horizon, action_dim]` 的动作。默认 `action_dim` 是 32，平台输出再切回 7（LIBERO）、8（DROID）或 14（ALOHA）。
8. `Policy.infer` 做输入变换、采样、反变换。`WebsocketPolicyServer` 用 msgpack 把观测收进来、把动作送出去。机器人进程只依赖 `openpi-client`。

π0-FAST 不走这条 flow。它用 `FASTTokenizer` 把动作压成 PaliGemma 词表末尾的 token，损失是下一个 token 的交叉熵。π0.5 的类仍然是 `Pi0`，用 `Pi0Config.pi05` 切换，没有单独的 `Pi05` 类。

## 怎么读

先读本页和 [仓库地图](repo-map.md)，再读 [π0.5 模型](pi05-model.md) 和 [flow matching](flow-matching.md)。要改机器人，读 [观测与动作](observation-and-actions.md)、[变换与归一化](transforms-and-normalization.md) 和对应的 example。要跑起来，直接看 [实验](experiments.md)。

## 页面

- [仓库地图](repo-map.md)：目录职责，以及 train / serve / client 的调用图。
- [观测与动作](observation-and-actions.md)：`Observation`、三个相机槽、`action_dim` 与 `action_horizon`。
- [变换与归一化](transforms-and-normalization.md)：repack、delta、分位数、`norm_stats.json`。
- [Tokenizer](tokenizer.md)：π0、π0.5 的离散状态、π0-FAST，以及 RoboArena 用的 binning / FSQ。
- [视觉编码器](vision-encoder.md)：真正用上的 SigLIP。`vit.py` 不在 π0.5 的前向里。
- [Gemma 骨干](gemma-backbone.md)：双专家、共享注意力、adaRMS、LoRA 冻结范围。
- [π0.5 模型](pi05-model.md)：和 π0、π0-FAST 的差别，注意力掩码，动作 chunk。
- [Flow matching](flow-matching.md)：损失、采样，以及和 π0 论文时间方向相反的约定。
- [PyTorch 端口](pytorch-port.md)：`PI0Pytorch`、transformers 补丁、不支持的功能。
- [训练循环](training-loop.md)：JAX `scripts/train.py` 和一个 step 里谁拿到梯度。
- [配置与 checkpoint](configs-and-checkpoints.md)：`TrainConfig` 名字、公开权重、EMA。
- [Policy 与 websocket](policy-and-server.md)：`create_trained_policy` 和 `serve_policy.py`。
- [客户端](client.md)：`openpi-client`、`ActionChunkBroker`。
- [LIBERO](example-libero.md)、[ALOHA](example-aloha.md)、[DROID](example-droid.md)、[UR5](example-ur5.md)、[无机器人客户端](example-simple-client.md)。
- [实验](experiments.md)：加载公开 π0.5 checkpoint、对比配置、改步数和 horizon、去掉 norm stats、冻结、JAX 与 PyTorch、接入新机器人。
