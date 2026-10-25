This page is the ALOHA path: real robot, simulator, and the π0.5 configs that reuse the 14-dimensional dual-arm state.

实机说明在 `examples/aloha_real/README.md`，仿真在 `examples/aloha_sim/README.md`。模型侧是 `src/openpi/policies/aloha_policy.py` 和 `LeRobotAlohaDataConfig`。

## 输入

`AlohaInputs` 期望 `state` 形状 `(14,)`，`images` 是 CHW uint8。相机名必须落在 `cam_high`、`cam_low`、`cam_left_wrist`、`cam_right_wrist`。进模型的是 high → `base_0_rgb`，左右腕进两个 wrist 槽。`cam_low` 不进三个槽。缺腕相机时填零且 mask 为 `False`。

`adapt_to_pi=True`（默认）会把标准 ALOHA 关节和夹爪换到 π 预训练时的坐标系。`_joint_flip_mask` 对部分关节乘 −1。夹爪在线位移和角度之间转换。`AlohaOutputs` 取前 14 维再变回去。关掉 `adapt_to_pi` 只在你的数据和 π 的坐标系已经一致时做，`pi05_base` 的 `trossen` 统计是按打开这个开关的空间算的。

默认 `use_delta_joint_actions=True`，mask `make_bool_mask(6, -1, 6, -1)`。仿真配置 `pi0_aloha_sim` 把它关掉。没有名为 `pi05_aloha_sim` 的配置。`serve_policy` 的 `ALOHA_SIM` 加载的是 π0 的 `pi0_aloha_sim` checkpoint。

## π0.5 配置

`pi05_aloha`：`Pi0Config(pi05=True)`，horizon 50，asset id `trossen`，没有 weight loader。`serve_policy --env ALOHA` 用这个配置加上 `gs://openpi-assets/checkpoints/pi05_base`。这是基座，不是在你的桌上微调过的专家。`policy_metadata["reset_pose"]` 是 `[0, -1.5, 1.5, 0, 0, 0]`。

`pi05_aloha_pen_uncap`：从 `pi05_base/params` 微调，数据 repo `physical-intelligence/aloha_pen_uncap_diverse`，norm stats 复用 `pi05_base/assets` 的 `trossen`，默认 prompt `uncap the pen`，20000 step，batch 64。对应的 π0 配置是 `pi0_aloha_pen_uncap`，加载的是 `pi0_base`。

毛巾和保鲜盒只有 π0 配置（`pi0_aloha_towel`、`pi0_aloha_tupperware`），checkpoint 在 README 的表里。没有 π0.5 版本的这两份专家。

## 命令

实机 Docker（`examples/aloha_real/README.md`）：

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

不用 Docker 时，机器人进程用 Python 3.10 的 venv，只装 `openpi-client`：`python -m examples.aloha_real.main`。另外要 `roslaunch` ALOHA 节点。相机序列号改在子模块 `third_party/aloha` 里，所以要初始化子模块。

仿真的 README 使用 `examples/aloha_sim` 的 compose，服务器命令是 `uv run scripts/serve_policy.py --env ALOHA_SIM`。那是 π0 的 `pi0_aloha_sim`。客户端 `examples/aloha_sim/main.py` 用 `ActionChunkBroker`，默认 `action_horizon=10`，模型配置的 horizon 是 50，所以每 10 步丢掉剩余动作再请求。实机 `examples/aloha_real/main.py` 的 broker 默认是 25，`pi05_aloha` 的模型 horizon 是 50。要在仿真里看 π0.5，需要自己把 `SERVER_ARGS` 指到 `pi05_aloha` 和一份 π0.5 权重，并确认观测键和 `AlohaInputs` 一致。仓库没有给这条组合单独的已验证说明。

## 要改的地方

自己的 ALOHA 数据：复制 `pi05_aloha_pen_uncap`，改 `repo_id` 和 `default_prompt`。若仍是 Trossen 双臂且 `adapt_to_pi=True`，可以继续用 `pi05_base` 的 `trossen` 统计；动作定义变了就重新 `compute_norm_stats.py`。转换脚本是 `examples/aloha_real/convert_aloha_data_to_lerobot.py`。

`cam_low` 要进模型的话，得替换三个槽里的一个，或改 `IMAGE_KEYS`。后者会动到所有平台。
