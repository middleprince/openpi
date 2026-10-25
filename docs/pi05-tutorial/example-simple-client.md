This page is the robot-free client: random observations against a policy server, used to time π0.5 inference.

`examples/simple_client/README.md` 和 `examples/simple_client/main.py`。它不计算任务成功率，只打推理时间。

## 命令

README 的无 Docker 流程，两个终端：

```bash
uv run examples/simple_client/main.py --env DROID
uv run scripts/serve_policy.py --env DROID
```

`--env` 的取值是 `EnvMode`：`ALOHA`、`ALOHA_SIM`、`DROID`、`LIBERO`。客户端默认 `ALOHA_SIM`。要打到 π0.5 的已发布专家，用 `DROID` 或 `LIBERO`。`ALOHA` 会让服务器加载 `pi05_aloha` 配 `pi05_base`。`ALOHA_SIM` 加载的是 π0 的仿真 checkpoint，不是 π0.5。

Docker 例子把 `SERVER_ARGS="--env ALOHA_SIM"` 放进 `examples/simple_client/compose.yml`。那是 π0。

客户端还有 `--host`、`--port`（默认 8000）、`--api-key`、`--num-steps`（默认 20）、`--timing-file`。先空跑 2 次 `infer` 让模型加载完，再记 `num_steps` 次。打印的列来自 `server_timing` 和 `policy_timing`：`client_infer_ms`、`server_infer_ms`、`server_prev_total_ms`、`policy_infer_ms`。`--timing-file` 写成 parquet。

这些命令未验证。第一次运行会把 checkpoint 下到 `~/.cache/openpi`，需要能访问 `gs://openpi-assets`。

## 随机观测长什么样

`_random_observation_libero` 和 `_random_observation_droid` 与 `make_libero_example`、`make_droid_example` 的键一致。ALOHA 的随机观测是 14 维 `state` 和四路 CHW 图像，含 `cam_low`。图像是 `randint` 的 uint8，状态是 `rand` 或全 1。这只能证明服务器能返回形状正确的 `actions`，不能证明动作有意义。

`simple_client` 不检查动作形状。按服务器配置，LIBERO 的 `actions` 在 `LiberoOutputs` 之后是 `(10, 7)`，`pi05_droid` 是 `(15, 8)`，`pi05_aloha` 是 `(50, 14)`。机器人侧的 `examples/droid/main.py` 另有断言要求 `(10, 8)`，和 `pi05_droid` 的 horizon 15 不一致；这个无机器人客户端没有那条断言。

## 要改的地方

测自己的 checkpoint：服务器用 `policy:checkpoint`，客户端 `--env` 必须仍和那份 config 的输入键一致。`pi05_libero` 配 `--env LIBERO`。不要用 DROID 的随机观测去打 LIBERO 服务器，键对不上，`LiberoInputs` 会 `KeyError`。
