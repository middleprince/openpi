This page is the robot-side package `openpi-client`: a websocket policy, an action-chunk broker, and the small runtime loop.

包在 `packages/openpi-client/`。机器人环境只装这个包，不装 JAX。`docs/remote_inference.md` 的安装是：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

写指南时没有执行，未验证。

## `WebsocketClientPolicy`

`websocket_client_policy.py`。构造参数 `host`、`port`、`api_key`。`host` 以 `ws` 开头就当成完整 URI，否则拼成 `ws://{host}:{port}`。默认 host 是 `0.0.0.0`，默认 port 是 `None`（URI 不带端口）。`simple_client` 传入 `port=8000`。

连接时如果给了 `api_key`，加上头 `Authorization: Api-Key ...`。`ConnectionRefusedError` 会每 5 秒重试。连上后第一包必须是 metadata，用 `msgpack_numpy` 解开，`get_server_metadata()` 能读到。

`infer(obs)` 把字典打包发出去。若收回来的是 `str`，当成服务器 traceback，抛 `RuntimeError`。否则解开成字典。里面有 `actions`，以及服务器加上的 `server_timing` 和策略加上的 `policy_timing`。

`reset` 是空操作。换任务时如果外面包了 `ActionChunkBroker`，要调 broker 的 `reset`，否则会把上一个 chunk 的剩余步发完。

## `ActionChunkBroker`

模型一次吐出 `action_horizon` 步。机器人控制环通常一步一条。`ActionChunkBroker(policy, action_horizon)` 第一次 `infer` 向内部 policy 要整块，之后每次只切 `x[cur_step]`。计数到达 `action_horizon` 就丢掉缓存，下一次重新推理。

非数组字段原样返回，所以 `policy_timing` 也会被切索引。它是字典，`slicer` 对非 ndarray 直接返回，不会坏。

这里的 `action_horizon` 是客户端打算消费的步数，不必等于模型吐出的整块长度。用完这么多步就丢掉剩余的，重新 `infer`。设得比返回的第一维还大，会在切片时越界。

三个例子没有共用同一种切块方式：

- `examples/aloha_sim/main.py` 用 `ActionChunkBroker`，默认 `action_horizon=10`。`pi0_aloha_sim` 的模型 horizon 是 50，broker 只用前 10 步。
- `examples/aloha_real/main.py` 同样用 broker，默认 `action_horizon=25`。`pi05_aloha` 的模型 horizon 是 50。
- `examples/libero/main.py` 不用 broker。它自己维护队列，`replan_steps` 默认 5，只把 `actions[:replan_steps]` 放进队列。模型若是 `pi05_libero`，返回的第一维是 10，断言要求它不小于 `replan_steps`。
- `examples/droid/main.py` 也不用 broker。`open_loop_horizon` 默认 8。文件里另有 `assert pred_action_chunk.shape == (10, 8)`。`pi05_droid` 的 `action_horizon` 是 15，按当前配置服务器会返回 15 而不是 10。这条断言和配置不一致，见 [DROID](example-droid.md)。

## 运行时

`runtime/runtime.py` 把 environment、agent、subscriber 串起来。`runtime/agents/policy_agent.py` 是把 `BasePolicy` 装进 agent 接口的薄包装。ALOHA 仿真和实机的 `main.py` 走这套。DROID 的 `examples/droid/main.py` 直接用 `WebsocketClientPolicy`，因为控制环在 DROID 自己的仓库里。

`image_tools.py` 提供客户端侧的 resize。远程推理文档的示例从这里 import。图像在进服务器之前就应是该机器人 `*Inputs` 能解析的形状：LIBERO 和 DROID 的例子是 uint8 HWC，ALOHA 是 uint8 CHW。

`msgpack_numpy.py` 是协议。服务器和客户端必须用同一份，numpy 数组才能过 websocket。

## 要改的地方

协议字段不要改。新机器人在客户端构造观测字典，键名和服务器配置里的 `*Inputs` 一致，然后 `policy.infer(obs)["actions"]`。

控制频率和 chunk 的关系在 broker 的 `action_horizon`。想每步都重新推理，就不要包 broker，每次把最新观测送进去，只用返回 chunk 的第 0 步。这会把服务器的 `infer_ms` 乘到控制频率上。`simple_client` 打印的就是这个时间。
