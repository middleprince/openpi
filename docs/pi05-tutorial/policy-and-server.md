This page is `Policy.infer` and the websocket server: how a checkpoint becomes an action chunk on port 8000.

两个文件：`src/openpi/policies/policy.py`、`src/openpi/policies/policy_config.py`、`src/openpi/serving/websocket_policy_server.py`、`scripts/serve_policy.py`。

## `create_trained_policy`

参数是 `train_config`、`checkpoint_dir`，以及可选的 `repack_transforms`、`sample_kwargs`、`default_prompt`、`norm_stats`、`pytorch_device`。

`checkpoint_dir` 先经过 `maybe_download`，所以可以写 `gs://openpi-assets/checkpoints/pi05_droid`。目录里有 `model.safetensors` 就 `load_pytorch`，否则 `restore_params(checkpoint_dir / "params", dtype=bfloat16)`。

norm stats 缺省时从 `checkpoint_dir / "assets" / asset_id` 加载。`asset_id` 来自数据配置。没有 asset id 会 `ValueError`。文件不存在会 `FileNotFoundError`。传入的 `norm_stats=None` 表示「请去加载」，不是「跳过归一化」。

返回的 `Policy` 把 `sample_kwargs` 存下来。推理时并进 `sample_actions` 的关键字参数。`num_steps` 从这里进，不从命令行进。

## `Policy.infer`

输入是一条观测字典，还没有 batch 维。函数会拷贝一份再变换。JAX 路径把每个数组变成 `jnp` 并在前面加一维 batch，然后拆一个 rng。PyTorch 路径用 `torch.from_numpy` 加 batch，再放到 `pytorch_device`。

`Observation.from_dict` 之后调用 `sample_actions`。输出字典先含 `state` 和 `actions`，去掉 batch 维，再跑 output transforms。最后加上：

```python
outputs["policy_timing"] = {"infer_ms": model_time * 1000}
```

`noise` 关键字可以传入形状 `[action_horizon, action_dim]` 或 `[1, action_horizon, action_dim]` 的噪声，用来固定采样。这是对比 `num_steps` 时该固定的东西。

`PolicyRecorder` 把每步的输入输出写成 `policy_records/step_*.npy`。`serve_policy.py --record` 会包上一层。

## 服务器

`WebsocketPolicyServer` 绑定 `0.0.0.0` 和给定端口。连接建立后先发一份 msgpack 的 metadata（`TrainConfig.policy_metadata`，ALOHA 配置里有 `reset_pose`）。之后每个二进制包是一条观测，`policy.infer` 的结果再打包发回。结果里追加 `server_timing.infer_ms`，以及上一包的 `prev_total_ms`。异常时把 traceback 字符串发回去再关掉连接。客户端看到字符串会当成错误。

`serve_policy.py` 用 tyro。两种写法都在仓库 README 和 `docs/remote_inference.md` 里：

```bash
uv run scripts/serve_policy.py --env LIBERO
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

`--port` 默认 8000。`--default-prompt` 在观测里没有 `prompt` 时注入。`--env` 只对默认策略有意义，对应 `EnvMode`。`--record` 打开 `PolicyRecorder`。

DROID 文档给出的等价命令是：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
uv run scripts/serve_policy.py --env=DROID
```

这些命令会下载 checkpoint 并加载模型。写指南时没有跑，未验证。README 的推理显存是 > 8 GB。

## 进程内调用

README 的 Python 片段用 `pi05_droid`：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
policy = policy_config.create_trained_policy(config, checkpoint_dir)
action_chunk = policy.infer(example)["actions"]
```

`example` 的键必须是 `DroidInputs` 要的那些，见 [DROID](example-droid.md)。换成 LIBERO 就用 `get_config("pi05_libero")` 和 `gs://openpi-assets/checkpoints/pi05_libero`，键改成 `LiberoInputs` 的键。同一段代码在 `examples/inference.ipynb` 里，README 这样指向它。

## 要改的地方

服务器协议本身不用改。新机器人改的是 config 指向的 `*Inputs`，这样 `infer` 收到的字典能被变换接住。

要在服务端改积分步数，现在的 CLI 做不到。在 `create_policy` 里把 `sample_kwargs={"num_steps": ...}` 传进 `create_trained_policy`，或者在进程内调用时传入。改 `Pi0.sample_actions` 的默认值会同时影响所有调用者。
