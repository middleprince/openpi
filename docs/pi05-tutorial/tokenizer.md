This page covers the text tokenizers: PaliGemma sentencepiece for π0 and π0.5, FAST tokens for π0-FAST, and the RoboArena binning and FSQ tokenizers that π0.5 does not use.

实现在 `src/openpi/models/tokenizer.py`。调用点是 `src/openpi/transforms.py` 的 `TokenizePrompt` 和 `TokenizeFASTInputs`。

## PaliGemma tokenizer

`PaligemmaTokenizer` 从 `gs://big_vision/paligemma_tokenizer.model` 加载 SentencePiece。`download.maybe_download` 使用匿名 GCS token。

`tokenize(prompt, state=None)` 先把 `_` 和换行换成空格。

`state is None` 时是 π0：`encode(prompt, add_bos=True)` 再接一个单独 encode 出来的 `"\n"`。注释把它叫做 answer 的起始 token。

`state` 不是 `None` 时是 π0.5 的离散状态。状态先被假定已经落在 `[-1, 1]`，再用 256 个 bin：

```python
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
```

然后拼成一句：

```text
Task: {cleaned_text}, State: {state_str};
Action: 
```

`state_str` 是 bin 编号用空格连起来。整句一次 `encode(..., add_bos=True)`。

序列短于 `max_len` 就在后面补 `False`（token 位置填的也是 `False`，因为 mask 会把它们挡住）。长于 `max_len` 就截断并打 warning，提示加大 `max_token_len`。

`Pi0Config.__post_init__` 在你没写 `max_token_len` 时：π0.5 用 200，π0 用 48。π0.5 更长，是因为状态 bin 的数字会占掉很多 token。200 仍可能不够，warning 是运行时才会看到的信号。

## `discrete_state_input` 决定 state 进不进 tokenizer

`TokenizePrompt` 只有 `discrete_state_input=True` 才把 `data["state"]` 传进 tokenizer，否则传 `None`。这个布尔值不进 `Pi0.compute_loss` 或 `sample_actions`，只被 `ModelTransformFactory` 读取。

`Pi0Config` 里如果没设，`discrete_state_input` 等于 `pi05`。所以 `pi05_droid`、`pi05_aloha`、`pi05_full_droid_finetune` 会把状态写进文本。

`pi05_libero` 写成 `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)`。文本里没有状态。同时 `pi05=True` 让 `embed_suffix` 不再建连续状态 token。这条配置下，网络前向不消费本体感觉。状态张量仍然会被归一化和 pad，只是没有模块去读它。这是配置事实，仓库没有另外解释原因。

## π0-FAST

`FASTTokenizer` 用同一个 PaliGemma 词表，再加上 Hugging Face 上的 `physical-intelligence/fast`（`trust_remote_code=True`）。

前缀是 `Task: {text}, State: {bins};\n`，双向注意力。动作用 FAST tokenizer 编码后映射到词表尾部：跳过最后 128 个特殊 token，`vocab_size - 1 - 128 - token`。后缀是 `"Action: "`、这些 token、再加 `"|"` 和 EOS。`token_ar_mask` 在前缀是 0、后缀是 1。`token_loss_mask` 只在后缀为真。

`extract_actions` 从解码文本里切 `Action: ` 和 `|` 之间的 token，再 `decode(..., time_horizon=, action_dim=)`。找不到 `Action: ` 就返回全零，形状 `(action_horizon, action_dim)`。

π0.5 不走这个类。仓库 README 写明 π0.5 只有 flow matching 头。

## Binning 和 FSQ

`BinningTokenizer` 和文件后部的 FSQ tokenizer 注释写着给 RoboArena 基线用，不给 pi0-style 模型用。`BinningTokenizer.tokenize` 在 `actions is not None` 时直接 `NotImplementedError`。配置出现在 `training/misc/roboarena_config.py` 和 `polaris_config.py`。改 π0.5 时不用动它们。

## 要改的地方

状态离散化的 bin 数和文本模板在 `PaligemmaTokenizer.tokenize` 的 `state is not None` 分支。改模板就必须和已有 checkpoint 一起改，否则已发布权重看的是现在这句 `Task: ... State: ...; Action:`。

只想让某个 π0.5 微调吃或不吃离散状态：改那个 `TrainConfig` 的 `discrete_state_input`，不要改 tokenizer 的默认参数。`max_token_len` 和状态维度一起涨，截断发生在动作提示 `"Action: "` 被切掉的时候，模型仍会跑，只是 prompt 不完整。
