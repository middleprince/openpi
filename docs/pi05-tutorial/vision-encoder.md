This page is the vision tower actually used by π0 and π0.5: SigLIP So400m/14 with no pooling, projected to the Gemma width. `src/openpi/models/vit.py` is not on this path.

`Pi0.__init__` 构造：

```python
_siglip.Module(
    num_classes=paligemma_config.width,
    variant="So400m/14",
    pool_type="none",
    scan=True,
    dtype_mm=config.dtype,
)
```

`paligemma_config.width` 在 `gemma_2b` 上是 2048。`decode_variant` 把 `"So400m/14"` 解成 SigLIP So400m 的宽度、深度、头数，以及 `patch_size=(14, 14)`。文件里的表：So400m 的 width 是 1152，depth 27，mlp 4304，heads 16。

## 一张图变成多少 token

`_Module.__call__` 用 stride 等于 patch 的卷积切图。224×224、patch 14，得到 `h = w = 16`，reshape 成 `[batch, 256, 1152]`，加上可学习位置编码。`pool_type="none"` 不做 GAP、不加 CLS。`num_classes` 不是分类，而是一个 `Dense(2048)`，把每个 token 投到 Gemma 宽度。返回的 `x` 仍是序列。

三个相机就是 3 × 256 = 768 个视觉 token，每个宽度 2048。某个相机的 `image_mask` 为假时，这 256 个 token 的 `input_mask` 全假，注意力里它们被 `valid_mask` 去掉。

卷积和位置编码在 float32 里做（源码注释写 “Kevin edit”），再 cast 到 `dtype_mm`。默认 `Pi0Config.dtype` 是 `"bfloat16"`。

## 训练时谁在更新

`embed_prefix` 调用 `self.PaliGemma.img(..., train=False)`。SigLIP 的 dropout 在训练 step 里也是关的。图像增强在 `preprocess_observation`，在进编码器之前。

SigLIP 参数挂在 `PaliGemma.img`，路径里没有 `llm`。`Pi0Config.get_freeze_filter` 的正则是 `.*llm.*`。LoRA 微调冻结 Gemma 时，SigLIP 和 `action_in_proj` / `time_mlp_*` / `action_out_proj` 仍然可训练。全量微调（`freeze_filter` 默认 `nnx.Nothing`）下 SigLIP 和两个专家一起更新。

## `vit.py`

`src/openpi/models/vit.py` 开头写明它改编自 `google-research/vision_transformer` 的 `vit_jax`。仓库里的 π0、π0.5、π0-FAST 前向不 import 它。改视觉编码器时改 `siglip.py` 和 `Pi0.__init__` 里的 `variant` / `pool_type`。PyTorch 侧的对应实现在 `models_pytorch/transformers_replace/models/siglip/`，要按 README 覆盖进已安装的 `transformers`。

## 要改的地方

分辨率：`IMAGE_RESOLUTION` 和 SigLIP patch 必须整除。改 224 而不改 patch，token 数会变，位置编码的形状也变，已有 checkpoint 对不上。

少一个相机：保持三个槽，把缺的槽 mask 设为 `False`。把槽从 `IMAGE_KEYS` 删掉会让 `preprocess_observation` 报 `images dict missing keys`。
