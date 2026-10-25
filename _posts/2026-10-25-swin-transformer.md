---
layout: post
title: "Swin Transformer: A Source Tour of Microsoft's Hierarchical Vision Backbone"
description: "Walk through the Swin source: shifted-window self-attention with relative position bias, patch merging downsampling, checkpoint weight remapping, and the training entry."
date: 2026-10-25
header-img: "img/post-bg.jpg"
permalink: /swin-transformer/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/swin-transformer/microsoft-swin-transformer-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: Swin Transformer, vision transformer, shifted windows, hierarchical backbone, image classification
author: "PyShine"
---

When the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) landed in 2021, most vision transformers were still computing attention over every pair of tokens in the image — quadratic cost, single-scale feature maps, and an awkward fit for dense prediction tasks that expect pyramid-shaped backbones. Microsoft's answer, written by Ze Liu and collaborators, was to restrict self-attention to small local windows and then shift those windows by half their size every other block, so information still flows across the whole image. The result became the default backbone for a generation of detection and segmentation systems, including the [Detectron2](/detectron2/) ecosystem, and its hierarchical four-stage design still shows up in modern architectures.

What makes this repository especially worth reading is that the ideas map onto the code with almost no abstraction layer. There is one file, `models/swin_transformer.py`, that contains the entire model: patch embedding, windowed attention, shifted-window masking, patch merging, and the classification head. Around it sit a thin yacs-based config system, a data loading module built on [PyTorch](/pytorch/) and timm primitives, and a distributed training entry that was clearly written to run on 8-GPU ImageNet boxes. Reading it top to bottom teaches you not just Swin, but how production vision training repos were structured in that era.

In this tour we walk the source the way a maintainer would: the model file first, then the shifted-window mechanics, the downsampling stage, the checkpoint loading quirks, and finally the training entry. Two diagrams generated from the actual repository tree anchor the discussion.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/swin-transformer/microsoft-swin-transformer-overview-architecture.svg" alt="Overview architecture of the Swin Transformer repository showing how main.py, config.py, models/build.py, and the SwinTransformer backbone fit together" style="min-width:900px;"></div>

<p><em>Overview of the repo: <code>main.py</code> orchestrates config, data, optimizer, and scheduler, while <code>models/build.py</code> instantiates the <code>SwinTransformer</code> backbone from <code>models/swin_transformer.py</code>.</em></p>

Reading the overview from left to right:

- [`SwinTransformer`, L485](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L485) — the top-level `nn.Module` that assembles patch embedding, four stages, final norm, and the linear head.
- [`PatchEmbed`, L437](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L437) — a strided 4×4 `Conv2d` that chops the image into non-overlapping patches and flattens them into tokens.
- [`BasicLayer`, L364](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L364) — one stage of the hierarchy: a stack of blocks followed by optional downsampling.
- [`SwinTransformerBlock`, L175](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L175) — the workhorse block that alternates regular and shifted window attention around an MLP.
- [`WindowAttention`, L77](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L77) — multi-head self-attention confined to an M×M window, with a learned relative position bias table.
- [`PatchMerging`, L315](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L315) — the resolution-halving layer that gives Swin its pyramid shape.
- [`build_model`, L15](https://github.com/microsoft/Swin-Transformer/blob/main/models/build.py#L15) — a config-driven dispatcher that also knows about Swin-V2, Swin-MoE, and Swin-MLP variants.
- [`main`, L90](https://github.com/microsoft/Swin-Transformer/blob/main/main.py#L90) — the distributed training and evaluation entry that wires everything together.
- [`load_pretrained`, L45](https://github.com/microsoft/Swin-Transformer/blob/main/utils.py#L45) — the fine-tuning loader that repairs mismatched weights, including bicubic resizing of the bias tables.

## Why Windows Instead of Global Attention

The motivating trade-off is cost. Global self-attention over H×W/16 tokens scales quadratically with image size, which is brutal for detection backbones that must process 800×1333 feature maps. Swin instead partitions the feature map into non-overlapping windows of default size M=7 and attends only within each window, making the cost linear in the number of tokens. The helper functions [`window_partition`, L45](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L45) and [`window_reverse`, L60](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L60) are just view/permute reshapes that move between `(B, H, W, C)` and a batch of `(num_windows*B, M, M, C)` tensors — no fancy kernels required for the baseline path.

The catch is that attention confined to fixed windows has no communication across window borders. Swin's fix is the *shifted window* scheme: even-numbered blocks use regular windows (W-MSA), odd-numbered blocks shift the feature map by ⌊M/2⌋ pixels along both axes first. After the shift, windows straddle the boundaries of the previous layer's windows, so tokens from different regions end up in the same attention group. You can see this in [`SwinTransformerBlock.forward` at the cyclic shift, L260](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L260), where `torch.roll` moves the feature map, and in `BasicLayer`'s construction at [L397](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L397), which alternates `shift_size=0` with `shift_size=window_size//2` across blocks.

Rolling creates one artifact: a window that wraps around the image edge contains tokens that are not actually spatial neighbors. Rather than wasting compute on those connections, the block precomputes an attention mask in its constructor ([`register_buffer("attn_mask", ...)`, L245](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L245)) that adds −100 to cross-region attention logits, effectively zeroing them after softmax. The mask is a buffer, not a parameter, so it is recomputed per resolution and — importantly — deleted by the checkpoint loader rather than restored from disk.

## The Anatomy of a Shifted Window Block

Each `SwinTransformerBlock` follows the classic pre-norm transformer recipe with two sublayers. The first applies LayerNorm, optionally rolls the feature map, partitions windows, and calls [`WindowAttention`, L77](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L77). The second is a GELU MLP with a 4× expansion, implemented in the modest [`Mlp`, L26](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L26) class. Both sublayers are wrapped in residual connections with stochastic-depth (`DropPath`) regularization.

Inside `WindowAttention`, three details deserve attention. First, the QKV projection is a single `nn.Linear(dim, dim*3)` followed by a reshape into heads — standard, but note the explicit `q * self.scale` before the matmul, since this code predates PyTorch's fused `scaled_dot_product_attention`. Second, the attention logits get a learned bias added: the [`relative_position_bias_table`, L101](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L101) is a `(2M−1)² × num_heads` parameter table, and a precomputed `relative_position_index` buffer maps each of the M²×M² token pairs inside a window to one row of that table. This relative bias is what lets the model know *where* tokens sit within their window without any absolute positional embedding. Third, the block exposes `no_weight_decay_keywords` at [L585](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L585) so the optimizer excludes this table from weight decay — a small but consequential training detail handled in [`build_optimizer`, L19](https://github.com/microsoft/Swin-Transformer/blob/main/optimizer.py#L19).

The optimizer code itself is worth a skim. [`set_weight_decay`, L59](https://github.com/microsoft/Swin-Transformer/blob/main/optimizer.py#L59) splits parameters into decay/no-decay groups based on parameter shape (1-D tensors like norms get zero decay) and the skip lists above. On the schedule side, [`build_scheduler`, L16](https://github.com/microsoft/Swin-Transformer/blob/main/lr_scheduler.py#L16) wraps timm's cosine scheduler with a linear warmup, and the repo also ships a homegrown [`LinearLRScheduler`, L66](https://github.com/microsoft/Swin-Transformer/blob/main/lr_scheduler.py#L66) for recipes that prefer linear decay. The entry point then linearly scales the learning rate with total batch size at [main.py L327](https://github.com/microsoft/Swin-Transformer/blob/main/main.py#L327), a convention from the large-batch ImageNet training era.

## PatchMerging and the Four-Stage Hierarchy

Where ViT keeps one resolution throughout, Swin builds a pyramid that CNN-based detectors expect: resolutions H/4, H/8, H/16, H/32 with channel widths doubling at each stage. The glue between stages is `PatchMerging`. Its forward pass at [`PatchMerging.forward`, L331](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L331) slices the feature map into even/odd rows and columns, concatenates the four neighbors along the channel dimension (2C becomes 4C), applies LayerNorm, and then a linear layer without bias that projects 4C back down to 2C. Resolution halves, width doubles — exactly the role a strided convolution plays in ResNet.

The `SwinTransformer` constructor at [L529](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L529) wires this up by instantiating one `BasicLayer` per stage, passing `downsample=PatchMerging` for every stage except the last. Each `BasicLayer` builds `depths[i]` blocks with alternating shift sizes, and optionally wraps each block in gradient checkpointing when `use_checkpoint` is set — a memory saver that matters for the 384-resolution and Swin-L configurations. After the final stage, [`forward_features`, L588](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py#L588) applies a final LayerNorm, average-pools over tokens, and hands the feature vector to the classification head. Because every piece is a plain module with a `flops()` method, the repo reports GFLOPs for any configuration at startup.

## Checkpoint Loading and Relative Position Remapping

The subtlety in this repo is that several "parameters" of a Swin model are actually resolution-dependent buffers: `relative_position_index`, `attn_mask`, and the size of `relative_position_bias_table` itself. That means a checkpoint trained at 224×224 cannot be loaded verbatim into a 384×384 model. The fine-tuning loader [`load_pretrained`, L45](https://github.com/microsoft/Swin-Transformer/blob/main/utils.py#L45) handles this with surgical string matching over state-dict keys.

First it deletes all `relative_position_index` and `attn_mask` entries, since those are always re-initialized from the new model's geometry ([L51](https://github.com/microsoft/Swin-Transformer/blob/main/utils.py#L51) onward). Then, for each `relative_position_bias_table` whose row count changed, it bicubically interpolates the pretrained table to the new `(2M−1)²` size — the remap logic lives at [L66](https://github.com/microsoft/Swin-Transformer/blob/main/utils.py#L66). Finally, if the classifier head mismatches, it does something clever for the ImageNet-22K → 1K transfer: rather than re-initializing, it reads `data/map22kto1k.txt`, a fixed 21841→1000 index map, and slices the pretrained head rows accordingly ([L111](https://github.com/microsoft/Swin-Transformer/blob/main/utils.py#L111)). That one file explains how the 22K-pretrained checkpoints reach their published 1K accuracies out of the box. The resume path, [`load_checkpoint`, L18](https://github.com/microsoft/Swin-Transformer/blob/main/utils.py#L18), is the ordinary strict-shape restore of model, optimizer, scheduler, and AMP scaler for interrupted runs.

## From Install to Your First Model

The [get_started guide](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) targets CUDA 10.2-era environments and pins old versions deliberately — the code relies on timm internals that changed after 0.4.12. The setup sequence is:

```bash
conda create -n swin python=3.7 -y
conda activate swin
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

The repo never grew a `requirements.txt` on its main branch; the guide pins each dependency explicitly instead, so treat those five lines as the source of truth. With a torch build of your choosing, the fastest way to touch the model is a single-GPU evaluation of the Swin-B checkpoint:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
  --cfg configs/swin/swin_base_patch4_window7_224.yaml \
  --resume swin_base_patch4_window7_224.pth --data-path <imagenet-path>
```

Training uses the same entry with `--cfg configs/swin/swin_tiny_patch4_window7_224.yaml` and `--nproc_per_node 8`, as documented in the guide's ImageNet-1K section. Under the hood, [`parse_option`, L36](https://github.com/microsoft/Swin-Transformer/blob/main/main.py#L36) merges CLI flags into the yacs defaults via [`get_config`, L352](https://github.com/microsoft/Swin-Transformer/blob/main/config.py#L352), which in turn loads the chosen YAML through [`_update_config_from_file`, L268](https://github.com/microsoft/Swin-Transformer/blob/main/config.py#L268). Data flows through [`build_loader`, L44](https://github.com/microsoft/Swin-Transformer/blob/main/data/build.py#L44), which selects between torchvision-style folders, a cached in-memory variant, and the ImageNet-22K archive reader in `data/`, then attaches mixup/CutMix from the timm data module. One epoch of the training loop, [`train_one_epoch`, L174](https://github.com/microsoft/Swin-Transformer/blob/main/main.py#L174), uses AMP mixed precision through a wrapped grad scaler and logs lr, loss, grad norm, and memory each step; [`validate`, L235](https://github.com/microsoft/Swin-Transformer/blob/main/main.py#L235) reports top-1/top-5 on the val split. Note that `torch.distributed.launch` is deprecated in modern PyTorch — with torch 2.x the code switches to reading `LOCAL_RANK` from the environment, per the comment at [main.py L70](https://github.com/microsoft/Swin-Transformer/blob/main/main.py#L70).

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/swin-transformer/microsoft-swin-transformer-architecture.svg" alt="Detailed architecture diagram of Swin Transformer internals including WindowAttention, relative position bias, PatchMerging, optimizer, LR scheduler, data loaders, and checkpoint utilities" style="min-width:900px;"></div>

<p><em>The detailed view adds the attention internals, data loading, optimizer/scheduler construction, and the checkpoint utilities in <code>utils.py</code>.</em></p>

## Honest Limits

Treat this repository as a paper-replica research artifact from 2021–2022, not a maintained library. It is effectively archived: the last significant updates landed in mid-2022. The training recipes assume serious hardware — the published ImageNet numbers come from 8× V100-class GPUs, and the ImageNet-22K pre-training runs are well beyond a single workstation, even with gradient checkpointing and `--accumulation-steps`. The dependency pins are the other friction point: `timm==0.4.12` is ancient, and importing the model with current timm versions breaks on the `timm.models.layers` import path (modern timm moved it to `timm.layers`). If you want Swin weights in a maintained stack, [timm](/timm/) re-implements every Swin variant with its own builder and pretrain tags, and the Hugging Face [torchvision](/torchvision/) releases include shifted-window backbones for detection work.

Architecturally, Swin has been superseded twice over: the same group's Swin-V2 (also in this repo, under `models/swin_transformer_v2.py`) replaced the bias table with a log-spaced coordinate function to handle larger windows, and the field has since moved to hybrid and plain ViT designs. None of this changes the historical verdict: Swin's windowed hierarchical design made transformers practical as drop-in CNN replacements, and this codebase remains the clearest executable specification of that idea.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Swin Transformer: A Source Tour of Microsoft's Hierarchical Vision Backbone",
  "description": "Walk through the Swin source: shifted-window self-attention with relative position bias, patch merging downsampling, checkpoint weight remapping, and the training entry.",
  "datePublished": "2026-10-25",
  "author": {"@type": "Person", "name": "PyShine"},
  "keywords": "Swin Transformer, vision transformer, shifted windows, hierarchical backbone, image classification",
  "publisher": {"@type": "Organization", "name": "PyShine"}
}
</script>
