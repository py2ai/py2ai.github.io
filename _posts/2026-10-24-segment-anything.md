---
layout: post
title: "Segment Anything: A Source Tour of Meta's Promptable Vision Model"
description: "Walk through the SAM source: the ViT image encoder with windowed attention, two-way transformer mask decoder, prompt embeddings, and the predictor flow."
date: 2026-10-24
header-img: "img/post-bg.jpg"
permalink: /segment-anything/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/segment-anything/facebookresearch-segment-anything-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: Segment Anything, SAM, promptable segmentation, vision transformer, mask decoder, Meta AI
author: "PyShine"
---

The Segment Anything Model (SAM) is Meta AI's answer to a simple but stubborn question: can a vision model segment any object in any image, given only a hint about what to look for? Published by the facebookresearch team in 2023, SAM was trained on the SA-1B dataset — 11 million images carrying 1.1 billion masks — and it generalizes to objects and image styles it never saw during training. Instead of predicting a fixed vocabulary of classes, SAM is promptable: you give it a click, a box, or a rough mask, and it returns precise object boundaries, class-agnostic and zero-shot.

What makes the repository unusually pleasant for a source tour is its size and discipline. The whole model lives in a handful of Python files under `segment_anything/`, written in plain PyTorch with almost no framework magic. There are exactly three neural components — a vision transformer image encoder, a prompt encoder, and a mask decoder — plus a thin `Sam` container, an interactive predictor, and an automatic mask generator that orchestrates the same pieces at scale. Every module is importable, readable in one sitting, and separated from the others by clean interfaces.

This post walks through that source the way we tour other inference stacks on this blog: first a structural map, then each subsystem with real line anchors, then a runnable setup, and finally an honest account of where the model hurts. If you have read our pieces on [PyTorch](/pytorch/) internals before, the shapes and tensor plumbing here will feel familiar.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/segment-anything/facebookresearch-segment-anything-overview-architecture.svg" alt="Segment Anything overview architecture: image encoder, prompt encoder, mask decoder, predictor, and model registry" style="min-width:900px;"></div>

<p><em>Overview of the SAM repository: the encoder consumes the image once, the prompt encoder turns clicks and boxes into tokens, and the mask decoder fuses both sides to emit candidate masks.</em></p>

Reading the overview from left to right:

- [`Sam`, L18](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/sam.py#L18) — the container module that owns all three submodules and runs the full image-plus-prompt forward pass in one call.
- [`ImageEncoderViT`, L17](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L17) — the heavyweight: a masked-autoencoder-style ViT that turns a 1024×1024 image into a 64×64 embedding grid.
- [`PatchEmbed`, L364](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L364) — the strided convolution that chops the image into 16×16 patches and adds positional embeddings.
- [`Block`, L119](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L119) — the transformer block, which switches between cheap windowed attention and expensive global attention by layer index.
- [`PromptEncoder`, L16](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L16) — converts points, boxes, and masks into sparse and dense token embeddings.
- [`MaskDecoder`, L16](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L16) — a deliberately lightweight transformer that fuses image and prompt tokens and emits masks plus quality scores.
- [`TwoWayTransformer`, L16](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/transformer.py#L16) — the decoder's core, where image tokens attend to prompts and prompts attend back to the image.
- [`SamPredictor`, L17](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py#L17) — the user-facing wrapper that caches encoder output so repeated prompts cost almost nothing.
- [`sam_model_registry`, L47](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/build_sam.py#L47) — the dict that maps `vit_b`, `vit_l`, and `vit_h` checkpoint names to builder functions.

## A ViT Encoder That Runs Once Per Image

The encoder is where nearly all of SAM's compute lives, and the design is a direct descendant of MAE pretraining. `ImageEncoderViT` takes a 1024×1024 RGB tensor, splits it with `PatchEmbed` into 4096 patch tokens of dimension 1280 (for ViT-H), and runs them through 32 transformer blocks. The key engineering trick is in `Block`: most layers use windowed attention over 14×14-token windows, and only four layers — those listed in `global_attn_indexes` — attend across the whole feature map. The helpers `window_partition` and `window_unpartition` do nothing more glamorous than reshaping the 64×64 token grid into 16 batched windows of 196 tokens and back again, but that reshape is what turns quadratic global attention into per-window attention for 28 of the 32 layers.

The output of the final block is a 64×64×1280 feature map that the neck at line 88 compresses with two rounds of convolution plus `LayerNorm2d` into 64×64×256 channels. That 256-channel embedding is the currency of the whole system: the decoder reads it, the prompt embeddings are shaped to match it, and the predictor caches it. Because the encoder is so expensive relative to everything downstream, SAM's API is built around a hard split — encode once per image, then decode as many masks as you like. The `SamPredictor` exists precisely to make that split ergonomic.

## From Install to Your First Model

Install directly from GitHub with pip:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Then download the largest checkpoint, `sam_vit_h_4b8939.pth` (about 2.4 GB), from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` — the registry treats the string `"vit_h"` and the alias `"default"` identically. A complete interactive session, verified against the README and the `SamPredictor.predict` signature in `predictor.py`, looks like this:

```python
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda")
predictor = SamPredictor(sam)

predictor.set_image(image_rgb)          # HxWx3 uint8 RGB array
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),         # 1 = foreground click
    multimask_output=True,
)
```

`set_image` resizes the long side to 1024 with `ResizeLongestSide`, normalizes with standard ImageNet statistics, runs the encoder once, and stores the 1×256×64×64 embedding on the predictor. Every subsequent `predict` call skips the encoder entirely: it transforms your coordinates back to the resized frame, embeds the prompts, and runs only the decoder. With `multimask_output=True` the decoder returns three candidate masks — roughly whole-object, part, and subpart granularity — together with predicted IoU scores in `scores`, so you can pick the best candidate without any ground truth. The low-resolution `logits` (256×256) can be fed back as a `mask_input` on the next call to refine a boundary iteratively.

## Prompts: Points, Boxes, and Masks as Tokens

The prompt encoder is small but full of design decisions. Points and boxes become "sparse" embeddings: `_embed_points` maps each click through a small MLP whose weights encode whether the point was labeled foreground or background, and `_embed_boxes` represents a box as just its two corner points sharing a dedicated corner embedding — the decoder learns to treat opposite corners as a bounding gesture. A padding token is appended when no points are supplied so that batches stay aligned.

Masks take a different route: `_embed_masks` downsamples a 256×256 low-resolution mask with convolutional layers into the same 64×64 dense grid as the image embedding, and `get_dense_pe` adds a sinusoidal positional encoding so the decoder knows where each dense value sits. This asymmetry — sparse tokens for geometric hints, dense tensors for spatial hints — is what lets one decoder interface accept such different prompt types. It is also why the iterative refinement trick works: the decoder's own 256×256 output can be re-entered as a mask prompt on the next turn, giving the model a correction channel without any retraining.

## The Two-Way Mask Decoder

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/segment-anything/facebookresearch-segment-anything-architecture.svg" alt="Segment Anything detail architecture: encoder blocks, prompt embedding paths, two-way transformer, hypernetwork MLPs, predictor and automatic mask generator flow" style="min-width:900px;"></div>

<p><em>Detail view: the decoder concatenates IoU and mask tokens with image and prompt tokens, runs the two-way transformer, then produces masks via upscaling and hypernetwork kernels.</em></p>

Inside `predict_masks`, the decoder concatenates a learned `iou_token` and four `mask_tokens` with the image embedding and the sparse prompt tokens, then runs the `TwoWayTransformer` — two layers, each one a `TwoWayAttentionBlock` with four steps: self-attention on image tokens, image-to-prompt cross-attention, an MLP, and prompt-to-image cross-attention. The bidirectionality matters: the prompt tokens "query" the image to find the object the user meant, while image tokens attend back to prompts so the picture context informs what each token should stand for.

Mask generation then happens in two parallel branches. The upscaled embedding branch doubles the 64×64 grid twice with transposed convolutions to 256×256. The hypernetwork branch passes each mask token through its own `MLP` to produce a dynamic 32×32 kernel, which is multiplied against the upscaled embedding — so each candidate mask is literally a dot product between a token-conditioned filter and the image features. Meanwhile the `iou_prediction_head` turns the IoU token's output into a quality estimate per mask. The whole decoder is only a few million parameters: nearly all of SAM's capacity is parked in the encoder, exactly so this part stays fast enough for interactive use.

## From Clicks to Whole-Image Coverage

The predictor and the automatic mask generator demonstrate two opposite usage modes over the same model. `SamPredictor` is built for a human in the loop: one `set_image`, then cheap per-click `predict` calls, optionally with a box prompt for crisper targets. `SamAutomaticMaskGenerator` has no human, so it manufactures prompts: `generate` splits the image into overlapping crops, `_process_crop` lays down a 32×32 grid of foreground points (1024 clicks per crop by default), predicts masks in batches through the predictor, and pushes everything through a chain of `MaskData` filters — stability scoring, deduplication, and small-region removal — before stitching masks back into full-image coordinates. Both paths share the same three modules; the difference is only in how prompts are sourced and how outputs are filtered.

## Honest Limits

The encoder cost is the first wall. Every new image pays a full 1024×1024 ViT-H forward pass — noticeable on a laptop GPU, several hundred milliseconds on a good desktop card — and there is no video support in this repository at all: it tracks nothing across frames, and Meta's SAM 2 lives in a separate codebase. The model also has no semantic vocabulary. Masks are class-agnostic; if you need labels you must attach your own classifier on top of each crop, or pair SAM with something like [CLIP](/clip/) for open-vocabulary naming.

The prompt types trade off differently than newcomers expect. A single click is ambiguous, which is exactly why `multimask_output` defaults to true; a box prompt resolves ambiguity well but biases the mask toward the box edges; mask prompts enable iterative refinement but are limited by the 256×256 feedback resolution, so hair-thin structures can wash out over repeated refinement rounds. Finally, the ONNX export path (`SamOnnxModel` in `segment_anything/utils/onnx.py`) covers only the prompt encoder and decoder — the ViT encoder stays in PyTorch, so browser-side demos still need a server or a separately exported encoder. If you are planning production serving around ONNX, our [ONNX](/onnx/) tour and the [torchvision](/torchvision/) model utilities are the natural next stops.

```json
{"@context": "https://schema.org", "@type": "TechArticle", "headline": "Segment Anything: A Source Tour of Meta's Promptable Vision Model", "description": "Walk through the SAM source: the ViT image encoder with windowed attention, two-way transformer mask decoder, prompt embeddings, and the predictor flow.", "datePublished": "2026-10-24", "keywords": "Segment Anything, SAM, promptable segmentation, vision transformer, mask decoder, Meta AI", "author": {"@type": "Person", "name": "PyShine"}, "publisher": {"@type": "Organization", "name": "PyShine"}}
```
