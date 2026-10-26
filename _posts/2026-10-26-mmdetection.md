---
layout: post
title: "MMDetection Source Tour: How OpenMMLab's Detection Toolbox Really Fits Together"
description: "A guided walk through the MMDetection 3.x source tree: detectors, heads, backbones, hooks and metrics, with two architecture diagrams and line-verified code anchors."
date: 2026-10-26
header-img: "img/post-bg.jpg"
permalink: /mmdetection/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/mmdetection/open-mmlab-mmdetection-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: mmdetection, object detection, mmengine, mmcv, faster r-cnn, mask r-cnn, rtmdet, coco
author: "PyShine"
---

MMDetection, maintained under the OpenMMLab umbrella, is probably the most complete open-source toolbox for object detection research. It ships well over a hundred detector recipes — from the classic Faster R-CNN and Mask R-CNN families, through Cascade R-CNN and the RetinaNet/ATSS line of one-stage models, to the RTMDet series and the query-based DETR/DINO branch — all expressed as composable config files running on top of MMEngine and MMCV. The [PyTorch](/pytorch/) foundation is assumed; everything detection-specific lives inside the repository we are touring today.

What makes the codebase genuinely interesting is that the model classes themselves are almost boringly small. `FasterRCNN` is an eight-line class. `MaskRCNN` adds an instance-segmentation branch in a handful more. The real engineering lives in the scaffolding around them: a registry that resolves every `type=` string in a config into a live Python object, a config-inheritance system where one file refines another by a `_base_` pointer, and a training loop you almost never write yourself. Reading the source therefore teaches you a framework design pattern as much as it teaches detection: this is the style that [Detectron2](/detectron2/) approaches from a different angle, and it rewards you every time you need to swap one component without touching the rest.

In this tour we read the tree the way a contributor would: entry points first, then the detector hierarchy, then the heads that do the actual work, and finally the data and evaluation layers that keep everything honest. Both diagrams below were generated directly from the current `main` branch of `open-mmlab/mmdetection`, and every class name and line anchor in this article was verified against that checkout, so the GitHub links should land on the exact definitions.

There is also a practical reason to study this particular tree: it has quietly become the baseline other papers are expected to beat. Survey papers on detection benchmark their tricks against MMDetection re-implementations, many competition entries start from its configs, and downstream libraries for rotated boxes, 3D detection and tracking were spun out of it. If you can navigate this repository, you can read the deltas in almost any modern detection paper's experimental setup — that fluency is worth more than any single model checkpoint.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/mmdetection/open-mmlab-mmdetection-overview-architecture.svg" alt="MMDetection architecture overview" style="max-width:100%;"></div>
<p><em>Figure 1 — the bird's-eye view: an inference API and a validation hook feed one abstract base that fans out into two-stage and single-stage families, each backed by prediction heads, a pyramid neck and a ResNet backbone.</em></p>

## Reading the overview from left to right:

- **[DetInferencer](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/apis/det_inferencer.py#L45)** is where most users first touch the library: it loads a config by name, builds the model, runs inference and can return either drawn images or JSON results in a single call.
- **[DetVisualizationHook](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/engine/hooks/visualization_hook.py#L22)** draws box overlays during validation and testing, so training runs leave inspectable images in your work directory without any extra plotting code.
- **[BaseDetector](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/base.py#L17)** defines the contract every model obeys — the `loss`, `predict` and `extract_feat` entry points that MMEngine's runner and tester call.
- **[TwoStageDetector](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/two_stage.py#L16)** composes backbone, neck, proposal head and RoI head into the classic Faster R-CNN shape; nearly every R-CNN variant in the repo is a thin subclass of it.
- **[FasterRCNN](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/faster_rcnn.py#L8)** is that composition made concrete — the class body is essentially just registry wiring, which is the whole point.
- **[RPNHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rpn_head.py#L22)** scans the pyramid for objectness and turns raw anchors into the proposals the second stage refines.
- **[StandardRoIHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/standard_roi_head.py#L17)** samples those proposals, pools their features, and delegates classification, box refinement and (optionally) masking to sub-heads.
- **[FPN](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py#L15)** fuses the stage outputs of [ResNet](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/resnet.py#L306) into a multi-scale pyramid — the same slot where a [Swin Transformer](/swin-transformer/) backbone or a PAFPN variant plugs in.
- **[RTMDet](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/rtmdet.py#L12)** shows the single-stage alternative in the same panel: one dense head reading a PAFPN-fed pyramid, no proposal step at all.

## The Detail Diagram: Heads, Hooks and Data

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/mmdetection/open-mmlab-mmdetection-architecture.svg" alt="MMDetection detailed architecture" style="max-width:100%;"></div>
<p><em>Figure 2 — the full map: detector families on the left, the dense-head and RoI-head hierarchies in the middle, backbones and task modules at the bottom, and the data-to-metric loop closing on the right.</em></p>

The detail view makes the two inheritance axes obvious. Down the left side, detectors specialize: [BaseDetector](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/base.py#L17) spawns [TwoStageDetector](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/two_stage.py#L16), [SingleStageDetector](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/single_stage.py#L13) and a shared [DetectionTransformer](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/base_detr.py#L14) for query-based models. Two-stage then splits into [FasterRCNN](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/faster_rcnn.py#L8), [MaskRCNN](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/mask_rcnn.py#L10) and [CascadeRCNN](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/cascade_rcnn.py#L8), while the single-stage line leads to [RTMDet](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/rtmdet.py#L12) and the transformer line to [DETR](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/detr.py#L16).

Down the middle run the heads. Dense heads all descend from [BaseDenseHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/base_dense_head.py#L23); its anchor-aware child [AnchorHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/anchor_head.py#L22) is the parent of [RPNHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rpn_head.py#L22), [RetinaHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/retina_head.py#L10) and [ATSSHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/atss_head.py#L19), whose descendant [RTMDetHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rtmdet_head.py#L22) powers the real-time family. Anchors themselves are laid out by [AnchorGenerator](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/prior_generators/anchor_generator.py#L18), a task module that heads receive as a configurable component rather than hardcode. On the RoI side, [BaseRoIHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/base_roi_head.py#L13) begets [StandardRoIHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/standard_roi_head.py#L17) and [CascadeRoIHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/cascade_roi_head.py#L22); the former delegates box work to [BBoxHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/bbox_heads/bbox_head.py#L23) and its two-shared-FC child [Shared2FCBBoxHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L221), and masks to [FCNMaskHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L29). A query-based model instead leans on [DETRHead](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/detr_head.py#L24), which turns decoder embeddings into boxes with no anchors anywhere.

The right edge of the diagram is the plumbing that makes training reproducible. [CocoDataset](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/coco.py#L14) yields per-image records wrapped in [DetDataSample](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/structures/det_data_sample.py#L7); [DetDataPreprocessor](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/data_preprocessors/data_preprocessor.py#L31) normalizes, pads and stacks them into batches before the model ever sees a tensor; predictions ride back inside the same container to [DetVisualizationHook](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/engine/hooks/visualization_hook.py#L22) and to [CocoMetric](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/evaluation/metrics/coco_metric.py#L23), which computes COCO-style mAP.

The bottom edge is where the pixels get read. Two-stage recipes mostly sit on [ResNet](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/resnet.py#L306), whose four stage outputs feed [FPN](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/fpn.py#L15) into five pyramid levels. RTMDet swaps in [CSPNeXt](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/cspnext.py#L18), a cross-stage partial backbone with depthwise-friendly blocks, and drains it through [PAFPN](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/pafpn.py#L11), which adds a bottom-up path so low-level localization detail survives the trip to the head. Because both sides of that exchange are just registry keys, changing the feature extractor for a whole family of models is a two-line config edit — the sort of experiment that used to require forking the entire training script.

## From Install to Your First Model

The blessed route is via OpenMMLab's own package manager:

```bash
pip install -U openmim
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
```

With that in place, inference is three lines — no config file hunting required:

```python
from mmdet.apis import DetInferencer

model = DetInferencer("rtmdet_s_8xb32-300e_coco")
result = model("demo/demo.jpg", out_dir="outputs")
```

`DetInferencer` resolves the config name against the OpenMMLab model zoo, downloads weights on first use, and handles the preprocessing and result drawing internally. Under the hood this is the same machinery you saw in Figure 1: the string is a config path, the config names every module, and the registry instantiates the whole stack from backbone to metric.

Training follows the same declarative logic. From a repository clone:

```bash
python tools/train.py configs/rtmdet/rtmdet_s_8xb32-300e_coco.py
python tools/test.py  configs/rtmdet/rtmdet_s_8xb32-300e_coco.py \
    checkpoints/rtmdet_s_8xb32-300e_coco.pth
```

The two commands differ only in which forward mode the runner drives. If you want a Faster R-CNN baseline instead, swap the config path — nothing else changes, because [TwoStageDetector](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/detectors/two_stage.py#L16) and its children already exist behind the registry.

The config you just passed around is itself an exercise in inheritance. It opens with `_base_ = '../_base_/schedules/schedule_1x.py'`-style pointers that pull in a standard schedule, a standard dataset wrapper and standard runtime defaults, then overrides only what makes RTMDet special: the head, the loss weights, the augmentation schedule. When you later wonder why your model trains with a particular learning-rate rule you never typed, the answer is almost always three `_base_` hops up the chain. Custom datasets usually mean reusing [CocoDataset](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/coco.py#L14) with a different annotation file and overriding `classes` in your config — genuinely most of the work for a standard detection project, with the framework absorbing the rest.

## How Training Really Runs

Nothing in your config is a Python training loop — MMEngine supplies one. Its runner reads the config, builds the model, the data loading stack and the optimizer, then walks epochs through train and validation loops. Every epoch, the runner pulls batches from the dataloader; the data loading stack — dataset, transforms, and the detector's data preprocessor — has already turned raw COCO dictionaries into `DetDataSample` objects with normalized, padded tensors. The model's `loss` method consumes those batches and returns a dict of named losses; the runner back-propagates the sum and lets the optimizer step.

The hook system is what makes this extensible. Checkpointing, logging, learning-rate scheduling and the visualization hook all register into a timeline that fires before and after epochs, iterations and validation passes. You can write a hook in ten lines, register it in your config, and it sits next to the built-ins — that is exactly how the semi-supervised and mean-teacher variants in the repo extend training without forking the loop.

Evaluation closes the loop with the same containers. After each validation pass, the tester collects `DetDataSample` predictions, hands them to [CocoMetric](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/evaluation/metrics/coco_metric.py#L23) together with the ground truth kept in the dataset, and reports mAP per class. Because predictions and annotations travel in the same structure end to end, adding a new metric or dataset rarely touches model code — which is precisely the separation the config-driven design was built to buy you.

Scaling follows the same pattern. The batch size and worker count are baked into config ids themselves — `8xb32` literally means eight GPUs at batch 32 — and switching to multi-machine training is a launcher flag on the tools, not a code change. Mixed precision, gradient accumulation and per-parameter optimizer groups are all schedule-level settings that live beside the model definition, which is why reproducing a paper number is usually a matter of picking the right config id rather than reverse-engineering an `args` namespace.

## Honest Limits

The design has costs, and they are worth naming before you build on it. Config sprawl is the first: because models are assembled declaratively, a Mask R-CNN recipe inherits from a generic two-stage base that inherits from a dataset definition, and by the time you have followed four `_base_` hops you are reading five files to answer one question. Second, the 2.x-to-3.x transition reorganized nearly every import path and config id; tutorials from 2021 are actively misleading today, and the `mmdet.apis` surface shrank to the inferencer you saw above. Third, MMCV version pinning is real friction — matching CUDA, PyTorch and mmcv wheels is the most common install failure, and it bites hardest on Windows.

Scope is also narrower than the name suggests. MMDetection is a detection toolbox, not a general vision platform: no classification, no diffusion, no LLM-adjacent tooling, and tracking lives in a separate repository. If you want a single-file training script you can fork in an afternoon, [YOLOv5](/yolov5/) or its Ultralytics successors are friendlier; if you want research-grade reference implementations with a different config philosophy, [Detectron2](/detectron2/) remains the rival worth reading next to this codebase. But for breadth of detectors, reproducibility of published numbers, and a registry/config architecture that scales across hundreds of models, MMDetection is still the reference — and now you know where its seams are.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "MMDetection Source Tour: How OpenMMLab's Detection Toolbox Really Fits Together",
  "description": "A guided walk through the MMDetection 3.x source tree: detectors, heads, backbones, hooks and metrics, with two architecture diagrams and line-verified code anchors.",
  "datePublished": "2026-10-26",
  "author": { "@type": "Person", "name": "PyShine" },
  "keywords": "mmdetection, object detection, mmengine, mmcv, faster r-cnn, mask r-cnn, rtmdet, coco"
}
</script>
