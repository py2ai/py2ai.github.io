---
layout: post
title: "torchvision Under the Hood: transforms v2 and the C++ Ops That Power Detection"
description: "A source tour of torchvision: the transforms v2 engine, tv_tensors, and the NMS, RoIAlign, and deformable convolution C++/CUDA kernels behind detection models."
date: 2026-10-22
header-img: "img/post-bg.jpg"
permalink: /torchvision/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/torchvision/pytorch-torchvision-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: torchvision, transforms v2, NMS, RoIAlign, deformable convolution, tv_tensors, object detection, data augmentation, PyTorch, CUDA kernels
author: "PyShine"
---

Most people treat torchvision as a convenience layer: a `Resize` here, a pretrained ResNet there. But if you train or deploy detection models, you are standing on a much more interesting machine. The `torchvision.transforms.v2` package rebuilt augmentation around a single `Transform` base class that understands whole samples — image, masks, bounding boxes, keypoints, labels — and moves them together through every staged operation. And the `torchvision.ops` package hides hand-written C++ and CUDA kernels for the operations that inference servers actually burn cycles on: non-maximum suppression, RoI alignment, deformable convolution. This post walks that machine from the Python surface down to the `.cu` files, continuing our series of source tours where we previously went inside [PyTorch itself](/pytorch/) and compared it against [TensorFlow](/tensorflow/).

The two diagrams below were generated directly from the `pytorch/vision` repository on the `main` branch. The overview shows the full arc: user-facing v2 transforms on the left, the type-driven kernel dispatch in the middle, compiled ops and their CUDA kernels on the right, and the Faster R-CNN detection stack that ties both halves together.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/torchvision/pytorch-torchvision-overview-architecture.svg" alt="torchvision architecture" style="max-width:100%;"></div>
<p><em>torchvision at a glance: the v2 transform engine on the left feeds type-dispatched kernels; on the right, Python op wrappers hand off to registered C++/CUDA kernels that detection models like Faster R-CNN consume.</em></p>

Reading the overview from left to right:

- [transforms.v2, L1](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/__init__.py#L1) — one flat namespace that re-exports every v2 transform, from `Compose` to `SanitizeBoundingBoxes`, so user code never has to know the internal module layout.
- [Compose, L11](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_container.py#L11) — the container that chains transforms; its [forward, L49](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_container.py#L49) passes the entire sample tuple through each step in order.
- [Transform, L17](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_transform.py#L17) — the base class whose [forward, L57](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_transform.py#L57) flattens the whole sample, samples parameters once, and transforms only the leaves that need it.
- [_get_kernel, L99](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_utils.py#L99) — the registry that resolves which functional kernel runs for a given input type; [register_kernel, L70](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_utils.py#L70) is the extension point.
- [RandomResizedCrop, L201](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_geometry.py#L201) — the workhorse of classification augmentation, part of a spatial catalog that also includes detection-specific [RandomIoUCrop, L1123](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_geometry.py#L1123).
- [ColorJitter, L72](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_color.py#L72) — the photometric family, plus [RandomPhotometricDistort, L185](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_color.py#L185), the SSD-style distorter.
- [nms, L20](https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L20) — the Python face of suppression, which calls the compiled op at [L54](https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L54) after checking the C extension is present.
- [nms_kernel, L165](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu#L165) — the CUDA entry point that launches a blockwise IoU computation and [gather_keep_from_mask, L104](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu#L104) collects the survivors.
- [RoIHeads, L504](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py#L504) — the second detection stage, whose [postprocess_detections, L680](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py#L680) is a direct consumer of the ops on the right side of the diagram.

## One Sample In, Everything Together

The v1 transforms API was built for a single image; the v2 API was built for a *sample*. The key decision lives in [Transform.forward, L57](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_transform.py#L57): whatever you pass — a tensor, a `(img, target)` pair, a dict with masks and boxes — gets flattened with torch's pytree machinery into a list of leaves plus a spec. Then `_needs_transform_list` decides which leaves are actually transformed. Pure tensors are passed through untouched when an explicit `Image` or `Video` is present in the sample, so the numeric labels in your detection target survive a `Resize` without being accidentally treated as pixels. That heuristic is documented at length in [_needs_transform_list, L75](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_transform.py#L75), and it is the reason v2 can keep backward compatibility for bare-tensor callers while still protecting structured targets.

The sample leaves are usually `tv_tensors` objects: tensor subclasses that carry just enough metadata for geometry to stay correct. [Image, L11](https://github.com/pytorch/vision/blob/main/torchvision/tv_tensors/_image.py#L11) remembers that it is CHW; [BoundingBoxes, L63](https://github.com/pytorch/vision/blob/main/torchvision/tv_tensors/_bounding_boxes.py#L63) remembers its format (`XYXY`, `XYWH`, `CXCYWH`) and its `canvas_size`, so a crop can re-coordinatize boxes instead of silently corrupting them. When a spatial transform runs, it needs the ground truth of "how big is the image" and "where are the boxes" — helpers like [get_bounding_boxes, L156](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_utils.py#L156) and [query_size, L195](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_utils.py#L195) extract that from the flattened sample, and [check_type, L221](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_utils.py#L221) implements the type matching used everywhere else.

Each subclass then overrides only two methods: `make_params` draws randomness once per call, and `transform` applies it to each flagged leaf. That split is what makes paired augmentation cheap — a flip decides its side once, then applies the same decision to image, boxes, and masks without re-rolling dice. Metadata-only transforms live in their own corner: [ConvertBoundingBoxFormat, L8](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_meta.py#L8) and [ClampBoundingBoxes, L27](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_meta.py#L27) mutate coordinates without touching pixels, and [SanitizeBoundingBoxes, L360](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_misc.py#L360) drops degenerate boxes after aggressive crops — an unglamorous but essential step for detection training stability.

## The Kernel Registry: Type-Driven Dispatch

A v2 transform never hardcodes "if PIL do X, if tensor do Y". Instead, when its `transform()` method needs real work done, it calls `self._call_kernel(functional, inpt, ...)`, which lands in [_get_kernel, L99](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_utils.py#L99). The registry is keyed by the pair of functional name and input type: `F.resize` has one kernel for `Image`, another for `Video`, another for PIL images, and a passthrough for anything it does not recognize. The public [register_kernel, L70](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_utils.py#L70) decorator lets third-party tensor subclasses plug into the same chain — wrap your type, register a kernel, and every built-in transform instantly works with it.

This design is why the catalog can stay small and readable. [RandomResizedCrop, L201](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_geometry.py#L201) is essentially parameter math plus a kernel call; the device- and type-specific branches live in the functional layer instead. The photometric side works the same way: [ColorJitter, L72](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_color.py#L72) composes brightness, contrast, saturation, and hue functionals, while [RandomPhotometricDistort, L185](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_color.py#L185) wraps that jitter with random channel permutation to reproduce SSD's augmentation recipe. Even the exotic family plays along — [TrivialAugmentWide, L445](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_auto_augment.py#L445) and its siblings in `_auto_augment.py` sample an op and magnitude, then dispatch through the same registry. Mixing recipes like [MixUp, L220](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_augment.py#L220) or [JPEG, L337](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_augment.py#L337) (synthetic compression artifacts) round out the catalog, and [ToImage, L35](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_type_conversion.py#L35) plus [ToDtype, L253](https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/_misc.py#L253) handle the canonical end-of-chain conversions.

## NMS From Python to CUDA

Non-maximum suppression is the highest-traffic op in any detection deployment, and torchvision implements it three times: once per device that matters. The Python wrapper [nms, L20](https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L20) validates the extension, then dispatches through `torch.ops.torchvision.nms` at [L54](https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L54) — the same call also routes to a quantized kernel when the inputs qualify. On the C++ side, [the op registration, L16](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/nms.cpp#L16) defines `torchvision::nms` in the stable library block at [L26](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/nms.cpp#L26), and the dispatcher picks the backend.

The CPU path in [nms_kernel_impl, L18](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp#L18) is the textbook algorithm: sort by score, walk the boxes, suppress overlaps above the IoU threshold. The CUDA path is where it gets clever. [nms_kernel, L165](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu#L165) tiles the boxes into blocks, computes a pairwise IoU matrix in shared memory, and then [gather_keep_from_mask, L104](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/nms_kernel.cu#L104) reduces the per-block suppression bitmasks into the final keep list — turning an O(n²) serial loop into a parallel scan that stays fast even at thousands of detections. Beyond plain NMS, [batched_nms, L57](https://github.com/pytorch/vision/blob/main/torchvision/ops/boxes.py#L57) offsets boxes per class index so a single suppression call handles all labels, and rotated boxes get their own compiled path through [box_iou_rotated, L22](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/box_iou_rotated.cpp#L22). All of this is invisible to the caller, which is exactly the point: when you export a model to [ONNX](/onnx/) or serve it behind [TorchServe](/torchserve/), the same registered op name resolves on whichever device the server picked.

## RoIAlign, Deformable Convolution, and the FPN Glue

Two more compiled ops carry the detection stack. RoIAlign fixes the quantization error that made older RoIPool lossy: [roi_align, L204](https://github.com/pytorch/vision/blob/main/torchvision/ops/roi_align.py#L204) passes `sampling_ratio` and `aligned` flags down through the registered op at [L283](https://github.com/pytorch/vision/blob/main/torchvision/ops/roi_align.py#L283), and the module wrapper [RoIAlign, L288](https://github.com/pytorch/vision/blob/main/torchvision/ops/roi_align.py#L288) makes it layer-composable. The C++ entry at [Tensor roi_align, L16](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/roi_align.cpp#L16) fans out to a scalar CPU kernel ([roi_align_forward_kernel_impl, L20](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/roi_align_kernel.cpp#L20)) and a GPU twin ([roi_align_forward_kernel_impl, L80](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/roi_align_kernel.cu#L80)) that assigns one CUDA thread per output bin and bilinearly samples four points.

Deformable convolution is the heavyweight: [deform_conv2d, L14](https://github.com/pytorch/vision/blob/main/torchvision/ops/deform_conv.py#L14) forwards offsets through the op call at [L92](https://github.com/pytorch/vision/blob/main/torchvision/ops/deform_conv.py#L92) into a C++ entry ([Tensor deform_conv2d, L16](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/deform_conv2d.cpp#L16)) and then a trio of CUDA kernels — [deformable_im2col_kernel, L162](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu#L162) gathers offset-sampled patches, [deformable_col2im_kernel, L404](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu#L404) scatters input gradients, and [deformable_col2im_coord_kernel, L673](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu#L673) computes the gradients *with respect to the offsets themselves*. The glue that assembles these into a model is [MultiScaleRoIAlign, L230](https://github.com/pytorch/vision/blob/main/torchvision/ops/poolers.py#L230), which maps each proposal to an FPN level by area and runs RoIAlign per level. Faster R-CNN consumes all of it: [FasterRCNN, L43](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py#L43) wires a [TwoMLPHead, L286](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py#L286) onto the pooled features, while [RoIHeads, L504](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py#L504) handles proposal sampling ([select_training_samples, L642](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py#L642)), the loss ([fastrcnn_loss, L12](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py#L12)), and final NMS. The in-model normalization stage, [GeneralizedRCNNTransform, L86](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/transform.py#L86), is itself a mini version of the v2 idea — resize and normalize, but track `image_sizes` so box decoding stays exact.

## From Install to Your First Model

Installation is one line alongside PyTorch, and the CUDA wheels bundle the prebuilt ops — no local toolchain needed:

```bash
pip install torchvision
```

The v2 API accepts plain tensors, but it shines on structured detection samples. Everything in this snippet is a public symbol verified in the `main` source:

```python
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

img = tv_tensors.Image(torch.rand(3, 256, 256))
target = {
    "boxes": tv_tensors.BoundingBoxes(
        [[20, 30, 200, 220]], format="XYXY", canvas_size=(256, 256)
    ),
    "labels": torch.tensor([7]),
}

t = v2.Compose([
    v2.RandomPhotometricDistort(p=0.5),
    v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
    v2.RandomIoUCrop(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.SanitizeBoundingBoxes(),
])

img_out, target_out = t(img, target)   # boxes stay valid through every step
```

For classification, a single `v2.RandomResizedCrop(size=(224, 224), antialias=True)` in the same chain covers the classic recipe. Because `Compose` is just an `nn.Module` subclass, the finished chain drops into a `DataLoader` worker unchanged, and the same ops run server-side if you later push the model through [ExecuTorch](/executorch/) or any runtime that links the torchvision extension.

## Honest Limits

The design carries real trade-offs. First, the compiled ops are a deployment tax: `torch.ops.torchvision.*` symbols are only available when the C/CUDA extension is built or a matching wheel is installed, and building from source requires a full CUDA toolchain pinned to your PyTorch version. Pure-Python fallbacks exist for some box math, but NMS and RoIAlign without the extension either fail or crawl. Second, the v1/v2 duality is genuine friction: both APIs live side by side, several v2 classes keep v1 aliases for backward compatibility, and v2 deliberately drops TorchScript support for containers like `Compose` — if your codebase still scripts transforms, migrating is a port, not a rename. Third, kernel coverage has edges: the CUDA NMS kernel targets standard boxes well, but rotated IoU and deformable paths are thinner (fewer tuned variants, occasional gaps on newer devices until the next release), and video transforms dispatch through fewer specialized kernels than their image counterparts. None of these are disqualifiers — they are the price of a Python API that compiles down to measured, device-specific kernels, and the reason torchvision remains the reference implementation the rest of the detection ecosystem measures against.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "torchvision Under the Hood: transforms v2 and the C++ Ops That Power Detection",
  "description": "A source tour of torchvision: the transforms v2 engine, tv_tensors, and the NMS, RoIAlign, and deformable convolution C++/CUDA kernels behind detection models.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-22",
  "keywords": "torchvision, transforms v2, NMS, RoIAlign, deformable convolution, tv_tensors, object detection, data augmentation, PyTorch, CUDA kernels",
  "publisher": {"@type": "Organization", "name": "PyShine"}
}
</script>
