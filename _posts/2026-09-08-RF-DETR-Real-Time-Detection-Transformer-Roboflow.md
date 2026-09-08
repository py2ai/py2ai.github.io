---
layout: post
title: "RF-DETR: Roboflow's Real-Time Detection Transformer for Object Detection, Segmentation, and Keypoints"
description: "RF-DETR is a real-time transformer architecture for object detection, instance segmentation, and keypoint detection developed by Roboflow. Built on a DINOv2 vision transformer backbone and a Deformable DETR decoder, RF-DETR delivers state-of-the-art accuracy and latency trade-offs on Microsoft COCO and RF100-VL. The open-source rfdetr package ships six model sizes from Nano to 2XLarge under Apache 2.0, with Plus components (XL/2XL) under PML 1.0. A single consistent API covers three tasks, and the same NAS method used to design the published models is available on the Roboflow platform. This post walks through the architecture, model size matrix, training and fine-tuning pipeline, and export and deployment targets."
date: 2026-09-08
header-img: "img/post-bg.jpg"
permalink: /RF-DETR-Real-Time-Detection-Transformer-Roboflow/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - RF-DETR
  - Object Detection
  - Instance Segmentation
  - Keypoint Detection
  - Transformer
  - DINOv2
  - DETR
  - Computer Vision
  - Open Source
  - Python
author: PyShine
---

## What is RF-DETR

RF-DETR is a real-time transformer architecture for object detection, instance segmentation, and keypoint detection developed by Roboflow. It is built on a DINOv2 vision transformer backbone and a Deformable DETR decoder, and it delivers state-of-the-art accuracy and latency trade-offs on Microsoft COCO and RF100-VL. The architecture was created with neural architecture search (NAS), and the same NAS method is now available on the Roboflow platform so a user can discover the best architecture for their own dataset.

The open-source `rfdetr` package and the Apache-designated model weights are released under Apache License 2.0. Plus components, including the `rfdetr_plus` extension and the RF-DETR-XL and RF-DETR-2XL detection models, are licensed under PML 1.0. The work was published at ICLR 2026 and is described in the paper [RF-DETR: Real-Time Detection Transformer](https://arxiv.org/abs/2511.09554). The code is on GitHub at [roboflow/rf-detr](https://github.com/roboflow/rf-detr) and the package is on [PyPI](https://pypi.org/project/rfdetr/).

## Model Architecture

RF-DETR is organized as a classic detection transformer with three stages. An input image at a task-appropriate resolution is passed through a DINOv2 vision transformer backbone, which is either frozen or partially tuned during training. The backbone produces multi-scale feature maps. These feature maps feed a Deformable DETR decoder, which uses object queries and cross-attention to reason about candidate detections. The decoder output is then routed to one of three task heads: a detection head that produces boxes and classes, a segmentation head that produces instance masks, and a keypoint head (currently in preview) that produces pose keypoints.

![RF-DETR model architecture](/assets/img/diagrams/rf-detr/rfdetr-architecture.svg)

The design is intentionally modular. The same backbone and decoder serve all three tasks, so a user who trains a detection model can later add a segmentation head without rebuilding the backbone. The DINOv2 backbone is pretrained on self-supervised data, which gives RF-DETR strong feature representations even on small datasets, and the Deformable DETR decoder keeps the cross-attention local and efficient through deformable sampling, which is what lets the model run in real time on a T4.

## Model Size Matrix

RF-DETR ships six model sizes, from Nano to 2XLarge, across detection, instance segmentation, and keypoint detection. The table below summarizes the detection sizes; the segmentation variants follow the same N-to-2XL progression with slightly larger parameter counts due to the mask head.

| Size | Detection class | Segmentation class | COCO AP50:95 (det) | COCO AP50:95 (seg) | Latency (ms) | Params (M) | Resolution | License |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| N | `RFDETRNano` | `RFDETRSegNano` | 48.4 | 40.3 | 2.3-3.4 | 30.5-33.6 | 384x384 | Apache 2.0 |
| S | `RFDETRSmall` | `RFDETRSegSmall` | 53.0 | 43.1 | 3.5-4.4 | 32.1-33.7 | 512x512 | Apache 2.0 |
| M | `RFDETRMedium` | `RFDETRSegMedium` | 54.7 | 45.3 | 4.4-5.9 | 33.7-35.7 | 576x576 | Apache 2.0 |
| L | `RFDETRLarge` | `RFDETRSegLarge` | 56.5 | 47.1 | 6.8-8.8 | 33.9-36.2 | 704x704 | Apache 2.0 |
| XL | `RFDETRXLarge` | `RFDETRSegXLarge` | 58.6 | 48.8 | 11.5-13.5 | 126.4-38.1 | 700x700 | PML 1.0 |
| 2XL | `RFDETR2XLarge` | `RFDETRSeg2XLarge` | 60.1 | 49.9 | 17.2-21.8 | 126.9-38.6 | 880x880 | PML 1.0 |

The matrix below shows how the six sizes map onto the three tasks and the two license tiers.

![RF-DETR model size matrix](/assets/img/diagrams/rf-detr/rfdetr-model-matrix.svg)

The Nano through Large sizes are Apache 2.0 and cover the latency-sensitive range from 2.3 ms to 8.8 ms on an NVIDIA T4 with TensorRT FP16. The XL and 2XL sizes are PML 1.0 and push accuracy further, reaching 60.1 COCO AP50:95 for detection and 49.9 for segmentation. The keypoint variant is currently a single preview model at 71.8 COCO AP50:95 (OKS-based).

## Benchmarks

RF-DETR achieves state-of-the-art results on both object detection and instance segmentation. All COCO accuracy numbers are measured in-house using pycocotools in single-artifact benchmarking (SAB) over the full 5,000-image `val2017` split, so every row is directly comparable. Latency is measured on an NVIDIA T4 using TensorRT, FP16, and batch size 1. Parameter counts are deployment-fused `nn.Module` counts. For full benchmarking methodology and reproducibility details, see [roboflow/sab](https://github.com/roboflow/single_artifact_benchmarking).

On detection, RF-DETR-L reaches 56.5 COCO AP50:95 at 6.8 ms, outperforming YOLO26-X (56.9 AP at 9.6 ms) and D-FINE-L (57.2 AP at 7.5 ms) in the latency-accuracy trade-off. On instance segmentation, RF-DETR-Seg-2XL reaches 49.9 COCO AP50:95 at 21.8 ms, against YOLO26-X-Seg at 46.8 AP at 12.92 ms. On keypoints (preview), RF-DETR reaches 71.8 COCO AP50:95 (OKS-based) at 9.7 ms, against YOLO26-pose-X at 71.0 AP at 9.8 ms.

## Training and Fine-Tuning

RF-DETR supports training for object detection, instance segmentation, and keypoint detection. The training pipeline is driven by YAML configs, one per model size and task, which set the backbone, decoder, resolution, augmentations, optimizer, and loss weights. The backbone is initialized from DINOv2 pretrained weights, the model is assembled from backbone, decoder, and task head, and the data loader applies augmentations like mosaic, mixup, and resize.

![RF-DETR training and fine-tuning pipeline](/assets/img/diagrams/rf-detr/rfdetr-training-pipeline.svg)

The forward pass produces task-specific predictions, and the loss is a combination of generalized IoU for boxes, classification loss, mask loss for segmentation, and OKS-based loss for keypoints. The optimizer is AdamW with a learning rate schedule, and training callbacks handle checkpointing, early stopping, and logging. The output is a checkpoint file and metrics. A user can fine-tune on a custom dataset in [Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb) or directly on the Roboflow platform.

## Export and Deployment

RF-DETR can export a trained checkpoint to four formats for deployment across cloud, edge, and browser targets. The `rfdetr.export` module produces ONNX for ONNX Runtime and in-browser inference, CoreML for iOS and macOS, TFLite for Android and edge, and ExecuTorch for mobile and edge. For NVIDIA targets, the model can be converted to TensorRT FP16 for the lowest latency on T4 and Jetson.

![RF-DETR export and deployment targets](/assets/img/diagrams/rf-detr/rfdetr-export-deployment.svg)

The export path means a single trained model can serve a cloud inference API, a Jetson edge device, and an in-browser demo without changing the training code. The same `rfdetr` API that runs inference on a server GPU also produces the artifacts needed for mobile and browser deployment.

## Installation

RF-DETR requires Python 3.10 or later. Install the `rfdetr` package with pip.

```bash
pip install rfdetr
```

For the Plus components (RF-DETR-XL and RF-DETR-2XL detection models), install the extension.

```bash
pip install rfdetr[plus]
```

To install from source for the latest unreleased features:

```bash
pip install https://github.com/roboflow/rf-detr/archive/refs/heads/develop.zip
```

## Quick Start

Run detection with a Medium model on a sample image.

```python
import supervision as sv
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRMedium()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
```

For instance segmentation, swap in `RFDETRSegMedium` and use `sv.MaskAnnotator()` instead of `sv.BoxAnnotator()`. For keypoints (preview), use `RFDETRKeypointPreview`.

## Licensing

Licensing is split by component. The open-source `rfdetr` package and the Apache-designated model weights (Nano through Large) are licensed under Apache License 2.0. Plus components, including the `rfdetr_plus` extension and the RF-DETR-XL and RF-DETR-2XL detection models, are licensed under PML 1.0. This split lets a user build and ship Apache-licensed detection and segmentation models for free, while reserving the largest sizes for the commercial Plus tier.

## Acknowledgements

RF-DETR is built upon [LW-DETR](https://arxiv.org/pdf/2406.03459), [DINOv2](https://arxiv.org/pdf/2304.07193), and [Deformable DETR](https://arxiv.org/pdf/2010.04159).

## Conclusion

RF-DETR is a pragmatic answer to the accuracy-latency trade-off in real-time vision. By pairing a DINOv2 backbone with a Deformable DETR decoder and three task heads, it covers object detection, instance segmentation, and keypoint detection under one API, with six model sizes that span from a 2.3 ms Nano to a 60.1 AP 2XLarge. The Apache 2.0 base, the Colab fine-tuning path, and the four export formats make it straightforward to train on a custom dataset and deploy from cloud to edge to browser. The source is on GitHub at [roboflow/rf-detr](https://github.com/roboflow/rf-detr), the package is on [PyPI](https://pypi.org/project/rfdetr/), and the documentation is at [rfdetr.roboflow.com](https://rfdetr.roboflow.com).
