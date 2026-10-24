---
layout: post
title: "PaddleOCR: A Source Tour of the Multilingual OCR Toolkit"
description: "Walk through the PaddleOCR source: DB text detection, CRNN-style recognition, the det-cls-rec flow, and the 3.x package restructure that reshaped the code."
date: 2026-10-24
header-img: "img/post-bg.jpg"
permalink: /paddleocr/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/paddleocr/paddlepaddle-paddleocr-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: PaddleOCR, OCR, text detection, DBNet, text recognition, PaddlePaddle
author: "PyShine"
---

Optical character recognition looks deceptively simple from the outside — an image goes in, a string comes out — but the systems that make it work reliably are chains of specialized neural networks. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), the OCR toolkit from Baidu's PaddlePaddle team, is one of the most starred OCR projects on GitHub because it packaged that full chain: a text detector that finds where the words are, a direction classifier that figures out whether a line is upside down, and a recognizer that turns cropped line images into characters across dozens of writing systems, with pretrained models in more than 80 languages.

This post is a source tour, not a tutorial. We read the code on the `main` branch as it exists today, after the 3.x restructure that turned the project from a pile of training scripts into a proper installable Python package. The new `paddleocr` package sits on top of the PaddleX inference runtime and exposes a small, tidy API, while the older training-era code still lives under `ppocr/` and `tools/`. That duality is the most interesting thing about the current tree: you can watch a project mid-migration, with a clean wrapper layer at the front and battle-tested modeling code behind it.

We will walk the repository from the public API inward: first the package surface and the CLI, then the DB text detector with its differentiable binarization head, then the CRNN- and SVTR-style recognition stack, and finally the end-to-end det→cls→rec flow that stitches all three stages together — both in the new package and in the legacy 2.x inference scripts that still ship in `tools/infer/`. Two diagrams generated from the actual module graph anchor the tour.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/paddleocr/paddlepaddle-paddleocr-overview-architecture.svg" alt="PaddleOCR overview architecture: public API, detection, recognition, direction classification, runtime and CLI" style="min-width:900px;"></div>

<p><em>Overview of the PaddleOCR 3.x package: the public API in <code>paddleocr/__init__.py</code>, the per-model wrappers, and the runtime plumbing they share.</em></p>

Reading the overview from left to right:

- [`PaddleOCR`, L36](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/__init__.py#L36) — the one class most users ever import. The package's `__init__` re-exports it along with every standalone model class, so `from paddleocr import PaddleOCR` is the entire integration surface for most projects.
- [`TextDetection`, L27](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/__init__.py#L27) and [`TextRecognition`, L30](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/__init__.py#L30) — the two stages of the OCR chain, exportable as independent models if you want to run detection without recognition or vice versa.
- [`class TextDetection`, L24](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_models/text_detection.py#L24) — the DB-style detector wrapper. In 3.x this is a thin subclass over the shared predictor wrapper, not a reimplementation of the network.
- [`class TextRecognition`, L23](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_models/text_recognition.py#L23) — the recognizer wrapper, same pattern: model naming, download and predictor construction are inherited, only model-specific defaults differ.
- [`TextLineOrientationClassification`, L21](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_models/textline_orientation_classification.py#L21) — the line-level 0/180° classifier that rescues text detected upside down.
- [`PaddleXPredictorWrapper`, L30](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_models/base.py#L30) — the shared base class that owns predictor lifecycle, so each model file stays short.
- [`parse_common_args`, L38](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_common_args.py#L38) — resolves `device`, precision, TensorRT and MKLDNN options into the engine config that `prepare_common_init_args` hands to the predictor.
- [`TextSystem`, L48](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_system.py#L48) — the legacy 2.x orchestrator, still in the tree, which chains detector, angle classifier and recognizer by hand.

## From Install to Your First Model

Installation is two package names. The engine comes first, then the toolkit:

```bash
python -m pip install "paddleocr[all]"
```

The `[all]` extra pulls in the document-parsing dependencies; a minimal recognition-only install needs far less. The quick-start documentation in the repo (`docs/quick_start.en.md`) shows the canonical first run:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

result = ocr.predict("./general_ocr_002.png")
```

The three boolean flags are a map of the OCR chain: `use_doc_orientation_classify` enables a whole-page rotation model (the [`DocImgOrientationClassification`, L21](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_models/doc_img_orientation_classification.py#L21) wrapper), `use_doc_unwarping` enables curved-document correction, and `use_textline_orientation` enables the per-line 0/180° classifier. Disable all three and `predict` runs pure detection plus recognition — the fastest path, and the clearest view of the core det→rec flow.

The same job from the shell:

```bash
paddleocr ocr -i ./general_ocr_002.png \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False
```

The CLI is built in [`paddleocr/_cli.py`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_cli.py#L282): a root `argparse` parser registers one subcommand per model class, each contributed through the abstract base at [`CLISubcommandExecutor`, L18](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_abstract.py#L18). That is why `paddleocr text_detection` and `paddleocr text_recognition` work as standalone commands — every wrapper registers itself, and the CLI never hardcodes a model list. Notably, recent releases let those subcommands run with `--engine transformers`, executing some models through the Hugging Face stack instead of PaddlePaddle.

The first `predict()` call triggers model downloads: the runtime resolves a model name to a hosted archive, fetches and unpacks it into a cache directory, and only then constructs the predictor. Engine defaults such as `DEFAULT_DEVICE`, `DEFAULT_PRECISION` and the MKLDNN cache size live in [`paddleocr/_constants.py`, L15](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_constants.py#L15), so tuning startup behavior on CPU-only machines is a matter of the `device` and `enable_mkldnn` arguments rather than source edits.

## Text Detection: Differentiable Binarization

PaddleOCR's default detector is a DB — differentiable binarization — network, and the clearest way to understand DB is to read its head. The modeling code behind it lives under `ppocr/modeling/`, the training-era half of the repo:

- [`class DBFPN`, L116](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/necks/db_fpn.py#L116) — the feature-preserving FPN neck that fuses multi-scale backbone features; detection needs both fine strokes and paragraph-scale context.
- [`class DBHead`, L168](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/heads/det_db_head.py#L168) — produces three maps: a probability map from the `binarize` head (built at [L180](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/heads/det_db_head.py#L180)), a threshold map from the `thresh` head, and approximate binary maps computed by the `step_function` at [L215](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/heads/det_db_head.py#L215).

The trick in the name: classic segmentation methods threshold a probability map at the end, a non-differentiable step that pushes all supervision through one hard bottleneck. DB instead learns the threshold itself and makes the binarization step soft inside the network, so both maps get gradient signal during training. The loss that does it, [`class DBLoss`, L29](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/losses/det_db_loss.py#L29), combines Dice loss on the probability map with an L1-style loss on the threshold map.

At inference the maps become rectangles. [`class DBPostProcess`, L29](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/db_postprocess.py#L29) thresholds the probability map, runs [`boxes_from_bitmap`, L109](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/db_postprocess.py#L109) to find contours, "unclips" each polygon outward to recover the full text region that the shrink-map training target deliberately under-covers, and rescales coordinates back to the original image. Curved text polygons are filtered through locality-aware NMS in [`ppocr/postprocess/locality_aware_nms.py`](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/locality_aware_nms.py), which merges overlapping boxes in a way that respects their shape rather than treating them as axis-aligned rectangles.

The legacy inference script [`TextDetector`, L36](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_det.py#L36) shows exactly how these pieces connect at runtime: it wires `postprocess_params["name"] = "DBPostProcess"` at [L77](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_det.py#L77), loads the exported static model, and returns boxes plus scores. If your task needs boxes around things that are not text, the same detect-then-crop shape appears throughout vision — see our [YOLOv5](/yolov5/) tour for the object-detection cousin.

## Recognition Heads: From CRNN to SVTR

Recognition is the older art here, and the repo preserves its whole evolution. The classic CRNN recipe still backs the default Chinese models: a CNN backbone turns a cropped line image into a feature sequence, a sequence neck flattens it, and a linear head classifies every timestep against a dictionary. Three files tell that story:

- [`class Im2Seq`, L33](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/necks/rnn.py#L33) — the "image to sequence" step, with [`BidirectionalLSTM`, L59](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/necks/rnn.py#L59) as the context encoder that made CRNN CRNN.
- [`class CTCHead`, L35](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/heads/rec_ctc_head.py#L35) — a per-timestep linear classifier. Combined with [`CTCLoss`, L23](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/losses/rec_ctc_loss.py#L23), it trains without character-level alignment: CTC sums over all monotonic alignments between the image and the label, so you never need to know which column of pixels is which letter.
- [`class CTCLabelDecode`, L201](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/rec_postprocess.py#L201) — greedy decoding at inference: collapse repeated timesteps, drop the blank symbol, then map indices through a dictionary. The parent class [`BaseRecLabelDecode`, L23](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/postprocess/rec_postprocess.py#L23) carries the character-map plumbing; the default Chinese dictionary ships as `ppocr/utils/ppocr_keys_v1.txt`, a roughly 6,600-character table that defines what the model can possibly output.

The modern upgrade is SVTR — a single-visual-model recognizer that borrows the ViT recipe for text. [`class SVTRNet`, L419](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/rec_svtrnet.py#L419) mixes local convolutional blocks with global attention blocks, keeping CNN speed where locality suffices and spending transformer capacity on full-line context. Configurations like `configs/rec/rec_svtrnet.yml` select it like any other backbone, because [`class BaseModel`, L27](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/architectures/base_model.py#L27) assembles backbone, neck and head from a YAML architecture description.

## The det→cls→rec Chain, Old and New

Both generations implement the same three-step dance, but the legacy scripts make it wonderfully explicit. [`TextSystem`, L48](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_system.py#L48) constructs a `TextDetector`, a `TextClassifier` (the angle model, defined at [L38](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_cls.py#L38)) and a `TextRecognizer` ([L39](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_rec.py#L39)), and its `__call__` at [L76](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/infer/predict_system.py#L76) reads like pseudocode: detect boxes, crop each region, flip the crops the classifier flags as upside down, batch them through the recognizer, then sort detections top-to-bottom before joining the strings.

The 3.x package implements the identical logic behind the `PaddleOCR` class, which composes the same three model wrappers plus optional document-level preprocessing. The stages are unchanged; what changed is who owns the plumbing. Model naming, download, predictor construction and result formatting moved into the shared wrapper layer, and legacy option names survive through shims like [`warn_deprecated_param`, L36](https://github.com/PaddlePaddle/PaddleOCR/blob/main/paddleocr/_utils/deprecation.py#L36), which translate old kwargs while printing a migration warning instead of failing.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/paddleocr/paddlepaddle-paddleocr-architecture.svg" alt="PaddleOCR detail architecture: 3.x package, legacy infer chain, DB detection modeling, recognition modeling, postprocess and training" style="min-width:900px;"></div>

<p><em>The detail view: the 3.x package on the left, the legacy 2.x infer chain on top, and the shared modeling, postprocess and training code underneath.</em></p>

## Weights, Checkpoints and the Static Graph

Because the training code still lives in-tree, you can follow a model from YAML config to deployed artifact without leaving the repository. A training run enters through [`main`, L46](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/train.py#L46), builds the architecture via `BaseModel`, and iterates loaders built by [`build_dataloader`, L83](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/data/__init__.py#L83) over datasets read by [`class SimpleDataSet`, L172](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/data/simple_dataset.py#L172), which applies the `imaug` augmentation chain per sample. Checkpointing goes through [`load_pretrained_params`, L199](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/save_load.py#L199) and [`save_model`, L241](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/save_load.py#L241), and [`main`, L28](https://github.com/PaddlePaddle/PaddleOCR/blob/main/tools/export_model.py#L28) freezes the trained dynamic graph into the static model files that both generations of predictors actually execute.

That static-graph contract is also the portability story. The documentation describes converting exported models to ONNX, after which they run anywhere ONNX does — our [ONNX](/onnx/) and [ONNX Runtime](/onnxruntime/) tours cover that side of the fence. The pretrained models fetched on first run are exactly these exported artifacts, hosted centrally so users never touch the training stack.

## Honest Limits

- **Chinese and English come first.** The default PP-OCR models are trained primarily on Chinese and English, and the default Chinese recognizer's output space is bounded by its ~6,600-character dictionary. Dozens of other languages ship dedicated dictionaries and models, but quality varies considerably by script, and some require a non-default `lang` parameter.
- **3.x broke the 2.x API.** The old `ocr.ocr()` call is deprecated, `use_angle_cls` became `use_textline_orientation`, the `det`/`rec`/`cls` boolean flags became per-feature options, and the result format changed from nested lists to dict-style objects. The shims translate and warn, but real code needs editing — the legacy scripts in `tools/infer/` remain the reference for old behavior.
- **The PaddlePaddle runtime is the price of admission.** The core path requires PaddlePaddle, a full deep-learning framework that dwarfs the OCR package itself. The newer `--engine transformers` option covers some models without Paddle, but only partially.
- **First run is a download event.** Each stage of the chain — document orientation, unwarping, detection, line classification, recognition — triggers its own model fetch, so a fully enabled setup pulls several archives before the first result appears. Air-gapped environments must pre-seed the model cache by hand.

For text specifically, though, the package remains the strongest open-source starting point we know: the detection stack is genuinely good at scene text, the multilingual model zoo is unmatched in breadth, and — as this tour shows — the whole chain is readable in an afternoon. If your boxes contain objects rather than letters, [YOLOv5](/yolov5/) is the sibling tool; if you want the exported models on the cheapest possible hardware, start with [ONNX Runtime](/onnxruntime/) or [ncnn](/ncnn/); and our [Transformers](/transformers/) tour pairs naturally with the `--engine transformers` story above.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "PaddleOCR: A Source Tour of the Multilingual OCR Toolkit",
  "description": "Walk through the PaddleOCR source: DB text detection, CRNN-style recognition, the det-cls-rec flow, and the 3.x package restructure that reshaped the code.",
  "datePublished": "2026-10-24",
  "author": {"@type": "Person", "name": "PyShine"},
  "publisher": {"@type": "Organization", "name": "PyShine"},
  "keywords": "PaddleOCR, OCR, text detection, DBNet, text recognition, PaddlePaddle",
  "about": ["OCR", "PaddlePaddle", "PaddleOCR"]
}
</script>
