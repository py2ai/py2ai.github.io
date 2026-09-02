---
layout: post
title: "Unlimited OCR: Constant KV-Cache for One-Shot Long-Horizon Document Parsing"
description: "Baidu's Unlimited-OCR uses Reference Sliding Window Attention to keep the KV cache constant throughout decoding, enabling transcription of dozens of pages in a single 32K-token forward pass."
date: 2026-09-02
header-img: "ai-coding-frameworks/ai-coding-frameworks"
permalink: /unlimited-ocr-constant-kv-cache-long-horizon-parsing/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [OCR, Baidu, DeepSeek, R-SWA, KV Cache, Document Parsing, LLM]
author: PyShine
---

# Unlimited OCR: Constant KV-Cache for Long-Horizon Parsing

End-to-end OCR models that use a large language model as the decoder have a well-known weakness: as the output sequence grows, the accumulated KV cache drives up memory consumption and progressively slows generation. Humans, by contrast, show no such decline in efficiency when copying long passages. Baidu's **Unlimited-OCR** tackles this gap head-on with a single architectural change — **Reference Sliding Window Attention (R-SWA)** — that keeps the KV cache constant for the entire decoding process, allowing the model to transcribe dozens of pages in a single 32K-token forward pass.

![Unlimited-OCR Overall Architecture](/assets/img/diagrams/unlimited-ocr/ocr-architecture.svg)

## The Problem: Linear KV-Cache Growth

When an LLM decoder generates text autoregressively, each new token appends its key-value pair to the KV cache. After N tokens the cache holds N entries. For short outputs this is fine, but document transcription routinely produces tens of thousands of tokens. The cache grows linearly, memory consumption rises, and generation slows — exactly the opposite of how a human copies text.

## The Solution: Reference Sliding Window Attention

Unlimited-OCR takes DeepSeek-OCR as its baseline and replaces **all attention layers** in the decoder with Reference Sliding Window Attention. R-SWA splits the attention context into two parts:

1. **Reference window** — a fixed-size set of persistent reference tokens that are never evicted. These act as a stable "working memory" anchor.
2. **Sliding window** — the most recent tokens, with older entries evicted in FIFO order once the window is full.

Because the combined size of the reference window plus the sliding window is fixed, the KV cache stays constant for the entire decoding run — regardless of whether the output is 100 tokens or 32,000.

![R-SWA Mechanism](/assets/img/diagrams/unlimited-ocr/ocr-rswa-mechanism.svg)

The result is an attention mechanism whose memory footprint does not grow with output length. Speed stays stable, the model never runs out of memory mid-transcription, and a single 32K-token forward pass can cover dozens of document pages.

Beyond OCR, R-SWA is a general-purpose parsing attention mechanism — it is equally applicable to ASR, translation, and other long-horizon generation tasks.

## Inference: Two Configs and a PDF Pipeline

Unlimited-OCR ships with two image configurations:

| Config | base_size | image_size | crop_mode | Use case |
|--------|-----------|------------|-----------|----------|
| **gundam** | 1024 | 640 | True | Single images needing fine detail (crop tiles) |
| **base** | 1024 | 1024 | False | Multi-page / PDF processing (full image) |

Single images can use either config. Multi-page and PDF processing always use **base**.

For PDFs, the pipeline is straightforward: PyMuPDF (`fitz`) converts each page to a PNG image at 300 DPI, then the model's `infer_multi` method processes all page images in a single pass.

A n-gram repetition prevention step (`no_repeat_ngram_size=35`) with a configurable window (128 for single images, 1024 for multi-page) suppresses the repetition loops that autoregressive decoders sometimes fall into during very long outputs.

![Inference Pipeline](/assets/img/diagrams/unlimited-ocr/ocr-inference-pipeline.svg)

### Quick Start with Transformers

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_name = 'baidu/Unlimited-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()

# Single image — gundam config
model.infer(
    tokenizer,
    prompt='<image>document parsing.',
    image_file='your_image.jpg',
    output_path='output/',
    base_size=1024, image_size=640, crop_mode=True,
    max_length=32768,
    no_repeat_ngram_size=35, ngram_window=128,
    save_results=True,
)

# Multi-page — base config only
model.infer_multi(
    tokenizer,
    prompt='<image>Multi page parsing.',
    image_files=['page1.png', 'page2.png', 'page3.png'],
    output_path='output/',
    image_size=1024,
    max_length=32768,
    no_repeat_ngram_size=35, ngram_window=1024,
    save_results=True,
)
```

### PDF Processing

```python
import tempfile, fitz  # PyMuPDF

def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix='pdf_ocr_')
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f'page_{i+1:04d}.png')
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths

model.infer_multi(
    tokenizer,
    prompt='<image>Multi page parsing.',
    image_files=pdf_to_images('your_doc.pdf', dpi=300),
    output_path='output/',
    image_size=1024,
    max_length=32768,
    no_repeat_ngram_size=35, ngram_window=1024,
    save_results=True,
)
```

## Three Deployment Backends

Unlimited-OCR can be served three ways, each targeting a different use case:

![Deployment Backends](/assets/img/diagrams/unlimited-ocr/ocr-deployment-backends.svg)

### 1. HuggingFace Transformers (Native)

The simplest path. Install the pinned dependencies (Python 3.12, CUDA 12.9, torch 2.10, transformers 4.57) and call `model.infer()` or `model.infer_multi()` directly on GPU. Best for development and prototyping.

### 2. vLLM (Production Serving)

The vLLM community contributed official support with ready-to-use Docker images:

```bash
# Default (CUDA 13.0)
docker pull vllm/vllm-openai:unlimited-ocr

# For Hopper GPUs (CUDA 12.9)
docker pull vllm/vllm-openai:unlimited-ocr-cu129
```

The official recipe is at [recipes.vllm.ai/baidu/Unlimited-OCR](https://recipes.vllm.ai/baidu/Unlimited-OCR). Best for production deployments that need PagedAttention and an OpenAI-compatible API.

### 3. SGLang (Streaming + Batch)

SGLang provides a streaming server with a custom logit processor for the n-gram repetition prevention:

```bash
python -m sglang.launch_server \
  --model baidu/Unlimited-OCR \
  --served-model-name Unlimited-OCR \
  --attention-backend fa3 \
  --page-size 1 \
  --mem-fraction-static 0.8 \
  --context-length 32768 \
  --enable-custom-logit-processor \
  --disable-overlap-schedule \
  --skip-server-warmup \
  --host 0.0.0.0 --port 10000
```

For batch inference, the included `infer.py` script starts the SGLang server automatically and sends concurrent requests:

```bash
# Image directory
python infer.py --image_dir ./examples/images --output_dir ./outputs \
  --concurrency 8 --image_mode gundam

# PDF pages
python infer.py --pdf ./examples/document.pdf --output_dir ./outputs \
  --concurrency 8 --image_mode gundam
```

## Key Design Decisions

**Why keep the reference window fixed?** The reference tokens act as a stable anchor — a "working memory" of the document's overall context. Without them, the model would lose sight of earlier pages when transcribing later ones. With them, page 50 can still reference the structure established on page 1.

**Why replace ALL attention layers?** Partial replacement would leave some layers with growing caches, defeating the purpose. By replacing every decoder attention layer with R-SWA, the entire pipeline maintains a constant memory footprint.

**Why n-gram repetition prevention?** Autoregressive decoders can enter repetition loops during very long generations, especially when the attention context is constrained. The `no_repeat_ngram_size=35` parameter bans any 35-gram from repeating within the configurable window, and the custom logit processor zeroes out the offending logits before sampling.

**Why is R-SWA general-purpose?** The constant-KV-cache property is not specific to OCR. Any task that requires long-horizon autoregressive generation — ASR transcription, long-form translation, code generation — benefits from a decoder whose memory does not grow with output length.

## Availability

| Resource | Link |
|----------|------|
| GitHub | [github.com/baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) |
| HuggingFace model | [huggingface.co/baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) |
| HuggingFace Spaces demo | [huggingface.co/spaces/baidu/Unlimited-OCR](https://huggingface.co/spaces/baidu/Unlimited-OCR) |
| arXiv paper | [arxiv.org/abs/2606.23050](https://arxiv.org/abs/2606.23050) |
| ModelScope | [modelscope.cn/models/PaddlePaddle/Unlimited-OCR](https://modelscope.cn/models/PaddlePaddle/Unlimited-OCR) |
| Baidu Cloud | [cloud.baidu.com/doc/OCR/s/fmr1p39gb](https://cloud.baidu.com/doc/OCR/s/fmr1p39gb) |
| vLLM recipe | [recipes.vllm.ai/baidu/Unlimited-OCR](https://recipes.vllm.ai/baidu/Unlimited-OCR) |
| License | MIT |

## Further Reading

- [DeepSeek-OCR (baseline)](https://github.com/deepseek-ai/DeepSeek-OCR) — the model Unlimited-OCR builds upon
- [DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2) — the next iteration from DeepSeek
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — Baidu's classic OCR toolkit
- [ms-swift](https://github.com/modelscope/ms-swift) — community training support for Unlimited-OCR
- [Unlimited-OCR paper (PDF)](https://arxiv.org/pdf/2606.23050)

## Summary

Unlimited-OCR is a focused, single-idea contribution: replace the decoder's attention with a mechanism whose KV cache does not grow. That one change unlocks long-horizon parsing that stays fast and memory-stable for the entire output — whether it is one page or fifty. The model is MIT-licensed, available on HuggingFace, and deployable via Transformers, vLLM, or SGLang.
