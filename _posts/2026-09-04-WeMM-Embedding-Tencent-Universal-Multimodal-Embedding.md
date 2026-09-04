---
layout: post
title: "WeMM-Embedding: Tencent's Universal Multimodal Embedding Family from WeChat Vision"
description: "WeMM-Embedding is a family of universal multimodal embedding models from Tencent's WeChat Vision team that provides unified representations for text, images, videos, visual documents, and interleaved multimodal inputs — achieving state-of-the-art on MMEB-v2 and MMEB-v3."
date: 2026-09-04
header-img: "ai-coding-frameworks/ai-coding-frameworks"
permalink: /wemm-embedding-tencent-universal-multimodal-embedding/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [WeMM-Embedding, Tencent, WeChat, Multimodal, Embedding, MRL, MMEB, RAG, vLLM, SGLang]
author: PyShine
---

# WeMM-Embedding: Tencent's Universal Multimodal Embedding Family

Embeddings are the backbone of modern search, retrieval, and RAG systems. But most embedding models handle only text, or at best text and images. Tencent's WeChat Vision team has released **WeMM-Embedding** — a family of universal multimodal embedding models that provides unified representations for text, images, videos, visual documents, and interleaved multimodal inputs. With three model sizes (2B, 4B, 9B) and Matryoshka Representation Learning support, WeMM-Embedding achieves state-of-the-art performance on both MMEB-v2 and MMEB-v3 benchmarks.

![WeMM-Embedding Architecture](/assets/img/diagrams/wemm-embedding/wemm-architecture.svg)

## What Is WeMM-Embedding?

WeMM-Embedding (WeChat Multi-Modal Embedding) is a family of universal multimodal embedding models developed by the WeChat Vision team at Tencent. The key word is **universal**: a single model produces embeddings for five input types — text, images, videos, visual documents, and interleaved multimodal content — all in the same vector space.

All embeddings are obtained from the last-layer hidden state at a dedicated `<embedding>` token position, followed by L2 normalization. Audio input is not currently supported.

## Model Family

| Model | Parameters | Matryoshka Dimensions | HuggingFace |
|-------|-----------|----------------------|-------------|
| **WeMM-Embedding-2B** | 2B | 64, 128, 256, 512, 1024, 2048 | [tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) |
| **WeMM-Embedding-4B** | 4B | 64, 128, 256, 512, 1024, 2560 | [tencent/WeMM-Embedding-4B](https://huggingface.co/tencent/WeMM-Embedding-4B) |
| **WeMM-Embedding-9B** | 9B | 64, 128, 256, 512, 1024, 2048, 4096 | [tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B) |

## Matryoshka Representation Learning (MRL)

One of the most practical features of WeMM-Embedding is **Matryoshka Representation Learning** — the ability to truncate the embedding to a smaller dimension with graceful quality degradation.

![Matryoshka Representation Learning](/assets/img/diagrams/wemm-embedding/wemm-matryoshka.svg)

For a supported dimension `d`, you simply truncate the full embedding and re-normalize:

```python
import torch.nn.functional as F

# Full embedding (e.g., 2048-dim for 2B model)
embedding = model.encode(input)

# Matryoshka at 256 dimensions
embedding_256 = F.normalize(embedding[..., :256], dim=-1)
```

On MMEB-v2, the 2B model at 256 dimensions retains **98.7% of its full-dimensional image and video performance**. This means you can reduce storage by 8x (2048 to 256) with less than 2% quality loss — a critical trade-off for large-scale vector databases.

## Benchmark Performance: State-of-the-Art

WeMM-Embedding achieves state-of-the-art performance on both MMEB-v2 (78 datasets) and MMEB-v3 (190 tasks), outperforming models from Qwen, VLM2Vec, GME, and others.

![Benchmark Performance](/assets/img/diagrams/wemm-embedding/wemm-benchmarks.svg)

### MMEB-v2 Results (78 datasets)

| Model | Size | AVG | Image | Video | VisDoc |
|-------|------|-----|-------|-------|--------|
| VLM2Vec | 2B | 47.8 | 59.7 | 29.0 | 44.0 |
| GME | 2B | 55.4 | 51.9 | 33.9 | 76.8 |
| VLM2Vec-V2 | 2B | 59.3 | 64.9 | 34.9 | 69.2 |
| Qwen3-VL-Embedding | 2B | 73.2 | 75.0 | 61.9 | 79.2 |
| DME-Small (closed) | 2B | 74.8 | 75.9 | 65.6 | 79.9 |
| **WeMM-Embedding** | **2B** | **77.9** | **79.6** | **70.8** | **80.7** |
| **WeMM-Embedding** | **4B** | **79.2** | **80.8** | **72.1** | **82.0** |
| VLM2Vec | 8B | 53.2 | 65.5 | 34.0 | 49.1 |
| GME | 8B | 59.2 | 56.0 | 38.6 | 79.3 |
| Qwen3-VL-Embedding | 8B | 77.8 | 80.1 | 67.1 | 82.4 |
| DME-Medium (closed) | 9B | 78.4 | 79.8 | 70.8 | 82.0 |
| **WeMM-Embedding** | **9B** | **80.6** | **81.9** | **74.3** | **83.3** |

Image and video tasks use Hit@1; visual-document tasks use NDCG@5. Higher is better.

### MMEB-v3 Results (190 tasks)

| Model | Size | V3-All | Text | Agent | MCMR | Audio |
|-------|------|--------|------|-------|------|-------|
| VLM2Vec-V2 | 2B | 38.3 | 24.5 | 28.7 | 4.1 | 0.0 |
| Omni-Embed-Nemotron | 3B | 43.5 | 39.2 | 36.5 | 26.1 | 36.5 |
| E5-Omni | 3B | 44.6 | 26.7 | 36.9 | 31.9 | 30.8 |
| Qwen3-VL-Embedding | 2B | 50.9 | 39.2 | 39.3 | 42.0 | 0.0 |
| **WeMM-Embedding** | **2B** | **56.0** | **45.3** | **45.1** | **42.5** | **0.0** |
| **WeMM-Embedding** | **4B** | **58.2** | **47.9** | **49.0** | **41.9** | **0.0** |
| Qwen3-VL-Embedding | 8B | 53.5 | 42.5 | 38.4 | 38.0 | 0.0 |
| Tianmu-Emb-Uni | 8B | 53.3 | 43.6 | 39.4 | 38.8 | 38.9 |
| **WeMM-Embedding** | **9B** | **59.5** | **48.8** | **51.0** | **49.3** | **0.0** |

V3-All includes 78 MMEB-v2 tasks, 53 text tasks, 47 agent tasks, 11 audio tasks, and MCMR. WeMM-Embedding does not support audio (scored 0.0).

## Serving Backends

WeMM-Embedding can be served four ways, each targeting different use cases.

![Serving Backends](/assets/img/diagrams/wemm-embedding/wemm-serving.svg)

### 1. Transformers (Native)

The simplest path for research and prototyping. The team recommends `transformers==5.2.0` for reproducibility.

```bash
pip install -r requirements.txt

python examples/transformers_inference.py \
  --model /path/to/WeMM-Embedding-2B \
  --image /path/to/image.jpg \
  --video /path/to/video.mp4 \
  --dimension 2048
```

### 2. Sentence Transformers

Ideal for RAG pipelines and vector database integration.

```bash
python examples/sentence_transformers_inference.py \
  --model /path/to/WeMM-Embedding-2B \
  --image /path/to/image.jpg \
  --video /path/to/video.mp4 \
  --dimension 2048
```

`SentenceTransformer` loads the model directly, so a HuggingFace model id like `tencent/WeMM-Embedding-2B` works in place of a local path. The `--dimension` flag selects the MRL dimension.

### 3. vLLM (Production Serving)

Tested with vLLM `0.27.0`:

```bash
MODEL_PATH=/path/to/WeMM-Embedding-2B
vllm serve "$MODEL_PATH" \
  --runner pooling \
  --chat-template "$MODEL_PATH/embedding_chat_template.jinja"
```

A one-command wrapper is available in `scripts/serve_vllm.sh`.

### 4. SGLang (Production Serving)

Tested with SGLang `0.5.9`:

```bash
MODEL_PATH=/path/to/WeMM-Embedding-2B
python scripts/patch_sglang_video.py
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --is-embedding \
  --enable-precise-embedding-interpolation
```

A one-command wrapper is available in `scripts/serve_sglang.sh`.

## Evaluation

The repository includes the official MMEB-v3 evaluation code in `mmeb_v3_eval/`, built on the [TIGER-AI-Lab/VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec) pipeline with a minimal diff:

- Multi-node multi-GPU inference (`torchrun --nnodes=N`)
- A `wemm_embedding` backbone implementing WeMM's preprocessing and batched inference
- Dataset instructions aligned with the released model
- 64-frame video sampling

```bash
cd mmeb_v3_eval
DATA_ROOT=/path/to/MMEB-V3 bash scripts/download_data.sh
MODEL_PATH=/path/to/WeMM-Embedding-2B DATA_BASEDIR=/path/to/MMEB-V3 \
OUTPUT_DIR=exps/wemm_embedding bash scripts/run_eval.sh
```

## Two-Stage Training

According to the [technical report (arXiv:2608.24053)](https://arxiv.org/abs/2608.24053), WeMM-Embedding is trained in two stages:

1. **Large-scale multimodal alignment** — the model learns to map all input types (text, images, videos, visual documents, interleaved content) into a shared embedding space using a large-scale training corpus.
2. **Refinement stage** — curated data, fine-grained relevance supervision, and cross-scale knowledge transfer are used to polish the embeddings and improve performance on diverse downstream tasks.

This two-stage approach is what enables the 2B model to surpass the previous 8B open-source baseline on MMEB-v2, and the 9B model to achieve a new state-of-the-art overall score of 80.6.

## Deployed at Scale across WeChat

WeMM-Embedding is not just a research model — it has been deployed at scale across WeChat's production applications:

- **WeChat Channels** (short video)
- **Official Accounts** (content feeds)
- **Moments** (social feed)
- **E-commerce services**

The technical report reports substantial gains on a 26-task in-house benchmark and consistent improvements across 14 online A/B tests.

## Key Design Decisions

**Why a dedicated `<embedding>` token?** Rather than pooling over all tokens, WeMM-Embedding extracts the hidden state at a dedicated embedding token position. This gives the model a single, consistent location to aggregate information from all input modalities.

**Why L2 normalization?** All embeddings are L2-normalized, making them directly compatible with cosine similarity — the standard metric for vector search and retrieval. This means you can drop WeMM-Embedding vectors directly into any vector database without additional preprocessing.

**Why Matryoshka dimensions?** Different applications need different trade-offs between embedding size and quality. MRL lets you choose the dimension at inference time: 64 dims for a quick filter, 256 for a mid-tier search index, 2048 for maximum accuracy — all from the same model.

**Why no audio?** The WeChat Vision team focused on the visual and text modalities where their expertise lies. Audio support is a natural extension for future work.

**Why 64-frame video sampling?** Videos are sampled at 64 frames during evaluation, providing enough temporal coverage for action recognition and video retrieval tasks without excessive computational cost.

## Installation

```bash
git clone https://github.com/Tencent/WeMM-Embedding.git
cd WeMM-Embedding
pip install -r requirements.txt
```

**Recommended:** `transformers==5.2.0` for inference and reproducibility (newer versions may differ in preprocessing behavior).

## Further Reading

- [GitHub: Tencent/WeMM-Embedding](https://github.com/Tencent/WeMM-Embedding) — Apache 2.0 license
- [Technical Report (PDF)](https://github.com/Tencent/WeMM-Embedding/blob/main/assets/WeMM_Embedding_tech_report.pdf)
- [arXiv: 2608.24053](https://arxiv.org/abs/2608.24053)
- [Performance Overview (PDF)](https://github.com/Tencent/WeMM-Embedding/blob/main/assets/performance-overview.pdf)
- [HuggingFace: tencent/WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B)
- [HuggingFace: tencent/WeMM-Embedding-4B](https://huggingface.co/tencent/WeMM-Embedding-4B)
- [HuggingFace: tencent/WeMM-Embedding-9B](https://huggingface.co/tencent/WeMM-Embedding-9B)
- [VLM2Vec (evaluation pipeline base)](https://github.com/TIGER-AI-Lab/VLM2Vec)

## Citation

```bibtex
@article{wemm-embedding,
  title={WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report},
  author={Junjie Zhou and Ke Mei and Lei Li and Tianyi Wang and Fengyun Rao and Jing Lyu},
  year={2026},
  eprint={2608.24053},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.24053},
}
```

## Summary

WeMM-Embedding is Tencent's answer to the multimodal embedding problem: one model, five input types (text, images, videos, visual documents, interleaved), three sizes (2B, 4B, 9B), and Matryoshka dimensions from 64 to 4096. It achieves state-of-the-art on MMEB-v2 (80.6 AVG at 9B) and MMEB-v3 (59.5 V3-All at 9B), outperforming Qwen3-VL-Embedding, VLM2Vec, GME, and others. With four serving backends (Transformers, Sentence Transformers, vLLM, SGLang), Apache 2.0 licensing, and HuggingFace integration, it is ready for production RAG pipelines, multimodal search systems, and cross-modal retrieval applications.
