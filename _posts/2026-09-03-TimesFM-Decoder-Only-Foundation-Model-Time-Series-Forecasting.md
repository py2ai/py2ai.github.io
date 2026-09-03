---
layout: post
title: "TimesFM: A Decoder-Only Foundation Model for Time-Series Forecasting"
description: "Google Research's TimesFM treats time-series forecasting like language modeling — a patched decoder-only transformer pretrained on a massive time-series corpus that achieves near-SOTA zero-shot performance across diverse datasets."
date: 2026-09-03
header-img: "ai-coding-frameworks/ai-coding-frameworks"
permalink: /timesfm-decoder-only-foundation-model-time-series-forecasting/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [TimesFM, Google Research, Time Series, Forecasting, Foundation Model, Transformer, PyTorch, JAX]
author: PyShine
---

# TimesFM: A Decoder-Only Foundation Model for Time-Series Forecasting

Large language models transformed NLP by pretraining on massive text corpora and then generalizing zero-shot to unseen tasks. Google Research's **TimesFM** applies the same recipe to time-series forecasting: pretrain a patched decoder-only transformer on a large time-series corpus, and the model can forecast unseen time series zero-shot — with accuracy approaching state-of-the-art supervised forecasting models trained individually on each dataset.

![TimesFM Architecture](/assets/img/diagrams/timesfm/timesfm-architecture.svg)

## The Key Idea: Patching Makes Time Series Like Text

The central insight is that a **time-series patch** can serve the same role as a **text token** in an LLM. Instead of processing individual time steps, TimesFM splits the input series into fixed-size patches (patch_len=32), embeds each patch into a hidden dimension, and feeds the resulting token sequence into a decoder-only transformer — exactly like GPT processes text.

![Patching Mechanism](/assets/img/diagrams/timesfm/timesfm-patching-mechanism.svg)

This analogy runs deep:

| LLM Concept | TimesFM Equivalent |
|------------|-------------------|
| Text token | Time-series patch (32 time steps) |
| Predict next token | Forecast next patch(es) |
| Decoder-only transformer | Same (causal self-attention) |
| Pretrain on text corpus | Pretrain on time-series corpus |
| Zero-shot text generation | Zero-shot forecasting |
| Context length (e.g. 4k tokens) | Context length (up to 16k time steps) |

The benefit: the same architectural innovations that made LLMs successful — scaling, pretraining, decoder-only generation — transfer directly to time-series forecasting.

## Version Evolution: Smaller But More Capable

TimesFM has evolved through three versions, and the trajectory is counterintuitive: the latest model is **smaller** but **more capable**.

![Version Evolution](/assets/img/diagrams/timesfm/timesfm-version-evolution.svg)

| Version | Parameters | Context Length | Key Changes |
|---------|-----------|----------------|------------|
| 1.0 (2024) | 200M | 2,048 | Initial patched decoder, ICML 2024 paper |
| 2.0 (2024-2025) | 500M | 2,048 | Scaled up, frequency indicator, Jax + PyTorch |
| **2.5 (Sept 2025)** | **200M** | **16,384** | Scaled down, 8x longer context, quantile head, no frequency indicator |

TimesFM 2.5 is the current recommended version. The reduction from 500M to 200M parameters — while simultaneously extending context from 2k to 16k — suggests that longer context matters more than raw parameter count for time-series forecasting. This aligns with the intuition that seeing more history leads to better predictions.

### New in 2.5

- **Continuous quantile forecast**: An optional 30M quantile head produces not just point forecasts but full quantile forecasts (mean, 10th through 90th quantiles) up to a 1k-token horizon.
- **No frequency indicator**: The model no longer needs to be told the sampling frequency of the time series — it infers this from the data itself.
- **Covariate support (XReg)**: External regressors can be incorporated via the XReg module (added October 2025).
- **Fine-tuning with LoRA**: HuggingFace Transformers + PEFT integration enables parameter-efficient fine-tuning.
- **Flax version**: A JAX/Flax implementation provides faster inference and TPU support.
- **HuggingFace Transformers**: Native integration as `google/timesfm-2.5-200m-transformers` (237k downloads).
- **Agent skill**: The repository includes an `AGENTS.md` entry point and `SKILL.md` for agent-driven forecasting workflows.

## Architecture: Patched Decoder-Only Transformer

The architecture is deliberately simple and LLM-like:

1. **Patching layer**: Split input time series into non-overlapping patches of length 32.
2. **Input embedding**: Linear projection from patch dimension to hidden dimension, plus positional encoding.
3. **Stacked transformer decoder**: Self-attention with causal masking over the patch sequence. 200M parameters.
4. **Output projection**: Linear layer mapping hidden states to patch-level forecasts.
5. **Optional quantile head**: 30M parameter head for continuous quantile forecasts.
6. **Unpatching**: Reshape patch forecasts into point and quantile forecast arrays.

The model is compiled with a `ForecastConfig` that controls inference behavior:

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,           # max history length
        max_horizon=256,            # max forecast horizon
        normalize_inputs=True,      # normalize each input series
        use_continuous_quantile_head=True,  # enable quantile forecasts
        force_flip_invariance=True,  # invariant to time reversal
        infer_is_positive=True,      # clamp negative forecasts to 0
        fix_quantile_crossing=True,  # ensure quantiles don't cross
    )
)

point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),       # series 1
        np.sin(np.linspace(0, 20, 67)),  # series 2
    ],
)

# point_forecast.shape  -> (2, 12)         batch=2, horizon=12
# quantile_forecast.shape -> (2, 12, 10)   mean + 10th to 90th quantiles
```

## Deployment Options

TimesFM can be deployed locally or through Google's first-party products.

![Deployment Options](/assets/img/diagrams/timesfm/timesfm-deployment-options.svg)

### Local Deployment

Three local paths cover different use cases:

**PyTorch (native)** — the simplest path for GPU inference:
```bash
pip install timesfm[torch]
```

**Flax (JAX)** — faster inference, TPU and Apple Silicon support:
```bash
pip install timesfm[flax]
```

**HuggingFace Transformers** — native integration, LoRA fine-tuning, 237k downloads:
```python
# The transformers-compatible checkpoint
# google/timesfm-2.5-200m-transformers
```

**XReg covariates** — for multivariate forecasting with external regressors:
```bash
pip install timesfm[xreg]
```

### Google First-Party Products

TimesFM is deployed inside three Google products:

- **[BigQuery ML](https://cloud.google.com/bigquery/docs/timesfm-model)** — enterprise-level SQL queries for scalability and reliability.
- **[Google Sheets](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html)** — forecasting in your daily spreadsheet via Connected Sheets + BigQueryML.
- **[Vertex Model Garden](https://pantheon.corp.google.com/vertex-ai/publishers/google/model-garden/timesfm)** — dockerized endpoint for agentic calling.

### Fine-Tuning

Fine-tuning is supported via HuggingFace Transformers + PEFT (LoRA). See the [`timesfm-forecasting/examples/finetuning/`](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning) directory for examples.

## Installation

```bash
# From PyPI
pip install timesfm[torch]       # PyTorch backend
pip install timesfm[flax]        # Flax/JAX backend
pip install timesfm[xreg]        # + covariate support

# From source (for development)
git clone https://github.com/google-research/timesfm.git
cd timesfm
uv venv
source .venv/bin/activate
uv pip install -e .[torch]
```

## Checkpoints

| Model | Parameters | Backend | HuggingFace |
|-------|-----------|---------|-------------|
| TimesFM 1.0 | 200M | JAX | [google/timesfm-1.0-200m](https://huggingface.co/google/timesfm-1.0-200m) |
| TimesFM 1.0 | 200M | PyTorch | [google/timesfm-1.0-200m-pytorch](https://huggingface.co/google/timesfm-1.0-200m-pytorch) |
| TimesFM 2.0 | 500M | JAX | [google/timesfm-2.0-500m-jax](https://huggingface.co/google/timesfm-2.0-500m-jax) |
| TimesFM 2.0 | 500M | PyTorch | [google/timesfm-2.0-500m-pytorch](https://huggingface.co/google/timesfm-2.0-500m-pytorch) |
| **TimesFM 2.5** | **200M** | **PyTorch** | [**google/timesfm-2.5-200m-pytorch**](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| **TimesFM 2.5** | **200M** | **Flax** | [**google/timesfm-2.5-200m-flax**](https://huggingface.co/google/timesfm-2.5-200m-flax) |
| **TimesFM 2.5** | **200M** | **Transformers** | [**google/timesfm-2.5-200m-transformers**](https://huggingface.co/google/timesfm-2.5-200m-transformers) |

Older versions (1.0, 2.0) are archived in the `v1` subdirectory. To install the legacy package: `pip install timesfm==1.3.0`.

## Key Design Decisions

**Why decoder-only (not encoder-decoder)?** A decoder-only architecture lets the model generate forecasts autoregressively — predicting the next patch given all previous patches. This mirrors how LLMs generate text and enables the same pretraining strategy (next-token prediction applied to patches).

**Why patches instead of individual time steps?** Processing individual time steps would make the sequence length impractically long. A patch length of 32 reduces the sequence length by 32x, making attention tractable while still capturing local temporal patterns within each patch.

**Why scale down from 500M to 200M?** Longer context (16k vs 2k) appears to matter more than parameter count for forecasting quality. A smaller model with longer context outperforms a larger model with shorter context — the same lesson the LLM community learned with context window extensions.

**Why drop the frequency indicator?** In v2.0, the model was told whether the data was hourly, daily, weekly, etc. In v2.5, the model infers this from the patch patterns themselves — removing a manual input and improving generality.

**Why is zero-shot forecasting important?** Traditional forecasting models (ARIMA, Prophet, etc.) need to be fit to each time series individually. A pretrained foundation model can forecast unseen series zero-shot — no fitting required — which matters when you have thousands or millions of time series to forecast (e.g., retail demand across all SKUs).

## Further Reading

- [Paper: A decoder-only foundation model for time-series forecasting (ICML 2024)](https://arxiv.org/abs/2310.10688)
- [Google Research blog post](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [TimesFM HuggingFace Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6)
- [BigQuery ML TimesFM docs](https://cloud.google.com/bigquery/docs/timesfm-model)
- [Google Sheets integration](https://workspaceupdates.googleblog.com/2026/02/forecast-data-in-connected-sheets-BigQueryML-TimesFM.html)
- [GitHub: google-research/timesfm](https://github.com/google-research/timesfm) — Apache-2.0 license

## Summary

TimesFM demonstrates that the decoder-only transformer recipe from NLP transfers to time-series forecasting. By treating patches of time steps as tokens, the model can be pretrained on a large time-series corpus and then forecast unseen series zero-shot — approaching the accuracy of supervised models trained individually on each dataset. The latest v2.5 release (200M params, 16k context, quantile head, covariate support, LoRA fine-tuning) is available on PyPI, HuggingFace, and inside Google's BigQuery ML, Sheets, and Vertex AI products.
