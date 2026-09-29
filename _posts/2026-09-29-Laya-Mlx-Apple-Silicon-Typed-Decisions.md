---
layout: post
title: "Laya-MLX: Typed Decisions at Native Apple Silicon Speed - Inside mizorewww/laya-mlx"
description: "A source tour of mizorewww/laya-mlx, the independent Python runtime that executes Laya typed-decision checkpoints natively on Apple Silicon through MLX. We walk the weight bridge, the ModernBERT reimplementation, the calibration guardrails, and the benchmark harness behind its documented 7-14 ms short decisions with zero output tokens."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Laya-Mlx-Apple-Silicon-Typed-Decisions/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/laya-mlx/mizorewww-laya-mlx-architecture.svg
tags:
  - MLX
  - Apple Silicon
  - Local AI
  - Decision Models
categories: [AI, Open Source]
keywords: "laya-mlx, MLX, Apple Silicon, typed decisions, Laya, ModernBERT, mmBERT, local inference, on-device AI, no autoregressive decoding, M3 Max benchmark, Python runtime, Apache-2.0, Hugging Face checkpoints, decision model"
author: "PyShine"
---

Ask most software for a routing choice, a rubric score, or a yes/no judgment and you will be handed an autoregressive language model that "thinks" one token at a time, wraps its answer in brittle JSON, and phones a data center to do it. The Laya family of models from Convai Innovations takes the opposite bet: a bidirectional encoder reads the state and the question together, and small decision heads emit calibrated probabilities directly — no generated text at all. That idea is compelling, but an idea is only as useful as the hardware it runs on, and the reference engine speaks PyTorch.

**Laya-MLX** (`mizorewww/laya-mlx`) is the bridge for the machine half of us actually carry every day: an independent, Apache-2.0 Python runtime that reimplements Laya's architecture in Apple's MLX framework and executes the original checkpoints natively on Apple Silicon. Its README documents **13.42 ms** median end-to-end for a short English typed decision on an M3 Max (7.39 ms with the multilingual checkpoint), with **zero output tokens**, no PyTorch, no Transformers runtime, and no cloud API. This is not a wrapper around someone else's server; it is a from-scratch neural port with its own benchmarks, validation suite, and performance research.

The source is worth a tour precisely because it is a port done in daylight. Weight-name mapping, RoPE subtleties, calibration quirks, and numerical parity are all handled in small, readable modules, and every latency claim in the README is backed by a JSON file of raw timing samples in the repository. If you have ever wondered what "running a model natively on Metal" actually involves beyond importing a framework, this codebase shows you, module by module.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/laya-mlx/mizorewww-laya-mlx-overview-architecture.svg" alt="Architecture overview of the mizorewww/laya-mlx repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the laya-mlx repository: callers on the tooling side converge on the Agent runtime, which prepares prompts and drives the MLX decision model, with Hugging Face checkpoints and the Rust tokenizer feeding it from the data side.*

Reading the overview from left to right: the tooling and ops lane contains the `laya_mlx/cli.py` command line, the checkpoint converter in `laya_mlx/convert.py`, and the benchmark harness under `benchmarks/`; they all funnel into the runtime API lane, where `Agent` in `laya_mlx/agent.py` is the single load-and-predict surface, flanked by the `Router` for checkpoint selection, the opt-in shortlist helper in `laya_mlx/shortlist.py`, and the Snake demo that calls the agent once per game move. The Agent pulls a checkpoint snapshot from Hugging Face, renders prompts through `laya_mlx/common.py`, tokenizes with the Rust tokenizer in `laya_mlx/tokenizer.py`, optionally reuses tokenized prefixes from `laya_mlx/prepared.py`, and finally executes batched forward passes through the MLX `DecisionModel` in `laya_mlx/model.py` — the neural core lane where all real computation happens.

## Why You Need This

The first problem is latency and shape mismatch. When your application needs a *decision* — which department handles this ticket, how urgent is it on a 1-5 rubric, does the customer want money back — an autoregressive LLM spends its budget generating words before it can emit a structured answer. Laya's typed questions (`choice`, `score`, `noul`) produce probabilities over named options in a single bidirectional forward pass, and question rows are batched independently. laya-mlx keeps that property intact on Apple hardware: the runtime's documented figures include 50-question throughput of 146.8 q/s (English) and 395.0 q/s (multilingual) at `batch_size=64`, measured end-to-end including prompt preparation, tokenization, synchronized inference, calibration, and result formatting.

The second problem is framework weight. The upstream engine and the usual inference path assume PyTorch and Hugging Face Transformers. On a Mac that means installing a multi-gigabyte stack to then run on MPS with varying kernel quality. laya-mlx's runtime dependencies are exactly four packages: `mlx`, `numpy`, `huggingface-hub`, and `tokenizers`. The neural network — ModernBERT/mmBERT encoder plus the decision Transformer and heads — is reimplemented directly in `mlx.nn`, attention runs through `mx.fast.scaled_dot_product_attention` and `mx.fast.rope`, and tokenization uses Hugging Face's Rust tokenizer without ever importing a Transformers model class. Peak MLX allocation for one short question is a documented 943.6 MiB (English) or 687.6 MiB (multilingual).

The third problem is language. The English checkpoint does not gracefully degrade off English — the routing notes in `laya_mlx/router.py` record near-random accuracy on 20-option intent classification for Hindi and Korean, with high reported confidence. Shipping one checkpoint and hoping is not a strategy. The built-in `Router` keeps all three Laya checkpoints (English ModernBERT-large 421M, multilingual mmBERT-base 322M, and the typed-decisions variant) behind one API and uses script detection as the primary routing signal — `laya_mlx/lang.py` even reports its evidence, including `language_undecided` and `diacritic_rate`, so unidentified Latin-script languages like Polish or Turkish route to the multilingual checkpoint on their non-English letters alone rather than being silently assumed English.

The fourth problem is trust. A port is only valuable if it computes the same function as the original. This repository treats that as a first-class engineering deliverable: loading validates every parameter name and shape, the export path refuses to overwrite existing checkpoints, and the published FP16 checkpoints on Hugging Face (under the `aac6fef` account) ship with model cards, provenance, licenses, and checksums — 36 published files that passed strict remote checksum verification per `benchmarks/results/hub-publication.json`.

## How It Works

One sentence sets the stage: every call funnels through a single `Agent` that turns Python state and question dictionaries into token sequences, executes a bidirectional encoder plus decision heads entirely in MLX, and returns calibrated probabilities — the detailed diagram below maps each stage to the file that owns it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/laya-mlx/mizorewww-laya-mlx-architecture.svg" alt="Detailed architecture of the mizorewww/laya-mlx repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of laya-mlx: entry points and the public API on top, the runtime core with prompting, caching, and routing in the middle, the MLX neural core below, and the demo, benchmark, and research tooling that keeps the whole thing honest.*

### Understanding the Architecture

**The native MLX encoder.** `laya_mlx/model.py` contains a complete, inference-only ModernBERT reimplementation: `EncoderConfig` validates the checkpoint's configuration and rejects unsupported encoders and non-default RoPE scaling outright; `EncoderAttention` applies rotary embeddings separately for full-attention and sliding-window layers (distinct RoPE bases, with 160000 global and 10000 local as defaults) and computes attention with MLX's fused kernels; `EncoderLayer` preserves ModernBERT's alternating local/global pattern every three layers, the inclusive sliding-window boundary, and the first-layer normalization behavior that deviates from the standard pre-norm layout.

**The decision heads.** In the same file, `DecisionModel` adds type embeddings per question type, then runs a small decision Transformer (`DecisionHead`) over the encoder output. One detail rewards close reading: `HeadLayer` uses ReLU in its feed-forward block, with a comment explaining that this matches the PyTorch `TransformerEncoderLayer` default even though the encoder itself uses GELU — fidelity over aesthetics. Marker positions index the option slots, a scorer head produces logits masked to real options, and an action head consumes confidence features (top probability, margin, normalized entropy) to reproduce the upstream `act_probability` field.

**The weight bridge.** `sanitize_weights` in `laya_mlx/model.py` maps upstream PyTorch parameter names to MLX conventions (fusing `in_proj_weight`/`in_proj_bias` into module weights, re-nesting `scorer` and `act_head` Sequentials), while `Agent` in `laya_mlx/agent.py` loads the result with `strict=True` so a single missing or renamed tensor fails loudly. `resolve_model` in the same file downloads only the required patterns from a Hugging Face snapshot — weights, agent config, encoder config, tokenizer, `mlx_config.json` — and validates checkpoint completeness before anything else runs.

**Prompting and calibration parity.** `laya_mlx/common.py` carries the upstream-adapted prompt construction: states serialize from text, JSON dictionaries, or conversation lists; options render in label-index order with structured criteria converted to compact JSON; each option occupies a mask-token slot within the shared `head_max_len` budget. Calibration follows upstream v0.3.5: fitted temperatures are clamped to `[0.5, 5.0]` before use, because the shipped `choice:11+` bucket of 0.1006 would sharpen logits roughly tenfold and report a coin flip as near-certainty. The raw checkpoint values remain inspectable as `agent.temperature_raw`, and a `RuntimeWarning` names every clamped bucket at load.

**The performance path.** `laya_mlx/prepared.py` implements a bounded prefix cache (128 questions) that reuses the tokenized question prefix while sharing CPU state tokenization — encoder states and predictions are never cached, so every question still gets its own encoder computation. Combined with the opt-in `compile=True`, `pad_to_multiple=16`, and `batch_size` chunking, this is the measured fast path: the documented Snake ablation shows 75.40 moves/s across 2,400 moves with the optimized settings, about 6.5% faster than the same-run eager control. For choice sets with hundreds of labels, `laya_mlx/shortlist.py` embeds the state and each label (mean-pooling the already-loaded encoder via `embed_fn_from_agent`, or accepting a dedicated bi-encoder), keeps the top-k by cosine similarity, and scores only the reduced set.

**Proof, not vibes.** The `benchmarks/` package measures each backend and checkpoint in a fresh process (`run.py`), validates numerical parity and stability with 100 repeated calls (`validate.py`), scores accuracy (`accuracy.py`), and assembles the report (`report.py`), with every timing sample stored as JSON under `benchmarks/results/`. The result documented in `BENCHMARKS.md`: all three checkpoints matched the upstream selected answer on 63/63 validation questions in both FP32 and FP16 — 378/378 comparisons — with zero measured active-memory growth across repeats.

End to end, one call looks like this: `agent.predict(state, questions)` validates each question definition, serializes the state, tokenizes with the Rust tokenizer, builds sequences with mask-token option markers inside the token budget, collates them into padded batches, runs the MLX encoder and decision heads on the GPU with a synchronous `mx.eval`, applies clamped temperature calibration and four-decimal rounding, and returns an answers dictionary — with `usage.output_tokens` permanently zero, because nothing was ever decoded token by token.

## Advantages

- **No autoregressive decoding.** Typed decisions complete in one bidirectional forward pass with zero output tokens; the documented P50 is 13.42 ms for the English 421M checkpoint and 7.39 ms for the multilingual 322M checkpoint on an M3 Max.
- **A genuinely lean native stack.** The runtime requires only `mlx`, `numpy`, `huggingface-hub`, and `tokenizers` — no PyTorch, no Transformers model classes — with attention and RoPE dispatched through MLX's fast Metal kernels.
- **Port fidelity as a feature.** Strict weight-name and shape validation, deliberate architectural detail preservation, and documented 378/378 parity with upstream selected answers across FP32 and FP16, plus deterministic repeated outputs.
- **Checkpoint intelligence built in.** The `Router` selects between English, multilingual, and typed-decisions checkpoints using script detection with reported evidence, guards model lifecycle with a re-entrant lock for concurrent threads, and keeps the specialized typed-decisions checkpoint strictly opt-in.
- **Measured everything.** A dedicated benchmark harness stores every timing sample in the repository; performance research notes in `docs/` (including `PERFORMANCE_RESEARCH.md`, `MATH_10X_RESEARCH.md`, and `ENGINEERING_10X_RESEARCH.md`) document compilation, quantization, and custom Metal kernel experiments down to their vendor headers in `experiments/engineering/vendor/`.
- **Honest failure modes.** Non-finite outputs raise a `FloatingPointError` with a retry suggestion, unsupported encoders fail at load, calibration clamping warns with specifics, and the research notes openly conclude that the investigated optimizations do not support a further universal 10× speedup on the same checkpoints.

## Benefits

- **Privacy and offline operation.** After the first checkpoint download, inference is fully local — no cloud API, no telemetry surface in the runtime path, and the Snake demo even runs with `HF_HUB_OFFLINE=1` against a cached snapshot.
- **Throughput you can tune.** `batch_size` caps questions per forward pass (default 16, larger requests chunk automatically), and the documented 50-question runs at `batch_size=64` reach 146.8 and 395.0 questions per second depending on checkpoint.
- **Production scaffolding included.** `laya_mlx/presets.py` ships ready question sets for triage, email, guard, moderation, and router workflows, while `laya_mlx/email.py` strips quotes, signatures, and disclaimers before the state ever reaches the model.
- **Reproducibility out of the box.** Pin a Hub revision, select a subfolder from the bundled repository, and export with `laya_mlx/convert.py`, which writes `model.safetensors`, both configs, tokenizer files, and a `mlx_config.json` provenance record — and never overwrites an existing output.
- **CI-verified on the target platform.** GitHub Actions runs small-model CPU tests on a macOS arm64 runner (`tests/` includes direct comparisons with Transformers and the pinned upstream decision head), so the port's guarantees are checked continuously, not just on one laptop.
- **A demo that doubles as evidence.** `laya-snake` plays terminal Snake where every move is a real Laya decision with a visible safety shield, and the replay renderer reproduces recorded runs at original speed — the README's GIF is an unretouched capture of that loop.

## Usage

Install from PyPI and make your first typed decision in a few lines:

```bash
pip install laya-mlx
```

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")
result = agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {
        "department": {
            "type": "choice",
            "instructions": "Who should handle this?",
            "criteria": ["billing", "technical", "sales"],
        }
    },
)
print(result["answers"]["department"])
```

Run the terminal Snake demo (download the checkpoint once for offline use):

```bash
pip install 'laya-mlx[demo]'
hf download aac6fef/laya-multilingual-mlx
laya-snake
```

Use the command line for prediction and checkpoint export:

```bash
uv run laya-mlx predict \
  --model aac6fef/laya-mlx \
  --state-file examples/state.json \
  --questions examples/questions.json

uv run laya-mlx convert \
  --model convaiinnovations/laya \
  --dtype float16 \
  --output models/laya-mlx-fp16
```

For development, the README's clone-and-run path:

```bash
gh repo clone mizorewww/laya-mlx
cd laya-mlx
uv sync --extra demo
uv run --extra demo laya-snake
```

The targets are Apple Silicon Macs on macOS 14+ with Python 3.11+; the measured environment in the README is an M3 Max (40 GPU cores, 128 GiB) with MLX 0.32.2.

## Conclusion

laya-mlx is what a serious platform port looks like: a small, legible set of modules — `laya_mlx/model.py` for the MLX reimplementation, `laya_mlx/agent.py` for the runtime contract, `laya_mlx/common.py` for prompt and calibration parity, `laya_mlx/prepared.py` and `laya_mlx/shortlist.py` for the performance path — wrapped in a benchmark and validation apparatus that publishes its raw data. It takes the Laya engine's elegant trick of answering typed questions without decoding a single token, and makes it a first-class citizen of Apple Silicon with documented single-digit-to-low-teens millisecond decisions. If you build decision-making software and your users carry M-series chips, the source tour is short, the benchmarks are auditable, and the checkpoints are already published and checksummed. Read the code, run the Snake demo, and time it yourself.

Links:

- [GitHub: mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)
- [Benchmarks and methodology (BENCHMARKS.md)](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md)
- [Pre-converted FP16 weights on Hugging Face](https://huggingface.co/aac6fef/laya-mlx)
- [Upstream Laya engine by Convai Innovations contributors](https://github.com/NandhaKishorM/laya)
