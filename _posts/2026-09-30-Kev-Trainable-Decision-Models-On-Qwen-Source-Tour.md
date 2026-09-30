---
layout: post
title: "Kev: Trainable Decision Models on Qwen - Inside jaredpalmer/kev"
description: "Kev is a family of small, open decision models trained on top of Qwen3.5 and Qwen3.8 base models, answering yes/no, multiple-choice and rating questions through a TypeSafe-compatible /v1/systemone API. We tour the source: the pointer-head architecture in kev/model.py, the LoRA training loop in kev.train, the calibrated FastAPI server in kev.serve, and the Modal-powered fine-tune and deploy skills. Learn how a rank-16 adapter and a tiny pointer head turn a frozen language-model backbone into a calibrated decision engine you can train and host yourself."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Kev-Trainable-Decision-Models-On-Qwen-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/kev/jaredpalmer-kev-architecture.svg
tags:
  - LLM
  - LoRA
  - Qwen
  - Fine-Tuning
categories: [AI, Open Source]
keywords: "Kev, jaredpalmer, decision model, Qwen3.5, Qwen3.8, LoRA fine-tuning, pointer head, System One API, TypeSafe Jev, calibration temperature, Brier score, FastAPI serving, Modal, MLX Apple Silicon, open source AI"
author: "PyShine"
---

Ask a large language model a yes/no question and you get prose you have to parse. Ask it for a probability and you get a number that may not mean anything. Kev, by Jared Palmer, takes a different route: it starts from a Qwen base model, trains a rank-16 LoRA adapter plus a tiny pointer head on top, and turns the whole thing into a decision engine that answers typed questions — yes/no, multiple-choice, and ordered rating — with calibrated probabilities, in a single forward pass that never generates a word. The project ships four sizes, from a 0.8B model that runs on a laptop to a 27B model that needs an 80 GB GPU.

Kev is a self-hostable member of the "Jev-like" family of decision models. Its API deliberately matches TypeSafe's System One contract, so if you already call Jev through the TypeSafe Python SDK, you can point the same client at a Kev server and keep the rest of your code. Everything is Apache-2.0, and the repo is unusually honest for a model release — its README documents where Kev trails Jev, and the numbers in it are checked in CI against committed reports.

The source is worth a tour because it is a complete, readable training-to-serving system in one Python package. The model, the request shapes, the checkpoint loader, the trainer, the benchmark harness, and the FastAPI server all live in `kev/`, each with a docstring that explains not just what it does but why. Question isolation, temperature calibration, and request batching are all there in a few hundred lines per file.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/kev/jaredpalmer-kev-overview-architecture.svg" alt="Architecture overview of the jaredpalmer/kev repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Kev repository: the model core, the serving stack, the training pipeline, and the evaluation harness that keeps the published numbers honest.*

Reading the overview from left to right: the training group turns labelled data into checkpoints — `kev/data.py` converts public datasets into labelled requests, `kev/train.py` adapts the LoRA and pointer head, and `modal_app.py` runs training studies on rented GPUs — while `kev/checkpoint.py` turns any run directory or Hub repo into a `DecisionModel` from `kev/model.py`. On the serving side, `kev/serve.py` loads a checkpoint through the same loader, validates requests against the shapes in `kev/api.py`, and scores them; the Next.js playground and the Hugging Face Space are clients of that same path. The evaluation group closes the loop: `kev/benchmark.py` scores checkpoints through `kev/predictors.py` and turns the rows into accuracy, Brier and calibration numbers via `kev/metrics.py`.

## Why You Need This

If you build any product that makes routine judgments — routing support tickets, flagging escalation, scoring urgency — you eventually face the same problem: you need probabilities attached to categories, not fluent text. Prompting a general LLM gives you uncalibrated guesses in an unpredictable format, and every model upgrade silently reshapes your output. Kev gives you a fixed, typed contract instead: each question declares its type and options, and the answer comes back as a probability distribution over exactly those options.

The second problem is trust. A routing system that automates decisions needs to know when its confidence is real. Every Kev checkpoint ships with a temperature fitted on held-out data, so the probabilities you threshold on are calibrated by default, and the benchmark tooling reports accuracy, Brier score, calibration error, and the share of decisions you could automate at a given error budget — the number that actually matters when deciding how much work a model can take over from humans.

The third problem is ownership. Hosted decision APIs are convenient until your questions are domain-specific, your data is sensitive, or your volume makes per-call pricing painful. Kev is built to be trained and run by you: fine-tuning starts from a released checkpoint with `--init_from` so you keep what the model already knows, the training format is byte-identical to the API request format, and the same code that produced the released checkpoints runs on your machine or on a rented GPU. The `kev-finetune` coding-agent skill automates the whole loop, and because the eval suites under `evals/` are frozen with checksums — with CI verifying the README's numbers against committed reports — the claims you read have not quietly moved underneath you.

## How It Works

Kev's entire inference side is one idea executed carefully: pack a document and its questions into one token sequence, isolate the questions from each other, and read out probabilities with a pointer head instead of generating text.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/kev/jaredpalmer-kev-architecture.svg" alt="Detailed architecture of the jaredpalmer/kev repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of jaredpalmer/kev: how the model core, serving stack, training pipeline, and evaluation harness connect, from request shapes and the branch mask to CUDA graphs, frozen suites, and parity tests.*

### Understanding the Architecture

**The typed-decision contract.** `kev/api.py` defines three Pydantic request shapes — `Noul` (yes/no), `Choice`, and `Score` — and maps all three onto a single primitive: options. A noul becomes two options named false and true; a choice becomes 1 to 255 named options, each optionally described; a score becomes ordered level descriptions. The module's `render()` flattens strings, objects, and arrays into labeled text the model sees, `to_record()` produces the internal record, and `user_tokens()` in `kev/model.py` rewrites any delimiter-like `<|...|>` sequence in user input before tokenization, so option boundaries are unforgeable.

**The encoder and the branch mask.** In `kev/model.py`, `encode()` packs a record as the state followed by one branch per question, reusing five existing Qwen special tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`, `<|fim_suffix|>`) as the state, question, option, option-end and decide delimiters — no new embedding rows need training; the LoRA adapter adapts their meaning. `branch_mask_batch()` builds an additive block-causal mask where a token may attend to the state and its own question but never to other questions, and each question's position IDs restart just after the state, so the model processes the state once and answers each question independently.

**The pointer head.** Also in `kev/model.py`, `PointerHead` projects each option's `</opt>` hidden state and the question's `<decide>` hidden state into a 256-dimensional space, scores options against the decision token, and applies a softmax. Because `<decide>` comes last in its branch, it can attend to the full option list. The head carries a `temperature` that divides the logits at inference only — training always sees temperature 1, so a fitted value stays meaningful.

**One row per question on hybrid backbones.** Qwen3.5 and Qwen3.8 mix attention layers with recurrent Gated DeltaNet layers, which cannot honor an attention mask. For these hybrid bases — every current Kev — `DecisionModel` runs each question as its own causal row via `rows_of()`, with the state computed once and reused from a cache for every row. `kev/shared_prefix.py` extends that trick to training with gradients, `kev/mlx_model.py` reimplements the same contract on Apple Silicon through mlx-lm's Metal kernels, and `tests/test_model.py` proves the packed and row forms equivalent.

**The checkpoint contract and calibration.** `kev/checkpoint.py` is the one module that knows how a checkpoint becomes a model: a directory or Hub repo containing an `adapter_config.json` is a LoRA adapter applied to the base recorded in `head.pt`; otherwise the full backbone ships in the checkpoint itself. `LoadOptions` controls the knobs — merging the adapter into the base weights, choosing the MLX backend on Apple Silicon, and restoring the temperature fitted by `scripts/calibrate_checkpoint.py`, which the head applies on load. `kev/calibrate.py` can later report what a temperature fitted on *your* workload's rows would do.

**The training loop.** `kev/train.py` fine-tunes the adapter and pointer head together with cross-entropy on the correct answer, holding the rest of the base fixed. It supports soft targets for ambiguous records, an ordinal penalty for score questions, KL anchoring against the base's zero-shot distribution, and a permutation-consistency loss that shuffles choice options. Data arrives through `kev/data.py`, which converts public labelled datasets into requests that go through the same `to_record()` mapping as the API — training and serving see byte-identical text — while `modal_app.py` wraps the trainer into reproducible GPU studies. The end-to-end serving flow is short: a request hits `kev/serve.py`, which validates it through `kev/api.py`, encodes it with the model, batches it with up to 64 other waiting requests on one model thread, replays captured CUDA graphs from `kev/cuda_graphs.py` when the GPU is the bottleneck, and returns per-question probabilities — with the state cached, so asking more questions about a document you have already sent only pays for the questions.

## Advantages

- **Open weights and an open recipe.** Four sizes (0.8B, 4B, 9B, 27B) with model cards listing every training stage, and weights published on Hugging Face and as a GitHub release with SHA-256 checksums.
- **Drop-in Jev compatibility.** The `/v1/systemone` endpoint and the `kev-latest` model name mean the TypeSafe Python SDK works against a local Kev server unchanged.
- **Question isolation by construction.** The branch mask and row layout in `kev/model.py` guarantee questions cannot read each other, and the repo's tests verify the packed and separated forms agree.
- **Calibrated confidence by default.** Each checkpoint carries a temperature fitted on held-out data; `KEV_TEMPERATURE=1` returns raw probabilities when you want them.
- **Laptop-scale training.** `--batch 1 --accum 8` in bf16 fits the 0.8B model on a 4 GB GPU, and `--init_from` starts your fine-tune from a released checkpoint.
- **One-command deployment.** `skills/kev-deploy/scripts/kev_serve.py` deploys your own HTTPS endpoint on Modal that scales to zero when idle, with bearer auth.

## Benefits

- **Probabilities, not prose.** Answers arrive as per-option distributions your code can threshold and audit — the confident cases get automated, the rest go to a person.
- **No generation cost.** Kev is a prefill-only encoder with a pointer head, so latency is measured in tens of milliseconds on data-centre GPUs.
- **Measurement built in.** `kev.benchmark` scores any checkpoint or remote endpoint on frozen suites and reports accuracy, Brier score, calibration error, and selective-prediction coverage; `kev.compare` compares two runs with paired bootstrap intervals.
- **Your data stays yours.** Fine-tuning takes a JSONL file in the same shape as an API request plus a label per question, and `kev.calibrate` reports what a temperature fitted on your rows would do.
- **Honest documentation.** The README and model cards state where Kev trails Jev, what its context limits are, and that option order can still change an answer.
- **A working reference UI.** The Next.js playground and the Gradio Space (`space/app.py`) demonstrate presets, packed-versus-separate comparisons, option permutation, and a chess demo driven entirely by decision questions.

## Usage

Run Kev-4B locally (Python 3.12 or 3.13 with [uv](https://docs.astral.sh/uv/)):

```bash
git clone https://github.com/jaredpalmer/kev.git && cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

Then send it typed questions with a plain `curl`:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "kev-latest",
  "questions": {
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                 "shipping": "Delivery status, delays, lost packages",
                                 "billing": "Charges, invoices, payment problems"}},
    "escalate":    {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }}'
```

Or use the TypeSafe SDK that ships with the `serve` extra:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:8009",
    model="kev-latest",
)
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(
            instructions="What is the customer's tone?",
            criteria={"calm": None, "frustrated": None, "angry": None},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["can wait", "this week", "today"],
        ),
    },
)
print(response.nouls["billing"].noul)
print(response.choices["tone"].choice)
print(response.scores["urgency"].score)
```

Fine-tune on your own labelled JSONL, starting from a released checkpoint:

```bash
uv run python -m kev.train --data train.jsonl --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b \
    --epochs 2 --lr 2e-5 --batch 1 --accum 8 --dtype bf16 --checkpointing 1 --device cuda --out runs/mine

uv run python -m kev.benchmark --run runs/mine --data heldout.jsonl --out runs/mine-eval
uv run --extra serve python -m kev.serve --run runs/mine --port 8009
```

Or let a coding agent drive the loop:

```bash
npx skills add jaredpalmer/kev@kev-finetune
```

For a public HTTPS endpoint instead of a local server, deploy the one-file Modal app:

```bash
pip install modal && modal setup
curl -LO https://raw.githubusercontent.com/jaredpalmer/kev/main/skills/kev-deploy/scripts/kev_serve.py
KEV_API_KEY=$(openssl rand -hex 24) modal deploy kev_serve.py
```

## Conclusion

Kev is a rare kind of open-source model release: a focused architecture (a frozen Qwen backbone, a rank-16 LoRA adapter, a pointer head), a typed API other tools already speak, a calibrated-by-default serving stack, and an evaluation harness strict enough that the README's own numbers are machine-checked. Whether you want a drop-in replacement behind an existing System One client, a decision model fine-tuned on your own routing rules, or a well-documented example of how pointer heads and block-causal masking turn a language model into a classifier, the source repays reading — starting with `kev/model.py`.

Links:

- [jaredpalmer/kev on GitHub](https://github.com/jaredpalmer/kev)
- [Kev model cards](https://github.com/jaredpalmer/kev/tree/main/docs/model-cards)
- [Kev on Hugging Face Spaces (live demo)](https://huggingface.co/spaces/jaredpalmer/kev)
- [TypeSafe System One API reference](https://docs.typesafe.ai/api)
