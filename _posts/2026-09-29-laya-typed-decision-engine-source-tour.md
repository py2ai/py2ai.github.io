---
layout: post
title: "Laya: Typed Decisions in a Single Forward Pass - Inside NandhaKishorM/laya"
description: "A source tour of NandhaKishorM/laya, the Python non-autoregressive System 1 decision engine that answers choice, score and yes-no questions over any state in one forward pass. We walk the typed question schema, the mask-marker scoring head, the sub-millisecond language router, and the serving surfaces that make fast agentic decision-making practical."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /laya-typed-decision-engine-source-tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/laya/nandhakishorm-laya-architecture.svg
tags:
  - Machine Learning
  - Python
  - AI Agents
  - Deep Learning
categories: [AI, Open Source]
keywords: "non-autoregressive decision engine, System 1 AI, laya decision engine, NandhaKishorM laya, ModernBERT classifier, typed decisions, choice score noul questions, language routing, calibrated confidence, RLCD training, single forward pass inference, Hugging Face checkpoints, MCP decision tools, Apache 2.0 AI project"
author: "PyShine"
---

Ask an LLM-backed agent to route a ticket and it will happily generate a paragraph explaining its reasoning, token by token, before you get the one word you actually needed. Laya, maintained by NandhaKishorM under Convai Innovations, takes the opposite path. It is a non-autoregressive "System 1" decision engine: instead of generating text, a bidirectional transformer encoder reads your state and your questions at once, and returns typed answers — a choice among labeled options, a score on a described scale, or a yes/no probability — with calibrated confidences, all in a single forward pass. The README documents 33 ms for one question and 7.2 ms per question when batched, measured on a T4 GPU.

What makes Laya interesting is not just the speed, but the shape of the interface. You declare questions with types and criteria as plain dictionaries, and the engine packs instructions, every option, and the entire state into one sequence where each option carries a [MASK] marker. The model scores those markers in parallel and softmaxes them into per-option probability distributions. No text generation means nothing to parse and, as the project puts it, nothing to hallucinate — the answer space is exactly the option set you declared.

The repository is worth a source tour because the whole pipeline is readable in an afternoon: the decision architecture in one module, the routing policy in another, and honest engineering everywhere you look — strict input validation with error messages that name the offending question, temperature-fitted confidence reported per answer, and an evaluation harness wired into CI as a regression gate. Let us walk it from the entry point down to the scoring head.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/laya/nandhakishorm-laya-overview-architecture.svg" alt="Architecture overview of the NandhaKishorM/laya repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Laya repository: a Router front end that detects script and language, an encoder-backed decision engine that answers typed questions in one forward pass, typed APIs that turn schemas and presets into question sets, and serving surfaces from a CLI to HTTP, MCP and framework integrations.*

Reading the overview from left to right: everything begins at the `Router` in `laya/router.py`, which calls the dependency-free detector in `laya/lang.py` to decide, before any model loads, whether the English checkpoint can read this state. The request then lands in the `Agent` runtime in `laya/agent.py`, which validates and encodes questions through `laya/common.py` — home of both the sequence builder and the `DecisionModel` architecture — applies the abstention gate from `laya/confidence.py`, and can swap its forward pass for the TileLang GPU fast path in `laya/fast.py`. On the API side, `laya/structured.py` turns JSON schemas or pydantic models into question sets and `laya/presets.py` ships ready-made ones. Finally, the ops column shows how the same engine reaches production: the `laya` CLI, the `laya-serve` HTTP server speaking a documented wire protocol, an MCP stdio server, and framework integrations for LangChain, LlamaIndex and CrewAI.

## Why You Need This

Agentic systems are drowning in inference calls that do not need generation. Classifying an incoming support message into a department, rating urgency on a scale, deciding whether a user is threatening to churn — these are decisions with finite answer spaces, yet the default recipe is a generative LLM call plus brittle parsing of whatever prose comes back. Laya replaces that whole pattern with a schema you declare and a distribution you can trust, which is why the repo describes itself as a System 1 engine: fast, reflexive judgment to complement slow, deliberate reasoning elsewhere in your stack.

The second problem is multilingual reliability. The repository's own benchmarks make the point uncomfortably well: the English checkpoint collapses on non-Latin scripts — the README's benchmark section records 0.000 accuracy at 0.952 confidence on Khmer — and because the model stays confident while wrong, confidence gating cannot rescue you after the fact. Laya's answer is to route before the forward pass. `laya/lang.py` detects script and applies a function-word heuristic in pure Python, in well under a millisecond, and non-English text goes to the multilingual checkpoint rather than being silently mangled. The Router's benchmark table shows the payoff: the multilingual checkpoint is usable on 45 of 51 MASSIVE languages versus 23 for the English one, while English intent accuracy stays on the strong checkpoint.

The third problem is trust in the numbers. A classifier that says "billing, 0.94" is only useful if 0.94 means something. Laya fits per-type, per-option-count temperature scaling — with additional per-language temperature maps — and reports an `answer_confidence` field that is the calibrated probability of the reported answer, on every question type. Its benchmark documentation reports an expected calibration error of 0.081 after temperature fitting. And when a deployment wants abstention rather than a guess, the `min_confidence` flag marks weak answers as `low_confidence` and the schema-driven `decide` call returns `None` for them.

Finally, there is the deployment problem. A decision engine is only worth adopting if it fits the surfaces you already run: Laya ships a CLI for one-liners, a FastAPI playground in `examples/server.py`, a Jev-compatible HTTP server in `laya/serve.py`, an MCP server for tool-calling clients, and integrations that expose the same engine to LangChain, LangGraph, LlamaIndex and CrewAI. There is even a TypeScript port, `laya-ts`, published to npm, that mirrors the engine for Node and browser use.

## How It Works

The whole system reduces to one idea: render every question, every option, and the state into a single sequence with a [MASK] marker per option, then let the model score all markers in parallel. The detailed diagram below maps that pipeline onto the actual files.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/laya/nandhakishorm-laya-architecture.svg" alt="Architecture of the NandhaKishorM/laya repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of NandhaKishorM/laya: routing and detection, the Agent runtime and DecisionModel head, the typed question APIs, the serving and integration surfaces, and the eval harness and checkpoint-integrity tooling that gate changes.*

### Understanding the Architecture

**The typed question schema.** Everything starts with three question types defined in `laya/common.py`: `QTYPES = {"choice": 0, "score": 1, "noul": 2}`. A `choice` question takes criteria as a dict of label to description, a `score` question takes an ordered list of level descriptions, and a `noul` question is a boolean with optional custom labels. The validation in `Agent._check_question` (`laya/agent.py`) is remarkably strict — null or duplicate choice labels, null score levels, and mistyped noul criteria are all refused with error messages that name the question and the fix. This is the kind of code that exists because the authors ran a server and got tired of caller errors surfacing as 500s three frames down.

**The sequence format.** `build_sequence` in `laya/common.py` assembles `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`, recording the position of every mask as a marker. Options are capped at 48 tokens each, and the whole question head shares a `head_max_len` budget; when options overflow it, the answer's `usage` block reports exactly which questions lost distinct option spans. The state is tokenized once per call and reused across every question, and conversation lists are truncated from the left so the newest turn survives.

**The DecisionModel head.** The architecture in `laya/common.py` is a bidirectional encoder (ModernBERT-large or mmBERT-base depending on the checkpoint) followed by a two-layer transformer decision head. A `type_emb` embedding tells the head which of the three question types it is reading. Then comes the elegant part: a `scorer` module — LayerNorm, Linear, GELU, Linear — is applied at every marker position, producing one logit per option, which becomes a softmax over exactly your option set. Alongside it, an `act_head` consumes the pooled CLS state plus four statistics (top probability, the top-two margin, normalized entropy, and the option count) to produce an action probability exposed as `act_probability` on every answer.

**Decoding and calibration.** `Agent._decode_answers` in `laya/agent.py` turns logits into typed answers: a `choice` returns the winning label plus full probabilities, a `score` returns the expected score over levels plus a legend, and a `noul` returns P(true). Each row is divided by a temperature drawn from per-question-type, per-option-count buckets — with per-language overrides — and both a distributional `confidence` and the calibrated `answer_confidence` are reported. The scoped OOM fallback is worth noticing too: an oversized request demotes to CPU for that one request under a lock, then restores the GPU runtime.

**Routing before inference.** The `Router` in `laya/router.py` holds three checkpoints — `english` (ModernBERT-large, 421M parameters, 512-token context), `multilingual` (mmBERT-base, 322M parameters, up to 8,192 tokens with `max_len=8192`) and `typed-decisions` — with an LRU over how many stay resident. Its `_route` method applies a clean precedence: explicit model, then task, then an opt-in workflow match against four known typed-decisions question schemas, then explicit language, then a caller-supplied `lang_guess`, then the built-in detector, then a default. Every `RouteDecision` carries a human-readable `reason`, so "why did this route to multilingual?" is one attribute away. `Router.predict_batch` groups a heterogeneous workload by checkpoint and by identical question schema before dispatching, so compatible requests share forward passes and results return in input order.

**The serving and eval perimeter.** Around the core, `laya/serve.py` exposes the engine over `POST /v1/systemone` on a wire protocol the docs keep compatible with TypeSafe's hosted Jev API; `laya/mcp/` speaks MCP over stdio; `laya/integrations/` adapts the same engine to LangChain, LlamaIndex and CrewAI. On the quality side, `laya/evals.py` defines evaluators like choice accuracy, score MAE and ECE, `laya-evals` (`laya/evals_cli.py`) gates builds on thresholds, and `laya/revisions.py` adds opt-in revision pinning and SHA-256 digest verification for checkpoints. Performance paths get their own modules: `laya/fast.py` and `laya/tl_kernels.py` implement TileLang fused kernels with 16-bit resident weights and fp32 accumulation plus CUDA graphs, while `laya/_compile.py` wraps the model in `torch.compile` with shape-independent dimensions to avoid per-request recompiles.

**The end-to-end flow.** A call like `router.predict(state, questions)` validates each question definition, normalizes it, detects the language and picks a checkpoint, encodes every question against the state into marker-carrying sequences, collates them into one batched tensor, runs a single bidirectional forward pass, scores all option markers in parallel, applies fitted temperatures, and returns typed answers with calibrated confidence — with no generated text anywhere in the loop. The checkpoints themselves are trained with reinforcement learning against strictly proper scoring rules (RLCD), which is what makes those probability outputs meaningful rather than decorative.

## Advantages

- **No autoregressive decoding.** All questions over a state are answered in one forward pass — the README documents 33 ms for a single question and 7.2 ms per question batched on a T4 — with no generation to parse and no chance of the model answering with an option you never declared.
- **Typed, schema-first interface.** The `choice` / `score` / `noul` question types in `laya/common.py` cover routing, rating and yes-no decisions, and `laya/structured.py` maps JSON schemas and pydantic models onto them with `decide()` and `decide_batch()`.
- **Routing that respects multilingual reality.** Script and language detection in `laya/lang.py` is dependency-free and runs in microseconds before any model loads, steering non-English traffic away from a checkpoint the benchmarks show cannot read it.
- **Calibrated confidence, twice over.** Temperature scaling is fitted per type and option bucket, and every answer carries both a distributional `confidence` and the calibrated `answer_confidence`, with opt-in `min_confidence` abstention.
- **Throughput-oriented batching everywhere.** `Agent.predict_batch`, `Router.predict_batch`, `predict_long` windowed scanning, and `sort_by_length` grouping share forward passes across states, schemas and checkpoints.
- **Multiple runtimes and surfaces.** PyTorch, a TileLang CUDA fast path, `torch.compile`, an ONNX Runtime path with INT8 quantization (`laya/onnx_agent.py`, `scripts/export_onnx.py`), plus CLI, HTTP, MCP and framework integrations out of the box.

## Benefits

- **Lower cost per decision.** Replacing generative calls with single-pass scoring makes per-decision latency and compute small and predictable, which matters when an agent makes thousands of routing, guardrail and triage decisions per hour.
- **Guardrails you can gate on.** The `LayaGuardrail` and triage runnables in `laya/integrations/langchain.py` wrap the same engine, so safety checks run in milliseconds instead of waiting on a generation round trip.
- **Production-grade operations.** Lazy or preloaded checkpoints with LRU eviction, per-request token budgets reaching every surface, CPU-fallback counters surfaced for health checks, and bearer-auth HTTP serving via `LAYA_*` environment variables.
- **Supply-chain hygiene.** `laya/revisions.py` supports pinned revisions and SHA-256 digest verification before any weight is parsed, a rare and welcome posture for Hub-downloaded models.
- **A real evaluation story.** `laya/evals.py` and the `laya-evals` CLI let you score a runner on a labelled dataset and gate CI on thresholds, and the repo's `research/` tree publishes the scripts and result files behind its benchmark claims.
- **A fine-tuning path that pays off.** The README reports the fine-tuned `laya-typed-decisions` checkpoint at 0.766 accuracy on the repo's 2,000-decision typed benchmark versus 0.362 for the base English checkpoint, with a Kaggle notebook (`notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`) covering the whole loop on free GPUs.

## Usage

Install from PyPI (Python 3.10 or newer):

```bash
python -m pip install laya
```

Declare typed questions and predict through the recommended `Router` entry point:

```python
from laya import Router

router = Router()  # downloads a checkpoint on first use; Router(preload=True) loads all three

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

result = router.predict(state, questions)
print(result["answers"]["department"]["choice"])  # billing
print(result["answers"]["churn_risk"]["noul"])    # probability the answer is yes
print(result["routing"]["model"])                 # english
```

From the command line, routing-only calls work offline and presets give you instant question sets:

```bash
laya "I was charged twice, please refund"            # routing decision only; no download
laya "My payment failed twice" --preset triage       # answer a ready-made preset
laya --batch tickets.txt --predict                   # score a file of requests in one batch
```

Or run the self-hosted HTTP server that speaks the Jev-compatible `/v1/systemone` protocol:

```bash
pip install "laya[serve]"
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve           # binds 0.0.0.0:8000, preloads all 3 checkpoints
```

## Conclusion

Laya is a refreshing counterpoint to the assumption that every AI decision needs a generative model. The source shows a coherent thesis carried through every layer: typed questions rendered into a single marker-carrying sequence, a bidirectional encoder that scores all options in parallel, calibration and routing handled before and after that one forward pass, and serving surfaces that treat strict validation and honest confidence reporting as features rather than afterthoughts. If you are building agents that make many small, structured decisions — triage, guardrails, routing, moderation — reading this codebase will change how you think about the "slow LLM call" in the middle of your loop. Start with `laya/common.py` and `laya/agent.py` for the engine, then `laya/router.py` to see how far a sub-millisecond language detector can carry you.

Links:

- GitHub repository: [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)
- Documentation: [nandhakishorm.github.io/laya](https://nandhakishorm.github.io/laya/)
- PyPI package: [laya](https://pypi.org/project/laya/)
- Hugging Face checkpoints: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
