---
layout: post
title: "NeoHorse: Recursive Self-Improvement via Agentic Post-Training - Inside TokenRhythm/NeoHorse"
description: "A source tour of TokenRhythm/NeoHorse, the open-weight agent model family built around a routing-harness evaluation-selection-update loop. We walk the serving examples, the NeoHorse-Jev prefill-only decision runtime, and the SGLang and vLLM adapters that make the self-improvement loop executable."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /NeoHorse-Recursive-Self-Improvement-Agentic-Post-Training-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/neohorse/tokenrhythm-neohorse-architecture.svg
tags:
  - Recursive Self-Improvement
  - Agentic AI
  - Post-Training
  - Open Source
categories: [AI, Open Source]
keywords: "NeoHorse, recursive self-improvement, agentic post-training, routing harness, NeoHorse-Jev, prefill-only inference, decision model, tool calling, vLLM, SGLang, open-weight LLM, agent workflows, TokenRhythm"
author: "PyShine"
---

Most agent models are trained once and then frozen, left to drift through whatever tool loops developers wrap around them. NeoHorse takes the opposite bet: the model itself sits inside a measurement loop, where a routing harness assigns tasks across a pool of models, records what happened, and feeds that evidence back into the next round of post-training. It is an early, concrete attempt to engineer recursive self-improvement as an ordinary training pipeline rather than a research slogan.

NeoHorse is TokenRhythm's family of open-weight models for agent workflows. The repository ships **NeoHorse-1**, a pair of 4B and 9B causal language models post-trained from Qwen3.5 for text-based agent harnesses, tool use, coding, and instruction following, plus **NeoHorse-Jev-4B**, a decision model built on NeoHorse-1-4B that answers structured questions with prefill-only inference. Both are Apache-2.0, with weights on Hugging Face and ModelScope plus quantized GGUF and MLX builds.

What makes the source worth a tour is the split between documentation and code. The routing harness and the training recipe are documented in the README and the bundled technical report (`TechnicalReport_NeoHorse_v1.pdf`), while the runnable code shows the other half of the loop: the serving stack that produces execution trajectories, and the Jev decision runtime that turns "which tool, which queue, which action" questions into cheap, probability-bearing answers. Reading the two together tells you exactly what the self-improvement loop consumes and what it emits.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/neohorse/tokenrhythm-neohorse-overview-architecture.svg" alt="Architecture overview of the TokenRhythm/NeoHorse repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the NeoHorse repository: model family documentation on the left, the NeoHorse-1 serving examples, the Jev decision runtime in the center, and the backend adapters on the right.*

Reading the overview from left to right: the README and the technical report define the model family and the routing-harness loop; the `examples` directory holds the chat and tool-call clients that exercise NeoHorse-1 through standard OpenAI-compatible servers; the Jev decision runtime (`jev/package`) centers on the `DecisionEngine`, fed by a bundle loader and a packed encoding model with a pointer head; and the `jev/infer` adapters wrap unmodified SGLang and vLLM servers so the same decision model can run behind production inference engines, with an optional vision path for image decisions.

## Why You Need This

If you build agent systems, you eventually discover that a lot of agent quality is really routing quality. Which model handles this request, which tool should be called, whether a workflow may proceed, how urgent an outcome is — these are decisions, and making a chat model generate JSON to answer them is expensive and fragile. NeoHorse-Jev attacks exactly this layer: given an application-defined state plus questions, it returns a selected candidate with full probabilities (Choice), a probability that a statement is true (Noul), or an expected rating over levels you define (Score), without generating free-form text.

The second problem is deployment realism. Many "agentic" releases stop at weights and a benchmark table. NeoHorse-1 ships with concrete serving instructions: both checkpoints run behind stock SGLang 0.5.17 or vLLM with a Qwen3-compatible reasoning parser and the `qwen3_coder` tool-call parser, a native 262,144-token context window, and two minimal clients in `examples/` that show the exact wire format, including thinking mode and automatic tool choice. That is enough to put a trajectory-producing agent endpoint up in minutes.

Third, the training story is written down rather than hand-waved. The README describes routing-guided curriculum SFT and routing-guided on-policy distillation as the two post-training stages, and names the data-quality machinery explicitly: exact and near-duplicate removal, evaluation decontamination, structural validation, six-dimensional semantic evaluation, and subscene-level Scene/Goal/Outcome labeling. The bundled technical report covers the routing harness, the post-training pipeline, and the evaluation protocol, including published sampling parameters, so comparisons against the reported numbers rest on something checkable.

Finally, the design is honest about being a prototype. The README states plainly that updated models can return to the harness to form a prototype evaluation-selection-update loop, and that extending the loop across successive iterations is the next step toward recursive self-improvement. You get a working slice of the loop today — heterogeneous model pool, recorded outcomes, capability-level feedback — without anyone claiming a finished perpetual-motion machine.

## How It Works

The repository splits into a documented training loop and an executable inference stack, and the detailed diagram follows the inference stack from the docs down to the pointer-head readout.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/neohorse/tokenrhythm-neohorse-architecture.svg" alt="Detailed architecture of the TokenRhythm/NeoHorse repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: documentation and report, the serving examples, the native neohorse_decision runtime, and the SGLang and vLLM adapter pairs with their shared JEV encoding runtimes.*

### Understanding the Architecture

**The loop, as documented.** According to `README.md`, the routing harness assigns tasks to a heterogeneous model pool, records tool interactions and outcomes, estimates capability demand, and feeds capability-level feedback into the next training mixture. Two post-training stages consume that feedback: routing-guided curriculum SFT and routing-guided on-policy distillation, which the README describes as turning execution trajectories into training signal while preserving execution and harness context. The bundled `TechnicalReport_NeoHorse_v1.pdf` and the linked arXiv report carry the full protocol.

**The trajectory-producing serving surface.** The two clients in `examples/` define what the harness records: `examples/chat.py` posts to `/v1/chat/completions` with `chat_template_kwargs.enable_thinking` set, and `examples/tool_call.py` adds a JSON-schema tool list with `tool_choice` on auto. On the server side, the README's launch commands enable the `qwen3` reasoning parser and the `qwen3_coder` tool-call parser on both SGLang and vLLM, so tool interactions come back in a parseable, structured form — exactly the raw material a routing harness needs to log and score.

**A decision model, not a small chat model.** The native runtime in `jev/package/src/neohorse_decision/` starts from `engine.py`, where `DecisionEngine` loads a complete model bundle — `backbone/`, `tokenizer/`, a separate `pointer_head.safetensors` head, and a `model_manifest.json` describing composition — via `_inference.py`. The engine is deliberately a serialized GPU worker: a `threading.Lock` guards each prediction, concurrent callers get a `BusyError` instead of queueing silently, and requests are bounded by explicit caps (1 MiB payload, at most 16 questions, a 32,768-token expanded budget).

**Packing many questions into one forward pass.** The core trick lives in `_vendor/model.py`. Five rarely used Qwen special tokens are repurposed as delimiters — state, question, option start, option end, and decide — and user text is rewritten so it can never forge them, making option boundaries unforgeable. `encode()` packs the state once, then appends one branch per question; a block-causal attention mask lets each question attend to the state and its own branch but not to sibling questions. A small `PointerHead` scores each option's end token against the decide token, producing logits per question. Because the state never sees the branches, its hidden states and KV cache are reusable across requests, which the prefix-reuse methods exploit; and because Qwen3.5's hybrid Gated DeltaNet layers cannot honor an arbitrary mask, hybrid backbones fall back to running each question as its own causal row continuing from the state.

**Answers that carry distributions.** On top of the model, `_vendor/schema.py` defines the three question types as pydantic models (up to 255 options for Choice, ordered levels for Score) and maps them onto the single pointer primitive: Noul becomes two options, false and true; Score becomes ordered level options whose expected value is the probability-weighted level index. `systemone.py` adds the wire adapter — model identity, validation rules such as two to ten Score levels, and confidence statistics derived from the local probability distributions, clearly labeled as local statistics rather than external calibration.

**Serving paths for every deployment shape.** `server.py` exposes `/v1/decision` and a System One-style `/v1/systemone` endpoint plus `/health`, with bearer-token auth, 8 MiB body limits, and distinct busy responses with `Retry-After`. When the loaded backbone is multimodal, the server automatically enables `vision.py`, which accepts one image (up to 4 megapixels, capped at 1,024 visual tokens) with one question. For engine-backed deployments, `jev/infer/sglang/launch.py` and `jev/infer/vllm/launch.py` start unmodified SGLang and vLLM servers: weight mapping modules build a config view over the original safetensors without tensor conversion, external loaders remap weight prefixes, and `head.py` performs the FP32 pointer readout on CPU after validating the manifest contract.

End to end: an application defines state and questions; the runtime validates and renders them into one packed sequence; the backbone runs a single (or per-question) forward pass; the pointer head produces a probability distribution per question; and the adapter layer returns typed answers — a chosen tool, a gating probability, an expected rating — that an agent harness, or NeoHorse's own routing harness, can act on immediately.

## Advantages

- **Prefill-only decisions.** Jev answers by pointing at options rather than generating text, so there is no sampling, no retry-for-valid-JSON, and the output format is valid by construction.
- **Many questions per forward pass.** The block-causal packing in `_vendor/model.py` answers up to 16 questions in one request while sharing the state prefix, and the prefix cache is reused exactly rather than approximately.
- **Stock inference engines.** The SGLang and vLLM adapters wrap unmodified engines with external loaders and weight-prefix remapping — no engine forks to maintain across releases.
- **No silent failures.** Explicit caps, strict load asserts in `_inference.py`, non-finite-logit checks in the CPU heads, and a serialized worker that fails loudly with `BusyError` instead of degrading.
- **A documented training recipe.** Routing-guided curriculum SFT, routing-guided on-policy distillation, decontamination, dedup, and structured data labeling are named in the README and expanded in the technical report.
- **Open weights, permissive license.** Apache-2.0 across the family, with 4B and 9B sizes plus GGUF, quantized, and MLX builds for local deployment.

## Benefits

- **Routing you can ship.** Choice distributions, Noul probabilities, and Score expectations are numeric signals your application can threshold, log, and feed into policy rules — the same shape of signal the NeoHorse routing harness consumes.
- **Lower cost per decision.** No generated output tokens; the runtime reports input tokens and serialized-answer tokens (`systemone.py`) so you can meter exactly what a decision costs.
- **Text and vision on one interface.** The same Choice/Noul/Score contract works over screenshots through `vision.py`, enabling UI-state checks and visual gating without a separate model.
- **Reproducible evaluation.** The README and `jev/README.md` publish sampling parameters, frozen benchmark subsets, per-benchmark scopes, and honest caveats about what is and is not ranked; the README reports NeoHorse-Jev-4B at 77.70 on its six-group text aggregate and 83.26% mean accuracy across Nimble, VitaminC, and MASSIVE.
- **Clear provenance.** `model_manifest.json` describes each bundle, and the vendored runtime keeps its `NOTICE.md` and license files, including credit to the Kev project from which parts of the decision inference code are adapted.
- **A template for self-improvement engineering.** Even as a prototype, the repository demonstrates the loop's contract: trajectories in, capability feedback out, updated models back into the pool.

## Usage

Serve NeoHorse-1-4B with SGLang, per the README:

```bash
pip install "sglang==0.5.17"
MODEL_PATH="/path/to/NeoHorse-1-4B"  # or /path/to/NeoHorse-1-9B
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name neohorse-1-4B \
  --host 0.0.0.0 --port 30000 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

Or with vLLM:

```bash
pip install -U vllm
MODEL_PATH="/path/to/NeoHorse-1-4B"  # or /path/to/NeoHorse-1-9B
vllm serve "$MODEL_PATH" \
  --served-model-name neohorse-1-4B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Then exercise the endpoint with the repository's own clients:

```bash
python examples/chat.py \
  --url http://127.0.0.1:8000 \
  --model neohorse-1-4B

python examples/tool_call.py \
  --url http://127.0.0.1:8000 \
  --model neohorse-1-4B
```

For the Jev decision model, install the native runtime from the `jev/` directory of the repository (after downloading the complete model bundle from Hugging Face or ModelScope) and use the CLI:

```bash
cd /path/to/NeoHorse/jev
python -m pip install --no-deps ./package

neohorse-decision predict --model-dir "$MODEL_DIR" --request example_request.json
neohorse-decision serve --model-dir "$MODEL_DIR" --port 8080
```

Query the decision service over HTTP:

```bash
curl -sS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"NeoHorse-Jev-4B","state":"I was charged twice for the same order. Please refund the extra charge today.","questions":{"refund":{"type":"noul","instructions":"Is the user requesting a refund?"}}}'
```

Or run the decision model behind the backend adapters, from the `jev/` directory:

```bash
CUDA_VISIBLE_DEVICES=0 python infer/vllm/launch.py \
  --bundle /path/to/model --port 30000

CUDA_VISIBLE_DEVICES=0 python infer/sglang/launch.py \
  --bundle /path/to/model --port 30000
```

## Conclusion

NeoHorse is interesting precisely because it refuses to overclaim. The README is candid that the evaluation-selection-update loop is a prototype, yet everything needed to participate in that loop is concrete: open 4B and 9B agent models with real tool-calling behavior, a decision model that answers routing questions in a single prefill pass with probability distributions, and serving code that integrates with stock SGLang and vLLM rather than forking them. For teams thinking about closed-loop agent improvement, this repository is one of the few places where the loop's inference side is executable today, and the training side is documented in enough detail to replicate.

Links:

- GitHub repository: [TokenRhythm/NeoHorse](https://github.com/TokenRhythm/NeoHorse)
- Technical report: [NeoHorse-1 on arXiv](https://arxiv.org/abs/2609.08183)
- Models: [NeoHorse-Jev-4B on Hugging Face](https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B) and the [NeoHorse-1 collection](https://huggingface.co/collections/TokenRhythm/neohorse-1)
- In-repo guides: `jev/README.md`, `jev/DEPLOYMENT.md`, and `jev/infer/README.md`
