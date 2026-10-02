---
layout: post
title: "Ollaya: Run Open Decision Models Locally, the Way Ollama Runs LLMs - Inside ollaya-dev/ollaya"
description: "Ollaya is a Rust daemon and CLI that serves open decision models locally: state plus typed questions in, calibrated probabilities out, in a single forward pass. We tour the source behind its Ollama-style UX, TypeSafe-compatible wire format, ONNX Runtime and llama.cpp engines, verified weight provenance, and MCP server."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /Ollaya-Run-Open-Decision-Models-Locally/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ollaya/ollaya-dev-ollaya-architecture.svg
tags:
  - Rust
  - Local AI
  - Machine Learning
  - MCP
categories: [AI, Open Source]
keywords: "ollaya, decision models, local inference, ONNX Runtime, llama.cpp, TypeSafe systemone API, Rust daemon, calibrated probabilities, MCP server, Ollama alternative, classification models, edge AI"
author: "PyShine"
---

Not every AI workload needs a generator. Classify this support ticket, score this churn risk, route this request by language, decide whether this email is urgent: these are questions with fixed answer spaces, and paying a token-by-token LLM to answer them is both slow and unnecessarily vague. [ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya) is built around that observation. It is a Rust daemon and CLI that runs open decision models locally the way Ollama runs LLMs: you pull a model by name, serve it from a local daemon, and send it a state plus typed questions, getting calibrated probabilities back in a single forward pass.

The project's positioning is precise. A decision model reads a state, a message, an email, a ticket, any JSON, plus typed questions such as choice, score, and noul, and returns calibrated probabilities in milliseconds. It never generates text. Ollaya speaks TypeSafe's /v1/systemone wire format, so existing clients work by changing one environment variable, and it packages a growing zoo of open models, from encoder classifiers to decoder-based deciders, behind the familiar Ollama verbs: run, pull, list, ps, show, rm, cp, stop, create.

The source is worth a tour because it solves the unglamorous half of local AI that most projects skip: weight provenance, per-device parity checks, calibration, and an honest, fully published benchmark story. This is infrastructure engineering with a measurement culture, and the code shows it. Let us look inside.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ollaya/ollaya-dev-ollaya-overview-architecture.svg" alt="Architecture overview of the ollaya-dev/ollaya repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Ollaya codebase: the CLI, daemon, and MCP server on the left feed the decision service, which resolves models from the registry, prepares typed questions from the decision schema, and dispatches to inference engines spanning ONNX Runtime and llama.cpp.*

Reading the overview from left to right: you enter through the CLI, which starts the daemon if it is not running, or through the MCP server that exposes models to coding agents. The HTTP API routes into the decision service, the component where every request settles: it resolves the model name against the registry, ensures a runner process exists through the scheduler, and hands prepared typed questions to the inference engine. The engine layer is pluggable, with ONNX Runtime for the encoder models and ONNX-graph deciders, llama.cpp for models whose authors publish GGUF files, and an MLX backend for Apple silicon. The Modelfile support closes the loop, letting you bake a question set into your own named model.

## Why You Need This

The first problem is latency at volume. If an agent or a backend needs a decision per request, per email, per ticket, generation-based answers are the wrong tool: they are variable in format, slow, and probabilistically uncalibrated. The project's published results page measures every model on its own GPUs and CPUs, and the numbers there make the case: encoder models answering five questions in single-digit milliseconds on a GPU, and even decoder-based decoders answering in hundreds of milliseconds, all through one local HTTP API.

The second problem is calibration. A probability that says 0.91 should mean something. The README highlights a striking comparison: on the same Nimble model weights, the calibration error is 0.022 on Ollaya versus 0.122 on Ollama, because Ollaya applies each model's fitted temperature at load time. That difference is the work of the decision crate's calibration module, and it is the difference between a score you can threshold on and a score you have to eyeball.

The third problem is trust in model provenance. Local model serving projects rarely answer the question of where the weights came from. Ollaya's answer is architectural: it never re-hosts weights. The small ONNX graphs it publishes, about 3 MB each, read the original weight files from the authors' Hugging Face repositories, pinned to a commit and verified by sha256. Models whose authors publish GGUF files run that file itself on llama.cpp. Attribution is explicit, per model, with each author named in the README.

The fourth problem is correctness across hardware. A model that answers differently on CPU than on GPU is not the same model. Before a model ships, the build-time Python pipeline checks its runtime question by question against the authors' own code, or against llama.cpp's server for GGUF models, on each device it runs on, and the fp32 exports are stated to give the same decision as the PyTorch reference on all questions per checkpoint. Parity is treated as a release gate, not a hope.

## How It Works

The repository is a Rust workspace of eight crates, plus a Python build pipeline, a TypeScript website, and a Tauri desktop app, and the detailed diagram maps the flow.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ollaya/ollaya-dev-ollaya-architecture.svg" alt="Detailed architecture of the ollaya-dev/ollaya repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of Ollaya: entry points on the left, the server core and API contract beside it, the inference engines on the right, with the decision schema, registry internals, and the build pipeline that packages the static model registry.*

### Understanding the Architecture

**One binary, Ollama verbs.** [crates/ollaya/src/main.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya/src/main.rs) and [commands.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya/src/commands.rs) implement the CLI surface: serve runs the daemon, and run, pull, list, ps, show, rm, cp, stop, and create behave the way Ollama users expect. [daemon.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya/src/daemon.rs) handles the lifecycle trick of starting the daemon on demand, and [modelfile.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya/src/modelfile.rs) parses Modelfiles so a question set and a precision parameter can be baked into a new named model with ollaya create.

**The service layer.** [crates/ollaya-server/src/http.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-server/src/http.rs) exposes both the TypeSafe-compatible endpoints, /v1/systemone, /v1/decisions, and /v1/models, and the native API with /api/decide, streaming NDJSON pulls, and the usual tags, show, and ps routes. The contract lives in [crates/ollaya-api](https://github.com/ollaya-dev/ollaya/tree/main/crates/ollaya-api/src), which ships the request and response types, a Rust client, and a set of preset packs, triage, router, guard, moderation, agent, and email, that attach curated question sets to any model. [service.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-server/src/service.rs) is where a decide request becomes a runner call, and [scheduler.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-server/src/scheduler.rs) keeps one runner process per loaded model.

**Questions as a schema.** [crates/ollaya-decision/src/question.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-decision/src/question.rs) defines the typed question grammar, [layout.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-decision/src/layout.rs) builds the token sequences each family of models expects, [calibration.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-decision/src/calibration.rs) applies the fitted temperatures, and [answer.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-decision/src/answer.rs) turns raw logits into rendered probability bars. The router models get script and language detection from [crates/ollaya-lang/src/route.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-lang/src/route.rs), which is how laya picks its English or multilingual variant per request.

**Pluggable engines.** [crates/ollaya-runner/src/engine.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-runner/src/engine.rs) abstracts the backends: [onnx.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-runner/src/onnx.rs) runs the ONNX graphs on ONNX Runtime with CPU and CUDA execution, [llama/mod.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-runner/src/llama/mod.rs) loads the author-published GGUF files through llama.cpp's shared libraries, and [mlx/mod.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-runner/src/mlx/mod.rs) targets the Apple GPU. Precision is chosen at load time, fp16 on GPU and fp32 on CPU.

**Registry and provenance.** [crates/ollaya-registry/src/pull.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-registry/src/pull.rs) implements name resolution and resumable pulls over [store.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya-registry/src/store.rs), reading manifests from the versioned static registry in registry/v2/library that the Python pipeline packages. The build pipeline in [convert/ollaya_convert/export.py](https://github.com/ollaya-dev/ollaya/blob/main/convert/ollaya_convert/export.py) exports each family from the authors' weights, [parity.py](https://github.com/ollaya-dev/ollaya/blob/main/convert/ollaya_convert/parity.py) verifies per-question parity, and [catalog.py](https://github.com/ollaya-dev/ollaya/blob/main/convert/ollaya_convert/catalog.py) assembles the model catalog. The end-to-end flow: install, pull a model by name, the daemon resolves its manifest, pulls pinned blobs with checksum verification, spawns a runner, and every decide request flows through the service, the question layout, the engine, calibration, and out as calibrated, rendered probabilities.

**Agents and surfaces.** [crates/ollaya/src/mcp.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya/src/mcp.rs) runs the MCP server so Claude Code, Claude Desktop, and Cursor can call the models as tools, with integration coverage in [crates/ollaya/tests/mcp.rs](https://github.com/ollaya-dev/ollaya/blob/main/crates/ollaya/tests/mcp.rs), the agent-facing skill in [skills/ollaya-decisions/SKILL.md](https://github.com/ollaya-dev/ollaya/blob/main/skills/ollaya-decisions/SKILL.md), the API contract documented in [docs/api.md](https://github.com/ollaya-dev/ollaya/blob/main/docs/api.md), and the Tauri desktop app in desktop/ wrapping the same daemon for a menu-bar experience.

## Advantages

- **Single forward pass decisions.** Typed questions with fixed answer spaces return calibrated probabilities in milliseconds, no token-by-token generation.
- **TypeSafe wire compatibility.** Existing clients work by pointing the base URL at localhost; the native API adds routing info and timings on top.
- **Verifiable weight provenance.** Small ONNX graphs read author-published weights pinned to commits with sha256 checks; GGUF models run the authors' own files; nothing is re-hosted.
- **Parity as a release gate.** Each model is checked question by question against the authors' code per device before it ships.
- **Calibration built in.** Fitted temperatures are applied at load, which is what makes the published calibration errors an order of magnitude tighter on identical weights.
- **Ollama-shaped UX.** The pull, run, ps, and Modelfile workflow means near-zero learning cost for anyone who has served a local model before.

## Benefits

- **Predictable cost per decision.** Millisecond-scale local inference replaces per-token API spend for classification-shaped workloads.
- **A curated model zoo.** Encoders, zero-shot NLI and instruction-following classifiers, decoder deciders, safety guards, and routers, each with its author, accuracy, and latency published.
- **Agent-native access.** MCP tools plus a skill file teach coding agents when to prefer a decision model over a generative call.
- **Runs where you are.** CPU, CUDA on NVIDIA, Metal via llama.cpp on Apple silicon, MLX on the Apple GPU, plus Docker images and a desktop app.
- **Measured in the open.** Accuracy, calibration, and per-device latency for every model, with raw result files in the repository.
- **Hackable model definitions.** Modelfiles let you package your own question sets and precision choices as shareable named models.

## Usage

Install on Linux or macOS:

```sh
curl -fsSL https://ollaya.dev/install.sh | sh
```

On Windows, the installer is a PowerShell one-liner:

```powershell
irm https://ollaya.dev/install.ps1 | iex
```

Pull and run the recommended model against a support ticket, using a preset question pack:

```sh
ollaya run winnow:e4b --preset triage "Third time this year you've double-charged me. Refund it today or I'm cancelling and moving to a competitor."
```

Serve the daemon, manage models, and check what is loaded:

```sh
ollaya serve
ollaya pull laya
ollaya list
ollaya ps
```

Bake your own question set into a named model with a Modelfile:

```
FROM laya
QUESTIONS ./triage.json
PARAMETER precision fp32
```

```sh
ollaya create triage -f Modelfile
ollaya run triage "..."
```

Expose the models to coding agents over MCP and install the decision-making skill:

```sh
claude mcp add ollaya -- ollaya mcp
npx skills add ollaya-dev/ollaya --skill ollaya-decisions
```

For servers, the CUDA Docker image is the fastest path:

```sh
docker run -d --gpus=all -p 11435:11435 ghcr.io/ollaya-dev/ollaya:cuda
```

## Conclusion

Ollaya is a disciplined piece of local-AI infrastructure: it takes a narrow, well-defined problem, calibrated decisions instead of generated text, and solves it end to end, from weight provenance and parity gates to calibration, wire compatibility, and agent access. The eight-crate workspace is cleanly layered, the measurement culture is visible in the published raw results, and the Ollama-shaped interface makes the whole thing feel familiar on the first try. If your product makes classification-shaped calls on every request, this repository deserves a place in your evaluation alongside the API meter.

Links:

- GitHub repository: [ollaya-dev/ollaya](https://github.com/ollaya-dev/ollaya)
- API reference: [docs/api.md](https://github.com/ollaya-dev/ollaya/blob/main/docs/api.md)
- Project website and results: [ollaya.dev](https://ollaya.dev)
