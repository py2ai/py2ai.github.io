---
layout: post
title: "dynolab: An AI Interpretability Workbench For Apple Silicon - Inside canivel/dynolab"
description: "Dyno Lab is a free, open-source SwiftUI Mac app that runs MLX models locally and lets you investigate them: activation capture, interventions, linear probes, SAE experiments and causal patch sweeps, with a bundled Python layer, an OpenAI-compatible router, an experimental cross-machine GPU pool, and an SDK, HTTP API and MCP bridge. We tour the repository from the instrumented inference server up."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /Dynolab-AI-Interpretability-Workbench-For-Apple-Silicon/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/dynolab/canivel-dynolab-architecture.svg
tags:
  - Interpretability
  - AI Safety
  - Machine Learning
  - Swift
categories: [AI, Open Source]
keywords: "dynolab, Dyno Lab, mechanistic interpretability, MLX, Apple Silicon, sparse autoencoders, linear probes, activation steering, causal patching, local LLM, SwiftUI, Prometheus metrics, MCP, open source"
author: "PyShine"
---

Most tools for looking inside a language model ask you to leave your GUI at the door: a notebook, a GPU cluster, and a pile of half-maintained scripts. [Dyno Lab](https://github.com/canivel/dynolab) takes the opposite route. It is a free, open-source Mac app - a real SwiftUI application, not an Electron shell - that runs MLX models locally on Apple Silicon and puts activation capture, interventions, probes and sparse-autoencoder experiments into the same visual workspace as ordinary chat.

The repository is a genuinely interesting piece of engineering because it refuses two easy outs. It does not reimplement inference in Swift to stay "pure native", and it does not wrap a Jupyter server to fake being an app. Instead the app is native where native matters - GPU telemetry, process scanning, model discovery, the entire UI - and the inference server is a Python child process that instruments the upstream `mlx_lm` server rather than forking it.

The research half is just as deliberate. Experiments live in saved studies with provenance, token limits and truncated runs are shown as what they are rather than passed off as answers, and the documentation draws a hard line: activation norms and probe scores are observations to investigate, not certificates of safety. That kind of honesty is exactly what a lab tool needs.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/dynolab/canivel-dynolab-overview-architecture.svg" alt="Architecture overview of the canivel/dynolab repository" style="max-width:100%;height:auto;" />
</div>

*High-level architecture overview of the canivel/dynolab repository, from the SwiftUI app and its DynoKit system layer through the instrumented inference server, the Lab research API, and the router and GPU pool.*

Reading the overview from left to right: the native app supervises models through the bundled `dyno` CLI, which starts the instrumented MLX server as a child process; timing hooks in the token stream feed a metrics registry exposed as Prometheus and JSON endpoints; the same server hosts the activation capture broker that the Lab research API drives; experiments run as causal patch sweeps against isolated workers, and results land in saved studies the app can reopen; a router fronts everything with one OpenAI-compatible endpoint, and an experimental pool extends a Mac with a Windows NVIDIA worker.

## Why You Need This

The first reason is access. Mechanistic interpretability has a reputation for requiring infrastructure before you can ask your first question. Dyno Lab inverts that: download the Apple Silicon DMG, pull a model in the Discover tab, ask something in Chat, open Execution to see the actual request, and then capture your first activation map - the repository's own first-experiment guide walks exactly that path. The Python and MLX runtimes are bundled, so there is no environment to break.

The second reason is that the toolchain is honest about what numbers mean. The built-in benchmark harness runs models one at a time so each sees the same memory situation, prints the spread between trials so you know when a median is not yet a measurement, and reports how often a quantised model produced byte-identical output to the reference at the same seed. Machine telemetry comes from IOReport, IOKit and Metal without a single `sudo`, and it deliberately ignores the IOKit "device utilization" figure that reads near 100% on idle Macs, using P-state residency for GPU busy instead.

The third reason is reproducibility as a product feature. Every experiment can be saved as a study - prompt, settings, results, observations - and rerun or exported with provenance. The repository itself models the practice: it ships a full reproduction workflow for a model's tendency to change harmless references, with held-out identifiers, a saved layer probe, and input-only controls, plus an approval-monitor investigation that candidly documents the false alarms its own probe produced.

## How It Works

The system splits cleanly into a native app that never touches Python, a bundled Python server that owns MLX, and a research API that connects them.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/dynolab/canivel-dynolab-architecture.svg" alt="Detailed architecture of the canivel/dynolab repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the canivel/dynolab repository, covering the SwiftUI workspace, the DynoKit hardware layer, the instrumented MLX server, the Lab research engine, the router and pool, and the SDK and MCP interfaces.*

### Understanding the Architecture

**The server is an instrumentation, not a fork.** [src/dyno/serve/instrument.py](https://github.com/canivel/dynolab/blob/main/src/dyno/serve/instrument.py) imports `mlx_lm.server`, swaps in instrumented subclasses of the response generator and HTTP handler, and hands control back - so batching, prompt caching and speculative decoding all remain upstream code. The timing hooks live inside the token stream rather than the HTTP layer, which means streaming and non-streaming requests are measured identically. The three internal seams it relies on are checked at startup, and if a future `mlx_lm` reshapes them the server fails loudly rather than silently serving without metrics.

**Metrics are measured at the source.** [src/dyno/serve/metrics.py](https://github.com/canivel/dynolab/blob/main/src/dyno/serve/metrics.py) and [exporters.py](https://github.com/canivel/dynolab/blob/main/src/dyno/serve/exporters.py) expose decode throughput that deliberately excludes prompt processing and queue time, time to first token, prefill rate, cache hits and allocator memory - as Prometheus text on `/metrics` and as JSON with the last ten requests on `/stats`. The app's [monitor](https://github.com/canivel/dynolab/blob/main/app/Sources/DynoKit/IOReport.swift) additionally reads GPU busy percentage from P-state residency, memory against Metal's working-set ceiling, and power for the GPU, CPU, DRAM and Neural Engine rails separately - all without root.

**Research runs against an isolated worker, not your chat server.** The Lab API in [src/dyno/lab/server.py](https://github.com/canivel/dynolab/blob/main/src/dyno/lab/server.py) exposes endpoints documented by a committed OpenAPI schema, and drives workers that load their own model copies. The causal patching module in [src/dyno/lab/patching.py](https://github.com/canivel/dynolab/blob/main/src/dyno/lab/patching.py) is a good example of the project's guardrails: patch sites are capped, clean and corrupted prompts must align token for token, target and foil must be distinct single tokens, and the result notes plainly that single-site patching is not a full circuit graph.

**Capture happens on the generation thread.** The activation broker in [src/dyno/serve/activations.py](https://github.com/canivel/dynolab/blob/main/src/dyno/serve/activations.py) is serviced from the same seam that runs between scheduler iterations, so resident capture reuses the serving model's weights without racing the HTTP layer. That is what makes "explore the layer-by-token activation magnitudes of the model you are chatting with" a button in an app rather than a weekend project.

**Routing is a policy you can interrogate.** [src/dyno/router/policy.py](https://github.com/canivel/dynolab/blob/main/src/dyno/router/policy.py) decides which model answers each request through four mechanisms in order - explicit rules, self-routing where the strongest model tags a conversation's difficulty, residency-aware cost that accounts for measured throughput and load time, and confidence escalation when a cheap model's own token probabilities say it was unsure. Every decision records the candidates it rejected and why, because, as the module puts it, a router you cannot interrogate is one you end up disabling.

**The GPU pool trusts nothing over the network.** The coordinator in [src/dyno/pool/coordinator.py](https://github.com/canivel/dynolab/blob/main/src/dyno/pool/coordinator.py) pairs with a Windows worker over a verified SSH tunnel and keeps raw RPC on loopback, so pooled GGUF inference can borrow a desktop NVIDIA card without opening the Mac's research endpoints to the LAN. Discovery uses zeroconf, and pairing has its own guided app.

Every piece of the chain - app panels over HTTP, CLI flags, research endpoints - records what it did, which is what makes the saved-study model workable: an experiment is only worth returning to if you can still see exactly how it was produced.

## Advantages

- **A real app, not a script wrapper.** The entire UI, telemetry and supervision layer is Swift with a bundled Python runtime; you never install or manage Python yourself.
- **Upstream inference, instrumented.** Because the server subclasses `mlx_lm` instead of reimplementing it, new model architectures arrive on upstream's schedule, not the project's.
- **Interpretability tools with guardrails.** Bounded patch sweeps, validation of prompt alignment, and explicit disclaimers that probes are observations rather than safety certifications.
- **Honest benchmarking.** Spread, agreement and recorded conditions turn the usual misleading tokens-per-second table into something you can actually compare.
- **One endpoint for your tools.** The router serves an OpenAI-compatible API, and the optional LAN sharing is opt-in and off by default.
- **Three automation surfaces.** A Python SDK, an HTTP API with a committed OpenAPI schema, and a local stdio MCP bridge for connecting an assistant.

## Benefits

- **Apple Silicon stops being the wrong platform for research.** MLX inference, resident activation capture and hardware telemetry are first-class on the machines many people already own.
- **Experiments survive their session.** Saved studies keep question, prompts, results and observations together, so a finding from last week can be reopened and rerun rather than reconstructed.
- **Failure is visible by design.** The server runs as a child process so an out-of-memory model load cannot take the app down, and incomplete runs are labeled as such instead of becoming silently truncated answers.
- **Cost of experimentation is near zero.** Everything runs locally; no API key is needed to probe, patch, or steer a model you downloaded.
- **A curriculum in a repository.** The first-experiment guide, the probe reproduction example, and the approval-monitor investigation double as a self-paced introduction to interpretability method.
- **MIT licensed and contributing-friendly.** Native and Python checks both run in CI, and the docs spell out the release and signing process.

## Usage

Serve a model with live metrics, from the command line:

```bash
dyno serve --model mlx-community/Qwen3-8B-4bit --port 8971
```

The server then answers an OpenAI-compatible API and two measurement endpoints:

```text
OpenAI API   http://127.0.0.1:8971/v1
Metrics      http://127.0.0.1:8971/metrics   (Prometheus)
Stats        http://127.0.0.1:8971/stats     (JSON)
```

Compare two builds of one model under identical conditions, with the unsafe-comparison warnings printed:

```bash
dyno bench --model A --model B --repeat 3 --csv results.csv
```

Inspect where a model hesitated, and see exactly what quantisation changed at the same seed:

```bash
dyno inspect "why do B-trees suit range queries" --port 8971 --port 8973
```

Run the rest of the toolbox:

```bash
dyno pull <repo-id>             # download a model from the hub
dyno route                      # one endpoint in front of them all
dyno harness install aider      # point a coding tool at your models
dyno top --json                 # live telemetry as newline-delimited JSON
```

Build the app and run both test suites from a checkout on an Apple Silicon Mac:

```bash
git clone https://github.com/canivel/dynolab.git
cd dynolab
./app/build.sh
uv sync --locked --extra mcp --extra docs --python 3.12
PYTHONPATH=src uv run --frozen python -m unittest discover -s tests -v
swift test --package-path app
```

## Conclusion

Dyno Lab is a rare combination: a polished native app, an inference stack that stays upstream-compatible by construction, and an interpretability toolkit that treats rigor and honesty as features rather than documentation chores. It will not replace a full research cluster, and it says so itself - but as a way to ask real questions about model behavior on the Mac on your desk, with results you can save, rerun and share, it stands nearly alone. Start with the first experiment guide, and treat every probe score as the beginning of an investigation, which is precisely what the project intends.

**Links:**

- Repository: [github.com/canivel/dynolab](https://github.com/canivel/dynolab)
- Website and app handbook: [dynolab.dev](https://dynolab.dev)
- Runtime and telemetry notes: [docs/runtime-notes.md](https://github.com/canivel/dynolab/blob/main/docs/runtime-notes.md)
- First experiment guide: [docs/first-experiment.md](https://github.com/canivel/dynolab/blob/main/docs/first-experiment.md)
- Python SDK: [dynolab.dev/sdk.html](https://dynolab.dev/sdk.html)
- HTTP API: [dynolab.dev/api.html](https://dynolab.dev/api.html)
- MCP bridge: [dynolab.dev/mcp.html](https://dynolab.dev/mcp.html)
- GPU pool guide: [docs/pool-guide.md](https://github.com/canivel/dynolab/blob/main/docs/pool-guide.md)
