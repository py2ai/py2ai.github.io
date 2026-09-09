---
layout: post
title: "Magnitude: An Open Source Local Model Inference Server Tuned for Apple Silicon"
description: "Magnitude is an open source inference server for Apple silicon that profiles your Mac, recommends the best models for it, then downloads, tunes, and runs them. It plugs into the agent you already use - Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline - or you can use the built-in harness. The core is an Inference Control Node (ICN), a Rust workspace with 11 crates that handles model lifecycle, hardware fit assessment, live inference with speculative decoding, and an OpenAI-compatible HTTP API. The native backend uses a pinned llama-cpp-rs bindings fork over llama.cpp with Apple Metal GPU offload. Models are loaded on demand and unloaded when idle or memory gets tight. Fully private and offline once models are downloaded. Apache 2.0 licensed, TypeScript CLI and Electron desktop, macOS 15+ on Apple silicon (M1 or later)."
date: 2026-09-08
header-img: "img/post-bg.jpg"
permalink: /Magnitude-Local-Model-Inference-Server-Apple-Silicon/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Magnitude
  - Local Inference
  - Apple Silicon
  - llama.cpp
  - Rust
  - AI Agents
  - Open Source
  - TypeScript
author: PyShine
---

## What is Magnitude

Magnitude is an open source inference server for Apple silicon. It profiles your Mac, recommends the best models for it, then downloads, tunes, and runs them. Plugs into Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline, or use the built-in harness. The code is on GitHub at [magnitudedev/magnitude](https://github.com/magnitudedev/magnitude), the package is on [npm](https://www.npmjs.com/package/@magnitudedev/cli), and the documentation is at [docs.magnitude.dev](https://docs.magnitude.dev).

## System Architecture

Magnitude has three layers: a TypeScript CLI and Electron desktop for the user interface, a Rust Inference Control Node (ICN) that serves an OpenAI-compatible HTTP API on port 8080, and native llama.cpp bindings over Apple Metal for GPU-accelerated inference.

![Magnitude system architecture](/assets/img/diagrams/magnitude/magnitude-architecture.svg)

The ICN is a Rust workspace with 11 crates, each with a defined responsibility:

- **icn-api** - the HTTP/OpenAPI boundary that receives chat completion requests
- **icn-server** - the composition root that wires everything together
- **icn-engine** - the live inference engine that runs the model
- **icn-models** - model lifecycle management (download, load, unload)
- **icn-hardware** - hardware fit assessment (profiles chip, memory, bandwidth)
- **icn-reasoning** - reasoning inspection and template handling
- **icn-catalog** - the model catalog with metadata and recommendations
- **icn-contracts** - transport- and backend-neutral contracts
- **icn-speculative** - speculative decoding support
- **icn-parity** - correctness parity testing against llama.cpp
- **icn-utils** - shared utilities

The native dependency chain is nested and pinned: `magnitude/inference/native/llama-cpp-rs` (the Rust bindings fork) wraps `llama-cpp-sys-2/llama.cpp` (the exact upstream llama.cpp revision). On Apple Silicon, the pinned bindings enable their macOS Metal backend for GPU layer offload.

## Setup Flow

Running `magnitude setup` walks through a five-stage flow: profile, recommend, download, tune, and connect.

![Magnitude setup flow](/assets/img/diagrams/magnitude/magnitude-setup-flow.svg)

1. **Hardware Profiling** - Magnitude profiles your chip, memory, and bandwidth to assess what your Mac can run.
2. **Model Recommendation** - models are ranked by speed, accuracy, intelligence, and memory, with tok/s estimates per model for your specific hardware.
3. **User picks a model** - you choose from the ranked recommendations.
4. **Download** - GGUF model files are downloaded to your Mac.
5. **Tune** - speculative decoding and Metal GPU layers are configured, along with context settings.
6. **Connect Harness** - your agent is switched to the local model via the OpenAI-compatible API.

You can also let your agent handle the setup. Send this to Pi, Claude Code, OpenCode, or whatever you use:

```text
Set up local models for me with the Magnitude CLI. Install it with `npm i -g @magnitudedev/cli` (or my package manager), then run `magnitude docs onboarding` and follow the instructions.
```

Your agent will profile your hardware, walk you through the best local models for it, download the ones you pick, and switch itself over to them.

## Agent Integrations

Magnitude plugs into eight agents. During setup, your agent connects your harness to the model you pick, or you can use Magnitude's built-in harness.

![Magnitude agent integrations](/assets/img/diagrams/magnitude/magnitude-agent-integrations.svg)

Supported agents: Pi, OpenCode, Hermes, OpenClaw, Codex, Claude Code, Oh My Pi, and Cline. Each connects to Magnitude's OpenAI-compatible API on port 8080, so any agent that speaks OpenAI's chat completions format works.

## Inference Request Lifecycle

Once running, Magnitude manages model memory automatically. Models are loaded on demand when your agent needs them and unloaded when idle or memory gets tight.

![Magnitude inference request lifecycle](/assets/img/diagrams/magnitude/magnitude-request-lifecycle.svg)

The lifecycle is:

1. **Chat Completion Request** - `POST /v1/chat/completions` with model, messages, and stream flag.
2. **icn-api** receives the request at the HTTP/OpenAPI boundary.
3. **Model loaded?** - if the requested model is not in memory, it is loaded via icn-models (downloaded if needed, GGUF loaded into memory, Metal GPU layers offloaded).
4. **icn-engine** runs live inference with speculative decoding via llama.cpp and Metal.
5. **Stream SSE Tokens** - tokens are streamed as `data: {token}` frames followed by `data: [DONE]`.
6. **Idle or memory tight?** - after the response, if the model is idle or memory is needed, it is unloaded; otherwise it stays in memory for the next request.

The API is OpenAI-compatible, so streaming completions look like this:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  --data '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

## Installation

```bash
npm i -g @magnitudedev/cli
magnitude setup
```

Requirements: any Apple silicon Mac (M1 or later, 2020 onward) running macOS 15 or newer. Intel Macs are not supported. More memory lets you run larger models.

## Why Magnitude

- **Knows your Mac** - profiles your hardware to assess fit and estimate tok/s per model
- **Recommends the best models** - ranked by speed, accuracy, intelligence, and memory
- **Tuned end to end** - speculative decoding and more, all set for your Mac
- **Easy setup** - one command and your agent is running local models
- **Free to run** - no token costs, API keys, or rate limits
- **Fully private and offline** - models, prompts, and files stay on your Mac
- **Models on demand** - loaded on request, unloaded when idle or memory fills
- **Open source** - Apache 2.0, yours to modify

## Monorepo Structure

Magnitude is a Bun monorepo with a TypeScript workspace for the CLI, desktop app, and packages, plus a Rust workspace for the inference engine:

| Directory | Contents |
|---|---|
| `cli/` | CLI source (TypeScript): commands, connections, inference runtime, docs, interactive runtime |
| `desktop/` | Electron desktop app (React + Vite) |
| `inference/` | Rust workspace: 11 ICN crates + native llama-cpp-rs bindings |
| `inference/crates/` | icn-api, icn-catalog, icn-contracts, icn-engine, icn-hardware, icn-models, icn-parity, icn-parity-probe, icn-reasoning, icn-server, icn-speculative, icn-utils |
| `inference/native/` | llama-cpp-rs bindings fork (pinned) |
| `inference/parity/` | parity cases, fixtures, profiles, model registry, target manifests |
| `inference/benchmark/` | versioned composite benchmark suite |
| `packages/` | 28 TypeScript packages: acn, acn-protocol, agent, ai, client-common, daemon-management, harness, icn-protocol, inference-benchmark, launcher, logger, openapi-effect, providers, roles, sdk, skills, storage, tracing, vcs, version |
| `integrations/` | agent-specific integrations (pi) |
| `web/` | web dashboard |

## Inference Testing Philosophy

Magnitude has three complementary validation categories:

1. **Correctness parity** - compares the smallest observable native and ICN operations: outputs, effective configuration, and state transitions.
2. **Performance parity** - times those same isolated operations only after both sides prove they performed equivalent work.
3. **Composite inference benchmarking** - sends controlled completion workloads to ICN and pinned `llama-server` endpoints to measure the complete engine, including scheduling, concurrency, prefix reuse, mixed prefill/decode work, latency, throughput, fairness, memory, and failures.

The primitive suites make failures attributable; the composite benchmark establishes whether the complete engine is competitive.

## Conclusion

Magnitude is a pragmatic answer to the question of running local models on Apple silicon. By profiling your hardware, recommending the best models, tuning them with speculative decoding and Metal GPU offload, and exposing an OpenAI-compatible API, it lets you plug local inference into the agent you already use without token costs, API keys, or rate limits. The Inference Control Node architecture - 11 Rust crates with clear responsibilities over a pinned llama.cpp native stack - ensures that model lifecycle, hardware fit, inference, and parity testing are each owned by a dedicated component. The source is on GitHub at [magnitudedev/magnitude](https://github.com/magnitudedev/magnitude), the package is on [npm](https://www.npmjs.com/package/@magnitudedev/cli), and it is Apache 2.0 licensed.
