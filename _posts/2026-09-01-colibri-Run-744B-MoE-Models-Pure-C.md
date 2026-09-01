---
layout: post
title: "colibri: Run 744B MoE Models on Consumer Hardware in Pure C"
description: "Learn how colibri streams frontier Mixture-of-Experts models (744B to 2.8T parameters) from disk using a single C file, treating VRAM, RAM, and NVMe as one inference hierarchy with a JIT-for-weights approach."
date: 2026-09-01
header-img: "img/post-bg.jpg"
permalink: /colibri-Run-744B-MoE-Models-Pure-C/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - C
  - LLM
  - Tutorial
author: "PyShine"
---

# colibri: Run 744B MoE Models on Consumer Hardware in Pure C

Frontier language models with hundreds of billions of parameters are normally gated behind hyperscaler-class GPU clusters. A 744B-parameter Mixture-of-Experts model like GLM-5.2 is expected to require multiple H100s and hundreds of gigabytes of VRAM. `colibri`, an open-source project from JustVugg, challenges that assumption by running the same model on a laptop with 25 GB of RAM and no GPU. Not fast, but correct - and the architecture behind that feat is worth understanding.

The core insight is structural. A 744B MoE model activates only about 40B parameters per token, and only about 11 GB of those change from token to token (the routed experts). The remaining 94.6% of experts sit idle for any given token. Colibri asks the obvious question: if most experts are not needed, why keep them all in memory? The engine treats VRAM, RAM, and NVMe as a single multitier inference hierarchy and streams experts from disk on demand, like a JIT compiler that only compiles the hot paths. The result is a pure-C engine with zero runtime dependencies that runs frontier models on hardware you already own.

## Project Overview

Colibri is released under the Apache 2.0 license and the current version is 1.10.1. The engine is a single C file (`c/colibri.c`) plus small headers, with no BLAS, no Python at runtime, and no GPU required.

| Property | Value |
|----------|-------|
| Repository | `JustVugg/colibri` |
| License | Apache 2.0 |
| Language | C (engine) + Python (converter / API gateway) |
| Version | 1.10.1 |
| Engine size | Single C file + small headers |
| Dependencies | gcc/clang + OpenMP (runtime); Python (converter only) |
| GPU required | No (CPU-only by default) |
| Model range | 7B to 2.8T parameters |
| Stars | ~26,500 |

## The Overall Architecture

The engine is organized around a central principle: placement only ever decides speed, never semantics. The router makes the same decisions and the weights keep the same precision whether an expert answered from VRAM, RAM, or disk. The architecture diagram below shows how the pieces fit together.

![colibri Overall Architecture](/assets/img/diagrams/colibri/colibri-architecture.svg)

### Understanding the Architecture

**The Engine (`colibri.c`)**
The runtime engine is a single C file. No BLAS, no Python at runtime, no GPU required. Python is used only by the one-time model converter and the optional OpenAI-compatible API gateway. The engine links against OpenMP (`libgomp` on Linux) for CPU parallelism and nothing else at runtime.

**Router**
For each token, the router picks 8 experts per layer across 75 layers (for GLM-5.2). The dense part - attention, shared experts, and embeddings - stays resident in RAM. Only the routed experts are candidates for streaming.

**JIT for Weights**
This is the central metaphor. A compiler JIT never compiles the whole program - it watches what actually runs and compiles the hot paths just in time. Colibri makes the same bet about a 744B parameter space: parameters are not resident state to be held, they are data to be staged across a heterogeneous storage hierarchy exactly when the router proves they are needed. Measured routing heat drives a per-layer LRU cache, a learned pinned hot-store, and one-layer-ahead prefetch.

**PILOT (Router-Lookahead Prefetch)**
A separate thread runs the router one layer ahead so that the next layer's experts are prefetched while the current layer computes. Routing is measurably 71.6% predictable one layer ahead, which means most prefetches hit before the demand read.

**PIPE (Async I/O Overlap)**
A bounded async I/O pool loads missing experts while resident ones compute. Batch-union reads ensure each unique expert is read only once per batch of positions, and each expert's three matrices are stored adjacent and read in a single `pread`.

**Backends**
The engine supports CPU (OpenMP), CUDA (with a resident pipeline that keeps the residual stream on-device across layers), Metal (Apple Silicon unified memory), and Vulkan (with MXFP4 decode and a second-device expert tier). The profitable combination depends on compute, bandwidth, residency, and workload - the engine does not force a single path.

**Semantic Guarantee**
This is the design constraint that ties everything together: insufficient fast memory may reduce speed, but the default policy never silently changes model precision or router semantics. The output is identical whether an expert answered from VRAM or from disk.

## The Per-Token Path

Every layer of every token walks the same five steps. The design goal is that placement only ever decides speed - the router's decisions and the weights' precision are the same whether an expert answered from VRAM or from disk.

![colibri Per-Token Path](/assets/img/diagrams/colibri/colibri-token-path.svg)

### The Five Steps

**1. ROUTE**
The router picks 8 experts of 256 per layer. The dense part (attention, shared experts, embeddings - about 17B params) stays resident in RAM at int4 (about 9.9 GB). Only the routed expert indices are candidates for staging.

**2. UNION**
Batch-union reads each unique expert once across all positions in the batch. If two positions in the same batch both need expert 142 in layer 37, that expert is read from disk exactly once, not twice. This is a critical optimization because disk reads are the bottleneck.

**3. PLACE**
The per-layer LRU cache and the learned pinned hot-store decide which tier each expert lands in: VRAM (if a GPU is present and the expert is hot enough), RAM (LRU cache or pinned), or NVMe (stream on demand). The placement decision is driven by measured routing heat recorded in `.coli_usage`.

**4. OVERLAP**
The PIPE async I/O pool loads missing experts while resident ones compute. The PILOT thread has already prefetched the next layer's experts. The goal is simple: never wait for the disk twice. Readahead and demand reads hit the same drive (deterministic hash routing) so nothing is cached twice.

**5. LEARN**
After compute, the engine records which experts your workload actually routed to (`.coli_usage`, updated every turn) and pins the hottest ones automatically. Colibri literally gets faster the more you use it - the learning cache is workload-specific and persistent across sessions.

## The Three-Tier Memory Hierarchy

The heart of colibri is its treatment of VRAM, RAM, and NVMe as a single multitier hierarchy rather than three separate memory requirements. The same engine spans the whole range: on a 25 GB laptop everything streams from disk (slow but correct); on a large host the entire expert set becomes resident and disk drops out of the decode path entirely.

![colibri Memory Hierarchy](/assets/img/diagrams/colibri/colibri-memory-hierarchy.svg)

### How the Tiers Work

**Dense Part (always resident in RAM)**
The dense part - attention, shared experts, and embeddings - is about 17B parameters. At int4 precision, it occupies about 9.9 GB and stays resident in RAM for the entire session. This never streams from disk.

**Routed Experts (19,456 total, streamed on demand)**
GLM-5.2 has 75 MoE layers with 256 routed experts each, plus the MTP head - 19,456 experts total. At int4, each expert is about 19 MB, so the full set is about 370 GB on disk. They are streamed on demand with a per-layer LRU cache, a learned pinned hot-store, and an optional VRAM tier.

**VRAM Tier (optional)**
If you have a GPU, the hottest experts can live in VRAM for fastest access. On a 6x RTX 5090 host with 251 GB RAM, the entire expert set becomes resident (`CUDA_EXPERT_GB=auto PIN_GB=all`) and disk drops out of the decode path entirely - delivering 5.8 to 6.8 tok/s.

**Dual-SSD Striping**
Decode is disk-bound on most machines, and expert reads are read-only. If you have a second SSD, you can put a full copy of the model on it and let the engine stream from both drives at once. Each expert is routed to one drive by a deterministic hash, weighted by the two drives' measured bandwidth. The aggregate bandwidth is the sum of both drives - a 9 GB/s + 3 GB/s pair reads experts about 33% faster than the fast drive alone. The mirror is validated at startup, never written, and a read error falls back to the primary (degrades instead of crashing).

**Learning Cache**
Between the tiers sits a learning cache: the engine records which experts your workload routes to (`.coli_usage`, updated every turn) and pins the hottest ones automatically. On multi-socket hosts, `COLI_NUMA=1` interleaves the resident weights across memory controllers.

### The Semantic Guarantee

This bears repeating because it is what distinguishes colibri from lossy quantization schemes: placement only ever decides speed. The router makes the same decisions whether an expert answered from VRAM or from disk. The weights have the same precision. The output is identical. If you cannot fit the model in fast memory, you get a slower model - not a different model.

## Supported Models and Backends

Colibri supports eight model families today, each implemented as a single C file that shares the same `coli chat` / `coli serve` / `coli web` front end. The engine spans from 7B (OLMoE) to 2.8T (Kimi K3) parameters.

![colibri Models and Backends](/assets/img/diagrams/colibri/colibri-models-backends.svg)

### Model Families

| Model | Parameters | Notes |
|-------|-----------|-------|
| GLM-5.2 | 744B MoE | Flagship, int4 streaming, 19,456 experts |
| GLM-5.3-Flash | 321B | With vision support |
| Inkling | 975B MoE | CUDA tier with bf16 residents |
| Kimi K3 | 2.8T | Largest supported model |
| DeepSeek V4 Flash | 284B | CUDA kernels as a tier, double-buffered prefetch |
| Qwen3.8-Flash-Next | 125B + 51B | n-gram, with tools and vision |
| Qwen3.6 | 35B-A3B | Smaller MoE |
| OLMoE | 7B | Smallest, good for testing |

### Backends

| Backend | Platform | Key Feature |
|---------|----------|-------------|
| CPU | All (OpenMP) | Portable, no GPU needed, default path |
| CUDA | NVIDIA | Resident pipeline (`COLI_CUDA_PIPE=2`), speculative expert kernels |
| Metal | Apple Silicon | Unified memory, batched expert math |
| Vulkan | Cross-platform | MXFP4 decode, second-device expert tier (`COLI_VK_DEV2`) |

### Local Cluster Mode

For distributed inference, colibri includes a coordinator-worker cluster mode. The coordinator keeps token generation, routing, and KV state local while disk-backed expert workers execute routed FFNs on other machines. A layer's routed batch-union is sent as one persistent TCP request, so a token does not incur one round trip per expert.

## Installation and Quick Start

### Option 1: Prebuilt Binary (No Compiler Needed)

Prebuilt archives are published for Linux, macOS, and Windows on the [Releases page](https://github.com/JustVugg/colibri/releases).

```bash
# Linux
mkdir colibri && tar xzf colibri-v1.10.1-linux-x86_64.tar.gz -C colibri && cd colibri
python3 coli info  # engine ready
```

On Windows, download the zip, extract it, install Python 3, and run from PowerShell:

```powershell
$env:COLI_MODEL="drive:/path/to/model/folder/"
py ./coli chat
```

### Option 2: Build from Source

Building from source produces the fastest binary for your CPU (`ARCH=native` unlocks the vector instructions your chip actually has).

```bash
# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install -y build-essential git python3

# macOS
xcode-select --install
brew install libomp git python

# Get the code and build
git clone https://github.com/JustVugg/colibri.git
cd colibri/c
./setup.sh
```

`setup.sh` checks your compiler and OpenMP, builds the engine, and runs a self-test. When it prints `engine self-test: 32/32`, the engine is working correctly.

### Get the Model

A pre-converted GLM-5.2 int4 container is available on HuggingFace. Use the group-scaled (gs64) container with int8 MTP heads - the older per-row int4 containers measure about 9pp worse on quality benchmarks, and plain int4 MTP heads collapse to near-zero draft acceptance.

```bash
# Download the recommended group-scaled int4 model
# https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp

# Or convert from FP8 shards yourself
pip install torch safetensors huggingface_hub numpy
./coli convert --model /path/to/your/model/directory
```

The conversion downloads FP8 shards one by one, converts them to the int4 container format, and prepares the MTP head for speculative decoding. It is resumable and does not require the full 756 GB FP8 checkpoint to exist on disk at once.

## Usage

### Interactive Chat

```bash
COLI_MODEL=/path/to/glm52_i4 ./coli chat
```

The engine automatically detects RAM budget, expert cache, and MTP settings. On startup it prints the model, precision, and resident memory:

```
colibri v1.10.1 - GLM-5.2 - 744B MoE - int4 - streaming CPU
ready in 32s - resident 9.9 GB
```

### OpenAI-Compatible API Server

```bash
COLI_MODEL=/path/to/glm52_i4 COLI_API_KEY=local-secret ./coli serve \
  --host 127.0.0.1 --port 8000 --model-id glm-5.2-colibri
```

Query it with any OpenAI-compatible client:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.2-colibri",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Web Dashboard

```bash
./coli web
```

The web dashboard shows live token metrics, a per-turn time breakdown, the VRAM/RAM/disk tier bar, and a live "Brain" view where all 19,456 experts are displayed as a cortex - color is the storage tier, brightness is routing heat, and every expert routed in a turn flashes white.

### Dual-SSD for Higher Bandwidth

```bash
COLI_MODEL=/fast/glm52_i4 COLI_MODEL_MIRROR=/second/glm52_i4 ./coli chat
```

For a second drive that cannot hold the whole model, colibri can rank a partial mirror from the expert history it already learns:

```bash
./coli mirror plan  --model /fast/glm52_i4 --mirror /second/glm52_i4 --budget-gib 200 --reserve-gib 20
./coli mirror stage --model /fast/glm52_i4 --mirror /second/glm52_i4 --budget-gib 200 --reserve-gib 20
./coli mirror verify --model /fast/glm52_i4 --mirror /second/glm52_i4
```

## Benchmark Numbers

All numbers are measured by the community, not estimated. The full table with hardware configs is in the [project benchmarks page](https://github.com/JustVugg/colibri/blob/main/docs/benchmarks.md).

| Machine | Speed |
|---------|-------|
| 6x RTX 5090, 251 GB RAM (full residency) | 5.8 to 6.8 tok/s |
| Apple M5 Max, 128 GB unified, Metal backend | 1.83 tok/s warm |
| Dell DGX Spark (Grace ARM), 121 GB | 2.4 tok/s |
| 128 GB CPU-only desktop | 1.8 tok/s warm |
| Single RTX 5070 Ti laptop | 1.07 tok/s |
| Ryzen 9 9950X, 123 GB, PCIe 5.0 NVMe | 1.23 tok/s |
| Intel Core Ultra 7, 24 GB RAM, WSL2 | 0.07 to 0.11 tok/s |
| Dev box baseline: 12 cores, 25 GB RAM | 0.05 to 0.1 tok/s cold |

The baseline (25 GB RAM, no GPU) is 0.05 to 0.1 tokens per second - roughly one word every 10 to 20 seconds. Slow, but it produces correct output from a 744B frontier model on hardware that costs less than a single H100 GPU fan.

## Advanced Features

### Speculative Decoding (MTP)

GLM-5.2 has a native Multi-Token Prediction (MTP) head that can draft tokens for the main model to verify. Colibri ships this with two rules learned through hard experience: the MTP head must be int8 (int4 heads collapse to near-zero draft acceptance), and draft and verify must compute the same function. The result is 2.2 to 2.8 tokens per forward pass when speculation pays off. When acceptance does not repay verification, MTP can be disabled.

### Compressed KV State

MLA attention stores a compressed KV state - 576 floats per token instead of 32,768, which is 57x smaller. The KV state persists across restarts, so conversations reopen warm with zero re-prefill. Token-exact forward validation against a `transformers` oracle confirms 32/32 test tokens match exactly.

### Open Hypotheses

Colibri treats an optimization as a hypothesis until a controlled end-to-end A/B shows otherwise. The project openly publishes what works, what might work, and what still needs testing - including negative results. This makes it both an inference engine and an open research platform for systems-level ML optimization.

## Why colibri Matters

Colibri matters for three reasons. First, accessibility: it democratizes access to frontier-class LLMs, allowing them to run on standard consumer machines. Second, engineering honesty: it never silently reduces quality - placement only decides speed, never semantics. Third, research openness: it treats optimizations as testable hypotheses and publishes negative results alongside positive ones.

The engine is deliberately small enough that the next useful optimization can come from anyone willing to measure it. Not renting intelligence behind an API - holding it: probing it, measuring it, improving it. That philosophy, combined with the technical elegance of treating storage as a first-class citizen of the inference pipeline, makes colibri a project worth watching.

## Troubleshooting

**Engine exits at startup on a fresh machine**
The engine links `libgomp.so.1` at runtime. A host that has never had a compiler installed may not carry it. Install the runtime package alone: `sudo apt install -y libgomp1`. Run `./coli doctor` to diagnose missing dependencies.

**MTP speculation not helping**
Make sure you have the int8 MTP heads, not the int4 ones. Plain int4 MTP heads collapse to near-zero draft acceptance. Use the group-scaled (gs64) container with int8 MTP heads, which also fixes the reasoning-loop and EOS-starvation failures seen with the older per-row int4 containers.

**Slow decode on a laptop**
Decode is disk-bound on most machines. Use a fast NVMe SSD, enable dual-SSD striping if you have a second drive, and let the learning cache warm up by running representative prompts first. The engine gets faster the more you use it.

**ARM64 Linux (Graviton, Raspberry Pi)**
There is no prebuilt ARM64 archive. Build from source - the engine is portable C with OpenMP and no x86-only intrinsics, so `./setup.sh` builds it on ARM64 without source changes or extra flags.

## Further Reading

- [colibri GitHub repository](https://github.com/JustVugg/colibri)
- [Project website](https://justvugg.github.io/colibri)
- [Quick Start guide](https://github.com/JustVugg/colibri/blob/main/docs/quickstart.md)
- [Benchmark protocol](https://github.com/JustVugg/colibri/blob/main/docs/benchmarks.md)
- [GPU backends guide](https://github.com/JustVugg/colibri/blob/main/GPU_BACKENDS.md)
- [Pre-converted GLM-5.2 int4 model (gs64)](https://huggingface.co/mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp)
