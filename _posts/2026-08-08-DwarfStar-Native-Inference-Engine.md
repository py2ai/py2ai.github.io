---
layout: post
title: "DwarfStar (ds4): A Native Inference Engine for DeepSeek V4 Flash"
description: "DwarfStar is a small native inference engine by Salvatore Sanfilippo, optimized first for DeepSeek V4 Flash. This guide covers its architecture, backends, performance, installation, and usage."
date: 2026-08-08
header-img: "img/post-bg.jpg"
permalink: /DwarfStar-Native-Inference-Engine/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI, Open Source, LLM, Inference]
tags: [DwarfStar, ds4, DeepSeek V4 Flash, GLM 5.2, LLM inference, Metal, CUDA, ROCm, GGUF, antirez, Salvatore Sanfilippo]
keywords: "DwarfStar ds4, antirez ds4 inference engine, DeepSeek V4 Flash local inference, GLM 5.2 GGUF, native LLM inference engine, Metal CUDA ROCm inference, SSD streaming LLM, tensor parallelism DeepSeek, GGUF imatrix quantization, Salvatore Sanfilippo DwarfStar"
author: "PyShine"
image: /assets/img/diagrams/dwarfstar/dwarfstar-architecture.svg
---

## Introduction

Running capable large language models locally used to mean leaning on a general-purpose GGUF runner. DwarfStar, also known by its repository name `ds4`, takes a different stance. It is a small, native inference engine optimized first for a single model family, DeepSeek V4 Flash, and it is built and tested as one cohesive unit rather than as a plug-and-play loader for arbitrary weights. The result is a system that is deliberately narrow but unusually fast and tightly integrated.

DwarfStar is the work of Salvatore Sanfilippo, better known as antirez, the creator of Redis. It is licensed under the MIT license and developed openly on GitHub at [antirez/ds4](https://github.com/antirez/ds4). Beyond the core engine, the repository ships the supporting ecosystem needed to make local inference practical: GGUF tooling, imatrix collection, quantization, quality testing, directional steering data, and a coding agent that runs against the live inference session.

The project is candid about how it was built. It was developed with strong assistance from GPT 5.5, 5.6, and Claude Fable, with humans leading the ideas, testing, and debugging. It also stands on the shoulders of [llama.cpp](https://github.com/ggml-org/llama.cpp) and GGML; while `ds4.c` does not link against GGML, the quantization layouts, kernels, and GGUF ecosystem pioneered by that project were an essential reference. This post walks through what DwarfStar is, how it works, the backends it targets, its measured performance, and how to install and use it.

## What is DwarfStar

DwarfStar is a self-contained native inference engine. Its primary target is DeepSeek V4 Flash, a routed mixture-of-experts model. It also supports GLM 5.2, and on very high-memory machines it can run DeepSeek V4 PRO. The defining design choice is that DwarfStar is not a general GGUF runner. Model loading, prompt rendering, tool calls, KV state management, the HTTP server, and the coding agent are all built and tested together for these specific models.

This narrow focus pays off in several ways. Because the engine knows the exact tensor layout, quantization mix, metadata, and optional multi-token-prediction state of the supported GGUFs, it can apply aggressive optimizations that a generic loader cannot safely assume. The 2-bit quantizations, for example, quantize only the routed mixture-of-experts weights, leaving shared experts, projections, and routing untouched to protect quality. The engine verifies that these aggressive quants behave well: they call tools reliably and work under coding agents.

Several features distinguish DwarfStar from a typical local runner:

- **SSD streaming** for low-RAM machines, where routed experts are cached and loaded from the GGUF file on demand while non-routed weights stay resident.
- **Tensor parallelism** over RDMA, so two Macs connected by Thunderbolt 5 can split a single decode across GPUs.
- **Pipeline parallelism** across multiple machines, gluing their RAM together to fit larger models and to accelerate long prefills.
- **Micro-batching** in the server, grouping decode rows across resident sessions for higher aggregate throughput.
- **GGUF tools** for offline imatrix collection, quantization, and quality scoring against official continuation fixtures.
- **Directional steering**, using single-vector activation directions to adjust model behavior without fine-tuning.

The software is described as beta quality and fast-changing, with a full QA run executed before each release. It is opportunistic about model support, following the best open weights for useful local machine sizes such as 128 GB laptops and 512 GB workstations, and a model may be removed when a better replacement arrives.

## How It Works

![DwarfStar Inference Architecture](/assets/img/diagrams/dwarfstar/dwarfstar-architecture.svg)

The architecture diagram above shows how a request flows through DwarfStar from input to executed tokens on a hardware backend. Read it top to bottom: inputs enter at the top, pass through the engine in the middle, and dispatch to a backend below, while offline tooling on the side prepares the model weights.

The top band is the input layer, colored green to mark user-facing entry points. Two sources feed the engine. The first is the user prompt, which can arrive from the interactive CLI, an agent session, or an OpenAI- and Anthropic-compatible HTTP endpoint.

The second source is the tool calls: tool schemas expressed in DSML, OpenAI, or Anthropic wire formats. These two inputs are the only externally visible surfaces of the system; everything below them is internal to the engine.

That internal placement is deliberate. Because there is no protocol boundary between the agent and the model, DwarfStar keeps latency low and avoids the translation overhead a socket-based stack would impose.

The middle band, colored blue for processes, is the DwarfStar engine. It holds four tightly coupled stages that share one process and one KV state. The first stage is prompt rendering, which converts messages and tool definitions into the exact DeepSeek chat template, including DSML tool blocks.

The second stage is model loading. It reads a supported GGUF and maps the specific tensor layout, quantization mix, and metadata into the resident graph. Knowing the exact layout in advance lets the engine apply optimizations a generic loader cannot, such as quantizing only the routed mixture-of-experts weights.

The third stage is the KV cache. It holds compressed attention state, including the indexer and compressor frontiers of DeepSeek's multi-head latent attention, and persists snapshots to disk so long agent sessions and server restarts resume without reprocessing a prefix.

The fourth stage is the inference graph. It runs prefill and decode, applies micro-batching when multiple server sessions are resident, and feeds token state back into the KV cache. Because all four stages live in one process, the agent can drive inference directly with no KV mismatch by construction.

The bottom-left band, colored purple for backends, is where the graph executes. The Metal backend targets Macs with 96 GB or more of unified memory and is the primary development target.

When the model does not fit, smaller Macs fall back to SSD streaming, where routed experts are cached and loaded from the GGUF on demand. The CUDA backend handles NVIDIA multi-GPU systems, including DGX Spark and eight-card servers such as 8xL40S, with tensor and pipeline parallelism.

The ROCm backend targets Strix Halo systems like the Framework Desktop. Each backend implements the same graph contract, so the engine above it does not change when you switch hardware.

The bottom-right band lists the supported model weights. DeepSeek V4 Flash is the primary target, with 2-bit routed quants that fit on 96 to 128 GB machines. GLM 5.2 is supported with routed Q2_K and Q4_K layouts.

DeepSeek V4 PRO is the high-memory target, runnable on 512 GB workstations and across distributed Mac Studio pairs. Model loading only accepts these specific GGUFs because the engine relies on their exact layout rather than parsing an arbitrary file.

The side band, colored orange for tools, holds the offline GGUF tooling. imatrix collects calibration data for the routed mixture-of-experts layers, which guides the quantization step so aggressive 2-bit quants stay accurate.

Quantization then produces the IQ2_XXS, Q2_K, MXFP4, and Q4_K GGUF files the engine loads. Quality testing scores those files against a 100-case fixture built from official DeepSeek continuations, so 2-bit quants are validated to call tools reliably before they ship.

The data flow is a closed loop: tooling feeds the supported models, the models feed model loading, model loading feeds inference, and inference dispatches to the backend. That vertical integration for a small set of models is the central reason DwarfStar can be both fast and reliable on consumer hardware.

## Backends

DwarfStar supports three GPU backends, each chosen for a class of machine where capable open-weight models actually fit.

**Metal** is the primary target and the default on macOS. It runs on Macs with 96 GB or more of unified memory, making the model resident for the fastest path. When the model does not fit, Metal also implements SSD streaming, where the non-routed weights stay resident while routed experts are kept in an in-memory cache and loaded from the GGUF file on cache misses. Two MacBooks can also be combined for tensor parallelism over RDMA on Thunderbolt 5.

**NVIDIA CUDA** targets multi-GPU systems, including DGX Spark and servers with older Ada Lovelace cards that newer frameworks may no longer support well. With `--cuda-tensor-parallel`, DeepSeek V4 Flash tensor and routed-expert work is split across an even number of GPUs, with paired devices holding a 50/50 split of routed experts and a row-sharded vocabulary head. GLM 5.2 uses normal layer placement across the selected devices.

**ROCm** targets AMD Strix Halo systems such as the Framework Desktop. It shares the same graph backend contract and KV session format as Metal and CUDA, and it supports SSD streaming for GLM 5.2 when the full model would not fit.

There is also a CPU reference and debug path, but the documentation is explicit that the CPU backend is not the production target. Normal inference should use Metal, CUDA, or ROCm. The CLI and server share the same KV session and snapshot format across all backends, so sessions remain portable across compatible builds.

## Performance

DwarfStar reports throughput at context frontiers rather than as a single whole-run average, using `ds4-bench` with a fixed token sequence and incremental prefill so each row measures only the newly added token interval.

On an 8xL40S NVIDIA server using the imatrix Q4 model with CUDA tensor parallelism, the server is reported to deliver 120 t/s aggregated generation and 2000 t/s prefill when serving multiple resident sessions with micro-batching. This is the configuration intended to turn a server with older Ada Lovelace cards into a multi-user LLM server for an organization.

Representative single-machine numbers from the project include the following, measured with the standard benchmark input and 128 greedy generation tokens at each frontier:

| Machine | Backend | Context | Prefill | Generation |
| --- | --- | ---: | ---: | ---: |
| MacBook Pro M5 Max, 128 GB | Metal | 2048 | 790.18 t/s | 39.35 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 65536 | 398.50 t/s | 27.64 t/s |
| DGX Spark GB10, 128 GB | CUDA | 2048 | 825.76 t/s | 18.05 t/s |
| DGX Spark GB10, 128 GB | CUDA | 65536 | 822.98 t/s | 13.84 t/s |

Distributed inference changes the picture. Pipeline parallelism across two M5 Max MacBooks on Thunderbolt 5 accelerates long prefills, with measured speedups of 1.38x at 9421 tokens up to 1.85x at 63819 tokens. Generation, however, is strictly autoregressive and pays at least one cross-machine hop per token, so distributed generation is slower than a single local process. Distributed inference is therefore mainly for fitting larger models and speeding up long prefills, not for faster decode.

Tensor parallelism over RDMA, in contrast, runs a single decode across two Macs at the same time. On two M5 Max 128 GB MacBooks running GLM 5.2, tensor parallel decode reaches about 16.8 t/s with full memory residency, compared to about 4.8 t/s for SSD streaming on one Mac. The `--power N` option lets users trade throughput for lower heat, fan noise, and battery drain by targeting a percentage of GPU usage.

## Installation

DwarfStar builds from source with `make`. First clone the repository and download a model. The `download_model.sh` script fetches verified GGUF files from Hugging Face, stores them under `./gguf/`, resumes partial downloads, and points `./ds4flash.gguf` at the selected main model.

```sh
git clone https://github.com/antirez/ds4.git
cd ds4

# Download a model (prefer the imatrix versions)
./download_model.sh ds4f-q2      # 96/128 GB RAM machines
./download_model.sh ds4f-q4      # >= 256 GB RAM machines
./download_model.sh glm-antirez-q2  # GLM 5.2 routed Q2_K
```

Then build for your platform. On macOS the default target is Metal. On Linux, plain `make` prints the available targets instead of selecting a CUDA target implicitly.

```sh
make                  # macOS Metal
make cuda-spark       # Linux CUDA, DGX Spark / GB10
make cuda-generic     # Linux CUDA, other local CUDA GPUs
make strix-halo       # Linux ROCm, AMD Strix Halo
make cpu              # CPU-only diagnostics build
```

For a known CUDA architecture, set `CUDA_ARCH` explicitly, for example when cross-building:

```sh
make cuda CUDA_ARCH=sm_89
make cuda CUDA_ARCH=native
```

The build produces the `ds4` CLI, the `ds4-server` HTTP server, the `ds4-agent` coding agent, the `ds4-bench` benchmark tool, and the `ds4-eval` capability evaluator. Run `./ds4 --help` and `./ds4-server --help` for the full flag list.

## Usage

### One-shot and interactive CLI

For a one-shot prompt, pass `-p`:

```sh
./ds4 -p "Explain Redis streams in one paragraph."
```

Without `-p`, the interactive multi-turn prompt starts. It keeps the rendered chat transcript and the live graph KV checkpoint, so each turn extends the previous conversation. Useful commands include `/help`, `/think`, `/think-max`, `/nothink`, `/ctx N`, `/read FILE`, and `/quit`. The CLI defaults to thinking mode; use `/nothink` for direct answers.

```sh
./ds4
ds4>
```

### SSD streaming for low-RAM machines

When the model does not fit in RAM, SSD streaming keeps non-routed weights resident while routed experts are loaded from the GGUF on cache misses. Start with the automatic cache budget:

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming --ctx 32768 --nothink
```

If startup reports the expert cache is too large, set the routed expert cache explicitly with a memory budget:

```sh
./ds4 -m ./ds4flash.gguf --ssd-streaming --ssd-streaming-cache-experts 32GB --ctx 32768
```

### HTTP server

Start a local OpenAI- and Anthropic-compatible server. Enabling the disk KV cache lets agent clients reuse expensive prefills across sessions and restarts.

```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

A minimal chat completion request against the local server:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

For multi-user serving on CUDA, keep multiple KV sessions resident so decode rows can be grouped across requests. The tested 8xL40S host is configured for up to 16 resident sessions:

```sh
./ds4-server --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000 \
  --batched-session 16 \
  --host 0.0.0.0
```

### Native coding agent

DwarfStar includes a native coding agent that controls inference from within the agent itself, with no socket or API boundary. The session is represented by the on-disk KV cache, tool calls are handled natively in the model's DSML format, and KV cache mismatches are impossible by construction. Agent sessions are stored under `~/.ds4/kvcache`; use `/save`, `/list`, and `/switch` to persist and resume sessions without a prefill stage.

```sh
./ds4-agent --cuda --cuda-tensor-parallel \
  --gpu-vram auto \
  --gpu-devices 0,2,4,6,1,3,5,7 \
  --model "$MODEL" \
  --ctx 100000
```

### Benchmarking

`ds4-bench` measures instantaneous prefill and generation throughput at context frontiers rather than a whole-run average, saving the live KV state to memory between frontiers and restoring it after a fixed greedy generation probe.

```sh
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

## Conclusion

DwarfStar is a focused experiment in what a native inference engine can be when it is built for a small set of models rather than for every GGUF in existence. By keeping model loading, prompt rendering, tool calls, KV state, the HTTP server, and the coding agent in one cohesive codebase, it removes the seams that usually cost latency and introduce KV mismatches. The aggressive 2-bit routed quantizations, validated against a 100-case quality fixture, let capable models like DeepSeek V4 Flash run on 96 GB Macs, while SSD streaming extends the same models to smaller machines.

The backends meet users where their hardware is: Metal on high-memory Macs, CUDA on multi-GPU NVIDIA servers including older Ada Lovelace cards, and ROCm on Strix Halo systems. The performance numbers are concrete, from 120 t/s aggregated generation on an 8xL40S server to roughly 16.8 t/s resident decode across two MacBooks using tensor parallelism over RDMA. The distributed pipeline parallelism mode shows a clear path to running the full DeepSeek V4 PRO across two 512 GB Mac Studios.

The project is also a candid example of modern AI-assisted development, built with strong assistance from GPT 5.5, 5.6, and Claude Fable while humans lead the ideas, testing, and debugging, and it openly acknowledges its debt to llama.cpp and GGML. It is beta quality and fast-changing, but for anyone who wants to run DeepSeek V4 Flash or GLM 5.2 locally with a tightly integrated, hackable engine, DwarfStar is well worth a close look. The code is on GitHub at [antirez/ds4](https://github.com/antirez/ds4) under the MIT license.
