---
layout: post
title: "Strata: Run a 125B MoE on a Gaming PC - Inside Niko1221/Strata"
description: "Strata runs the 125-billion-parameter Qwen3.8-Flash-Next on a 12 GB gaming GPU by splitting 24,576 MoE experts across VRAM, RAM, CPU and SSD. A source tour of the C++ engine, the expert memory tiers, and the speculative decoding that makes it fast."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /Strata-Run-A-125B-MoE-On-A-Gaming-PC/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/strata/niko1221-strata-architecture.svg
tags: [LLM, Inference, MoE, C++]
categories: [AI, Open Source]
keywords: Strata, Qwen3.8-Flash-Next, MoE inference, local LLM, CUDA, HIP, SYCL, speculative decoding, expert offloading
author: "PyShine"
---

Model sizes have grown far faster than consumer graphics cards. The largest open-weight models now ship as
hundreds of billions of parameters, and the usual answer is "rent a server". Strata, an MIT-licensed project by
Niko1221, takes the opposite road: it runs Qwen3.8-Flash-Next, a 125-billion-parameter mixture-of-experts model,
on an ordinary gaming PC with a 12 GB card, 32 GB of RAM and a big SSD. Nothing leaves the machine. The trick is
not compression alone - it is a scheduling system that treats your whole PC as one heterogeneous accelerator and
keeps every tier of memory busy at the same time.

The numbers in the README make the case better than any pitch. On an RTX 5070 (12 GB), Strata writes answers at
94 tokens per second in the Q2_0 size and reads 32K-token prompts at 2,650 tokens per second. On an RX 9070 XT
(16 GB), the same model writes 60 tokens per second. Those are speeds most people associate with hosted APIs,
not with a desktop tower. The codebase that delivers them is a serious piece of systems work: a C++ engine with
hand-written CUDA, HIP and SYCL kernels, an AVX-512 CPU expert pool, and a Python serving layer that speaks the
OpenAI, Anthropic and Responses APIs over localhost.

In this tour we walk the actual source tree: the installer that adapts to your hardware, the engine driver that
composes a token from 48 captured layer graphs, the three expert memory tiers, and the speculative decoding
controller that decides, step by step, whether guessing ahead will pay off. Everything cited here is a real file
in the repository - paths, structures and behaviors as they exist in the code today.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/strata/niko1221-strata-overview-architecture.svg" alt="Architecture overview of the Strata repository, from the installer through the serving layer to the engine and its expert memory tiers" style="min-width:720px;width:100%;max-width:1100px;" />
</div>
*Architecture overview of the repository: the installer launches the serving layer, the serving layer feeds the resident engine, and the engine draws on three expert memory tiers before streaming tokens back.*

Reading the overview from left to right:

- **Installation and entry.** `setup.py` is the installer: it detects your graphics card and RAM, picks a model
  size that fits, downloads it with resume support, and launches the stack. `START-HERE.bat` on Windows and
  `setup.sh` on Linux wrap the same flow, and `chat.py` gives you a terminal client once the server is up.
- **Python serving layer.** `serve/server.py` exposes OpenAI-compatible chat completions, the Anthropic messages
  endpoint and the Responses API, plus health, model and slot introspection. `serve/mcp.py` adds tools from MCP
  servers, and `serve/web/index.html` is the built-in browser app with its live monitor.
- **Engine core.** The server keeps one resident engine process and talks to it over stdin and stdout. The driver
  in `src/program/generate.cpp` composes each token, and `src/core/session.cpp` walks it through all 48 layers,
  deciding the order of work so nothing waits idly.
- **Expert memory tiers.** `src/core/expert_cache.cpp` keeps the most-used experts in VRAM, `src/kernels/cpu/pool.cpp`
  computes misses on the CPU in parallel with the GPU, and `src/ngram/ple_reader.cpp` fetches rows of a large
  n-gram table from the SSD. Results flow back into the engine and out to the client as a token stream.

## Why You Need This

The obvious question: why not just quantize harder and squeeze the model onto the card? Because a dense 125B
model will not fit in 12 GB no matter how hard you press, and aggressive quantization of a dense network costs
real quality. Strata sidesteps the dilemma by exploiting the structure of mixture-of-experts models. Qwen3.8-
Flash-Next contains 24,576 expert networks, but each token only needs 10 of them. You never have to hold all the
experts on the graphics card - you have to hold the right ten within microseconds of the router asking for them.

That changes the hardware conversation. Instead of "which GPU can fit this model", the question becomes "how do
I use the VRAM, RAM, CPU cores and SSD I already own together". A 12 GB card plus 32 GB of RAM plus a decent SSD
is hardware millions of people already have. Strata is the missing scheduler that turns that combination into a
125B-parameter reasoning system. Privacy is the other win: chats, code and documents never leave the machine,
which matters for anyone working with client code, medical text or anything covered by an NDA.

There is also a practical openness argument. The project is MIT licensed, uses parts of llama.cpp and ggml, and
ships an honest, detailed breakdown of what runs where. For developers building agentic tools, the included MCP
server support and the OpenAI-compatible endpoint mean existing tooling - Claude Code, Codex CLI, Cursor-style
clients - plugs in by changing a base URL.

## How It Works

The engine is the heart of the project, so the detailed view below focuses on how one token travels through the
48-layer network while three memory tiers cooperate underneath it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/strata/niko1221-strata-architecture.svg" alt="Detailed architecture of the Strata engine: serving layer, engine driver, session loop, speculative decoding, expert memory tiers and compute kernels" style="min-width:760px;width:100%;max-width:1200px;" />
</div>
*Detailed architecture of the engine: the serving layer fronts a resident process whose driver and session loop route each token through layer operations, speculative drafters, and the VRAM, CPU and SSD expert tiers.*

### Understanding the Architecture

**Weights come in through the artifact layer.** `src/artifact/gguf_reader.cpp` reads the GGUF container the model
ships in, and `src/core/weights.cpp` turns it into resident engine state. Notably, the large n-gram table that
serves as the SSD tier is not part of the main pack: as documented in `include/strata/kernels/ngram.hpp`, the
per-layer token embedding tensor is a 51.2-billion-element IQ4_NL tensor, about 28.8 GB, kept in its own GGUF
shard and read a few rows at a time through the operating system's cache.

**One driver composes one token.** The entry point in `src/program/generate.cpp` describes the whole job in one
line of its own comments: embed the token, run the 48 captured layer graphs with the CPU expert pool behind a
doorbell, apply the output head, sample, repeat. Captured graphs are the performance keystone. As
`include/strata/core/session.hpp` explains, a graph bakes in its arguments, so the graph unit is one layer and
per-token values arrive through fixed-address device buffers - and because the host, not the GPU, decides when
each layer runs, the CPU expert pool can start working on layer N while the graphics card is still finishing
layer N-1.

**The session loop owns state that outlives a layer.** `src/core/session.cpp` holds the gated residual stack that
every block reads and writes, plus two very different kinds of layer state: the model's 48 layers split into 12
QSA layers with KV-cache-style state and 36 GDN layers carrying recurrence and convolution history. The same file
implements the multi-GPU "layer-range carve", which is how two or three cards can share one model.

**The expert handoff is a verified contract.** `include/strata/core/expert_source.hpp` documents the interface
between the session loop, the router and the CPU pool clause by clause - the activation buffer must be the same
pinned memory every layer, the CPU pool must not apply the router weight because `moe_combine` in
`src/core/layer.cpp` does that summation on the device, and a wrong expert geometry must be refused rather than
mis-indexed. Reading comments like these is a masterclass in defensive systems design.

**Three tiers, one residency table.** `src/core/expert_cache.cpp` fills spare VRAM with the most frequently
routed experts and keeps adapting as your conversation shifts topics - roughly 700 more experts fit per extra
gigabyte of card memory. Misses go to `src/kernels/cpu/pool.cpp`, which computes them in place on pinned RAM
using AVX-512 and AVX2 kernels such as `src/kernels/cpu/iq_avx512.cpp`. The SSD tier, read by
`src/ngram/ple_reader.cpp`, hashes the last three token ids into 16 row indices per token and pulls only those
rows of the 28.8 GB table. The GPU kernels on the CUDA side - `src/kernels/cuda/router_top10.cu` for routing,
`src/kernels/cuda/native_moe.cu` for cached experts, `src/kernels/cuda/native_qsa.cu` and
`src/kernels/cuda/native_gdn.cu` for the two layer families - have matching SYCL ports under `sycl/` for Intel
Arc cards.

**Guess, then check, but only when it pays.** The speculative decoding stack is unusually disciplined.
`src/core/mtp.cpp` uses the model's own multi-token-prediction layer to draft up to 3 tokens, and
`src/spec/suffix_drafter.cpp` can draft up to 5 tokens by looking up repeated text such as quoted code. The
controller in `src/spec/controller.cpp` carries a cost model of exactly what each draft depth costs in dense
work, CPU misses and sync, starts from measured acceptance priors, and learns per session - it only drafts when
the expected gain beats the measured cost. Drafts are checked in one pass by the verify window in
`src/core/verify.cpp`, which the header documents as bit-for-bit identical to greedy decoding, with a commit step
that replays the accepted prefix into the recurrence state. Because the big model alone decides every word, the
output equals plain decoding - only sooner: 2.4 to 3.2 tokens per pass on average, a 1.6 to 1.8x speedup.

**Prompts are read in batched pieces.** Long documents go through `src/prefill/prefill.cpp` in up to 8,192-token
chunks at over 1,000 tokens per second, with the next layer's experts streaming over PCIe while the current
layer's attention runs. After the first message, `src/core/conversation_memory.cpp` keeps the conversation so
follow-ups start in seconds instead of re-reading everything.

**The serving layer is deliberately conventional.** `serve/server.py` implements chat completions with and
without streaming, the Anthropic messages format, the Responses API for Codex CLI, and a health/slot surface -
one sequence at a time behind a FIFO unless you enable batching, and requests that exceed the context are
rejected with a 400 rather than silently truncated. `serve/mcp.py` speaks JSON-RPC 2.0 to MCP servers over stdio
or Streamable HTTP, exposes their tools with server-prefixed names, and restarts a crashed tool server on its
next call. Vision support routes images through the model's mmproj projector, converting formats with Pillow
when needed.

End to end: a request lands on the local HTTP endpoint, the template in `serve/frontend.py` turns it into token
ids, the resident engine driver embeds them and walks the layers, the router names ten experts per token, the
VRAM cache answers hits while the CPU pool computes misses in parallel and the SSD supplies its few rows, the
drafters propose continuations that the verifier accepts or rejects, and tokens stream back out - 94 of them per
second on a card that technically holds less than one percent of the model.

## Advantages

- **Server-class model on consumer hardware.** A 125B-parameter MoE runs on a 12 GB card and 32 GB of RAM -
  hardware you may already own - instead of a multi-GPU rental.
- **No quality tax from the offloading.** The MTP and prompt-lookup drafts are verified bit-for-bit against
  greedy decoding, and the model's experts are preserved across all sizes except the code-focused one.
- **Every tier works simultaneously.** GPU, CPU, RAM and SSD all contribute concurrently, which is why miss
  latency does not dominate the token rate.
- **Honest, measurable scheduling.** The draft controller weighs expected gain against a measured cost model and
  learns from your actual sessions rather than guessing.
- **Real API compatibility.** OpenAI chat completions, Anthropic messages and the Responses API are all served
  locally, so coding agents and chat clients connect by pointing at a base URL.
- **Tool calling through MCP.** Local tools from configured MCP servers are offered to the model with clean
  namespacing and automatic crash recovery.

## Benefits

- **Privacy by default.** Conversations, code and documents never leave the PC; there is no account, no telemetry
  requirement and no per-token bill.
- **Fixed cost.** The model download happens once with resume support; after that, usage is free regardless of
  how many millions of tokens you generate.
- **Hardware flexibility.** NVIDIA through CUDA, AMD through HIP, and experimental Intel Arc support through a
  SYCL build, with community-tested notes for older GPUs.
- **Responsive long-context work.** Batched prefill reads 30,000-token prompts in about a minute the first time,
  and follow-ups start in seconds because the conversation state is retained.
- **A reference implementation worth studying.** The headers explain their invariants - the expert handoff
  contract, the n-gram hash's edge cases, the verify window's state replay - in unusual, verifiable detail.
- **Watchable operation.** The built-in web monitor shows GPU, CPU and RAM activity live, so you can see the
  expert tiers doing their work while an agent codes.

## Usage

Installation is a single entry point. Clone the repository, then on Windows run `START-HERE.bat` or on Linux:

```bash
./setup.sh
```

The installer asks which model size and context length you want, whether image input should be enabled, then
downloads roughly 70 GB (resumable if interrupted) and starts the app at `http://127.0.0.1:8080`. To add a
different model later, run `SETUP.bat` on Windows or `./setup.sh --setup` on Linux. `UPDATE.bat` and
`./update.sh` refresh the code without restarting the model.

To use Strata from your own tools, add an OpenAI-compatible provider with base URL `http://127.0.0.1:8080/v1` -
any API key and any model name are accepted. For Anthropic-protocol clients, set the base URL environment
variable:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080
```

For Codex CLI and other Responses-API clients, the same server exposes the `/v1/responses` route. To expose the
server to your phone or another machine, always pair it with a key:

```bash
START-HERE.bat --setup --host 0.0.0.0 --api-key YOURSECRET
```

Tools from MCP servers are enabled by listing them under `mcp_servers` in the run config or passing a separate
config file with `--mcp-config`. If you prefer to run the serving layer on its own, `serve/server.py` can also
start against a scripted mock engine, which is how the project tests every API path without a GPU. Reasoning
effort (off, low, medium, high) is selectable in the chat menu or through your client's reasoning-effort setting,
and enabling batching in the config lets a second request run in parallel at the cost of slower answers on a
12 GB card.

## Conclusion

Strata is a quiet rebuttal to the idea that frontier-scale inference belongs exclusively to datacenters. Its
premise is simple - a mixture-of-experts model only needs a few experts at a time - but the execution is deep:
captured per-layer graphs, a doorbell protocol between host and device, a residency table that learns your
conversations, a cost-model-driven drafting controller, and parity-tested kernels on three GPU platforms. The
source tree reads like a systems course, with headers that state their invariants and tests that observe them
rather than trust them.

If you have been waiting for local models to catch up with what hosted APIs offer, this is the project to watch.
Clone it, point your coding agent at localhost, and see how much of a 125B-parameter model your existing PC was
already capable of running.

**Links:**

- Repository: <https://github.com/Niko1221/Strata>
- Model: <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
- Compressed weights: <https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF>
- License: <https://github.com/Niko1221/Strata/blob/main/LICENSE>
