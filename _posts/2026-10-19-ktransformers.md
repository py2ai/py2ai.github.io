---
layout: post
title: "KTransformers Source Tour: How kt-kernel Runs Giant MoE Models on CPU Plus GPU"
description: "A guided walk through the kt-kernel source tree: AMX and AVX2 expert kernels, Quest-style sparse attention, a NUMA worker pool, and the CUDA helpers that tie CPU and GPU together."
date: 2026-10-19
header-img: "img/post-bg.jpg"
permalink: /ktransformers/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ktransformers/kvcache-ai-ktransformers-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: [KTransformers, kt-kernel, AMX, MoE inference, sparse attention, NUMA, GPTQ Marlin, CPU GPU hybrid]
author: "PyShine"
---

KTransformers earned its reputation by doing something that sounds impossible: running DeepSeek-class mixture-of-experts models on a workstation GPU paired with a big pile of ordinary DDR memory. The trick is placement, not magic — dense, attention-critical work lives on the GPU, while the enormous expert feed-forward layers, which see only a few experts per token, stream from CPU RAM through heavily optimized kernels. A README benchmark line puts DeepSeek-R1-0528 (FP8) at 227.85 tokens/s total throughput and 87.58 output tokens/s at 8-way concurrency on 8×L20 GPUs plus a Xeon Gold 6454S. If you have followed our earlier CPU-inference tours of [llama.cpp](/llama.cpp-2/) and [BitNet](/bitnet/), this project sits at an interesting midpoint: llama.cpp keeps everything on CPU, while KTransformers treats the CPU as a first-class co-processor beside the GPU.

There is a second reason this tour is timely. The repository rebuilt itself. Today's live tree is **kt-kernel**, a slimmer C++/Python library whose README promises "CPU-optimized kernel operations for heterogeneous LLM inference" plus a fine-tuning track with LlamaFactory. The historic Python servers — the OpenAI-compatible API, the balance-serve scheduler, the old kvc2 prefix cache — now live under `archive/`, preserved but no longer the point. Reading the old tree to understand the new one wastes effort, so this walk sticks to the files that ship today. That also makes KTransformers a useful contrast to our [DeepSpeed](/deepspeed-2/) coverage: DeepSpeed moves tensors to CPU as an offload afterthought, while kt-kernel was designed around the CPU from the first line.

The map below shows the whole engine. A thin Python CLI and loader family hand weights to a pybind11 bridge; the bridge exposes operator interfaces implemented twice — once with Intel AMX tiles, once in portable AVX2 — while a paged KV cache serves Quest-style sparse attention, and a NUMA-aware worker pool keeps dozens of cores fed.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/ktransformers/kvcache-ai-ktransformers-overview-architecture.svg" alt="KTransformers kt-kernel overview architecture" style="max-width:100%;"></div>

<p><em>Overview of kt-kernel: the Python frontend and loaders feed the native bridge; operator interfaces dispatch to AMX or AVX2 expert kernels and the block-sparse attention path, all running over a NUMA worker pool, with CUDA ops handling GPU-side GPTQ and gating.</em></p>

Reading the overview from left to right:

- `kt run` ([run.py, L85](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/cli/commands/run.py#L85)) resolves a model, sanity-checks environment variables, and either boots the engine directly or builds a [SGLang command](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/cli/commands/run.py#L606) that loads kt-kernel as its CPU expert backend — the same integration path our [SGLang](/sglang-3/) tour covered from the serving side.
- The loader family in [loader.py, L816](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/loader.py#L816) reads GGUF checkpoints (a format our [llama.cpp](/llama.cpp-2/) tour traced back to its C++ origin) and a whole zoo of safetensor precisions, then translates tensor names into kt-kernel's slots.
- The native bridge ([ext_bindings.cpp, L561](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/ext_bindings.cpp#L561)) publishes the operator contracts from [common.hpp, L418](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/common.hpp#L418) and stamps `__cpu_variant__` attributes onto the module so Python can pick AMX, AVX-512, or AVX2 code at import time.
- On the GPU side, a separate extension ([binding.cpp, L22](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cuda/binding.cpp#L22)) registers GPTQ Marlin GEMM, top-k softmax gating, and GGUF dequantization kernels — a vocabulary shared with [vLLM](/vllm-2/), from which the gating kernel is explicitly adapted.
- Underneath everything, [task_queue.h, L22](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/task_queue.h#L22) and [worker_pool.h, L138](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.h#L138) provide the NUMA-aware execution fabric every kernel runs on.

Now let's descend into each layer.

## A Framework That Rebuilt Itself Around kt-kernel

The first surprise in the repo listing is how small the live surface is. `archive/` holds several generations of the old project, including the C++ sources of the original KTransformers servers. The active tree is `kt-kernel/`, and its shape tells you the architecture story immediately: `operators/` for the math, `cpu_backend/` for threading, `cuda/` for GPU helpers, `python/` for the CLI and loaders, plus one top-level `ext_bindings.cpp` gluing it together.

The Python entry ([run.py, L170](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/cli/commands/run.py#L170)) shows how deliberately thin the frontend is. `_run_impl` checks for conflicting environment variables ([L586](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/cli/commands/run.py#L586)), picks between local serving and delegating to SGLang, and otherwise stays out of the way. This is the opposite philosophy from [Triton Inference Server](/triton-server/), which owns the whole serving surface; kt-kernel is a kernel library that deliberately lets a real server own the HTTP.

The import-time variant detection is the cleverest bridge detail. [ext_bindings.cpp, L565](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/ext_bindings.cpp#L565) attaches attributes like `"amx"`, `"avx512_bf16"`, `"avx512_vnni"`, and `"avx2"` to the module, alongside `__int8_weight_layout__` at [L583](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/ext_bindings.cpp#L583). Python code inspects these flags once and binds the right specialization, so no runtime dispatch tax is paid inside the hot loop. The config structs that flow through the bridge live in [common.hpp, L143](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/common.hpp#L143): a `GeneralConfig` baseline, `QuantConfig` at [L222](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/common.hpp#L222), `GeneralMOEConfig` at [L230](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/common.hpp#L230), and the two abstract contracts — `MLA_Interface` at [L412](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/common.hpp#L412) and `MoE_Interface` at [L418](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/common.hpp#L418) — that every backend implementation must satisfy. Nothing in the Python layer knows whether AMX or AVX2 will answer.

## CPU MoE Experts: AMX Tiles and the CRTP Stack

The heart of kt-kernel is its mixture-of-experts execution on CPU, and the AMX path is the flagship. Intel's Advanced Matrix Extensions add tile registers and TMUL dot-product instructions; exploiting them requires weights staged in exactly the right layout. The class hierarchy uses curiously-recurring templates: [AMX_MOE_BASE, L39](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/moe_base.hpp#L39) provides the shared machinery, and each quantization flavor derives from it, while [AMX_MOE_TP, L82](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/moe.hpp#L82) wraps per-NUMA-partition instances behind the tensor-parallel facade of `TP_MOE` at [L427](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/moe.hpp#L427).

Weight loading is half the performance battle. [load_weights, L263](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/moe.hpp#L263) repacks raw quantized expert bytes into the tile layout AMX demands, with integrity checks like `verify_load_right` at [L154](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/moe.hpp#L154) to catch silent layout corruption. The B-side tile construction lives in [amx_buffers.hpp, L499](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/la/amx_buffers.hpp#L499), where `BufferBInt4Impl` copies each 16×16 tile and applies the VNNI transpose (self-inverse, so the same routine handles both directions) visible at [L704](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/la/amx_buffers.hpp#L704). On the A side, [BufferAImpl, L20](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/la/amx_buffers.hpp#L20) stages activations with k-group variants for different block-quantized weight shapes.

The forward pass is two GEMMs. [do_gate_up_gemm, L238](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/moe.hpp#L238) projects each token's hidden state through the selected experts' gate and up matrices, then applies SwiGLU in AVX-512 registers — the activation function sits in [amx.hpp, L47](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/la/amx.hpp#L47), and `init_tile` at [L157](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/la/amx.hpp#L157) configures the TMUL tile dtypes once per compute type. [do_down_gemm, L251](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/amx/moe.hpp#L251) finishes the expert FFN by projecting back down and accumulating into per-partition outputs. Because each NUMA partition owns its own experts' weights, most memory traffic stays on one socket — which is exactly why the thread pool below matters so much.

For hosts without AMX — AMD EPYC machines, older Intel boxes — [AVX2_MOE_BASE, L39](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/avx2/moe_base.hpp#L39) implements the same contract in portable intrinsics, with its own weight staging at [L185](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/avx2/moe_base.hpp#L185). The release notes flag AVX2-only support as the path that brought AMD servers into LoRA fine-tuning, and the Python-side wrapper [GeneralMoEWrapper, L29](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/moe_kernel.py#L29) is where a serving framework's MoE layer gets bound to whichever backend the CPU supports.

## Quest-Style Sparse Attention Over Paged KV

Long-context MoE models die on attention long before they die on experts, so kt-kernel's second pillar is a block-sparse KV cache in the Quest style: keep the full cache, but let each decode step look closely at only the most promising blocks. The manager side is [KVCache, L178](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache.h#L178), sized by [KVCacheConfig, L111](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache.h#L111), with `ThreadResize`, `BatchResize`, and `BlockResize` entry points at [L201](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache.h#L201) through [L223](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache.h#L223) handling the three axes of concurrency.

The attention driver is [attn_with_kvcache, L499](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache_attn.cpp#L499), which selects between the layer-oriented and KV-head-oriented code paths (the latter starting at [attention_kvhead_, L18](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache_attn.cpp#L18) for multi-head variants). The sparse sequence then unfolds in four steps. First, [quantize_q_, L532](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache_attn.cpp#L532) compresses queries to integers so block scoring becomes cheap integer dot products. Second, [calculate_block_similarity_layer_, L608](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache_attn.cpp#L608) scores every cached block against those queries. Third, [select_block_layer_, L737](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache_attn.cpp#L737) keeps only the top blocks under the sparsity budget, and [retrieval_kvcache_layer_, L799](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache_attn.cpp#L799) gathers their K/V pages into a compact working set for the real (full-precision) attention kernel. Finally, [calculate_sparsity_layer_, L849](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache_attn.cpp#L849) records how much of the cache actually got touched, and [get_attn_sparsity, L1514](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/kvcache/kvcache_attn.cpp#L1514) exposes that number upward so schedulers can adapt budgets.

GPU readers of our [FlashInfer](/flashinfer/) tour will recognize the shape of this idea: approximate, structure-aware attention that trades a little recall for a large bandwidth win. The difference is substrate. FlashInfer optimizes what the GPU touches; kt-kernel optimizes what leaves DRAM at all, which on a CPU-bound hybrid stack is the resource that matters.

## The NUMA Backend: Queues, Pools, Work Stealing

Every kernel above assumes a pool of workers that understand socket topology, and that fabric is worth its own section. [ThreadState, L63](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.h#L63) is the per-thread record — cache-line aligned with `alignas(64)` to stop false sharing, holding core and NUMA node ids plus scratch space. `InNumaPool` at [L72](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.h#L72) allocates those states from the right node, and [NumaJobDistributor, L105](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.h#L105) splits jobs across nodes so an expert computes on the socket that holds its weights.

The [WorkerPool, L138](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.h#L138) constructor wires it together, `init` at [L159](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.h#L159) pins threads, and the escape hatch for uneven work is [do_work_stealing_job, L153](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/worker_pool.h#L153): when the static NUMA split leaves one socket idling, its threads steal remaining indices from the busy one. Above the pool, [TaskQueue, L22](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/task_queue.h#L22) offers the tiny API kernels actually use — `enqueue` at [L27](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/task_queue.h#L27) to submit, `sync` at [L29](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cpu_backend/task_queue.h#L29) to drain — keeping synchronization out of the kernel bodies.

## From Install to Your First Model

Installation follows the kernel-library pattern — no service daemon, no config zoo:

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers/kt-kernel
pip install .
```

`kt --help` then reveals the CLI surface: `run` for serving, `chat` for interactive sessions, `bench` for throughput, `doctor` for hardware capability checks, plus `quant` and `sft` for the fine-tuning track. A minimal local launch looks like:

```bash
kt run --model-path /models/DeepSeek-R1-0528 --gguf-path /models/gguf
```

The loader negotiates precisions at startup. [loader.py, L19](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/loader.py#L19) enumerates the GGML quantization types it understands, and the safetensor family covers the modern zoo: FP8 at [L332](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/loader.py#L332), GPTQ at [L1081](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/loader.py#L1081), MXFP4 at [L1198](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/loader.py#L1198), NVFP4 at [L1312](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/loader.py#L1312), and MXFP8 at [L1405](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/loader.py#L1405). Name translation — mapping a checkpoint's tensor names into kt-kernel's expectations — is centralized in `translate_name_to_gguf` at [L89](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/utils/loader.py#L89), which is the single most useful file to read when a new model architecture lands.

GPU experts get their weights through the CUDA extension. [binding.cpp, L94](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cuda/binding.cpp#L94) registers `gptq_marlin_gemm`, backed by the warp-specialized Marlin kernel at [gptq_marlin.cu, L60](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cuda/gptq_marlin/gptq_marlin.cu#L60) — the same INT4 tensor-core design our [vLLM](/vllm-2/) tour saw in production — and [binding.cpp, L97](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cuda/binding.cpp#L97) registers `topk_softmax`, whose fused gating kernel ([topkGatingSoftmax, L164](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/cuda/moe/moe_topk_softmax_kernels.cu#L164)) computes softmax, max, argmax, and top-k selection in one launch, adapted from vLLM's kernels as the file header notes. The placement story this enables is exactly what the README advertises: hot experts on GPU, cold experts on CPU.

The fine-tuning track deserves its own tour, but the shape is worth knowing: KTransformers × LlamaFactory integration lets MoE experts stay quantized on CPU while LoRA adapters train on GPU, and the team reports 6-12× faster training than ZeRO-Offload baselines on the same hardware — a claim we flag as vendor-reported, not independently verified.

## Honest Limits and Where kt-kernel Fits

Limits first, since they decide whether this tool belongs in your stack. It is x86-centric: the AMX path needs Sapphire Rapids-or-newer Intel, and while the AVX2 fallback widens coverage to AMD, ARM is off the table for the CPU side. The vendor list in `cpu_backend/vendors/` hints at Ascend NPU and HIP ports, but the mature path is CUDA-plus-Intel-CPU. Attention sparsity is approximate — block scoring can miss what full attention would catch, so quality-sensitive deployments should measure, not assume. And the repository is mid-pivot: tutorials and tooling still straddle the old servers and the new kernel, so documentation occasionally points at `archive/` paths that are no longer load-bearing.

As for fit: kt-kernel is the answer when model size, not request volume, is your constraint — the single-node case where a 600GB-class MoE model must serve a handful of users on hardware you already own. For multi-tenant serving with routing and batching at scale, the SGLang integration hands those problems to a dedicated server, and for cloud-scale deployments you would compare against the hosted paths in our [inference overview](/inference/) or the serving-stack tours of [Triton](/triton-server/) and [MLC-LLM](/mlc-llm-2/). Read it as a specialization of the same lesson llama.cpp taught: when the bottleneck is memory bandwidth, the cheapest spare memory in the building is usually the DIMMs you already have — and KTransformers shows what it takes to make those DIMMs fast.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "KTransformers Source Tour: How kt-kernel Runs Giant MoE Models on CPU Plus GPU",
  "description": "A guided walk through the kt-kernel source tree: AMX and AVX2 expert kernels, Quest-style sparse attention, a NUMA worker pool, and the CUDA helpers that tie CPU and GPU together.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-19",
  "keywords": "KTransformers, kt-kernel, AMX, MoE inference, sparse attention, NUMA, GPTQ Marlin, CPU GPU hybrid"
}
</script>
