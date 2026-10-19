---
layout: post
title: "FlashInfer Source Tour: The Plan/Run Machine Behind Fast Attention"
description: "A source tour of FlashInfer, the kernel library behind vLLM and SGLang: the plan/run wrapper contract, host-side split-KV scheduling, JIT-compiled decode and prefill kernels, paged KV cache appends, cascade merges, and the AutoTuner."
date: 2026-10-19
header-img: "img/post-bg.jpg"
permalink: /flashinfer/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/flashinfer/flashinfer-ai-flashinfer-overview-architecture.svg
tags:
  - LLM
  - Python
  - Open Source
categories:
  - AI Inference
keywords: "FlashInfer, paged KV cache, attention kernels, split-KV scheduling, JIT compilation, CUDA, decode, prefill, cascade attention, autotuner, vLLM, SGLang"
author: "PyShine"
---

Every time you chat with a model served by [vLLM](/vllm-2/) or [SGLang](/sglang-3/), there is a good chance the attention step was executed by FlashInfer. The library sits one layer below the engines: engines decide *what* to run, FlashInfer decides *how* a batch of ragged, differently-lengthed requests gets turned into a handful of well-shaped CUDA launches. We toured the compiler side of this world with [Triton](/triton-2/) and the raw kernel side with [FlashAttention](/flash-attention/); FlashInfer is the connective tissue between the two, and its source is a masterclass in an API style that deserves more attention than it gets: plan once, then run many times.

The reason for that style is structural. A decode step over a paged KV cache cannot be described by tensor shapes alone — it depends on sequence lengths, page tables, and how much scratch memory the GPU can spare. FlashInfer's answer is a two-phase contract: a `plan()` call inspects all of that on the CPU, picks a schedule, allocates workspaces, and records a plan; a `run()` call then launches the prepared kernel with almost no host-side branching. In this tour we follow that contract through the wrappers, the host-side scheduler, the JIT kernel modules, the paged KV machinery, and the autotuner that quietly picks which compiled variant to trust.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/flashinfer/flashinfer-ai-flashinfer-overview-architecture.svg" alt="FlashInfer architecture overview" style="max-width:100%;"></div>

<p><em>The FlashInfer stack: plan/run wrappers at the top, host-side planning and JIT module loading in the middle, split-KV CUDA kernels below, with the AutoTuner steering tactic choices.</em></p>

Reading the overview from left to right:

- [BatchDecodeWithPagedKVCacheWrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L1194) and [BatchPrefillWithPagedKVCacheWrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L2719) are the two workhorse classes every engine talks to.
- Their [plan](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L1895) methods consult the host-side [DecodePlan scheduler](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh#L512) before any GPU work happens.
- [get_batch_decode_module](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L537) and [get_batch_prefill_module](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L571) load or JIT-compile the actual kernel binaries.
- The kernels live in headers: [decode.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh#L624) and [prefill.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/prefill.cuh#L5040).
- [page.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/page.py#L580) and [page.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/page.cuh#L399) handle writing new tokens into the paged cache.
- [AutoTuner](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/autotuner/autotuner.py#L1535) benchmarks candidate tactics per shape and caches the winners.

## The Plan/Run Contract

Open [decode.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L1194) and the wrapper reads like a state machine. `plan()` takes the whole batch description — sequence lengths, page table indirection, kv cache layout — and returns nothing you can multiply; what it returns is a filled workspace and a plan info blob. Then `run()` ([line 2787](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L2787)) does the real launch with the arguments the plan froze. The split exists because engines like vLLM run *many* decode steps against an unchanged batch structure: paying the scheduling cost once per step instead of once per token is the whole point.

The prefill side mirrors it in [BatchPrefillWithPagedKVCacheWrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L2719). Its [plan](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L3388) has more to decide — query lengths vary per request, custom masks are allowed, and ragged (non-paged) KV is supported too — while [run](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L4217) stays thin. There is also a single-request convenience path, [single_prefill_with_kv_cache](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L1271), for workloads that never batch.

One subclass shows why this design survives production: [CUDAGraphBatchDecodeWithPagedKVCacheWrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L3536) exists so the same decode step can be captured into a CUDA graph. Graph replay forbids host-side decisions at launch time, which is exactly what the plan/run split guarantees — after `plan()`, `run()` is deterministic enough to record.

The plan phase is also where backend selection happens, and the source is unusually transparent about it. On the prefill side, [get_fmha_module](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L213) loads the canonical cuDNN-free FMHA modules, while [get_customize_batch_prefill_module](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py#L279) exposes a mechanism for callers to register their own kernel modules under a name the wrapper can resolve later. Engines exploit this to swap in architecture-specific kernels without forking the wrapper code. The plan call resolves which module actually fits the current dtype, head dimension, and device — so a vLLM deployment on Hopper and one on Blackwell can run identical Python while hitting completely different binaries underneath.

## Planning on the Host, Executing on the Device

The scheduler is where the interesting arithmetic lives, and it is pure C++ in [scheduler.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh#L512). Decode work per request is proportional to its KV length, so the scheduler partitions the KV dimension into chunks and assigns chunks to thread blocks — the split-KV strategy [FlashAttention](/flash-attention/) popularized, but computed per batch on the host rather than baked into the kernel. [DecodePlanImpl](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh#L426) walks batch sizes against the available smem and registers budget, producing a `DecodePlanInfo` ([struct at line 366](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh#L366)) that later serializes into a plain int64 vector — deliberately boring, deliberately stable.

Prefill planning needs a search instead of a partition, because a single long request can dominate a batch. [PrefillBinarySearchKVChunkSize](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh#L101) binary-searches the largest KV chunk that still fills the GPU, balancing load across requests of wildly different lengths. The elegance here is that all of this is host code compiled into the Python extension — no separate scheduler service, no graph compiler, just a well-tested C++ function the wrapper calls before every fresh batch.

## The Paged KV Cache Layer

Engines do not store KV contiguously; they store it as pages in a pool so sequences can grow without reallocation. FlashInfer accepts that layout as ground truth. On the Python side, [page.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/page.py#L309) builds batch indices and positions from the page indirection, and [append_paged_kv_cache](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/page.py#L580) is the entry point engines call after every generated token. There is a parallel MLA variant ([append_paged_mla_kv_cache](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/page.py#L411)) for the compressed latent layout DeepSeek models use.

The CUDA side is refreshingly small for how load-bearing it is: [AppendPagedKVCacheKernel](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/page.cuh#L399) and its decode-specialized twin ([line 363](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/page.cuh#L363)) copy new K/V rows into their page slots, handling the case where one token's data straddles a page boundary. This is unglamorous code, but it is the boundary contract between an engine's memory manager and FlashInfer's kernels — get it wrong and every downstream attention result is silently wrong.

Newer hardware brings a twist: writing the cache can be fused with quantizing it. [nvfp4_quantize_append_paged_kv_cache](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/page.py#L710) and its CUDA counterpart ([NVFP4QuantizeAppendPagedKVCache at line 621](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/page.cuh#L621)) quantize incoming FP16/BF16 tensors to NVFP4 blocks during the append itself, halving cache traffic on Blackwell-class GPUs without a separate quantization launch. The same trick exists for the MLA layout. It is a small example of a big theme: on modern accelerators, memory movement *is* the compute budget, so FlashInfer keeps collapsing separate memory passes into the ops that were touching the data anyway.

## One Kernel Family, Many Backends

FlashInfer is often described as "one library," but the source shows a family. The canonical headers compile per (head_dim, dtype, pos-encoding) combination through the JIT path: [get_batch_decode_module](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L537) checks a cache, and on miss invokes nvcc through the `jit/` machinery to produce a `.so` for exactly your shapes. First call is slow; every call after is a dlopen away.

Beyond the canonical kernels sit the vendor paths. [trtllm_batch_decode_with_kv_cache](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L3891) routes to NVIDIA's trtllm-gen kernels — the ones behind [TensorRT-LLM](/tensorrt-llm-2/) — while [xqa_batch_decode_with_kv_cache](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L4773) exposes the XQA kernel, and [fast_decode_plan](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py#L4978) is the planner that understands when each of these backends is eligible. The decode dispatcher in [decode.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh#L756) shows the same pattern inside a single backend: the standard kernel ([line 624](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh#L624)) for GQA shapes, and a dedicated MLA kernel ([line 910](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/decode.cuh#L910)) when the cache is in DeepSeek's compressed layout. Prefill mirrors all of it in [prefill.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/prefill.cuh#L5040) with a ragged-KV twin at [line 4799](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/prefill.cuh#L4799).

If you read [our ONNX Runtime tour](/onnxruntime-2/), this is the same "many execution providers, one API" idea rebuilt for attention — except here the provider choice happens per batch at plan time, not per model at load time.

## Cascades, Merges, and the Autotuner

Two quieter subsystems round out the library. The first is cascade attention: when thousands of requests share a long system prompt, recomputing attention over the shared prefix per request is waste. [MultiLevelCascadeAttentionWrapper](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cascade.py#L226) runs attention once per prefix level and then merges the partial results — because attention outputs with their log-sum-exp statistics form a mathematically clean algebra, merging two partial attentions is exact, not approximate. The merge kernels in [cascade.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/cascade.cuh#L45) ([batched version at line 220](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/cascade.cuh#L220), persistent variant at [line 374](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/cascade.cuh#L374)) implement that algebra on-device, and the Python side exposes it through [merge_state](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cascade.py#L42) and the in-place [merge_state_in_place](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/cascade.py#L109).

The second is the AutoTuner ([class at line 1535](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/autotuner/autotuner.py#L1535)). Many kernels have several valid tactics — tile choices, split counts — and which one wins depends on your exact shapes. The tuner wraps tunable ops, profiles each tactic the first time a shape appears ([TuningConfig](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/autotuner/autotuner.py#L454) and [DynamicTensorSpec](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/autotuner/autotuner.py#L420) describe the shape buckets), and persists winners to disk so the cost is paid once per machine. vLLM and SGLang both flip it on in production; it is the difference between "FlashInfer is fast" and "FlashInfer is fast on *your* shapes."

## From Install to Your First Decode

Getting started is `pip install flashinfer-python` plus a CUDA 12+ toolkit for the JIT path. A minimal decode loop follows the contract directly: allocate a workspace buffer, construct `BatchDecodeWithPagedKVCacheWrapper`, call `plan()` with your float and int workspaces, then call `run()` per step. Auxiliary ops follow the same pattern — [rope kernels](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/pos_enc.cuh#L650) for positional encoding, and a full sampling family in [sampling.cuh](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/sampling.cuh#L859) (top-k, top-p, min-p kernels) so engines never round-trip sampling through slow fallbacks.

Being honest about the limits, because the source is: the JIT path needs a working nvcc at runtime, so first-plan latency on a fresh shape can be seconds — autotuning multiplies that if you enable it cold. The MLA and FP8/FP4 paths are tuned for specific architectures, and [the attention core](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/attention/_core.py#L65) shows the API still migrating between the classic wrappers and newer module forms, which means engine integrations pin versions carefully. This is infrastructure that assumes you run an engine, not a chat app.

That positioning is what makes the source worth reading. [llama.cpp](/llama.cpp-2/) sells portability by avoiding CUDA abstractions; [MLC](/mlc-llm-2/) and [TVM](/tvm-2/) compile everything ahead of time; FlashInfer sits deliberately in between — ahead-of-time architecture, just-in-time kernels, per-batch planning, and an autotuner closing the loop. For the bigger picture of where it sits among serving stacks, see [the inference index](/inference/), and for the engine that consumes it most visibly, [the vLLM tour](/vllm-2/).

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "FlashInfer Source Tour: The Plan/Run Machine Behind Fast Attention",
  "description": "A source tour of FlashInfer: plan/run wrappers, host-side split-KV scheduling, JIT-compiled attention kernels, paged KV cache appends, cascade merges, and the AutoTuner.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-19",
  "keywords": "FlashInfer, paged KV cache, attention kernels, split-KV scheduling, JIT compilation, CUDA, decode, prefill, cascade attention, autotuner"
}
</script>
