---
layout: post
title: "One Booster, Every Parallelism - Inside hpcaitech/ColossalAI"
description: "A source-level tour of hpcaitech/ColossalAI: how the Booster facade, a plugin family spanning 3D parallelism and ZeRO, chunk-based Gemini memory, and the Shardformer make large model training fit on the hardware you have."
date: 2026-10-16
header-img: "img/post-bg.jpg"
permalink: /colossalai/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/colossalai/hpcaitech-colossalai-overview-architecture.svg
tags: [ColossalAI, distributed training, ZeRO, tensor parallelism, HPC-AI, Gemini, Shardformer]
categories: [AI, Open Source]
keywords: "ColossalAI source code, hpcaitech ColossalAI, Booster plugin, HybridParallelPlugin, GeminiDDP chunk manager, LowLevelZeroOptimizer, Shardformer policies, DTensor layout, FP8 quantization, colossalai run CLI"
author: "PyShine"
---

Training a modern large model is a combinatorics problem disguised as a systems problem: data parallel or tensor parallel or stage parallel, optimizer states on GPU or CPU, activations recomputed or offloaded, and every combination trades memory against throughput differently. hpcaitech/ColossalAI, the Apache-2.0 project from HPC-AI Technology, is one of the most complete open answers: a single training facade called Booster, behind which a family of plugins can turn on any mix of data, tensor, and stage parallelism, ZeRO-style optimizer sharding, chunk-based CPU offloading, FP8 mixed precision, and mixture-of-experts, all without rewriting the model. The project's own README benchmark is aggressive about it: a 70B Llama-like model at 811.79 TFLOPS per GPU on 16 H200s with a hybrid layout, and B200 nodes beating H200 by 50 to 70 percent on throughput. As always in this series, what follows is an educational tour of published source code.

The architecture is a stack of increasingly clever wrappers. At the top sits Booster, a facade class whose boost method accepts any model, optimizer, and plugin and returns wrapped versions that behave like the originals. Plugins decide everything: the flagship HybridParallelPlugin composes data, tensor, and stage parallelism with three optimizer tiers, while GeminiPlugin brings the chunked memory system that migrates parameters between GPU and CPU in large blocks. Underneath, the Shardformer rewrites transformer architectures module by module for tensor parallel execution, guided by per-model policy files, and the DTensor layout engine supplies the distributed math. Around the core live an inference system with its own KV cache and server, Triton kernels for attention primitives, and MoE operations. Reading the overview from left to right follows one training step from the facade, through plugin selection, into the memory and sharding machinery.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/colossalai/hpcaitech-colossalai-overview-architecture.svg" alt="ColossalAI repository architecture overview" style="max-width:100%;"/></div>

<p><em>Overview of ColossalAI: the Booster facade delegates to a plugin family, Gemini chunks migrate parameter memory, and the Shardformer plus DTensor engine handles tensor-parallel surgery.</em></p>

Reading the overview from left to right:

- [Booster](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/booster.py) is the facade at line 33: boost, backward, no_sync, LoRA helpers, and model, optimizer, and scheduler save and load all route through it.
- [colossalai.initialize](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/initialize.py) is the 157-line boot sequence that builds the parallel context from launcher environment variables.
- [Plugin](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/plugin_base.py) is the 74-line contract every parallel strategy implements, deciding how models, optimizers, and dataloaders get wrapped.
- [HybridParallelPlugin](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/booster/plugin/hybrid_parallel_plugin.py) is the flagship at line 929, composing data, tensor, and stage parallelism with naive, AMP, and ZeRO optimizer tiers.
- [GeminiDDP](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/zero/gemini/gemini_ddp.py) is the 875-line memory wrapper that keeps parameters in chunks which migrate on demand.
- [ChunkManager](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/zero/gemini/chunk/manager.py) plus chunk.py pack parameters and optimizer states into large, pinned, movable memory blocks.
- [LowLevelZeroOptimizer](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/zero/low_level/low_level_optim.py) partitions gradients and optimizer states across the data-parallel group, ZeRO stage 1 and 2 style.
- [Shardformer](https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/shardformer/shard/shardformer.py) is the 43-line engine that, given a policy file such as the 487-line LlamaPolicy, splits a transformer for tensor parallel training.

## Why You Need This

The benchmark table in the README is the honest sales pitch, and it is worth reading closely. On 8 H200 GPUs training a 7B Llama-like model with ZeRO-2 data parallelism, Colossal-AI reports 17.13 samples per second at 534.18 TFLOPS per GPU. Switch to a 70B model on 16 H200s with the same ZeRO setup and it holds 3.27 samples per second at 469.1 TFLOPS. The hybrid layouts are where it gets interesting: zero1 plus tensor-parallel-2 plus 4-way stage splitting reaches 805.69 TFLOPS per GPU on 8 B200s for the 7B model and 811.79 on 16 H200s for the 70B, with peak memory around 100GB per card. The project's headline numbers, 50 percent higher throughput for the 7B and over 70 percent for the 70B when moving to B200, are all measured with these in-repo configurations, and the benchmark scripts sit in the examples tree for verification.

The second reason is architectural: Booster is one of the cleanest plugin systems in the training-framework space. The facade at line 33 of booster.py exposes about thirty public methods, from boost at line 126 through backward at 175, load_model at 291, save_model at 315, optimizer save and load at 352 and 372, and LoRA helpers enable_lora at 240 and save_lora_as_pretrained at 418. Every one of those delegates to the active plugin, which means the difference between pure data parallel, FSDP, ZeRO, Gemini offloading, or full 3D parallelism is a config object, not a code change. A 44-line accelerator shim abstracts the hardware calls underneath, and the 157-line initialize module wires the parallel context from the launcher environment before any of it runs.

The third reason is the memory science. Gemini, the chunked memory system, is Colossal-AI's answer to the question of what to do when parameters, gradients, and optimizer states do not fit: pack them into large chunks rather than moving tensors one by one. chunk.py, at 560 lines, implements a chunk as a contiguous memory region with a GPU window and a pinned CPU shard, and manager.py, at 270 lines, allocates and tracks them. A placement policy module, 240 lines, decides what lives where, and gemini_hook.py moves data at the four boundaries of a training step: before and after forward, before and after backward. The companion LowLevelZeroOptimizer partitions gradients through a bookkeeping layer of bucket and gradient stores, so stage 1 and 2 sharding work with asynchronous reduction hooks. Reading these files is a graduate course in memory management.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/colossalai/hpcaitech-colossalai-architecture.svg" alt="ColossalAI detailed architecture with plugins and Gemini chunks" style="max-width:100%;"/></div>

<p><em>Detail view: the Booster facade and its entry methods, the full plugin family, the Gemini chunk machinery from manager to hooks, the Shardformer stack, and the MoE, FP8, and kernel support layer.</em></p>

The HybridParallelPlugin file is where the composition happens, and its class layout tells the story. HybridParallelModule, at line 60 of hybrid_parallel_plugin.py, is the model wrapper that knows about tensor-parallel groups and stage boundaries; HybridParallelNaiveOptimizer at line 279 is the plain data-parallel path; HybridParallelAMPOptimizer at line 491 adds fp16 and bf16 loss scaling; and HybridParallelZeroOptimizer at line 667 extends the LowLevelZeroOptimizer so that ZeRO sharding works on top of the other two parallel axes simultaneously. The plugin itself, at line 929, extends the stage-parallel plugin base, which is why the same config can express the zero1-plus-TP2-plus-4-way-stage layout from the benchmark table. A separate MoE hybrid variant exists for expert models, and there are baseline torch DDP and FSDP plugins for apples-to-apples comparisons. The facade covers the interaction patterns a multi-axis run actually needs: Booster.no_sync at line 223 creates accumulation windows where gradients skip cross-rank reduction, and the scheduler save and load methods at lines 400 and 409 round-trip learning-rate state alongside the optimizer, so a resumed 3D run restarts warm rather than cold.

The project's ecosystem is evidence the abstraction survives contact with real workloads. Open-Sora, the team's video-generation effort, trains its diffusion transformers on this stack; ColossalChat used it for the full RLHF sequence on chat models; Colossal-LLaMA-2 demonstrated domain-specific fine-tuning in a half-day of training on rented hardware; and a February 2025 guide walked through fine-tuning DeepSeek's 671B-parameter model with the upgraded toolchain. Those projects exercise very different corners of the plugin matrix, video generation favoring activation-heavy stage splits, RLHF favoring the inference side, and 671B-scale tuning leaning entirely on the chunked memory system. A library that hosts all four under one facade has put its abstraction where the workloads are.

The Shardformer is the component outsiders should study first, because tensor parallelism is usually the most painful part to implement by hand. The engine, shardformer.py, is only 43 lines: it takes a model, a ShardConfig of 110 lines of flags, and a policy, then performs the module surgery. The policies folder holds one file per architecture, and LlamaPolicy at 487 lines shows the pattern: each attention, MLP, and normalization sublayer is annotated with the sharding spec it should receive, plus optimizations like fused kernels where the groups allow. The DTensor engine underneath, in the d_tensor package with api, comm_spec, sharding_spec, and layout_converter modules, computes and converts the distributed layouts so that a matrix split four ways reduces correctly. Getting this wrong produces silent divergence; the policy files are effectively reviewed recipes.

The support layer rounds out the picture. The quantization package ships fp8.py and fp8_hook.py, the one-line FP8 mixed-precision integration the project blog credits with cutting training cost by up to 30 percent, plus bitsandbytes integration in bnb.py. The kernel package holds Triton implementations of the primitives large models need, from rms_layernorm and fused rotary embeddings to flash decoding and KV cache copy routines. The inference package is a serving system of its own, with core, executor, kv_cache, and server subpackages, and the auto_parallel module carries the compiler side: passes, checkpoint planning, offload planning, and profilers that reason about sharding choices before you commit them. The mixture-of-experts support lives in a compact _operation module plus its plugin variant, enough to see how experts are dispatched across ranks.

## From Install to a Launched Run

The Llama example's Gemini benchmark script shows the launcher in real form:

```sh
pip install colossalai
git clone https://github.com/hpcaitech/ColossalAI
cd ColossalAI/examples/language/llama

# one command starts 8 workers and applies the Gemini chunked-memory config
colossalai run --nproc_per_node 8 --hostfile hosts.txt benchmark.py -g -x -b 16
```

Inside the training script, the only structural change is the Booster pattern: construct a plugin, call booster.boost(model, optimizer), and train through booster.backward instead of loss.backward. Everything else, from chunk migration to gradient bucketing, happens inside the wrappers.

Honest limits: Colossal-AI's breadth is also its operating cost. The plugin matrix multiplies, and exotic combinations such as MoE plus stage parallelism plus FP8 carry compatibility caveats that only the test suite fully maps. The custom kernels and CUDA extensions mean installation is heavier than a pure-Python library, with build toolchains that matter on older driver stacks. The headline benchmark numbers come from the project's own hardware and scripts; reproducing 800-plus TFLOPS per GPU requires the exact H200 or B200 configurations listed. And the Gemini chunk system, while elegant, tunes through several interacting knobs, chunk size and placement policy chief among them, that reward experimentation. What the source demonstrates is a serious systems argument: parallelism strategies are composable objects, and a good facade makes the combinatorics navigable.
