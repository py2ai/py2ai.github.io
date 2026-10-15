---
layout: post
title: "The Engine That Sharded BLOOM - Inside microsoft/DeepSpeed"
description: A guided tour of the DeepSpeed training engine: ZeRO stages one through three, the DeepSpeedEngine forward-backward-step loop, pipe parallelism, the multi-node launcher, and the accelerator layer that spans six hardware vendors.
date: 2026-10-15
header-img: "img/post-bg.jpg"
permalink: /deepspeed/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/deepspeed/microsoft-deepspeed-overview-architecture.svg
tags: [deep learning, distributed training, ZeRO, open source]
categories: [AI, Open Source]
keywords: "DeepSpeed ZeRO ZeRO-Infinity microsoft distributed training BLOOM open source"
author: "PyShine"
---

When Microsoft's AI at Scale initiative committed to training language models measured in hundreds of billions of parameters, the binding constraint was not model design but memory arithmetic. A 176-billion-parameter model in mixed precision needs hundreds of gigabytes just to hold its weights, and several times more once optimizer states and gradients join the party. DeepSpeed is the open-source training system Microsoft built to break that wall, and its track record is written into the repository's own README: Megatron-Turing NLG at 530 billion parameters, BLOOM at 176 billion, Jurassic-1 at 178 billion, YaLM at 100 billion, GPT-NeoX at 20 billion, and Turing NLG at 17 billion all passed through this engine. The whole system ships under the Apache 2.0 license, which means every trick that made those runs fit is sitting in plain sight in the source tree.

What surprises first-time readers is how much of the system lives in plain Python. Beneath a thin layer of fused CUDA kernels, DeepSpeed is an orchestrator: a launcher that SSHes into your cluster, an engine class that re-implements the training loop with sharded bookkeeping, three ZeRO stages that progressively cut memory, a pipe-parallel subsystem, an autotuner, and a hardware abstraction that now speaks NVIDIA, AMD, Ascend, Gaudi, Intel XPU, and plain CPU. Reading it teaches you how modern distributed training actually works, one hook and one partition at a time. As always in this series, what follows is an educational tour of published source code.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/deepspeed/microsoft-deepspeed-overview-architecture.svg" alt="DeepSpeed overview architecture diagram" style="max-width:100%;"/></div>

<p><em>DeepSpeed overview: two entry points feed the DeepSpeedEngine, which drives the ZeRO stages, pipe and MoE parallelism, and a platform layer of collectives, accelerator backends, and JIT-compiled ops.</em></p>

Reading the overview from left to right:

- [the initialize() entry point](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/engine.py) — one call that wraps a plain PyTorch model and a JSON config into a managed distributed engine.
- [the launcher runner](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/launcher/runner.py) — parses a hostfile into a resource pool and fans out training processes across machines over SSH.
- [DeepSpeedEngine](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/engine.py) — the 5,153-line heart of the library where forward, backward, and step acquire their distributed meaning.
- [ZeRO stage 1 and 2](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py) — partitions optimizer states and gradients across data-parallel ranks.
- [ZeRO stage 3](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py) — shards the parameters themselves, with CPU and NVMe offload in the ZeRO-Infinity style.
- [partition_parameters.py](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/partition_parameters.py) — the zero.Init machinery that builds models no single rank can hold.
- [the pipe engine](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/engine.py) — layer-split parallelism driven by micro-batch schedules across a process topology.
- [the JIT op builder](https://github.com/microsoft/DeepSpeed/blob/master/op_builder/builder.py) — a registry of fused kernels compiled on first use through ninja.

## Why You Need This

The first reason is practical: the memory wall is the wall you hit. Train anything past a few billion parameters on commodity GPUs and PyTorch's default data-parallel replicate-everything model stops fitting long before your compute does. ZeRO's insight, implemented in three escalating stages, is that a training cluster wastes staggering amounts of memory by keeping identical optimizer states on every rank. Stage one partitions those states, stage two adds gradient partitioning, and stage three goes all the way and partitions the model parameters, gathering each layer on the fly only for the instants it is needed. When even stage three is not enough, the offload machinery spills states and parameters to CPU memory or NVMe storage, which is how a machine with modest GPUs can co-author a 176-billion-parameter run.

The second reason is the engineering itself. This codebase is a masterclass in the unglamorous systems work that papers skip: a prefetch coordinator that issues asynchronous parameter fetches before a layer needs them, a contiguous memory allocator that packs fragmented shards into bandwidth-friendly flat buffers, a tiling module that slices single enormous layers so they can pass through a sharded step, and a swap layer that moves optimizer state asynchronously so the GPU never stalls waiting for storage. None of these ideas is exotic in isolation, but seeing them composed inside one coherent engine is the difference between reading the ZeRO papers and understanding them.

The third reason is leverage. The abstractions pioneered here became the default vocabulary of the field: Hugging Face Transformers, Accelerate, PyTorch Lightning, MosaicML Composer, Determined, and MMEngine all ship DeepSpeed integrations, and the research has kept moving with DeepSpeed-MoE, ZeRO++, Ulysses sequence parallelism, ZenFlow, SuperOffload, and DeepCompile arriving year after year. If you work with large models in any capacity, the patterns in this tree — partitioned parameters, hook-driven offload, schedule-driven collectives — reappear everywhere from inference servers to fine-tuning frameworks, and recognizing them by name makes every other system easier to read.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/deepspeed/microsoft-deepspeed-architecture.svg" alt="DeepSpeed detailed architecture diagram" style="max-width:100%;"/></div>

<p><em>The detailed view: launcher and collectives at the top, the ZeRO internals block with its offload and prefetch machinery, the pipe-parallel subsystem, training support modules, and the platform layer underneath.</em></p>

Start where every user starts: deepspeed/runtime/engine.py. The file runs to 5,153 lines, and the class DeepSpeedEngine opens at line 516 after a long module-level initialize() function has already read your config JSON and made the load-bearing decisions. From there the engine is a conventional training loop wearing a distributed coat: forward begins at line 2987, backward at 3386, the optimizer step at 3571, and checkpointing at save_checkpoint (line 4873) and load_checkpoint (line 4399), with train and eval entry points near lines 2887 and 2893. Along the way the engine wires in a timer module for the wallclock breakdowns you see in logs, and a small monitor dispatcher that routes metrics to TensorBoard, WandB, or CSV depending on your config. The elegance is that the user's training script never changes shape — the engine intercepts the same three calls and adds bookkeeping between them.

The ZeRO block is where the memory actually gets saved. stage_1_and_2.py, at 3,464 lines, implements DeepSpeedZeroOptimizer: it flattens gradients, reduces and partitions them by rank, and hands the update to a fused optimizer — fused_optimizer.py, 456 lines, keeps FP16 master weights in a single flat buffer updated by one kernel call. Stage three is a different beast entirely. stage3.py, 3,435 lines, registers every parameter as a partition, and its companion partition_parameters.py, 2,093 lines, provides the zero.Init context so a model can be constructed from the start without any rank materializing the full weights. The runtime choreography lives in parameter_offload.py (552 lines), which pushes parameters to CPU or NVMe and swaps them back ahead of each forward and backward, and partitioned_param_coordinator.py (587 lines), which peeks at upcoming module executions to issue asynchronous prefetches so gathers overlap with compute. Supporting cast includes contiguous_memory_allocator.py for flat buffers, tiling.py (230 lines) for layers too big for one sharded pass, offload_states.py (87 lines) as a scoped context that lends the GPU to another workload, and a partitioned optimizer swapper in swap_tensor that moves states without blocking the step.

Parallelism beyond data-parallel lives in two places. The pipe subsystem mirrors a miniature framework: pipe/engine.py (1,153 lines) subclasses DeepSpeedEngine and buffers micro-batches, module.py (580 lines) lets you declare which layers live on which stage, schedule.py (486 lines) encodes the clock-cycle choreography of forwards, backwards, and buffer exchanges, topology.py (378 lines) models the process mesh where data, pipe, and model-parallel axes intersect, and p2p.py (136 lines) supplies the raw send and recv. Alongside it, sharded_moe.py (635 lines) implements DeepSpeed-MoE's top-K expert routing with capacity-aware dispatch, and module_inject (replace_module.py at 632 lines plus a 26-line policy base) can swap standard transformer blocks for optimized kernels at load time. Combine data, pipe, and tensor axes and you get the 3D parallelism that carried the 530-billion-parameter runs.

Two subsystems make all of this operable. The launcher splits into launch.py (250 lines), the command-line front door, and runner.py (523 lines), which reads a hostfile into a resource pool, checks each host's reachability over SSH, and honors a no-SSH mode for single-box runs — it is the piece that turns one command into a synchronized cluster. The autotuner (autotuner.py at 957 lines with a 352-line scheduler) runs training trials over a search space of ZeRO stages, offload targets, and micro-batch sizes, scores them with an exponential-median scheme, and writes the winning configuration back for production use. Underneath everything, the accelerator package defines a 326-line abstract device contract with concrete implementations for CUDA, CPU, Intel XPU, Ascend NPU, HPU for Gaudi, Apple MPS, and more, while comm.py (818 lines) wraps the collectives with timing hooks and a oneCCL backend so even CPU-only clusters can participate. The op builder registry, 839 lines, JIT-compiles the fused kernels on first use through ninja and torch's extension machinery, caching results so the compile cost is paid once.

The research frontier is unusually visible in the tree. superoffload_stage3.py (320 lines) implements the SuperOffload engine for superchip hardware that earned an ASPLOS 2026 honorable mention; a zenflow package carries the stall-free offloading engine from its 2025 write-up; inference v2 has its own engine module. Recent news items trace the pace: Muon optimizer support and SDMA offloaded collectives for AMD GPUs landed in May 2026, a core API update in December 2025 added PyTorch-style backward hooks and low-precision master states, and ZeRO++ powered large-scale distillation for recommendation models at LinkedIn. Arctic Long Sequence Training pushed context windows into the millions of tokens, DeepNVMe made NVMe bandwidth a first-class citizen, and DeepCompile and AutoTP brought compiler-style and automatic tensor-parallel optimizations into the mix. The repository doubles as a living index of systems-for-ML research.

## From Install to First Sharded Step

Installation is one pip command, and the ops compile on demand:

```
pip install deepspeed
ds_report
deepspeed --num_gpus=4 train.py --deepspeed_config ds_config.json
```

The ds_report command prints which compiled kernels are available and which will be built on first use, which is invaluable for debugging environments. Your training script changes by three lines: import deepspeed, wrap the model with deepspeed.initialize, and hand a JSON config that names the ZeRO stage and offload targets. A minimal stage-two config with CPU optimizer offload is a dozen lines of JSON. On Windows the project offers a build_win.bat script for pre-building, with a documented caveat that the asynchronous I/O and GDS offload backends are unsupported there. The launcher accepts a hostfile listing one hostname per line with slot counts, and from that moment your four-GPU workstation and your twelve-node cluster are driven by the same command.

Honest limits: DeepSpeed's power is proportional to its config surface, and the many-knobbed JSON can feel like tuning a ship's engine room for users who just want training to run; the JIT compilation path trades a slow first step for a fast install, which occasionally surfaces as environment-specific build errors; the pipe-parallel subsystem, while instructive, is the oldest architectural layer and shows its age next to newer schedule implementations elsewhere in the ecosystem; and the research features move quickly enough that a given release may carry experimental modules still settling down. None of that dims the core achievement: this is the codebase that made parameter sharding a solvable engineering problem rather than a research paper, and it repays every hour you spend inside it.

