---
layout: post
title: "Megatron-LM: A Source Tour of NVIDIA's Large-Scale Training Framework"
description: "Walk through the Megatron core source: tensor and sequence parallel primitives, the transformer stack, sharded distributed optimizer, and the training entry loop."
date: 2026-10-24
header-img: "img/post-bg.jpg"
permalink: /megatron-lm/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/megatron-lm/nvidia-megatron-lm-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: Megatron-LM, tensor parallelism, sequence parallelism, distributed optimizer, TransformerEngine, NVIDIA
author: "PyShine"
---

Megatron-LM is NVIDIA's reference implementation for training large language models on clusters of GPUs, and it is the codebase behind many of the record-setting scaling results you see in model announcements. The repository actually ships two things: **Megatron-LM**, a set of pre-configured training scripts, and **Megatron Core**, a composable library of GPU-optimized building blocks that other frameworks (including NeMo) build on. If you have ever wondered how a model with hundreds of billions of parameters can be split across thousands of devices without the code turning into an unreadable mess, this is the place to look.

The remarkable thing about the source is how much of the magic reduces to one idea: *everything is a process group*. Tensor parallelism, data parallelism, context parallelism, and even the sharded optimizer are all just different ways of slicing ranks into groups and agreeing on which collective to run over which group. Once you internalize that, the code stops being intimidating. In this tour we read the actual files — the group bookkeeping in `parallel_state.py`, the communication primitives in `mappings.py`, the sharded linear layers, the transformer stack, the distributed optimizer, and the training loop that ties it all together.

We focus deliberately on tensor and sequence parallelism, the transformer implementation, and the optimizer. Megatron also contains sophisticated stage-wise 1F1B scheduling machinery, but that subsystem lives in files outside this tour's path, so we mention it only where the code forces us to.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/megatron-lm/nvidia-megatron-lm-overview-architecture.svg" alt="Overview architecture of NVIDIA Megatron-LM: process groups, tensor-parallel layers, transformer stack, distributed optimizer and training loop" style="min-width:900px;"></div>

<p><em>The overview map: CLI arguments feed the training loop; process groups back every collective; column- and row-parallel layers compose into the transformer stack that the loop trains, with sharded optimizer state stepping the gradients.</em></p>

Reading the overview from left to right:

- [`parse_args`, L130](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py#L130) — the single argparse entry point that collects every flag from grouped config factories.
- [`validate_args`, L421](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py#L421) — derives data-parallel size from world size and asserts the world divides evenly.
- [`initialize_model_parallel`, L630](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L630) — carves the tensor, context and data process groups every later collective relies on.
- [`copy_to_tensor_model_parallel_region`, L496](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py#L496) — the identity forward / all-reduce backward glue for column-parallel inputs.
- [`ColumnParallelLinear`, L1107](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L1107) — the GEMM whose weight matrix is split along the output dimension.
- [`MLP`, L157](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/mlp.py#L157) — the fc1/fc2 pair that shows the two parallel layer types working together.
- [`SelfAttention`, L1813](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py#L1813) — routes QKV and output projections through the same parallel machinery.
- [`train`, L4552](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L4552) — the top-level loop that owns iterations, logging, checkpointing and evaluation.

## Process Groups: The Coordinate System

Start with `megatron/core/parallel_state.py`, because every other file imports it. The module is essentially a registry of process groups: tensor-parallel, data-parallel, context-parallel, and combinations of those, each stored as a global that gets initialized exactly once per process. The workhorse is [`initialize_model_parallel`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L630), which takes the requested sizes and computes which ranks sit in which group. Rank layout matters more than it sounds: Megatron packs tensor, context and data parallel ranks into each node so that the fastest interconnect (NVLink) carries the chattiest collectives.

The accessor functions read like a map of the design space — [`get_tensor_model_parallel_world_size`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L2060), context- and data-parallel equivalents, and group getters for every collective type. Teardown is symmetric via [`destroy_model_parallel`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py#L2616), which recent versions carefully null out to keep repeated initialization in tests honest. The recent codebase has also consolidated much of this bookkeeping into a `ProcessGroupCollection` object that gets passed explicitly instead of relying purely on globals — a quiet but meaningful refactor for anyone embedding Megatron Core in another framework.

## The Tensor-Parallel Primitives

The actual math of tensor parallelism lives in `megatron/core/tensor_parallel/`. Two files do the heavy lifting. `mappings.py` defines the autograd-aware collectives: [`copy_to_tensor_model_parallel_region`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py#L496) is a forward identity whose backward all-reduces gradients across the tensor group, and its mirror [`gather_from_sequence_parallel_region`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py#L528) all-gathers inputs along the sequence dimension and scatter-reduces gradients back. That second primitive is what makes *sequence parallelism* work: with the flag on, layer norms and dropout operate on sequence-sharded activations, and the gathers happen exactly at the boundaries where the parallel GEMMs need full rows.

The GEMMs themselves live in `layers.py`. [`ColumnParallelLinear`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L1107) splits the weight along its output dimension, so each rank computes a slice of the output features from the full input; [`RowParallelLinear`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L1528) splits along the input dimension and all-reduces partial sums at the end. Chaining a column layer into a row layer — the classic Megatron trick — means one all-reduce per MLP or attention block instead of one per matrix multiply. Both layers funnel through [`LinearWithGradAccumulationAndAsyncCommunication`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L744), a custom autograd Function that overlaps weight-gradient accumulation with the collectives and handles sequence-parallel input gathering transparently.

## The Transformer Stack

With the primitives in place, the transformer files read almost like pseudocode. `megatron/core/transformer/mlp.py` defines [`MLP`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/mlp.py#L157): a `ColumnParallelLinear` fc1, a fused GELU (see the `bias_gelu_impl` call at [L311](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/mlp.py#L311)), and a `RowParallelLinear` fc2. Gated variants (SwiGLU and friends) swap the activation for a tripled intermediate size and a multiplication. Note the naming drift from older releases: this class is now `MLP`, not the historical `ParallelMLP`.

`attention.py` is one of the largest files in the tree. The abstract [`Attention`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py#L288) base wires up projections and holds the shared plumbing, while [`SelfAttention`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/attention.py#L1813) builds the packed QKV column-parallel projection and the row-parallel output projection. The core math delegates to `dot_product_attention.py`, whose [`DotProductAttention`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/dot_product_attention.py#L26) dispatches to fused backends — the local PyTorch path, or the TransformerEngine fused and FlashAttention kernels when available. Variants like multi-latent attention subclass the same scaffolding, which is why the file grew protocol-style builder interfaces: the class you get is resolved from a submodule spec, not hardcoded.

Stacking happens in two files. [`TransformerLayer`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_layer.py#L334) sequences the pre-norm attention branch and MLP branch with residual connections and dropout, and [`TransformerBlock`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py#L268) materializes the list of layers from a `ModuleSpec` and runs them. The spec indirection is the extension point: model files under `megatron/core/models/` declare which implementations (TransformerEngine or native) each sublayer should use, which is how [`GPTModel`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py#L55) assembles itself from the same block.

## From Install to Your First Model

Installation follows the README. The library is on PyPI:

```bash
uv pip install megatron-core
```

or, if you want the training scripts and to read the source alongside:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

The README warns that building from source is memory-hungry and suggests capping compile parallelism with `MAX_JOBS=4` if the build runs out. Most production users instead start from the NVIDIA GPU Cloud PyTorch container, where the CUDA toolchain and TransformerEngine are preinstalled; the official installation guide walks through that route.

The entry script is `pretrain_gpt.py` at the repo root. A minimal two-GPU tensor-parallel run looks like:

```bash
torchrun --nproc-per-node 2 pretrain_gpt.py \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --micro-batch-size 2 \
  --global-batch-size 32 \
  --num-layers 12 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --seq-length 2048 \
  --use-distributed-optimizer
```

Every flag here is real: the parallel sizes come from the `ModelParallelConfig` dataclass that argparse factories expose as CLI options, the model geometry comes from `TransformerConfig` fields, and `--use-distributed-optimizer` is defined explicitly in `megatron/training/arguments.py`. In practice you would add dataset, tokenizer, and checkpoint flags — the README's quickstart link covers data preparation — but this much already exercises the whole parallel stack from the diagrams above.

## Sharded Optimizer State and the Training Loop

The optimizer subsystem is where Megatron's memory engineering pays off. `megatron/core/optimizer/optimizer_config.py` defines [`OptimizerConfig`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py#L140), a dataclass holding learning rate, clipping, weight decay, and precision options. The factory [`get_megatron_optimizer`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/__init__.py#L1037) inspects that config plus the model's parameter groups and decides which wrapper to build. With the distributed flag on, you get [`DistributedOptimizer`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py#L153), which partitions the exp_avg and exp_avg_sq moment buffers across the data-parallel group instead of replicating them on every rank — roughly an 8x cut in optimizer memory per GPU at large DP counts compared with naive replication, and the reason the flag is on in essentially every real config.

The surrounding pieces matter too. `megatron/core/distributed/distributed_data_parallel.py` implements gradient bucketing with overlap of the reduce and accumulation, and `finalize_model_grads` ([L651](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/finalize_model_grads.py#L651) in `megatron/core/distributed/finalize_model_grads.py`) is the hook that all-reduces gradients across data-parallel ranks and syncs embedding gradients between the ranks that own the input and output word embeddings when those live on different model stages.

On top of all this, `megatron/training/training.py` is the orchestration layer: [`setup_model_and_optimizer`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L2810) builds model, DDP wrapper and optimizer in one call; [`train`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L4552) owns the iteration loop, timers, logging and fault-tolerant checkpointing; [`train_step`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L3242) runs one forward-backward cycle through a user-supplied `forward_step_func`; and [`evaluate_and_print_results`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L5776) handles validation passes. The loop is deliberately framework-neutral about what happens inside a step — that is how the same engine drives GPT, multimodal, and hybrid architectures.

## Honest Limits

A few caveats worth stating plainly. First, this is a multi-node-first framework: the abstractions assume an NCCL-capable cluster, and running on a single GPU or a laptop is possible but exercises none of the interesting machinery — if that is your situation, a smaller trainer is a better teacher. Second, the setup experience is Docker/NGC-centric; building from source works, but the pinned CUDA, NCCL and TransformerEngine versions make bare-metal installs fiddly. Third, Megatron Core is tightly coupled to TransformerEngine for its fastest paths — the fused kernels and FP8 support assume TE containers, so plans to swap in other kernel libraries need spec-level work.

Fourth, the stage-wise 1F1B scheduling machinery (interleaved schedules, microbatch splitting) is a major part of why Megatron scales, and this tour deliberately stayed out of those files; treat the diagrams here as covering the tensor-, sequence- and data-parallel story only. Finally, the codebase moves fast — classes get renamed (the old `ParallelMLP` is now `MLP`), argument groups get refactored into config dataclasses, and line numbers drift, so anchor links in older write-ups decay faster than in most projects.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/megatron-lm/nvidia-megatron-lm-architecture.svg" alt="Detailed architecture of NVIDIA Megatron-LM: parallel state, tensor-parallel collectives, transformer components, optimizer wrappers, distributed grad handling and GPT model assembly" style="min-width:900px;"></div>

<p><em>The detail map adds the surrounding machinery: RNG tracking for consistent dropout, vocab-parallel cross entropy, the optimizer wrapper hierarchy, DDP bucketing and gradient finalization, and the GPT model assembly driven by layer specs.</em></p>

If you want to go deeper, three adjacent reads pair well with this one: our [DeepSpeed](/deepspeed/) tour covers the data-parallel and sharded-optimizer alternatives from Microsoft's side, the [FlashAttention](/flash-attention/) breakdown explains the kernels Megatron's attention backend calls into, and the [transformers](/transformers/) piece covers the Hugging Face ecosystem that Megatron Bridge now converts checkpoints to and from. For a minimal contrast on how far you can get without any of this machinery, see [nanoGPT](/nanogpt/). Megatron-LM is what those ideas look like when they grow up to fill a datacenter.

<script type="application/ld+json">
{"@context":"https://schema.org","@type":"TechArticle","headline":"Megatron-LM: A Source Tour of NVIDIA's Large-Scale Training Framework","description":"Walk through the Megatron core source: tensor and sequence parallel primitives, the transformer stack, sharded distributed optimizer, and the training entry loop.","datePublished":"2026-10-24","dateModified":"2026-10-24","author":{"@type":"Person","name":"PyShine"},"publisher":{"@type":"Organization","name":"PyShine"},"keywords":"Megatron-LM, tensor parallelism, sequence parallelism, distributed optimizer, TransformerEngine, NVIDIA","url":"https://pyshine.com/megatron-lm/","image":"https://pyshine.com/assets/img/diagrams/megatron-lm/nvidia-megatron-lm-overview-architecture.svg"}
</script>
