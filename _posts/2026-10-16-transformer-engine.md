---
layout: post
title: "FP8 Without Rewriting Your Stack - Inside NVIDIA/TransformerEngine"
description: "A source-level tour of NVIDIA/TransformerEngine: how recipes, an autocast region manager, a family of quantized tensors, fused modules from Linear to TransformerLayer, and expert-parallel MoE machinery put FP8, MXFP8, and NVFP4 training behind a one-line context manager for PyTorch and JAX."
date: 2026-10-16
header-img: "img/post-bg.jpg"
permalink: /transformer-engine/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/transformer-engine/nvidia-transformer-engine-overview-architecture.svg
tags: [Transformer Engine, FP8, MXFP8, NVFP4, mixed precision, NVIDIA, Megatron-LM, mixture of experts]
categories: [AI, Open Source]
keywords: "TransformerEngine source code, NVIDIA FP8 training, fp8_autocast, DelayedScaling recipe, Float8CurrentScaling, MXFP8BlockScaling, NVFP4BlockScaling, Float8Tensor, DotProductAttention, TransformerLayer, expert parallel MoE"
author: "PyShine"
---

Training large models in FP8 promises roughly double the throughput and half the memory traffic of BF16, and the 2022 paper FP8 Formats for Deep Learning is the reason the industry believes it. The catch has always been numerics: eight-bit formats have tiny dynamic ranges, so every matmul needs scale factors, running maximums, and careful bookkeeping, and hand-rolling that across a transformer is a research project of its own. NVIDIA/TransformerEngine, the Apache-2.0 library behind FP8 training in Megatron-LM, NeMo, Hugging Face Accelerate, Lightning, and Amazon SageMaker, is the industrial answer: version 2.19, released in September 2026, carries FP8 recipes for Hopper through Rubin silicon plus MXFP8 and NVFP4 block scaling for Blackwell, fused kernels for attention, GEMMs, norms, and MoE, and a two-line API that hides all of it. As always in this series, what follows is an educational tour of published source code.

The repository splits into four packages: common holds the recipe definitions shared by every framework, pytorch is the 137-file flagship, jax mirrors the API across fifty files, and debug is a twenty-one-file numerics debugger. The design principle is separation of policy from mechanism. Recipes in common declare what precision policy you want; an autocast context manager in pytorch/quantization.py applies it region by region; a family of quantized tensor types carries the actual scaled data through the graph; and fused modules, from a plain Linear to a full TransformerLayer, call kernels that consume that data natively. Reading the overview from left to right traces one forward pass from recipe selection to quantized math to distributed infrastructure.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/transformer-engine/nvidia-transformer-engine-overview-architecture.svg" alt="TransformerEngine repository architecture overview" style="max-width:100%;"/></div>

<p><em>Overview of TransformerEngine: a shared recipe family feeds fp8_autocast, which quantizes work through fused modules into a QuantizedTensor contract implemented by Float8, MXFP8, and NVFP4 types, all resting on distributed and MoE infrastructure with a mirrored JAX API.</em></p>

Reading the overview from left to right:

- [The Recipe family](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py) is the policy base at line 110, with DelayedScaling at 176, Float8CurrentScaling at 289, MXFP8BlockScaling at 345, and NVFP4BlockScaling at 502.
- [fp8_autocast](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py) at line 980 of a 2044-line module is the context manager that switches quantizers on when you enter a region.
- [TransformerEngineBaseModule](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py) at line 948 is the contract every fused module extends.
- [Linear](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py) at line 1904 of 1904-plus lines wraps its custom autograd function at 1807 for FP8 GEMMs.
- [LayerNormMLP](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/layernorm_mlp.py) at line 2156 fuses normalization with the gated MLP block so intermediate activations stay in low precision.
- [DotProductAttention](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py) at line 594 dispatches fused attention across supported backends.
- [QuantizedTensor](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantized_tensor.py) defines the storage at line 54 and the Quantizer ABC at 371 that every format implements.
- [distributed.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/distributed.py) is the 2271-line infrastructure file with activation recomputation at 241 and an RNG state tracker at 832.

## Why You Need This

The first reason is that low precision is a moving target that no application wants to chase alone. In three years the field went from delayed-scaling FP8 on Hopper to per-tensor current scaling, to MXFP8 microscaling blocks, to NVFP4 four-bit training on Blackwell, with hybrid quantization and FP8 attention support arriving release by release. Each format has different scale layouts, kernel requirements, and numerics traps. TransformerEngine absorbs that churn behind one recipe interface, so moving your training from FP8 to MXFP8 is changing a constructor argument, not rewriting quantization logic across your model.

The second reason is integration depth with the frameworks you already use. The autocast API is deliberately shaped like torch.autocast, so adopting it is a wrapper around the forward pass rather than a restructure. Megatron-LM uses it for tensor, sequence, context, and expert parallelism with communication overlap; Accelerate and Lightning expose it as precision plugins; SageMaker's model parallel library builds on the same primitives. Even better, the pytorch and jax packages share the recipe definitions in common, so a numerics decision made once applies to both ecosystems, and the JAX side ships its own dropless MoE path tuned for GB200-class systems. The attention package even carries inference plumbing, with a KVCacheManager at line 21 of attention/inference.py and a paged variant at 596, so the same library serves decode-time serving and training-time throughput.

The third reason is that the library treats fused kernels and quantization as one problem, not two. A naive FP8 implementation quantizes weights and activations, runs a fast GEMM, then pays the savings back in elementwise glue. The modules here fuse the glue into the kernels: LayerNormLinear folds the norm into the GEMM epilogue, LayerNormMLP keeps the entire norm-plus-expansion block in low precision, GroupedLinear batches expert weights into single calls, and the attention stack dispatches to fused backends including FP8 attention. CUDA graph support in a 1735-line graph.py then captures the steady state so launch overhead disappears at scale. Throughput comes from the whole chain, and the whole chain is here.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/transformer-engine/nvidia-transformer-engine-architecture.svg" alt="TransformerEngine detailed architecture" style="max-width:100%;"/></div>

<p><em>The detailed map: the recipe hierarchy and autocast region manager, the fused module family, the quantized tensor zoo, MoE routing and permutation, runtime infrastructure from CUDA graphs to fused optimizers, and the JAX mirror with debugging tools.</em></p>

The recipe layer is the policy brain, and it is refreshingly small. [Recipe](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/recipe/__init__.py) at line 110 is a dataclass-style base, and the four public strategies are subclasses in the same file: DelayedScaling at 176 tracks amax history over a window and is the classic Hopper-era recipe; Float8CurrentScaling at 289 computes scales from the current step's maximums; MXFP8BlockScaling at 345 and NVFP4BlockScaling at 502 encode Blackwell's hardware microscaling formats. Policy objects carry no tensors; they are declarative. The mechanism lives in [quantization.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py), which opens by probing hardware support with functions like check_fp8_support at 191, then provides fp8_model_init at 886 so parameters are born ready for quantization, and fp8_autocast at 980, which on entry pushes the recipe's quantizers into a global manager and on exit restores the previous state. The QuantizerRole and QuantizerRequest types at lines 59 and 117 make the handoff explicit.

The quantized tensor zoo is where formats become data. [QuantizedTensorStorage](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantized_tensor.py) at line 54 holds the raw bytes and scale planes, and the [Quantizer](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantized_tensor.py) ABC at 371 defines the quantize and dequantize contract. Each format then gets its own file: [Float8Tensor](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/tensor/float8_tensor.py) at line 440 with its current-scaling quantizer at 197, Float8BlockwiseQTensor at 258 of the blockwise file, plus MXFP8 and NVFP4 siblings in their own modules, and a HybridTensor and GroupedTensor for mixed and batched cases. Two functions in quantized_tensor.py, prepare_for_saving at 291 and restore_from_saved at 318, manage the awkward autograd moment when quantized activations must be stashed for backward without defeating the memory savings that motivated them.

The fused modules are the biggest files in the package and the least negotiable to hand-write. [TransformerEngineBaseModule](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/base.py) at 948 centralizes weight initialization, quantization mode selection, and reset mechanics. [Linear](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/linear.py) reaches line 1904 because its autograd function at 1807 must handle every combination of quantized and unquantized inputs, weight caching, and gradient accumulation in FP8. LayerNormLinear at 1546 and LayerNormMLP at 2156 push fusion further, and [GroupedLinear](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/module/grouped_linear.py) at 1605 exists for MoE, where dozens of expert GEMMs must batch into few kernel launches. On top sit [DotProductAttention](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py) at 594, a torch-compatible MultiheadAttention at 48 of multi_head_attention.py, and the [TransformerLayer](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py) at line 71 that assembles the canonical block with DropPath at 46.

The MoE and infrastructure layers round out the system. Token routing lives in a 561-line [router.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/router.py), expert placement and collectives in a 1245-line [ep.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/ep.py), and the token sort-and-unsort dance in permutation.py's 1347 lines, all exposed through an operation graph whose dispatch and combine operations sit in the 38-file ops package. Around everything, [distributed.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/distributed.py) provides activation recomputation at 241 and deterministic RNG tracking at 832, cpu_offload.py moves state to host memory across 944 lines with a 751-line v1 variant kept for compatibility, and the fused optimizers package ships fused Adam, SGD, and multi-tensor apply. The JAX mirror keeps the same recipe objects and adds its own fp8_autocast at line 814 of jax/quantize/helper.py. A 21-file debug package, wired through debug/__init__.py, rounds it out with buffers and hooks for inspecting exactly what the quantizers did to your tensors, which is the difference between guessing at a loss spike and reading it.

## From Install to FP8 Training

The package builds from source against your CUDA toolchain, so the README recommends no-build-isolation and prebuilt NGC containers:

```sh
pip install --no-build-isolation "transformer_engine[pytorch]"
# or, in a ready environment:
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:YY.MM-py3
```

The canonical usage is a context manager around the forward pass, straight from the project's getting-started flow:

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling

linear = te.Linear(768, 3072, bias=True)

recipe = DelayedScaling()
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    out = linear(torch.randn(16, 128, 768, device="cuda"))
```

Inside the context, the Linear's weights and the input quantize through the active recipe, the GEMM runs on FP8 tensor cores, and scales update automatically. Swap DelayedScaling for Float8CurrentScaling, MXFP8BlockScaling, or NVFP4BlockScaling and the same two lines address three generations of hardware formats.

Honest limits: the hardware floor is real. FP8 needs Hopper or newer, MXFP8 and NVFP4 need Blackwell, and on Ampere you get FP16 and BF16 optimizations only, so the library is exactly as universal as NVIDIA's product line. Building from source ties you to specific CUDA, cuDNN, and PyTorch version combinations, which is why the NGC container is the advertised path rather than a convenience. The JAX API, while sharing recipes, is a separate surface with its own maturity curve. Debugging low-precision numerics stays genuinely hard; the debug package helps, but scale-factor bugs still look like mysterious loss spikes. And the abstraction rewards standard architectures: highly custom blocks may fall back to unquantized paths silently. None of this dims the achievement, which is making an eight-bit world navigable with a context manager, and doing it for both of the frameworks that matter.
