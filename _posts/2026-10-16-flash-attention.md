---
layout: post
title: "The Kernel That Rewrote Attention - Inside Dao-AILab/flash-attention"
description: "A source-level tour of Dao-AILab/flash-attention: how the flash_attn Python API binds to the mha_fwd C++ entry, the generated SM80 kernel grid, the TMA warp-specialized Hopper FA3 stack, the CuTeDSL FA4 rewrite, and the fused GPT reference model."
date: 2026-10-16
header-img: "img/post-bg.jpg"
permalink: /flash-attention/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/flash-attention/dao-ailab-flash-attention-overview-architecture.svg
tags: [FlashAttention, GPU kernels, CUDA, attention, Triton, ROCm, Hopper, Blackwell]
categories: [AI, Open Source]
keywords: "flash-attention source code, flash_attn_func, mha_fwd flash_api.cpp, online softmax kernel_traits.h, flash_attn_with_kvcache paged KV, flash_attn_varlen_func, hopper mainloop TMA GMMA, CuTeDSL flash_attn.cute, composable kernel ROCm, Tri Dao"
author: "PyShine"
---

Every transformer trained since 2022 owes part of its existence to a paper proof that attention could be computed without ever writing the full sequence-by-sequence matrix to HBM. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness by Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Re appeared at NeurIPS 2022, and its successor FlashAttention-2, a single-author work Tri Dao presented at ICLR 2024, became the de facto attention kernel for the industry. The BSD 3-Clause licensed repository that carries both is one of those rare codebases where a systems insight, memory traffic dominates math, is visible in every directory: a Python front end that never allocates what it can fuse, a C++ binding layer that dispatches on head dimension, and three kernel generations for three GPU eras. As always in this series, what follows is an educational tour of published source code.

The repository is organized as an archaeology of GPU hardware. The main package implements FlashAttention-2 for Ampere, Ada, and Hopper GPUs with a generated grid of CUDA files. The hopper folder is a beta FlashAttention-3 that rewrites the mainloop around TMA copies, warp specialization, and Hopper's GMMA tensor core instructions. The flash_attn.cute package is FlashAttention-4, a CuTeDSL implementation for Hopper and Blackwell that expresses the whole algorithm in Python-embedded tile primitives. Around them sit a fused GPT reference model, optimized rotary and norm layers, a training folder for GPT2 and GPT3, and a ROCm port with two AMD backends. Reading the overview from left to right follows one attention call from the Python signature down to a specific compiled kernel file.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/flash-attention/dao-ailab-flash-attention-overview-architecture.svg" alt="FlashAttention repository architecture overview" style="max-width:100%;"/></div>

<p><em>Overview of FlashAttention: the flash_attn Python API binds to the C++ mha_fwd entry, which launches the generated SM80 kernel grid; MHA modules and the GPT reference model sit on top, while FA3, FA4, the ROCm CK backend, the Triton variant, and the test suite surround them.</em></p>

Reading the overview from left to right:

- [flash_attn_interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py) is the public surface, 1629 lines with flash_attn_func at 1156, flash_attn_qkvpacked_func at 1019, flash_attn_varlen_func at 1391, and flash_attn_with_kvcache at 1485.
- [flash_api.cpp](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp) is the C++ binding at 1538 lines, opening with mha_fwd at line 370.
- [The SM80 kernel grid](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src) holds 116 files enumerating head dims, dtypes, causal flags, and split-KV variants.
- [mha.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py) is the module layer at 993 lines, with the full projection-carrying MHA at line 373.
- [models/gpt.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/models/gpt.py) builds a complete GPT at 1080 lines, GPTLMHeadModel at 577.
- [hopper/flash_attn_interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py) is FlashAttention-3 at 1146 lines, flash_attn_func at 809.
- [cute/interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py) is FlashAttention-4, 4970 lines with flash_attn_func at 4145.
- [tests/test_flash_attn.py](https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py) enforces the project's correctness bar: max numerical error at most twice a PyTorch reference.

## Why You Need This

Standard attention materializes an N-by-N score matrix in GPU memory, so a 32-thousand-token sequence in bfloat16 would need gigabytes before the softmax even starts. FlashAttention computes exact attention in linear memory by tiling: load K and V blocks, maintain running max and sum statistics, rescale as you go. The README's memory chart shows 10x savings at 2K tokens and 20x at 4K, which is what unlocked long-context training before anyone sold you a sparse approximation. Because the result is exact, every downstream user from Mistral's sliding-window models to Gemma-2's softcapping could adopt it without changing their math, and the changelog reads like a history of transformer research: variable-length batches in 2.0, bottom-right causal alignment in 2.1, split-KV decode in 2.2 contributed with the xformers team's Daniel Haziza, sliding windows in 2.3 with Mistral AI, ALiBi in 2.4 thanks to Sanghun Cho at Kakao Brain, deterministic backward from Meituan engineers, paged KV cache in 2.5 matching the PagedAttention paper, softcapping in 2.6 from HuggingFace's Narsil and lucidrains, and torch.compile compatibility in 2.7.

The decode side is where serving lives or dies. flash_attn_with_kvcache accepts a pre-allocated key and value cache with cache_seqlens, updates it in place with new tokens, applies rotary embedding inside the same kernel, and supports paged caches through a block_table where each page must be a multiple of 256 elements, the exact shape of work vLLM-style servers produce. When the query length is one, the kernel splits cache loading across thread blocks and combines partial results, a collaboration with the xformers team that turned single-token generation into a bandwidth-bound rather than latency-bound problem. Grouped-query attention is handled by letting K and V carry fewer heads than Q as long as the counts divide evenly, so Llama-class models need no special mode.

The training story is just as concrete. The repository ships a full GPT implementation plus fused MLP, LayerNorm, cross-entropy, and rotary layers, and its training folder measures 3-5x speedup over the HuggingFace baseline, reaching up to 225 TFLOPs per second per A100, equivalent to 72 percent model FLOPs utilization, without activation checkpointing. FA3 pushes Hopper hardware further with FP8 forward support, and FA4 extends to Blackwell. Adoption is catalogued in the repo's usage.md, and if compiling is too heavy, the HuggingFace kernels library serves prebuilt wheels: get_kernel("kernels-community/flash-attn2") returns a ready flash_attn_func.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/flash-attention/dao-ailab-flash-attention-architecture.svg" alt="FlashAttention repository architecture detail" style="max-width:100%;"/></div>

<p><em>Detail view: the Python API wraps autograd functions and flattening utilities, the C++ entry dispatches into trait-templated SM80 kernels generated by script, FA3 pairs a warp-specialized mainloop with 451 instantiations, FA4 splits forward, backward, and MLA kernels in CuTeDSL, and modules, model, ops, and training consume the API.</em></p>

The Python layer is thin by design. The autograd classes FlashAttnQKVPackedFunc at line 461 and FlashAttnFunc at 828 store the output's log-sum-exp statistics and the dropout RNG state, then recompute nothing during backward except through the C++ kernel. bert_padding.py, 218 lines, provides unpad_input and pad_input, the flattening step that turns a padded batch into packed total-token arrays so the varlen entry points never waste threads on padding. The packed variants exist because stacking Q, K, V into one tensor lets backward skip three separate gradient concatenations, a small API decision with measurable throughput value. flash_attn_with_kvcache at 1485 rounds out the set, refusing a backward pass since decode never needs one.

The C++ middle layer does shape policing and kernel selection. flash_api.cpp validates strides and head dims, then calls mha_fwd at 370, mha_varlen_fwd at 540, mha_bwd at 802, mha_varlen_bwd at 1012, and mha_fwd_kvcache at 1245. Below it, the src folder's 116 files are enumerations: for each head dimension in 32, 64, 96, 128, 192, 256, for fp16 and bf16, causal or not, plain or split, a generated .cu file instantiates the templates in kernel_traits.h, which fix tile sizes and shared-memory staging per combination. The actual math lives in flash_fwd_kernel.h, an online-softmax loop over K and V tiles, flash_bwd_kernel.h, which recompute-then-accumulate gradients using the saved LSE, and softmax.h, the running-state struct both share. generate_kernels.py writes the instantiation files so the compiler never has to chew a monolithic template, a build-time trick mirrored in the Hopper folder.

The third and fourth generations justify their own directories. FA3's mainloop_fwd_sm90_tma_gmma_ws.hpp name says it all: TMA asynchronous copies feed a warp-specialized stage machine where producer warps load tiles and consumer warps run GMMA tensor-core instructions, with epilogue_fwd.hpp handling final rescaling and tile_scheduler.hpp assigning work, including a persistent scheduler for variable-length batches; 451 files in hopper/instantiations cover the dtype, head-dim, and packing matrix. FA4 replaces C++ with CuTeDSL, the Python-embedded tile language from CUTLASS: interface.py at 4970 lines dispatches to flash_fwd.py and flash_bwd.py, Blackwell paths in flash_fwd_sm100.py use tcgen05 MMA with two-CTA backward variants for head dim 256, and flash_fwd_mla_sm100.py implements multi-head latent attention decode for DeepSeek-style compressed caches, with softmax.py reimplementing the running-max trick in tile form.

The model layer shows how the kernel composes into a real training stack. mha.py's MHA at 373 carries QKV and output projections, ParallelMHA at 707 adds tensor-parallel versions, and block.py, 397 lines, wires attention, MLP, and norms into a residual block. models/gpt.py assembles everything: create_mixer_cls at 62, create_block at 262, GPTModel at 409, GPTLMHeadModel at 577, with shard_state_dict_tp at 698, a HuggingFace GPT2 weight remap at 930, and a Megatron remap at 987. The ops package supplies fused_dense.py at 688 lines, layer_norm.py at 800, rms_norm.py at 174, and layers/rotary.py at 482, each backed by its own small CUDA extension under csrc. The AMD port lives in csrc/flash_attn_ck, the Composable Kernel backend supporting MI200 through MI355X and RDNA 3/4, while the Triton backend rides in through the aiter submodule controlled by the FLASH_ATTENTION_TRITON_AMD_ENABLE flag, with an autotuning switch and a config override environment variable for tuning individual launches.

## From Install to Causal Attention

Install the stable FlashAttention-2 wheel; ninja is required, since without it compilation takes around two hours on a single core versus 3-5 minutes on a 64-core machine, and MAX_JOBS caps parallelism on smaller RAM machines:

```sh
pip install flash-attn --no-build-isolation
# small-RAM machines
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

Run exact, causal attention with sliding window and grouped-query heads, then decode against a paged KV cache:

```python
import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache

q = torch.randn(1, 2048, 32, 128, dtype=torch.float16, device="cuda")
k, v = torch.randn(2, 1, 2048, 8, 128, dtype=torch.float16, device="cuda")
out = flash_attn_func(q, k, v, causal=True, window_size=(1023, 0))

# incremental decoding: cache updated in place, rotary applied inside
cache_seqlens = torch.tensor([1024], dtype=torch.int32, device="cuda")
out = flash_attn_with_kvcache(q[:, -1:], k_cache, v_cache,
    cache_seqlens=cache_seqlens, causal=True)
```

For Hopper FP8-era performance, install the beta package from the hopper subdirectory and import flash_attn_3.flash_attn_interface; for Blackwell and beyond, pip install flash-attn-4 and import flash_attn.cute.

Honest limits: compilation is a genuine cost, Linux is the only well-tested platform, and the SM80 kernels require Ampere or newer, leaving Turing owners to a community fork. Head dims above 256 are unsupported, dropout determinism trades speed for reproducibility, and FA3 remains a separately installed beta while FA4's CuTeDSL stack is young enough that its API surface may still move. The varlen path expects pre-flattened inputs, so integrating with padded batches means adopting the unpad and pad helpers rather than dropping in a replacement for torch's attention call. But for what it promises, exact attention at bandwidth speed, the repository remains the reference point the entire ecosystem measures against.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "The Kernel That Rewrote Attention - Inside Dao-AILab/flash-attention",
  "description": "A source-level tour of Dao-AILab/flash-attention: how the flash_attn Python API binds to the mha_fwd C++ entry, the generated SM80 kernel grid, the TMA warp-specialized Hopper FA3 stack, the CuTeDSL FA4 rewrite, and the fused GPT reference model.",
  "author": { "@type": "Person", "name": "PyShine" },
  "datePublished": "2026-10-16",
  "keywords": "FlashAttention, CUDA, online softmax, FA3, FA4, KV cache",
  "image": "https://pyshine.com/assets/img/diagrams/flash-attention/dao-ailab-flash-attention-overview-architecture.svg"
}
</script>
