---
layout: post
title: "The Compressor That Feeds vLLM - Inside vllm-project/llm-compressor"
description: "A source tour of vllm-project/llm-compressor, the quantization workbench that turns Hugging Face checkpoints into compressed-tensors models for vLLM, with GPTQ, AWQ, SmoothQuant, SpinQuant, and FP4 observers."
date: 2026-10-16
header-img: "img/post-bg.jpg"
permalink: /llm-compressor/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/llm-compressor/vllm-project-llm-compressor-overview-architecture.svg
tags: [llm-compressor, quantization, vLLM, GPTQ, AWQ, compressed-tensors]
categories: [Open Source, Deep Learning]
keywords: "llm-compressor source code, GPTQ quantization, AWQ, SmoothQuant, NVFP4 MXFP4, compressed-tensors, vLLM deployment"
author: "PyShine"
---

Every quantized model you load into vLLM was made by something, and for most of the official checkpoints that something is [llmcompressor](https://github.com/vllm-project/llm-compressor) - the vLLM project's own optimization library. It takes a Hugging Face checkpoint plus a recipe, runs a short calibration pass, and saves a compressed-tensors model that vLLM consumes natively: W8A8 FP8 and INT8, W4A16 with GPTQ or AWQ, NVFP4 and MXFP4 microscaling formats, KV-cache and attention quantization, even expert pruning with REAP. The Red Hat AI team uses it to publish production checkpoints for models like Nemotron, GLM-5.3, Qwen3.8, and Kimi-K3, and the library itself just landed a Triton GPTQ kernel that is roughly 15 times faster than the previous eager path with batched layer quantization on top.

What distinguishes the codebase is its architecture rather than any single algorithm. Everything is a Modifier - a small state machine that reacts to lifecycle events - orchestrated by a global CompressionSession and driven by one-call entrypoints. Add an algorithm by writing one class; compose algorithms by listing them in a recipe. In this tour we read the session core, the quantization mixin and observer system, the GPTQ and AWQ implementations, and the sequential calibration machinery that traces subgraphs through Hugging Face models. As always in this series, what follows is an educational tour of published source code.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/llm-compressor/vllm-project-llm-compressor-overview-architecture.svg" alt="llm-compressor overview architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Overview of the llm-compressor architecture: the oneshot entrypoint resolves a Recipe into Modifiers, the session core sweeps lifecycle events, and the quantization group supplies mixins, observers, and the GPTQ engine.</em>

Reading the overview from left to right:

- [entrypoints/oneshot.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/entrypoints/oneshot.py) is the front door at 501 lines: the Oneshot class at line 55 and the decorated oneshot function at line 330.
- [core/session.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/core/session.py) is the global CompressionSession at 213 lines: initialize at line 74, finalize at line 147, and event dispatch at line 165.
- [modifiers/modifier.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/modifier.py) is the base contract at 306 lines: Modifier at line 14 with on_initialize at line 201 through on_calibration_end at line 298.
- [modifiers/gptq/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/gptq/base.py) is the flagship algorithm at 433 lines: GPTQModifier at line 50.
- [modifiers/gptq/helpers.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/gptq/helpers.py) carries the math at 453 lines: accumulate_hessian at line 40 and factorize_hessian at line 283.
- [modifiers/transform/awq/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/transform/awq/base.py) is AWQ at 1056 lines: AWQModifier at line 55.
- [observers/grid_search.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/observers/grid_search.py) is the scale search at 702 lines: the eager observer at line 40 and Triton kernels from line 242.
- [modifiers/quantization/quantization/mixin.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/quantization/quantization/mixin.py) is the shared engine at 546 lines: QuantizationMixin at line 58.

## Why You Need This

If you serve models with vLLM, this library is the supported path from a full-precision Hugging Face checkpoint to something vLLM actually accelerates. The compressed-tensors format it emits stores the quantization scheme, group sizes, scales, and zero points inside the model config, so loading needs no extra flags. Where the [torchao quantization stack](https://pyshine.com/ao/) targets training-time low precision and [onnxruntime's int4 kernels](https://pyshine.com/onnxruntime/) target the ONNX world, llm-compressor targets exactly one goal: accurate weights-only and weight-activation quantized checkpoints that vLLM runs at full speed on H100 and Blackwell hardware.

Second, the recipe model makes quantization reproducible. A recipe is a list of modifier declarations - or a YAML file with stages - that travels with the model, so anyone can see exactly how a published checkpoint was produced and re-run it on a different model. Recipes compose: SmoothQuant followed by GPTQ, AWQ plus RTN, NVFP4 with an MSE observer, REAP expert pruning before quantization. The library validates the combination up front and writes the recipe back into the saved checkpoint as metadata.

Third, the engineering inside is worth stealing for any calibration job. Sequential calibration traces actual subgraphs through the Hugging Face forward code using AST analysis, stages intermediate activations through an offload-capable cache so 700B-parameter models compress on modest hardware, and supports distributed data-parallel hessian reduction. The observer framework includes a Triton grid-search kernel with buffered error tolerance that reaches bitwise parity with the eager path - a pattern of "eager reference plus fused kernel with parity tests" that generalizes far beyond quantization.

<div style="text-align:center; margin: 20px 0;">
  <img src="https://pyshine.com/assets/img/diagrams/llm-compressor/vllm-project-llm-compressor-architecture.svg" alt="llm-compressor detailed architecture diagram" style="max-width:100%; height:auto;">
</div>
<em>Detailed view: entry and data plumbing, the session core, the quantization machinery with observers and the grid-search kernel, the algorithm roster, sequential staging, and native model definitions.</em>

The entrypoint is the best place to start reading because it wires everything. [oneshot.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/entrypoints/oneshot.py) defines Oneshot at line 55 whose __call__ at line 203 fetches a calibration dataloader, runs apply_recipe_modifiers at line 227, and hands off to post_process for saving. Inside apply_recipe_modifiers the code linearizes any grouped-expert MoE modules it detects at lines 248 through 254, enters calibration contexts (norm wrappers, MoE all-experts mode, eager attention) in an ExitStack at lines 258 through 263, then calls session.initialize and lets a calibration flow - resolved from the modifiers at lines 277 through 279 - drive the model with data. The validate_model method at line 289 refuses checkpoints already quantized by other toolkits. [entrypoints/utils.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/entrypoints/utils.py) supplies pre_process at line 41 and post_process at line 99, which load models from paths and save them back with the recipe embedded, and [recipe/recipe.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/recipe/recipe.py) defines Recipe at line 27 with from_modifiers at line 45, stage-aware from_dict at line 168, and yaml serialization at line 242. The [ModifierFactory](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/factory.py) resolves recipe strings against the registry, while [args/dataset_arguments.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/args/dataset_arguments.py) and [datasets/utils.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/datasets/utils.py) handle the calibration data side - wikitext, Ultrachat, C4, custom JSON - with sequence lengths and split control.

The session core is a compact state machine worth emulating. [session.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/core/session.py) implements CompressionSession at line 43 as a module-level singleton with a _CallbackContainer at line 25; initialize at line 74 builds a CompressionLifecycle and State, event at line 165 fans an Event to the lifecycle, and finalize at line 147 returns a ModifiedState per modifier. [lifecycle.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/core/lifecycle.py) runs the ordered sweep - initialize at line 73, event at line 151 with _validate_event_order guarding at line 215, finalize at line 117 - and [state.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/core/state.py) holds the model, hardware, and recipe in State at line 74. Every algorithm subclasses [Modifier](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/modifier.py) at line 14, whose hooks span the whole arc: on_initialize at line 201, on_event at line 225, on_start at line 237, on_calibration_start at line 273, on_sequential_epoch_end at line 283, and on_calibration_end at line 298. This is why algorithms compose without knowing about each other: they all react to the same event stream, and the lifecycle guarantees ordering.

Quantization proper lives in the mixin and the observers. [mixin.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/quantization/quantization/mixin.py) defines QuantizationMixin at line 58: initialize_quantization at line 224 attaches quantizer modules to every targeted linear, start_calibration at line 243 switches observers into collecting mode, end_calibration at line 263 freezes scales, and sync_obs_act_stats at line 303 reduces activation statistics across data-parallel ranks. resolve_quantization_config at line 367 merges the scheme, targets, and ignore lists into a compressed-tensors config. The hooks that feed observers are in [calibration.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/quantization/calibration.py): observe at line 104, input and output forward hooks at lines 184 and 192, and dedicated calibrate_query_hook, calibrate_key_hook, and calibrate_value_hook at lines 206 through 214 so KV-cache quantization can collect per-QKV statistics. Observers themselves register by name: [observers/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/observers/base.py) defines Observer at line 29 with get_qparams at line 99 and sync_activation_stats at line 147; the min-max and MSE variants are compact, while [imatrix.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/observers/imatrix.py) adds IMatrixMSEObserver at line 25 with the NVFP4ExpandedIMatrixObserver at line 253 whose grid search is a strict superset of the fouroversix scale choices. The heavy lifting is [grid_search.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/observers/grid_search.py): the eager reference at line 40 with _calculate_mse_error at line 136 and _calculate_imatrix_error at line 176, then the Triton implementation from _grid_search_observer_triton at line 242 with packed, split, and split-reduce kernels at lines 407, 568, and 668, tuned by the triton_error_buffer default of 100 percent for FP4 and 30 percent otherwise.

The algorithm roster shows each paper as one modifier. [gptq/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/gptq/base.py) implements GPTQModifier at line 50 with on_initialize at line 187, calibrate_module at line 247, compress_modules at line 280 with batched layers sharing one shape, a hessian rank-reduction helper at line 365, and an RTN fallback summary at line 391 when hessian input runs out. Its math lives in [helpers.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/gptq/helpers.py): make_empty_hessian at line 30, accumulate_hessian at line 40, apply_activation_ordering at line 237 for act-order calibration, factorize_hessian at line 283 for the Cholesky path, column_scale_window at line 343, and get_triton_gptq_config at line 426 selecting the kernel configuration. The block update itself is [gptq_quantize.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/gptq/gptq_quantize.py): quantize_weight at line 28, the Triton dispatch at _gptq_block_update_triton at line 212, and the fused variant at line 261. [awq/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/transform/awq/base.py) is the largest single algorithm at 1056 lines - AWQModifier at line 55 with its activation-aware scale search, plus a get_lowest_common_ancestor_with_avoid helper at line 1015 that resolves which preceding layer's activations feed each target. [smoothquant/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/transform/smoothquant/base.py) contributes SmoothQuantModifier at line 63 with _apply_smoothing at line 322 migrating activation outliers into weights, [spinquant/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/transform/spinquant/base.py) adds rotation-based quantization with norm mappings, [autoround/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/autoround/base.py) wraps the AutoRound sign-gradient optimizer in AutoRoundModifier at line 106, and [qad/base.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/qad/base.py) provides the quantization-aware-distillation QADModifier at line 33. REAP expert pruning is supported through [pruning/helpers.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/pruning/helpers.py), and the same GPTQ engine powers FP8: the Nemotron Lightning FP8 checkpoints cited in the README are GPTQ-based.

The sequential staging machinery is what makes big models tractable, and it is the most unusual code in the repo. The registry package infers from the modifier set whether calibration can be datafree, independent per module, or sequential - the abstract flow class at line 17 of registry.py resolves the concrete one from the modifiers with from_modifiers at line 28 and warns when a user override disagrees with the inference - and sequential mode then runs the model subgraph by subgraph. The sequential helpers module and the 1745-line transformers_helpers.py partition the model's forward code into traced subgraphs, with an AST-based auto-wrapper in the ast_utils package that intercepts Hugging Face forward methods without editing them. The cache module implements IntermediatesCache at line 69 - per-input activation buffers with CPU offload and record_stream-aware device management plus an OverrideEqMode dispatch mode at line 507 - so only one subgraph's activations live on the GPU at a time. The modeling package completes the story: [modeling/moe/linearize.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/moe/linearize.py) converts grouped-expert MoE layers into the linear-expert form (531 lines of support in linear_experts.py) that batched GPTQ needs, and the repo even ships native model definitions like [modeling/kimi_k3/modeling_kimi_k3.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/kimi_k3/modeling_kimi_k3.py) at 1362 lines so architectures without upstream Transformers support can still be quantized. Shared plumbing - distributed reductions in [utils/dist.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/utils/dist.py), Triton launch helpers in [utils/triton_utils.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/utils/triton_utils.py), and the norm wrappers in [modeling/offset_norm.py](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/offset_norm.py) - rounds out the runtime.

## From Install to Compressed Checkpoint

The README's quick tour quantizes Qwen3-30B-A3B to block FP8 in about twenty lines:

```sh
pip install llmcompressor
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-30B-A3B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)
oneshot(model=model, recipe=recipe)

model.save_pretrained(MODEL_ID.split("/")[1] + "-FP8-BLOCK")
tokenizer.save_pretrained(MODEL_ID.split("/")[1] + "-FP8-BLOCK")
```

The saved folder then loads directly in vLLM with no special arguments:

```python
from vllm import LLM

model = LLM("Qwen3-30B-A3B-FP8-BLOCK")
output = model.generate("My name is")
```

Honest limits: this is a quantization workbench, not a general trainer - there is no end-to-end finetuning loop beyond the distillation modifier, and accuracy recovery rides on calibration data quality, so garbage calibration text yields garbage scales. Multi-modal and exotic architectures sometimes need the in-repo native definitions or patches rather than loading straight from Transformers, the Triton paths require NVIDIA GPUs, and the recipe format evolves fast enough that old YAML files occasionally need updating after upgrades. Sequential tracing through arbitrary forward code is genuinely hard, and the team documents edge cases where tracing falls back to eager. None of that changes the verdict: if your model's next stop is vLLM, this is the compressor to feed it with.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "The Compressor That Feeds vLLM - Inside vllm-project/llm-compressor",
  "description": "A source tour of vllm-project/llm-compressor: the Modifier lifecycle, CompressionSession, GPTQ with Triton kernels, AWQ and SmoothQuant transforms, FP4 grid-search observers, and sequential subgraph calibration.",
  "datePublished": "2026-10-16",
  "author": { "@type": "Person", "name": "PyShine" },
  "publisher": { "@type": "Organization", "name": "PyShine" },
  "mainEntityOfPage": "https://pyshine.com/llm-compressor/",
  "keywords": "llm-compressor, vLLM, GPTQ, AWQ, SmoothQuant, NVFP4, MXFP4, compressed-tensors, quantization"
}
</script>
