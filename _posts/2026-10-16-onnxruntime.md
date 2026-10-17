---
layout: post
title: "One API, Twenty-Five Accelerators - Inside microsoft/onnxruntime"
description: "A source-level tour of microsoft/onnxruntime: the versioned OrtApi C surface, the InferenceSession execution engine, graph fusion transformers, the MLAS CPU kernel zoo, the CUDA provider with seven fused attention backends, the new plugin execution-provider ABI, and the training stack."
date: 2026-10-16
header-img: "img/post-bg.jpg"
permalink: /onnxruntime/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/onnxruntime/microsoft-onnxruntime-overview-architecture.svg
tags: [ONNX Runtime, inference, MLAS, execution providers, CUDA, quantization]
categories: [AI, Open Source]
keywords: "onnxruntime source code, onnxruntime_c_api.h OrtApi, inference_session.cc RunImpl, graph transformer fusion attention, MLAS qnbitgemm sqnbitgemm, cuda_execution_provider partitioning, flash_attention cutlass_fmha vendored, MatMulNBits, execution provider plugin ABI, orttraining module.cc"
author: "PyShine"
---

Somewhere behind your bank's fraud model, your phone's photo search, a Windows Copilot feature, and half the vector databases on the market, the same C library is running the inference: ONNX Runtime, Microsoft's cross-platform inference and training accelerator, MIT-licensed, at version 1.32.0 in this tree. Its README pitches exactly that reach: models from PyTorch, TensorFlow, scikit-learn, LightGBM, and XGBoost, optimal performance on whatever hardware is present, acceleration through graph optimizations and hardware-specific execution providers. The repository is one of the largest in this series, and its structure makes the scope legible: a public C API designed for long-term binary compatibility, an engine that loads and rewrites computation graphs, a CPU kernel library with its own SIMD dispatch, twenty-five built-in execution providers, and language bindings from C# to Rust. As always in this series, what follows is an educational tour of published source code.

What makes ONNX Runtime architecturally interesting is that it refuses to pick a side. It is neither a pure graph compiler like the TVM tree we toured nor a hand-tuned kernel library like CUTLASS; it is both, arranged in layers so each can be replaced. The graph layer fuses operators before execution, the kernel layer dispatches to per-architecture machine code, and the provider layer hands whole subgraphs to vendor libraries from TensorRT to Core ML. The 1.32 tree shows the newest evolution: a plugin execution-provider ABI that lets hardware vendors ship their backends as separate libraries, and a training stack sharing the same engine. Reading the overview from left to right walks one model from a language binding through graph rewriting and kernel dispatch to whichever accelerator wins the partition.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/onnxruntime/microsoft-onnxruntime-overview-architecture.svg" alt="ONNX Runtime repository architecture overview" style="max-width:100%;"/></div>

<p><em>Overview of ONNX Runtime: language bindings call the versioned C API and the newer EP plugin ABI, the InferenceSession loads and rewrites the graph through fusion transformers, the kernel framework dispatches to MLAS on CPU and partitions subgraphs across the CUDA and twenty-four other execution providers, contrib_ops supplies fused attention and quantized GEMM, and the training stack reuses the engine.</em></p>

Reading the overview from left to right:

- [onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_c_api.h) is the 9234-line public contract, with the OrtApi struct of function pointers declared at 1375 and a C++ RAII wrapper in onnxruntime_cxx_api.h at 3925 lines.
- [inference_session.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/inference_session.cc) is the 5103-line engine: LoadOrtModel at 2156, Run at 3698, and RunImpl at 3713, with provider registration beside it.
- [graph.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/graph/graph.cc) implements the in-memory model at 7411 lines, under a 2240-line graph.h whose class Graph sits at 722.
- [attention_fusion.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/optimizer/attention_fusion.cc) at 1369 lines is the largest of the seventy-seven graph transformer files, flanked by layer_norm_fusion.cc at 922 and bias_gelu_fusion.cc at 113.
- [mlas.h](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/inc/mlas.h) declares the Microsoft Learning Acceleration Service in 2781 lines, backed by 151 kernel files spanning nineteen architecture folders.
- [cuda_execution_provider.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cuda/cuda_execution_provider.cc) runs 3804 lines of capability matching and stream orchestration for the NVIDIA path.
- [attention.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/contrib_ops/cuda/bert/attention.cc) routes fused attention among seven backends, including vendored FlashAttention and CUTLASS FMHA, both of which we toured upstream earlier in this series.
- [onnxruntime_pybind_state.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/onnxruntime_pybind_state.cc) at 3597 lines is the Python binding, with Session at 246, InferenceSession at 618, OrtValue at 1260, and IOBinding at 1067 of the collection module.

## Why You Need This

The first reason is deployment reality. Training happens in a handful of frameworks on a handful of machines, but inference happens everywhere: iOS and Android, Windows and Linux, browsers via WebAssembly, cars, microcontrollers, and every cloud in between. ONNX Runtime is the layer that makes one exported graph run on all of them without per-platform rewrites. The public C API is versioned like an operating system ABI, function tables you query at runtime, which is why binaries built years ago still load, and why the same 9234-line header serves C, C++, C#, Java, JavaScript, Go, Objective-C, and Rust. When a new accelerator appears, the engine does not change shape; a provider plugs in. That stability is the whole product.

The second reason is that the CPU path is deceptively deep. The Microsoft Learning Acceleration Service, MLAS, is a complete SIMD math library living under onnxruntime/core/mlas, and its 151 files cover nineteen architecture folders from scalar and x86 through AVX2, AVX512, AMX, NEON, SVE, S390x, RISCV, LoongArch, and WASM SIMD. Beyond the classic sgemm and convolve kernels sit the LLM-era ones: flashattn.cpp with its GQA and QKV variants, rotary_embedding, linear_attention, layernorm, and halfgemm for fp16. The quantization story is the deepest part, and it connects directly to the weight-quantization machinery in the torchao post: qgemm carries per-ISA int8 kernels, q4gemm handles 4-bit weights, and qnbitgemm with its sqnbitgemm family implements block-quantized N-bit GEMM down to 2-bit block-length headers for AVX2 and AVX512. Your quantized model runs fast on a laptop CPU because of these files.

The third reason is the graph optimizer, the quiet pass that makes exported models fast before any vendor library sees them. Seventy-seven transformer files fuse what export tools scatter: attention_fusion.cc rebuilds scaled dot-product attention blocks, layer_norm_fusion collapses norm chains, gelu and bias-gelu fusions remove glue, matmul_add fuses linear layers, constant folding and common subexpression elimination clean the rest. The GraphTransformerManager applies them in level steps through ApplyTransformers at line 26 of its 100-line manager, with the full suite assembled by GenerateTransformers at 210 of graph_transformer_utils.cc and a reduced minimal-build set at 555. Then partitioning happens per execution provider, and the CUDA path alone dispatches fused attention among seven backends: cudnn_fmha, the vendored flash_attention tree with its per-head-dimension template instantiations, cutlass_fmha, lean_attention, xqa, fastertransformer decoding, and TensorRT's fused kernels.

<div style="text-align:center;"><img src="https://pyshine.com/assets/img/diagrams/onnxruntime/microsoft-onnxruntime-architecture.svg" alt="ONNX Runtime repository architecture detail" style="max-width:100%;"/></div>

<p><em>Detail view: the OrtApi tables and EP plugin ABI front the engine, sessions load models into Graph objects viewed read-only during partitioning, transformer levels apply fusions and QDQ handling, the kernel framework plans execution and dispatches to MLAS families, the CUDA provider routes attention backends, and bindings plus training close the loop.</em></p>

The execution engine rewards reading in order. inference_session.cc loads via LoadOrtModel at 2156 or raw bytes at 2178, runs graph transformations, resolves kernels through kernel_registry_manager, and produces an execution plan: sequential_execution_plan.h at 212 lines fixes node order and arena lifetimes, allocation_planner.cc and bfc_arena.cc do the bin-freelist memory choreography, and execution_frame.cc threads OrtValues through it. class OpKernel at line 41 of the 457-line public op_kernel.h is the contract every operator implements on every provider, registered through KernelRegistry at 23 of kernel_registry.h and matched against the partitioned graph through graph_viewer.cc read-only views. When a subgraph lands on a different device, data_transfer_manager.cc moves tensors across, and IOBinding.cc lets you pin inputs and outputs to specific devices to avoid the copies entirely.

The provider layer shows what the plugin ABI changes. The CUDA provider at 3804 lines performs capability matching per node, and its contrib_ops companions carry the frontier: attention.cc at 326 lines with a 1190-line attention_impl.cu choosing among the seven fused backends, MatMulNBits at line 173 of a 1596-line CPU file with CUDA, WebGPU, and JS siblings, beam search and sampling generation ops, and a MoE folder. The other twenty-four provider directories span TensorRT, OpenVINO, QNN, NNAPI, Core ML, DNNL, XNNPACK, WebGPU, WebNN, DirectML, SNPE, CANN, MIGraphX, RKNPU, and Azure, with plugin-ep-cuda and plugin-ep-webgpu already moved out of tree at the root to prove the new onnxruntime_ep_c_api.h model, 3326 lines that define how external providers join. Session-side extras show where the product is heading: lora_adapters.cc manages swappable adapters, model_package_api.cc loads packaged ORT-format models, and the flatbuffers path skips protobuf parsing entirely.

The periphery completes the ecosystem picture. Python's pybind_state.cc at 3597 lines registers every kernel type and OrtValue semantics, while onnxruntime_inference_collection.py adds the friendly surface: Session at 246, InferenceSession at 618, ModelCompiler at 891, IOBinding at 1067, and OrtValue at 1260. The quantization toolkit under python/tools/quantization produces the QDQ models the graph transformer consumes, with quantize.py, base_quantizer.py, qdq_quantizer.py, matmul_nbits_quantizer.py, and matmul_bnb4_quantizer.py, plus calibration and preprocessing. orttraining mirrors the inference engine for learning: training_api holds module.cc, checkpoint.cc, optimizer.cc, and lr_scheduler.cc behind ort_training_apis.h, training_ops adds gradient kernels, and the whole thing is exposed to PyTorch users as a one-line accelerator. A core/dlpack folder even lets tensors flow zero-copy to and from PyTorch, the same DLPack currency the cuVS C API spoke.

## From Install to a Running Session

Install the wheel, load any ONNX model, and let the optimizer and provider selection do their work:

```sh
pip install onnxruntime
# GPU: pip install onnxruntime-gpu
```

```python
import numpy as np
import onnxruntime as ort

# sessions pick available providers in the order given
sess = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print([i.name for i in sess.get_inputs()])
results = sess.run(None, {"input": np.zeros((1, 3, 224, 224), np.float32)})
```

Quantize an existing model to the QDQ format the engine optimizes for with two lines from the same tree's tools, and the fused int8 kernels in MLAS take over on CPU.

Honest limits: the repository is enormous, so building from source is a genuine project with its own build.py machinery and per-platform prerequisites, and the plugin execution-provider ABI is new enough that provider coverage still varies. Some provider directories are vendor-gated by hardware you may not own, and contrib op compatibility across versions is looser than the core ONNX set. None of that changes the core fact: if a model runs on a device you own, the odds are good that this exact engine, this kernel zoo, and this fusion stack are what made it fast, all MIT-licensed in one tree.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BlogPosting",
  "headline": "One API, Twenty-Five Accelerators - Inside microsoft/onnxruntime",
  "description": "A source-level tour of microsoft/onnxruntime: the versioned OrtApi C surface, the InferenceSession execution engine, graph fusion transformers, the MLAS CPU kernel zoo, the CUDA provider with seven fused attention backends, the new plugin execution-provider ABI, and the training stack.",
  "author": { "@type": "Person", "name": "PyShine" },
  "datePublished": "2026-10-16",
  "keywords": "ONNX Runtime, MLAS, execution providers, graph optimization, CUDA EP, MatMulNBits, quantization",
  "image": "https://pyshine.com/assets/img/diagrams/onnxruntime/microsoft-onnxruntime-overview-architecture.svg"
}
</script>
