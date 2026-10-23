---
layout: post
title: "coremltools: How Apple Compiles PyTorch Models Onto the Neural Engine"
description: "A source tour of apple/coremltools: the conversion flow from PyTorch and TensorFlow into Core ML, the MIL intermediate representation, graph passes, and post-training quantization."
date: 2026-10-23
header-img: "img/post-bg.jpg"
permalink: /coremltools/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/coremltools/apple-coremltools-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: coremltools, Core ML, MIL intermediate representation, model conversion, quantization, palettization, pruning, Apple Neural Engine, PyTorch to Core ML, on-device inference
author: "PyShine"
---

If you have ever trained a network in [PyTorch](/pytorch/) and then wondered how it ends up running inside an iOS app with a 40% smaller footprint and zero Python interpreter in sight, the answer is almost always **coremltools**. Apple's open-source toolkit takes a model expressed in a training framework, rewrites it into an intermediate representation called MIL (Model Intermediate Language), and serializes it into the `.mlpackage` format that Core ML — the on-device runtime baked into iOS, iPadOS and macOS — can load, schedule across CPU/GPU/Neural Engine, and execute natively. The repo lives at `apple/coremltools`, it is a large but readable Python codebase, and in this post we walk its main streets: the conversion entry point, the MIL IR, the graph passes that optimize programs, the backend that serializes them, and the post-training compression stack.

It is worth placing the tool in Apple's ecosystem before diving into code. [MLX](/mlx/) is Apple's NumPy-like array framework for *training and research* on Apple silicon — it wants to be your modeling language. coremltools is the *shipping* layer: it assumes you already have trained weights and cares about freezing them into a format that Xcode, `CoreML.framework`, and the ANE compiler stack understand. The two are siblings, not competitors, and a common workflow is train in PyTorch (or MLX, exporting through PyTorch), then hand the frozen module to coremltools.

The tour below is based on a checkout of `main`. We will follow one conversion end to end, look at how op coverage and deployment targets interact, run an actual `ct.convert`, then compress a model after the fact. Two diagrams anchor the navigation: a compact overview and a detailed graph with 21 nodes.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/coremltools/apple-coremltools-overview-architecture.svg" alt="coremltools architecture" style="max-width:100%;"></div>

<p><em>Overview of coremltools: user API, frontends, the MIL IR, graph passes, backend serialization, the MLModel runtime wrapper, and the post-training optimization stack.</em></p>

Reading the overview from left to right:

- [`convert`, L69](https://github.com/apple/coremltools/blob/main/coremltools/converters/_converters_entry.py#L69) — the public `ct.convert()` signature: source auto-detection, `inputs`/`outputs` types, `compute_units`, `minimum_deployment_target`, and a states argument for KV-cache-style models.
- [`mil_convert`, L148](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/converter.py#L148) — the dispatcher. A `ConverterRegistry` maps a `convert_from` string (`pytorch`, `tensorflow`, `tensorflow2`) to a frontend class and `convert_to` (`mlprogram`, `neuralnetwork`) to a backend.
- [`load`, L27](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/frontend/torch/load.py#L27) — the PyTorch frontend's gate: it accepts a TorchScript module, a `.pt`/`.pth` path, or a `torch.export` `ExportedProgram`, and wraps it in a `TorchConverter`.
- [`convert_single_node`, L125](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/frontend/torch/ops.py#L125) — the heart of transcription: for every node in the internal torch graph it looks up a registered handler, binds scope/debug metadata, and emits MIL ops through the `mb.*` builder.
- [`SSAOpRegistry`, L17](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/ops/registry.py#L17) — the versioned op registry: `core_ops` (keyed by opset, from iOS13 up), `dialect_ops` (framework-specific), and `custom_ops`.
- [`Program`, L26](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/program.py#L26) — the SSA IR container that frontends emit into: functions, blocks, `Var`s, plus hooks for the pass machinery.
- [`MILProtoExporter`, L75](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/backend/mil/load.py#L75) — the mlprogram backend: converts a PyMIL `Program` into the MIL protobuf, streaming large weights into external `.bin` files inside the package.
- [`MLModel`, L308](https://github.com/apple/coremltools/blob/main/coremltools/models/model.py#L308) — the runtime wrapper returned to users: `save`, `predict`, compute-unit pinning, state objects, and spec introspection.
- [`linear_quantize_weights`, L56](https://github.com/apple/coremltools/blob/main/coremltools/optimize/coreml/_post_training_quantization.py#L56) — the post-training path: int8/int4 weight quantization, palettization and pruning applied to an already-converted `.mlpackage`.

## The MIL IR: One Intermediate for Three Frontends

The design decision that shapes the whole repository is MIL: a single static single-assignment language that every frontend lowers into and every backend consumes. The [Builder, L69](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/builder.py#L69) is what frontend code actually calls — `mb.conv`, `mb.mul`, `mb.matmul` — and each call resolves the correct *version* of the op for the deployment target you asked for. The same op name can exist at multiple opsets; the registry picks the right class from `core_ops[op_type]` based on the current function's opset version. That is how one conversion target string (`ct.target.iOS17`, for example) changes the shape of the generated program.

Op definitions are physically organized by opset folder, which makes platform evolution easy to read in the source tree. [`conv`, L17](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/ops/defs/iOS17/conv.py#L17) subclasses the iOS15 implementation and registers itself for the iOS17 target; newer machine-learning ops live in the iOS18 folder — for instance [`scaled_dot_product_attention`, L18](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/ops/defs/iOS18/transformers.py#L18), the fused attention primitive that transformer models need to avoid materializing full attention matrices on device.

The TensorFlow side reuses exactly the same IR. [`TF2Loader`, L43](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/frontend/tensorflow2/load.py#L43) consumes Keras models, SavedModel directories, frozen graphs and concrete functions, and emits MIL just like the torch frontend does. So if your origin story is [TensorFlow](/tensorflow/) rather than PyTorch, the back half of the flow — passes, serialization, runtime — is byte-for-byte the same machinery. On the torch side, handlers are registered with [`register_torch_op`, L100](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/frontend/torch/torch_op_registry.py#L100), which also supports aliases (one torch op kind handled by several names) and inplace variants that the context tracks so SSA semantics are preserved.

## Graph Passes: Where Models Get Faster

After a frontend produces a raw MIL program, the optimization stage rewrites it in place. Every rewrite implements [`AbstractGraphPass`, L41](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/mil/passes/graph_pass.py#L41) — a small interface with a `apply_to_program`-style entry plus `PassOption` knobs that users can override through `pass_options` when calling `ct.convert`.

A representative example is [`fuse_activation_silu`, L66](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/backend/mil/passes/fuse_activation_silu.py#L66): it pattern-matches `sigmoid` feeding a `mul` and replaces the pair with a single `silu` op. Fewer ops means fewer dispatches at runtime and a better shot at ANE scheduling. The same folder contains fusions for pow/sqrt chains, constant decomposition, and name sanitization. Two passes deserve special mention because they touch *correctness* rather than speed: [`adjust_io_to_supported_types`, L20](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/backend/mil/passes/adjust_io_to_supported_types.py#L20) casts inputs and outputs to dtypes the chosen opset actually supports (fp16 I/O for mlprogram targets, for example), and [`insert_image_preprocessing_ops`, L18](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/backend/mil/passes/insert_image_preprocessing_op.py#L18) bakes scale/bias and channel reordering into the graph when you declare an input as `ct.ImageType`, so the app can hand the model a raw `CVPixelBuffer` and get correct numbers.

Serialization then happens in the mlprogram backend. The `MILProtoExporter` writes per-op protobuf records, and `should_use_weight_file` (same file, L59) decides whether a constant is small enough to inline or big enough to stream into `Weight.bin` inside the package — this is the mechanism that makes multi-gigabyte LLMs storable and lazily mappable by the OS. The companion class [`CoreMLProtoExporter`, L590](https://github.com/apple/coremltools/blob/main/coremltools/converters/mil/backend/mil/load.py#L590) layers on the model description: input/output feature types, flexible-shape enumerations, classifier wiring, and state features.

## From Install to Your First Model

Installation is one pip line (Python 3.9+ is what current releases target):

```bash
pip install coremltools
```

You will also need a PyTorch install if your source model is torch-based. The minimal flow is: script or export the module, call `ct.convert` with typed inputs, save the package:

```python
import coremltools as ct
import torch

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.head = torch.nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.head(torch.flatten(x, 1))

model = TinyNet().eval()
scripted = torch.jit.script(model)

mlmodel = ct.convert(
    scripted,
    inputs=[ct.TensorType(shape=(1, 3, 8, 8))],
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
)
mlmodel.save("TinyNet.mlpackage")
```

`ct.convert` (the function at the top of the overview diagram) returns an [`MLModel`](https://github.com/apple/coremltools/blob/main/coremltools/models/model.py#L308) immediately. Two arguments matter most in practice. `compute_precision=ct.precision.FLOAT16` tells the converter to store and compute in half precision — the default for `mlprogram` targets and what the Neural Engine expects. `compute_units` accepts `ct.ComputeUnit.ALL`, `CPU_ONLY`, or `CPU_AND_GPU`; `ALL` lets the runtime schedule on the ANE when ops allow it. The same wrapper then loads the package and runs inference on a Mac via [`predict`, L826](https://github.com/apple/coremltools/blob/main/coremltools/models/model.py#L826) — a dict in, a dict out — which is also how you sanity-check numerics before deploying to a phone. One subtlety worth knowing: `predict` requires macOS 12.0+ for `.mlpackage` models, and the spec version you target must be supported by the Core ML framework installed on the machine.

## Weight Compression After the Fact

Conversion is only half the toolkit. `coremltools.optimize` operates on already-converted `.mlpackage` models and applies post-training compression to their weight constants. The config objects live in `_config.py` — [`OpLinearQuantizerConfig`, L131](https://github.com/apple/coremltools/blob/main/coremltools/optimize/coreml/_config.py#L131) exposes `mode` (`linear_symmetric`/`linear`), `dtype` (`int8`, `int4`), `granularity` (`per_tensor`, `per_channel`, `per_block`) and `block_size` — and the public functions `linear_quantize_weights`, `palettize_weights`, `prune_weights`, and `get_weights_metadata` are what you call from Python:

```python
from coremltools.optimize.coreml import (
    OpLinearQuantizerConfig,
    linear_quantize_weights,
)

config = OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int4",
    granularity="per_block",
    block_size=32,
)
compressed = linear_quantize_weights(mlmodel, config=config)
compressed.save("TinyNet_int4.mlpackage")
```

Under the hood each function instantiates a registered MIL pass — for quantization that is [`linear_quantize_weights`, L1170](https://github.com/apple/coremltools/blob/main/coremltools/optimize/coreml/_quantization_passes.py#L1170) — which rewrites each weight `const` into a compressed `constexpr_` dequantize form, so the on-device runtime materializes weights on the fly in the compute unit's preferred format. The same registry mechanism that served conversion is reused here, which is a genuinely elegant piece of reuse. For LLM-sized models, `per_block` int4 is the option Apple's own export recipes lean on: it keeps outliers tolerable by giving every 32-element block its own scale, typically landing a 3-4x weight-size reduction versus fp16 with modest quality loss.

## Compute Units: CPU, GPU, or the Neural Engine

`compute_units` is a *hint to the scheduler*, not a guarantee. `CPU_ONLY` and `CPU_AND_GPU` are hard constraints the runtime respects; `ALL` permits ANE scheduling but the actual placement is decided by Core ML at load time based on op support, dtype, and shape characteristics. The practical consequences show up in conversion behavior: the `adjust_io_to_supported_types` pass exists precisely because ANE-friendly models want fp16 boundaries, and the converter will warn when symbolic shapes or unsupported dtypes force fallbacks. If you have worked with [ONNX](/onnx/) or other interchange formats, the mental model is similar — a target-agnostic graph — except here the "target" is a family of Apple chips with a compiler stack you cannot inspect, so coremltools encodes a lot of hard-won knowledge about what the ANE actually accepts into passes and opset gating.

## Honest Limits

The macOS/Xcode coupling is real. `predict()` runs only on macOS (10.13 minimum, 12.0+ for packages), the specification version you target must be supported by the installed Core ML framework, and new opsets (iOS17/iOS18) require new-enough OSes to load at all — you cannot ship an iOS18-spec model to an iOS15 device. Keeping a matrix of target OS versions and testing each exported package against the oldest one you support is not optional.

ANE coverage is a moving target. `compute_units=ALL` frequently produces a model that *partially* runs on the Neural Engine with the rest falling back to GPU or CPU, and the fallback decisions happen inside the closed-source runtime — coremltools cannot tell you exactly which subgraphs landed where. Op coverage in the torch frontend is similarly incomplete: any node kind without a registered handler raises `NotImplementedError`, and the frontend explicitly rejects dynamically quantized torch models, pointing you at the `coremltools.optimize` APIs instead (see the check around L136 of the torch `ops.py` transcription module). Complex control flow, exotic dtype tricks, and very new torch releases tend to need a wait-and-retry.

Quantization is not free. Int8/int4 and palettization reduce weight size substantially, but activation distributions that were fine in fp16 can degrade noticeably in int4, especially for small models or aggressive `per_tensor` granularity — the config surface exists precisely so you can trade granularity against size. Budget time for `get_weights_metadata` plus an eval harness, and consider keeping sensitive layers at higher precision. Finally, note what this stack is not: it is not a training runtime (that is MLX or PyTorch again), it is not cross-platform (for that world you would look at [Executorch](/executorch/) or plain ONNX tooling), and its Python-only predict path makes it a poor fit for server-side serving outside Apple hardware.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "coremltools: How Apple Compiles PyTorch Models Onto the Neural Engine",
  "description": "A source tour of apple/coremltools: the conversion flow from PyTorch and TensorFlow into Core ML, the MIL intermediate representation, graph passes, and post-training quantization.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-23",
  "keywords": "coremltools, Core ML, MIL intermediate representation, model conversion, quantization, palettization, pruning, Apple Neural Engine, PyTorch to Core ML, on-device inference",
  "publisher": {"@type": "Organization", "name": "PyShine"}
}
</script>
