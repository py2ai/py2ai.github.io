---
layout: post
title: "Inside TensorFlow 2: Eager Execution, tf.function Tracing, and the Grappler Machine"
description: "A source tour of TensorFlow's session-less runtime: how eager ops execute through the TFE C API, how tf.function traces and caches concrete functions, and how Grappler optimizes the graph before it runs."
date: 2026-10-22
header-img: "img/post-bg.jpg"
permalink: /tensorflow/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/tensorflow/google-tensorflow-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: TensorFlow, eager execution, tf.function, Grappler, FuncGraph, ConcreteFunction, XLA, deep learning framework, source code tour
author: "PyShine"
---

TensorFlow has a strange reputation problem. Ten years ago it was the framework everyone complained about: graphs, sessions, feed dictionaries, a learning curve shaped like a wall. TensorFlow 2 deleted almost all of that. There is no `tf.Session` anymore, no placeholder, no manual graph building. You write ordinary Python, ops run immediately, results come back as tensors you can `print()`. And yet the graph machinery never went away — it just moved underneath you, so well hidden that most users never meet it.

That hidden machinery is what this post tours. We will read the actual source of the TensorFlow repository, branch `master`, and follow two execution paths that every training step takes. The first is the eager path: what literally happens between `tf.add(a, b)` and a number on your screen. The second is the traced path: what happens when you slap `@tf.function` on that same code, where the Python body is re-executed in a special graph-capturing mode, and where the legendary Grappler optimizer gets its chance to rewrite everything before a single kernel runs. If you know [Keras](/keras/), you have used the top of this stack; here we go three layers below it. And if you have read our tours of [JAX](/jax/) or PyTorch, you will notice the same evolutionary pressure at work: everyone converged on "eager first, trace later."

The overview below is the whole story in one picture. The left groups are Python, the middle is the C boundary, the right is the C++ runtime plus Grappler, the graph optimizer that runs whenever a traced function is defined.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/tensorflow/google-tensorflow-overview-architecture.svg" alt="TensorFlow runtime architecture" style="max-width:100%;"></div>

<p><em>TensorFlow 2 runtime: Python front-end, the tf.function tracing layer, the TFE C boundary, the C++ eager executor, and Grappler feeding optimized FunctionDefs back into execution.</em></p>

Reading the overview from left to right:

- [tf.function, L1319](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/polymorphic_function/polymorphic_function.py#L1319) — the decorator you actually call; it wraps your Python callable in a `Function` object with a per-signature cache.
- [_maybe_define_function, L188](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/polymorphic_function/tracing_compilation.py#L188) — on every call, computes a cache key and decides whether to trace a new concrete function or reuse an existing one.
- [FuncGraph, L134](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/func_graph.py#L134) — a subclass of `ops.Graph` that records every op your Python body executes while "taping" the run.
- [quick_execute, L28](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/execute.py#L28) — the pybind entry point for single eager ops; `execute = quick_execute` is literally an alias at line 131.
- [TFE_Execute, L924](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/eager/c_api.cc#L924) — the C API call that crosses the Python/C++ boundary.
- [EagerOperation::Reset, L330](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/eager/eager_operation.cc#L330) — builds the mutable op object: name, attrs, inputs, device.
- [EagerExecutor::AddOrExecute, L135](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/eager/eager_executor.cc#L135) — runs the node synchronously in eager mode or queues it when async execution is enabled.
- [KernelAndDeviceFunc::Run, L437](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/eager/kernel_and_device.cc#L437) — the final hop: invoke the FunctionLibraryRuntime and registered kernels on a device.
- [MetaOptimizer::OptimizeGraph, L782](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/meta_optimizer.cc#L782) — Grappler's scheduler, which runs a fixed roster of graph rewriting passes.

## Eager by Default: Anatomy of a Single Op

In TF1, an op call appended a node to a graph and did nothing else. In TF2, `tf.add(a, b)` computes. The generated Python wrapper eventually lands in [execute.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/execute.py#L28), whose `quick_execute` receives the op name, inputs, attrs, and the global [Context, L519](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/context.py#L519) — the process-wide object that knows whether you are in eager mode, which devices exist, and how tensors are placed. From there the call crosses into C++ through `TFE_Execute`, where an `EagerOperation` is reset and filled with the op's metadata, then handed to the executor.

The executor is where TensorFlow's async mode lives. By default, `EagerExecutor::SyncExecute` runs the node on the calling thread and blocks. If you enable async execution, `AddOrExecute` enqueues the node and returns immediately — the op appears to have completed, but the actual kernel launch happens on a background thread, and errors surface later, when you touch the output. This is the same trick [PyTorch](/pytorch/) plays with its CUDA streams: keep the Python thread responsive, let the device lag behind.

At the bottom of the eager path sits `KernelAndDeviceOp::Run` in [kernel_and_device.cc, L287](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/eager/kernel_and_device.cc#L287), which resolves the op's registered kernel for the target device and launches it. Outputs come back wrapped as `TensorHandle`s — lazy references that may or may not yet contain computed data. Every eager object you see in Python, including the `EagerTensor` you print, is a thin Python skin over one of these handles.

## tf.function: Tracing, Caches, and Concrete Functions

The eager path is convenient but slow per-op: every call pays Python dispatch, attribute validation, and a C boundary crossing. `tf.function` is the escape hatch. When you call a decorated function, `Function.__call__` at [polymorphic_function.py, L809](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/polymorphic_function/polymorphic_function.py#L809) consults a cache keyed on argument types and shapes. A miss triggers `_maybe_define_function` in [tracing_compilation.py, L188](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/polymorphic_function/tracing_compilation.py#L188), which is where the interesting transformation happens.

Tracing re-executes your Python body, but inside a [FuncGraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/func_graph.py#L134). Tensors flowing through the graph are placeholders; every TensorFlow op you call is appended to the FuncGraph instead of executing. Python control flow is unrolled at trace time — an `if` on a tensor value either works via AutoGraph's rewriting or just takes one branch forever, which is the single most common source of `tf.function` bugs. The captured graph is then packaged by `from_func_graph` at [atomic_function.py, L471](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/polymorphic_function/atomic_function.py#L471) into an `AtomicFunction`, a lean executable wrapper, and stored in a `ConcreteFunction` ([concrete_function.py, L1019](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/polymorphic_function/concrete_function.py#L1019)) pinned to one input signature.

This design explains the behavior everyone learns the hard way: call your function with a `[None, 784]` tensor and a `[32, 784]` tensor and you get two traces, two graphs, two copies of your weights' references. Pin shapes with `input_signature` and the cache stays small. When you suspect tracing trouble, `tf.config.run_functions_eagerly(True)` — implemented in [eager_function_run.py, L32](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/eager/polymorphic_function/eager_function_run.py#L32) — flips every function back to plain eager execution, which is usually enough to isolate the bug. The trace-then-cache pattern should be familiar from [JAX](/jax/): both frameworks bet that the first call is expensive and every later call should be pure C++.

## Grappler: The Optimizer That Runs Before You Notice

Here is the part of TF2 that has no PyTorch equivalent. The moment a FuncGraph is finalized, it can be handed to Grappler, TensorFlow's graph optimizer. The item builder at [grappler_item_builder.cc, L296](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/grappler_item_builder.cc#L296) wraps the GraphDef together with its fetch tensors and init ops into a `GrapplerItem`, and [MetaOptimizer::OptimizeGraph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/meta_optimizer.cc#L782) runs the passes in a fixed schedule: constant folding, arithmetic simplification, op fusion, function inlining, layout optimization, and more, each implemented as a self-contained rewriter.

Constant folding ([constant_folding.cc, L4057](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/constant_folding.cc#L4057)) precomputes subgraphs whose inputs are all known. The arithmetic optimizer ([arithmetic_optimizer.cc, L4489](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/arithmetic_optimizer.cc#L4489)) removes redundant multiplications and reassociates the data flow so that common patterns collapse. Remapper ([remapper.cc, L5021](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/remapper.cc#L5021)) fuses Conv2D with BatchNorm and activations into single kernels — a direct speedup for every CNN trained through Keras. The function optimizer ([function_optimizer.cc, L1528](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/function_optimizer.cc#L1528)) works on the function library itself, inlining and specializing FunctionDefs. Several passes lean on static shape inference from [GraphProperties, L2538](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/costs/graph_properties.cc#L2538), and when the meta-optimizer needs a real device to measure against, [SingleMachine::Initialize, L142](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/clusters/single_machine.cc#L142) provisions a local cluster for the job.

The result is that a `@tf.function` you wrote naively often runs significantly faster than the same code run eagerly, before you touch XLA at all. Other stacks expose similar machinery — [ONNX Runtime](/onnxruntime/) and [TVM](/tvm-2/) both do graph-level rewrites — but Grappler's framing is unusual: it is always on, mostly invisible, and tuned more for training graphs than for deployment-time minification.

## From Install to Your First Model

The install is the boring part:

```bash
pip install tensorflow
```

Now a minimal snippet that deliberately walks both paths from the diagrams — an eager function, a traced function with a pinned signature, and the eager-mode debug switch:

```python
import tensorflow as tf

def dense(x, w, b):
    return tf.add(tf.matmul(x, w), b)

w = tf.Variable(tf.random.normal([784, 32]))
b = tf.Variable(tf.zeros([32]))

# Eager path: each call goes quick_execute -> TFE_Execute -> kernels.
y1 = dense(tf.ones([1, 784]), w, b)

# Traced path: first call traces a FuncGraph, later calls reuse it.
fast = tf.function(dense, input_signature=[
    tf.TensorSpec([None, 784], tf.float32), w, b])
y2 = fast(tf.ones([1, 784]))   # trace happens here
y3 = fast(tf.ones([5, 784]))   # cache hit: same concrete function

# Debug: pretend it is 2026 but execute like 2020-eager.
tf.config.run_functions_eagerly(True)
y4 = fast(tf.ones([2, 784]))   # no tracing, pure eager again
tf.config.run_functions_eagerly(False)
```

Three calls, three different stories: `y1` crosses the C boundary twice per op, every time. `y2` and `y3` hit the same cached concrete function — the `None` in the signature means both batch sizes share one trace. `y4` skips the cache entirely. Add `print(w.shape)` after any of these and you are back to ordinary Python debugging.

## Honest Limits

**Binary size is enormous.** The CPU-only pip wheel approaches 600 MB installed, and with CUDA libraries the footprint grows by gigabytes. Compare [llama.cpp](/llama.cpp-2/) or [GGML](/ggml/), which run models from a single binary. TensorFlow's runtime carries ten years of op registry, Grappler passes, and kernel implementations for every device it might meet.

**Graph-mode complexity did not disappear; it moved.** The session is gone, but retracing surprises, AutoGraph edge cases, and the "Python side effects run only at trace time" rule are all still there, now documented in caveats rather than fixed by design. Teams shipping large `tf.function` codebases still maintain shape-signature discipline by hand.

**XLA compile latency is real.** `tf.function(jit_compile=True)` can transform performance on TPU and GPU, but the first call pays a full compile, and dynamic shapes force recompilation. For latency-sensitive serving, this trade needs the same budgeting you would give any ahead-of-time compiler.

**Windows is a second-class platform.** Since release 2.11, native Windows pip packages are CPU-only; GPU support requires WSL2. Linux remains the target the build system actually optimizes for, and CI behavior reflects it.

**Export is graph-shaped.** SavedModel serializes concrete functions, which means deployment tooling inherits the traced view of your model — one more reason the tracing internals above matter even if you never read them in anger. When the target is a leaner runtime, exporting to [ONNX](/onnx/) and serving with specialized runtimes remains the pragmatic route.

## Where TensorFlow Sits in the Stack

TensorFlow today is a full-stack compiler wearing a Python API as a mask: eager mode for development speed, tracing for execution efficiency, Grappler for free wins, XLA for the last mile. The layers are genuinely separable — Keras users live entirely above them, JAX proved you can expose only the traced layer and still win, and the serving ecosystem downstream ([vLLM](/vllm-2/), [TGI](/tgi/)) mostly consumes the exported graph form. Reading the source makes the whole stack feel less like magic: it is one long chain from `def function(...)` to a kernel launch, with a very determined optimizer standing guard in the middle.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Inside TensorFlow 2: Eager Execution, tf.function Tracing, and the Grappler Machine",
  "description": "A source tour of TensorFlow's session-less runtime: how eager ops execute through the TFE C API, how tf.function traces and caches concrete functions, and how Grappler optimizes the graph before it runs.",
  "author": {"@type": "Person", "name": "PyShine"},
  "datePublished": "2026-10-22",
  "keywords": "TensorFlow, eager execution, tf.function, Grappler, FuncGraph, ConcreteFunction, XLA, deep learning framework, source code tour",
  "publisher": {"@type": "Organization", "name": "PyShine"}
}
</script>
