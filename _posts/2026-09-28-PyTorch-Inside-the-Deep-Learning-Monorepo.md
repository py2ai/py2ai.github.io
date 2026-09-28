---
layout: post
title: "PyTorch: From torch.add Down to the C++ Kernel - Inside pytorch/pytorch"
description: "A source-level tour of pytorch/pytorch: how the monorepo layers the torch Python frontend over torch/csrc bindings, the c10 dispatcher, the ATen tensor library and its CPU/CUDA kernels, plus the autograd engine, torch.compile, torch.export and torch.distributed machinery."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /PyTorch-Inside-the-Deep-Learning-Monorepo/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/pytorch/pytorch-pytorch-architecture.svg
tags:
  - PyTorch
  - Deep Learning
  - C++
  - Open Source
categories: [AI, Open Source]
keywords: "PyTorch architecture, pytorch pytorch repository, inside PyTorch source code, ATen tensor library, c10 dispatcher, autograd engine, torch.compile Dynamo Inductor, torch.export, torch distributed c10d, PyTorch monorepo, torchgen codegen, build PyTorch from source, PyTorch C++ kernel dispatch, TorchScript mobile runtime, deep learning framework internals"
author: "PyShine"
---

Every day, millions of `import torch` statements run code whose interior most users never see. You call `torch.add`, a gradient appears during `loss.backward()`, `torch.compile` makes a model mysteriously faster - and all of it feels like one seamless library. It is not. It is a carefully layered machine where Python, C++, CUDA, and generated code cooperate through a dispatch system that most people never learn exists.

That machine lives in [pytorch/pytorch](https://github.com/pytorch/pytorch), one of the largest open-source repositories on GitHub: more than 22,000 tracked files, roughly 5,000 Python modules and 4,400 C++ headers and sources, currently carrying version 2.15.0a0 on its `main` branch and a BSD-style license. The README describes the package in two lines - tensor computation with strong GPU acceleration, and neural networks on a tape-based autograd system - but the repository behind those lines is organized like a small operating system for tensors.

This post is not another usage tutorial. The blog already has install guides and model examples; what it does not have is a source tour. We will walk the repository the way its own developers navigate it: from the `torch/` Python frontend, across the `torch/csrc` pybind boundary, into the dispatcher, down to `aten/src/ATen` and its CPU and CUDA kernels, with stops at autograd, `torch.compile`, `torch.export`, and the distributed stack.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/pytorch/pytorch-pytorch-overview-architecture.svg" alt="Architecture overview of the pytorch/pytorch repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview: the torch Python package fronts nn/optim, torch.compile and torch.export, crosses the torch/csrc pybind boundary into the autograd engine and TorchScript, dispatches into the ATen tensor library, and rests on the c10 core - while torch.distributed binds c10d collectives and tools/ + torchgen generate the glue that holds it together.*

Reading the overview from left to right: everything a user touches is Python - `torch/__init__.py` (the package entry, where `torch.compile` is defined), `torch/nn` and `torch/optim` (the neural network and optimizer layers), and two compile-and-package surfaces, `torch.compile` backed by Dynamo and Inductor, and `torch.export` for whole-program capture. The first crossing is the pybind boundary in `torch/csrc`, where Python objects become C++ tensors and the autograd engine and TorchScript live. Below that sits the real computational core: `aten`, the ATen tensor library, resting on `c10/core` - TensorImpl, storage, and the dispatch keys that decide which kernel runs. Two side towers complete the picture: `torch/distributed` over the c10d C++ collectives, and the `tools/` plus `torchgen` codegen system that generates a large share of the bindings and kernel plumbing you see in the other boxes. Keep the layering in mind - Python, C++ bindings, dispatch, kernels - because every section below zooms into one of those bands.

## Why You Need This

First, the error messages. Every serious PyTorch user has met a stack trace that ends in `aten/src/ATen/...` or a warning that mentions the dispatcher, and half of debugging is knowing which layer is lying to you. When a tensor silently changes device, when `backward()` complains about a graph, or when a custom `autograd.Function` misfires, the answer is almost always in the boundary code this tour covers. Reading the source turns those cryptic frames into a map.

Second, the repository is the specification. The docs describe behavior; the source defines it. `native_functions.yaml` under `aten/src/ATen/native/` lists 2,590 operator definitions - the real contract of the tensor API - and the files in `tools/autograd/`, including `derivatives.yaml`, define exactly how each operation differentiates. When two sources of documentation disagree, that YAML is what wins.

Third, performance work is source work. `torch.compile` is not a black box: its docstring in `torch/_dynamo/__init__.py` says plainly that it hooks CPython's frame-evaluation API (PEP 523), rewrites bytecode, and extracts an FX graph for a customizable backend. If you want to predict when compilation will help - or why it graph-breaks - you need the layering, not the marketing.

Finally, this codebase is one of the best-maintained large C++/Python hybrids in open source, and it is BSD-licensed. Reading how it manages a 2,590-op surface with generated bindings, how `c10::Dispatcher` routes calls, and how the build system generates thousands of files is a free education in scaling a framework - whether or not you ever patch a kernel.

## How It Works

The detailed diagram maps the real directories of the monorepo onto the call path of a single tensor operation, from the Python frontend to the kernels.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/pytorch/pytorch-pytorch-architecture.svg" alt="Detailed architecture of the pytorch/pytorch repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the torch Python package (nn, optim, package) fronts torch.compile (TorchDynamo over torch/csrc/dynamo frame hooks, torch.fx, TorchInductor) and torch.export; torch/csrc is the pybind boundary that runs the autograd engine and enters the c10::Dispatcher, which reads dispatch keys from c10/core and routes into ATen and its native CPU/CUDA kernels; TorchScript, its lite interpreter and serialization target deployment, while torch.distributed binds c10d collectives and RPC, and tools/ with torchgen generate the bindings from native_functions.yaml and derivatives.yaml.*

### Understanding the Architecture

**The Python frontend.** Everything starts in `torch/__init__.py`, a 3,000-plus-line module that imports the compiled `torch._C` extension and re-exports it as friendly Python - this is where `torch.compile` itself is defined. On top of it live the layers users actually touch: `torch/nn` for modules and losses, `torch/optim` for update rules, and `torch/func`, which is a thin re-export of the functorch transforms in `torch/_functorch` (`grad`, `vmap`, `jacrev`, `jacfwd`, `hessian`, `functionalize`). `torch/package` rounds out the frontend by bundling Python code and weights into a self-contained archive.

**The pybind boundary and the autograd engine.** The crossing from Python into C++ happens in `torch/csrc`, the pybind layer that turns Python objects into `at::Tensor` values. Its most interesting residents are `torch/csrc/dynamo` - the C++ half of the frame-evaluation hook that TorchDynamo uses to intercept and rewrite bytecode before it runs - and `torch/csrc/autograd`, home of `engine.cpp`. The engine there is a thread-pool-based work queue: it walks the graph of `Node`s recorded during the forward pass, executes them on background threads, and accumulates gradients into leaves. Fork-safety handling in that file tells you how seriously this code takes its own concurrency.

**The dispatcher.** Every tensor operation funnels through `c10::Dispatcher`, declared in `aten/src/ATen/core/dispatch/Dispatcher.h`. Operators are registered under dispatch keys held in `c10/core/DispatchKeySet.h` - CPU, CUDA, Autograd keys, backend-fallback keys - and the dispatcher picks the right kernel by intersecting each tensor's key set. Registration is an event the dispatcher publishes through `OpRegistrationListener`, and the code is explicit that events fire on `def` registrations, not on `impl` or `fallback` calls. This one mechanism is why eager mode, autograd, `vmap`, and tracing can all observe the same call.

**The tensor library.** Below the dispatcher sits ATen itself, the C++ tensor library in `aten/src/ATen`. Its `TensorIterator.cpp` is the shared engine behind element-wise, reduction, and comparison kernels - it settles broadcasting, type promotion, and parallelization once so every kernel inherits the behavior. Hardware paths branch from there: vectorized CPU loops live in `aten/src/ATen/native/cpu`, CUDA kernels in `aten/src/ATen/native/cuda`. The operator surface itself is declared in `aten/src/ATen/native/native_functions.yaml`, the same file the build reads to generate bindings.

**The compile stack.** `torch.compile` is a pipeline, not a single pass. `torch/_dynamo` rewrites Python bytecode - using the frame-evaluation hook that `torch/csrc/dynamo` implements - into a `torch.fx` graph, and `torch/_inductor` lowers that graph into Triton GPU kernels or C++ CPU code. `torch/export` shares the tracing machinery but stops at a serializable `ExportedProgram`: a graph plus tensor constants with explicit guards, meant for deployment rather than speedup. And `torch/_functorch` implements the transforms that `torch.func` re-exports, so `vmap` and `grad` are ordinary library code, not interpreter magic.

**The codegen machine.** Much of this repository writes itself. `torchgen/gen.py` reads `native_functions.yaml` and emits dispatcher registrations, Python bindings, and kernel stubs throughout `torch/csrc` and `aten/src`, while `tools/autograd/derivatives.yaml` defines how every operator differentiates. That is how a surface of thousands of operators stays consistent across Python, C++, and CUDA - and why a from-source build spends so much time generating files before compiling anything.

Follow one call end to end: `torch.add(a, b)` starts in the Python frontend, crosses the pybind boundary in `torch/csrc`, and reaches `c10::Dispatcher`. The dispatcher reads the key set stamped on each tensor, hits the Autograd key first - queuing a node for the engine in `torch/csrc/autograd/engine.cpp` - then redispatches to a kernel under `aten/src/ATen/native`. One call, four layers, four directories - and that path is the whole architecture in miniature.

## Advantages

- **One dispatcher for everything.** Eager execution, autograd, `vmap`, tracing, and TorchScript all observe the same call through `c10::Dispatcher`, so features compose instead of fighting each other.
- **Generated plumbing, hand-written kernels.** Schemas in `native_functions.yaml` generate the bindings and registrations; engineers spend their time on kernels rather than boilerplate.
- **Eager and compiled execution share one graph.** `torch.compile` speeds up hot paths while plain eager mode stays available, and both run on the same `torch.fx` representation.
- **Export is a first-class concern.** `torch.export` produces guarded, serializable graphs that run outside Python on servers and mobile devices.
- **A real C++ API.** `torch/csrc/api` mirrors much of the Python surface, so C++ applications can use the tensor library without a Python runtime.
- **Distribution built in.** `torch/distributed` with the c10d C++ collectives ships in the core repository, not as an add-on.

## Benefits

- **Debuggability by layer.** Stack traces that end in `aten/src/ATen` or mention the dispatcher become navigable once you know which layer owns what.
- **Portable deployment.** TorchScript, its lite interpreter, and serialized programs move models off the training machine without a rewrite.
- **Extensibility without forking.** A new kernel or a custom `autograd.Function` plugs into the same dispatcher the built-ins use.
- **Performance you can reason about.** Knowing where dispatch, broadcasting, and kernel selection happen turns slow operations into diagnosable ones.
- **Permissive licensing.** The BSD-style license lets commercial products build on the entire stack.
- **A reference architecture.** The layering - Python frontend, pybind boundary, dispatcher, kernels - is a template worth studying for any large hybrid codebase.

## Usage

Everything in this tour is observable on your own machine, because the repository is built to be built. The README asks for Python 3.10 or later, a compiler with full C++20 support, at least 10 GB of free disk space, and 30-60 minutes for the first build. After cloning and updating submodules, install the development dependencies and build the package in editable mode:

```bash
pip install --group dev
export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python -m pip install --no-build-isolation -v -e .
```

Useful variations from the same document: export `USE_CUDA=0` for a CPU-only build, and for AMD ROCm run `python tools/amd_build/build_amd.py` before installing. If you would rather not build at all, the Docker route mirrors the wheel experience, and the repository ships its own image recipe:

```bash
docker run --gpus all --rm -ti --ipc=host pytorch/pytorch:latest
```

```bash
make -f docker.Makefile
```

Contributors who only want the documentation can skip the C++ build entirely:

```bash
cd docs/
pip install -r requirements.txt
make html
```

Once the build finishes, `import torch` loads the compiled `torch._C` extension, and every path described above - `torch/csrc`, `aten/src/ATen`, `torch/_dynamo` - is live in your environment, ready to be read or patched.

## Conclusion

pytorch/pytorch is not a Python library with a C++ accent; it is a layered system where Python is only the front door. The frontend in `torch/`, the bindings in `torch/csrc`, the dispatcher in `aten/src/ATen/core/dispatch`, the kernels under `aten/src/ATen/native`, and the codegen in `torchgen/` and `tools/` each do one job and hand off cleanly to the next. Read it once and the framework stops being magic: stack traces become maps, compile behavior becomes predictable, and the distance between `torch.add` and a CUDA kernel shrinks to a path you can walk.

Links:

- [pytorch/pytorch on GitHub](https://github.com/pytorch/pytorch)
- [PyTorch documentation](https://pytorch.org/docs/)
