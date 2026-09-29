---
layout: post
title: "Keras: One Deep Learning API, Four Backends - Inside keras-team/keras"
description: "A guided source tour of Keras 3, the multi-backend deep learning framework that runs the same model code on JAX, TensorFlow, and PyTorch. We walk the keras/src tree to see how the backend dispatch layer, KerasTensor symbolic graphs, the trainer loop, and the .keras serialization stack fit together."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Keras-Multi-Backend-Deep-Learning-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/keras/keras-team-keras-architecture.svg
tags:
  - Keras
  - Deep Learning
  - JAX
  - PyTorch
categories: [AI, Open Source]
keywords: "Keras 3, multi-backend deep learning, JAX, TensorFlow, PyTorch, OpenVINO, KerasTensor, functional API, keras.ops, trainer loop, model serialization, open source Python framework"
author: "PyShine"
---

Most deep learning libraries ask you to make a lifelong commitment on day one. Pick PyTorch and your research code is welded to `torch.Tensor`; pick TensorFlow and your deployment story runs through SavedModel; pick JAX and you live in a world of pure functions. Keras 3, the project we are touring today, takes the opposite position: the engine underneath your model should be a configuration detail, not an architectural one. You write a layer once, call `model.fit()` once, and the same code executes on JAX, TensorFlow, or PyTorch — with OpenVINO available for inference-only deployments.

Keras 3 lives at [keras-team/keras](https://github.com/keras-team/keras) and ships on PyPI as the `keras` package. It is written almost entirely in Python (3.11+), licensed under Apache 2.0, and its dependencies are refreshingly light — `numpy`, `absl-py`, `h5py`, `optree`, `ml-dtypes`, `namex`, `rich`, and `packaging` per `pyproject.toml`. The backend itself is not a dependency at all; you install TensorFlow, JAX, or PyTorch alongside it and select one with the `KERAS_BACKEND` environment variable before importing.

What makes this repository genuinely worth reading is that the multi-backend trick is not marketing gloss — it is engineered, and the seams are visible in the source. The `keras/src` tree implements one abstraction layer that five backends plug into, a symbolic tensor system that predates any real data, and a trainer loop that gets jitted differently for each engine. Reading it teaches you how to build a framework-shaped library, not just how to use one.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/keras/keras-team-keras-overview-architecture.svg" alt="Architecture overview of the keras-team/keras repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Keras 3 architecture: the generated public API surface, the model and graph core, the backend abstraction, the training stack, and serialization.*

Reading the overview from left to right: everything begins at the generated `keras/api` package, which exposes `keras.Model`, `keras.ops`, and the saving utilities to users. The `Model` class extends `Functional`, which itself builds on the stateless `keras.Function` DAG — the machinery that records how layers connect, one `Node` and one `KerasTensor` at a time. On the right side, `keras.ops` delegates every mathematical operation to the backend dispatch in `keras/src/backend`, which binds one of the concrete engines at import time. Below, the `Trainer` composes fit/evaluate/predict loops on top of `EpochIterator` and its data adapters, while the saving group serializes models into the `.keras` format and rebuilds those graphs on load.

## Why You Need This

The obvious pain point Keras 3 solves is framework lock-in. Research code written against `tf.keras` a few years ago could not run on PyTorch without a rewrite, and JAX adopters had to leave the comfortable `model.fit()` world entirely. In Keras 3 the model definition, the layers, the losses, and the metrics are all backend-agnostic by construction, so the README's promise is concrete: if your `tf.keras` model uses no custom components, it can run on JAX or PyTorch immediately, and custom layers usually need only a few minutes of conversion to use `keras.ops` instead of backend-native calls.

The second problem is data pipeline heterogeneity. Every team has its own way of feeding data — NumPy arrays, `tf.data.Dataset` pipelines, PyTorch `DataLoader`s, Python generators. Keras 3's trainer accepts all of them regardless of which backend is selected, because `keras/src/trainers/data_adapters/` contains a dedicated adapter for each format, including a `grain_dataset_adapter` and a `py_dataset_adapter` for multi-worker loading. Your input pipeline becomes orthogonal to your compute engine.

Third, there is the performance question. Different backends win on different architectures — the project's own README points to a public benchmark suite for choosing — and with Keras 3 you can measure rather than guess, because switching engines is an environment variable change, not a port. The same code that trains on your laptop with the NumPy-backed eager path can be pointed at a TPU cluster through JAX without touching the model file.

Finally, Keras 3 is a pragmatic migration path rather than a rewrite trap. It is designed as a drop-in replacement for `tf.keras` when running on the TensorFlow backend, with Keras 2 preserved separately as the `tf-keras` package. Teams can keep existing `.keras`-format checkpoints, keep their `tf.data` pipelines, and adopt a second backend only where it pays off.

## How It Works

The whole framework is best understood as three concentric layers: a public API shell, a symbolic graph core that knows nothing about hardware, and a backend abstraction that binds to exactly one engine per process.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/keras/keras-team-keras-architecture.svg" alt="Detailed architecture of the keras-team/keras repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of keras-team/keras: the generated API surface, the KerasTensor/Node graph machinery, the five backend packages, the trainer and its data adapters, and the serialization and export stack.*

### Understanding the Architecture

**The import-time backend dispatch.** Everything starts in `keras/src/backend/config.py`, which reads the `KERAS_BACKEND` environment variable at import time (defaulting to `tensorflow`) alongside settings like `floatx` and the image data format. Then `keras/src/backend/__init__.py` performs a one-way star import of the chosen engine's package — `keras/src/backend/tensorflow`, `keras/src/backend/jax`, `keras/src/backend/torch`, `keras/src/backend/numpy`, or `keras/src/backend/openvino` — so that every name Keras uses internally, from `add` to `matmul`, is bound to one concrete implementation for the lifetime of the process. This is why the docs insist the backend must be configured before `import keras` and cannot be changed afterwards. Each backend package provides the same surface: an `ops/` subpackage implementing the NumPy-like API over core, math, nn, linalg, and image operations, plus backend-specific `layer.py`, `trainer.py`, `random.py`, and `export.py` modules. The shared contracts they all honor live in `keras/src/backend/common/` — `variables.py` defines the backend-agnostic `Variable`, `dtypes.py` normalizes dtype handling, and `stateless_scope.py` enables the pure-functional execution that JAX-style transformations require.

**KerasTensor and the symbolic graph.** The heart of the Functional API is `KerasTensor` in `keras/src/backend/common/keras_tensor.py` — a symbolic placeholder carrying only a shape, a dtype, and sparse/ragged flags, with no data behind it. When you call a layer on one, `Operation.__call__` in `keras/src/ops/operation.py` detects symbolic inputs and routes to `symbolic_call()`, which runs `compute_output_spec()` for static shape inference and then records the event. That record is the `Node` class in `keras/src/ops/node.py`: a `Node` captures the operation, its arguments (wrapped as `SymbolicArguments`), and its outputs, wiring itself into the operation's `_inbound_nodes` and `_outbound_nodes` lists, while every output tensor gets a `_keras_history` marker pointing back to the node that created it. Nodes form the vertices of the DAG and `KerasTensor`s are its edges, so walking the history metadata reconstructs the entire computation graph — no data ever flowed, yet the topology is complete.

**Functional models are Functions with state.** `keras/src/ops/function.py` defines `keras.Function`, a stateless container for such a DAG that can be re-applied to new inputs. `keras/src/models/functional.py` then defines `Functional(Function, Model)` — a Model that is literally a captured graph of layer calls, which is why you can slice a `backbone` or an `activations` sub-model out of intermediate tensors and have weights stay shared. `keras/src/models/sequential.py` implements `Sequential` as a special case of `Functional`. The piece that ties it together is the class definition in `keras/src/models/model.py`: `class Model(Trainer, base_trainer.Trainer, Layer)`, where the first `Trainer` in the MRO was chosen at import time from the backend package — `TensorFlowTrainer`, `JAXTrainer`, `TorchTrainer`, `NumpyTrainer`, or `OpenVINOTrainer`. A Keras model is simultaneously a Layer, a symbolic graph, and a backend-specific training engine.

**One trainer, five engines.** The backend-agnostic half of training lives in `keras/src/trainers/trainer.py`. Its `compile()` method accepts an optimizer, loss, metrics, `jit_compile`, and `steps_per_execution`, wrapping user-facing losses and metrics into `CompileLoss` and `CompileMetrics` from `compile_utils.py`. The iteration machinery is deliberately layered, as the module docstring in `keras/src/trainers/epoch_iterator.py` spells out: `DataAdapter`s normalize the raw data type, `EpochIterator` handles batching, shuffling, and steps, and the `Trainer` owns callbacks, validation, distribution, and epochs. The backend-specific half decides how a step executes. In `keras/src/backend/jax/trainer.py`, for example, the train step is wrapped in `jax.jit` (or `flax.nnx.jit` when NNX mode is enabled) and multiple batches are fused with `jax.lax.scan` when `steps_per_execution` is greater than one — while the PyTorch and TensorFlow trainers implement the same contract with their native compilation stacks. Callbacks from `keras/src/callbacks/callback.py` and optimizers from `keras/src/optimizers/optimizer.py` plug in at the same points on every engine.

**Serialization as a first-class subsystem.** Saving is not an afterthought bolted onto the model class; it is its own group of modules. `keras/src/saving/saving_api.py` provides the public `save_model`/`load_model` entry points, `keras/src/saving/saving_lib.py` writes and reads the `.keras` format as a zip archive (config JSON, metadata, and weights), and `keras/src/saving/serialization_lib.py` implements `serialize_keras_object`/`deserialize_keras_object` with a safe-mode scope that gates arbitrary code execution when loading untrusted files. Custom classes are resolved through the registry in `keras/src/saving/object_registration.py` — the `@keras.saving.register_keras_serializable` decorator — which is what makes a checkpoint containing your custom layer portable. A separate `keras/src/export/` package handles deployment-oriented exports: SavedModel, ONNX, LiteRT, OpenVINO, and TorchScript-style artifacts, each with a backend-appropriate implementation.

**The end-to-end flow.** Put it together and a call to `model.fit(x, y)` travels like this: `Model` (a backend-selected `Trainer`) validates that `compile()` was called, builds an `EpochIterator`, asks the data adapter registry to normalize `x` and `y` into batches, and then for each batch invokes the model's forward pass — where every layer's `call()` runs through `keras.ops` onto the bound backend — computes the compiled loss and metrics, applies the optimizer's updates to the shared `Variable` objects, fires `on_train_batch_end` callbacks, and finally jits or compiles whole multi-step chunks depending on `jit_compile` and `steps_per_execution`. When training is done, `model.save("model.keras")` walks the same `Node` graph it built at construction time, serializes every layer's config through the registry, and zips it with the weights — ready to be rebuilt on a completely different backend.

## Advantages

- **True backend portability.** One model definition runs on JAX, TensorFlow, PyTorch, and OpenVINO (inference-only), switched via `KERAS_BACKEND` — no code forks, no translation layers.
- **Portable custom components.** A custom layer or metric written with `keras.ops` works in native PyTorch `Module` hierarchies and JAX model functions, not just inside `model.fit()`.
- **Static shape inference before any data.** `KerasTensor` plus `compute_output_spec()` catches shape and dtype mistakes at graph-construction time, seconds after you write the bug.
- **A complete, coherent op library.** `keras.ops` covers NumPy-style core/math/linalg/image/nn operations uniformly across all backends, so you rarely need backend-native code.
- **Backend-appropriate performance.** Each engine gets a hand-tuned trainer — `jax.jit` and `lax.scan` on JAX, native compilation on TF and Torch — instead of one lowest-common-denominator loop.
- **A real serialization format.** The zip-based `.keras` format with safe-mode deserialization and a custom-object registry makes checkpoints portable and loading auditable.

## Benefits

- **No lock-in, lower risk.** Framework bets age badly; Keras 3 turns your engine choice into a reversible decision, which matters for multi-year ML systems.
- **Easier debugging.** With PyTorch or JAX eager execution under a familiar high-level API, you get readable stack traces instead of graph-mode mystery errors.
- **Scale without rewrites.** The `distribution_lib` present in each backend package means the same model scales from laptop to GPU/TPU clusters by configuration.
- **Keep your data pipeline.** `tf.data.Dataset`, PyTorch `DataLoader`, Grain, PyDataset, generators, and plain arrays all feed the same trainer through dedicated adapters.
- **A drop-in `tf.keras` successor.** Existing TensorFlow users can migrate by adopting the `.keras` save format, with Keras 2 still available as `tf-keras`.
- **A reference codebase for framework design.** The clean separation of symbolic graph, state, and execution makes `keras/src` one of the best places to study how modern ML frameworks are layered.

## Usage

Install Keras 3 and at least one backend package (`tensorflow`, `jax`, or `torch`; `openvino` supports inference only):

```
pip install keras --upgrade
```

Select the backend before importing — via environment variable:

```
export KERAS_BACKEND="jax"
```

or in a notebook, where it must be set before `import keras`:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
```

For GPU setups, the repository ships per-backend requirement files. As an example, the README's JAX CUDA environment via conda:

```shell
conda create -y -n keras-jax python=3.10
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
python pip_build.py --install
```

To work on the Keras source itself, install the development dependencies and the local build (note the project recommends WSL2 for Windows users):

```
pip install -r requirements.txt
python pip_build.py --install
```

and regenerate the public API layer with:

```
./shell/api_gen.sh
```

## Conclusion

Keras 3 is the rare framework whose source matches its pitch. The import-time dispatch in `keras/src/backend`, the symbolic `KerasTensor`/`Node` graph that powers the Functional API, the MRO-composed `Model` that grafts a backend-specific trainer onto a portable layer tree, and the registry-driven `.keras` serialization stack together demonstrate how to run one API over many engines without diluting any of them. Whether you want to choose your compute engine empirically, keep your ML code portable for the long haul, or simply learn how a mature framework is layered, the `keras` repository repays the read.

Links:

- [keras-team/keras on GitHub](https://github.com/keras-team/keras)
- [Keras documentation at keras.io](https://keras.io/)
