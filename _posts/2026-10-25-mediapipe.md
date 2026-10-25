---
layout: post
title: "MediaPipe: A Source Tour of Google's On-Device ML Framework"
description: "Walk through the MediaPipe source: the Tasks Python API, base options and running modes, landmarker and detector graphs, and how models ship to phones and desktops."
date: 2026-10-25
header-img: "img/post-bg.jpg"
permalink: /mediapipe/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/mediapipe/google-mediapipe-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: MediaPipe, on-device ML, hand landmarks, pose detection, Tasks API, Bazel
author: "PyShine"
---

If you have ever pointed a phone camera at your own hand and watched a wireframe skeleton snap to your knuckles in real time, you have almost certainly met MediaPipe. The framework started life inside Google as a C++ engine for streaming perception and grew into the company's default way to ship on-device ML features: face mesh, hand tracking, body pose, object detection, segmentation, and more recently on-device generative audio and text. The repository now lives under the google-ai-edge organization on GitHub, and the master branch is one of the larger codebases we have toured on this blog — thousands of build targets spanning C++, Java, Objective-C, JavaScript, and Python.

What makes MediaPipe worth reading is not any single algorithm. The landmark models are compact TFLite graphs you could also run with [TensorFlow Lite](/tensorflow/) or [ONNX Runtime](/onnxruntime/). The interesting part is everything around the model: a scheduler that moves timestamped packets between calculators, a task layer hiding it behind five friendly Python classes, and a ctypes bridge letting pure Python drive the C++ core. We read the code top-down, from the Tasks API to the framework layer, with every symbol linked to the master tree.

Two caveats up front: the PyPI package is a prebuilt wheel — building from source is a Bazel affair we treat honestly later — and you will meet two generations of Python API side by side, the classic `mp.solutions` modules and the newer Tasks API.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/mediapipe/google-mediapipe-overview-architecture.svg" alt="MediaPipe overview architecture: five Python vision task classes, BaseOptions, ctypes bindings, the C++ calculator graph runtime, and the legacy solutions layer" style="min-width:900px;"></div>

<p><em>Overview of the MediaPipe source: the Tasks API classes on the left are configured by BaseOptions and running modes, talk to the native core through ctypes bindings, and all of it lands on the CalculatorGraph runtime. The legacy <code>mp.solutions</code> layer sits apart on the right.</em></p>

Reading the overview from left to right:

- [HandLandmarker, L330](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py#L330) — the flagship task class: 21 hand keypoints plus handedness, with a symmetric trio of entry methods we will meet below.
- [BaseOptions, L59](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/base_options.py#L59) — the shared config every task accepts: where the model bytes come from and which delegate runs them.
- [VisionTaskRunningMode, L30](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/core/vision_task_running_mode.py#L30) — the IMAGE / VIDEO / LIVE_STREAM enum that decides which native function signature gets called.
- [Image, L200](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/core/image.py#L200) — the frame container handed to every task, with a zero-copy [numpy_view, L376](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/core/image.py#L376) for pixel access.
- [mediapipe_c_bindings, load_shared_library, L71](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/mediapipe_c_bindings.py#L71) — the ctypes bridge that loads the bundled C library and wires every entry point.
- [CalculatorGraph, L109](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/calculator_graph.h#L109) — the C++ heart: validates a graph config, schedules nodes, and moves packets between them.
- [FlowLimiterCalculator, L72](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/core/flow_limiter_calculator.cc#L72) — the unassuming calculator that keeps live streams real time by dropping stale frames.
- [SolutionBase, L186](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solution_base.py#L186) — the legacy wrapper behind everything you reach via `mp.solutions`.

## From Install to Your First Model

The repository README is no longer a documentation hub — it is a short stub that forwards you to the developer site at developers.google.com/edge/mediapipe. That is a small but telling change: Google treats the GitHub tree as the source of truth for *code*, while usage docs live elsewhere. The practical entry point remains the PyPI wheel:

```python
pip install mediapipe
```

Then the canonical Tasks API incantation, assembled entirely from symbols you can verify in the tree:

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image.create_from_file('image.jpg')
    result = landmarker.detect(mp_image)
```

Every piece of that snippet has a home in the source. `from mediapipe.tasks import python` works because [tasks/python/__init__.py, L25](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/__init__.py#L25) re-exports `BaseOptions`; the `vision.RunningMode` alias is defined at [vision/__init__.py, L78](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/__init__.py#L78). The context manager is not docs sugar — [__enter__, L593](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py#L593) really closes the underlying graph when the block exits. And `mp.Image.create_from_file` reaches a pybind factory registered in `mediapipe/python/pybind/image.cc`.

MediaPipe tasks consume `.task` bundles — TFLite model bytes plus metadata. The `model_asset_path` field points at a file on disk; the docstring at [BaseOptions, L59](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/base_options.py#L59) lists `model_asset_buffer` for raw bytes as the only other supported source. Our earlier tour of [TFLite Micro](/tflite-micro/) covers the flatbuffer format from the microcontroller side.

## The Tasks Layer: Five Classes, One Shape

The modern API's most striking property is its uniformity. [PoseLandmarker, L334](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/pose_landmarker.py#L334), [FaceLandmarker, L3083](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/face_landmarker.py#L3083), [ObjectDetector, L211](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/object_detector.py#L211) and [ImageSegmenter, L232](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/image_segmenter.py#L232) are all built to the same template as the hand landmarker: a `Result` dataclass, an `Options` dataclass, a `create_from_model_path` convenience, a `create_from_options` factory, and three detect-style methods.

The trio of methods is the API's real design decision. `detect` takes one image and blocks. `detect_for_video` takes an image plus a timestamp and blocks until that frame's inference completes. `detect_async` takes a timestamp and a callback, returns immediately, and delivers results later — it exists only in LIVE_STREAM mode. Pass `running_mode=LIVE_STREAM` without a `result_callback` and [create_from_options, L384](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py#L384) refuses to construct the task, validating the combination before any native call happens.

Results are plain dataclasses converted out of C structs. A [HandLandmarkerResult, L246](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py#L246) holds lists of [NormalizedLandmark, L67](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/components/containers/landmark.py#L67) — x, y, z, visibility, presence — plus world landmarks in meters and handedness categories built from [Category, L24](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/components/containers/category.py#L24). Nothing in a result holds a reference to native memory, which is precisely the point of the conversion layer we look at next.

## BaseOptions, Running Modes and the ctypes Bridge

Here is the part of the source that most MediaPipe users never see. In the current tree, the task classes do not import any pybind module. Instead, [hand_landmarker.py's import block](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py#L330) pulls in `mediapipe_c_bindings`, `base_options_c`, and the dispatch helpers, and every native interaction goes through `ctypes`.

The mechanics are worth pausing on. When you construct a task, your `BaseOptions` dataclass is flattened into `MpBaseOptionsC` — a `ctypes.Structure` mirroring a C struct, defined in `base_options_c.py` — because the C ABI cannot accept a Python object. The running mode enum travels as a plain integer; the `_CTYPE_VALUE_MAP` at the top of the running-mode module maps IMAGE/VIDEO/LIVE_STREAM to 1/2/3. Then the C library — a single shared object shipped inside the wheel — exposes functions like `hand_landmarker_detect` that take those structs.

Two helpers make this safe rather than terrifying. `SerialDispatcher` at [SerialDispatcher, L25](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/serial_dispatcher.py#L25) runs all C calls on a dedicated worker thread so that callbacks fire in order and the GIL never deadlocks against a native thread. `AsyncResultDispatcher` at [AsyncResultDispatcher, L66](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/async_result_dispatcher.py#L66) wraps live-stream callbacks so `detect_async` can hand you a Python-level future that resolves when the native side reports results. The `Delegate` enum at [Delegate, L78](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/core/base_options.py#L78) offers CPU and GPU, but GPU support ultimately depends on what the shipped C library was built with — one of the platform drifts we return to later.

## Under the Tasks: Graphs, Calculators and Packets

Below the C ABI sits the actual MediaPipe Framework, and it is a different style of system. A [CalculatorGraph, L109](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/calculator_graph.h#L109) consumes a [CalculatorGraphConfig, L237](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/calculator.proto#L237) — a protobuf listing calculator nodes, their input and output streams, and side packets — then validates it and schedules the nodes. Data moves between calculators as [Packet, L88](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/packet.h#L88) objects: immutable, timestamped, reference-counted payloads.

The calculator library is enormous, but one file deserves special mention for anyone doing real-time video. [FlowLimiterCalculator, L72](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/calculators/core/flow_limiter_calculator.cc#L72) implements back-pressure for live streams: with `max_in_flight: 1`, only one frame is being processed at a time, and every frame that arrives while the model is busy is dropped rather than queued. The comment block at the top of the file explains the trade-off directly — queueing video frames produces stale output and growing latency, while dropping them keeps results fresh. Every camera-facing demo graph you have seen implicitly relies on this.

The graphs behind each task are not hand-assembled at runtime. The build system composes them from `.proto` text graphs — the framework's [CalculatorGraphTemplate, L38](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/tool/calculator_graph_template.proto#L38) rule system expands one parametrized template into concrete configs — and ships them as binary resources inside the wheel. When a task initializes, the C library assembles the right graph and loads the model; only then does your `detect` call push a packet through.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/mediapipe/google-mediapipe-architecture.svg" alt="Detailed MediaPipe architecture: vision task classes, result containers, Python core dispatchers, ctypes C binding layer, vision core types, the C++ framework of calculator graphs and packets, and the legacy solutions stack" style="min-width:900px;"></div>

<p><em>The full map: task classes and their options, the container dataclasses returned as results, the serial and async dispatchers, the ctypes C-ABI layer, and the C++ framework underneath — with the legacy <code>mp.solutions</code> stack kept in its own group.</em></p>

## The Legacy solutions API Is Still There

Type `mp.solutions.hands` today and you get the old API — still in the tree, still shipped, but clearly from a different era. [Hands, L73](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/hands.py#L73) subclasses [SolutionBase, L186](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solution_base.py#L186), which loads a prebuilt binary graph and feeds it numpy frames through pybind bindings. Its `process(image)` method at [process, L300](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solution_base.py#L300) takes a BGR numpy array and returns a named tuple, a shape that predates the dataclass discipline of the Tasks layer.

The legacy modules bundle their own conveniences — [draw_landmarks, L119](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py#L119) draws annotated keypoints straight onto a frame — and they remain the quickest way to prototype. But they are in maintenance mode: the Tasks classes are where GPU delegates, new models, and cross-platform parity land first. If you start a new project on `mp.solutions`, you are building on the one generation of the API that Google has explicitly stopped evolving.

## Honest Limits

Reading the source also means reading its friction, and MediaPipe has more than its share.

Building from source is a Bazel project of unusual size. The wheel you `pip install` is prebuilt by Google; reproducing it locally means wiring Bazel, a C++ toolchain, and platform SDKs. Contributors routinely spend more time on the build than on the code change itself.

The legacy story is genuinely confusing. `mp.solutions` still exists in the tree, but the model zoo for Tasks is versioned and pinned — each landmarker has a specific `.task` release with its own accuracy/speed variants — and Google's docs interlink the two generations heavily. Tutorials on the internet mix APIs from both eras without saying so.

Cross-platform API drift is real. The Android Java API, the iOS Swift API, the Python Tasks API, and the web JavaScript API are separately maintained bindings over the same C++ core, and they disagree about naming, option availability, and delegate support. GPU inference, in particular, is configured differently on each platform, and a feature announced for one binding often arrives in the others late or not at all.

Finally, the Python wrapper is only as stable as its C ABI. The ctypes layer is efficient — no pybind11 shim for tasks — but it means a mismatch between the Python code and the bundled shared library fails at struct-packing time rather than import time. Pin your `mediapipe` version, exactly as you would pin a model.

MediaPipe remains the reference design for shipping perception ML to devices where every millisecond and megabyte counts. If you want to compare approaches, [ONNX Runtime](/onnxruntime/) covers the general cross-platform inference story, [TFLite Micro](/tflite-micro/) pushes the same flatbuffer ideas down to microcontrollers, and [ncnn](/ncnn/) is the community-built alternative with a similar mobile focus. The graph-plus-calculators architecture, though, is the part worth stealing regardless of which runtime you end up deploying.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "MediaPipe: A Source Tour of Google's On-Device ML Framework",
  "description": "Walk through the MediaPipe source: the Tasks Python API, base options and running modes, landmarker and detector graphs, and how models ship to phones and desktops.",
  "datePublished": "2026-10-25",
  "author": {"@type": "Person", "name": "PyShine"},
  "publisher": {"@type": "Organization", "name": "PyShine"},
  "keywords": "MediaPipe, on-device ML, hand landmarks, pose detection, Tasks API, Bazel"
}
</script>
