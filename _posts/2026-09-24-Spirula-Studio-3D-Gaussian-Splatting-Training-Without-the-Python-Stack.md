---
layout: post
title: "Spirula Studio: 3D Gaussian Splatting Training Without the Python Stack"
description: "Spirula Studio is a GPL-3.0 C++ project that trains 3D Gaussian Splatting models from raw photos or video in one self-contained binary - no Python, no PyTorch, no separate COLMAP. Cross-vendor Vulkan compute, quantized training for 10 million Gaussians in 8 GB of VRAM, native 360-degree camera support, built-in SfM, AI masking, and meshing. We tour the architecture."
date: 2026-09-24
header-img: "img/post-bg.jpg"
permalink: /Spirula-Studio-3D-Gaussian-Splatting-Training-Without-the-Python-Stack/
tags:
  - Graphics
  - C++
  - Vulkan
  - 3D
  - Open Source
author: "PyShine"
---
# Spirula Studio: 3D Gaussian Splatting Training Without the Python Stack

Turning a folder of photos into a 3D model has been possible for years, but the recipe still reads like a research lab checklist: a Python environment, PyTorch matched to the right CUDA toolkit, a separate COLMAP install for camera poses, half a dozen preprocessing scripts, and an NVIDIA card to run it all on. [Spirula Studio](https://github.com/harry7557558/spirula-studio) - 638 stars, GPL-3.0 licensed, written in C++ - collapses that checklist into one self-contained binary. Raw photos or video go in; a trained 3D Gaussian Splat model comes out, viewable in any modern splat viewer, and it can be converted into a textured mesh along the way. The trainer runs on NVIDIA, AMD, Intel, and Apple Silicon GPUs through Vulkan compute, fits up to 10 million full-spherical-harmonics Gaussians into 8 GB of VRAM via quantized training, and natively understands fisheye and 360-degree camera data with no undistortion step. Built-in structure-from-motion, AI masking, and frame extraction mean there is no external tooling to babysit either. Formerly known as spirulae-splat, the project is developed and maintained almost entirely by one person, and the discipline of its engineering shows in every directory.

![Architecture overview of the Spirula Studio repository showing the application surface, training engine, and GPU backends](/assets/img/diagrams/spirula-studio/spirula-overview-architecture.svg)

## Why You Need This

The reference implementations behind most splat tutorials were written to reproduce papers, not to ship. That shows up as friction you pay every single time: conda environments that break when a CUDA toolkit updates, a COLMAP binary that is its own installation project, preprocessing scripts that assume exactly one camera model, and VRAM requirements that quietly exclude half the GPUs in the world. Spirula Studio removes the entire class of problems by refusing the Python dependency outright - the same conviction we saw on the inference side in [colibri running MoE models in pure C](https://pyshine.com/colibri-Run-744B-MoE-Models-Pure-C/), here applied to training. Because it targets Vulkan compute, it is one of the few splat trainers that runs on AMD and Intel GPUs and on Apple Silicon Macs, not just CUDA hardware; when you are choosing between local AI machines, that vendor flexibility matters more than peak specs, a theme from our [DGX Spark versus Ryzen AI Halo comparison](https://pyshine.com/AMD-Ryzen-AI-Halo-vs-NVIDIA-DGX-Spark-Which-Local-AI-Machine-Should-You-Buy/). VRAM pressure is met with quantized training that fits 10 million full-SH Gaussians into 8 GB, which covers gaming laptops and mid-range desktop cards. And the workflow is genuinely end to end: a built-in SfM module replaces the separate COLMAP install, SAM-based AI masking replaces hand-run cleanup scripts, and frame extraction pulls training stills straight out of video files - including native GoPro MAX 360-degree footage that most tools would make you undistort first.

## How It Works

The repository is a single C++ codebase organized around one executable, two interchangeable GPU backends, and a set of learned subsystems that share one inference runtime.

![Detailed architecture diagram of the Spirula Studio source tree from the repository files](/assets/img/diagrams/spirula-studio/spirula-architecture.svg)

Everything hangs off `src/app/Main.cpp`, a subcommand table that dispatches to `train`, `sfm`, `mesh`, `sam`, `geometry`, and `encode` submains under `src/app/cli`. The same binary hosts a Dear ImGui desktop GUI under `src/app/gui` for interactive editing, and an HTTP web viewer server under `src/app/webviewer` that streams training progress to a browser while the model trains. Interface strings come from `src/i18n`, where every message exists in exactly 13 languages and the compiler itself rejects an incomplete translation - the message type holds all 13 strings or it does not build. The training heart is [TrainerCore](https://github.com/harry7557558/spirula-studio/blob/master/src/app/TrainerCore.cpp), the single driver loop shared by GUI and CLI, stepping through `src/engine` scene state, saving and resuming through `src/checkpoint`, and pulling cameras and images from `src/data`, whose parsers read COLMAP, Nerfstudio, and Metashape datasets. Every training option funnels through [TrainConfig.h](https://github.com/harry7557558/spirula-studio/blob/master/src/config/TrainConfig.h), an X-macro that acts as the single source of truth and expands into the CLI flags, GUI controls, and serialization code. The math lives in `src/kernels` - projection, rasterization, tiling, loss, optimization, and densification families, each with a launcher plus device-side kernel code - and every kernel call is dispatched through a generated backend API in `src/backend/api` to exactly one of two implementations. The Vulkan backend covers NVIDIA, AMD, Intel, and Apple Silicon through MoltenVK; the CUDA backend is the legacy NVIDIA-only option. Both compile from the same Slang shader sources in `src/shaders`, and a dedicated parity test suite under `src/backend/tests` numerically compares the two so results do not drift between vendors. What would normally require PyTorch runs instead on `src/nn`, a reusable GPU inference layer with its own Vulkan runtime: SAM 2 and SAM 3 segmentation for masking (checkpoints are fetched at first use, terms shown before download, never bundled), ALIKED with LightGlue and [LoMa](https://github.com/davnords/LoMa) feature matching for the SfM module, Metric3D v2 for depth and normals, and MoGe-2 point maps as the default geometry model. `src/sfm` turns raw images into a COLMAP-format sparse model with no external dependencies, `src/video` decodes footage on the GPU through the VK_KHR video extensions with an ffmpeg fallback, and `src/mesh` converts trained splats into meshes via Delaunay triangulation with a UV texture atlas. The build system ties it together: `cmake/` holds modules whose `sources.txt` file list feeds both backends, and four Python codegen tools under `tools/codegen` regenerate the backend forwarders and config expansions at development time, so committed generated code never goes stale by accident.

## Advantages

- **One binary, zero Python.** No PyTorch, no conda, no environment rot - a single executable carries training, SfM, masking, and meshing.
- **Cross-vendor by default.** Vulkan-first design runs on NVIDIA, AMD, Intel, and Apple Silicon; CUDA remains as a legacy option for NVIDIA-only setups.
- **Extreme VRAM efficiency.** Quantized training squeezes up to 10 million full-SH Gaussians into 8 GB of VRAM, opening large scenes to ordinary hardware.
- **Native 360-degree support.** Fisheye and equirectangular inputs load directly - no undistortion scripts before training.
- **End-to-end workflow.** Frame extraction, AI masking, SfM, splat training, and textured meshing live in the same tool with GUI and CLI access to all of it.
- **Engineered, not glued.** Generated backend forwarders, X-macro configuration, cross-backend parity tests, and compile-checked 13-language translations.

## Benefits

The practical benefit starts with what you never have to install: no Python environment to repair, no COLMAP to configure, no preprocessing scripts to maintain - the same self-contained philosophy behind [meshoptimizer's dependency-free rendering library](https://pyshine.com/meshoptimizer-Mesh-Optimization-GPU-Rendering/), taken to the whole training tool. Hardware you already own becomes eligible: an AMD gaming laptop, an Intel desktop, or an M-series Mac can all train production-quality splats, and Apple support arrived with a double-clickable app bundle and disk image. During training you are not flying blind - the binary serves a viewer on an HTTP port, so a quick SSH forward shows live progress from any browser, and the same viewer ships standalone on GitHub Pages for presenting finished splats. Output stays interoperable: splats open directly in tools like SuperSplat, meshes export with texture atlases for game engines, and a public gallery of user-trained scenes shows what the quality looks like outside marketing renders. Finally, the project is a case study in solo-maintained engineering discipline - generated code is committed and refreshed by codegen tools, both backends are held to numerical parity, and translations are verified at compile time - so the codebase reads like a masterclass in taming C++ complexity without a team.

## Usage

The quickest path is a prebuilt binary from the [Releases page](https://github.com/harry7557558/spirula-studio/releases) - downloads exist for Windows, Linux, and macOS, and unzipping gives you the `spirula` executable plus a double-clickable GUI. To build from source, install the Vulkan SDK and pick the recommended Vulkan backend:

```bat
cd spirula-studio\
.\build_develop.bat -DSS_BACKEND=vulkan -DSS_ENABLE_PATENTED=ON
```

That produces `build_vulkan\spirula.exe` on Windows; on Linux and macOS the equivalent is `bash build_develop.bash -DSS_BACKEND=vulkan -DSS_ENABLE_PATENTED=ON`, and macOS additionally wraps the binary in an app bundle and disk image with MoltenVK statically linked. The `-DSS_ENABLE_PATENTED=ON` flag enables GPU-side AVC/HEVC video decoding - about 15 times faster than the default ffmpeg fallback, though the README notes the bitstream parsers carry third-party patent exposure you accept by enabling it; omit the flag if that is a concern. The CUDA backend builds the same way with `-DSS_BACKEND=cuda` on Windows or Linux.

For training on a remote or cloud GPU, the CLI is the intended path: run `spirula --help` to see the subcommands, start `spirula train` on your dataset, and the command serves a viewer on an HTTP port you can forward over SSH to watch convergence live. The GUI downloads a SAM masking checkpoint on first use after showing its license terms, while on the command line you point `--model` at a file you downloaded yourself. Finished results view in any splat viewer, and the project's own [web viewer](https://harry7557558.github.io/spirula-studio/viewer/) plus galleries on [Megascapes](https://library.getmegascapes.com/) and [SuperSplat](https://superspl.at/explore/software/spirula-studio) show reference-quality scenes trained with it.

## Conclusion

Gaussian splatting is graduating from research code to shipped software, and Spirula Studio is the clearest example of what that transition looks like: one self-contained binary where the community stack needed five tools, two GPU backends kept honest by parity tests, and graphics work that finally runs on whatever GPU you happen to own. If you have a folder of photos, a 360-degree camera full of footage, or a remote GPU with no patience for environment setup, this is the shortest path from pixels to a 3D model - and the [source on GitHub](https://github.com/harry7557558/spirula-studio) is worth reading even if you never train a scene, because almost everything in it is done the disciplined way.
