---
layout: post
title: "Brush: 3D Reconstruction for All - Open Source Gaussian Splatting in Rust"
description: "Learn how Brush brings 3D reconstruction to everyone with open-source Gaussian Splatting implemented in Rust. This guide covers installation, training models, and real-time 3D reconstruction from images."
date: 2026-05-18
header-img: "img/post-bg.jpg"
permalink: /Brush-3D-Reconstruction-For-All/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [3D Reconstruction, Rust, Computer Vision]
tags: [Brush, 3D reconstruction, Gaussian Splatting, Rust, computer vision, NeRF, neural rendering, open source, 3D graphics, photogrammetry]
keywords: "how to use Brush 3D reconstruction, Brush Gaussian Splatting tutorial, 3D reconstruction from images, Gaussian Splatting Rust implementation, Brush vs NeRF comparison, open source 3D reconstruction tool, Brush installation guide, real-time 3D reconstruction, neural radiance fields alternative, photogrammetry with Gaussian Splatting"
author: "PyShine"
---

## What is Brush?

Brush is an open-source 3D reconstruction engine built in Rust that implements [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) -- a technique that represents 3D scenes as collections of 3D Gaussian primitives that can be rendered in real time. With over 4,300 stars on GitHub and growing at 78 stars per day, Brush has rapidly become one of the most exciting projects in the neural rendering space.

What sets Brush apart from other Gaussian Splatting implementations is its commitment to universal accessibility. It runs on **macOS, Windows, Linux** with **NVIDIA, AMD, and Intel GPUs**, on **Android**, and even **in a web browser** via WebAssembly. This is possible because Brush is built on the [Burn](https://github.com/tracel-ai/burn) machine learning framework and uses WebGPU-compatible technology, eliminating the CUDA dependency that locks most ML tools to NVIDIA hardware.

> **Key Insight:** Brush produces simple, dependency-free binaries that run on nearly all devices without any setup. This is a fundamental shift from most machine learning tools that require complex CUDA installations and are limited to specific GPU vendors.

## How Gaussian Splatting Works

Traditional 3D reconstruction approaches like NeRF (Neural Radiance Fields) use neural networks to implicitly represent scenes. Gaussian Splatting takes a different approach: it represents a scene as a set of 3D Gaussian primitives, each with position, covariance (shape), color (via Spherical Harmonics), and opacity. These Gaussians are projected onto 2D image planes and alpha-blended front-to-back to produce photorealistic renderings.

The result is a representation that can be rendered at real-time frame rates (often 100+ FPS) while maintaining high visual quality. Brush implements this entire pipeline in Rust with custom GPU compute shaders, achieving performance that matches or exceeds the reference Python/CUDA implementations.

![Brush Architecture Diagram](/assets/img/diagrams/brush/brush-architecture.svg)

The architecture diagram above illustrates Brush's layered design. At the top, input data flows in through standard formats like COLMAP datasets, Nerfstudio transforms, PLY files, and image masks. The core Rust crates handle dataset loading (`brush-dataset`), Gaussian Splat rendering (`brush-render`), MCMC training (`brush-train`), backward pass autodiff (`brush-render-bwd`), and loss computation (`brush-loss`). Supporting crates provide GPU radix sorting, parallel prefix sums, virtual file systems, and COLMAP parsing. The GPU compute layer uses Burn with CubeCL for JIT-compiled compute kernels and WebGPU/wgpu for cross-platform GPU abstraction. The output layer provides desktop, CLI, web, and mobile applications.

## Key Features

| Feature | Description |
|---------|-------------|
| **Cross-Platform** | Runs on Windows, macOS, Linux, Android, and Web (WASM) |
| **Multi-GPU Vendor** | Supports NVIDIA, AMD, and Intel GPUs via WebGPU |
| **MCMC Training** | Auto-growing splats with scene exploration and pruning |
| **Zero CUDA Dependency** | Built on Burn + WebGPU, no CUDA installation needed |
| **Real-Time Viewer** | Interactive egui-based viewer during training |
| **Web Demo** | Train and view directly in Chrome/Edge browser |
| **CLI Interface** | Full command-line control with `--with-viewer` option |
| **Dynamic Splats** | Support for sequence playback and delta frame animations |
| **Loss Functions** | L1, SSIM (separable convolution), and LPIPS perceptual loss |
| **Scalable Data** | Handles datasets larger than RAM with streaming data loader |
| **PLY Export** | Standard and compressed PLY format support |
| **Rerun Integration** | Visualize training dynamics and memory usage |
| **NPM Package** | brush-js WASM module for JavaScript integration |
| **Single Binary** | No complex dependencies, just download and run |

![Brush Features and Pipeline](/assets/img/diagrams/brush/brush-features.svg)

The features diagram shows the complete reconstruction pipeline. Step 1 handles data input and loading from COLMAP, Nerfstudio, PLY, and mask formats. Step 2 runs the MCMC training loop with splat initialization, MCMC-like optimization with auto-growing and pruning, and the custom AdamScaled optimizer with exponential learning rate scheduling. Step 3 is the GPU rendering pipeline that projects Gaussians, performs tile-based rasterization with radix sorting and prefix sums, computes Spherical Harmonics for view-dependent color, and alpha-blends the result -- with a backward pass for autodiff gradients via Burn. Step 4 produces outputs including PLY export, real-time viewer, web deployment, and Rerun visualization. The feature cards below highlight Brush's cross-platform support, zero CUDA dependency, MCMC training, interactive viewer, web demo, loss functions, dynamic splats, scalable data handling, CLI, Android support, Rerun integration, and NPM package.

## Installation

### Prerequisites

Brush requires **Rust 1.88+** installed on your system. If you don't have Rust yet, install it using [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Building from Source

Clone the repository and build an optimized release binary:

```bash
git clone https://github.com/ArthurBrussee/brush.git
cd brush
cargo run --release
```

For a debug build (faster compilation, slower runtime):

```bash
cargo run
```

> **Important:** Always use `--release` for training. Debug builds are significantly slower and the CLI will warn you about this.

### Running the CLI

Brush provides a full command-line interface. View available commands:

```bash
brush --help
```

Train a model from a dataset:

```bash
brush train /path/to/dataset --with-viewer
```

The `--with-viewer` flag opens the interactive UI alongside training, which is invaluable for debugging and monitoring progress.

Export a trained model:

```bash
brush export /path/to/dataset --export-path ./output
```

Evaluate model quality:

```bash
brush eval /path/to/dataset
```

### Web Build

Brush can be compiled to WebAssembly for browser deployment. You need `wasm-pack` and Node.js:

```bash
# Install wasm-pack
cargo install wasm-pack

# Build and run the web demo
cd brush/apps/brush-app/web
npm install
npm run dev
```

The web version currently supports Chrome 134+ and Edge on Windows and macOS.

### Android Build

For Android deployment, set up the Android SDK and NDK:

```bash
# Add Android target
rustup target add aarch64-linux-android

# Install cargo-ndk
cargo install cargo-ndk

# Build native library (release mode for performance)
cargo ndk -t arm64-v8a -o crates/brush-app/app/src/main/jniLibs/ build --release

# Build and install the APK
cd crates/brush-app
./gradlew build
./gradlew installDebug
adb shell am start -n com.splats.app/.MainActivity
```

## Training with Brush

### Dataset Preparation

Brush accepts datasets in two primary formats:

1. **COLMAP format** -- The standard output from COLMAP photogrammetry software, including sparse/dense reconstructions with camera poses and point clouds.

2. **Nerfstudio format** -- A `transforms.json` file paired with image directories, commonly used in the NeRF ecosystem.

If your dataset includes an `init.ply` file, Brush will automatically use it as the initial point cloud for training. You can also include a folder of `masks` to ignore specific image regions, or use images with alpha channels to force the output splat to match transparency.

### Training Process

Brush v0.3 uses an MCMC-like training technique with its own variation that still grows splats automatically. This combines the best of both worlds: splats grow where they are needed, while also exploring the scene like in MCMC to improve quality.

```bash
# Basic training with viewer
brush train /path/to/colmap/dataset --with-viewer

# Training with maximum splat count
brush train /path/to/dataset --max-splats 1000000

# Export checkpoints every N steps
brush train /path/to/dataset --export-every 1000 --export-path ./checkpoints

# Save evaluation images to disk
brush train /path/to/dataset --eval-save-to-disk --export-path ./eval_results
```

During training, you can interact with the scene in the viewer, rotate the model with arrow keys, and compare the current rendering against input views as training progresses.

> **Amazing:** Brush can train on datasets larger than RAM. Only a configurable amount of data is cached in memory, while the rest is streamed by the data loader during training. Training also starts instantly -- no lengthy preprocessing step required.

### Loss Functions

Brush supports multiple loss functions that can be combined during training:

- **L1 Loss** -- Basic pixel-wise absolute difference
- **SSIM Loss** -- Structural Similarity Index with separable convolution for efficiency
- **LPIPS Loss** -- Learned Perceptual Image Patch Similarity for perceptual quality
- **Alpha Loss** -- Weighted loss for transparent image regions, controllable via `--alpha-loss-weight`

## Viewing and Exporting Results

### Interactive Viewer

Brush includes a full-featured viewer built with egui, supporting:

- **Orbit controls** -- Rotate around the scene center
- **FPS controls** -- First-person navigation through the scene
- **Flythrough controls** -- Smooth camera path animation
- **Panning** -- Translate the view
- **Arrow keys** -- Rotate model and move up/down
- **F key** -- Toggle fullscreen mode
- **FOV slider** -- Adjust field of view
- **Background color picker** -- Change the scene background
- **Splat scale slider** -- Adjust Gaussian splat sizes

### Loading Splats

The viewer can load `.ply` and `.compressed.ply` files. For web deployment, you can stream data from a URL:

```
https://arthurbrussee.github.io/brush-demo/?url=https://example.com/scene.ply&zen=true
```

The `?zen=true` parameter enables fullscreen mode for immersive viewing.

### Dynamic Splats

Brush supports animated splat sequences:

- **ZIP of PLY files** -- Load a sequence of splat files as an animation
- **Delta frame PLY** -- Custom format with incremental frames (used by [Cat4D](https://cat-4d.github.io/) and [Cap4D](https://felixtaubner.github.io/cap4d/))
- **Play/pause controls** -- Control animation playback in the viewer

## Architecture Deep Dive

Brush is organized as a Rust workspace with 20+ crates, each with a focused responsibility:

### Core Crates

| Crate | Purpose |
|-------|---------|
| `brush-render` | Forward Gaussian Splat rendering with GPU kernels |
| `brush-render-bwd` | Backward pass for autodiff gradient computation |
| `brush-train` | MCMC training loop with AdamScaled optimizer |
| `brush-dataset` | Scene loading, batching, and data management |
| `brush-loss` | Loss functions: L1, SSIM, LPIPS |
| `brush-sort` | GPU radix sort for tile-based rendering |
| `brush-prefix-sum` | Parallel prefix sum for GPU computation |
| `brush-vfs` | Virtual file system abstraction |
| `brush-serde` | Serialization layer for splat data |
| `brush-async` | Async runtime utilities |
| `brush-cube` | CubeCL compute abstractions |
| `colmap-reader` | COLMAP format parser |
| `lpips` | LPIPS perceptual similarity model |

### Application Crates

| Crate | Purpose |
|-------|---------|
| `brush-app` | Desktop GUI application (egui) |
| `brush-cli` | Command-line interface |
| `brush-js` | WebAssembly + NPM package for web |
| `brush-c` | C FFI bindings |

### GPU Compute Stack

Brush's GPU compute stack is what enables its cross-platform capability:

1. **Burn Framework** -- Provides the autodiff system and tensor operations
2. **CubeCL** -- JIT-compiled compute kernels written in Rust that compile to SPIR-V, WGSL, PTX, and more
3. **wgpu** -- Cross-platform GPU abstraction supporting Vulkan, Metal, DX12, and WebGPU
4. **Custom Forks** -- Brush maintains forks of wgpu and CubeCL with WebGPU subgroup operations needed for the backward rasterizer

> **Takeaway:** The use of Burn and CubeCL means Brush's compute kernels are written once in Rust and automatically compiled to the optimal GPU instruction set for each platform. This is why a single codebase can run on NVIDIA, AMD, Intel, and even in the browser.

## Brush vs Other 3D Reconstruction Tools

| Feature | Brush | gsplat (Python) | Original 3DGS | NeRF Studio |
|---------|-------|-----------------|----------------|-------------|
| Language | Rust | Python/CUDA | Python/CUDA | Python/CUDA |
| GPU Vendors | NVIDIA, AMD, Intel | NVIDIA only | NVIDIA only | NVIDIA only |
| Web Support | Yes (WASM) | No | No | No |
| Mobile Support | Android | No | No | No |
| CUDA Required | No | Yes | Yes | Yes |
| Training Speed | Faster than gsplat | Baseline | Baseline | Varies |
| Quality (PSNR) | Higher than gsplat | Baseline | Baseline | Varies |
| Binary Size | Small, standalone | Large (Python env) | Large (Python env) | Large (Python env) |
| Interactive Training | Yes | Limited | Limited | Yes |
| Dynamic Splats | Yes | No | No | No |

## Troubleshooting

### Common Issues

**Build fails with Rust version error:**

Brush requires Rust 1.88 or later. Update your toolchain:

```bash
rustup update stable
rustup default stable
```

**Web demo only works in Chrome/Edge:**

WebGPU is still an emerging standard. Firefox and Safari support is not yet available. Use Chrome 134+ or Edge for the web demo.

**Training is slow:**

Make sure you are running a release build:

```bash
cargo run --release
```

Debug builds are significantly slower. The CLI will display a warning if you are running in debug mode.

**Out of memory on large datasets:**

Brush can handle datasets larger than RAM by streaming data. If you still encounter memory issues, try reducing the batch size or limiting the maximum number of splats:

```bash
brush train /path/to/dataset --max-splats 500000
```

**Android build fails:**

Ensure the following environment variables are set:
- `ANDROID_NDK_HOME` pointing to your NDK installation
- `ANDROID_HOME` pointing to your Android SDK

Also verify you have added the correct target:

```bash
rustup target add aarch64-linux-android
```

**wgpu panics on startup:**

This may indicate your GPU driver does not support the required features. Try updating your GPU drivers to the latest version. On Linux, ensure you have the latest Mesa or proprietary drivers installed.

## Getting Started: Your First Reconstruction

Here is a complete walkthrough to create your first 3D reconstruction with Brush:

1. **Capture images** -- Take 50-200 photos of an object or scene from different angles. More coverage leads to better results.

2. **Run COLMAP** -- Generate camera poses and a sparse point cloud:

```bash
colmap automatic_reconstructor \
  --workspace_path ./workspace \
  --image_path ./images
```

3. **Train with Brush** -- Point Brush at your COLMAP output:

```bash
brush train ./workspace/sparse --with-viewer
```

4. **Monitor progress** -- Watch the training in the interactive viewer. You will see the scene gradually appear as splats are created and refined.

5. **Export the result** -- Once training converges (typically 7,000-30,000 iterations), export the PLY file:

```bash
brush export ./workspace/sparse --export-path ./output
```

6. **Share on the web** -- Upload the PLY file and share via URL:

```
https://arthurbrussee.github.io/brush-demo/?url=https://your-server.com/scene.ply&zen=true
```

> **Important:** Brush's MCMC-like training automatically grows splats where they are needed and prunes redundant ones. You do not need to manually tune the number of Gaussians -- the algorithm handles this for you. However, you can set a maximum splat count with `--max-splats` if you want to limit memory usage.

## Performance Benchmarks

Brush's rendering and training are generally faster than gsplat, the reference Python implementation. The project includes built-in benchmarks:

```bash
cargo bench
```

Key performance characteristics:

- **Clean builds** compile in approximately 1.5 minutes on modern hardware
- **Training speed** exceeds gsplat on comparable hardware
- **Memory usage** is optimized with streaming data loading for large datasets
- **Web training** is approaching feature parity with the desktop version

## Community and Resources

- **GitHub Repository:** [ArthurBrussee/brush](https://github.com/ArthurBrussee/brush)
- **Web Demo:** [arthurbrussee.github.io/brush-demo](https://arthurbrussee.github.io/brush-demo)
- **Discord Community:** [Join the Brush Discord](https://discord.gg/TbxJST2BbC)
- **License:** Apache-2.0

Brush is not an official Google product. It is a forked public version of the [google-research repository](https://github.com/google-research/google-research/tree/master/brush_splat), significantly extended with cross-platform support, MCMC training, and many other features.

## Conclusion

Brush represents a paradigm shift in 3D reconstruction accessibility. By implementing Gaussian Splatting entirely in Rust on top of the Burn ML framework and WebGPU, it eliminates the CUDA dependency that has limited similar tools to NVIDIA GPUs. The result is a tool that runs everywhere -- from high-end workstations to mobile phones to web browsers -- while delivering training quality that exceeds the reference implementation.

Whether you are a researcher exploring neural rendering, a developer building 3D applications, or a hobbyist creating photorealistic 3D models from photos, Brush provides a powerful yet accessible entry point into the world of Gaussian Splatting.