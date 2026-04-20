---
layout: post
title: "LingBot-Map: Streaming 3D Reconstruction with the Geometric Context Transformer"
description: "Deep dive into LingBot-Map - the feed-forward 3D foundation model that reconstructs camera poses, depth maps, and point clouds at 20 FPS from streaming video using a novel Geometric Context Transformer with two-stream paged KV cache"
header-img: "img/posts/ai-coding-frameworks/ai-coding-frameworks.jpg"
permalink: /2026/04/20/lingbot-map-streaming-3d-reconstruction-geometric-context-transformer/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
tags: [AI, 3D Reconstruction, Computer Vision, Transformer, DINOv2, Point Cloud, Streaming]
author: "PyShine"
---

## Introduction

Reconstructing 3D structure from video sequences has long been dominated by iterative optimization methods like COLMAP and SfM, which require hundreds of bundle adjustment iterations and can take hours to process even moderate-length clips. Feed-forward models such as VGGT demonstrated that a single forward pass through a Vision Transformer could predict camera poses, depth maps, and point clouds from a handful of images -- but they were limited to short, fixed-length sequences processed in batch mode, making them impractical for real-time or streaming applications.

LingBot-Map, developed by the Robbyant team and introduced in the paper "Geometric Context Transformer for Streaming 3D Reconstruction" (arXiv:2604.14141), changes this equation entirely. It is a feed-forward 3D foundation model that performs streaming 3D reconstruction at approximately 20 FPS on 518x378 resolution images, handling sequences exceeding 10,000 frames without any iterative optimization. The key innovation is the Geometric Context Transformer (GCT), which architecturally unifies three critical mechanisms -- Anchor Context, Pose-Reference Window, and Trajectory Memory -- within a single streaming framework powered by a novel two-stream paged KV cache.

Building on the VGGT architecture, LingBot-Map replaces VGGT's bidirectional batch attention with a causal streaming design that processes frames one at a time while maintaining geometric consistency through carefully designed attention patterns and cache management. The result is a model that achieves state-of-the-art reconstruction quality on diverse benchmarks while running fast enough for real-time applications, all released under the permissive Apache 2.0 license.

## GCTStream Architecture

![LingBot-Map GCT Architecture](/assets/img/diagrams/lingbot-map/lingbot-map-gct-architecture.svg)

The GCTStream architecture implements the full Geometric Context Transformer pipeline in a streaming configuration. At the input stage, the `PatchEmbed` module uses a DINOv2 ViT-L/14 backbone with registers to extract patch-level features from each input image. The DINOv2 encoder produces 1024-dimensional embeddings for each 14x14 image patch, and these pretrained weights initialize both the frame blocks and global blocks of the aggregator, providing a strong visual foundation without training from scratch.

The core of the architecture is the `AggregatorStream`, which alternates between two types of attention blocks in a round-robin pattern. The 24 Frame Blocks perform per-frame self-attention with 2D Rotary Position Embeddings (RoPE), processing each frame's tokens independently -- there is no cross-frame communication in these blocks, which keeps the per-frame computation efficient and parallelizable. The 24 Global Blocks then perform cross-frame causal attention, where each frame attends to all previously cached frames through the FlashInfer paged KV cache or PyTorch's SDPA fallback. This is where the Geometric Context Transformer's three mechanisms operate: Anchor Context through scale frames that persist in the cache, Pose-Reference Window through the sliding window of recent frames, and Trajectory Memory through the append-only special token stream.

Three types of special tokens are prepended to each frame's patch tokens before entering the global blocks: a `camera_token` (1 token) that accumulates geometric information for pose prediction, `register_tokens` (4 tokens) that serve as additional learnable memory slots, and a `scale_token` (1 token) that activates bidirectional attention among the initial scale frames. The `patch_start_idx` is set to 6 (1 camera + 4 register + 1 scale), meaning the first 6 tokens of each frame are special and the remaining tokens are image patches. The `CameraCausalHead` extracts the camera token from the aggregator's output and iteratively refines the camera pose through 4 rounds of DiT-style adaptive modulation. Finally, two `DPTHead` modules -- one for depth (output_dim=2, activation="exp") and one for 3D points (output_dim=4, activation="inv_log") -- fuse multi-scale features from aggregator layers [4, 11, 17, 23] to produce dense depth maps and world-coordinate point clouds with associated confidence estimates.

## Two-Stream Paged KV Cache

![LingBot-Map Two-Stream Cache](/assets/img/diagrams/lingbot-map/lingbot-map-two-stream-cache.svg)

The two-stream paged KV cache is the key innovation that enables LingBot-Map's streaming capability at scale. Implemented in the `FlashInferKVCacheManager`, it partitions each frame's key-value pairs into two logically distinct streams that share a single physical page pool per transformer layer, with fundamentally different lifecycle policies.

The **Patch Stream** is recyclable: each frame's patch tokens (e.g., 972 tokens for 518x378 resolution at patch_size=14) occupy exactly one page in the cache. Patch pages are divided into two categories: scale pages and window pages. Scale pages (default 8, controlled by `kv_cache_scale_frames`) hold the initial bidirectional frames and are never evicted -- they serve as the Anchor Context that grounds the global coordinate system. Window pages (default 64, controlled by `kv_cache_sliding_window`) hold the most recent streaming frames and follow a FIFO eviction policy: when the window exceeds its capacity, the oldest window page is recycled back to the free list and its slot is reused for the next incoming frame. This implements the Pose-Reference Window mechanism, ensuring each frame always attends to a bounded, recent context.

The **Special Stream** is append-only and never recycled: every frame's 6 special tokens (camera + 4 registers + scale) are written sequentially into special pages. Since special tokens are only 6 per frame and pages hold `floor(page_size / 6)` frames worth of specials (e.g., 42 frames per page at page_size=256), the special stream grows very slowly. Critically, special tokens from every frame ever processed remain accessible, implementing the Trajectory Memory mechanism. This means the camera token from frame 1000 can still attend to the camera token from frame 1, providing long-range drift correction that would be impossible with a purely sliding-window approach.

The physical layout per block is `[max_num_pages, 2, page_size, H, D]`, where dim 1 separates K (index 0) and V (index 1), H is the number of attention heads (16), and D is the head dimension (64). Pages 0 through `max_patch_pages - 1` form the patch page pool (recyclable), and pages `max_patch_pages` through `max_num_pages - 1` form the special page pool (append-only). During attention computation, the visible page table is constructed in strict order: scale patch pages first, then window patch pages, then all special pages last. Placing special pages last means only the final page may be partially full, so FlashInfer's `paged_kv_last_page_len` naturally describes the partial special tail without requiring a custom attention mask.

## Two-Phase Inference

![LingBot-Map Two-Phase Inference](/assets/img/diagrams/lingbot-map/lingbot-map-two-phase-inference.svg)

LingBot-Map employs a two-phase inference strategy that bridges bidirectional and causal attention modes, enabling both accurate scale estimation and efficient streaming processing.

**Phase 1** processes the initial scale frames (default 8, configurable via `num_scale_frames`) together as a single block. These frames receive bidirectional attention among themselves through the scale token mechanism: the `scale_token` is set to its "first" variant for scale frames and "rest" variant for streaming frames, and when scale frames are processed with `num_frame_per_block` equal to the number of scale frames, they attend to each other in both directions. This bidirectional processing is critical for establishing a stable global coordinate system -- the scale frames jointly determine the scene's metric scale and provide a geometric anchor that all subsequent streaming frames reference. The scale frames' patch pages are permanently retained in the KV cache (never evicted), and their special tokens are appended to the special stream like any other frame.

**Phase 2** processes all remaining frames one at a time in strict causal order. Each streaming frame attends to the cached KV pairs from scale frames (Anchor Context), the sliding window of recent frames (Pose-Reference Window), and all accumulated special tokens (Trajectory Memory). The frame produces full predictions -- camera pose, depth map, and point cloud -- in a single forward pass, and its KV pairs are either stored in the cache (if it is a keyframe) or discarded after attention (if it is a non-keyframe). For sequences longer than 320 frames, the keyframe interval is automatically set to `ceil(num_frames / 320)` to keep the KV cache within the model's trained RoPE range, reducing memory by approximately `1/keyframe_interval`. For very long sequences exceeding 3000 frames, windowed inference splits the video into overlapping windows, each processed with a fresh KV cache, and then aligns and stitches the per-window predictions using pairwise similarity transforms estimated from overlapping depth maps and camera poses.

## Iterative Camera Pose Refinement

![LingBot-Map Iterative Camera Refinement](/assets/img/diagrams/lingbot-map/lingbot-map-iterative-camera.svg)

The `CameraCausalHead` refines camera pose estimates through an iterative loop inspired by the DiT (Diffusion Transformer) architecture. This refinement process runs 4 iterations by default (configurable via `camera_num_iterations`), progressively improving the 9-dimensional pose encoding that represents absolute translation (3D), quaternion rotation (4D), and field of view (2D).

At each iteration, the current pose estimate is embedded via a linear layer and fed into the `poseLN_modulation` network -- a SiLU activation followed by a linear projection that produces `3 * dim_in` outputs. These are split into shift, scale, and gate components for adaptive layer normalization. The camera token from the aggregator is normalized through an affine-free `LayerNorm`, then modulated as `gate * modulate(LN(pose_tokens), shift, scale) + pose_tokens`, where `modulate(x, shift, scale) = x * (1 + scale) + shift`. This DiT-style adaptive modulation allows the current pose estimate to condition how the camera token is processed, creating a feedback loop where better estimates lead to better feature extraction.

The modulated camera token then passes through a 4-layer `CameraBlock` trunk with causal attention and its own KV cache. Each `CameraBlock` attends to the cached camera tokens from all previous frames (via the special stream's trajectory memory) using 3D RoPE for temporal position encoding. The trunk output is normalized and passed through an MLP `pose_branch` that predicts a delta update. In the first iteration, the delta is used directly as the initial pose (starting from a learned `empty_pose_tokens` parameter). In subsequent iterations, the delta is added to the previous estimate: `pred_pose_enc = pred_pose_enc + pred_pose_enc_delta`. The previous prediction is detached before embedding to prevent gradient flow across iterations. After each iteration, the pose encoding is activated: translation uses linear activation, quaternion uses linear activation (normalized to unit norm), and field of view uses ReLU to ensure positive focal lengths. Setting `camera_num_iterations` to 1 skips three refinement passes and reduces the camera head's KV cache by 4x, trading a small amount of pose accuracy for significantly faster inference.

## Practical Features

LingBot-Map includes several practical features that make it suitable for real-world deployment scenarios:

**Sky Masking** uses an ONNX-based sky segmentation model (`skyseg.onnx`) to filter out sky points from the reconstructed point cloud, which significantly improves visualization quality for outdoor scenes. The model is automatically downloaded from HuggingFace on first use, and masks are cached in `<image_folder>_sky_masks/` for reuse. Install `onnxruntime` (CPU) or `onnxruntime-gpu` (GPU) to enable this feature.

**CPU Offloading** (`--offload_to_cpu`) moves per-frame predictions to CPU during inference, dramatically reducing GPU memory usage for long sequences. This is enabled by default; use `--no-offload_to_cpu` only if you have sufficient GPU memory.

**Keyframe Intervals** reduce KV cache memory by only storing every N-th frame's KV pairs. Non-keyframe frames still produce full predictions but their KV pairs are discarded after attention computation. The model auto-selects a keyframe interval when the sequence exceeds 320 frames (the trained RoPE range).

**Windowed Inference** (`--mode windowed`) handles sequences exceeding 3000 frames by splitting them into overlapping windows, each processed independently with a fresh KV cache. Cross-window alignment uses pairwise similarity transforms estimated from overlapping depth maps and camera poses, with median depth ratio scaling to correct metric scale differences.

**torch.compile Optimization** can be applied to the model for additional speedup using the `reduce-overhead` mode, leveraging CUDA graph capture for reduced kernel launch overhead.

**3D Visualization** is provided through the `viser` library, launching a browser-based interactive viewer at `http://localhost:8080` with configurable confidence thresholds, point sizes, and downsampling factors.

**GLB Export** enables saving the reconstructed 3D point cloud and camera trajectory as a GLB file for use in external 3D applications.

## Getting Started

### Installation

```bash
# Create conda environment
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map

# Install PyTorch (CUDA 12.8)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Install lingbot-map
pip install -e .

# Install FlashInfer (recommended for paged KV cache)
pip install flashinfer-python -i https://flashinfer.ai/whl/cu128/torch2.9/

# Install visualization dependencies (optional)
pip install -e ".[vis]"
```

### Running Inference on Images

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder /path/to/images/ --mask_sky
```

### Running Inference on Video

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --video_path video.mp4 --fps 10
```

### Keyframe Mode Configuration

For long sequences that exceed 320 frames, use keyframe mode to reduce memory:

```bash
# Manual keyframe interval
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --keyframe_interval 6

# Windowed inference for very long sequences (>3000 frames)
python demo.py --model_path /path/to/checkpoint.pt \
    --video_path video.mp4 --fps 10 \
    --mode windowed --window_size 128
```

### Visualization with viser

```bash
# Launch interactive 3D viewer with sky masking
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky \
    --port 8080 --conf_threshold 1.5 \
    --downsample_factor 10 --point_size 0.00001
```

### Faster Inference

Reduce camera refinement iterations for faster processing:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --camera_num_iterations 1
```

### Limited GPU Memory

```bash
# Offload predictions to CPU (default) and reduce scale frames
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ \
    --offload_to_cpu --num_scale_frames 2
```

## Conclusion

LingBot-Map represents a significant advance in feed-forward 3D reconstruction by demonstrating that streaming, real-time inference is achievable without sacrificing reconstruction quality. The Geometric Context Transformer's three mechanisms -- Anchor Context for scale grounding, Pose-Reference Window for local geometric consistency, and Trajectory Memory for long-range drift correction -- are elegantly unified through the two-stream paged KV cache design, where recyclable patch pages handle bounded recent context and append-only special pages accumulate trajectory information indefinitely.

The practical implications are substantial: a model that runs at 20 FPS on sequences exceeding 10,000 frames opens the door to real-time 3D reconstruction from drone footage, autonomous vehicle perception, AR/VR scene capture, and robotic navigation -- all without the computational burden of traditional SfM pipelines. The Apache 2.0 license and availability of multiple model variants (lingbot-map for balanced performance, lingbot-map-long for extended sequences, lingbot-map-stage1 for VGGT-compatible bidirectional inference) make this a versatile tool for both research and production applications. As the field moves toward foundation models for 3D vision, LingBot-Map's streaming architecture provides a compelling blueprint for how geometric understanding can scale from a few images to hours of video.