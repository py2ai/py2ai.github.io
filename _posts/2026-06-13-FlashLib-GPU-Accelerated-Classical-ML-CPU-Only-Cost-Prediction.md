---
layout: post
title: "FlashLib: GPU-Accelerated Classical ML with CPU-Only Cost Prediction"
date: 2026-06-13
categories: [ai, gpu, machine-learning, performance]
tags: [flashlib, gpu, triton, cutedsl, kmeans, pca, hdbscan, umap, tsne, roofline, cost-estimation, pareto, sklearn]
image: ai-coding-frameworks/ai-coding-frameworks
permalink: /FlashLib-GPU-Accelerated-Classical-ML-CPU-Only-Cost-Prediction/
---

Classical machine learning has a GPU problem. Scikit-learn is the lingua franca of classical ML -- every data scientist knows its API, every textbook teaches its classes, every production pipeline uses its estimators. But sklearn runs on CPU, and for large datasets, that means waiting minutes or hours for operations that could complete in seconds on a GPU.

NVIDIA's cuML accelerates some of these operations on GPU, but its coverage is limited, and there is no way to predict costs before you run the code. You cannot answer simple questions like "how long will KMeans take on 500K points with 64 features on an H200?" without actually running it.

FlashLib asks a different question: **what if you could get sklearn simplicity, GPU speed, and cost predictability in one library?**

Built by researchers at the Berkeley RISE Lab -- including Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica, and Song Han -- FlashLib is a GPU-accelerated library for classical ML that ships 16 high-level primitives, a dual-backend architecture (Triton + CuteDSL), and a CPU-only cost estimation API that predicts runtime, FLOPs, memory usage, and bottlenecks before you ever touch a GPU.

## What is FlashLib?

FlashLib is a GPU-accelerated classical machine learning library built on Triton and CuteDSL. It provides 16 high-level primitives -- from KMeans and PCA to UMAP and t-SNE -- all with sklearn-compatible APIs, while delivering significant speedups over cuML and sklearn.

Key characteristics:

- **16 sklearn-compatible GPU primitives**: Clustering, nearest neighbors, decomposition, manifold learning, regression, classification, and preprocessing
- **Dual-backend architecture**: Triton (portable, all CUDA GPUs) and CuteDSL (Hopper-specific, FA3-style fused TMA+WGMMA for up to 1.64x additional speedup)
- **CPU-only cost estimation API**: Predict runtime, FLOPs, memory, and bottlenecks without a GPU
- **Apache-2.0 licensed**, v0.2.0

![FlashLib Core Architecture](/assets/img/diagrams/flashlib/flashlib-architecture.svg)

The architecture diagram above shows FlashLib's layered design. At the top, 16 sklearn-compatible application classes provide the user-facing API. The `_route()` dispatch layer uses `HwProps` to auto-select the optimal backend. Two parallel backends -- Triton for portability and CuteDSL for Hopper-specific optimizations -- feed into low-level kernel primitives. Critically, the `flashlib.info` and `flashlib.diagnose` modules branch off as a CPU-only path that never touches the GPU layers.

## The 16 Primitives

FlashLib covers the broadest surface area of classical ML on GPU in a single library:

| Family | Primitives |
|--------|-----------|
| Clustering | `flash_kmeans`, `flash_dbscan`, `flash_hdbscan`, `flash_spectral_clustering` |
| Nearest Neighbors | `flash_knn`, `flash_ivf_flat` (IVF-Flat ANN) |
| Decomposition | `flash_pca`, `flash_truncated_svd` |
| Manifold Learning | `flash_umap`, `flash_tsne` |
| Regression | `flash_linear_regression`, `flash_ridge`, `flash_logistic_regression` |
| Classification | `flash_multinomial_nb`, `flash_random_forest` |
| Preprocessing | `flash_standard_scaler` |

Each primitive has both a `flash_*` function and an sklearn-style class, making migration from sklearn trivial:

```python
# sklearn
from sklearn.cluster import KMeans
km = KMeans(n_clusters=256, max_iter=20)
km.fit(X)

# FlashLib - same API, GPU-accelerated
from flashlib import KMeans
km = KMeans(n_clusters=256, max_iter=20)
km.fit(X)  # runs on GPU automatically
```

![FlashLib 16 Primitives - Family Coverage and Backend Support](/assets/img/diagrams/flashlib/flashlib-primitive-coverage.svg)

Every primitive ships with a Triton backend (portable across all CUDA GPUs). KMeans is the only primitive that also has a full CuteDSL backend, providing up to 1.64x additional speedup on H100/H200 for large-shape problems.

## Dual-Backend Architecture: Triton + CuteDSL

FlashLib's dual-backend architecture is its performance engine. Every primitive ships with two implementations:

**Triton backend**: Portable, works on all CUDA GPUs (Ampere and later). Triton kernels are written in Python and compiled to PTX, making them easy to maintain and extend. This is the default backend for most operations.

**CuteDSL backend**: Hopper-specific (H100/H200), uses FA3-style fused TMA (Tensor Memory Access) + WGMMA (Warp Group Matrix Multiply-Accumulate). This backend delivers up to 1.64x faster KMeans than Triton for large-shape problems, according to FlashLib's own benchmarks.

The `_route()` function handles backend selection automatically. It checks `HwProps` -- the GPU's SM architecture, batch size, metric, K, and D -- to pick the optimal backend. Users never need to specify the backend manually.

For KMeans, the CuteDSL FA3 sweet spot is:

- `sm_arch >= 90` (Hopper)
- `B == 1` (single batch)
- Euclidean metric
- `K >= 4096`
- `D >= 256` with `D % 16 == 0`

When all these conditions are met, FlashLib routes to the CuteDSL backend for maximum performance. Otherwise, it falls back to Triton.

```python
from flashlib import KMeans

# FlashLib auto-selects CuteDSL on H100/H200 for large shapes
km = KMeans(n_clusters=8192, max_iter=20)
km.fit(X)  # _route() picks CuteDSL if Hopper + large K/D

# On non-Hopper GPUs, Triton is used automatically
# No code changes needed
```

## The Info API: CPU-Only Cost Prediction

This is FlashLib's most innovative feature. No other ML library provides anything like it.

The `flashlib.info` module is a pure-stdlib cost estimation API that works without importing torch or triton. It runs on any machine -- laptops, CI servers, cloud functions -- and produces estimates in approximately 5 microseconds, fast enough for LLM agents to call in tool loops.

![FlashLib Info API - CPU-Only Cost Estimation Workflow](/assets/img/diagrams/flashlib/flashlib-info-api.svg)

### What It Predicts

The `info.estimate()` function returns an `Estimate` dataclass containing:

- **`runtime_ms`**: Predicted wall-clock time in milliseconds
- **`FLOPs`**: Total floating-point operations
- **`HBM bytes`**: High-bandwidth memory bytes transferred
- **`peak memory`**: Peak GPU memory usage
- **`bottleneck`**: Whether the operation is compute-bound, memory-bound, or latency-bound
- **`confidence`**: Which of the 4 confidence tiers the estimate falls into

### The 4 Confidence Tiers

| Tier | Color | Meaning |
|------|-------|---------|
| `calibrated` | Green | Within approximately 20% of wall-clock time, based on measured benchmarks |
| `measured` | Blue | Based on direct measurement data |
| `roofline` | Yellow | Computed from peak throughput x default efficiency |
| `heuristic` | Orange | Conservative fallback estimate |

### Code Examples

```python
import flashlib.info as info

# Estimate KMeans runtime on H200
est = info.estimate("kmeans", shape=(100_000, 64), params={"K": 256, "max_iters": 20}, device="H200")
print(est.summary_line())
# kmeans  4.42 ms  bound=memory  410 GB/s  ( 11% peak)  [calibrated]

# Pareto-optimal GEMM variants
info.pareto("gemm", shape=(8192, 8192, 8192))

# Compare flashlib vs cuML/sklearn
info.compare("kmeans", shape=(500_000, 64), params={"K": 64})
```

### The Subops Tree

Every `Estimate` carries a `subops` attribute that contains the full decomposition tree. For KMeans, this breaks down into distance GEMM, reduction, and assignment steps:

```python
est = info.estimate("kmeans", shape=(100_000, 64), params={"K": 256, "max_iters": 20}, device="H200")
est.print_tree()
# kmeans
#   distance_gemm
#   reduce
#   assign
```

This decomposition is critical for understanding where time goes and for tolerance-driven dispatch -- you can specify an acceptable error tolerance and have the system automatically select the cheapest variant that meets it.

### Why CPU-Only Estimates Matter

The info API enables a new class of applications:

- **LLM agents** can call `info.estimate()` in tool loops to plan computations before executing them
- **Cloud cost estimation** -- predict runtime and cost before provisioning a GPU instance
- **CI/CD gates** -- reject operations that would exceed time or memory budgets
- **Pre-flight checks** -- use `flashlib.diagnose` to verify GPU capabilities, then `info.estimate` to verify feasibility

All of this works without a GPU. You can install FlashLib on a laptop and get cost estimates for H200 workloads.

## Calibrated Roofline Model

FlashLib's cost estimation is grounded in a calibrated roofline model, not theoretical peak numbers.

### Hardware Specs

The model includes peak throughput data for H100, H200, and A100 across all precision levels:

| Metric | H100 | H200 | A100 |
|--------|------|------|------|
| FP64 TFLOPS | 34 | 34 | 9.7 |
| TF32 TFLOPS | 989 | 989 | 312 |
| FP16/BF16 TFLOPS | 1979 | 1979 | 624 |
| INT8 TOPS | 3958 | 3958 | 1248 |
| HBM Bandwidth | 3352 GB/s | 4807 GB/s | 2039 GB/s |

### Per-Op-Class Default Efficiency

Not all operations achieve peak throughput. FlashLib applies default efficiency factors by operation class:

| Op Class | Default Efficiency |
|----------|-------------------|
| GEMM | 0.40 |
| Elementwise | 0.80 |
| Reduction | 0.70 |
| Solver | 0.25 |
| KNN | 0.45 |

### Calibrated Overrides

When FlashLib has measured benchmark data for a specific op+device combination, it overrides the peak x efficiency default with the actual sustained throughput. This is why `calibrated` estimates are within approximately 20% of wall-clock time.

A launch overhead floor of 50 microseconds per kernel launch is also applied, ensuring that very small operations are not underestimated.

## Multi-Precision GEMM Pareto Frontier

FlashLib ships 12+ GEMM variants that trade accuracy for speed, each with a known `expected_residual`:

| Variant | Precision | Notes |
|---------|-----------|-------|
| `gemm_fp32` | FP32 | Full precision baseline |
| `gemm_tf32` | TF32 | 1.5x faster, approximately 7-bit mantissa loss |
| `gemm_3xtf32` | TF32x3 | Triple TF32 compensation |
| `gemm_bf16` | BF16 | 2x faster, approximately 3-bit mantissa loss |
| `gemm_3xbf16` | BF16x3 | Triple BF16 compensation |
| `gemm_fp16` | FP16 | Fastest half precision |
| `gemm_3xfp16` | FP16x3 | Triple FP16 compensation |
| `gemm_fp16_x9` | FP16x9 | 9-iteration compensation |
| `gemm_fp16_x3_kahan` | FP16x3 Kahan | Kahan compensated summation |
| `gemm_tf32_x6` | TF32x6 | 6-iteration TF32 compensation |
| `gemm_ozaki2_int8` | Ozaki INT8 | Ozaki algorithm with INT8 decomposition |

![FlashLib Multi-Precision GEMM Pareto Frontier](/assets/img/diagrams/flashlib/flashlib-gemm-pareto.svg)

The Pareto API filters to only the optimal variants -- those where no other variant is both faster and more accurate:

```python
import flashlib.info as info

# Get only Pareto-optimal GEMM variants
pareto_variants = info.pareto("gemm", shape=(8192, 8192, 8192))

# Tolerance-driven dispatch: specify accuracy, get fastest variant
est = info.estimate("gemm", shape=(8192, 8192, 8192), params={"tol": 1e-3}, device="H200")
```

With tolerance-driven dispatch, you specify your accuracy requirement and FlashLib automatically selects the fastest GEMM variant that satisfies it. You do not need to understand the full trade-off space -- `info.pareto()` returns only the optimal points.

## Hardware-Aware Routing

`flashlib._hw.HwProps` is a frozen dataclass that captures the GPU's capabilities:

```python
from flashlib._hw import HwProps

hw = HwProps.detect()  # auto-detect current GPU
# HwProps(device_tag='H200', sm_arch=90, sm_count=132,
#          l2_bytes=52848640, smem_per_sm_bytes=228352,
#          total_mem_bytes=141733920768,
#          is_hopper=True, is_blackwell=False,
#          is_ampere=False, is_cuda=True)
```

Each primitive's `_route()` function uses `HwProps` to pick the backend and variant. The routing logic is deterministic and transparent -- you can trace exactly why FlashLib chose a particular backend for your workload.

## Lazy Import System

FlashLib uses a lazy import system to keep startup fast and support CPU-only environments:

- `_LAZY_ATTRS` dict maps 155+ top-level names to `(module_path, attr_name)` pairs
- `__getattr__` lazy-loads on first access
- `flashlib.info` and `flashlib.diagnose` are eager (pure stdlib, no GPU)
- All GPU primitives load lazily

This means importing FlashLib on a CPU-only machine only loads the info and diagnose modules. You can use the cost estimation API without ever touching GPU code:

```python
# On a laptop with no GPU
import flashlib.info as info  # works - pure stdlib
est = info.estimate("kmeans", shape=(100_000, 64), params={"K": 256}, device="H200")

# GPU primitives load only when accessed
from flashlib import KMeans  # now torch/triton are imported
```

## Low-Level Primitives

Beyond the 16 high-level primitives, FlashLib exposes low-level kernels for advanced users:

- **Specialized GEMM**: `cov_gemm`, `gram_gemm`, `ab_gemm`
- **Eigendecomposition**: `eigh` with 5 variants (cusolver, qdwh, qdwh_ns, jacobi, halko)
- **Polar decomposition**: `polar` with 4 variants (qdwh_hybrid, express, express_warm, zolo)
- **Orthonormalization**: `cholqr2`, `split_basis`
- **Graph/distance kernels**: `pairwise_l2`, `connected_components`, `flash_mst`
- **Normalization**: `flash_rmsnorm`, `flash_layernorm`

These low-level primitives are the building blocks that the high-level applications compose. They are also available for direct use in custom pipelines.

## Why FlashLib Matters

FlashLib is the first library to combine three capabilities in one package:

1. **sklearn-compatible GPU ML**: 16 classical ML primitives with the same API you already know, running 10-100x faster on GPU
2. **CPU-only cost prediction**: The info API lets you predict runtime, memory, and bottlenecks before you run code -- without a GPU
3. **Dual-backend architecture**: A portable Triton baseline plus a Hopper-specific CuteDSL fast path, with automatic routing

The info API enables a new class of applications: LLM agents that plan before they compute, cloud cost estimators that predict before they provision, and CI gates that reject before they fail. This is the feature that sets FlashLib apart from every other GPU ML library.

The dual-backend architecture is a model for other GPU libraries: a portable baseline that works everywhere, plus an architecture-specific fast path that delivers extra performance on the latest hardware. The `_route()` function makes this transparent -- users never need to know which backend is running.

### Getting Started

```bash
pip install flashlib
```

```python
# GPU-accelerated classical ML
from flashlib import KMeans, PCA, HDBSCAN, UMAP, TSNE

# CPU-only cost estimation (works without GPU)
import flashlib.info as info
est = info.estimate("kmeans", shape=(100_000, 64), params={"K": 256}, device="H200")
print(est.summary_line())
```

FlashLib is open source under the Apache-2.0 license. The repository is at [github.com/FlashML-org/flashlib](https://github.com/FlashML-org/flashlib).

*Performance claims in this post are attributed to FlashLib's own benchmarks. The 1.64x CuteDSL speedup figure is from the FlashLib README.*