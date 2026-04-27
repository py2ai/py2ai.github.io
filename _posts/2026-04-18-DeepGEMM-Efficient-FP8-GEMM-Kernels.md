---
layout: post
title: "DeepGEMM: Clean and Efficient FP8 GEMM Kernels with Fine-Grained Scaling"
description: "A comprehensive guide to DeepGEMM, DeepSeek's high-performance tensor core kernel library for FP8 GEMM operations, MoE optimizations, and JIT compilation on NVIDIA SM90/SM100 GPUs."
date: 2026-04-18
header-img: "img/post-bg.jpg"
permalink: /DeepGEMM-Efficient-FP8-GEMM-Kernels/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - CUDA
  - Deep Learning
  - GPU Computing
  - FP8
author: "PyShine"
---

# DeepGEMM: Clean and Efficient FP8 GEMM Kernels with Fine-Grained Scaling

DeepGEMM is a unified, high-performance tensor core kernel library developed by DeepSeek that brings together the key computation primitives of modern large language models -- GEMMs (FP8, FP4, BF16), fused MoE with overlapped communication (Mega MoE), MQA scoring for the lightning indexer, HyperConnection (HC), and more -- into a single, cohesive CUDA codebase. All kernels are compiled at runtime via a lightweight Just-In-Time (JIT) module, requiring no CUDA compilation during installation.

Despite its lightweight design with only a limited number of core kernel functions, DeepGEMM's performance matches or exceeds expert-tuned libraries across various matrix shapes, achieving up to **1550 TFLOPS** on H800 GPUs. The library leverages concepts from CUTLASS and CuTe but avoids heavy reliance on their templates or algebras, making it a clean and accessible resource for learning NVIDIA GPU kernel optimization techniques.

![DeepGEMM Architecture](/assets/img/diagrams/deepgemm/deepgemm-architecture.svg)

### Understanding the DeepGEMM Architecture

The architecture diagram above illustrates the layered design of DeepGEMM, showing how the library organizes its functionality from the Python API layer down to the GPU hardware. Let us break down each component:

**Python API Layer**

The topmost layer provides the user-facing interface through the `deep_gemm` Python module. This is where developers interact with the library -- calling functions like `fp8_gemm_nt`, `m_grouped_fp8_gemm_nt_contiguous`, or `fp8_fp4_mega_moe`. The API is designed to be minimal and intuitive, exposing only the essential parameters while hiding the complexity of kernel selection, JIT compilation, and GPU resource management. Utility functions for configuring SM counts, tensor core utilization, and PDL (Programmatic Dependent Launch) are also exposed at this level.

**Kernel Types**

DeepGEMM organizes its kernels into several categories based on the computation pattern:

- **Dense GEMMs** -- Standard matrix multiplications (`D = C + A @ B`) supporting FP8, BF16, and TF32 data types. These handle the bulk of attention and projection computations in transformer models.
- **Grouped GEMMs** -- Specialized for Mixture-of-Experts (MoE) models where multiple experts share the same weight dimensions but process different numbers of tokens. Available in contiguous and masked layouts.
- **MQA Logits Kernels** -- Optimized multi-query attention scoring for the DeepSeek V3.2 lightning indexer, supporting both paged and non-paged variants.
- **Mega MoE** -- A fused kernel that overlaps EP dispatch, two FP8xFP4 linear layers, SwiGLU activation, and EP combine into a single mega-kernel, hiding NVLink communication latency behind tensor core computation.

**JIT Compilation Engine**

The Just-In-Time compilation engine is a critical architectural component. Rather than shipping pre-compiled CUDA binaries, DeepGEMM compiles kernels at runtime using a lightweight CPP JIT module. This approach offers several advantages:

1. **Zero-installation CUDA compilation** -- No need for a CUDA compiler during package installation; kernels are compiled on first use.
2. **Shape-specific optimization** -- The JIT engine can specialize kernels for the exact matrix dimensions being used, enabling optimizations that pre-compiled libraries cannot achieve.
3. **Architecture adaptation** -- Kernels are compiled for the specific GPU architecture (SM90 or SM100) present on the system, ensuring optimal instruction usage.
4. **Reduced binary size** -- Only the kernels actually needed are compiled and loaded, keeping memory footprint minimal.

The JIT engine supports both NVCC and NVRTC compilation backends, with NVRTC offering up to 10x faster compilation at a potential minor performance cost for some shapes.

**GPU Support**

DeepGEMM targets NVIDIA's latest architectures:

- **SM90 (Hopper)** -- H100, H800, H200 GPUs. Supports FP8 GEMM with 1D-1D and 1D-2D scaling modes, BF16 GEMM, and TF32 HC pre-norm GEMM.
- **SM100 (Blackwell)** -- B200, B300 GPUs. Adds FP8xFP4 mixed precision GEMM, Mega MoE with overlapped communication, and supports all memory layout variants (NT, TN, NN, TT).

This architecture enables DeepGEMM to serve as a one-stop kernel library for LLM training and inference, covering everything from dense projections to MoE routing and attention scoring.

---

## Why FP8 GEMM Matters

General Matrix Multiplication (GEMM) operations are the computational backbone of large language models. During a single forward pass of a model like DeepSeek-V3, billions of floating-point multiplications occur across attention projections, feed-forward networks, and MoE expert layers. The choice of numerical precision for these operations directly impacts both computational throughput and model quality.

**The FP8 Advantage**

Traditional training and inference pipelines use BF16 (Brain Float 16) or FP32 for GEMM operations. While BF16 provides excellent numerical stability, it only utilizes half of the available tensor core throughput compared to FP8. NVIDIA's Hopper and Blackwell architectures introduced native FP8 tensor core support, doubling the theoretical FLOPS per clock cycle compared to BF16.

However, FP8's reduced dynamic range (only 3 bits of exponent for E4M3, 2 bits for E5M2) creates a fundamental challenge: values outside the representable range are either saturated to the maximum value or underflow to zero. This is where **fine-grained scaling** becomes essential.

**Fine-Grained Scaling**

DeepGEMM implements per-block scaling factors that allow each small tile of the matrix to use its own optimal scale. This approach:

1. **Preserves accuracy** -- Small tiles can represent local value distributions more precisely than a single global scale.
2. **Reduces quantization error** -- The per-tile scaling minimizes the information loss from FP8 casting.
3. **Enables training** -- Fine-grained scaling makes FP8 viable for both forward and backward passes, not just inference.

DeepGEMM supports two scaling modes:

- **1D-1D scaling** -- Both the row and column scaling factors are 1D vectors, providing per-row and per-column scaling. This is the simpler mode and works well for most matrix shapes.
- **1D-2D scaling** -- Combines 1D row scaling with 2D sub-tile scaling for even finer granularity. This mode is particularly beneficial for matrices where value distributions vary significantly across both dimensions.

![DeepGEMM Kernel Pipeline](/assets/img/diagrams/deepgemm/deepgemm-kernel-pipeline.svg)

### Understanding the FP8 GEMM Kernel Pipeline

The kernel pipeline diagram above shows the complete data flow through an FP8 GEMM operation in DeepGEMM, from input tensors to the final output. Let us examine each stage in detail:

**Input Stage**

The pipeline begins with three primary inputs:

- **Matrix A (FP8 E4M3)** -- The left-hand side matrix, stored in row-major format for the NT layout. Each element is an 8-bit floating-point number in E4M3 format (4 bits exponent, 3 bits mantissa), providing a dynamic range of approximately [1/64, 448].
- **Matrix B (FP8 E4M3)** -- The right-hand side matrix, stored in column-major format (transposed). Uses the same E4M3 format as Matrix A.
- **Scaling Factors** -- Per-block scaling factors that enable fine-grained quantization. On SM90, these are stored as FP32 values. On SM100, they use the packed UE8M0 format, where 4 scaling values are packed into a single `torch.int` for memory efficiency.

The scaling factors are critical for maintaining numerical accuracy. Without them, the limited dynamic range of FP8 would cause significant information loss. With per-block scaling, each 128x128 (or similar) tile of the matrix has its own scale, allowing the FP8 values to utilize the full representable range.

**TMA Load Stage**

Tensor Memory Accelerator (TMA) is a hardware feature introduced in NVIDIA's Hopper architecture that offloads memory transfers from the streaming multiprocessor (SM). DeepGEMM uses TMA to asynchronously load matrix tiles from global memory into shared memory, overlapping data movement with computation. This is a key optimization that hides memory latency and keeps the tensor cores fed with data.

The TMA loads are configured based on the kernel's tile sizes, which are selected by the JIT engine based on the matrix dimensions. For small matrices, smaller tiles reduce waste; for large matrices, larger tiles maximize throughput.

**Tensor Core MMA Stage**

The Matrix Multiply-Accumulate (MMA) operation is the heart of the GEMM kernel. DeepGEMM uses NVIDIA's tensor cores to perform the actual matrix multiplication in FP8 precision. On SM90, this uses the `wgmma.mma_async` instruction (Warpgroup Matrix Multiply-Accumulate), which can perform a 16x16x16 or 16x8x32 FP8 MMA operation per clock cycle per tensor core.

The MMA stage operates on the scaled FP8 values, accumulating results in higher precision (FP32) to prevent numerical drift during the accumulation. This mixed-precision approach -- FP8 inputs with FP32 accumulation -- is essential for maintaining model quality.

**Epilogue Stage**

After the MMA completes, the epilogue stage handles:

1. **Scale application** -- The per-block scaling factors are applied to the accumulated results to produce the final FP32 values.
2. **Bias addition** -- If a bias term `C` is provided, it is added to the result.
3. **Output casting** -- The FP32 results are cast to the output dtype (typically BF16 for downstream consumption).
4. **Output store** -- The final results are written back to global memory, potentially using TMA stores for efficiency.

The entire pipeline is orchestrated to maximize overlap between memory transfers and computation, ensuring that the tensor cores are never starved for data.

---

## Key Features

![DeepGEMM Features](/assets/img/diagrams/deepgemm/deepgemm-features.svg)

### Understanding DeepGEMM's Key Features

The features diagram above organizes DeepGEMM's capabilities into three major categories. Let us explore each in depth:

**Precision Support**

DeepGEMM provides comprehensive precision coverage for different computational needs:

- **FP8 GEMM (E4M3)** -- The flagship feature, offering the highest throughput for LLM workloads. FP8 E4M3 provides 4 exponent bits and 3 mantissa bits, giving a dynamic range suitable for weights and activations. DeepGEMM supports both 1D-1D and 1D-2D fine-grained scaling modes, allowing users to choose the right trade-off between accuracy and performance for their specific model.

- **FP8xFP4 Mixed Precision** -- A cutting-edge feature for SM100 (Blackwell) GPUs that stores weights in FP4 (2 exponent bits, 1 mantissa bit) while keeping activations in FP8. This halves the memory bandwidth for weight loading, which is critical for MoE models where the weight-to-computation ratio is high. The FP4 weights use UE8M0 scaling factors packed into 32-bit integers for memory efficiency.

- **BF16 GEMM** -- Traditional Brain Float 16 support for workloads that require higher numerical precision or for backward compatibility with existing models. BF16 provides 8 exponent bits and 7 mantissa bits, matching FP32's dynamic range with reduced precision.

- **TF32 HC Pre-Norm GEMM** -- TensorFloat-32 with HyperConnection pre-normalization. This specialized kernel supports the HyperConnection architecture used in DeepSeek models, where pre-normalization is fused into the GEMM operation to reduce kernel launch overhead and improve numerical stability.

**MoE Optimizations**

Mixture-of-Experts models present unique computational challenges that DeepGEMM addresses with specialized kernels:

- **Mega MoE** -- The most advanced MoE optimization in DeepGEMM. It fuses the entire MoE computation -- EP dispatch, two FP8xFP4 linear layers, SwiGLU activation, and EP combine -- into a single mega-kernel. By overlapping NVLink communication with tensor core computation, Mega MoE can hide most of the communication latency that would otherwise bottleneck MoE inference. This requires multi-process launch with symmetric memory buffers.

- **MQA Logits** -- Multi-Query Attention scoring kernels for the DeepSeek V3.2 lightning indexer. These kernels compute token-to-token logits using weighted ReLU attention, supporting both paged (for decoding) and non-paged (for prefilling) variants. The FP8 implementation reduces memory bandwidth requirements while maintaining scoring accuracy.

- **Grouped GEMM** -- Specialized for MoE forward and backward passes where multiple experts share the same weight dimensions. DeepGEMM supports two layouts:
  - **Contiguous layout** -- Tokens from each expert are concatenated into a single tensor, suitable for training forward passes and inference prefilling.
  - **Masked layout** -- A mask tensor specifies which portions of the output are valid, designed for inference decoding with CUDA graphs where the CPU does not know the exact token distribution.

**Engineering Quality**

DeepGEMM's engineering philosophy prioritizes simplicity and accessibility:

- **~300 Lines Core** -- The core GEMM kernel is remarkably concise at approximately 300 lines of CUDA code. This is achieved by leveraging concepts from CUTLASS and CuTe without the template metaprogramming overhead that makes those libraries complex. The result is a codebase that is easy to understand, modify, and extend.

- **JIT Compilation** -- All kernels are compiled at runtime, eliminating the need for CUDA compilation during installation. The JIT engine uses a lightweight CPP module with low CPU overhead, supporting both NVCC and NVRTC backends. NVRTC compilation offers up to 10x speedup for kernel compilation.

- **Programmatic Dependent Launch (PDL)** -- A Hopper-architecture feature that allows the CPU to launch dependent kernels without waiting for the previous kernel to complete. DeepGEMM supports PDL for reducing kernel launch overhead in pipelined workloads.

- **Configurable SM Usage** -- The `set_num_sms` and `set_tc_util` functions allow fine-grained control over GPU resource allocation, enabling users to reserve SMs for other tasks or optimize for specific workload patterns.

---

## Performance Benchmarks

DeepGEMM achieves exceptional performance across a range of matrix shapes, matching or exceeding expert-tuned libraries:

| Metric | Value |
|--------|-------|
| Peak FP8 Performance (H800) | Up to 1550 TFLOPS |
| Core Kernel Size | ~300 lines of CUDA |
| JIT Compilation Speedup (NVRTC) | Up to 10x faster |
| Supported Architectures | SM90 (Hopper), SM100 (Blackwell) |
| Memory Layouts (SM100) | NT, TN, NN, TT |

The 1550 TFLOPS figure on H800 represents near-theoretical-peak utilization of the FP8 tensor cores. This level of performance is achieved through:

1. **TMA-based data movement** -- Offloading memory transfers to dedicated hardware.
2. **Fine-grained scaling** -- Minimizing quantization error while maintaining FP8 throughput.
3. **Warp-specialized kernel design** -- Separating data movement and computation into distinct warp groups.
4. **Software pipelining** -- Overlapping multiple stages of the GEMM pipeline to hide latency.

For MoE workloads, the Mega MoE kernel provides significant speedups by fusing multiple operations and overlapping communication with computation. The FP8xFP4 mixed precision mode further reduces memory bandwidth requirements, which is often the bottleneck in MoE inference.

---

## Getting Started

### Requirements

Before installing DeepGEMM, ensure your system meets these requirements:

- **GPU**: NVIDIA SM90 (H100, H800, H200) or SM100 (B200, B300) architecture
- **Python**: 3.8 or higher
- **Compilers**: C++20 support required
- **CUDA Toolkit**: 12.3+ for SM90 (12.9+ recommended), 12.9+ for SM100
- **PyTorch**: 2.1 or higher
- **CUTLASS**: 4.0 or higher (cloned via Git submodule)
- **{fmt} library** (cloned via Git submodule)

### Installation

For development with full submodule support:

```bash
# Clone with submodules
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Link essential includes and build the CPP JIT module
cat develop.sh
./develop.sh
```

For production installation:

```bash
# Install via the install script
cat install.sh
./install.sh
```

After installation, simply import `deep_gemm` in your Python project:

```python
import deep_gemm
```

### Basic Usage

#### Dense FP8 GEMM

The most common operation is a standard dense FP8 GEMM:

```python
import torch
import deep_gemm

# Prepare FP8 inputs with scaling factors
# Matrix A: [M, K], row-major (NT layout)
# Matrix B: [N, K], column-major (NT layout, transposed)
# Scaling factors for A and B
a_fp8 = ...  # FP8 E4M3 tensor, shape [M, K]
b_fp8 = ...  # FP8 E4M3 tensor, shape [N, K]
a_sf = ...   # Scaling factors for A
b_sf = ...   # Scaling factors for B

# Perform GEMM: D = C + A @ B.T
# Output is BF16 by default
result = deep_gemm.fp8_gemm_nt(a_fp8, a_sf, b_fp8, b_sf)
```

#### Grouped GEMM for MoE

For Mixture-of-Experts models, use the grouped GEMM interface:

```python
# Contiguous layout: tokens from each expert are concatenated
# m_indices specifies which rows belong to which expert
result = deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
    a_fp8, a_sf, b_fp8, b_sf, m_indices
)
```

#### Mega MoE (Fused MoE Kernel)

The Mega MoE kernel fuses the entire MoE computation:

```python
# Allocate symmetric memory buffer
# Requires PyTorch >= 2.9
buffer = deep_gemm.get_symm_buffer_for_mega_moe(
    group, num_experts, num_max_tokens_per_rank,
    num_topk, hidden, intermediate_hidden
)

# Transform weights (FP4 with UE8M0 SF) into required layout
transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe(
    l1_weights, l2_weights
)

# Copy inputs into the buffer before each call
buffer.x[:num_tokens].copy_(x_fp8)
buffer.x_sf[:num_tokens].copy_(x_sf)
buffer.topk_idx[:num_tokens].copy_(topk_idx)
buffer.topk_weights[:num_tokens].copy_(topk_weights)

# Run the fused mega MoE kernel
y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_fp4_mega_moe(y, transformed_l1, transformed_l2, buffer)
```

#### MQA Logits for Lightning Indexer

For DeepSeek V3.2's lightning indexer scoring:

```python
# Non-paged version (for prefilling)
logits = deep_gemm.fp8_mqa_logits(
    q,           # [seq_len, num_heads, head_dim] E4M3
    kv,          # [seq_len_kv, head_dim] E4M3 with float SF
    weights,     # [seq_len, num_heads] float
    cu_seq_len_k_start,  # [seq_len] int
    cu_seq_len_k_end,    # [seq_len] int
    clean_logits=True     # Clean unfilled logits to -inf
)
```

### Configuration Utilities

DeepGEMM provides several utility functions for fine-tuning kernel behavior:

```python
# Control SM usage (reserve SMs for other tasks)
deep_gemm.set_num_sms(100)  # Use at most 100 SMs
current_sms = deep_gemm.get_num_sms()

# Set tensor core utilization ratio
deep_gemm.set_tc_util(0.8)  # Target 80% TC utilization

# Enable Programmatic Dependent Launch
deep_gemm.set_pdl(True)

# Configure M/K alignment for contiguous grouped GEMM
deep_gemm.set_mk_alignment_for_contiguous_layout(128)

# Transform scaling factors into required layout
transformed_sf = deep_gemm.transform_sf_into_required_layout(sf_tensor)
```

### Environment Variables

DeepGEMM supports several environment variables for debugging and configuration:

```bash
# Print JIT debugging information
export DG_JIT_DEBUG=1

# Print selected configs for each shape
export DG_PRINT_CONFIGS=1

# Use NVRTC for faster compilation (up to 10x)
export DG_JIT_USE_NVRTC=1

# Set custom JIT cache directory
export DG_JIT_CACHE_DIR="/path/to/cache"

# Dump PTX and SASS for analysis
export DG_JIT_DUMP_ASM=1

# Skip CUDA extension build during installation
export DG_SKIP_CUDA_BUILD=1
```

---

## Integration Workflow

![DeepGEMM Integration Workflow](/assets/img/diagrams/deepgemm/deepgemm-integration-workflow.svg)

### Understanding the Integration Workflow

The integration workflow diagram above illustrates the seven-step process for integrating DeepGEMM into an existing LLM training or inference pipeline. Let us walk through each step:

**Step 1: Install DeepGEMM**

The first step is straightforward -- clone the repository with submodules and run the install script. The JIT compilation model means there is no need for a CUDA compiler during installation. Kernels are compiled on first use, making the installation process fast and simple.

```bash
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM && ./install.sh
```

**Step 2: Prepare FP8 Tensors**

Before calling DeepGEMM kernels, you must cast your model weights and activations to FP8 E4M3 format and compute the corresponding scaling factors. DeepGEMM requires TMA-aligned tensors with specific layout constraints:

- For SM90: Scaling factors in FP32 format, with TMA-aligned and transposed layout for the LHS.
- For SM100: Scaling factors in packed UE8M0 format (4 values per `torch.int`).

DeepGEMM provides utility functions like `get_mn_major_tma_aligned_tensor` and `transform_sf_into_required_layout` to handle the alignment and layout transformations.

**Step 3: Configure Kernel Parameters**

Set up the kernel configuration based on your workload:

- Choose the appropriate GEMM function (dense, grouped contiguous, grouped masked, Mega MoE).
- Configure SM count and tensor core utilization.
- Enable or disable PDL based on your pipeline structure.
- Set M/K alignment for grouped GEMM operations.

**Step 4: Call GEMM Kernels**

Invoke the appropriate DeepGEMM function with your prepared inputs. The JIT engine will compile the kernel on first use and cache it for subsequent calls with the same shape. This means the first call has a compilation overhead, but all subsequent calls with matching dimensions execute at full speed.

**Step 5: Handle Output**

DeepGEMM outputs are typically in BF16 format, ready for consumption by downstream operations. For grouped GEMMs, the output layout matches the input layout (contiguous or masked). For Mega MoE, the output is a BF16 tensor of shape `[num_tokens, hidden]`.

**Step 6: Integrate with Training/Inference**

DeepGEMM kernels are designed to be drop-in replacements for existing GEMM operations in your training or inference framework. The key integration points are:

- **Attention projections** (Q, K, V, O) -- Replace BF16 GEMMs with FP8 GEMMs.
- **FFN/MLP layers** -- Use grouped GEMMs for MoE expert projections.
- **MoE routing** -- Use MQA logits kernels for the lightning indexer.
- **Full MoE layers** -- Use Mega MoE for end-to-end fused computation.

**Step 7: Optimize and Benchmark**

Use DeepGEMM's built-in benchmarking tools to measure performance:

```python
from deep_gemm.testing import bench

# Benchmark a specific shape
bench.benchmark_fp8_gemm(M=4096, N=4096, K=4096)
```

Monitor performance with environment variables like `DG_PRINT_CONFIGS` and `DG_JIT_DEBUG` to ensure kernels are being compiled and executed optimally. Adjust SM count, TC utilization, and PDL settings based on your specific workload characteristics.

---

## Architecture Comparison: SM90 vs SM100

DeepGEMM supports both NVIDIA Hopper (SM90) and Blackwell (SM100) architectures, with feature differences tailored to each platform:

| Feature | SM90 (Hopper) | SM100 (Blackwell) |
|---------|---------------|-------------------|
| FP8 GEMM | Yes (1D-1D, 1D-2D) | Yes (1D-1D) |
| FP8xFP4 Mixed Precision | No | Yes |
| BF16 GEMM | Yes | Yes |
| TF32 HC Pre-Norm | Yes | Yes |
| MQA Logits | Yes (FP8) | Yes (FP8, FP8xFP4) |
| Mega MoE | No | Yes |
| Memory Layouts | NT only | NT, TN, NN, TT |
| Scaling Factor Format | FP32 | Packed UE8M0 |
| PDL Support | Yes | Yes |

The SM100 (Blackwell) support adds several important capabilities:

- **FP8xFP4 mixed precision** enables storing weights in FP4 format, halving memory bandwidth requirements for weight-heavy operations like MoE expert projections.
- **Mega MoE** fuses the entire MoE computation into a single kernel, overlapping NVLink communication with tensor core computation.
- **All memory layouts** (NT, TN, NN, TT) are supported, removing the need for explicit transposition in many cases.

---

## Troubleshooting

### Common Issues

**JIT Compilation Errors**

If you encounter JIT compilation errors, check the following:

```bash
# Enable debug output
export DG_JIT_DEBUG=1

# Verify CUDA toolkit version
nvcc --version  # Should be 12.3+ for SM90, 12.9+ for SM100

# Check compiler path
export DG_JIT_NVCC_COMPILER="/usr/local/cuda/bin/nvcc"
```

**TMA Alignment Issues**

DeepGEMM requires TMA-aligned tensors. If you see alignment errors:

```python
# Use DeepGEMM's alignment utilities
aligned_tensor = deep_gemm.get_mn_major_tma_aligned_tensor(your_tensor)
aligned_sf = deep_gemm.transform_sf_into_required_layout(your_sf)

# Check required alignment size
alignment = deep_gemm.get_tma_aligned_size()
```

**Performance Not Meeting Expectations**

```python
# Check and configure SM usage
deep_gemm.set_num_sms(128)  # Use all SMs on H100

# Enable PDL for pipelined workloads
deep_gemm.set_pdl(True)

# Try NVRTC for faster compilation (may have minor perf impact)
# Set DG_JIT_USE_NVRTC=1 before importing deep_gemm
```

**Grouped GEMM Alignment Errors**

For grouped GEMM operations, ensure your data is properly aligned:

```python
# Get the required alignment for contiguous layout
alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()

# Ensure each expert segment is aligned to the block size
# If not, pad your data accordingly
```

---

## Conclusion

DeepGEMM represents a significant contribution to the open-source GPU kernel ecosystem. By combining high performance (up to 1550 TFLOPS on H800), clean code (~300 lines core), and comprehensive feature coverage (FP8, FP4, BF16, MoE, MQA), it provides a production-ready foundation for LLM training and inference on NVIDIA's latest architectures.

The library's JIT compilation model eliminates installation friction, while its fine-grained scaling approach makes FP8 viable for both training and inference. The specialized MoE kernels -- particularly Mega MoE with its fused computation and overlapped communication -- address the unique performance challenges of Mixture-of-Experts models.

Whether you are building the next generation of LLMs or optimizing existing model deployments, DeepGEMM offers the performance, flexibility, and simplicity needed to push the boundaries of GPU computing.

**Links:**

- GitHub: [https://github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- License: MIT
- Citation: See repository for BibTeX entry
