---
layout: post
title: "DeepEP: Efficient Expert-Parallel Communication Library for MoE"
date: 2026-04-25 09:50:00 +0800
categories: [AI, Deep Learning, GPU]
tags: [deepseek, moe, expert-parallelism, gpu, cuda, rdma, nvlink, distributed-training, inference]
keywords: "DeepEP, Mixture of Experts, MoE communication, expert parallelism, GPU all-to-all, DeepSeek, NVLink, RDMA, low-latency kernel, FP8 dispatch"
description: "DeepEP is a high-performance communication library for Mixture-of-Experts (MoE) and expert parallelism, providing high-throughput and low-latency all-to-all GPU kernels with FP8 support."
author: "PyShine"
---

## Introduction

**DeepEP** is a communication library purpose-built for **Mixture-of-Experts (MoE)** and **expert parallelism (EP)** from [DeepSeek](https://github.com/deepseek-ai). It delivers high-throughput and low-latency all-to-all GPU kernels—commonly known as **MoE dispatch and combine**—with support for low-precision operations including **FP8**. Designed to align with the group-limited gating algorithm in the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) paper, DeepEP is a critical building block for scaling MoE models across multiple GPUs and nodes.

---

## What is DeepEP?

In MoE architectures, tokens must be routed (dispatched) to the appropriate expert GPUs, and the computed results must be gathered back (combined). This all-to-all communication pattern is the bottleneck in large-scale MoE training and inference. DeepEP solves this with two classes of optimized kernels:

1. **Normal Kernels** – optimized for training and inference prefilling, supporting both NVLink (intranode) and RDMA (internode) forwarding with asymmetric-domain bandwidth.
2. **Low-Latency Kernels** – designed for latency-sensitive inference decoding, using pure RDMA with a hook-based communication-computation overlap that occupies **zero SM resources**.

![DeepEP Architecture](/assets/img/diagrams/deep-ep/deep-ep-architecture.svg)

---

## Architecture Overview

### Normal Kernels (Training / Prefilling)

Normal kernels are optimized for throughput-heavy scenarios:

| Feature | Description |
|---------|-------------|
| **NVLink Intranode** | Up to ~153 GB/s dispatch / ~158 GB/s combine on H800 |
| **RDMA Internode** | ~43–58 GB/s across EP16–EP64 configurations |
| **SM Number Control** | Fine-grained control over Streaming Multiprocessor usage |
| **FP8 Dispatch + BF16 Combine** | Mixed-precision for memory and compute efficiency |

### Low-Latency Kernels (Inference Decoding)

For decoding scenarios where every microsecond counts:

| Feature | Description |
|---------|-------------|
| **Pure RDMA** | Direct NIC-to-NIC communication without CPU involvement |
| **Ultra-Low Latency** | 77–194 μs dispatch, 114–369 μs combine |
| **Hook-Based Overlap** | Background RDMA traffic with **zero SM occupation** |
| **CUDA Graph Compatible** | Enables static graph capture for repeated inference |

---

## MoE Dispatch and Combine Flow

The core data flow in DeepEP follows the standard MoE pattern with highly optimized communication primitives:

![MoE Data Flow](/assets/img/diagrams/deep-ep/deep-ep-moe-flow.svg)

### Dispatch Forward

```python
import torch
import torch.distributed as dist
from deep_ep import Buffer, EventOverlap

Buffer.set_num_sms(24)

def dispatch_forward(x, topk_idx, topk_weights, num_experts, previous_event=None):
    # Calculate dispatch layout
    num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event = \
        _buffer.get_dispatch_layout(topk_idx, num_experts,
                                    previous_event=previous_event, async_finish=True,
                                    allocate_on_comm_stream=previous_event is not None)
    
    # Execute dispatch kernel
    recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        _buffer.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank,
                         num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank,
                         num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=True,
                         allocate_on_comm_stream=True)
    return recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event
```

### Combine Forward

```python
def combine_forward(x, handle, previous_event=None):
    combined_x, _, event = _buffer.combine(
        x, handle, async_finish=True, previous_event=previous_event,
        allocate_on_comm_stream=previous_event is not None
    )
    return combined_x, event
```

### Low-Latency Dispatch for Decoding

```python
def low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts):
    recv_hidden_states, recv_expert_count, handle, event, hook = \
        _buffer.low_latency_dispatch(
            hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts,
            async_finish=False, return_recv_hook=True
        )
    # hook() triggers background RDMA receive without SM occupation
    return recv_hidden_states, recv_expert_count, handle, event, hook
```

---

## Performance Benchmarks

DeepEP is benchmarked on **H800 GPUs** with **CX7 InfiniBand 400 Gb/s** network cards, following the DeepSeek-V3/R1 pretraining settings (4096 tokens per batch, 7168 hidden, top-4 groups, top-8 experts, FP8 dispatching and BF16 combining).

![Performance Benchmarks](/assets/img/diagrams/deep-ep/deep-ep-performance.svg)

### Normal Kernel Results

| Type | EP Size | Dispatch Bandwidth | Combine Bandwidth |
|------|---------|-------------------|-------------------|
| Intranode | 8 | 153 GB/s (NVLink) | 158 GB/s (NVLink) |
| Internode | 16 | 43 GB/s (RDMA) | 43 GB/s (RDMA) |
| Internode | 32 | 58 GB/s (RDMA) | 57 GB/s (RDMA) |
| Internode | 64 | 51 GB/s (RDMA) | 50 GB/s (RDMA) |

### Low-Latency Kernel Results

| EP Size | Dispatch Latency | Dispatch BW | Combine Latency | Combine BW |
|---------|-----------------|-------------|-----------------|------------|
| 8 | 77 μs | 98 GB/s | 114 μs | 127 GB/s |
| 16 | 118 μs | 63 GB/s | 195 μs | 74 GB/s |
| 32 | 155 μs | 48 GB/s | 273 μs | 53 GB/s |
| 64 | 173 μs | 43 GB/s | 314 μs | 46 GB/s |
| 128 | 192 μs | 39 GB/s | 369 μs | 39 GB/s |
| 256 | 194 μs | 39 GB/s | 360 μs | 40 GB/s |

> **News (2025.04.22)**: Tencent Network Platform Department contributed optimizations that improved normal kernel performance by up to **30%**.
>
> **News (2025.06.05)**: Low-latency kernels now leverage NVLink for intranode communication where possible, further reducing latency.

---

## Network Configuration

DeepEP is fully tested with **InfiniBand** and theoretically compatible with **RoCE**. Proper network configuration is essential for peak performance:

![Network Configuration](/assets/img/diagrams/deep-ep/deep-ep-network-config.svg)

### Traffic Isolation

Use **InfiniBand Virtual Lanes (VL)** to segregate traffic:

- Normal kernel workloads
- Low-latency kernel workloads
- Other cluster workloads

Control via the `NVSHMEM_IB_SL` environment variable.

### Adaptive Routing

| Load Condition | Recommendation |
|----------------|---------------|
| Heavy network load | Enable adaptive routing |
| Light network load | Use static routing |

### Congestion Control

Disabled in production environments—no significant congestion has been observed in DeepSeek's deployments.

---

## Installation

### Requirements

- **GPU**: Ampere (SM80), Hopper (SM90), or architectures with SM90 PTX ISA support
- **Python**: 3.8+
- **CUDA**: 11.0+ (SM80) or 12.3+ (SM90)
- **PyTorch**: 2.1+
- **Network**: NVLink (intranode) + RDMA (internode)

### Build from Source

```bash
# Install NVSHMEM dependency first (see third-party/README.md)

# Build and create symbolic links
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py build
ln -s build/lib.linux-x86_64-cpython-38/deep_ep_cpp.cpython-38-x86_64-linux-gnu.so

# Run tests
python tests/test_intranode.py
python tests/test_internode.py
python tests/test_low_latency.py
```

### Install

```bash
NVSHMEM_DIR=/path/to/installed/nvshmem python setup.py install
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NVSHMEM_DIR` | Path to NVSHMEM (disables internode/low-latency if unset) |
| `DISABLE_SM90_FEATURES` | Set to 1 to disable SM90-specific features |
| `TORCH_CUDA_ARCH_LIST` | Target architectures, e.g., `"9.0"` |
| `DISABLE_AGGRESSIVE_PTX_INSTRS` | Set to 1 to disable aggressive PTX load/store instructions |

---

## Experimental Branches

DeepEP has an active community contributing optimizations:

| Branch | Description |
|--------|-------------|
| **Zero-copy** | Removes PyTorch tensor-to-buffer copies, reducing SM usage significantly |
| **Eager** | Low-latency protocol removing extra RTT from RDMA atomic operations |
| **Hybrid-EP** | TMA instructions for minimal SM usage, PCIe kernel support, NVFP4 support |
| **AntGroup-Opt** | SM-free RDMA path, SBO overlap, rail-optimized forwarding |
| **Mori-EP** | ROCm/AMD GPU support via MORI backend |

---

## Community and Ecosystem

- **[uccl/uccl-ep](https://github.com/uccl-project/uccl/tree/main/ep)** – Heterogeneous GPU and NIC support (Nvidia, AMD, EFA, Broadcom, CX7)
- **[Infrawaves/DeepEP_ibrc_dual-ports_multiQP](https://github.com/Infrawaves/DeepEP_ibrc_dual-ports_multiQP)** – Multi-QP and dual-port NIC support
- **[antgroup/DeepXTrace](https://github.com/antgroup/DeepXTrace)** – Diagnostic analyzer for slow rank localization
- **[ROCm/mori](https://github.com/ROCm/mori)** – AMD's next-gen communication library for AI workloads

---

## Roadmap

- [x] AR support
- [x] A100 support (intranode only)
- [x] BF16 low-latency dispatch
- [x] NVLink for intranode low-latency kernels
- [ ] TMA copy for internode and low-latency kernels
- [ ] SM-free kernels and refactors
- [ ] Fully remove undefined-behavior PTX instructions

---

## Conclusion

DeepEP represents a significant advancement in MoE communication efficiency, providing both **throughput-optimized** and **latency-optimized** kernels for different phases of model training and inference. With support for FP8, fine-grained SM control, and zero-SM-overlap hooks, it enables scaling MoE models to hundreds of GPUs while maintaining high hardware utilization. The active community contributions from Tencent, AntGroup, and others continue to push its performance boundaries.

For more details, visit the [DeepEP GitHub repository](https://github.com/deepseek-ai/DeepEP).

---

## Citation

```bibtex
@misc{deepep2025,
  title={DeepEP: an efficient expert-parallel communication library},
  author={Chenggang Zhao and Shangyan Zhou and Liyue Zhang and Chengqi Deng and Zhean Xu and Yuxuan Liu and Kuai Yu and Jiashi Li and Liang Zhao},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/deepseek-ai/DeepEP}},
}
```
