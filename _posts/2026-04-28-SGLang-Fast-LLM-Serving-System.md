---
layout: post
title: "SGLang: The High-Performance LLM Serving Framework Powering 400K+ GPUs"
description: "Learn how SGLang delivers fast LLM inference with RadixAttention, PD disaggregation, and speculative decoding. This guide covers architecture, installation, and real-world usage for the framework powering 400K+ GPUs worldwide."
date: 2026-04-28
header-img: "img/post-bg.jpg"
permalink: /SGLang-Fast-LLM-Serving-System/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Infrastructure, Open Source, Python]
tags: [SGLang, LLM inference, RadixAttention, KV cache, serving framework, GPU optimization, speculative decoding, PD disaggregation, vLLM alternative, open source]
keywords: "SGLang LLM serving framework tutorial, how to use SGLang for LLM inference, SGLang vs vLLM comparison, RadixAttention KV cache reuse, SGLang installation guide, fast LLM serving Python, SGLang speculative decoding setup, PD disaggregation LLM inference, SGLang OpenAI compatible API, best open source LLM serving framework"
author: "PyShine"
---

SGLang is a fast and powerful LLM serving framework designed to deliver high-throughput, low-latency inference for large language models at production scale. Developed by the SGLang team at UC Berkeley and the open-source community, SGLang has rapidly become one of the most widely deployed inference engines in the world, powering over 400,000 GPUs and processing trillions of tokens daily. Adopted by organizations including xAI, AMD, NVIDIA, LinkedIn, Cursor, Oracle, Google Cloud, Microsoft Azure, and AWS, SGLang represents a fundamental shift in how LLM serving infrastructure is built -- combining innovative caching algorithms, disaggregated compute architectures, and advanced decoding strategies into a single, cohesive system.

Whether you are deploying a single model on a workstation or orchestrating thousands of GPUs across a data center, SGLang provides the performance primitives and flexible APIs needed to serve LLMs efficiently. In this post, we will explore the architecture behind SGLang, its core innovations like RadixAttention and PD disaggregation, practical installation and usage examples, and how it compares to alternatives like vLLM.

## How SGLang Works

![SGLang System Architecture](/assets/img/diagrams/sglang/sglang-architecture.svg)

The SGLang system is organized into four distinct layers, each responsible for a critical aspect of the inference pipeline. At the top sits the **API Layer**, which exposes multiple interfaces for client applications. This layer provides an OpenAI-compatible REST API server, an Anthropic-compatible API, and an Ollama-compatible API, allowing seamless integration with existing toolchains and SDKs without any code changes on the client side. The API layer handles request parsing, authentication, rate limiting, and routing, ensuring that incoming requests are properly formatted and dispatched to the engine below.

Beneath the API layer lies the **Core Engine**, which is the brain of SGLang. The engine consists of two primary components: the **Scheduler** and the **Frontend**. The Scheduler is responsible for batch formation, memory management, and request prioritization. It groups incoming requests into batches that maximize GPU utilization while respecting memory constraints and latency targets. The Frontend handles the SGLang programming language runtime, compiling `sgl.function` and `sgl.gen` primitives into optimized execution plans. Together, these components ensure that the system makes efficient use of available compute resources while maintaining responsiveness for individual requests.

The **Inference Layer** sits below the engine and contains the model execution logic. This layer manages tensor parallelism, pipeline parallelism, and data parallelism across multiple GPUs. It handles weight loading, CUDA graph capture, and kernel dispatch for attention, MLP, and normalization operations. The inference layer also integrates with SGLang's attention backends -- including FlashInfer, FlashAttention 3/4, Triton, TRTLLM, and FlashMLA -- selecting the optimal kernel based on the hardware and model configuration.

At the bottom is the **Hardware Layer**, which abstracts GPU communication through NCCL, GLOO, and custom transport protocols. This layer supports NVIDIA GPUs (H100, H200, B200, GB200, GB300), AMD GPUs (MI300X, MI325X), Intel accelerators, Ascend NPUs, and Google TPUs. The hardware abstraction ensures that SGLang can run on diverse infrastructure without requiring changes to the upper layers, making it truly portable across the modern AI compute landscape.

## RadixAttention: The Core Innovation

![RadixAttention and Unified Cache](/assets/img/diagrams/sglang/sglang-radix-attention.svg)

RadixAttention is the foundational innovation that sets SGLang apart from other serving frameworks. Traditional LLM serving systems treat each request independently, computing the full key-value (KV) cache from scratch for every prompt. This approach wastes enormous amounts of compute when multiple requests share common prefixes -- a pattern that occurs frequently in multi-turn conversations, system prompts, few-shot examples, and agentic workflows.

RadixAttention solves this problem by organizing the KV cache as a **radix tree**, a data structure where each node represents a token prefix and its associated KV tensors. When a new request arrives, the system traverses the radix tree to find the longest matching prefix. The matched portion's KV cache is reused directly, and only the remaining suffix tokens need to be computed. This automatic prefix matching happens transparently, with no changes required to the application code.

The radix tree structure enables several powerful capabilities. First, it supports **automatic KV cache sharing** across requests that share system prompts or conversation history. Second, it handles **tree-structured reasoning** patterns, where multiple completions branch from a common prefix -- a pattern common in chain-of-thought, tree-of-thought, and best-of-N sampling strategies. Third, it provides **cascade eviction priority**, where leaf nodes (least recently used suffixes) are evicted first while preserving shared prefixes that benefit multiple active requests.

SGLang further extends caching with its **Unified Cache** architecture, which consolidates three cache types into a single managed pool: the **Full Attention Cache** for standard transformer models, the **Sliding Window Attention (SWA) Cache** for models like Mistral that use windowed attention, and the **Mamba State Cache** for hybrid architectures that combine attention with state-space models. The unified cache allows SGLang to serve heterogeneous model architectures without separate memory pools, reducing fragmentation and improving overall memory utilization.

In production benchmarks, RadixAttention delivers up to **5x faster inference** on workloads with high prefix reuse, such as multi-turn chat, agentic loops, and structured generation tasks. The performance gain scales with the ratio of shared prefix length to total sequence length, making it especially valuable for real-world serving scenarios.

## PD Disaggregation: Separating Prefill and Decode

![PD Disaggregation Architecture](/assets/img/diagrams/sglang/sglang-pd-disaggregation.svg)

Prefill and decode are two fundamentally different computational phases in LLM inference, and PD disaggregation recognizes that they have opposing hardware requirements. **Prefill** is compute-bound: it processes all prompt tokens in parallel using matrix multiplications that saturate GPU FLOPS. **Decode** is memory-bound: it generates one token at a time, with each step requiring a read of the full model weights and KV cache from GPU memory. When these two phases run on the same GPU, they interfere with each other -- prefill consumes memory bandwidth that decode needs, and decode occupies compute units that prefill could use.

SGLang's PD disaggregation architecture separates these phases onto dedicated instances. **Prefill instances** are configured with high-compute GPUs (such as H100 or B200) optimized for parallel token processing. **Decode instances** use GPUs with high memory bandwidth (such as H200 or MI300X) optimized for single-token generation. When a request arrives, the prefill instance processes the entire prompt, building the KV cache. The completed KV cache is then transferred to a decode instance via a high-speed interconnect, where token generation proceeds without interruption.

The KV cache transfer between instances uses SGLang's optimized transport layer, which supports NCCL, Mooncake, and custom RDMA protocols. The transfer is pipelined: as soon as the first layers of the KV cache are ready, they begin transmitting to the decode instance, overlapping communication with computation. This pipelining minimizes the latency overhead of the handoff.

The performance gains from PD disaggregation are substantial. On standard benchmarks, SGLang achieves **3.8x faster prefill** and **4.8x faster decode** compared to collocated serving. On the NVIDIA GB300 NVL72 platform, which provides 72 GPUs with NVLink interconnect, PD disaggregation delivers **25x inference performance** improvement by allowing each phase to run at its optimal configuration without resource contention. This architecture also enables independent scaling: you can add more prefill instances to handle bursty prompt traffic, or more decode instances to increase throughput for long-running generations, without over-provisioning either resource.

## Speculative Decoding for Lower Latency

![Speculative Decoding Pipeline](/assets/img/diagrams/sglang/sglang-speculative-decoding.svg)

Speculative decoding is a technique that accelerates autoregressive generation by predicting multiple future tokens in parallel, then verifying them against the target model. Instead of generating one token per forward pass, speculative decoding drafts several candidate tokens using a smaller or faster model, then validates the entire batch in a single forward pass through the target model. Tokens that pass verification are accepted, and the process repeats from the first rejected position.

SGLang supports multiple speculative decoding methods, each suited to different model architectures and performance requirements:

- **EAGLE-2 and EAGLE-3**: These are model-based draft generators that use a lightweight autoregressive head trained to predict the next-token distribution of the target model. EAGLE-3, the latest version, achieves **373.25 tokens per second** on Llama-3.1-70B compared to 158.34 tokens per second at baseline -- a 2.36x speedup. EAGLE-3 improves over EAGLE-2 by using a deeper draft model with better feature extraction from the target model's hidden states.

- **MTP (Multi-Token Prediction)**: This method trains the target model itself to predict multiple tokens at each step, rather than relying on a separate draft model. MTP is particularly effective for models like DeepSeek-V3 that natively support multi-token prediction heads.

- **NGRAM**: A lightweight, model-free approach that uses n-gram statistics from the prompt to predict likely continuations. NGRAM requires no additional model weights and provides modest speedups (1.2-1.5x) with zero overhead for model loading.

- **DFLASH**: A draft-model approach that uses a small, distilled version of the target model for speculation. DFLASH is useful when a compatible draft model is available and provides good acceptance rates.

The speculative decoding pipeline in SGLang is tightly integrated with RadixAttention. When a speculative draft is rejected at a certain position, the KV cache for the accepted prefix is preserved in the radix tree, and only the rejected suffix needs to be recomputed. This integration means that speculative decoding benefits from prefix caching just like regular generation, further reducing redundant computation.

## Key Features Overview

| Feature | Description |
|---------|-------------|
| RadixAttention | Automatic KV cache reuse via radix tree prefix matching |
| PD Disaggregation | Separate prefill/decode instances for optimized throughput |
| Speculative Decoding | EAGLE-2/3, MTP, NGRAM, DFLASH methods |
| 140+ Models | Llama, Qwen, DeepSeek, Gemma, Mistral, and more |
| 13+ Attention Backends | FlashInfer, FlashAttention 3/4, Triton, TRTLLM, FlashMLA |
| Multi-API Support | OpenAI, Anthropic, Ollama compatible APIs |
| Structured Outputs | JSON schema, regex, EBNF constraints |
| Multi-LoRA | Dynamic adapter management with overlap loading |
| torch.compile | Custom passes with breakable CUDA graphs |
| Multi-Hardware | NVIDIA, AMD, Intel, Ascend, Google TPU |

## Installation

SGLang can be installed in several ways depending on your environment and requirements.

### Install via pip

The simplest way to get started is with pip:

```bash
pip install "sglang[all]"
```

This installs SGLang along with all recommended dependencies, including FlashInfer for optimized attention kernels.

### Install from Source

For the latest development version or to contribute to the project, install from the cloned repository:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e ".[all]"
```

Building from source ensures you have the most recent commits and allows you to modify the codebase directly.

### Install via Docker

For containerized deployments, SGLang provides pre-built Docker images:

```bash
docker pull lmsys/sglang:latest
docker run --gpus all \
  --shm-size 32g \
  -p 30000:30000 \
  lmsys/sglang:latest \
  python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 30000
```

The `--shm-size 32g` flag is important for models that require large shared memory for tensor parallel communication.

## Usage Examples

### Launching a Server

Start an OpenAI-compatible server with a single command:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 30000
```

For multi-GPU serving with tensor parallelism:

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tp 4 \
  --port 30000
```

### OpenAI-Compatible API

Once the server is running, you can query it using the standard OpenAI client or curl:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain RadixAttention in simple terms."}
    ],
    "max_tokens": 512
  }'
```

### Python API with Engine

For programmatic access without a server, use the SGLang Engine directly:

```python
import sglang as sgl

# Launch the engine
engine = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

# Generate completions
outputs = engine.generate(
    prompts=["What is RadixAttention?", "Explain PD disaggregation."],
    sampling_params={"max_new_tokens": 256, "temperature": 0.7}
)

for output in outputs:
    print(output["text"])

# Shut down the engine
engine.shutdown()
```

### SGLang Frontend Language

SGLang also provides a domain-specific language for composing complex LLM programs with automatic KV cache reuse:

```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, question1, question2):
    s += sgl.user(question1)
    s += sgl.assistant(sgl.gen(max_tokens=256))
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen(max_tokens=256))

# Run with automatic prefix caching
state = multi_turn_chat.run(
    question1="What is SGLang?",
    question2="How does RadixAttention work?"
)
print(state.text())
```

The `sgl.function` decorator compiles the program into an optimized execution plan. When the same function is called with a shared prefix (for example, the same system prompt), RadixAttention automatically reuses the cached KV tensors, avoiding redundant computation.

## Performance Benchmarks

SGLang has consistently pushed the boundaries of LLM serving performance across multiple milestones:

| Milestone | Performance |
|-----------|-------------|
| RadixAttention (Jan 2024) | Up to 5x faster inference |
| Compressed FSM (Feb 2024) | 3x faster JSON decoding |
| Llama3 Serving (Jul 2024) | Faster than TensorRT-LLM and vLLM |
| DeepSeek MLA (Sep 2024) | 7x faster DeepSeek MLA |
| GB200 NVL72 (Sep 2025) | 2.7x higher decode throughput |
| GB300 NVL72 (Feb 2026) | 25x inference performance |
| EAGLE-3 Speculative | 373.25 tok/s vs 158.34 baseline |

These benchmarks reflect real-world serving scenarios, not synthetic micro-tests. The RadixAttention gains come from multi-turn conversations and agentic workloads with high prefix reuse. The DeepSeek MLA result demonstrates SGLang's ability to optimize for novel architectures like Multi-head Latent Attention. The GB300 NVL72 result showcases the combined power of PD disaggregation and hardware-aware scheduling on NVIDIA's latest platform.

## SGLang vs vLLM

SGLang and vLLM are the two most popular open-source LLM serving frameworks, and they share some architectural similarities while differing in key areas:

**KV Cache Management**: vLLM uses PagedAttention, which manages KV cache memory at the page level to prevent fragmentation. SGLang uses RadixAttention, which adds automatic prefix matching and reuse on top of paged memory management. This means SGLang can share KV cache across requests with common prefixes, while vLLM would recompute those prefixes for each request.

**PD Disaggregation**: SGLang has first-class support for PD disaggregation, allowing prefill and decode to run on separate GPU instances with optimized KV cache transfer. vLLM does not natively support disaggregated prefill and decode, requiring both phases to run on the same GPU.

**Speculative Decoding**: Both frameworks support speculative decoding, but SGLang offers a wider range of methods including EAGLE-2, EAGLE-3, MTP, NGRAM, and DFLASH. vLLM primarily supports EAGLE and Medusa-style draft models.

**Model Coverage**: SGLang supports over 140 models including the latest architectures like DeepSeek-V3, Llama-4, Qwen3, and Gemma 3. vLLM has comparable model coverage but may lag on the newest architectures.

**Hardware Support**: Both frameworks support NVIDIA and AMD GPUs. SGLang additionally supports Intel accelerators, Ascend NPUs, and Google TPUs, providing broader hardware coverage for organizations with diverse infrastructure.

## Conclusion

SGLang has established itself as a leading LLM serving framework by combining three critical innovations: RadixAttention for automatic KV cache reuse, PD disaggregation for separating compute-bound and memory-bound phases, and speculative decoding for reducing generation latency. These features work together synergistically -- RadixAttention preserves shared prefixes that benefit speculative decoding, PD disaggregation ensures each phase runs on optimal hardware, and speculative decoding fills the compute gap during decode-bound generation.

With support for 140+ models, 13+ attention backends, and deployment across NVIDIA, AMD, Intel, Ascend, and TPU hardware, SGLang provides the flexibility and performance needed for production LLM serving at any scale. The framework's adoption by major organizations processing trillions of tokens daily validates its readiness for the most demanding workloads.

If you are building LLM-powered applications and need a serving framework that delivers high throughput, low latency, and efficient resource utilization, SGLang is worth serious consideration.

**Links:**

- GitHub: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
- Documentation: [https://sgl-project.github.io/](https://sgl-project.github.io/)