---
layout: post
title: "AirLLM: Run 70B LLMs on a 4GB GPU Without Quantization"
description: "Learn how AirLLM runs 70B large language models on a single 4GB GPU card without quantization, distillation, or pruning. Explore MoE streaming, FP8 support, and real-world performance."
date: 2026-08-08
header-img: "assets/img/ai-coding-frameworks/ai-coding-frameworks"
permalink: /airllm-70b-llm-4gb-gpu/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
categories: [AI, LLM, Inference, GPU, Open Source]
tags: [AirLLM, LLM, GPU, Inference, MoE, Mixture of Experts, 70B, DeepSeek, Llama, FP8, CPU, MacOS, Apache 2.0]
keywords: "AirLLM tutorial, run 70B on 4GB GPU, LLM inference optimization, Mixture of Experts streaming, AirLLM installation guide, how to run large models on small GPU, DeepSeek-V3 single card, Llama 3.1 405B 8GB, Kimi K3 2.8T"
author: "PyShine"
---

## Introduction

The GPU memory bottleneck is the elephant in the room of modern AI. As large language models scale from 7B to 70B to 400B and beyond, the VRAM required just to hold the model weights has become a showstopper for most developers. A full-precision 70B model needs roughly 140GB of VRAM -- that's four A100-80GB cards or a single H100, neither of which is within reach of the average developer, student, or small team.

The conventional wisdom says the only path forward is quantization (shrinking model weights from FP16 to 4-bit or 8-bit), distillation (training a smaller model to mimic a larger one), or pruning (removing weights). Each approach trades quality for size, and quantization in particular can degrade model behavior in subtle ways that are hard to detect.

**AirLLM** by Gavin Li challenges this assumption. It lets you run 70B models on a single 4GB GPU card -- **without quantization, distillation, or pruning**. Even more remarkably, it scales to 405B Llama 3.1 on 8GB, 671B DeepSeek-V3 on ~12GB, and Kimi K3 (2.8T parameters) on under 4GB. The trick is not about making the model smaller -- it's about making the model loading smarter.

## What Is AirLLM?

AirLLM is a Python library that performs **layer-by-layer model streaming** for inference. Instead of loading the entire model into GPU memory at once, it processes one transformer layer at a time, keeping only the current layer's weights on the GPU. For Mixture of Experts (MoE) models, it goes even further: it streams one expert at a time within each layer, loading only the expert weights that the current token actually routes to.

This approach exploits a key insight about transformer inference: during a forward pass, each layer processes the hidden state sequentially. You don't need all layers in memory simultaneously -- you only need the one you're currently computing. The hidden state (which is relatively small, typically a few MB per token) passes between layers, and each layer's weights are loaded and discarded as needed.

### The Technical Approach

AirLLM's architecture centers on three core techniques:

**1. Layer-wise streaming.** The model is decomposed into individual transformer layers, each saved as a separate file. During inference, layers are loaded one at a time onto the GPU. After a layer processes the hidden state, the layer's weights are freed from GPU memory and the next layer is loaded. This creates a constant memory footprint proportional to the size of a single layer, not the total model.

**2. MoE expert streaming.** For Mixture of Experts models (like Mixtral, DeepSeek-V3, Qwen3-MoE), AirLLM further decomposes each layer. MoE layers contain multiple "expert" feed-forward networks, and a router selects which expert(s) to activate for each token. AirLLM loads only the selected expert(s) at inference time, rather than loading all experts. This dramatically reduces memory for models with many experts per layer.

**3. Hidden state accumulation.** The hidden state (the intermediate representation of the input) is maintained on the GPU throughout the forward pass. Since the hidden state size is determined by the hidden dimension and sequence length -- not the model size -- it remains manageable even for trillion-parameter models. For a 4096-token sequence with a 4096 hidden dimension, the hidden state is roughly 4096 × 4096 × 2 bytes ≈ 32MB, trivial compared to the model weights.

The result: VRAM requirements scale with **per-layer size**, not **total model size**. A 70B model and a 7B model with the same layer architecture use roughly the same amount of GPU memory, because each layer has the same dimensions. The 70B model just has more layers, processed sequentially.

## Key Features

### Run 70B Models on 4GB GPU

The headline feature: full-precision 70B inference on a consumer-grade 4GB GPU card (like the RTX 3050, RTX 3060 4GB, or the 4GB M-series Macs). No quantization required -- the model runs in its original FP16 or BF16 precision.

### Run 405B Llama 3.1 on 8GB

Llama 3.1 405B is the flagship model in the Llama family, with 405 billion parameters in its full version. AirLLM runs it on a single 8GB GPU by streaming layers one at a time.

### Run 671B DeepSeek-V3 on ~12GB

DeepSeek-V3 is a Mixture of Experts model with a total of 671 billion parameters (activated parameters are much smaller per token). AirLLM leverages MoE expert streaming to load only the active experts, fitting the inference pipeline into ~12GB.

### Run 2.8T Kimi K3 on Under 4GB

Kimi K3 is the largest open-source model released to date at 2.8 trillion parameters. Thanks to its sparse MoE architecture and AirLLM's per-expert streaming, it runs in just 3.72GB of VRAM on an RTX 6000 Ada.

### Sparse MoE Expert Streaming

For MoE models, AirLLM loads only the expert(s) that each token's router selects. If a layer has 64 experts and a token routes to 2, only those 2 experts' weights are loaded. This is the key enabler for running trillion-parameter MoE models on small GPUs.

### FP8 Model Support (v3.0)

Version 3.0 adds support for FP8 models, further reducing memory footprint. When combined with expert streaming, FP8 models like Qwen3-235B fit into approximately 3GB of VRAM.

### CPU Inference

Don't have a GPU? AirLLM supports CPU inference as well. While slower, it makes running large models accessible on any machine with sufficient RAM and disk space.

### MacOS Support

AirLLM runs natively on Apple Silicon (M1, M2, M3) using the MLX framework. This means MacBook and Mac mini users can run 70B+ models without any NVIDIA GPU.

### Automatic Model Detection (AutoModel)

AirLLM's `AutoModel` class automatically detects the model architecture from its HuggingFace configuration, so you don't need to specify a model class explicitly. One line of code handles Llama, Qwen, Mistral, Mixtral, DeepSeek, Phi, Gemma, ChatGLM, Baichuan, InternLM, and more.

### Apache 2.0 Licensed

AirLLM is released under the Apache 2.0 license, making it free for both personal and commercial use.

## Installation

Getting started with AirLLM is straightforward:

```bash
pip install airllm
```

That's it. AirLLM works with PyTorch and automatically detects your hardware (GPU or CPU).

## Usage Examples

### Basic Inference

The simplest way to use AirLLM is through the `AutoModel` class:

```python
from airllm import AutoModel

MAX_LENGTH = 128

model = AutoModel.from_pretrained("meta-llama/Llama-2-70b-hf")

input_text = ["What is the capital of France?"]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False
)

generation_output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])
print(output)
```

### Running a 70B Model on 4GB GPU

The same code works for the 70B version -- AirLLM handles the layer streaming automatically:

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3-70B-Instruct")

input_text = ["Explain quantum computing in simple terms."]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=256,
    padding=False
)

generation_output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=100,
    use_cache=True,
    return_dict_in_generate=True
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```

### Running 405B Llama 3.1 on 8GB

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.1-405B-Instruct")

input_text = ["Write a Python function that sorts a linked list."]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=256,
    padding=False
)

generation_output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=200,
    use_cache=True,
    return_dict_in_generate=True
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```

### Running DeepSeek-V3 (671B) on ~12GB

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-V3")

input_text = ["Explain the difference between supervised and unsupervised learning."]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=256,
    padding=False
)

generation_output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=200,
    use_cache=True,
    return_dict_in_generate=True
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```

### Running Kimi K3 (2.8T) on Under 4GB

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("moonshotai/Kimi-K3")

input_text = ["Write a haiku about artificial intelligence."]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=256,
    padding=False
)

generation_output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=50,
    use_cache=True,
    return_dict_in_generate=True
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```

### CPU Inference

No GPU? No problem:

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

input_text = ["Hello, how are you?"]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False
)

generation_output = model.generate(
    input_tokens["input_ids"],  # .cpu() is the default
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```

### MacOS Inference

On Apple Silicon, install MLX and use the same code:

```bash
pip install airllm mlx torch
```

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-135M")

input_tokens = model.tokenizer(
    ["What is the meaning of life?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False
)

generation_output = model.generate(
    input_tokens["input_ids"],
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```

### Using Compression for Speed

AirLLM also supports optional block-wise quantization (4-bit or 8-bit) to speed up inference by up to 3x when disk loading is the bottleneck:

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "garage-bAInd/Platypus2-70B-instruct",
    compression="4bit"
)

input_text = ["Explain the transformer architecture."]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=256,
    padding=False
)

generation_output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=100,
    use_cache=True,
    return_dict_in_generate=True
)

print(model.tokenizer.decode(generation_output.sequences[0]))
```

### Profiling and Configuration

Enable profiling to see time spent in each stage:

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    profiling_mode=True
)
```

Specify a custom path for layer shards:

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    layer_shards_saving_path="/path/to/save/layer/shards"
)
```

Use a HuggingFace token for gated models:

```python
model = AutoModel.from_pretrained(
    "meta-llama/Llama-3-70B",
    hf_token="HF_API_TOKEN"
)
```

## Performance Benchmarks

The following table summarizes what AirLLM users have achieved across different model sizes and configurations:

| Model | Total Parameters | Minimum VRAM | Notes |
|---|---|---|---|
| Mistral 7B / Phi-3 8B | 7B | ~1-2 GB | Standard dense model |
| Qwen3-32B | 32B | ~2-3 GB | Dense model |
| Mixtral 8x7B (MoE) | 47B | ~1-3 GB | MoE expert streaming |
| Qwen3-235B (MoE) | 235B | ~3 GB | MoE + FP8 support |
| Llama 3 70B | 70B | ~4 GB | Full FP16 precision |
| Llama 3.1 405B | 405B | ~8 GB | Layer streaming |
| DeepSeek-V3 | 671B | ~12 GB | MoE + FP8 |
| Kimi K3 | 2.8T | ~3.7 GB | Sparse MoE, per-expert streaming |

The key insight is that VRAM requirements do not scale with total model size -- they scale with **per-layer size**. This is why a 2.8T MoE model can run in less VRAM than a 70B dense model: each layer in the 2.8T model is actually smaller when only the active experts are loaded.

### Speed Considerations

The trade-off for this memory efficiency is speed. Because layers are loaded sequentially from disk, inference is slower than with a fully loaded model. However, AirLLM mitigates this with several optimizations:

- **Prefetching**: The next layer is loaded while the current layer is being computed, overlapping I/O and compute by up to 10%.
- **Block-wise quantization**: Optional 4-bit or 8-bit compression reduces layer size, speeding up disk loading by up to 3x with minimal accuracy loss.
- **MoE expert streaming**: Loading only the needed experts means less data to transfer per layer.

For many use cases -- research, exploration, prototyping -- the speed difference is acceptable given the ability to run models that would otherwise require multi-GPU clusters.

## Supported Models

AirLLM works out of the box with virtually every popular open LLM through its `AutoModel` class:

- **Llama**: Llama 2, Llama 3, Llama 3.1, Llama 3.3, Llama 4
- **Qwen**: Qwen 1, Qwen 2, Qwen 2.5, Qwen 3 (including MoE and FP8 variants)
- **DeepSeek**: DeepSeek V2, V3, R1
- **Mistral & Mixtral**: All variants
- **Phi**: Phi-3, Phi-4
- **Gemma**: Gemma 1, Gemma 2
- **ChatGLM**: ChatGLM series
- **Baichuan**: Baichuan series
- **InternLM**: InternLM series
- **Yi**: Yi series

New models are typically supported the day they're released, as `AutoModel` reads the HuggingFace configuration and adapts automatically.

## Who Should Use AirLLM?

### Developers without High-End GPUs

If you've ever wanted to experiment with 70B+ models but couldn't justify the cost of an A100 or H100, AirLLM is for you. A laptop with a 4GB GPU or an Apple Silicon Mac can now run state-of-the-art models.

### Students and Researchers

AirLLM democratizes access to large models for academic research. Students can fine-tune, evaluate, and experiment with 70B+ models on personal hardware without needing cluster access.

### Cost-Conscious Teams

For small teams and startups, AirLLM eliminates the need for expensive multi-GPU deployments in many scenarios. A single RTX 3060 (4GB) can handle inference workloads that previously required a full server rack.

### Privacy-Sensitive Applications

Running large models locally -- without sending data to cloud APIs -- is critical for confidential data processing. AirLLM makes local inference viable for organizations that cannot use cloud-based LLM services.

### Rapid Prototyping

When you need to quickly test a model's capabilities without setting up a full inference infrastructure, AirLLM provides the fastest path from "HuggingFace model ID" to "running inference."

## Conclusion

AirLLM represents a fundamentally different approach to LLM inference. Instead of trying to fit the entire model into GPU memory through compression, it asks a simpler question: do you need the entire model in memory at once? The answer, for transformer inference, is no -- and the implications are staggering.

By streaming layers one at a time and further decomposing MoE layers into individual experts, AirLLM breaks the VRAM bottleneck that has defined LLM deployment for years. A 4GB GPU -- once capable of running only 7B models -- can now handle 70B. An 8GB card can run 405B. And the 2.8T Kimi K3 runs on under 4GB, a feat that seemed impossible just months ago.

The trade-off is speed: sequential layer loading means slower inference than a fully loaded multi-GPU setup. But for the vast majority of use cases -- research, prototyping, exploration, privacy-sensitive workloads -- this is an acceptable trade. The ability to run state-of-the-art models on commodity hardware without quantization or quality loss changes what's possible for individual developers and small teams.

As models continue to grow larger, AirLLM's approach becomes increasingly compelling. The layer-streaming technique scales with model depth rather than model width, making it inherently future-proof for trillion-parameter architectures. Combined with FP8 support, CPU inference, MacOS compatibility, and automatic model detection, AirLLM is a vital tool in the inference optimization toolkit.

**Resources:**
- [AirLLM on GitHub](https://github.com/lyogavin/airllm)
- [AirLLM on PyPI](https://pypi.org/project/airllm/)
- [Colab Example Notebook](https://colab.research.google.com/github/lyogavin/airllm/blob/main/air_llm/examples/run_all_types_of_models.ipynb)
- [AirLLM Paper on arXiv](https://arxiv.org/abs/2212.09720)