---
layout: post
title: "DFlash: Block Diffusion for Lightning-Fast LLM Speculative Decoding"
description: "Learn how DFlash uses block diffusion models to accelerate LLM inference with speculative decoding, achieving 2-3x speedups across Transformers, SGLang, vLLM, and MLX backends."
date: 2026-04-17
header-img: "img/post-bg.jpg"
permalink: /DFlash-Block-Diffusion-Speculative-Decoding/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - LLM
  - Speculative Decoding
  - Python
author: "PyShine"
---

# DFlash: Block Diffusion for Lightning-Fast LLM Speculative Decoding

Large Language Models have become the backbone of modern AI applications, but their autoregressive decoding process remains a fundamental bottleneck. Each token must be generated sequentially -- the model cannot produce the next token until the current one is finalized. This serial dependency limits throughput and makes real-time applications costly, especially for large models with billions of parameters.

DFlash, developed by Z Lab, tackles this problem head-on with an elegant approach: **block diffusion for speculative decoding**. Instead of drafting tokens one at a time like traditional speculative decoding methods, DFlash generates an entire block of draft tokens in parallel using a lightweight diffusion-style model. These draft tokens are then verified against the target LLM in a single forward pass, dramatically increasing the number of tokens accepted per verification step.

The result is a practical, production-ready system that achieves 2-3x speedups across four major inference backends -- Transformers, SGLang, vLLM, and MLX (Apple Silicon) -- with 14+ pre-trained draft models available on HuggingFace for popular LLMs including Qwen3, Qwen3.5, Kimi-K2.5, LLaMA-3.1, and more.

## The Speculative Decoding Problem

Autoregressive language models generate text one token at a time. For a model with 70 billion parameters, each forward pass through the GPU produces exactly one token, regardless of how much parallel compute capacity sits idle. This is fundamentally wasteful -- the GPU could process hundreds of tokens in parallel during prefill, but during decoding it processes just one.

Speculative decoding addresses this by introducing a small, fast **draft model** that proposes multiple candidate tokens. The large **target model** then verifies these candidates in a single forward pass. If the draft model is good, most tokens are accepted, and the effective throughput increases by the average acceptance length.

Traditional speculative decoding uses an autoregressive draft model, which still generates tokens one at a time -- just faster. DFlash reimagines this: what if the draft model could generate all its proposed tokens **in parallel**, like a diffusion model denoising a block? This is the core insight behind block diffusion.

## How It Works

![Speculative Decoding Architecture](/assets/img/diagrams/dflash/dflash-speculative-decoding-architecture.svg)

### Understanding the Speculative Decoding Architecture

The architecture diagram above illustrates the complete DFlash speculative decoding pipeline, from input prompt through draft generation, target verification, and final output. This system operates as a tight loop that alternates between the lightweight draft model and the powerful target model.

**Input Processing and Prefill**

The pipeline begins when the input prompt is fed into the target model for prefill. During this initial forward pass, the target model processes all input tokens and produces two critical outputs: the first decoded token (via the language model head) and the hidden states from strategically selected intermediate layers. These hidden states are not normally exposed during inference, but DFlash captures them to condition the draft model. The `build_target_layer_ids` function computes which layers to extract from, distributing selections evenly across the target model's depth. For example, if the target has 64 layers and the draft has 4 layers, DFlash extracts hidden states from layers near positions 1, 21, 42, and 61.

**Context Feature Extraction**

The extracted hidden states from multiple target layers are concatenated along the feature dimension and then projected through a linear layer (`fc`) followed by RMSNorm (`hidden_norm`). This projection reduces the concatenated dimensionality back to the draft model's hidden size, creating a compact context feature vector. This context feature carries rich semantic information from the target model -- it knows what the target model "is thinking" at multiple levels of abstraction, which allows the draft model to produce tokens that closely match the target's distribution.

**Block Draft Generation**

The draft model takes the context feature along with a block of mask tokens and generates draft tokens in parallel. Unlike autoregressive drafting where each token depends on the previous one, the block diffusion approach allows all positions in the block to be refined simultaneously. The draft model's attention layers concatenate context keys/values (from the target hidden states) with noise/proposal keys/values (from the draft's own embeddings), enabling cross-architecture attention that bridges the two models.

**Target Verification and Acceptance**

The target model then processes the entire block of draft tokens in a single forward pass. The verification step compares the draft tokens against the target model's posterior distribution. Tokens are accepted sequentially from the beginning of the block until the first mismatch. The acceptance length is computed as a cumulative product of match indicators, ensuring that only a contiguous prefix of matching tokens is accepted. After the last accepted token, the target model's own prediction for the next position is appended, guaranteeing at least one new token per iteration.

**KV-Cache Management**

Both the target and draft models maintain KV caches that are carefully managed across iterations. After each verification step, the target KV cache is cropped to remove entries beyond the accepted prefix. The draft KV cache is similarly cropped and then reused in the next iteration, avoiding redundant recomputation. This `DynamicCache` with trimming and rollback is essential for memory efficiency, especially in long-context scenarios.

## Block Diffusion Process

![Block Diffusion Process](/assets/img/diagrams/dflash/dflash-block-diffusion-process.svg)

### Understanding the Block Diffusion Process

The block diffusion process diagram above shows how DFlash generates multiple draft tokens in parallel through a diffusion-inspired denoising procedure. This is the key innovation that separates DFlash from traditional autoregressive speculative decoding.

**From Noise to Tokens**

The process starts with a block of mask tokens -- placeholder identifiers that represent "unknown" positions. These mask tokens are embedded using the target model's `embed_tokens` layer (shared with the draft model in the MLX backend, or separately loaded in the Transformers backend). The resulting noise embeddings serve as the initial input to the draft model, analogous to the noisy input in a standard diffusion model.

**Single-Step Denoising**

Unlike iterative diffusion models that require many denoising steps, DFlash performs a single forward pass through the draft model to produce the entire block of tokens. The draft model's architecture is specifically designed for this: each attention layer receives both the context features (from the target model) and the noise embeddings (from the mask tokens). The context features provide the "guidance signal" that steers the denoising, while the noise embeddings provide the positional structure for the block.

**Non-Causal Attention Within the Block**

A critical design choice is that the draft model uses **non-causal attention** within the block (`is_causal=False`). This means each position in the block can attend to every other position, allowing the model to resolve inter-token dependencies in parallel. This is fundamentally different from autoregressive drafting, where token N cannot influence token N-1. With non-causal attention, the draft model can produce globally coherent blocks where all tokens are mutually consistent.

**Position Encoding and KV-Cache Reuse**

The draft model uses position IDs that account for both the previously accepted context and the current block positions. The KV cache from previous iterations is reused: context keys and values from earlier accepted tokens are prepended to the current block's keys and values. This means the draft model has full access to the conversation history without reprocessing it, making each draft step efficient. After generating the block, the draft KV cache is cropped to the start of the next iteration's expected position.

**Block Size as a Hyperparameter**

The block size (B) is a configurable hyperparameter that trades off parallelism against acceptance rate. Larger blocks mean more tokens generated in parallel, but also more opportunities for the draft to diverge from the target distribution. In practice, block sizes of 8-16 tokens work well for most models, with the optimal value depending on the specific target-draft pair and the task domain.

## Cross-Attention Mechanism

![Cross-Attention Mechanism](/assets/img/diagrams/dflash/dflash-cross-attention-mechanism.svg)

### Understanding the Cross-Attention Mechanism

The cross-attention mechanism diagram above details how DFlash bridges the target model and the draft model through a novel attention pattern that concatenates context and proposal key-value pairs. This is the architectural heart of the system.

**Dual Key-Value Sources**

Each DFlash attention layer receives two distinct inputs: the context hidden states (projected from the target model's intermediate layers) and the noise/proposal embeddings (from the draft model's own processing). The query vectors are computed from the proposal embeddings, while the key and value vectors are computed from **both** sources. Specifically, the key projection `k_proj` is applied to both the context hidden states and the proposal embeddings, producing `k_ctx` and `k_noise` respectively. The same is done for the value projection, producing `v_ctx` and `v_noise`.

**Concatenation Strategy**

The context keys/values and noise keys/values are concatenated along the sequence dimension before being passed to the attention function. This means the attention matrix has shape `(block_length, context_length + block_length)`, where the first portion attends to the target model's context and the second portion attends to the draft's own proposals. The context portion provides grounding -- it tells the draft model what the target has already produced -- while the proposal portion enables inter-token coordination within the block.

**Query-Norm and Key-Norm**

DFlash applies RMSNorm to both queries and keys before computing attention scores. This QK-norm technique stabilizes training and improves the quality of the attention distribution, especially when combining keys from two different sources (target context and draft proposals) that may have different statistical properties. The norm ensures that the dot-product attention scores are well-scaled regardless of the source.

**Rotary Position Embeddings**

Rotary position embeddings (RoPE) are applied to both queries and keys. The context keys receive position offsets corresponding to their original positions in the target model's sequence, while the proposal keys receive offsets starting after the context length. This ensures that the position encoding correctly reflects the absolute and relative positions of all tokens, whether they come from the target context or the draft block.

**Cache Integration**

The concatenated key-value pairs are stored in the draft model's KV cache. On subsequent iterations, the cached context keys/values are reused, and only the new block's keys/values need to be computed. The `cache.update_and_fetch` method handles this transparently, returning the full key-value history up to the current position. After each iteration, the cache is cropped to remove entries beyond the accepted prefix, maintaining memory efficiency.

## Multi-Backend Support

![Multi-Backend Support](/assets/img/diagrams/dflash/dflash-multi-backend-support.svg)

### Understanding the Multi-Backend Architecture

The multi-backend support diagram above illustrates how DFlash provides a unified interface across four distinct inference backends, each with its own architecture, optimizations, and deployment characteristics. This cross-platform design makes DFlash one of the most versatile speculative decoding solutions available.

**Transformers Backend (PyTorch)**

The Transformers backend is the reference implementation, built directly on HuggingFace Transformers. It supports Qwen3 and LLaMA-3.1-8B-Instruct models and provides the most straightforward API for research and experimentation. The draft model is loaded as a `DFlashDraftModel` that inherits from `Qwen3PreTrainedModel`, ensuring compatibility with the Transformers ecosystem. This backend uses `DynamicCache` for KV-cache management and supports both Flash Attention 2 and PyTorch SDPA as attention implementations. Multi-GPU evaluation is supported via `torchrun` with NCCL distributed initialization.

**SGLang Backend**

SGLang is a high-performance serving framework, and DFlash integrates with it through the `--speculative-algorithm DFLASH` flag. The SGLang backend supports advanced features like schedule overlapping (experimental) and tensor-parallel inference. It uses the TRT-LLM MHA attention backend for the target model and Flash Attention 4 (FA4) for the draft model, optimizing for throughput in serving scenarios. The Mamba scheduler strategy with extra buffering handles the hybrid attention patterns that DFlash requires.

**vLLM Backend**

vLLM is the most widely deployed LLM serving engine, and DFlash support is available through the nightly build. The integration uses vLLM's native speculative decoding framework, configured via the `--speculative-config` JSON parameter. Users specify the method as `"dflash"`, provide the draft model path, and set the number of speculative tokens. This backend is ideal for production deployments that already use vLLM, as it requires minimal changes to existing infrastructure.

**MLX Backend (Apple Silicon)**

The MLX backend brings DFlash to Apple Silicon Macs, leveraging Apple's MLX framework for efficient Metal GPU acceleration. This implementation has several unique features: it shares `embed_tokens` and `lm_head` between the target and draft models (via the `bind` method) to reduce memory usage, supports `RotatingKVCache` for sliding window KV management, and includes special handling for GatedDeltaNet (GDN) layers through a monkey-patching mechanism. The `_GDNStateCapture` class temporarily patches GDN layer calls to intercept their internal states, enabling DFlash to work with models like Qwen3.5 that use GDN layers.

**Unified Benchmarking CLI**

All four backends share a common benchmarking interface through `python -m dflash.benchmark`. The CLI accepts a `--backend` flag and adjusts its behavior accordingly: the Transformers and MLX backends run models locally, while the SGLang and vLLM backends connect to a running server via HTTP. This unified interface makes it easy to compare performance across backends and identify the best option for a given deployment scenario.

## Installation

DFlash supports four backends, each with its own dependency set. Use a separate virtual environment for each backend to avoid conflicts.

**Transformers (PyTorch)**

```bash
uv pip install -e ".[transformers]"
```

**SGLang**

```bash
uv pip install -e ".[sglang]"
```

**vLLM (Nightly Build Required)**

```bash
uv pip install -e ".[vllm]"
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

**MLX (Apple Silicon)**

```bash
pip install -e ".[mlx]"
```

For optimal performance with the Transformers backend, install Flash Attention:

```bash
pip install flash-attn --no-build-isolation
```

## Usage Examples

### vLLM Serving

Launch a vLLM server with DFlash speculative decoding for Qwen3.5-27B:

```bash
vllm serve Qwen/Qwen3.5-27B \
  --speculative-config '{"method": "dflash", "model": "z-lab/Qwen3.5-27B-DFlash", "num_speculative_tokens": 15}' \
  --attention-backend flash_attn \
  --max-num-batched-tokens 32768
```

### SGLang Serving

Start an SGLang server with DFlash for Qwen3.5-35B-A3B:

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-35B-A3B \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path z-lab/Qwen3.5-35B-A3B-DFlash \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --attention-backend trtllm_mha \
    --speculative-draft-attention-backend fa4 \
    --mem-fraction-static 0.75 \
    --mamba-scheduler-strategy extra_buffer \
    --trust-remote-code
```

### Transformers (Python API)

Use DFlash directly with HuggingFace Transformers for Qwen3 and LLaMA-3.1 models:

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

draft = AutoModel.from_pretrained(
    "z-lab/Qwen3-8B-DFlash-b16",
    trust_remote_code=True,
    dtype="auto",
    device_map="cuda:0",
).eval()

target = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype="auto",
    device_map="cuda:0",
).eval()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
input_ids = tokenizer.apply_chat_template(
    messages, return_tensors="pt",
    add_generation_prompt=True,
    enable_thinking=False,
).to(draft.device)

output = draft.spec_generate(
    input_ids=input_ids,
    max_new_tokens=2048,
    temperature=0.0,
    target=target,
    stop_token_ids=[tokenizer.eos_token_id],
)
print(tokenizer.decode(output[0], skip_special_tokens=False))
```

### MLX (Apple Silicon)

Run DFlash on Apple Silicon with streaming output:

```python
from dflash.model_mlx import load, load_draft, stream_generate

model, tokenizer = load("Qwen/Qwen3.5-4B")
draft = load_draft("z-lab/Qwen3.5-4B-DFlash", sliding_window_size=None)

messages = [{"role": "user", "content": "How many positive whole-number divisors does 196 have?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

tps = 0.0
for r in stream_generate(model, draft, tokenizer, prompt, block_size=16, max_tokens=2048, temperature=0.6):
    print(r.text, end="", flush=True)
    tps = r.generation_tps
print(f"\nThroughput: {tps:.2f} tok/s")
```

For ultra-long-context or agentic use cases, bound the draft KV history with a sliding window:

```python
draft = load_draft("z-lab/Qwen3.5-4B-DFlash", sliding_window_size=4096)
```

## Supported Models

DFlash provides 14+ pre-trained draft models on HuggingFace, with more on the way:

| Target Model | DFlash Draft Model | Status |
|---|---|---|
| Qwen3.6-35B-A3B (Preview) | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) | Available |
| Kimi-K2.5 | [z-lab/Kimi-K2.5-DFlash](https://huggingface.co/z-lab/Kimi-K2.5-DFlash) | Available |
| Qwen3.5-4B | [z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) | Available |
| Qwen3.5-9B | [z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) | Available |
| Qwen3.5-27B | [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash) | Available |
| Qwen3.5-35B-A3B | [z-lab/Qwen3.5-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash) | Available |
| Qwen3-Coder-Next | [z-lab/Qwen3-Coder-Next-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-Next-DFlash) | Available |
| Qwen3-Coder-30B-A3B | [z-lab/Qwen3-Coder-30B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3-Coder-30B-A3B-DFlash) | Available |
| gpt-oss-20b | [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) | Available |
| gpt-oss-120b | [z-lab/gpt-oss-120b-DFlash](https://huggingface.co/z-lab/gpt-oss-120b-DFlash) | Available |
| Qwen3-4B (non-thinking) | [z-lab/Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) | Available |
| Qwen3-8B (non-thinking) | [z-lab/Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16) | Available |
| LLaMA-3.1-8B-Instruct | [z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat) | Available |
| Qwen3.5-122B-A10B | -- | Coming Soon |
| Qwen3.5-397B-A17B | -- | Coming Soon |
| GLM-5.1 | -- | Coming Soon |

The training recipe will be open-sourced soon, enabling users to train custom DFlash draft models for any LLM.

## Benchmarking

DFlash includes a built-in benchmarking suite that evaluates performance across five standard datasets: GSM8K, MATH-500, HumanEval, MBPP, and MT-Bench. Datasets are automatically downloaded and cached as JSONL on first run.

**vLLM Benchmark**

```bash
python -m dflash.benchmark --backend vllm \
    --base-url http://127.0.0.1:8000 --model Qwen/Qwen3.5-27B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --enable-thinking
```

**SGLang Benchmark**

```bash
python -m dflash.benchmark --backend sglang \
    --base-url http://127.0.0.1:30000 --model Qwen/Qwen3.5-35B-A3B \
    --dataset gsm8k --num-prompts 128 --concurrency 1 --enable-thinking
```

**Transformers Benchmark (Multi-GPU)**

```bash
torchrun --nproc_per_node=8 -m dflash.benchmark --backend transformers \
    --model Qwen/Qwen3-8B --draft-model z-lab/Qwen3-8B-DFlash-b16 \
    --dataset gsm8k --max-samples 128
```

**MLX Benchmark**

```bash
python -m dflash.benchmark --backend mlx \
    --model Qwen/Qwen3.5-4B --draft-model z-lab/Qwen3.5-4B-DFlash \
    --dataset gsm8k --max-samples 128 --enable-thinking \
    --draft-sliding-window-size 4096
```

The benchmarking output reports baseline throughput (autoregressive), DFlash throughput, decoding speedup ratio, and average acceptance length. The acceptance length histogram provides insight into how often the draft model's proposals match the target distribution at each position within the block.

## Key Technical Details

**Context Feature Projection**

The `extract_context_feature` function selects hidden states from specific target model layers and concatenates them. The `build_target_layer_ids` function distributes layer selections evenly across the target model's depth. For a single-layer draft model, it selects the middle layer; for multi-layer drafts, it spans from layer 1 to layer N-3, avoiding the very first and last layers which tend to carry less task-specific information.

**GatedDeltaNet Support (MLX)**

Models like Qwen3.5 use GatedDeltaNet (GDN) layers instead of standard attention. DFlash's MLX backend handles this through a monkey-patching mechanism: the `_GDNStateCapture` class temporarily replaces GDN layer `__call__` methods with capturing wrappers that intercept internal states. This allows the draft model to extract context features from GDN-based models without modifying the original model code.

**Experimental Sliding Window**

For ultra-long-context or agentic use cases where the KV cache grows unboundedly, DFlash offers an experimental `sliding_window_size` option. When enabled, the draft model uses `RotatingKVCache` instead of standard `KVCache`, bounding the committed draft KV history to a fixed window size. This trades off some acceptance rate for predictable memory usage, which is critical for long-running agent loops.

**Model Binding (MLX)**

On Apple Silicon, the MLX backend shares the `embed_tokens` and `lm_head` layers between the target and draft models via the `bind` method. This reduces memory overhead by avoiding duplicate copies of these large embedding matrices, which is especially valuable on devices with unified memory architecture.

## Conclusion

DFlash represents a significant advance in speculative decoding for LLMs. By replacing the traditional autoregressive draft model with a block diffusion model, it achieves higher parallelism and better acceptance rates. The cross-architecture attention mechanism elegantly bridges the target and draft models, while the multi-backend support ensures that DFlash can be deployed in virtually any inference environment.

With 14+ pre-trained draft models, support for four major inference backends, and a comprehensive benchmarking suite, DFlash is ready for production use today. The upcoming training recipe will further expand its applicability, enabling custom draft models for any target LLM.

**Links:**

- **GitHub:** [https://github.com/z-lab/dflash](https://github.com/z-lab/dflash)
- **Paper:** [arXiv:2602.06036](https://arxiv.org/abs/2602.06036)
- **Blog:** [https://z-lab.ai/projects/dflash/](https://z-lab.ai/projects/dflash/)
- **Models:** [HuggingFace Collection](https://huggingface.co/collections/z-lab/dflash)
- **License:** MIT

**Citation:**

```bibtex
@article{chen2026dflash,
  title   = {{DFlash: Block Diffusion for Flash Speculative Decoding}},
  author  = {Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal = {arXiv preprint arXiv:2602.06036},
  year    = {2026}
}
```

## Related Posts

- [BitNet: 1-bit LLMs for Efficient Inference](/BitNet-1-bit-LLMs-Efficient-Inference/)
- [TimesFM: Time Series Foundation Model by Google Research](/TimesFM-Time-Series-Foundation-Model/)
- [Harbor: Open Source Container Registry](/Harbor-Open-Source-Container-Registry/)