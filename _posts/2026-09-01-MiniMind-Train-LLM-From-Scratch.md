---
layout: post
title: "MiniMind: Train a 64M-Parameter LLM From Scratch in 2 Hours"
description: "Learn how MiniMind implements the complete LLM training pipeline - from tokenizer training through pretraining, SFT, LoRA, DPO, and GRPO/Agentic RL - using pure PyTorch with a Qwen3-aligned architecture."
date: 2026-09-01
header-img: "img/post-bg.jpg"
permalink: /MiniMind-Train-LLM-From-Scratch/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - LLM
  - PyTorch
  - Tutorial
author: "PyShine"
---

# MiniMind: Train a 64M-Parameter LLM From Scratch in 2 Hours

Most introductions to large language models stop at calling a `transformers` API: load a pretrained checkpoint, attach a LoRA adapter, and fine-tune on a small dataset. While useful, this skips over the actual mechanics of how a transformer learns. MiniMind, an open-source project by jingyaogong, takes the opposite approach. It trains a roughly 64M-parameter language model from scratch in about two hours on a single NVIDIA 3090 GPU, with every core algorithm implemented in pure PyTorch - no `transformers` training wrappers, no `peft` abstractions, no `trl` recipe boilerplate.

The repository covers the complete LLM lifecycle: tokenizer training, pretraining, supervised fine-tuning, LoRA, DPO, RLAIF (PPO / GRPO / CISPO), tool use, Agentic RL, adaptive thinking, and white-box distillation. The architecture is aligned with the Qwen3 / Qwen3-MoE ecosystem, supports both Dense and Mixture-of-Experts variants, and interoperates with `llama.cpp`, `vllm`, and `ollama` for inference. This article walks through MiniMind's architecture, training pipeline, and the implementation choices that make a from-scratch LLM tractable on consumer hardware.

## Project Overview

MiniMind is released under the Apache 2.0 license and the current main line is `minimind-3` (released 2026-04-01). The Dense variant has approximately 64M parameters and the MoE variant has 198M total parameters with 64M active per token. For context, the smallest minimind-3 is roughly 1/2700th the size of GPT-3, which is what makes end-to-end training feasible on a single GPU.

Key facts at a glance:

| Property | Value |
|----------|-------|
| Repository | `jingyaogong/minimind` |
| License | Apache 2.0 |
| Architecture | Qwen3-aligned (Dense + MoE) |
| Parameters (Dense) | 64M |
| Parameters (MoE) | 198M total / 64M active |
| Hidden size | 768 |
| Layers | 8 |
| Attention heads | 8 Q / 4 KV (GQA) |
| Vocabulary | 6400 tokens |
| Context length | up to 32768 (YaRN scaling) |
| Framework | Pure PyTorch (transformers for I/O only) |

## Architecture

The MiniMind model is a standard decoder-only transformer with the modern features you would expect from a 2025-era design: Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE) with YaRN long-context extension, RMSNorm with Q/K normalization, SwiGLU feed-forward networks, and tied input/output embeddings. The MoE variant replaces the dense FFN with a Top-K routing layer over four SwiGLU experts.

![MiniMind Transformer Architecture](/assets/img/diagrams/minimind/minimind-architecture.svg)

### Understanding the Architecture

The diagram above shows the high-level data flow through the model, along with the internal structure of a single `MiniMindBlock`. Each component is described below.

**Input Tokens and Embedding Layer**
The model accepts token IDs from a 6400-token vocabulary trained with a custom BPE + ByteLevel tokenizer. An `nn.Embedding` layer projects each token to a 768-dimensional vector. The embedding matrix is tied to the LM head weights (`tie_word_embeddings=True`), which both reduces parameter count and often improves generalization on small models. This is the same tying strategy used in modern Qwen, Gemma, and Llama models.

**Stack of MiniMindBlocks**
The backbone is a stack of 8 identical `MiniMindBlock` modules. Each block follows the now-standard pre-norm transformer pattern: normalize, attend, residual add, normalize, feed-forward, residual add. The residual connections are critical for stability when training from scratch because they provide a gradient highway that bypasses the nonlinear transformations.

**Inside a MiniMindBlock**
The block expands into two sub-modules separated by residual connections. The first sub-module is Grouped Query Attention: 8 query heads share 4 key/value heads via `repeat_kv`, reducing the KV cache size by half compared to multi-head attention while preserving quality. RoPE is applied to queries and keys after a per-head RMSNorm (`q_norm` / `k_norm`), a trick popularized by Qwen2 that stabilizes attention logits at long sequence lengths. Flash Attention (via `torch.nn.functional.scaled_dot_product_attention`) is used whenever the sequence length is greater than 1 and no padding mask is required.

The second sub-module is the feed-forward network. The Dense variant uses a single SwiGLU FFN: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`. The MoE variant replaces this with a `MOEFeedForward` module that routes each token to one of four experts. The intermediate size is derived as `ceil(hidden_size * pi / 64) * 64`, a neat trick that approximates the 4x expansion ratio used in larger models while staying multiple-of-64 for hardware efficiency.

**Final RMSNorm and LM Head**
After the last block, a final RMSNorm normalizes the hidden states and the tied LM head projects them back to vocabulary logits. Cross-entropy loss is computed with label shifting (`logits[..., :-1]` vs `labels[..., 1:]`) and `ignore_index=-100` to skip padding tokens.

**Key Insights**
The architecture deliberately mirrors Qwen3's structure rather than reinventing it. This means weights can be mapped into `transformers` checkpoints and served by `llama.cpp`, `vllm`, and `ollama` without modification - a practical decision that trades a bit of novelty for substantial ecosystem compatibility. The `generate` method inherits from `transformers.GenerationMixin`, so the same sampling parameters (temperature, top-k, top-p, repetition penalty) work as users expect.

### RoPE and YaRN Long-Context Extension

MiniMind precomputes cosine and sine frequency tables at initialization using a base frequency of 1,000,000 and supports up to 32768 positions. When `inference_rope_scaling` is enabled, the YaRN algorithm rescales frequencies outside the original 2048-token training window using a linear ramp between `beta_fast=32` and `beta_slow=1`. This is the same mechanism used by Qwen2.5 and Mistral to extend context without retraining.

## The Full Training Pipeline

MiniMind's distinguishing feature is that it ships working training scripts for every stage of LLM development. Each script is self-contained, uses the same model and dataset abstractions, and can be run on a single GPU or distributed across multiple GPUs with DDP and DeepSpeed.

![MiniMind Training Pipeline](/assets/img/diagrams/minimind/minimind-training-pipeline.svg)

### Understanding the Training Pipeline

The pipeline diagram traces the journey from raw data to a final aligned model. Each stage is described in detail below.

**1. Data Preparation**
MiniMind ships cleaned and deduplicated datasets for every stage: `pretrain_t2t.jsonl` for pretraining, `sft_t2t.jsonl` for supervised fine-tuning, `rlaif.jsonl` for RLHF, and `agent_rl.jsonl` / `agent_rl_math.jsonl` for Agentic RL. The data pipeline uses `datasketch` for MinHash deduplication, `jieba` for Chinese tokenization, and `simhash` for near-duplicate detection. This mirrors how production LLM teams clean pretraining corpora.

**2. Tokenizer Training**
The tokenizer is a BPE + ByteLevel model trained with a 6400-token vocabulary. The chat template uses special tokens (`<|im_start|>`, `<|im_end|>`) for conversation boundaries and dedicated tokens for tool calls (`<tool_call>`, `</tool_call>`) and reasoning (`<think>`, `</think>`). Buffer tokens are reserved for future extensions.

**3. Pretraining (`train_pretrain.py`)**
Pretraining uses next-token prediction with cross-entropy loss. The script supports mixed-precision training (bfloat16 or float16) with `torch.cuda.amp.GradScaler`, gradient accumulation, gradient clipping (`grad_clip=1.0`), and a custom cosine learning-rate scheduler with warmup. The MoE auxiliary loss (load-balancing) is added to the language modeling loss during pretraining of the MoE variant. Checkpoints are saved as half-precision state dictionaries to minimize disk usage.

**4. Supervised Fine-Tuning (`train_full_sft.py`)**
SFT uses the same training loop as pretraining but with a different dataset and loss masking strategy. The `SFTDataset` class applies the chat template via `tokenizer.apply_chat_template` and masks labels so that only assistant responses contribute to the loss. System prompts are randomly injected with a 20% probability to improve robustness. Tool-call capabilities are mixed into the main SFT data, so a single SFT run yields a model that can already perform basic tool use.

**5. LoRA Fine-Tuning (`train_lora.py`)**
The LoRA implementation is written from scratch rather than using `peft`. The `LoRA` module wraps an `nn.Linear` with two low-rank matrices A (Gaussian init) and B (zero init), ensuring the adapter starts as an identity transformation. The adapter is attached only to square Linear layers (where `in_features == out_features`), which corresponds to the attention projections. Training optimizes only the LoRA parameters; the base model weights are frozen.

**6. DPO (`train_dpo.py`)**
Direct Preference Optimization is implemented natively in PyTorch. The loss function `dpo_loss` computes the difference between chosen and rejected log-probabilities, subtracts reference-model log-probabilities, and applies the sigmoid-based loss `-F.logsigmoid(beta * logits)`. This is the standard DPO objective - no reward model is needed because the implicit reward is the policy itself.

**7. RLAIF: PPO, GRPO, CISPO**
MiniMind implements three on-policy RL algorithms from scratch. The GRPO trainer (`train_grpo.py`) is particularly instructive: it generates G responses per prompt using a decoupled rollout engine, computes a group-relative advantage (reward minus the group mean), and optimizes a clipped surrogate objective with a KL penalty against a frozen reference model. The reference model also serves as the reward model - a pragmatic design choice that simplifies the training stack.

**8. Agentic RL (`train_agent.py`)**
This is the most advanced stage. The agent is trained to use tools (calculator, weather, time, currency converter, translator) across multi-turn conversations. Reward signals include tool-call format validity, argument correctness, answer correctness, length penalties, and repetition penalties. The rollout engine is abstracted behind a `RolloutEngine` interface with a PyTorch implementation and an optional `sglang` backend for faster inference.

**9. Distillation (`train_distillation.py`)**
White-box distillation transfers knowledge from a larger teacher model to MiniMind by matching hidden states and logits. This is how the minimind2-DeepSeek-R1 distilled models were created.

**Key Insights**
The pipeline is deliberately ordered so that each stage builds on the previous one: pretraining produces a base model with language understanding, SFT adds instruction following, DPO/RL aligns with human preferences, and Agentic RL adds tool-use capabilities. Skipping stages (e.g., going straight from pretrain to RL) tends to produce unstable training because the policy has no prior on what a reasonable response looks like.

## Mixture of Experts Routing

The MoE variant (`use_moe=True`) replaces the dense SwiGLU FFN with a Top-1 routing layer over four experts. This is the same design used by Qwen3-MoE and Mixtral, scaled down.

![MiniMind MoE Routing](/assets/img/diagrams/minimind/minimind-moe-routing.svg)

### Understanding MoE Routing

**Router / Gate**
A single linear layer (`Linear(768, 4)`) produces logits for each of the four experts. The logits are passed through softmax to produce a probability distribution, and `torch.topk` selects the highest-scoring expert per token. With `num_experts_per_tok=1`, each token is routed to exactly one expert, which keeps inference FLOPs close to the dense model.

**Weight Normalization**
After Top-K selection, the routing weights are renormalized to sum to 1 (`norm_topk_prob=True`). This is important for numerical stability - without it, a low-probability expert selection could be effectively ignored.

**Expert Computation**
Each expert is a standard `FeedForward` (SwiGLU) module with its own `gate_proj`, `up_proj`, and `down_proj`. The implementation is efficient: instead of running all experts and masking, it uses `index_add_` to scatter the weighted expert outputs back to the original token positions. Tokens not routed to a given expert are skipped entirely.

**Auxiliary Load-Balancing Loss**
A critical problem in MoE training is router collapse, where the gate learns to send all tokens to a single expert. MiniMind addresses this with an auxiliary loss: the average load per expert is multiplied by the average router probability per expert, and the sum is scaled by `router_aux_loss_coef=5e-4`. This term encourages balanced routing and is added to the main language modeling loss during training.

**Key Insights**
The MoE design follows the principle that sparse expert models can scale parameters without scaling FLOPs. MiniMind's 198M-A64M model has three times the parameters of the dense variant but uses roughly the same compute per token. The trade-off is memory: all four expert weights must be resident in GPU memory even though only one is active per token. This makes MoE particularly suitable for inference-floored but memory-rich deployments.

## GRPO Reinforcement Learning Flow

GRPO (Group Relative Policy Optimization) is the algorithm behind DeepSeek-R1 and is one of the most popular RL methods for LLMs in 2025. MiniMind implements it from scratch with a clean, decoupled rollout engine.

![MiniMind GRPO Flow](/assets/img/diagrams/minimind/minimind-grpo-flow.svg)

### Understanding the GRPO Flow

**Prompt Batch and Rollout**
Training starts with a batch of B prompts. For each prompt, the rollout engine generates G candidate responses (typically 4-8) by sampling from the current policy. The `TorchRolloutEngine` uses the model's `generate` method directly; an alternative `SGLangRolloutEngine` connects to a running sglang server for faster batched inference.

**Reward Computation**
Rewards are a combination of rule-based and model-based signals:
- Length reward: +0.5 if the response is between 20 and 800 characters, -0.5 otherwise
- Thinking format reward: +1.0 if the `<think>...</think>` block is between 20 and 300 characters
- Single-thinking reward: +0.25 if exactly one `</think>` token appears
- Repetition penalty: subtracts a normalized trigram repetition score
- Reward model score: a separate `LMForRewardModel` provides a learned scalar reward

The reward model is built on the same MiniMind architecture - a pragmatic choice that avoids pulling in a large external reward model.

**Group-Relative Advantage**
Unlike PPO, which estimates a value baseline with a critic network, GRPO computes the advantage as `reward_i - mean(rewards_in_group)`. This eliminates the need for a value function and substantially reduces memory usage. The advantage is computed per group of G responses to the same prompt.

**Policy and Reference Log-Probabilities**
For each generated response, the policy model and a frozen reference model both compute per-token log-probabilities. The `compute_per_token_logps` function uses `logits_to_keep` to avoid recomputing logits for the prompt portion, which is a significant optimization.

**Clipped Surrogate Loss**
The ratio `pi_theta(a|s) / pi_ref(a|s)` is clipped to a range (typically 0.8 to 1.2) and multiplied by the advantage. A KL divergence penalty between the policy and reference model provides additional regularization. The total loss is `clip_loss + beta * KL + aux_loss`.

**Backpropagation and Policy Update**
Gradients are computed only with respect to the policy model parameters (the reference model is frozen). The implementation uses `DistributedDataParallel` for multi-GPU training and supports `torch.compile` for additional speedup.

**Key Insights**
GRPO's key advantage over PPO is the removal of the value function. This eliminates an entire model worth of memory and compute, which matters enormously when you are training on a single 3090. The trade-off is higher variance in the advantage estimate, which GRPO mitigates by averaging over the group of G responses. MiniMind's implementation also demonstrates a practical pattern: the reward model and reference model can be the same checkpoint, which halves the memory footprint of the RL training stack.

## Installation and Quick Start

The repository is structured for direct cloning and immediate training.

```bash
# Clone the repository
git clone https://github.com/jingyaogong/minimind.git
cd minimind

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt
```

The key dependencies are:

| Dependency | Version | Purpose |
|------------|---------|---------|
| `torch` | 2.6.0+ | Core training framework |
| `transformers` | 4.57.6 | Tokenizer and model I/O |
| `datasets` | 3.6.0 | Data loading |
| `swanlab` | 0.9.8 | Training visualization (drop-in wandb replacement) |
| `streamlit` | 1.50.0 | Web demo UI |
| `sentencepiece` | 0.2.0 | Tokenizer training |

Note that `torch` is commented out in `requirements.txt` because the correct version depends on your CUDA setup. Install PyTorch separately from the official wheel index matching your CUDA version.

## Training a Model From Scratch

The fastest path to a trained model is:

```bash
# Step 1: Pretrain (the "2 hours on a 3090" claim)
cd trainer
python train_pretrain.py \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --save_dir ../out

# Step 2: Supervised fine-tuning
python train_full_sft.py \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --from_weight pretrain \
    --save_dir ../out

# Step 3 (optional): LoRA fine-tuning on a custom task
python train_lora.py \
    --epochs 1 \
    --batch_size 16 \
    --data_path ../dataset/sft_t2t_mini.jsonl \
    --from_weight full_sft \
    --save_dir ../out
```

For multi-GPU training, launch with `torchrun`:

```bash
torchrun --nproc_per_node=4 trainer/train_pretrain.py \
    --epochs 1 \
    --batch_size 8 \
    --data_path ../dataset/pretrain_t2t_mini.jsonl
```

The scripts automatically detect distributed mode via `init_distributed_mode()` and wrap the model in `DistributedDataParallel`. DeepSpeed is supported for larger configurations.

## Inference and Serving

MiniMind provides three inference paths.

### Web Demo (Streamlit)

```bash
cd scripts
streamlit run web_demo.py
```

The web UI supports thinking display, tool selection, and multi-turn tool calls.

### OpenAI-Compatible API Server

```bash
cd scripts
python serve_openai_api.py
```

This starts a Flask server that implements the OpenAI Chat Completions API, including support for `reasoning_content`, `tool_calls`, and `open_thinking` fields. It can be plugged directly into Open-WebUI, FastGPT, or any other OpenAI-compatible client.

### Programmatic Chat

```bash
cd scripts
python chat_api.py
```

This is a minimal interactive REPL that shows the chat template handling, thinking tokens, and tool-call parsing in a few dozen lines.

### Third-Party Inference Engines

Because the model is `transformers`-compatible, you can export and serve it with production-grade engines:

```bash
# llama.cpp (GGUF quantization)
python convert_model.py --model_path ../out/pretrain_768.pth --output minimind-3
llama-cli -m minimind-3.gguf -p "Hello, who are you?"

# vLLM (high-throughput serving)
vllm serve minimind-3 --dtype bfloat16

# ollama
ollama create minimind -f Modelfile
ollama run minimind
```

## Evaluation

MiniMind includes an evaluation harness that runs the model on standard benchmarks:

```bash
python eval_llm.py --model_path ../out/full_sft_768.pth
```

Supported benchmarks include C-Eval, C-MMLU, and OpenBookQA. Results are written to a JSON report and can be visualized with the included radar chart generator. YaRN-based long-context evaluation is also supported for testing RoPE extrapolation.

## Troubleshooting

**Out of memory during pretraining**
Reduce `batch_size` to 8 or 4 and increase `accumulation_steps` to 4 or 8. The effective batch size is `batch_size * accumulation_steps`, which preserves training dynamics while fitting in memory.

**Tokenizer loading errors**
Ensure `model/tokenizer.json` and `model/tokenizer_config.json` are on the Python path. The tokenizer is loaded via `AutoTokenizer.from_pretrained('model')` - if you run from a different directory, pass the full path.

**NaN loss during RL training**
Reduce the learning rate (RL typically uses 1e-6 to 5e-6) and ensure the reference model is frozen. If the KL penalty coefficient is too low, the policy can drift away from the reference and collapse.

**MoE training imbalance**
If the auxiliary loss is not decreasing, increase `router_aux_loss_coef` from `5e-4` to `1e-3`. Also check that `norm_topk_prob=True` - without normalization, routing can become unstable.

**Slow generation on CPU**
Use KV caching (`use_cache=True`, the default) and consider exporting to GGUF via `convert_model.py` for llama.cpp-based CPU inference, which is typically 5-10x faster than the native PyTorch path on CPU.

**Checkpoint resume fails**
The `lm_checkpoint` utility saves optimizer state, scaler state, epoch, step, and wandb ID. If you change the model configuration (e.g., `hidden_size`), the checkpoint will not load - start a fresh run or delete the checkpoint directory.

## Conclusion

MiniMind is a remarkable educational project because it refuses to hide the mechanics of LLM training behind abstractions. Every algorithm - from RMSNorm to GRPO - is implemented in plain PyTorch, readable in an afternoon, and trainable on a single consumer GPU. Whether you want to understand how RoPE works, see a real MoE router in action, or experiment with Agentic RL without renting an H100 cluster, MiniMind provides a complete, working reference.

The fact that the architecture is aligned with Qwen3 means that the lessons learned here transfer directly to production-scale models. The same code patterns - GQA with q/k_norm, SwiGLU FFN, YaRN RoPE scaling, Top-K MoE routing, clipped surrogate RL objectives - appear in the largest open-weight models being trained today. MiniMind just makes them small enough to actually run.

## Related Posts

- [yaml-cpp: A YAML Parser and Emitter in C++](/yaml-cpp-YAML-Parser-Emitter-Cpp/)
- [Catch2: A Natural C++ Testing Framework](/Catch2-Natural-Cpp-Testing-Framework/)
- [meshoptimizer: Making 3D Meshes Smaller and Faster](/meshoptimizer-Mesh-Optimization-Library/)

## Links

- [GitHub Repository: jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- [Hugging Face Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- [ModelScope Profile](https://www.modelscope.cn/profile/gongjy)
