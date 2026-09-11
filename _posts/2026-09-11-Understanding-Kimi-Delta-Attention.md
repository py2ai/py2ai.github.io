---
layout: post
title: "Understanding Kimi Delta Attention: The Linear Attention Powering Kimi K3"
description: "Kimi Delta Attention (KDA) is Moonshot AI's linear attention module that extends Gated DeltaNet with fine-grained, channel-wise diagonal gating and a specialized DPLR chunkwise algorithm. It keeps a fixed-size recurrent state instead of a growing KV cache, and when layered in a 3:1 hybrid with full Multi-Head Latent Attention it outperforms full attention while cutting KV cache by 75% and lifting decoding throughput 6x at a 1M-token context. KDA is the backbone of both Kimi Linear (48B/3B activated) and the 2.8T-parameter Kimi K3. This post breaks down the recurrence formula, the hybrid stack, the K3 architecture, and the chunkwise DPLR algorithm that makes it fast."
date: 2026-09-11
header-img: "img/post-bg.jpg"
permalink: /Understanding-Kimi-Delta-Attention/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Kimi Delta Attention
  - Linear Attention
  - Moonshot AI
  - Kimi K3
  - LLM Architecture
  - Deep Learning
  - Open Source
author: PyShine
---

## What is Kimi Delta Attention

Kimi Delta Attention (KDA) is a linear attention mechanism introduced by Moonshot AI in the [Kimi Linear technical report](https://arxiv.org/abs/2510.26692). It is a refinement of [Gated DeltaNet](https://arxiv.org/abs/2412.06464) that replaces the coarse, per-head scalar decay gate with a **fine-grained, channel-wise diagonal gate**. The result is a recurrent state that can selectively forget and retain information along each feature dimension independently, which makes far better use of a fixed-size memory matrix.

Two properties make KDA stand out. First, it is the first linear attention variant to **outperform full softmax attention under fair comparisons** across short-context, long-context, and reinforcement-learning regimes. Second, because it is a recurrent operator with a constant-size state, its memory does not grow with sequence length, so it drops the linearly expanding key-value (KV) cache that dominates cost in long-context inference.

KDA is not just a research module. It is the workhorse of two shipped models: **Kimi Linear** (48B total / 3B activated, 1M context, MIT-licensed weights on [Hugging Face](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)) and the **2.8-trillion-parameter Kimi K3**, where 69 of the 93 transformer layers are KDA layers. The KDA kernel is open-sourced in the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda) library.

## The KDA Recurrence: Scalar Gate vs Channel-wise Gate

The cleanest way to understand KDA is to write its state update next to the Gated DeltaNet baseline it extends.

Gated DeltaNet keeps a memory matrix `S_t` and updates it with a rank-1 write and a scalar decay:

```text
S_t = alpha_t * (I - beta_t k_t k_t^T) S_(t-1) + beta_t k_t v_t^T
```

Here `alpha_t` is a single scalar, so the entire state decays at one rate per head. Kimi Delta Attention replaces that scalar with a diagonal matrix:

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_(t-1) + beta_t k_t v_t^T
```

`Diag(alpha_t)` is a per-channel gate: every feature dimension now has its own forgetting rate. The diagram below traces how a token's key, value, learning rate, and gate flow through the update.

![KDA recurrence mechanism](/assets/img/diagrams/kimi-linear/kda-recurrence-mechanism.svg)

### Understanding the Recurrence

**The alpha gate - the core innovation.** In Gated DeltaNet the decay `alpha_t` is shared across all channels of a head, which is a blunt instrument: a single rate must simultaneously let some features persist and let others fade. KDA's `Diag(alpha_t)` makes the gate a vector the same width as the head dimension, so each channel independently decides how much of the previous state to keep. This is the difference between one volume knob and a full graphic equalizer. The network learns, per channel, how sticky or forgetful each slice of memory should be, which is what lets a small fixed-size recurrent state hold useful information over very long horizons.

**The delta rule as online gradient descent.** The term `(I - beta_t k_t k_t^T) S_(t-1) + beta_t k_t v_t^T` is the classic delta rule. It can be read as a rank-1 online gradient step that minimizes a reconstruction loss between the stored value associated with key `k_t` and the new value `v_t`. KDA keeps this rule intact, which is important: it preserves the well-understood stability and convergence behavior of the delta rule while only changing the decay structure around it. This is why the authors describe KDA as "more consistent with the classical delta rule" than other linear-attention variants.

**The role of beta.** `beta_t` is a learned learning rate that controls how aggressively the new write `k_t v_t^T` overwrites the existing state along the direction of `k_t`. A larger beta means the new token's value takes precedence; the `(I - beta_t k_t k_t^T)` factor first erases the projection of the old state onto `k_t` so the write does not simply add on top of stale content.

**Reading the output.** The output at step `t` is produced by querying the updated state with the same key: `o_t = S_t k_t`. Because `S_t` is a matrix of fixed shape (head dimension by head dimension) rather than a list of past tokens, this read costs the same work regardless of how long the sequence is. That is the structural reason KDA's memory stays constant as context grows to a million tokens.

**Why fine-grained gating matters in practice.** Real text mixes many kinds of information that should age at different rates: syntax that is relevant for one clause, entity identity that must persist for the whole document, and numeric facts that need to survive until a question references them. A single scalar decay forces a compromise. Per-channel decay lets the model allocate sticky channels to persistent facts and ephemeral channels to local context, which is the mechanism behind the reported quality gains over both Gated DeltaNet and full attention.

## The Hybrid 3:1 Stack

KDA alone has one weakness shared by all linear-attention RNNs: a finite-state memory cannot, in theory, retrieve an arbitrary token from the distant past with the precision that full attention can. Moonshot AI's answer is not to pick one mechanism but to **interleave** them. Kimi Linear stacks KDA layers and Multi-Head Latent Attention (MLA) layers in a 3:1 ratio: every three KDA layers are followed by one full-attention MLA layer.

![Kimi Linear hybrid stack](/assets/img/diagrams/kimi-linear/kimi-linear-hybrid-stack.svg)

### Understanding the Hybrid Architecture

**The 27-layer layout.** Kimi Linear's base model has 27 transformer layers. Twenty of them are KDA layers and seven are Gated MLA layers, placed at layers 4, 8, 12, 16, 20, 24, and 27. That placement is deliberate: the full-attention layers are evenly spaced so that every block of KDA recurrence is never more than three layers away from a global-attention calibration step. The model gets the throughput of linear layers for the bulk of its compute and the exact global recall of full attention at regular checkpoints.

**Why 3:1 and not 1:1 or 7:1.** The report includes ablations on this ratio. A KDA-heavy stack like 7:1 trains well (the loss drops nicely) but generalizes worse on held-out evaluation, which is the classic signature of a recurrent model overfitting its finite state. A 1:1 ratio is stable but throws away most of the efficiency benefit because half the layers are full attention. Three-to-one is the Pareto point: enough full-attention layers to anchor global retrieval, with enough KDA layers to dominate the cost and memory profile.

**No positional encoding on the MLA layers.** Kimi Linear removes all explicit positional encodings (RoPE or otherwise) from the full-attention layers. Positional and ordering information is carried entirely by the KDA layers, whose recurrent structure is inherently order-aware. The report shows that this NoPE-plus-KDA combination extrapolates to long sequences more robustly than RoPE-augmented full attention, because the model never has to reason about out-of-distribution rotation angles.

**The KV cache payoff.** This is where the hybrid earns its keep. Full attention stores a key and a value for every token seen, so the KV cache grows linearly with sequence length and becomes the dominant memory cost at long context. KDA stores only a fixed recurrent state per layer. With a 3:1 split, three quarters of the layers never grow their memory, and the measured reduction at 1M tokens is about 75% less KV cache than an equivalent full-MLA model. At decode time this translates to roughly 6x higher throughput for a million-token context.

## Scaling to Kimi K3

KDA graduates from a 48B research model to the production backbone of **Kimi K3**, a 2.8-trillion-parameter Mixture-of-Experts model with a 1M-token context window. K3 keeps the same hybrid philosophy but scales every axis: depth, width, expert count, and the role of KDA within the stack.

![Kimi K3 architecture](/assets/img/diagrams/kimi-linear/kimi-k3-architecture.svg)

### Understanding the Kimi K3 Architecture

**Layer composition.** K3 has 93 transformer layers: 69 KDA layers, 24 Gated MLA layers, and 1 dense layer. That is roughly the same 3:1 KDA-to-MLA rhythm as Kimi Linear, scaled to frontier depth. The attention is wide (96 heads, attention hidden dimension 7168) and the MoE is extremely sparse: of 896 routed experts, only 16 fire per token, plus one shared expert. With 104B activated parameters out of 2.8T, K3 keeps its per-token compute tractable while spreading capacity across a huge expert pool.

**Stable LatentMoE.** The expert block is a LatentMoE that operates in a compressed latent space rather than the full hidden dimension. To stabilize optimization at 16-of-896 sparsity, Moonshot AI pairs it with SiT Unit GLU activations and a Quantile Balancing loss that keeps expert load evenly distributed. The combination is what the team calls Stable LatentMoE, and it is what makes extreme sparsity trainable without routing collapse.

**Attention Residuals (AttnRes).** K3 adds a second information-flow innovation alongside KDA. AttnRes replaces the standard single-source residual stream with a learned mixing operation: each layer can attend to a weighted combination of representations from all preceding layers rather than only the immediately previous one. Where KDA improves information flow across the sequence (the time/sequence axis), AttnRes improves information flow across depth (the layer axis). The two are complementary and are both credited with the reported 2.5x scaling-efficiency gain over Kimi K2.

**Inside one KDA block.** A single KDA block is Norm, a Linear projection, a ShortConvolution (a short causal depthwise convolution that injects local order bias), and then the KDA recurrence, followed by the Stable LatentMoE expert block. The ShortConvolution is part of why the full-attention layers can safely drop positional encoding: local ordering is handled by the conv, long-range recurrence by KDA, and global recall by the periodic MLA layers.

**Deployment and scale.** K3 is served on vLLM with a single hybrid KV-cache manager that holds both the fixed recurrent state of KDA layers and the growing KV cache of MLA layers side by side. At 1M tokens this mixed manager is what makes the context affordable; the [vLLM day-0 support post](https://vllm.ai/blog/2026-07-27-k3) describes the prefill, decode, and speculative-decoding paths built specifically for KDA. Full weights are released on [Hugging Face](https://huggingface.co/moonshotai/Kimi-K3).

## The Chunkwise DPLR Algorithm

A recurrent rule is only useful if it can be computed efficiently. KDA's state update is sequential by nature, which would make prefill of long prompts slow. Moonshot AI solves this with a bespoke **chunkwise parallel algorithm** built on a specialized Diagonal-Plus-Low-Rank (DPLR) parameterization of the transition dynamics.

![KDA chunkwise DPLR efficiency](/assets/img/diagrams/kimi-linear/kda-chunkwise-dplr-efficiency.svg)

### Understanding the Chunkwise Algorithm

**Two execution modes.** The KDA kernel switches between two algorithms based on sequence length. For long inputs (prefill), it runs in **chunkwise mode**, which processes the recurrence in fixed-size chunks that can be parallelized across the chunk dimension, turning a long sequential scan into a sequence of small matrix operations the GPU can batch. For short or streaming decode steps, it runs in **fused recurrent mode**, a step-by-step RNN update that uses constant memory per step. The dispatch happens inside the kernel so the model code does not change between training and serving.

**The DPLR specialization.** The general DPLR matrix form admits a chunkwise algorithm, but it is expensive. KDA uses a **specialized variant** of DPLR for its transition matrices that substantially reduces the computation while staying consistent with the classical delta rule. This is the key engineering contribution: by constraining the structure of the transition matrix, the chunkwise parallelization keeps the math equivalent to the sequential recurrence but cheap enough to run at speed on Hopper-class GPUs. The report frames this as "more consistent with the classical delta rule" than competing linear-attention chunk schemes, which is both a correctness and an efficiency claim.

**The efficiency outcome.** Measured against full Multi-Head Latent Attention with an identical training recipe, the KDA-based hybrid delivers three simultaneous wins: about 75% less KV cache at 1M tokens, up to 6x decoding throughput at that context (roughly 1.84ms per output token versus 11.48ms for MLA), and higher quality on MMLU-Pro, RULER, and RL-style benchmarks. On RULER at 128k context, Kimi Linear is Pareto-optimal: top accuracy at 3.98x speedup. The point of the chunkwise DPLR algorithm is that these are not purchased with a quality tax; they come from a faster, leaner recurrence that also models better.

**Where the kernel lives.** The production kernel (`chunk_kda`, `fused_recurrent_kda`, and the channel-wise `fused_kda_gate`) is open-sourced inside the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda) `fla.ops.kda` package, and the model ships with a vLLM serving path, so the speedups above are reproducible rather than paper-only numbers.

## Usage: Run Kimi Linear

Kimi Linear ships as drop-in Hugging Face checkpoints and a vLLM serving path. Install the linear-attention kernel, load the instruction-tuned model, and generate.

```bash
pip install -U fla-core
```

Inference with Hugging Face Transformers (requires `trust_remote_code=True` for the KDA modeling code):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
    {"role": "user", "content": "Is 123 a prime?"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
generated_ids = model.generate(inputs=input_ids, max_new_tokens=500)
print(tokenizer.batch_decode(generated_ids)[0])
```

For production, serve an OpenAI-compatible API with vLLM across multiple GPUs and the full 1M context window:

```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 1048576 \
  --trust-remote-code
```

## Key Features

| Feature | Description |
|---------|-------------|
| Channel-wise diagonal gating | `Diag(alpha_t)` gives each feature dimension its own decay rate instead of one scalar per head |
| Delta-rule update | State update is a rank-1 online gradient step, consistent with the classical delta rule |
| Specialized DPLR chunkwise algorithm | Cheap parallel prefill that stays mathematically equivalent to the sequential recurrence |
| Hybrid 3:1 KDA-to-MLA stack | Linear layers dominate cost; periodic full-attention layers anchor global recall |
| NoPE on full-attention layers | Positional information is carried by KDA's recurrence, improving long-sequence extrapolation |
| Constant recurrent memory | KV usage stops growing with sequence length, ~75% cache reduction at 1M tokens |
| 6x decoding throughput | Up to 6.3x faster time-per-output-token than MLA at 1M-token context |
| Outperforms full attention | Higher quality than full MLA on MMLU-Pro, RULER, and RL benchmarks under matched training |
| Scales to 2.8T | Backbone of 69 of Kimi K3's 93 layers (24 Gated MLA + 1 dense) |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: fla` | Missing linear-attention kernel | `pip install -U fla-core` (>= 0.4.0) |
| Slow long-context decode | Running full-attention only / wrong kernel path | Ensure KDA layers use `chunk_kda`/`fused_recurrent_kda` from the `fla` package |
| OOM at 1M context with MLA | KV cache grows linearly | Use the KDA hybrid model; verify the mixed KV-cache manager is active |
| Quality drop on retrieval tasks | KDA ratio too high (e.g. 7:1) | Keep the 3:1 KDA-to-MLA ratio; full-attention layers are needed for global recall |
| `trust_remote_code` required | KDA modeling is not yet in upstream Transformers | Pass `trust_remote_code=True` on model and tokenizer load |
| Poor long-sequence extrapolation | RoPE on full-attention layers | Use the NoPE configuration; let KDA carry positional information |

## Conclusion

Kimi Delta Attention is a small, well-motivated change with outsize consequences. By promoting the decay gate from a scalar to a per-channel diagonal, it turns a fixed-size recurrent state into a memory that can selectively persist the facts worth keeping, which is enough to let a linear-attention layer match and exceed full attention. Paired with a 3:1 hybrid schedule and a specialized DPLR chunkwise algorithm, it removes the long-context tax of softmax attention: constant memory instead of a growing KV cache, and 6x decode throughput at a million tokens, with better quality rather than worse.

The same module scales from a 48B open model to the 2.8T Kimi K3, where it forms the majority of the backbone and is one of two architectural updates credited with a 2.5x scaling-efficiency gain. For anyone building long-context or agentic systems where context length and decode cost are the binding constraints, KDA is currently the most production-ready linear-attention design to study and deploy.

## Links

- [Kimi Linear paper (arXiv:2510.26692)](https://arxiv.org/abs/2510.26692)
- [Kimi Linear code (GitHub)](https://github.com/MoonshotAI/Kimi-Linear)
- [Kimi K3 weights (Hugging Face)](https://huggingface.co/moonshotai/Kimi-K3)
- [Kimi K3 announcement (kimi.com)](https://www.kimi.com/blog/kimi-k3)
- [vLLM Kimi K3 day-0 support](https://vllm.ai/blog/2026-07-27-k3)
- [KDA kernel in flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)
- [Gated DeltaNet (arXiv:2412.06464)](https://arxiv.org/abs/2412.06464)
