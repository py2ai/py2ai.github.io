---
layout: post
title: "OpenMythos: Reconstructing the Claude Mythos Architecture from First Principles"
description: "An in-depth exploration of OpenMythos, an open-source theoretical reconstruction of the Claude Mythos architecture featuring Recurrent-Depth Transformers, LTI-stable injection, dual attention modes, and adaptive computation time."
date: 2026-04-20
header-img: "assets/img/diagrams/openmythos/openmythos-architecture.svg"
permalink: "/OpenMythos-Reconstructing-Claude-Mythos-Architecture/"
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, LLM, Transformer, Architecture, Open-Source, Python, Deep-Learning, Claude, Recurrent-Depth]
author: PyShine
---

## Introduction

Large language models have transformed the landscape of artificial intelligence, yet the internal architectures powering the most capable systems remain closely guarded secrets. Among these, Anthropic's Claude has demonstrated reasoning capabilities that suggest something fundamentally different is happening beneath the surface -- something beyond simple token-prediction transformers. OpenMythos is an open-source project that attempts to reconstruct the theoretical architecture behind Claude from first principles, proposing that the key innovation lies in what it calls a Recurrent-Depth Transformer.

The central hypothesis of OpenMythos is striking: Claude does not reason primarily through chain-of-thought tokens visible to the user. Instead, reasoning happens silently in latent space through recurrent iterations within the model's hidden layers. Each forward pass executes multiple loops of computation over the same block, allowing the model to "think deeper" without producing additional output tokens. This is a paradigm shift from the prevailing assumption that more reasoning requires more visible tokens.

Why does this matter? If correct, the Recurrent-Depth approach explains several observed phenomena: Claude's ability to produce nuanced responses without lengthy chain-of-thought, its capacity for complex multi-step reasoning in a single generation, and the computational efficiency that allows it to compete with much larger models. OpenMythos provides a complete, implementable specification of this architecture -- from the mathematical guarantees that prevent training instability to the adaptive computation mechanisms that allocate thinking time where it is needed most. By making this reconstruction open source, the project invites the community to test, validate, and build upon these ideas.

---

## The Three-Stage Architecture

![OpenMythos Architecture](/assets/img/diagrams/openmythos/openmythos-architecture.svg)

The OpenMythos architecture divides the forward pass into three distinct stages: Prelude, Recurrent Block, and Coda. Each stage serves a specific purpose in the overall computation pipeline, and together they form a system that can iteratively refine its internal representations before producing output.

The **Prelude** is the entry point. It processes the input token embeddings through a standard transformer layer -- applying attention and feed-forward operations -- to produce an initial hidden state `e`. Critically, this representation `e` is computed once and then frozen. It is not updated during the recurrent iterations that follow. This frozen input serves as an anchor point, a stable reference that the recurrent block can always fall back on.

The **Recurrent Block** is where the core reasoning happens. It takes the hidden state and iterates over it multiple times, refining the representation through each pass. At every iteration, the frozen input `e` from the Prelude is injected back into the computation. This is the key design decision: by constantly re-injecting the original input, the model prevents hidden state drift -- a failure mode where repeated transformations cause the representation to wander away from the original semantic content. The update rule can be expressed as:

```
h_{t+1} = A * h_t + B * e + TransformerBlock(h_t, e)
```

Here, `A` is a learned state-transition matrix with stability guarantees (discussed in detail later), `B` is a learned input-injection matrix, and the TransformerBlock applies its standard attention and feed-forward operations conditioned on both the current state and the frozen input.

The **Coda** is the exit stage. After the recurrent block has completed its iterations (either by reaching a maximum count or through adaptive early exit), the Coda processes the final hidden state to produce the output logits. Like the Prelude, it uses a standard dense transformer layer -- no recurrence, no MoE, just a clean mapping from hidden state to vocabulary predictions.

This three-stage design is elegant in its separation of concerns. The Prelude encodes, the Recurrent Block reasons, and the Coda decodes. The frozen input `e` acts as a gravitational center that keeps the recurrent iterations grounded in the original problem, while the state-transition matrix `A` ensures that the iterative process remains numerically stable across arbitrarily many loops.

---

## Inside the Recurrent Block

![Recurrent Block Internals](/assets/img/diagrams/openmythos/openmythos-recurrent-block.svg)

The Recurrent Block is the computational heart of OpenMythos, and its internal design reveals several sophisticated mechanisms working in concert. Understanding each component is essential to appreciating how the architecture achieves both depth and stability.

**Loop Index Embedding** provides the model with awareness of its current iteration depth. Before each pass through the block, a sinusoidal position encoding -- analogous to the position encodings used in standard transformers -- is added to the hidden state based on the current loop index. This tells the model "you are on iteration 3 of 16," allowing it to adapt its computation based on depth. Early iterations might focus on broad pattern matching, while later iterations can specialize in fine-grained reasoning. Without this signal, the model would be unable to distinguish between its first and tenth pass through the same weights.

**TransformerBlock with Dual Attention** applies either Grouped Query Attention (GQA) or Multi-Latent Attention (MLA), depending on the configuration. Both modes support the same fundamental operation: allowing tokens to attend to one another. The choice between them trades off between KV-cache efficiency (MLA wins) and implementation simplicity (GQA wins). The feed-forward portion uses Mixture-of-Experts (MoE), where a router selects a subset of expert networks for each token, dramatically increasing parameter count without proportionally increasing compute.

**LoRA Adapter** applies depth-wise Low-Rank Adaptation at each loop iteration. Rather than sharing identical weights across all iterations, the LoRA adapter injects iteration-specific modifications. The formula is:

```
delta(x, t) = down(x) * scale[t] @ B
```

Here, `down(x)` projects the input to a lower dimension, `scale[t]` is a per-iteration scaling vector learned during training, and `B` is the up-projection matrix. This means each loop iteration applies a slightly different transformation, allowing the model to specialize its behavior at different depths without the cost of maintaining entirely separate parameter sets.

**LTI Injection** (Linear Time-Invariant) is perhaps the most critical innovation for training stability. Drawing on the Parcae architecture, the state-transition matrix `A` is parameterized to guarantee a spectral radius strictly less than 1. The parameterization uses:

```
A = exp(-exp(log_dt + log_A))
```

This double-exponential form ensures that every element of `A` is positive and strictly less than 1, regardless of the learned values of `log_dt` and `log_A`. This is a Zero-Order Hold (ZOH) discretization of a continuous-time system with a negative diagonal matrix, and it mathematically guarantees that the recurrent loop cannot explode. The `get_A()` method provides direct access to verify this constraint at any time.

**ACT Halting** (Adaptive Computation Time) allows the model to decide, on a per-position basis, when it has converged. Each token position maintains a halting probability, and once the cumulative probability exceeds a threshold, that position stops computing and its output is fixed as a weighted average of all previous iterations. This means simple tokens might exit after 2-3 loops, while complex reasoning tokens might use all 16 or more. The result is a computation graph that adapts its depth to the difficulty of the input -- efficient for easy inputs, thorough for hard ones.

---

## Dual Attention Modes: GQA vs MLA

![Attention Modes](/assets/img/diagrams/openmythos/openmythos-attention-modes.svg)

OpenMythos supports two distinct attention mechanisms, each with its own trade-offs. The choice between them is a configuration option, and the architecture is designed so that both modes are drop-in compatible within the same TransformerBlock.

**Grouped Query Attention (GQA)** is the more conventional approach. In standard Multi-Head Attention, every query head has its own key and value head. GQA reduces this by grouping multiple query heads to share a single key-value head. For example, with 32 query heads and 4 KV heads, each KV head serves 8 query heads. This reduces the KV cache size by a factor of 8 compared to full MHA, with minimal quality degradation. GQA is the default in many modern architectures including LLaMA-2 and Mistral, and it represents a well-understood, low-risk choice.

**Multi-Latent Attention (MLA)**, introduced in DeepSeek-V2, takes a fundamentally different approach to KV cache compression. Instead of caching full key and value vectors, MLA compresses them through a low-rank latent vector `c_kv`. The key and value projections are reconstructed on the fly from this compressed representation:

```
c_kv = W_dkv * x          # compress to latent (low rank)
k = W_k * c_kv + RoPE     # reconstruct key + rotary position encoding
v = W_v * c_kv             # reconstruct value
```

The critical insight is that only `c_kv` and the RoPE-applied key need to be cached. Since `c_kv` has a much smaller dimension than the full key and value vectors, the KV cache shrinks by 10-20x compared to standard MHA. This is a massive advantage for long-context inference, where KV cache size often becomes the memory bottleneck.

The trade-off is that MLA requires additional computation at inference time to reconstruct the full K and V from the latent. However, this reconstruction cost is typically small compared to the memory bandwidth savings from the reduced cache. For models serving long contexts (128K+ tokens), MLA can be the difference between fitting a request on a single GPU or requiring multi-GPU partitioning.

In OpenMythos, both modes are implemented with a clean abstraction layer. The `attn_type` parameter in `MythosConfig` selects between `"gqa"` and `"mla"`, and the rest of the architecture remains identical. This allows researchers to directly compare the two approaches under controlled conditions, using the same recurrent depth, same MoE configuration, and same training data.

---

## Model Scale Variants

![Model Variants](/assets/img/diagrams/openmythos/openmythos-model-variants.svg)

OpenMythos defines seven model variants spanning three tiers, from 1 billion to 1 trillion parameters. Each tier scales not just in size but in architectural configuration, reflecting the different computational budgets and deployment targets appropriate for each scale.

**Small Tier (1B, 3B):** These models are designed for research, experimentation, and edge deployment. The 1B variant uses a dimension of 1024, 8 attention heads, 4 experts (with 2 active), and 4 loop iterations. The 3B variant increases the dimension to 1536, heads to 12, experts to 8 (with 2 active), and loop iterations to 8. Both use a context length of 4096 tokens. These models can be trained on a single GPU and are ideal for testing architectural hypotheses.

**Medium Tier (10B, 50B):** The 10B model steps up to 2048 dimensions, 16 heads, 16 experts (4 active), and 12 loop iterations with an 8192-token context. The 50B model pushes further to 3072 dimensions, 24 heads, 32 experts (4 active), 16 loop iterations, and a 16384-token context. These models represent the sweet spot for many practical applications -- large enough to exhibit emergent reasoning behaviors, small enough to serve efficiently.

**Large Tier (100B, 500B, 1T):** The frontier-scale models. The 100B uses 4096 dimensions, 32 heads, 64 experts (8 active), 20 loop iterations, and 32768-token context. The 500B and 1T variants scale dimensions to 6144 and 8192 respectively, with up to 128 experts and 32 loop iterations. The 1T model supports a 131072-token context window. These models require distributed training infrastructure but represent the scale at which the Recurrent-Depth architecture's advantages become most pronounced.

A key feature across all variants is **depth extrapolation**: models are trained with `N` loop iterations but can be evaluated with `N+k` iterations at inference time. Because the LTI injection guarantees stability, adding extra iterations does not cause the hidden state to diverge. Instead, the model continues refining its representation, often producing better results on complex tasks. This is analogous to giving a person more time to think about a hard problem -- the same cognitive machinery, just more applications of it.

The scaling laws for Recurrent-Depth Transformers differ from standard dense models. Increasing loop iterations is far cheaper than increasing layer count, because the same parameters are reused. A 3B model with 8 loops effectively has the computational depth of a 24-layer model but with only 3B unique parameters. This parameter efficiency is one of the central advantages of the architecture.

---

## Training Stability Breakthrough

The history of looped and recurrent transformers is littered with training failures. The fundamental problem is residual explosion: when you apply the same transformation repeatedly, small perturbations can compound exponentially. After 10 or 20 iterations, the hidden state can grow to astronomical values, producing NaN losses and collapsed gradients.

OpenMythos solves this through the LTI injection mechanism. The state-transition matrix `A` is not learned directly. Instead, the model learns `log_dt` and `log_A`, and the actual transition matrix is computed as:

```
A = exp(-exp(log_dt + log_A))
```

The inner exponential ensures that the argument to the outer exponential is always negative (since `exp(anything)` is positive, and `-exp(anything)` is negative). The outer exponential of a negative number is always between 0 and 1. Therefore, every element of `A` is strictly in the range (0, 1), guaranteeing that the spectral radius of `A` is less than 1.

This is a Zero-Order Hold discretization of a continuous-time linear system where the continuous state matrix has a negative diagonal. In control theory, such systems are known to be stable -- they decay toward zero over time. The ZOH discretization preserves this stability property in the discrete domain.

The practical impact is dramatic: training proceeds smoothly regardless of the number of loop iterations. Gradients flow backward through 4, 16, or 50 iterations without exploding or vanishing. The `get_A()` method allows verification at any point during training:

```python
# Verify stability guarantee
A = model.recurrent_block.lti_injector.get_A()
spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(A)))
assert spectral_radius < 1.0, f"Stability violated: {spectral_radius}"
```

This guarantee is not an approximation or a heuristic -- it is a mathematical certainty derived from the parameterization. It removes the single biggest obstacle to training recurrent-depth transformers and makes the entire architecture viable.

---

## Mixture of Experts in the Loop

The Recurrent Block employs Mixture-of-Experts (MoE) in its feed-forward layers, while the Prelude and Coda use standard dense FFNs. This design choice reflects the different roles of each stage: the Prelude and Coda are simple mappings that do not benefit from the increased capacity of MoE, while the Recurrent Block's iterative reasoning benefits enormously from a larger parameter space.

The MoE implementation uses a top-K router with bias-based load balancing. For each input token, the router computes affinity scores against all expert networks and selects the top-K (typically K=2 or K=4, depending on the model variant) experts with the highest scores. The token's representation is then processed by these selected experts, and their outputs are combined using the router's softmax weights.

**Routed experts** specialize in different patterns. Through training, some experts learn to handle mathematical reasoning, others natural language understanding, others code generation, and so on. This specialization emerges naturally from the training process -- no manual assignment is needed.

**Shared experts** run alongside the routed experts and process every token regardless of the routing decision. They capture cross-domain patterns that are useful across all inputs: basic linguistic structure, common logical patterns, and general knowledge. This ensures that even when the router makes a suboptimal selection, the shared experts provide a baseline of competent processing.

**Router bias** is added to the affinity scores before top-K selection. This bias is updated during training to encourage balanced expert utilization. Without it, the router tends to collapse: most tokens get routed to a small subset of experts, leaving the rest idle. The bias term gently pushes underutilized experts to receive more tokens, maintaining a more even distribution.

The interaction between MoE and recurrence is particularly interesting. Because the same MoE layer is applied at each loop iteration, the router can route the same token to different experts at different depths. On the first iteration, a math problem token might be routed to a "pattern matching" expert. By the fifth iteration, after the problem structure has been partially resolved, the same token might be routed to a "computation" expert. This depth-dependent routing is enabled by the Loop Index Embedding, which provides the router with information about the current iteration.

---

## The MoDA Alternative

OpenMythos also includes an experimental attention variant called MoDA (Mixture-of-Depths Attention), implemented in `moda.py`. MoDA takes a different approach to depth-aware computation: instead of iterating over the same layer multiple times, it allows each attention head to attend not only to the current layer's key-value pairs but also to the depth KV from all preceding layers.

In MoDA, each head maintains a depth-augmented KV cache that accumulates representations from every layer in the stack. When computing attention at layer L, a head can attend to keys and values from layers 1 through L-1 in addition to the current layer. This creates a form of "vertical" attention that complements the standard "horizontal" attention across the sequence dimension.

The MoDA implementation also incorporates DeepSeek-V3 style Mixture-of-Experts with sigmoid gating, where the gating function uses a sigmoid activation rather than the more common softmax. Sigmoid gating allows multiple experts to contribute independently -- each expert's gate is computed without reference to the others -- which can lead to more diverse expert utilization patterns.

While MoDA is included as an alternative, the primary architecture uses the Recurrent Block with GQA or MLA attention. MoDA represents a research direction that may prove valuable in future iterations of the architecture, particularly for models where the recurrent-depth approach is not feasible due to latency constraints.

---

## Getting Started

OpenMythos is designed to be accessible and configurable. The Python API provides both low-level configuration and pre-defined model variants, making it easy to get started regardless of whether you want to experiment with architectural details or simply train a standard model.

### Creating a Model with MLA Attention

```python
from open_mythos import OpenMythos, MythosConfig

# Create model with MLA attention
config = MythosConfig(
    dim=2048,
    n_heads=16,
    attn_type="mla",
    max_loop_iters=16
)
model = OpenMythos(config)
```

### Creating a Model with GQA Attention

```python
from open_mythos import OpenMythos, MythosConfig

# Create model with GQA attention
config = MythosConfig(
    dim=2048,
    n_heads=16,
    n_kv_heads=4,       # 4 KV heads for GQA grouping
    attn_type="gqa",
    max_loop_iters=16
)
model = OpenMythos(config)
```

### Using Pre-Defined Variants

```python
from open_mythos import OpenMythos
from open_mythos.variants import mythos_3b, mythos_10b, mythos_50b

# Use a pre-defined variant
model = OpenMythos(mythos_3b())

# Or the medium-tier variant
model = OpenMythos(mythos_10b())
```

### Verifying Training Stability

```python
# Check the spectral radius of the LTI transition matrix
A = model.recurrent_block.lti_injector.get_A()
eigenvalues = torch.linalg.eigvals(A)
max_radius = torch.max(torch.abs(eigenvalues))
print(f"Spectral radius: {max_radius:.6f}")
# Should always be < 1.0
```

### Running Inference with Extra Depth

```python
# Train with 8 loops, evaluate with 12 for deeper reasoning
model.config.max_loop_iters = 12
output = model(input_ids)
# The LTI guarantee ensures stability even with extra iterations
```

The configuration system is fully documented and all parameters have sensible defaults. The `MythosConfig` dataclass exposes every architectural knob: dimension, heads, KV heads, expert count, active experts, loop iterations, context length, attention type, LoRA rank, and more. This makes OpenMythos an ideal testbed for ablation studies and architectural research.

---

## Conclusion

OpenMythos represents a significant contribution to the open-source AI community by providing a complete, implementable reconstruction of what may be the most important architectural innovation in modern language models: the Recurrent-Depth Transformer. The project's central insight -- that reasoning can happen silently in latent space through stable iterative refinement, rather than through visible chain-of-thought tokens -- challenges fundamental assumptions about how language models think.

The technical achievements of the project are substantial. The LTI injection mechanism solves the long-standing training stability problem that plagued recurrent transformers, providing a mathematical guarantee rather than a heuristic fix. The dual attention modes (GQA and MLA) offer flexibility for different deployment scenarios. The adaptive computation time mechanism ensures that compute is allocated where it matters most. And the seven model variants provide a clear scaling roadmap from research-grade 1B models to frontier-scale 1T systems.

Perhaps most importantly, OpenMythos is open source. The architecture specifications, the mathematical derivations, and the implementation are all available for inspection, modification, and extension. This transparency is crucial for the scientific process: hypotheses must be testable, results must be reproducible, and claims must be verifiable. By providing a concrete implementation of the Recurrent-Depth hypothesis, OpenMythos enables the community to validate or refute the architectural assumptions behind Claude's capabilities.

The future directions are clear: depth extrapolation experiments to test whether more iterations consistently improve reasoning, comparisons between MoE and dense variants at matched compute budgets, exploration of the MoDA attention alternative, and scaling studies across all seven model variants. OpenMythos has laid the foundation -- the community will build upon it.