---
layout: post
title: "LLM Alignment: RLHF, DPO, and How Models Learn to Be Helpful"
description: "Learn how modern LLMs go from predicting the next token to being helpful, harmless, and honest assistants. Explore the three-stage alignment pipeline: Supervised Fine-Tuning (SFT), Reward Model training, and PPO. Understand the modern alternative DPO that skips the reward model entirely."
date: 2026-08-17
header-img: "img/post-bg.jpg"
permalink: /LLM-Alignment-RLHF-DPO-How-Models-Learn-to-Be-Helpful/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - LLM
  - Alignment
  - RLHF
  - DPO
  - InstructGPT
  - Reward-Model
  - Tutorial
author: "PyShine"
image: /assets/img/diagrams/llm-alignment/llm-llm-alignment-rlhf-dpo.svg
---

# LLM Alignment: RLHF, DPO, and How Models Learn to Be Helpful

A pretrained large language model is an incredible statistical achievement. It has absorbed trillions of tokens of human knowledge and can predict the next token with astonishing accuracy. But a pretrained model does not know what to say. If you ask it "How do I hack into a system?", it will answer. If you ask it to write a poem about existential dread, it will do that too -- with equal enthusiasm, because from its perspective, every completion is just one more statistical prediction.

Alignment is the process of teaching a pretrained model what it *should* say. The goal is to make models helpful (they understand intent and provide useful answers), harmless (they refuse dangerous requests), and honest (they do not fabricate information). This post explains how alignment works, from the original RLHF (Reinforcement Learning from Human Feedback) pipeline popularized by InstructGPT, to the modern DPO (Direct Preference Optimization) approach that has become the standard for open-source models.

This post is the fourteenth in our series understanding LLMs from the inside out, following [the attention mechanism](/LLM-Attention-Mechanism-Heart-of-Transformer/), [positional encoding](/LLM-Positional-Encoding-RoPE-ALiBi-Sinusoidal/), [Mixture of Experts](/LLM-Mixture-of-Experts-MoE-Sparse-Scaling/), [Parameter-Efficient Fine-Tuning](/LLM-Parameter-Efficient-Fine-Tuning-LoRA-QLoRA/), and earlier installments on [tokenization](/LLM-Tokenization-How-Text-Becomes-Numbers/), [quantization](/LLM-Quantization-FP16-INT8-INT4-GGUF-AWQ-GPTQ/), and [training pipelines](/LLM-Training-Pipeline-Pretraining-SFT-RLHF/).

![LLM Alignment (RLHF, DPO) Diagram](/assets/img/diagrams/llm-alignment/llm-llm-alignment-rlhf-dpo.svg)

## Why Alignment Is Needed

Consider the difference between a base model and an aligned model when asked the same question:

**Prompt**: "How can I make a bomb?"

**Base model response**: A detailed, step-by-step guide with chemical formulas and safety notes. The model is doing what it was trained to do: predict the most likely continuation given the context. From its training data (which includes public chemistry textbooks, history of weapons, etc.), the most likely completion happens to be dangerous.

**Aligned model response**: "I'm sorry, I can't help with that. If you're interested in chemistry, I'd be happy to explain safe household chemical reactions or the science behind explosives at a high level." The aligned model recognizes the dangerous intent, refuses the request, and offers a constructive alternative.

This is the alignment problem in a nutshell: pretrained models optimize for *next-token prediction accuracy*, but we want them to optimize for *helpfulness, harmlessness, and honesty*. These two objectives are not the same, and bridging the gap requires a separate training pipeline after pretraining.

## The Three-Stage Alignment Pipeline

The standard alignment approach, refined through the InstructGPT and GPT-4 programs [1][2], consists of three sequential stages:

1. **Supervised Fine-Tuning (SFT)**: Teach the model to follow instructions on high-quality demonstration data
2. **Reward Model (RM) Training**: Learn a scoring function from human preference comparisons
3. **Reinforcement Learning (PPO or DPO)**: Optimize the policy toward high-reward responses while staying close to the base model

Each stage addresses a different aspect of alignment. SFT teaches format and basic helpfulness. The reward model encodes nuanced human preferences. PPO or DPO pushes the model toward those preferences while preventing reward hacking.

## Stage 1: Supervised Fine-Tuning (SFT)

SFT is the simplest alignment stage. You take a high-quality dataset of (instruction, response) pairs and train the model to generate the response given the instruction. The loss function is standard cross-entropy:

```
L_SFT = -sum_t log P_theta(y_t | x, y_<t)
```

where `x` is the instruction/prompt and `y` is the target response.

### What SFT Achieves

SFT teaches the model to:
- Understand instruction-following format (respond directly, don't just continue)
- Adopt a helpful, conversational tone
- Follow multi-step instructions
- Know when to say "I don't know" or ask for clarification

The SFT dataset is curated: responses are written by skilled annotators or come from high-quality sources like Stack Overflow, code repositories, and human-written tutorials. The quality of the SFT data directly determines the quality ceiling of the aligned model.

### What SFT Does Not Achieve

SFT has limitations. It teaches the model what good responses look like, but it does not teach it to *choose* between two good responses. When two responses are both valid but one is more helpful, more harmless, or more honest than the other, SFT cannot distinguish. This is because SFT treats all training examples as equally correct -- it learns to match each one, not to prefer one over another.

SFT also teaches the model to reproduce the *average* quality of its training data. If some responses are mediocre, the model learns to produce mediocre responses some of the time. The model does not learn to consistently produce the *best* response.

## Stage 2: The Reward Model

To go beyond SFT, we need a way to teach the model *preferences* -- that response A is better than response B for the same prompt. This is where the reward model comes in.

### Collecting Human Preference Data

The first step is to collect pairwise comparison data. Human annotators are given a prompt and two responses (A and B) and asked to choose which one is better. The dimensions of "better" can include helpfulness, harmlessness, honesty, factuality, and following instructions.

For each prompt X, the annotator produces a pair `(Y_w, Y_l)` where `Y_w` is the winner and `Y_l` is the loser. A large-scale alignment dataset might contain 100K-500K such pairs.

### Training the Reward Model

The reward model `r_phi(X, Y)` is a neural network that takes a prompt X and a response Y and outputs a scalar score. It shares the same architecture as the language model (typically the same backbone) but replaces the language modeling head with a single-output scalar head.

The reward model is trained on the preference pairs using the Bradley-Terry loss [3]:

```
loss = -log sigma( r_phi(X, Y_w) - r_phi(X, Y_l) )
```

This loss encourages the reward model to assign a higher score to the winner than to the loser. The sigmoid function converts the score difference into a probability, and the loss is minimized when the winner's score is much higher than the loser's.

### What the Reward Model Encodes

The reward model implicitly learns a multi-dimensional value function. Different aspects of a response contribute to its score:
- **Helpfulness**: Does the response address the user's intent?
- **Harmlessness**: Is the response safe and non-dangerous?
- **Honesty**: Does the response avoid fabricating information?
- **Instruction following**: Does the response follow the format requested?
- **Clarity**: Is the response well-organized and easy to understand?
- **Tone**: Is the tone appropriate for the context?

The reward model is trained to assign a high score to responses that satisfy all these criteria and a low score to responses that violate them. Once trained, it serves as an automated proxy for human judgment.

## Stage 3: PPO -- Reinforcement Learning with Human Feedback

With a reward model in hand, we can now use reinforcement learning to optimize the language model policy. The algorithm of choice is Proximal Policy Optimization (PPO) [4], adapted for the language modeling setting.

### The PPO Objective

The PPO loss for language model alignment is:

```
L_PPO = E[ r_phi(X, Y) - beta * KL(pi_theta || pi_ref) ]
```

This combines two terms:
1. **Reward maximization**: The model generates responses Y, scores them with the reward model, and tries to maximize the average score `r_phi(X, Y)`
2. **KL penalty**: The KL divergence between the current policy `pi_theta` and the reference policy `pi_ref` (typically the SFT model) penalizes the model for deviating too far from what it already learned

The `beta` hyperparameter controls the strength of the KL penalty. It is typically set so that the KL divergence stays around 0.01-0.1 nats per token.

### Why the KL Penalty Is Essential

Without the KL penalty, the model can exploit the reward model. This is called *reward hacking*: the model discovers quirks in the reward model and produces responses that score high but are not actually better. For example:
- The model might learn that very long responses score higher (because longer responses appear more thorough to the reward model)
- The model might learn to include certain phrases that the reward model associates with high quality
- The model might drift away from natural language and start producing text that looks good to the reward model but is nonsensical to humans

The KL penalty keeps the model within a "trust region" around the SFT distribution, preventing these pathologies.

### The PPO Algorithm

For each training batch, PPO follows this sequence:

1. Generate responses Y from the current policy pi_theta for a batch of prompts X
2. Score each response with the reward model r_phi(X, Y)
3. Compute the KL divergence between pi_theta and pi_ref for each response
4. Compute the importance sampling ratio: `ratio = pi_theta(Y|X) / pi_old(Y|X)`
5. Clip the ratio to `[1 - eps, 1 + eps]` (typically eps = 0.2) to prevent large policy updates
6. Compute the clipped surrogate objective and update pi_theta

The clipping in step 5 is a key innovation of PPO. It ensures that the policy does not change too much in a single update, stabilizing training and preventing reward hacking through sudden large shifts.

### Implementation Sketch

```python
import torch
import torch.nn.functional as F

def ppo_step(policy, ref_model, reward_model, prompts, beta=0.1, eps=0.2):
    # Generate responses
    responses = policy.generate(prompts, max_new_tokens=256)
    full_text = [p + " " + r for p, r in zip(prompts, responses)]

    # Compute reward model scores
    with torch.no_grad():
        rewards = reward_model(full_text)  # scalar per example

    # Compute log probs for current and reference policies
    log_probs = policy.log_probs(full_text)
    ref_log_probs = ref_model.log_probs(full_text)

    # Compute KL divergence
    kl = (log_probs - ref_log_probs).mean(dim=-1)

    # Compute importance ratio (simplified: use mean log prob ratio)
    ratio = torch.exp(log_probs.mean(dim=-1) - log_probs.mean(dim=-1).detach())

    # Clipped surrogate
    surr1 = ratio * (rewards - beta * kl)
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * (rewards - beta * kl)
    loss = -torch.min(surr1, surr2).mean()

    # Update policy
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

This is a simplified sketch -- a production implementation needs to handle token-level granularity, advantage estimation, and multiple optimization epochs per batch.

## DPO: Bypassing the Reward Model

In 2023, Rafailov et al. [5] introduced Direct Preference Optimization (DPO), which showed that the reward model can be bypassed entirely. DPO achieves alignment directly from preference pairs without training an intermediate reward model or running PPO.

### The DPO Loss

DPO derives its loss from the same Bradley-Terry preference framework but avoids explicitly modeling a reward function. The loss is:

```
L_DPO = -log sigma( beta * [
    log pi_theta(Y_w|X) - log pi_theta(Y_l|X)
    - log pi_ref(Y_w|X) + log pi_ref(Y_l|X)
] )
```

This looks complex, but the intuition is straightforward:
- The model should assign higher log-probability to the winner `Y_w` than to the loser `Y_l`
- The comparison is relative to the reference model `pi_ref`, which anchors the optimization
- `beta` controls how strongly the model should prefer the winner over the loser

### Why DPO Works

The DPO loss can be interpreted as finding the policy that maximizes the reward model's implicit objective without explicitly training a reward model. The key insight is that the reward model's optimal policy can be written in closed form as:

```
pi*(Y|X) = (1/Z(X)) * pi_ref(Y|X) * exp(1/beta * r_phi(X, Y))
```

where `Z(X)` is a normalization constant. Taking the log and rearranging, we get exactly the DPO loss formula.

In practice, DPO's simplicity is its greatest strength:
- No separate reward model to train and maintain
- No PPO with complex hyperparameters (learning rate scheduling, clip range, number of epochs)
- No reward hacking to worry about
- Implementation is straightforward: it is essentially a cross-entropy loss with a preference formulation

### DPO vs RLHF Comparison

| Aspect | RLHF | DPO |
|--------|------|-----|
| Reward model | Required (separate training) | Not needed |
| Complexity | High (RM + PPO with tuning) | Low (single loss function) |
| Stability | Moderate (PPO can be unstable) | High (standard supervised optimization) |
| Data format | Prompt + response + preference | Prompt + preference pair |
| Compute cost | ~3x SFT (RM + PPO) | ~1x SFT |
| Quality | State-of-the-art | Matches or exceeds RLHF on benchmarks |
| Inference | Same as base model | Same as base model |

DPO has become the default alignment method for open-source projects. The standard recipe for open-source model alignment today is:

1. Train base model on text corpus
2. SFT on instruction-following data
3. DPO on preference pairs

## Alternative Alignment Approaches

Several other alignment methods have been developed, each with different trade-offs:

### Constitutional AI (2022)

Constitutional AI [6] uses a set of principles (a "constitution") rather than human comparisons to guide alignment. The model generates responses, then self-critiques them against constitutional rules, and revises if needed. This reduces reliance on human annotation and improves scalability.

### Direct Alignment Language Models (2023)

DALM [7] reformulates alignment as a direct language modeling problem. Instead of training a separate reward model or using PPO, it converts the preference optimization into a next-token prediction task by constructing preference-aware sequences.

### Iterative DPO (2024)

Iterative DPO [8] improves DPO by using the aligned model itself to generate preference data, creating an alignment feedback loop. The model generates candidate responses, a judge (or the model itself) ranks them, and DPO is retrained on the new preferences.

### On-Policy DPO (2024)

Standard DPO uses the same data for multiple optimization steps, which can degrade quality over time. On-Policy DPO [9] generates fresh responses from the current policy for each DPO update, maintaining the on-policy property that PPO was designed to preserve.

## Practical Alignment Workflow

Here is how you would align an open-source model in 2026:

### Step 1: SFT

```python
from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=sft_data,
    args=TrainingArguments(
        output_dir="./sft-output",
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        num_train_epochs=3,
    ),
    packing=True,
    max_seq_length=1024,
)
trainer.train()
```

### Step 2: DPO

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # the SFT model, frozen
    args=DPOConfig(
        output_dir="./dpo-output",
        beta=0.1,
        learning_rate=1e-5,
        num_train_epochs=1,
        per_device_train_batch_size=2,
    ),
    train_dataset=preference_data,  # {prompt, chosen, rejected}
)
trainer.train()
```

The `trl` library (Transformers Reinforcement Learning) provides a unified interface for SFT, DPO, and PPO. The preference dataset format is simple: each example has a `prompt`, a `chosen` response, and a `rejected` response.

### Step 3: Evaluating Alignment Quality

Alignment quality is evaluated using benchmarks like:
- **MT-Bench**: Multi-turn dialogue evaluation using GPT-4 as a judge
- **AlpacaEval**: Single-turn evaluation comparing model responses to a reference
- **Open-Ended Chain-of-Thought (OE-CoT)**: Tests reasoning quality with human judgment

These benchmarks measure helpfulness, instruction following, factuality, and safety across diverse tasks.

## Why Alignment Is More Important Than Pretraining

For many years, the prevailing view was that pretraining scale was the dominant factor in model capability. While scale still matters, alignment has emerged as an equally important factor for real-world usefulness:

- A well-aligned 7B model can outperform a poorly aligned 70B model on user-facing tasks
- Alignment enables a base model to be adapted to specific domains or interaction styles without retraining
- Safety alignment prevents models from being misused, which is critical for deployment

The alignment stage is where pretrained capability is converted into practical utility. A model that can do the task but refuses to help is not useful. A model that can do the task but sometimes produces dangerous or false information is not trustworthy. Alignment bridges this gap.

## Takeaways

1. **Alignment transforms a base model into an assistant.** Pretraining teaches knowledge and capability; alignment teaches helpfulness, harmlessness, and honesty.

2. **The three-stage pipeline is SFT -> RM -> PPO/DPO.** SFT teaches format; the reward model encodes preferences; PPO or DPO optimizes the policy.

3. **DPO is the modern default.** It achieves alignment quality matching or exceeding RLHF with a simpler pipeline (no reward model, no PPO). For open-source projects, DPO is the preferred approach.

4. **Data quality matters more than algorithm sophistication.** Both SFT and preference training succeed or fail based on the quality and coverage of the data.

5. **Alignment is never complete.** New capabilities (tool use, multi-modal reasoning) require new alignment data and methods. The alignment pipeline is continuously evolving as model capabilities expand.

## References

[1] Ouyang, L. et al. "Training language models to follow instructions with human feedback." NeurIPS 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

[2] Bubeck, S. et al. "Sparks of Artificial General Intelligence: Early experiments with GPT-4." 2023. [arXiv:2303.12712](https://arxiv.org/abs/2303.12712)

[3] Bradley, R.A., Terry, M.E. "Conditional logit analysis of choice behavior." 1952.

[4] Schulman, J. et al. "Proximal Policy Optimization Algorithms." 2017. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[5] Rafailov, R. et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

[6] Bai, Y. et al. "Constitutional AI: Harmlessness from AI Feedback." 2022. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)

[7] Xu, C. et al. "DALM: Direct Alignment Language Models." 2023. [arXiv:2312.02364](https://arxiv.org/abs/2312.02364)

[8] Yuan, L. et al. "Iterative Preference Learning from Human Feedback." 2024. [arXiv:2402.10097](https://arxiv.org/abs/2402.10097)

[9] Chen, Y. et al. "On-Policy DPO: Aligning Large Language Models with Fewer Data." 2024. [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)
