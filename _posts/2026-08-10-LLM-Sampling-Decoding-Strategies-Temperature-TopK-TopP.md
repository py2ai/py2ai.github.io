---
layout: post
title: "LLM Sampling and Decoding Strategies: Temperature, Top-k, Top-p, Min-p, and Beam Search"
description: "A practical guide to LLM decoding covering greedy search, beam search, temperature, top-k, top-p (nucleus), min-p sampling, and repetition penalties. Learn how to tune generation for code, chat, and creative writing."
date: 2026-08-10
header-img: "img/post-bg.jpg"
permalink: /LLM-Sampling-Decoding-Strategies-Temperature-TopK-TopP/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - LLM
  - Inference
  - Decoding
  - Tutorial
author: "PyShine"
---

# LLM Sampling and Decoding Strategies: Temperature, Top-k, Top-p, Min-p, and Beam Search

A language model does not output text. At every generation step it outputs a vector of raw scores -- one logit per token in its vocabulary, often 32,000 to 256,000 numbers. Turning those logits into the next token is the job of the decoding strategy, and the choice matters more than most people expect. The same model can produce deterministic, factual answers or wild, creative prose depending entirely on a handful of sampling parameters. Worse, the wrong settings cause the two most common complaints about LLMs: endless repetition loops and incoherent rambling. This post explains how each decoding strategy works, when to use it, and which settings to start from.

This is the seventh post in our LLM internals series, following our coverage of [tokenization](/LLM-Tokenization-How-Text-Becomes-Numbers/), [attention](/LLM-Attention-Mechanism-Heart-of-Transformer/), [KV-cache](/LLM-Decode-KV-Cache-GPU-VRAM-Deep-Dive/), [inference phases](/LLM-Prompt-vs-Decode-Understanding-Two-Phases-Inference/), the [training pipeline](/LLM-Training-Pipeline-Pretraining-SFT-RLHF/), and [quantization](/LLM-Quantization-FP16-INT8-INT4-GGUF-AWQ-GPTQ/).

## Why Decoding Strategy Matters

![LLM Sampling and Decoding Strategies](/assets/img/diagrams/llm-sampling/llm-sampling-strategies.svg)

### Understanding the Diagram

The diagram above breaks decoding into six panels: the logits-to-probability conversion, greedy versus beam search, temperature scaling, truncation methods, the full decoding loop, and recommended starting settings. Let us walk through each panel.

**Panel 1: From Logits to Probabilities**

The top-left panel shows the two-step conversion that happens at every generation step. The model's final layer produces one raw logit per vocabulary token -- for the prompt "The cat sat on the", the candidates might be "mat" (3.1), "floor" (2.6), "roof" (1.8), "chair" (1.4), and "grass" (0.7). Logits are unbounded and do not sum to anything, so softmax exponentiates and normalizes them into a probability distribution: 46%, 28%, 13%, 9%, 4%. The key property of softmax is that it never changes the ranking -- the highest logit always gets the highest probability. It only controls how peaked the distribution is, which is exactly the lever temperature pulls in panel 3.

**Panel 2: Greedy vs Beam Search**

The top-middle panel contrasts the two deterministic strategies. Greedy decoding picks the single highest-probability token at every step, producing one fixed output for a given prompt. Beam search instead keeps the k best partial sequences (here k=2): after "The" it tracks both "The cat" (-0.3) and "The dog" (-0.9), pruning "A cat" (-1.3). At the end it returns the complete sequence with the best cumulative log-probability -- "The cat sat" at -0.5 beats the greedy sequence's -1.02. Greedy is simply beam search with width 1. Beam finds higher-likelihood sequences, but for open-ended chat those sequences tend to be generic and repetitive, which is why beam is mostly used for translation and summarization rather than conversation.

**Panel 3: Temperature Sampling**

The top-right panel shows the same distribution at three temperatures. Temperature divides every logit before softmax: T=0.2 sharpens the distribution until "mat" takes 91% (nearly greedy), T=1.0 preserves the model's raw distribution (46/28/13/9/4), and T=2.0 flattens it toward uniform (33/26/17/14/10). Low temperature is what you want for code and factual QA; high temperature increases diversity at the cost of coherence. Above roughly T=1.5 the tail tokens start derailing the output.

**Panel 4: Top-k and Top-p Truncation**

The bottom-left panel shows why sampling from the full distribution is dangerous: the tail contains thousands of unlikely tokens, and even a 0.1% chance of a bad token compounds over hundreds of generation steps. Truncation removes the tail before sampling. Top-k keeps a fixed number of candidates (k=3 keeps mat, floor, roof). Top-p (nucleus sampling) keeps the smallest set whose cumulative probability reaches p -- at p=0.9 it keeps four tokens here (96% cumulative mass), but would keep only one token if the model were confident, and dozens if it were uncertain. Min-p, a newer method, sets the cutoff relative to the top token's probability, making it robust at high temperatures. After truncation, the kept probabilities are renormalized to sum to 1.

**Panel 5: The Decoding Loop**

The bottom-middle panel shows the full generation cycle: the context (prompt plus generated tokens) goes through one forward pass, the logits are scaled by 1/T, filtered by top-k/top-p/min-p, one token is sampled from the renormalized distribution, appended to the context, and the loop repeats until an EOS token, a stop string, or max_tokens. Every generated token costs one forward pass -- this is the decode phase from our [prompt vs decode post](/LLM-Prompt-vs-Decode-Understanding-Two-Phases-Inference/), and it is why the KV-cache exists. Replace sampling with argmax in step 5 and this loop becomes greedy decoding.

**Panel 6: Recommended Starting Points**

The bottom-right panel gives starting settings by task: near-greedy (T=0.0-0.3) for code and factual QA, moderate (T=0.6-0.8, top-p=0.95) for general chat, higher (T=0.9-1.2) for creative writing, and the highest (T=1.2-1.5, top-p=1.0) for brainstorming. It also lists the three repetition-control knobs: llama.cpp's repetition_penalty, and OpenAI's frequency_penalty and presence_penalty. The tuning order matters: adjust temperature first, then top-p, and reach for penalties only when loops appear.

## From Logits to Probabilities

The final layer of a transformer produces a vector of logits -- one floating-point score per vocabulary entry. Logits are convenient for the model (they are unbounded, so the network can express confidence with arbitrary margin) but useless for sampling until softmax converts them:

```
p_i = exp(z_i / T) / sum_j exp(z_j / T)
```

where `z_i` is the logit for token i and T is the temperature. At T=1 this is plain softmax. The formula reveals everything about decoding: the only inputs are the logits and T, and the only output is a probability distribution. Every decoding strategy is a way of choosing (or narrowing down) a token from this distribution.

One subtlety: softmax is computed in floating point, and the division by T happens before exponentiation. This is why T must be positive -- T=0 would divide by zero, so "temperature zero" in every real implementation actually means "skip sampling and take argmax".

## Greedy and Beam Search

### Greedy Decoding

Greedy decoding takes argmax at every step. It is deterministic (same prompt, same output, every time), cheap to implement, and the default in Hugging Face Transformers when you do not pass `do_sample=True`. Its weaknesses show up on longer outputs: greedy text tends to fall into repetition loops ("the movie was great, the movie was great, ...") because locally optimal choices do not add up to globally good text.

Use greedy for: code generation, factual extraction, math word problems, and any task where there is one right answer and diversity is a liability.

### Beam Search

Beam search keeps the k highest-scoring partial sequences at each step and returns the best complete one. It approximates finding the globally most probable sequence, which greedy cannot do -- greedy's locally optimal picks can lead to a dead end.

Beam search shines in machine translation and summarization, where the output space is constrained and high likelihood correlates with quality. It fails at open-ended generation for an interesting reason: the highest-probability text is often the most generic. The [nucleus sampling paper](https://arxiv.org/abs/1904.09751) showed that human text has much higher variance in per-token probability than beam-search text -- people systematically avoid the most predictable word. Maximizing likelihood produces text that is fluent but boring and strangely repetitive.

Practical notes: beam search multiplies compute and KV-cache memory by the beam width (typically 4-5), and it does not combine with sampling -- you pick one family or the other. Most chat APIs do not expose beam search at all.

## Temperature Sampling

Temperature is the single most influential sampling parameter. Dividing logits by T before softmax reshapes the distribution without changing the token ranking:

- **T < 1** sharpens: the gap between the top token and the rest widens. At T=0.2, a 0.5-logit lead becomes a 91% probability.
- **T = 1** leaves the model's learned distribution untouched.
- **T > 1** flattens: unlikely tokens gain probability mass. At T=2.0 the fifth-ranked token still has 10% probability.

The failure modes sit at both extremes. Too low, and the model becomes brittle and repetitive -- the same prompt always yields the same completion, and any tendency to loop is amplified. Too high, and tail tokens with genuinely wrong continuations get sampled, producing grammar errors, topic drift, and eventual incoherence. The useful range for most models is 0.3 to 1.3.

A common misconception is that temperature controls "creativity". It controls diversity. Whether that diversity reads as creativity or as noise depends on the model's quality and on whether truncation (next section) keeps the tail in check.

## Truncation: Top-k, Top-p, and Min-p

Raw sampling from the full distribution is dangerous because the tail is long: a 128k-vocabulary model might assign 2% of total probability to thousands of junk tokens collectively. Over a 500-token completion, you will sample from that tail repeatedly. Truncation methods delete the tail before sampling.

### Top-k

Keep the k highest-probability tokens, discard the rest, renormalize. Simple and predictable, but blind to the distribution's shape: when the model is confident (peaked distribution), k=50 keeps 49 junk candidates that collectively get sampled surprisingly often; when the model is uncertain (flat distribution), k=50 may still cut off legitimate options. Typical values are 40-100. Top-k is rarely used alone today, but survives as a safety net combined with top-p.

### Top-p (Nucleus Sampling)

Keep the smallest set of tokens whose cumulative probability reaches p, discard the rest, renormalize. This adapts to model confidence: at a confident step the nucleus might contain 2 tokens; at an uncertain step it might contain 200. Top-p=0.9 to 0.95 is the de facto standard for chat and creative generation, and the method introduced by [Holtzman et al.](https://arxiv.org/abs/1904.09751) remains the default in most frameworks.

Its weakness appears at high temperature: flattening the distribution inflates the nucleus, so top-p=0.95 at T=1.5 still admits a large tail of dubious tokens.

### Min-p

Min-p sets an absolute floor: keep tokens whose probability is at least `min_p * p_max`, where p_max is the top token's probability. With min_p=0.05, a confident step (p_max=0.9) keeps tokens above 4.5%, while an uncertain step (p_max=0.1) keeps everything above 0.5%. The cutoff scales with confidence automatically, and unlike top-p it stays tight even at high temperatures. The [min-p paper](https://arxiv.org/abs/2407.01082) (ICLR 2025) showed improved quality and diversity at T up to 3.0, and the method is now built into Hugging Face Transformers, vLLM, and llama.cpp.

If your framework supports min-p, a strong modern default is T=0.7-1.0 with min_p=0.05-0.1 and no top-p.

## Repetition Penalties

Penalties modify logits of tokens that have already appeared, discouraging loops:

- **repetition_penalty** (llama.cpp, HF): divides the logits of all tokens already present in the context by a fixed factor (1.0 = off, typical 1.05-1.2). Simple and effective, but it also penalizes legitimate repetition -- technical terms, variable names, and the topic of the conversation itself.
- **frequency_penalty** (OpenAI): the penalty grows with the number of times a token has appeared. Good for curbing repeated words in long outputs.
- **presence_penalty** (OpenAI): a flat penalty applied once a token has appeared at all, regardless of count. Pushes the model toward new topics and vocabulary.

Penalties are a patch, not a cure. If outputs loop at moderate temperature with top-p=0.95, the better fix is often a small repetition_penalty (1.05) rather than cranking temperature, which trades the loop problem for an incoherence problem.

## Practical Examples

### Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "Write a one-line tagline for a coffee shop."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

# Balanced chat settings: sampling on, moderate temperature, nucleus truncation
output = model.generate(
    inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.05,
)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

Set `do_sample=False` (and drop temperature/top_p) for greedy decoding. Beam search is `num_beams=4` with `do_sample=False`. Min-p is available as `min_p=0.05` in recent Transformers versions.

### llama.cpp

```bash
./build/bin/llama-cli \
    -m ./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf \
    -p "Write a one-line tagline for a coffee shop." \
    -n 128 \
    --temp 0.7 \
    --top-p 0.9 \
    --min-p 0.05 \
    --repeat-penalty 1.05
```

llama.cpp applies its sampler chain in a fixed order (penalties, then temperature, then truncation), so you can combine flags safely. Setting `--temp 0` switches to greedy.

### OpenAI-Compatible APIs

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Write a one-line tagline for a coffee shop."}],
    temperature=0.7,
    top_p=0.9,
    frequency_penalty=0.3,
    presence_penalty=0.0,
)
print(response.choices[0].message.content)
```

This works against any OpenAI-compatible server (vLLM, llama.cpp's llama-server, Ollama, LM Studio). Note that the OpenAI API does not expose top-k or min-p; for those, use the engine's native configuration.

## Recommended Starting Points

| Task | Temperature | top-p | Repetition penalty |
|------|------------|-------|--------------------|
| Code / factual QA | 0.0 - 0.3 | 0.9 | off |
| General chat | 0.6 - 0.8 | 0.95 | 1.0 - 1.05 |
| Creative writing | 0.9 - 1.2 | 0.95 - 1.0 | 1.05 - 1.15 |
| Brainstorming | 1.2 - 1.5 | 1.0 | 1.1 - 1.2 |

Treat these as starting points, not laws. Models differ: a well-aligned instruct model tolerates higher temperature than a base model, and some models ship with recommended sampling settings in their model card -- check there first.

## Common Pitfalls

**Pitfall 1: Blaming the model for decoding problems.** Repetition, blandness, and incoherence are often sampling misconfigurations, not model quality issues. Before switching models, try T=0.7 with top-p=0.9 and repetition_penalty=1.05.

**Pitfall 2: Stacking aggressive truncation with low temperature.** T=0.2 with top-k=10 is nearly greedy and highly loop-prone on long outputs. If you need determinism, use T=0 and accept the loops, or fix loops with a penalty instead of more truncation.

**Pitfall 3: Expecting temperature = 0 to guarantee reproducibility across systems.** Argmax is deterministic only on identical software and hardware. Different GPU kernels, batching, or quantization can flip near-tied logits. For strict reproducibility you also need fixed seeds, pinned versions, and often `temperature=0` plus greedy on the same stack.

**Pitfall 4: Using beam search for chat.** Beam maximizes likelihood, and maximum-likelihood open-ended text is generic and repetitive. Beam search is for translation and summarization; use sampling for conversation.

**Pitfall 5: Penalizing technical content.** A repetition penalty of 1.2 on a code assistant will actively corrupt outputs -- repeated variable names and syntax tokens get suppressed. Keep penalties at or below 1.05 for code, or disable them entirely.

## Related Posts

- [LLM Tokenization: How Text Becomes Numbers](/LLM-Tokenization-How-Text-Becomes-Numbers/)
- [LLM Attention Mechanism: The Heart of the Transformer](/LLM-Attention-Mechanism-Heart-of-Transformer/)
- [LLM Decode: KV-Cache and GPU VRAM Deep Dive](/LLM-Decode-KV-Cache-GPU-VRAM-Deep-Dive/)
- [LLM Prompt vs Decode: Understanding the Two Phases of Inference](/LLM-Prompt-vs-Decode-Understanding-Two-Phases-Inference/)
- [LLM Training Pipeline: From Pretraining to RLHF and DPO](/LLM-Training-Pipeline-Pretraining-SFT-RLHF/)
- [LLM Quantization: Running 70B Models on a Laptop with FP16, INT8, and INT4](/LLM-Quantization-FP16-INT8-INT4-GGUF-AWQ-GPTQ/)

## Further Reading

- [The Curious Case of Neural Text Degeneration (nucleus sampling paper)](https://arxiv.org/abs/1904.09751)
- [Turning Up the Heat: Min-p Sampling (paper)](https://arxiv.org/abs/2407.01082)
- [Hugging Face: How to Generate](https://huggingface.co/blog/how-to-generate)
- [Hugging Face Transformers: Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [OpenAI API Reference (temperature, penalties)](https://platform.openai.com/docs/api-reference/chat/create)
- [llama.cpp GitHub repository](https://github.com/ggml-org/llama.cpp)

## Conclusion

Decoding is the cheapest lever you have on LLM output quality -- no retraining, no extra VRAM, just a different choice of token at every step. The mental model is simple: softmax turns logits into probabilities, temperature reshapes those probabilities, truncation removes the unreliable tail, and penalties discourage repetition. Start with T=0.7 and top-p=0.9 for general use, drop to near-greedy for code and factual work, raise temperature with min-p for creative tasks, and add a gentle repetition penalty only when loops appear. Get these four knobs right and even a small local model will feel noticeably sharper.
