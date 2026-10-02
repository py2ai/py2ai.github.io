---
layout: post
title: "AnyJev: Typed Decisions With Calibrated Probabilities - Inside nokia-applied-research/AnyJev"
description: "AnyJev turns any open-source LLM into a Jev-style decision model: ask typed questions, read calibrated probabilities from a single prefill, and automate more traffic safely. We tour the source to see how permutation marginalization, label-free priors and closed-form heads make it work."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /AnyJev-Turn-Any-LLM-Into-A-Calibrated-Decision-Model/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/anyjev/nokia-applied-research-anyjev-architecture.svg
tags:
  - AI Decisions
  - LLM Calibration
  - Model Serving
  - Open Source
categories: [AI, Open Source]
keywords: "AnyJev, typed decisions, LLM calibration, Jev-style decision model, permutation marginalization, temperature scaling, closed-form head, vLLM serving, position bias, probability calibration"
author: "PyShine"
---

If you have ever asked a language model to pick between a few options, you have probably noticed that the confidence it reports is more decoration than measurement. Reorder the options and the answer changes. Read the probability it puts on each choice and the numbers look decisive even when the model is guessing. For a demo that is amusing; for a router that decides which team handles a support ticket, or a gate that decides whether an agent acts without asking a human, it is disqualifying.

AnyJev, published by Nokia's applied research group with a collaborator from Tencent Hunyuan, attacks exactly this gap. It is a Python library that turns any causal LLM into a decision model: you ask typed questions (a choice among options, a yes/no, a score on a scale) and you get back a probability distribution you can threshold, read from a single prefill with nothing generated and nothing parsed. The project's reported headline is the one that matters in production: on a twenty-way banking classification task with Qwen3-8B, the share of traffic that can be safely automated at a five percent error budget goes from 7.7 percent with raw logits to 52.0 percent once the probabilities are calibrated.

The source is worth a tour because it is unusually honest research code. Every number in the README is regenerated from committed JSON artifacts, the research log documents negative results alongside wins, and the library ships in a 0.2.0 state where the only hard dependency is NumPy. It also deliberately positions itself in a fast-moving space: it implements the Jev-style contract of typed, leveled decisions without being affiliated with any of the proprietary models in that family.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/anyjev/nokia-applied-research-anyjev-overview-architecture.svg" alt="Architecture overview of the nokia-applied-research/AnyJev repository" style="max-width:100%;height:auto;" />
</div>

*The repository at a glance: a thin public interface around a prompt layer, four decision levels fed by a calibration core, interchangeable model backends, and a serving pipeline that turns a Hugging Face checkpoint into a measured decision endpoint.*

Reading the overview from left to right: your application imports the package and constructs a `Decider` with typed `Question` objects; the prompt layer renders states and builds the exact prompts whose label tokens are scored; the decision levels (raw, L0, L1, L2) progressively add calibration machinery on top of the same readout; the backend interface abstracts whether the model runs locally through transformers or behind a vLLM server; and the serving pipeline glues truncation, serving and measurement together for deployment.

## Why You Need This

The core problem is position bias. When an LLM answers a multiple-choice question by reading its next-token distribution over the letters A, B, C, the option listed first tends to attract probability mass. The README's banner figure demonstrates it on a real banking-intent item: reverse the order of the options and a raw logit readout flips its answer, while the L0 readout gives the same answer both ways. Any measured flip rate of 0.230 dropping to 0.073 with zero labels is the difference between a demo and a component you can reason about.

The second problem is calibration, and it is subtler. Even when the ranking is right, the probabilities themselves do not mean what they say. A reported 0.9 from raw logits is not trustworthy enough to act on, so in practice every decision escalates to a human, and the automation rate collapses. This is why the project keeps insisting that accuracy and auto-decidable coverage are different axes: the accuracy moves only a few points across levels, but the share of decisions you can safely automate moves by a factor of several. Once the number means what it says, a simple threshold converts a watchful supervisor into a supervision-free pipeline.

There is also a deployment-shaped problem. Many teams cannot fine-tune a model per task, and few want a separate small classifier for every question they ask. AnyJev's design answers both constraints. The zero-label level (L0) requires nothing but extra prefills of a model you already serve. The strongest level (L2) needs only 100 to 300 labeled examples per question and fits its correction in closed form in seconds on a CPU, leaving the model's weights untouched. And the same API runs against transformers in-process or against a vLLM server, so the code you validate locally is the code you ship.

Finally, the project is refreshingly explicit about limits, which is exactly what you want in a component that makes automated decisions. It documents when the position-bias correction does not help (when one label dominates the batch), that heads do not transfer across questions or models, and that calibration cannot rescue a model that genuinely cannot answer the question. Knowing the failure envelope before adopting a library is rare and valuable.

## How It Works

The whole library is organized around one idea: a decision is read, not generated. One prompt goes in, the forward pass is stopped at a chosen point, and a probability vector over the question's options comes out.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/anyjev/nokia-applied-research-anyjev-architecture.svg" alt="Detailed architecture of the nokia-applied-research/AnyJev repository" style="max-width:100%;height:auto;" />
</div>

*The detailed architecture: typed question specs and result objects frame the Decider, which orchestrates the prompt layer, the calibration modules and the closed-form heads; three backends implement a common scoring interface; the deployment pipeline and evaluation harness surround the core.*

### Understanding the Architecture

**Typed questions define the contract.** In `anyjev/question.py`, the frozen `Question` dataclass supports exactly three kinds: `choice` (2 to 26 options, the letter-readout limit), `noul` (a yes/no), and `score` (2 to 10 bins over a scale, or explicit ordered levels). Constructors validate everything up front — duplicate options, bad bin counts, inverted scales all raise at build time. Because every downstream component keys off this small vocabulary, the rest of the system stays comprehensible.

**The readout turns a question into scored prompts.** `anyjev/readout.py` builds the chat prompt (with a default system message), resolves the label token IDs for each option permutation, and handles the shared-prefix logic that makes scoring all rotations of one state cheap when the backend supports it. `anyjev/state.py` renders the state — whatever context your application provides — into the text the model will read. Nothing is generated: the decision comes from logits over label tokens in one prefill.

**L0 removes position bias with zero labels.** `anyjev/calibrate/permute.py` generates the cyclic shifts of the option order, so that every option occupies every position exactly once across K reads, and marginalizes the scores. On top of that, `anyjev/calibrate/contextual.py` estimates and divides out the label prior — by default the batch mean of the decisions, optionally from content-free probes — with a prior-strength exponent (0.75 for the batch prior by default) chosen from a sweep over hundreds of model-question points and documented with its failure mode in the repository's docs.

**L1 adds honest probabilities.** `anyjev/calibrate/posthoc.py` implements temperature scaling fit on a labeled set for the question. The README's table shows what it buys: calibration error on the banking task drops from 0.184 at L0 to 0.095 at L1, using 100 to 500 labels. The ranking stays the same; the confidence becomes actionable.

**L2 replaces the readout with a fitted head.** `anyjev/heads.py` fits a closed-form linear head — diff-means, LDA, ridge, or reduced-rank regression, selected by cross-validation — on the hidden state captured partway down the network, with a temperature fit on out-of-fold scores so the head's own optimism is accounted for. No gradients, no weight updates. Because a middle block is a better feature space than the last one, `anyjev/truncate.py` can also cut the model to the blocks a decision needs and the serving pipeline (`anyjev/pipeline.py`) automates the whole convert-serve-measure loop. The repository ships fitted head banks for five Qwen3 models in `anyjev-heads/`, 23 heads per model in a file of a few megabytes.

**Backends hide the serving topology.** `anyjev/backends/base.py` defines the scoring interface; `anyjev/backends/hf.py` runs transformers in-process, `anyjev/backends/vllm.py` speaks to a vLLM server — a generate server with restricted tokens for raw/L0/L1, an embed server whose pooler returns the last hidden state for L2 — and `anyjev/backends/fake.py` provides a synthetic model so the entire pipeline runs in under a second on a laptop with no GPU. The result objects in `anyjev/result.py` carry their level, and `require("L1")` raises unless the decision meets the bar, so downstream code cannot silently act on a weaker calibration than it needs.

End to end: a state and a typed question enter the `Decider`; the readout renders the prompt(s); the backend scores label tokens or exposes hidden states; the active level applies shift marginalization, prior correction, a temperature, or a fitted head; and a `Decision` with a distribution, a confidence, and an honest level comes out. Enable the rotation budget (`adaptive_shifts=True`) and `anyjev/calibrate/stopping.py` decides after two or three shifts whether the leader is far enough ahead to stop — certifying agreement with the full-K answer via a log-odds margin and a confidence bound, which the README reports cuts 18 shifts to about 7.2 on average while multiplying decision throughput.

## Advantages

- **Zero-label debiasing.** L0 removes most of the position bias by construction, with cyclic-shift marginalization and a label-free prior — no calibration data required for the first win.
- **Calibrated, thresholdable outputs.** With a few hundred labels, L1 and L2 turn probabilities into numbers you can gate automation on, which is where the real coverage gains live.
- **No fine-tuning, no weight changes.** L2 heads are closed-form solves on a small labeled set; the model itself is never touched, so you can swap models without retraining anything but the few-kilobyte heads.
- **Cheap at serving time.** With the rotation budget enabled, most decisions stop after a couple of shifts; at L2 the head reads one truncated forward — less than a plain forward pass.
- **Backend-agnostic.** The same API runs on transformers in-process, on a vLLM server (both generate and embed shapes), or on a synthetic fake backend for tests and demos.
- **Research-grade reproducibility.** Every number in the docs is regenerated from committed JSON artifacts, and negative results are documented rather than hidden.

## Benefits

- **Higher safe automation rates.** The project's own measurement on a 20-way task shows auto-decidable coverage at a 5 percent risk going from under 8 percent with raw logits to over half of traffic once calibrated.
- **Order-invariant answers.** Because no option is favored by its position, upstream code can list options in any order without changing the decision — a property raw logit readouts famously lack.
- **Graceful level escalation.** A `Decider` can start at L0 on day zero, collect labels with `d.observe()`, and have the L2 head solve itself at 30 labels and re-solve as more arrive.
- **Deployment-friendly footprint.** A numpy-only core, optional torch extras, heads that fit in megabytes, and a one-command pipeline that measures accuracy, calibration error and latency on your own hardware.
- **Honest contracts between components.** Decisions carry their level and `require()` enforces minimum calibration where it matters, preventing the classic bug of trusting an uncalibrated probability in production code.
- **Clear limits, documented.** The README and docs state plainly where the method does not help — dominating labels, questions the model cannot answer, heads that do not transfer — so integration teams can plan around real constraints.

## Usage

Install with the Hugging Face extras, truncate a model to the blocks a decision needs, and serve it:

```bash
pip install "anyjev[hf]"

# keep the blocks a decision needs - usually about two thirds
python -m anyjev.truncate Qwen/Qwen2.5-7B-Instruct 18 ./qwen-b18

# serve it. L2 reads a hidden state, so the pooler hands one back untouched
vllm serve ./qwen-b18 --task embed \
  --override-pooler-config '{"pooling_type":"LAST","normalize":false,"softmax":false}'
```

Then define a typed question, fit a head on a small labeled set, and decide:

```python
from anyjev import Decider, Question
from anyjev.backends.vllm import VLLMBackend

d = Decider(VLLMBackend("http://localhost:8000", "./qwen-b18"), level="L2")
route = Question.choice("Which team should handle this?",
                        ["billing", "technical", "sales", "other"], name="route")

d.fit_head(route, states, labels, layers=[-1])   # 100-300 labels, one closed-form solve
d.decide(ticket, [route])["route"].distribution
```

Without any labels, turn on the rotation budget and calibrate it against the model's own full-strength readout:

```python
d = Decider(VLLMBackend("http://localhost:8000", "./qwen-b18"), adaptive_shifts=True)
d.calibrate_adaptive(route, unlabelled_tickets, target=0.01)   # a few hundred states, no labels
d.decide_batch(tickets, route)      # diagnostics: shifts_used, stop_threshold
```

Measure everything on your own box, or run the no-GPU synthetic demo:

```bash
python -m anyjev.pipeline Qwen/Qwen2.5-7B-Instruct --labels-from banking20
python -m demo.jev_mode --backend fake
```

## Conclusion

AnyJev is a well-scoped piece of engineering: it takes one narrow, well-known failure of LLM-based decision making — uncalibrated, position-biased readouts — and fixes it in layers whose costs and label requirements are explicit. The zero-label level makes answers order-invariant out of the box; the label-hungry levels make probabilities worth thresholding; and the closed-form heads keep the whole thing cheap enough to run in front of a model you already serve. For any team building routers, gates, or agent checkpoints on top of open models, it is a library that solves the right problem at the right layer, and the source makes the case for itself.

**Links:**

- Repository: [github.com/nokia-applied-research/AnyJev](https://github.com/nokia-applied-research/AnyJev)
- PyPI package: [pypi.org/project/anyjev](https://pypi.org/project/anyjev/)
- Levels contract: [docs/levels.md](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/levels.md)
- Benchmark results: [docs/results_bench.md](https://github.com/nokia-applied-research/AnyJev/blob/main/docs/results_bench.md)
