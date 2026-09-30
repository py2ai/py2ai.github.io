---
layout: post
title: "NanoJev: A Nano Replica of Jev With Parallel Decisions and a Full Training Pipeline - Inside TianyuCodings/NanoJev"
description: "A source-level tour of TianyuCodings/NanoJev, a compact Python reimplementation of the Jev System-One decision model. We trace how one Qwen3-0.6B backbone with parallel decision heads answers Boolean, Choice and Score questions without decoding tokens, and how the repo builds candidate data, rolls out game episodes, and trains the shared checkpoint end to end."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /NanoJev-Nano-Replica-Parallel-Decisions-Training-Pipeline-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/nanojev/tianyucodings-nanojev-architecture.svg
tags:
  - Python
  - Machine Learning
  - Decision Models
  - Open Source
categories: [AI, Open Source]
keywords: "NanoJev, Jev replica, System-One decision model, parallel decisions, decision heads, Qwen3-0.6B, candidate scoring, ViZDoom, maze navigation, snake, supervised fine-tuning, Brier loss, probability distributions, source code tour, TianyuCodings"
author: "PyShine"
---

Most language-model agents decide by talking: they generate reasoning, then an answer token by token, then parse the text back into an action. NanoJev, a nano replica of the Jev decision model from TypeSafe, takes the opposite road. A state and a set of questions go in, complete probability distributions come out, and no output token is ever decoded. The whole thing is assembled from one 0.6B backbone, a LayerNorm, a linear scalar head, and an optional set-attention module. That economy is exactly why the repository is worth reading.

TianyuCodings/NanoJev is not a stub or a paper-only artifact. The tree contains a working inference service, programmatic data generators, environment adapters for Maze, Snake and ViZDoom, a unified supervised training script with several loss objectives, an n-step TD extension, and independent evaluation and replay tooling. The README reports that the released checkpoint plays four games — ViZDoom Basic, ViZDoom Predict Position, a 50×50 maze and Snake — from a single shared model, with results such as 128/128 on the Basic test set. Whether or not you chase those exact numbers, the code shows you how such a system is actually built.

The source is worth a tour for a second reason: it is unusually honest about its own machinery. Question encodings, probability targets and loss semantics are pinned down by offline contract tests in scripts/test_question_contract.py, invariance checks in scripts/check_trained_invariance.py, and a long compatibility audit in docs/TYPESAFE_CONTRACT.md. You can read a small codebase and see, in one sitting, what "a decision model without decoding" means at the tensor level — and what it deliberately does not claim.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/nanojev/tianyucodings-nanojev-overview-architecture.svg" alt="Architecture overview of the TianyuCodings/NanoJev repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the NanoJev pipeline: a shared DecisionModel feeds a persistent predictor and an HTTP service, environments and expert policies generate decision data, and a unified trainer warm-starts the model with optional TD targets before evaluation.*

Reading the overview from left to right: the browser replay UI (web/app.js) posts state and question batches to the decision HTTP service (scripts/serve_decisions.py), which loads a DecisionPredictor exactly once; the predictor wraps the DecisionModel defined in scripts/train_toy_decisions.py. Below, the episode pipeline (scripts/unified_game_pipeline.py) hosts the Maze and Snake environments (scripts/unified_grid_envs.py) and the headless ViZDoom adapter (scripts/unified_doom_env.py), querying the predictor for policy probabilities during rollouts. A frozen vision expert policy (scripts/sonic_predict_policy.py) contributes Predict Position decision rows alongside episode data, all converging on the unified games trainer (scripts/train_unified_games.py), which can pull n-step targets from scripts/unified_td.py and hands selected checkpoints to the evaluator (scripts/evaluate_unified_checkpoint.py).

## Why You Need This

If you build agents that must react fast and often, autoregressive decoding is an awkward substrate. NanoJev addresses that directly: each decision is a forward pass whose output is already a normalized distribution over supplied candidates, so ranking, selecting or sampling an action needs no generation loop. For a robot loop, a game bot or any controller that makes many judgments per second, that architectural difference is the point, not an optimization footnote.

The repo also solves a data problem that plagues decision-model work: where do labeled decisions come from? NanoJev answers with fully programmatic generators. scripts/game_tasks.py implements self-solvable toy games (tic-tac-toe and grid path tasks) with a pure-stdlib solver, and scripts/build_game_decisions.py turns those positions into decision records whose Choice targets are a uniform optimal-action policy, split into train, dev, calibration, test and ood. The harder Predict Position task gets its supervision from a different, verifiable source: scripts/sonic_predict_policy.py is a frozen pure-vision policy whose episodes scripts/sonic_predict_data.py records into decision rows, and scripts/prepare_sonic_supervision.py validates every row against the shared schema before training ever sees it.

A third problem is evaluation integrity. It is easy to fool yourself with a game agent that survives by doing nothing. NanoJev's environments (scripts/unified_grid_envs.py, scripts/unified_doom_env.py) make success a terminal property — a positive KILLCOUNT delta in ViZDoom, reaching the goal or collecting the target food in the grid games — and the collector refuses to treat external truncation as failure. scripts/replay_unified_episodes.py replays recorded trajectories through the simulators, and scripts/verify_nanojev_comparison.py keeps NanoJev, Jev and an untuned Qwen baseline on the same seeds, candidates and epsilon-greedy controller.

Finally, the training side is a compact case study in learning from mixed supervision. One checkpoint must answer action Choice questions for the policy and action-conditioned Boolean questions about what actually happened. scripts/train_unified_games.py shows how to balance those pools, warm-start from an existing checkpoint, and mix losses — including a Monte Carlo estimator for a Brier-style objective — without letting any quarantined or imagined label slip in.

## How It Works

The whole system hangs off one model class and one input convention; everything else in the tree exists to feed that pair or to check it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/nanojev/tianyucodings-nanojev-architecture.svg" alt="Detailed architecture of the TianyuCodings/NanoJev repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of NanoJev: data generators and environments produce decision rows, the unified trainer warm-starts the shared DecisionModel through a persistent predictor, and serving, contract checks and evaluation all read from the same checkpoint format.*

### Understanding the Architecture

**The DecisionModel itself is deliberately tiny.** In scripts/train_toy_decisions.py, DecisionModel wraps a Hugging Face backbone — the default is Qwen/Qwen3-0.6B-Base — with a LayerNorm and a single Linear(hidden, 1) scalar head. When the checkpoint config sets set_head to "attention", Choice questions additionally get a set-scoring path: a Linear(hidden+1, 128) projection that receives the candidate embedding plus log of the candidate count, a 4-head MultiheadAttention over the candidates of the same question, and a final Linear(128, 1) whose weights start at zero so training begins from the scalar path. Boolean questions reuse the scalar logit as the two-vector [0, z]; invalid candidate slots are masked to -1e9 before softmax.

**Parallel decisions are a batch shape, not a trick.** Every (state, question, candidate) triple becomes one "leaf path" of tokens — the state, the question type and instructions, then "Candidate:\n...\nDecision:" and an EOS — as built by prepare_examples in scripts/predict_toy_decisions.py. A single padded forward pass encodes all leaf paths of all questions at once, the last-token hidden states are gathered into a (questions, max_candidates, hidden) tensor, and the scalar and set heads turn that into one logit per candidate. The predictor then softmaxes each row into a distribution. The result object reports autoregressive_decode_steps: 0 because there is genuinely no decode loop anywhere in the path.

**Dynamic candidates are just different leaf sets.** The request schema, validated by validate_request in scripts/predict_toy_decisions.py, accepts Boolean questions (with optional false/true textual criteria), Choice questions with 2–255 named textual candidates, and Score questions with 2–10 ordered level descriptions whose value is the probability-weighted level index. Because candidate identifiers and question IDs never enter the token stream — a property scripts/test_question_contract.py actively verifies — the same model scores a 4-action game state and a 100-option workflow question with no retraining of the output shape.

**Serving loads weights once and answers batches.** scripts/serve_decisions.py is a small standard-library HTTP server that constructs a DecisionPredictor at startup, exposes POST /api/evaluate, and enforces a local demo ceiling of 32 states, 96 questions and 256 candidate paths per request. The browser UI in web/app.js posts the same {states: [...]} JSON to that endpoint, and the response carries per-question probabilities, the argmax decision value, and an execution block that counts forward passes and candidate paths. Inference is CUDA-bound by design: the predictor requires a CUDA device, keeps parameters in float32, and autocasts the forward pass to bfloat16.

**Training mixes policy and observed outcomes.** The unified trainer in scripts/train_unified_games.py warm-starts from a local checkpoint bundle via DecisionPredictor, then trains one shared model on two roles. Policy rows are Choice questions supervised with either API-style distributions or expert action targets under cross entropy; outcome rows are action-conditioned Boolean questions whose gold label is the actual terminal episode success under a frozen continuation policy — never an inferred counterfactual for actions that were not executed. Losses include plain CE, a direct Brier objective, and paired_brier_pg, a Monte Carlo gradient estimator over sampled predictive categories; a BalancedQuestionSampler applies population weights such as 1/3 maze, 1/3 snake and 1/6 per shooting task, as declared in configs/sonic_unified_sft_v1.json, and scripts/unified_td.py can add n-step targets computed for the frozen behavior policy.

**The end-to-end flow closes the loop.** Generators and environments produce decision rows and complete episodes; the trainer warm-starts and fine-tunes the single shared DecisionModel; the selected checkpoint goes back into DecisionPredictor, which serves the web replay pages and powers the next rollout round through scripts/unified_game_pipeline.py; and evaluation scripts — evaluate_unified_checkpoint.py, complete_sonic_evaluation.py, replay_unified_episodes.py and verify_nanojev_comparison.py — re-run the recorded episodes through independent simulator replay before any result is reported. Data, model and verdicts all speak the same question contract, which is what makes the small codebase coherent.

## Advantages

- **Zero-token decisions.** Every answer is a probability distribution read straight from a forward pass, eliminating autoregressive decode steps, answer parsing and their latency.
- **One backbone, many questions.** Boolean, Choice (2–255 candidates) and Score (2–10 levels) all flow through the same scalar-plus-set-attention heads, so new question shapes need no architectural changes.
- **Truly parallel inference.** Candidate paths from independent states and questions are batched into one backbone forward, and the execution block returned by the predictor makes the parallelism measurable.
- **Programmatic, auditable data.** Toy games ship with a real solver (scripts/game_tasks.py), and expert supervision comes from a frozen policy with a pinned source commit and checkpoint hash in scripts/sonic_predict_policy.py.
- **Honest evaluation discipline.** Observed outcomes only, no fabricated counterfactual labels, simulator replay of recorded episodes, and paired seeds across the three-model comparison in scripts/verify_nanojev_comparison.py.
- **Lightweight serving.** The whole inference service is one standard-library HTTP file plus a persistent predictor object, with no provider calls and no network fallback during serving.

## Benefits

- **Fast reaction loops.** Controllers that query the model many times per step — aiming, turning, navigating — get decisions from batched forward passes rather than token streams.
- **Reusable across tasks.** The same checkpoint answers questions for Maze, Snake and both ViZDoom scenarios, because tasks differ only in the text of states, questions and candidates.
- **Clear upgrade path.** The trainer's stage and loss flags, plus the TD target provider, let you extend from behavior cloning toward outcome-driven objectives without new infrastructure.
- **Portable question contract.** The encoding rules are documented against the TypeSafe semantics in docs/TYPESAFE_CONTRACT.md and enforced by runnable offline tests, so integrators know exactly what is and is not compatible.
- **Reproducible releases.** Checkpoint bundles carry config.json, backbone_config, tokenizer and best.safetensors, and the trainer records data, implementation and weight hashes into every run config.
- **Learnable codebase.** With roughly 140 Python scripts, explicit module docstrings and contract tests, the repository reads like a worked example of building a decision model from scratch.

## Usage

The README's quick start clones the repository and installs the toy requirements (torch, transformers, safetensors, numpy pinned versions; inference requires a CUDA device):

```bash
git clone https://github.com/TianyuCodings/NanoJev.git
cd NanoJev
python -m pip install -r requirements-toy.txt huggingface_hub
```

Download the released checkpoint and dataset from Hugging Face under the `unified-games-v1` revision:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="C-Tianyu/NanoJev",
    revision="unified-games-v1",
    local_dir="checkpoints/NanoJev-unified",
    allow_patterns=["best.safetensors", "config.json", "tokenizer/*", "backbone_config/*"],
)
snapshot_download(
    repo_id="C-Tianyu/NanoJev-Data",
    repo_type="dataset",
    revision="unified-games-v1",
    local_dir="data/NanoJev-unified",
)
```

Start the persistent decision service, which loads the model once and serves the replay UI:

```bash
python scripts/serve_decisions.py \
  --checkpoint-dir checkpoints/NanoJev-unified \
  --web-root web --port 8765 --disable-native-triton
```

Decision batches then go to `POST http://127.0.0.1:8765/api/evaluate`. To browse the recorded games without the model, the README also serves the static replay site:

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory web
```

## Conclusion

NanoJev earns the "nano" in its name honestly: the decision model is a backbone, a norm, a scalar head and one optional attention module, and everything else in the repository exists to generate honest data for it, train it under mixed supervision, or prove that its answers behave correctly. For anyone curious how a System-One style decision model — parallel heads, dynamic candidate sets, direct probability readout — is actually wired to environments, trainers and evaluation, this is one of the few places where the entire loop fits in your head at once.

Links:

- GitHub repository: [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)
- Model checkpoint: [C-Tianyu/NanoJev on Hugging Face](https://huggingface.co/C-Tianyu/NanoJev)
- Dataset: [C-Tianyu/NanoJev-Data on Hugging Face](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)
- Jev background: [Introducing System-One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
