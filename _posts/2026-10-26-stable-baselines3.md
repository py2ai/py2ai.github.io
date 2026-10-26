---
layout: post
title: "Stable-Baselines3 Source Tour: How PyTorch Reinforcement Learning Is Wired"
description: "A guided walk through the DLR-RM/stable-baselines3 codebase: BaseAlgorithm, the on-policy and off-policy training loops, buffers, policies, vectorized environments, and the hooks that make PPO, SAC, and DQN tick."
date: 2026-10-26
header-img: "img/post-bg.jpg"
permalink: /stable-baselines3/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/stable-baselines3/dlr-rm-stable-baselines3-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: stable baselines3, reinforcement learning, ppo sac, pytorch rl, gymnasium, dqn, source code tour, actor critic
author: "PyShine"
---

Reinforcement learning has a reputation for being hard to implement correctly. Small details — how advantage estimates are normalized, when a target network copies weights, how entropy is folded into the objective — decide whether an agent learns or silently diverges. [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) from DLR-RM is the reference-quality answer to that problem: a faithful PyTorch reimplementation of the classic Baselines algorithms, built on [Gymnasium](/gymnasium/) environments, with readable code and consistent structure across every learner it ships. If you have ever trained PPO, SAC, DQN, or A2C and wondered what actually happens between `model.learn()` and a saved zip file, this tour is for you.

What makes this codebase worth reading is not just the algorithms — it is the shared scaffolding underneath them. One abstract base class defines the contract for all learners; two loop templates cover the on-policy and off-policy families; a small set of buffers, policy factories, and vectorized environment wrappers fills in the rest. Because the scaffolding is shared, reading one algorithm teaches you the shape of all of them. That consistency is also why the library has become the de facto baseline in academic RL papers: when a paper says "SB3 default hyperparameters," reviewers know exactly what that means.

This post walks the source top-down. We start with a high-level architecture diagram, zoom into a detail diagram that maps every major module, trace a first training run line by line, and finish with an honest look at where the library's abstractions bend. Everything referenced below links to a real symbol at a real line number in the `master` branch, so you can follow along in your editor.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/stable-baselines3/dlr-rm-stable-baselines3-overview-architecture.svg" alt="Stable-Baselines3 architecture overview" style="max-width:100%;"></div>

<p><em>The overview graph shows how two loop templates and five algorithm classes hang off one abstract base, with buffers, policies, vectorized environments, and callbacks as shared services.</em></p>

**Reading the overview from left to right:**

- [BaseAlgorithm, L67](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/base_class.py#L67) is the abstract parent of every learner: it owns the environment reference, device selection, seeding, logging, and the save/load machinery that all children inherit.
- [OnPolicyAlgorithm, L21](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py#L21) implements the rollout loop — collect a batch of fresh experience, train on it, throw the data away, repeat.
- [OffPolicyAlgorithm, L27](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/off_policy_algorithm.py#L27) implements the replay loop — store every transition forever (up to capacity), sample mini-batches, and interleave environment steps with gradient updates.
- [PPO, L18](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py#L18) and [A2C, L16](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py#L16) are the on-policy children; most of their files are just objective functions and hyperparameters.
- [SAC, L19](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py#L19) and [DQN, L19](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py#L19) are the off-policy children, adding entropy regularization and target networks respectively.
- [RolloutBuffer, L343](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L343) and [ReplayBuffer, L158](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L158) implement the two storage disciplines — fixed-size batches versus a large ring of transitions.
- [ActorCriticPolicy, L416](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L416) is the default neural policy: a feature extractor plus actor and value heads, reused by PPO and A2C.
- [VecEnv, L50](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/base_vec_env.py#L50) defines the vectorized environment interface — the reason a single `learn()` call can step eight or sixteen simulators in lockstep.
- [BaseCallback, L31](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py#L31) provides the hook points that let you inject evaluation, checkpointing, or custom logic without touching algorithm code.

## From Install to Your First Model

Getting running takes three lines beyond the install. The idiomatic first program looks like this:

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=8)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_cartpole")
```

Each line maps to a specific module. [make_vec_env, L40](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/env_util.py#L40) builds a vectorized environment, wrapping each copy in a `Monitor` that records episode rewards and lengths. The `PPO` constructor walks up the inheritance chain to [BaseAlgorithm._setup_model](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/base_class.py#L252), which resolves the policy string `"MlpPolicy"` into a real network, creates the rollout buffer, and configures the learning-rate schedule — the library expresses schedules as a simple [Schedule, L29](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/type_aliases.py#L29) callable that maps a progress fraction to a float.

The call to `model.learn()` lands at [learn, L512](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/base_class.py#L512) in the base class, which normalizes arguments, wires callbacks, and delegates to the concrete subclass. From there the rollout loop takes over: `collect_rollouts` runs the policy against the vectorized environment until the buffer holds `n_steps * n_envs` transitions, then `train()` performs the clipped-surrogate updates. Finally [save, L819](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/base_class.py#L819) bundles parameters and metadata into a zip artifact, and [load, L643](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/base_class.py#L643) reconstructs the agent from it. The round trip is deliberately boring — no pickle of whole objects, just data you can inspect.

One habit worth copying from the docs early: evaluate deterministically, and do it automatically with [EvalCallback, L342](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py#L342), which runs [evaluate_policy, L12](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py#L12) every `eval_freq` steps, keeps the best checkpoint, and can stop training early when rewards plateau. The callback API is small — `_on_training_start`, `_on_step`, `_on_rollout_end` — but it covers essentially every integration need without forking the library.

Reproducibility gets first-class treatment too. `set_random_seed` threads a seed through PyTorch, NumPy, and the environment stack at construction time, and the base class picks the device automatically — CPU, GPU, or Apple silicon via MPS — while exposing it as a constructor argument. The [logger](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/logger.py#L1) module is the quiet hero here: every algorithm reports episode reward, loss terms, and exploration statistics through the same key-value interface, so swapping `verbose=1` console output for a TensorBoard dashboard is a one-line `TensorBoardCallback` away. When a run misbehaves, you can diff two experiments' logged keys rather than guessing from stdout noise.

## On-Policy vs Off-Policy: Two Loops, One Base Class

The single most instructive contrast in the codebase is between the two loop templates. The on-policy loop lives in `OnPolicyAlgorithm.learn`, and its heart is [collect_rollouts, L162](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py#L162): step every environment, store observations, actions, rewards, dones, and log-probabilities into the [RolloutBuffer](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L343), and let the policy's value function bootstrap the terminal state. When the buffer fills, [train, L270](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py#L270) computes advantages with generalized advantage estimation and runs several epochs of minibatch gradient descent — for PPO that means the famous clipped ratio between new and old policy probabilities. The data is then dead weight: the next rollout starts from a fresh buffer, because the policy that generated it no longer exists.

The off-policy loop inverts the economics. [OffPolicyAlgorithm.learn, L324](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/off_policy_algorithm.py#L324) keeps a persistent [ReplayBuffer](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py#L158) of `buffer_size` transitions and, each step, performs `gradient_steps` updates sampled uniformly from history. Its own [collect_rollouts, L526](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/off_policy_algorithm.py#L526) also handles exploration strategies and warmup: until `learning_starts` transitions exist, actions come from the exploration noise rather than the policy. DQN plugs epsilon-greedy and target-network syncing into [train, L190](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py#L190); SAC replaces exploration noise with an entropy-regularized stochastic policy tuned automatically in [train, L205](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py#L205); TD3 adds twin critics and delayed policy updates at [train, L160](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/td3.py#L160).

The practical consequence: on-policy algorithms spend most of their wall-clock time collecting environment steps and are sample-inefficient but stable; off-policy algorithms squeeze many gradient steps out of each environment interaction and excel when simulation is expensive. The base class keeps both loops honest — progress reporting, callback dispatch, logging cadence, and evaluation hooks behave identically whether you are training PPO or SAC, because the loop templates share that code.

## Inside the Detail Diagram

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/stable-baselines3/dlr-rm-stable-baselines3-architecture.svg" alt="Stable-Baselines3 detail architecture" style="max-width:100%;"></div>

<p><em>The detail graph expands each overview box into real modules: policy factories and their network pieces, the buffer family including hindsight replay, concrete vectorized environments, and the tooling belt of callbacks, evaluation, and plotting.</em></p>

Zooming in, three subsystems deserve attention. First, the policy stack: [BasePolicy, L280](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L280) is the root class, and concrete families specialize it — [SACPolicy, L181](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py#L181) for squashed Gaussian actors, [TD3Policy, L86](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/policies.py#L86) for deterministic actors with twin critics (which DDPG reuses verbatim), and [DQNPolicy, L88](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/policies.py#L88) for Q-networks. Under them sit shared neural building blocks in `torch_layers` (the NatureCNN for Atari-style pixels, MlpExtractor for actor-critic heads) and a clean probability layer in `distributions` that handles discrete, Gaussian, and state-dependent-noise action spaces uniformly.

Second, the environment stack. [DummyVecEnv, L15](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/dummy_vec_env.py#L15) steps simulators sequentially in-process and is the right default; [SubprocVecEnv, L79](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py#L79) moves each environment into its own process when Python-side simulation cost dominates. Wrappers compose freely: [VecNormalize, L15](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_normalize.py#L15) maintains running statistics to normalize observations and rewards, and because it is a wrapper, it composes with either backend. The `Monitor` wrapper records per-episode statistics to csv so that `plot_results` can turn training logs into learning curves later.

Third, the buffer family. Beyond the two core buffers, [HerReplayBuffer, L15](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/her/her_replay_buffer.py#L15) implements hindsight experience replay for goal-conditioned tasks — after an episode ends, failed trajectories are relabeled so the achieved goal becomes the intended one, turning sparse-reward failures into dense learning signal. The type-safe sample structures live in `type_aliases`, which is also where the [Schedule](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/type_aliases.py#L29) alias used for learning rates and clip ranges is defined.

## Honest Limits

Stable-Baselines3 is deliberately a library of proven classics, not a catalog of the frontier. You will not find model-based RL, distributional methods like Rainbow, population-based training, or the latest sequence-modeling agents here — the maintainers explicitly prioritize correctness, documentation, and consistency over algorithm count. If your research needs something exotic, expect to write it yourself, though the loop templates make that less painful than it sounds: a new on-policy method is usually a subclass that overrides `train()`.

The abstraction also has a cost. Everything is tuned for the vectorized-environment regime, so single-environment training takes a configuration detour, and custom observation wrappers must respect the Gymnasium API precisely or you will hit shape errors deep inside buffer indexing. Recurrent policies exist but are more involved than the feedforward default, and large-scale distributed training is out of scope — the library assumes one machine, one learner, many simulators. For RLHF-scale infrastructure where hundreds of workers feed a central learner, projects like [OpenRLHF](/openrlhf/) live in a different weight class, even though they still lean on SB3-style algorithmic conventions.

Documentation and testing, to be fair, are where this project shines rather than struggles. Every algorithm ships with the same benchmark plots, docstring hyperparameter references, and type hints. The main practical pitfall is subtler: default hyperparameters are tuned per environment suite, so PPO defaults that ace Atari can underperform on continuous control unless you adjust `n_steps`, `batch_size`, and the entropy coefficient. Reading the source — especially the annotated `train()` implementations — is the fastest way to build the intuition for which knob does what.

## Where to Go Next

A good reading order inside the repo: start with `base_class.py` for the lifecycle, then `on_policy_algorithm.py` next to `buffers.py` to see the rollout loop and its storage as one picture, then `ppo.py` as the most complete on-policy objective, and finish with `sac.py` plus `sac/policies.py` to watch entropy regularization thread through an off-policy learner. Along the way you will notice how much heavy lifting the Gymnasium integration does — the observation and action space objects drive buffer shapes, network input sizes, and distribution choices automatically, which is why respecting the [Gymnasium](/gymnasium/) API contract matters so much.

If this tour was useful, the natural companions on this blog are the [Gymnasium](/gymnasium/) environment tour, the [PyTorch](/pytorch/) internals walkthrough for the tensor-level machinery underneath, and [scikit-learn](/scikit-learn/) for the classical-ML contrast in API design. And if your interest in RL is really about aligning large models, [OpenRLHF](/openrlhf/) shows where the same on-policy loop ideas scale to PPO-based post-training of LLMs — same concepts, very different plumbing.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Stable-Baselines3 Source Tour: How PyTorch Reinforcement Learning Is Wired",
  "description": "A guided walk through the DLR-RM/stable-baselines3 codebase: BaseAlgorithm, on-policy and off-policy training loops, buffers, policies, vectorized environments, PPO, SAC, and DQN.",
  "datePublished": "2026-10-26",
  "author": "PyShine",
  "keywords": "stable baselines3, reinforcement learning, ppo sac, pytorch rl, gymnasium, dqn"
}
</script>
