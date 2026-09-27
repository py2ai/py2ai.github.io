---
layout: post
title: "Jev Gym: A System-One Decision Agent Controls Any Gymnasium Environment"
description: "jev-gym is a System-One decision agent that controls any gymnasium environment by answering three typed questions at every step - direction, at_risk, instability - while a motor layer turns those answers into actions. A source-level tour of pyshine-labs/jev-gym: the typed question contract, three decision engines with graceful fallback, the per-env motor translation, a Flask console with a live decision view, and the three-stage fine-tuning that made the answers control-grade."
date: 2026-09-27
header-img: "img/post-bg.jpg"
permalink: /Jev-Gym-A-System-One-Decision-Agent-Controls-Any-Gymnasium-Env/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/jev-gym/pyshine-labs-jev-gym-architecture.svg
tags:
  - Reinforcement Learning
  - Python
  - AI Agents
  - Open Source
  - Gymnasium
categories: [AI, Open Source]
keywords: "jev-gym, Jev agent, System One decision agent, gymnasium environments, typed questions, decision engine, CartPole, MountainCar, LunarLander, BipedalWalker, ModernBERT encoder, laya decision model, OpenRouter Decisions API, DAgger, imitation learning, Flask WebUI, motor layer, control law, choose_action"
author: "PyShine"
---

Ask a large language model to balance a pole and you get a paragraph. Text generation is the wrong output shape for control: a controller needs a verdict at every tick, not an essay. [jev-gym](https://github.com/pyshine-labs/jev-gym) takes the opposite approach. It wraps a **System-One decision agent (Jev)** that never generates text - instead, at every step of a [gymnasium](https://gymnasium.farama.org/) environment it answers three typed questions about the state: `direction` (a choice: left or right), `at_risk` (a noul: a yes/no belief returned as a probability) and `instability` (a score: how unstable, 0 to N). A separate motor layer translates that assessment into the actual environment action. The same stack drives CartPole, MountainCar, MountainCarContinuous, Acrobot, Pendulum, LunarLander and BipedalWalker, and a Flask WebUI shows the entire loop live: state in, typed decision, action out with the reason, and the reward curve climbing from step zero.

The design is worth studying because it splits a problem that is usually smeared together. Assessing a situation - "the pole is leaning left, risk is low, instability moderate" - is separated from acting on it, and the assessment is expressed as a fixed, typed schema with probabilities and a confidence attached to every answer. That structure is what lets one decision stack serve seven environments with wildly different action spaces, what lets a tiny offline head stand in for a GPU model without changing a line of calling code, and what makes the agent's behavior inspectable at every step instead of hidden inside a policy network's logits.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jev-gym/pyshine-labs-jev-gym-overview-architecture.svg" alt="Architecture overview of the jev-gym repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview: the environment session frames every observation into a canonical 4-D state, the decision stack answers typed questions through one of three engines, the motor layer turns the answers into the env's native action, and the Flask console renders the whole loop as it runs.*

Reading the overview left to right is reading one step of the agent. The env session loop in `webui.py` steps a real gymnasium environment and hands the raw observation to `canonical_state` in `jev_agent.py`, which maps any supported observation - CartPole's four floats, Pendulum's cosine/sine pair, LunarLander's eight values - into one shared cart frame `[x, x_dot, theta, theta_dot]`. That single framing decision is what makes the rest of the system env-agnostic: the decision stack always speaks the same language no matter what it is watching. The framed state then goes to the engine layer, where one of three interchangeable backends answers the typed question set: a fully offline physics-informed head, the open laya decision model loaded from a local checkpoint, or the hosted Jev model behind the OpenRouter Decisions API.

The answers come back as typed objects - choice, noul and score values, each with per-option probabilities, a confidence and the name of the engine that decided. From there the motor layer takes over: `choose_action` in `webui.py` translates the answers into the env's native action space, with the tuned CartPole law in `controller.py` adding a track-limit guard on top of the direction answer. Whatever happens, the loop closes through the Flask endpoints into the single-page console, which draws the state bars, the Jev node and the action-out box in real time. The sections below zoom into each of those boxes.

## Why You Need This

If you have trained or served reinforcement learning policies, you know the standard failure modes. A PPO network controls well but explains nothing - you cannot ask it why it pushed left, and you cannot partially trust it. An LLM agent explains everything but is too slow and too verbose for per-step control, and its free-form output is fragile to parse. And a hand-written control law is fast and transparent but lives in its own little world: the CartPole solver tells you nothing about the lander.

jev-gym positions itself exactly in the gap. The typed questions are the contract: `direction`, `at_risk`, `instability` (plus `side_engine` for LunarLander, and momentum-phrased `pump` choices for MountainCar and Acrobot). Because the contract is fixed, the engine behind it is swappable - the same calling code runs against the offline head, the open laya checkpoint, or a hosted model - and because the answers carry probabilities and confidence, the consumer decides how much to trust each answer at each step. The motor layer then does the env-specific part once, in one readable function, instead of hiding it inside training.

This is useful for three audiences. If you build agents, it is a working template for System-One decision loops - assess with typed questions, act with a translation layer, fall back gracefully when a backend fails. If you teach or study control and imitation learning, the repository ships a complete, reproducible three-stage fine-tuning recipe that turns a pre-trained decision model's answers into control-grade ones. And if you just want a genuinely watchable demo, the console shows the agent thinking: probability bars, risk sparkline, decision trail, reward curve - every step accounted for.

## How It Works

The diagram below maps the real subsystems of the repository and how they connect, from the HTTP surface down to the training scripts that produced the shipped checkpoint.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jev-gym/pyshine-labs-jev-gym-architecture.svg" alt="Detailed architecture of jev-gym: Flask WebUI, Jev agent core, motor layer, training stages and CLI tools" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the Flask routes and session loop, the agent core with its typed contract and three engines, the motor translation with its laws and learned motors, the three training stages, and the CLI utilities around them.*

### Understanding the Architecture

**The typed contract: Question and Answer.** Everything starts in `jev_agent.py` with two small dataclasses. A `Question` carries a name, a type - `choice`, `score` or `noul` - natural-language instructions, and optional criteria (an options dict for choices, a label list for scores). An `Answer` carries the value, per-option probabilities, a confidence and the engine that produced it. No text is ever generated; the model fills a schema. This is the entire interface between "thinking" and "acting", and its smallness is the point: any backend that can fill this schema can drive the agent.

**Framing any observation: canonical_state.** The `canonical_state` function maps each supported observation family into the agent's 4-D cart frame `[x, x_dot, theta, theta_dot]`. CartPole passes through unchanged; Pendulum's `[cos, sin, omega]` becomes `atan2` angle plus angular velocity; Acrobot contributes its second-link angle and velocity; MountainCar maps position and velocity with neutral angles; LunarLander contributes horizontal position, lateral velocity, angle and angular velocity. One decision stack, one state language, seven environments.

**The question sets.** `questions_for(env_id)` selects the set that matches each family's control contract: the default trio `direction` / `at_risk` / `instability` for CartPole and Pendulum, a `side_engine` choice (left-engine / right-engine / none) for LunarLander, and momentum-phrased `pump` choices for MountainCar and Acrobot whose instructions describe energy pumping in plain physics language. The instructions matter - they are the model's entire view of the task, and they read like a physics teacher's wording rather than a prompt-engineering trick.

**Three engines, one contract.** The `make_engine` factory returns one of three backends. `LocalDecisionHead` is fully offline and depends on nothing beyond numpy: it computes shared features once per call (a lean estimate, a momentum alignment, a normalized risk) and emits calibrated probabilities through sigmoids - the "single forward pass" of a physics-informed head with no network at all. `LayaBackend` loads the open laya decision model, preferring the task-tuned checkpoint at `.models/laya_gym/` and falling back to the base `.models/laya/` download; on a GPU it answers in roughly 38 ms per step. `JevBackend` sends the state and the typed questions to `typesafe/jev-1.13` through the OpenRouter Decisions API (about 70-500 ms per step, needs `OPENROUTER_API_KEY`). Both remote backends degrade gracefully: an unparseable laya answer falls back per-question, any jev error falls back entirely, and every answer records which engine decided - the pole never drops because of a network hiccup.

**The session loop and the HTTP surface.** `webui.py` exposes six routes: `GET /` serves the console page, `GET /api/envs` lists the served environments by filtering the gymnasium registry, `GET /api/arch` returns real layer facts read from the checkpoint configs, and `POST /api/start`, `/api/step`, `/api/stop` build, drive and close a single locked `Session`. The session owns the gym environment, applies the decimate setting (ask the engine every N-th step and hold the answers in between), accumulates the reward, keeps a rolling decision trail of the last steps, and reports each snapshot as JSON: frame, step count, reward, average decision time, average confidence, risk, and the per-env success verdict computed from gymnasium's own terminated/truncated semantics.

**The motor layer.** `choose_action` is the translation from typed answers to the env's native action, and it is honest about what each env actually needs. With **Jev drives** on: CartPole's direction answer drives push 0/1 with a wall guard from `controller.py` that pushes back toward the track center near the ends; MountainCar pumps 0/2 and MountainCarContinuous thrusts -1/+1 on the direction answer; Acrobot fires hip torque 0/2; Pendulum is the subtle one - the instability score gates the pump intensity of an energy-pumping swing-up while a PD law always catches near upright; LunarLander's side_engine answer fires the 1/3 laterals with the main engine reserved for the fall-speed bound; and BipedalWalker keeps its learned PPO gait, because 6-D joint torques cannot come from a left/right answer. With **Jev drives** off, the per-family physics laws or the learned PPO motors (`LEARNED`, loaded from stable-baselines3 checkpoints when present) drive directly while Jev still assesses every step - the same stack assesses without controlling.

**What the console shows.** `templates/index.html` is a single page with a canvas flow diagram: state-in vertical bars on the left, the Jev node in the middle glowing with the active engine's color (direction plus confidence, at_risk and instability bars), and the action-out box on the right with a human-readable explanation of why that action was selected. Around it: per-component state and action strip charts with legends, the cumulative reward curve drawn from step 0, typed decision cards with per-option probabilities, a risk sparkline, a step I/O panel, the decision trail table, and a model architecture panel whose layer counts, attention types and hidden sizes are read from the real checkpoint configs - frozen encoder versus task-trained head versus motor, not a marketing diagram.

**The training stages.** The shipped `.models/laya_gym` checkpoint is the product of three scripts, each reproducible. `train_laya_gym.py` (stage 1) runs each env's passing control law, labels every visited state with typed answers, and fine-tunes the frozen ModernBERT encoder's top layers plus the typed-decisions head on all envs at once. `refine_gym.py` (stage 2) drops the law's own flip-ambiguity zones - like CartPole states where the control signal is razor-thin - oversamples the decisive states and trains env-specific epochs on top. `refine_lander_dagger.py` (stage 3) is the DAgger fix: it flies LunarLander episodes driven by the tuned model itself, labels those exact states with the passing law, and retrains on the mixture, closing the compounding-error gap that took the lander from -264 to +256 mean return. `scoreboard.py` then verifies in pure-laya mode: 5 episodes per env, the model's own answers driving everything, no confidence fallback and no substitution.

**Data flow, one step at a time.** Follow a single tick through the diagram: the browser posts `/api/step`, the session asks the active engine the typed question set for the env, the engine frames the state (canonically, or as a per-env physics payload for laya) and returns typed answers with probabilities, `choose_action` maps the answers through the motor layer to the env action, the gym environment steps, and the snapshot - frame, reward, trail entry, decision timing - returns to the console, which pulses the flow canvas and appends the charts. Every arrow in the diagram is exercised once per step, and every intermediate value is visible in the UI.

## Advantages

- **Typed answers, not prose.** Every decision is a fixed schema - choice, noul or score - with probabilities and confidence. Consumers get calibrated trust signals instead of text to parse, and the UI can chart anything the agent believes.
- **One canonical frame across environments.** Because every observation maps to the same 4-D cart state, adding an environment is a mapping plus a translation, not a new agent. The README's "adding a new env" recipe is three steps.
- **Engines that fail soft.** The local head has zero dependencies and answers instantly; laya and the hosted jev model fall back to it per question or on error, and every answer is tagged with the engine that decided. The agent never stalls waiting for a backend.
- **Assess without controlling.** The **Jev drives** toggle separates assessment from action: switch it off and the physics laws or PPO motors drive while the typed answers still stream to the UI - a built-in mode for comparing model beliefs against ground-truth controllers.
- **A training recipe you can rerun.** Law imitation, margin-filtered refinement and DAgger are three plain scripts over the same label builders, taking roughly 25-30 minutes on an RTX 4060 Ti. The encoder stays frozen, so only the typed head and top layers train and the 421M-parameter base is untouched.
- **Verification is part of the design.** `scoreboard.py` runs pure-laya episodes per env, `verify_all.py` walks every served env through the live HTTP API, and `_verify_seeds.py` sweeps hundreds of seeds before a learned motor is exposed in the dropdown.

## Benefits

- **You can watch the reasoning.** Probability bars, confidence, risk sparkline, decision trail and the action-out explanation mean every step is accountable - ideal for teaching, demos and debugging, and rare even in commercial agent tooling.
- **No cloud required to start.** `pip install -r requirements.txt` and the local head drive CartPole offline with nothing but numpy and gymnasium. The laya checkpoint and the OpenRouter key are optional upgrades, not prerequisites.
- **Fast enough for real control.** The local head answers in microseconds and laya averages around 38 ms per step on a consumer GPU - comfortably inside a control loop, where a text-generating LLM is not.
- **Control-grade out of the box.** The shipped `.models/laya_gym` checkpoint passes 5/5 episodes on all 7 served envs with pure-laya answers: CartPole the full 500 steps, MountainCar solved, MountainCarContinuous at 92.1, Acrobot and Pendulum to their goals, LunarLander around 256 and BipedalWalker around 319.
- **Honest engineering throughout.** Fallbacks are structural rather than silent, the architecture panel reads real checkpoint configs, success verdicts come from each env's own objective, and a learned policy is only exposed after it passes the seed sweep.
- **Open and inspectable.** The whole system is a handful of readable Python files under an open repository - the decision contract, the motors and the training stages can each be read in an afternoon.

## Usage

Clone and set up. The core stack needs only four packages; the optional extras add the Box2D environments, CUDA torch for fast laya inference, and the laya backend itself:

```bash
git clone https://github.com/pyshine-labs/jev-gym.git && cd jev-gym
python -m venv .venv
.venv\Scripts\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Optional but recommended:
pip install swig "gymnasium[box2d]"     # LunarLander + BipedalWalker envs
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU
pip install laya                        # laya backend for --engine laya
```

Fetch the laya decision-model checkpoint (a resumable download into `.models/laya/`):

```bash
python _fetch_laya.py
```

Launch the console and open http://127.0.0.1:7860. Pick an environment, an engine, the **Jev drives** toggle, min-confidence, decimate, seed and speed, then press START:

```bash
python webui.py            # http://127.0.0.1:7860
```

Prefer the terminal? `run_agent.py` runs the agent's home task, CartPole-v1, with flags for engine, episodes, seed, min-confidence, decimate, rendering, verbose per-step printing and a CSV decision trail:

```bash
python run_agent.py --engine local --episodes 3
python run_agent.py --engine laya --render --verbose --log trail.csv
```

To reproduce the shipped decision-engine training - three stages, each building on the last, ending with the pure-laya verification - run:

```bash
# stage 1 - multi-task law imitation -> .models/laya_gym/
python train_laya_gym.py

# stage 2 - margin-filtered refinement on top of stage 1
python refine_gym.py

# stage 3 - DAgger: laya-driven flights labeled by the law
python refine_lander_dagger.py

# verify - 5 episodes per env, pure-laya answers drive everything
python scoreboard.py
```

For learned motor layers, `_train_ppo.py` and `_train_continue.py` train PPO policies for any continuous or discrete task, and `verify_all.py` plus `_verify_seeds.py 300` check them end to end before you add a checkpoint to the `LEARNED` list in `webui.py`.

## Conclusion

jev-gym is a clean argument that control does not need text generation - it needs typed questions, calibrated answers and a motor layer that respects each environment's reality. The repository demonstrates the full arc in one place: a schema small enough to hold in your head, three interchangeable engines with graceful degradation, a per-env translation layer you can actually read, a console that makes every step watchable, and a reproducible three-stage fine-tuning recipe that lifted the model's own answers to control grade across all seven environments. Clone it, press START on CartPole, and watch the typed answers drive the pole - then flip **Jev drives** off and see the difference between assessing and acting.

**Links:**

- Repository: [https://github.com/pyshine-labs/jev-gym](https://github.com/pyshine-labs/jev-gym)
- Gymnasium documentation: [https://gymnasium.farama.org](https://gymnasium.farama.org)
- OpenRouter: [https://openrouter.ai](https://openrouter.ai)
- OpenRouter Jev tutorial: [https://openrouter.ai/blog/tutorials/how-to-use-jev/](https://openrouter.ai/blog/tutorials/how-to-use-jev/)
- laya model card: [https://huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
- More projects and write-ups: [https://pyshine.com](https://pyshine.com)
