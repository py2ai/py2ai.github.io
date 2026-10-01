---
layout: post
title: "quackd: One CLI For All Your Robots - Inside rokbenko/quackd"
description: "quackd is an open-source Python CLI that puts an LLM pilot in front of seven different robot bodies - a walking duck, an SO-101 arm, dual-arm carts, a humanoid, and ROS 2 bases - behind one manifest-driven verb registry and an executor that does not trust the model. We tour the repository: the three loops, the .duck task files, the flock modes, and the honest hardware transcripts."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /Quackd-One-CLI-For-All-Your-Robots/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/quackd/rokbenko-quackd-architecture.svg
tags:
  - Robotics
  - LLM Agents
  - MCP
  - Python
categories: [AI, Open Source]
keywords: "quackd, robot LLM agent, LeRobot SO-101, Microduck, embodied AI, MCP server, MuJoCo simulator, robot flock, .duck task files, decision LLM, Apache 2.0, open source robotics, Rok Benko"
author: "PyShine"
---

Every robot you can buy today already knows how to move. A walking duck balances itself with an onboard policy, a desktop arm picks with a learned skill, a wheeled base drives - and none of it needs an AI model. What the robot lacks is any idea of what those skills are *for*. [quackd](https://github.com/rokbenko/quackd) is an open-source attempt to close that gap: a Python CLI that connects the robots you own, gives each one an LLM for a pilot, and lets a goal stated in plain English drive bodies as different as a bipedal duck and a six-joint arm.

The project's promise is right in its description - one CLI for all your robots, each with a brain. Seven bodies ship as first-party adapters: the Microduck walking robot, an Open Duck Mini v2, LeRobot SO-101-class arms, the dual-arm XLeRobot cart, the two-arms-on-a-lift AlohaMini, the ToddlerBot humanoid, and any wheeled base speaking ROS 2 over rosbridge. The same sentence - "find the ball and kick it" - runs against any of them, because each adapter declares what its body can actually do, and the pilot works only from that declaration.

What makes the repository worth a close reading is not the robot count but the discipline underneath it. The layer that talks to the model never touches the motors directly. A safety executor sits between every model decision and every movement, task files carry machine-checked budgets, and the project publishes the transcript of its own real hardware runs - including the parts that went wrong - with a candor that is rare anywhere in AI tooling.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/quackd/rokbenko-quackd-overview-architecture.svg" alt="Architecture overview of the rokbenko/quackd repository" style="max-width:100%;height:auto;" />
</div>

*High-level architecture overview of the rokbenko/quackd repository, from the CLI and .duck task contract through the deliberation loop, the safety executor, and the manifest-driven robot adapters.*

Reading the overview from left to right: the CLI loads a task contract and resolves the `--robot` flag through an adapter factory that discovers installed bodies via entry points; the agent loop builds an observation and hands the model exactly one turn to answer with one tool call; the executor checks that call against the allowlist, the feasibility verdict, the budgets, and the confirmation rules before any verb runs; verbs exist only if the body's manifest declares them, and they exchange intents and camera frames with the adapter that owns the hardware.

## Why You Need This

The first gap quackd fills is the one between skills and goals. Robots ship with controllers, not intentions: the duck's own firmware walks, kicks and stands up, the arm's servos hold positions, but nothing in that stack can take the sentence "wave to the camera" and turn it into a plan. quackd's answer is to let an LLM choose, once per turn, from verbs that are the robot's real capabilities and nothing more - the model plans, the robot's own controllers execute, and neither pretends to do the other's job.

The second gap opens with the second robot. Every body speaks its own protocol, so a household with three robots ends up commanded from three terminals that do not know each other exist. quackd puts them behind one command line, under names you choose in a persistent registry, and its flock mode lets one goal go to several bodies at once - each pilot is told what the others can do, so the work is divided on data rather than on guesses.

The third reason is trust. A model that can move a physical object is a model you should not fully trust, and the repository takes that seriously in code rather than in disclaimers: verbs that are not in the manifest do not exist anywhere in the system, verbs that move the body are refused until the pilot has judged the task feasible against the body's datasheet, a human is asked where the contract says so, and a kill switch plus a heartbeat guarantee that a stalled model still ends in a stop.

## How It Works

The architecture organizes everything into three loops running at three different speeds, from the robot's own reflexes up to the deliberating pilot.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/quackd/rokbenko-quackd-architecture.svg" alt="Detailed architecture of the rokbenko/quackd repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the rokbenko/quackd repository, covering the CLI surface, the .duck contract, the agent loop and its providers, the executor's gate chain, flock coordination, and the seven adapter packages.*

### Understanding the Architecture

**The deliberation loop runs on a strict diet.** [quackd/agent/loop.py](https://github.com/rokbenko/quackd/blob/main/quackd/agent/loop.py) owns the shape of a turn: one observation in, exactly one tool call out. The observation carries a text summary of what the detector saw, the robot's state, and the last result; frames are kept for the two most recent observations only, so a long run does not balloon the context. Zero tool calls earns one re-prompt and then failure; several tool calls means only the first is honored. The model never issues joint targets - verbs send *intents*, and the adapters own the motion.

**Everything the model may do is decided by a manifest.** [quackd/adapters/manifest.py](https://github.com/rokbenko/quackd/blob/main/quackd/adapters/manifest.py) defines the pydantic contract every adapter returns from `connect()`: what body it is, which sensors and intents it has, which verbs it provides, and a datasheet - height, payload, joint travel, and how sure the project is about each number. The verb registry is built from that manifest at connect time, so a `kick` verb on an arm is not refused; it simply never appears in the tool list.

**The executor is the layer that does not trust the LLM.** [quackd/safety.py](https://github.com/rokbenko/quackd/blob/main/quackd/safety.py) implements a gate chain that every call passes through, whichever door it came from - the agent loop, an MCP client, or the CLI itself: the abort flag, the allowlist, the feasibility verdict, parameter validation, human confirmation, the budget, machine-enforced abort conditions, preconditions, and dry-run. The budget counts steps, wall minutes, and model calls, and `stop` is deliberately exempt from the abort gate so the brake still works after an abort has fired.

**The task contract is a file format, not a prompt.** [quackd/duckfile/](https://github.com/rokbenko/quackd/blob/main/quackd/duckfile/parser.py) parses `.duck` files with strict pydantic frontmatter: the goal, the verbs the pilot may use, and the budgets. Validation checks a task against one or more manifests before anything connects, so a task that needs a camera cannot be run on a body without one. The repository ships fourteen starter tasks at its root and over two hundred example arm tasks under its documentation tree, from "wave" to "rock-paper-scissors".

**One provider interface covers the whole model market.** [quackd/agent/providers/](https://github.com/rokbenko/quackd/blob/main/quackd/agent/providers/factory.py) speaks to Anthropic, OpenAI, Gemini, Grok, Mistral, DeepSeek, Cohere, Qwen, Kimi, GLM and Meta, plus local models through Ollama, vLLM, llama.cpp and LM Studio, plus a `fake` provider for testing with no key at all. A single catalogue module is the source of truth for model ids and prices, and the run's cost is computed from real token counts - a model with no published rate prints `cost unpriced` rather than `$0`, because a frontier model that reads as free is the dangerous failure.

**A run that cannot be argued about is worth nothing.** Every step lands in `runs/<timestamp>/` as a line-by-line JSON transcript: each prompt, each gate that fired, every intent sent to the robot, the three clocks the run was measured by, and what the model cost. `quackd log` replays a finished run through the same renderer the live terminal used. The README applies this honesty to itself - its hero section dissects the project's own hardware run, frame staleness and detector mistakes included.

Put together, a turn flows like this: the adapter reports state and a camera frame, the detector turns the frame into detections, the prompt builder assembles the observation and the manifest-derived tool list, the model returns one verb call, the executor runs its gate chain, the verb sends intents while the steering loop corrects against fresh frames, and the result becomes the next observation - until the pilot declares success, declares failure, or a budget ends the run.

## Advantages

- **Bodies are plugins, not forks.** Each of the seven robots lives in its own distribution installed by an extra like `quackd[lerobot]`; the bare `quackd` package installs none of them, and a third party can publish a new body by declaring one entry point - no pull request required.
- **The three-loop separation defends itself.** Reflexes stay on the robot at its own rate, steering runs in quackd's process, and the LLM deliberates at roughly one turn per few seconds - the model is never in a control loop, and composite verbs never call the model.
- **Safety is enforced, not suggested.** The allowlist, the feasibility verdict against the datasheet, confirmations, budgets, heartbeats and the kill switch are all machine-checked in one place for every caller.
- **Flocks come in two honest flavors.** A pilot flock runs one loop per robot on the wall clock with a shared message bus; a coordinator flock replaces it with a deterministic referee on a lockstep clock and auctions work the way distributed systems actually negotiate.
- **Rehearsal before hardware.** The arm's simulator runs the real backend code over a physics model, `quackd robot twin` clones a calibrated arm into it, and `quackd preflight` sweeps a task file seed after seed and refuses to touch the real arm.
- **It meets you where your agent already lives.** `quackd serve-mcp` exposes nine `robot_*` tools over the Model Context Protocol, so Claude Code or any MCP client can drive the arm one verb at a time through the same executor.

## Benefits

- **One mental model for a heterogeneous fleet.** Learn the verb registry and the `.duck` format once, and the same knowledge drives a duck, an arm, and a ROS base.
- **Switchable brains per run.** The same task file runs against a frontier model, a local model with no API key, or the fake provider in CI - the executor does not care who the pilot is.
- **Cost and time are measured per run.** Token buckets, per-call costs, and three separate clocks land in every transcript, so the question "what did that wave cost" has a file-backed answer.
- **The hallucination surface is structural.** A verb outside the manifest is not a refused call; it is a call the model was never offered, which removes a whole class of prompt-injection failures.
- **Honest documentation you can audit.** The repository states plainly which bodies have run on real hardware and which have not, and walks through the recorded runs including what the robot got wrong.
- **Apache-2.0 and Python 3.11+.** The core install is deliberately light, and every SDK is an optional extra, so a test machine never pulls a deep-learning stack it does not need.

## Usage

Try the duck in a physics simulator with no robot and no API key, using the scripted pilot:

```bash
uvx --from "quackd[mujoco]" quackd run --goal "walk in a square" --robot microduck:mujoco --llm fake
```

Or the cartoon simulator, which needs no download at all:

```bash
uvx --from "quackd[microduck]" quackd run find-and-kick --llm fake
open runs/*/run.gif
```

Set up the SO-101 arm, which is the body the project has actually run on hardware. Python 3.12 is required for the LeRobot extra:

```bash
uv venv --python 3.12
uv pip install "quackd[lerobot,openai]"
quackd doctor --robot lerobot:real --address COM3
quackd robot add arm-01 lerobot:real --address COM3 --llm openai:gpt-6-astra
quackd robot rest-pose arm-01
```

Rehearse the task with the built-in fake pilot, then dry-run it before letting the arm move:

```bash
quackd run lerobot-lookout --robot arm-01 --llm fake
quackd run --goal "Wave to the camera with an extended arm" --robot arm-01 --max-steps 10 --dry-run
quackd run --goal "Wave to the camera with an extended arm" --robot arm-01 --max-steps 10
```

Drive the arm from Claude instead, over the Model Context Protocol - the server refuses to start unless it can reach the rest pose, and the same executor gates every tool call:

```bash
claude mcp add arm -- .venv/Scripts/quackd.exe serve-mcp --robot arm-01
```

Inspect what a run did, after the fact, from its transcript:

```bash
quackd log
```

## Conclusion

quackd is a rare kind of robotics repository: it is less about teaching robots new tricks than about giving the tricks they already have a competent, well-guarded decision maker. The manifest-driven verb registry, the executor that treats every model answer as a request rather than a command, and the transcripts that record everything make it a strong reference implementation for anyone building LLM-piloted hardware. If you own a compatible body - or just want to watch a duck walk a square in the browser - the simulator runs in sixty seconds with no key and no robot, and the code rewards the read.

**Links:**

- Repository: [github.com/rokbenko/quackd](https://github.com/rokbenko/quackd)
- Browser simulator: [quackd.org/simulator](https://www.quackd.org/simulator)
- Architecture guide: [docs/architecture.md](https://github.com/rokbenko/quackd/blob/main/docs/architecture.md)
- MCP integration: [docs/mcp.md](https://github.com/rokbenko/quackd/blob/main/docs/mcp.md)
- Flock modes: [docs/flock.md](https://github.com/rokbenko/quackd/blob/main/docs/flock.md)
- Decision LLMs: [docs/decision-llms.md](https://github.com/rokbenko/quackd/blob/main/docs/decision-llms.md)
- The arm's documentation: [docs/adapters/lerobot.md](https://github.com/rokbenko/quackd/blob/main/docs/adapters/lerobot.md)
