---
layout: post
title: "Microduck RL: Sim2Real Locomotion for an 800-Gram Biped - Inside pollen-robotics/microduck_rl"
description: "How Pollen Robotics trains walking, recovery, and trick policies for its ~800 g Microduck biped: mjlab and MuJoCo Warp at 50 Hz, BAM actuator physics, backlash twins, a 61-dimensional hot-swappable observation contract, and a clean ONNX path from GPU training to the real robot."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /Microduck-RL-Sim2Real-Locomotion-Pollen-Robotics/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/microduck_rl/pollen-robotics-microduck_rl-architecture.svg
tags:
  - Reinforcement Learning
  - Robotics
  - Sim2Real
  - MuJoCo
categories: [AI, Open Source]
keywords: "microduck_rl, pollen robotics, bipedal robot, reinforcement learning, sim2real, mjlab, mujoco warp, PPO, rsl_rl, domain randomization, BAM actuator model, ONNX policy export, Dynamixel XL330, reward design, Hugging Face Jobs"
author: "PyShine"
---

Bipedal balance is brutal at every scale, but there is a special circle of difficulty reserved for robots the size of a bottle of water. At roughly 800 grams and 25 centimeters tall, the Microduck biped from Pollen Robotics — the team behind the Reachy robots — is carried by fourteen Dynamixel XL330 hobby servos whose gearboxes have real play, whose encoders sit on the wrong side of that play, and whose friction behavior refuses to look like the textbook. When a policy trained in simulation meets hardware like that, the sim-to-real gap is not a footnote; it is the whole project. `pollen-robotics/microduck_rl` is the repository where that gap gets fought, policy by policy.

Microduck RL is the training half of the Microduck project: a family of reinforcement learning environments for the little biped, built on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) and trained with PPO through rsl_rl. Policies are trained at 50 Hz, exported to ONNX, and then run on the real robot by the onboard runtime that lives in the sibling `pollen-robotics/microduck` repository. The task list reads like a tiny circus program: velocity-tracked walking on flat and rough terrain, walking-plus-fall-recovery, standing up from any flop, commanded sit-to-stand, crouching to touch the ground with the mouth tip, kicking a small ball, a forward roll over the head, and an entire roller-skating branch with passive wheels under the feet — gliding, swizzling, crouching on wheels, even standing up onto them.

What makes the source worth a tour is that it encodes a complete, opinionated sim2real recipe rather than a pile of hyperparameters. The README points you at the actuator modeling, the domain randomization, and the backlash simulation; `CLAUDE.md` distills the reward-design lessons learned across the whole project, written for humans and coding agents alike. Between the two, plus roughly six thousand lines of custom MDP functions in `src/mjlab_microduck/tasks/mdp.py`, you can watch every hard-won decision in context: why the observation vector is shaped the way it is, why domain randomization must never accumulate across resets, and why a penalty with the wrong sign quietly teaches the robot to farm the violation.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/microduck_rl/pollen-robotics-microduck_rl-overview-architecture.svg" alt="Architecture overview of the pollen-robotics/microduck_rl repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the microduck_rl architecture: the training entry and remote job submission on the left, the task registry and environment recipes in the middle, the MDP library and the robot/actuator models feeding them, and the ONNX export plus CPU deployment rehearsal on the right.*

Reading the overview from left to right: the `train` entry point (`src/mjlab_microduck/train_cli.py`) either forwards to mjlab's trainer or, with the `--hf-jobs` flag, submits the whole run to Hugging Face Jobs via `src/mjlab_microduck/hf_jobs.py`. The task registry (`src/mjlab_microduck/tasks/__init__.py`) builds environment configs and registers the `-Backlash-` twins that `src/mjlab_microduck/tasks/backlash.py` manufactures by wrapping any base config. Every recipe pulls its rewards, events, and observations from the MDP library and its robot definition from `src/mjlab_microduck/robot/microduck_constants.py`, which in turn loads the MJCF models and configures the BAM actuator from `src/mjlab_microduck/actuator/friction_dr_bam.py`. On the far right, `scripts/export.py` turns a trained checkpoint into a deployment-ready ONNX file, and `scripts/infer_policy.py` rehearses that exact deployment in CPU MuJoCo before anything touches hardware.

## Why You Need This

The first problem this repository solves is the one that dominates small-robot sim2real: actuator fidelity. At this scale the servos are most of the physics, so the project refuses to model the XL330 as an ideal PD controller. All tasks use the BAM M6 actuator model — a voltage control law with back-EMF and Coulomb, Stribeck, and load-dependent friction — and the thin subclass in `src/mjlab_microduck/actuator/friction_dr_bam.py` adds per-environment domain randomization on top of it. `FrictionDRBamActuator` rescales the friction budget every episode, and the environment recipes randomize battery voltage, voltage sag under load, command delay, and encoder bias. The result is a simulation whose hardest uncertainty is deliberately the one you will meet on the hardware.

The second problem is orchestration: a walking robot needs more than one skill. Walking, recovering from a fall, sitting down gently, and performing a trick cannot be one giant policy, but swapping policies at runtime only works if every network speaks the same language. This repo enforces a shared 61-dimensional actor observation — 48 proprioceptive values plus a command block of twist, head pose, and body pose — across the entire policy family. Environments that do not use a command slot zero-pad it instead of dropping it, so the runtime can hot-swap a walking policy for a recovery policy or a roulade policy mid-stride. `scripts/infer_policy.py` lets you rehearse exactly that swap, keyboard-driven, before the robot is anywhere near the loop.

The third problem is mechanical honesty. Real servos have gear play, and the real encoder sits on the output side of it, so the firmware's position loop closes through the backlash. Every main task has a backlash twin — trained on a model with plus-or-minus one degree of play in series with each servo — generated by `src/mjlab_microduck/robot/microduck/add_backlash.py` and wired up by `make_backlash_variant()` in `src/mjlab_microduck/tasks/backlash.py`. The actuator's firmware PD emulation reads through the play, the joint observations read through the play, and because observation and action dimensions are unchanged, the ONNX export and the runtime need no modifications at all.

Finally, the repository solves the iteration-cost problem that quietly kills most RL projects: long runs that die at 3 a.m. or configs that crash after hours. A five-iteration smoke test at 64 environments is documented as mandatory before any long run, the CPU-only test suite in `tests/` locks in joint-index mappings, reward sign conventions, and NaN guards, and the `--hf-jobs` flag moves the whole training loop onto Hugging Face Jobs — with `scripts/hf/uploader.py` streaming checkpoints to a private model repo while the run proceeds.

## How It Works

Everything funnels from a single command into a registry-driven training stack, and the detailed diagram below traces that flow file by file.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/microduck_rl/pollen-robotics-microduck_rl-architecture.svg" alt="Detailed architecture of the pollen-robotics/microduck_rl repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: how the train CLI, the task registry and per-family environment configs, the shared MDP library, the MJCF robot models with the BAM actuator, and the export/deployment scripts interconnect, with remote Hugging Face Jobs and the CPU test suite around them.*

### Understanding the Architecture

**The task registry.** `src/mjlab_microduck/tasks/__init__.py` is the spine of the repo: it registers every task id — the thirteen task families from the README plus their flat/rough variants and fifteen `-Backlash-` twins — through mjlab's `register_mjlab_task`. Each registration binds an environment config factory, an RL config, and a runner class. That runner, `MicroduckOnPolicyRunner`, exists mostly to fix a serialization corner case so run configs can be dumped to YAML cleanly; it subclasses mjlab's velocity runner, which tells you how much of the locomotion machinery is inherited and how little is reinvented.

**The environment recipes.** One config module per task family lives in `src/mjlab_microduck/tasks/`, and `microduck_velocity_env_cfg.py` is the main walking recipe and the shared base. Its top of file is a wall of `ENABLE_*` booleans — CoM randomization, mass/inertia randomization, joint friction randomization, velocity pushes, IMU orientation randomization, encoder bias — each with a comment explaining whether it is on, off, or "was True". The other families build on it rather than starting fresh: the VelStand and StandUp recipes import from the velocity recipe, the BallKick recipe borrows its head-body names, and the RollerSlope recipe composes the rollers recipe with the custom ramp terrain in `src/mjlab_microduck/tasks/slope_terrain.py`.

**The MDP library.** Essentially every custom reward, event, observation, command term, and curriculum lives in `src/mjlab_microduck/tasks/mdp.py`, grouped by task. Two monkey-patches at the top are quietly load-bearing: they sanitize NaN rewards in the reward manager and NaN advantages in PPO's return computation, because a single NaN physics state can otherwise propagate into a crashed optimizer. The file also carries the `passive_*` convention — unactuated joints (wheels, backlash hinges) are excluded from actuator, observation, and reward selections by regex — plus `_servo_joint_ids` helpers so no MDP function ever hardcodes joint indices that would break on interleaved roller or backlash models.

**The robot and its actuators.** `src/mjlab_microduck/robot/microduck_constants.py` defines the robot configs, the HOME frame (a carefully tuned standing pose whose comments record why each angle moved), and which MJCF each task family uses: the stripped `robot_walk.xml` where falling is cheap, the full-collision model where the body may physically lie on the ground, the rollers model with passive wheel hinges, and their backlash twins. The actuator side, `src/mjlab_microduck/actuator/friction_dr_bam.py`, provides both the friction-randomized BAM actuator and `BacklashEncoderBamActuator`, whose PD feedback reads the encoder through the backlash joints — reproducing, at the control-law level, exactly what the real firmware sees.

**From checkpoint to ONNX.** `scripts/export.py` resolves a checkpoint from local logs or a W&B run, rebuilds the play environment from the registry, and calls the runner's ONNX export, which bakes the observation normalizer into the graph — the README is emphatic that you should never hand-convert a checkpoint, or the deployed policy will see unnormalized observations. The exporter also attaches metadata to the file so the runtime knows what it is holding.

**The deployment rehearsal.** `scripts/infer_policy.py` loads the scene XMLs and runs ONNX policies in plain CPU MuJoCo, with terminal keypresses for velocity commands and trick triggers, support for hot-swapping walking, standing, sit-stand, and roulade policies in one session, and `--save-csv`/`--record` hooks that feed the sim2real comparison plots in `scripts/plot_observations_comparison_plotly.py`. Smaller utilities round it out: `scripts/play_latest.py` resolves the most recent W&B run through `scripts/wandb_utils.py` and launches the sim viewer, while `scripts/validate_bam_testbench.py` replays real XL330 testbench recordings against the BAM actuator model using the dedicated testbench MJCF under `src/mjlab_microduck/robot/xl330_test_bench/`.

One end-to-end pass ties it together. You run `uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096`: the CLI resolves the task in the registry, mjlab builds thousands of parallel MuJoCo Warp environments from the velocity recipe, and each iteration collects 24 steps per environment through the shared MDP stack — rewards gated by contact and orientation checks, domain-randomization events re-sampled without accumulating, curricula mutating ranges through the managers. PPO updates the policy, W&B logs every per-term reward so you can verify no penalty has gone positive, and `model_XXXX.pt` checkpoints accumulate in `logs/` (or stream to Hugging Face if the run is remote). When the gait looks right, `scripts/export.py` freezes it into an ONNX file with the normalizer baked in, and `scripts/infer_policy.py` lets you drive the exported policy with your keyboard in CPU MuJoCo — the same 61-dimensional observation contract the real robot's runtime will use.

## Advantages

- **Actuator-first sim2real.** The BAM M6 voltage-level actuator model with per-env friction, voltage, and delay randomization attacks the dominant source of the gap instead of hoping PPO absorbs it.
- **One observation contract, many policies.** The 61-dimensional layout is shared across the whole family, which is what makes runtime hot-swapping of walk/recover/trick policies a design property rather than a retrofit.
- **Backlash A/B testing with zero code changes.** Because backlash twins keep observation and action dimensions identical, comparing a policy trained with and without gear play requires only changing the task id.
- **Massively parallel training on ordinary hardware workflows.** MuJoCo Warp runs thousands of environments on one CUDA GPU, `uv` handles the environment, and `--hf-jobs` relocates the identical run to Hugging Face Jobs with checkpoint streaming included.
- **Engineering guardrails that survive long runs.** NaN sanitization at two layers, a CPU-only regression suite for config invariants, and a documented smoke-test ritual catch the failure modes that otherwise waste GPU-days.
- **A written playbook, not tribal memory.** `CLAUDE.md` turns the project's reward-design rules, curriculum pacing, and sim2real footguns into explicit, greppable guidance.

## Benefits

- **A head start on your own small-robot policies.** The template-and-extend workflow — every task family building on the velocity recipe's domain-randomization and observation stack — means a new behavior starts from a proven base instead of a blank config.
- **Fewer burned training runs.** Sign-convention rules, jackpot prevention, and reward-mass comparisons are all encoded in the docs and in `tests/`, so the classic reward-hacking failures are caught before they cost iterations.
- **Safer hardware bring-up.** The CPU rehearsal script replicates the runtime's policy hot-swapping and command-slot semantics, so integration bugs surface in simulation, not on the robot's servos.
- **Honest physics, measurable trust.** Validating the actuator kernel against real testbench recordings (`scripts/validate_bam_testbench.py`) gives the simulator an evidence trail instead of a vibe.
- **Reproducible experiment history.** W&B integration, resumable checkpoints, and the checkpoint uploader mean any result can be traced back to its exact config and weights.
- **Open licensing.** The code is Apache 2.0, so the environments, actuator modeling, and tooling can be studied, adapted, and reused in other projects.

## Usage

The repository manages its environment with `uv` and expects a CUDA GPU for training (training runs through MuJoCo Warp):

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

Resume from a checkpoint:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

`uv run list-envs` prints the live task registry, and rehearsal of the runtime's hot-swapping looks like this:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

The test suite runs on CPU with no GPU required:

```bash
uv run --with pytest pytest tests/
```

## Conclusion

`pollen-robotics/microduck_rl` is a small repository with an unusually high density of judgment. It takes one robot, one actuator, and one observation contract, and then builds everything else — thirteen task families, backlash twins, remote training, ONNX deployment, deployment rehearsal, and a regression suite — as consequences of those three commitments. For anyone training RL policies for real hardware, especially small robots where actuator physics dominates, it is one of the most instructive source tours available: the code shows you the mechanism, and `CLAUDE.md` shows you the reasoning. Even if your robot has wheels instead of legs and a hundred grams more mass, the recipe generalizes: model the actuator honestly, keep the observation contract sacred, verify physics assumptions before training, and rehearse deployment on the cheapest machine you own.

Links:

- [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) — the repository covered in this post
- [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) — the Microduck project home and onboard runtime
- [mujocolab/mjlab](https://github.com/mujocolab/mjlab) — the training framework (MuJoCo Warp + rsl_rl)
- [Rhoban/bam](https://github.com/Rhoban/bam) — the better-actuator-models project behind the BAM actuator
