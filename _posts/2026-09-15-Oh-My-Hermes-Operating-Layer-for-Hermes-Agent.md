---
layout: post
title: "Oh My Hermes: The Operating Layer That Makes Your AI Coding Agent Professional"
description: "Oh My Hermes (OMH) is a MIT-licensed operating layer for NousResearch's Hermes Agent that routes every request to the right model at the right effort, tunes prompts per model family, runs parallel work lanes with verification gates, and keeps reviewer-gated long-term memory. The project's own measurement: the same coding tasks solved for $0.66 instead of $4.29, in 5 minutes instead of 23. Here is how the routing, fanout, memory, and evidence systems actually work."
date: 2026-09-15
header-img: "img/post-bg.jpg"
permalink: /Oh-My-Hermes-Operating-Layer-for-Hermes-Agent/
featured-img: ai-coding-frameworks/ai-coding-framework
image: https://pyshine.com/assets/img/diagrams/oh-my-hermes/ohmyhermes-architecture.svg
tags:
  - Oh My Hermes
  - Hermes Agent
  - AI Agents
  - Open Source
  - Model Routing
  - Claude Code
  - Codex
  - Developer Tools
author: "PyShine"
---

You have an AI coding agent. It is brilliant, tireless, and slightly chaotic. It bills like a law firm on tasks a junior could do. It says "done!" before anyone has checked. And it forgets everything you taught it the moment the session ends. We covered the agent itself before - [Hermes Agent](https://github.com/NousResearch/hermes-agent), NousResearch's self-improving agent with a quarter of a million GitHub stars. Today's project is what sits on top of it.

[Oh My Hermes](https://github.com/rlaope/oh-my-hermes) (OMH, MIT licensed, around 2,000 stars and on the trending charts right now) is an operating layer for Hermes Agent that fixes exactly those three failures. It routes every request to the right model at the right reasoning effort - the project's own measurement claims the same coding tasks solved for **$0.66 instead of $4.29, in 5 minutes instead of 23**. It never says "done" unless it watched a gate pass. And it keeps a long-term project memory where nothing is remembered silently - every lesson goes through a review card, carries provenance, and expires without re-confirmation.

The philosophy is summed up in two rules that hold everywhere in the codebase: model choice and coding ownership are separate decisions, and nothing prepared is ever reported as executed. Let's unpack how a plugin manages to enforce either.

![Oh My Hermes architecture](/assets/img/diagrams/oh-my-hermes/ohmyhermes-architecture.svg)

### Understanding the Architecture

The diagram above shows one request's journey through OMH, and the layers explain why this is an "operating layer" rather than a fork. OMH never patches Hermes or hides a coding executor behind it - Hermes stays the natural-language surface, and native skills run as capabilities inside OMH's governed path.

**1. Routing first, always.** Every incoming request is scored *before* dispatch, and every signal that moved the score is named. A rename request scores light and lands in the quick lane. "Find every reference to X" trips the exhaustive-search signal and gets a model that will not miss a reference. This single decision - cost should follow difficulty - is where most of the savings come from.

**2. Prompts are tuned per model family, and measured.** Thirteen model families get one calibration block each, every sentence written against a documented trait of that family: Claude is told the checklist is complete, Gemini is warned that a claim without tool output is not evidence, Qwen3-Coder is never allowed to emit thinking tags, DeepSeek is reminded that version and thinking mode are contract fields. And the blocks are held accountable: the project notes that GPT-6 Astra's first calibration draft made it keep working on tasks it would not pass, cost 10% more for the same answers, and was cut on that number.

**3. Executors are interchangeable.** OMH's own skills catalog carries 108 `omh-*` specialist skills - frontend, backend, Rust, native debugging, verification gates, security review, performance budgets - and the right ones load into the run automatically as tool calls; you never invoke an expert, say it in English or Korean, and the router picks the specialists. When a lane should run on a different CLI entirely, the Maestro lane hands the work to Codex or Claude Code with readiness probes and per-run model selection - explicitly opt-in, never the default path.

**4. The HUD keeps it honest.** One row per delegated lane shows model, effort, turns, tokens, and cost provenance - a cost of zero renders only when the host confirmed it; an unpriced call says "unknown", never "$0". Above the prompt sits the phase todo, the run's own checklist, not a summary written afterwards.

## The Router: Where the Money Is

![Oh My Hermes routing](/assets/img/diagrams/oh-my-hermes/ohmyhermes-routing.svg)

### Understanding the Routing

**Categories you own.** Work lands in categories - `ultrabrain`, `deep`, `architect`, `quick`, `writing`, `visual-engineering`, `artistry`, and a few more - and each category is an editable, ordered chain of model-plus-effort entries. The chains live in one JSON file (`~/.omh/routing/model-chains.json`); editing them requires no code, or you can skip the file entirely and use the arrow-key picker (`omh model`, or `/omh-model` inside the Hermes TUI) to walk categories, step head models, and adjust effort.

**Fallback, not downgrade.** When a provider rejects a model, the chain advances to the next entry. But there is a sharper rule underneath: a dispatch that would inherit a provider which cannot serve the chosen model is *refused* rather than silently downgraded. Silent model-swapping is how agents quietly produce worse answers; OMH makes the refusal visible instead.

**Setup respects what you actually have.** During `omh setup`, OMH reads which providers Hermes is already linked to - provider IDs and variable names only, never keys or tokens - and reorders every chain so models your providers can serve lead the list, with the rest marked in the pickers. Your interview answers override detection, and a Claude Code subscription is handled specially: it only seeds the Maestro lane preference, because Hermes itself cannot spend it.

**The measured claim.** On the project's own corpus of coding tasks with the same model (GPT-6 Astra), routing produced the same solve rate - 18 of 30 - at roughly one-seventh the cost and one-quarter the wall clock. The honest footnote: these are the project's own measurements on its own tasks, and its dedicated A/B benchmark harness (`benchmarks/product-ab/v1`) is built and published in the repo, with the project explicitly stating that no independent measured run has been released yet. That kind of claim boundary is rare and worth crediting.

## Parallel Work: Safe Fan-Out, Typed Results

![Oh My Hermes parallel fanout](/assets/img/diagrams/oh-my-hermes/ohmyhermes-fanout.svg)

### Understanding the Fan-Out

The `ulw-work` engine turns an accepted plan into parallel lanes, and the design prevents the classic multi-agent failure modes:

- **Units never share a file.** The plan is split so each unit owns disjoint files - no two lanes editing the same code.
- **Every unit branches from one pinned SHA.** All worktrees start from the same commit, so lanes cannot drift apart mid-run.
- **Results are typed, not vibes.** Each unit returns a sidecar with four states: process exited, schema valid, verification observed, integration ready. Exit code zero with no evidence stays "reported done" until a gate checks it.
- **Receipts are earned.** A verification receipt is reused only when the revision, command, and environment all match - you cannot carry a stale "tests passed" across code changes.
- **Failure is displayed honestly.** A child that did no work shows failed, never a green row; admission control holds units back when the provider is under load.

One more nice touch: batched tool calls run concurrently and get branded on the `[OMH]` line as `parallel shot xN`, so you can see parallelism happening rather than guessing.

## Memory and Evidence: The Character of the Thing

![Oh My Hermes memory and evidence](/assets/img/diagrams/oh-my-hermes/ohmyhermes-memory-evidence.svg)

### Understanding the Memory Model

Most "agent memory" is a silent key-value sink: whatever the model decides to save gets saved, and six months later you are arguing with an agent that half-remembers a decision you reversed. OMH's memory is deliberately slower and more accountable:

1. **A candidate is captured** from the session and placed on a review card.
2. **You decide**: remembered, refused, or deferred - with the reason written down either way.
3. **An approved record carries provenance** and a review-due date; confirming it resets the clock, and silence ages it from active to reference to archive.
4. **The next session gets a recall pack** - memories ranked for the task at hand and cut to a token budget, with conflicts and duplicates resolved.

The store is file-backed and belongs to OMH; Hermes' own memory is never read or patched. And when a turn does carry recalled context, Hermes announces it on every surface it speaks through - "OMH - recalled 2 memories" - so you always know when history is influencing the answer.

### Understanding the Evidence Ladder

The four-state status vocabulary is the quiet heart of the project: **Plan - not run** (nothing has happened yet), **Code - running** (watched now), **Code - reported done** (the executor *said* so, nobody checked), and **Test - verified** (a gate actually passed). Most tools collapse the middle two into "complete". OMH refuses to, and backs it with a completion-integrity gate that refuses stubs and skipped tests as evidence, plus guardrails where you write your own toolcall rules to block off-script actions and set approval tiers for risky ones.

## Getting Started

One line on macOS and Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/rlaope/oh-my-hermes/main/install.sh | sh
```

Or on Windows (PowerShell 5.1+):

```powershell
irm https://raw.githubusercontent.com/rlaope/oh-my-hermes/main/install.ps1 | iex
```

Package-manager people get their own paths: `brew install rlaope/tap/omh`, `bun install -g oh-my-hermes`, or `npm install -g oh-my-hermes`, plus a Hermes-native route (`hermes skills tap add rlaope/oh-my-hermes`). Then:

```bash
omh setup     # guided setup: providers, model chains, entitlement interview
omh doctor    # verify or troubleshoot the installation
omh           # open Hermes wearing the OMH identity
```

Updates are one command (`omh update`) and detect how OMH was installed, upgrading through the owning package manager. The [agent-install protocol](https://github.com/rlaope/oh-my-hermes/blob/main/INSTALL_FOR_AGENTS.md) is worth reading on its own merits: it has the agent pin the install to a full commit SHA before reading instructions - a supply-chain-aware pattern more projects should copy.

## Why This Matters

AI coding agents crossed the line from demo to daily driver a while ago; what has not caught up is management. We accept agents that bill a premium for trivial work, declare victory before testing, and forget yesterday's decisions. Oh My Hermes treats those as engineering problems with engineering solutions: difficulty-based routing with owned fallback chains, typed evidence instead of claimed completion, and memory that earns its place through review.

The project is also a study in honest boundaries - publishing its own cost measurements alongside the explicit statement that its independent A/B benchmark has not run yet, and pinning its agent-install instructions to commit SHAs. If you run Hermes Agent (or delegate to Claude Code and Codex through it), OMH is the upgrade that makes the whole operation feel staffed rather than improvised. And if you are building your own agent stack, the [architecture documentation](https://github.com/rlaope/oh-my-hermes/blob/main/docs/ARCHITECTURE.md) doubles as a design manual for evidence-gated orchestration.

## Related Posts

- [Hermes Agent: The Self-Improving AI Agent](https://pyshine.com/Hermes-Agent-Self-Improving-AI-Agent/)
- [NVIDIA SkillSpector: AI Agent Skills Security Scanner](https://pyshine.com/NVIDIA-SkillSpector-AI-Agent-Skills-Security-Scanner/)
- [DSPy Agent Skills: Production-Grade DSPy 3.2 Skills for Coding Agents](https://pyshine.com/DSPy-Agent-Skills-Production-Grade-DSPy-3.2-Skills-for-Coding-Agents/)
