---
layout: post
title: "Easel: An Open-Source AI Agent for Social Media Creation - Inside ZJU-REAL/Easel"
description: "Easel is an open-source AI agent for social media from ZJU-REAL that ties trend discovery, content planning, multimedia production, multi-platform publishing, and performance attribution into one continuous loop. This source tour walks through its OpenClaw agent runtime, 113-skill library, Playwright platform adapters for seven Chinese platforms, and the profile memory that carries lessons back into the next creation cycle."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Easel-Open-Source-AI-Agent-For-Social-Media-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/easel/zju-real-easel-architecture.svg
tags:
  - AI Agent
  - Social Media
  - Python
  - Open Source
categories: [AI, Open Source]
keywords: "Easel, ZJU-REAL, AI agent, social media automation, OpenClaw, content workflow, Xiaohongshu publishing, Douyin, Bilibili, Zhihu, WeChat Official Account, Playwright, FastAPI, content calendar, Apache 2.0"
author: "PyShine"
---

Most AI writing tools stop at the draft. They hand you text, maybe an image prompt, and then leave the hard parts - formatting for each platform, uploading, checking whether anything actually worked - to you. Easel, from the ZJU-REAL research organization (the README carries Zhejiang University and Peking University lab branding), takes a different position: an agent should carry a piece of content all the way from a trending topic to a published post, and then learn from how that post performed. The result is a Python workspace where discovery, planning, production, publishing, and attribution are not five separate products but five layers of one continuous job.

Easel describes itself as an open-source content workspace for social media creators. It connects an agent runtime, per-account profiles, a large library of executable skills, and real media tooling so that the agent does not just explain what to do - it produces the files, adapts them per platform, publishes them through logged-in accounts, and records what happened. The project is Apache 2.0 licensed, requires Python 3.10 or newer, and ships with a FastAPI web workspace, a CLI, and a browser-automation layer built on Playwright.

What makes the repository worth a source tour is that very little of it is vaporware. The skills are directories containing SKILL.md playbooks plus runnable Python scripts; the seven supported platforms each map to a concrete automation script; publishing success is not trusted until a read-back from the platform confirms it. Reading the code gives you an unusually honest picture of what "AI runs my social media" actually requires in practice - guards, gates, queues, and verification at every step.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/easel/zju-real-easel-overview-architecture.svg" alt="Architecture overview of the ZJU-REAL/Easel repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Easel architecture: entry points feed a single OpenClaw agent, which routes work through a large skill library and shared tool scripts, stages deliverables in an outputs pipeline, and publishes through platform adapters - with account profiles closing the learning loop.*

Reading the overview from left to right: users reach Easel through the CLI (`easel/cli.py`), the FastAPI backend (`web/app.py`), or the React web workspace (`web/frontend/src/App.tsx`), and all three converge on the same OpenClaw gateway configured in `openclaw/openclaw.json5`. The gateway loads its operating instructions from `openclaw/workspace/AGENTS.md` and routes each task into the skill library under `skills/openclaw`, whose deterministic tooling lives in `skills/shared/scripts`. Produced work lands in `outputs/` under manifest contracts, scheduled posts flow through the publish queue into the platform publishers, and whatever the attribution layer learns is written back into `profiles/` so the next session starts smarter.

## Why You Need This

If you run one or more social media accounts, you already know the workflow is fragmented. Trend research happens in one app, drafting in another, scheduling in a third, and analytics in a dashboard none of them talk to. Easel's core bet is that a single agent with a persistent memory of your account can collapse that fragmentation: the same context that picks a topic also writes the script, renders the cards, and fills in the publish form.

The second problem is format fragmentation. The same idea needs to become a Xiaohongshu card note, a Douyin short video, a Zhihu long-form article, or a short WeChat post - each with different length limits, aspect ratios, and title conventions. Easel treats this as a first-class concern: the cross-platform publishing skill regenerates each variant from one master asset while respecting per-platform constraints, and the publisher scripts enforce hard limits like Xiaohongshu's twenty-full-width-character title cap before anything is submitted.

The third problem is that agents forget. Most LLM sessions start from zero, so your positioning, audience, boundaries, and hard-won lessons about what works have to be restated every time. Easel solves this with account profiles - a directory per persona containing `identity.md`, `style.md`, `audience.md`, `platforms.md`, `preferences.md`, and `memory.md` - that are injected into every request. After publishing, verified lessons flow back into that profile, so the loop tightens over time instead of resetting.

Finally, there is the trust problem. Autonomous posting to real accounts is risky, and the project is refreshingly blunt about it: the README warns that automated Xiaohongshu publishing can trigger platform risk controls, and the code answers with dry-run preflight checks, a deterministic secret scanner, a persona-consistency gate, and a read-back reconciliation that refuses to report success without platform-side evidence. If you have been hesitant to let an agent touch production accounts, this is the part of the codebase to study.

## How It Works

Easel is a thin Python integration layer around an OpenClaw agent runtime, and everything interesting happens in how that agent is instructed, constrained, and equipped.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/easel/zju-real-easel-architecture.svg" alt="Detailed architecture of the ZJU-REAL/Easel repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the Easel source: the five workflow layers, the shared tool scripts behind them, the platform adapters with their read-back verification, and the attribution path back into account profiles.*

### Understanding the Architecture

**The entry funnel.** Three doors lead into the same room. The CLI (`easel/cli.py`) offers `chat`, `skill`, `web`, `gateway`, `doctor`, and `ping` subcommands; `easel/commands/skill.py` routes single-skill requests to the agent rather than executing anything locally; and the web backend (`web/app.py`, a single large FastAPI module with SSE streaming) serves the React workspace whose API client lives in `web/frontend/src/lib/api.ts`. Because OpenClaw derives its gateway port from a hash of the profile name, `easel/gateway_endpoint.py` exists as the single source of truth for resolving that port - the docstring walks through the FNV-1a hash scheme that lands the `easel` profile on its dedicated port. The backend keeps the agent resident in the gateway process over an HTTP transport specifically to avoid paying a client cold start on every conversational turn.

**The prompt stack.** The agent's behavior is assembled from layers documented in `docs/prompt-stack.md`: `openclaw/workspace/SOUL.md` defines personality and a category-level capability overview, `openclaw/workspace/AGENTS.md` carries the real business logic (five-layer routing rules, when to plan before acting, production self-checks, publication safety), a generated `CONTEXT.md` pins the project root path, and each triggered skill loads its own `SKILL.md` plus reference files on demand. Notably, personas are not stored in a global user file - `easel/persona.py` inlines the profile into each message as a prefix, which the docs explain avoids concurrency races between parallel sessions, and it appends a per-turn reminder to counteract the well-known drift where a long conversation stops following its system prompt.

**The skill library and its tooling.** Under `skills/openclaw` sit 113 skill directories (the project's own badge and `docs/skill-function-mapping.md` put the number at 113), each combining an agent-readable playbook with optional reference material and scripts. Discovery skills like `skill-trending-topics` fetch real-time hot lists from Weibo, Douyin, Zhihu, Toutiao, and Bilibili through documented public JSON APIs - with explicit instructions not to scrape the platforms directly - and filter them against the account's niche. Planning skills score topics and maintain the calendar; production skills generate copy, cards, posters, and video. The heavy lifting happens in `skills/shared/scripts`, roughly forty deterministic Python modules covering media generation, `calendar_ops.py` for calendar reads and writes, and `model_registry.py` for checking which paid media models are configured before spending money.

**The production discipline.** Anything the agent creates goes into `outputs/<topic>/`, with final deliverables at the project root, intermediates in an `assets/` subdirectory, and system state in underscore-prefixed directories. `output_paths.py` validates the layout so nothing scatters into the outputs root. Cross-layer handoffs are formalized by `skills/shared/scripts/manifest.py`, which records each step - layer, skill, status, outputs, a one-line conclusion - into a hidden `.easel.json` per project, so a downstream layer reads its upstream's conclusions with a `latest` command instead of re-deriving them, and a failed step leaves a breakpoint the workflow can resume from.

**The publishing gauntlet.** Before anything goes public, content passes two scripted gates: `persona_gate.py` scores the draft against the account persona (eighty or above passes; below that it warns but, by design, never blocks), and `content_guard.py` deterministically scans for secrets and internal details - API keys, internal hostnames, proxy addresses, environment variable names - and exits with a failure code on any hit. Then the platform adapters take over: `xhs_publish.py` drives Xiaohongshu's creator studio through Playwright with persistent login state and anti-detection measures, `douyin_publish.py` does the same for Douyin, and `web_publisher.py` is a config-driven engine for platforms that only expose web forms, such as Kuaishou, WeChat Channels, and Zhihu, where each platform is a step configuration shared across one Playwright framework. Crucially, none of these trust the click of a submit button: `platform_readback.py` returns to the creator center, reads the published-works list, and only reports `verified` when the new post matches by title and time window - otherwise it returns one of three honest failure states.

**The scheduling and learning loop.** Batch posting runs through `skills/openclaw/skill-publish-scheduler/scripts/publish_queue.py`, a standard-library-only queue that imports a schedule of content-by-platform-by-time rows, computes which items are due, and delegates each to the right platform publisher before marking it done; recurring triggers are left to the gateway's cron rather than a resident daemon. After publication, the attribution side collects the evidence - `account_stats.py` for account and content data, `xhs_comment.py` for comment threads, the postmortem skill for structured reviews - and distills reusable lessons back into `profiles/<name>/memory.md`, but only with the user's explicit consent and only for genuinely reusable preferences, boundaries, and validated patterns.

Put together, an end-to-end run looks like this: the trending-topics skill surfaces a hot story relevant to the account's niche; the topic evaluator scores it and the calendar script checks the posting context; a production skill drafts copy and renders cards or video into `outputs/<topic>/`; the content guard and persona gate screen the draft; the publish queue or a direct dispatch hands it to the right Playwright adapter; the read-back module confirms the post actually exists on the platform; and the postmortem path folds the performance data back into the profile's memory file for the next cycle.

## Advantages

- **One agent across the whole loop.** Discovery, planning, production, publishing, and attribution share one runtime, one context, and one set of instructions in `openclaw/workspace/AGENTS.md`, so nothing is lost between tools.
- **Skills execute instead of advising.** The skill library pairs every playbook with runnable scripts, and finished work lands as real files in `outputs/` rather than as chat messages you have to copy out.
- **Publishing is verified, not assumed.** The read-back reconciliation in `skills/shared/scripts/platform_readback.py` distinguishes verified, unverified, login-required, and read-back-error outcomes, so a failed post is never reported as success.
- **Profiles are structured, not improvised.** Six fixed markdown files per account, injected per request, keep parallel sessions isolated and make long-term memory an inspectable file rather than hidden state.
- **Safety gates are code, not prompts.** The secret scanner and persona gate are deterministic scripts with explicit exit codes, which makes the pre-publish pipeline testable and auditable.
- **Honest failure handling.** The manifest records failed steps with breakpoints, the queue marks completed items to prevent double posting, and the code routinely documents its own limitations and platform risks in comments and skill docs.

## Benefits

- **Time back from manual formatting.** One master asset becomes platform-correct variants - card notes, short videos, long articles - with per-platform limits enforced by the publisher scripts themselves.
- **Compounding account knowledge.** Lessons validated by real performance accumulate in `memory.md`, so topic selection and style guidance improve from evidence rather than guesswork.
- **A calmer relationship with platform risk.** Dry-run plan modes, headless-after-verification workflows, and human confirmation before publishing give you checkpoints where automation would otherwise go straight live.
- **Hackable by design.** Platform selectors are centralized in single dictionaries, skill behavior is plain markdown, and every layer boundary is documented in `docs/SKILL-SPEC.md` and `docs/prompt-stack.md`, so adapting a flow is a file edit, not a fork.
- **A reference implementation for agent builders.** The gateway endpoint resolution, per-request persona injection, anti-drift reminders, and step manifests are all independently useful patterns for anyone wiring an LLM agent to real-world tools.
- **Runnable self-diagnostics.** `easel doctor` checks the environment and `easel ping` verifies gateway connectivity, which shortens the path from clone to working install.

## Usage

The project targets Linux or macOS with a Windows PowerShell installer (`setup.ps1`) also present; the guided installer checks Node.js, FFmpeg, and Playwright/Chromium along the way. From the repository's quick start:

```bash
git clone git@github.com:ZJU-REAL/Easel.git
cd Easel
bash setup.sh
source .venv/bin/activate    # easel is installed in .venv; activate it first (Windows: .venv\Scripts\activate)
easel web
# Or: easel chat
```

Under the hood the installer covers the Python-side dependencies, which you can also run directly:

```bash
pip install -e .
python -m playwright install chromium
```

The minimum configuration is a usable LLM in the project-root `.env` file:

```bash
ANTHROPIC_API_KEY=your_api_key
CLAUDE_MODEL=anthropic/claude-sonnet-4-6
```

Then open `http://localhost:7860` for the web workspace - the recommended entry point, since it adds conversations, assets, profiles, the content library, and publishing management on top of the CLI. Two diagnostics are worth running first:

```bash
easel doctor    # check the environment
easel ping      # verify gateway and agent connectivity
```

Single skills can also be invoked directly from the CLI, with an optional account profile:

```bash
easel skill trending-topics -i "今天有什么值得追的热点" -p 户外达人
```

## Conclusion

Easel is one of the more complete open-source attempts to answer a question a lot of people are asking right now: what does it actually take for an AI agent to run a real social media account end to end? Its answer is not a cleverer prompt but an honest system - a layered prompt stack that resists drift, a hundred-plus skills backed by deterministic scripts, publisher adapters that verify their own success, and a profile memory that turns performance data into durable account knowledge. The seven-platform publishing layer is necessarily China-centric, and the project is candid that web-automation publishing carries real platform risk, but the engineering patterns here travel well beyond any one platform or market. If you are building agents that must act in the world rather than merely talk about it, reading this source is time well spent.

Links:

- GitHub repository: [https://github.com/ZJU-REAL/Easel](https://github.com/ZJU-REAL/Easel)
- Project page: [https://zju-real.github.io/Easel/](https://zju-real.github.io/Easel/)
- Capability map (113 skills): [docs/skill-function-mapping.md](https://github.com/ZJU-REAL/Easel/blob/main/docs/skill-function-mapping.md)
- English README: [README_EN.md](https://github.com/ZJU-REAL/Easel/blob/main/README_EN.md)
