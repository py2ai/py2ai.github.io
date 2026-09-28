---
layout: post
title: "Paperclip: The Company Layer for Your AI Agents - Inside paperclipai/paperclip"
description: "Paperclip is an open-source control plane that turns a pile of AI agents into a managed organization. A source-level tour of paperclipai/paperclip: the Node.js server core, DB-backed heartbeat wake queue, atomic task checkout, budget hard-stops, approval gates, adapters for Claude Code and Codex, company portability, and the governed MCP gateway."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /Paperclip-The-Company-Layer-for-Your-AI-Agents/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/paperclip/paperclipai-paperclip-architecture.svg
tags:
  - AI Agents
  - Node.js
  - Orchestration
  - Open Source
categories: [AI, Open Source]
keywords: "Paperclip, paperclipai, AI agent orchestration, agent org chart, OpenClaw, Claude Code, Codex, agent budgets, heartbeat execution, agent governance, AI task manager, multi-agent control plane, MCP gateway, agent adapters, autonomous business"
author: "PyShine"
---

The scenario the repository opens with is uncomfortably familiar: twenty Claude Code terminals open, each mid-task, none of them aware of the others. One of them is quietly burning tokens in a loop. A reboot erases the lot. You are not running agents at that point - you are babysitting tabs. [Paperclip](https://github.com/paperclipai/paperclip) from Paperclip Labs is a bet that the missing piece is not a better agent but an organization around them: goals, org charts, budgets, approval gates, and a heartbeat scheduler that keeps everyone working while you sleep.

The project's own framing is the sharpest one-line pitch you will read this month: *if OpenClaw is an employee, Paperclip is the company.* It is a Node.js server and React dashboard where you define a business goal ("Build the #1 AI note-taking app to $1M MRR"), hire a team of agents - CEO, CTO, engineers, marketers, any bot from any provider - approve the strategy, set budgets, and then supervise the work from one place, including from your phone.

What makes the source worth a tour is that it treats agent orchestration as an operations problem with real engineering behind it: atomic task checkout so no two agents grab the same job, DB-backed wakeup queues with coalescing and orphan recovery, budget hard-stops that pause agents mid-flight, revisioned governance with rollback, and full company export/import. This post walks through how those pieces actually fit together in the code.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/paperclip/paperclipai-paperclip-overview-architecture.svg" alt="Architecture overview of the paperclipai/paperclip repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview: the React dashboard and CLI drive a Node.js control plane; a service layer persists to PostgreSQL; a heartbeat wake queue wakes agent adapters, and a governed MCP server exposes tools.*

Reading the overview from left to right: two front doors - the React dashboard (`ui/`) for humans and the `paperclipai` CLI for setup and onboarding - meet the same REST surface. Inside the server, a large service layer carries the domain logic and persists everything to PostgreSQL, which ships embedded for zero-setup first runs. The distinctive piece is the heartbeat pipeline: a database-backed wake queue that fires agents on schedule and on events, feeding an adapter registry that speaks to Claude Code, Codex, Cursor, OpenClaw-style webhook bots, and more - the README's rule is "if it can receive a heartbeat, it's hired." Keep this shape in mind; the detailed diagram is where the orchestration subtleties live.

## Why You Need This

Individual agents got good fast. Coordination did not. The repository's problem table is worth reading because every row is a war story: context scattered across places so your bot never knows what you are actually doing; folders of agent configs with no task management; runaway loops maxing your quota before you notice; recurring work (support, reports, social) that depends on you remembering to kick it off. These are not model problems. They are management problems - the kind companies solved a century ago with org charts, budgets, and approval workflows - applied to software that never sleeps.

Paperclip's answer is structural rather than clever. Tasks trace up through projects to company goals, so an agent always sees the *why* along with the *what*. Agents have roles, titles, reporting lines, and permissions - governance in the literal sense of who can do what. Every conversation and decision lands in a ticket with full tool-call tracing and an immutable audit log. Money is a first-class constraint: monthly budgets per agent, warning thresholds, hard stops that pause the agent and cancel queued work when the limit hits.

The second reason is deployment honesty. Paperclip is self-hosted, open source under MIT, needs no account, and runs from a single Node.js process with an embedded PostgreSQL on first start - then lets you point at your own Postgres for production. One deployment can run multiple companies with complete data isolation, which is the difference between a demo and a control plane. And the design keeps its boundaries explicitly: it is not a chatbot, not an agent framework, not a workflow builder. It uses the agents you already have and manages the organization they work in.

Finally, the coordination details the README calls out are the ones every homegrown multi-agent setup gets wrong: atomic checkout with execution locks so no double-work, persistent agent state so heartbeats resume context instead of restarting, and goal-aware execution so delegation up and down the org chart carries intent, not just titles.

## How It Works

The diagram below maps the real subsystems of the repository, from the interfaces through the server core and heartbeat pipeline down to the agent runtimes and data layer.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/paperclip/paperclipai-paperclip-architecture.svg" alt="Detailed architecture of Paperclip: interfaces, server core with identity and secrets, heartbeat pipeline with wake queue and watchdog, budgets and approvals, agent adapters, and the database and extension layer" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the dashboard and CLI hit REST routes; the server core hosts identity, services, secrets and storage; a wake queue fires run dispatch through agent adapters under a watchdog; budgets throttle, approvals gate, and everything persists to PostgreSQL.*

### Understanding the Architecture

**The server core.** Everything lives under `server/src/`, a TypeScript Node.js application whose `app.ts` boots the HTTP API, mounts `routes/`, and hosts the `services/` directory - hundreds of focused modules, one per concept (agents, issues, goals, budgets, routines, approvals, activity). Identity is handled in `auth/`: two deployment modes (trusted local loopback for fast starts, or authenticated with board users), agent API keys, and short-lived run JWTs so every mutating request traces to an actor. `secrets/` keeps sensitive values encrypted and out of prompts unless a scoped run explicitly needs them, and `storage/` handles attachments and work products in provider-backed object storage. For first-run ergonomics, `embedded-postgres-supervisor.ts` boots and owns a managed PostgreSQL so `pnpm dev` needs no database setup.

**The heartbeat pipeline.** Agents do not run continuously by default; they wake. `modules/wake-queue` is a DB-backed wakeup queue with coalescing, so a burst of triggers produces one sensible wake instead of five. When a wake fires, `modules/run-dispatch` does the real work: budget checks, workspace resolution, secret injection, skill loading, and finally adapter invocation. Runs produce structured logs, cost events, session state, and audit trails - and because agents crash and hosts reboot, `modules/active-run-watchdog` detects orphaned runs and recovers them automatically. The result is the property the README promises: agents resume the same task context across heartbeats instead of starting from scratch.

**Work, money, and governance.** The work system is deliberately boring in the good way. Issues carry company/project/goal/parent links with blocker dependencies, comments, documents, and labels; checkout is atomic with execution locks, so two agents can never hold the same task. `services/budgets.ts` enforces token and cost tracking by company, agent, project, goal, issue, provider, and model - with warning thresholds and hard stops that pause agents and cancel queued work on overspend. `services/routines.ts` turns recurring jobs into tracked issues on cron, webhook, or API triggers, so support rotas and weekly reports do not depend on memory. On the governance side, `services/approvals.ts` and issue approvals implement board review workflows, revisioned config with rollback, and pause/resume/terminate for any agent - nothing ships without sign-off when a gate is configured.

**Agent runtimes.** `packages/adapters/` is the "bring your own agent" layer: local terminal harnesses for Claude Code, Codex, Cursor, Gemini, Grok, Kimi and opencode, a pi adapter, plus gateway adapters for OpenClaw-style webhook bots and the Hermes gateway. The adapter contract is deliberately thin - receive a heartbeat, do work, report back - which is why the roster grows so fast. `packages/paperclip-runner` hosts sandboxed execution for cloud and isolated runs, and `packages/skills-catalog` feeds runtime skill injection so agents learn Paperclip workflows and project context without retraining.

**Data and extensions.** `packages/db` holds the schema and migrations for PostgreSQL, with `packages/shared` carrying the cross-package contracts. Extensibility comes from two directions: `packages/plugins` runs instance-wide plugins as out-of-process workers with capability-gated host services, and `packages/mcp-server` exposes a governed tool gateway so agents get MCP tools under the same permission and budget regime as everything else. Company portability ties the bow - `services/company-export-readme.ts` and friends implement export/import of entire organizations with secret scrubbing and collision handling, so a deployment is a portfolio, not a trap.

**A heartbeat in flight.** Follow one task end to end. A routine fires at 9:00, or a human assigns an issue and the assignment itself creates a wake. The wake queue coalesces and dequeues; run dispatch checks the agent's remaining budget, resolves the project workspace, injects scoped secrets, and loads the org's skills. The adapter - say `claude-local` - receives the heartbeat, works inside an execution workspace (git worktree, operator branch), and reports diffs, screenshots, or tests back. The watchdog supervises: if the run dies, it is recovered; if it finishes, the work product lands in the ticket, the cost event lands in the ledger, and the activity log records who did what and why. If a review gate is configured, the board sees the result before anything ships.

## Advantages

- **Organization as the primitive.** Goals, roles, reporting lines and permissions are data-model citizens - coordination comes from structure, not prompt hope.
- **Atomic execution.** Task checkout and budget enforcement are atomic, eliminating double-work and runaway spend at the concurrency boundary where most agent setups fall apart.
- **Heartbeats with recovery.** The DB-backed wake queue, coalescing, and orphan-recovery watchdog give scheduled autonomy the reliability of a real job system.
- **Governance with rollback.** Approval gates, revisioned configuration, and immutable audit logs make agent operations reviewable - and reversible.
- **Provider-neutral.** Adapters for a dozen harnesses plus webhook bots mean the org outlives any single model or tool vendor.
- **Portable and isolated.** Full org export/import with secret scrubbing, and strict company scoping so one deployment safely runs many businesses.

## Benefits

- **You stop babysitting terminals.** Sessions persist across reboots, conversations are threaded into tickets, and the dashboard shows who is doing what - from a laptop or a phone.
- **Budgets become enforceable, not aspirational.** Cost tracking to the issue and model level, with hard stops that fire automatically when agents overspend.
- **Recurring work runs itself.** Routines create tracked issues and wake the right agent on schedule, with catch-up policies for missed runs.
- **Delegation carries context.** Goal ancestry means every task explains its why, so agents act consistently even when nobody re-briefs them.
- **Switching costs stay low.** MIT-licensed, self-hosted, embedded Postgres for day one, external Postgres for year five, and org export if you ever leave.

## Usage

The recommended install is the checksummed installer (it also sets up Node.js 24.11+ if needed):

```bash
curl -fsSLO https://paperclip.ing/install.sh
curl -fsSLO https://paperclip.ing/install.sh.sha256
sha256sum -c install.sh.sha256
bash install.sh
```

Non-interactive install, or a throwaway test drive with no permanent install:

```bash
paperclipai onboard --yes

# isolated test instance with a pre-seeded CEO agent:
ANTHROPIC_API_KEY=... npx paperclipai test-drive
OPENAI_API_KEY=... npx paperclipai test-drive --harness codex
```

From source (Node.js 24.11+, pnpm 9.15+):

```bash
git clone https://github.com/paperclipai/paperclip.git
cd paperclip
pnpm install
pnpm dev
```

This starts the API at `http://localhost:3100` with an embedded PostgreSQL created automatically. Day-to-day development commands cover the full loop:

```bash
pnpm test        # Vitest unit suite (Playwright e2e runs separately)
pnpm typecheck   # type checking
pnpm db:migrate  # apply database migrations
pnpm dev:mobile  # phone-friendly UI on :3101
```

One operational note from the README: anonymous usage telemetry is enabled by default. It collects no prompts or content, and disabling it is a one-liner - set `PAPERCLIP_TELEMETRY_DISABLED=1` or `DO_NOT_TRACK=1`, or set `telemetry.enabled: false` in the config.

## Conclusion

Paperclip's thesis is that the hard part of agentic work is not the agents - it is the company around them - and the source executes that thesis with unusual discipline: a control plane where checkout is atomic, budgets are hard stops, heartbeats survive crashes, governance is revisioned, and entire organizations are exportable artifacts. It is also an honest repository: it tells you what it is not, documents its telemetry and how to turn it off, and keeps its adapter layer thin enough that "any agent" is a design constraint rather than a slogan. If your multi-agent setup has outgrown a folder of scripts and a row of terminal tabs, run the test-drive, hire two agents, give them a goal and a budget, and watch the wake queue do its job. The company metaphor is not marketing - it is the architecture.

**Links:**

- Repository: [https://github.com/paperclipai/paperclip](https://github.com/paperclipai/paperclip)
- Documentation: [https://docs.paperclip.ing](https://docs.paperclip.ing)
- Community plugins: [https://github.com/gsxdsm/awesome-paperclip](https://github.com/gsxdsm/awesome-paperclip)
