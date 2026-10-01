---
layout: post
title: "DSCODE: A Terminal Coding Agent That Works In A Team - Inside qiz029/dscode"
description: "DSCODE is an MIT-licensed macOS terminal coding agent built on DeepSeek Harness, where sessions see each other, hand over tasks, delegate on a kanban board, and get reviewed by an independent read-only model. We tour the source."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /DSCODE-A-Terminal-Coding-Agent-That-Works-In-A-Team/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/dscode/qiz029-dscode-architecture.svg
tags:
  - AI Agents
  - Coding Agents
  - Developer Tools
  - Open Source
categories: [AI, Open Source]
keywords: "DSCODE, terminal coding agent, DeepSeek Harness, agent delegation, session bridge, independent AI review, kanban agent board, agent-to-agent, CI agent runs, open source"
author: "PyShine"
---

Coding agents have quietly converged on a lonely design: one session, one context window, one conversation, alone with your repository. Everything else on your machine is invisible to it. The result is familiar — a second terminal session duplicating work a first one already finished, a review that never happens because the author and the reviewer are the same model, and a five-minute question that pollutes the main thread where you were mid-refactor.

DSCODE, by Todd Zheng, is a terminal coding agent for macOS built on DeepSeek Harness that starts from the opposite assumption: sessions on your machine are visible to each other. Start a task in the TUI, then add requirements or read progress from another terminal. One session hands a task to another, which can read the sender's transcript and answer. A `/btw` side question runs in its own read-only child session so the main conversation stays clean. After a code change passes its checks, an independent, read-only reviewer model examines the diff before the turn is allowed to end.

The source rewards the tour because the interesting parts are not the model calls — they are the plumbing that makes multi-agent behavior safe: a persisted mailbox with retry de-duplication and a finite budget, child agents editing in isolated Git worktrees, a permission preset reviewed by a separate model from your instructions rather than a rule table, and a non-interactive `exec` mode that turns all of it into something CI can call.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/dscode/qiz029-dscode-overview-architecture.svg" alt="Architecture overview of the qiz029/dscode repository" style="max-width:100%;height:auto;" />
</div>

*Architecture overview of the qiz029/dscode repository: a launcher opens the TUI and CLI on one shared session runtime, feature plugins compose the harness, and an independent review plugin guards every change.*

Reading the overview from left to right: the launcher starts sessions that the TUI, the CLI, and scripted `exec` runs all share. The session runtime plans delegated work on a board, composes its capabilities from the plugin set pinned by a versioned preset, and submits finished diffs to the independent reviewer, which decides approvals through its policy. A build-and-check layer runs the test suite that verifies the runtime.

## Why You Need This

The first reason is context hygiene. The `/btw` command answers a side question in a read-only child session and shows the answer in a panel, and the exchange never enters the main conversation. Anyone who has watched an important refactor thread get buried under three curiosity questions knows the value: your main context stays about the task you chose, not every tangent you met on the way.

The second reason is delegation with receipts. `/delegate <task>` turns the main agent into a coordinator: it plans the task on a board with priorities and dependencies, runs ready parts as child agents in isolated Git worktrees created from a clean `HEAD`, verifies each result, and merges the accepted ones in dependency order, staged and uncommitted. `/delegate-dashboard` shows the board as a color-coded kanban. Parents run up to five children at once below Ultra effort and twenty at Ultra, and each child gets its own effort level chosen by the parent.

The third reason is honest review. Most agents review their own work, which is structurally the same as not reviewing. In DSCODE, once a change passes its relevant checks, the diff — or, outside a repository, the changes since a snapshot taken when the task began — goes to a separate read-only model, and concrete findings get fixed before the turn ends. `/review` runs the same reviewer by hand, scoped to a staging area, a base branch, a commit, or a path. `/review-usage` reports what the reviewer allowed and what it cost.

The fourth reason is operability. `dscode exec "prompt"` runs a full turn in scripts and CI: the reply streams to stdout, tool activity and the session id go to stderr, and the exit code reflects the turn, with `--json` and `--resume` supported. Trigger definitions add durable cron and delay jobs and can start sessions unattended, installed as a launchd agent. An agent that only lives in a TUI is a toy; an agent with a scripted contract is infrastructure.

## How It Works

DSCODE is architected as one shared session runtime with pluggable capabilities, a launcher that keeps the installation pinned, and review and memory systems that operate beside the main loop.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/dscode/qiz029-dscode-architecture.svg" alt="Detailed architecture of the qiz029/dscode repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the qiz029/dscode repository, from the launcher and TUI through the core runtime, feature plugins, review guardrails, and the test and evaluation layers.*

### Understanding the Architecture

**One runtime, many doors.** The launcher — `bin/dscode.mjs` resolving into `packages/launcher/` — starts sessions whose runtime lives in `plugins/dscode/index.mjs`. The TUI (`packages/tui/`), the command line, and `plugins/exec/cli.mjs` for scripted runs all enter that same runtime and context. A persistent shell keeps its working directory, environment and background jobs across the session, so a task you start interactively is continuous with what scripts and other terminals see. Readers never take the session write lock.

**Capabilities are plugins over a pinned preset.** The feature set composes from `plugins/` — account, credentials, openrouter, email, i18n, memory, auto-review, exec, and more — with versions fixed by `config/preset.json`, which the first launch installs from the DSH Plugin Hub. `scripts/harness.mjs` and `scripts/composition.mjs` are the tooling that generates and verifies that composition, so the installed bundle is reproducible rather than assembled by hand. No MCP server is mounted by default; you add your own in `config/mcp.local.yml`.

**Agent-to-agent is a designed protocol.** `docs/session-communication.md` specifies how the agent finds and messages other sessions through `list_sessions`, `read_session`, `send_session` and `reply_session`, choosing between `queue`, `steer` and `defer` delivery. A persisted mailbox with retry de-duplication and a finite budget bounds message loss, double processing and wake-up loops — the failure modes that turn naive multi-agent systems into echo chambers. Session cards advertise each session's project, workspace, and the topics of its last five user requests, so an agent can pick a collaborator without reading its transcript.

**Memory is a background pipeline.** The memory plugin — `plugins/memory/` with `index.mjs`, `pipeline.mjs` and `store.mjs` — extracts reusable experience after sessions and retrieves it later together with its workspace and source messages. Model usage is tracked separately, and memory can be disabled per session or globally, so the feature is a dial rather than a tax.

**Guardrails are structural, not advisory.** The auto-review plugin — `index.mjs`, `policy.mjs`, `audit.mjs` under `plugins/auto-review/` — implements the independent reviewer: a read-only model that examines diffs and decides approvals from your instruction. The default sandbox is `workspace-write`; Ultra grants no extra permissions; Computer Use keeps its separate human authorization. Credentials for the DeepSeek route are stored locally in `~/.dscode/credentials.yaml` with `0600` permissions and never sent to the agent.

**The QA layer treats agent behavior as testable.** `tests/` covers everything from `auto-review.test.mjs` and `apply-patch.test.mjs` to `communication.test.mjs` and `btw.test.mjs`, `scripts/checks.mjs` runs the suite, `scripts/e2e.mjs` exercises full sessions end to end, and `eval/` holds analysis and experiments. The Makefile ties the targets together.

The end-to-end flow: install the pinned harness, start `dscode` in a project, work in the TUI while other terminals steer or watch the session, delegate parallel subtasks to worktree-isolated children, and end each turn only after an independent read-only model has reviewed the diff and its findings are fixed.

## Advantages

- **Sessions that cooperate.** Hand tasks between sessions, steer live work from another terminal, and subscribe to progress — no copy-paste bridges.
- **Reviews by a different model.** The independent read-only reviewer breaks the author-reviews-own-work pattern that undermines most agent output.
- **Parallel children without collisions.** Git-worktree isolation and dependency-ordered merges make concurrent agent edits mergeable instead of chaotic.
- **Clean main context.** Side questions run in disposable read-only children, keeping the primary thread on the primary task.
- **A real CI contract.** `dscode exec` with exit codes, stdout streaming and JSON output makes agent turns scriptable and observable.
- **Reproducible installation.** The pinned, versioned preset means every machine gets the same harness, verified together.

## Benefits

- **Parallelism you can trust.** The delegate board, worktree isolation and verification ordering turn "run five agents at once" from a leap of faith into a workflow.
- **Context discipline compounds.** A main thread that stays on-task produces better code than one diluted by tangents.
- **Governance built in.** Approval decisions come from your instruction through an independent model, with an audit trail and usage accounting.
- **Operable unattended.** Durable triggers and cron jobs let sessions start on events, with the same workspace and approval boundaries as interactive runs.
- **Multilingual by default.** Six interface languages — including Chinese, Japanese, Korean and Spanish — selectable per machine.
- **Model flexibility.** DeepSeek official, OpenRouter, and OpenCode Go routes with a curated model list and per-model reasoning efforts.

## Usage

Install globally and start in a project:

```sh
npm install -g @toddzheng024/dscode
cd /path/to/project
dscode
```

Enter `/login` and paste your DeepSeek API key — it is stored locally with `0600` permissions, never sent to the agent. Then work as usual:

```sh
dscode --continue                 # continue the last session
dscode --resume SESSION_ID        # resume a specific session
dscode --cwd /another/project     # work in another directory
dscode trigger list               # event-driven runs defined in this project
dscode doctor                     # analyse recent logs and session traces
```

Inside the TUI, hand a task to another session and let a side question run without polluting the thread:

```
/btw why is the cache cold on the first turn?
dscode send <session-id> --steer "review the change in parser.ts and reply"
```

For scripts and CI, run a single turn non-interactively:

```sh
dscode exec "prompt"
git diff | dscode exec "review this"
```

## Conclusion

DSCODE's thesis is that the unit of AI-assisted coding is not the session but the team of sessions — and the repository backs that thesis with the unglamorous machinery real collaboration needs: message budgets, worktree isolation, dependency-ordered merges, independent review, and a scripted entry point that CI can call. If you have outgrown single-session agents but found multi-agent setups to be glorified chaos, this codebase is worth studying precisely because it treats the failure modes as first-class engineering problems.

Links:

- Repository: https://github.com/qiz029/dscode
- Releases: https://github.com/qiz029/dscode/releases
- Changelog: https://github.com/qiz029/dscode/blob/main/docs/CHANGELOG.md
- Session communication design: https://github.com/qiz029/dscode/blob/main/docs/session-communication.md
