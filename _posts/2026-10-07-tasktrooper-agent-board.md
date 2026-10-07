---
layout: post
title: "TaskTrooper: Put It on the Board, the Agents Ship It - Inside makifbaysal/tasktrooper"
description: "A source-level tour of TaskTrooper, the local-first agent platform that turns a Kanban board into an orchestration layer for coding agents, with self-evolving role agents, embedded Postgres, and usage-limit resilience."
date: 2026-10-07
header-img: "img/post-bg.jpg"
permalink: /tasktrooper-agent-board/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/tasktrooper/makifbaysal-tasktrooper-architecture.svg
tags: [AI Agents, Go, Kanban, Developer Tools]
categories: [AI, Open Source]
keywords: TaskTrooper, local-first agent platform, Kanban board agents, Claude Code, Cursor, OpenCode, Go, Electron, embedded Postgres, self-evolution
author: "PyShine"
---

Agent CLIs like Claude Code, Cursor, Antigravity, and OpenCode are superb at doing work you hand them, one task at a time. What they do not do is coordinate a whole software team's worth of work: decide what ships next, hand a task from analysis to implementation to review to release, and keep going when a usage limit interrupts the middle of a run. TaskTrooper, an Apache-2.0 project by makifbaysal, was built for exactly that gap. It is a local-first agent platform for software teams of one: a Kanban board of tasks, a set of role agents such as product manager, architect, backend developer, and release engineer, and a runtime that hands each task to an agent CLI on your own machine, or to a model API you bring a key for.

The slogan on the repository's banner captures the design: "Put it on the board. The agents ship it." You do not drive the agents; the board does. A task that lands in a column is dispatched to that column's agent automatically, the agent does the work, and the task moves on to the next column, where the next agent takes over. Nobody presses run. Everything runs locally: the Electron desktop app starts an embedded Postgres and the Go backend, serves the UI, and runs the agent sessions on your machine, with no account, no cloud, and no login.

The result is one of the more complete agent-orchestration codebases you can clone today: a Go backend with a clean hexagonal layout, an Electron and Vite desktop shell, seven role agents with a catalog-driven library of skills and rules, a self-evolution loop with a golden-gate regression check, and careful engineering around the messy realities of agent runs, from blocked tasks to rate limits to per-task branches and pull requests. This tour walks the repository and shows how the pieces fit.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/tasktrooper/makifbaysal-tasktrooper-overview-architecture.svg" alt="Architecture overview of the TaskTrooper repository, showing the Electron desktop shell, the Go agent server with embedded Postgres, the HTTP API, the task orchestrator, agent CLI runners, the LLM factory, and the catalog and Postgres adapters" style="max-width:100%;">
</div>
<p><em>Architecture overview of the TaskTrooper repository, from the desktop shell down to the orchestrator and its adapters.</em></p>

Reading the overview from left to right:

- The **Electron desktop shell** in [desktop/src/main](https://github.com/makifbaysal/tasktrooper/tree/main/desktop/src/main) boots the whole stack: it loads the workspace UI from [desktop/src/renderer](https://github.com/makifbaysal/tasktrooper/tree/main/desktop/src/renderer) and spawns the Go backend as a child process.
- The **agent server entry** in [server/cmd/agent-server/main.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/cmd/agent-server/main.go) starts the embedded Postgres from [server/internal/platform/embeddedpg](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/platform/embeddedpg), runs schema migrations, and then boots the runtime in [server/internal/platform/runtime/runtime.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/platform/runtime/runtime.go).
- The **HTTP API** in [server/internal/adapter/http/handler.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/adapter/http/handler.go) exposes every capability to the UI and routes task and agent commands into the orchestrator.
- The **task orchestrator** in [server/internal/application/orchestrator](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/orchestrator) is the heart: it moves tasks between columns, spawns agent CLIs such as the Claude Code runner in [server/internal/adapter/cli/claudecode](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/cli/claudecode), or routes runs to the LLM provider factory in [server/internal/adapter/llm/factory.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/adapter/llm/factory.go) when an agent runs on an API key instead.
- The **catalog and state** layer keeps the source of truth in the [catalog/](https://github.com/makifbaysal/tasktrooper/tree/main/catalog) directory and syncs it into the Postgres adapter in [server/internal/adapter/storage/postgres](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/storage/postgres) at boot and on an interval, while the tool belt in [server/internal/adapter/tools](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/tools) gives agents concrete capabilities.

## Why You Need This

If you have ever pasted the same context into an agent CLI five times a day, you already feel the problem TaskTrooper solves. Agent sessions are stateless from the team's point of view: the CLI does not know your acceptance criteria, your column policy, or that the last run already failed code review twice. A board fixes that. Tasks carry their own acceptance criteria, and a task cannot move forward out of a review column until every criterion has a verdict. Tasks can block each other, and a blocked task simply waits until its blocker is done.

The second reason is resilience, which is rare among agent orchestrators. When an agent CLI runs out of its usage limit in the middle of a task, TaskTrooper parks the task on Blocked instead of failing it, holds other runs on the same CLI so they do not hit the same wall, and picks the task back up by itself once the limit resets. On a CLI that can resume a session, such as Claude Code with its resume flag, the agent carries on from where it stopped instead of starting over. That one behavior is the difference between a demo and a tool you can leave running overnight.

The third reason is the self-evolution loop. Agents rewrite their own playbooks from how their work actually went: on a schedule, whenever a task is sent back for revision, or on demand, an agent reviews its runs, chat messages, scores, and current skills, rules, and memories, and proposes changes. With the golden gate enabled, a golden task suite runs before and after a proposed change and an independent judge model decides whether to keep it; if the pass rate drops, the whole change set is reverted automatically. Later, each applied change is classified as effective, regressed, or neutral. Most self-improving-agent projects talk about this; this repository implements it end to end.

## How It Works

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/tasktrooper/makifbaysal-tasktrooper-architecture.svg" alt="Detailed architecture of the TaskTrooper repository, showing the Electron desktop layer, server boot, HTTP API handlers, the application core with orchestrator and services, the execution adapters for agent CLIs and LLM providers, the catalog and VCS integrations, and the state layer" style="max-width:100%;">
</div>
<p><em>Detailed architecture of the TaskTrooper repository, including execution adapters, integrations, and the state layer.</em></p>

### Understanding the Architecture

**Boot is a choreography between the desktop shell and the Go server.** The Electron main process in [desktop/src/main](https://github.com/makifbaysal/tasktrooper/tree/main/desktop/src/main) launches the agent server, whose entry in [server/cmd/agent-server/main.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/cmd/agent-server/main.go) starts an embedded Postgres via [server/internal/platform/embeddedpg](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/platform/embeddedpg) unless an external DSN is provided, applies migrations from [server/internal/platform/database](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/platform/database), and then calls into the runtime bootstrap in [server/internal/platform/runtime/runtime.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/platform/runtime/runtime.go), which mounts the HTTP handler in [server/internal/adapter/http/handler.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/adapter/http/handler.go) and wires the application services.

**The application core is where the board becomes a machine.** The orchestrator in [server/internal/application/orchestrator](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/orchestrator) contains planners, executors, verifiers, and replanners that turn a board task into agent work and judge the results; the board service in [server/internal/application/board](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/board) owns columns, transitions, and blocking; and the session service in [server/internal/application/session](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/session) tracks each agent run end to end. Prompts are rendered by [server/internal/application/prompt](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/prompt) from the embedded catalog in [catalog/embed.go](https://github.com/makifbaysal/tasktrooper/blob/main/catalog/embed.go), and the agent registry in [server/internal/application/registry](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/registry) resolves which agent owns which column.

**Execution is pluggable per agent, not per install.** Four runner adapters live under [server/internal/adapter/cli/](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/cli): Claude Code in [server/internal/adapter/cli/claudecode](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/cli/claudecode), Cursor in [server/internal/adapter/cli/cursor](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/cli/cursor), Antigravity in [server/internal/adapter/cli/antigravity](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/cli/antigravity), and OpenCode in [server/internal/adapter/cli/opencode](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/cli/opencode). The Claude Code runner renders its prompt context through the prompt service and hands child processes to the guard in [server/internal/platform/proctree](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/platform/proctree). Agents that run on an API instead go through the LLM factory in [server/internal/adapter/llm/factory.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/adapter/llm/factory.go), which selects between the native Anthropic client in [server/internal/adapter/llm/anthropic.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/adapter/llm/anthropic.go) and the OpenAI-compatible client in [server/internal/adapter/llm/openai_compat.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/adapter/llm/openai_compat.go), with the quota guard in [server/internal/adapter/llm/ratelimit.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/adapter/llm/ratelimit.go) watching the limits.

**Delivery is part of the loop, not an afterthought.** Every task gets its own branch and pull request through the Git client in [server/internal/adapter/vcs/git/client.go](https://github.com/makifbaysal/tasktrooper/blob/main/server/internal/adapter/vcs/git/client.go) and the GitHub client in [server/internal/adapter/vcs/github](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/vcs/github). Code review reads the PR; once review is signed off, the release engineer in [server/internal/application/release](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/release) merges it, ships it per the component's delivery profile, verifies production afterward, and finishes or rolls back the release, never trusting a green deploy alone.

**State and skills are synchronized deliberately.** The catalog adapter in [server/internal/adapter/catalogrepo](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/adapter/catalogrepo) reads the agent definitions from the repository's catalog directory and syncs them into Postgres at boot and on an interval, while the domain model in [server/internal/domain](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/domain) defines tasks, transitions, workflows, and the evolution machinery everything else shares. The self-evolution service in [server/internal/application/evolution](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/evolution) writes its proposals and scores back to the same store, and the board service persists every transition so the golden gate has ground truth to compare against.

End to end, a task travels like this: you drag a card into a column, the orchestrator resolves the column's owner from the registry, renders the prompt from the embedded catalog plus the task's acceptance criteria, and dispatches the run to the agent's runtime, whether that is a spawned CLI guarded by the process tree or an LLM provider call through the factory. The agent works in its own branch with the tool belt, reports findings, and the orchestrator moves the card forward, parks it on Blocked when a limit bites, or sends it back for revision, which the evolution service later turns into better playbooks.

## Advantages

- **The board is the orchestrator.** Columns, transitions, and blocking rules in [server/internal/application/board](https://github.com/makifbaysal/tasktrooper/tree/main/server/internal/application/board) decide what happens next, so coordination does not live in your head or in a brittle script.
- **Seven role agents out of the box.** Product manager, system architect, backend developer, frontend developer, mobile developer, QA agent, and release engineer ship with catalog-driven skills, rules, and tool policies, and every agent is editable or replaceable with your own.
- **One board, several runtimes.** The runtime is chosen per agent, so a developer on Claude Code, a reviewer on Cursor, a QA agent on OpenCode, and an API-only agent can share the same board.
- **Usage limits do not lose work.** Parked tasks wait on Blocked until a limit resets, sibling runs on the same CLI are held, and Claude Code sessions resume where they stopped.
- **Self-evolution with a regression gate.** Reflection proposes skill, rule, and memory changes; the golden gate and an independent judge model revert the whole change set if the pass rate drops.
- **Verification is enforced, not suggested.** Review columns cannot pass without criteria verdicts, the QA agent cannot pass a task without running something, and releases are verified in production rather than declared done on a green build.

## Benefits

- **Local-first privacy.** Embedded Postgres, local sessions, no account, no cloud, no login; your keys and your code stay on your machine.
- **Real team simulation for solo work.** The role split gives you review pressure and release discipline even when the team is one person with a board.
- **Lower token waste.** Resumable sessions and blocked-task parking mean an interrupted run does not throw away everything the agent already read and wrote.
- **Inspectability.** Agent manuals, skills, rules, memories, and KPI results are stored and editable, so you can audit why an agent behaved the way it did.
- **Clean extension points.** The hexagonal layout, with adapters for CLIs, LLM providers, tools, VCS, and storage, makes adding a new runner or provider a contained change rather than a fork.
- **Cross-platform reach.** macOS, Windows, and Linux builds ship from the same Electron and Go codebase, with an install script, Homebrew cask, setup executable, and AppImage.

## Usage

Install on macOS with the official script, or with Homebrew:

```sh
curl -fsSL https://raw.githubusercontent.com/makifbaysal/tasktrooper/main/scripts/install.sh | bash
```

```sh
brew tap makifbaysal/tasktrooper https://github.com/makifbaysal/tasktrooper
brew install --cask tasktrooper
```

On Windows, download the setup executable from Releases; on Linux, use the AppImage or the .deb package. First launch downloads the Postgres binaries and the embedding model into the app's data directory, and you need git plus at least one agent runtime: an agent CLI, or an API key for a model provider.

To run from source:

```sh
make setup      # go mod download + npm ci
make desktop    # Electron app in dev mode
make dev        # backend + UI dev server on http://localhost:3200
```

Once the app is open, connect an agent CLI or paste a provider key, create a project, and put a task on the board. Pick a column, add acceptance criteria, and the owning agent takes it from there, from analysis through code review to a merged, verified release.

## Conclusion

TaskTrooper is a thoughtful answer to a question the agent ecosystem keeps circling: what does it take to let agents run a software project instead of just assisting with one? The repository's answer is a real orchestration substrate, a board that owns the workflow, role agents with editable playbooks, execution adapters that respect each CLI's quirks, and a self-evolution loop guarded by an objective regression gate. The engineering is disciplined, the failure modes are handled, and everything runs on hardware you control. For solo developers and small teams who want agents to carry the whole delivery loop, this is one of the most complete open starting points available.

Links:

- Repository: [https://github.com/makifbaysal/tasktrooper](https://github.com/makifbaysal/tasktrooper)
- Website: [https://tasktrooper.ai](https://tasktrooper.ai)
- Docs: [https://tasktrooper.ai/docs](https://tasktrooper.ai/docs)
- Releases: [https://github.com/makifbaysal/tasktrooper/releases](https://github.com/makifbaysal/tasktrooper/releases)
- License: Apache-2.0
