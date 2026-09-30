---
layout: post
title: "ZCode: Z.ai's Open-Source Coding Agent Harness, From Turn Loop to Tool Dispatch - Inside zai-org/ZCode"
description: "A source-level tour of zai-org/ZCode, Z.ai's open-source AI coding workspace that ships terminal, web, and desktop clients over one TypeScript agent runtime. We trace the turn loop, tool scheduler, permission gate, subagents, and the plugin/MCP extension system through real file paths. Includes two architecture diagrams rendered from the actual repository tree."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /ZCode-Z-Ai-Coding-Agent-Harness-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/zcode/zai-org-zcode-architecture.svg
tags:
  - ZCode
  - AI Agents
  - Coding Agents
  - TypeScript
categories: [AI, Open Source]
keywords: "ZCode, zai-org, Z.ai coding agent, coding agent harness, agent CLI architecture, TypeScript agent runtime, agent turn loop, tool dispatch scheduler, subagents, MCP servers, plugin marketplace, open source coding agent, agent permissions, context compaction"
author: "PyShine"
---

Most coding agents you can open up are a chat loop with a handful of tools bolted to its side. The interesting ones are the projects that treat the *harness* — the machinery that turns a model stream into safe, observable, interruptible work — as the actual product. When we cloned `zai-org/ZCode` and started reading, that is the shape we found: not a demo wrapper around an API, but a full agent runtime with a phase state machine, a dependency-aware tool scheduler, foreground and background subagents, and a plugin marketplace, all wrapped in three different user interfaces.

ZCode is Z.ai's open-source AI coding workspace. The repository says it plainly in `README.en.md`: an AI coding workspace with desktop, browser, and terminal interfaces, containing "the clients, backend services, shared UI, and Agent CLI and runtime source code." Everything is TypeScript on Node.js (24.14.0, pinned in `mise.toml`), organized as a pnpm workspace, released under the Apache-2.0 license. One `zcode` command fronts all of it: with no arguments it starts a terminal UI, `zcode --web` serves a browser client from a local backend, and the Electron desktop shell in `packages/desktop` embeds the same server package.

That combination is exactly why the source is worth a tour. If you have read other open-source agent harnesses, the vocabulary here is familiar — session loop, tool registry, permission gate, MCP — but ZCode makes structural choices you rarely see together: a single `AgentRuntime` core shared by every client surface, an explicit turn state machine instead of a while-loop, a scheduler that reasons about tool safety metadata before executing anything, and an extension system that spans plugins, skills, MCP servers, and lifecycle hooks. Reading how those pieces fit is a free education in agent harness design, whether or not you end up running ZCode daily.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/zcode/zai-org-zcode-overview-architecture.svg" alt="Architecture overview of the zai-org/ZCode repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the ZCode repository: clients and gateways on the left feed one shared agent core, which dispatches into the tool and extension layer on top of a typed contracts foundation.*

Reading the overview from left to right: the `zcode` CLI entry in `apps/zcode-cli/packages/cli/src/main.ts` either launches the terminal UI or boots the agent runtime directly; the web client in `packages/web/src` and the Electron desktop shell in `packages/desktop/src` both reach the agent through the protocol server in `packages/server/src`, which the desktop embeds as a workspace dependency. The server drives `AgentRuntime` (`apps/zcode-cli/packages/core/src/runtime/agent-runtime.ts`), the single session brain. From there the turn loop streams completions through the model runner in `apps/zcode-cli/packages/adapters/src/model/runner.ts` and dispatches tool calls into the registry and scheduler, which can spawn subagents via the `Task` tool. Along the bottom, the `contracts` package carries typed session events and payload shapes that every layer shares.

## Why You Need This

First, the obvious one: you want an agent you can actually inspect. ZCode ships its entire stack as source — the agent CLI, the TUI, the web client, the backend, the Electron host — under Apache-2.0. There is no "open core with a closed brain" arrangement; the harness itself is the repository. When a turn misbehaves, every decision point between your prompt and the model request is code you can read, starting from `apps/zcode-cli/packages/core/src/runtime/`.

Second, agent logic tends to rot when it is duplicated per interface. ZCode attacks that directly: the TUI, web client, and desktop shell are all thin surfaces over one `AgentRuntime`. The runtime's public methods live in `apps/zcode-cli/packages/core/src/runtime/methods/` — turns, compaction, MCP management, subagent control, background tasks — and every surface calls the same core. The repo even enforces this with an architecture checker (`pnpm architecture:check`, backed by `scripts/architecture/`), so structural drift is a build failure, not a slow surprise.

Third, tool execution reliability. The hard part of an agent harness is not calling tools; it is deciding which tools may run together, which need approval, and what happens when a call fails mid-turn. ZCode models this explicitly — every tool entry carries metadata like `readOnly`, `destructive`, `concurrentSafe`, and `needsApproval` (`apps/zcode-cli/packages/core/src/tool/registry.ts`), and a scheduler turns that metadata into a parallel execution plan. If you are building your own harness, that layer alone is worth studying.

Finally, extensibility without forking. Coding agents live or die by how much of the surrounding environment they can reach: other tools via MCP, project knowledge via skills, policy via hooks, whole capabilities via plugins. ZCode ships all four mechanisms as first-class subsystems with on-disk conventions and CLI commands, so adapting it to a team's workflow is configuration and authoring rather than patching the core.

## How It Works

To see the machinery up close, we walk the detailed diagram below and then zoom into the five subsystems that do the real work.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/zcode/zai-org-zcode-architecture.svg" alt="Detailed architecture of the zai-org/ZCode repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of zai-org/ZCode: the clients group connects through the server and RPC layer into the agent core runtime, which coordinates tools, subagents, and the MCP/plugin/skill extension stack over the shared contracts package.*

### Understanding the Architecture

**The session loop is a first-class state machine.** A turn in ZCode is not a loose `while` loop. `apps/zcode-cli/packages/core/src/agent/turn-machine.ts` defines explicit phases — processing input, awaiting the model, streaming, scheduling tools, awaiting permission, executing tools, aggregating results, completing — and illegal transitions throw. The driver, `runRegularTurnLoop` in `apps/zcode-cli/packages/core/src/runtime/methods/turn-loop.ts`, walks that machine on every iteration: it drains pending runtime commands and steering messages, runs microcompaction, triggers auto-compact near the context limit (with a "rapid refill" guard in `turn-loop-state.ts` against compact-thrash loops), initializes MCP, filters the tool list for the turn, and only then issues the model request. The machine's `getNextPhase` decides whether results loop back to another model round or the turn completes.

**Tool dispatch is dependency-aware and safety-aware.** The registry in `apps/zcode-cli/packages/core/src/tool/registry.ts` stores each tool with rich metadata — read-only, destructive, concurrent-safety, approval requirements, timeouts, output budgets — and exposes provider-visible contracts augmented with usage instructions. When the model returns a batch of calls, `apps/zcode-cli/packages/core/src/tool/scheduler.ts` builds a dependency graph, topologically sorts it, groups calls into parallel waves, and validates there are no cycles. Destructive tools never run in parallel, read-only tools do by default, and concurrency is capped (the default is 10). Permission resolution is woven through the same pipeline: a call that needs approval parks the turn in the `AwaitingPermission` phase until the permission service (`core/src/permission/service.ts`) or the user resolves it.

**Subagents are child runtimes, not prompt strings.** The dispatch tool is named `Task` (with a legacy `Agent` alias, per `core/src/tool/compat.ts`). The runner in `apps/zcode-cli/packages/core/src/subagent/runner.ts` can run tasks in the foreground or background, auto-backgrounds long-running work, watches for inactivity, and notifies the parent session when a background task completes. Two built-in profiles ship in the core: a read-only Explore specialist whose prompt and tool whitelist are defined in `core/src/subagent/explore.ts` and `explore-tools.ts`, and a general-purpose researcher in `general-purpose.ts`. In `core/src/runtime/methods/subagent.ts`, each child gets its own `AgentRuntime` with a filtered tool set, borrowed MCP access, mirrored tool events, and a message-steering sink — recursion all the way down.

**Extensions plug in at three depths.** Depth one is MCP: servers of type `stdio`, `http`, or `sse` are configured in `~/.zcode/cli/config.json`, connected before the first model request, and surfaced as `mcp__<server>__<tool>` tools. Depth two is plugins: a plugin is a directory with a `.zcode-plugin/plugin.json` manifest contributing skills, markdown commands, and MCP servers; the official marketplace (`zcode-plugins-official`) seeds built-in plugins like browser-use and document-skills and fetches CDN-distributed zips with sha256 verification (`apps/zcode-cli/packages/adapters/src/plugins/`). Depth three is lifecycle hooks — seven events from `SessionStart` to `Stop` — where process hooks exchange JSON on stdin/stdout and exit code 2 acts as a hard deny. The trust machinery for workspace hooks lives in `core/src/hooks/workspace-hook-trust-*.ts`.

**Context is managed, not merely appended.** The prompt is assembled by `core/src/context/builder.ts` from static sections plus dynamic sources, while the compaction subsystem (`core/src/compact/` and the runtime's `compact.ts` method) summarizes and trims history as it grows — manual compaction, microcompaction between steps, and policy-driven auto-compact at the context limit. On top of that sits a persistent project-memory loop (`core/src/memory/memory-agent-loop.ts`) that extracts and recalls durable knowledge across sessions.

**The model layer absorbs provider mess.** `apps/zcode-cli/packages/adapters/src/model/` wraps streaming execution in a retry budget, stream idle timeouts, a streaming tool-call assembler, strict tool-schema handling, and tool-call validation, with OAuth flows for the official gateway under `adapters/src/provider/`. The core loop sees a clean stream of events; the adapter deals with everything that can break between here and the provider.

The end-to-end flow ties it together: a prompt enters through a client, is serialized onto the runtime's command queue (`core/src/runtime/command-queue.ts`), and starts a turn. The context builder assembles the request from message history, the model step streams a response, any tool calls are scheduled into safe parallel waves, gated by permission and hooks, executed, and their results are aggregated back into history — then the loop returns to the model until a round produces no further tool calls. Subagents launched by `Task` run this same loop inside child runtimes, reporting back to the coordinator that spawned them.

## Advantages

- **One harness, three surfaces.** The TUI, web client, and desktop app all sit on the same `AgentRuntime` and the same tool/subagent stack, so a capability added once works everywhere.
- **Explicit turn semantics.** The phase state machine in `turn-machine.ts` makes interruption, permission waits, and tool failures structured states rather than ad-hoc flags — a pattern worth copying in any agent project.
- **Safety-first scheduling.** Read-only/destructive/concurrent metadata plus topological scheduling gives deterministic, auditable tool execution instead of fire-and-forget parallelism.
- **A real extension ecosystem.** Plugins with manifests and a marketplace, MCP servers, skills, and seven lifecycle hooks cover capability, knowledge, and policy extension points separately and cleanly.
- **Context engineering built in.** Microcompaction, auto-compact with anti-thrash guards, and a persistent memory loop address the failure mode that kills long coding sessions: running out of usable context.
- **Engineering discipline you can verify.** TypeScript everywhere, an architecture checker in the build, dependency-graph scripts, and zero production dependencies in the CLI runtime keep the core auditable.

## Benefits

- **For agent builders,** ZCode is a working reference for the pieces that are hardest to get right: turn state, tool scheduling, approval flow, and subagent lifecycles — all readable under Apache-2.0.
- **For engineering teams,** the CLI distribution (`pnpm build:zcode` producing a tarball, checksum, and `install.sh`) plus configurable data directories make self-hosting and pinning a coding agent straightforward.
- **For tool authors,** one MCP server or plugin manifest reaches the terminal, web, and desktop clients at once, with per-plugin data directories and user-config expansion handling the hygiene.
- **For platform and security reviewers,** the permission broker, rule matching, workspace hook trust evaluation, and bash read-only policy files (`core/src/tool/handlers/bash-readonly-policy-*.ts`) give you concrete code to audit rather than promises.
- **For daily drivers,** the trade is the reverse of lock-in: a fast local TUI when you want it, a shareable web surface when you need it, and the same sessions behind both.
- **For contributors,** the monorepo boundaries are enforced by tooling, so you can tell immediately whether a change belongs in `core`, `adapters`, or a client package.

## Usage

Bootstrap the workspace (requires Git, Node.js 24.14.0, and pnpm 10.33.2 per `mise.toml`):

```bash
pnpm bootstrap
```

Develop against the desktop or web surfaces:

```bash
pnpm dev:desktop
pnpm dev:web
```

Work directly with the Agent CLI source:

```bash
pnpm --filter @zcode/cli dev --help
pnpm --filter @zcode/cli... build
node apps/zcode-cli/packages/cli/dist/zcode.cjs --help
```

The unified `zcode` command starts the TUI by default; `--web` serves the browser client from a local backend:

```bash
zcode
zcode --web
zcode --web --workspace /path/to/project --port 3030 --no-open
zcode --help
```

Manage plugins from the same CLI:

```bash
zcode plugins list
zcode plugins enable ios-simulator
zcode plugins disable browser-use
```

To package the full CLI distribution (TUI, backend, and web client behind one command):

```bash
pnpm build:zcode --base-url https://downloads.example.com/zcode/
```

## Conclusion

ZCode is one of the more complete open-source agent harnesses currently readable end to end: a phase-driven turn loop, a metadata-driven tool scheduler, recursive subagents, and a three-layer extension model, all shared by terminal, web, and desktop clients. Structurally it occupies an interesting point in the open-harness landscape — CLI-first agents give you the loop and the tools, and ZCode keeps all of that while adding the server, client SDK, and shared UI layers that turn a solo assistant into a workspace. If your interest is building agents rather than merely using them, the tour from `main.ts` through `turn-loop.ts` to `scheduler.ts` is time well spent.

Links:

- GitHub repository: [zai-org/ZCode](https://github.com/zai-org/ZCode)
- ZCode CLI and runtime sources: [apps/zcode-cli](https://github.com/zai-org/ZCode/tree/main/apps/zcode-cli)
- Community: Discord and Feishu links are available in the repository README.
