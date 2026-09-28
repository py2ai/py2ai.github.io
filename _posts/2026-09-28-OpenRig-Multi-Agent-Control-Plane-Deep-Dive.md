---
layout: post
title: "OpenRig: One Control Plane for Your AI Agent Team - Inside mvschwarz/openrig"
description: "A source tour of mvschwarz/openrig, an open-source TypeScript multi-agent harness that turns AI coding agents into persistent, organized teams. We walk the real architecture: a Hono HTTP daemon, RigSpec YAML topologies, durable SQLite state, a queue/inbox/outbox coordination model, tmux runtime adapters for Claude Code and Codex, a TUI, and an MCP server."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /OpenRig-Multi-Agent-Control-Plane-Deep-Dive/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/openrig/mvschwarz-openrig-architecture.svg
tags:
  - AI Agents
  - Multi-Agent
  - TypeScript
  - Open Source
categories: [AI, Open Source]
keywords: "OpenRig, mvschwarz, multi-agent harness, AI coding agents, Claude Code, Codex, tmux, RigSpec, agent topology, MCP server, SQLite, TypeScript, open source, agent orchestration, local daemon"
author: "PyShine"
---

Build anything real with AI coding agents and the pattern is familiar: one terminal for the implementer, another for the reviewer, a third for the planner — until serious work means a dozen tabs. Each session is independent, none share state, a reboot erases the arrangement, and nobody can answer "what is my team doing right now?" The tools that wrap the models are excellent, but the layer above them — the system your agents form when they run *together* — is still ad-hoc scripts and muscle memory.

**OpenRig** from `mvschwarz/openrig` is a multi-agent harness aimed at exactly that layer. Its framing is sharp: a harness wraps a model; a rig wraps your harnesses. You define the team in a YAML **RigSpec** — pods, members, edges, continuity policies — and boot it with one command, `rig up`, with Claude Code and Codex managed as one system. Under the hood it is a local daemon plus a CLI, a terminal UI, and an MCP server, all built on tmux: a TypeScript monorepo (`packages/daemon`, `packages/cli`, `packages/tui`, plus a React web UI in maintenance mode), published as `@openrig/cli`, version 0.5.17, Apache 2.0 licensed.

The source is worth a tour even if you never launch a rig: it is an unusually complete reference control plane, with SQLite state behind an 85-migration chain, a queue/inbox/outbox coordination model, snapshot/restore explicit about what resumed and what did not, watchdog policies, and a declarative workflow runtime. Reading how these fit together teaches what "managing agents as a system" actually requires — far more than a loop around a chat API.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openrig/mvschwarz-openrig-overview-architecture.svg" alt="Architecture overview of the mvschwarz/openrig repository" style="max-width:100%;height:auto;" />
</div>

*Overview of OpenRig: the rig CLI, MCP server, and TUI all talk to a Hono HTTP daemon, whose domain services instantiate RigSpec YAML into running rigs, drive Claude Code and Codex adapters through tmux, and persist everything to SQLite while an event bus keeps the TUI live.*

Reading the overview from left to right: the **Client Surfaces** group is where operators and agents enter — the `rig` CLI (`packages/cli/src/index.ts`), an MCP server (`packages/cli/src/mcp-server.ts`), and the terminal UI (`packages/tui/src/main.ts`). All three converge on the **Daemon Core**: `packages/daemon/src/server.ts` builds a Hono HTTP app whose routes delegate to domain services, including the RigSpec engine (`packages/daemon/src/domain/rigspec-instantiator.ts`) that turns the YAML specs under `packages/daemon/specs/rigs` into running topologies, and an event bus (`packages/daemon/src/domain/event-bus.ts`) that fans out change events. The **State Layer** persists rig, session, queue, and workflow state in SQLite (`packages/daemon/src/db/connection.ts`), and the **Agent Runtimes** group is where work happens: dedicated adapters (`packages/daemon/src/adapters/claude-code-adapter.ts`, `packages/daemon/src/adapters/codex-runtime-adapter.ts`) drive real tmux sessions (`packages/daemon/src/adapters/tmux.ts`).

## Why You Need This

The first problem is fragility. A team assembled by hand exists only in your terminal multiplexer's memory and your head: close the laptop and the topology is gone. OpenRig makes it a durable object — the daemon records every rig, pod, seat, and edge in SQLite, `rig down --snapshot` captures the full state, and a later `rig up <name>` restores by name, reporting per node whether it resumed, came up fresh, or failed. Reboots become a recoverable event instead of a rebuild-from-scratch exercise.

The second problem is heterogeneity. Teams deliberately run more than one harness — one model family for implementation, another for review — and then the tooling diverges: startup flags, permission surfaces, transcript formats. OpenRig normalizes this behind a five-method runtime adapter contract, with native adapters for Claude Code and Codex (plus terminal nodes and a Pi adapter). It can even adopt what already exists: `rig discover` fingerprints sessions already running in tmux, and `rig adopt` binds them into a managed rig rather than forcing a restart.

The third problem is coordination. Copy-pasting work between terminal panes does not survive contact with reality — messages get lost, pending tasks are remembered by nobody, and there is no audit trail. OpenRig ships a durable coordination primitive: a shared queue with per-seat inbox and outbox, exposed through `rig send`, `rig broadcast`, and `rig chatroom`. Every message and task lands in the daemon's database, so handoffs survive restarts and can be listed later. The starter rigs operationalize this: `first-project` pairs an owner seat with a checker seat, and the owner is asked to track work in the queue and request a checker review before reporting back.

The fourth problem is trust. Giving an agent a terminal is granting real power, and "just run it" is not a governance strategy. OpenRig's defaults go the other way: YOLO mode is off — full-permission bypass requires an explicit `OPENRIG_YOLO=1` or a deliberate seat policy — and launching a rig writes provider hooks and workspace trust settings that the README documents in a dedicated "what OpenRig changes on your machine" section, with `rig setup --dry-run` showing the plan before anything is applied. Conservative defaults, explicit opt-in, honest disclosure: the posture more agent tooling should copy.

## How It Works

OpenRig runs as a local daemon that the CLI, TUI, and MCP server all reach over HTTP, with tmux as the substrate where agent sessions live.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openrig/mvschwarz-openrig-architecture.svg" alt="Detailed architecture of the mvschwarz/openrig repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the OpenRig daemon: Hono routes mounted in packages/daemon/src/server.ts delegate to domain services — rig lifecycle, RigSpec codec, node launcher, seat service, queue repository, workflow runtime, discovery, watchdog, snapshots — backed by an SQLite migration chain, with runtime adapters driving tmux panes.*

### Understanding the Architecture

**The client surfaces.** Everything a human or an agent touches is a thin client over the daemon. The CLI entry (`packages/cli/src/index.ts`) provides commands for launching teams, inspecting state, sending messages, and managing context, wired up through `packages/cli/src/bin-wrapper.ts`. The TUI (`packages/tui/src/main.ts`) is a genuine cockpit: a topology graph of the running rig, seat tables with runtime, model, and context state, plus Specs, Projects, Terminals, and Feed views. The MCP server (`packages/cli/src/mcp-server.ts`) wraps the same `DaemonClient` (`packages/cli/src/client.ts`) and exposes tools such as `rig_up`, `rig_down`, `rig_ps`, `rig_snapshot_create`, `rig_restore`, `rig_discover`, and `rig_bind` — so an agent can manage its own team topology through the Model Context Protocol.

**The daemon.** `packages/daemon/src/server.ts` constructs a Hono application and mounts a large route surface — rigs, sessions, events, snapshots, restore, discovery, transcripts, queue, workflow, mission control, and dozens more — each route module receiving its dependencies as injected domain services. Authentication middleware (`packages/daemon/src/middleware/auth-bearer-token.ts`) guards the API, and a WebSocket integration (`@hono/node-ws`) carries terminal and event streams. The daemon is a plain local process: it runs on your machine, owns your tmux server, and its HTTP API is the single point of control.

**The domain core.** The real logic lives in `packages/daemon/src/domain/`, and the file names read like a control-plane checklist: `rig-lifecycle-service.ts` for boot and teardown, `rigspec-instantiator.ts` for turning YAML specs into concrete rigs, `node-launcher.ts` for starting seats, `queue-repository.ts` with inbox and outbox handlers for durable handoff, `workflow-runtime.ts` for declarative workflow specs, `discovery-coordinator.ts` for fingerprinting existing sessions, `snapshot-capture.ts` and `restore-orchestrator.ts` for the snapshot cycle, and `watchdog-scheduler.ts` with a policy engine. This is where "manage agents as a system" is actually implemented — typed services, explicit state machines, no magic.

**The state layer.** Everything survives restarts because everything is written down. The daemon persists to SQLite through `better-sqlite3` (`packages/daemon/src/db/connection.ts`), with a schema evolved by exactly 85 versioned migrations applied by `packages/daemon/src/db/migrate.ts` — a chain that tells the project's own history, from core schema and bindings through workflow instances, watchdogs, and permission policies. The declarative inputs ship alongside the code: `packages/daemon/specs/rigs/launch/first-project/rig.yaml` defines the two-seat starter, `packages/daemon/specs/rigs/launch/conveyor/rig.yaml` the four-seat mixed-harness conveyor, and `packages/daemon/specs/rigs/preview/product-team/rig.yaml` a larger product squad.

**The agent runtimes.** The adapter layer (`packages/daemon/src/adapters/`) is where OpenRig meets the outside world. `claude-code-adapter.ts` and `codex-runtime-adapter.ts` implement a common runtime adapter contract for their harnesses, `tmux.ts` manages the sessions and panes every seat lives in, and `cmux.ts` supports the cmux terminal provider as an alternative view. Because every agent is a real tmux session, you are never locked out: attach, inspect scrollback, work with any seat directly — OpenRig manages the team without taking away the terminal.

**The end-to-end flow.** Run `rig up first-project --cwd .` and the CLI posts to the daemon's `/api/up` route; the instantiator reads the shipped RigSpec, the node launcher creates tmux sessions, and the Codex adapter boots the owner and checker seats with managed startup (trust, hooks, identity environment). Readiness is checked, events flow over the bus, and `rig ps --nodes --rig first-project` or the TUI show exactly what is running. A `rig send` travels through the durable queue into the owner's inbox; `rig down --snapshot` then freezes the arrangement for the next `rig up first-project` to restore.

## Advantages

- **Declarative topologies.** A RigSpec YAML file — pods, members, edges, continuity policies — beats shell history as a definition of your team: reviewable, versionable, reusable.
- **Real terminals, not abstractions.** Every seat is a tmux session you can attach to and inspect; the harness coordinates the team without walling off the agents.
- **Harness-agnostic by contract.** Claude Code, Codex, terminal nodes, and Pi sit behind one runtime adapter contract, so mixing model families is a spec edit, not a rewrite.
- **Durable coordination.** Queue, inbox, and outbox live in SQLite, so handoffs and operator messages survive restarts and remain listable and auditable.
- **Honest snapshot/restore.** `rig down --snapshot` plus `rig up <name>` reports per node what resumed, what came up fresh, and what failed — no pretending everything recovered.
- **Self-management via MCP.** The bundled MCP server exposes the same control plane to the agents themselves, enabling agent-driven topology changes with the operator in the loop.

## Benefits

- **Reclaims time lost to session archaeology.** No more recreating a dozen terminal sessions after a reboot; the topology is restored by name from a snapshot.
- **Fits existing teams.** Discovery and adoption bring already-running tmux sessions under management, organizing what you have instead of demanding a clean slate.
- **Ships proven team patterns.** Starter rigs like the owner/checker pair and the conveyor pipeline give you working team shapes — intake, planning, build, review — on day one.
- **Keeps autonomy accountable.** YOLO off by default, explicit permission configuration, and a documented inventory of every machine change make broad access a choice, not an accident.
- **Maintains observability.** The TUI's topology graph, seat details, feed, and mission-control views answer "what is my team doing?" at a glance, and `rig queue list` shows the work trail.
- **Stays simple to run.** A local daemon, one npm install, and tmux — no cloud account, no orchestration cluster, no vendor lock-in.

## Usage

The README states the requirements plainly: Node.js 20, 22, or 24 and tmux, on macOS or Linux (native Windows is not supported yet, and WSL2 has not been tested). Install and inspect the setup plan first:

```bash
npm install -g @openrig/cli
rig setup --dry-run
```

Check prerequisites — the first-project starter needs tmux and an authenticated Codex:

```bash
tmux -V
codex --version
codex login status
```

From a repository, preview the plan, then boot the two-seat starter and open the shared dashboard:

```bash
cd /path/to/your/repository
rig up first-project --cwd . --plan
rig up first-project --cwd .
rig tui --shared
```

Check seat readiness, then give the owner one bounded outcome — the README's own example:

```bash
rig ps --nodes --rig first-project
rig send dev-owner@first-project 'Implement <one useful change>. Track the task in the queue and return its ID. Keep it local, verify the behavior, ask dev-check@first-project to check the exact candidate, and record the result and how I can try it.'
rig queue list --destination dev-owner@first-project --limit 1000
```

Browse the other shipped topologies — `product-team` (two orchestrators, implementation, QA, design, two independent reviewers), the four-seat mixed-harness `conveyor`, plus `implementation-pair`, `adversarial-review`, `research-team`, and `secrets-manager`:

```bash
rig specs preview product-team --kind rig
rig up product-team
rig specs ls
```

Finally, make the team recoverable:

```bash
rig down --snapshot
rig up first-project
```

## Conclusion

OpenRig occupies a layer of the AI-coding stack that is easy to overlook and impossible to do without: the layer that manages agents as a system rather than as individual chats. Its choices — a Hono daemon over durable SQLite state, a declarative RigSpec, an adapter contract over tmux, a queue-based coordination primitive, snapshots that tell the truth about what resumed — read like a field guide to agent infrastructure that survives daily work. If you run more than one AI coding agent, point it at a repository, boot the two-seat starter, and watch a rig wrap your harnesses.

### Links

- [GitHub: mvschwarz/openrig](https://github.com/mvschwarz/openrig)
- [npm: @openrig/cli](https://www.npmjs.com/package/@openrig/cli)
- [Apache 2.0 License](https://github.com/mvschwarz/openrig/blob/main/LICENSE)
