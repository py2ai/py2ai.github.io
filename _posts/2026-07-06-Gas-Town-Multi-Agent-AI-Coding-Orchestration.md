---
layout: post
title: "Gas Town: Orchestrate 30+ AI Coding Agents in One Git-Backed Workspace"
date: 2026-07-06 12:00:00 +0800
categories: [ai, multi-agent, developer-tools]
tags: [gas-town, multi-agent, ai-coding, orchestration, claude-code, github-copilot, open-source]
permalink: /Gas-Town-Multi-Agent-AI-Coding-Orchestration/
featured-img: ai-coding-frameworks/ai-coding-frameworks
---

## Introduction

The era of running a single AI coding agent is ending. Teams now run 10, 20, even 30 agents simultaneously, but coordination breaks down fast. Agents lose context on restart, overwrite each other's work, duplicate effort, and drift apart. The productivity gains of adding more agents collapse beyond a handful without real orchestration.

[Gas Town](https://github.com/gastownhall/gastown) is the workspace manager that makes multi-agent AI coding reliable and scalable. It is not another coding agent - it is a coordinator that manages existing agents (Claude Code, GitHub Copilot, Codex, Gemini, Cursor, and more) with git-persistent state, a 3-tier watchdog monitoring system, and a Bors-style bisecting merge queue.

The numbers tell the story: 16.5k GitHub stars, 1.5k forks, 7,655 commits showing active development, MIT license, and cross-platform support for macOS, Linux, and Windows. The project is trending at roughly 51 stars per day. Written in Go for single-binary distribution, Gas Town turns the chaos of many agents into a fleet that works like a team.

The one-line value proposition: coordinate multiple AI coding agents with git-persistent state, built-in monitoring, and a Bors-style merge queue - all from one workspace directory.

## The Multi-Agent Problem

Running a few AI coding agents in parallel sounds great until you actually try it. The problems appear quickly and compound as you scale.

**State loss on crash.** When an agent process dies, its in-memory work state vanishes. There is no recovery path. You restart the agent and it begins from scratch, losing hours of context, partial edits, and reasoning. At 10 agents, this happens daily.

**Overwrites and conflicts.** Multiple agents editing the same repository produce conflicting branches. Without a queue, merges become a manual conflict-resolution marathon. Two agents fix the same bug independently and one overwrites the other.

**Duplicate work.** Without a central coordinator, agents have no visibility into what others are doing. Three agents independently research the same library. Two implement the same feature. Wasted API calls multiply.

**Identity and handoff chaos.** Tracking which agent is working on what, who finished, and who needs the next task becomes a full-time job. Manual handoffs do not scale past a handful of agents.

**API rate limit exhaustion.** Uncoordinated agents hammer the same provider simultaneously. You hit rate limits not because of volume but because of poor scheduling. One agent's burst blocks everyone else.

**The scaling wall.** Beyond roughly 10 agents, productivity per agent drops sharply without orchestration. The overhead of coordination exceeds the value of parallelism. This is the wall Gas Town was built to break through.

## What is Gas Town?

Gas Town is a workspace manager, not a coding agent itself. It coordinates existing AI coding agents rather than replacing them. This is a critical distinction - you keep the agents you already use and add a management layer on top.

**Multi-runtime support.** Gas Town works with Claude Code, GitHub Copilot, Codex, Gemini, Cursor, and more. Each agent runtime is configured separately, so you can mix providers in a single workspace. One rig might use Claude Code while another uses Codex, all coordinated by the same Mayor.

**Git as the source of truth.** All work state persists in git-backed hooks. When an agent crashes and restarts, it reattaches to its git worktree and resumes from the persisted state. No lost work, no starting over. The entire workspace state is versioned, auditable, and recoverable.

**The town metaphor.** Gas Town uses a consistent civic vocabulary: a Mayor coordinates, Rigs hold projects, Polecats do the work, Hooks persist state, a Refinery merges results, and a Deacon watches over everything. The metaphor is not decoration - it maps directly to the architecture and makes the system intuitive once you learn the terms.

**Written in Go.** A single binary distributes to macOS, Linux, and Windows. No runtime dependencies, no container required (though Docker Compose is supported). The Go choice prioritizes deployment simplicity.

**Open source under MIT.** Free to use, modify, and contribute. The repository is active with 7,655 commits and a growing community.

## Core Concepts - The Town Vocabulary

Gas Town's architecture is built on a civic metaphor. Each term maps to a concrete component. Learning the vocabulary is learning the system.

- **The Mayor** - The primary AI coordinator. Typically a Claude Code instance with full workspace context. The Mayor assigns work, monitors progress, and summarizes results. It is the brain of the operation.
- **Town** - The workspace directory, conventionally `~/gt/`. It contains all projects, agents, and configuration. One Town is your entire multi-agent operation.
- **Rigs** - Project containers. Each rig wraps a git repository with its own lifecycle, agents, merge queue, and monitoring. A Town holds many rigs.
- **Crew Members** - A personal workspace within a rig for hands-on human or agent work. The crew member is where direct interaction happens.
- **Polecats** - Worker agents with persistent identity but ephemeral sessions. A polecat spins up, does work, persists its state to hooks, and spins down. The identity persists across sessions; the process does not.
- **Hooks** - Git worktree-based persistent storage. This is the core innovation. Every agent's working state is a git worktree branch that survives process death, crashes, and restarts.
- **Convoys** - Work tracking units. A convoy bundles multiple beads assigned to one or more polecats for coordinated execution.
- **Beads** - Git-backed issue tracking. Each bead is a structured work item stored as git objects (commits, trees, data) with state, assignee, and history.
- **Molecules** - Workflow templates defined in TOML. A molecule coordinates multi-step work - for example, "implement feature" broken into research, implement, test, review.
- **Refinery** - Per-rig merge queue processor using a Bors-style bisecting algorithm. Batches candidate merges, tests them together, and bisects on failure.
- **Wasteland** - Federated work coordination network built on DoltHub. Enables multiple Town instances to share work, beads, and convoys.

## Architecture Overview

Gas Town uses a layered architecture. The Town sits at the top as the workspace container. The Mayor sits above all rigs with cross-workspace context. Each rig is self-contained with its own git repo, agents, merge queue (Refinery), and monitoring (Witness). Hooks provide the persistence substrate at the bottom - every agent's state is a git worktree.

![Architecture](/assets/img/diagrams/gastown/gastown-architecture.svg)

### Component Breakdown

The diagram shows the hierarchy from top to bottom. The Town node (dark blue) at the top represents the `~/gt/` workspace directory - the root container for everything. Below it, the Mayor (amber) is the AI coordinator, typically a Claude Code instance that holds full workspace context and routes work.

The Mayor branches to multiple Rigs (green). Each rig is a project container wrapping a git repository. Rig 1, Rig 2, and Rig N show that a Town can hold many projects. Each rig contains Crew Members (purple) for direct interaction and Polecats (red) for autonomous work.

The Polecats all converge downward to Hooks (teal) - the git worktree persistent storage layer. This is the substrate that makes state survive crashes. The Refinery (orange) attaches to each rig as the merge queue, and the Witness (pink) attaches as the lifecycle manager.

### Context

This architecture solves the state-loss problem that plagues multi-agent setups. By making git worktrees the persistence layer, Gas Town ensures that no matter how many agents crash or restart, their work state is recoverable. The Mayor's position above all rigs gives it the cross-project context needed to route work intelligently.

### Technical Details

The layered design means each rig is independent. A failure in one rig does not cascade to others. The Refinery and Witness are per-rig services, so merge queues and monitoring are scoped appropriately. The Hooks layer is shared infrastructure - all polecats persist to git worktrees, but each worktree is isolated, preventing file conflicts between agents.

### Value-Added

The key insight is that Gas Town separates coordination from execution. The Mayor coordinates; the Polecats execute; the Hooks persist; the Refinery merges; the Witness monitors. Each layer has a single responsibility. This separation is what allows the system to scale to 30+ agents without collapsing into chaos.

### Conceptual Understanding

Think of Gas Town as a city government. The Town is the city limits. The Mayor runs the city. Each Rig is a department with its own projects. Polecats are workers who come and go but whose desk (Hooks) stays set up. The Refinery is the quality-control checkpoint. The Witness is the supervisor. The metaphor makes a complex distributed system feel familiar.

## Installation and Setup

Gas Town offers three installation paths covering all major platforms.

### Homebrew (macOS)

```bash
brew install gastown
```

### npm (cross-platform)

```bash
npm install -g @gastown/gt
```

### From source (Go)

```bash
go install github.com/steveyengge/gastown/cmd/gt@latest
```

### Shell completions

Gas Town provides shell completions for bash, zsh, and fish. After installation, generate completions for your shell:

```bash
# bash
gt completion bash > /etc/bash_completion.d/gt

# zsh
gt completion zsh > "${fpath[1]}/_gt"

# fish
gt completion fish > ~/.config/fish/completions/gt.fish
```

### Docker Compose

For containerized deployments, Gas Town ships with Docker Compose support:

```bash
docker compose up -d
```

### Initial setup

After installing the `gt` binary, initialize your Town:

```bash
# Create the Town directory structure
gt init

# Add a rig wrapping an existing git repository
gt rig add /path/to/your/repo

# Configure agent runtimes (Claude Code, Codex, Copilot, Gemini, Cursor)
gt config set-runtime claude-code
gt config set-runtime codex
```

### Verification

Confirm your installation:

```bash
gt version
gt status
gt rig list
```

The `gt status` command shows the health of your Town, including active rigs, running polecats, and any stuck agents detected by the monitoring system.

## Multi-Agent Workflows - Convoys, Beads, and Molecules

Gas Town's work tracking follows a three-level hierarchy: Molecule (template) becomes Convoy (work unit) which contains Beads (individual tasks).

![Workflow](/assets/img/diagrams/gastown/gastown-workflow.svg)

### Component Breakdown

The diagram flows left to right. A Molecule (purple) is a TOML-defined workflow template - for example, "implement feature" defined as four steps. The Molecule expands into a Convoy (blue), which is the actual work tracking unit containing a bundle of beads.

The Convoy produces four Beads (green): Research, Implementation, Testing, and Review. Each bead is a git-backed work item with its own state, assignee, and history. The beads flow to the Mayor (amber), which routes them to available Polecats (red).

The Mayor assigns beads based on agent capacity and skill. Polecat A gets the research bead, Polecat B gets implementation, Polecat C gets testing. Each polecat persists its working state to Hooks (teal) as it progresses. When all beads complete, the results flow to the Refinery (orange) for merging.

### Context

This pipeline shows how Gas Town turns a workflow template into coordinated multi-agent execution. The Molecule-to-Convoy-to-Bead hierarchy means you define workflows once as templates and reuse them. The Mayor handles the routing intelligence - you do not manually assign tasks to agents.

### Technical Details

Beads are stored as git objects - commits, trees, and structured data. This means the entire work history is versioned and auditable. You can query the state of any bead at any point in time. When a polecat completes a bead, the next bead in the convoy is automatically assigned, enabling chained workflows without manual handoffs.

### Value-Added

The separation of template (Molecule) from execution (Convoy) from task (Bead) gives you reuse, tracking, and granularity. You can define a Molecule once and run it across many rigs. You can inspect any bead's state. You can reassign beads when agents fail. This is workflow engineering, not just task dispatch.

### Conceptual Understanding

A Molecule is a recipe. A Convoy is a batch cooked from that recipe. Beads are the individual dishes. The Mayor is the head chef assigning dishes to cooks (Polecats). Each cook keeps their station (Hooks) set up between shifts. When all dishes are ready, they go to quality control (Refinery) before serving.

## Git-Backed Persistence - Hooks Explained

Hooks are Gas Town's core innovation. Work state lives in git worktrees, not in agent memory. This single design choice eliminates the most painful failure mode of multi-agent systems: lost work on crash.

Each polecat's working state is a git worktree branch. When the polecat process dies - whether from a crash, a restart, or a deliberate spin-down - the worktree remains. On restart, the agent reattaches to its worktree and resumes from the persisted state. No lost context, no partial edits discarded, no reasoning thrown away.

Beads themselves are git objects: commits, trees, and structured data stored in the repository. This means the entire workspace state is versioned, auditable, and recoverable. You can reconstruct the exact state of any agent's work at any point in time by checking out the worktree branch.

Compare this to in-memory state, the default for most agent setups. An in-memory agent that crashes loses everything. You start over. With Hooks, the agent's "memory" is the git worktree, which is durable by definition.

Worktree isolation is the other benefit. Each agent works in its own worktree, a separate working directory pointing at the same repository. Agents cannot overwrite each other's files because they are physically separate directories. Conflicts are deferred to the Refinery's merge queue, where they are handled systematically rather than chaotically.

This approach means Gas Town treats git not just as version control but as a persistence substrate and coordination mechanism. Git is the database, the message queue, and the audit log - all in one.

## The Monitoring System - 3-Tier Watchdog

Running 30 agents means something is always stuck, crashed, or drifting. Gas Town's 3-tier watchdog system detects, diagnoses, and recovers failed agents automatically.

![Monitoring](/assets/img/diagrams/gastown/gastown-monitoring.svg)

### Component Breakdown

The diagram shows three horizontal tiers. At the top, the Deacon (dark blue) is the background supervisor running continuous patrol cycles across all rigs. The Deacon is Tier 1 - the top-level monitor that watches everything.

The Deacon patrols each rig (green): Rig 1, Rig 2, Rig N. Each rig has its own Witness (pink) at Tier 2 - the per-rig lifecycle manager. The Witness detects stuck agents, triggers recovery actions, and reports up to the Deacon.

Within a rig, the Witness monitors individual polecats. The diagram shows a stuck polecat (red, "STUCK - No Progress") and a healthy polecat (green, "Working"). The bold red edge labeled "detects stuck" shows the Witness identifying the problem.

When a stuck agent is detected, the Witness dispatches Dogs (amber) at Tier 3 - infrastructure workers that handle cleanup and health checks. The Dogs execute a Recovery Action (teal) - restart or reassign - shown as a dashed edge back to the stuck polecat.

The Deacon also branches to two observability outputs: the Activity Feed TUI (purple, `gt feed`) for real-time user-facing monitoring, and OpenTelemetry (orange) for metrics and logs export to external backends.

### Context

This 3-tier design separates concerns: the Deacon patrols globally, the Witness manages per-rig, and the Dogs execute recovery. A problem detected at one tier escalates to the next. This prevents both over-monitoring (one component watching everything) and under-monitoring (no one watching specific rigs).

### Technical Details

The Deacon runs continuous patrol cycles - periodic sweeps across all rigs checking for anomalies. The Witness uses configurable thresholds to determine "stuck" - for example, no commit activity for N minutes. The recovery flow is: nudge (gentle restart), handoff (reassign beads to another polecat), escalate (P0/P1/P2 severity routing).

The Activity Feed TUI (`gt feed`) provides a real-time view of all agent activity with a problems view for stuck detection. The Web Dashboard offers browser-based monitoring. OpenTelemetry integration exports metrics and logs to observability backends like Prometheus, Grafana, or Jaeger.

### Value-Added

Without monitoring, a 30-agent fleet silently degrades. Stuck agents consume resources without producing work. The 3-tier watchdog ensures problems are detected and recovered automatically, keeping the fleet productive. The escalation system (P0/P1/P2) routes different failure classes to appropriate responses - a stuck agent is not an emergency, but a crashed Deacon is.

### Conceptual Understanding

The Deacon is the night watchman making rounds. The Witness is the floor manager watching their department. The Dogs are the maintenance crew called in to fix problems. When the watchman finds a problem, the floor manager investigates, and the maintenance crew executes the fix. The Activity Feed is the security monitor on the wall, and OpenTelemetry is the log book sent to headquarters.

## The Refinery - Bors-Style Merge Queue

Multiple agents produce branches that need merging. Without a queue, this is chaos. The Refinery is Gas Town's per-rig merge queue processor using a Bors-style bisecting algorithm.

![Refinery](/assets/img/diagrams/gastown/gastown-refinery.svg)

### Component Breakdown

The diagram flows left to right. Four agents (red) - Agent A, B, C, D - each produce a feature branch. These branches enter the Refinery Queue (amber), which batches candidate merges together rather than testing each individually.

The batch (blue) runs all merged branches through CI as a single combined test. This is the key efficiency: instead of four separate CI runs, one run tests all four merges stacked together.

The decision diamond (green) asks "CI Pass?" If YES (bold green edge), all MRs in the batch merge to main at once. If NO (bold pink edge), the batch is bisected - split in half - and each half is tested separately to isolate the failing commit.

The failure path flows: Bisect (pink) splits the batch, Test Half (blue) runs the reduced batch, Isolate (red) identifies the problematic commit, and Reject + Notify (orange) sends feedback to the responsible agent (dashed edge back to Agent A).

The Scheduler (purple) attaches from the side as a capacity governor, throttling the queue to prevent API rate limit exhaustion. This is shown as a dashed edge labeled "throttle."

### Context

The bisecting merge queue dramatically reduces CI runs compared to testing each merge individually. If you have 10 candidate merges and one fails, a naive approach runs 10 CI checks. The bisecting approach runs 1 (all together), then bisects to find the failure in roughly log2(N) additional runs. For 10 merges, that is about 4 runs instead of 10.

### Technical Details

The Refinery integrates with Hooks: merged state persists in git, so agents see the updated baseline after merges complete. The Scheduler is a capacity governor that prevents API rate limit exhaustion by throttling agent dispatch - it ensures agents are not all spun up simultaneously. Seance is the session discovery and continuation feature that resumes interrupted agent sessions.

### Value-Added

The merge queue solves the "multiple agents editing the same repo" problem systematically. Instead of manual conflict resolution, the Refinery batches, tests, and bisects automatically. The Scheduler prevents the rate-limit problem by governing capacity. Together, they make parallel merges safe and efficient.

### Conceptual Understanding

The Refinery is an assembly line quality gate. Agents bring their parts (branches). The line batches them together and tests the whole assembly. If it passes, all parts ship. If it fails, the line splits the batch and tests each half to find the defective part, which is sent back to its maker with a note. The Scheduler is the floor manager controlling how many workers are on the line at once to avoid overloading the power supply.

## Federation - The Wasteland Network

Wasteland is Gas Town's federated work coordination network built on DoltHub. It enables multiple Town instances to share work, beads, and convoys across organizational boundaries.

The use case is distributed teams. Imagine two organizations, each running their own Town instance, coordinating on a shared project. Without federation, they would need manual synchronization. With Wasteland, beads can be assigned to agents in federated Towns automatically.

DoltHub serves as the federation layer. It provides versioned data sharing with SQL query access, meaning federated work state is queryable, auditable, and version-controlled. Work routing across Towns uses the same bead/convoy model as local work - a bead assigned to a remote Town flows through the same pipeline as a local one.

This enables elastic capacity. When one Town is overloaded, overflow work can be dispatched to remote Towns with available capacity. The federation handles the routing transparently. A solo developer running a small Town can temporarily borrow capacity from a partner's Town during a crunch.

Wasteland is an advanced feature that most users will not need initially, but it represents Gas Town's ambition: not just coordinating agents within one workspace, but coordinating work across a network of workspaces. It positions Gas Town as infrastructure for federated AI software development at scale.

## Use Cases and Real-World Scenarios

Gas Town fits anywhere you need multiple AI agents working in parallel on code.

**Solo developer scaling.** A single developer runs 5-10 agents on different features simultaneously. The Mayor assigns work, the Refinery merges results, and the developer reviews the output. What used to take a week of sequential coding becomes a day of parallel agent work.

**Team coordination.** A team runs 20-30 agents across a monorepo. The Refinery manages merges so agents do not conflict. The monitoring system catches stuck agents before they waste resources. The team reviews bead completions rather than managing agents directly.

**CI/CD augmentation.** Agents fix failing tests and address code review comments automatically. Beads are created from the CI failure or review comment, assigned to polecats, and the fix flows through the Refinery back to main. The loop from failure to fix to merge is automated.

**Large refactoring.** Molecules define multi-step refactoring workflows executed by agent fleets. A molecule might define "rename module" as: update imports, update tests, update docs, update build config - each step a bead assigned to a different polecat, all coordinated by the Mayor.

**Bug triage.** Beads are created from an issue tracker and assigned to agents by severity. P0 bugs go to the fastest polecats; P2 bugs wait in the queue. The Mayor balances load across available agents.

**Documentation generation.** Agents write docs in parallel - one per module - and the Refinery merges the results. What would be days of sequential writing becomes hours of parallel generation.

**Federated development.** Multiple organizations coordinate via Wasteland. A core team and a community team each run their own Town, sharing beads for issues that cross organizational boundaries.

## Comparison with Alternatives

| Tool | Approach | Multi-Agent | Git-Persistent | Merge Queue | Federation |
|------|----------|-------------|----------------|-------------|------------|
| Gas Town | Workspace manager + orchestration | 20-30 agents | Yes (Hooks) | Yes (Refinery) | Yes (Wasteland) |
| Claude Code (standalone) | Single agent in one repo | No | No | No | No |
| GitHub Copilot Workspace | Cloud-based single session | No | No | No | No |
| Aider | Single agent pair programming | No | Partial | No | No |
| AutoGPT/AgentGPT | Autonomous task agents | Limited | No | No | No |
| CrewAI | Agent framework (library) | Yes | No | No | No |
| OpenDevin/OpenHands | Single autonomous agent | Limited | Partial | No | No |

The key differentiators: Gas Town is a *manager* (not an agent), uses git as the persistence substrate, includes a merge queue, and supports federation. Most alternatives are either single-agent tools or agent frameworks that lack persistence, merge queues, and federation. Gas Town occupies a unique niche: orchestration infrastructure for fleets of existing agents.

## Configuration and Customization

Gas Town is configured through TOML files at multiple levels.

### Town configuration

Global settings live in `~/gt/config.toml`:

```toml
[town]
name = "my-town"
default_runtime = "claude-code"

[scheduler]
max_concurrent_agents = 10
rate_limit_window = "60s"

[monitoring]
deacon_patrol_interval = "30s"
witness_stuck_threshold = "5m"
escalation_p0 = "restart"
escalation_p1 = "reassign"
escalation_p2 = "notify"
```

### Per-rig configuration

Each rig has its own configuration for agent runtimes, Refinery settings, and Witness thresholds:

```toml
[rig]
name = "my-project"
repo = "/path/to/repo"

[rig.refinery]
batch_size = 5
bisect_enabled = true

[rig.witness]
stuck_threshold = "10m"
heartbeat_interval = "3m"

[rig.runtime]
type = "claude-code"
model = "claude-sonnet-4"
api_key = "${CLAUDE_API_KEY}"
```

### Molecule definitions

Molecules are TOML files defining multi-step workflow templates:

```toml
[molecule]
name = "implement-feature"
description = "Research, implement, test, and review a feature"

[[molecule.steps]]
name = "research"
bead_type = "research"

[[molecule.steps]]
name = "implement"
bead_type = "code"
depends_on = "research"

[[molecule.steps]]
name = "test"
bead_type = "test"
depends_on = "implement"

[[molecule.steps]]
name = "review"
bead_type = "review"
depends_on = "test"
```

### OpenTelemetry configuration

Export metrics and logs to observability backends:

```toml
[otel]
enabled = true
endpoint = "http://localhost:4317"
service_name = "gas-town"
export_interval = "10s"
```

## Conclusion and Links

Gas Town solves the multi-agent coordination problem with four key innovations: git-persistent state via Hooks, a 3-tier watchdog monitoring system, a Bors-style bisecting merge queue, and federation through Wasteland. Together, these make running 30+ AI coding agents reliable rather than chaotic.

The paradigm shift is from single-agent coding to fleet-managed AI software development. Instead of one agent in one repo, you run a fleet coordinated by a Mayor, persisted in git, monitored by a Deacon, and merged through a Refinery. This is the infrastructure layer that makes multi-agent AI coding production-ready.

Gas Town is open source under the MIT license, written in Go, and actively developed with 7,655 commits. Whether you are a solo developer scaling to 10 agents or a team coordinating 30, Gas Town provides the orchestration layer to make it work.

### Getting started

- **GitHub repository**: [https://github.com/gastownhall/gastown](https://github.com/gastownhall/gastown)
- **Install via Homebrew**: `brew install gastown`
- **Install via npm**: `npm install -g @gastown/gt`
- **Install from source**: `go install github.com/steveyengge/gastown/cmd/gt@latest`

### Key features recap

- Multi-runtime support: Claude Code, Codex, GitHub Copilot, Gemini, Cursor
- Git worktree-based persistent storage (Hooks)
- Beads integration for git-backed issue tracking
- 3-tier watchdog monitoring (Deacon, Witness, Dogs)
- Bors-style bisecting merge queue (Refinery)
- Federation via DoltHub (Wasteland)
- OpenTelemetry support for metrics and logs
- Web dashboard and TUI activity feed (`gt feed`)
- Docker Compose support
- Shell completions for bash, zsh, and fish

The era of single-agent coding is ending. Gas Town is the workspace manager for the fleet era.
