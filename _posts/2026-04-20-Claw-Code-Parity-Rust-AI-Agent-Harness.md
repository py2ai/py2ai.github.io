---
layout: post
title: "Claw Code Parity: The Rust AI Agent Harness Built Autonomously by Claws"
description: "An in-depth look at claw-code-parity — a Rust port of the Claude Code CLI system that is autonomously maintained by AI agents, featuring a 9-crate architecture, worker boot state machines, recovery recipes, and a three-part autonomous development ecosystem."
date: 2026-04-20
header-img: "assets/img/diagrams/claw-code-parity/claw-code-parity-architecture.svg"
permalink: /Claw-Code-Parity-Rust-AI-Agent-Harness/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [Rust, AI-Agent, Claude-Code, Autonomous-Development, Open-Source]
author: PyShine
---

## Introduction

Claw Code Parity is a Rust port of the Claude Code CLI system -- and it is autonomously maintained by lobsters and claws, not by human hands. The project, hosted at [ultraworkers/claw-code-parity](https://github.com/ultraworkers/claw-code-parity), set out to prove a provocative thesis: an open coding harness can be built autonomously, in public, at high velocity, and the result can be production-grade. Within two hours of its launch, the repository reached 50K stars, signaling that the developer community recognized the significance of what was being attempted.

The core premise of claw-code-parity is straightforward but ambitious. Rather than having human engineers manually port the TypeScript-based Claude Code CLI to Rust, the project deploys AI agents -- referred to as "claws" -- to do the porting work. Each agent operates in its own lane, making changes, running tests, and merging code without human intervention. The project is not affiliated with Anthropic; it is an independent effort to demonstrate that autonomous software development is not a future possibility but a present reality.

What makes claw-code-parity particularly interesting from an engineering perspective is its architecture. The codebase is organized into nine Rust crates, each with a clear responsibility boundary. It implements a typestate pattern for worker boot sequences, a recovery recipe system that handles failures before escalating to humans, and a policy engine that codifies operational decisions as executable rules. These are not toy implementations -- they are designed to handle the real-world complexity of running AI agents at scale, where failures are frequent, state machines drift, and the only sustainable approach is to make the system self-correcting.

This post walks through the architecture, the worker lifecycle, the autonomous development ecosystem, and the permission and policy engine that together make claw-code-parity a compelling case study in autonomous software engineering.

## The Rust Crate Architecture

![Claw Code Parity Rust Crate Architecture](/assets/img/diagrams/claw-code-parity/claw-code-parity-architecture.svg)

The claw-code-parity workspace is organized into nine crates, each serving a distinct role in the system. At the top level, `rusty-claude-cli` is the main CLI binary -- invoked as `claw` -- that ties everything together. It depends on the `runtime` crate, which is the core of the system. The runtime crate contains session management, configuration handling, the permission system, MCP (Model Context Protocol) integration, and prompt construction. If you need to understand how the system works end-to-end, the runtime crate is where you start.

The `api` crate handles the Anthropic API client, including SSE (Server-Sent Events) streaming for real-time token delivery. It abstracts away the HTTP layer and provides a clean interface for sending prompts and receiving streamed responses. The `commands` crate provides the slash-command registry, allowing users to interact with the CLI through commands like `/help`, `/status`, and `/config`. The `tools` crate is the largest, implementing over 40 built-in tools that the agent can invoke: `bash`, `read_file`, `write_file`, `edit_file`, `glob_search`, `grep_search`, `WebFetch`, `WebSearch`, `TodoWrite`, and many more. Each tool is registered with a name, description, and schema, and the runtime dispatches tool calls through this registry.

The `plugins` crate manages the plugin registry and hook wiring. Plugins can extend the system by registering new tools, hooks, or command handlers. The `telemetry` crate handles session tracing and usage telemetry, providing observability into what each agent session does. The `compat-harness` crate extracts TypeScript manifests from the original Claude Code codebase, enabling the Rust port to maintain behavioral parity with the TypeScript implementation. Finally, the `mock-anthropic-service` crate provides a deterministic mock `/v1/messages` endpoint for testing, ensuring that tests are reproducible and do not depend on external API availability.

The dependency graph flows downward: `rusty-claude-cli` depends on `runtime`, which depends on `api`, `commands`, `tools`, `plugins`, and `telemetry`. The `compat-harness` and `mock-anthropic-service` crates are used during development and testing. The workspace uses Cargo's `resolver = 2` for dependency resolution and enforces `#![deny(unsafe_code)]` across all crates, meaning no unsafe blocks are permitted anywhere in the codebase. This is a deliberate choice: when AI agents are writing code autonomously, eliminating an entire class of memory-safety bugs through a crate-level deny directive is a pragmatic safety net.

## Worker Boot State Machine and Recovery

![Claw Code Parity Worker Lifecycle](/assets/img/diagrams/claw-code-parity/claw-code-parity-worker-lifecycle.svg)

The worker boot process in claw-code-parity follows a typestate pattern, where the state of a worker is encoded in its type rather than in runtime flags. The `WorkerStatus` enum defines the lifecycle of a worker from spawn to completion:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    Spawning,
    TrustRequired,
    ReadyForPrompt,
    Running,
    Finished,
    Failed,
}
```

When a worker is first created, it enters the `Spawning` state. During this phase, the runtime allocates resources, initializes the MCP connection, and loads the plugin registry. Once initialization is complete but before the worker can accept prompts, it transitions to `TrustRequired`. This state exists because the first action a new worker must take is to accept a trust prompt -- a confirmation that the worker is authorized to operate in the current context. Only after the trust prompt is resolved does the worker transition to `ReadyForPrompt`, the state from which prompts can be dispatched. This typestate pattern ensures that prompts are never sent to a worker that is not ready to receive them, eliminating an entire category of race conditions.

When a prompt is dispatched, the worker transitions to `Running`. Upon completion, it moves to `Finished`. If something goes wrong at any point, the worker transitions to `Failed`. The `WorkerFailureKind` enum categorizes failures into four types: `TrustGate` (the trust prompt was not resolved), `PromptDelivery` (the prompt could not be delivered to the agent), `Protocol` (a protocol-level error occurred during execution), and `Provider` (the underlying API provider failed).

What sets claw-code-parity apart from a naive retry system is its Recovery Recipe mechanism. Rather than simply retrying on failure, the system maps each failure scenario to a specific recovery recipe. There are seven recipes, each targeting a particular failure mode:

- `TrustPromptUnresolved` maps to `AcceptTrustPrompt` -- re-present the trust prompt for resolution.
- `PromptMisdelivery` maps to `RedirectPromptToAgent` -- route the prompt through an alternative delivery channel.
- `StaleBranch` maps to `RebaseBranch` -- the worker's branch has fallen behind main; rebase it.
- `CompileRedCrossCrate` maps to `CleanBuild` -- a crate failed to compile; clean and rebuild.
- `McpHandshakeFailure` maps to `RetryMcpHandshake` -- the MCP connection failed to establish; retry with a timeout.
- `PartialPluginStartup` maps to `RestartPlugin` -- a plugin failed to start; restart it by name.
- `ProviderFailure` maps to `RestartWorker` -- the API provider failed; restart the entire worker.

Each recovery recipe is attempted up to a configurable maximum number of times. If the maximum is exceeded, the system escalates to a human operator. The `WorkerRegistry` maintains an in-memory map of all active workers, their current states, and their recovery attempt counts, providing a single source of truth for the orchestrator.

## The Autonomous Development Ecosystem

![Claw Code Parity Autonomous Ecosystem](/assets/img/diagrams/claw-code-parity/claw-code-parity-autonomous-ecosystem.svg)

Claw-code-parity does not operate in isolation. It is part of a three-component autonomous development ecosystem: OmX, clawhip, and OmO. Together, these three systems form a closed loop that can develop, review, and merge code without human intervention -- though humans remain in the loop through Discord notifications and manual overrides.

OmX (oh-my-codex) is the workflow layer. It provides two primary modes. In `$team` mode, multiple agents work in coordinated parallel review, each examining different aspects of a change set. In `$ralph` mode, a single agent executes a task with persistent verification -- it keeps running, checking, and fixing until the task is complete. OmX is responsible for breaking down objectives into task packets, assigning them to workers, and collecting results.

Clawhip is the event and notification router. Its critical function is to keep monitoring outside the agent context window. When an agent is deep in a coding session, its context window fills up, and it may lose track of external events -- a branch going stale, a CI pipeline failing, or a dependency being updated. Clawhip watches for these events and routes them to the appropriate handler, ensuring that no important signal is lost simply because the agent was busy.

OmO (oh-my-openagent) is the multi-agent coordination layer. It handles planning, task handoffs between agents, and disagreement resolution. When two agents produce conflicting changes, OmO resolves the conflict using a configurable strategy -- merging, selecting the higher-confidence result, or escalating to a human. OmO also manages the overall workflow, deciding which tasks to run in parallel and which must be sequential.

The full loop works as follows: a human issues a request through Discord. OmX picks up the request, decomposes it into tasks, and assigns them to workers via OmO. Workers execute their tasks, producing LaneEvents -- typed event objects that represent state changes in the system. Clawhip routes these events back to Discord, closing the loop. The `LaneEventName` enum defines over 15 typed events:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaneEventName {
    #[serde(rename = "lane.started")]
    Started,
    #[serde(rename = "lane.ready")]
    Ready,
    #[serde(rename = "lane.green")]
    Green,
    #[serde(rename = "lane.failed")]
    Failed,
    #[serde(rename = "branch.stale_against_main")]
    BranchStaleAgainstMain,
    // ... 15+ more event types
}
```

These typed events replace the fragile approach of scraping tmux panes for status information. Each event carries structured data, making it machine-parseable and enabling automated responses. The philosophical insight that emerges from this architecture is worth stating directly: when agent systems can rebuild a codebase in hours, the scarce resource becomes architectural clarity, task decomposition, judgment, taste, and conviction about what is worth building. The code is no longer the bottleneck -- the decisions about what code to write are.

## Permission and Policy Engine

![Claw Code Parity Permission Policy Flow](/assets/img/diagrams/claw-code-parity/claw-code-parity-permission-policy.svg)

Claw-code-parity implements a dual-flow system for controlling what agents can do: the Permission flow and the Policy flow. These are separate concerns that operate at different levels of abstraction, and understanding both is essential to understanding how the system achieves autonomous operation without sacrificing safety.

The Permission flow governs tool invocations. When an agent attempts to use a tool -- say, `write_file` or `bash` -- the request passes through `PermissionPolicy.authorize()`. This method first checks hook overrides. Hooks are user-configurable rules that can short-circuit the permission flow in three ways: a hook deny blocks the tool invocation immediately, bypassing all further checks. A hook ask forces an interactive prompt even when the current mode would otherwise allow the action. A hook allow permits the action but still respects ask rules as a safety net -- this is important because it means that even if a hook says "allow this," any ask-level rules will still trigger a prompt.

If no hook overrides apply, the system falls through to the Allow/Deny/Ask rules. These rules use pattern matching with three match types: `Any` (matches everything), `Exact` (matches an exact string), and `Prefix` (matches a string prefix). For example, a rule might allow read access to any file (`Any`), deny write access to `/etc/` (`Prefix`), or ask before executing `rm` (`Exact`). If the rules produce an `Ask` result, the `PermissionPrompter` presents an interactive prompt to the user, and the result is a `PermissionOutcome` of `Allow` or `Deny`. The system supports three permission modes: `read-only` (no write operations permitted), `workspace-write` (write operations permitted within the workspace), and `danger-full-access` (all operations permitted). The mode sets the default behavior, but individual rules can override it.

The Policy flow operates at a higher level. It governs what actions the system takes in response to lane events. The `PolicyEngine` evaluates a `LaneContext` -- which includes the current state of the lane, the worker, and any relevant events -- against a set of `PolicyCondition` rules. These conditions can be combined using `And` and `Or` combinators, and include specific checks like `GreenAt` (the lane is green at a specific commit), `StaleBranch` (the branch is behind main), `StartupBlocked` (the worker failed to start), and `LaneCompleted` (the lane has finished its task).

When conditions match, the engine produces `PolicyAction` directives: `MergeToDev` (merge the lane's changes to the dev branch), `MergeForward` (merge changes to the next branch in the chain), `RecoverOnce` (attempt a single recovery using the recovery recipe system), `Escalate` (escalate to a human), `CloseoutLane` (close the lane and mark it as complete), and `Chain` (execute multiple actions in sequence). The `RecoveryStep` enum provides the concrete actions:

```rust
pub enum RecoveryStep {
    AcceptTrustPrompt,
    RedirectPromptToAgent,
    RebaseBranch,
    CleanBuild,
    RetryMcpHandshake { timeout: u64 },
    RestartPlugin { name: String },
    RestartWorker,
    EscalateToHuman { reason: String },
}
```

The key insight is that the PolicyEngine enables autonomous operation by codifying operational decisions as executable rules. Rather than requiring a human to decide what to do when a branch goes stale or a worker fails, the system applies a deterministic policy. Humans set the policy; the system executes it. This is what makes autonomous development possible at scale -- not the absence of human judgment, but the encoding of that judgment into the system itself.

## The 9-Lane Parity Checkpoint

The claw-code-parity project uses a lane-based development model where each lane is an independent workstream that produces a specific set of changes. The project reached its parity checkpoint when all nine lanes were merged, each representing a major subsystem of the Rust port:

- **Lane 1: Bash validation** -- the foundational lane that established the bash tool and its validation pipeline.
- **Lane 2: CI fix (sandbox probe)** -- fixed CI issues and implemented the sandbox probe that detects whether the CLI is running in a sandboxed environment.
- **Lane 3: File-tool edge cases** -- added 744 lines of code handling edge cases in file tools (read, write, edit), including symlink resolution, binary file detection, and permission errors.
- **Lane 4: TaskRegistry** -- implemented the task registry (335 LOC) that tracks all running tasks and their states.
- **Lane 5: Task wiring / tools dispatch** -- connected the task registry to the tools dispatch system, enabling tools to be invoked through the task pipeline.
- **Lane 6: Team+Cron registries** -- added the team registry for multi-agent coordination and the cron registry for scheduled tasks (363 LOC).
- **Lane 7: MCP lifecycle bridge** -- implemented the MCP lifecycle bridge (406 LOC) that manages the full lifecycle of MCP connections, from handshake to teardown.
- **Lane 8: LSP client dispatch** -- added the Language Server Protocol client dispatch (438 LOC), enabling the CLI to interact with language servers for code intelligence.
- **Lane 9: Permission enforcement** -- implemented the permission enforcement system (340 LOC) that gates all tool invocations through the policy engine.

Each lane is described by a `TaskPacket` that specifies the objective, scope, repository, branch policy, acceptance tests, commit policy, reporting contract, and escalation policy:

```rust
pub struct TaskPacket {
    pub objective: String,
    pub scope: String,
    pub repo: String,
    pub branch_policy: String,
    pub acceptance_tests: Vec<String>,
    pub commit_policy: String,
    pub reporting_contract: String,
    pub escalation_policy: String,
}
```

The `TaskPacket` is the contract between the orchestrator and the worker. It tells the worker exactly what to do, what constitutes success, and what to do if things go wrong. This structured approach to task definition is what enables autonomous agents to work independently -- the task packet removes ambiguity and provides a clear definition of done.

## Getting Started

To build and run claw-code-parity, you need a recent Rust toolchain (1.75 or later). The following commands will clone the repository, build the workspace, and run the CLI:

```bash
# Clone the repository
git clone https://github.com/ultraworkers/claw-code-parity.git
cd claw-code-parity/rust

# Build the workspace
cargo build --workspace

# Run the CLI
cargo run --bin claw

# One-shot prompt mode
cargo run --bin claw -- prompt "Explain this codebase"

# Run mock parity tests
cargo test --workspace
```

Authentication uses an OAuth login with PKCE (Proof Key for Code Exchange) flow, which avoids storing API keys on disk. The CLI supports model aliases for convenience: `opus` maps to the most capable model, `sonnet` to the balanced model, and `haiku` to the fast model. Sessions are persisted in JSONL format, enabling resumption and audit logging. Each session records the full conversation history, tool invocations, and their results, providing a complete audit trail of what the agent did and why.

## Conclusion

Claw-code-parity represents a significant milestone in autonomous software development. It is not just a Rust port of an existing tool -- it is a demonstration that an open coding harness can be built, maintained, and extended by AI agents operating in public. The architecture reflects this reality: it is "clawable" in the sense that every component is designed for machine-first operation. Typed events replace ad-hoc logging. State machines replace implicit state. Recovery recipes replace manual intervention. Policy engines replace operational runbooks.

The broader implication is that autonomous software development is not a future possibility but a present reality. The nine lanes that reached parity were not coded by human hands -- they were coded by claws, each working in its own branch, each subject to the same permission and policy engine that governs human operation. The system does not need to be perfect; it needs to be self-correcting, and that is exactly what the recovery recipe system and policy engine provide.

For developers interested in the intersection of AI agents and systems programming, claw-code-parity offers a concrete, production-grade reference implementation. The code is open source, the architecture is documented, and the development process is transparent. Whether you are building your own agent harness or evaluating the state of autonomous development, the repository is worth studying: [https://github.com/ultraworkers/claw-code-parity](https://github.com/ultraworkers/claw-code-parity)