---
layout: post
title: "Prime Agent: A Self-Improving RLM Harness for Long-Running Coding and Research Work"
description: "Prime Agent is an open-source coding and research agent built around two core abstractions: the Recursive Language Model (RLM), which treats context as variables and tools like recursive subagents as function calls inside a persistent Python REPL, and the Continual Harness, which stores supplemental prompts, memories, skill descriptions, and reusable subagent specifications as durable state that can be refined through small, evidence-backed updates. Prime Agent combines a persistent Python control environment with durable harness state, daemon-backed session continuity, direct agent-to-agent communication, heartbeats, schedules, persistent goals, bounded autonomous mode, and automatic compaction. Released under the MIT License by PrimeIntellect, it is built on top of the pi terminal framework. This post walks through the system architecture, the RLM programming model, the Continual Harness and /refine self-improvement flow, and the long-running agent lifecycle."
date: 2026-09-08
header-img: "img/post-bg.jpg"
permalink: /Prime-Agent-Self-Improving-RLM-Harness-PrimeIntellect/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Prime Agent
  - RLM
  - AI Agents
  - Self-Improving
  - Coding Agent
  - Open Source
  - TypeScript
  - Python
  - PrimeIntellect
author: PyShine
---

## What is Prime Agent

Prime Agent is an open-source coding and research agent for general and long-running work, developed by PrimeIntellect. It is designed around two core abstractions: the **Recursive Language Model (RLM)**, which treats context as variables (prompt-as-a-variable) and tools like recursive subagents as function calls (programmatic tool and sub-agent calling) inside a persistent REPL, and the **Continual Harness**, which stores supplemental prompts, memories, skill descriptions, and reusable subagent specifications as durable state that Prime Agent can refine through small, evidence-backed updates, local to the session by default.

Prime Agent combines a persistent Python control environment with durable harness state, so useful working context and reusable operating patterns can outlive a single chat window. The project is fully open source and released under the MIT License. The code is on GitHub at [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent), and the paper is available at [arXiv:2608.23552](https://arxiv.org/abs/2608.23552). The agent and TUI are built on top of [pi](https://github.com/earendil-works/pi).

## System Architecture

Prime Agent separates terminal presentation, process coordination, agent execution, model-facing Python, and persisted state. The client (TUI or headless) owns rendering and input but does not own execution. A daemon supervisor owns discovery, routing, attachments, worker health, and cross-agent message delivery. Each worker owns one root runtime, its scheduler, kernels, and all descendants below that root. The `AgentSession` owns provider calls, queues, tools, compaction, goals, child lifecycles, and transcript writes. The Python REPL is the model-facing control environment.

![Prime Agent system architecture](/assets/img/diagrams/prime-agent/prime-agent-architecture.svg)

Workers and kernels are separate processes for lifecycle and failure containment, not security sandboxes. They normally run with the same operating-system permissions as the client. Normal interactive sessions use the daemon-backed path; explicit SDK and fallback integrations can run the same `AgentSessionRuntime` in process.

## The RLM Programming Model

The Recursive Language Model treats context as variables and tools as function calls inside a persistent Python REPL. When a user sends a prompt or steer, the AgentSession streams a request to a model provider. The provider responds with either text or a Python tool call. If it is a tool call, the Python kernel executes it. A typed host request returns an authoritative operation to the TypeScript session; ordinary execution returns a result, stdout, or error. The kernel can also spawn child agents via `rlm(...)`, which returns child results programmatically.

![Prime Agent RLM programming model](/assets/img/diagrams/prime-agent/prime-agent-rlm-model.svg)

The design is intentionally programmatic. Everything is a Python call: file operations, shell commands, tool use, subagents, and context management all happen through code. Subagents are built in: `rlm(...)` spawns real child agents for parallel or background work and returns their results programmatically. Skills are executable: skills are importable Python packages, and the built-in skill creator can turn recurring workflows into project or personal skills.

## Continual Harness and /refine

The Continual Harness stores supplemental prompts, memories, skill descriptions, and reusable subagent specifications as durable state. The `/refine` command reviews the current trajectory and can apply small, evidence-backed updates to this supplemental state. It never rewrites the immutable base system prompt, and recorded snapshots support rollback.

![Prime Agent Continual Harness and /refine](/assets/img/diagrams/prime-agent/prime-agent-refine-harness.svg)

The flow is: the agent runs, accumulating a trajectory of transcript and artifacts. When `/refine` is invoked, it reviews the trajectory and proposes a small, focused, evidence-backed update. The update is recorded as a snapshot (which supports rollback) and then applied to the supplemental harness state. The next turn uses the updated harness, while the base system prompt remains immutable. This lets the agent improve its working context and reusable operating patterns over time without rewriting its core instructions.

## Long-Running Agent Lifecycle

Prime Agent is built for long-running work, especially for evaluations in research. Active sessions, Python REPL state, schedules, and subagents keep running when the terminal detaches and can be reattached later. The daemon supervisor manages background continuity.

![Prime Agent long-running agent lifecycle](/assets/img/diagrams/prime-agent/prime-agent-lifecycle.svg)

The lifecycle features include:

- **Daemon-backed continuity:** active sessions, Python REPL state, schedules, and subagents keep running when the terminal detaches and can be reattached later.
- **Direct agent-to-agent communication:** running agents and retained subagents can discover one another, exchange messages, and steer active work without routing everything through the user.
- **Heartbeats and schedules:** `/heartbeat`, `rlm_heartbeat`, and `prime-agent schedule` can re-enter a session periodically or at a specific time.
- **Persistent goals:** `/goal` keeps an objective and its progress active across turns until it is completed, paused, or cleared.
- **Bounded autonomous mode:** `/autonomous` continues within configured turn, token, and time budgets and can run user-defined quality gates. A passed gate checks only what that gate verifies; reaching a limit does not imply task success.
- **Automatic compaction:** context is preserved across turns and terminal sessions through compaction.

## Installation

Install the latest stable release on macOS or Linux:

```bash
curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh
```

The installer downloads a versioned release, verifies its SHA-256 checksum, installs the `prime-agent` command, and can prepare the Python runtime used by the agent.

Start Prime Agent from the repository or directory you want it to work in:

```bash
cd /path/to/project
prime-agent
```

On first launch, run `/login` to choose a subscription or API-key provider. Prime Agent works in the current directory and can run commands and modify files there.

## Useful Commands

```bash
prime-agent agents                   # Browse running, idle, and saved sessions
prime-agent attach <agent>           # Reattach to a running session
prime-agent --resume [path|id]       # Browse sessions or resume one directly
prime-agent status                   # Inspect background service state
prime-agent doctor [--fix]           # Inspect or repair background services
prime-agent update [--force]         # Update Prime Agent
prime-agent shutdown [--force]       # Stop every agent, worker, and background service
```

## Safety

Prime Agent executes model-generated Python and project commands with your user permissions. Its worker and kernel processes improve lifecycle isolation and recovery; they are not a security sandbox. Review changes and use trusted repositories, instructions, skills, and extensions only. Run untrusted code or instructions in an external sandbox or restricted environment.

## Documentation

- [Quickstart](https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/quickstart.md) — install, authenticate, and run a first session
- [Usage and CLI reference](https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/usage.md) — commands, sessions, autonomous limits, and output modes
- [Long-running and background agents](https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/long-running-agents.md) — detach and reattach, goals, heartbeats, and schedules
- [RLM programming model](https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/rlm.md) — the persistent Python REPL, subagents, skills, and the trust model
- [Architecture overview](https://github.com/PrimeIntellect-ai/prime-agent/blob/main/packages/coding-agent/docs/architecture.md) — daemon, worker, kernel, and persistence boundaries

## Related Projects

- [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) — verification tools for Prime Agent
- [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) — reinforcement learning framework
- [pi](https://github.com/earendil-works/pi) — the terminal framework Prime Agent's agent and TUI are built on

## Conclusion

Prime Agent is a pragmatic answer to the problem of context loss in long-running coding and research work. By pairing the Recursive Language Model with the Continual Harness, it lets useful working context and reusable operating patterns outlive a single chat window, and the `/refine` command lets the agent improve its supplemental state through small, evidence-backed updates without rewriting its core instructions. The daemon-backed continuity, direct agent-to-agent communication, heartbeats, schedules, persistent goals, and bounded autonomous mode make it suited for work that takes minutes to hours rather than a single turn. The source is on GitHub at [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent), and the paper is at [arXiv:2608.23552](https://arxiv.org/abs/2608.23552).
