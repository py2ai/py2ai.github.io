---
layout: post
title: "SwarmForge: Disciplined AI Agent Orchestration with tmux"
description: "SwarmForge is a tmux-based AI agent orchestration platform by Uncle Bob that turns swarms of AI agents into reliable software engineers using git worktrees, layered constitutions, and file-based handoffs."
date: 2026-05-27
header-img: "img/post-bg.jpg"
permalink: /SwarmForge-Disciplined-AI-Agent-Orchestration-tmux/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, Open Source]
tags: [SwarmForge, Uncle Bob, AI agent orchestration, tmux, multi-agent coordination, Clean Code, TDD, mutation testing, git worktrees, AI coding agents]
keywords: "tmux-based AI agent orchestration, how to orchestrate multiple AI agents with tmux, SwarmForge tutorial Uncle Bob, multi-agent AI coding workflow, git worktree AI agent isolation, AI agent constitution layer, tmux swarm coordination, SwarmForge installation guide, disciplined AI agent orchestration, file-based AI agent handoffs"
author: "PyShine"
---

## Introduction

Tmux-based AI agent orchestration has emerged as a practical solution for coordinating multiple AI coding agents without the chaos of unmanaged concurrent workflows. When several AI agents work on the same codebase simultaneously, they overwrite each other's files, lack structured communication channels, and produce code that ignores engineering standards. SwarmForge, created by Robert C. Martin (Uncle Bob) and Justin Martin, provides a disciplined orchestration platform that uses tmux sessions, git worktrees, and layered constitutions to turn swarms of AI agents into reliable, professional software engineers. Built as a pure shell project with no runtime dependencies beyond tmux and git, SwarmForge enforces Clean Code principles through its constitution system -- requiring TDD, mutation testing, CRAP analysis, and Gherkin acceptance testing from every agent in the swarm.

## How It Works

SwarmForge operates through a clear orchestration pipeline that begins with a single command and ends with a fully coordinated multi-agent environment. The entry point is the `swarm` command, a thin zsh wrapper that delegates to `swarmforge.sh`, the 550-line orchestrator script. This script parses the `swarmforge/swarmforge.conf` configuration file, which defines the swarm topology with lines of the form `window <role> <agent> <worktree>`. For each configured role, the orchestrator creates a git worktree under `.worktrees/`, launches a tmux session, opens a macOS Terminal window via AppleScript (osascript), and starts the specified AI backend (claude or codex) in the assigned worktree.

The inter-agent communication system relies on `notify-agent.sh`, a helper script generated in the project-local `swarmtools/` directory during startup. When one agent needs to send a message to another, it writes the complete handoff message to a temporary file and invokes `notify-agent.sh <target-role> --file <message-file>`. The notify script resolves the target role to its tmux session name using the `.swarmforge/sessions.tsv` file, then injects the message as keystrokes into that session using `tmux send-keys`. If the target agent is busy, incoming messages are saved as files in a local `pending-messages/` directory with priority-prefixed filenames like `PP-YYYYMMDD-HHMMSS-source.txt`, where `PP` is a two-digit priority (00 for architect handoffs, 50 for normal messages).

The watchdog system, implemented in `swarm-window-watchdog.sh`, continuously monitors all Terminal windows. If a non-cleanup window is closed accidentally, the watchdog reopens it attached to the same tmux session. If the cleanup window (the first window listed in the config) is closed, the watchdog triggers `swarm-cleanup.sh`, which kills all tmux sessions and closes all remaining Terminal windows, providing a clean shutdown mechanism.

![SwarmForge Architecture](/assets/img/diagrams/swarm-forge/swarm-forge-architecture.svg)

The architecture diagram above illustrates the complete SwarmForge orchestration flow from configuration input to agent execution and lifecycle management. At the top, the green `swarmforge.conf` node represents the input that defines the swarm topology -- each line specifies a window role, the AI backend to use (claude or codex), and the git worktree assignment. The blue `swarmforge.sh` node is the central orchestrator, a 550-line zsh script that parses the configuration, validates role prompts, creates git worktrees, and coordinates the entire startup sequence. The teal tmux node represents the session manager, which creates one isolated terminal session per configured role. The purple Terminal + osascript node shows the macOS window management layer, where AppleScript opens a separate Terminal window for each role and attaches it to the corresponding tmux session. The orange claude and codex nodes represent the AI backends, selected per-role from the configuration file -- each agent runs in its own tmux session with its own system prompt. The coral worktree nodes (`.worktrees/coder`, `.worktrees/refactorer`, `.worktrees/architect`) show the git worktree isolation that prevents agents from overwriting each other's changes. The amber `notify-agent.sh` node represents the inter-agent communication layer, which sends keystrokes to tmux sessions for real-time message delivery. The pink `pending-messages/` node shows the priority-based message queue that ensures no handoff is lost when an agent is busy. The red watchdog diamond monitors Terminal windows and reopens closed ones, while the gray `swarm-cleanup.sh` node terminates all sessions when the cleanup window closes. This architecture effectively functions as a tmux-based operating system for AI agents, where each agent has its own isolated workspace, its own communication channel, and its own lifecycle managed by the watchdog system.

> **Key Insight:** SwarmForge solves the fundamental coordination problem in multi-agent AI workflows by giving each agent its own git worktree and a structured communication protocol. Without worktree isolation, agents would overwrite each other's changes; without file-based handoffs, agents would have no disciplined way to delegate work. The constitution layer ensures every agent follows the same engineering standards -- TDD, mutation testing, and CRAP analysis are not optional.

## Key Features

| Feature | Description |
|---------|-------------|
| Config-Driven Topology | The swarm shape comes from `swarmforge/swarmforge.conf`, not hardcoded shell variables. Define as many roles as your project needs. |
| Project-Local Roles | Each role is defined by `swarmforge/<role>.prompt` in the working tree being orchestrated, keeping agent instructions close to the code. |
| Layered Constitution | `swarmforge/constitution.prompt` delegates to subordinate files with explicit precedence: project > engineering > workflow. |
| Backend Selection Per Role | A role can launch `claude` or `codex` as its AI backend, allowing mixed backends within a single swarm. |
| Observable Swarm | One Terminal window per role opens automatically, letting you watch every agent think and work in real time. |
| Self-Hosted and Lightweight | Runs locally in tmux and Terminal with minimal machinery. No cloud dependencies, no Docker containers. |
| Git Worktree Isolation | Each role works in its own worktree under `.worktrees/`, preventing file conflicts between concurrent agents. |
| Message Queueing with Priority | Busy agents queue incoming messages in `pending-messages/` with priority-prefixed filenames, ensuring no handoff is lost. |

![SwarmForge Features](/assets/img/diagrams/swarm-forge/swarm-forge-features.svg)

The features diagram above presents SwarmForge's eight core capabilities in a hub-and-radial layout, with the blue center node representing the platform itself and each colored branch representing a distinct feature. The green Config-Driven Topology branch highlights that the swarm shape is defined entirely by `swarmforge.conf`, allowing projects to choose their own role composition instead of being locked to a fixed set. The teal Project-Local Roles branch emphasizes that each role prompt lives in the working tree, making agent instructions version-controlled and project-specific. The purple Layered Constitution branch shows the three-level precedence system -- `project.prompt` (highest priority), `engineering.prompt` (TDD, mutation testing, CRAP analysis), and `workflow.prompt` (handoff protocols, message queueing) -- that separates concerns without forcing everything into one large prompt. The orange Backend Selection branch illustrates the per-role choice between `claude` and `codex`, enabling mixed-backend swarms where different roles use different AI providers. The coral Observable Swarm branch represents the real-time visibility provided by one Terminal window per role, where developers can watch agents reason, write code, and communicate. The amber Self-Hosted and Lightweight branch underscores the zero-dependency philosophy -- only tmux, git, and an AI backend are required. The pink Git Worktree Isolation branch shows how each role gets its own worktree under `.worktrees/`, preventing the file conflicts that plague unmanaged multi-agent workflows. The red Message Queueing branch depicts the priority-based file queueing system in `pending-messages/`, where messages are saved with priority prefixes (00 for architect handoffs, 50 for normal) and processed in sorted order after the current job completes. Together, these features form a cohesive discipline system that transforms raw AI coding speed into reliable, maintainable engineering output.

> **Amazing:** The entire SwarmForge platform is a 550-line zsh script with zero runtime dependencies. No npm install, no pip install, no Docker containers -- just tmux, git, and your chosen AI backend. The swarm topology is defined in a simple four-column config file, and the constitution system enforces Clean Code principles that Uncle Bob has advocated for decades, now applied to AI agents instead of human developers.

## Installation

SwarmForge requires macOS with Terminal, tmux, git, and at least one AI backend installed.

**Prerequisites:**

- macOS (Terminal + osascript for window management)
- tmux (session manager)
- git (worktree isolation and version control)
- An AI backend: `claude` (Claude Code) or `codex` (OpenAI Codex CLI)

**Install SwarmForge into your project directory:**

```bash
curl -L https://github.com/unclebob/swarm-forge/archive/refs/heads/main.tar.gz | tar -xz --strip-components=1
```

This command pulls the SwarmForge repository contents directly into your current working directory without creating a git remote. The key files you need are `swarm`, `swarmforge.sh`, `swarm-cleanup.sh`, `swarm-window-watchdog.sh`, and `swarmlog.sh`.

**Add the SwarmForge scripts to your PATH:**

```bash
export PATH="/path/to/your/project:$PATH"
```

Or add `swarmforge.sh` to your shell PATH before startup, as recommended in the project documentation.

## Usage

### Configuration

Create a `swarmforge/` directory in your target working directory with the following structure:

```text
swarmforge/
  swarmforge.conf
  constitution.prompt
  constitution/
    project.prompt
    engineering.prompt
    workflow.prompt
  specifier.prompt
  coder.prompt
  refactorer.prompt
  architect.prompt
```

Define your swarm topology in `swarmforge/swarmforge.conf`. Each line specifies a window with a role, an AI backend, and a worktree assignment:

```conf
window specifier codex master
window coder codex coder
window refactorer codex refactorer
window architect codex architect
```

The first window listed is the cleanup window. When its Terminal window closes, SwarmForge shuts down all sessions. If a role uses `master` as its worktree, that agent runs in the main working directory on the `master` branch. Otherwise, SwarmForge creates a git worktree under `.worktrees/<worktree>`.

### Constitution System

The `constitution.prompt` file is the entry point for the constitution layer. It defines precedence and directs agents to read subordinate files in order:

```text
1. swarmforge/constitution/project.prompt
2. swarmforge/constitution/engineering.prompt
3. swarmforge/constitution/workflow.prompt
```

If two subordinate files conflict, the earlier file wins. This separation lets you keep project-specific rules (language, tooling, naming conventions) separate from engineering rules (TDD, mutation testing, CRAP analysis) and workflow rules (worktree isolation, handoff format, message queueing).

### Running the Swarm

Just type `swarm`:

```bash
swarm
```

The windows should all pop up. Each Terminal window attaches to its tmux session, and the AI backend for each role starts in its assigned worktree.

### Inter-Agent Handoffs

Agents communicate through the file-based handoff system. To send a message to another agent:

```bash
./swarmtools/notify-agent.sh <target-role> --file ./tmp/<target-role>-handoff.txt
```

Every handoff message must start with: `Re-read your role and constitution.` After the opening line, include exactly these fields: sender role, specifier handoff name, branch name, and commit hash. Do not tell the receiving role how to do its job -- the normal request is: `Apply your own role rules to this state.`

When an agent is busy and receives a message, it saves the message as a file in its local `pending-messages/` directory with this naming format:

```text
pending-messages/PP-YYYYMMDD-HHMMSS-source.txt
```

Priority `00` is used for architect handoffs to coder or refactorer. Priority `50` is the default for normal messages. After the current job completes, queued messages are processed in sorted filename order.

![SwarmForge Workflow](/assets/img/diagrams/swarm-forge/swarm-forge-workflow.svg)

The workflow diagram above shows the step-by-step process for using SwarmForge, from initial setup through agent execution to final cleanup. Step 1 (green) is creating the `swarmforge/` directory in the target project, which holds all configuration and role prompt files. Step 2 (blue) is writing the `swarmforge.conf` topology definition and the individual `<role>.prompt` files that instruct each agent on its responsibilities. Step 3 (teal) is running the `swarm` command, which delegates to `swarmforge.sh` to begin the orchestration. Step 4 (purple) shows tmux sessions launching and Terminal windows opening via osascript, one per configured role. Step 5 (orange) depicts each AI agent starting in its isolated git worktree under `.worktrees/`, where it can work independently without conflicting with other agents. Step 6 (coral) illustrates the file-based handoff system, where `notify-agent.sh` sends messages between agents via tmux keystroke injection, and busy agents queue incoming messages in `pending-messages/` with priority-prefixed filenames. Step 7 (amber) represents the user observing all agent activity in real time through the Terminal windows that SwarmForge opened. The pink decision diamond asks "Work complete?" -- if not, agents continue the handoff cycle, delegating work back and forth through the structured communication protocol. When work is complete, the gray `swarm-cleanup.sh` node terminates all tmux sessions and closes all Terminal windows. This workflow creates a tight feedback loop where the human operator maintains full visibility into every agent action, can intervene at any time, and can shut down the entire swarm cleanly by closing the cleanup window.

> **Takeaway:** With a single `swarm` command, an entire multi-agent orchestration environment springs to life -- Terminal windows open for each role, tmux sessions launch in isolated git worktrees, and agents begin communicating through a file-based handoff system with priority queueing. The observable nature of tmux means you can watch every agent think and work in real time.

## Conclusion

SwarmForge brings a discipline-first philosophy to multi-agent AI coding workflows. By combining git worktree isolation, file-based handoffs with priority queueing, a layered constitution system, and real-time observability through tmux, it transforms the chaos of concurrent AI agents into a coordinated engineering process. The constitution layer ensures that every agent follows TDD, mutation testing, CRAP analysis, and Gherkin acceptance testing -- the same Clean Code practices that Uncle Bob has taught for decades, now enforced at the prompt level for AI agents. The self-dogfooding requirement, where SwarmForge uses itself to build its own features, demonstrates that the platform is reliable enough for production use, not just experimentation.

> **Important:** SwarmForge's self-dogfooding requirement -- that SwarmForge should use itself to build its own features -- is more than a clever meta-exercise. It is a statement of confidence in the discipline-first approach: if the orchestration platform is not reliable enough to build itself, it is not reliable enough for your project. The constitution layer ensures that this reliability is enforced at the prompt level, not just the process level.

**Links:**
- GitHub: [https://github.com/unclebob/swarm-forge](https://github.com/unclebob/swarm-forge)