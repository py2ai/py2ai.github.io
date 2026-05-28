---
layout: post
title: "Oh-My-Pi: AI Coding Agent for Terminal with Hash-Anchored Edits"
description: "Learn how oh-my-pi brings AI-powered coding to the terminal with hash-anchored edits, LSP integration, Python support, browser tools, and subagent orchestration. Installation guide, architecture, and real-world examples."
date: 2026-05-28
header-img: "img/post-bg.jpg"
permalink: /Oh-My-Pi-AI-Coding-Agent-Terminal/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, TypeScript, Developer Tools]
tags: [oh-my-pi, AI coding agent, terminal, hash-anchored edits, LSP, Python, browser automation, subagents, Claude Code, developer tools]
keywords: "oh-my-pi AI coding agent tutorial, terminal AI coding assistant, hash-anchored edits AI, how to use oh-my-pi, oh-my-pi vs Claude Code, AI agent terminal tool, LSP integration AI coding, Python AI agent terminal, browser automation AI agent, oh-my-pi installation guide"
author: "PyShine"
---

## What Is Oh-My-Pi?

Oh-My-Pi (omp) is a terminal-based AI coding agent that takes the standard "LLM in the loop" pattern and adds four distinguishing primitives: **hash-anchored edits** that prevent concurrent agent runs from clobbering each other, **LSP integration** so edits are symbol-aware rather than blind text replacement, a **controllable browser surface** for live documentation and web-testing, and **subagent orchestration** that fans out work across isolated workers with typed results back to the parent.

Created by [Can Boeluek](https://github.com/can1357) as a fork of [Mario Zechner](https://github.com/mariozechner)'s [Pi](https://github.com/badlogic/pi-mono), omp extends the original with a batteries-included coding workflow. The project has earned over 7,800 stars on GitHub and is growing at more than 2,500 stars per week -- a clear signal that developers are finding real value in its approach.

At its core, omp is built with **TypeScript** (running on the Bun runtime) and **Rust** (approximately 27,000 lines for the native engine). It provides unified access to **40+ AI model providers**, **32 built-in tools**, **13 LSP operations**, and **27 DAP operations** -- all from a single terminal interface.

> "The most capable agent surface that ships. Continuously tuned by real-world use -- complete out of the box, open all the way down." -- oh-my-pi README

## Architecture Overview

Oh-My-Pi's architecture is organized into four distinct layers that work together to deliver a seamless coding experience from the terminal.

![Oh-My-Pi Architecture]({{ site.baseurl }}/assets/img/diagrams/oh-my-pi/oh-my-pi-architecture.svg)

The diagram above illustrates the full architecture of oh-my-pi. At the top, four entry points -- Terminal TUI, Node SDK, RPC Mode, and ACP Mode -- all funnel into the Agent Core, which manages sessions, tool calling, and state. The Agent Core dispatches work to the Tool Surface, which is divided into four categories: Files and Search (read, write, edit, ast_edit, search, find, ast_grep), Runtime (bash, eval, recipe, ssh), Code Intelligence (lsp, debug), and Coordination (task, irc, todo_write, ask, job). Below the Tool Surface sits the Rust Native Engine, which provides in-process implementations of shell (brush), grep, AST parsing, and structural summarization -- eliminating fork/exec overhead on the hot path. On the right, the Agent Core connects to 40+ providers across three tiers (Frontier APIs, Coding Plans, and Self-Hosted), and at the bottom, the Hindsight Memory system provides persistent, project-scoped knowledge retention across sessions.

### Entry Points

Oh-My-Pi offers four ways to interact with the same engine:

- **Interactive TUI** -- The default terminal interface with tool cards, edit previews, and structured option pickers
- **Node SDK** -- Embed the agent directly in JavaScript/TypeScript applications using `@oh-my-pi/pi-coding-agent`
- **RPC Mode** -- Drive the agent over stdio with NDJSON commands for non-Node embedders
- **ACP Mode** -- Speak the [Agent Client Protocol](https://github.com/zed-industries/agent-client-protocol) over JSON-RPC for editor integration

### The Rust Native Engine

What sets omp apart from other terminal agents is its native Rust engine. Instead of shelling out to external tools like `rg`, `grep`, `find`, or `bash`, omp links the real implementations directly into the process:

| Module | What It Does | Lines of Rust |
|--------|-------------|---------------|
| shell | Embedded bash with persistent sessions | ~3,700 |
| grep | Regex search with parallel/sequential modes | ~1,900 |
| keys | Kitty keyboard protocol with PHF lookup | ~1,490 |
| text | ANSI-aware width, truncation, SGR-preserving wrap | ~1,450 |
| summarize | Tree-sitter structural source summaries | ~1,040 |
| ast | ast-grep pattern matching and structural rewrites | ~1,000 |
| fs_cache | Mtime-keyed file cache shared by read, grep, lsp | ~840 |
| highlight | Syntax highlighting with 11 semantic categories | ~470 |
| pty | Native PTY allocation for sudo and ssh interactive prompts | ~455 |

This means no fork/exec on the hot path. The same binary runs on macOS, Linux, and Windows without requiring WSL.

## Key Features

![Oh-My-Pi Features]({{ site.baseurl }}/assets/img/diagrams/oh-my-pi/oh-my-pi-features.svg)

The features diagram above shows the eight major feature categories of oh-my-pi, each with three concrete capabilities. Hashline Edits provide content-hash anchors that eliminate whitespace battles and string-not-found loops. LSP Integration gives the agent everything your IDE knows -- renames, diagnostics, navigation, and code actions. Subagents enable parallel task fan-out with isolated worktrees and schema-validated results. Time-Travel Rules inject course-correction mid-stream without paying context tax. Browser and Web tools offer stealth browsing and 14 search providers with structured extraction. Hindsight Memory provides project-scoped persistent knowledge. The Native Engine delivers in-process ripgrep, brush shell, and 5-platform support. And 40+ Providers give you one `/model` command to switch between any model.

### 01. Hashline: Edit by Content Hash

The signature feature of oh-my-pi is its hash-anchored edit system. Instead of retyping the lines it wants to change, the model points at anchors -- content hashes of the original lines. This eliminates the two most common edit failures in AI coding agents:

1. **Whitespace battles** -- The model's indentation does not match the file, causing `str_replace` to fail
2. **Stale file corruption** -- The model edits a file that has changed since it was last read, silently overwriting new content

When a file has been modified since the agent last read it, the anchors diverge and omp rejects the patch before it corrupts anything. According to the project's benchmarks, Grok 4 Fast spends **61% fewer output tokens** on the same work when using hash-anchored edits instead of traditional diff-based approaches.

### 02. LSP Wired Into Every Write

Ask for a rename and you get a rename. The call goes through `workspace/willRenameFiles`, so re-exports, barrel files, and aliased imports update before the file moves. The agent has access to 13 LSP operations including diagnostics, navigation, symbols, renames, code actions, and raw requests. Everything your IDE knows, the agent knows.

### 03. Drives a Real Debugger

While most AI agents are still sprinkling print statements, omp attaches real debuggers:

- A C binary segfaults: attach `lldb`, step to the bad pointer, read the frame
- A Go service hangs: attach `dlv` and walk the goroutines
- A Python process is wedged: `debugpy`, pause, inspect, evaluate

The `debug` tool provides 27 DAP operations for breakpoints, stepping, threads, stack inspection, and variable evaluation.

### 04. Time-Traveling Stream Rules

Your rules sit dormant until the model goes off-script. A regex match aborts the stream mid-token, injects the rule as a system reminder, and retries from the same point. You get course-correction without paying context tax on every turn. Injections survive compaction, so the fix sticks across the entire session.

### 05. First-Class Subagents

The `task` tool fans out into isolated worktrees, each worker runs its own tool surface, and the final yield is a schema-validated object the parent reads directly. No prose to parse, no merge conflicts between siblings, no orphaned edits. Subagents can even communicate via IRC-style direct messages for coordination.

### 06. Web Search with Structured Extraction

`web_search` chains 14 ranked providers and hands whatever URLs it finds straight to `read`. Arxiv PDFs, GitHub pages, Stack Overflow threads come back as structured markdown with anchors intact. The agent can cite, follow, and quote without losing where it came from.

### 07. Native Performance, Even on Windows

Other agents shell out to `rg`, `grep`, `find`, and `bash`. On many machines those binaries do not exist, and on the ones where they do, every call costs a fork-exec round-trip. omp links the real implementations into the process. ripgrep, glob, find: in-process. brush is the bash, with sessions that survive across calls. The same binary runs on macOS, Linux, and Windows -- no WSL bridge required.

### 08. Code Review with Priorities

`/review` spawns dedicated reviewer subagents that sweep branches, single commits, or uncommitted work in parallel. Every issue is ranked P0 through P3 and scored for confidence. You tackle what blocks release first; nothing important hides in a wall of prose.

### 09. Hindsight: Memory the Agent Curates

The agent remembers your codebase between sessions. It writes facts mid-run with `retain`, pulls them back with `recall`, and compresses each session into a mental model that loads on the first turn of the next one. Project-scoped by default, so what it learns about this repo stays with this repo.

### 10. Inherits Existing Config

Every other agent ships an importer and expects you to convert. omp reads the eight formats already on disk in their native shape -- Cursor MDC, Cline `.clinerules`, Codex `AGENTS.md`, Copilot `applyTo`, and the rest. No migration script, no YAML-to-TOML port, no "supported subset" footnotes.

## The 32 Built-In Tools

Oh-My-Pi ships 32 tools in the same namespace as `read` and `bash`. Pin the active set with `--tools read,edit,bash,...` and the rest stay hidden but indexed -- `search_tool_bm25` pulls them back in mid-session when `tools.discoveryMode` says so.

| Category | Tools | Description |
|----------|-------|-------------|
| Files and Search | `read`, `write`, `edit`, `ast_edit`, `ast_grep`, `search`, `find` | Files, dirs, archives, SQLite, PDFs, URLs through one path |
| Runtime | `bash`, `eval`, `recipe`, `ssh` | Shell, Python/JS cells, task runners, remote commands |
| Code Intelligence | `lsp`, `debug` | Diagnostics, navigation, renames, DAP debugging |
| Coordination | `task`, `irc`, `todo_write`, `job`, `ask` | Subagents, inter-agent messaging, session state |
| Outside the Box | `browser`, `web_search`, `github`, `generate_image`, `inspect_image`, `render_mermaid` | Web, GitHub, vision, diagrams |
| Memory and State | `checkpoint`, `rewind`, `retain`, `recall`, `reflect` | Session state, context pruning, persistent memory |
| Misc | `calc`, `resolve`, `search_tool_bm25` | Arithmetic, preview actions, tool discovery |

## 40+ Providers, One Command Away

Roles route work by intent. `default` for normal turns. `smol` for cheap subagent fan-out. `slow` for deep reasoning. `plan` for plan mode. `commit` for changelogs. Override at launch with `--smol`, `--slow`, or `--plan`; cycle through the configured models for the active role with `Ctrl+P`. Swap the active model mid-session with the `/model` slash command.

### Frontier APIs

Anthropic, OpenAI, OpenAI Codex, Google Gemini, xAI, Mistral, Groq, Cerebras, Fireworks, Together, Hugging Face, NVIDIA, OpenRouter, Perplexity, and more.

### Coding Plans

Cursor, GitHub Copilot, GitLab Duo, Kimi Code, MiniMax, Alibaba, Qwen, Xiaomi, and others -- subscription-routed with `/login`.

### Self-Hosted

Ollama, LM Studio, llama.cpp, vLLM, LiteLLM -- OpenAI-compatible `/v1/models` endpoints with optional keys.

### Four Knobs for Routing

- **Custom providers** -- Declare anything that speaks OpenAI, Anthropic, or Google protocols
- **Fallback chains** -- Per-role chains under `retry.fallbackChains`; when the primary hits 429, the next entry takes over
- **Path-scoped roles** -- Pin a heavier `default` on one repo without touching global config
- **Round-robin credentials** -- Stack API keys per provider; the runtime rotates with session affinity

## Installation

### macOS and Linux

```bash
curl -fsSL https://omp.sh/install | sh
```

### Bun (Recommended)

```bash
bun install -g @oh-my-pi/pi-coding-agent
```

### Windows (PowerShell)

```powershell
irm https://omp.sh/install.ps1 | iex
```

### Pinned Versions (mise)

```bash
mise use -g github:can1357/oh-my-pi
```

**Requirements**: macOS, Linux, or Windows with Bun >= 1.3.14.

## Quick Start

After installation, launch omp in your project directory:

```bash
cd your-project
omp
```

On first run, omp inherits whatever is already on disk: rules, skills, and MCP servers from `.claude`, `.cursor`, `.windsurf`, `.gemini`, `.codex`, `.cline`, `.github/copilot`, and `.vscode`. No migration script needed.

### Switch Models

Use `/model` to cycle through configured providers, or set environment variables:

```bash
export ANTHROPIC_API_KEY=your-key
export OPENAI_API_KEY=your-key
```

### Run a One-Shot Command

```bash
omp -p "list all TypeScript files in src/"
```

### Fan Out Subagents

```bash
omp --task "refactor the auth module" --task "add tests for utils"
```

## Oh-My-Pi vs. Other Terminal Agents

| Feature | oh-my-pi | Claude Code | Aider | Cursor Agent |
|---------|----------|-------------|-------|--------------|
| Edit method | Hash-anchored | str_replace | search/replace | diff |
| LSP integration | 13 ops | Limited | None | Full IDE |
| Debugger (DAP) | 27 ops | None | None | IDE debugger |
| Subagents | First-class | None | None | None |
| Browser tool | Built-in | None | None | Built-in |
| Native Rust engine | ~27k LoC | None | None | None |
| Providers | 40+ | Anthropic only | Multiple | Multiple |
| Memory | Hindsight | None | None | Project context |
| Stream rules | Time-traveling | None | None | None |
| Config inheritance | 8 formats | Claude only | Aider only | Cursor only |

## Monorepo Structure

Oh-My-Pi is organized as a monorepo with the following packages:

| Package | Description |
|---------|-------------|
| `@oh-my-pi/pi-ai` | Multi-provider LLM client with streaming and model/provider integration |
| `@oh-my-pi/pi-agent-core` | Agent runtime with tool calling and state management |
| `@oh-my-pi/pi-coding-agent` | Interactive coding agent CLI and SDK |
| `@oh-my-pi/pi-tui` | Terminal UI library with differential rendering |
| `@oh-my-pi/pi-natives` | N-API bindings for grep, shell, image, text, syntax highlighting |
| `@oh-my-pi/omp-stats` | Local observability dashboard for AI usage statistics |
| `@oh-my-pi/pi-utils` | Shared utilities (logging, streams, dirs/env/process helpers) |
| `@oh-my-pi/swarm-extension` | Swarm orchestration extension package |

### Rust Crates

| Crate | Description |
|-------|-------------|
| `pi-natives` | Core Rust native addon (N-API cdylib) |
| `pi-shell` | Embedded shell / PTY / process management |
| `pi-ast` | Tree-sitter-based code summarizer and AST utilities |
| `pi-iso` | Task isolation backend (APFS, btrfs, zfs, overlayfs) |
| `brush-core-vendored` | Vendored fork of brush-shell for embedded bash |
| `brush-builtins-vendored` | Vendored bash builtins |

## Real-World Benchmarks

The project's benchmark data tells a compelling story about the impact of the harness on model performance:

| Model | Metric | Improvement |
|-------|--------|-------------|
| Grok Code Fast 1 | 6.7% to 68.3% | Tenfold lift when edit format stops eating the model alive |
| Gemini 3 Flash | +5 pp | Over str_replace -- beats Google's own best attempt at the format |
| Grok 4 Fast | -61% tokens | Output collapses once the retry loop on bad diffs disappears |
| MiniMax | 2.1x | Pass rate more than doubles with same weights and prompt |

These numbers demonstrate that the harness -- not just the model -- is a first-class lever for coding agent performance.

## Extensibility

An extension is a TypeScript module with the same tool API, slash-command registry, hotkey table, and TUI primitives the built-ins use. Nothing is reserved. Ask omp to write the piece you are missing, then `/reload-plugins`. Keep it local, ship it in a marketplace, or publish it to npm.

## License

Oh-My-Pi is released under the **MIT License**.

- Original work (c) 2025 Mario Zechner
- Extensions and modifications (c) 2025-2026 Can Boeluek

## Links

- **Website**: [omp.sh](https://omp.sh)
- **GitHub**: [github.com/can1357/oh-my-pi](https://github.com/can1357/oh-my-pi)
- **npm**: [@oh-my-pi/pi-coding-agent](https://www.npmjs.com/package/@oh-my-pi/pi-coding-agent)
- **Discord**: [discord.gg/4NMW9cdXZa](https://discord.gg/4NMW9cdXZa)
- **Changelog**: [CHANGELOG.md](https://github.com/can1357/oh-my-pi/blob/main/packages/coding-agent/CHANGELOG.md)