---
layout: post
title: "CodeGraph v2: 38K Stars Update — Token Optimization, New Platforms, and 5x Growth"
description: "CodeGraph exploded from 7,600 to 38,745 stars in two weeks. This update covers token optimization (fewer tokens, fewer tool calls), 8+ platform support (Gemini CLI, AntiGravity, Kiro, Hermes Agent), and why developers are adopting it so rapidly."
date: 2026-06-04
header-img: "img/post-bg.jpg"
permalink: /CodeGraph-v2-38000-Stars-Update-Token-Optimization-New-Platforms/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, TypeScript]
tags: [CodeGraph, Claude Code, code knowledge graph, AI coding agents, token optimization, Gemini CLI, Cursor, Codex CLI, MCP server, code intelligence, static analysis]
keywords: "CodeGraph v2 update, CodeGraph 38000 stars, CodeGraph token optimization, CodeGraph Gemini CLI, CodeGraph new platforms, AI coding agent token efficiency, CodeGraph vs manual code exploration, fewer tokens fewer tool calls, CodeGraph MCP server, pre-indexed code knowledge graph"
author: "PyShine"
---

# CodeGraph v2: 38K Stars Update — Token Optimization, New Platforms, and 5x Growth

Two weeks ago, we covered CodeGraph when it had 7,600 stars. Today it has 38,745 — a 5x explosion driven by one insight: **fewer tokens, fewer tool calls, 100% local**. This update covers what changed, what's new, and why the developer community is adopting CodeGraph at breakneck speed.

## The Growth Story

![CodeGraph Growth Timeline](/assets/img/diagrams/codegraph-v2/codegraph-growth-timeline.svg)

CodeGraph went from 7,600 to 38,745 stars in approximately two weeks, with +10,793 stars this week alone. That makes it one of the fastest-growing developer tools on GitHub.

Why the explosive growth? Three reasons:

1. **Token efficiency matters more than ever.** As AI coding agents proliferate, context window waste is the bottleneck. CodeGraph eliminates wasteful file-scanning loops.
2. **Platform expansion.** CodeGraph now supports 8+ AI coding agent platforms, up from 4 in the original release.
3. **Zero configuration.** One command installs and configures everything. No API keys, no external services, no data leaving your machine.

## Token Optimization: The New Focus

![Token Optimization Flow](/assets/img/diagrams/codegraph-v2/codegraph-token-optimization.svg)

The original CodeGraph pitch was "94% fewer tool calls, 77% faster exploration." The v2 message sharpens the focus: **fewer tokens consumed**.

### Why Tokens Matter

Every tool call an AI agent makes consumes tokens — both in the request and the response. When Claude Code explores an unfamiliar codebase, it spawns Explore agents that scan files using grep, glob, and Read tool calls. For a large project like VS Code, answering a single architecture question can require 52 tool calls consuming 50K+ tokens.

CodeGraph replaces this wasteful discovery loop with structured graph queries. The same VS Code architecture question takes 3 tool calls and ~2K tokens with CodeGraph.

### How CodeGraph Reduces Token Consumption

| Approach | Tool Calls | Tokens | Time |
|----------|-----------|--------|------|
| Without CodeGraph | 52 calls | 50K+ tokens | 1m 37s |
| With CodeGraph | 3 calls | ~2K tokens | 17s |
| **Improvement** | **94% fewer** | **~96% fewer** | **82% faster** |

The `codegraph_context` tool is the token-saving powerhouse. It combines search, navigation, and code retrieval into a single call that returns everything an agent needs to understand a code area. This replaces the 40-50 tool calls that agents normally make during exploration.

## Expanded Platform Ecosystem

![Platform Ecosystem](/assets/img/diagrams/codegraph-v2/codegraph-platform-ecosystem.svg)

CodeGraph v2 expands from 4 to 8+ supported AI coding agent platforms:

| Platform | Status | Integration |
|----------|--------|-------------|
| Claude Code | Original | MCP server + CLAUDE.md instructions |
| Cursor | Original | MCP server + .cursor/rules/codegraph.mdc |
| Codex CLI | Original | MCP server + AGENTS.md instructions |
| opencode | Original | MCP server integration |
| **Gemini CLI** | **New** | MCP server integration |
| **AntiGravity** | **New** | MCP server integration |
| **Kiro** | **New** | MCP server integration |
| **Hermes Agent** | **New** | MCP server integration |

All platforms connect through the same MCP server interface. The `npx @colbymchenry/codegraph` installer auto-detects installed agents and configures each one appropriately.

## Architecture Overview

CodeGraph builds a semantic knowledge graph of your codebase using tree-sitter AST parsing, stores it in a local SQLite database with FTS5 full-text search, and exposes it to AI agents through an MCP server with 8 specialized tools.

![CodeGraph Architecture](/assets/img/diagrams/codegraph/codegraph-architecture.svg)

The architecture follows a 5-layer pipeline:

1. **ExtractionOrchestrator** — tree-sitter parses source code into ASTs; language-specific queries extract 22 NodeKinds and 12 EdgeKinds
2. **ReferenceResolver** — connects function calls to definitions, resolves imports, establishes inheritance chains
3. **Framework Detection** — recognizes web-framework routing files across 13 frameworks (Django, Flask, FastAPI, Express, Laravel, Rails, Spring, Gin, chi, gorilla/mux, Axum, actix, Rocket, ASP.NET, Vapor, React Router, SvelteKit)
4. **SQLite Storage** — local `.codegraph/codegraph.db` with FTS5 full-text search, using `better-sqlite3` (native) or `node-sqlite3-wasm` (fallback)
5. **Auto-Sync** — native OS file events (FSEvents/inotify/ReadDirectoryChangesW) with 2-second debounced incremental sync

## Benchmark Results

Tested across 6 real-world codebases comparing Claude Code's Explore agent with and without CodeGraph:

| Codebase | With CodeGraph | Without CodeGraph | Improvement |
|----------|---------------|-------------------|-------------|
| VS Code (TypeScript) | 3 calls, 17s | 52 calls, 1m 37s | 94% fewer, 82% faster |
| Excalidraw (TypeScript) | 3 calls, 29s | 47 calls, 1m 45s | 94% fewer, 72% faster |
| Claude Code (Python+Rust) | 3 calls, 39s | 40 calls, 1m 8s | 93% fewer, 43% faster |
| Claude Code (Java) | 1 call, 19s | 26 calls, 1m 22s | 96% fewer, 77% faster |
| Alamofire (Swift) | 3 calls, 22s | 32 calls, 1m 39s | 91% fewer, 78% faster |
| Swift Compiler (Swift/C++) | 6 calls, 35s | 37 calls, 2m 8s | 84% fewer, 73% faster |

> **Key observation:** With CodeGraph, agents **never fell back to reading files** — they trusted the graph results completely. The Swift Compiler benchmark tested the largest codebase (25,874 files, 272,898 nodes) and CodeGraph indexed it in under 4 minutes.

## MCP Tools

CodeGraph exposes 8 tools through its MCP server:

| Category | Tool | Purpose |
|----------|------|---------|
| Search | `codegraph_search` | Find symbols by name using FTS5 full-text search |
| Search | `codegraph_files` | Get indexed file structure |
| Search | `codegraph_status` | Check index health and statistics |
| Navigation | `codegraph_callers` | Trace incoming call chains |
| Navigation | `codegraph_callees` | Trace outgoing call chains |
| Navigation | `codegraph_node` | Get symbol details with optional source code |
| Impact | `codegraph_impact` | Analyze what code is affected by changing a symbol |
| Context | `codegraph_context` | Build complete context for a task in one call |

The `codegraph_context` tool is the most powerful — it combines search, navigation, and code retrieval into a single call, replacing the 40-50 tool calls that agents normally make during exploration.

## Supported Languages

CodeGraph supports 19+ languages through tree-sitter grammar parsing:

| Category | Languages |
|----------|-----------|
| Web/Scripting | TypeScript, JavaScript, Python, Ruby, PHP, Dart, Svelte, Liquid |
| Systems | Go, Rust, C, C++ |
| JVM | Java, Kotlin |
| Apple | Swift |
| .NET | C# |
| Other | Pascal/Delphi |

## Getting Started

### Quick Install

```bash
npx @colbymchenry/codegraph
```

The interactive installer will:
1. Auto-detect installed agents (Claude Code, Cursor, Codex CLI, opencode, Gemini CLI, AntiGravity, Kiro, Hermes Agent)
2. Install `codegraph` on your PATH
3. Configure each agent's MCP server settings and instructions
4. Set up auto-allow permissions for Claude Code
5. Initialize your current project

### Non-Interactive Install

```bash
# Auto-detect agents, install global
codegraph install --yes

# Explicit target list
codegraph install --target=cursor,claude --yes

# Print config snippet without writing files
codegraph install --print-config codex
```

### Initialize Projects

```bash
cd your-project
codegraph init -i
```

This builds the per-project knowledge graph index. Restart your agent for the MCP server to load.

## CLI Reference

```bash
codegraph                         # Run interactive installer
codegraph install                 # Run installer (explicit)
codegraph init [path]             # Initialize in a project (--index to also index)
codegraph uninit [path]           # Remove CodeGraph from a project
codegraph index [path]            # Full index (--force to re-index)
codegraph sync [path]             # Incremental update
codegraph status [path]           # Show statistics
codegraph query <search>          # Search symbols (--kind, --limit, --json)
codegraph files [path]            # Show file structure
codegraph context <task>          # Build context for AI
codegraph affected [files...]     # Find test files affected by changes
codegraph serve --mcp             # Start MCP server
```

### Affected Files for CI

```bash
# Pass files as arguments
codegraph affected src/utils.ts src/api.ts

# Pipe from git diff
git diff --name-only | codegraph affected --stdin

# Custom test file pattern
codegraph affected src/auth.ts --filter "e2e/*"
```

## Library Usage

```typescript
import CodeGraph from '@colbymchenry/codegraph';

const cg = await CodeGraph.init('/path/to/project');

await cg.indexAll({
  onProgress: (p) => console.log(`${p.phase}: ${p.current}/${p.total}`)
});

const results = cg.searchNodes('UserService');
const callers = await cg.getCallers('UserService.login');
const impact = await cg.getImpactRadius('UserService.login');
const context = await cg.buildContext('implement user authentication');

await cg.watch();  // Auto-sync on file changes
await cg.close();
```

## Key Features Summary

| Feature | Description |
|---------|-------------|
| Token Optimization | Fewer tokens consumed — agents query structured graph instead of reading raw files |
| Smart Context Building | One tool call returns entry points, related symbols, and code snippets |
| Full-Text Search | Find code by name instantly across your entire codebase, powered by FTS5 |
| Impact Analysis | Trace callers, callees, and the full impact radius of any symbol |
| Always Fresh | File watcher uses native OS events with debounced auto-sync |
| 19+ Languages | TypeScript, JavaScript, Python, Go, Rust, Java, C#, PHP, Ruby, C, C++, Swift, Kotlin, Dart, Svelte, Liquid, Pascal/Delphi |
| Framework-aware Routes | Recognizes web-framework routing across 13 frameworks |
| 100% Local | No data leaves your machine. No API keys. No external services |
| 8+ Platform Support | Claude Code, Cursor, Codex CLI, opencode, Gemini CLI, AntiGravity, Kiro, Hermes Agent |

## What Changed Since Our Original Post

| Aspect | Original Post (May 20) | This Update (June 4) |
|--------|------------------------|----------------------|
| Stars | 7,600 | 38,745 |
| Tagline | "94% fewer tool calls, 77% faster exploration" | "fewer tokens, fewer tool calls, 100% local" |
| Platforms | 4 (Claude Code, Cursor, Codex CLI, opencode) | 8+ (added Gemini CLI, AntiGravity, Kiro, Hermes Agent) |
| Focus | Tool call reduction | Token optimization + tool call reduction |
| Diagrams | 2 (architecture, features) | 3 new (token flow, ecosystem, growth) |

## Conclusion

CodeGraph's 5x growth in two weeks signals a shift in how developers think about AI coding agents. The bottleneck isn't just tool calls — it's **tokens**. Every unnecessary file read, every redundant grep search, every wasteful exploration loop consumes context window capacity that could be used for actual coding.

CodeGraph solves this by pre-indexing codebases into a semantic knowledge graph with tree-sitter, giving AI agents instant structural understanding that replaces dozens of file-scanning tool calls with a single graph query. With 19+ language support, 13 framework-aware route detectors, 8+ platform integrations, 100% local processing, and automatic file watching, CodeGraph is a zero-configuration productivity multiplier for any developer using AI coding agents.

**Repository:** [https://github.com/colbymchenry/codegraph](https://github.com/colbymchenry/codegraph)

**npm:** [@colbymchenry/codegraph](https://www.npmjs.com/package/@colbymchenry/codegraph)

**License:** MIT