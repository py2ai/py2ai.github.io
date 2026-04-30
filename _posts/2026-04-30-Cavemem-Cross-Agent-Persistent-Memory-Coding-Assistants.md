---
layout: post
title: "Cavemem: Cross-Agent Persistent Memory for Coding Assistants with Compressed SQLite Storage"
description: "Discover how cavemem gives AI coding assistants persistent memory across sessions using compressed SQLite storage, hybrid search, and MCP tools. Local-first, privacy-aware, and supports Claude Code, Cursor, Gemini CLI, and more."
date: 2026-04-30
header-img: "img/post-bg.jpg"
permalink: /Cavemem-Cross-Agent-Persistent-Memory-Coding-Assistants/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, Open Source]
tags: [cavemem, persistent memory, AI coding agent, SQLite, MCP, Claude Code, Cursor, compression, hybrid search, local-first]
keywords: "cavemem persistent memory setup, how to give AI coding agents memory, cross-agent memory SQLite MCP, cavemem vs other memory tools, Claude Code persistent memory, local-first AI agent storage, caveman grammar compression, hybrid search BM25 vector, AI coding assistant session continuity, privacy-aware agent memory"
author: "PyShine"
---

# Cavemem: Cross-Agent Persistent Memory for Coding Assistants

Cavemem is a cross-agent persistent memory system that gives AI coding assistants the ability to remember across sessions. Using deterministic compression, local SQLite storage, and Model Context Protocol (MCP) tools, cavemem captures what happened during coding sessions and makes that knowledge available the next time you start work. No network. No cloud. No daemon required on the write path.

## The Problem: AI Agents Have Amnesia

Every time you start a new session with Claude Code, Cursor, Gemini CLI, or any other AI coding assistant, it starts from scratch. The model has no memory of what it did yesterday, what bugs it fixed last week, or what architectural decisions were made in previous sessions. This amnesia forces developers to re-explain context, re-describe codebases, and re-establish coding standards at the start of every conversation.

Cavemem solves this by hooking into session boundaries, compressing observations with the caveman grammar (approximately 75% fewer prose tokens while preserving code byte-for-byte), and writing to a local SQLite database. Agents query their own history through three MCP tools: `search`, `timeline`, and `get_observations`.

## How It Works: The Write Path

![Write Path](/assets/img/diagrams/cavemem/cavemem-write-path.svg)

### Understanding the Write Path

The write path is designed for speed and reliability. Hook handlers must complete in under 150 milliseconds, which means no network calls, no model invocations, and no waiting on background processes. Every observation is written synchronously to SQLite before the hook returns.

**Step 1: IDE Session Event**

When a coding session event occurs (session start, tool use, session end, user prompt), the IDE fires a hook that invokes the cavemem CLI. This integration works across five IDEs: Claude Code, Cursor, Gemini CLI, OpenCode, and Codex. Each IDE has its own installer that registers the appropriate hooks and MCP configuration.

**Step 2: Privacy Redaction**

Before any text is stored, the redaction layer strips content inside `<private>...</private>` tags. This is enforced at the write boundary, meaning private content never appears in the database, in logs, or in search results. Additionally, paths matching `excludePatterns` in settings are never read in the first place.

**Step 3: Caveman Compression**

The compressor transforms prose using a deterministic grammar that removes pleasantries, hedges, fillers, and articles while preserving technical tokens byte-for-byte. Code blocks, URLs, file paths, shell commands, version numbers, dates, and numeric identifiers pass through untouched. The compression is lossy on filler words by design, but every technical detail is preserved exactly.

**Step 4: SQLite + FTS5 Storage**

Compressed observations are committed to a local SQLite database. FTS5 (Full-Text Search) indexes are updated via triggers, enabling fast keyword search immediately. The database lives at `~/.cavemem/` by default and requires no configuration.

**Step 5: Background Embedding (Async)**

When embedding is enabled, a local worker auto-spawns in the background to compute vector embeddings. This worker uses Transformers.js by default (no network calls), with optional support for Ollama or OpenAI providers. The worker self-exits when idle and binds to `127.0.0.1` only. If the worker is down, writes still succeed; only semantic search is degraded (BM25 keyword search keeps working).

## How It Works: The Read Path

![Read Path](/assets/img/diagrams/cavemem/cavemem-read-path.svg)

### Understanding the Read Path

The read path uses progressive disclosure to minimize token consumption. Rather than dumping full observation bodies into the agent's context, cavemem returns compact results first and lets the agent request full details only when needed.

**MCP Server (stdio)**

The MCP server runs on stdio and exposes four tools to the agent:

| Tool | Returns | Purpose |
|------|---------|---------|
| `search(query, limit?)` | Compact results with IDs, scores, snippets, session IDs, timestamps | Find relevant memories using BM25 keyword search plus optional cosine re-ranking |
| `timeline(session_id, around_id?, limit?)` | Compact results with IDs, kinds, timestamps | Browse observations in chronological order within a session |
| `get_observations(ids[], expand?)` | Full observation bodies, expanded by default | Fetch complete details for specific observations |
| `list_sessions(limit?)` | Session metadata with IDE, working directory, timestamps | Discover available sessions |

**Hybrid Search: BM25 + Vector**

Cavemem combines two search strategies:

1. **BM25 (SQLite FTS5)**: Fast keyword search that works immediately, even without embeddings. This is the baseline that never degrades.

2. **Cosine Re-ranking**: When embeddings are available, search results are re-ranked using vector similarity. The `alpha` setting (default 0.5) controls the blend between BM25 and vector scores.

The tunable ranker lets you adjust the balance: set `alpha` closer to 1.0 for keyword-heavy search, or closer to 0.0 for semantic-heavy search.

**Web Viewer**

A read-only HTTP viewer at `http://localhost:37777` provides a human-readable interface for browsing sessions. The viewer serves expanded text (the compression is reversed for display), making it easy to inspect what the agent has remembered without using the MCP tools.

## The Compression Engine

![Compression Pipeline](/assets/img/diagrams/cavemem/cavemem-compression.svg)

### Understanding the Compression Pipeline

The caveman compression engine is the signature innovation of cavemem. It compresses prose deterministically and offline, never invoking a model. The contract is strict: deterministic output, byte-for-byte technical token preservation, and round-trip-guaranteed expansion for human readability.

**Tokenizer Segmentation**

The tokenizer splits input into two categories:

- **Preserved tokens**: Code blocks, inline code, URLs, file paths, shell commands, version numbers, dates, numeric literals, and identifiers. These segments pass through the compressor untouched.

- **Prose tokens**: Everything else. These are the segments that get compressed.

The tokenizer recognizes 10 distinct kinds: `fence` (triple-backtick code blocks), `inline-code`, `url`, `path`, `version`, `date`, `number`, `identifier`, `heading`, and `prose`.

**Prose Transforms**

Prose segments go through three transforms in order:

1. **Remove pleasantries, hedges, fillers, and articles** (intensity-driven)
2. **Apply the abbreviations map** (intensity-driven)
3. **Collapse whitespace**

**Intensity Levels**

| Level | Articles | Fillers | Hedges | Abbreviations |
|-------|----------|---------|--------|---------------|
| `lite` | Keep | Minimal | Keep | Minimal |
| `full` | Drop | Broad | Drop | Broad |
| `ultra` | Drop | Aggressive | Drop | Aggressive (includes `w/`, `b/c`, `&`) |

**Compression Example**

```
Input:  "The auth middleware throws a 401 when the session token expires; we should add a refresh path."
Stored: "auth mw throws 401 @ session token expires. add refresh path."
Viewed: "The auth middleware throws a 401 when session token expires. Add a refresh path."
```

Notice that `401` (a numeric literal) and `session token` (an identifier) are preserved exactly. Only the filler words and hedging language are removed. The expansion step restores known abbreviations but does not restore dropped words, since the stored form has already committed to brevity.

**Performance**

The compression engine achieves a throughput of at least 5 MB/s on a single core. Hook handlers complete in under 150ms p95. Average token reduction on the benchmark corpus is at least 30% (target is 40% at full intensity, 55% at ultra).

## The Caveman Ecosystem

![Caveman Ecosystem](/assets/img/diagrams/cavemem/cavemem-ecosystem.svg)

### Understanding the Caveman Ecosystem

Cavemem is part of a three-tool ecosystem built on a shared philosophy: agents should do more with less. Each tool stands alone but they compose powerfully when used together.

**caveman: Output Compression**

The first tool in the ecosystem compresses what the agent says. Using the same deterministic grammar, caveman reduces output tokens by approximately 75% across Claude Code, Cursor, Gemini, and Codex. This means the agent communicates more efficiently, consuming fewer tokens per response.

**cavemem: Persistent Memory (This Tool)**

The second tool compresses what the agent remembers. Session observations are compressed at write time and stored in SQLite. The agent queries its own history through MCP tools, receiving compressed results by default and expanding only when needed.

**cavekit: Autonomous Build Loop**

The third tool compresses what the agent guesses. Instead of letting the agent improvise solutions, cavekit uses a spec-driven approach: natural language specifications are decomposed into kits, built in parallel, and verified against the original spec. This eliminates the common failure mode of agents building the wrong thing.

**How They Compose**

When used together, cavekit orchestrates the build process, caveman compresses the agent's output to save tokens, and cavemem stores the decisions and outcomes for future sessions. The result is an agent that builds correctly, communicates efficiently, and remembers everything.

## Installation and Setup

### Install Globally

```bash
npm install -g cavemem
```

### Register Hooks for Your IDE

```bash
cavemem install                    # Claude Code (default)
cavemem install --ide cursor       # Cursor
cavemem install --ide gemini-cli   # Gemini CLI
cavemem install --ide opencode     # OpenCode
cavemem install --ide codex        # Codex
```

### Verify Installation

```bash
cavemem status
cavemem doctor
```

The `status` command shows a single dashboard with wiring status, database counts, embedding backfill progress, and worker process ID. The `doctor` command runs a full verification of the installation.

### Open the Memory Viewer

```bash
cavemem viewer                     # Opens http://127.0.0.1:37777
```

### Search Memory from CLI

```bash
cavemem search "auth middleware bug" --limit 10
cavemem search "deployment config" --no-semantic   # BM25 only
```

## Configuration

Settings are stored at `~/.cavemem/settings.json` and can be managed through the CLI:

| Key | Default | Description |
|-----|---------|-------------|
| `dataDir` | `~/.cavemem` | SQLite database location |
| `compression.intensity` | `full` | Compression level: `lite` / `full` / `ultra` |
| `compression.expandForModel` | `false` | Return expanded text to the model |
| `embedding.provider` | `local` | Embedding provider: `local` / `ollama` / `openai` |
| `workerPort` | `37777` | Local viewer port |
| `search.alpha` | `0.5` | BM25 / vector blend ratio |
| `search.defaultLimit` | `10` | Default search result count |
| `privacy.excludePatterns` | `[]` | Paths never captured |

```bash
cavemem config show                # View all settings
cavemem config set embedding.provider ollama
cavemem config set compression.intensity ultra
cavemem config open                # Open settings in editor
```

## Privacy and Security

Cavemem is designed with a local-first, privacy-aware philosophy:

- **No network calls by default**: The local embedding provider uses Transformers.js with no outbound connections
- **Private content stripping**: `<private>...</private>` tags are stripped at the write boundary
- **Path exclusion**: Configure glob patterns to exclude entire directories from capture
- **Loopback binding**: The worker binds to `127.0.0.1` only, never exposing data to the network
- **No cloud dependency**: All data stays on your machine in SQLite

## CLI Reference

| Command | Description |
|---------|-------------|
| `cavemem install [--ide <name>]` | Register hooks + MCP for an IDE |
| `cavemem uninstall [--ide <name>]` | Remove hooks + MCP |
| `cavemem status` | Dashboard: wiring, DB counts, embedding backfill, worker PID |
| `cavemem config show\|get\|set\|open` | View/edit settings |
| `cavemem start\|stop\|restart` | Control the worker daemon |
| `cavemem viewer` | Open the memory viewer in browser |
| `cavemem doctor` | Verify installation |
| `cavemem search <query>` | Search memory (BM25 + cosine re-rank) |
| `cavemem compress <file>` | Compress a file with caveman grammar |
| `cavemem reindex` | Rebuild FTS5 + vector index |
| `cavemem export <out.jsonl>` | Dump observations to JSONL |
| `cavemem mcp` | Start MCP server (stdio) |

## Who Should Use Cavemem

Cavemem is designed for developers who use AI coding assistants regularly and want those assistants to maintain context across sessions:

- **Claude Code users** who want their agent to remember what it did yesterday without re-explaining the entire project
- **Cursor users** who switch between projects and need session-specific memory
- **Teams using multiple IDEs** who want cross-IDE memory that works regardless of which tool they're using
- **Privacy-conscious developers** who want local-only storage with no cloud dependency
- **Anyone managing complex codebases** where session continuity saves hours of re-briefing

## Conclusion

Cavemem addresses one of the most fundamental limitations of AI coding assistants: session amnesia. By combining deterministic compression, local SQLite storage, hybrid search, and MCP-based progressive disclosure, it gives agents the ability to remember what happened in previous sessions without sacrificing privacy or requiring network access. The caveman grammar compression ensures that stored memories are compact (approximately 75% fewer prose tokens) while preserving every technical detail byte-for-byte. As part of the broader Caveman ecosystem, cavemem complements output compression (caveman) and autonomous building (cavekit) to create AI agents that say less, remember more, and build better.

**Links:**

- GitHub Repository: [https://github.com/JuliusBrussee/cavemem](https://github.com/JuliusBrussee/cavemem)
- npm Package: [https://www.npmjs.com/package/cavemem](https://www.npmjs.com/package/cavemem)
- Caveman Ecosystem: [https://github.com/JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman)