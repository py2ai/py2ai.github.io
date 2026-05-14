---
layout: post
title: "AgentMemory: Persistent Memory for AI Coding Agents"
date: 2026-05-14
categories: [ai, open-source, developer-tools, memory]
tags: [agentmemory, ai-agents, persistent-memory, claude-code, cursor, mcp, coding-assistant, llm]
image: /assets/img/diagrams/agentmemory/agentmemory-memory-pipeline.svg
featured-img: ai-coding-frameworks/ai-coding-frameworks
description: "AgentMemory gives AI coding agents persistent memory across sessions. Learn how it captures, compresses, and injects context for Claude Code, Cursor, and more."
header-img: "img/post-bg.jpg"
permalink: /AgentMemory-Persistent-Memory-AI-Coding-Agents/
keywords: [persistent memory for AI coding agents, AgentMemory MCP integration, Claude Code persistent memory, AI agent memory management, coding assistant context retention, Cursor AI memory plugin, LLM agent session memory, AI coding agent long-term memory, MCP memory server for agents, agent memory compression pipeline]
author: "PyShine"
---

Every coding agent forgets everything when the session ends. You waste the first 5 minutes of every session re-explaining your stack, your preferences, your architecture decisions. **AgentMemory** ([rohitg00/agentmemory](https://github.com/rohitg00/agentmemory)) fixes this — it's persistent memory for AI coding agents that silently captures what your agent does, compresses it into searchable memory, and injects the right context when the next session starts.

With **7,550+ GitHub stars** and **1,335 stars today**, agentmemory is the #1 trending solution for giving coding agents long-term memory. It works with Claude Code, Cursor, Codex CLI, Gemini CLI, Cline, Windsurf, and any MCP-compatible agent.

## The Problem: Agents Forget Everything

Built-in agent memory (CLAUDE.md, .cursorrules) caps out at ~200 lines and goes stale. Every new session means:

- Re-explaining your architecture decisions
- Re-discovering the same bugs
- Re-teaching your coding preferences
- Copy-pasting context from previous conversations

At 240 observations, CLAUDE.md consumes **22K+ tokens** — and still can't search semantically.

## How AgentMemory Works

![AgentMemory Memory Pipeline Architecture](/assets/img/diagrams/agentmemory/agentmemory-memory-pipeline.svg)

The memory pipeline has four phases:

### 1. Capture (12 Auto Hooks)

AgentMemory uses **12 lifecycle hooks** that silently capture everything your agent does — zero manual effort required:

- **SessionStart** — loads project profile (top concepts, files, patterns)
- **UserPromptSubmit** — captures user intent (privacy-filtered)
- **PreToolUse** — records file access patterns with enriched context
- **PostToolUse** — captures tool name, input, and output
- **PostToolUseFailure** — records error context
- **PreCompact** — re-injects memory before compaction
- **Stop / SessionEnd** — summarizes the session

### 2. Process

Raw observations go through a processing pipeline:

1. **SHA-256 Dedup** — 5-minute window eliminates duplicate captures
2. **Privacy Filter** — strips API keys, secrets, and `<private>` tags before storage
3. **LLM Compression** — compresses raw observations into structured facts, concepts, and narratives
4. **Vector Embedding** — supports 6 providers plus local embeddings (all-MiniLM-L6-v2, free and offline)

### 3. Store (Triple Index + 4-Tier Consolidation)

![AgentMemory 4-Tier Memory Consolidation](/assets/img/diagrams/agentmemory/agentmemory-memory-tiers.svg)

AgentMemory uses a **4-tier memory consolidation** system inspired by how human brains process memory during sleep:

| Tier | What | Analogy |
|------|------|---------|
| **Working** | Raw observations from tool use | Short-term memory |
| **Episodic** | Compressed session summaries | "What happened" |
| **Semantic** | Extracted facts and patterns | "What I know" |
| **Procedural** | Workflows and decision patterns | "How to do it" |

Memories decay over time (Ebbinghaus curve). Frequently accessed memories strengthen. Stale memories auto-evict. Contradictions are detected and resolved.

All data is stored in **SQLite** — zero external database dependencies.

### 4. Retrieve (Hybrid Search)

AgentMemory uses **triple-stream retrieval** fused with Reciprocal Rank Fusion (RRF, k=60):

| Stream | What it does | When |
|--------|-------------|------|
| **BM25** | Stemmed keyword matching with synonym expansion | Always on |
| **Vector** | Cosine similarity over dense embeddings | Embedding provider configured |
| **Graph** | Knowledge graph traversal via entity matching | Entities detected in query |

Results are session-diversified (max 3 per session) and delivered within a **~2,000 token budget** — that's **92% fewer tokens** than pasting full context.

## Multi-Agent Ecosystem

![AgentMemory Multi-Agent Ecosystem](/assets/img/diagrams/agentmemory/agentmemory-agent-ecosystem.svg)

One memory server, shared across all your coding agents. AgentMemory works with **any agent that supports MCP or HTTP**:

| Agent | Integration | Notes |
|-------|------------|-------|
| **Claude Code** | 12 hooks + MCP + skills | Plugin marketplace available |
| **Codex CLI** | 6 hooks + MCP + skills | Plugin marketplace available |
| **Cursor** | MCP server | `~/.cursor/mcp.json` |
| **Gemini CLI** | MCP server | `gemini mcp add` |
| **Cline / Roo Code** | MCP server | Settings UI config |
| **Windsurf** | MCP server | `mcp_config.json` |
| **Claude Desktop** | MCP server | `claude_desktop_config.json` |
| **OpenClaw** | MCP + plugin | Memory slot integration |
| **Hermes** | MCP + plugin | 6-hook memory provider |
| **Aider** | REST API | `curl :3111` endpoints |
| **Any agent** | REST API | 104 endpoints available |

All agents share the **same memory server** — your Claude Code session teaches your Cursor session, and vice versa.

## 51 MCP Tools

AgentMemory provides the most comprehensive MCP memory toolkit available:

**Core tools** (always available):
- `memory_recall` — search past observations
- `memory_save` — save an insight, decision, or pattern
- `memory_smart_search` — hybrid semantic + keyword search
- `memory_sessions` — list recent sessions
- `memory_profile` — project profile (concepts, files, patterns)
- `memory_export` — export all memory data

**Extended tools** (50 total with `AGENTMEMORY_TOOLS=all`):
- Knowledge graph: `memory_graph_query`, `memory_relations`
- Team memory: `memory_team_share`, `memory_team_feed`
- Governance: `memory_audit`, `memory_governance_delete`
- Actions: `memory_action_create`, `memory_action_update`, `memory_frontier`, `memory_next`
- Coordination: `memory_lease`, `memory_signal_send`, `memory_signal_read`
- Workflows: `memory_routine_run`, `memory_checkpoint`, `memory_sentinel_create`
- And more...

## Benchmarks That Matter

### Retrieval Accuracy

On the **LongMemEval-S** benchmark (ICLR 2025, 500 questions):

| System | R@5 | R@10 | MRR |
|--------|-----|------|-----|
| **agentmemory** | **95.2%** | **98.6%** | **88.2%** |
| BM25-only fallback | 86.2% | 94.6% | 71.5% |

### Token Savings

| Approach | Tokens/year | Cost/year |
|----------|------------|-----------|
| Paste full context | 19.5M+ | Impossible (exceeds window) |
| LLM-summarized | ~650K | ~$500 |
| **agentmemory** | **~170K** | **~$10** |
| agentmemory + local embeddings | ~170K | **$0** |

## Quick Start

```bash
# Terminal 1: start the memory server
npx @agentmemory/agentmemory

# Terminal 2: seed sample data and see recall in action
npx @agentmemory/agentmemory demo
```

Open `http://localhost:3113` to watch the memory build live in the real-time viewer.

### For Claude Code

```
Install agentmemory: run `npx @agentmemory/agentmemory` in a separate terminal to start the memory server. Then run `/plugin marketplace add rohitg00/agentmemory` and `/plugin install agentmemory` — the plugin registers all 12 hooks, 4 skills, AND auto-wires the MCP server.
```

### For Cursor / Cline / Windsurf / Any MCP Agent

Add this to your agent's MCP config:

```json
{
  "mcpServers": {
    "agentmemory": {
      "command": "npx",
      "args": ["-y", "@agentmemory/mcp"],
      "env": {
        "AGENTMEMORY_URL": "http://localhost:3111"
      }
    }
  }
}
```

## vs Competitors

| Feature | agentmemory | mem0 (53K ⭐) | Letta/MemGPT (22K ⭐) | Built-in (CLAUDE.md) |
|---------|------------|--------------|----------------------|---------------------|
| **Type** | Memory engine + MCP server | Memory layer API | Full agent runtime | Static file |
| **Retrieval R@5** | **95.2%** | 68.5% | 83.2% | N/A (grep) |
| **Auto-capture** | 12 hooks (zero effort) | Manual `add()` calls | Agent self-edits | Manual editing |
| **Search** | BM25 + Vector + Graph (RRF) | Vector + Graph | Vector (archival) | Loads everything |
| **Multi-agent** | MCP + REST + leases + signals | API (no coordination) | Within Letta only | Per-agent files |
| **External deps** | None (SQLite + iii-engine) | Qdrant / pgvector | Postgres + vector DB | None |
| **Token cost** | ~1,900/session ($10/yr) | Varies | Core memory in context | 22K+ at 240 obs |

## Key Features

- **Automatic capture** — 12 hooks record every tool use, zero manual effort
- **Semantic search** — BM25 + vector + knowledge graph with RRF fusion
- **Memory evolution** — versioning, supersession, relationship graphs
- **Auto-forgetting** — TTL expiry, contradiction detection, importance eviction
- **Privacy first** — API keys, secrets, `<private>` tags stripped before storage
- **Self-healing** — circuit breaker, provider fallback chain, health monitoring
- **Claude bridge** — bi-directional sync with MEMORY.md
- **Knowledge graph** — entity extraction + BFS traversal
- **Team memory** — namespaced shared + private across team members
- **Citation provenance** — trace any memory back to source observations
- **Git snapshots** — version, rollback, and diff memory state
- **Session replay** — scrub through past sessions with play/pause and speed control

## Programmatic Access

AgentMemory registers core operations as iii functions (`mem::remember`, `mem::observe`, `mem::context`, `mem::smart-search`, `mem::forget`). Any language with an iii SDK can call them:

```python
from iii import register_worker

iii = register_worker("ws://localhost:49134")
iii.connect()

iii.trigger({
    "function_id": "mem::smart-search",
    "payload": {"project": "demo", "query": "how do tokens refresh"},
})
```

## Conclusion

AgentMemory solves the fundamental problem with AI coding agents: they forget everything between sessions. With 95.2% retrieval accuracy, 92% token savings, 51 MCP tools, and support for every major coding agent, it's the most comprehensive persistent memory solution available. One command to start, one config entry to connect — and your agents never forget again.

**Links:**
- GitHub: [rohitg00/agentmemory](https://github.com/rohitg00/agentmemory)
- npm: [@agentmemory/agentmemory](https://www.npmjs.com/package/@agentmemory/agentmemory)
- Website: [agent-memory.dev](https://agent-memory.dev)
- License: Apache-2.0