---
layout: post
title: "MemPalace: Local-First AI Memory That Hits 96.6% Recall Without a Single API Call"
description: "Learn how MemPalace stores your conversation history as verbatim text, retrieves it with 96.6% R@5 on LongMemEval using zero LLM calls, and organizes everything with an intuitive Palace metaphor of Wings, Rooms, and Drawers."
date: 2026-04-19
header-img: "img/post-bg.jpg"
permalink: /MemPalace-Local-First-AI-Memory/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - AI Agent
  - Memory System
  - Tutorial
author: "PyShine"
---

# MemPalace: Local-First AI Memory That Hits 96.6% Recall Without a Single API Call

If you use AI coding agents like Claude Code, Gemini CLI, or Cursor, you have probably experienced the same frustration: every new session starts from scratch. The agent has no memory of what you discussed yesterday, what decisions you made last week, or why you chose a particular architecture three months ago. You end up repeating context, re-explaining decisions, and losing hours to re-discovery.

**MemPalace** is an open-source, local-first AI memory system that solves this problem completely. Created by the MemPalace team and boasting over 48,000 GitHub stars, it stores your conversation history as verbatim text and retrieves it with semantic search -- achieving 96.6% R@5 on LongMemEval without requiring a single API call or cloud connection.

The core philosophy is simple but powerful: do not summarize, extract, or paraphrase. Store the original content verbatim, organize it structurally, and retrieve it with precision. This approach preserves every detail and nuance that summarization would inevitably lose.

![MemPalace Architecture](/assets/img/diagrams/mempalace/mempalace-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates how MemPalace processes and serves your AI memory. Let us break down each layer:

**Data Sources (Top Row)**

Three primary data sources feed into MemPalace. Project Files include your codebase, documentation, and configuration files. Claude Code Sessions capture the full conversation history from your AI coding sessions. CLI and API Commands allow programmatic access for scripting and automation. All three sources flow into the Miner, which handles content extraction and indexing.

**Mining and Storage (Center)**

The Miner processes raw content from all sources, extracting meaningful chunks and building the index. It feeds into The Palace, which is the structured memory store. The Palace organizes content hierarchically: Wings represent people and projects, Rooms represent topics and themes, and Drawers hold the original verbatim content. This three-level hierarchy enables scoped search rather than flat corpus retrieval.

**Backend Layer (Below Palace)**

The Palace connects to pluggable storage backends. ChromaDB is the default vector store, providing local-first semantic search. The Custom Backend option means you can swap in any vector database that implements the interface defined in `mempalace/backends/base.py` without touching the rest of the system.

**Search and Output (Bottom)**

The Search and Retrieval layer combines semantic and hybrid search across the structured hierarchy. Results flow to two destinations: AI Agents (Claude Code, Gemini CLI, MCP tools) receive context for their sessions, and the wake-up command loads relevant context at the start of each new session so your agent picks up exactly where you left off.

## The Palace Metaphor: Wings, Rooms, and Drawers

What makes MemPalace unique is its organizational structure. Instead of dumping everything into a flat vector database, it uses an intuitive palace metaphor borrowed from the ancient memory technique known as the method of loci.

![Palace Structure](/assets/img/diagrams/mempalace/mempalace-palace-structure.svg)

### Understanding the Palace Structure

The palace structure diagram shows the three-level hierarchy that organizes all your memories. Here is a detailed walkthrough:

**The Palace (Root)**

The Palace is your entire memory store. Think of it as the building that contains everything. When you run `mempalace wake-up`, the Palace determines what context to load for your current session based on recency, relevance, and your established preferences.

**Wings (Level 1: People and Projects)**

Each Wing represents a distinct person or project in your life. The diagram shows three examples: Alice (a person you collaborate with), ProjectX (a specific codebase), and Team (a group context). Wings provide the first level of scoping -- when you search within a Wing, you only see content related to that person or project. This prevents cross-contamination between unrelated contexts.

**Rooms (Level 2: Topics and Themes)**

Within each Wing, Rooms organize content by topic. Under the ProjectX Wing, you see three Rooms: API Design, Database, and Deployment. Each Room groups related conversations and decisions together. When you ask "why did we switch to GraphQL?", the search can scope to the API Design Room within the ProjectX Wing, dramatically improving precision.

**Drawers (Level 3: Verbatim Content)**

Drawers hold the original, unmodified content. Under the API Design Room, you find Meeting Notes (verbatim), Code Decisions (verbatim), and Slack Threads (verbatim). The verbatim storage principle is critical: MemPalace never summarizes or paraphrases. The exact words from your conversations and documents are preserved, ensuring no detail is lost.

**Scoped Search (Dashed Lines)**

The red dashed lines converging on the Scoped Search node illustrate that you can search at any level of the hierarchy. Search across the entire Palace for broad queries, within a Wing for project-specific context, within a Room for topic-focused results, or within a Drawer for exact content. This scoping is what enables the high recall rates without requiring an LLM.

## The Retrieval Pipeline: From Raw to Reranked

MemPalace offers a three-stage retrieval pipeline that lets you trade off between simplicity and maximum accuracy. The remarkable thing is that even the rawest mode achieves 96.6% recall without any LLM involvement.

![Retrieval Pipeline](/assets/img/diagrams/mempalace/mempalace-retrieval-pipeline.svg)

### Understanding the Retrieval Pipeline

The retrieval pipeline diagram shows the three-stage process from query to ranked results. Here is how each stage works:

**Stage 1: Raw Semantic Search (96.6% R@5)**

The pipeline starts with a user query like "Why did we switch to GraphQL?" This query enters Stage 1, which performs pure semantic search using ChromaDB vector embeddings. No LLM is involved at any point. No API key is needed. No cloud connection is required. The result is 96.6% recall at 5 on the LongMemEval benchmark with 500 questions. This is the number that matters most because it represents the zero-dependency baseline -- your memory works entirely offline with no external costs.

**Stage 2: Hybrid v4 (98.4% R@5)**

Stage 2 adds three heuristic boosters on top of raw semantic search. Keyword boosting ensures that exact term matches get a relevance lift. Temporal proximity boosting recognizes that recent conversations are more likely to be the ones you need. Preference pattern extraction learns from your past search behavior to improve future results. The hybrid v4 pipeline achieves 98.4% R@5 on held-out questions (450 questions not seen during tuning), which is the honest generalizable figure. Still no LLM required.

**Stage 3: LLM Rerank (99%+ R@5)**

For users who want maximum accuracy and are willing to use an LLM, Stage 3 takes the top-20 retrieved sessions and asks a capable model to rerank them. This works with any reasonably capable model -- Claude Haiku, Claude Sonnet, or even local models via Ollama. There is no Anthropic dependency. The project deliberately does not headline a "100%" number because the last 0.6% was reached by inspecting specific wrong answers, which the benchmark documentation flags as teaching to the test.

## Integrations: Works With Your Existing Tools

MemPalace is not a walled garden. It exposes 29 MCP tools and integrates directly with the AI agents you already use.

![Integrations](/assets/img/diagrams/mempalace/mempalace-integrations.svg)

### Understanding the Integrations

The integrations diagram shows how MemPalace connects to your existing workflow. Here is a detailed breakdown:

**MCP Server (Center)**

The MemPalace MCP Server is the central integration point, exposing 29 tools that cover palace reads and writes, knowledge-graph operations, cross-wing navigation, drawer management, and agent diaries. Any MCP-compatible client can connect to this server and access the full range of memory operations. The MCP protocol ensures that MemPalace works with any current or future agent that supports the standard.

**Agent Integrations (Top Row)**

Five primary agent integrations are shown. Claude Code has the deepest integration with a dedicated plugin plus hooks for auto-save, meaning your Claude Code sessions are automatically stored in the Palace without any manual action. Gemini CLI connects via an extension. Cursor and VS Code connect as MCP clients. Local models running through Ollama or similar tools can also access the Palace through the MCP server.

**The Palace Core (Middle)**

Behind the MCP Server sits The Palace itself with its Wings, Rooms, and Drawers hierarchy plus the Knowledge Graph. The Knowledge Graph adds temporal entity-relationship tracking with validity windows, so you can query not just what was decided but when it was valid and whether it has been superseded.

**Storage Layer (Bottom)**

ChromaDB handles vector storage for semantic search, while SQLite stores the Knowledge Graph and metadata. Both are local-first and require no external services. The pluggable backend architecture means you can replace ChromaDB with any vector store that implements the backend interface.

**Key Features (Bottom Right)**

The feature callout highlights five core capabilities: Verbatim Storage ensures no information is lost through summarization. Scoped Search enables precision retrieval within Wings, Rooms, or Drawers. Agent Diaries give each specialist agent its own wing and diary in the palace. Temporal Graph tracks when facts become valid and when they expire. Zero API Calls in raw mode means your memory works entirely offline with no external dependencies.

## Getting Started

Installation takes two commands:

```bash
pip install mempalace
mempalace init ~/projects/myapp
```

Then mine your content:

```bash
# Mine project files
mempalace mine ~/projects/myapp

# Mine Claude Code sessions
mempalace mine ~/.claude/projects/ --mode convos

# Search your memory
mempalace search "why did we switch to GraphQL"

# Load context for a new session
mempalace wake-up
```

The `wake-up` command is particularly powerful. It loads the most relevant context from your Palace into the current session, so your AI agent starts with full knowledge of past decisions, conversations, and project context.

## Knowledge Graph

Beyond the Palace hierarchy, MemPalace includes a temporal entity-relationship graph backed by local SQLite. This graph tracks not just what facts are true, but when they became true and when they stopped being true. You can add entities, query relationships, invalidate outdated facts, and view timelines. This is essential for long-running projects where architectural decisions change over time and you need to know not just the current state but the history of how you got there.

## Auto-Save Hooks

For Claude Code users, MemPalace provides two hooks that automatically save your conversation context. One saves periodically during a session, and the other saves before context compression occurs. This means your Palace is always up to date without any manual intervention. The hooks are configured in your Claude Code settings and require no ongoing maintenance.

## Benchmarks Summary

| Benchmark | Metric | Score | Notes |
|---|---|---|---|
| LongMemEval (raw, no LLM) | R@5 | 96.6% | 500 questions, zero API calls |
| LongMemEval (hybrid v4, held-out) | R@5 | 98.4% | 450 held-out questions, no LLM |
| LongMemEval (hybrid v4 + LLM rerank) | R@5 | 99%+ | Any capable model |
| LoCoMo (session, top-10) | R@10 | 60.3% | 1,986 questions |
| LoCoMo (hybrid v5, top-10) | R@10 | 88.9% | Same set |
| ConvoMem (all categories) | Avg recall | 92.9% | 250 items, 50 per category |
| MemBench (ACL 2025) | R@5 | 80.3% | 8,500 items |

All results are reproducible from the repository with the commands provided in `benchmarks/BENCHMARKS.md`.

## Requirements

- Python 3.9+
- A vector-store backend (ChromaDB by default)
- Approximately 300 MB disk for the default embedding model
- No API key required for the core benchmark path

## Why MemPalace Matters

The AI agent ecosystem has a memory problem. Every session starts blank. Every conversation is forgotten. Every decision must be re-explained. Existing solutions like Mem0, Mastra, and Zep either summarize your data (losing detail), require cloud connections (compromising privacy), or depend on LLM calls for basic retrieval (adding cost and latency).

MemPalace takes a different approach: store everything verbatim, organize it structurally, and retrieve it with precision. The result is a local-first memory system that achieves 96.6% recall without any LLM, any API key, or any cloud connection. When you need maximum accuracy, the hybrid and rerank pipelines push that to 98.4% and 99%+ respectively.

The Palace metaphor makes the system intuitive to use. Wings for people and projects. Rooms for topics. Drawers for original content. Scoped search within any level. It is the method of loci applied to AI memory, and it works.

With 29 MCP tools, integrations for Claude Code, Gemini CLI, Cursor, and VS Code, plus auto-save hooks and a temporal knowledge graph, MemPalace is not just a memory store. It is a complete memory infrastructure for the AI-first development workflow.

Check out the [MemPalace GitHub repository](https://github.com/MemPalace/mempalace) to get started.