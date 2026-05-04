---
layout: post
title: "Stash: Persistent AI Memory System with 8-Stage Consolidation and MCP Integration"
description: "Learn how Stash gives AI agents long-term memory with episodic storage, semantic recall, 8-stage consolidation, contradiction detection, and MCP server integration. Built in Go with PostgreSQL and pgvector."
date: 2026-05-04
header-img: "img/post-bg.jpg"
permalink: /Stash-Persistent-AI-Memory-System/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Go, Developer Tools]
tags: [Stash, AI memory, persistent memory, MCP, pgvector, Go, LLM agents, semantic recall, consolidation, developer tools]
keywords: "Stash AI persistent memory system, how to give AI agents long-term memory, Stash MCP server setup, AI memory consolidation pipeline, pgvector semantic search for AI, Stash vs other AI memory tools, persistent memory for Claude Code, AI agent memory Go tutorial, contradiction detection AI memory, Stash installation guide"
author: "PyShine"
---

# Stash: Persistent AI Memory System with 8-Stage Consolidation and MCP Integration

## Introduction

Stash is a persistent AI memory system that gives LLM agents long-term memory across sessions, solving one of the most fundamental limitations of current AI architectures: statelessness. Every time an LLM agent starts a new conversation, it begins with a blank slate, unable to recall past interactions, learned facts, or accumulated knowledge. Stash changes this by providing a structured memory layer that persists between sessions, enabling AI agents to remember, recall, consolidate, and learn from their experiences over time.

Built in Go with PostgreSQL and the pgvector extension, Stash exposes a full MCP (Model Context Protocol) server with over 20 tools that any compatible AI client can use. The system implements a sophisticated memory hierarchy inspired by human cognition: raw interactions are stored as episodic memories, then progressively distilled through an 8-stage consolidation pipeline into facts, relationships, causal links, patterns, and hypotheses. This mirrors how human memory works -- we do not remember every conversation verbatim, but we extract and retain the important lessons.

The architecture addresses several critical challenges in AI memory management. Contradiction detection automatically identifies conflicting facts and resolves them based on confidence scores. Confidence decay implements a computational version of Ebbinghaus's forgetting curve, ensuring that unverified information gradually fades while well-established knowledge persists. The hypothesis lifecycle provides a structured approach to learning, moving from proposed ideas through testing to confirmed facts or rejected claims. Together, these mechanisms create a memory system that is not just a database, but an active reasoning substrate for AI agents.

> **Key Insight:** Stash's 8-stage consolidation pipeline processes memories through episodic extraction, relationship mapping, causal linking, contradiction detection, confidence decay, goal inference, failure detection, and hypothesis scanning -- all automatically and incrementally since the last checkpoint.

## How It Works

![Stash Architecture](/assets/img/diagrams/stash/stash-architecture.svg)

### Understanding the Stash Architecture

The architecture diagram above illustrates how Stash orchestrates the flow of information from raw AI interactions through progressive memory consolidation stages, ultimately exposing structured knowledge through an MCP server interface. The system is designed as a pipeline where each stage builds upon the outputs of previous stages, creating increasingly refined and abstract representations of knowledge.

**MCP Client Layer**

At the top of the architecture, AI clients such as Claude Code, Cursor, or any MCP-compatible tool connect to Stash via the MCP protocol. Stash supports both SSE (Server-Sent Events) and stdio transport modes, making it compatible with virtually any AI client that implements the MCP specification. Clients invoke one of the 20+ exposed tools to store episodes, recall facts, track goals, or query relationships. The MCP layer handles authentication, request routing, and response formatting transparently.

**Episodic Memory Store**

When an AI agent interacts with a user or system, the raw interaction is captured as an episode. Each episode contains the conversation content, metadata (timestamp, namespace, source), and an automatically generated vector embedding using OpenAI's text-embedding-3-small model (1536 dimensions). These embeddings are stored in PostgreSQL with the pgvector extension, which provides HNSW indexes for efficient cosine similarity search using the `<=>` operator. The episodic store serves as the foundation layer -- it captures everything, similar to how human episodic memory records raw experiences.

**Consolidation Pipeline**

The consolidation pipeline is the core intelligence of Stash. It runs incrementally, processing only new data since the last checkpoint, which keeps computational costs manageable. The pipeline consists of 8 stages that progressively extract higher-order knowledge from raw episodes. Each stage uses LLM-based reasoning (via gpt-4o-mini) to analyze and transform the data. The stages are: episode-to-fact extraction, relationship extraction, causal link detection, contradiction detection, confidence decay, goal progress inference, failure pattern detection, and hypothesis evidence scanning.

**Semantic Recall Engine**

When an AI agent needs to recall information, Stash uses a two-tier search strategy. First, it searches the facts store using cosine similarity, which returns distilled, high-confidence knowledge. If the facts search does not yield sufficient results, it falls back to searching the raw episodes store. This priority-based approach ensures that consolidated, verified knowledge is preferred over raw, potentially contradictory episode data. The HNSW indexes enable sub-second retrieval even across large datasets.

**PostgreSQL + pgvector Backend**

All data in Stash is stored in PostgreSQL 16 with the pgvector extension. The database schema includes tables for episodes, facts, relationships, causal links, goals, failures, hypotheses, and working context. Goose migrations (20 files) manage schema evolution. The pgvector extension provides native vector operations and HNSW indexing for similarity search. An embedding cache backed by pgx avoids re-embedding identical text, reducing API costs and latency for repeated queries.

## The 8-Stage Consolidation Pipeline

![Consolidation Pipeline](/assets/img/diagrams/stash/stash-consolidation-pipeline.svg)

### Understanding the Consolidation Pipeline

The consolidation pipeline is the most distinctive feature of Stash, implementing a multi-stage knowledge distillation process that transforms raw episodic memories into structured, verified knowledge. Each stage operates incrementally, processing only new or updated data since the last consolidation checkpoint, which ensures efficiency and avoids redundant computation.

**Stage 1: Episodes to Facts**

The first stage clusters related episodes using cosine similarity (threshold of 0.85) and extracts factual statements from them. When multiple episodes discuss the same topic, the system identifies common assertions and creates fact records. Deduplication ensures that the same fact is not stored multiple times. Each extracted fact receives an initial confidence score and is tagged with its source episodes for traceability. This stage effectively compresses verbose conversations into concise, actionable knowledge units.

**Stage 2: Facts to Relationships**

The second stage analyzes facts to identify subject-predicate-object triples, creating a knowledge graph structure. For example, if a fact states "Redis is used for caching in the application," the relationship extractor creates a triple: (Redis, used_for, caching). These relationships enable graph traversal queries and provide context for understanding how different pieces of knowledge connect to each other. The system uses LLM-based reasoning to identify relationship types and validate their semantic coherence.

**Stage 3: Facts to Causal Links**

The third stage detects cause-and-effect relationships between facts. Unlike simple correlations, causal links represent directional influence: one fact causes another. Stash uses recursive CTEs (Common Table Expressions) in PostgreSQL to support forward and backward causal chain traversal. This means an AI agent can ask "what caused X?" (backward traversal) or "what are the consequences of X?" (forward traversal), enabling sophisticated reasoning about system behavior and decision outcomes.

**Stage 4: Contradiction Detection**

The fourth stage identifies conflicting facts and classifies them into three categories: replacement (a newer fact supersedes an older one), contradiction (two facts genuinely conflict and cannot both be true), and compatible (apparent conflict that is actually context-dependent). When the system detects a contradiction with confidence of 0.9 or higher, it automatically supersedes the older fact. Below that threshold, both facts are retained with their confidence scores, allowing the AI agent to make informed decisions about which to trust.

> **Amazing:** Stash's contradiction detection automatically classifies conflicts as replacement, contradiction, or compatible, and auto-supersedes older facts when confidence reaches 0.9 or higher -- implementing a computational model of belief revision that mirrors how humans update their understanding.

**Stage 5: Confidence Decay**

The fifth stage implements a computational model of Ebbinghaus's forgetting curve. Facts lose confidence over time with a decay factor of 0.95 per consolidation window. When a fact's confidence drops below the 0.1 threshold, it is marked as expired and excluded from recall results. However, facts that are repeatedly referenced or reinforced have their confidence restored, creating a natural mechanism where frequently used knowledge persists while unused knowledge gradually fades. This prevents the knowledge base from growing unbounded with stale information.

**Stage 6: Goal Progress Inference**

The sixth stage examines facts and episodes to infer progress on tracked goals. Stash supports hierarchical goals where parent goals automatically complete when all sub-goals are finished. The system analyzes recent interactions to detect goal-relevant actions, completions, or blockers, and updates goal status accordingly. This enables AI agents to maintain long-running objectives across sessions without manual status updates.

**Stage 7: Failure Pattern Detection**

The seventh stage identifies patterns in failed attempts and records them as failure memories. Each failure record includes what was attempted, why it failed, and what alternative approach should be tried instead. This anti-knowledge mechanism prevents AI agents from repeating the same mistakes. When an agent encounters a similar situation in the future, the failure memory is recalled alongside positive knowledge, providing guidance on what to avoid.

**Stage 8: Hypothesis Evidence Scanning**

The eighth and final stage scans for evidence that supports or refutes active hypotheses. Stash implements a full hypothesis lifecycle: proposed, testing, confirmed, and rejected. When sufficient supporting evidence accumulates, a hypothesis is confirmed and automatically creates a new Fact record. When evidence consistently contradicts a hypothesis, it is rejected. This implements a simplified version of the scientific method, enabling AI agents to form, test, and validate theories about the domains they operate in.

## Memory Model

![Memory Model](/assets/img/diagrams/stash/stash-memory-model.svg)

### Understanding the Stash Memory Model

The memory model diagram illustrates how Stash organizes knowledge into a hierarchical structure that mirrors human cognitive memory systems. This is not a flat key-value store; it is a multi-layered knowledge architecture where information flows from raw experiences through progressively refined abstractions, each layer providing different guarantees and access patterns.

**Episodic Memory Layer**

The base layer stores raw interactions as episodes. Each episode captures the full context of an interaction: who said what, when, in what namespace, and with what metadata. Episodes are the most granular memory type and retain the richest contextual information. They are embedded as 1536-dimensional vectors using OpenAI's text-embedding-3-small model, enabling semantic similarity search. However, episodes are also the most verbose and least processed memory type -- they contain noise alongside signal, just as human episodic memories contain irrelevant details alongside important information.

**Semantic Memory Layer**

The semantic layer stores distilled facts extracted from episodes through the consolidation pipeline. Facts are concise, declarative statements that represent verified knowledge. Each fact has a confidence score, a creation timestamp, a last-reinforced timestamp, and source episode references. Facts are also vector-embedded for similarity search. The key difference from episodes is that facts have been validated, deduplicated, and structured. They represent the "what I know" layer of memory, stripped of conversational context but enriched with confidence metadata.

**Relational Memory Layer**

The relational layer stores subject-predicate-object triples that connect facts into a knowledge graph. Relationships encode how entities relate to each other: "X depends on Y," "X is a subtype of Y," "X causes Y." This layer enables graph traversal queries and provides the structural context that makes individual facts meaningful in relation to each other. Without the relational layer, facts would exist in isolation; with it, they form an interconnected web of knowledge that supports reasoning about complex systems.

**Causal Memory Layer**

The causal layer stores directed cause-effect links between facts. Unlike relationships, which can be any type of association, causal links specifically represent influence: one fact causes another. This layer supports forward chaining (predicting effects from causes) and backward chaining (identifying causes from observed effects) using recursive CTEs in PostgreSQL. Causal memory is particularly valuable for debugging, root cause analysis, and decision-making, where understanding why something happened is as important as knowing what happened.

**Working Context Layer**

The working context layer provides TTL-based short-term memory for session-specific information. Unlike the persistent layers above, working context entries automatically expire after their time-to-live elapses. This is useful for storing temporary state, session preferences, or context that is relevant now but should not persist indefinitely. Working context acts as the "scratchpad" of the memory system, analogous to human working memory that holds information temporarily during active problem-solving.

**Namespace Hierarchy**

All memory types are organized within a path-based namespace hierarchy. Namespaces like `/self` store agent-specific knowledge, `/projects/myapp` store project-specific knowledge, and so on. When querying, namespace matching includes descendants, so a query on `/projects` also returns facts from `/projects/myapp` and `/projects/myapp/backend`. This hierarchical organization enables multi-tenant usage and scoped knowledge management, allowing a single Stash instance to serve multiple agents or projects without cross-contamination.

> **Takeaway:** By organizing memory into namespaces like `/self` and `/projects/myapp`, a single Stash instance can serve multiple AI agents or projects simultaneously, with descendant matching ensuring scoped queries return relevant results from all sub-namespaces.

## MCP Integration

![MCP Integration](/assets/img/diagrams/stash/stash-mcp-integration.svg)

### Understanding MCP Integration

The MCP integration diagram shows how Stash exposes its memory capabilities through the Model Context Protocol, enabling any MCP-compatible AI client to access persistent memory without custom integration code. MCP is an open protocol that standardizes how AI applications connect to external tools and data sources, and Stash leverages it to provide a universal memory interface.

**Transport Modes**

Stash supports two MCP transport modes: SSE (Server-Sent Events) and stdio. The SSE mode runs Stash as a standalone HTTP server that clients connect to over the network, making it suitable for multi-client deployments and remote access. The stdio mode runs Stash as a subprocess of the AI client, communicating through standard input and output streams, which is simpler for single-client local setups. Both modes expose the same set of tools and produce identical results; the choice depends entirely on the deployment architecture.

**Tool Categories**

The 20+ MCP tools exposed by Stash are organized into functional categories. Memory tools handle storing and recalling episodes and facts. Relationship tools manage the knowledge graph. Causal tools query cause-effect chains. Goal tools track objectives and progress. Hypothesis tools manage the hypothesis lifecycle. Failure tools record and query failure patterns. Working context tools manage short-term session memory. Consolidation tools trigger and monitor the pipeline. Namespace tools manage the organizational hierarchy. This categorization makes it easy for AI agents to discover and use the right tools for their needs.

**Client Compatibility**

Because MCP is an open standard, Stash works with any compatible client. This includes Claude Code, Cursor, Windsurf, Cline, and any other tool that implements the MCP client specification. The client simply needs to be configured with the Stash server URL (for SSE mode) or command (for stdio mode), and it automatically gains access to all memory capabilities. No custom plugins, API wrappers, or client-specific code is required. This vendor-neutral approach ensures that Stash remains useful even as the AI tooling landscape evolves.

**OpenAI-Compatible Embedding**

Stash uses OpenAI's embedding API by default but is designed to work with any OpenAI-compatible endpoint. This includes OpenRouter, Ollama, local model servers, and any service that implements the OpenAI API format. The embedding model, dimensions, and base URL are all configurable through environment variables. This abstraction makes Stash vendor-agnostic: you can switch embedding providers without changing any application code, and you can run entirely on local models for privacy-sensitive deployments.

**Prometheus Metrics**

Stash exposes Prometheus metrics for monitoring memory usage, consolidation performance, API latency, and embedding cache hit rates. These metrics enable operations teams to track the health and efficiency of the memory system, identify bottlenecks, and plan capacity. The metrics endpoint is available alongside the MCP server, making it easy to integrate into existing observability infrastructure.

> **Important:** Stash exposes 20+ MCP tools across memory, relationship, causal, goal, hypothesis, failure, and working context categories, making it compatible with Claude Code, Cursor, Windsurf, and any MCP-compatible client without custom integration code.

## Installation

### Docker Compose (Recommended)

The easiest way to get started with Stash is using Docker Compose, which sets up both PostgreSQL with pgvector and the Stash server:

```bash
# Clone the repository
git clone https://github.com/alash3al/stash.git
cd stash

# Copy the example environment file
cp .env.example .env

# Edit .env with your OpenAI API key and configuration
# Required: STASH_OPENAI_API_KEY
# Optional: STASH_OPENAI_BASE_URL (for OpenRouter, Ollama, etc.)

# Start the services
docker compose up -d
```

The Docker Compose configuration includes a pgvector-enabled PostgreSQL container and the Stash server built with a multi-stage Dockerfile that produces a distroless image. The `.env` file controls all configuration including the OpenAI API key, embedding model, and database connection parameters.

### Build from Source

For development or customization, you can build Stash from source:

```bash
# Clone the repository
git clone https://github.com/alash3al/stash.git
cd stash

# Build the CLI binary
go build -o stash ./cmd/cli

# Ensure PostgreSQL 16 with pgvector is running, then:
./stash serve
```

### Configure Your AI Client

After Stash is running, configure your MCP-compatible AI client to connect to it. For Claude Code, add the following to your MCP configuration:

```json
{
  "mcpServers": {
    "stash": {
      "url": "http://localhost:8080/sse"
    }
  }
}
```

For stdio mode, use:

```json
{
  "mcpServers": {
    "stash": {
      "command": "stash",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STASH_OPENAI_API_KEY` | OpenAI API key for embeddings and reasoning | Required |
| `STASH_OPENAI_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `STASH_EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `STASH_EMBEDDING_DIMS` | Embedding dimensions | `1536` |
| `STASH_REASONING_MODEL` | LLM model for consolidation reasoning | `gpt-4o-mini` |
| `STASH_DB_DSN` | PostgreSQL connection string | `postgres://stash:stash@localhost:5432/stash?sslmode=disable` |
| `STASH_SERVER_ADDR` | Server listen address | `:8080` |
| `STASH_TRANSPORT` | MCP transport mode (sse or stdio) | `sse` |

## Features

| Feature | Description |
|---------|-------------|
| Episodic Memory | Store raw interactions as episodes with vector embeddings for semantic search |
| Semantic Recall | Two-tier cosine similarity search: facts first, then episodes as fallback |
| 8-Stage Consolidation | Automatic pipeline distills episodes into facts, relationships, causal links, and more |
| Contradiction Detection | Auto-detects conflicting facts, classifies as replacement/contradiction/compatible, auto-supersedes at >=0.9 confidence |
| Hypothesis Lifecycle | Structured flow: proposed -> testing -> confirmed (creates Fact) -> rejected |
| Goal Tracking | Hierarchical goals with auto-completion when all sub-goals finish |
| Failure Memory | Records what did not work, why, and what to do instead |
| Causal Chain Tracing | Recursive CTEs for forward/backward causal chain traversal |
| Confidence Decay | 0.95 factor per window, facts expire below 0.1 threshold (Ebbinghaus forgetting curve) |
| Namespace Hierarchy | Path-based organization (/self, /projects/myapp) with descendant matching |
| Working Context | TTL-based short-term memory for session-specific context |
| MCP Server | 20+ tools exposed via SSE or stdio, compatible with all major AI clients |
| Embedding Cache | pgx-backed cache to avoid re-embedding identical text |
| OpenAI-Compatible | Works with OpenAI, OpenRouter, Ollama, any OpenAI-compatible endpoint |
| Prometheus Metrics | Built-in metrics for monitoring memory usage and performance |
| Docker Support | Multi-stage build producing distroless image, Docker Compose for easy setup |

## Troubleshooting

### PostgreSQL pgvector Extension Not Available

If you see errors related to the pgvector extension, ensure your PostgreSQL instance has the extension installed and enabled:

```sql
-- Check if pgvector is installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- If not installed, create it
CREATE EXTENSION IF NOT EXISTS vector;
```

For Docker deployments, use the `pgvector/pgvector:pg16` image which includes the extension pre-installed. The Docker Compose file in the Stash repository uses this image by default.

### Embedding API Errors

If you encounter errors when generating embeddings, verify the following:

- Your `STASH_OPENAI_API_KEY` is valid and has sufficient quota
- The `STASH_OPENAI_BASE_URL` is correct if using a non-OpenAI provider
- The `STASH_EMBEDDING_MODEL` matches a model available at your configured endpoint
- Network connectivity to the API endpoint is working

For local deployments using Ollama, ensure the embedding model is pulled before starting Stash:

```bash
# Pull the embedding model in Ollama
ollama pull nomic-embed-text

# Configure Stash to use Ollama
# Set STASH_OPENAI_BASE_URL=http://localhost:11434/v1
# Set STASH_OPENAI_API_KEY=ollama
# Set STASH_EMBEDDING_MODEL=nomic-embed-text
```

### Consolidation Pipeline Not Running

The consolidation pipeline runs incrementally based on checkpoints. If it appears stuck:

- Check that new episodes have been stored since the last consolidation
- Verify the reasoning model (`STASH_REASONING_MODEL`) is accessible and has quota
- Check the Stash logs for LLM API errors during consolidation stages
- Ensure the database connection is stable and not timing out during long consolidation runs

### Memory Recall Returns No Results

If semantic recall returns empty results:

- Verify that episodes or facts have been stored in the target namespace
- Check that the query embedding was generated successfully
- Ensure the HNSW indexes have been built (they are created automatically by migrations)
- Try a broader namespace query (e.g., `/projects` instead of `/projects/myapp`) since descendant matching applies
- Check if confidence decay has expired the facts (confidence below 0.1)

### Docker Compose Connection Issues

If the Stash container cannot connect to PostgreSQL:

- Ensure both containers are on the same Docker network
- Verify the `STASH_DB_DSN` uses the correct hostname (the PostgreSQL service name in Docker Compose)
- Check that PostgreSQL is healthy before Stash starts (the Docker Compose file includes a health check)
- Inspect container logs: `docker compose logs stash` and `docker compose logs postgres`

## Conclusion

Stash represents a significant step forward in AI agent infrastructure by providing a structured, persistent memory system that goes far beyond simple key-value storage. Its 8-stage consolidation pipeline implements a computational model of human memory formation, transforming raw experiences into verified knowledge through fact extraction, relationship mapping, causal linking, and contradiction resolution. The confidence decay mechanism based on Ebbinghaus's forgetting curve ensures that the knowledge base remains relevant and does not grow unbounded with stale information.

The MCP server integration makes Stash immediately accessible to any compatible AI client without custom integration code. With 20+ tools covering memory, relationships, causality, goals, hypotheses, and failures, AI agents gain a comprehensive memory substrate that supports not just recall but active reasoning. The OpenAI-compatible abstraction layer ensures vendor neutrality, allowing deployments with OpenAI, OpenRouter, Ollama, or any compatible endpoint.

For developers building AI agent systems, Stash provides the missing piece that enables agents to learn from experience, avoid repeating mistakes, track long-running goals, and build increasingly sophisticated models of the domains they operate in. The namespace hierarchy supports multi-tenant deployments, and the Prometheus metrics integration enables production-grade monitoring.

**Links:**

- GitHub: [https://github.com/alash3al/stash](https://github.com/alash3al/stash)