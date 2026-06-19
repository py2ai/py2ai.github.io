---
layout: post
title: "TencentDB Agent Memory: Fully Local Long-Term Memory for AI Agents"
description: "Learn how TencentDB Agent Memory gives AI agents layered long-term memory with symbolic short-term compression. Cuts tokens by 61%, improves pass rates by 51%, fully local with SQLite."
date: 2026-06-19
header-img: "img/post-bg.jpg"
permalink: /TencentDB-Agent-Memory-Local-Long-Term-Memory-for-AI-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Memory Systems, Open Source]
tags: [TencentDB, Agent Memory, Long-term Memory, AI Agents, Local Storage, TypeScript, Open Source]
keywords: "TencentDB Agent Memory tutorial, how to install memory-tencentdb plugin, AI agent long-term memory system, TencentDB Agent Memory vs Mem0, local agent memory SQLite, OpenClaw memory plugin setup, Hermes agent memory integration, symbolic memory Mermaid canvas, L0 L1 L2 L3 memory layering, fully local AI agent memory"
author: "PyShine"
---

TencentDB Agent Memory is a fully local long-term memory system for AI agents that combines symbolic short-term memory with layered long-term memory. Built as a plugin for OpenClaw and Hermes Agent, it cuts token usage by up to 61.38%, improves task pass rates by 51.52%, and raises persona accuracy from 48% to 76% -- all running locally with SQLite and sqlite-vec, no external API required.

## The Agent Memory Problem

AI agents are powerful but forgetful. They lose context between sessions, drown in verbose tool logs, and force you to re-explain the same SOPs, project background, and output formats every single time. This repetition is not just annoying -- it wastes tokens, degrades performance, and caps how far an agent can go in a single session.

The core question TencentDB Agent Memory asks is simple: what if the Agent could remember what should be remembered, so humans can focus on judgment, creation, and work that truly matters?

With 5,689 GitHub stars and over 5,000 new stars in a single month, the community has spoken clearly -- better agent memory is one of the most pressing needs in the AI tooling ecosystem. The benchmark results back this up: a 61.38% token reduction on WideSearch, a 51.52% relative pass-rate improvement, and PersonaMem accuracy jumping from 48% to 76%.

> **Key Insight**: TencentDB Agent Memory cuts token usage by up to 61.38% while improving task pass rates by 51.52% -- measured over continuous long-horizon sessions, not isolated turns. This is not a marginal optimization; it is a structural rethink of how agents manage context.

## What is TencentDB Agent Memory?

TencentDB Agent Memory is a fully local long-term memory system for AI agents, built as a plugin for the OpenClaw and Hermes Agent frameworks. It rests on two pillars: **symbolic short-term memory** for in-task information overload, and **layered long-term memory** for cross-session experience.

The project explicitly rejects two common but flawed approaches:

- **Flat vector stores** (like traditional RAG or Mem0): These shred data into fragments and dump them into a flat vector store, where recall degenerates into a blind search across disconnected fragments with no macro-level guidance.
- **Irreversible lossy summarization**: Collapsing history into a summary saves tokens but destroys the evidence trail, making debugging impossible.

Instead, TencentDB Agent Memory builds a semantic pyramid where lower layers preserve evidence and upper layers preserve structure. Everything runs locally with SQLite and sqlite-vec as the default backend -- zero external API required, though remote OpenAI-compatible endpoints are supported for embeddings.

The project is MIT licensed, currently at version 0.3.6, published to npm as `@tencentdb-agent-memory/memory-tencentdb`, and requires Node.js >= 22.16.0. It is written in TypeScript with ESM modules, built with tsdown, and tested with Vitest.

## Architecture: The L0-L3 Semantic Pyramid

![Architecture](/assets/img/diagrams/tencentdb-agent-memory/tencentdb-agent-memory-architecture.svg)

The architecture diagram above shows the complete memory system from top to bottom. At the very top sits the Agent Context -- the only thing the LLM actually sees in its prompt window. This context contains a lightweight Mermaid canvas (condensed task state) and recalled memories injected before each turn.

Below the Agent Context, the system splits into two parallel memory tracks:

**Short-term memory** (green) handles in-task information overload through a three-layer stack. The bottom layer archives raw tool outputs to `refs/*.md` files. The middle layer extracts step-level summaries into JSONL. The top layer condenses everything into a Mermaid canvas with `node_id` references. The Agent only attends to the top-layer structure and drills down via `node_id` when an error occurs -- this is what enables the 61% token reduction.

**Long-term memory** (blue) builds the semantic pyramid: L0 Conversation (raw dialogue) feeds into L1 Atom (atomic facts extracted every N conversations), which feeds into L2 Scenario (scene blocks aggregated from atoms), which feeds into L3 Persona (user profile distilled every N new memories). A deterministic drill-down path runs in reverse: from L3 Persona back to L1 Atom when details are needed.

The storage layer (purple) uses heterogeneous storage: SQLite + sqlite-vec for the bottom layer (facts, logs, traces) enabling full-text and vector retrieval, and Markdown files for the top layer (personas, scenes, canvases) enabling human-readable white-box inspection.

The Gateway (orange) is an HTTP sidecar listening on port 8420 with endpoints for `/capture`, `/search`, `/recall`, and `/health`. It supports optional API key authentication and CORS allow-listing for network deployment.

Finally, the integration layer (red) shows the two supported frameworks: OpenClaw (zero-config plugin install) and Hermes Agent (Docker greenfield or attach to existing install).

> **Takeaway**: The L0-L3 semantic pyramid replaces flat vector piles with a hierarchical structure where recall is guided by macro-level persona and scenario context, not blind fragment search. Lower layers preserve evidence; upper layers preserve structure.

## Key Features and Capabilities

![Features](/assets/img/diagrams/tencentdb-agent-memory/tencentdb-agent-memory-features.svg)

The features diagram organizes the system's capabilities into five categories:

**Memory Architecture** (green) includes the core design patterns: Memory Layering (the L0-L3 semantic pyramid), Symbolic Memory (Mermaid canvas with `node_id` tracing), Heterogeneous Storage (SQLite DB for evidence, Markdown for structure), Full Traceability (deterministic drill-down path from persona to raw text), and Progressive Disclosure (the Agent only attends to the top layer, drilling down on demand).

**Recall and Search** (blue) provides three retrieval strategies. The default is Hybrid recall, which uses Reciprocal Rank Fusion (RRF) to combine keyword and embedding results. Keyword search uses BM25 with jieba tokenization for Chinese and standard tokenization for English. Vector search uses sqlite-vec for similarity matching. A 5000ms recall timeout ensures the conversation is never blocked -- on timeout, injection is skipped gracefully.

**Integration** (orange) covers the dual-framework support. OpenClaw gets a zero-config plugin install. Hermes Agent supports both Docker greenfield deployment and attaching to an existing install. The Gateway HTTP API on port 8420 enables multi-agent collaboration. Auto-discovery launches the Gateway via `Popen()` on the first conversation. Bearer token authentication secures all routes except `/health`.

**Pipeline** (purple) automates the memory lifecycle: Conversation Capture runs automatically on every dialogue. L1 Extraction triggers every N conversations (default 5) to extract atomic facts with deduplication. L2 Aggregation builds scene blocks from atoms with a minimum 900-second interval. L3 Persona generates the user profile every N=50 new memories. The warmup strategy starts from turn 1 and doubles each time (1 to 2 to 4).

**Configuration** (red) offers three levels of tuning. Level 1 covers 90% of use cases with daily parameters like timezone, recall strategy, and pipeline intervals. Level 2 adds advanced tuning for long tasks and sessions. Level 3 provides the full parameter reference in `openclaw.plugin.json`. Zero-config defaults mean everything works out of the box. Offload compression supports mild (0.5 ratio) and aggressive (0.85 ratio) modes. Migration tools (`migrate-sqlite-to-tcvdb`, `export-tencent-vdb`, `read-local-memory`) handle data portability. Local LLM support via `node-llama-cpp` enables fully offline extraction.

## How It Works: From Install to Recall

![Workflow](/assets/img/diagrams/tencentdb-agent-memory/tencentdb-agent-memory-workflow.svg)

The workflow diagram traces the complete journey from installation to active memory recall. Here is how each step works:

**Step 1 - Prerequisites**: Verify OpenClaw >= 2026.3.13 and Node.js >= 22.16.0 are installed. These version requirements come from the `openclaw.plugin.json` compatibility constraints.

**Step 2 - Install Plugin**: Run `openclaw plugins install @tencentdb-agent-memory/memory-tencentdb` to install the plugin from npm, then restart the gateway.

**Step 3 - Enable in Config**: Add `"memory-tencentdb": {"enabled": true}` to `~/.openclaw/openclaw.json`. This activates the plugin with zero-config defaults (SQLite + sqlite-vec backend, hybrid recall, 5 results per recall).

**Step 4 - Decision: Short-term Compression?** A decision diamond asks whether you want to enable short-term context compression. If yes, set `offload.enabled: true` and register the `contextEngine` slot so OpenClaw routes context-offload requests to this plugin. If no, you still get the full L0-L3 long-term memory pipeline.

**Step 5 - Restart Gateway**: Run `openclaw gateway restart` to apply the configuration changes.

**Step 6 - Verify**: Check for `[memory-tdai]` log entries and confirm the data directory `~/.openclaw/state/memory-tdai/` was created. Run `curl http://localhost:8420/health` to confirm the Gateway is responding.

**Step 7 - Smoke Test**: Have a 2-3 turn conversation with memorable information, then start a new turn and check whether recall injection surfaces the earlier context.

**Step 8 - Conversation Capture**: Every conversation is automatically captured to L0. No manual intervention is needed -- the `auto-capture` hook in `src/core/hooks/auto-capture.ts` handles this.

**Step 9 - L1 Extraction**: Every N conversations (default 5), the pipeline extracts atomic facts from L0 conversations. Deduplication is enabled by default to prevent redundant memories.

**Step 10 - L2 Aggregation**: Scene blocks are built from atoms with a minimum 900-second interval between passes within the same session. These are stored as human-readable Markdown files.

**Step 11 - L3 Persona**: The user profile is generated every N=50 new memories. This persona carries day-to-day preferences, voice, and long-term goals.

**Step 12 - Recall Before Next Turn**: Before the next conversation turn, hybrid search (keyword + embedding via RRF fusion) retrieves the top-5 relevant memories and injects them into the Agent context. A 5000ms timeout ensures the conversation is never blocked.

The optional short-term offload feedback loop (teal) shows how tool logs are offloaded to `refs/*.md`, a Mermaid canvas is injected into context, and `node_id` drill-down retrieves full raw text when needed.

## Use Cases

TencentDB Agent Memory shines in scenarios where agents need to maintain context over long horizons or across multiple sessions:

**Long-horizon coding tasks**: On SWE-bench, which runs 50 consecutive tasks per session to simulate real-world context accumulation pressure, the plugin improved pass rates from 58.4% to 64.2% (+9.93% relative) while cutting token usage by 33.09%. The symbolic short-term memory offloads verbose build logs and error traces into Mermaid symbols, keeping the context window lean.

**Multi-session research**: On the WideSearch benchmark, pass rates jumped from 33% to 50% (+51.52% relative) with a 61.38% token reduction. The agent remembers what it already searched, what it found, and what it still needs to explore.

**Personalized AI assistants**: PersonaMem accuracy went from 48% to 76% (+59% relative). The L3 Persona layer captures user preferences, communication style, and recurring goals -- so the assistant adapts without being re-prompted every session.

**Enterprise knowledge retention**: SOPs, project context, tool conventions, and output format preferences are captured once and reused forever. New team members benefit from accumulated organizational memory without manual knowledge transfer.

**Multi-agent collaboration**: The Gateway HTTP API on port 8420 enables multiple agents to share a common memory store. One agent captures knowledge; another recalls it. This is particularly useful for agent orchestration pipelines.

## Integration with AI Agents

TencentDB Agent Memory supports two AI agent frameworks with distinct integration paths:

### OpenClaw Integration

OpenClaw is the primary integration path with zero-config defaults:

```bash
# Install the plugin
openclaw plugins install @tencentdb-agent-memory/memory-tencentdb
openclaw gateway restart
```

Enable it in your config:

```json
// ~/.openclaw/openclaw.json
{
  "memory-tencentdb": {
    "enabled": true
  }
}
```

For optional short-term compression (requires version >= 0.3.4):

```json
{
  "memory-tencentdb": {
    "config": {
      "offload": {
        "enabled": true
      }
    }
  },
  "plugins": {
    "slots": {
      "contextEngine": "memory-tencentdb"
    }
  }
}
```

### Hermes Agent Integration

Hermes offers two paths. The Docker greenfield approach bundles `hermes-agent` and the `memory_tencentdb` provider together:

```bash
cd docker/opensource
docker build -f Dockerfile.hermes -t hermes-memory .

docker run -d \
  --name hermes-memory \
  --restart unless-stopped \
  -p 8420:8420 \
  -e MODEL_API_KEY="your-api-key" \
  -e MODEL_BASE_URL="https://api.lkeap.cloud.tencent.com/v1" \
  -e MODEL_NAME="deepseek-v3.2" \
  -e MODEL_PROVIDER="custom" \
  -v hermes_data:/opt/data \
  hermes-memory

curl http://localhost:8420/health
```

For attaching to an existing Hermes install, you download the plugin package, install dependencies, symlink to the Hermes plugin directory, and declare the provider in `config.yaml`:

```yaml
# ~/.hermes/config.yaml
memory:
  provider: memory_tencentdb
```

The Gateway auto-launches on the first conversation via `Popen()` -- no manual startup needed.

### Gateway Security

The Gateway supports optional API key authentication and CORS allow-listing:

```bash
# Set API key (all routes except /health require Bearer token)
export TDAI_GATEWAY_API_KEY="your-secret-key"

# CORS allow-list (comma-separated origins)
export TDAI_CORS_ORIGINS="https://your-app.example.com"

# Client-side: plugin sends Bearer token to Gateway
export MEMORY_TENCENTDB_GATEWAY_API_KEY="your-secret-key"
```

API key comparison uses constant-time comparison to prevent timing attacks. The `/health` endpoint stays open for orchestrator probes like Docker healthchecks and Kubernetes liveness checks.

## Performance and Scalability

The benchmark results are measured over continuous long-horizon sessions, not isolated turns. This is critical context -- SWE-bench runs 50 consecutive tasks per session to simulate the context-accumulation pressure of real-world agents:

| Memory Capability | Benchmark | OpenClaw Success | With Plugin | Relative Delta | OpenClaw Tokens | With Plugin Tokens | Relative Delta |
|---|---|---|---|---|---|---|---|
| Short-term | WideSearch | 33% | 50% | +51.52% | 221.31M | 85.64M | -61.38% |
| Short-term | SWE-bench | 58.4% | 64.2% | +9.93% | 3474.1M | 2375.4M | -33.09% |
| Short-term | AA-LCR | 44.0% | 47.5% | +7.95% | 112.0M | 77.3M | -30.98% |
| Long-term | PersonaMem | 48% | 76% | +59% | -- | -- | -- |

The token reductions are substantial across all benchmarks: 61.38% on WideSearch, 33.09% on SWE-bench, and 30.98% on AA-LCR. The pass-rate improvements compound these savings -- the agent is not just using fewer tokens, it is also succeeding more often.

The recall system is designed for production resilience: a 5000ms timeout ensures the conversation is never blocked by a slow recall. The warmup strategy triggers from turn 1 and doubles each time (1 to 2 to 4), so new sessions get immediate memory activation without overwhelming the system.

## Getting Started

### Prerequisites

- OpenClaw >= 2026.3.13
- Node.js >= 22.16.0

### Quick Install (OpenClaw)

```bash
openclaw plugins install @tencentdb-agent-memory/memory-tencentdb
openclaw gateway restart
```

### Enable in Config

```json
{
  "memory-tencentdb": {
    "enabled": true
  }
}
```

### Level 1 Daily Tuning (90% of use cases)

```json
{
  "memory-tencentdb": {
    "config": {
      "timezone": "Asia/Shanghai",
      "storeBackend": "sqlite",
      "recall": {
        "strategy": "hybrid",
        "maxResults": 5
      },
      "pipeline": {
        "everyNConversations": 5
      },
      "persona": {
        "triggerEveryN": 50
      },
      "offload": {
        "enabled": false
      }
    }
  }
}
```

### Verify the Gateway

```bash
curl http://localhost:8420/health
# Should return {"status":"ok"} or {"status":"degraded"}
```

## Comparison with Alternatives

| Approach | TencentDB Agent Memory Advantage |
|---|---|
| **Mem0 / flat vector stores** | Layered L0-L3 semantic pyramid vs flat vector pile; heterogeneous storage (DB + Markdown) vs single store; symbolic memory for short-term |
| **Traditional RAG** | Hierarchical recall guided by macro-level persona/scenario structure vs blind fragment search; full traceability vs lossy chunks |
| **Brute-force context accumulation** | 61% token reduction via Mermaid offloading vs full log injection; `node_id` drill-down preserves traceability |
| **Cloud memory services** | Fully local (SQLite + sqlite-vec) vs external API dependency; zero-config vs complex setup; no vendor lock-in |
| **Irreversible summarization** | Deterministic drill-down path from persona to raw text vs information loss; white-box Markdown vs opaque embeddings |

> **Important**: Full traceability is the design principle that sets this system apart. Compression never sacrifices the ability to drill down. A deterministic path always exists from high-level abstractions (Persona/canvas) back to ground-truth evidence (L0 Conversation/refs). Debugging becomes a walk along the chain, not a probe into an opaque database.

## Tech Stack

| Component | Technology |
|---|---|
| Language | TypeScript (ESM modules) |
| Runtime | Node.js >= 22.16.0 |
| Build | tsdown |
| Storage | SQLite + sqlite-vec (0.1.7-alpha.2) |
| LLM SDK | Vercel AI SDK (`ai` package, `@ai-sdk/openai`) |
| Tokenizer | js-tiktoken |
| Chinese tokenizer | @node-rs/jieba |
| Config | yaml, json5 |
| Validation | zod |
| HTTP | undici |
| Testing | Vitest |
| Optional LLM | node-llama-cpp (local LLM support) |
| Optional observability | opik |

The tech stack is modern and fully TypeScript-native. The Vercel AI SDK provides a unified interface for LLM calls, supporting both local models (via `node-llama-cpp`) and remote OpenAI-compatible endpoints. The `@node-rs/jieba` package handles Chinese tokenization for BM25 keyword search, making the system bilingual out of the box.

## Conclusion

TencentDB Agent Memory brings layered, symbolic memory to AI agents with measurable results: 61% token reduction, 51% pass-rate improvement, and 76% persona accuracy. It is fully local, zero-config, MIT licensed, and works with both OpenClaw and Hermes Agent frameworks.

The L0-L3 semantic pyramid replaces flat vector stores with a hierarchical structure where recall is guided by macro-level context. The symbolic short-term memory offloads verbose tool logs into Mermaid symbols with `node_id` drill-down, cutting tokens while preserving full traceability. Heterogeneous storage (SQLite for evidence, Markdown for structure) gives you both robust retrieval and white-box debuggability.

With 5,700+ stars and active development, the project is growing fast. The roadmap includes portable memory (cross-agent, cross-framework, cross-device migration), automatic skill generation, and a visual debugging dashboard.

> **Amazing**: The entire system runs locally with SQLite and sqlite-vec -- no external API required, no cloud dependency, no vendor lock-in. Zero-config defaults work out of the box. This is agent memory that respects your privacy, your budget, and your autonomy.

Get started in two commands:

```bash
openclaw plugins install @tencentdb-agent-memory/memory-tencentdb
openclaw gateway restart
```

The Agent will start remembering what should be remembered, so you can focus on what truly matters.