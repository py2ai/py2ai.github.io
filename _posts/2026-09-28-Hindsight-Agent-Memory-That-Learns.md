---
layout: post
title: "Hindsight: Agent Memory That Learns - Inside vectorize-io/hindsight"
description: "Hindsight is an open-source agent memory system built to make agents learn, not just remember. A source-level tour of vectorize-io/hindsight: the retain/recall/reflect operations, four-way parallel retrieval with rank fusion, observations and mental models, memory banks, the provider layer, and the built-in MCP endpoints."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /Hindsight-Agent-Memory-That-Learns/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/hindsight/vectorize-hindsight-architecture.svg
tags:
  - Python
  - AI Agents
  - Memory
  - PostgreSQL
  - Open Source
categories: [AI, Open Source]
keywords: "Hindsight, vectorize-io hindsight, agent memory, LongMemEval benchmark, retain recall reflect, agent memory system, mental models, observations, memory banks, pgvector, MCP server, LiteLLM, cross-encoder reranking, reciprocal rank fusion, hindsight-api"
author: "PyShine"
---

Every conversation with an agent starts from zero. You explain your project, your preferences, your constraints - and next session the agent knows none of it. Most attempts to fix this bolt a vector database onto the chat loop and call it memory: embeddings go in, similar chunks come out. That is recall of text, not learning. [Hindsight](https://github.com/vectorize-io/hindsight), an open-source memory system from Vectorize.io, takes a different position: memory should be a structured, evolving model of the world the agent operates in - facts, experiences, consolidated beliefs, and synthesized understanding - with retrieval that knows about time, causality, and entity relationships, not just cosine similarity.

The project backs the claim with results: state-of-the-art performance on the LongMemEval benchmark, with numbers independently reproduced by researchers at the Virginia Tech Sanghani Center and The Washington Post, and live per-model accuracy, latency and cost published on its benchmark site. But what makes Hindsight worth reading is the architecture itself. It is a real memory *system* - a server with an engine that extracts, normalizes, links, and consolidates; four retrieval strategies that race in parallel; and a reasoning loop that can form new beliefs about what it knows.

This post walks through the repository as it is actually implemented: the clients and entry points, the three operations every memory system needs (retain, recall, reflect), what happens inside each one, the background worker that turns raw facts into evidence-backed observations and self-rewriting mental models, and the operational surface - banks, webhooks, admin tooling, and per-bank MCP endpoints.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/hindsight/vectorize-hindsight-overview-architecture.svg" alt="Architecture overview of the vectorize-io/hindsight repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview: SDKs, a Rust CLI, an embedded in-process mode, and a web dashboard all talk to one API server, which delegates to the memory engine; the engine uses LLM providers for extraction and reasoning, runs a background worker, and persists everything to PostgreSQL.*

Reading the overview from left to right: four front doors lead into the same server. The Python, Node.js and Go SDKs and a Rust CLI speak REST over the network; the `hindsight-all` package boots the entire stack in-process for applications that want memory without running a server; and the control plane dashboard manages banks, operations, and metrics visually. Inside, the API server exposes both REST routes and Model Context Protocol endpoints - one MCP server per memory bank, enabled by default - so coding agents can use memory as tools. Everything routes into the memory engine, which leans on a provider layer covering 25+ LLM backends for the extraction and reasoning work, offloads consolidation and refresh jobs to a background worker, and stores all state in PostgreSQL with pgvector. Keep that shape in mind; the rest of the post zooms into each box.

## Why You Need This

If your agent touches the same users, projects, or tasks more than once, memory is not optional - it is the difference between an assistant and a tool you have to re-brief daily. But the common implementations have structural problems. Pure vector search retrieves text that *looks like* your query, which fails exactly when phrasing drifts: "Alice got promoted" does not match "Who is the senior engineer on the platform team?" unless something upstream normalized both into the same facts and entities. Time is another blind spot - "what happened in June?" is a filter question, not a similarity question. And conversation-history replay bloats context windows with stale text instead of consolidated knowledge.

Hindsight attacks each of these in the data model rather than in prompts. Memories are extracted into typed structures - world facts, the agent's own experiences, observations consolidated from many memories, and mental models synthesized on top - and represented as entities, relationships, and time series alongside dense and sparse vectors. Recall then runs four genuinely different searches in parallel and fuses them. The result the project advertises, and benchmarks corroborate, is that agents answer questions about *what they have learned* rather than merely *what was recently said*.

The second reason is operational shape. Memory is stateful, and stateful services fail in boring ways: schema drift, stuck jobs, secrets leaking into storage. Hindsight treats those as first-class problems - migrations ship with the server, a background worker has watchdogs and liveness probes, an admin CLI handles bank repair, and an opt-in Memory Defense policy scans every write against 45 secret/PII patterns to redact or block before storage. That is production posture, not a demo.

Finally, the ecosystem bet: 60+ integrations from Claude Code and Cursor to LangGraph and n8n, SDKs in three languages, a 2-line LLM wrapper that adds memory to an existing OpenAI or Anthropic client, and MCP endpoints built in. Memory becomes infrastructure you can swap in, not a rewrite.

## How It Works

The diagram below maps the real subsystems of the repository and how they connect, from the entry points through the three memory operations to storage and the operational edges.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/hindsight/vectorize-hindsight-architecture.svg" alt="Detailed architecture of Hindsight: entry points, HTTP and MCP layers, retain path, four-way recall path, reflect agent, background worker, providers, and storage operations" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: SDKs, CLI, embedded mode and dashboard feed the server, which dispatches retain, recall and reflect through the memory engine; retain extracts and resolves, recall races four strategies through rank fusion and a cross-encoder, reflect reasons with tools, and a worker consolidates observations and refreshes mental models in the background.*

### Understanding the Architecture

**The core package layout.** The heart of the system lives in `hindsight-api-slim/hindsight_api/`, a Python package whose `engine/` directory carries the memory machinery: `memory_engine.py` orchestrates operations, `retain/`, `search/`, `reflect/` and `consolidation/` implement them, and `providers/` wraps the LLM backends. The `hindsight-api` package you install from PyPI is a thin metapackage over this core, and `hindsight-all` adds `hindsight/embedded.py`, which boots the same server inside your process - so embedded mode and distributed mode are literally the same code path. First-run storage uses `pg0`, a zero-configuration embedded PostgreSQL, with external PostgreSQL or Oracle AI Database 23ai as drop-in upgrades for production.

**Retain: from text to structure.** The retain operation (`engine/retain/`) is where raw input becomes memory. `fact_extraction.py` sends the content through an LLM to pull out facts, temporal data, entities and relationships; `entity_processing.py` normalizes mentions into canonical entities so "Alice", "A. Chen" and a user ID resolve to one node; `embeddings.py` produces the dense and sparse vector representations that later recall will match against; and `chunk_storage.py` keeps the original documents searchable. Input language is detected and preserved end to end, so facts stay in their original language and entities keep their native script rather than being transliterated. Retain is asynchronous by design - writes are queued, and the background worker chews through them.

**Recall: four searches in parallel.** On the read path (`engine/search/`), `retrieval.py` fans a query out to four strategies simultaneously: semantic vector similarity, BM25 keyword matching, graph traversal over entity/temporal/causal links, and temporal range filtering - with `query_analyzer.py` parsing intent and time expressions like "last June" up front. `fusion.py` merges the result lists using reciprocal rank fusion, then `reranking.py` applies a cross-encoder (`cross_encoder.py`) to rescore the merged candidates before the token budget trims them. The point of racing four imperfect retrievers and fusing them is robustness: a paraphrase that defeats the vectors still hits the graph, an exact term still hits BM25, and a date-anchored question still hits the temporal filter.

**Reflect: reasoning, not lookup.** `reflect/agent.py` implements a genuine agent loop over the memory bank: it gets a toolset (`reflect/tools.py`) to search and read memories, iterates - search, read, think, search again - and produces answers that connect memories rather than quoting one of them. This is the operation for questions like "why do my outreach messages get responses?" where the answer is a synthesis across many scattered facts.

**Observations and mental models.** Two background layers turn accumulated facts into knowledge. The consolidator (`engine/consolidation/consolidator.py`) merges related facts into observations - deduplicated beliefs that keep their supporting evidence as exact quotes with a proof count, and that get *refined* rather than overwritten when new evidence arrives. On top of those, `mental_model_refresh.py` maintains standing answers to questions you define once ("What are this user's preferences?"): Hindsight writes the answer, stores it, and rewrites it in the background as the bank learns. Reading a mental model is a database read - no retrieval, no LLM call - so an agent can boot each session with a page of settled knowledge instead of rediscovering it.

**Providers and operations.** The `engine/providers/` directory wraps 25+ LLM backends - hosted APIs, fully local runtimes like Ollama and llama.cpp, and existing subscriptions (ChatGPT Plus/Pro, Claude Pro/Max, Cursor, GitHub Copilot) that need no API key - with `litellm_llm.py` routing the long tail through LiteLLM. On the operational side, `alembic/` carries schema migrations, `webhooks/` emits lifecycle events for retain, consolidation and refresh, and `admin/` provides the CLI for migrations, bank repair and stuck operations. Every server also ships per-bank MCP endpoints (`api/mcp.py`), so any MCP client can use retain, recall and reflect as tools without writing integration code.

**A memory in flight.** Follow one statement - "Alice got promoted to senior engineer" - through the boxes. The SDK posts it to a bank's retain endpoint; the server queues it and returns. The worker picks it up, the extraction LLM pulls out the fact, the temporal data and the entities; entity resolution links this Alice to the one from last month; embeddings make it findable by meaning; storage persists the fact with its evidence. Weeks later, "Who is the senior engineer?" triggers recall, the four strategies race, the fusion ranks the retained fact high, and the reranker confirms it. And if you defined a mental model for "team roster", the background refresh has already folded this fact into a standing answer your agent reads at boot - no query required.

## Advantages

- **Structure over similarity.** Facts, entities, relationships and time series are first-class - retrieval inherits that structure instead of guessing it from text at query time.
- **Four retrievers, one answer.** Semantic, keyword, graph and temporal searches run in parallel and are fused with reciprocal rank fusion plus cross-encoder reranking, so no single failure mode sinks a query.
- **Learning, not hoarding.** The consolidation layer compresses many facts into evidence-backed observations that are refined over time, and mental models expose them as zero-latency standing answers.
- **Disposition-aware reasoning.** Banks carry traits like skepticism, literalism and empathy that shape how reflect reasons over their memories, and isolation is strict - no cross-bank leakage.
- **Boring-ops first.** Migrations, liveness probes, Prometheus metrics, webhooks, an admin CLI and opt-in Memory Defense redaction of secrets and PII are all in the box.
- **Same code everywhere.** Embedded in-process mode, self-hosted Docker, Kubernetes via Helm, or the managed cloud all run the same engine, and MCP endpoints ship built in.

## Benefits

- **Agents that improve with use.** The longer a bank lives, the better its observations and mental models get - session boot costs stay flat while knowledge compounds.
- **No re-briefing tax.** Per-project memory for coding agents is one install: the coding-agents integration builds a bank from git history and past sessions automatically and injects it as the agent starts.
- **Two lines to try it.** The LLM wrapper (`wrap_openai` / `wrap_anthropic`) adds retain-and-recall around an existing client with no other code changes - memory before the call, retention after it.
- **Privacy and multilingual defaults.** Secrets and PII can be redacted or blocked before storage, and non-English memories keep their original language and script end to end.
- **Escape hatch from vector-only designs.** If your current RAG memory misses paraphrases, dates, or relationship questions, the four-way retrieval model is the concrete alternative to evaluate against - with public benchmarks to compare on.

## Usage

Start a server with Docker (embedded PostgreSQL included):

```bash
export OPENAI_API_KEY=sk-xxx

docker run -it --pull always --name hindsight --restart unless-stopped -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v hindsight-data:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

Or bare metal with pip, or Kubernetes with Helm:

```bash
pip install hindsight-api
export HINDSIGHT_API_LLM_API_KEY=sk-xxx
hindsight-api
```

Then use the three operations from Python:

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

# Retain: store information
client.retain(bank_id="my-bank", content="Alice works at Google as a software engineer")

# Recall: search memories
results = client.recall(bank_id="my-bank", query="What does Alice do?")

# Reflect: disposition-aware reasoning over memories
answer = client.reflect(bank_id="my-bank", query="What should I know about Alice?")
```

Node.js and Go SDKs follow the same shape, and a fully embedded mode needs no server at all:

```python
import os
from hindsight import HindsightServer, HindsightClient

with HindsightServer(
    llm_provider="openai",
    llm_model="gpt-5-mini",
    llm_api_key=os.environ["OPENAI_API_KEY"]
) as server:
    client = HindsightClient(base_url=server.url)
    client.retain(bank_id="my-bank", content="Alice works at Google")
```

The fastest integration for an existing agent is the LLM wrapper - swap your client and memory happens automatically around every call:

```python
from openai import OpenAI
from hindsight_litellm import wrap_openai

client = wrap_openai(
    OpenAI(),
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

# Recalls relevant memories before the call and retains the conversation after it
response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "What do you know about me?"}],
)
```

For CLI coding agents, one package wires long-term project memory into a dozen tools with no setup command:

```bash
npx @vectorize-io/hindsight-coding-agents install all
```

And any MCP client can talk to a bank directly - every server exposes one MCP endpoint per bank:

```
http://localhost:8888/mcp/{bank_id}/
```

## Conclusion

Hindsight's bet is that agent memory is a data-modeling problem before it is a retrieval problem, and the repository is that bet executed consistently: extraction turns text into typed structure, four parallel retrieval strategies cover each other's blind spots, a background worker compresses history into evidence-backed observations and self-updating mental models, and the operational surface - banks, webhooks, migrations, admin tooling, MCP endpoints - treats memory as the production service it is. If your agents repeat themselves, forget users, or re-derive what they knew last week, clone the repository, run one bank through retain-recall-reflect, and compare the answers against your current vector store. The benchmark site publishes the numbers; your own traffic will tell you the rest.

**Links:**

- Repository: [https://github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)
- Documentation: [https://hindsight.vectorize.io](https://hindsight.vectorize.io)
- Benchmarks: [https://benchmarks.hindsight.vectorize.io/](https://benchmarks.hindsight.vectorize.io/)
- PyPI: [https://pypi.org/project/hindsight-api/](https://pypi.org/project/hindsight-api/)
