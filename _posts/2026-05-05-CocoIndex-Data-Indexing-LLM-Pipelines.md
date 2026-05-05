---
layout: post
title: "CocoIndex: Incremental Data Indexing Framework for LLM Pipelines and AI Agents"
description: "CocoIndex is an open-source Python framework with a Rust core that provides incremental data indexing for LLM pipelines, RAG systems, and AI agents with delta-only reprocessing and end-to-end lineage."
date: 2026-05-05
header-img: "img/post-bg.jpg"
permalink: /CocoIndex-Data-Indexing-LLM-Pipelines/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI, Data Engineering]
tags: [cocoindex, incremental-indexing, rag, llm-pipelines, data-engineering, vector-database, python, rust, ai-agents, knowledge-graph]
keywords: "cocoindex incremental data indexing, LLM pipeline framework, RAG incremental indexing, AI agent data pipeline, delta-only data processing, Python Rust data framework, vector database indexing, knowledge graph pipeline"
author: "PyShine"
---

## What is CocoIndex?

CocoIndex is an open-source incremental data indexing framework designed to keep AI agents and LLM applications supplied with continuously fresh context. Built with a Python SDK backed by a high-performance Rust core engine, CocoIndex transforms codebases, meeting notes, PDFs, Slack messages, and other data sources into live, always-up-to-date indexes -- recomputing only the delta on every change. The project has rapidly gained traction on GitHub, earning over 160 stars per day at its peak, reflecting the strong demand for production-grade incremental data pipelines in the AI ecosystem.

The core philosophy behind CocoIndex is simple yet powerful: you declare *what* your target state should look like, and the engine keeps it in sync with the latest source data and code -- forever, at minimal cost. Think of it as React for data engineering. Just as React re-renders only the components that changed, CocoIndex reprocesses only the data that changed.

## Architecture Overview

![CocoIndex Architecture](/assets/img/diagrams/cocoindex/cocoindex-architecture.svg)

The CocoIndex architecture follows a layered design that separates the user-facing Python API from the high-performance Rust core engine. At the top layer, developers write declarative Python applications using the `@coco.fn` decorator and `coco.App` configuration. These declarations flow through the CocoIndex Python SDK, which provides the public API for mounting sources, declaring targets, and orchestrating processing components.

The Rust core engine sits beneath the Python SDK, connected via PyO3 bindings. This engine is responsible for the heavy lifting: change detection, fingerprint comparison, memoization, task scheduling, and version tracking. The memoization subsystem caches results by computing a hash of both the input data and the transformation code, meaning that if neither the source data nor the code has changed, the engine skips re-execution entirely. The lineage tracking subsystem records end-to-end provenance, so every target row, vector, or graph node traces back to its exact source byte. The task scheduler executes processing components in parallel by default, enabling CocoIndex to scale from a single repository to petabyte-scale data stores.

On the periphery, source connectors ingest data from systems like LocalFS, Amazon S3, Google Drive, PostgreSQL, and Apache Kafka. Target connectors write processed data to stores including pgvector, LanceDB, Neo4j, Qdrant, SurrealDB, and Kafka. This connector architecture makes CocoIndex adaptable to virtually any data pipeline scenario.

> **Key Insight**: The Rust core is not an afterthought -- it is the foundation. Parallel chunking, zero-copy transforms where possible, and failure isolation ensure that one bad record never stalls the entire flow. Production-grade from day zero.

## Connector Ecosystem

![CocoIndex Ecosystem](/assets/img/diagrams/cocoindex/cocoindex-ecosystem.svg)

CocoIndex ships with a rich connector ecosystem that spans both source and target systems. On the source side, the framework supports local file systems, Amazon S3 object storage, Google Drive, PostgreSQL databases, Apache Kafka message queues, and Oracle Cloud Infrastructure (OCI) object storage. Each source connector implements a standard interface that the incremental engine uses to detect changes and pull only the modified data.

On the target side, CocoIndex supports a diverse set of storage backends optimized for different AI workloads. pgvector (PostgreSQL with vector extensions) is the go-to choice for RAG pipelines requiring vector similarity search. LanceDB provides a serverless vector database option. Neo4j and FalkorDB serve knowledge graph use cases where entities and relationships need to be stored and queried. Qdrant offers a high-performance vector similarity engine. SurrealDB provides a multi-model database that supports both document and graph queries. Kafka serves as a target for event streaming patterns. Turbopuffer and Apache Doris round out the options for specialized workloads.

The ecosystem also includes optional dependencies for embedding models: `sentence-transformers` for local embeddings, `litellm` for API-based LLM calls, and `colpali-engine` for vision-based document retrieval. This modular dependency design means you install only what your pipeline needs.

> **Takeaway**: With 20+ working examples in the repository -- from code embedding and PDF RAG to knowledge graph construction and Kafka streaming -- CocoIndex provides battle-tested starters for virtually every common AI data pipeline pattern.

## Incremental Workflow

![CocoIndex Workflow](/assets/img/diagrams/cocoindex/cocoindex-workflow.svg)

The incremental workflow is the heart of CocoIndex. When source data enters the pipeline, the engine first performs change detection by comparing fingerprints of the current data against previously stored versions. This fingerprint comparison is efficient and avoids re-reading unchanged data.

If the source data is unchanged, the engine hits the cache and skips processing entirely. This is the "cache hit" path, and it is the most common case in production: in a 10,000-row corpus, typically only 0.1% of rows change between runs, meaning 99.9% of the data stays cached. This translates to roughly 10x cost savings at scale compared to full recomputation.

When a change is detected -- the "delta" path -- the engine executes the transform function decorated with `@coco.fn(memo=True)`. This function might split text into chunks, generate embeddings, extract entities, or perform any other Python transformation. The memoization decorator ensures that even within the delta, sub-computations with unchanged inputs are skipped.

After transformation, the target state sync phase compares the newly declared target states with those from the previous run. New target states are created, changed ones are updated, and missing ones are deleted. This reconciliation happens atomically per processing component, ensuring consistency. Finally, the lineage record maps each target state back to its exact source byte, providing full provenance for debugging, auditing, and regulatory compliance.

> **Amazing**: Sub-second freshness is achievable because CocoIndex tracks per-row provenance. When a single file changes, only the target rows derived from that file are re-synced. The rest of the index remains untouched.

## Key Features

![CocoIndex Features](/assets/img/diagrams/cocoindex/cocoindex-features.svg)

| Feature | Description | Benefit |
|---------|-------------|---------|
| Incremental Processing | Delta-only recompute -- only changed data is reprocessed | 10x cheaper at scale; 99.9% of corpus stays cached |
| Declarative API | Target = F(Source) -- declare what you want, not how to update it | Code is as simple as the one-off version; engine handles the rest |
| End-to-End Lineage | Every target byte traces back to its exact source byte | Debuggable, auditable, regulator-friendly AI pipelines |
| Rust Core Engine | Production-grade from day zero with parallel chunking and failure isolation | One bad record never stalls the flow; zero-copy transforms where possible |
| Parallel by Default | Any scale from single repo to petabyte stores | Horizontal scaling without configuration changes |
| Smart Memoization | Cache keyed by hash(input) + hash(code) | Skip execution when both inputs and transformation code are unchanged |

The declarative programming model deserves special attention. In traditional ETL pipelines, developers must write both the transformation logic and the incremental update logic -- tracking what changed, computing diffs, and applying updates. CocoIndex eliminates the second half entirely. You write your transformation as if it runs once on the full dataset, and the engine automatically handles incremental updates. This is analogous to how React developers declare UI as a function of state, and the framework handles re-rendering only what changed.

The memoization system adds another layer of efficiency. When you decorate a function with `@coco.fn(memo=True)`, CocoIndex computes a fingerprint from both the input data and the function code. If neither has changed since the last run, the cached result is returned instantly. This means that even when source data changes, any sub-computation whose inputs are unaffected is skipped. The memoization also handles code changes: if you modify your transformation function, only the outputs that depend on the changed code are recomputed.

> **Important**: CocoIndex is not just for RAG. While vector search pipelines are the most common use case, the framework equally supports knowledge graph construction, structured data extraction, event streaming to Kafka, and multi-repository summarization. Any pipeline where source data changes over time and derived indexes must stay fresh is a candidate for CocoIndex.

## Installation and Quick Start

Getting started with CocoIndex takes just a few minutes:

```bash
pip install -U cocoindex
```

For optional dependencies based on your use case:

```bash
pip install -U "cocoindex[postgres]"      # PostgreSQL / pgvector target
pip install -U "cocoindex[lancedb]"       # LanceDB vector store target
pip install -U "cocoindex[neo4j]"         # Neo4j knowledge graph target
pip install -U "cocoindex[qdrant]"        # Qdrant vector DB target
pip install -U "cocoindex[kafka]"         # Kafka streaming target
pip install -U "cocoindex[sentence_transformers]"  # Local embeddings
pip install -U "cocoindex[all]"           # All optional dependencies
```

Here is a minimal example that indexes documents into PostgreSQL with vector search:

```python
import cocoindex as coco
from cocoindex.connectors import localfs, postgres
from cocoindex.ops.text import RecursiveSplitter

@coco.fn(memo=True)                          # cached by hash(input) + hash(code)
async def index_file(file, table):
    for chunk in RecursiveSplitter().split(await file.read_text()):
        table.declare_row(text=chunk.text, embedding=embed(chunk.text))

@coco.fn
async def main(src):
    table = await postgres.mount_table_target(PG, table_name="docs")
    table.declare_vector_index(column="embedding")
    await coco.mount_each(index_file, localfs.walk_dir(src).items(), table)

coco.App(coco.AppConfig(name="docs"), main, src="./docs").update_blocking()
```

Run once to backfill the entire index. Re-run anytime -- only the changed files are re-embedded. The `@coco.fn(memo=True)` decorator ensures that files whose content has not changed since the last run are skipped entirely.

## What Can You Build?

CocoIndex ships with 20+ working examples in the repository, covering a wide range of AI data pipeline patterns:

- **Code Embedding** -- Walk a git repo, chunk source files with an AST-aware splitter, embed with sentence-transformers, and upsert to pgvector or LanceDB. Fully incremental: only files touched by the latest commit re-embed.
- **PDF RAG Index** -- Ingest PDFs from local storage, S3, or Google Drive, extract text, chunk with a recursive splitter, embed each chunk, and upsert into pgvector or LanceDB with a vector index.
- **Knowledge Graph from Conversations** -- Pull people, topics, decisions, and action items from meeting transcripts, Slack, or podcasts using an LLM extractor, and upsert into Neo4j or Kuzu. Only changed turns re-extract.
- **Hacker News Trending Topics** -- Fetch HN threads via the Algolia API, recursively pull nested comments, LLM-extract typed topic lists with Gemini, and rank topics by weighted mention count.
- **Multi-Repo Summarization** -- Walk N git repositories, extract READMEs and public APIs, LLM-summarize each one, and roll up into a single top-level summary. Only repos with new commits re-run.
- **CSV to Kafka Streaming** -- Watch a folder of CSV files and publish each row as a JSON message to a Kafka topic. Sub-second incremental: only changed rows publish.
- **Structured Extraction** -- Read messy forms, PDFs, or invoices and extract typed, schema-validated fields with BAML or DSPy, then write rows into Postgres. Only changed documents re-extract.

## Technology Stack

CocoIndex is built on a hybrid Python-Rust architecture:

- **Python SDK** (3.11+) -- Declarative API, async-first design, type-safe with full mypy strict mode support
- **Rust Core Engine** -- PyO3 bindings, Tokio async runtime, production-grade error handling with retries, exponential back-off, and dead-letter queues
- **Build System** -- Maturin for Python wheel builds with Rust extension, uv for dependency management
- **Testing** -- pytest with async support, testcontainers for database integration tests, cargo test for Rust

The Rust workspace consists of five crates: `core` (the main engine), `py` (Python bindings), `py_utils` (Python-Rust utility helpers), `utils` (general utilities), and `ops_text` (text processing operations like splitting and language detection).

## Links

- [CocoIndex GitHub Repository](https://github.com/cocoindex-io/cocoindex)
- [CocoIndex Documentation](https://cocoindex.io/docs)
- [CocoIndex Quickstart Guide](https://cocoindex.io/docs/getting_started/quickstart)
- [CocoIndex Core Concepts](https://cocoindex.io/docs/programming_guide/core_concepts)
- [CocoIndex Examples](https://github.com/cocoindex-io/cocoindex/tree/main/examples)
- [CocoIndex on PyPI](https://pypi.org/project/cocoindex/)
- [CocoIndex Discord Community](https://discord.com/invite/zpA9S2DR7s)