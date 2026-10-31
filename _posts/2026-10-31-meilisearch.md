---
layout: post
title: "Meilisearch Source Tour: How the Rust Search Engine Works Under the Hood"
description: "A guided architecture tour of the Meilisearch codebase: actix-web handlers, the async task scheduler, and the milli engine's ranking and vector search internals."
date: 2026-10-31
header-img: "img/post-bg.jpg"
permalink: /meilisearch/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/meilisearch/meilisearch-meilisearch-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: meilisearch, rust search engine, full-text search, vector search, hybrid search, LMDB, index scheduler, milli, charabia, actix-web, typo tolerance, source code tour
author: "PyShine"
---

Meilisearch is one of those projects that feels deceptively small from the outside. You POST some JSON documents to it, you POST a query, and half a second later you get typo-tolerant, ranked results with facet counts. Under the hood, though, it is a carefully layered Rust system: an HTTP layer built on actix-web, an asynchronous task scheduler that persists every write to LMDB before acknowledging it, and a search engine crate called milli that owns inverted indexes, facet databases, and vector stores. This post is a source tour of that system, based on a fresh read of the repository at its default `main` branch.

The layout itself tells you a lot before you read a single function. Everything lives under `crates/`: `crates/meilisearch` is the HTTP server you actually launch, `crates/index-scheduler` is the durable work queue, and `crates/milli` is the engine core with no knowledge of HTTP at all. That separation is deliberate — milli could be reused by a different frontend entirely, which is exactly what makes the codebase pleasant to navigate. If you have read our earlier tours of [Elasticsearch](/elasticsearch/) or the pure-Rust [Tantivy](/tantivy/) library, you will notice Meilisearch sits between the two philosophies: more opinionated and turnkey than Tantivy, far lighter than Elasticsearch.

Two diagrams accompany this post. The overview shows the three crates and the main arrows between HTTP handlers, the scheduler, and the engine. The detail diagram goes one level down into the scheduler's batch creation and milli's ranking-rule machinery. Every node in the interactive SVG links to the real file on GitHub, so you can keep this page open in one tab and the source in another.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/meilisearch/meilisearch-meilisearch-overview-architecture.svg" alt="Meilisearch architecture overview: actix-web REST layer feeding the index scheduler and milli engine" /></div>

<p><em>Meilisearch at a glance: the actix-web REST layer turns every write into a queued task, the index-scheduler applies batches to milli, and search reads milli directly. Click any node to open its source file.</em></p>

Reading the overview from left to right:

- Everything starts at [`main.rs` setup](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/main.rs#L56), which parses the `Opt` configuration, wires tracing, and starts the actix HTTP server. The application itself is assembled by [`create_app`, L137](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/lib.rs#L137), and [`setup_meilisearch`, L220](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/lib.rs#L220) constructs the scheduler and auth controller that every handler shares.
- Index lifecycle endpoints live in one module: [`create_index`, L231](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/routes/indexes/mod.rs#L231) on `POST /indexes`, plus get, update and delete handlers in the same file.
- Document writes are handled by [`replace_documents`, L1367](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/routes/indexes/documents.rs#L1367), which delegates to the shared [`document_addition`, L1534](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/routes/indexes/documents.rs#L1534) helper. Crucially, this handler returns HTTP `202 Accepted` — the write has only been *enqueued*.
- The enqueuing point is [`IndexScheduler::register`, L930](https://github.com/meilisearch/meilisearch/blob/main/crates/index-scheduler/src/lib.rs#L930) in the index-scheduler crate, which persists a `Task` into an LMDB-backed [`Queue`, L129](https://github.com/meilisearch/meilisearch/blob/main/crates/index-scheduler/src/queue/mod.rs#L129).
- On the read path, [`search_with_post`, L866](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/routes/indexes/search.rs#L866) deserializes a [`SearchQuery`, L87](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/search/mod.rs#L87) struct and calls [`perform_search`, L1852](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/search/mod.rs#L1852) synchronously — search never goes through the task queue.
- The scheduler resolves which physical database a request touches through [`IndexMapper`, L60](https://github.com/meilisearch/meilisearch/blob/main/crates/index-scheduler/src/index_mapper/mod.rs#L60), which opens and caches per-index LMDB environments.
- Each index is a [`milli::Index`, L131](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/index.rs#L131): a bundle of LMDB databases for word-to-document postings, exact-attribute postings, synonyms, prefix databases, facet trees, and vectors.
- Query execution bottoms out in milli's [`execute_search`, L809](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/search/new/mod.rs#L809), with a parallel [`execute_vector_search`, L744](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/search/new/mod.rs#L744) path for hybrid and semantic queries backed by the [vector store](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/vector/db.rs#L314).

One subtlety worth calling out before the detail diagram: the search path deliberately bypasses the scheduler entirely. Reads run synchronously against the LMDB environment with bounded concurrency — `crates/meilisearch/src/search_queue.rs` implements a permit-based queue whose capacity you can tune, and `main.rs` passes it into the app state. Writes, by contrast, always funnel through the task queue even for tiny updates. This asymmetry is the whole design in one sentence: reads are cheap and direct, writes are durable and serialized. It also explains a common beginner surprise — you POST a hundred documents and immediately search for one of them and get nothing back. The task was accepted, not applied; wait for the scheduler to drain its batch and the document appears.

## From Install to Your First Model

The README keeps the onboarding honest: three steps and you have search. First, get an instance — either the hosted cloud offering or a self-hosted binary. Locally, the fastest route is the official install script, which downloads a single static binary:

```sh
curl -L https://install.meilisearch.com | sh
./meilisearch
```

Docker works just as well if you prefer containers: `docker run -it --rm -p 7700:7700 -v $(pwd)/meili_data:/meili_data getmeili/meilisearch:latest`. With no master key set, the local instance runs in an unrestricted dev mode; setting `MEILI_MASTER_KEY` switches on the API-key security layer implemented in `crates/meilisearch-auth`.

Second, add documents. The example from the README creates an index called `movies` implicitly and sends three records:

```sh
curl -X POST "http://localhost:7700/indexes/movies/documents" \
  -H 'Content-Type: application/json' \
  --data-binary '[
    { "id": 1, "title": "Carol", "genres": ["Romance", "Drama"] },
    { "id": 2, "title": "Wonder Woman", "genres": ["Action", "Adventure"] },
    { "id": 3, "title": "Life of Pi", "genres": ["Adventure", "Drama"] }
  ]'
```

The response is a JSON object with a `taskUid`, and this is the moment the architecture we just mapped becomes visible: your `curl` hit `replace_documents`, which enqueued a `DocumentAddition` task and returned immediately. Indexing happens in the background. Poll `GET /tasks/{taskUid}` until the status flips to `succeeded`, then issue the third request, a search:

```sh
curl -X POST "http://localhost:7700/indexes/movies/search" \
  -H 'Content-Type: application/json' \
  --data-binary '{ "q": "wondre woman" }'
```

Meilisearch returns *Wonder Woman* despite the typo in `wondre`. That is not a gimmick — typo tolerance is a first-class ranking rule computed against the prefix and exact-posting databases you saw in the `Index` struct. From Python this same flow takes three lines with the `meilisearch` client package, which makes the engine a natural companion for [FastAPI](/fastapi/) backends and retrieval-augmented applications — add a dedicated vector store only when you need heavier semantic search alongside keyword search.

## How Indexing Actually Happens

The scheduler is the most interesting subsystem for anyone writing concurrent services in Rust. Its run loop lives in a dedicated thread ([`run`, L630](https://github.com/meilisearch/meilisearch/blob/main/crates/index-scheduler/src/lib.rs#L630)) and each iteration executes a [`tick`, L168](https://github.com/meilisearch/meilisearch/blob/main/crates/index-scheduler/src/scheduler/mod.rs#L168). A tick reads newly registered tasks, groups compatible ones via [`create_next_batch`, L540](https://github.com/meilisearch/meilisearch/blob/main/crates/index-scheduler/src/scheduler/create_batch.rs#L540), and hands the batch to [`process_batch`, L53](https://github.com/meilisearch/meilisearch/blob/main/crates/index-scheduler/src/scheduler/process_batch.rs#L53). Because tasks are persisted in LMDB *before* the HTTP response is sent, a crash between "accepted" and "indexed" loses nothing — on restart the scheduler replays the queue. This design is why the API can promise asynchronous durability without a separate message broker.

Inside `process_batch`, document additions reach milli's write side. The merged change set flows into the parallel indexer entry point [`extract`, L99](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/update/new/indexer/document_changes.rs#L99), which fans extraction work across rayon threads: field ids, word docids, facet values, and — when the index settings declare embedders — document embeddings. Settings changes travel the same road: the [`update_all`, L577](https://github.com/meilisearch/meilisearch/blob/main/crates/meilisearch/src/routes/indexes/settings.rs#L577) settings handler enqueues a `SettingsUpdate` task, and the scheduler re-indexes the affected data. This is also where tokenization enters: milli delegates segmentation and normalization to the `charabia` crate (declared in `crates/milli/Cargo.toml`), which gives the engine its multi-language coverage including Chinese, Japanese, Hebrew and Latin-script languages.

## Search Execution Inside milli

On the read side, `perform_search` translates the HTTP query into a `SearchContext` and calls into milli. [`extract_tokens`, L909](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/search/new/mod.rs#L909) splits the user query into terms and builds a query graph — a small DAG of possible words including typo variants. [`execute_search`, L809](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/search/new/mod.rs#L809) then resolves the candidate universe and asks [`get_ranking_rules_for_query_graph_search`, L510](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/search/new/mod.rs#L510) to assemble the rule chain: words, typo, proximity, attribute, sort, and exactness, mirroring the documented default ranking order. If the request carries a `sort` parameter, [`resolve_sort_criteria`, L651](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/search/new/mod.rs#L651) inserts the sort rules ahead of the relevance rules.

Worth a brief mention too is the federation layer above all of this: the same `crates/meilisearch/src/search/federated/` module powers the multi-search endpoint, where one HTTP request fans out into per-index queries and merges the responses with weighted scores. In network mode it can even forward sub-queries to remote Meilisearch instances, which is the project's answer to read scaling without full distribution. None of that machinery lives in milli — it is pure HTTP-layer composition on top of `perform_search`, which keeps the engine crate simple and the composability at the API level where it belongs.

Facets are computed by [`FacetDistribution`, L84](https://github.com/meilisearch/meilisearch/blob/main/crates/milli/src/search/facet/facet_distribution.rs#L84), which walks the facet-number and facet-string databases restricted to the result universe. For hybrid queries — where the client sends both `q` and an `embedder` reference in the request body — `execute_vector_search` queries the ANN-enabled vector store and merges semantic and keyword candidates through the score-merging logic in `crates/meilisearch/src/search/federated/`. The detail diagram traces both of these paths end to end:

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/meilisearch/meilisearch-meilisearch-architecture.svg" alt="Meilisearch detailed architecture: HTTP handlers, scheduler batch creation, milli ranking rules and indexer extraction" /></div>

<p><em>The detailed view: task registration and batch creation on the scheduler side, ranking-rule construction and facet distribution on the search side, and the parallel extraction stages on the write side.</em></p>

## Honest Limits

Meilisearch makes deliberate trade-offs, and the README is upfront about several of them. It is a single-binary, single-node engine: there is no built-in distributed cluster, and scaling past one machine means using the experimental network/federated search features or sharding yourself. The dataset fits where LMDB fits — indexes live on your local disk, so very large corpora need multiple instances or a distributed engine like Elasticsearch. Note also that LMDB caps the *read* transaction count, which is precisely why `crates/meilisearch/src/search_queue.rs` exists to throttle concurrent searches. Async indexing is a feature, but it means writes are not immediately visible — clients must poll tasks, which adds a bit of bookkeeping compared to synchronous engines like [Typesense](/typesense/). Finally, benchmark figures in the README (sub-50-millisecond searches, etc.) are the project's own claims measured on its own workloads under `workloads/`; treat them as directional, not as independent results. For pure algorithmic detail on inverted indexing in Rust, Tantivy remains the deeper rabbit hole, and for end-to-end local chat-plus-search stacks you can wire an Open WebUI retrieval layer in front of Meilisearch yourself.

```json
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Meilisearch Source Tour: How the Rust Search Engine Works Under the Hood",
  "description": "A guided architecture tour of the Meilisearch codebase: actix-web handlers, the async task scheduler, and the milli engine's ranking and vector search internals.",
  "datePublished": "2026-10-31",
  "dateModified": "2026-10-31",
  "author": { "@type": "Person", "name": "PyShine" },
  "keywords": "meilisearch, rust, search engine, LMDB, vector search, source code",
  "about": { "@type": "SoftwareApplication", "name": "Meilisearch", "applicationCategory": "Search Engine" }
}
```
