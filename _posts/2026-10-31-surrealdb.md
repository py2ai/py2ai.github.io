---
layout: post
title: "SurrealDB: One Rust Binary for Documents, Graphs, and Vectors"
description: "A source-code tour of SurrealDB's multi-model engine: KV storage backends, the SurrealQL query executor, document processing, and the HTTP/RPC layer."
date: 2026-10-31
header-img: "img/post-bg.jpg"
permalink: /surrealdb/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/surrealdb/surrealdb-surrealdb-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: surrealdb, multi-model database, rust, kv store, query engine
author: "PyShine"
---

If you have ever built an AI-flavored application, you know the drill: documents go into a document store, the relationship graph goes into a graph database, embeddings go into a dedicated vector engine such as [Qdrant](/qdrant/) or [Milvus](/milvus/), and then you spend weeks wiring four systems together, keeping them consistent, and paying four infrastructure bills. Multi-model databases attack this sprawl at the root. Instead of bolting a JSON column onto a relational engine or gluing a vector index onto a key-value cache, they commit to one storage core that treats documents, graph edges, and high-dimensional vectors as first-class citizens.

SurrealDB is one of the most ambitious entries in this category. It is written entirely in Rust, ships as a single binary, and — according to its README — unifies "document, graph, relational, time-series, geospatial and key-value data types with powerful search and retrieval functionalities (full-text, vector, hybrid)" plus real-time capabilities. It can run embedded in your process, compiled to WebAssembly for the browser, on the edge, as a self-hosted node, or as a distributed cluster. SurrealDB even doubles as an API backend: row-level permissions, scope-based user signup, and live queries that push changes to clients are built in, so the traditional backend service between your app and the database can shrink or disappear. The core database code is licensed under the Business Source License 1.1, with SDKs and libraries under Apache 2.0 or MIT.

What does that look like inside? In this post we walk the actual source of `surrealdb/surrealdb` on the `main` branch (commit `18971ffba781`) and trace the path a query takes: from the CLI that starts the server, through the HTTP and WebSocket surfaces, into the SurrealQL parser, the executor, the per-record document engine, and finally the key-value storage layer where RocksDB, SurrealKV, memory, and TiKV live. Two diagrams anchor the tour: an overview of the major layers, and a detailed map of the modules and entry points.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/surrealdb/surrealdb-surrealdb-overview-architecture.svg" alt="SurrealDB overview architecture" style="min-width:900px;width:100%;max-width:1100px;" loading="lazy"></div>

<p><em>Overview: CLI boots the server, the Axum HTTP router fans out to RPC and SQL endpoints, queries flow through the executor and document engine, and every read and write lands in a pluggable KV backend.</em></p>

Reading the overview from left to right:

- **[Commands enum, L167](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/cli/mod.rs#L167)** — the Clap-derived subcommand table. `Start` is described as "Start the database server" and `Sql` as "Start an SQL REPL in your terminal with pipe support", so the same binary is both server and client tool.
- **[start::init, L212](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/cli/start.rs#L212)** — server bootstrap. The `--bind` flag defaults to `127.0.0.1:8000` (see [the arg definition, L158](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/cli/start.rs#L158)), overridable via the `SURREAL_BIND` environment variable.
- **[community_router, L151](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/ntw/mod.rs#L151)** — an Axum router that merges the [ `/rpc` handler, L38](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/ntw/rpc.rs#L38), the [ `/sql` handler, L28](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/ntw/sql.rs#L28), `/signin`, `/signup`, `/key`, export/import, GraphQL, and MCP routes into one service.
- **[RpcProtocol trait, L95](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/rpc/protocol.rs#L95)** — the protocol-agnostic JSON-RPC surface. Its `execute` method matches the wire method and dispatches to typed handlers like `query` and `signin`; the [Method enum, L3](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/rpc/method.rs#L3) enumerates them.
- **[Datastore, L210](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/ds.rs#L210)** — the central object. [Datastore::new, L951](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/ds.rs#L951) takes a URL such as `memory`, `rocksdb://`, `surrealkv://`, or `tikv://127.0.0.1:2379` and builds the engine; [Datastore::execute, L3437](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/ds.rs#L3437) parses SurrealQL text and runs it.
- **[Executor, L51](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/dbs/executor.rs#L51)** — drives each parsed statement through its `compute` method and collects per-statement results, timings, and observability events.
- **[Iterator::process, L1206](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/dbs/iterator.rs#L1206)** — the bridge from scanning to document semantics: for every candidate record it constructs a [Document](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/doc/document.rs#L21) and invokes `select`, `create`, `update`, `relate`, or `delete` on it.
- **[Transactor, L37](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/tr.rs#L37)** — the transaction facade every backend implements, with `TransactionType` (Read/Write) and `LockType` (Pessimistic/Optimistic) chosen per statement.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/surrealdb/surrealdb-surrealdb-architecture.svg" alt="SurrealDB detailed architecture" style="min-width:900px;width:100%;max-width:1100px;" loading="lazy"></div>

<p><em>Detail map: HTTP surface, RPC and auth modules, parser and statement types, the executor with its index planners, the document engine, all four KV backends, and the full-text and vector index subsystems.</em></p>

## One Storage Contract, Many Backends

SurrealDB's storage design is a classic "KV core, many engines" play — the same philosophy that made [Redis](/redis/) successful, but applied to a full database rather than a cache. Everything above the storage layer only ever sees byte keys and byte values inside transactions. The concrete engine is chosen by a URL scheme.

The dispatch happens in [the flavour match, L620](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/ds.rs#L620): `memory` (with `mem` accepted as an alias), `rocksdb`, `surrealkv`, `indxdb` (the browser/WASM engine), and `tikv` for distributed deployments. A nice archaeological detail: the legacy `file:` scheme was removed, and the code now fails with a targeted message — "The `file://` scheme is no longer supported; use `rocksdb://` or `surrealkv://` instead" — instead of a generic error.

Each backend implements the same transaction contract. The in-memory engine is a compact [Datastore struct, L34](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/mem/mod.rs#L34); the RocksDB engine initializes in [rocksdb::Datastore::new, L349](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/rocksdb/mod.rs#L349) and carries its own background flusher, commit coordinator, memory manager, and garbage collector; SurrealKV — the project's own disk engine — does the same in [surrealkv::Datastore::new, L68](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/surrealkv/mod.rs#L68). Transactions themselves are expressed in [kvs/tr.rs](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/tr.rs): a `Transactor` wraps any backend-specific `Transactable` implementation, exposes `get`, `set`, `scan`, `commit`, and `cancel`, and even logs an error if a writeable transaction is dropped without being committed — a small guard that catches real bugs.

The plumbing that assembles a `Datastore` is worth a glance too: [Builder::build_with_path, L195](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/ds/builder.rs#L195) threads a `TransactionBuilder`, a live-query broker, capabilities, and index stores into the final struct. The `TransactionBuilder` trait exists precisely so external crates can plug in custom backends while reusing everything above it.

## From Install to Your First Model

The README keeps installation to one command per platform. On Linux or macOS:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://install.surrealdb.com | sh
```

On macOS you can alternatively `brew install surrealdb/tap/surreal`, and on Windows `iwr https://windows.surrealdb.com -useb | iex`. Docker works too, and the README's dev-mode example is the fastest way to poke at the system: `docker run --rm --pull always --name surrealdb -p 8000:8000 surrealdb/surrealdb:latest start --log info --user root --pass root memory` — an in-memory server on port 8000 with root credentials.

Once installed, `surreal start` boots the server (default bind address `127.0.0.1:8000`) and `surreal sql` drops you into an interactive SurrealQL shell. Then the multi-model part begins. Define a schemafull table with fields, a unique index, and an event, exactly as the README shows:

```sql
DEFINE TABLE user SCHEMAFULL;
DEFINE FIELD email ON TABLE user TYPE string ASSERT string::is_email($value);
DEFINE INDEX email ON TABLE user COLUMNS email UNIQUE;
DEFINE EVENT email ON TABLE user WHEN $before.email != $after.email THEN (
    CREATE event SET user = $value, time = time::now(), action = 'email_changed'
);
```

Graph edges come from `RELATE`, which the README demonstrates as `RELATE user:tobie->write->article:surreal SET time.written = time::now();` — no join tables, no separate graph engine. If you later need analytical workloads that SurrealDB is not aiming for, that is where column-oriented engines such as DuckDB or ClickHouse complement rather than replace it.

## How a Query Actually Runs

The execution path is a chain of well-separated stages. Text arrives at [Datastore::execute, L3437](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/kvs/ds.rs#L3437), which first checks session expiry and anonymous-access policy, then calls [syn::parse_with_capabilities, L107](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/syn/mod.rs#L107). The parser — a hand-written lexer and recursive-descent parser under `surrealdb/core/src/syn/` — validates the query against the datastore's capability flags before producing an `Ast`.

Statements then reach the [Executor, L51](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/dbs/executor.rs#L51), whose design the code summarizes in one line: "An executor which relies on the `compute` methods of the logical expressions." Each statement type carries its own compute logic. For a `SELECT`, [SelectStatement::compute in select.rs](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/expr/statements/select.rs) creates a `QueryPlanner` and an `Iterator`; the borrowed [Statement enum, L38](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/dbs/statement.rs#L38) then matches the statement variant at [Iterator::process, L1206](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/dbs/iterator.rs#L1206) and hands each record to the right document operation.

Index selection lives in two generations of planners. The streaming executor's [Planner, L124](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/exec/planner.rs#L124) — with its `plan` method at [L528](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/exec/planner.rs#L528) — turns expressions into physical operator trees, while [IndexAnalyzer, L19](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/exec/index/analysis.rs#L19) produces candidates and [select_access_path, L235](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/exec/index/access_path.rs#L235) chooses between B-tree scans, full-text scans, and KNN scans. The module documentation in `exec/index.rs` even includes an ASCII diagram of this planner-then-execute split.

The search subsystems are genuine multi-model proof. Full-text search is a real inverted index with BM25 scoring — see [FullTextIndex, L151](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/idx/ft/fulltext.rs#L151) and its `Bm25Params`. Vector search ships two independent index families: [HnswIndex, L172](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/idx/trees/hnsw/index.rs#L172) and a DiskANN implementation under `idx/trees/diskann/`, with a bounded top-k helper in [KnnPriorityList, L11](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/idx/planner/knn.rs#L11). You do not need a sidecar vector database for moderate-scale semantic search — the same reasons teams reach for [pgvector](/pgvector/) apply here, minus the PostgreSQL dependency.

## Auth, Realtime, and the RPC Surface

The document engine's per-record lifecycle is where SurrealDB's "database as API backend" claim gets real. [Document::create, L10](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/doc/create.rs#L10) reads like a checklist: permission gate, view-table rejection, input computation, record-id generation, table-type check, default fields, schema field processing, create permissions, storage via [store_record_data, L10](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/doc/store.rs#L10), index maintenance, table references, views, events, live-query fan-out, and changefeeds — before output projection runs. Graph edges follow the same discipline in [Document::relate, L11](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/doc/relate.rs#L11).

Authentication is layered the same way. The HTTP endpoints [/signin, L41](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/ntw/signin.rs#L41) and [/signup, L39](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/ntw/signup.rs#L39) delegate to the IAM module: [iam::signin, L92](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/iam/signin.rs#L92) verifies access grants at root, namespace, or database level and supports bearer records, while [iam::signup, L79](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/core/src/iam/signup.rs#L79) provisions scoped users. Over WebSocket, the server implements the same [RpcProtocol trait on Websocket, L717](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/rpc/websocket.rs#L717), with explicit `begin`, `commit`, and `cancel` handlers and live-query registration in [handle_live, L857](https://github.com/surrealdb/surrealdb/blob/main/surrealdb/server/src/rpc/websocket.rs#L857). There is even a stdio MCP server subcommand "for AI assistant integration" — SurrealDB is clearly aiming at the agent-tooling ecosystem, not just human developers.

## Honest Limits

A few caveats worth knowing before you commit. First, licensing: the core database code is BSL 1.1, not a classic OSS license — fine for most products, but read the terms if you plan to offer SurrealDB itself as a service. Second, storage backends are a moveable feast: `file://` is gone, TiKV support depends on build features, and the README's Docker examples lean on `memory` or `rocksdb`, so pick `rocksdb://` or `surrealkv://` deliberately for persistence. Third, the GraphQL interface is listed in the README as "coming soon", so HTTP `/sql` and JSON-RPC over WebSocket are the stable query surfaces today. Fourth, "multi-model" does not mean "best at everything": heavy analytical scans are still the territory of dedicated column-oriented engines, and extremely large-scale vector workloads may still justify a dedicated engine — SurrealDB gives you the integrated default, not necessarily the extreme-end specialist. Finally, the two vector index families (HNSW and DiskANN) and the enterprise/community split visible in the code (the `CommunityComposer` versus enterprise brokers in `kvs/ds/builder.rs`) mean feature availability varies by edition and build flags — verify against the docs for your exact version.

None of these limits are unusual for a system this young and this broad. What is unusual is how coherent the implementation is: one KV contract, one parser, one executor, one document lifecycle — and four storage engines, three index families, and two wire protocols that all slot into the same seams. If your next project needs documents, relationships, and embeddings in one store, SurrealDB deserves a spot on your shortlist next to Redis stack hybrids and Qdrant-backed stacks.

{"@context":"https://schema.org","@type":"TechArticle","headline":"SurrealDB: One Rust Binary for Documents, Graphs, and Vectors","description":"A source-code tour of SurrealDB's multi-model engine: KV storage backends, the SurrealQL query executor, document processing, and the HTTP and RPC layer.","author":{"@type":"Person","name":"PyShine"},"datePublished":"2026-10-31","keywords":"surrealdb, multi-model database, rust, kv store, query engine","publisher":{"@type":"Organization","name":"PyShine"}}
