---
layout: post
title: "ClickHouse: A Source Tour of MergeTree and the Query Pipeline - Inside ClickHouse/ClickHouse"
description: "A guided source-tour of the ClickHouse/ClickHouse repository, walking the real C++ code behind the MergeTree storage engine, the Processors query pipeline, and distributed execution across shards and replicas. Learn how the real-time OLAP database actually works by reading the directories, entry points, and data structures that power it."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /ClickHouse-MergeTree-Query-Pipeline-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/clickhouse/clickhouse-clickhouse-architecture.svg
tags:
  - ClickHouse
  - C++
  - Databases
  - Architecture
categories: [AI, Open Source]
keywords: "clickhouse source code, mergetree storage engine, clickhouse internals, columnar database, OLAP database, query pipeline processors, clickhouse query execution, replicated merge tree, distributed query execution, data parts and granules, primary key index, C++ database engine, clickhouse architecture, source tour"
author: "PyShine"
---

A dashboard that aggregates billions of rows and refreshes before your coffee cools is no longer unusual, and when it happens, the engine behind it is very often ClickHouse. The [ClickHouse/ClickHouse](https://github.com/ClickHouse/ClickHouse) repository is where that speed actually lives: a large C++ monorepo containing the columnar storage engine, the vectorized query execution pipeline, and the machinery that stitches many servers into one distributed database. Reading it is the fastest way to stop treating sub-second analytics over terabytes as magic.

The project's README describes it plainly: ClickHouse is an open-source column-oriented database management system that allows generating analytical data reports in real-time. The codebase is organized as a classic monorepo — `src/` holds the database itself, `programs/` holds the user-facing binaries such as `server`, `client`, `local`, and `keeper`, and `base/`, `contrib/`, and `tests/` carry the foundations, third-party dependencies, and an enormous functional test suite. The whole thing is licensed under Apache 2.0, and the `src/` tree is cleanly partitioned into subsystems: `Core` for the data model, `Parsers` for SQL, `Analyzer` and `Planner` for query compilation, `Processors` for execution, `Storages` for table engines, and `QueryPipeline` for the glue between them.

The source is worth a tour because ClickHouse concentrates three ideas — columnar storage with sorted immutable parts, pull-based vectorized execution, and shared-nothing distribution — into one coherent codebase, and each idea is legible in specific files. Once you can trace an INSERT into a part on disk and a SELECT down through index pruning into a pipeline of processors, you understand not just ClickHouse but the design vocabulary of most modern analytical systems. Every claim in this post points at a real path in the tree, so you can follow along in the repository.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/clickhouse/clickhouse-clickhouse-overview-architecture.svg" alt="Architecture overview of the ClickHouse/ClickHouse repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the ClickHouse/ClickHouse source layout: protocol front-ends in `src/Server` hand queries to the interpreters, the SQL layer in `src/Analyzer` and `src/Planner` compiles them into a `QueryPlan`, that plan unfolds into the processor pipeline under `src/Processors`, and the MergeTree engine in `src/Storages/MergeTree` feeds it data while distributed tables fan queries out to shards.*

Reading the overview from left to right: the `clickhouse-server` entry point in `programs/server` boots the protocol handlers in `src/Server`, which route queries into the interpreters of `src/Interpreters`. The SQL layer builds an analyzer query tree in `src/Analyzer`, hands it to `src/Planner`, and gets back a plan made of steps in `src/Processors/QueryPlan`. That plan unfolds into a `QueryPipeline`, which the executor in `src/Processors/Executors` runs as a graph of threads and transforms. On the storage side, the table engines of `src/Storages` center on MergeTree — its sources feed chunks of columns into the pipeline — while `StorageDistributed` scatters queries to remote MergeTree shards and pulls the streams back through the pipeline.

## Why You Need This

The first problem ClickHouse solves is one every growing data team hits: you have far more rows than you can realistically scan, but your dashboards and ad-hoc queries still need answers in interactive time. The repository's answer lives in `src/Storages/MergeTree`, where data is stored as immutable parts sorted by the table's primary key and cut into granules by `MergeTreeIndexGranularity.cpp`. Because parts are sorted, a query can consult `KeyCondition.cpp` and skip whole ranges of the key without touching them, which is the mechanical foundation of ClickHouse's famous scan speed.

The second problem is the memory and CPU cost of row-by-row processing. The code solves it with a strictly columnar data model — `src/Core/Block.cpp` and `src/Columns/IColumn.h` define blocks of typed column chunks — and with an execution engine built from processors that transform whole blocks at a time. Joins, aggregations, and sorts in `src/Processors/Transforms` operate on vectors rather than rows, which keeps CPU caches busy and lets the engine saturate modern hardware. This is why analytical queries that would take minutes elsewhere complete in fractions of a second.

The third problem is keeping a distributed system of replicas and shards manageable. Instead of one monolithic cluster brain, the repository composes distribution at the table level: `src/Storages/StorageReplicatedMergeTree.cpp` layers replication onto MergeTree with a replicated operation log, and `src/Storages/StorageDistributed.cpp` scatters inserts and queries across shards. Understanding those two files tells you exactly what ZooKeeper-style coordination provides and what remains local, which is invaluable when you operate or debug a cluster.

Finally, the source matters for anyone building on the platform rather than merely running it. Schema design in ClickHouse is really engine design — partition keys, sorting keys, TTLs, and the MergeTree engine family all map directly onto code paths you can read. The TTL machinery in `src/Processors/TTL/ITTLAlgorithm.cpp`, the skip indexes registered by `MergeTreeIndices.cpp`, and the merging modes in `MergeTreeData.h`'s `MergingParams` are the ground truth behind the documentation. Touring the source turns "best practices" from folklore into mechanism.

## How It Works

Every query, whether it arrives over TCP, HTTP, or a wire-compatibility protocol, funnels through one dispatch chain that compiles SQL into a plan and then executes it as a pipeline.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/clickhouse/clickhouse-clickhouse-architecture.svg" alt="Detailed architecture of the ClickHouse/ClickHouse repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of ClickHouse: protocol handlers feed the executeQuery dispatcher, the analyzer and planner compile the query into QueryPlan steps, those steps unfold into processors run by the PipelineExecutor, and the MergeTree trees cover both the insert path through parts and the read path through pruning and range readers, with replication and distributed execution on the right.*

### Understanding the Architecture

**The server is a set of protocol front-ends over one dispatcher.** `programs/server/Server.cpp` is the entry point: its `Server::main` loads configuration, initializes the global context, and starts listeners for the native TCP protocol, HTTP, and the MySQL and PostgreSQL wire protocols. Each session is handled by a class like `src/Server/TCPHandler.cpp` or `src/Server/HTTPHandler.cpp`, which collects SQL text and calls the central dispatcher declared in `src/Interpreters/executeQuery.h`. That dispatcher owns the common flow — parsing, analysis, execution, and streaming results back — no matter which protocol the client speaks.

**SQL becomes a query tree, then a plan of steps.** The parser in `src/Parsers` produces an abstract syntax tree, and `src/Analyzer/QueryTreeBuilder.cpp` lowers that AST into a query tree — an explicit graph of query nodes. `src/Planner/Planner.cpp` then walks the tree and emits a `QueryPlan`, built from step objects in `src/Processors/QueryPlan` such as `AggregatingStep.cpp`, `SortingStep.cpp`, and, most importantly for this tour, `ReadFromMergeTree.cpp`. The plan is a tree of logical operations; only at the end does it "unfold" into actual processors. The interpreter entry point in `src/Interpreters/InterpreterSelectQueryAnalyzer.cpp` orchestrates exactly this handoff.

**MergeTree stores data as immutable, sorted parts.** The heart of the engine is `src/Storages/MergeTree/MergeTreeData.cpp`, which manages a set of data parts described by `IMergeTreeDataPart.cpp` and `MergeTreePartInfo.h` — a name encoding partition, block range, and merge level. Each part stores columns sorted by the table's sorting key and cut into granules tracked by `MergeTreeIndexGranularity.cpp`, with per-partition min/max ranges in `MergeTreeIndexMinMax.cpp` and optional skip indexes alongside. During reads, `KeyCondition.cpp` turns WHERE clauses into an RPN condition over the primary key and `PartitionPruner.cpp` discards whole partitions first, so most parts are never even opened.

**Writes append new parts; background merges compact them.** An INSERT flows from `src/Interpreters/InterpreterInsertQuery.cpp` through the pipeline into `MergeTreeSink.cpp`, which splits the block per partition and calls `MergeTreeDataWriter::writeTempPart` — the writer computes the partition, builds the min/max index, sorts the block by the sorting key, and writes columns plus checksums into a temporary `tmp_insert_` directory before it is committed under its final part name. In the background, `MergeTreeDataMergerMutator.cpp` selects parts to merge (`selectPartsToMerge`) and the executors in `MergeTreeBackgroundExecutor.cpp` run the merge and mutation tasks, gradually collapsing many small parts into fewer large ones. TTL rules from `src/Processors/TTL/ITTLAlgorithm.cpp` are applied during those same merges, and the `MergingParams` modes in `MergeTreeData.h` — Ordinary, Collapsing, Summing, Aggregating, Replacing, VersionedCollapsing, Graphite — give the same machinery per-table semantics.

**Queries execute as a graph of processors.** When the plan is unfolded, `ReadFromMergeTree` — via `MergeTreeDataSelectExecutor.cpp`, whose `readFromParts` turns pruned parts and ranges into a plan node — creates `MergeTreeSource.cpp` processors, each wrapping a select processor that walks `MergeTreeRangeReader.cpp`. That range reader chains PREWHERE steps, filtering rows on a cheap condition before fully deserializing the remaining columns through readers like `MergeTreeReaderWide.cpp`. All sources and transforms are connected into a pipeline by `src/QueryPipeline/QueryPipelineBuilder.cpp`, and the runtime in `src/Processors/Executors/Runtime/PipelineExecutor.cpp` builds an `ExecutingGraph.cpp` from it, spawning threads that pull chunks through ports until the result reaches the client.

**Replication and distribution are table-level composites.** `src/Storages/StorageReplicatedMergeTree.cpp` extends the MergeTree machinery with a replicated log: writes go through `ReplicatedMergeTreeSink.cpp`, operations are enqueued in `ReplicatedMergeTreeQueue.cpp`, and coordination happens through the ZooKeeper client in `src/Common/ZooKeeper/ZooKeeper.cpp` — the constructor flatly refuses to create a replicated table without it. On top of that, `src/Storages/StorageDistributed.cpp` presents a cluster of shards as one table: inserts are scattered to remote shards, and selects are executed by `src/QueryPipeline/RemoteQueryExecutor.cpp`, which sends the query over the native protocol to each shard's `TCPHandler` and merges the returned streams locally.

Follow one query end to end and the design clicks: `clickhouse-client` sends SQL over the native protocol to `TCPHandler`, `executeQuery` parses it into an AST, the analyzer and planner compile it into a QueryPlan, and `ReadFromMergeTree` prunes partitions and key ranges down to a handful of granules. The pipeline executor spins up a graph of MergeTree sources and transforms, PREWHERE filters rows before full columns are read, aggregations run over whole column chunks, and the distributed variants of these steps simply fan the same machinery out to replicas and shards before the coordinator merges the streams. Storage, planning, and execution all speak the same language of blocks and columns, which is exactly what makes the engine fast.

## Advantages

- **Columnar storage with real pruning.** Sorted parts, granules, primary-key conditions in `KeyCondition.cpp`, and partition pruning in `PartitionPruner.cpp` mean queries touch only the data ranges that can actually match.
- **Vectorized block execution.** The processor model in `src/Processors` pushes whole blocks of columns through transforms, so aggregation, sorting, and joining run at memory-bandwidth speeds rather than row-loop speeds.
- **Composable table engines.** `src/Storages/StorageFactory.cpp` registers dozens of engines, and MergeTree's `MergingParams` modes reuse one storage core for Replacing, Summing, Aggregating, Collapsing, and more semantics.
- **Background maintenance built in.** Merges, mutations, TTL expirations, and part cleanups are scheduled by `MergeTreeBackgroundExecutor.cpp`, so tables stay compact and policy-compliant without operator intervention.
- **Replication and sharding without a cluster brain.** `StorageReplicatedMergeTree.cpp` and `StorageDistributed.cpp` compose distribution at the table level, keeping the core server simple and the failure modes local.
- **Multiple protocols, one engine.** Native TCP, HTTP, and MySQL/PostgreSQL wire compatibility in `src/Server/` all funnel into the same execution path, so tooling choices do not fork the architecture.

## Benefits

- **Read the engine, not the marketing.** Tracing an INSERT through `MergeTreeSink.cpp` and `MergeTreeDataWriter.cpp` gives you ground truth about part sizes, sorting, and deduplication without intermediaries.
- **Design schemas that map to the machinery.** Knowing how `KeyCondition.cpp` and `MergeTreeIndexGranularity.cpp` work tells you why ORDER BY choice and partitioning dominate real-world performance.
- **Debug with a mental model.** When inserts are slow or merges lag, the files in `src/Storages/MergeTree/` show exactly which background executor, queue, or settings knob is responsible.
- **Learn production-grade C++ at scale.** The repository demonstrates how a large systems codebase structures ports, processors, thread pools, and caches — patterns that transfer to far smaller projects.
- **Understand distributed trade-offs.** The replicated queue in `ReplicatedMergeTreeQueue.cpp` and the scatter-gather logic in `StorageDistributed.cpp` are reference implementations of log-based replication and sharded execution you can study directly.
- **Contribute with confidence.** Clean subsystem boundaries — Parsers, Analyzer, Planner, QueryPlan, Processors, Storages — mean a new contributor can locate the exact file that matters for a fix in minutes.

## Usage

The README's quick install path for Linux, macOS, and FreeBSD is a single curl command:

```
curl https://clickhouse.com/ | sh
```

That script installs the standard ClickHouse binaries, after which you can start the server and open `clickhouse-client` to run SQL. The README also links the [official documentation](https://clickhouse.com/docs/) for in-depth setup, a [tutorial](https://clickhouse.com/docs/getting_started/tutorial/) for setting up and querying a small ClickHouse cluster, and a [code browser](https://github.dev/ClickHouse/ClickHouse) powered by github.dev for exploring this source tree with syntax highlighting. Source-level readers will want to start in `src/Storages/MergeTree/` and `src/Processors/` — the two directories this tour leaned on most.

## Conclusion

Reading ClickHouse/ClickHouse rewards the effort precisely because the code matches the architecture. Storage is a forest of immutable sorted parts that background merges keep tidy; planning turns SQL into a tree of explicit steps; execution pulls columns through a graph of processors; and replication and sharding are table-level composites rather than hidden platform magic. Once you have walked `programs/server/Server.cpp`, `MergeTreeData.cpp`, `ReadFromMergeTree.cpp`, and `PipelineExecutor.cpp`, ClickHouse stops being a black box and becomes a well-organized codebase you can navigate on demand.

Links:

- Repository: [github.com/ClickHouse/ClickHouse](https://github.com/ClickHouse/ClickHouse)
- Documentation: [clickhouse.com/docs](https://clickhouse.com/docs/)
- Tutorial: [clickhouse.com/docs/getting_started/tutorial](https://clickhouse.com/docs/getting_started/tutorial/)
