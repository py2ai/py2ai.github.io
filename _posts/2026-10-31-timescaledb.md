---
layout: post
title: "TimescaleDB Source Tour: How Hypertables, Chunk Routing, Columnstore, and Continuous Aggregates Actually Work"
description: "An annotated architecture tour of the timescale/timescaledb C codebase: hypertable creation, tuple routing into chunks, the planner hook, columnstore compression, continuous aggregates, and background workers."
date: 2026-10-31
header-img: "img/post-bg.jpg"
permalink: /timescaledb/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/timescaledb/timescale-timescaledb-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: timescaledb, postgresql, time-series, hypertable, chunk, tuple routing, columnstore, compression, continuous aggregates, background workers, planner hook, source code tour
author: "PyShine"
---

TimescaleDB is a PostgreSQL extension for high-performance real-time analytics on time-series and event data, developed by Timescale (now Tiger Data). Unlike a fork of Postgres or a separate storage engine, it is a loadable extension written in C that hooks deeply into the planner, the executor, and the DDL machinery of the database itself. That design choice explains almost everything you see when you open the repository: there is a `src/` directory that talks to PostgreSQL core through well-defined hooks, a `sql/` directory that ships the extension's user-visible functions and catalog definitions, and a `tsl/` directory that holds the licensed "TSL" features — columnstore compression and continuous aggregates — behind a small cross-module interface.

This post is a source tour of the `main` branch (version 2.32.0-dev at the time of writing). We rendered two interactive architecture diagrams directly from the source tree: an overview graph that fits on one screen, and a detail graph that separates the extension into eight functional groups. Every symbol mentioned below was located with grep in the actual sources, so the line-number links take you to exact code in the `timescale/timescaledb` repository on GitHub.

The mental model to keep in mind: a "hypertable" is the user-facing table, but it is backed by many PostgreSQL child tables called chunks, one per time (and optionally space) interval. Writes arrive at the parent table, get translated into a point in a multi-dimensional space, and are routed into the right chunk. Everything else — columnstore compression, continuous aggregates, retention and refresh policies — is machinery layered on top of that single idea.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/timescaledb/timescale-timescaledb-overview-architecture.svg" alt="TimescaleDB overview architecture: SQL interface, hypertable core, insert routing, planner, and TSL features" /></div>

<p><em>Overview of the timescale/timescaledb codebase. SQL entry points become hypertable metadata in the catalog; every write is partitioned and routed to a chunk; the planner wraps DML so routing happens at execution time; TSL code adds columnstore compression and continuous aggregates.</em></p>

Reading the overview from left to right:

- The journey starts in SQL: [`create_hypertable`, L21](https://github.com/timescale/timescaledb/blob/main/sql/ddl_api.sql#L21) in `sql/ddl_api.sql` is the user-facing function that turns an ordinary PostgreSQL table into a hypertable.
- That SQL wrapper calls into C: [`ts_hypertable_create`, L1576](https://github.com/timescale/timescaledb/blob/main/src/hypertable.c#L1576) validates options and drives [`ts_hypertable_create_internal`, L1482](https://github.com/timescale/timescaledb/blob/main/src/hypertable.c#L1482), which registers the hypertable in the metadata tables exposed by [`catalog_get_table`, L569](https://github.com/timescale/timescaledb/blob/main/src/ts_catalog/catalog.c#L569).
- Dimensions come next: [`ts_dimension_calculate_default_slice`, L433](https://github.com/timescale/timescaledb/blob/main/src/dimension.c#L433) computes the interval boundaries for the time dimension, and [`ts_dimension_slice_scan_limit`, L440](https://github.com/timescale/timescaledb/blob/main/src/dimension_slice.c#L440) finds which slice covers a given coordinate.
- On every insert, [`ts_partitioning_func_apply`, L271](https://github.com/timescale/timescaledb/blob/main/src/partitioning.c#L271) maps column values to an int64 coordinate, and [`ts_chunk_tuple_routing_find_chunk`, L72](https://github.com/timescale/timescaledb/blob/main/src/chunk_tuple_routing.c#L72) picks — or creates — the target chunk via [`ts_chunk_create_for_point`, L1665](https://github.com/timescale/timescaledb/blob/main/src/chunk.c#L1665).
- Before a row lands in a child table, [`ts_chunk_insert_state_create`, L466](https://github.com/timescale/timescaledb/blob/main/src/chunk_insert_state.c#L466) builds the per-chunk insert state that carries triggers, indexes, and result relation info.
- For queries, the planner hook [`timescaledb_planner`, L631](https://github.com/timescale/timescaledb/blob/main/src/planner/planner.c#L631) rewrites plans against hypertables, and DML gets wrapped by [`ts_modify_hypertable_path_create`, L797](https://github.com/timescale/timescaledb/blob/main/src/nodes/modify_hypertable.c#L797) so that [`modify_hypertable_exec`, L290](https://github.com/timescale/timescaledb/blob/main/src/nodes/modify_hypertable.c#L290) can route tuples at execution time.
- In the licensed `tsl/` tree, [`compress_chunk`, L300](https://github.com/timescale/timescaledb/blob/main/tsl/src/compression/compression.c#L300) rewrites row chunks into compressed columnar batches, while [`continuous_agg_refresh`, L970](https://github.com/timescale/timescaledb/blob/main/tsl/src/continuous_aggs/refresh.c#L970) incrementally materializes continuous aggregates from invalidation ranges.

## Why the Codebase Is Shaped This Way

TimescaleDB cannot patch PostgreSQL itself, so it attaches at every seam the extension API exposes. During library load, [`_PG_init`, L89](https://github.com/timescale/timescaledb/blob/main/src/init.c#L89) installs hooks for planning, DDL, COPY, and executor callbacks, and registers the `timescaledb.*` GUC settings (for example `enable_direct_compress_insert` in `src/guc.c`). Until the user runs `CREATE EXTENSION`, all of that machinery stays dormant: [`ts_extension_is_loaded`, L285](https://github.com/timescale/timescaledb/blob/main/src/extension.c#L285) is the gate checked throughout the code, which is why the extension can sit safely loaded inside any Postgres instance.

Because hooks alone do not give you a clean internal API, the code routes shared state through caches. The generic pinned-cache infrastructure lives in `src/cache.c` ([`ts_cache_pin`, L122](https://github.com/timescale/timescaledb/blob/main/src/cache.c#L122)), and the most important user of it is the hypertable cache: [`ts_hypertable_cache_get_entry`, L189](https://github.com/timescale/timescaledb/blob/main/src/hypertable_cache.c#L189) turns a relation OID into a `Hypertable` object once per transaction and pins it so executor callbacks can reuse it without re-scanning the catalog. Chunk lookups get the same treatment via a hypercube-keyed memo in [`ts_subspace_store_get`, L198](https://github.com/timescale/timescaledb/blob/main/src/subspace_store.c#L198), which avoids hitting the chunk catalog table on every routed row.

DDL interception deserves its own mention. A `CREATE TABLE ... WITH (tsdb.hypertable)` statement, or an `ALTER TABLE ... SET (timescaledb.compress)`, flows through the process-utility hook [`timescaledb_ddl_command_start`, L6593](https://github.com/timescale/timescaledb/blob/main/src/process_utility.c#L6593). That single function is one of the largest files in the extension for a reason: every DDL statement that touches a hypertable (dropping chunks, truncating, adding columns) must be translated so that the same operation happens on the right set of child tables and catalog rows.

## From Install to Your First Model

The repository README's quick start is genuinely one-command. With Docker running, the fastest path is:

```sh
curl -sL https://tsdb.co/start-local | sh
```

That script starts a tuned container, exposes PostgreSQL on port 6543, and sets up a persistent volume. If you prefer to do it manually — the README notes this is also the Windows path — the equivalent is:

```bash
docker run -d --name timescaledb \
    -p 6543:5432 \
    -e POSTGRES_PASSWORD=password \
    timescale/timescaledb-ha:pg18
```

The `timescale/timescaledb-ha:pg18` image ships TimescaleDB pre-installed and pre-loaded. Connect with `psql -h localhost -p 6543 -U postgres` (password `password`). If you are using a self-managed PostgreSQL instead of the Docker image, the standard sequence is `CREATE EXTENSION timescaledb;` after adding `shared_preload_libraries = 'timescaledb'` — the preload is what lets [`_PG_init`](https://github.com/timescale/timescaledb/blob/main/src/init.c#L89) install the hooks described above.

Now create your first data model, straight from the README's Step 3:

```sql
CREATE TABLE sensor_data (
    time TIMESTAMPTZ NOT NULL,
    sensor_id TEXT NOT NULL,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    pressure DOUBLE PRECISION
) WITH (tsdb.hypertable);
```

The `tsdb.hypertable` parameter is the modern sugar over the classic two-step form `CREATE TABLE ...; SELECT create_hypertable('sensor_data', 'time');` — both end in the same C entry point. Inside the extension, that triggers the full creation path from the overview diagram: catalog rows, a time dimension on the `time` column, and the first chunk.

Then load data and aggregate. The README's Step 4 enables direct columnstore inserts and generates ~7.8 million rows with `generate_series`:

```sql
SET timescaledb.enable_direct_compress_insert = on;
INSERT INTO sensor_data (time, sensor_id, temperature, humidity, pressure)
SELECT time,
       'sensor_' || ((random() * 9)::int + 1),
       20 + (random() * 15),
       40 + (random() * 30),
       1000 + (random() * 50)
FROM generate_series(NOW() - INTERVAL '90 days', NOW(), INTERVAL '1 seconds') AS time;
```

Finally, query it the time-series way — bucketing is a first-class SQL function, and the planner knows how to push `time` predicates down to individual chunks:

```sql
SELECT time_bucket('1 hour', time) AS hour,
       sensor_id,
       ROUND(AVG(temperature)::numeric, 2) AS avg_temp
FROM sensor_data
WHERE time > NOW() - INTERVAL '24 hours'
GROUP BY hour, sensor_id
ORDER BY hour DESC, sensor_id
LIMIT 20;
```

## The Detail View: Write Path, Columnstore, Aggregates, Workers

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/timescaledb/timescale-timescaledb-architecture.svg" alt="TimescaleDB detail architecture: bootstrap and caches, hypertable core, chunk write path, planner and executor, TSL columnstore, continuous aggregates, background workers" /></div>

<p><em>The detail graph separates the sources into eight groups. Follow the write path from partitioning through tuple routing to chunk insert state; follow the read path from the planner hook through hypertable expansion into chunk pruning.</em></p>

The write path in detail. The executor asks for a routing plan once per statement, then for every row the partitioning functions produce a point; existing chunks are found through the subspace store memo, and a miss calls `ts_chunk_create_for_point`, which builds a [`ts_hypercube_add_slice_from_range`, L129](https://github.com/timescale/timescaledb/blob/main/src/hypercube.c#L129) hypercube and materializes the child table. Bulk loading via `COPY FROM` reuses the same machinery from [`copy_chunk_state_create`, L158](https://github.com/timescale/timescaledb/blob/main/src/copy.c#L158) instead of spawning a separate code path — a deliberate reuse that keeps COPY semantics identical to INSERT.

The columnstore. When a chunk is compressed, [`tsl_compress_chunk`, L899](https://github.com/timescale/timescaledb/blob/main/tsl/src/compression/api.c#L899) wraps the engine-level [`compress_chunk`](https://github.com/timescale/timescaledb/blob/main/tsl/src/compression/compression.c#L300): rows are sorted by the configured `order_by` columns, grouped by `segment_by` columns, and written as compressed columnar batches into a companion table. The settings themselves are catalog objects — [`ts_compression_settings_get`, L400](https://github.com/timescale/timescaledb/blob/main/src/ts_catalog/compression_settings.c#L400) fetches them per chunk. Setting them up from DDL is [`tsl_process_compress_table`, L1529](https://github.com/timescale/timescaledb/blob/main/tsl/src/compression/create.c#L1529), and the README's `convert_to_columnstore` loop maps to [`tsl_move_to_columnstore`, L1182](https://github.com/timescale/timescaledb/blob/main/tsl/src/compression/api.c#L1182). Inserts into already-compressed chunks trigger on-demand decompression of the affected batches during routing.

Continuous aggregates. Declaring `CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous)` is parsed by [`tsl_process_continuous_agg_viewstmt`, L783](https://github.com/timescale/timescaledb/blob/main/tsl/src/continuous_aggs/create.c#L783), which builds a hidden materialization hypertable. When rows in raw chunks change, [`continuous_agg_dml_invalidate`, L921](https://github.com/timescale/timescaledb/blob/main/tsl/src/continuous_aggs/insert.c#L921) appends affected time ranges to a persistent log via [`invalidation_cagg_log_add_entry`, L230](https://github.com/timescale/timescaledb/blob/main/tsl/src/continuous_aggs/invalidation.c#L230). A scheduled job later calls [`continuous_agg_refresh`](https://github.com/timescale/timescaledb/blob/main/tsl/src/continuous_aggs/refresh.c#L970), which re-aggregates only the invalidated windows — that is what makes refreshes cheap on large datasets. Policies like [`add_continuous_aggregate_policy`, L99](https://github.com/timescale/timescaledb/blob/main/sql/policy_api.sql#L99) simply register jobs.

Background workers. Policies need a scheduler, and TimescaleDB ships one inside the database. The launcher ([`ts_bgw_cluster_launcher_main`, L860](https://github.com/timescale/timescaledb/blob/main/src/loader/bgw_launcher.c#L860)) runs under the postmaster, spawns a per-database scheduler ([`ts_bgw_scheduler_main`, L1028](https://github.com/timescale/timescaledb/blob/main/src/bgw/scheduler.c#L1028)), which polls the jobs catalog and executes due jobs with [`ts_bgw_job_start`, L80](https://github.com/timescale/timescaledb/blob/main/src/bgw/job.c#L80). Retention, compression, and continuous-aggregate refresh policies are all just rows in that jobs table, which is why the whole policy system survives restarts with no external daemon.

## Honest Limits

- This tour covers the extension sources, not the storage engine: TimescaleDB inherits all of PostgreSQL's storage semantics, including vacuum, WAL, and replication behavior, from the core server it runs inside.
- The README's performance claims (for example, columnstore compression of "90%+ typical" and sub-100ms skipscan queries) are quoted from its quick start, not independently benchmarked here; your numbers will depend on data shape and hardware.
- Compression and continuous aggregates live in `tsl/` under the Timescale License (the community edition, free to use and self-host, but not open-source-licensed in the Apache sense); the core in `src/` and `sql/` is Apache 2. The license split is enforced by the cross-module function interface, visible as `*_stub` fallbacks in `src/cross_module_fn.c`.
- The diagrams reflect `main` at version 2.32.0-dev. The codebase moves quickly — line numbers will drift in future releases, though the module layout has been stable for years.
- Writing directly into compressed chunks works, but the README recommends the batched `convert_to_columnstore` flow after ingestion; direct inserts pay a decompression cost on touched batches.

## Where to Go Next

If you are evaluating storage for machine-learning workloads, TimescaleDB pairs naturally with the rest of the Postgres ecosystem: dashboards over time-series data are a common Grafana use case, and long-term metric storage is exactly what the PromQL-compatible tooling around Prometheus wants from a remote database. If you are comparing engines, our ClickHouse tour covers the columnar-native alternative with a very different architecture trade-off, and if your AI stack needs embeddings next to your time-series rows, pgvector in the same PostgreSQL instance keeps the operational story to one system. For the source itself, start at [github.com/timescale/timescaledb](https://github.com/timescale/timescaledb) and follow the diagram links into the real code.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "TimescaleDB Source Tour: How Hypertables, Chunk Routing, Columnstore, and Continuous Aggregates Actually Work",
  "description": "An annotated architecture tour of the timescale/timescaledb C codebase with interactive diagrams.",
  "datePublished": "2026-10-31",
  "author": { "@type": "Person", "name": "PyShine" },
  "keywords": "timescaledb, postgresql, time-series, hypertable, chunk, columnstore, continuous aggregates",
  "about": "TimescaleDB PostgreSQL extension architecture"
}
</script>
