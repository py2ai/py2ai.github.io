---
layout: post
title: "Netdata: Real-Time Infrastructure Monitoring - Inside netdata/netdata"
description: "A guided source-tour of netdata/netdata, the open-source real-time infrastructure monitoring agent. We trace per-second metric collection through the dbengine storage tier, parent-child streaming with replication, and the edge-based machine learning that flags anomalies on every sample."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Netdata-Real-Time-Infrastructure-Monitoring-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/netdata/netdata-netdata-architecture.svg
tags:
  - Netdata
  - Monitoring
  - Observability
  - Open Source
categories: [AI, Open Source]
keywords: "netdata, real-time monitoring, infrastructure monitoring, dbengine, time-series database, anomaly detection, netdata cloud, streaming replication, per-second metrics, open source monitoring, prometheus alternative, metric collection, observability pipeline"
author: "PyShine"
---

Most monitoring tools assume fine-grained data is expensive, so they sample, average, and round down. By the time a spike reaches your dashboard it has been diluted into a five-minute mean. Netdata, the open-source project behind `netdata/netdata`, starts from the opposite premise: collect every metric every second, keep it locally, and make "what changed right now?" instantly answerable. The repository hosts the Netdata Agent — the C program that runs on servers, containers, and edge devices — and it is one of the most substantial monitoring codebases you can read today.

What you get when you install it is a full observability pipeline in a single agent: automatic collection from systems, containers, apps, and hardware sensors; a purpose-built time-series database called dbengine; unsupervised machine learning that scores every value as normal or anomalous; hundreds of pre-configured alerts; streaming and replication to parent nodes; export connectors for Prometheus, Graphite, OpenTSDB, and others; and a web API with auto-generated dashboards on port 19999. Netdata Cloud is optional, adds multi-node views, and does not centralize metric storage — the agent is designed to be fully useful with its data staying on your infrastructure.

This repository deserves a source tour because almost nothing in it is a thin wrapper around someone else's storage or transport: the database, the streaming protocol, the ML inference path, and the alerting engine are all implemented in-tree — mostly C, a Go plugin for application collectors, and a slice of C++ for the ML runtime. Following one metric sample from a collector thread to a compressed on-disk extent, with an anomaly bit riding along on every point, teaches you more about high-resolution monitoring than any product page.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/netdata/netdata-netdata-overview-architecture.svg" alt="Architecture overview of the netdata/netdata repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the netdata/netdata architecture: collectors feed the RRD metric core, which fans samples out to dbengine storage, machine learning, alerting, streaming parents, export connectors, and the web API, with ACLK linking the agent to Netdata Cloud.*

Reading the overview from left to right: the collection group gathers data three ways — in-process C collectors under `src/collectors`, the embedded Go plugin booting at `src/go/cmd/godplugin/main.go`, and external processes speaking the plugins.d protocol parsed by `src/plugins.d/plugins_d.c`. All converge on the RRD metric core (`src/database/rrd.h`). From there the flow branches: pages go to the dbengine storage engine (`src/database/engine/rrdengine.c`), every sample is scored by ML (`src/ml/ml.cc`), alert expressions run in the health engine (`src/health/health.c`), and the same stream is duplicated outward to parents (`src/streaming/stream-sender.c`), third-party backends (`src/exporting/exporting_engine.c`), and Netdata Cloud (`src/aclk/aclk.c`), while the web API (`src/web/api/web_api_v2.c`) closes the loop for dashboards and automation.

## Why You Need This

The first problem Netdata solves is resolution. Traditional stacks poll on minute-scale intervals — tolerable for capacity planning, useless for triage: a two-second CPU stall does not survive minute averaging. Netdata's collectors run per-second by default, and everything downstream preserves that fidelity: storage, queries, and dashboards all treat one-second granularity as the norm, as the README's key-features table states.

The second problem is setup cost. A conventional metrics stack asks you to pick an agent, provision a TSDB, wire exports, and build dashboards before you see a single chart. Netdata's collectors auto-detect most of what they monitor — the catalog in `src/collectors/COLLECTORS.md` advertises 850+ integrations covering databases, web servers, message queues, Kubernetes, and cloud providers. Start the agent, open `http://localhost:19999`, and the dashboard is already populated, because charts are created programmatically as collectors register metrics.

The third problem is diagnosis rather than collection. With thousands of metrics, the difficulty shifts from gathering signals to noticing which one misbehaved. That is where the ML module earns its place: Netdata trains lightweight unsupervised models per metric at the edge, and every sample carries an anomaly bit you can overlay on any chart. Finally, durability and scale: agents stream to parents in real time, replication backfills whatever a parent missed during an outage, and tiered storage trades resolution for retention as data ages — one agent on a Raspberry Pi and a tree of parents ingesting from hundreds of children use the same code paths.

## How It Works

Follow one metric sample from a collector to a disk extent and back out through a query.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/netdata/netdata-netdata-architecture.svg" alt="Detailed architecture of the netdata/netdata repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source-tour map of netdata/netdata: collection plugins, the RRD metric core, dbengine internals, the ML pipeline, streaming and replication, and the delivery subsystem for alerts, exports, APIs, and Netdata Cloud.*

### Understanding the Architecture

**Collection is deliberately heterogeneous.** Under `src/collectors` sit the platform plugins: `proc.plugin` and `cgroups.plugin` for Linux kernel and container metrics, `apps.plugin` for per-process accounting, and the FreeBSD, macOS, and Windows plugins. Application collectors run externally — the Go plugin boots from `src/go/cmd/godplugin/main.go` into the runtime at `src/go/plugin/agent/agent.go`, with discovery under `src/go/plugin/go.d/discovery` — and anything not in-process speaks the plugins.d protocol parsed by `src/plugins.d/plugins_d.c`. The supervising daemon starts in `src/daemon/main.c`, bringing up the service threads (streaming, ML workers, the health event loop, dbengine flushers), while `src/daemon/service.c` archives obsolete charts and dimensions when collectors stop updating them — essential in dynamic environments like Kubernetes.

**The RRD core is the meeting point of the whole system.** Every host is an RRDHOST (`src/database/rrdhost.c`), every chart an RRDSET, every series an RRDDIM; the per-second update pass lives in `src/database/rrdset-collection.c` and `src/database/rrddim-collection.c`. Right in the collection loop sits a call to `ml_dimension_is_anomalous()`: the hook where ML judges each freshly collected value before it is stored. The verdict travels as a flag bit packed into Netdata's custom storage-number format (`src/libnetdata/storage_number/storage_number.c`), so anomaly info rides on the data itself. A storage-engine abstraction in `src/database/storage-engine.c` dispatches to the backing database — dbengine by default, `ram`/`alloc` modes for ephemeral setups.

**dbengine turns per-second samples into compressed, append-only files.** The core in `src/database/engine/rrdengine.c` appends points to per-metric hot pages (`src/database/engine/page.c`) held in the page cache (`src/database/engine/pagecache.c`). When a page fills it becomes dirty and is queued for flushing; dbengine packs up to 64 dirty pages into one extent, compresses it with LZ4, and appends it to a datafile (`.ndf`, `src/database/engine/datafile.c`), while journal files (`src/database/engine/journalfile.c`) record where every extent landed — journal v1 for crash recovery, journal v2 as a memory-mapped index. The LRU machinery in `src/database/engine/cache.c` manages memory via "memory ballooning", sized from the actively collected metric count, and `src/database/engine/mrg.c` tracks every retained metric. Above the raw storage sit three tiers: tier 0 keeps per-second points, tier 1 aggregates sixty of them, tier 2 sixty of tier 1 — updated in real time and backfilled on agent start.

**ML is many small models, not one big one.** Each dimension's work flows through the queue (`src/ml/ml_queue.cc`), into feature vectors (`src/ml/ml_features.cc`), and out to k-means training and scoring (`src/ml/ml_kmeans.cc`, using the lightweight dlib library). Netdata trains several models per metric over overlapping windows covering roughly the last two days; a sample is flagged anomalous only when all models agree — an ensemble design the ML docs credit with suppressing false positives. Detection compares the Euclidean distance of the current feature vector to the learned cluster centers against a 99th-percentile threshold of training distances, and the API surfaces the result via `options=anomaly-bit`.

**Streaming turns any agent into a hub or a spoke.** A child's per-host sender (`src/streaming/stream-sender.c`) connects to its parent over Netdata's custom binary protocol — the same port 19999 that serves the web API — pushing samples and metadata through a circular buffer (`src/streaming/stream-circular-buffer.c`). On the receiving side, `src/streaming/stream-receiver.c` feeds incoming commands through the same plugins.d parser used by local plugins, so a parent stores a child's data through an identical path. The replication pair (`src/streaming/stream-replication-sender.c`, `stream-replication-receiver.c`) handles history: after a disconnect, the sender reads missing extents from its dbengine datafiles and backfills the parent.

**Delivery is where data becomes useful to humans and other systems.** The health engine (`src/health/health.c`, event loop in `src/health/health_event_loop.c`) evaluates alert expressions against RRD values, with stock alerts in `src/health/health.d` and notifications under `src/health/notifications`. The exporting engine (`src/exporting/exporting_engine.c`) mirrors metrics to Prometheus, Graphite, OpenTSDB, MongoDB, AWS Kinesis, and Pub/Sub. Queries resolve through the context registry (`src/database/contexts/rrdcontext.c`) and the query target planner (`src/database/contexts/query_target.c`), which picks the storage tier matching the zoom level. For fleets, `src/claim/claim.c` registers the agent with Netdata Cloud and `src/aclk/aclk.c` maintains the MQTT-based link carrying contexts, alert events, and cloud-initiated queries.

End to end, one sample's journey: a collector reads a value, the chart loop in `src/database/rrdset-collection.c` stamps it, ML attaches the anomaly bit via `src/libnetdata/storage_number/storage_number.c`, `src/database/rrddim-collection.c` hands it to dbengine where it lands on a hot page and eventually a compressed extent, the streaming sender replicates it to parents while the exporting engine mirrors it outward, the health engine checks it against alert expressions, and a dashboard query reads it back seconds later from whichever tier matches the zoom level — all without a query language or a server outside the node.

## Advantages

- **Per-second resolution everywhere.** Collection, storage, ML, alerting, and dashboards all operate at one-second granularity, so brief incidents stay visible.
- **Purpose-built storage engine.** dbengine (`src/database/engine`) implements page compression, tiered retention, journal recovery, and memory ballooning in-tree, tuned for one workload: a monitoring agent.
- **Edge-first machine learning.** Unsupervised k-means models train per metric on the node itself, the anomaly bit costs no extra storage, and multi-model consensus keeps noise low.
- **Streaming with replication.** Parent-child streaming over a dedicated binary protocol includes historical backfill, so parents hold complete data even across child outages.
- **Your data stays yours.** The agent runs and stores locally; Netdata Cloud is optional and does not centralize metric storage, compatible with strict or air-gapped deployments.

## Benefits

- **Faster root-cause analysis.** Correlated anomaly rates across every metric, surfaced by the Anomaly Advisor, turn "something is wrong" into "these subsystems changed together at 14:02".
- **Lower total cost of monitoring.** One agent replaces the usual agent-plus-TSDB-plus-dashboard assembly, and compact storage keeps high-resolution retention affordable.
- **Gentle resource footprint.** The agent is engineered for always-on operation on production nodes and small devices, with self-monitoring charts exposing what it consumes.
- **Resilient by construction.** Append-only storage with journal recovery, circular send buffers, and replication mean crashes and disconnects degrade gracefully instead of losing your window of evidence.
- **Open and hackable.** GPLv3+ agent code, clear boundaries between subsystems, and a documented plugins.d protocol make it practical to extend or embed.

## Usage

Quickest install on Linux, via the repository's kickstart script:

```sh
wget -O /tmp/netdata-kickstart.sh https://get.netdata.cloud/kickstart.sh && sh /tmp/netdata-kickstart.sh --dont-wait
```

To build the agent from this source tree (CMake with Ninja, per `packaging/installer/methods/source.md`):

```sh
cmake -S . -B build -G Ninja
cmake --build build
cmake --install build
```

The README notes the UI is available at `http://localhost:19999` (or `http://NODE:19999` remotely). To centralize a fleet, enable streaming on each child in `stream.conf`:

```conf
[stream]
    enabled = yes
    destination = parent1:19999 parent2:19999
```

and tune retention by allocating disk space per dbengine tier in `netdata.conf`, as described in `src/database/CONFIGURATION.md`.

## Conclusion

Netdata is one of those projects where the source code substantiates the pitch. The claims about per-second collection, compact storage, edge ML, and replicated parents all trace back to concrete, readable implementations: the collection loop in `src/database/rrdset-collection.c`, the page and extent machinery in `src/database/engine/`, the k-means ensemble in `src/ml/`, and the streaming code in `src/streaming/`. Whether you deploy it as a single-node dashboard or a multi-tier pipeline, the agent is a complete, self-contained monitoring stack — and an instructive codebase to read.

Links:

- [netdata/netdata on GitHub](https://github.com/netdata/netdata)
- [Netdata documentation](https://learn.netdata.cloud)
- [Netdata Agent license (GPLv3+)](https://github.com/netdata/netdata/blob/master/LICENSE)
