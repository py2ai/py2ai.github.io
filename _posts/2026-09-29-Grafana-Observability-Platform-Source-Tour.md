---
layout: post
title: "Grafana: Query, Visualize, and Alert on Any Data Source - Inside grafana/grafana"
description: "A guided source tour of grafana/grafana, the open-source observability platform. We walk the Go backend, the TSDB query layer, the datasource plugin model, the React dashboard runtime, and the ngalert alerting engine with two architecture diagrams."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Grafana-Observability-Platform-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/grafana/grafana-grafana-architecture.svg
tags:
  - Observability
  - Go
  - Data Visualization
  - Open Source
categories: [AI, Open Source]
keywords: "Grafana, observability, dashboards, data visualization, datasource plugins, Go backend, React frontend, Grafana alerting, tsdb, open source monitoring, Grafana architecture, plugin SDK, ngalert"
author: "PyShine"
---

If you have ever stared at a wall of dashboards during an incident, you have almost certainly used Grafana. What is easy to forget is that Grafana is not a metrics database at all. It stores no time series itself; it is the connective tissue — a query, visualization, and alerting layer that speaks to whatever storage you already run. Reading its source is one of the best lessons in how to build a system whose whole job is to adapt to other systems.

This repository, `grafana/grafana`, is the open-source platform for monitoring and observability described in its own README: query, visualize, alert on, and understand your metrics no matter where they are stored, with dynamic dashboards, Explore views for ad-hoc queries, and mixed data sources in a single panel. The backend is written in Go (see `go.mod`, currently targeting Go 1.26), the frontend is a React/TypeScript application under `public/app`, and the two halves meet at a well-defined HTTP boundary and the Grafana plugin SDK.

Why bother with a source tour? Because the questions that matter in real deployments — how a panel query reaches your database, how credentials stay inside the server, how alert rules get evaluated on schedule — are all answered in concrete files here. Grafana is also a masterclass in plugin architecture: nearly everything, including core datasources, flows through the same backend plugin contract. Follow the code once and you will understand every datasource, every panel, and the alerting engine as a bonus.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/grafana/grafana-grafana-overview-architecture.svg" alt="Architecture overview of the grafana/grafana repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the grafana/grafana architecture: the Go entry point and server lifecycle, the HTTP query path into the plugin layer, the React dashboard runtime, the alerting engine, and the SQL store that ties them together.*

Reading the overview from left to right: the `grafana` binary starts in `pkg/cmd/grafana/main.go`, which boots the service lifecycle in `pkg/server/server.go` and the SQL store in `pkg/services/sqlstore`. The server runs the HTTP API in `pkg/api/http_server.go`, which both serves the React frontend and routes query traffic from `pkg/api/ds_query.go` into the query service in `pkg/services/query/query.go`. From there, requests either hit the expression engine in `pkg/expr` or are handed to the plugin integration layer in `pkg/services/pluginsintegration`, which executes built-in backends under `pkg/tsdb`; the alerting engine in `pkg/services/ngalert` reuses that same query path on a schedule. The frontend mirrors the flow: dashboard scenes under `public/app/features/dashboard-scene` issue queries through the runner in `public/app/features/query/state/runRequest.ts`, landing on the same `/ds/query` endpoint.

## Why You Need This

The first problem Grafana solves is heterogeneity. Modern infrastructure does not store telemetry in one place: metrics live in Prometheus or a cloud vendor, logs in Loki or Elasticsearch, traces in Tempo or Jaeger, and the rest in assorted SQL databases and SaaS APIs. Grafana's answer is the datasource abstraction — any backend implementing the SDK's query interface becomes a first-class citizen, and a panel can compose queries across sources, including the special "mixed" datasource (`public/app/plugins/datasource/mixed/MixedDataSource.ts`) that fans one panel out to several backends at once.

The second problem is operational: turning raw telemetry into shared, reviewable artifacts. Dashboards with template variables, library panels, snapshots, folders, and permissions are server-side objects, persisted through the dashboard service (`pkg/services/dashboards/service`) and the SQL store, and importable through provisioning. Because dashboards live on the server, they can be version-controlled, provisioned into clusters, and shared across teams — the "dashboard as code" workflow.

The third problem is vigilance. Humans do not watch graphs at 3 a.m.; alert rules do. Grafana ships a full alerting engine under `pkg/services/ngalert`, which schedules rule groups, evaluates conditions against any datasource, tracks state transitions over time, and routes firing alerts to contact points through its own Alertmanager implementation (`pkg/services/ngalert/notifier/alertmanager.go`). Because alerts evaluate through the same query pipeline as panels, an alert and the graph that justifies it can never silently disagree about the data.

Finally, there is the extensibility problem. Every organization eventually has a system nobody else monitors. Grafana's plugin SDK (`github.com/grafana/grafana-plugin-sdk-go`, imported throughout `pkg/tsdb` and `pkg/plugins`) defines a stable process-based contract: implement `QueryDataHandler`, declare a `plugin.json`, and your service appears in the datasource picker with signed verification and lifecycle management handled by the host. Reading how Grafana consumes its own contract is the fastest way to write good plugins for it.

## How It Works

The whole system hangs off one idea: a single dependency-injected server whose services cooperate through narrow interfaces, with queries flowing through one plugin pipeline regardless of destination.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/grafana/grafana-grafana-architecture.svg" alt="Detailed architecture of the grafana/grafana repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of grafana/grafana: the Wire bootstrapping and config layer, HTTP routes and the unified /apis server, the query pipeline with server-side expressions, the plugin integration stack with built-in TSDB backends, the React dashboard runtime, and the ngalert scheduler-evaluator-notifier trio.*

### Understanding the Architecture

**The boot sequence is dependency injection, not globals.** `pkg/cmd/grafana/main.go` builds a `urfave/cli` application whose side-effect import of `pkg/server/bootstrap/wire` registers the OSS Wire graph. `pkg/server/server.go` owns the lifecycle: its `Server` type initializes and runs registered services — the HTTP server, background services, tracing, feature toggles — and shuts everything down cleanly. Configuration lives in `pkg/setting`, driven by `conf/defaults.ini` with environment overrides, while `pkg/services/featuremgmt` implements the feature toggles that gate new behavior across both halves of the codebase.

**HTTP is the front door, in two flavors.** The classic surface lives in `pkg/api/http_server.go` with route registration in `pkg/api/api.go` — dashboards, folders, users, annotations, and the frontend's boot data. The newer surface is the unified API server in `pkg/services/apiserver`, which mounts Kubernetes-style `/apis/...` endpoints for app modules registered under `pkg/registry/apis` (the `apps/` tree at the repo root is built the same way). Query traffic arrives at `POST /ds/query` in `pkg/api/ds_query.go`; when the `queryServiceRewrite` feature flag is on, the same handler rewrites requests to the `datasource.grafana.app` query endpoint.

**The query service is a scheduler for queries, not a database driver.** In `pkg/services/query/query.go`, an incoming multi-query request is parsed into individual queries grouped by datasource UID, stamped with diagnostic headers (`X-Plugin-Id`, `X-Dashboard-Uid`, `X-Panel-Id`), and executed with bounded concurrency — tunable from the `[query]` config section, defaulting to the number of CPUs. Expressions like math, reduce, and threshold operations are recognized and delegated to the server-side expression engine in `pkg/expr`, which builds a small evaluation graph (`pkg/expr/graph.go`) and calls back into the plugin client to fetch the inputs its nodes need.

**Everything below the query service is a plugin.** `pkg/services/pluginsintegration` is the integration heart: it loads and validates plugins (`pluginstore`, signature verification, CDN resolution via `pluginscdn`), manages their processes (`pkg/plugins/backendplugin`), resolves per-request plugin context in `plugincontext` (org, user, datasource settings, decrypted secrets), and wraps every outgoing call in the client middleware chain in `clientmiddleware` — HTTP middleware's pattern, applied to plugin calls. Built-in backends live in `pkg/tsdb` — Azure Monitor, CloudWatch, Graphite, the test-data source, and `grafanads`, the pseudo-datasource in `pkg/tsdb/grafanads/grafana.go` handling the `-- Grafana --` internal source. Datasources not on that list arrive as external plugins through the identical contract, which is the point.

**The frontend is a runtime in its own right.** `public/app/app.ts` wires up the platform singletons from `@grafana/runtime` (in `packages/grafana-runtime`) — the data source service, query runner factory, location services, plugin hooks — before any page renders. Dashboards are modeled as reactive "scenes": `public/app/features/dashboard-scene/scene/DashboardScene.tsx` owns the dashboard object graph, variables, and panel layout, hosting the panel runtime in `public/app/features/panel` where each panel plugin, from the workhorse `TimeSeriesPanel.tsx` down to canvas and geomap, receives transformed data frames. The query runner in `public/app/features/query/state/runRequest.ts` orchestrates retries, cancellation, and mixed-datasource splitting before issuing `POST /api/ds/query`.

**Alerting is a second, quieter query consumer.** Inside `pkg/services/ngalert`, `schedule/schedule.go` ticks rule groups — each rule runs on its own channel and routine — handing conditions to `eval/eval.go`, which executes them through the same query service panels use. `state/manager.go` persists state transitions so restarts do not lose history, and `notifier/alertmanager.go` implements the Alertmanager protocol for routing, grouping, and silencing notifications — a design Prometheus operators will find familiar.

One end-to-end walk ties it together: you open a dashboard, the scene runtime resolves template variables and asks `runRequest.ts` to run each panel's queries; the runner splits any mixed queries and posts to `/api/ds/query`; `pkg/api/ds_query.go` hands the payload to the query service, which groups queries per datasource, evaluates expressions in `pkg/expr`, and dispatches the rest through the plugin integration client with middlewares and decrypted credentials attached; a built-in TSDB backend or an external plugin process returns data frames; the browser renders them. If one of those queries is instead an alert condition, the scheduler evaluates it on its tick and the notifier wakes the on-call rotation — the same path, a different consumer.

## Advantages

- **Genuinely datasource-agnostic.** Every backend, built-in or external, implements the same `QueryDataHandler` contract from the plugin SDK, so visualization, alerting, and exploration work uniformly across Prometheus, Loki, CloudWatch, SQL, and whatever you bolt on.
- **Server-side expressions close the loop.** The `pkg/expr` engine composes math, reduce, and threshold operations across sources on the server, so derived queries do not depend on client compute and can be reused by alerts.
- **One plugin pipeline, many middlewares.** Caching, tracing, metrics, and OAuth token propagation are applied as client middlewares in `pkg/services/pluginsintegration/clientmiddleware`, keeping datasource implementations small and cross-cutting concerns centralized.
- **Alerting shares the query brain.** Because `ngalert` evaluates through the same query service, alert definitions and dashboards stay consistent, and rule conditions can reference any datasource a panel can.
- **Process-isolated plugins.** Plugin backends run out-of-process behind a versioned contract in `pkg/plugins/backendplugin`, with signature checks and lifecycle stages that contain misbehaving third-party code.
- **Modern, hackable frontend.** The React/TypeScript app with the scene model in `public/app/features/dashboard-scene` and a real runtime layer in `packages/grafana-runtime` makes dashboard behavior inspectable and extensible, with an ongoing webpack-to-rspack build migration documented in the developer guide.

## Benefits

- **Faster incident response.** Mixed datasources, Explore split views, and correlated metrics-logs-traces navigation mean fewer tool switches when something is on fire.
- **One platform for many teams.** Folders, permissions, library panels, and org-scoped configuration in the SQL store let a single deployment serve multiple teams without dashboard sprawl.
- **Alerting you can reason about.** Scheduled per-rule evaluation, persistent state management, and a Prometheus-compatible notifier design make Grafana alerting predictable under failure and restart.
- **A stable extension business.** The signed plugin model plus the plugin CDN lets organizations distribute internal datasources safely.
- **Operational transparency.** Diagnostic headers on every query (`X-Dashboard-Uid`, `X-Panel-Id`, `X-Query-Group-Id`) let you trace a slow graph back to the exact panel that caused it.
- **A reference architecture for Go services.** Wire-based dependency injection, feature toggles, and the unified API server pattern in `pkg/services/apiserver` are reusable lessons for your own Go platform.

## Usage

The quickest way to run Grafana is Docker, as documented in the repository's installation guide (`docs/sources/setup-grafana/installation/docker/index.md`):

```bash
docker run -d -p 3000:3000 --name=grafana grafana/grafana-enterprise
```

Point your browser at `http://localhost:3000`; the container serves the platform on port 3000, persisting to its embedded SQLite database unless you mount a volume.

To build and run from source, follow `contribute/developer-guide.md`. First install frontend dependencies and start the asset build:

```bash
corepack enable
corepack install
yarn install --immutable
yarn start
```

Then, in a second terminal, compile and run the Go backend from the repository root:

```bash
make run
```

Log in with the default credentials `admin` / `admin` (the password change is requested on first login). For a hot-reload frontend workflow, the guide documents the rspack path: run `yarn start:rspack` alongside `RSPACK=1 make run`.

## Conclusion

Grafana's source answers the question every observability tool eventually faces: how do you stay neutral about where data lives while still delivering fast, reliable experiences on top of it? The answer in `grafana/grafana` is architectural discipline — a lifecycle-managed Go server, a single query pipeline, a plugin contract that even Grafana's own backends obey, and a React runtime that treats dashboards as composable scene graphs. Whether you are choosing an observability stack, writing a datasource plugin, or designing your own extensible platform, this codebase repays a careful reading.

Links:

- GitHub repository: [https://github.com/grafana/grafana](https://github.com/grafana/grafana)
- Official documentation: [https://grafana.com/docs/](https://grafana.com/docs/)
- Developer guide (in-repo): [https://github.com/grafana/grafana/blob/main/contribute/developer-guide.md](https://github.com/grafana/grafana/blob/main/contribute/developer-guide.md)
