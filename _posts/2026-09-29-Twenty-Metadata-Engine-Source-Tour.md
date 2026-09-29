---
layout: post
title: "Twenty: A Source Tour of the Metadata Engine Behind the Open-Source CRM - Inside twentyhq/twenty"
description: "A deep source-tour of twentyhq/twenty, the open-source Salesforce alternative CRM built with TypeScript, NestJS, PostgreSQL, and React. Explore its object and field metadata engine, per-workspace PostgreSQL schemas, generated GraphQL APIs, and the workspace migration pipeline that turns UI clicks into DDL."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Twenty-Metadata-Engine-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/twenty/twentyhq-twenty-architecture.svg
tags:
  - CRM
  - Open Source
  - TypeScript
  - GraphQL
categories: [AI, Open Source]
keywords: "twenty CRM, open source CRM, Salesforce alternative, twentyhq/twenty, NestJS, GraphQL schema generation, PostgreSQL multi-tenant, metadata driven architecture, React CRM frontend, twenty ORM, workspace migration, TypeScript monorepo, self-hosted CRM, BullMQ worker, twenty SDK"
author: "PyShine"
---

Most CRMs hide their data model behind a vendor's admin console. You click through screens, wait for support tickets, and accept whatever schema the product team decided you should have. Twenty takes the opposite route: it is an open-source CRM — positioned explicitly as an alternative to Salesforce — where the data model itself is a first-class, user-editable artifact, and where AI agents, workflows, and apps all build on the same primitives. That design decision is exactly what makes its repository worth reading.

The code lives in [twentyhq/twenty](https://github.com/twentyhq/twenty), a large TypeScript monorepo orchestrated with Nx and Yarn 4. The backend (`packages/twenty-server`) is a NestJS application backed by PostgreSQL, Redis, and BullMQ; the frontend (`packages/twenty-front`) is a React application built with Vite, Jotai, and Apollo Client. Alongside them sit a component library (`twenty-ui`), a shared kernel (`twenty-shared`), an application SDK (`twenty-sdk`), a CLI, a Zapier integration, and the marketing site — all in one workspace.

What makes the source genuinely instructive is the machinery that powers "customize your CRM": a metadata engine that stores the definition of every object, field, view, and page layout as data, validates changes, generates database DDL and a GraphQL schema from those definitions, and keeps dozens of caches coherent while it does so. In this tour we walk through that engine, the GraphQL layer that consumes it, and the frontend architecture built on top — citing real paths from the tree so you can follow along.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/twenty/twentyhq-twenty-overview-architecture.svg" alt="Architecture overview of the twentyhq/twenty repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the twentyhq/twenty architecture: the React frontend talks to a NestJS API layer, which splits into core platform modules and the metadata engine; the metadata engine plans changes that the migration pipeline applies to per-workspace PostgreSQL schemas through the Twenty ORM and workspace datasource.*

Reading the overview from left to right: the React entry point (`packages/twenty-front/src/index.tsx`) hydrates a client-side metadata store and issues GraphQL through an Apollo factory; the API layer (`packages/twenty-server/src/engine/api/graphql`) serves both a data API and a metadata API, guarded by the core modules for auth, sessions, and workspaces; metadata requests land in `packages/twenty-server/src/engine/metadata-modules`, which plans every structural change as a workspace migration; and the data side — the Twenty ORM plus the workspace datasource — turns those plans into SQL against a PostgreSQL deployment that keeps a dedicated schema for every workspace, with a BullMQ worker draining asynchronous jobs alongside.

## Why You Need This

If you have ever tried to bend a hosted CRM to your business, you know the pain: fields you cannot add, objects you cannot relate, automations trapped in a proprietary scripting language, and exports that never quite contain what you need. Twenty's answer is to make the CRM schema itself programmable. Companies, contacts, opportunities, notes, and tasks are not hardcoded tables — they are "standard objects" defined through the same metadata system that custom objects use, so the line between product and platform effectively disappears.

The second problem Twenty solves is deployment control. The stack is deliberately conventional — TypeScript, NestJS, PostgreSQL, Redis — so a team can self-host the whole thing with the Docker Compose file shipped in `packages/twenty-docker`, which wires up a `server` container, a `worker` container, a PostgreSQL database, and Redis. Because the license is AGPLv3 for the core (with an explicit enterprise-licensed subset and MIT-licensed SDK/UI packages), you can read every line that handles your customer data and satisfy yourself about what it does.

Third, there is the integration surface. The server exposes GraphQL as its primary API, but `packages/twenty-server/src/engine/api` also contains a REST adapter and an MCP (Model Context Protocol) adapter, so both conventional tooling and AI agents can operate on the same objects the UI uses. The `twenty-sdk` extends this to development: you can define objects, fields, views, agents, and logic functions as TypeScript code, publish them as apps, and version the whole CRM configuration like any other piece of your stack.

Finally, for engineers, the repository is a working reference for a hard problem: multi-tenant systems where each tenant can invent its own schema at runtime. Whether you are building a low-code platform, a headless CMS, or any product that lets users define entities, the patterns in this codebase — flat metadata maps, migration orchestration, schema-per-workspace PostgreSQL — translate directly.

## How It Works

The architecture only makes sense once you see that "metadata" is not a side feature but the spine of the server: every object you see in the UI is a row in metadata tables, and every capability is derived from those rows.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/twenty/twentyhq-twenty-architecture.svg" alt="Detailed architecture of the twentyhq/twenty repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the twentyhq/twenty codebase: the frontend modules, the NestJS API and its generated-schema machinery, the metadata modules and workspace migration pipeline, and the data layer that executes DDL and queries against per-workspace PostgreSQL schemas.*

### Understanding the Architecture

**The metadata catalog.** Everything starts in `packages/twenty-server/src/engine/metadata-modules`, a directory of roughly sixty modules covering objects, fields, views, view filters and sorts, page layouts and widgets, navigation menu items, webhooks, roles, permissions, and more. The `object-metadata` and `field-metadata` modules are the heart of it: their services and GraphQL resolvers accept create/update/delete inputs, validate them, and translate them into operations on *flat* entity structures. Standard objects like companies and people are seeded from `packages/twenty-server/src/engine/workspace-manager/standard-objects-prefill-data` and the `twenty-standard-application` module, so they go through exactly the same pipeline as anything a user creates.

**Flat entity maps and caching.** Rather than querying metadata tables on every request, the server projects them into in-memory "flat" maps — `FlatObjectMetadata`, `FlatFieldMetadata`, and dozens of siblings, all under `packages/twenty-server/src/engine/metadata-modules/flat-entity` and the `flat-*` modules. These maps are loaded and cached through `packages/twenty-server/src/engine/workspace-cache` and invalidated atomically when metadata changes; a metadata version counter (`workspace-metadata-version`) lets the GraphQL layer know when a workspace's generated schema needs to be rebuilt. This is the mechanism that keeps a runtime-editable schema fast enough to serve production traffic.

**The migration pipeline.** When `ObjectMetadataService.updateManyObjects` or `FieldMetadataService.createManyFields` runs (both in `packages/twenty-server/src/engine/metadata-modules`), they do not write SQL directly. Instead they call `WorkspaceMigrationValidateBuildAndRunService` in `packages/twenty-server/src/engine/workspace-manager/workspace-migration/services`, which computes from/to flat-entity maps, expands the operation matrix through a metadata side-effect engine (creating a field also creates its views, search vectors, and relation companions), and hands a validated migration to `WorkspaceMigrationRunnerService`. The runner applies each action via a registry of handlers, then invalidates the affected caches and bumps the metadata version — a clean, auditable transaction around every structural change.

**A GraphQL schema generated per workspace.** Because each workspace invents its own objects, there cannot be one static GraphQL schema. `packages/twenty-server/src/engine/api/graphql/workspace-schema-builder` generates one at runtime: `WorkspaceGraphQLSchemaGenerator` and its `GqlTypeGenerator` walk the flat object/field metadata and emit object types, input types, filter and order-by types, Relay-style connections, aggregations, and group-by types via the generators under `graphql-type-generators`. Execution then flows through `graphql-query-runner`, whose parsers translate GraphQL filters, sorts, and selected fields into TypeORM expressions. The same engine also powers the REST and MCP adapters, so all three APIs share one semantic core.

**The Twenty ORM and per-workspace schemas.** The data itself lives in PostgreSQL with a schema per workspace — `packages/twenty-server/src/engine/workspace-datasource/workspace-schema.service.ts` creates (or drops) a schema named from the workspace ID, and the datasource resolves connections with the right `search_path`. Above it, `packages/twenty-server/src/engine/twenty-orm` provides workspace-scoped repositories (injected via `@InjectWorkspaceScopedRepository`), select and mutation query builders, and the `workspace-schema-manager`, whose table, column, index, foreign-key, and enum manager services are the actual DDL executors the migration runner calls. One database, many tenants, zero shared mutable tables.

**The frontend reads metadata like data.** On the client, `packages/twenty-front/src/index.tsx` hydrates the Jotai-based metadata store (`packages/twenty-front/src/modules/metadata-store`) before rendering, so navigation, views, and fields are available synchronously. The Apollo setup in `packages/twenty-front/src/modules/apollo` wires the generated client and optimistic effects, while `packages/twenty-front/src/modules/object-record` — with submodules like `record-table`, `record-show`, `record-board`, and `record-calendar` — renders any object generically from that metadata. The settings-side `object-metadata` module is, fittingly, just another consumer: it drives the object designer through ordinary metadata mutations.

Trace one request end to end and the design clicks: a user drags a new field onto a view; the frontend sends a metadata mutation; `FieldMetadataService` plans the change; the side-effect engine expands it; the migration builder diffs flat maps; the runner executes DDL through the schema manager inside the workspace's PostgreSQL schema; caches invalidate and the metadata version ticks up; the next GraphQL request regenerates or reuses the per-workspace schema; and the record table refetches with the new column — all without a deploy, a lockstep migration file, or a restart.

## Advantages

- **Metadata-driven everything.** Objects, fields, views, page layouts, and roles are data, not code, so customization happens at runtime through the same system the product itself uses.
- **A real API-first core.** GraphQL is generated per workspace from live metadata, with REST and MCP adapters layered on the same engine — one source of truth for the UI, integrations, and AI agents.
- **Isolated multi-tenancy.** A dedicated PostgreSQL schema per workspace (`workspace-datasource`, `workspace-schema-manager`) keeps tenant data physically separated while remaining a single database to operate.
- **Transactional schema changes.** The validate/build/run migration pipeline in `workspace-manager/workspace-migration` makes structural edits atomic, side-effect-aware, and observable, instead of scattering ad-hoc DDL.
- **Developer workflow as code.** The `twenty-sdk`'s `defineObject`/`defineField` DSL, `create-twenty-app` scaffolding, and the `twenty` CLI let teams version CRM configuration in git and publish it like software.
- **Familiar, boring-on-purpose stack.** NestJS, TypeORM, PostgreSQL, Redis, BullMQ, React, and Apollo are all mainstream choices, which lowers the barrier for self-hosters and contributors alike.

## Benefits

- **Escape the per-seat black box.** Self-hosting the AGPLv3 core means full visibility into and control over the system that holds your customer relationships.
- **Adapt the CRM to the business, not the reverse.** Custom objects and relations are created and evolved in production through the metadata engine, without vendor tickets or downtime.
- **Automate with confidence.** Workflows, logic functions, and AI agents operate on the same metadata-defined objects and the same GraphQL API, so automations keep working as the schema evolves.
- **Integrate anything.** Webhooks, the Zapier package, the REST/GraphQL/MCP surfaces, and the client SDK give every consumer a contract that is generated from the actual schema.
- **Learn transferable architecture.** The flat-entity caching, per-workspace schema generation, and migration orchestration patterns are directly reusable in any platform that lets users define entities.
- **Contribute meaningfully.** The Nx monorepo layout, package-level READMEs, and clear module boundaries (`engine`, `modules`, `queue-worker`) make the codebase navigable for outside contributors.

## Usage

Scaffold a new Twenty app with the CLI, defining objects as typed code:

```bash
npx create-twenty-app my-app
```

```ts
import { defineObject, FieldType } from 'twenty-sdk/define';

export default defineObject({
  nameSingular: 'deal',
  namePlural: 'deals',
  labelSingular: 'Deal',
  labelPlural: 'Deals',
  fields: [
    { name: 'name', label: 'Name', type: FieldType.TEXT },
    { name: 'amount', label: 'Amount', type: FieldType.CURRENCY },
    { name: 'closeDate', label: 'Close Date', type: FieldType.DATE_TIME },
  ],
});
```

Publish it to your workspace:

```bash
npx twenty app:publish --private
```

For self-hosting, the repository ships `packages/twenty-docker/docker-compose.yml`, which defines `server`, `worker`, `db` (PostgreSQL), and `redis` services around the `twentycrm/twenty` image; copy `packages/twenty-docker/.env.example`, set `SERVER_URL`, `PG_DATABASE_URL`, and `REDIS_URL`, then start the stack:

```bash
docker compose up -d
```

For local development of the monorepo itself, the root `package.json` provides a `start` script that boots `twenty-server` and `twenty-front` concurrently and then launches the worker; the README links the full local setup guide in the documentation.

## Conclusion

Twenty is more than an open-source CRM checklist — it is a demonstration that a CRM can treat its own schema as a living, versioned artifact. The source tour above shows how `metadata-modules` catalog structure, how the `workspace-migration` pipeline turns edits into safe DDL, how a GraphQL schema is generated per workspace on top of flat metadata maps, and how the React frontend consumes the same metadata to render anything generically. If you are evaluating a Salesforce alternative, building a low-code platform, or just want to study a serious TypeScript implementation of runtime-extensible data models, cloning [twentyhq/twenty](https://github.com/twentyhq/twenty) and following the paths in this post is time well spent.

Links:

- GitHub repository: [https://github.com/twentyhq/twenty](https://github.com/twentyhq/twenty)
- Documentation: [https://docs.twenty.com](https://docs.twenty.com)
- App development guide: [https://docs.twenty.com/developers/extend/apps/getting-started](https://docs.twenty.com/developers/extend/apps/getting-started)
