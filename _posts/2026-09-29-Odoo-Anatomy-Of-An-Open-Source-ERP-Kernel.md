---
layout: post
title: "Odoo: Anatomy of an Open-Source ERP Kernel - Inside odoo/odoo"
description: "A guided source-tour of the odoo/odoo monorepo: the per-database ORM registry, BaseModel and its field system, the module dependency loader behind 642 addons, the JSON-RPC HTTP layer, and the OWL component framework that renders the web client."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Odoo-Anatomy-Of-An-Open-Source-ERP-Kernel/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/odoo/odoo-odoo-architecture.svg
tags:
  - Odoo
  - Python
  - ERP
  - Architecture
categories: [AI, Open Source]
keywords: "Odoo, odoo/odoo, open source ERP, Odoo ORM, BaseModel, OWL framework, Odoo architecture, Python monorepo, JSON-RPC, module registry, PostgreSQL, addons, source code tour"
author: "PyShine"
---

Most business software hides its architecture behind a marketing site. Odoo does the opposite: the entire thing — server, ORM, HTTP framework, module system, and web client — is one Python-and-JavaScript monorepo you can download and read. If you have ever wondered how a company ships CRM, accounting, inventory, eCommerce, manufacturing, and point-of-sale from a single codebase, the answer is sitting in `odoo/orm/`, `odoo/modules/`, `odoo/http/`, and `addons/web/`, and it is remarkably legible.

Odoo, published as `odoo/odoo` on GitHub under the LGPLv3 license, describes itself in its README as "a suite of web based open source business apps" whose apps also integrate seamlessly into a full ERP. The tree we toured is the `master` branch, version 20.1 by `odoo/release.py`. It contains 642 addon modules discovered by manifest file, 575 Python files inside the core `odoo/` package, and a vendored JavaScript framework for the frontend. This is not a framework you adopt; it is a platform you extend.

That last point is exactly why the source is worth a tour. Odoo is one of the few large systems where every business feature is written against the same public extension surface it ships with: a module is a directory with a `__manifest__.py`, models are Python classes, views are data records, and the web client is a component framework any addon can plug into. Reading the kernel therefore teaches you two things at once — how Odoo works, and a battle-tested design for plugin architectures in general.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/odoo/odoo-odoo-overview-architecture.svg" alt="Architecture overview of the odoo/odoo repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the odoo/odoo source layout: the `odoo-bin` launcher and server processes, the HTTP/RPC layer, the addon loading machinery, the ORM kernel that talks to PostgreSQL, and the OWL-based web client.*

Reading the overview from left to right: the `odoo-bin` script hands off to the CLI package in `odoo/cli/`, which boots the threaded or pre-forked server processes in `odoo/service/server.py`. That server hosts the HTTP framework in `odoo/http/`, which routes JSON-RPC traffic into the model dispatch service at `odoo/service/model.py`. Requests resolve against the per-database model registry in `odoo/orm/registry.py`, which has been populated by the module loader in `odoo/modules/loading.py` from the 642 manifests under `addons/` plus the special kernel module at `odoo/addons/base`. Models then talk to PostgreSQL through `odoo/sql_db.py`, while the browser side — the OWL web client under `addons/web/static/src` — calls the same HTTP endpoints.

## Why You Need This

If you build any backend that other developers extend, Odoo's kernel solves a set of problems you will eventually face, and it solves them in the open. The first problem is multi-tenancy with per-tenant schema and behavior. Odoo hosts many databases on one server, and each database can have a different set of installed modules — meaning a different set of model classes. The file `odoo/orm/registry.py` opens with the design in one sentence: "A `Registry` object is instantiated per database, and exposes all the available models for its database." Instead of one global class hierarchy, every database gets its own assembled one.

The second problem is integrating dozens of business domains without them collapsing into each other. Sales needs inventory, invoicing needs sales, accounting needs both. Odoo's answer is the module dependency graph in `odoo/modules/module_graph.py`, whose source literally diagrams `base` at the root with everything else pointing up, and the loader in `odoo/modules/loading.py`, which forces `base` to load first (`_FORCED_MODULES = ('base',)`) and then installs modules in dependency order. Cross-module behavior is expressed through model inheritance rather than code forks.

The third problem is the frontend. An ERP needs hundreds of list, form, kanban, and pivot views, all dynamically assembled per user, per company, and per installed module. Odoo ships its own component framework, OWL, vendored at `addons/web/static/lib/owl/owl.js`, and builds the entire web client on top of it in `addons/web/static/src/`. Even the data layer on the browser side is abstracted: `addons/web/static/src/model/relational_model` keeps client-side records in sync with the server.

Finally, there is the integration problem. Every ERP must expose its data to other systems. Odoo's HTTP layer in `odoo/http/` provides JSON-RPC endpoints dispatched through `odoo/service/model.py`, so the same `search`, `read`, `write`, and `create` methods the web client uses are callable from any language. The kernel is the API.

## How It Works

The best way to understand Odoo's kernel is to follow one request from the browser down to SQL and back, and the detailed diagram below lays out every file on that path.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/odoo/odoo-odoo-architecture.svg" alt="Detailed architecture of the odoo/odoo repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of odoo/odoo: from `odoo-bin` through the CLI, server processes, HTTP dispatchers, module loading, the ORM core with its registry, environments, fields and query compiler, down to psycopg2, and over to the OWL web client.*

### Understanding the Architecture

**The entry point and server shell.** Everything starts with `odoo-bin`, a five-line script that calls `odoo.cli.main()`. The `odoo/cli/` package defines subcommands such as `server`, `shell`, `scaffold`, and `db`; `odoo/cli/server.py` is the default. Booting creates one of the server classes in `odoo/service/server.py` — `ThreadedServer`, `GeventServer`, or `PreforkServer` — and that file also defines `WorkerCron` for scheduled jobs and `WorkerHTTP` for the pre-forking mode. The server does not know about business logic at all; it just hosts a WSGI application built by the HTTP package.

**The HTTP and RPC layer.** The `odoo/http/` package is a compact web framework in its own right: `odoo/http/router.py` implements the `@route` decorator controllers use to declare endpoints, `odoo/http/routing_map.py` matches URLs, `odoo/http/requestlib.py` wraps the incoming request, and `odoo/http/session.py` handles sessions and CSRF validation. `odoo/http/dispatcher.py` defines an abstract `Dispatcher` with three concrete flavors — `HttpDispatcher`, `JsonRPCDispatcher`, and `Json2Dispatcher` — so plain HTTP pages, JSON-RPC, and the newer JSON/2 protocol share one error, session, and security pipeline. Controllers live inside addons; the web client's backend endpoints are in `addons/web/controllers/`, including `webclient.py` and `dataset.py`, and generic model calls funnel into `odoo/service/model.py`, whose `call_kw` function resolves the target model and method before invoking it.

**The module system.** An Odoo module is a directory with a `__manifest__.py` describing its name, data files, and dependencies. Discovery lives in `odoo/modules/module.py` (`get_manifest`, `initialize_sys_path`), which scans `addons/` — 642 modules carry a manifest in this tree — plus the package-internal `odoo/addons/` where the `base` module itself lives. Dependencies are resolved into a directed graph by `odoo/modules/module_graph.py`, and `odoo/modules/loading.py` orchestrates the actual load: create or upgrade schemas, run migrations via `odoo/modules/migration.py`, and import data files (XML, CSV) through `odoo/tools/convert.py`. After loading, the registry signal notifies workers to reload their model classes.

**The ORM kernel.** The heart of the tree is `odoo/orm/`, and its `models.py` — 6,613 lines — defines `MetaModel` and `BaseModel`, the class every business model inherits. `BaseModel` provides the CRUD surface (`create`, `read`, `write`, `unlink`, `search`, `search_read`, `browse`) that both the RPC layer and the web client call. Records are always handled through recordsets bound to an `Environment`, defined alongside `Transaction` and the cache layer in `odoo/orm/environments.py`, which carries the database cursor, user id, and context. The per-database `Registry` in `odoo/orm/registry.py` assembles final model classes from all installed modules and manages the named caches you can see listed in its source.

**Fields, domains, and SQL generation.** Field definitions were split into a package-style family: `odoo/orm/fields.py` holds the 2,080-line `Field` base class, while `odoo/orm/fields_relational.py` (1,799 lines) implements `Many2one`, `One2many`, and `Many2many`, with sibling files for temporal, numeric, textual, selection, and binary types. When you search, an expression like `[('state', '=', 'done')]` is parsed by `odoo/orm/domains.py` and compiled into parameterized SQL by `odoo/orm/query.py`, which executes against the psycopg2 connection pool in `odoo/sql_db.py`. The ORM never strings-concatenates user input into SQL; domains are structured data end to end.

**The OWL web client.** On the browser side, `addons/web/static/src/webclient/webclient.js` mounts the root component using the OWL framework from `addons/web/static/lib/owl/owl.js`. Views — list, form, kanban, graph, pivot, calendar — live in `addons/web/static/src/views/` and read their data through the client-side model in `addons/web/static/src/model/relational_model`, which issues `call_kw` requests to the dataset controller. Assets such as JavaScript bundles are declared per-module; `addons/web/__manifest__.py` defines the `web.assets_backend` bundle that pulls everything together.

Follow one click end to end: a user edits a kanban record, the OWL view updates its `relational_model` state and POSTs a JSON-RPC call; `JsonRPCDispatcher` authenticates the session, wraps the request, and routes it through `odoo/service/model.py`; `call_kw` asks the database's `Registry` for the model and invokes `write`; the ORM applies field constraints, updates the cache and the row via psycopg2, and returns a recordset serialized to JSON; the web client re-renders the changed record. One path, five subsystems, no magic.

## Advantages

- **A real plugin architecture, not a bolt-on.** Every feature, including the web client, ships as a module loaded by the same registry and dependency graph — the kernel eats its own dog food.
- **Per-database model registries.** Because `odoo/orm/registry.py` assembles classes per database, one server safely serves tenants with different installed modules.
- **Structured domains instead of raw SQL.** `odoo/orm/domains.py` and `odoo/orm/query.py` turn declarative expressions into parameterized SQL, keeping ORM code readable and injection-safe.
- **A disciplined HTTP layer.** Three dispatchers sharing one pipeline in `odoo/http/dispatcher.py` means CSRF, sessions, and error handling behave identically for pages and RPC.
- **Deterministic module loading.** The dependency graph in `odoo/modules/module_graph.py` plus forced `base` loading in `odoo/modules/loading.py` makes install and upgrade order explicit and reproducible.
- **A purpose-built frontend framework.** OWL gives the web client a reactive component model sized for the problem, without pulling a large external UI stack into the kernel.

## Benefits

- **Study value for backend engineers.** Few open-source systems let you read a complete ERP kernel — registry, loader, ORM, RPC — in a single tree with this level of internal documentation in docstrings.
- **A proven extension surface.** Custom apps use exactly the primitives described here: a `__manifest__.py`, model classes inheriting `BaseModel`, and data files; `odoo-bin scaffold` generates the skeleton.
- **Multi-database deployment flexibility.** The worker and server classes in `odoo/service/server.py` let you scale from a laptop (`python odoo-bin -d mydb`) to a pre-forked production host with separate cron workers.
- **Integration-ready from day one.** The JSON-RPC endpoints used by the web client are the same ones available to external tools, so scripting and integrations need no new server code.
- **Debuggability through layering.** Each concern has one obvious file: routing in `router.py`, dispatch in `dispatcher.py`, loading in `loading.py`, SQL in `query.py` — which narrows any bug hunt quickly.
- **Free software with commercial-grade scope.** LGPLv3 licensing plus 642 modules spanning accounting to manufacturing make it equally viable as a learning resource and a deployment target.

## Usage

Install from source and run a local server, per the repository's own scripts and configuration:

```bash
git clone https://github.com/odoo/odoo.git
cd odoo
pip install -r requirements.txt
```

```bash
python odoo-bin --addons-path=addons -d mydb
```

The `-d/--database`, `--addons-path`, and `--db-filter` options are declared in `odoo/tools/config.py`. Then open `http://localhost:8069` and install apps from the UI.

To start a new custom module, use the built-in scaffolding command from `odoo/cli/scaffold.py`:

```bash
python odoo-bin scaffold my_library ./addons-extra
```

And for an interactive REPL against a running database, `odoo/cli/shell.py` provides:

```bash
python odoo-bin shell -d mydb
```

## Conclusion

Odoo's monorepo is one of the most instructive large Python codebases you can read today. The overview you saw at the top — launcher, servers, HTTP dispatch, module loader, ORM registry, PostgreSQL, OWL client — is not an idealized architecture diagram; every node in it is a file or directory you can open in the repository. If you want to understand how a modern ERP is really built, or you are designing your own extensible platform, clone the repo, start at `odoo-bin`, and follow a request through `odoo/http/` into `odoo/orm/`. The code is the documentation.

Links:

- GitHub repository: [https://github.com/odoo/odoo](https://github.com/odoo/odoo)
- Official documentation: [https://www.odoo.com/documentation/master](https://www.odoo.com/documentation/master)
