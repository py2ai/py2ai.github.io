---
layout: post
title: "MOVO: Turn A Local Agent Runtime Into An Enterprise Platform - Inside himovo/movo"
description: "MOVO wraps the DeepSeek Harness (DSH) agent runtime in a complete self-hosted product: user workspace, admin console, document intelligence, enterprise knowledge, approvals, audit and a twelve-service Docker Compose deployment. A source tour of the FastAPI services, the Node.js runtime host and the governance layer."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /movo-turn-the-dsh-agent-runtime-into-an-enterprise-platform/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/movo/himovo-movo-architecture.svg
tags: [AI Agents, Self-Hosted, Enterprise, Open Source]
categories: [AI, Open Source]
keywords: MOVO, DeepSeek Harness, DSH, agent platform, self-hosted, Docker Compose, FastAPI, Vue 3, enterprise AI
author: "PyShine"
---

Running an agent demo on a laptop and operating an agent platform for a team are two different sports. The demo needs a clever prompt; the platform needs identity and access control, knowledge management, model administration, approval workflows, audit trails, file delivery and a deployment story that survives contact with an IT department. MOVO Community Edition, from himovo, is an answer to the second sport. It takes the DeepSeek Harness (DSH) agent runtime, with its planning, tool calls, Skills and sub-agents, and wraps it in everything an enterprise actually needs around it: a user workspace, an administration console, document intelligence, enterprise knowledge with citations, governance, and a one-command Docker Compose deployment.

The project is refreshingly direct about division of labor. In their own words: DSH runs the Agent; MOVO brings the Agent into enterprise production. The self-hosted edition ships the complete web workspace and admin console, conversation and agent APIs, the DSH Runtime Host, a document parser with a retrieval worker, and the deployment configuration as a single repository. You connect your own model providers, your data and runtime services stay inside your deployment, and the community tenant created by the setup flow has no member-count limit and no billing enforcement. Some capabilities, namely the Browser Agent and Code Agent that drive a local browser session and a project terminal, live in the separately distributed MOVO Desktop application, which connects to your self-hosted service; the source in this repository covers the web platform.

What makes MOVO worth a source tour is that it is one of the few projects where the boring parts, approvals, receipts, audit, quotas, migrations, secret bootstrap, are first-class citizens of the architecture rather than an afterthought. Let us walk through how the pieces fit.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/movo/himovo-movo-overview-architecture.svg" alt="Architecture overview of the MOVO repository" style="min-width:640px;width:100%;">
</div>

*Architecture overview of the MOVO repository: the Vue 3 frontends, the Nginx gateway, the FastAPI services, the DSH runtime and the deployment tooling.*

Reading the overview from left to right: both frontends, the user workspace at `apps/user-web` and the admin console at `apps/admin-web`, are Vue 3 applications served through the Nginx gateway defined in `deploy/docker/nginx.conf`. The gateway routes API traffic to the conversation service, whose FastAPI entry point is `services/chat-api/app/main.py`, and to the management service at `services/admin-api/app/main.py`. The chat API drives agent turns through the Node.js DSH Runtime Host at `services/chat-api/dsh/runtime-host/src/host.mjs`, consults the governance layer in `services/chat-api/app/governance/approval_runtime.py` for sensitive actions, and uses the retrieval and citation machinery in `services/chat-api/app/knowledge`. Both APIs share the schema maintained under `services/chat-api/app/migrations`. The whole stack of twelve services is orchestrated by the `movo` launcher script acting on `docker-compose.yml`.

## Why You Need This

- **A complete product, not a skeleton.** Authentication, organizations, roles, quotas, audit and file delivery ship in the box. You do not assemble them from three other projects.
- **Stay native to DSH.** The official runtime, Skills, Tools, MCP integrations and sub-agents are used as they are, so you are not locked into a disconnected proprietary re-implementation.
- **Your models, your data.** Connect any compatible model API during setup; application data, knowledge and runtime services remain in your own deployment, and provider credentials are encrypted before storage.
- **Governance that works at the tool level.** Sensitive tool actions can require human approval, executions are traced, generated artifacts are retained, and every action can be traced to a receipt.
- **Document intelligence included.** PDF, DOCX, XLSX, PPTX, CSV and Markdown parsing, including images and charts, runs in a dedicated service, which is what makes enterprise knowledge bases actually usable.
- **One command to production-shaped deployment.** The launcher pulls prebuilt images, retries network failures, waits for health checks and prints the setup address; no environment file or local build is required.

## How It Works

MOVO is a composition of focused services. Two Vue 3 frontends talk to three FastAPI application services and one Node.js runtime through an Nginx gateway, with MongoDB, Redis and Weaviate behind them. The default Compose file starts twelve services, including a one-time secret bootstrap container. The next diagram follows the code from the gateway down into the agent runtime and the governance layer.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/movo/himovo-movo-architecture.svg" alt="Detailed architecture of the MOVO codebase" style="min-width:720px;width:100%;">
</div>

*Detailed architecture of the MOVO codebase, from the gateway and endpoints to the DSH Runtime Host, governance and document parsing.*

### Understanding the Architecture

**The gateway and the frontends.** The Nginx configuration at `deploy/docker/nginx.conf`, packaged by `deploy/docker/gateway.Dockerfile`, terminates HTTP for both `apps/user-web` and `apps/admin-web` and routes API paths to the backend services. Keeping the routing in one gateway configuration is what allows the whole platform to sit behind `http://localhost:3000` in development or a reverse proxy with your own TLS in production; DNS and certificates stay the operator's responsibility by design.

**The chat API.** `services/chat-api/app/main.py` is a FastAPI application that wires CORS, structured request logging with an `X-Request-ID` for every call, and request context propagation. Its REST surface lives in `services/chat-api/app/api/endpoints`. The interesting machinery is in `services/chat-api/app/dsh_runtime/`: `turn_admission.py` controls when a new agent turn may start, `turn_runner.py` drives a single turn end to end, `gateway.py` speaks to the runtime host process, and `host_manager.py` manages that process's lifecycle, with dedicated modules for turn cancellation, recovery and finalization so that an interrupted turn has a defined path back to a consistent state. Model calls flow through the provider abstraction in `services/chat-api/app/llm`, context assembly through `services/chat-api/app/context_engine`, and token metering through `services/chat-api/app/token_usage`.

**The DSH Runtime Host.** The Node.js side, under `services/chat-api/dsh/runtime-host/src/`, is the process that actually executes agent behavior. `host.mjs` is the entry point; `kernel-runtime.mjs` runs the agent kernel, `runtime-manager.mjs` manages session lifecycles, and `host-protocol.mjs` frames the event stream exchanged with the Python side. Every execution is appended to `event-journal.mjs`'s journal, which is what makes traces inspectable after the fact, and `skill-bundle-materializer.mjs` writes Skill bundles to disk where the runtime can load them, including Skills installed from ZIP packages. A registry of native plugins, including the desktop approval broker, extends the host without changing its core.

**Governance.** The modules under `services/chat-api/app/governance/` are the compliance spine. `approval_runtime.py` suspends turns that want to execute a sensitive tool action until a human approves, `action_receipt_store.py` records receipts for executed actions, and `audit.py` maintains the trail. The admin console's audit views are served from `services/admin-api/app/system_audit`, which queries the same records. This is the layer that turns "the agent did something" into "the agent did this approved thing, here is the evidence".

**Knowledge and documents.** Enterprise knowledge lives in `services/chat-api/app/knowledge`, with retrieval, citation projection, a research focus builder for multi-round research, and parsers for skill catalogs and SKILL.md files. Documents enter through the separate `services/document-parser` service: `app/main.py` exposes parse and preview APIs, `app/workers` processes them in the background, and `app/integrations` adapts the heavy format engines (Playwright, LibreOffice and Docling are pulled in when building from source). Both the chat API and the admin API delegate parsing here rather than each growing their own file handling.

**Deployment and operations.** The `movo` launcher orchestrates `docker-compose.yml`: `./movo up` pulls the published images from GHCR sequentially with continuous retry, waits for services to become healthy and prints the setup address. Day-two operations are covered by `deploy/cli/`: `backup.sh` for volume backups, `migrations.sh`, `images.sh`, `pull.sh` and `sequence-fix.sh` for the upgrade edge cases that `./movo fix` addresses. Configuration is a four-line environment file based on `.env.example`: port, volume prefix, image prefix and version.

Follow one conversation end to end: a user sends a message in `apps/user-web`; the gateway routes it to `services/chat-api/app/main.py`, whose endpoints admit a turn through `turn_admission.py`; `turn_runner.py` assembles context from the knowledge service and the document parser's extracted content, then drives the turn through the DSH Runtime Host, where `kernel-runtime.mjs` plans, calls tools and invokes Skills while `event-journal.mjs` records everything. When the model wants a sensitive action, `approval_runtime.py` suspends the turn until a human approves; on completion the answer streams back through the gateway with citations and a receipt in the audit trail.

## Advantages

- **Production concerns are built in.** Approvals, receipts, audit trails and token metering are part of the turn path, not bolted on afterwards.
- **Process isolation for the agent runtime.** The Node.js runtime host runs as its own managed process with a framed protocol, so a misbehaving turn cannot take the API down with it.
- **Recovery is explicit.** Turn cancellation, recovery and finalization each have dedicated modules, which is the difference between a demo and a platform you can restart safely.
- **Background document processing.** Parsing happens in a worker-based service with format integrations isolated, so a pathological PPTX cannot block a chat response.
- **Deterministic deployment.** Twelve well-defined services, one Compose file, a launcher with retry and health waits, and pinned image versions for production.
- **Honest licensing.** The MOVO Community License states exactly what is restricted (hosted multi-tenant SaaS, rebranding) instead of hiding behind vague terms.

## Benefits

- **Faster path to real users.** Employees get a workspace with chat, research, knowledge, files and content generation on day one, not after a quarter of integration work.
- **Admins get real levers.** Organizations, users, roles, models, knowledge, Skills, Tools, quotas and runtime health are all managed from the admin console and its API.
- **Traceability by default.** Event journals, action receipts and audit records mean every agent outcome can be explained after the fact, which is what compliance conversations actually require.
- **Vendor-neutral model access.** Any compatible chat model API works, plus optional embedding, reranking, vision, image generation and web search providers configured in the setup wizard.
- **Community-friendly economics.** No member-count limit and no billing enforcement in the Community Edition; you scale by adding hardware, not seats.
- **A readable reference architecture.** The separation of gateway, APIs, runtime host, governance and document services is a template you can borrow for any agent platform work.

## Usage

Requirements are Git, Docker Desktop (or Docker Engine with Compose v2), at least 8 GB of memory, 20 GB of disk, network access to GHCR and Docker Hub, and credentials for one compatible model API. Then:

```bash
git clone https://github.com/himovo/movo.git
cd movo
chmod +x movo
./movo up
```

Open `http://localhost:3000/admin/setup`. The wizard checks the deployment, creates your organization and initial accounts, connects a default chat model, and optionally configures embedding, reranking, vision, image generation and web search providers. After setup, the user workspace lives at `http://localhost:3000/` and the admin console at `http://localhost:3000/admin/`.

Day-two operations from the repository root:

```bash
./movo status
./movo logs chat-api
./movo restart
./movo update
./movo backup /path/on/a/large-disk/movo-backup
./movo down        # stop, keep data
./movo down -v     # stop and delete data after confirmation
```

To customize the port, canonical URL, image version or volume prefix, copy `.env.example` to `.env` and set `MOVO_PORT`, `PUBLIC_BASE_URL`, `MOVO_VERSION` and `MOVO_VOLUME_PREFIX`; keep the volume prefix stable after first startup. On Windows, run the stack inside an Ubuntu WSL 2 distribution, or start the same prebuilt images directly with `docker compose up -d`. Contributors can build locally with `./movo up --build`, which additionally pulls Playwright, LibreOffice, Docling and model assets, and the repository hygiene gate is `python3 scripts/check_open_source_hygiene.py`.

## Conclusion

MOVO is a pragmatic answer to the question every team eventually asks after their first agent demo: now what? By keeping DSH as the untouched runtime and concentrating its own code on the enterprise surface, gateway, APIs, knowledge, document intelligence, approvals, audit and deployment, the project delivers something that is genuinely operable rather than merely impressive. The source layout makes the boundaries obvious, the governance modules make the compliance story concrete, and the launcher makes the first hour almost boring, which is exactly what you want from infrastructure. If you are evaluating how to bring agents to a team without renting someone else's platform, MOVO Community Edition is one of the most complete self-hosted starting points available today.

Links:

- Repository: [github.com/himovo/movo](https://github.com/himovo/movo)
- Documentation: [himovo.com/en/guide/introduction.html](https://www.himovo.com/en/guide/introduction.html)
- Docker deployment guide: [docs/docker-deployment.md](https://github.com/himovo/movo/blob/main/docs/docker-deployment.md)
- Windows installation: [docs/windows-installation.md](https://github.com/himovo/movo/blob/main/docs/windows-installation.md)
- Releases: [github.com/himovo/movo/releases](https://github.com/himovo/movo/releases)
