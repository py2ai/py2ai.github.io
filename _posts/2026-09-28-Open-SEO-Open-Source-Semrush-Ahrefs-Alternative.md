---
layout: post
title: "OpenSEO: An Open-Source Alternative to Semrush and Ahrefs - Inside every-app/open-seo"
description: "OpenSEO is an MIT-licensed, open-source alternative to Semrush and Ahrefs built on Cloudflare Workers. We tour the source: a TanStack Start UI, a DataForSEO-powered keyword, backlink and site-audit engine, an MCP server with fifty-plus tools, and SAM, an in-app AI agent — all self-hostable with Docker or Cloudflare."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /Open-SEO-Open-Source-Semrush-Ahrefs-Alternative/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/open-seo/every-app-open-seo-architecture.svg
tags:
  - SEO
  - Cloudflare Workers
  - MCP
  - Open Source
categories: [AI, Open Source]
keywords: "OpenSEO, open source SEO tool, Semrush alternative, Ahrefs alternative, Cloudflare Workers, MCP server, AI agent, keyword research, rank tracking, site audit, backlinks, DataForSEO, self-hosting, TanStack Start, Drizzle ORM"
author: "PyShine"
---

Ask any founder or indie developer what their SEO stack costs and you will usually hear a number with three digits and the word "per month" attached to it. The dominant suites have earned their reputations, but they have also turned search visibility into a subscription tax on small teams. OpenSEO, the project behind the GitHub repository `every-app/open-seo`, starts from a different premise: the core workflows of SEO — keyword research, rank tracking, competitor insight, backlinks, site audits, and AI visibility — should be software you can run, read, and modify yourself.

OpenSEO describes itself in its README as "an open source alternative to Semrush and Ahrefs," and that framing is accurate. It is a complete, MIT-licensed web application written in TypeScript, built for Cloudflare Workers, with a modern React interface on top and a DataForSEO API connection underneath for the raw search data. You bring your own DataForSEO key and pay only for what you use, or you subscribe to the hosted version at openseo.so. The repository also ships an MCP server and a set of pre-built agent skills, so tools like Claude Code can operate on your SEO data directly rather than through screen scraping or fragile export files.

What makes the source worth a tour, though, is not just the product category. It is rare to find a single repository that demonstrates, in clean and current TypeScript, how to build an AI-era SaaS on serverless infrastructure: a Worker entry point that routes everything, durable Cloudflare Workflows for long-running jobs, Drizzle ORM over D1 or Postgres, a full MCP server with OAuth, and an in-app AI agent — all in one readable codebase. Whether you want the tool or the architecture lesson, this repository rewards the read.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/open-seo/every-app-open-seo-overview-architecture.svg" alt="Architecture overview of the every-app/open-seo repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the OpenSEO architecture: a TanStack Start frontend and Cloudflare Worker app, feature services backed by DataForSEO and Drizzle, an MCP and SAM agent layer, and a dedicated audit engine built on durable workflows.*

Reading the overview from left to right: a browser request lands in the Worker entry point at `src/server.ts`, which either serves the server-rendered TanStack Start UI, dispatches a server function RPC, hands `/agents/*` traffic to the SAM chat Durable Object, or serves the MCP endpoint behind its OAuth provider. Server functions and MCP tools both funnel into the feature services under `src/server/features`, which read and write through the Drizzle layer (`src/db/index.ts` over D1 by default, Postgres optionally) and pull SEO data through the metered DataForSEO client (`src/server/lib/dataforseo/client.ts`). The audit engine sits deliberately apart in `src/audit-worker.ts` — a second Worker hosting the `SiteAuditWorkflow` — so memory-hungry crawls and Lighthouse payloads never threaten the main app's isolate. Finally, the published agent skills in `plugins/openseo/skills` are just callers of the same MCP server, which is the key to the project's "for you and your AI agent" promise.

## Why You Need This

The first problem OpenSEO solves is economic. Most commercial SEO suites price themselves for marketing departments, not for a developer who needs to check rankings on a handful of domains or audit a site before launch. OpenSEO flips the model: you supply a DataForSEO API key, the software charges you nothing beyond your own usage of that API, and the hosted service is transparent about its margin — the README states it plainly by noting the hosted plan charges 28% extra per DataForSEO request. When you self-host, there is no margin at all, just your own metered costs.

The second problem is control. SEO data — your keywords, your competitors, your crawl histories — is commercially sensitive, and the README's "Fork and vibe code your own custom tool" line is not marketing fluff. Because the entire stack is in this one repository, you can change ranking-check schedules, add a custom report, swap the data provider, or bolt the MCP server onto internal tooling. Docker self-hosting runs in `local_noauth` mode behind your own reverse proxy, so a personal install is genuinely a `docker compose up -d` away; Cloudflare self-hosting works on the free plan for internet-facing, multi-device setups.

The third problem is workflow. Modern SEO work increasingly happens inside coding agents, and OpenSEO embraces that instead of fighting it. It exposes a real MCP server with a full OAuth provider for hosted deployments and API-key auth elsewhere, so Claude Code, OpenClaw, Hermes, or any MCP client can call fifty-plus SEO tools — from `research-keywords` to `run-site-audit` — as first-class capabilities. Pre-built agent skills in `plugins/openseo/skills` (keyword research, keyword clustering, link prospecting, local SEO, SEO coaching, and more) package the methodology, not just the API surface, into workflows your agent can follow.

Finally, there is the focus problem. Instead of a hundred-tab dashboard, OpenSEO ships focused workflows — the README's main list is keyword research, rank tracking, competitor insights, backlinks, site audits, and AI visibility — plus SAM, an in-app AI agent that can drive all of those features conversationally. If your current tool makes you export CSVs to do anything interesting, this is the corrective experience.

## How It Works

The detailed diagram below maps the main data paths through the codebase, from UI routes down to the database clients and workflow engine.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/open-seo/every-app-open-seo-architecture.svg" alt="Detailed architecture of the every-app/open-seo repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of OpenSEO: server functions and MCP tools converge on feature services, the DataForSEO client meters every call through the billing module, Drizzle abstracts D1 and Postgres, and the audit and rank-check workflows run as durable, retryable Cloudflare Workflows.*

### Understanding the Architecture

**The entry point is one Worker that routes everything.** `src/server.ts` is the `main` declared in `wrangler.jsonc`, and its `fetch` handler is a small, readable dispatcher: TanStack Start's handler with a `frame-ancestors 'self'` CSP wrapper for app documents, a GDPR storage-erasure endpoint, `/agents/*` routed to chat Durable Objects with per-session authorization against the session's project, the Autumn billing webhook in hosted mode, the OAuth provider wrapper, and the self-hosted MCP route. The same file exports the `scheduled` handler that runs the two cron triggers from `wrangler.jsonc` — every five minutes for rank checks and stale-audit reconciliation, daily at 03:17 for OAuth KV garbage collection and referral sweeps.

**Auth is middleware, not an afterthought.** `src/middleware/ensureUser.ts` and its companions under `src/middleware/ensure-user/` implement three distinct auth modes declared in the repo: `cloudflare_access` (validating `cf-access-jwt-assertion` JWTs with a team domain and policy AUD), `hosted` (Better Auth-backed email and password), and `local_noauth` (trusted local mode that injects an admin user). Because the resolve step is centralized, the Worker, server functions, and the SAM agent authorization path in `src/server.ts` all share the same notion of a user and organization.

**Every feature is a slice, with thin RPC on top.** Under `src/server/features` you find one directory per domain — keywords, rank-tracking, backlinks, audit, dashboard, domain, gsc, ga4, projects, reports, sam and more — each typically split into `repositories` and `services`. The files in `src/serverFunctions/` (for example `src/serverFunctions/keywords.ts`, `src/serverFunctions/rank-tracking.ts`, `src/serverFunctions/audit.ts`) are TanStack Start server functions: a thin, typed RPC layer the React client calls, which delegate immediately to those services.

**All SEO data flows through one metered client.** `src/server/lib/dataforseo/client.ts` is the single gateway to DataForSEO, with endpoint modules beside it — `labs.ts` for keyword research, `backlinks.ts`, `serp.ts`, `lighthouse.ts`, `ai.ts` for LLM visibility metrics, plus `google-ads.ts` and `business.ts`. The `meter()` helper in the client wraps every fetcher with the billing layer from `src/server/billing/subscription.ts`: it asserts the organization has usage credits available and tracks the spend per call, attributing it to a credit feature mapped from the DataForSEO path. That is how the "pay only for what you use" promise is enforced in code rather than in a pricing page.

**Long jobs are durable Cloudflare Workflows.** Rank checks run through `RankCheckWorkflow` (exported from `src/server.ts`), started either by the `RankTrackingService` or by the cron-driven `scheduledRankChecks` service, which admits work in task units sized against DataForSEO's rate limits. Site audits run through `SiteAuditWorkflow` in a separate auxiliary worker defined by `src/audit-worker.ts` and `wrangler.audit.jsonc` — its multi-MB Lighthouse payloads and in-flight HTML would otherwise threaten the app worker's heap. Each step is durable and retried independently, crawl scratch state lives in the `AuditScratchpad` Durable Object, and `src/server/features/audit/services/auditReconciler.ts` acts as a cron watchdog that reaps audits stuck in "running."

**The AI layer is two-sided: MCP for external agents, SAM for humans.** `src/server/mcp/server.ts` registers the tool suite (backlinks, keyword research, SERP, rank tracking, Google Search Console, Google Analytics, local SEO, reports, site audits, projects) with Zod schemas and output contracts, while `src/server/mcp/oauth-provider.ts` fronts it for hosted deployments. SAM, the in-app agent, lives in `src/server/features/sam/SamChatAgent.ts` — a Durable Object built on the Agents SDK whose tools in `samChatTools.ts` reuse the very same MCP tool handlers, with model access through OpenRouter (`src/server/lib/openrouter.ts`). One tool implementation therefore serves the UI chat, external coding agents, and the published skills.

Follow one request end to end and the design clicks into place. You click "run audit" in the UI; the TanStack Start server function in `src/serverFunctions/audit.ts` passes through `ensureUser` middleware into `AuditService`, which calls `create()` on the cross-script `SITE_AUDIT_WORKFLOW` binding; the `SiteAuditWorkflow` instance starts in the audit worker, validating context against the audit row and running crawl and Lighthouse phases as retryable steps, storing results through the shared Drizzle layer; the UI polls status via the audit server functions; and if the workflow instance ever dies, the next five-minute cron tick sends the reconciler after it.

## Advantages

- **Honest, metered cost model.** The DataForSEO client meters every call through `src/server/billing/subscription.ts`, so costs map to actual usage instead of a flat suite subscription.
- **Two credible self-hosting paths.** Docker via `Dockerfile.selfhost` and `compose.yaml` for personal installs, or Cloudflare through `alchemy.run.ts` for internet-facing deployments — the same codebase serves both.
- **Durable background jobs without infrastructure.** Crawl-and-audit jobs and rank-check runs are Cloudflare Workflows with independently retried steps, plus a cron watchdog, so no separate queue or worker fleet is needed.
- **Deliberate workload isolation.** The audit engine lives in its own worker (`src/audit-worker.ts`) with its own scratchpad Durable Object, keeping memory spikes away from the request-serving isolate.
- **One tool set, three consumers.** MCP tools in `src/server/mcp/tools` power the public MCP server, the SAM chat agent, and the shipped agent skills, so behavior stays consistent across interfaces.
- **Portable persistence.** Drizzle schemas are kept in parity across D1 and Postgres (enforced by `src/db/schema-parity.test.ts`), giving installs a genuine scale-up path.

## Benefits

- **You own the deployment and the data.** Self-hosting keeps keywords, competitor data, and audit history inside infrastructure you control, with GDPR storage-erasure handling built in.
- **Your AI agent becomes the SEO operator.** With the MCP server and pre-built skills, agents can research keywords, run audits, and pull rankings directly — no browser automation required.
- **Focused workflows instead of dashboard sprawl.** The feature slices under `src/server/features` map to concrete SEO tasks, which keeps both the UI and the code navigable.
- **Transparent economics for the hosted plan.** The project publishes how the hosted service prices DataForSEO usage, and the code that implements the metering is right there in the repo.
- **A fork-friendly foundation.** Clean boundaries between RPC, services, data providers, and workflows make it realistic to fork and reshape the tool for a niche.
- **A reference architecture for AI-era SaaS.** Even if you never run an SEO tool, the patterns here — metered API gateways, durable workflows, MCP with OAuth, agent Durable Objects — are directly reusable.

## Usage

The repository documents two quick paths. For local development (from `docs/LOCAL_DEVELOPMENT.md`):

```sh
corepack enable
pnpm install --frozen-lockfile
cp .env.example .env.local
pnpm run db:migrate:local
pnpm run dev
```

Your DataForSEO key goes into `.env.local` as a base64-encoded `login:password` value:

```sh
printf '%s' 'YOUR_LOGIN:YOUR_PASSWORD' | base64
```

For the simplest self-host, Docker (from `docs/SELF_HOSTING_DOCKER.md`):

```bash
cp .env.example .env
docker compose up -d
```

That pulls the published image `ghcr.io/every-app/open-seo:latest` and serves the app on port 3001 by default; set `DATAFORSEO_API_KEY` in `.env` first, and optionally `OPENROUTER_API_KEY` for SAM's AI features.

## Conclusion

OpenSEO is a genuinely useful tool wrapped around an exceptionally instructive codebase. It replaces the subscription-tax model of commercial SEO suites with your own metered API key, gives both humans and AI agents a first-class interface to the same workflows, and demonstrates — in real, shipping TypeScript — how to build all of it on serverless infrastructure. If you have been paying for an SEO suite you barely use, or you simply want to see what a modern agent-native product looks like from the inside, cloning this repository is an afternoon well spent.

Links:

- [GitHub repository: every-app/open-seo](https://github.com/every-app/open-seo)
- [OpenSEO MCP setup docs](https://openseo.so/docs/mcp)
- [OpenSEO Agent Skills setup docs](https://openseo.so/docs/skills/setup)
- [Hosted version](https://openseo.so)
