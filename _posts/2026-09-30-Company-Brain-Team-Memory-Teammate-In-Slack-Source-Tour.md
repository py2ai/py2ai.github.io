---
layout: post
title: "Company Brain: A Teammate in Slack That Remembers Everything - Inside supermemoryai/company-brain"
description: "A source tour of supermemoryai/company-brain, the open-source Slack teammate from the supermemory team that ingests channel conversations, extracts durable memories, and answers questions with retrieval over your team's own knowledge. We walk the Cloudflare Workers architecture, the memory containers, and the retrieval-augmented answering loop."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Company-Brain-Team-Memory-Teammate-In-Slack-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/company-brain/supermemoryai-company-brain-architecture.svg
tags:
  - Slack
  - AI Agents
  - Memory
  - Open Source
categories: [AI, Open Source]
keywords: "company brain, supermemory, slack ai teammate, team memory, cloudflare workers, durable objects, retrieval augmented generation, memory extraction, slack bot, open source ai, mcp tools, knowledge base, d1 database, hono, typescript"
author: "PyShine"
---

Most team knowledge never makes it into a document. It lives in Slack threads, gets decided in a hallway conversation that someone summarizes in a channel, and then evaporates the moment everyone scrolls past it. Search cannot save you, because you have to already know the words to search for. The supermemory team ran into this so hard that they built a paid product around it, and when they discontinued that product, they did something unusually generous: they open-sourced the whole thing.

That product is Company Brain, hosted in the supermemoryai/company-brain repository. It is a Slack teammate that sits in your channels, quietly remembers the durable things your team says, and answers questions from that memory. Ask it whether production is down and it can answer from context nobody explicitly gave it, because it has been distilling the channel conversation in the background. It can also act rather than advise, since it connects to GitHub, Linear, Notion, Google Workspace and other tools over MCP, and it can run code in its own sandbox when a question needs computation.

What makes the repository worth a source tour is that this is not a wrapper around a chat model. The code shows a complete memory pipeline with explicit write scopes, permission-aware read scopes, an ambient distillation scheduler, a triage layer that decides when the bot should speak at all, and an approval gate for consequential tool writes. It is TypeScript deployed to Cloudflare Workers, Durable Objects and D1, and almost every interesting design decision is visible in a file you can open and read.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/company-brain/supermemoryai-company-brain-overview-architecture.svg" alt="Architecture overview of the supermemoryai/company-brain repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Company Brain architecture: Slack events land on a Hono worker, get dispatched to a per-organization CompanyBrainAgent Durable Object, which triages traffic, computes answers, and feeds both turn-level and ambient memory writes into the memory layer.*

Reading the overview from left to right: Slack posts signed events to the gateway worker, whose route handlers verify and deduplicate them before handing work to the agent; the agent classifies each event as an explicit turn, a chime candidate, or ambient context; explicit and chime traffic flows through turn computation, which retrieves scoped memories and assembles tools; meanwhile the ambient distiller watches channels that the bot does not reply in and pushes distilled memory documents into the same memory layer; the tool runtime gives the brain MCP connectors and a code sandbox so answers can include live data, not just recall.

## Why You Need This

The first problem Company Brain solves is capture. Humans are unreliable scribes, and the moments that matter, like who owns a decision or why an approach was rejected, are usually typed in passing. The repository attacks this with an observation pipeline in src/brain/slack/channel-observe.ts: after a channel goes quiet, a distiller job wakes up, pulls the recent message batch, and curates only the durable material such as ownership changes, decisions, commitments and status changes. Raw chatter is explicitly discarded by the distillation prompt, so the memory does not turn into a transcript of jokes.

The second problem is recall with permission boundaries. A shared team brain is dangerous if a private channel's contents leak into a public answer. Company Brain models memory as scoped containers rather than one bucket, and the read path in src/brain/memory/read-scope.ts guarantees that an answer can only draw on containers the asker could read themselves. In a DM, the brain reads your personal container, the shared brain, and every private-channel container you are a member of. It cannot leak something you could not see, which is the correct invariant for an organizational memory.

The third problem is knowing when to speak. A bot that replies to everything gets muted in week one. Company Brain routes every non-mentioned message through a triage model in src/brain/slack/triage.ts that returns one of four decisions: answer, ack, investigate, or pass. Parse failures and model errors fall back to pass, meaning the failure mode is silence rather than noise, and chime budgets in src/brain/slack/chime-budget.ts cap how often it may intrude per channel.

The fourth problem is that answers alone are often not enough. When the right response is "opened the issue" rather than "you should open an issue," the brain needs hands. Tool assembly in src/brain/turn/tools.ts wires in MCP connectors from a large built-in directory, embedded Google tools, a web search layer, and a sandbox that clones repositories and runs scripts. Consequential writes do not happen silently: they suspend and render a Block Kit Approve/Deny card, and only the person who made the request can approve it.

## How It Works

The whole system is a pipeline from a signed Slack webhook to a permission-scoped answer, with memory being written continuously along the way.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/company-brain/supermemoryai-company-brain-architecture.svg" alt="Detailed architecture of the supermemoryai/company-brain repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the Company Brain source: the gateway worker, the Slack interface modules, the agent turn core inside the Durable Object, the memory layer with its containers and tags, the tool runtime, and the D1, KV and DO storage platform underneath.*

### Understanding the Architecture

**The gateway is a thin, defensive Hono worker.** The entry point in src/worker.ts mounts the Slack routes, the auth session middleware, the setup wizard, and the API under /brain, then falls through to static assets for the React dashboard built from web/. Every Slack event posted to the routes in src/routes/slack/ first passes HMAC signature verification in src/brain/slack/verify.ts, then deduplication: each event id is recorded in the BRAIN_KV namespace under a key like slack:evt:<event_id> with a one-day expiry, so Slack retries cannot double-process a message. The handler returns an immediate acknowledgment and does the real work with waitUntil.

**One Durable Object owns one organization.** The worker resolves the Slack team id to an org and dispatches to a CompanyBrainAgent instance looked up by name in src/brain/turn/agent.ts, built on the Agents SDK Durable Object base class. This per-org object owns all mutable state: Slack turns, chime scheduling, ambient observation jobs, approvals, leases, scheduled automations and team invites. The wrangler.jsonc configuration declares it with SQLite-backed storage, alongside a Sandbox DO class, the D1 database and the KV namespace, which means a single deployment is fully provisioned by Cloudflare.

**Triage decides the shape of every turn.** Explicit traffic such as mentions, DMs and thread replies skips triage entirely and becomes an answer turn. Everything else flows through the triage model in src/brain/slack/triage.ts, which can answer with a priority, acknowledge with an emoji, mark the thread for investigation, or pass. The turn entry in src/brain/slack/turn.ts then drives computeTurn in src/brain/turn/compute.ts, which builds the message layout with src/brain/turn/context.ts, selects the reply depth, and streams output through src/brain/slack/stream.ts in DMs or via progress cards and a final message in channels, using the Slack Web API client in src/brain/slack/client.ts.

**Memory writes are structured documents, not log lines.** Both the post-turn reflection pass in src/brain/turn/post-turn-reflect.ts and the ambient distiller produce documents matching MemoryDocSchema in src/brain/memory/writeback.ts: a title, self-contained content, optional sources, an optional event date, and one to a few canonical tags from src/brain/memory/tags.ts. Each write gets a deterministic custom id derived from a SHA-256 hash of its key material, which gives the pipeline deduplication for free. The write request is stamped with exactly one container tag: the shared team brain, a personal user container for DMs, or a slack_channel_ container for private channels.

**Retrieval is hybrid, scoped, and fan-out limited.** When a turn needs context, searchBrain in src/brain/memory/search-brain.ts resolves the readable container set through src/brain/memory/read-scope.ts, then issues hybrid searches of up to forty results per container with a 0.3 relevance threshold from src/brain/search-brain-format.ts. Container searches run with a bounded concurrency of six so a DM from someone in many private channels cannot flood the vector service, results are deduplicated by id, and the top forty survive into the prompt. The tagged metadata on each document lets retrieval focus on a person, project or topic when the question calls for it.

**Tools and approvals close the loop.** Tool assembly in src/brain/turn/tools.ts combines memory search helpers, MCP catalog and execution under src/brain/tools/mcp/ with execute.ts at its core, the sandbox clients under src/brain/tools/sandbox/, and web search and extraction under src/brain/tools/web/. Write-class operations route through the approval machinery in src/brain/turn/approval.ts, which pauses the turn until the requesting human taps Approve or Deny on a Block Kit card. Scheduled work such as digests and automations is handled by the scheduler modules in src/brain/tools/, and persisted in D1 through the drizzle schema in src/db/schema/.

The end-to-end flow looks like this: a teammate asks a question in a channel; Slack signs and posts the event; the worker verifies, deduplicates and acknowledges it; the org's Durable Object classifies it and runs the turn; the turn searches the containers the asker may read, optionally calls MCP tools or the sandbox, streams an answer back into the thread; and after the answer lands, a reflection pass extracts any durable facts into the correctly scoped memory container, so the next question starts from a smarter brain.

## Advantages

- **Ambient capture with no workflow change.** The channel-observe distiller in src/brain/slack/channel-observe.ts turns ordinary conversation into durable memory; nobody has to remember to document anything.
- **Permission-faithful retrieval.** The container model in src/brain/memory/read-scope.ts and src/memory/container-tags.ts means an answer can never exceed the asker's own access, which is the property most "team knowledge" tools get wrong.
- **Deliberate proactivity.** Triage in src/brain/slack/triage.ts plus chime budgets in src/brain/slack/chime-budget.ts make the brain's silence a designed outcome, not a missing feature.
- **Real tool agency with human brakes.** MCP execution under src/brain/tools/mcp/ reaches hundreds of external tools, while the approval gate in src/brain/turn/approval.ts keeps consequential writes under explicit human control.
- **Serverless from edge to brain.** One wrangler.jsonc provisions the worker, D1, KV, Durable Objects and the optional container sandbox; there is no cluster to babysit.
- **Model freedom.** The brain accepts an Anthropic, OpenAI, Google, xAI or OpenRouter key and resolves the provider itself, so you are not locked into one vendor.

## Benefits

- **Institutional memory that survives attrition.** Decisions, owners and rationale persist in scoped containers, so "why did we do it this way" stays answerable after the people move on.
- **Faster incident response.** Because the brain watches channels continuously, status questions can be answered from the freshest distilled context without anyone re-explaining.
- **Onboarding that actually works.** New teammates can ask the brain in a DM and get answers assembled from the shared brain plus every private channel they already belong to.
- **A trustworthy audit trail.** Structured memory documents with sources, event dates and deterministic ids in src/brain/memory/writeback.ts make every stored fact traceable.
- **Self-hosted privacy.** The entire system deploys to your own Cloudflare account under the Apache-2.0 license; your conversations never have to live in someone else's SaaS.
- **An extensible agent platform.** Skills, a workspace prompt, automations and proactivity settings give teams a way to shape behavior without forking the code.

## Usage

Deployment is designed to be a one-click affair: the repository's README offers a Deploy to Cloudflare button that provisions D1, KV, Durable Objects and Workers AI, and asks for exactly two secrets, SUPERMEMORY_API_KEY for the memory backend and MODEL_API_KEY for the model provider. After deployment you open /setup on your worker, which checks the keys, generates a Slack app manifest with your URLs filled in, and walks you through installing it.

For local development, the README's commands are:

```sh
bun install
cp .dev.vars.example .dev.vars   # fill in the two keys
bun run dev
```

The project scripts in package.json cover the rest of the lifecycle:

```sh
bun run deploy            # wrangler deploy
bun run db:generate       # generate drizzle migrations and bundle them
wrangler d1 migrations apply DB --local
```

If you change the schema under src/db/schema, you regenerate migrations with db:generate; the worker applies pending migrations on its next request. Slack must be able to reach a dev server, so the README recommends pointing a tunnel at it and setting PUBLIC_URL in .dev.vars before creating the Slack app from the tunnel's /setup page. On the free Workers plan the brain runs fine, with an optional Daytona key enabling the code sandbox; on Workers Paid you can instead uncomment the containers block in wrangler.jsonc to get a built-in sandbox container.

## Conclusion

Company Brain is one of the more complete open-source agent codebases you can read: a defensive event gateway, a per-organization Durable Object brain, triage and chime budgeting that treat restraint as a feature, a memory pipeline with structured documents and permission-faithful containers, and a tool runtime that can both fetch live data and act, with approvals where it matters. If you are building any system that needs to remember a team rather than a user, the modules under src/brain/memory/ and src/brain/slack/ are worth an afternoon of reading on their own. And if you just want the teammate, the five-minute deploy is real.

Links:

- GitHub repository: https://github.com/supermemoryai/company-brain
- User guide: https://github.com/supermemoryai/company-brain/blob/main/docs/guide/README.md
- Architecture notes: https://github.com/supermemoryai/company-brain/blob/main/docs/architecture.md
- supermemory: https://supermemory.ai
