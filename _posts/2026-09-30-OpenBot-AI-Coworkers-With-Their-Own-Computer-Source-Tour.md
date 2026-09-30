---
layout: post
title: "OpenBot: AI Coworkers With Their Own Computer - Inside CopilotKit/OpenBot"
description: "A source tour of CopilotKit/OpenBot, the self-hosted agent platform where every AI coworker gets its own browser, files and tools behind one governed gateway. We walk the CEL policy engine, the per-Bot computer supervisor, and the action stream that makes every step visible."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /OpenBot-AI-Coworkers-With-Their-Own-Computer-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/openbot/copilotkit-openbot-architecture.svg
tags:
  - AI Agents
  - Open Source
  - Agent Governance
  - TypeScript
categories: [AI, Open Source]
keywords: "OpenBot, CopilotKit, AG-UI, AI coworkers, agent governance, CEL policy, browser automation agent, per-agent sandboxing, self-hosted AI agents, MCP, audit trail, LangGraph, TypeScript"
author: "PyShine"
---

Most agent demos stop at the moment that matters: the model calls a tool, and the demo asks you to trust it. OpenBot, built by the CopilotKit team, is built around the opposite posture. Each AI coworker it hosts gets a computer of its own — a real browser with its own logins, its own files under a private workspace, and only the tools an administrator granted — and every action it takes is decided before it happens and recorded after. The README puts the difference in one line: an agent that can use your tools is not the same thing as an agent you can let near them.

Concretely, OpenBot is an open-source agent platform that runs entirely inside your own infrastructure. A Bot is any endpoint that speaks the AG-UI protocol, so coworkers built with LangGraph, Mastra, CrewAI, Pydantic AI, Google ADK — or written by hand — all arrive the same way. The stack is TypeScript on Bun: a React app on port 3010, a Hono API server on port 3001, Chromium-driven computer containers on 4100, a container supervisor on 4500, and PostgreSQL with pgvector underneath. The example tenant package in `examples/fintech` ships thirteen coworkers as configuration, from a General Assistant to a Risk Analyst, and the whole repository is MIT-licensed and explicitly a template to clone rather than a hosted product.

That framing is exactly why the source is worth a tour. There is no model in the box and no hosted service to hide behind, so the interesting engineering sits in plain sight: how a "computer per coworker" is actually provisioned, how a browser click is refused without ever trusting the model that requested it, and how a person watching a channel can see what the Bot is doing in real time. The code reads like a security engineering exercise that happens to be a product, and the comments explain why each boundary exists.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openbot/copilotkit-openbot-overview-architecture.svg" alt="Architecture overview of the CopilotKit/OpenBot repository" style="max-width:100%;height:auto;" />
</div>

*Overview of CopilotKit/OpenBot: the React UI reaches the Hono API server and its CopilotKit runtime, which exchanges AG-UI turns with Bot endpoints; every browser, file and shell action returns through the action gateway, where CEL policy and the audit trail decide and record it before anything reaches that Bot's own computer, provisioned by the supervisor.*

Reading the overview from left to right: a conversation starts in the React UI (`app/src/routes`) and lands on the Hono API server (`server/src/app.ts`), which mounts the CopilotKit runtime (`server/src/copilot.ts`). The runtime sends each turn over AG-UI to the selected Bot endpoint, and when the Bot answers with a tool call — a browser navigation, a file write, a shell command — that call comes back to the action gateway (`server/src/computer/gateway.ts`) rather than reaching any machine directly. The gateway asks the CEL policy engine, writes the audit row first, and only then forwards the action to the Bot's computer (`agent-computer/src/index.ts`), a container the supervisor (`supervisor/src/index.ts`) created for exactly that Bot. PostgreSQL holds product data, snapshots and audit rows; CopilotKit Intelligence holds the durable threads; and a small worker process (`worker/src/index.ts`) fires scheduled routines through the same server paths.

## Why You Need This

The first problem OpenBot solves is the oldest one in agent engineering: capability versus control. An agent that may not click anything is a chat box, and an agent that may click anything on a machine holding your logins is an incident waiting for a calendar slot. OpenBot refuses the trade-off by splitting the two halves. The model proposes; the gateway (`server/src/computer/gateway.ts`) disposes. Its header comment states the rule plainly: the record is not a report written alongside the work, it is the thing the action goes through, so an action that was not recorded did not happen — there is no path that acts without writing the audit row first.

The second problem is that "what did it actually click?" is unanswerable if the caller supplies the answer. OpenBot's snapshot flow makes element references opaque to the Bot: `/snapshot` stamps every interactive element with a ref from the accessibility tree, and `/click` and `/type` must present one of those refs back. The server holds the mapping in its own snapshot store (`server/src/computer/snapshot-store.ts`, database-backed so the ref resolves on any replica behind a load balancer), which closes the evasion where a model sends a ref pointing at one element while describing another.

The third problem is framework lock-in. Agent stacks churn, and a governance layer welded to one framework is a governance layer you will rip out. Because a Bot is just an AG-UI endpoint, OpenBot's controls ride the protocol rather than the runtime: the same gateway, policy engine and audit trail sit in front of a LangGraph Bot (`agent-langgraph`), a proof-of-concept Bot (`agent-bot`), or an endpoint you registered from `/agents`. Endpoint registration is validated with the same private-address and URL checks used for browser navigation, at registration and again on every redirect.

The fourth problem is unattended work. A scheduled routine runs on somebody's authority while nobody watches, and that case is worth finding later. Every audit row in OpenBot carries an initiator kind — `person`, `deployment`, `routine` or `handoff` — recorded from a signed run assertion the Bot cannot relabel. The audit screen's "Nobody watching" filter collects the routine and handoff rows, which is precisely the question an operator asks the morning after.

## How It Works

The whole system is organized around one idea — every path from a model's decision to a real machine passes through a single, auditable choke point — and the detailed graph shows how each piece of that path earns its place.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openbot/copilotkit-openbot-architecture.svg" alt="Detailed architecture of the CopilotKit/OpenBot repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of CopilotKit/OpenBot: from the React live screen and activity panels, through the CopilotKit runtime and granted-tool registry, into the action gateway with its CEL engine, snapshot store and target checks, and out to the supervisor's Docker lifecycle and the agent-computer process holding sessions, navigation, ARIA snapshots, screencast, shell and workspace.*

### Understanding the Architecture

**The gateway is the only way in.** `server/src/computer/gateway.ts` does three jobs in order: resolve the ref the caller sent into the element it actually points at, using the snapshot the server fetched; ask the policy; write the audit row whichever way the decision went, and only then act. The class interface exposes navigate, click, type, key, scroll, file reads and writes, command runs, control handovers and secret requests — and nothing reaches the computer except through it. A refused action raises `ActionRefusedError` carrying the rule that refused it, so the transcript can name the boundary instead of mumbling "denied".

**A computer per Bot, provisioned by the only process that holds the Docker socket.** `supervisor/src/index.ts` exposes exactly four verbs — ensure, stop, reset, list — with no passthrough and no way to name a container directly; container names are derived from the validated Bot id in `supervisor/src/names.ts`. The comment is explicit about the threat model: putting Docker access inside the API server would put every request-handler bug one mistake away from the host, so the worst a compromised server can do here is cycle a computer that already belongs to a Bot. Each computer is one container with its own `/workspace` volume and its own browser profile, optionally under gVisor via `COMPUTER_RUNTIME=runsc`, bound to loopback with a required per-container token.

**One long-lived browser, addressed by reference.** `agent-computer/src/index.ts` keeps a single Playwright-driven browser open per Bot so state survives between turns — a session signed in an hour ago is still signed in now. Its ARIA snapshot module (`agent-computer/src/aria-snapshot.ts`) turns the accessibility tree into a compact ref-stamped element list, which is why filling a form needs no vision model: the Bot reads the fields rather than guessing coordinates from pixels. Files stay confined to the workspace volume by `agent-computer/src/workspace.ts`, and shell commands run through `agent-computer/src/shell.ts`, inheriting only PATH, locale, terminal and proxy variables — never the deployment's environment.

**Policy that fails closed.** `server/src/computer/policy.ts` evaluates CEL expressions over a context that includes `tool.name`, `intent`, `bot.id`, `actor.id`, `page.url`, `page.host`, `element.*`, `key`, `command`, `file.*`, `mcp.*` and `initiator.*`. Deny rules are evaluated before allow rules; a missing or empty policy permits nothing; a broken rule refuses rather than opens; and a malformed configured policy stops server startup outright. The policy lives in `server/src/computer/policy-store.ts` and is edited live from the `/admin/boundaries` screen, so a rule added at runtime is in force on the next action.

**Visibility is a product surface, not a log file.** Beside every conversation the app renders two panels from `app/src/components/computer/`: the live screen (`live-screen.tsx`) proxies the Bot's actual browser over a websocket through the channels layer (`server/src/channels/socket.ts`), and the activity log (`activity-log.tsx`) lists what the Bot ran away from the browser — every command with its output and exit code, every file read, write and listing, newest first. A saved file contributes its path and size and never its contents, matching the write route, which declines to echo them.

**Humans can take the wheel, and secrets stay out of the transcript.** When a Bot hits a login wall or a two-factor prompt it calls for help; control handover is recorded as `computer.help_requested`, `computer.control_taken` and `computer.control_released`, and while a person is driving, Bot actions are refused rather than queued (`agent-computer/src/control.ts`). Secret entry is a separate flow from chat: the audit trail records that a secret was requested and its character count, never its value.

Tracing one turn end to end: a person types into a channel route (`app/src/routes/_authed/_app/channel/$channelId.tsx`); the server resolves the actor through `server/src/auth/index.ts` and builds the run in `server/src/copilot.ts`, offering only the tools granted to that Bot; the AG-UI endpoint streams its reply and tool calls back; each acting call enters the gateway, resolves its ref from the snapshot store, passes (or fails) CEL policy, and lands in the audit trail before the computer is touched; `agent-computer` executes it in that Bot's private browser and workspace, streaming frames to the live screen; and results flow back through the runtime into both the conversation and the Intelligence thread that survives restarts.

## Advantages

- **A genuine sandbox per coworker.** Each Bot gets its own container, `/workspace` volume and browser profile, with optional gVisor isolation and loopback-only binding behind a per-container token — not a shared browser with renamed tabs.
- **Ref resolution the model cannot talk its way around.** The server owns the element mapping (`server/src/computer/snapshot-store.ts`), so describing one button while sending the ref of another resolves against reality, not against the description.
- **Fail-closed governance you can read.** Deny-before-allow CEL rules with a documented context; a broken rule refuses, an absent policy permits nothing, and refusals carry the name of the rule.
- **Framework freedom through AG-UI.** Built-in Bots and remote endpoints are the same thing to the control plane; swapping LangGraph for a hand-written server changes nothing about policy or audit.
- **The audit trail is structural, not optional.** Rows are written before actions, refusals name their rule, and initiator kinds distinguish a person's session from a scheduled routine.
- **Answers can be components.** Bots can call governed React components from `app/src/components/gallery/` instead of answering only in prose, with per-component data-function grants.

## Benefits

- **Operational trust from day one.** `/admin/audit` shows permitted, refused and failed actions with the deciding rule, so answering "what did the Bot do?" is a filter, not an investigation.
- **Interruption is a designed path.** Taking the wheel at a login wall is one click, audited, and Bot actions stop rather than queue while a human drives.
- **Secrets hygiene is enforced in code.** Credentials are encrypted at rest, never returned by an API, redacted from audit events, and shell commands inherit a minimal environment.
- **Unattended work has an owner of record.** Routines run through a Postgres work queue (`server/src/work/queue.ts`) with a fifteen-minute floor, a cap on enabled routines, an auto-off after repeated failures, and signed initiator attribution on every row.
- **Memory survives restarts.** Conversations and threads live in CopilotKit Intelligence with per-deployment stamping, so channels persist across deploys and restarts.
- **It runs where you run it.** A single Docker image with an embedded PostgreSQL option takes the same `.env` from a laptop to a company deployment, and nothing is published as a package to depend on — you own the fork.

## Usage

Clone the repository, then create the environment file and install dependencies (Bun 1.3+ and Docker are required):

```sh
cp .env.example .env
bun install
```

Connect a fresh CopilotKit Intelligence project and fill in the model key in `.env`:

```sh
npx --yes copilotkit@latest login
npx --yes copilotkit@latest project select
bun scripts/setup-learning.ts
```

Start the whole stack — Docker services, migrations, the API server on port 3001 and the app on port 3010:

```sh
bash scripts/start.sh
```

Then open `http://localhost:3010` and try `/bot` with a prompt like "Open news.ycombinator.com and tell me the top story", or inspect `/admin/audit` after asking the Bot to fill out a web form. For a single-container deployment with embedded PostgreSQL:

```sh
docker run -p 3001:3001 --env-file .env \
  -e EMBEDDED_POSTGRES=on -v openbot-data:/var/lib/postgresql \
  ghcr.io/copilotkit/openbot:latest
```

## Conclusion

OpenBot's contribution is not another agent framework — it deliberately refuses to be one. What the source delivers is the missing half of the agent story: a computer abstraction per coworker that is isolated by construction, a gateway where every action is resolved against server-held truth, decided by fail-closed policy, and recorded before it happens, and visibility surfaces that make the action stream a first-class part of the product. Reading it is a lesson in how to give an AI real capability without giving up the audit trail, and since it is a template meant to be cloned, the best way to evaluate it is to run it and change it.

Links:

- GitHub repository: [CopilotKit/OpenBot](https://github.com/CopilotKit/OpenBot)
- Project page: [copilotkit.ai/openbot](https://copilotkit.ai/openbot)
- AG-UI protocol: [ag-ui-protocol/ag-ui](https://github.com/ag-ui-protocol/ag-ui)
- Repository documentation: [docs/README.md](https://github.com/CopilotKit/OpenBot/blob/main/docs/README.md)
