---
layout: post
title: "Doop: Humans and AI Agents Designing on One Canvas - Inside kgoedecke/doop"
description: "Doop is the open-source, self-hostable alternative to Paper.design: a multiplayer design canvas where people edit in the browser and AI agents stream designs in live over a built-in MCP server. We tour the source to see how one Express server, a WebSocket room, and a resident agent team make it work."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /Doop-A-Multiplayer-Design-Canvas-Where-Agents-Design-Too/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/doop/kgoedecke-doop-architecture.svg
tags:
  - Design Tools
  - AI Agents
  - MCP
  - Open Source
categories: [AI, Open Source]
keywords: "doop, multiplayer design canvas, open source figma alternative, MCP server, Claude Code design, AI agent canvas, real-time collaboration, self-hosted design tool, PGlite, Tauri desktop app"
author: "PyShine"
---

Design tools have been quietly converging on the same realization: the next collaborator on a design file is not another human. Most products bolt an AI panel onto the side and call it done — you prompt, it generates, you download, you re-import. The loop is clunky precisely because the agent is not *in* the room. It cannot see your cursor, you cannot watch it work, and neither of you can react to the other in real time.

Doop takes the opposite approach. It is an open-source multiplayer design canvas — pitched openly as the alternative to Paper.design — where every canvas is a live WebSocket room that humans and AI agents join as equals. People edit in the browser; agents connect through a built-in MCP server using the standard OAuth flow and stream their designs into frames section by section, with presence avatars, working status, and an activity feed showing everyone exactly who is doing what. The repository, kgoedecke/doop, is a single TypeScript codebase: one Express server hosting the API, the WebSocket room, and the MCP endpoint; a React client; an embedded Postgres; and a resident agent team that picks up design cards on its own.

The source rewards a tour because it is a complete, opinionated answer to a question half the industry is fumbling: what does it mean for an AI agent to be a first-class participant in a creative tool rather than a bolted-on generator? Doop's answer spans identity (agents act under a human's access), etiquette (agents narrate status and self-review screenshots), economics (bring-your-own model accounts with a carefully staged free tier), and architecture (everything an agent does flows through the same mutation path a human click does).

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/doop/kgoedecke-doop-overview-architecture.svg" alt="Architecture overview of the kgoedecke/doop repository" style="max-width:100%;height:auto;" />
</div>

*The system at a glance: a React client and a Tauri desktop shell talk to one Express server that hosts the WebSocket room, the MCP tool surface and the canvas core; a resident agent queue and connected model accounts staff the AI side, and Drizzle persistence sits underneath everything.*

Reading the overview from left to right: the browser app proxies its API, WebSocket and MCP traffic to the single server on port 4400; the MCP surface and the HTTP handlers both funnel through the canvas actions layer, which mutates the in-memory store that broadcasts to every connected viewer and mirrors writes to the embedded Postgres; and the resident agent team picks up queued cards, runs them through routed model credentials, and edits the canvas exactly like a human-connected agent would.

## Why You Need This

If you have ever tried to get an AI to help with UI work, you know the failure mode: you describe a screen, it generates a static mockup or a wall of code, and the interesting part — iterating with feedback — happens across a copy-paste chasm. Doop collapses that loop to zero distance. You connect Claude Code (or any MCP client) once with a single command, approve the OAuth flow, and from then on the agent works on your canvas as you, streaming a hero section or a pricing table into a frame while you watch it land chunk by chunk.

The second reason is multiplayer itself. Real-time collaboration is notoriously hard to self-host — it usually means CRDTs, operational transforms, or a second sync service. Doop keeps it deliberately simple: one WebSocket room per canvas carries cursors, presence, per-frame editing indicators, comments pinned to elements, and an activity feed. There is no separate realtime infrastructure to run, no vendor lock-in, and no reason the same room cannot seat both a designer and an agent.

Third, it is genuinely self-hostable in a way few "open-source alternatives" manage. `bun run dev` with zero configuration boots a working stack — data persists to an embedded Postgres (PGlite) in a local folder, and every optional integration degrades gracefully rather than erroring. `docker compose up` gives you the production shape. Accounts, private-by-default canvases, email invites, link sharing, workspaces, and even optional Stripe billing for team seats are all in the code, not promised on a roadmap.

Finally, the agent economics are worth studying even if you never deploy it. The built-in Doop Agent runs on the server's key for a configurable handful of free tasks, then transparently hands off to a model account the user connects — a ChatGPT subscription via OAuth, an OpenAI or Anthropic API key, an OpenRouter key, or a Gemini key. Runs are attributed to the human whose card or comment requested them, so the person who asked for the work is the person whose account pays for it. That is a thoughtful answer to the "who pays for the agent" question, implemented rather than debated.

## How It Works

Everything lives in one server process. The web client is a React app on Vite (dev port 4300) that proxies `/api`, `/ws` and `/mcp` to the server on port 4400 — and in production the same server serves the built client, so deployment is one process.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/doop/kgoedecke-doop-architecture.svg" alt="Detailed architecture of the kgoedecke/doop repository" style="max-width:100%;height:auto;" />
</div>

*The detailed architecture: the Express host mounts the MCP surface and routers, the canvas core funnels every mutation through actions into the write-through store, the agent team layers a tool loop over the same actions, and imports, sync and screenshot rendering feed the canvas from outside.*

### Understanding the Architecture

**One server, three protocols.** `server/index.ts` is the whole host: an Express app wired to better-auth for identity, a `ws` WebSocketServer for the live room, and the MCP transport mounted alongside them. The startup banner even tells you which agent mode you are in, because with no server key and no connected accounts the resident agent is deliberately off — queued work simply waits.

**The MCP surface is the product.** `server/mcp.ts` registers the tools an agent drives: `get_guide` (a built-in usage contract the agent must read first), `get_canvas`, `create_frame`, and then the streaming pair that makes designs feel live — `append_frame_html` streams complete HTML sections in small chunks (the instructions suggest roughly one to four kilobytes each) that render the moment they arrive, while `edit_frame_html` does exact find-and-replace edits that morph into the rendered frame in place. Tools for image search, icon search, backgrounds, asset upload and generated imagery round out the kit, and the tool instructions end with the kind of rule you wish every agent had: never ship a placeholder tile where a real logo belongs.

**Every mutation goes through one door.** `server/actions.ts` is the single mutation layer for canvases and frames — humans reach it from HTTP and WebSocket handlers, agents reach it from MCP tools, and the resident agents reach it from their tool loop. Because the path is shared, attribution, access checks, activity feed entries and thumbnails all behave identically no matter who is holding the pen. The in-memory `server/store.ts` holds the hot state in plain Maps and mirrors every committed mutation through `server/db/persist.ts` to Drizzle-backed Postgres, hydrating from there at boot — write-through simplicity instead of a sync engine.

**Agents act as their human.** The MCP endpoint authenticates through the standard OAuth flow, and `server/access.ts` is the same gate for browser and MCP traffic — an agent inherits exactly its human's access to a canvas. Self-issued agent keys are handled in `server/agentKeys.ts` for clients that cannot do OAuth. The instructions require agents to `set_status` when they start or shift focus, so watchers see a narrated, attributed presence rather than a black box.

**The resident team runs the same loop server-side.** `server/resident.ts` implements a Claude-shaped tool loop per agent role with queued work: `shared/agents.ts` defines the roster — a generalist "Doop" builder plus specialists for UX, copy, brand, accessibility and polish — and a board card names an ordered pipeline of roles it walks one stage at a time. Which credential each run uses is `server/agentModel.ts`'s job, and `server/openaiAgent.ts` translates that Anthropic-shaped loop onto OpenAI's Responses API so a connected ChatGPT subscription or OpenAI key can drive it. `server/distill.ts` adds the memory half: it proposes durable style guidelines from your canvas that every agent then follows.

**Self-review is mandatory, not optional.** After every create or significant edit the tool contract requires a `get_frame_screenshot` call — backed by `server/screenshot.ts` rendering the frame — and fixes for whatever looks wrong before moving on. The first-canvas welcome performance (`server/demo.ts`) is a pre-authored frame replayed through this same machinery, so a brand-new install demonstrates the entire experience with zero configuration.

**Imports and integrations feed the canvas.** `server/importer.ts` brings in websites and GitHub repos as starting frames, `server/ingest.ts` implements the live-app sync snippet, and `server/linear` turns delegated Linear tickets into canvases with design cards and reports results back. The test suite in `tests/` covers this surface densely — dozens of files spanning access, agent routing, MCP tools, frame replay and billing — and a Tauri shell in `desktop/src-tauri` wraps the client as a native app.

End to end: a request — human click, MCP tool call, or resident card — reaches the actions layer; the action validates access, mutates the store, and fans out to the WebSocket room while persisting through Drizzle; watchers see the edit land instantly; and if the actor was an agent, its status, presence and eventual screenshot self-review were all rendered in the same feed as everyone else's cursors.

## Advantages

- **Agents are participants, not plugins.** The same mutation path, the same presence feed, and the same access rules serve humans and agents, which is what makes watching an agent design feel like watching a colleague.
- **Streaming edits that render live.** Section-by-section `append_frame_html` delivery means you see the design assemble in real time instead of waiting for a final artifact.
- **True single-process self-hosting.** Embedded Postgres, graceful degradation of every optional integration, and one Docker command for the production shape — no external services required to try it.
- **Sane agent economics.** Free-tier tasks on the server's key, then bring-your-own model accounts with per-human attribution, so nobody subsidizes anyone else's agent usage.
- **Design memory built in.** Pinned exemplar frames, captured decisions, and a distiller that proposes durable style rules the whole agent roster follows.
- **A real desktop app.** The Tauri shell wraps the same client with native save panels for frame exports.

## Benefits

- **Zero-distance iteration.** Ask for a change in the canvas chat, watch the agent stream it in, leave a pinned comment on the element that bothers you — the feedback loop is measured in seconds, not exports.
- **Privacy by default.** Canvases are private until invited; the WebSocket rejects unauthenticated joins; and agents can never exceed the access of the human whose identity they carry.
- **No realtime infrastructure to babysit.** One Node process, one WebSocket room per canvas, write-through persistence — the collaboration stack is small enough to actually understand.
- **Choice of intelligence.** The same agent team runs on Anthropic, Azure OpenAI, a ChatGPT subscription, OpenAI, OpenRouter, Gemini or a Claude key, with per-user model tier selection in Settings.
- **Team-ready out of the box.** Workspaces with owner/admin/member roles, per-seat Stripe billing that is off unless you enable it, email-domain signup restrictions, and optional OIDC SSO.
- **Honest engineering surface.** An unusually dense test suite, a single shared access gate, and documented failure states (like SMTP ports blocked by PaaS hosts) make the system legible to operators.

## Usage

Run it locally with zero configuration:

```bash
git clone https://github.com/kgoedecke/doop && cd doop
bun install
bun run dev
```

The web app serves on http://localhost:4300 and the API, WebSocket and MCP server on http://localhost:4400 — data persists to an embedded Postgres with no setup.

Or self-host the production build with Docker:

```bash
BETTER_AUTH_SECRET=$(openssl rand -hex 32) docker compose up -d
```

Connect Claude Code to your canvas in one command:

```bash
claude mcp add --transport http doop http://localhost:4300/mcp
```

Approve the OAuth window that opens, and the agent works as you. To turn on the built-in Doop Agent's free tier, give the server an Anthropic key:

```bash
ANTHROPIC_API_KEY=sk-ant-...   # in .env, or your deployment's environment
```

## Conclusion

Doop is what an AI-native creative tool looks like when the integration is treated as a first-class design problem. One process hosts the canvas, the room and the agent protocol; one mutation path keeps humans and agents honest; and the economics are attributed per human instead of hand-waved. Whether you deploy it as your team's design canvas or just read it as a reference implementation for building MCP-native collaborative apps, the source is a genuinely instructive piece of work — complete, self-hostable, and opinionated in the right places.

**Links:**

- Repository: [github.com/kgoedecke/doop](https://github.com/kgoedecke/doop)
- Hosted version: [doop.design](https://doop.design)
- Linear setup guide: [docs/linear.md](https://github.com/kgoedecke/doop/blob/main/docs/linear.md)
