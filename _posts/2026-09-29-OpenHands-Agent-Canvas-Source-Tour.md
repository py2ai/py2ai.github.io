---
layout: post
title: "OpenHands: A Control Center for Coding Agents - Inside OpenHands/OpenHands"
description: "A guided source tour of OpenHands/OpenHands, the open-source Agent Canvas control center for running coding agents across local, Docker, and cloud backends. We trace the React frontend, the multi-backend registry, the WebSocket event stream, and the launcher scripts that wire the whole stack together."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /OpenHands-Agent-Canvas-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/openhands/openhands-openhands-architecture.svg
tags:
  - OpenHands
  - AI Agents
  - Coding Agents
  - Architecture
categories: [AI, Open Source]
keywords: "OpenHands, Agent Canvas, coding agents, AI software engineer, agent server, WebSocket event stream, React frontend, Zustand stores, Docker sandbox, self-hosted AI agents, ACP agents, automation server, software agent SDK, TypeScript client"
author: "PyShine"
---

If you have ever let an AI agent loose in a terminal, you know the uncomfortable silence that follows: did it run the command? did it edit the right file? is it stuck? OpenHands, the project that began life as OpenDevin and grew into one of the best-known open-source agent platforms, has spent years answering that question with a simple idea — give the developer a real control room, not just a chat box. The repository OpenHands/OpenHands is that control room, and reading its source is one of the better ways to understand how a production agent system is actually wired.

What you find in the repo today goes by the name Agent Canvas. It is the self-hosted developer control center for coding agents: a React and TypeScript application, published as the npm package `@openhands/agent-canvas`, that lets you start conversations and automations with the open-source OpenHands agent as well as third-party ACP-compatible agents such as Claude Code, Codex, and Gemini. Crucially, it is backend-agnostic — the same UI can drive an agent running on your laptop, inside a Docker sandbox, on a remote VM, or on OpenHands Cloud, and it can flip between those backends without losing your place. The heavy lifting of actually executing the agent loop happens in the Python Agent Server, which lives in the companion `software-agent-sdk` repository and is reached through a generated TypeScript client.

That division of labor is exactly why the source is worth a tour. Many agent projects collapse everything into one opaque loop; OpenHands instead draws a clean boundary — the frontend in this repo never executes agent actions itself — and then solves the genuinely hard problems on the client side: multi-backend management, a resumable event stream that merges streaming tokens with REST history, and a launcher stack that proxies frontend, agent server, and automation traffic behind a single origin. Let's walk the tree.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openhands/openhands-openhands-overview-architecture.svg" alt="Architecture overview of the OpenHands/OpenHands repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the OpenHands Agent Canvas: a CLI launcher and ingress proxy bring up the stack, the React frontend talks to a registry of backends through typed API services, and a WebSocket provider feeds a Zustand event store that drives the conversation UI.*

Reading the overview from left to right: the `agent-canvas` CLI (`bin/agent-canvas.mjs`) is the front door — it launches the ingress proxy (`scripts/ingress.mjs`) and serves the built UI, so one command stands up the whole local stack. On the right of the launcher, `src/root.tsx` boots the React application and its route table (`src/routes.ts`), mounting the conversation UI that dominates day-to-day use. Beneath the routes sits the API services layer: a backend registry that tracks every configured agent server and picks a healthy active one, an agent-server adapter, and focused services for events, settings, and automations that all read their host and credentials from that registry. The distinctive piece is the realtime link — a WebSocket provider (`src/contexts/conversation-websocket-context.tsx`) that appends live agent events into a deduplicating Zustand store (`src/stores/use-event-store.ts`), which in turn drives what the conversation panel renders. The automation service is a peer consumer, reached through the ingress proxy's path-based routing rather than through the agent server itself.

## Why You Need This

The first problem Agent Canvas solves is fragmentation. If you try one agent on your laptop, another in a CI container, and a third behind your company's VPN, you end up with three chat windows, three sets of credentials, and no shared history. This repo's answer is the backend registry under `src/api/backend-registry/`: every backend — local agent server, VM, or cloud — is a first-class entry with its own health tracking, persisted selection, and URL resolution, and the UI can add, switch, or fall away from backends gracefully. The fallback logic in `active-store.ts` even prefers a healthy local backend when your saved selection has died, so the app never strands you on a dead endpoint.

The second problem is trust and isolation. Running an agent with full filesystem access on your machine is exactly as dangerous as it sounds, and the README says so in blunt warning blocks. Agent Canvas therefore ships a Docker sandbox mode in which the agent only sees a `PROJECTS_PATH` directory you mount into the container, and it binds local listeners to loopback by default with an auto-generated session key — deliberately refusing to inject that key when you expose the port beyond `127.0.0.1`. For internet-facing installs, the self-hosting guide walks through hardening. Security posture here is a set of concrete code decisions, not a footnote.

The third problem is drudgery. A lot of agent work is not interactive at all: generate a weekly report to Slack, decompose new GitHub issues into tasks, keep branches in sync. The automations feature — the routes under `automations/` in `src/routes.ts` plus `src/api/automation-service/automation-service.api.ts` — connects the canvas to a separate automation backend (itself open source, in the `OpenHands/automation` repo) so runs can fire on schedules or webhooks and dispatch conversations to any of your registered agent servers. Your control center becomes a place where unattended agents live, not just where you chat.

Finally, you need this because agents are only useful when you can see what they are doing. The conversation surface renders not just messages but the agent's terminal commands, browser screenshots, file edits, and task tracking — with the terminal panel powered by xterm and the browser panel fed by real observation events from the runtime. Watching the loop happen, step by step, is what turns "the AI did something" into "the AI ran `pytest`, got three failures, and patched the imports."

## How It Works

The whole system is easiest to understand as three planes — launch, state, and render — connected by one event stream; the detailed diagram follows a single conversation's path through them.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openhands/openhands-openhands-architecture.svg" alt="Detailed architecture of the OpenHands/OpenHands repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of OpenHands: the launcher scripts and ingress proxy on one side, the app shell and route table on the other, the typed API services in the middle, and the WebSocket event stream fanning out into Zustand stores that the feature panels subscribe to.*

### Understanding the Architecture

**The stack launcher.** `bin/agent-canvas.mjs` is the npm binary entry point, and its docstring is a precise spec of the production layout: an agent server via `uvx`, an automation backend via `uvx`, and a pre-built static frontend, all behind one origin. The companion `scripts/dev-with-automation.mjs` documents the same topology in ASCII art — the ingress proxy on port 8000 routes `/api/automation/*` to the automation backend, `/api/*` and `/sockets` to the agent server, and everything else to the frontend dev server. `scripts/ingress.mjs` is that proxy: a small, deliberately backend-independent HTTP router that matches routes by longest path prefix and can append runtime service information into `/server_info` so the frontend discovers where things live instead of guessing ports.

**The backend registry.** Everything the UI does goes to "the active backend," and `src/api/backend-registry/` is where that concept lives. `active-store.ts` keeps a persisted snapshot of configured backends and the current selection, with a carefully documented fallback chain that prefers a healthy local backend; `health-store.ts` tracks liveness; `url-selection.ts` can even read the selection from the URL. On top of this, `src/api/agent-server-compatibility.ts` caches the agent server's `/server_info` so features can be gated by server version, and `src/api/agent-server-client-options.ts` builds the shared connection options — host, session API key, workspace defaults — that every service hands to the generated `@openhands/typescript-client` classes. The service-layer convention is written down in `src/api/README.md`: services are plain objects of async methods, and cloud-specific calls go through `src/api/cloud/proxy.ts` rather than local clients.

**The event stream.** This is the heart of the frontend. When you open a conversation route, `src/contexts/conversation-websocket-context.tsx` opens a WebSocket (via `src/hooks/use-websocket.ts` and `src/utils/websocket-url.ts`) and becomes the single consumer of everything the agent does. Incoming payloads are classified by the type guards in `src/types/agent-server/type-guards.ts` — action events, observations, agent errors, conversation state updates, streaming deltas — and dispatched to the right destination: REST event history is loaded and paginated through `src/api/event-service/event-service.api.ts` (which in cloud mode splits history between the App API and the per-conversation runtime sandbox, authenticating the latter with an `X-Session-API-Key` header), while live events flow into the stores.

**The state stores.** `src/stores/use-event-store.ts` is a small masterclass in event-sourcing hygiene on the client: it deduplicates events by id in a Set, merges consecutive streaming deltas from the same sender instead of appending a node per token, bulk-inserts REST history pages and re-sorts them by timestamp, and maintains a parallel `uiEvents` list shaped for rendering. Around it sit purpose-built stores — `conversation-state-store.ts` keeps execution status scoped per conversation so a planning helper conversation can never clobber the main one's status, while `browser-store.ts` and `command-store.ts` hold browser and terminal observations. The WebSocket context also handles the human side of the loop: optimistic user messages echoed back and matched, and confirmation requests that turn dangerous actions into a yes/no prompt via the event service.

**The render surface.** The conversation UI under `src/components/conversation/` and the chat message list in `src/components/conversation-events/chat/messages.tsx` subscribe to those stores and render the loop as it happens. The panels are honest windows onto the runtime: `src/components/terminal/` wraps xterm for command output, `src/components/browser/` shows pages the agent actually visited, and `src/components/files/` exposes the workspace. Notably, the data flow is bidirectional — the agent server can invoke client tools back into the UI (`src/api/canvas-ui-client-tool.ts`, handled by `src/services/canvas-ui.ts`), and actions like `launch-child-conversation-client-tool.ts` let an agent spawn a helper sub-conversation whose status is tracked separately. The whole thing is packaged unusually well: `npm run build:lib` produces library entrypoints (`browser`, `conversation`, `files`, `settings`, `sidebar`, `terminal`, `i18n`) so other applications can embed pieces of Agent Canvas.

**End to end.** Follow one prompt: you type in the conversation route, the WebSocket context sends it through the typed client to the agent server, the Python-side agent loop executes tools inside its runtime — a local process, a Docker sandbox, or a cloud workspace — and each step comes back as an event. The WebSocket provider classifies it, the event store merges it, and the panels redraw: a command appears in the terminal, a screenshot in the browser panel, a state update flips the execution badge. When you scroll up, older events arrive by REST pagination and slot into timestamp order. The frontend never runs the agent; it makes the agent legible — which is precisely what a control center is for.

## Advantages

- **Backend-agnostic by construction.** The registry in `src/api/backend-registry/` means local, remote, Docker, and cloud agent servers are interchangeable targets; `src/api/agent-server-compatibility.ts` negotiates feature differences by reading the server's own `/server_info`.
- **A real-time event stream with replay.** Live WebSocket events merge with paginated REST history in `src/stores/use-event-store.ts`, so a reload or scroll-up reconstructs the full timeline instead of losing it.
- **Security-conscious defaults.** Loopback-only binding, auto-generated session keys that are withheld in non-loopback and `--public` modes, explicit Docker sandbox instructions, and a self-hosting hardening guide.
- **Multi-agent, multi-runtime.** Any ACP-compatible agent works alongside the built-in OpenHands agent, and child conversations with isolated status tracking enable planner/worker patterns inside one UI.
- **Unattended automations.** Scheduled and webhook-triggered runs dispatch through the automation backend, turning the canvas into an always-on agent host rather than a chat window.
- **Embeddable components.** The library build exposes conversation, terminal, files, settings, and i18n modules for reuse in host applications — the frontend is a product, not just a demo.

## Benefits

- **One place to supervise every agent.** Conversations, terminal output, browser views, files, metrics, and settings live behind a single origin served by the ingress proxy, so you stop context-switching between tools.
- **Lower risk when experimenting.** Sandbox modes and scoped project directories mean you can hand an agent a repository without handing it your home directory.
- **Faster debugging of agent behavior.** Because every action and observation is an event you can see and search, odd agent behavior becomes traceable rather than mysterious.
- **Works the way teams actually run.** Share an agent server for code review bots, keep personal agents on your laptop, and move conversations between them — the backend registry absorbs the change.
- **Honest about its boundaries.** `docs/architecture.md` states plainly what Agent Canvas does not do — no direct action execution, no sandboxing in-process — which makes it easier to reason about failures and to contribute.
- **MIT-licensed and open end to end.** The frontend here, the Python SDK and Agent Server, and the automation backend are all open source, so nothing in the stack is a black box.

## Usage

The quickest start is the npm package, which runs the full local stack (agent server and automation backend via `uvx`, plus the pre-built frontend). Prerequisites are Node.js 24 or later and `uv`:

```sh
npm install -g @openhands/agent-canvas
agent-canvas
```

The launcher can also be split when you want to run pieces separately:

```sh
agent-canvas --frontend-only  # static frontend + ingress only
agent-canvas --backend-only   # agent server + automation backend + ingress only
```

For an isolated sandbox, use the Docker image and mount a projects directory (see the repo's `README.windows.md` for the PowerShell equivalent):

```sh
export PROJECTS_PATH="$HOME/projects"  # directory containing your project folders
mkdir -p "$PROJECTS_PATH" "$HOME/.openhands"

docker run -it --rm \
  -p 127.0.0.1:8000:8000 \
  -e AGENT_CANVAS_ALLOW_LAN_SESSION_KEY=true \
  -v "$HOME/.openhands:/home/openhands/.openhands" \
  -v "${PROJECTS_PATH}:/projects" \
  ghcr.io/openhands/agent-canvas
```

And to work on the source itself:

```sh
git clone https://github.com/OpenHands/OpenHands.git
cd OpenHands
npm install
npm run dev
```

Point your browser at `http://localhost:8000` (or `/canvas` for the Docker image), configure an LLM in the settings page, add extra backends from the UI, and start a conversation. The repo's `docs/architecture.md`, `docs/DEVELOPMENT.md`, and `docs/SELF_HOSTING.md` cover the rest, from runtime modes to internet-facing hardening.

## Conclusion

OpenHands has changed shape more than once — from OpenDevin to a full agent platform, and now to Agent Canvas, a control center that treats coding agents as a fleet to be managed. What makes the repository rewarding to read is that its architecture matches its ambition: a launcher that composes a stack from small pieces, a registry that makes backends swappable, a service layer over a generated TypeScript client, and an event-stream design that treats every agent action as a first-class, replayable fact. If you want to understand how a serious agent system presents itself to humans — without the illusion that the UI runs the loop — clone this repo and start at `bin/agent-canvas.mjs`; the whole journey from there to `src/stores/use-event-store.ts` is well signed.

Links:

- GitHub repository: [OpenHands/OpenHands](https://github.com/OpenHands/OpenHands)
- Documentation: [docs.openhands.dev](https://docs.openhands.dev/openhands/usage/agent-canvas/setup)
- Agent Server / SDK: [OpenHands/software-agent-sdk](https://github.com/OpenHands/software-agent-sdk)
- Automation backend: [OpenHands/automation](https://github.com/OpenHands/automation)
