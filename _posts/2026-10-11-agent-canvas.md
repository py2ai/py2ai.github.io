---
layout: post
title: "OpenHands: Agent Canvas, a Control Center for Coding Agents - Inside All-Hands-AI/OpenHands"
description: "A source-code tour of OpenHands Agent Canvas, the MIT-licensed TypeScript control center that runs OpenHands, Claude Code, Codex, and Gemini through the Agent-Client Protocol, with cron automations, webhook integrations, and self-hosted agent servers."
date: 2026-10-11
header-img: "img/post-bg.jpg"
permalink: /agent-canvas/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/agent-canvas/All-Hands-AI-agent-canvas-overview-architecture.svg
tags: [AI, Coding Agents, TypeScript, Automation]
categories: [AI, Open Source]
keywords: "OpenHands, Agent Canvas, ACP, coding agents, automation, self-hosted, open source, architecture"
author: "PyShine"
---

The first wave of AI coding tools gave every agent its own window; the next wave is about orchestration. OpenHands, one of the most recognized names in open source coding agents, has evolved its flagship repository into Agent Canvas, which the README describes as a self-hosted developer control center for coding agents and automations. One React application lets you run the OpenHands agent, Claude Code, Codex, Gemini, or any agent speaking the Agent-Client Protocol (ACP), against backends that can be your laptop, a Docker container, a VM, or OpenHands Cloud. The project ships as the npm package @openhands/agent-canvas under an MIT license, and the repository is one of the most instructive TypeScript codebases in the agent ecosystem.

What makes the architecture worth studying is that the frontend is the product. There is no heavyweight control-plane server in the repo; instead, a carefully layered client - React Router routes, typed API services, and small stores - speaks to agent-server backends and ACP agent processes over documented protocols. On top of the conversations sit automations with cron schedules and webhook triggers, integrations with Slack, GitHub, and Linear, and enough operational plumbing - version compatibility checks, LLM balance tracking, device-flow authentication - that the whole thing behaves like real infrastructure rather than a demo.

As always in this series, this is an educational tour, and agent orchestration deserves exactly that caution: automations can post to public services and agents can execute real commands on real machines. The maintainers say it plainly in their self-hosting guide when they warn you to treat an agent VM as you would any exposed server. Study the code, run it on machines you control, and keep credentials scoped.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/agent-canvas/All-Hands-AI-agent-canvas-overview-architecture.svg" alt="OpenHands Agent Canvas overview architecture diagram" style="max-width:100%;"></div>
<p><em>Agent Canvas at a glance: a React and Electron frontend, a client API layer of adapters and automations, and pluggable agent backends from local servers to ACP CLIs and cloud.</em></p>

Reading the overview from left to right:

- The interface starts at [src/routes](https://github.com/All-Hands-AI/OpenHands/blob/main/src/routes), the React Router route tree for conversations, automations, and settings.
- [electron/main.mjs](https://github.com/All-Hands-AI/OpenHands/blob/main/electron/main.mjs) wraps the same app as a desktop application.
- [src/stores/conversation-store.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/stores/conversation-store.ts) is one of the small client stores that hold conversation state.
- [src/api/agent-server-adapter.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/agent-server-adapter.ts) adapts conversations, runtime info, and tags across backend deployments.
- [src/api/acp-service/acp-service.api.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/acp-service/acp-service.api.ts) manages ACP providers, credentials, and models.
- [src/api/automation-service/automation-service.api.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/automation-service/automation-service.api.ts) drives the cron and webhook automation layer.
- [src/api/llm-balance-service.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/llm-balance-service.ts) tracks LLM balance and subscription state.
- The agent-server backends the adapter talks to can be local, in Docker, or on a VM you own.
- [src/constants/acp-providers.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/constants/acp-providers.ts) registers the ACP agent families, from Claude Code and Codex to Gemini and generic CLIs.
- OpenHands Cloud and OpenHands Enterprise appear as optional remote deployments for the same frontend.
- [src/api/git-provider-items-service.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/git-provider-items-service.ts) lists repositories from your connected Git providers.
- [src/api/hooks-service.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/hooks-service.ts) connects automations to third-party services.

## Why You Need This

The first reason is the Agent-Client Protocol made concrete. ACP is emerging as the common way for editors and control centers to drive agent CLIs, and this repository is one of the largest real implementations of the client side. The ACP service and its provider registry show how a single UI represents heterogeneous agents - Claude Code, Codex, Gemini, or any generic CLI - with per-provider credentials, auth status, model discovery, and conflict warnings when two providers overlap. If you are building tooling around agents, this is the reference for treating agents as swappable backends rather than hardcoded features.

The second reason is the automation design. Conversations are treated as first-class records with tag keys for automation trigger, automation id, name, and run id, which means an automated run is auditable in the same timeline as a human chat. The automation service wires cron schedules and webhooks to conversation starts, then routes results outward through webhooks to services like Slack, GitHub, Linear, and Notion. Reading the automation folder teaches how to turn a chat agent into a scheduled engineering team member without inventing a separate execution model.

The third reason is operational maturity in a client codebase. The adapter negotiates backend compatibility versions before talking to a server, tracks runtime services and deployment mode, caches conversation metadata locally, and splits LLM state into balance and subscription services so the UI can warn before a job fails mid-run. The self-hosting documentation then assembles the pieces - static frontend, agent server, automation backend, ingress proxy - with concrete ports and a systemd unit. Few agent projects show this much respect for the operational half of the problem.

## How It Works

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/agent-canvas/All-Hands-AI-agent-canvas-architecture.svg" alt="OpenHands Agent Canvas detailed architecture diagram" style="max-width:100%;"></div>
<p><em>Inside Agent Canvas: frontend surfaces, client state stores, the API service layer, the automation layer, and the backends they reach.</em></p>

### Understanding the Architecture

**One React app, three shells.** The same React Router application in [src/routes](https://github.com/All-Hands-AI/OpenHands/blob/main/src/routes) runs as a browser app served statically, inside an Electron desktop shell whose main process lives in [electron/main.mjs](https://github.com/All-Hands-AI/OpenHands/blob/main/electron/main.mjs), and behind the self-hosted ingress proxy. Route files map the product areas one to one - conversations, automation lists and details, agent settings, ACP credential screens - which keeps navigation readable even as the surface grows.

**Small stores and a single adapter.** Client state is split into focused stores such as [src/stores/agent-store.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/stores/agent-store.ts), [src/stores/conversation-store.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/stores/conversation-store.ts), and [src/stores/event-message-store.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/stores/event-message-store.ts), so agent lifecycle, conversation records, and streamed events each have one home. The agent-state service publishes state transitions into the event store, and the chat service funnels user messages through the adapter, which owns the actual HTTP and event-stream conversation with the selected backend.

**The adapter as the compatibility boundary.** [src/api/agent-server-adapter.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/agent-server-adapter.ts) is the largest and most load-bearing module: it converts backend conversation payloads into app types, exposes runtime services info, resolves the deployment mode, and stamps conversations with reserved tag keys including the ACP server marker and the agentcanvas client source. A dedicated compatibility module guards against protocol drift between frontend releases and agent-server releases, and the endpoint configuration module decides which backend URL applies for local, remote, or cloud modes.

**ACP as a pluggable agent bus.** [src/api/acp-service/acp-service.api.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/acp-service/acp-service.api.ts) manages provider credentials and model lists for ACP agents, with the provider registry in [src/constants/acp-providers.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/constants/acp-providers.ts) covering claude-code, codex, gemini, and a generic CLI fallback. Authentication ranges from stored secrets to an OAuth-style device flow in [src/api/device-flow-client.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/device-flow-client.ts), and the settings screens surface auth status banners and conflict warnings so users can see exactly which agent will run which model.

**Automations as scheduled conversations.** The automation service in [src/api/automation-service/automation-service.api.ts](https://github.com/All-Hands-AI/OpenHands/blob/main/src/api/automation-service/automation-service.api.ts) talks to a backend at /api/automation, creating automations whose trigger specs include cron schedules with webhook entry points. Each run starts a real conversation on the agent server, tagged with its automation identity, and results flow out through the hooks service to third-party tools. Git provider integration feeds repository context into the automation so tasks like issue triage or report generation operate on the right codebase.

**The supporting cast.** LLM balance and subscription services keep model spending visible, the file upload API moves documents into conversations, the conversation metadata store persists local conversation records, canvas extensions let the UI be extended at runtime, and telemetry is a separate service that product decisions can be made from without polluting the domain logic. None of these know about agents directly, which is what keeps the layering clean.

**The end-to-end flow.** A user (or a cron trigger) creates a conversation; the adapter posts it to the chosen agent server, which may wrap the OpenHands agent or an ACP CLI agent; events stream back through the adapter into the stores, rendering in the chat panel; automations repeat this flow unattended and push outcomes to webhooks. The frontend never runs an agent itself - it is a control surface, and every execution stays on a backend you configured.

## Advantages

- **Agent-agnostic by protocol.** ACP support plus the adapter pattern let OpenHands, Claude Code, Codex, Gemini, and custom CLIs coexist in one interface.
- **Backend portability.** The same frontend drives local, Docker, VM, and cloud backends, and switching does not lose your conversation history.
- **Real automation primitives.** Cron schedules, webhook triggers, and per-run tagging turn agents into unattended workers with an audit trail.
- **Operational honesty.** Compatibility checks, LLM balance tracking, and documented ports show production thinking rarely found in agent frontends.
- **Desktop and web in one codebase.** The Electron shell reuses the exact React app, so there is one feature surface to maintain.
- **MIT and readable.** The MIT license and a clean src/ layout make the client layer genuinely reusable in your own products.

## Benefits

- **One pane of glass.** Managing several agents no longer means several windows; conversations, runs, and settings live in one control center.
- **Always-on engineering work.** Server deployments keep automations running when your laptop is closed, which is the difference between a toy and a teammate.
- **Cost visibility.** Balance and subscription services surface LLM spend before a scheduled run silently fails.
- **Self-hosting with real documentation.** The self-hosting guide walks through ports, API keys, systemd units, and security hardening in concrete terms.
- **Extensibility without forking.** Canvas extensions and the hooks service add capabilities through configuration rather than code changes.
- **A pattern library for agent UIs.** The store, adapter, and service layering is directly portable to any team building their own agent console.

## Usage

The self-hosting guide's core command starts the static frontend, the agent server, and the automation backend behind an ingress proxy:

```bash
openssl rand -base64 32
export LOCAL_BACKEND_API_KEY=<key>
npx @openhands/agent-canvas --public
```

That single command brings up the ingress on port 8000, the agent server on 18000, the automation backend on 18001, and the static server on 3001. For development from a clone of the repository, the package.json scripts cover the loop:

```bash
npm install
npm run dev      # frontend + backend in watch mode
npm run build    # production build
```

Point the UI at a backend - local, Docker, VM, or OpenHands Cloud - connect a Git provider for repository context, add an ACP agent such as Claude Code under settings, and schedule your first automation with a cron trigger and a Slack webhook for delivery.

## Conclusion

Agent Canvas is the clearest signal yet that the future of AI coding tools is orchestration, not just completions. The OpenHands team turned a famous agent into a control center where agents are pluggable backends, conversations are auditable records, and automations run on real schedules against real integrations. Read the adapter, then the automation service, then the self-hosting guide, and you will come away with a complete mental model of how agent infrastructure is built in practice.

Links:

- [github.com/All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands)
- [docs.openhands.dev](https://docs.openhands.dev)
- [npmjs.com/package/@openhands/agent-canvas](https://www.npmjs.com/package/@openhands/agent-canvas)
