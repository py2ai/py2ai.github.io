---
layout: post
title: "ENZO: Chat With 300+ Models on Your Own Keys - Inside theguysudo/ENZO"
description: "A source-level tour of ENZO, the self-hosted, bring-your-own-key AI workspace that unifies 300+ models, self-drafting agents, deep research, and a generated-project sandbox behind one Express server."
date: 2026-10-07
header-img: "img/post-bg.jpg"
permalink: /enzo-ai-workspace/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/enzo/theguysudo-enzo-architecture.svg
tags: [AI Agents, Self-Hosted, TypeScript, LLM]
categories: [AI, Open Source]
keywords: ENZO, self-hosted AI workspace, bring your own key, model catalog, agent loop, deep research, Groq, OpenRouter, Express, TypeScript
author: "PyShine"
---

Most AI tools today ask you to rent intelligence through someone else's meter. You sign up, you get a token, you get rate limits shaped by someone else's margins, and every prompt you write passes through a middleman that takes a cut or logs a copy. ENZO, published by theguysudo under the Apache-2.0 license, takes the opposite route. It is a self-hosted AI workspace that runs entirely on your infrastructure, speaks to more than 300 models across nine providers, and uses exactly the keys you paste into it. When you send a message, the request flows from your browser through the ENZO server to the provider you picked, and you pay that provider their normal price. Nothing sits in between.

The project is written in strict-mode TypeScript end to end, from the Express backend in [index.ts](https://github.com/theguysudo/ENZO/blob/main/index.ts) to the React workspace under [synthetic-nature/src/App.tsx](https://github.com/theguysudo/ENZO/blob/main/synthetic-nature/src/App.tsx). There is no account system, no usage meter, and no subscription. The first live-validated key you paste claims the instance: it is written to the container environment and sealed into a persistent volume, which unlocks every server-side feature, including agents, skills, and memory. Keys are protected by a passphrase-encrypted vault in the browser, with a downloadable recovery file, and you can wipe them at any time.

What makes ENZO interesting as a codebase is how much it packs into one coherent design. It combines streaming chat with thinking and research modes, a unified model marketplace, an agent builder that drafts its own operating manual from a plain-English task, seventy-four injectable skills, a deep research engine, a long-term memory store, and a sandboxed project runtime that generates and runs small web applications on request. This tour walks through the repository structure, explains how the pieces cooperate, and shows you how to run it yourself in a few commands.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/enzo/theguysudo-enzo-overview-architecture.svg" alt="Architecture overview of the ENZO repository, showing the React workspace, the Express API server, the model catalog and health monitor, the agent loop with skills and memory, and the project sandbox" style="max-width:100%;">
</div>
<p><em>Architecture overview of the ENZO repository, from the browser workspace down to the agent loop, memory, and skills.</em></p>

Reading the overview from left to right:

- The **React workspace UI** lives in [synthetic-nature/src/App.tsx](https://github.com/theguysudo/ENZO/blob/main/synthetic-nature/src/App.tsx) and is the single entry surface for chat, the model marketplace, agents, and projects. The key vault client in [synthetic-nature/src/lib/keyVault.ts](https://github.com/theguysudo/ENZO/blob/main/synthetic-nature/src/lib/keyVault.ts) encrypts provider keys behind a passphrase before anything leaves the browser.
- The **Express API server** in [index.ts](https://github.com/theguysudo/ENZO/blob/main/index.ts) exposes every capability over REST and server-sent events on port 5001. It verifies vault sessions through [src/core/vault-token.ts](https://github.com/theguysudo/ENZO/blob/main/src/core/vault-token.ts) and rate-limits each route group.
- The **model catalog** in [src/models/model-sync.ts](https://github.com/theguysudo/ENZO/blob/main/src/models/model-sync.ts) merges provider listings into one searchable catalog of 300+ models, while the health monitor in [src/models/health.ts](https://github.com/theguysudo/ENZO/blob/main/src/models/health.ts) probes them live and feeds the toolbar's health trace.
- The **agent loop** in [src/agent/agent-tools.ts](https://github.com/theguysudo/ENZO/blob/main/src/agent/agent-tools.ts) executes chat and agent runs, injecting playbooks from the bundled skill library in [src/skills/bundled-skills.ts](https://github.com/theguysudo/ENZO/blob/main/src/skills/bundled-skills.ts) and building its context from the memory store in [src/core/memory.ts](https://github.com/theguysudo/ENZO/blob/main/src/core/memory.ts).
- The **project sandbox** in [src/projects/project-runtime.ts](https://github.com/theguysudo/ENZO/blob/main/src/projects/project-runtime.ts) takes generated code, verifies it, and runs it so you can see results inside the workspace rather than copy-pasting snippets.

## Why You Need This

If you work with several providers, you already know the friction. Each vendor has its own playground, its own billing page, its own quirks, and none of them show your model usage side by side. Subscribing to a unified commercial assistant solves that, but you trade away control of your keys, your data, and your bill. ENZO removes that trade. Because it is self-hosted and bring-your-own-key, the catalog, the agents, and the research engine all run on infrastructure you control, and the provider relationship is direct.

The second reason is agent quality. Most "agent builders" hide their prompts and lock their tools. ENZO's two-pass builder in [src/agents/agents.ts](https://github.com/theguysudo/ENZO/blob/main/src/agents/agents.ts) is different: you describe a task in plain English, the system drafts an operating manual for the agent with a live key, and then runs it. The manual is inspectable and editable, and the agent's work is grounded by seventy-four bundled skills in [skills-bundled/](https://github.com/theguysudo/ENZO/tree/main/skills-bundled), which cover domains from API design to Kubernetes so the loop does not hallucinate procedures from scratch.

The third reason is trust through verification. The repository ships a continuous-integration setup that includes a black-box security pentest on every push, with assertions for auth bypass, insecure direct object references, hostile payloads, and stream integrity, plus unit and security test suites covering agents, the vault, crypto, and model sync. That level of paranoia is rare in self-hosted tools, and it matters when the tool is holding provider keys that can spend your money.

## How It Works

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/enzo/theguysudo-enzo-architecture.svg" alt="Detailed architecture of the ENZO repository, showing the React frontend components, the Express API surface, the agent and research core, the model layer, the vault, memory and projects layer, and the deployment and hardening tooling" style="max-width:100%;">
</div>
<p><em>Detailed architecture of the ENZO repository, including the agent core, the model layer, and the hardening tooling.</em></p>

### Understanding the Architecture

**The frontend is a themed workspace, not a wrapper.** The [synthetic-nature/](https://github.com/theguysudo/ENZO/tree/main/synthetic-nature) application provides a terminal-style chat surface in [synthetic-nature/src/components/terminal/](https://github.com/theguysudo/ENZO/tree/main/synthetic-nature/src/components/terminal), a marketplace theme for browsing model cards, and supporting libraries such as [synthetic-nature/src/lib/codeExtract.ts](https://github.com/theguysudo/ENZO/blob/main/synthetic-nature/src/lib/codeExtract.ts) for lifting code blocks out of answers. The vault client encrypts keys locally, and [synthetic-nature/src/lib/vaultRecovery.ts](https://github.com/theguysudo/ENZO/blob/main/synthetic-nature/src/lib/vaultRecovery.ts) produces the downloadable recovery file. During a chat, an ECG-style trace in the toolbar reflects the live health of the model catalog and flatlines red when it becomes unreachable.

**The backend is one disciplined Express server.** The [index.ts](https://github.com/theguysudo/ENZO/blob/main/index.ts) monolith registers every route group: chat and research endpoints, the vault session flow, music search and streaming, voice endpoints, and the agent and project routers in [src/agents/agentRoutes.ts](https://github.com/theguysudo/ENZO/blob/main/src/agents/agentRoutes.ts) and [src/projects/project.ts](https://github.com/theguysudo/ENZO/blob/main/src/projects/project.ts). Feature modules such as the Cloudflare tunnel in [src/features/tunnel.ts](https://github.com/theguysudo/ENZO/blob/main/src/features/tunnel.ts) and the UI-UX search filter in [src/features/ui-ux-search.ts](https://github.com/theguysudo/ENZO/blob/main/src/features/ui-ux-search.ts) plug into the same server, and live app previews are registered through [src/core/preview.ts](https://github.com/theguysudo/ENZO/blob/main/src/core/preview.ts).

**The model layer treats providers as peers, not gods.** The catalog in [src/models/model-sync.ts](https://github.com/theguysudo/ENZO/blob/main/src/models/model-sync.ts) pulls listings from nine providers into one unified store, and [src/models/model-info.ts](https://github.com/theguysudo/ENZO/blob/main/src/models/model-info.ts) supplies metadata for each card. The health monitor in [src/models/health.ts](https://github.com/theguysudo/ENZO/blob/main/src/models/health.ts) probes availability continuously. When a request comes in, the throttle module in [src/models/throttle.ts](https://github.com/theguysudo/ENZO/blob/main/src/models/throttle.ts) acquires a provider, tracks daily remaining quota, and places cooled-down providers aside so the workspace degrades gracefully instead of failing loudly.

**The agent core is where the workspace earns its keep.** The loop in [src/agent/agent-tools.ts](https://github.com/theguysudo/ENZO/blob/main/src/agent/agent-tools.ts) streams completions through [fetchOpenAIStream](https://github.com/theguysudo/ENZO/blob/main/src/agent/agent-tools.ts) and executes tools through [executeTool](https://github.com/theguysudo/ENZO/blob/main/src/agent/agent-tools.ts), with a small but delightful touch: [findMatchingDraft](https://github.com/theguysudo/ENZO/blob/main/src/agent/agent-tools.ts) detects email-draft requests and hands back a structured draft instead of raw text. Web search is handled by [src/agent/search.ts](https://github.com/theguysudo/ENZO/blob/main/src/agent/search.ts), whose [shouldAutoSearch](https://github.com/theguysudo/ENZO/blob/main/src/agent/search.ts) decides when a query needs fresh results, and the deep research engine in [src/agent/research-engine.ts](https://github.com/theguysudo/ENZO/blob/main/src/agent/research-engine.ts) chains those searches into multi-step investigations. The agent ecosystem in [src/agents/](https://github.com/theguysudo/ENZO/tree/main/src/agents) adds the builder, a data gatherer in [src/agents/gather.ts](https://github.com/theguysudo/ENZO/blob/main/src/agents/gather.ts), neural operations in [src/agents/neural.ts](https://github.com/theguysudo/ENZO/blob/main/src/agents/neural.ts), a scheduler in [src/agents/scheduler.ts](https://github.com/theguysudo/ENZO/blob/main/src/agents/scheduler.ts), and a trainer in [src/agents/trainer.ts](https://github.com/theguysudo/ENZO/blob/main/src/agents/trainer.ts).

**State is split deliberately.** Provider keys persist through the environment manager in [src/core/env-manager.ts](https://github.com/theguysudo/ENZO/blob/main/src/core/env-manager.ts) and the claim flow in [src/core/vault-boot.ts](https://github.com/theguysudo/ENZO/blob/main/src/core/vault-boot.ts). Conversation memory lives in [src/core/memory.ts](https://github.com/theguysudo/ENZO/blob/main/src/core/memory.ts) with functions to build context, record turns, and remember or forget facts. Skills are stored and learned through [src/skills/skills.ts](https://github.com/theguysudo/ENZO/blob/main/src/skills/skills.ts), which can import the bundled library and learn new playbooks from a repository. Generated applications are verified by [src/core/build-verify.ts](https://github.com/theguysudo/ENZO/blob/main/src/core/build-verify.ts) before the runtime in [src/projects/project-runtime.ts](https://github.com/theguysudo/ENZO/blob/main/src/projects/project-runtime.ts) spawns them, so a broken build never reaches your screen silently.

**Deployment is a two-file affair.** The [Dockerfile](https://github.com/theguysudo/ENZO/blob/main/Dockerfile) builds the image published to the GitHub container registry, and [docker-compose.yml](https://github.com/theguysudo/ENZO/blob/main/docker-compose.yml) mounts three named volumes for generated projects, installed skills, and claimed memory so your data survives restarts and upgrades.

End to end, a message travels like this: the React workspace encrypts your request context locally, posts it to the Express server, which validates the vault session, resolves the requested model through the catalog and throttle, and enters the agent loop. The loop streams tokens from the chosen provider, may invoke web search, skills, or memory tools along the way, and returns a streamed answer to the terminal surface. If the answer contains a runnable application, the project router extracts the files, verifies the build, and registers a live preview you can open directly.

## Advantages

- **Direct provider relationship.** Requests go from your browser through ENZO to the provider you picked at their normal price, with no middleman account, no usage meter, and no subscription.
- **One catalog, nine providers.** The sync layer in [src/models/model-sync.ts](https://github.com/theguysudo/ENZO/blob/main/src/models/model-sync.ts) merges 300+ models into a single searchable marketplace, so switching models is a click instead of a new tab and a new login.
- **Self-drafting, inspectable agents.** The two-pass builder in [src/agents/agents.ts](https://github.com/theguysudo/ENZO/blob/main/src/agents/agents.ts) writes each agent's operating manual from a plain-English description, and the manual stays editable rather than hidden.
- **Bundled expertise.** Seventy-four skills in [skills-bundled/](https://github.com/theguysudo/ENZO/tree/main/skills-bundled) give the agent loop concrete domain playbooks, reducing improvisation on unfamiliar topics.
- **Security taken seriously.** Forty-four pentest assertions run on every push through [scripts/pentest.sh](https://github.com/theguysudo/ENZO/blob/main/scripts/pentest.sh), alongside unit and security suites for the agent, vault, crypto, and model layers.
- **Survivable state.** Named volumes for projects, skills, and memory mean a container upgrade never wipes your workspace.

## Benefits

- **Cost control.** You see exactly what each provider charges because you pay them directly, and the throttle module helps you stay within daily quota instead of discovering overages later.
- **Privacy by architecture.** Keys are encrypted in your browser vault and sealed into your own volumes; there is no vendor database to leak because there is no vendor in the loop.
- **Practical research.** The deep research engine in [src/agent/research-engine.ts](https://github.com/theguysudo/ENZO/blob/main/src/agent/research-engine.ts) turns scattered searches into structured findings without leaving the workspace.
- **Runnable outputs.** Generated applications are built, verified, and served live through the preview registry, closing the gap between "the model wrote code" and "the code actually runs."
- **One language, strict mode.** The whole stack is strict TypeScript, so contributors move between frontend and backend without a context switch, and the type system catches integration drift early.
- **Low-friction hosting.** Docker compose, a one-line docker run, or a Google Colab notebook in [notebooks/enzo-colab.ipynb](https://github.com/theguysudo/ENZO/blob/main/notebooks/enzo-colab.ipynb) cover the machine, the server, and the no-install cases.

## Usage

The quickest path is Docker compose:

```bash
git clone https://github.com/theguysudo/ENZO.git
cd enzo
docker compose up -d
# open http://localhost:5001
```

Prefer a single command without cloning? This mounts the same named volumes, so your data and the claimed instance survive restarts and upgrades:

```bash
docker run -d -p 5001:5001 --name enzo \
  -v enzo-projects:/app/generated-projects \
  -v enzo-skills:/app/src/skills/skills \
  -v enzo-memory:/app/data \
  ghcr.io/theguysudo/enzo:latest
```

To develop from source, you need Node 20 or newer, then run the server in watch mode and the test suites:

```bash
git clone https://github.com/theguysudo/ENZO.git
cd ENZO
npm install
npm run dev            # tsx watch index.ts, serving on port 5001
npm test               # agent, vault, crypto, and model suites
npm run build:frontend # builds the workspace UI
```

Once the app is up, press Login, paste a key from a provider such as OpenRouter, Google AI Studio, NVIDIA NIM, Groq, or HuggingFace, and the instance claims itself. From there you can stream a chat, browse the model marketplace, describe a task to build an agent, or ask for a small web app and watch it build and run in the sandbox.

## Conclusion

ENZO is a reminder that the "AI workspace" does not need to be a hosted product with a subscription page. With a strict TypeScript backend, a themed React frontend, a unified catalog across nine providers, self-drafting agents, and a hardened continuous-integration loop, the repository shows what a bring-your-own-key future looks like in practice. The architecture is legible, the state boundaries are deliberate, and the security story is tested rather than promised. If you have been looking for a workspace that treats your keys and your data as yours, this is a codebase worth cloning.

Links:

- Repository: [https://github.com/theguysudo/ENZO](https://github.com/theguysudo/ENZO)
- Live demo: [https://enzo-hub.duckdns.org](https://enzo-hub.duckdns.org)
- Container image: [https://github.com/theguysudo/ENZO/pkgs/container/enzo](https://github.com/theguysudo/ENZO/pkgs/container/enzo)
- License: Apache-2.0
