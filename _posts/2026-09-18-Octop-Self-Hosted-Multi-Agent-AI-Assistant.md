---
layout: post
title: "Octop: A Self-Hosted AI Assistant Where Every User Gets an Agent Team"
description: "TencentCloud's Octop is a self-hosted, multi-user, multi-agent AI assistant in a single Python process: web dashboard, IM channels, cron, browser automation, and ACP coding-agent delegation. MIT licensed."
date: 2026-09-18
header-img: "img/post-bg.jpg"
permalink: /Octop-Self-Hosted-Multi-Agent-AI-Assistant/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/octop/octop-architecture.svg
tags: [AI agents, self-hosted, Octop, multi-agent systems, open source]
author: "PyShine"
---

Ask around about AI assistants and you will hear the same complaint again and again: every capable one wants to live in somebody else's cloud. Your conversations, your documents, your schedule, your family's questions — all of it flows through a rented server with a privacy policy you did not write. Octop, an open-source project from TencentCloud that has been climbing the GitHub trending charts with roughly 3,800 stars, makes a different bet: a full multi-user, multi-agent assistant that runs entirely on your own machine, boots with one command, and still reaches you on the chat platforms you already use.

The pitch in the project's README is charmingly direct. Octop is not just a tool, it says, but a digital life form that can operate in parallel. Strip away the poetry and the substance is real. One Python process serves a web dashboard, a CLI, IM channels (Feishu, DingTalk, QQ, Discord, WeCom), and cron automation. Every user of the household or small team gets their own agents, each agent gets its own workspace, memory, and skills, and nothing ever has to leave the machine it runs on.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/octop/octop-architecture.svg" alt="Octop architecture: clients and surfaces, HTTP API layer, domain core, harness libraries, persistence and data" style="max-width:100%;height:auto;" />
</div>

*The architecture map above is drawn from the real repository tree, so every node you can click links to a real file. Notice how the whole right side of the diagram — the harness libraries — is imported rather than reimplemented.*

## One process, no queue, no ceremony

The most opinionated decision in Octop is architectural, and the project defends it in an explicit design record: there is no message broker, no worker fleet, no sidecar database cluster. Web UI, IM channels, and cron jobs all funnel through a single in-process processor. The entire state of the system lives in a control-plane database — SQLite with WAL mode by default, PostgreSQL if you outgrow it — and a restart simply rebuilds everything from that database.

For anyone who has stood up a "simple" self-hosted stack that somehow ended up with Redis, a queue, two workers, and a reverse proxy, this is refreshingly boring engineering. The project layout keeps the discipline visible: the domain core under `infra/` never imports the HTTP layer, the HTTP routers stay thin adapters, and a single composition root wires every singleton at boot. The project even ships an [AGENTS.md](https://github.com/TencentCloud/Octop/blob/main/AGENTS.md) handbook so AI coding agents working on the codebase respect the same module boundaries — a pattern we looked at in [the Pi agent harness](https://pyshine.com/Pi-Agent-Harness-Self-Extensible-Coding-Agent/), and it is good to see a major project making it a hard rule rather than a suggestion.

## A team of specialists, not one generalist

Where Octop departs from the chat-app mold is in how it treats agents. A deployment is multi-user with JWT-based isolation: one admin account, shared household or team, and every request resolved to a specific user row with ownership enforced at the database level. Inside each user's account you can run multiple agents, and each agent is more than a system prompt.

Every agent carries a workspace on disk under `~/.octop/`, a persona drawn from sixteen MBTI personality templates (there is an interactive quiz to pick one), an expert profile from a library that is scanned at boot, and its own set of skills. The workspace is where memory lives — powered by the project's harness-memory library — so an agent that has been helping you with weekly reports actually remembers last week's reports, and the memory travels with the workspace if you switch storage backends.

Speaking of storage: workspaces can sit on local disk, inside Docker containers, or on remote object stores like S3 and COS, all behind one adapter interface. The agent operates inside those boundaries, and risky actions such as shell commands pass through user-editable guardrail rules, with tool approval and PII redaction available before anything sensitive leaves the workspace.

## The harness stack under the hood

Octop does not try to reinvent the agent runtime. It composes four focused libraries, shown as hexagons on the right of the diagram:

- **harness-agent** — the LangGraph-based chat runtime: model routing, tools, skills, and conversation checkpointing
- **harness-gateway** — the bridge that normalizes messages from every IM platform into one pipeline
- **harness-memory** — hierarchical recall with full-text search, so memory migrates with the workspace
- **harness-browser** — CDP-driven browser automation with persistent profiles

That last one deserves a highlight. With Playwright Chromium installed, agents get headless browser sessions for filling forms, capturing screenshots, and gathering information — the same class of capability we covered in [BrowserSkill](https://pyshine.com/BrowserSkill-Let-Your-Coding-Agent-Use-Your-Logged-In-Browser/), but wired directly into a self-hosted assistant instead of a coding agent. There is also browser-plus for remote browsing, an interactive terminal in the dashboard, and live remote desktop streaming for GUI apps.

## ACP: the bridge in both directions

The most forward-looking feature is the Agent Client Protocol integration, and it runs both ways. Inbound, `octop acp` exposes any of your agents as a stdio ACP server, which means editors like Zed or agent tools like OpenCode can use your self-hosted assistant as their brain. Outbound, you can delegate coding tasks from a chat to external coding agents — OpenCode, Claude Code, Codex, and CodeBuddy are the built-in runners — with permission gates in between. Your assistant becomes a coordinator that can dispatch work to specialists and pull the results back into the conversation.

For teams, this turns Octop into something closer to a dispatch desk than a chat window. A request arrives on Feishu, the agent reasons about it, delegates the code change to a coding agent, and posts the outcome back to the group chat — all without a line of glue code you had to write.

## Knowledge, plugins, and cron that speaks English

Two more pieces round out the platform. The knowledge base provides RAG over your own documents: upload files, and semantic retrieval grounds agent answers in your private corpus rather than the open internet. Plugins extend Octop with third-party toolkits, seeded from a bundled set that ships in the repository and toggled from the dashboard.

Cron deserves a mention because of how it fits the story: scheduled jobs are configured in natural language or slash commands, run on APScheduler inside the same process, and can either push a message or run an agent first and push the result. A household assistant that reminds you of things and actually does the follow-up work is the classic demo here, and it works exactly as the architecture suggests it would.

## Getting started

Installation is deliberately forgiving. The one-line curl script provisions its own isolated Python environment, so there is no system Python to break:

```bash
curl -fsSL https://finnie-1258344699.cos.ap-guangzhou.myqcloud.com/octop/install.sh | bash
octop init
octop run
```

Then open `http://127.0.0.1:8088`. If you prefer containers, a Docker Compose file is included, the image writes a generated admin password to a credential file on first boot, and a native desktop app plus an FnOS NAS package are on the release page. Everything — configuration, database, agent workspaces, secrets — lives under one `~/.octop/` directory, which makes backups trivial: there is one folder to protect.

Octop is MIT licensed and on PyPI, and the roadmap points toward shared expert libraries between users and self-evolving skills distilled from everyday conversations. The honest comparison set right now is small, because most self-hosted assistants are single-user wrappers around a chat completion call. Octop is the one that took the harder path — real multi-user isolation, a genuine multi-agent runtime, and a no-queue single-process design — and made it installable in an evening. If you have been waiting for an AI assistant that treats your machine as the boundary, this is the week to try it.
