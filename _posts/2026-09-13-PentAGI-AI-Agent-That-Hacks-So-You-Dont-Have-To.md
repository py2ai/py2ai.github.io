---
layout: post
title: "PentAGI: The Open Source AI Agent That Hacks So You Don't Have To"
description: "PentAGI is a fully autonomous, self-hosted AI penetration testing system. A team of AI agents researches your target, plans the attack, runs 600+ Kali Linux tools in isolated Docker sandboxes, remembers what worked, and writes you a full vulnerability report. 22.7k stars, MIT licensed, Go backend, React frontend, 10+ LLM providers, GraphQL + REST APIs. This is what happens when multi-agent AI meets offensive security."
date: 2026-09-13
header-img: "img/post-bg.jpg"
permalink: /PentAGI-AI-Agent-That-Hacks-So-You-Dont-Have-To/
featured-img: ai-coding-frameworks/ai-coding-framework
tags:
  - PentAGI
  - Penetration Testing
  - Open Source
  - AI Agents
  - Security
  - Multi-Agent
  - Go
  - GraphQL
  - Docker
  - Self-Hosted
author: PyShine
image: ai-coding-frameworks/ai-coding-framework
---

Imagine handing a security intern a target, a Kali Linux terminal, and the instruction "find every way in, then write me a report." Now imagine that intern never sleeps, never gets bored scanning the same ports, remembers every successful exploit from past engagements, and runs everything inside a disposable sandbox so nothing touches your real machine. That intern is **PentAGI** — short for *Penetration testing Artificial General Intelligence* — and it is one of the fastest-growing open source security projects on GitHub right now, sitting at over **22,700 stars** with a clean MIT license.

Let's unpack what makes it tick, why security teams are paying attention, and how you can run it tonight on a laptop with Docker.

## The One-Paragraph Pitch

PentAGI is a **self-hosted, fully autonomous AI penetration testing platform** built by [VXControl](https://github.com/vxcontrol). You give it a target and an objective through a web UI or API. A **multi-agent AI system** decomposes that goal into subtasks, delegates each piece to a specialist agent, runs real pentest tools (nmap, sqlmap, metasploit, and 600+ more) inside **sandboxed Docker containers**, stores everything it learns in a vector database, and hands you back a **detailed vulnerability report** with exploitation guides. It supports over 10 LLM providers, exposes both REST and GraphQL APIs, and ships with Grafana dashboards and Langfuse observability out of the box.

If you have followed the rise of agentic AI coding tools, think of PentAGI as the same idea, pointed at offensive security instead of source code.

## A Team of Specialists, Not One Brain

The single most important design choice in PentAGI is that it does not throw one giant prompt at one model and hope for the best. It runs a **delegation system** of specialized agents, each with a clear job.

![PentAGI multi-agent workflow](/assets/img/diagrams/pentagi/pentagi-multi-agent.svg)

The flow is refreshingly human:

1. **Orchestrator** — receives your pentest *Flow*, breaks it into *Tasks* and *SubTasks*, and before doing anything else it queries the vector store for similar past runs so it benefits from prior experience. It then routes work to the specialists.
2. **Researcher** — analyzes the target surface, searches the knowledge base for known vulnerabilities, identifies open ports and weak points, and stores findings back to memory.
3. **Developer** — plans the actual attack: which tools, which exploit vectors, which Docker image fits the job. It pulls tool capabilities and exploit techniques from the knowledge base.
4. **Executor** — runs the plan. It calls nmap, sqlmap, metasploit, and friends inside the sandbox, loads tool guides from the knowledge base, and streams real-time status back to the orchestrator.

The data model underneath is tidy and relational: a **Flow** contains **Tasks**, which contain **SubTasks**, which produce **Actions**, which yield **Artifacts** and **Memories**. Every command and its output is persisted to PostgreSQL, so nothing is lost and everything is auditable.

## The Architecture: Microservices, Not a Script

A common trap with "AI + security" projects is that they are a thin Python wrapper around an API call. PentAGI is the opposite — it is a full microservices stack designed for horizontal scaling and production observability.

![PentAGI container architecture](/assets/img/diagrams/pentagi/pentagi-architecture.svg)

A few things worth calling out:

- **Backend in Go** with both GraphQL and REST APIs, protected by Bearer token auth. It is multi-tenant via a `tenant_id`, so multiple teams can share one deployment.
- **PostgreSQL + pgvector** is the persistent store. This is not just a log — it is a *semantic memory* database. The agent can ask "have I seen a target like this before?" and get ranked, meaningful answers.
- **Task queue** does async, event-driven dispatch so agent work scales out rather than blocking the API.
- **Knowledge graph via Graphiti + Neo4j** is an optional but powerful layer: it tracks entities and their relationships across engagements, giving the agents genuine context awareness instead of flat keyword matching.
- **Observability is first-class.** OpenTelemetry feeds VictoriaMetrics (metrics), Jaeger (traces), and Loki (logs), all surfaced in Grafana. **Langfuse** separately tracks every LLM call — tokens, costs, prompts — backed by ClickHouse, Redis, and MinIO. You can literally watch the agents think.

The frontend is a modern **React 19 + TypeScript** app (Apollo Client v4, Vite 8) with a file manager, flow control, and real-time task status. Recent releases also added user resource libraries with MD5-deduplicated storage and a first-class, user-manageable knowledge base with semantic search and text anonymization.

## Memory: How It Gets Smarter Over Time

This is the part that separates a toy from a tool. PentAGI has a **three-tier memory system**, and it is the reason the second engagement on a similar target is faster than the first.

![PentAGI memory and context management](/assets/img/diagrams/pentagi/pentagi-memory-system.svg)

- **Long-term memory** lives in the pgvector store — embeddings of past actions, results, success patterns, and domain knowledge. It persists across flows and supports semantic similarity search.
- **Working memory** holds the current context, active goals, and system state for the session in flight. This is what drives each agent's in-the-moment decisions.
- **Episodic memory** records the actual command history and outcomes — what was tried, what worked — and feeds those patterns back into long-term storage so the system learns from every run.

Because LLM context windows are finite, PentAGI also ships a **chain summarization system**. As a conversation grows, older messages are selectively summarized through a multi-step pipeline (convert to a Chain AST, section summarization, QA summarization, rebuild a smaller chain). This prevents token overflow while keeping the conversation coherent — a detail that matters a lot when an attack chain runs for hundreds of tool calls.

## Tools, Search, and the Sandbox That Keeps You Safe

Here is the part that should make any security engineer breathe easy: **every tool runs inside a disposable Docker sandbox**. Nothing executes on your host.

![PentAGI tools, search, and sandboxing](/assets/img/diagrams/pentagi/pentagi-tools-search.svg)

The default sandbox image is `vxcontrol/kali-linux`, which comes preloaded with **600+ professional pentesting tools** — not the "20+" the marketing bullet undersells. The agents do not install tools at runtime; they just call them. Categories include:

- **Network scanning**: nmap, masscan
- **Exploitation**: metasploit, sqlmap, commix
- **Web testing**: nikto, whatweb, gobuster, ffuf, wpscan, zaproxy
- **Password cracking and recon**: hydra, john, hashcat, subfinder, amass

Containers are created dynamically (`pentagi-terminal-1`, `-2`, ...) and cleaned up after the task. A **smart container manager** picks the right image per job, and you can swap in custom images like Parrot Security if you prefer. Host network mode is supported for cases that need it.

Beyond the local toolbox, the agents reach out to **external search systems** for live intelligence: Tavily, Firecrawl, Traversaal, Perplexity, DuckDuckGo, Google Custom Search, Sploitus (for CVE/exploit data), and Searxng. A built-in **isolated web scraper** (headless browser) gathers information from pages directly. The result is an agent that combines the freshest public intel with 600+ local tools — no stale knowledge cutoffs.

## LLM Flexibility: Bring Your Own Brain

PentAGI does not lock you into one vendor. It supports over 10 providers out of the box: **OpenAI, Anthropic, Google Gemini, AWS Bedrock, Ollama, DeepSeek, GLM, Kimi, Qwen, MiniMax**, plus aggregators like OpenRouter, DeepInfra, Atlas Cloud, and OpenCode. You can also point it at any OpenAI-compatible endpoint. For fully air-gapped deployments, the project ships a [vLLM + Qwen3.5-27B-FP8 guide](https://github.com/vxcontrol/pentagi/blob/main/examples/guides/vllm-qwen35-27b-fp8.md) for running a local model. Different agents can even use different models — cheap fast models for research, heavier ones for planning.

## Quick Start: Running It Tonight

You need Docker and an LLM API key. That is genuinely it.

```bash
mkdir pentagi && cd pentagi
curl -O https://raw.githubusercontent.com/vxcontrol/pentagi/main/docker-compose.yml
curl -o .env https://raw.githubusercontent.com/vxcontrol/pentagi/main/.env.example
# Edit .env: add your LLM provider + API key, set a secret key
docker compose up -d
```

Then open the web UI, configure your provider, and submit your first pentest flow. The project also supports giving agents Docker access without exposing your host (useful for advanced workflows), running multiple isolated instances via `tenant_id`, GitHub and Google OAuth for login, and optional Graphiti/Neo4j for the knowledge graph.

The latest release at the time of writing is **v2.1.0**, which added a complete file management layer (user resource libraries with MD5 dedup and per-flow workspace files that sync into worker containers), a first-class user-manageable knowledge base with semantic search and text anonymization, real-time ToolCall observability, and assistant tools to monitor and steer running flows without leaving the chat.

## Why This Matters

There are two ways to react to "AI that hacks." The reflexive one is fear. The productive one is recognition that the same agentic patterns transforming coding, data analysis, and trading are inevitably coming for security testing — and an open, auditable, self-hostable version is far healthier for the industry than a closed black box.

PentAGI's value is not that it replaces a skilled pentester. It is that it handles the tedious 80% — the port scans, the enumeration, the "try this exploit, try that one" — so a human can focus on the 20% that actually requires judgment. It remembers what worked, it runs in isolation, it writes everything down, and it hands you a report. For a small security team that cannot afford a 24/7 red team, that is a serious capability.

A few honest boundaries from the maintainers themselves: PentAGI today is an autonomous and assistant-guided pentest platform, not a CALDERA-style breach-and-attack simulator with predefined campaigns. Flow reports export to web view, clipboard, Markdown, and PDF — JSON export is not yet a supported format. Treat it as a powerful, fast-moving tool, not a finished product.

## Where to Go Next

- **Repository and full documentation**: [github.com/vxcontrol/pentagi](https://github.com/vxcontrol/pentagi)
- **Homepage**: [pentagi.com](https://pentagi.com/)
- **Releases** (including the v2.1.0 notes): [github.com/vxcontrol/pentagi/releases](https://github.com/vxcontrol/pentagi/releases)

If multi-agent AI applied to real-world work interests you, these related posts are worth a read:

- [CloddsBot: Open Source AI Trading Agent Across 1000+ Markets](/CloddsBot-Open-Source-AI-Trading-Agent-1000-Markets/)
- [CowAgent: Open Source Super AI Assistant](/cowagent-open-source-super-ai-assistant/)
- [MathModelAgent: The AI That Turns a 3-Day Math Competition Into 1 Hour](/MathModelAgent-AI-Math-Modeling-3-Days-to-1-Hour/)
- [LLM Wiki: A Personal Knowledge Base That Builds Itself](/LLM-Wiki-Personal-Knowledge-Base-That-Builds-Itself/)

PentAGI is MIT licensed, 22.7k stars, and actively maintained. Pull it down, point it at a deliberately vulnerable target in your lab, and watch a team of AI agents do what used to take a human all night. The future of security testing is autonomous, and it is open source.
