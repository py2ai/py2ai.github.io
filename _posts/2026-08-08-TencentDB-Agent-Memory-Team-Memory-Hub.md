---
layout: post
title: "TencentDB Agent Memory: Team-Level Memory Hub for AI Agents"
description: "TencentDB Agent Memory is a team-level memory hub for AI Agents that stores conversations, docs, decisions, and workflows. Supports Claude Code, CodeBuddy, multiple LLM providers, and Docker deployment with MIT license."
date: 2026-08-08
categories: [AI Agents, Memory, Developer Tools, Multi-Agent]
tags: [AI, Agent, Memory, LLM, Multi-Agent, TencentDB, Team Collaboration, Node.js, MIT, open-source]
keywords: "TencentDB Agent Memory, AI agent memory hub, team memory for AI agents, persistent memory multi-agent systems, memory-core memory-hub proxy, AI agent conversation storage, TencentCloud agent memory, Claude Code memory integration, CodeBuddy memory, AI team collaboration memory"
author: "PyShine"
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/tencent-agent-memory/tencent-agent-memory-architecture.svg
---

# TencentDB Agent Memory: Team-Level Memory Hub for AI Agents

Every AI agent user knows the drill. You open your coding assistant, spend 10 minutes re-explaining your project structure, your preferred patterns, the architecture decisions you made last week, and the API quirks you discovered. Then you work for an hour. The session ends. Tomorrow, you do it all over again.

Now imagine a **team** of agents — a coder, a researcher, a documenter — each forgetting everything the moment their session closes. They can't build on each other's work. They can't reference past decisions. They can't learn as a group. Every single interaction starts from scratch.

**TencentDB Agent Memory** ([TencentCloud/TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory)) solves this problem. It's a **team-level memory hub** for AI Agents — a centralized system that remembers conversations, documents, decisions, and workflows across your entire agent team. Built in Node.js, MIT-licensed, with 17k+ GitHub stars, it's the infrastructure layer that turns a flock of stateless agents into a coordinated, learning team.

## The Problem: Agents Forget, Teams Suffer

The current generation of AI coding agents are powerful but stateless. They operate in isolation:

- **No cross-session memory** — every conversation starts with a blank context
- **No cross-agent memory** — agents can't share what they've learned with each other
- **No persistent knowledge base** — decisions, workflows, and discovered patterns vanish when the session ends
- **Repeated work** — you re-explain your stack, your codebase, and your preferences to every new agent interaction

For individual developers, this means lost productivity. For teams using multi-agent systems, it means entire classes of knowledge — architectural decisions, incident response playbooks, onboarding workflows — that can't accumulate or compound.

TencentDB Agent Memory addresses this by giving agents a shared, persistent memory layer that lives between sessions and across agent boundaries. It's not just a memory for one agent — it's a **memory hub** for an entire team.

## What TencentDB Agent Memory Is

TencentDB Agent Memory is an open-source, Node.js-based system that provides persistent, shareable memory for AI agents. It consists of three core services working together:

- **memory-core** — the core memory engine that manages storage, retrieval, and lifecycle of memories
- **memory-hub** — the centralized hub that enables team-level memory sharing and collaboration
- **memory-proxy** — the integration layer that connects memory to your favorite coding agents like Claude Code and CodeBuddy

The philosophy is simple: **agents remember, humans innovate**. By letting agents retain context across sessions and share it across the team, you stop wasting time on repetitive explanations and start focusing on creative, high-value work.

## Key Features

### Team Memory Beta

The standout feature — **Team Memory** allows agents within a team to share memory in real time. When one agent discovers a pattern, documents a decision, or captures a workflow, all agents in the team can access it. This transforms isolated agents into a collaborative unit.

### Memory Hub: Conversations, Docs, Decisions, Workflows

The Memory Hub is where everything lives:

| Memory Type | What It Stores | Use Case |
|-------------|---------------|----------|
| **Conversations** | Full dialogue history with context | Resume discussions seamlessly |
| **Documents** | Project docs, specs, READMEs | Agents reference project knowledge |
| **Decisions** | Architecture choices, trade-offs | Understand why the code is the way it is |
| **Workflows** | Proven multi-step processes | Replicate successful patterns |

### Memory Proxy: Integrate with Claude Code and CodeBuddy

The Memory Proxy acts as a bridge between the memory services and your coding agents. It intercepts agent interactions, stores relevant data in the memory hub, and retrieves context when needed.

Supported integrations include:

- **Claude Code** — persistent memory for your Claude Code sessions
- **CodeBuddy** — team memory for Tencent's CodeBuddy agent
- **Any MCP-compatible agent** — works with any tool implementing the Model Context Protocol

### Multiple LLM Providers

TencentDB Agent Memory is provider-agnostic. It supports various LLM backends, so you can use whatever model works best for your use case — whether that's a cloud provider, an open-source model, or a local deployment.

### Docker Deployment

Get up and running in minutes with Docker:

```bash
# Clone the repository
git clone https://github.com/TencentCloud/TencentDB-Agent-Memory.git
cd TencentDB-Agent-Memory

# Start all services with Docker Compose
docker-compose up -d
```

This spins up the memory-core, memory-hub, and memory-proxy services with a single command. No complex configuration, no manual dependency management.

## Architecture: Three Services, One Memory Hub

TencentDB Agent Memory uses a three-tier architecture that separates concerns cleanly while providing a unified memory experience.

### memory-core — The Memory Engine

The **memory-core** service is the foundation. It handles:

- **Memory storage** — persists conversations, documents, decisions, and workflows
- **Memory retrieval** — fast, accurate recall when agents need context
- **Memory lifecycle** — manages creation, updates, expiry, and versioning of memories
- **Semantic search** — finds relevant memories by meaning, not just keyword matching

Think of memory-core as the brain's hippocampus — the region responsible for forming and retrieving memories.

### memory-hub — The Team Memory Hub

The **memory-hub** service is where collaboration happens:

- **Team namespaces** — organize memories by team, project, or domain
- **Shared memory** — agents within a team access the same memory pool
- **Permission management** — control which agents can read, write, or modify memories
- **Memory synchronization** — ensures all team members see the latest state

The hub is what turns isolated memory into team memory. When one agent learns something, every agent on the team can access it.

### memory-proxy — The Integration Layer

The **memory-proxy** service is the bridge to your existing agent tools:

- **Claude Code integration** — automatically captures and recalls context
- **CodeBuddy integration** — connects Tencent's coding agent to the memory hub
- **MCP server** — works with any MCP-compatible agent
- **API gateway** — RESTful API for custom integrations

The proxy is what makes the system practical. You don't need to change how you work with agents — the proxy handles memory transparently in the background.

### How the Three Services Work Together

![TencentDB Agent Memory Architecture](/assets/img/diagrams/tencent-agent-memory/tencent-agent-memory-architecture.svg)

Data flows from your agent tools through the proxy, which handles protocol translation and authentication. The hub routes memory operations to the right team namespace and enforces permissions. The core stores, indexes, and retrieves memories with semantic search capabilities.

## Quick Start Guide

### Prerequisites

- Node.js 18+
- Docker and Docker Compose (for containerized deployment)
- An API key from your preferred LLM provider

### Option 1: Docker Deployment (Recommended)

```bash
git clone https://github.com/TencentCloud/TencentDB-Agent-Memory.git
cd TencentDB-Agent-Memory

# Configure environment
cp .env.example .env
# Edit .env with your LLM provider settings

# Launch all services
docker-compose up -d
```

### Option 2: Manual Setup

```bash
# Install dependencies for each service
cd memory-core && npm install && cd ..
cd memory-hub && npm install && cd ..
cd memory-proxy && npm install && cd ..

# Start each service (in separate terminals or using a process manager)
cd memory-core && npm start
cd memory-hub && npm start
cd memory-proxy && npm start
```

### Connecting Claude Code

Add the memory proxy as an MCP server in your Claude Code configuration:

```json
{
  "mcpServers": {
    "tencent-memory": {
      "url": "http://localhost:3000/mcp"
    }
  }
}
```

Once configured, Claude Code will automatically store conversations and decisions in the memory hub, and recall relevant context when you start new sessions.

### Creating a Team

```bash
# Create a team namespace
curl -X POST http://localhost:3001/api/teams \
  -H "Content-Type: application/json" \
  -d '{
    "name": "backend-team",
    "description": "Backend engineering team"
  }'

# Add a memory entry
curl -X POST http://localhost:3001/api/teams/backend-team/memories \
  -H "Content-Type: application/json" \
  -d '{
    "type": "decision",
    "title": "Use PostgreSQL for user data",
    "content": "After evaluating MongoDB and PostgreSQL, the team chose PostgreSQL for user data storage due to strong ACID compliance and relational constraints.",
    "tags": ["database", "architecture", "decision"]
  }'

# Search team memory
curl http://localhost:3001/api/teams/backend-team/memories/search?q=database+decision
```

## Who Should Use It

### Development Teams

If you're building software with AI coding assistants, TencentDB Agent Memory eliminates the "constant re-explanation" problem. Your team's agents will remember your architecture, your conventions, your decisions — forever.

### Multi-Agent Orchestrators

If you're building or using multi-agent systems, Memory Hub provides the shared state layer that makes agents work as a team instead of competing silos. Each agent can build on what the others have learned.

### DevOps and SRE Teams

Capture incident response workflows, runbooks, and troubleshooting patterns. When an incident occurs, your AI agent can recall the exact steps that worked last time instead of starting from scratch.

### Technical Writers and Documentation Teams

Agents can reference existing documentation, maintain consistency across documents, and automatically update docs when code changes — without re-reading the entire codebase every time.

### Research Teams

Build a persistent knowledge base that accumulates findings, hypotheses, and experimental results. New agents can immediately access the full research history instead of rediscovering old conclusions.

## Why TencentDB Agent Memory Stands Out

| Feature | TencentDB Agent Memory | Other Solutions |
|---------|----------------------|-----------------|
| **Memory Scope** | Team-level shared memory | Often single-agent only |
| **Architecture** | 3-service modular (core/hub/proxy) | Monolithic or single-service |
| **LLM Support** | Multiple providers | Limited to one provider |
| **Integrations** | Claude Code, CodeBuddy, MCP | Varies by tool |
| **Deployment** | Docker Compose, one-command | Often complex setup |
| **License** | MIT (permissive) | Varies |
| **Stars** | 17k+ | Varies |

> **Key Differentiator:** TencentDB Agent Memory is designed from the ground up for **team-level memory**, not just single-agent memory. The memory-hub service provides the collaboration layer that other solutions lack.

## Conclusion

TencentDB Agent Memory fills a critical gap in the AI agent ecosystem: persistent, shareable memory for agent teams. By separating concerns into three services — memory-core for storage, memory-hub for collaboration, and memory-proxy for integration — it provides a modular, scalable foundation for any multi-agent system.

With Docker deployment, multi-LLM support, and MIT licensing, it's accessible to everyone from individual developers to large engineering organizations. The 17k+ stars on GitHub speak to the community's need for a solution that lets agents truly remember — and teams truly collaborate.

The future of AI-assisted development isn't about individual agents working in isolation. It's about **teams** of agents that learn together, build on each other's work, and accumulate institutional knowledge. TencentDB Agent Memory is the memory layer that makes this future possible.

**Links:**

- GitHub: [TencentCloud/TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory)
- License: MIT
- Language: Node.js