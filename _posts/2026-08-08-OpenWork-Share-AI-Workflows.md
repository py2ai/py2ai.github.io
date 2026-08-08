---
layout: post
title: "OpenWork: Share AI Workflows Across Agents and Teams"
description: "OpenWork is a free, open-source desktop app for sharing AI workflows. Learn how to install it, wire it into Claude Code, Codex, Cursor, and OpenCode, and share skills, MCPs, and connected services across your team."
date: 2026-08-08
permalink: /OpenWork-Share-AI-Workflows/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/openwork/openwork-architecture.svg
categories: [AI, Open Source, Developer Tools]
tags: [OpenWork, MCP, Claude Code, Codex, Cursor, OpenCode, AI workflows, open source, desktop app, Model Context Protocol]
keywords: "OpenWork, OpenWork MCP, share AI workflows, open source Claude Cowork alternative, OpenWork desktop app, OpenWork Den, Model Context Protocol, Codex MCP, Claude Code MCP, Cursor MCP, OpenCode, team AI workflows"
author: "PyShine"
---

## Introduction

OpenWork is a free, open-source desktop application built for sharing AI workflows across the agents and tools you already use. It is an open-source alternative to Claude Cowork and Codex, available for macOS, Windows, and Linux. The core idea is simple but powerful: add one OpenWork MCP to Codex, Claude Code, Cursor, OpenCode, or any other compatible agent, and you can reuse the same skills, MCP servers, and connected services across your tools, your teammates, and your machines.

The desktop app gives you a dedicated workspace when you want one, but it is not required. You can drive OpenWork entirely from the agent you already have installed. For larger organizations, an admin interface called OpenWork Den lets you publish capabilities, manage access, and configure shared or per-user connections from a single control plane.

Under the hood, OpenWork is built on a modern TypeScript stack: React and Tailwind for the interface, shadcn/ui components, TanStack Query for server state, Zustand for client state, Zod for validation, Drizzle for database access, and Better-Auth for authentication. The project is managed with pnpm workspaces and is fully open source, hosted at [github.com/different-ai/openwork](https://github.com/different-ai/openwork). You can download a prebuilt build from [openworklabs.com/download](https://openworklabs.com/download) or read the official documentation at [openworklabs.com/docs](https://openworklabs.com/docs).

What makes OpenWork different from a single-vendor assistant is its agent-agnostic design. Instead of locking you into one model provider, it works with 50+ LLMs from any provider and runs locally so your files stay on your machine. Teams can package skills, MCP servers, plugins, and configurations into a single shareable link that teammates import in one click, with no terminal or setup guide required.

## How It Works

![OpenWork Architecture](/assets/img/diagrams/openwork/openwork-architecture.svg)

The architecture diagram above shows how OpenWork sits between your AI agents and the capabilities you want to reuse. At the top, four compatible agents are shown in green: Claude Code, Codex, Cursor, and OpenCode. These represent the input layer, the clients where you actually type prompts and run tasks. Each agent connects to the central OpenWork MCP, which is a remote Streamable HTTP MCP server hosted at `https://api.openworklabs.com/mcp/agent`.

The OpenWork MCP is the blue process layer in the middle of the diagram. It exposes exactly two tools to every connected agent: `search_capabilities`, which lets the agent discover what skills, plugins, and connected services are available to it, and `execute_capability`, which actually runs a chosen capability. This minimal two-tool surface is the whole integration contract. After you add the MCP, your client opens a browser so you can sign in and choose your OpenWork organization, and access is then scoped by your org membership, roles, policies, and exposure allowlists.

Below the MCP layer, the diagram splits into two clusters. On the bottom-left, in orange, is the OpenWork Desktop App. This is the local tooling surface that manages your workspace (files and sessions), hosts your skills (portable `SKILL.md` packs), and connects your plugins and MCP connections. The desktop app is what produces the capabilities that flow up into the MCP. When you create or import a skill on your desktop, it becomes something the MCP can expose to any agent you have authorized.

On the bottom-right, in purple, is OpenWork Den, the backend control plane. Den handles administration of organizations and members, groups people into teams, enforces access control through policies and roles, and provisions shared resources such as inference providers and the extension marketplace. The dashed arrows show the governance and data flow: Den scopes and governs what the MCP is allowed to expose, sets desktop policies and publishes marketplaces down to the desktop app, and the desktop app in turn serves its capabilities up to the MCP.

The dashed cross-connections are the important part of the data flow. The desktop app serves capabilities into the MCP (orange dashed edge), while Den governs and scopes that same MCP (purple dashed edge) and pushes policies and published marketplace items back down to the desktop (purple dashed edge). The result is a clean separation of concerns: the desktop is where capabilities are created and run locally, Den is where an organization decides who can use what, and the MCP is the single, thin bridge that any agent can call into.

Because the MCP is a standard remote Streamable HTTP server with OAuth, any client that speaks that protocol can use it. OpenCode is verified with the native remote MCP OAuth flow, while Codex, Claude Code, Cursor, ChatGPT Desktop, and VS Code have documented setup guides that point at the same URL. This means a team can standardize on one set of shared skills and connections without forcing everyone onto the same agent.

## Installation

OpenWork is a desktop app, so the fastest path is to download a prebuilt installer from [openworklabs.com/download](https://openworklabs.com/download) for macOS, Windows, or Linux. After signing in with your OpenWork Cloud account, the guided flow walks you through creating a workspace and opening it ready to use.

If you already use an AI agent that can run commands on your computer, you can paste this prompt into Claude Code, Cursor, Codex, ChatGPT, or any compatible agent and let it perform the installation for you:

```text
Install OpenWork on my computer, set up my first workspace, and open it ready to use. Follow the steps in https://openworklabs.com/start.md?v=hero
```

That single prompt installs OpenWork, creates your first workspace, and opens it ready to run. For local development from source, the repository uses pnpm workspaces. Clone the repo and start the dev server:

```bash
# Clone the repository
git clone https://github.com/different-ai/openwork.git
cd openwork

# Install dependencies and start the dev server
pnpm install
pnpm dev
```

If you need to run multiple git worktrees at once, use the worktree dev script, which derives a stable profile name from the worktree path and lets Electron and Vite pick free ports automatically:

```bash
pnpm dev:worktree
```

Dev startup prints a banner such as `[openwork] dev profile=... cdp=http://127.0.0.1:9223` that you can use to locate the profile directory and pass the CDP URL to local tooling.

## Usage

The OpenWork MCP is the integration point you add to each agent. It brings your assigned skills, plugins, MCP connections, Google Workspace, and Microsoft 365 capabilities into any compatible agent through the two tools described above.

### Codex

Add the remote MCP server, then sign in with Codex's MCP login command:

```bash
codex mcp add openwork --url https://api.openworklabs.com/mcp/agent
codex mcp login openwork
```

To reconnect or switch organizations, log out and back in:

```bash
codex mcp logout openwork
codex mcp login openwork
```

### Claude Code

Add the remote HTTP server, then use the `/mcp` command inside Claude Code to complete the client OAuth flow:

```bash
claude mcp add --transport http openwork https://api.openworklabs.com/mcp/agent
```

### OpenCode

OpenCode is verified with the native remote MCP OAuth flow. Add this entry to your `opencode.json` and then authenticate:

```json
{
  "mcp": {
    "openwork": {
      "type": "remote",
      "enabled": true,
      "url": "https://api.openworklabs.com/mcp/agent",
      "oauth": {}
    }
  }
}
```

```bash
opencode mcp auth openwork
```

### Any MCP client

For any other client that supports remote Streamable HTTP MCP servers with OAuth, use this server URL:

```text
https://api.openworklabs.com/mcp/agent
```

Once the MCP is connected, your agent can call `search_capabilities` to find what is available and `execute_capability` to run it. Access is scoped by your OpenWork organization membership, roles, policies, and exposure allowlists, so two people on the same MCP URL can see very different capabilities depending on what their admin has assigned.

## Features

OpenWork ships with a focused set of features aimed at making AI workflows portable and shareable.

**Desktop app for macOS, Windows, and Linux.** The app is a dedicated workspace where you manage files, sessions, skills, and plugins locally. In desktop mode your files stay on your machine and prompts are sent directly to the LLM provider you choose. Hosted cloud workers are optional and run on sandboxed infrastructure.

**OpenWork MCP integration.** A single remote MCP exposes `search_capabilities` and `execute_capability` to every connected agent. This is the entire contract an agent needs to learn, which keeps integrations thin and stable across agent updates.

**Multi-agent support.** Codex, Claude Code, Cursor, OpenCode, ChatGPT Desktop, and VS Code all point at the same MCP URL. OpenCode is verified with the native remote MCP OAuth flow, and setup guides are provided for the others.

**OpenWork Den control plane.** Den is the admin surface for teams and enterprises. You can provision inference at scale, control which members and teams can use each model provider, invite teammates, create teams, set desktop policies, restrict local model access, control which app versions your organization can use, and publish skills and plugins through marketplaces.

**Google Workspace and Microsoft 365 integration.** Connect Gmail, calendar, Drive, Slack, and other managed services through the Connect flow so that capabilities backed by those services become available to your agents through the MCP.

**Organization management and team access control.** Skills, MCP servers, plugins, and configs can be packaged into a single link that teammates import in one click, with no terminal or setup guide required. Anthropic-compatible plugins can be imported so their supported skills and remote MCPs become available through the OpenWork MCP.

**Bring your own keys and 50+ LLMs.** You connect your own API keys, or use a managed provider on the cloud plans. Any model OpenCode supports works, including OpenAI, Anthropic, Google, and local models across 50+ providers.

## Conclusion

OpenWork solves a real and growing problem: the proliferation of AI agents has made it hard to keep skills, MCPs, and connected services consistent across tools and teammates. By exposing a single, standard MCP that any compatible agent can call, OpenWork lets you create a capability once and reuse it everywhere. The desktop app gives you a local-first workspace, while OpenWork Den gives organizations the governance they need to share safely.

The project is fully open source, built on a modern TypeScript stack, and works with the agents and LLM providers you already use. Whether you are an individual developer who wants to stop re-configuring every agent, or a team that needs to publish and revoke capabilities centrally, OpenWork gives you a clean, agent-agnostic path to share AI workflows.

To get started, download the app from [openworklabs.com/download](https://openworklabs.com/download), read the docs at [openworklabs.com/docs](https://openworklabs.com/docs), or clone the source at [github.com/different-ai/openwork](https://github.com/different-ai/openwork).
