---
layout: post
title: "OpenChatX MCP: Turn ChatGPT into a Local Agent Runtime - Inside XiaoPuOuO/openchatx-mcp"
description: "A source tour of XiaoPuOuO/openchatx-mcp (OpenChatX), a TypeScript local MCP server that bridges ChatGPT to real computer control through OpenAI's Secure MCP Tunnel. We map its HTTP boundary, tool registration pipeline, external MCP server discovery, and permission model with two architecture diagrams."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Openchatx-MCP-ChatGPT-Local-Agent-Runtime-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/openchatx-mcp/xiaopuouo-openchatx-mcp-architecture.svg
tags:
  - ChatGPT
  - MCP
  - AI Agents
  - Open Source
categories: [AI, Open Source]
keywords: "OpenChatX, openchatx-mcp, ChatGPT local agent, MCP server, OpenAI Secure MCP Tunnel, local agent runtime, MCP server discovery, computer control agent, TypeScript MCP, AI agent tools, capability runtime, ChatGPT developer mode"
author: "PyShine"
---

ChatGPT has become a remarkably capable planner, but for most of its life it has been a planner locked inside a browser tab. It can describe how to refactor your repository or run your test suite; it just cannot touch either itself. The Model Context Protocol (MCP) was supposed to close that gap, yet wiring a local MCP server into ChatGPT has historically meant tunnels, reverse proxies, and API plumbing that most people never get quite right. OpenChatX, the project behind the `XiaoPuOuO/openchatx-mcp` repository, makes that bridge a product: it turns ChatGPT into a local agent runtime that can operate your computer, use your tools, and manage your other MCP servers — through one officially supported connection.

OpenChatX is a batteries-included local agent platform written in TypeScript for Node.js (the package targets Node 22.18+ and is MIT licensed). Locally it ships coding-grade file and shell tools, web fetching, custom Toolbox tools, Skills, Rules, Projects, Subagents, and cross-session Summaries, plus an aggregator that folds external MCP servers into the same tool surface. All of it is exposed through a single MCP server that ChatGPT reaches via OpenAI's Secure MCP Tunnel — the documented integration path, not a reverse-engineered hack — and a desktop app wraps the runtime for macOS and Windows. ChatGPT remains the planner; OpenChatX is the local tool and runtime layer underneath it.

The source is worth a tour because "give a cloud chatbot my computer" is exactly the kind of claim that deserves code-level scrutiny. This repository takes its own trust model seriously: the MCP endpoint binds to loopback, remote calls are gated against a persisted OpenAI subject, and the security documentation is honest about what is and is not a sandbox. Reading it tells you how tool exposure stays small enough for ChatGPT's context window, and where the real permission boundaries sit.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openchatx-mcp/xiaopuouo-openchatx-mcp-overview-architecture.svg" alt="Architecture overview of the XiaoPuOuO/openchatx-mcp repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the OpenChatX runtime: a bootstrap layer feeds a loopback HTTP boundary that authorizes remote calls, composes per-request MCP servers, and dispatches into local machine tools and the capability layer.*

Reading the overview from left to right: the static configuration in `src/config.ts` is loaded by the process entry point in `src/index.ts`, which assembles the runtime and starts the HTTP boundary. That boundary (`src/server/http-server.ts`) is the only network door — it authorizes remote calls through the auth store, logs traffic to the audit trail, serves the dashboard under `/ui`, and hands MCP requests to the server factory. The factory (`src/mcp/server-factory.ts`) installs the tool registration boundary, which dispatches into local machine tools — shell and terminal, file and patch, web fetch — and registers the capability layer: the external MCP registry and the Toolbox registry.

## Why You Need This

The first problem OpenChatX solves is tool-context arithmetic. Every MCP server you connect contributes tool schemas to the prompt, and a dozen servers can crowd out the actual work. OpenChatX attacks this from two directions: built-in platform tools, custom toolbox tools, and external MCP tools are registered lazily — the model sees compact discovery tools (`tool_search` and `tool_call`, defined in `src/tools/catalog/catalog-tools.ts`) instead of every schema up front — and the routing instructions in `buildMcpInstructions()` (`src/config.ts`) push the model toward specific tools (`file_read`, `glob`, `grep`), reserving `bash` for genuine shell work.

The second problem is conversation mortality. Long agent work routinely outlives a ChatGPT conversation's practical context limits, and reconstructing state by hand is miserable. OpenChatX's `summarize` tool (in `src/tools/summarize/`, backed by `src/summaries/summary-registry.ts`) stores a compacted summary plus the recent tail of the conversation locally and returns a UUID. Hand that UUID to a fresh conversation and `summarize` returns the stored context — then deletes the handoff automatically.

The third problem is server management itself. If your agent needs a new MCP server mid-task, OpenChatX lets ChatGPT do the wiring. The `mcp_server_manage` tool (`src/tools/mcp-server-management/mcp-server-management-tools.ts`) creates, updates, enables, or disables entries in `mcp-servers.json` with validation and redacted secrets, hot-reloading the registry without a restart. The external registry (`src/external-mcp/registry.ts`) watches the config file on disk and rebuilds connections when it changes.

Finally, there is the packaging problem. A raw stack of Node.js, PM2, and a tunnel client is a lot of moving parts. The desktop app (a Swift host on macOS, a C# host on Windows under `desktop/`) manages the local runtime and bundled tunnel client, stores the control-plane API key in the OS credential store, and runs through regular ChatGPT chat rather than consuming Work or Codex agentic allowance.

## How It Works

Everything starts from a single Node process whose composition order is laid out entirely in `src/index.ts`.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openchatx-mcp/xiaopuouo-openchatx-mcp-architecture.svg" alt="Detailed architecture of the XiaoPuOuO/openchatx-mcp repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of OpenChatX: transport and auth on the loopback boundary, the MCP composition core, focused local machine tools, the capability and store layer, workflow management services, and durable state.*

### Understanding the Architecture

**The process entry composes services exactly once.** `src/index.ts` builds the durable registries — toolbox, project, summary, external MCP, subagent runtime, job manager, context budget guard — and passes them into `createMcpServerFactory()` from `src/mcp/server-factory.ts` before starting the HTTP server. The factory snapshots an immutable runtime profile and produces short-lived MCP server instances on demand, so per-request state never leaks across conversations.

**The transport is loopback-only with subject binding.** `MCP_CONFIG.host` in `src/config.ts` is `127.0.0.1`, so the HTTP server in `src/server/http-server.ts` never exposes a public inbound port; the remote path is OpenAI's Secure MCP Tunnel, an outbound connection made by the official tunnel client. Requests arriving with an `x-openai-subject` header are checked by `OpenChatXAuthStore.authorizeToolCall()` (`src/auth/store.ts`), which binds the first observed OpenAI subject on disk (mode `0600`) and rejects later mismatches with a JSON-RPC 403. `McpAuditLogger` (`src/server/audit/audit-log.ts`) records `tools/list` requests and completed `tools/call` metadata.

**A boundary wraps every tool call.** The factory installs `installToolRegistrationBoundary()` from `src/mcp/tool-registration-boundary.ts`, overriding `registerTool` so all tools share one execution pipeline. Until the model calls `start_here` — the bootstrap tool in `src/tools/start-here/start-here.ts` that injects AGENTS.md instructions, project context, and always-applied rules — other tools fail with `INITIALIZATION_REQUIRED`. While Project routing is pending, only `glob` stays available; everything else gets `PROJECT_ROUTING_REQUIRED`. The boundary also compacts results, appends progress heartbeats, and enforces the context budget from `src/mcp/context-budget.ts`.

**Local machine control is a set of focused tools, gated by Project scopes.** The shell group pairs a one-shot `bash` tool (`src/tools/shell/bash-tool.ts`) with interactive terminal sessions built on `node-pty` (`src/tools/shell/interactive-shell.ts`), capped at four concurrent sessions with a one-megabyte transcript buffer apiece. File read/write/edit (`src/tools/file/file-tools.ts`), the first-class `apply_patch` editor (`src/tools/apply-patch/apply-patch.ts`), glob and grep search, image viewing, and web fetching with bounded document retention (`src/tools/web/web-open.ts`) round out the surface. Path resolution flows through `ProjectScope` (`src/projects/project-scope.ts`): a Project declares read/write/shell permissions and path roots, and anything outside the root raises `PROJECT_EXTERNAL_ACCESS_REQUIRED` until granted. SECURITY.md is explicit that scopes are a policy layer, not an OS sandbox — shell commands run with your user's permissions.

**MCP server discovery is hot and lazy.** `createExternalMcpRegistry()` in `src/external-mcp/registry.ts` reads `mcp-servers.json` (see `mcp-servers.example.json` for the shape: local `command` entries and remote `url` entries with optional headers, environment, and timeouts), connects enabled servers over stdio or Streamable HTTP, and exposes their tools under `mcp:<server>:<tool>` identifiers. A filesystem watcher debounces config edits and rebuilds the connection set without a restart. The `tool_search`/`tool_call` pair is how ChatGPT reaches those tools: search across `builtin`, `toolbox`, and `mcp` sources, then invoke by returned id, while `mcp_server_list` reports servers with redacted secrets and `mcp_server_manage` performs validated writes.

**Capabilities are user-extensible at every layer.** Toolboxes are folders under `toolboxes/` — nineteen ship with the repo — each with a `toolbox.json` manifest and optional custom TypeScript tools, loaded by `src/toolbox/registry.ts`. Skills are portable `SKILL.md` files found on demand by `skill_search`; Rules are `.mdc` files with Always, Auto Attached, Agent Requested, and Manual modes. The Capability Store (`src/store/store-service.ts` with `src/store/github-community-store.ts`) installs community capabilities from GitHub repos tagged `openchatx-capability`, pinned to an immutable commit and rejecting symlinks and git submodules. Subagents (`src/subagents/runtime.ts` with `SmartModelRouter`) delegate work to explicitly configured model profiles.

The end-to-end flow ties it together: ChatGPT calls a tool, the request travels through the Secure MCP Tunnel to the loopback `/mcp` route, `src/server/http-server.ts` resolves the session into an agent identity via `AsyncLocalStorage`, the factory spins up a short-lived MCP server, the registration boundary gates and audits the call, the handler does the real work — say, spawning a persistent terminal through `node-pty` — and the compacted result flows back through the same official channel.

## Advantages

- **One connection, many servers.** Local and remote MCP servers, custom Toolbox tools, and built-ins all arrive through a single MCP connection, so ChatGPT configuration stays simple.
- **Small tool context by design.** Lazy registration via `tool_search`/`tool_call` and compressed tool output keep the model's prompt focused instead of drowning it in schemas.
- **Official transport, no scraping.** The bridge uses OpenAI's documented Secure MCP Tunnel path — no undocumented endpoints, no session-cookie reuse, no traffic interception.
- **Hot-reconfigurable from chat.** `mcp_server_manage` validates and hot-reloads server configuration, and the filesystem watcher in the external registry picks up manual edits — no restarts.
- **Auditable and observable.** Timestamped audit logs, a localhost dashboard with live tool activity, and tool-call presentation make agent behavior inspectable after the fact.
- **Cross-platform desktop packaging.** Signed and notarized macOS builds and Windows installers hide the Node/PM2/tunnel-client machinery behind a normal app.

## Benefits

- **Real computer control, not a demo.** Persistent interactive terminals, a first-class patch editor, glob/grep search, and file tools cover the daily loop of an agent working on your machine.
- **Work that survives the conversation.** UUID-based summary handoffs let multi-day tasks hop across fresh ChatGPT conversations without losing state.
- **Permission semantics that match reality.** Project scopes make agent intent explicit per folder, with grant-once and grant-for-session escapes, and remote calls are bound to a single OpenAI subject.
- **An honest security story.** SECURITY.md tells you plainly that localhost access is unauthenticated by design, scopes are not an OS sandbox, and community capabilities are third-party code.
- **Extensibility without forking.** Toolboxes for TypeScript tools, `SKILL.md` workflows, importable `.mdc` rules, and the community Capability Store let you grow the agent from inside.
- **A hackable codebase.** Biome linting, strict TypeScript, a test suite, and a thorough `wiki/` of architecture notes make the source navigable.

## Usage

The recommended path is the desktop app: grab the DMG (`OpenChatX-macos-arm64.dmg` or `-x64.dmg`) or the Windows installer (`OpenChatX-Setup-x64.exe` / `-arm64.exe`) from the [latest release](https://github.com/XiaoPuOuO/openchatx-mcp/releases/latest), then connect OpenChatX Desktop to ChatGPT via More → Connect Tunnel… using a tunnel ID from the OpenAI Platform and a Runtime API key stored in the OS credential store. In ChatGPT, create an MCP app with connection type Tunnel and select that tunnel.

For a manual source install (requires Node.js 22.18+, npm, ripgrep, and the official OpenAI `tunnel-client`):

On macOS:

```bash
brew install ripgrep
git clone https://github.com/XiaoPuOuO/openchatx-mcp.git
cd openchatx-mcp
npm ci
npm run setup -- --config-only
```

On Windows PowerShell:

```powershell
winget install BurntSushi.ripgrep.MSVC
git clone https://github.com/XiaoPuOuO/openchatx-mcp.git
Set-Location openchatx-mcp
npm ci
npm run setup -- --config-only
```

Then install the official OpenAI `tunnel-client`, create the `openchatx` tunnel profile, export `CONTROL_PLANE_API_KEY`, and run:

```bash
npm run setup
npm start
```

Handy management commands include `npm run status` (runtime/tunnel status), `npm run logs`, `npm run print-url` (local MCP/UI/tunnel URLs), `npm run restart`, `npm run stop`, and `npm run auth:reset` (clear the bound ChatGPT subject).

## Conclusion

OpenChatX is one of the more complete answers yet to "what should a ChatGPT-to-your-computer bridge actually look like?" The answer in `XiaoPuOuO/openchatx-mcp` is architectural: a loopback-only transport with subject binding, one registration boundary every tool must pass through, lazy tool exposure that respects the model's context budget, and permission scopes explicit about being policy rather than sandbox. It is MIT licensed, written in TypeScript, and available as polished desktop builds if you would rather not touch the source at all — though the source is well worth the read.

Links:

- GitHub repository: https://github.com/XiaoPuOuO/openchatx-mcp
- OpenAI Secure MCP Tunnel guide: https://developers.openai.com/api/docs/guides/secure-mcp-tunnels
- ChatGPT Developer Mode and MCP apps: https://help.openai.com/en/articles/12584461-developer-mode-and-mcp-apps-in-chatgpt
