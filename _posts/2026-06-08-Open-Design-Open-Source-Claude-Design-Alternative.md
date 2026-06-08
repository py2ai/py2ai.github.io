---
layout: post
title: "Open Design: The Open-Source Claude Design Alternative with 261 Plugins and 21 Agent CLIs"
description: "Open Design is a local-first, agent-native design studio that runs 21 CLIs, 150+ design systems, and 261 plugins on your laptop with zero cloud lock-in."
date: 2026-06-08
header-img: "img/post-bg.jpg"
permalink: /Open-Design-Open-Source-Claude-Design-Alternative/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Design, Open Source, Developer Tools]
tags: [Open Design, Claude Design alternative, AI design tool, open source design, coding agent CLI, design systems, HyperFrames, local-first design, agent-native design, BYOK proxy]
keywords: "Open Design open source Claude Design alternative, how to use Open Design locally, Open Design vs Claude Design comparison, AI design tool with coding agents, Open Design 261 plugins tutorial, local-first AI design studio, Open Design design systems setup, agent-native design tool for developers, Open Design installation guide, BYOK proxy AI design tool"
author: "PyShine"
---

## Introduction

In April 2026, Anthropic released Claude Design -- the first LLM that delivers design artifacts directly inside a chat interface. It was a watershed moment for AI-assisted design, but it came with a catch: vendor lock-in, cloud-only storage, and a single model provider. **Open Design**, the open-source Claude Design alternative, removes every one of those constraints. It is a local-first, agent-native design product that turns your existing coding-agent CLI into a design engine -- shipping 261 ready-to-use plugins, 150+ brand-grade design systems, 21 coding-agent CLIs, and 6 artifact types, all running on your laptop with SQLite storage and zero cloud round-trip.

> **Key Insight:** Open Design ships 261 ready-to-use plugins and 150 brand-grade design systems out of the box -- including Linear, Stripe, Vercel, Apple, Tesla, Notion, and Anthropic -- making it the largest open design ecosystem that runs entirely on your laptop.

## What is Open Design

Open Design is what happens when the agent-native design loop stops being closed and becomes a filesystem of skills, design systems, and plugins that coding agents can read, write, and remix. It is the Figma alternative for the agent era: instead of pushing pixels on a canvas, it delivers single-page artifacts in real CSS, real fonts, and real components.

Three composable planes form the foundation:

- **Plugins** -- Runnable workflows that extend generation on demand (261 official plugins)
- **Skills** -- Agent design taste encoded in SKILL.md files (130+ skills)
- **Design Systems** -- Brand contracts encoded in DESIGN.md files (150+ systems)

All three are plain files anyone can author, version, and publish. The DESIGN.md brand contract uses a 9-section schema covering color, typography, spacing, layout, motion, voice, anti-patterns, components, and tokens.

| Feature | Open Design | Claude Design | Figma | Lovable/v0/Bolt |
|---------|-------------|---------------|-------|------------------|
| Local-first | Yes (SQLite) | No (cloud) | No (cloud) | No (cloud) |
| Agent-native | 21 CLIs | Single model | None | Single model |
| Design systems | 150+ (DESIGN.md) | None | Component library | None |
| Skills | 130+ (SKILL.md) | None | Plugins | None |
| Plugins | 261 official | None | Community | Templates |
| BYOK proxy | Yes | No | No | No |
| Open source | Apache-2.0 | Proprietary | Proprietary | Proprietary |
| Desktop app | Yes (Electron) | No (web) | Yes | No (web) |
| MCP integration | Yes (1-line) | No | No | No |

> **Takeaway:** With a single `od mcp install claude` command, your existing Claude Code installation gains a complete design studio -- skills, design systems, plugins, and artifact streaming -- without any additional configuration or API keys.

## Architecture -- How Open Design Works

Open Design supports three deployment topologies, each designed for a different level of local control versus cloud convenience.

**Topology A -- Fully Local (Default)**

Everything runs on the user's machine. The browser connects to a Next.js dev server, which communicates with the OD daemon (a long-running Node/Express process) via HTTP on localhost:7456. The daemon manages sessions, loads skills and design systems, spawns agent CLIs as child processes, and streams their output back as Server-Sent Events (SSE). A single `pnpm tools-dev run web` starts both the Next.js app and the daemon. Zero configuration, zero accounts.

**Topology B -- Vercel + Daemon (Hybrid)**

For teams that want a shared web URL, the Next.js frontend deploys to Vercel while the daemon stays on the user's laptop. The user runs `od daemon --expose` which creates a tunnel URL (e.g., via cloudflared). They paste this URL into the deployed web app's "Connect daemon" screen. The daemon holds all secrets; Vercel holds nothing sensitive.

**Topology C -- Vercel + Direct API (No Daemon)**

The lightest deployment: no local CLI, no daemon. The browser talks directly to the Anthropic or OpenAI API using keys stored in localStorage. This is the "just try it" path with a degraded experience -- no Claude Code skills, no filesystem artifacts (stored in IndexedDB instead), no PPTX export. But it requires zero local setup.

**Daemon Internals**

The daemon is the heart of Topologies A and B. It contains eight core subsystems: the Session Manager tracks active conversations; the Skill Registry loads and caches SKILL.md files; the Agent Adapter Pool manages the lifecycle of spawned CLI processes; the Design-System Resolver reads and caches DESIGN.md files; the Artifact Store persists generated HTML/CSS; the Export Pipeline handles PPTX/PDF/MP4 conversion; the Preview Compile Pipeline renders artifacts into the sandboxed iframe; and the Detection Service scans PATH for installed agent CLIs with a 24-hour cache TTL.

**Data Flow**

All three topologies share the same web bundle. The prompt composition follows a three-layer stack: BASE_SYSTEM_PROMPT (output contract) + active DESIGN.md (brand specification) + active SKILL.md (workflow rules). The daemon composes these layers and delivers them to the agent, which streams back text containing `<artifact>` tags that the parser extracts and renders in the preview iframe.

![Open Design Architecture](/assets/img/diagrams/open-design/open-design-architecture.svg)

The architecture diagram above illustrates the three deployment topologies side by side, along with the daemon internals and data flow.

- Topology A (Fully Local): Browser connects to Next.js dev server on localhost, which communicates with the OD daemon via HTTP on port 7456, which spawns agent CLIs as child processes
- Topology B (Vercel + Daemon): Next.js frontend deploys to Vercel while the daemon stays on the user's laptop, connected via a cloudflared tunnel; Vercel holds no secrets
- Topology C (Vercel + Direct API): Browser talks directly to Anthropic or OpenAI API using keys in localStorage; no daemon, no local CLI, degraded experience
- The daemon contains eight subsystems: Session Manager, Skill Registry, Agent Adapter Pool, Design-System Resolver, Artifact Store, Export Pipeline, Preview Compile Pipeline, and Detection Service
- Data stores include SQLite at .od/app.sqlite for persistence, the filesystem at .od/projects/ for project files, and in-memory caches for skills and design systems
- The prompt composition follows a three-layer stack: BASE_SYSTEM_PROMPT defines the output contract, the active DESIGN.md provides brand specification, and the active SKILL.md provides workflow rules
- The daemon composes all three layers and delivers them to the agent, which streams back text containing artifact tags that the parser extracts and renders in the preview iframe
- A single `pnpm tools-dev run web` starts both the Next.js app and the daemon with zero configuration
- Adding `pnpm tools-dev` also launches the Electron desktop shell for the full local experience
- The Detection Service scans PATH for installed agent CLIs with a 24-hour cache TTL and re-detects on daemon SIGHUP
- The Agent Adapter Pool manages the full lifecycle of spawned CLI processes, including start, stream, cancel, and resume
- The Artifact Store persists generated HTML/CSS to SQLite, while the Export Pipeline handles conversion to PPTX, PDF, and MP4 formats
- The Preview Compile Pipeline renders artifacts into a sandboxed iframe using the srcdoc attribute for live preview
- All three topologies share the same web bundle, ensuring feature parity regardless of deployment mode
- Topology B preserves the full agent-native experience with a cloud-accessible UI by keeping the daemon on the user's laptop
- Topology C stores artifacts in IndexedDB instead of the filesystem and cannot export to PPTX, but requires zero local setup
- The daemon's Skill Registry loads and caches SKILL.md files on demand, with hot-reload when files change on disk
- The Design-System Resolver reads and caches DESIGN.md files, making brand specifications available for prompt composition
- Server-Sent Events (SSE) stream the agent's output back to the browser in real time for live preview updates
- The Session Manager tracks active conversations, ensuring context persistence across multiple interactions within a session

## Skills and Design Systems Ecosystem

### Skills

Skills are the atomic units of design capability. Each skill is a directory containing a SKILL.md file with YAML frontmatter (name, description, triggers, od.mode, od.preview, od.design_system, od.inputs) and a free-form Markdown body describing the workflow the agent should follow. Skills can also include an assets/ directory for templates and a references/ directory for knowledge files.

The 130+ bundled skills are organized by mode:

- **Prototype skills** -- Generate single-page HTML artifacts (web-prototype, saas-landing, dashboard, pricing-page, docs-page, blog-post, mobile-app)
- **Deck skills** -- Create horizontal-swipe presentations (simple-deck, magazine-web-ppt, guizang-ppt)
- **Template skills** -- Handle specialized output formats
- **Design-system skills** -- Apply brand specifications to generated output

A key design decision is backward compatibility: any Claude Code skill works in Open Design without modification, and OD-specific extensions (od.mode, od.preview, od.design_system, od.inputs, od.craft) are entirely optional.

### Design Systems

Design systems are the brand contracts. Each is a DESIGN.md file with a 9-section schema covering color, typography, spacing, layout, motion, voice, anti-patterns, components, and tokens. The 150+ bundled systems span seven visual categories:

- **AI/LLM** -- Claude, OpenAI, Anthropic, Cursor, Hugging Face, Together AI
- **Developer Tools** -- Linear, Vercel, GitHub, Raycast, Supabase, Sentry
- **Productivity** -- Notion, Slack, Figma, Miro, Canva, Loom
- **Fintech** -- Stripe, Coinbase, Revolut, Wise, Shopify, Binance
- **Automotive** -- Tesla, BMW, Ferrari, Lamborghini, Renault, Bugatti
- **Media** -- Spotify, Discord, Pinterest, RunwayML
- **Lifestyle** -- Airbnb, Nike, Starbucks, Duolingo

Drop a new folder into design-systems/ and the picker discovers it automatically.

![Open Design Ecosystem](/assets/img/diagrams/open-design/open-design-ecosystem.svg)

The ecosystem diagram maps the three composable planes that make Open Design extensible: Skills, Design Systems, and Plugins.

- The Skills Plane contains 130+ skills organized by mode: Prototype skills (web-prototype, saas-landing, dashboard, pricing-page, docs-page, blog-post, mobile-app)
- Deck skills create horizontal-swipe presentations (simple-deck, magazine-web-ppt, guizang-ppt)
- Template and Design-system skills handle specialized output formats and brand application
- Each skill is a directory with a SKILL.md file containing YAML frontmatter (name, description, triggers, od.mode, od.preview, od.design_system, od.inputs) and a Markdown workflow body
- Skills can also include an assets/ directory for templates and a references/ directory for knowledge files
- A key design decision is backward compatibility: any Claude Code skill works in Open Design without modification
- OD-specific extensions (od.mode, od.preview, od.design_system, od.inputs, od.craft) are entirely optional and ignored by plain Claude Code
- The Design Systems Plane contains 150+ brand-grade systems, each a DESIGN.md file with a 9-section schema
- The 9 sections cover: color, typography, spacing, layout, motion, voice, anti-patterns, components, and tokens
- Design systems span seven visual categories: AI/LLM (Claude, OpenAI, Anthropic, Cursor, Hugging Face, Together AI)
- Developer Tools category includes Linear, Vercel, GitHub, Raycast, Supabase, and Sentry
- Productivity category includes Notion, Slack, Figma, Miro, Canva, and Loom
- Fintech category includes Stripe, Coinbase, Revolut, Wise, Shopify, and Binance
- Automotive category includes Tesla, BMW, Ferrari, Lamborghini, Renault, and Bugatti
- Media category includes Spotify, Discord, Pinterest, and RunwayML
- Lifestyle category includes Airbnb, Nike, Starbucks, and Duolingo
- Drop a new folder into design-systems/ and the picker discovers it automatically with no configuration
- The Plugins Plane carries 261 official plugins across six categories: Scenarios (12), Image-templates, Video-templates, Design-systems (142), Atoms (13), and Examples
- Scenario plugins include od-code-migration, od-figma-migration, od-react-export, and od-vue-export
- The Composition Flow connects all three planes: when a user sends a brief, the daemon composes the system prompt from three layers
- The three layers are: the base output contract, the active DESIGN.md body, and the active SKILL.md body
- Swap the skill or design system in the top bar and the next send uses the new stack immediately
- Bodies are cached in-memory per session for fast switching between skills and design systems

> **Amazing:** Open Design includes 150 brand-grade design systems spanning 9 visual categories -- from minimalist Linear and Stripe to luxury Ferrari and Lamborghini, from retro Nintendo to futuristic SpaceX -- all as plain DESIGN.md files that any coding agent can read and apply without rendering software.

## Agent Integration -- 21 CLIs and BYOK Proxy

Open Design is agent-native and model-agnostic: it does not ship an agent. Instead, it uses the CLIs already on your PATH. This is the most load-bearing design decision in the architecture.

### Supported Agents

Open Design supports 21 coding-agent CLIs: Claude Code, Codex CLI, Cursor, VS Code + GitHub Copilot, GitHub Copilot CLI, Gemini CLI, OpenCode, OpenClaw, Antigravity, Cline, Trae, Kimi CLI, Pi Agent, Mistral Vibe CLI, and Hermes Agent.

### One-Line MCP Install

```bash
od mcp install claude       # Install for Claude Code
od mcp install codex        # Install for Codex CLI
od mcp install cursor      # Install for Cursor
od mcp install copilot     # Install for GitHub Copilot
od mcp install --print     # Dry-run preview
od mcp install --help      # Full list of agents
```

### Detection Strategy

When the daemon starts, the Detection Service runs all agent adapters' `detect()` methods in parallel. Each adapter uses two signals: a PATH scan (which `<binary>` for each known executable name, completing in under 10ms) and a config-dir probe (checking for ~/.claude/, ~/.codex/, ~/.cursor/, etc.). Results are cached in ~/.open-design/agents.json with a 24-hour TTL and re-detected on daemon SIGHUP.

### Agent Adapter Interface

Each agent adapter implements a TypeScript interface with five methods:

- `detect()` -- Returns installation status and auth state
- `capabilities()` -- Reports what the agent supports (surgical edits, native skill loading, streaming, resume, permission mode, context window)
- `run()` -- Starts an execution and returns an async iterable of events
- `cancel()` -- Terminates a running execution
- `resume()` -- Continues an interrupted run

Adding a new CLI requires only one entry in the adapters registry.

### BYOK Proxy

When no local CLI is available, the daemon falls back to API mode. The user provides their API key and selects a provider. The daemon normalizes the provider's SSE stream into a unified delta/end/error format, then feeds it through the same artifact parser and preview renderer.

![Open Design Agent Integration](/assets/img/diagrams/open-design/open-design-agent-integration.svg)

The agent integration workflow diagram shows how Open Design delegates the entire agent loop to the user's existing coding-agent CLI, rather than implementing its own LLM client.

- The Detection Phase runs when the daemon starts: all agent adapters' detect() methods execute in parallel
- Each adapter uses two signals: a PATH scan (which binary for each known executable, completing in under 10ms) and a config-dir probe (checking for ~/.claude/, ~/.codex/, ~/.cursor/, etc.)
- Detection results are cached in ~/.open-design/agents.json with a 24-hour TTL and re-detected on daemon SIGHUP
- The Rescan button in Settings also triggers a fresh detection at any time
- In the Execution Phase (Local CLI Mode), the daemon composes the system prompt from three layers: base output contract, active DESIGN.md body, and active SKILL.md body
- The daemon then spawns the agent CLI as a child process with the composed prompt, user prompt, and working directory
- The agent's stdout is streamed back as Server-Sent Events (SSE) to the web UI in real time
- The stream parser extracts artifact tags and the preview renderer displays the HTML in a sandboxed iframe using the srcdoc attribute
- In the Execution Phase (BYOK API Mode), when no local CLI is available, the daemon falls back to API mode
- The user provides their API key and selects a provider: Anthropic, OpenAI, Azure OpenAI, Google Gemini, Ollama, LM Studio, vLLM, or any OpenAI-compatible endpoint
- The daemon normalizes the provider's SSE stream into a unified delta/end/error format, then feeds it through the same artifact parser and preview renderer
- The BYOK proxy includes per-target SSRF protection that blocks internal IPs, link-local addresses, and CGNAT ranges at the daemon edge
- For MCP Server Integration, Open Design provides a one-line install: od mcp install agent writes the MCP server configuration into the agent's config file
- This allows the agent to call Open Design tools natively from within its own interface
- The --print flag provides a dry-run preview, and --uninstall removes the configuration cleanly
- The Adapter Interface defines five methods: detect() returns installation status and auth state
- capabilities() reports what the agent supports: surgical edits, native skill loading, streaming, resume, permission mode, and context window
- run() starts an execution and returns an async iterable of events for real-time streaming
- cancel() terminates a running execution gracefully
- resume() continues an interrupted run, preserving context and conversation state
- Adding a new CLI requires only one entry in the adapters registry, making the system easily extensible

> **Important:** The BYOK proxy at POST /api/proxy/{provider}/stream supports OpenAI, Anthropic, Azure OpenAI, Google Gemini, Ollama, LM Studio, vLLM, or any OpenAI-compatible endpoint -- with per-target SSRF protection that blocks internal IPs, link-local, and CGNAT addresses at the daemon edge.

## Artifact Types -- What You Can Create

Open Design supports six artifact types, each designed for a specific output format:

- **Prototypes** -- Web, desktop, and mobile single-page HTML artifacts rendered in a sandboxed iframe with real CSS, real fonts, and real components
- **Live Dashboards** -- KPI walls, decision rooms, and GitHub-style dashboards with an editable tweaks panel
- **Decks** -- Magazine layouts, pitch decks, and weekly updates. Export to HTML, PDF, PPTX, ZIP, or Markdown
- **Images** -- Brand-grade visual assets with 93 ready-to-replicate prompt templates
- **Video and HyperFrames** -- Agent-native motion graphics. HTML+CSS+GSAP rendered to MP4 via headless Chrome + FFmpeg. 11 templates + 39 Seedance prompts for SaaS promos, brand reels, data races, and logo outros
- **Export Formats** -- HTML (single file, inlined assets), PDF, PPTX, MP4, ZIP, Markdown

## Plugin System -- 261 Official Plugins

The 261 official plugins span six categories:

- **Scenarios (12)** -- od-code-migration, od-default, od-design-refine, od-figma-migration, od-media-generation, od-new-generation, od-nextjs-export, od-plugin-authoring, od-react-export, od-share-to-community, od-tune-collab, od-vue-export
- **Image-templates** -- Ready-to-replicate visual asset templates
- **Video-templates** -- Motion graphics templates for HyperFrames
- **Design-systems (142)** -- Brand packs that pair with DESIGN.md files
- **Atoms (13)** -- Reusable component primitives
- **Examples** -- Reference implementations

Community plugins are extensible through the plugins/community/ directory. The plugin spec is defined in docs/plugins-spec.md, and the Automation page lets users orchestrate repetitive design workflows into reusable, schedulable automations.

## Installation

### Method 1: Desktop App (Recommended)

Download from open-design.ai or GitHub Releases. Available for macOS (Apple Silicon + Intel x64), Windows (x64), and Linux (AppImage). After install, the app auto-detects coding-agent CLIs on PATH and loads skills and design systems.

### Method 2: Install into Your Coding Agent (No UI)

```bash
# One-line install into the agent you are using:
curl -fsSL https://open-design.ai/install.sh | sh -s <agent>
# <agent> = claude | codex | cursor | copilot | openclaw | antigravity | gemini
#         | pi | vibe | hermes | cline | kimi | trae | opencode
```

Then inside the agent:

```
> Use open-design to generate a landing page with the Linear design system
```

### Method 3: Docker

```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design/deploy
cp .env.example .env
echo "OD_API_TOKEN=$(openssl rand -hex 32)" >> .env
docker compose up -d
# open http://localhost:7456
```

### Method 4: Run from Source

```bash
git clone https://github.com/nexu-io/open-design.git
cd open-design
corepack enable && pnpm install
pnpm tools-dev run web
```

Requirements: Node ~24, pnpm 10.33.2+

### MCP Server Install

```bash
od mcp install claude       # Install for Claude Code
od mcp install codex        # Install for Codex CLI
od mcp install cursor      # Install for Cursor
od mcp install copilot     # Install for GitHub Copilot
od mcp install --print     # Dry-run preview
od mcp install --help      # Full list of agents
```

## Features

| Feature | Description |
|---------|-------------|
| Agent-native design | Uses 21 existing coding-agent CLIs as the design engine -- no built-in agent, no lock-in |
| 150+ design systems | Brand-grade DESIGN.md systems including Linear, Stripe, Vercel, Apple, Tesla, Notion, Anthropic, Figma |
| 261 official plugins | Scenarios, image-templates, video-templates, design-systems, atoms, examples |
| 130+ skills | SKILL.md-based capabilities for prototypes, decks, dashboards, images, video, HyperFrames |
| Local-first storage | SQLite at .od/app.sqlite, project files at .od/projects/, no cloud round-trip |
| Desktop app | Native macOS (Apple Silicon + Intel) and Windows (x64) apps |
| BYOK proxy | Supports OpenAI, Anthropic, Azure, Google, Ollama, vLLM, any OpenAI-compatible endpoint |
| MCP integration | One-line `od mcp install <agent>` wires into any supported agent config |
| HyperFrames | HTML+CSS+GSAP to MP4 via headless Chrome + FFmpeg for agent-native motion graphics |
| Multi-format export | HTML, PDF, PPTX, MP4, ZIP, Markdown export from any artifact |
| Sandboxed preview | Iframe-based live preview with real CSS, real fonts, real components |
| SSRF protection | Per-target blocking of internal IPs, link-local, and CGNAT at daemon edge |
| Three deployment topologies | Fully local, Vercel+daemon hybrid, Vercel+direct API |
| Automation | Orchestrate repetitive design workflows into reusable, schedulable automations |
| Figma migration | Dedicated plugins migrate Figma/Pencil workflows into React/Next.js/Vue code |

## Troubleshooting

- **"no agents found on PATH"** -- Install one of the supported CLIs or switch to API mode
- **Claude Code exits with code 1** -- Check `claude auth status`, re-authenticate with `/login`
- **better-sqlite3 ABI mismatch after Node version change** -- Run `pnpm --filter @open-design/daemon rebuild better-sqlite3`
- **OD_BIN missing or daemon URL :0** -- Rebuild daemon CLI, restart, reopen project from app
- **Docker Authorization error on macOS** -- Enable host networking in Docker Desktop
- **Artifact never renders** -- The model produced text without wrapping in `<artifact>` tags. Try a stricter skill or more capable model
- **Codex loads too much plugin context** -- Start with `OD_CODEX_DISABLE_PLUGINS=1`

## Conclusion

Open Design delivers the open-source, local-first, agent-native design studio that the market has been waiting for. With 261 plugins, 150+ design systems, 130+ skills, and 21 agent CLIs, it offers the largest open design ecosystem that runs entirely on your laptop. The composable architecture -- skills, design systems, and plugins as plain files -- means anyone can author, version, and publish extensions without touching proprietary APIs. And the privacy advantage is absolute: everything runs locally with SQLite storage, and the BYOK proxy supports any OpenAI-compatible endpoint with per-target SSRF protection. Whether you are a solo developer prototyping a landing page or a team migrating from Figma to code, Open Design gives you the tools without the lock-in.

## Links

- GitHub Repository: https://github.com/nexu-io/open-design