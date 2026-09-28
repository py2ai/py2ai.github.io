---
layout: post
title: "Codex ChatGPT Web: Run Codex on Your ChatGPT Plan - Inside miuuyy/codex-chatgpt-web"
description: "A source tour of miuuyy/codex-chatgpt-web, a local Responses bridge that lets OpenAI Codex run tasks on ChatGPT Web models from your own account. We map the Electron launcher, the loopback Responses daemon, the Playwright browser workers, and the MCP tunnel harness that wire it all together."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /Codex-ChatGPT-Web-Bridge-ChatGPT-Web-Models-Into-Codex/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/codex-chatgpt-web/miuuyy-codex-chatgpt-web-architecture.svg
tags:
  - Codex
  - ChatGPT
  - Open Source
  - AI Coding
categories: [AI, Open Source]
keywords: "codex chatgpt web, codex-chatgpt-web, chatgpt web models, codex harness, responses api bridge, mcp connector, openai tunnel, browser automation, electron launcher, playwright automation, local ai proxy, miuuyy"
author: "PyShine"
---

If you pay for ChatGPT and you also use Codex, you have probably felt the odd split between the two: your ChatGPT plan includes strong models with their own usage allowances, yet Codex only talks to the models it ships with, metered against a separate quota. miuuyy/codex-chatgpt-web closes that gap in a clever, slightly audacious way. It is an unofficial desktop tool that presents the ChatGPT Web models available on your own account — including Pro-tier entries when your plan exposes them — directly inside Codex's native model picker, while keeping Codex's familiar interface, tasks, images, and streaming intact. No API keys for a second provider, no foreign chat UI: the models simply show up as additional rows ending in "(Web)".

Under the hood, the project is not a wrapper around some private endpoint. It is a focused local Responses bridge, as the package description puts it: a daemon that listens on loopback, speaks the Responses API that Codex already understands, and then fulfills each request by driving a real, authenticated ChatGPT web session through browser automation. Because the session is yours, the usage comes out of your ChatGPT plan's separate limits rather than your Work or Codex quota. And because everything runs locally — an Electron launcher, a Bun-powered daemon, Playwright-driven browser workers — there is no hosted middleman inspecting your prompts.

That combination makes the source unusually worth a tour. This is systems programming applied to an awkward problem: HTTP servers, SSE encoding, DOM automation, subprocess supervision, MCP tool transport, and transactional config editing, all cooperating to make one app believe another app is a model provider. The repository is candid about what it does — the README and docs/architecture.md document the design in remarkable detail — and the code backs every claim with a concrete module. Let's walk through it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/codex-chatgpt-web/miuuyy-codex-chatgpt-web-overview-architecture.svg" alt="Architecture overview of the miuuyy/codex-chatgpt-web repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the codex-chatgpt-web architecture: the Electron launcher supervises a loopback Responses daemon, which drives authenticated ChatGPT browser sessions and, in full mode, an MCP harness reached through an outbound OpenAI tunnel.*

Reading the overview from left to right: the Electron launcher (launcher/electron/main.cjs) hosts the React control center (launcher/src/App.tsx) and owns the pool of up to five task-bound browser tabs (launcher/electron/browser-host.cjs). It spawns the bridge runtime whose entry point is the CLI (src/cli.ts); setup (src/codex-integration.ts) points Codex's Responses route at the daemon's loopback listener (src/server.ts). Incoming turns stream out through the Responses SSE bridge (src/bridge.ts) and are dispatched to the ChatGPT Web adapter (src/adapters/chatgpt-web/index.ts), which drives the ChatGPT UI via the Playwright worker (src/adapters/chatgpt-web/browser-worker.ts) and publishes tool rounds through the turn broker (src/adapters/chatgpt-web/turn-broker.ts). In full harness mode, the stdio MCP server (src/adapters/chatgpt-web/mcp-server.ts) exposes local Codex tools to ChatGPT through an outbound tunnel connection (src/tunnel.ts), so nothing listens on a public port.

## Why You Need This

The first problem this solves is economic. Codex usage and ChatGPT usage are metered separately, and heavy Codex sessions can burn through quota long before your ChatGPT allowance is touched. By routing Codex turns through your authenticated ChatGPT web session, codex-chatgpt-web lets you spend the plan you already pay for. Pro accounts get separate Pro model rows, and the launcher detects which models your account can actually use instead of promising a fixed list.

The second problem is continuity of workflow. Alternative approaches usually mean leaving Codex for a browser tab: you copy context, paste it into ChatGPT, lose tool access, and lose the task binding. Here, the conversation stays tied to your Codex task. Images and full context travel with the request, streamed output comes back through the normal Codex interface, and compaction — the process of summarizing a long context into a new epoch — is handled natively rather than left to you.

The third problem is tool access, and this is where the project goes beyond a copy-paste bridge. In full harness mode, ChatGPT's tool calls are connected back to your current Codex task: files, terminal, and approvals all live on your machine, and the MCP plumbing in src/adapters/chatgpt-web/mcp-server.ts plus the outbound tunnel in src/tunnel.ts makes them reachable from the ChatGPT conversation without exposing a public IP or opening inbound ports. A separate Zero Risk mode exists for cautious setups: it never reads or operates the ChatGPT page at all — you paste and send the prepared prompt yourself, and local tools still work through a dedicated connector.

Finally, it solves the trust problem explicitly rather than by marketing. The docs are blunt that this is unofficial browser automation, not an OpenAI API, and that UI changes on ChatGPT can break selectors — so the code is engineered to fail explicitly, with drift detected instead of silently switching model or transport. Security invariants are written down in docs/security-model.md: loopback-only binding, browser state stored with restrictive file modes, bearer-token-protected lifecycle endpoints, and a five-tab cap to avoid tripping account abuse controls.

## How It Works

At its core, the system is a protocol translator sandwiched between Codex and a live ChatGPT web page, with a supervisor keeping both sides healthy.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/codex-chatgpt-web/miuuyy-codex-chatgpt-web-architecture.svg" alt="Detailed architecture of the miuuyy/codex-chatgpt-web repository" style="max-width:100%;height:auto;" />
</div>

*Detailed component graph of codex-chatgpt-web: launcher internals, the Responses daemon and its protocol layer, model catalog and passthrough, the browser adapter stack, the MCP/tunnel harness, and the Codex integration and operations tooling.*

### Understanding the Architecture

**The launcher is the sole supervisor.** The Electron main process (launcher/electron/main.cjs) owns the React renderer (launcher/src/App.tsx), a local control server (launcher/electron/control-server.cjs), the browser tab host (launcher/electron/browser-host.cjs), and the ChatGPT sign-in session (launcher/electron/chatgpt-auth-session.cjs). Before anything runs, launcher/electron/runtime-install.cjs verifies the embedded runtime — a pinned Bun executable plus the bridge, MCP server, and browser helper — against a deterministic manifest of paths, sizes, and SHA-256 hashes, and only then accepts the private versioned directory it will actually execute. launcher/electron/runtime-supervisor.cjs then starts the optional tunnel first, waits for healthy evidence, starts the Responses daemon, and waits for its versioned health payload.

**The daemon speaks fluent Responses.** src/cli.ts is the single binary entry: it parses subcommands like setup, login, doctor, serve, mcp, service, and tunnel, loads configuration through src/config.ts, and starts the loopback HTTP listener in src/server.ts (the default Responses port is 17841). Requests are parsed by src/responses/parser.ts, prior-turn state is expanded from src/responses/state.ts, and src/bridge.ts re-encodes adapter events into the exact Responses SSE event stream Codex expects — reasoning envelopes, usage accounting with cached-token details, and error payloads classified into proper HTTP statuses. There is even a deliberate HTTP 426 response to Codex's WebSocket prewarm, which nudges Codex into its HTTP/SSE transport without any model fallback.

**The adapter turns protocol into UI actions.** src/adapters/chatgpt-web/index.ts assembles the ChatGPT Web adapter, which delegates to src/adapters/chatgpt-web/browser-worker.ts — a Playwright-core driver that signs in on the launcher's persistent Electron partition, compiles the Codex context through src/adapters/chatgpt-web/prompt.ts into an inline JSON envelope with images attached natively, submits it to the ChatGPT composer, and reads the streamed answer back. Binding is done on ChatGPT's logical data-turn-id rather than a display index, so virtualized-list rerenders cannot be mistaken for new turns. The incoming markdown is normalized by src/adapters/chatgpt-web/markdown.ts (with a GFM plugin), token budgets are estimated with the GPT-5 tokenizer, and src/adapters/chatgpt-web/turn-execution.ts tracks each session so cancellation, retries, and stall timeouts behave predictably.

**Models and catalogs are honest about limits.** src/model-catalog.ts augments the authenticated native model catalog with the chatgpt-web/ namespace rows defined in src/chatgpt-web-models.ts — Luna/Think for accounts without a reasoning selector, Instant through High for reasoning-capable ones, plus separate Pro rows when available. The advertised context window and a compaction reserve come from real account inspection, not guesswork, and the catalog generation rejects effort groupings whose budgets disagree. Native authenticated endpoints (Search, Image Gen) are forwarded transparently through src/native-passthrough.ts so routed sessions do not lose capabilities.

**The harness connects ChatGPT to your machine — carefully.** In full mode, src/adapters/chatgpt-web/mcp-main.ts runs the stdio MCP server (src/adapters/chatgpt-web/mcp-server.ts) that the ChatGPT custom connector reaches through the official OpenAI tunnel client, downloaded, version-pinned, and SHA-256-verified by src/tunnel.ts. Every connector call presents one turn-bound capability; the turn broker in src/adapters/chatgpt-web/turn-broker.ts keeps that binding private and dispatches actions to the live Codex task, while unexpected approval prompts fail closed unless you explicitly opt into per-call "Allow once" clicks. Compaction across epochs is orchestrated by src/adapters/chatgpt-web/compaction-handoff.ts together with the codec in src/responses/compaction.ts, so a long task continues in a fresh browser chat without losing history.

**Integration is transactional, not destructive.** src/setup.ts routes Codex to the daemon by editing the built-in provider's base URL through src/codex-integration.ts, and every changed line is recorded in src/codex-integration-journal.ts so disconnect or uninstall restores your original configuration byte-for-byte. src/doctor.ts provides end-to-end health checks, and src/service.ts manages the daemon lifecycle with an authenticated drain contract: lifecycle operations only proceed when active HTTP requests and active browser sessions both reach zero.

Following one request end to end: Codex posts a Responses request to the loopback daemon on port 17841; src/server.ts parses it via src/responses/parser.ts and expands remembered state, then hands the turn to the ChatGPT Web adapter. The adapter compiles the full context envelope, leases one task-bound tab from the launcher's browser host, and the Playwright worker submits it to the ChatGPT composer on your authenticated session. Streamed answer fragments flow back through the markdown normalizer into src/bridge.ts, which emits Responses SSE events that Codex renders as native reasoning, commentary, and final output. If ChatGPT decides to call a tool mid-answer, the call travels down the outbound tunnel to the MCP server, the turn broker binds it to the current turn, Codex executes it locally, and the result re-enters the same ChatGPT response — tools, browser, and protocol stitched into one turn.

## Advantages

- **Your ChatGPT plan, inside Codex.** Web models appear as native rows in Codex's model picker, with usage drawn from your ChatGPT allowance instead of your Codex or Work quota.
- **Full harness tool access.** ChatGPT can operate your current task's files, terminal, and tools through MCP, with results and tool calls staying inside one ChatGPT response.
- **Outbound-only connectivity.** The tunnel is outbound; no public IP, no inbound port, no router forwarding — a rarity for "let a cloud model touch my machine" setups.
- **Explicit failure over silent drift.** UI changes, missing connectors, or ambiguous turn identities fail loudly with specific errors rather than quietly switching models or retrying forever.
- **Transactional configuration.** Every Codex config edit is journaled and restored byte-for-byte on disconnect or uninstall, so trying the tool is reversible.
- **Hardened local runtime.** The embedded runtime is verified against a SHA-256 manifest before launch, and lifecycle endpoints require an application-owned bearer token.

## Benefits

- **Lower effective cost per task.** Codex sessions draw on ChatGPT limits you already pay for, which matters for long-running agentic work.
- **No workflow switch.** Tasks, images, streaming, and compaction remain native Codex experiences — there is no second UI to babysit.
- **Pro-tier headroom.** Accounts with Pro access get dedicated Pro model entries, keeping their context budgets separate from lower-effort rows.
- **Safer automation defaults.** Temporary Chat by default, a five-tab cap, fail-closed approvals, and a Zero Risk mode that never reads the ChatGPT page give you graded exposure.
- **Honest diagnostics.** The doctor command, browser smoke tests, and an opt-in local limits estimate make it clear what is healthy and what is being counted.
- **Cross-platform packaging.** The launcher ships for macOS (arm64/x64), Windows x64, and Linux x64, with its own browser and runtime — no separate Chrome, Node, or Bun installation needed.

## Usage

Install the launcher with the prebuilt installer for your platform, or use the terminal installer. On macOS and Linux:

```bash
curl -fsSL https://github.com/miuuyy/codex-chatgpt-web/releases/latest/download/install-launcher.sh | sh
```

On Windows PowerShell:

```powershell
irm https://github.com/miuuyy/codex-chatgpt-web/releases/latest/download/install-launcher.ps1 | iex
```

Then sign in to ChatGPT in the embedded browser, install the models, and restart Codex once — models ending in "(Web)" appear in the picker. For coding with tools, complete the Full harness setup on the launcher's MCP page (create the tunnel and API key, enable ChatGPT Developer Mode, create the Codex Native2 connector, then press Connect harness).

If you prefer to run from source, the README documents this path with Bun 1.4.0:

```bash
git clone https://github.com/miuuyy/codex-chatgpt-web.git && \
cd codex-chatgpt-web && \
bun run app
```

Useful development and maintenance commands from the same README and package scripts:

```bash
bun run dev:launcher          # isolated dev profile under ~/.codex-chatgpt-web-dev
bun run src/cli.ts dev status
bun run verify                # full verification suite
bun run smoke:subagents       # subagent protocol smoke tests (V1 + V2)
bun run app:package           # package the desktop launcher
```

Subagent protocol selection is explicit and can be switched from the terminal:

```bash
codex-chatgpt-web subagents status
codex-chatgpt-web subagents compatibility-v1
codex-chatgpt-web subagents native
```

## Conclusion

codex-chatgpt-web is a rare kind of open-source project: an unofficial integration that treats honesty as a feature. It does exactly one thing — make the ChatGPT Web models on your own account usable from Codex's native harness — and it does it with a layered architecture (supervising launcher, loopback Responses daemon, browser automation, MCP tunnel harness) that fails loudly, restores cleanly, and documents every invariant it enforces. If you live in Codex and pay for ChatGPT, reading this codebase is worthwhile even before you run it; if you do run it, start with browser-only mode, read the security model before enabling full harness, and keep it on a trusted workstation with your own account.

Links:

- GitHub repository: [miuuyy/codex-chatgpt-web](https://github.com/miuuyy/codex-chatgpt-web)
- Architecture documentation: [docs/architecture.md](https://github.com/miuuyy/codex-chatgpt-web/blob/main/docs/architecture.md)
- Security model: [docs/security-model.md](https://github.com/miuuyy/codex-chatgpt-web/blob/main/docs/security-model.md)
- Troubleshooting and video walkthroughs: [TROUBLESHOOTING.md](https://github.com/miuuyy/codex-chatgpt-web/blob/main/TROUBLESHOOTING.md)
