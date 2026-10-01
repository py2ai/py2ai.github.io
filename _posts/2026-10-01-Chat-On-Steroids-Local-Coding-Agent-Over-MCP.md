---
layout: post
title: "Chat On Steroids: Turning ChatGPT Into A Local Coding Agent Over MCP - Inside totec448-spec/chat-on-steroids"
description: "Chat On Steroids is an open-source Electron workspace that gives ChatGPT hands on your real projects: an MCP server exposes files, shells, terminals, and desktop control, a Chrome companion extension bridges the chat page, and worker teams plus a Goal loop keep long tasks moving. We tour the source to see how the pieces fit."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /Chat-On-Steroids-Local-Coding-Agent-Over-MCP/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/chat-on-steroids/totec448-spec-chat-on-steroids-architecture.svg
tags:
  - MCP
  - ChatGPT
  - Coding Agent
  - Electron
categories: [AI, Open Source]
keywords: "Chat On Steroids, ChatGPT coding agent, MCP server, Model Context Protocol, Electron, TypeScript, local tools, worker agents, Chrome extension, desktop automation, open source"
author: "PyShine"
---

ChatGPT can write impressive code in its chat window, but the code never touches your machine unless you copy it there yourself. Codex-style agentic coding fixed that for people with the right subscription and CLI, and it left an obvious question hanging: why should the chat you already pay for not be able to read your files, run your tests, and keep your terminals open? Chat On Steroids is a community answer to exactly that question, built as a desktop workspace that wires the ChatGPT you already use to a carefully fenced set of local tools.

[Chat On Steroids](https://github.com/totec448-spec/chat-on-steroids) (CoS) is an MIT-licensed Electron application, currently at version 2.1.22, that turns ChatGPT into a local coding agent over the Model Context Protocol. A local MCP server exposes approved capabilities — files, shell, terminals, desktop control, browser tabs, plugins — and tunnel adapters make that server reachable from ChatGPT's cloud. A companion Chrome extension bridges the conversation page itself: it watches the chat, injects the app's tool results into the thread, and sends the model's replies back. Around this core, the app adds worker teams, an unattended Goal loop, and a Compact and Resume mechanism that carries a session into a fresh chat.

The source is worth a tour because it solves the unglamorous half of the agent problem with unusual care. Anyone can call a model API; the hard parts are the capability boundaries (which folders, which commands, which confirmations), the session machinery (who owns a conversation, what happens when a worker stops, how history survives a compaction), and the bridge to a web page that was never meant to be automated. The repository documents all three in its module headers, and the code matches the commentary.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/chat-on-steroids/totec448-spec-chat-on-steroids-overview-architecture.svg" alt="Architecture overview of the totec448-spec/chat-on-steroids repository" style="max-width:100%;height:auto;" />
</div>

*High-level architecture overview of the totec448-spec/chat-on-steroids repository, from the Electron app and the ChatGPT bridge to the tool layer and work management.*

Reading the overview from left to right: the Electron main process owns the window, configuration, and secrets, while the workspace UI drives everything through IPC; the companion Chrome extension pairs with the connection module, which manages tunnel adapters that expose the local MCP server; inside the server, a shared tool machinery registers core file-and-shell tools and desktop tools; and the work management group — worker families, the Goal loop, and the session store — keeps multi-step jobs coherent across turns.

## Why You Need This

If you have ever pasted a stack trace into a chat, copied the suggested fix back into your editor, and re-run the tests by hand, you already know the tax that Chat On Steroids removes. With the workspace connected, you describe the task in the chat you already have open, and the model reads the actual project, edits actual files, runs the actual test suite, and reports real output — not a reconstruction of what it guesses your project looks like. The gap between "the model's idea of your code" and "your code" is where most AI-assisted fixes quietly go wrong.

The second reason is delegation. Real work is rarely one prompt: refactor the module, then update the tests, then regenerate the fixtures. CoS gives the prime conversation a team — independent worker families, each with its own conversation and its own context, governed by admission limits rather than eviction, so a worker that proves it was still running stays recognized. When a family's last worker stops, its full history parks under its prime conversation and wakes intact later. That is a durability model, not a demo.

The third reason is unattended progress. Long tasks outlive a single turn: the model finishes, but two of the three things you asked for are still open. The Goal loop addresses this with a second model standing in for you — given the recorded conversation and a strict continuation gate, it decides whether the requested work is clearly finished or the chat should keep moving, and it drafts the next nudge. Your approval, your folders, and your usage remain yours; the README is explicit that the tool organizes work and does not grant extra quota or override provider rules.

## How It Works

The application is an Electron app whose main process hosts every capability, a renderer workspace UI, and a Chrome extension that bridges the ChatGPT page, with the MCP server as the model's front door.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/chat-on-steroids/totec448-spec-chat-on-steroids-architecture.svg" alt="Detailed architecture of the totec448-spec/chat-on-steroids repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the totec448-spec/chat-on-steroids repository, tracing the bridge, the MCP tool layer, capability modules, and session machinery.*

### Understanding the Architecture

**The bridge is a paired extension, not a scraper.** The extension's manifest (extension/manifest.json) registers a content script and a MAIN-world relay on chatgpt.com plus a service worker; src/main/bridge.ts holds the app side, and src/main/connection.ts tracks pairing and status. Content flows through extension/chatgpt-dom.js, which reads the page structurally, and extension/fiber.js, which relays events from the page's main world. The app records the authoritative conversation locally in src/main/session/recorder.ts, so features build on a stable transcript rather than a scrolled, virtualized DOM.

**Reaching a loopback server from the cloud takes adapters.** The MCP server in src/main/mcp/server.ts knows nothing about networking beyond a loopback URL; the adapters in src/main/tunnel/index.ts make that URL reachable — one for OpenAI's Secure MCP Tunnel, which is outbound-only, and one for a generic cloudflared quick tunnel whose secret path token is the privacy boundary. Adding a third provider means adding one function, not touching the tools.

**Tools share one kernel and differ only by connector.** src/main/mcp/kernel.ts implements the machinery every tool sits on — error mapping, call timing, recording context, and result formatting — while the tool families in tools-core.ts, tools-desktop.ts, tools-browser.ts, and tools-plugins.ts split along connector boundaries. Exposure is monotonic per endpoint: a revoked permission makes the live handler return a disabled signal rather than unregistering, so a cached tool snapshot on ChatGPT's side never breaks. Read-only tools carry explicit annotations, because ChatGPT treats an unannotated tool as a write and asks the user to confirm every call.

**File and shell access is fenced twice.** Path policy lives in src/main/sandbox.ts, which resolves every virtual path against the approved workspace roots, and execution lives in src/main/exec.ts. Longer sessions go through the unified exec manager in src/main/codex/manager.ts, which keeps persistent terminals via src/main/workspace-terminal.ts and edits files with the apply-patch engine in src/main/codex/apply-patch/ — a streaming patch format with its own parser and error taxonomy. Desktop control in src/main/computer/ captures screens and sends keys through native APIs.

**Work management is where the agents live.** src/main/agents.ts implements the worker families: each active incarnation has its own admission limit, spawn and finish plans cross a durable snapshot barrier before publication, and identity is resolved by conversation, never by local names. src/main/goal.ts implements the Goal loop — the continuation credential stays in the main process next to the other secrets, and the page receives a validated draft reply rather than an API key. src/main/session/handoff.ts is the Compact and Resume path that carries the session and worker history into a fresh chat.

**Extensibility is a first-class surface.** The plugin manager in src/main/plugins/manager.ts installs and runs connector plugins (with an OAuth flow and a bundled uv runtime for Python-based servers), and src/main/skills.ts manages the skill library the model can pull in. Even the code-mode tool — src/main/mcp/code-mode-tool.ts, which runs model-authored JavaScript against the tool surface inside a QuickJS sandbox — is just another capability with the same kernel underneath.

End to end: you write a task in the workspace UI, the extension delivers it to the ChatGPT page, the model calls tools through the tunnel-backed MCP server, the kernel routes each call to the right connector, results stream back into the recorded conversation, and workers or the Goal loop keep the job moving until you close it out.

## Advantages

- **Your existing plan becomes an agent plan.** The workspace drives the ChatGPT conversation you already have rather than requiring a separate agentic product, and it says so plainly in the README.
- **Capability boundaries are enforced in code.** Approved folders, per-capability tools, monotonic exposure, and a path sandbox mean the model's reach is a configured fact, not a prompt suggestion.
- **Workers are durable, not disposable.** Admission limits, parked histories under the prime conversation, and durable snapshots before publication mean a team of workers survives restarts and admissions changes.
- **Read-only tools stay read-only.** Explicit annotations stop ChatGPT from demanding confirmation for every harmless query, which is the difference between an agent and a permission-clicking simulator.
- **Two tunnel options out of the box.** OpenAI's outbound-only secure tunnel for supported accounts, and a cloudflared quick tunnel with a secret path for everyone else.
- **Genuinely cross-platform.** Windows, macOS, and Linux builds ship from the same TypeScript tree, with native desktop capture handled per platform.

## Benefits

- **A working MCP reference implementation.** The kernel/connector split, the monotonic exposure rule, and the inbound request handling are directly reusable patterns for anyone building their own MCP server.
- **Session machinery you can learn from.** Recorder, resume gate, correlation, blocked chats, usage accounting — the session folder is a compact course in building stateful agent UIs on top of a stateless chat page.
- **Unattended runs without lost context.** The Goal loop and Compact and Resume together mean a long brief can progress while you are away and land in a fresh chat with its history intact.
- **Local-first secrets.** Keys live in the OS-protected vault inside the main process; the extension receives replies, never credentials.
- **Extensible in three directions.** Plugins for new connectors, skills for reusable knowledge, and code mode for model-driven tool batching — each isolated from the core.
- **Multilingual and inspectable.** Nine UI languages ship in the renderer, and the MIT license plus thorough module documentation make the whole system auditable.

## Usage

Install a signed-in ChatGPT-capable browser and grab the desktop build for your platform from the releases page (Windows x64 installer, macOS Apple-silicon DMG, or Linux DEB):

- Windows: https://github.com/totec448-spec/chat-on-steroids/releases/latest/download/Chat-On-Steroids-Setup-x64.exe
- macOS (Apple silicon): https://github.com/totec448-spec/chat-on-steroids/releases/latest/download/Chat-On-Steroids-macOS-arm64.dmg
- Linux: https://github.com/totec448-spec/chat-on-steroids/releases/latest/download/Chat-On-Steroids-Linux-x64.deb

Then follow the README's four steps: approve your project folder under Settings, Workspace; connect Core under Settings, Setup and register it in ChatGPT as a custom MCP app; load the companion extension via Chrome's Load unpacked with the folder the app opens for you; pick a model and send your first task. Pairing with the extension is automatic.

To build and verify the source yourself, the repository is a standard Node project:

```bash
git clone https://github.com/totec448-spec/chat-on-steroids.git
cd chat-on-steroids
npm install
npm run dev
```

The package.json scripts mirror the project's own CI discipline:

```bash
npm run typecheck   # strict TypeScript across main, renderer, shared
npm test            # vitest suites
npm run verify      # ripgrep fetch, privacy and notices checks, typecheck, tests
npm run dist        # packaged builds for all three platforms
```

The project requires Chrome 125 or newer for the companion extension, a ChatGPT account whose workspace allows custom MCP apps, and — per the README's responsible-use notice — a commitment to stay within your provider's terms rather than treat the tooling as a way around them.

## Conclusion

Chat On Steroids reads like the codebase of a team that has thought carefully about where agent systems actually fail: not in the model, but in the seams — the page that must not be misread, the shell that must not run wild, the worker that must not lose its history. It turns a chat subscription into a capable local coding agent, fences every capability behind explicit code, and documents its own reasoning in the source. If you are building anything MCP-shaped, or you just want your ChatGPT tab to finally touch your filesystem, this repository deserves an afternoon of your attention.

Links:

- GitHub repository: https://github.com/totec448-spec/chat-on-steroids
- Releases: https://github.com/totec448-spec/chat-on-steroids/releases/latest
- Setup documentation: https://github.com/totec448-spec/chat-on-steroids/blob/main/docs/setup.md
- Security notes: https://github.com/totec448-spec/chat-on-steroids/blob/main/SECURITY.md
