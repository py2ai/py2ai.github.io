---
layout: post
title: "TradingView MCP: Give Your AI Assistant Eyes and Hands on Your Trading Charts"
description: "TradingView MCP connects Claude to your locally running TradingView Desktop over the Chrome DevTools Protocol for Pine Script development, chart navigation, and workflow automation - all on your machine."
date: 2026-09-19
header-img: "img/post-bg.jpg"
permalink: /TradingView-MCP-AI-Assistant-For-Your-Charts/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/tradingview-mcp/tvmcp-architecture.svg
tags:
  - MCP
  - TradingView
  - AI Agents
  - Pine Script
  - Open Source
author: "PyShine"
---

If you have ever watched an AI write flawless code but fumble in the dark the moment the task involves your trading chart, you already understand the gap this project closes. [TradingView MCP](https://github.com/tradesdontlie/tradingview-mcp) is an open-source bridge, MIT licensed, that connects an AI assistant such as Claude Code to your locally running TradingView Desktop application. Once wired up, the model can change symbols and timeframes, read indicator values, write and debug Pine Script, draw support levels, manage alerts, and even step through historical bars in replay mode — by calling tools instead of you clicking around.

The design principle that makes it worth studying: it never talks to TradingView's servers. It speaks the Chrome DevTools Protocol to the Electron app already running on your machine, on localhost, and every byte of data stays local. The overview below shows the whole system in one glance: the AI on one side, the bridge in the middle, your own TradingView instance on the other.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/tradingview-mcp/tvmcp-overview-architecture.svg" alt="TradingView MCP high-level overview architecture" style="min-width:900px;width:100%;">
</div>

*High-level overview: the MCP server and CLI share one core that drives TradingView Desktop over a local debugging port.*

## Why You Need This

Chart work is repetitive in ways machines are good at. Ask yourself how much time you spend re-drawing session levels every morning, switching between twenty tickers to check one indicator on each, or rebuilding a Pine Script after a one-character compile error. Common experience shows that automating even half of that loop returns hours per week to actual analysis.

You need this tool specifically if you live in three workflows. First, Pine Script development: instead of copy-pasting code into the editor and squinting at error messages, your assistant injects the script, compiles it, reads the exact errors, and iterates until it compiles clean. Second, multi-chart workflows: set up a 2x2 grid with different symbols per pane in a single sentence. Third, research and journaling: screenshots, indicator tables, and price levels can be captured as JSON for your own tooling, since every MCP tool is also a plain `tv` CLI command with JSON output that pipes into `jq`.

The honesty matters too. This is an interface layer, not a trading bot: it executes no real trades, and the project is explicit that it requires a valid TradingView subscription and that you remain responsible for complying with [TradingView's policies](https://www.tradingview.com/policies/).

## How It Works

The repository is a compact Node.js codebase with only two dependencies — the official MCP SDK and a Chrome remote-interface client — and the detailed diagram below maps every layer of it.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/tradingview-mcp/tvmcp-architecture.svg" alt="TradingView MCP detailed architecture" style="min-width:1100px;width:100%;">
</div>

*Detailed architecture: from MCP clients and the CLI, through the core dispatcher and domain modules, to the debug port on TradingView Desktop.*

On the surface layer, `src/server.js` runs the MCP server over stdio and registers more than 75 tools defined in `src/tools/` — chart control, data reading, Pine development, replay, drawing, alerts, tabs, and UI automation. The same core powers `src/cli/index.js`, where a small router maps the familiar `tv status`, `tv quote`, or `tv pine compile` verbs onto identical functions, so humans in a terminal and agents over MCP share one tested code path.

The core layer in `src/core/` is where the real domain logic lives: chart navigation, OHLCV and study extraction, Pine Script injection and compilation, bar-by-bar replay practice, a poll-and-diff streaming module that emits JSONL, and UI automation for clicks and keyboard input. All of it converges on `src/connection.js`, a thin wrapper that opens the Chrome DevTools Protocol socket to your TradingView Desktop on localhost port 9222 — a debugging interface you must deliberately enable with a standard Chromium flag, or via the per-platform launch scripts in `scripts/`.

One more layer deserves attention: the agent guidance. `CLAUDE.md` ships a complete decision tree mapping natural language requests to exact tool sequences, and the `skills/` directory plus a specialized performance-analyst agent teach the model how to chain tools into full chart-analysis workflows. Context management is engineered throughout: outputs are deduplicated and compact by default, so a full chart analysis costs a few kilobytes of context instead of eighty.

## Advantages

Against raw screenshot-pasting approaches, the advantage is structure. The model reads typed data — quote objects, indicator values, Pine drawings as deduplicated price levels — rather than interpreting pixels, which is why the tool distinguishes so sharply between reading tables, labels, lines, and boxes.

Against browser automation frameworks, the advantage is the deliberate scope. Everything runs through one standard debugging interface on your own machine, with no server contact, no data storage, and no trade execution. That narrow surface is exactly why it is auditable: with two dependencies and small modules, you can review the whole thing in an afternoon.

The dual surface is an engineering win as well. Because the CLI and the MCP tools share the core, the 29 tests that exercise the code path benefit both interfaces, and scripts you build on `tv stream` today keep working as the agent layer evolves.

## Benefits

The immediate benefit is speed in Pine Script development — compile-check-fix cycles that took minutes of manual editing collapse into a single conversational turn. The second is consistency: drawing the same session levels across charts, or configuring an indicator identically on a dozen symbols, stops depending on your memory or discipline.

The third is privacy, which is rare in this category. Your chart data, symbol choices, and strategy experiments never leave your machine; the assistant sees your chart because it is allowed to, through a port you enabled, on a loopback connection. For traders handling proprietary strategies, that local-only guarantee is the difference between adopting AI assistance and forbidding it.

## Usage

Clone, install, and launch TradingView with the debug port:

```shell
git clone https://github.com/tradesdontlie/tradingview-mcp.git
cd tradingview-mcp
npm install
./scripts/launch_tv_debug_mac.sh   # or launch_tv_debug.bat / _linux.sh
```

Register the server with Claude Code in your MCP config (`~/.claude/.mcp.json`):

```json
{
  "mcpServers": {
    "tradingview": {
      "command": "node",
      "args": ["/path/to/tradingview-mcp/src/server.js"]
    }
  }
}
```

Verify with a health check, then work in natural language: ask for the current chart state, request a Pine Script for a session-levels indicator, or tell it to start replay at a specific date. From a terminal, the CLI offers the same power — `tv quote`, `tv pine compile`, `tv pane layout 2x2`, `tv stream quote | jq '.close'` for continuous monitoring. Requirements are modest: Node.js 18 or newer, TradingView Desktop with a subscription, and any of macOS, Windows, or Linux.

## Conclusion

TradingView MCP is a well-scoped piece of the new tooling wave around the Model Context Protocol: small, dependency-light, and honest about what it does and does not do. It turns your trading application into something an AI can see and act on, while keeping every byte on your own machine. If you write Pine Script or manage multi-chart workflows, the repository deserves a spot next to your editor.

Links:

- Repository: [github.com/tradesdontlie/tradingview-mcp](https://github.com/tradesdontlie/tradingview-mcp)
- TradingView: [tradingview.com](https://www.tradingview.com/)
- TradingView policies: [tradingview.com/policies](https://www.tradingview.com/policies/)
- Model Context Protocol: [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)
