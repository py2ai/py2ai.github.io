---
layout: post
title: "x64dbg-mcp: An MCP Server Embedded Inside the Debugger - Inside duty1g/x64dbg-mcp-server"
description: "A source tour of duty1g/x64dbg-mcp-server, a Zig-native MCP plugin that embeds a full Model Context Protocol server inside the x64dbg debugger. Learn how its runtime SDK bridge, HTTP/SSE transports, and 80-tool registry let AI agents drive breakpoints, memory inspection, and PE analysis safely."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /X64dbg-MCP-Server-Native-Debugger-Plugin-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/x64dbg-mcp/duty1g-x64dbg-mcp-architecture.svg
tags:
  - MCP
  - x64dbg
  - Reverse Engineering
  - Zig
categories: [AI, Open Source]
keywords: "x64dbg MCP server, Zig plugin, Model Context Protocol, AI debugging agent, reverse engineering education, x64dbg plugin development, JSON-RPC debugger, MCP tools, Streamable HTTP SSE, debugger automation, binary analysis AI, x64bridge.dll, malware analysis training, duty1g"
author: "PyShine"
---

Most "AI controls a debugger" demos you see are glue: a Python script shelling out to a command line, polling text output, hoping nothing desynchronizes. `duty1g/x64dbg-mcp-server` takes a fundamentally different route — it is a plugin, written in Zig, that loads directly into x64dbg's address space and runs an MCP (Model Context Protocol) server on a thread inside the debugger itself. There is no wrapper process and no screen scraping; the AI agent's tool calls land a few function pointers away from the debugger core. For anyone studying how agentic systems are given safe, structured control over a complex native tool, this repository is a compact and unusually clean case study.

The project, known as x64dbg-MCP Server, speaks MCP 2024-11-05 over both Streamable HTTP and SSE transports with JSON-RPC 2.0 payloads. Its tool registry in `src/mcp/tools.zig` defines 80 tools spanning the whole x64dbg workflow: loading binaries, setting INT3/hardware/conditional/memory/exception breakpoints, stepping, dumping registers, reading and patching memory, scanning for byte patterns, walking import and export tables, and even reading PE headers from live memory to detect the original entry point of a packed binary. Twenty-two debugger event callbacks feed a ring buffer and a notification queue so the agent learns about breakpoint hits and exceptions the moment they happen.

The source is worth a tour because it answers, in about five Zig files, the questions that matter when you embed a network server in a GUI application: how do you bind to a closed-source host's API without import libraries, how do you serve HTTP from a thread inside someone else's process, and how do you design a tool surface that an LLM can use without corrupting the debug session? Everything below is read straight from the code — `src/main.zig`, `src/core/bridge.zig`, `src/core/mcp_server.zig`, `src/core/config.zig`, `src/mcp/tools.zig`, and `src/mcp/json.zig`.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/x64dbg-mcp/duty1g-x64dbg-mcp-overview-architecture.svg" alt="Architecture overview of the duty1g/x64dbg-mcp-server repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the x64dbg-mcp-server architecture: the Zig plugin lifecycle and event callbacks on the host side, the HTTP/JSON-RPC transport layer, the agent-facing tool registry, and the config/build scaffolding around them.*

Reading the overview from left to right: the plugin lifecycle node (`src/main.zig`) is the anchor — at load time it resolves the x64dbg SDK through the runtime bridge (`src/core/bridge.zig`), registers 22 event callbacks, builds its Plugins menu with icons from `src/resources/icons.zig`, and loads `mcp_config.json` via `src/core/config.zig` before auto-starting the HTTP server thread (`src/core/mcp_server.zig`). That server encodes JSON-RPC through the zero-allocation writer in `src/mcp/json.zig` and dispatches `tools/call` requests into the 80-tool registry in `src/mcp/tools.zig`, which is the only path through which an agent actually touches the debugger — every tool bottoms out in bridge calls. Around the core, `build.zig` cross-compiles both the 32-bit and 64-bit plugin DLLs, and `SKILL.md` documents the state-discipline workflow agents are expected to follow.

## Why You Need This

If you have ever tried to script x64dbg, you know the friction. The debugger is a GUI application, and its automation surface was designed for humans clicking menus — not for a program that wants to set a breakpoint, run, wait for the hit, dump registers, and decide the next step on its own. An external script also suffers a synchronization problem: it never really knows whether the target is running or paused, so it either polls blindly or races the debugger. This plugin dissolves both problems by living inside the process: tool handlers call `bridge.isDebugging()` and `bridge.isRunning()` before every sensitive operation, and every response ends with a `[state]` line (`NO_TARGET`, `RUNNING — call WaitForPause before inspecting`, or `PAUSED at 0x... (module) | disassembly`) so the agent's model of the world is refreshed on every single call.

The second problem is the event one. A debugging session is event-driven by nature — breakpoints hit, exceptions fire, DLLs load — but a request/response protocol like HTTP has no way to push that news to the client. The project solves it twice. Debugger callbacks registered in `src/main.zig` call `notifyEvent` in `src/core/mcp_server.zig`, which queues events for HTTP clients (surfaced as `[event:*]` lines on the next tool response and via the long-poll `WaitForEvent` tool, capped at 120 seconds) and pushes JSON-RPC `notifications/message` frames to SSE clients in real time. The `initialize` response even carries a twelve-rule instruction block telling agents to call `GetDebugState` first, to always pair `run` with `WaitForEvent`, and never to act on stale assumptions.

Third, there is the capability gap. An agent with only "run" and "read memory" can narrate a session, but it cannot do reverse engineering. This tool surface can: `FindPattern` scans module memory with `??` wildcards, `GetReferences` finds CALL/JMP xrefs, `DisassembleFunction` relies on x64dbg's analysis (the descriptions tell the agent to run `analr` first), `DetectOEP` walks the PE header chain at `base+0x3C` through `DbgMemRead` to locate a packed binary's original entry point, and `DumpModule` writes a reconstructed module to disk. That is the difference between a chatbot watching a debugger and an agent actually doing guided binary analysis — in an educational or malware-analysis-training context, precisely the repetitive baseline work a student needs demonstrated.

Finally, the plugin removes the deployment tax that kills most tooling adoption. Because it is Zig with zero dependencies — no .NET runtime, no Python interpreter, no DLL hell beyond the host's own `x64bridge.dll` — a single `zig build` cross-compiles both `x64dbg-MCP-Server.dp32` and `x64dbg-MCP-Server.dp64` from any host OS, and installing is a file copy into x64dbg's plugin folders. The server starts automatically with the debugger, so "AI-assisted session" becomes: launch x64dbg, point your MCP client at the port, go.

## How It Works

The whole system is a pipeline from x64dbg's plugin ABI down to a JSON-RPC handler, with the tool registry as the only bridge between network and debugger — here is the detailed view.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/x64dbg-mcp/duty1g-x64dbg-mcp-architecture.svg" alt="Detailed architecture of the duty1g/x64dbg-mcp-server repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of x64dbg-mcp-server: plugin lifecycle and callbacks feeding the event ring buffer, the bearer-authenticated HTTP listener with JSON-RPC dispatch and SSE stream, the ToolDef registry partitioned into breakpoint, memory, flow-control, and PE-analysis tool groups, all grounded in the runtime SDK bridge.*

### Understanding the Architecture

**The runtime bridge is the load-bearing trick.** x64dbg plugins are DLLs loaded into the debugger process, and the canonical way to use its SDK is linking against `x64bridge.lib`. Instead, `src/core/bridge.zig` calls `GetModuleHandleA` on `x64bridge.dll`/`x32bridge.dll` and `x64dbg.dll`/`x32dbg.dll` and resolves every symbol with `GetProcAddress` at `pluginit` time — mandatory functions panic if missing, while extended ones like `DbgDisasmFastAt`, `DbgGetBpList`, `DbgMemMap`, `DbgGetThreadList`, and `GuiGetDisassembly` resolve optionally. Because the code never names a bitness-specific import, one source tree builds both architectures, and the file even documents a subtle ABI detail: bridge functions return a C++ `bool` in the low byte of EAX, so on x86 the upper garbage bytes must be masked before testing.

**The plugin lifecycle wires everything at startup.** `src/main.zig` exports the three functions x64dbg's plugin SDK expects. `pluginit` resolves the bridge and registers the `StartMCPServer`/`StopMCPServer` debugger commands; `plugsetup` builds the Plugins menu (start/stop toggle, "Configure MCP Server...", About) with embedded PNG icons, registers the 22 event callbacks in one `inline for` over a comptime tuple, logs every tool in the registry to the x64dbg log pane, loads the saved config, and auto-starts the server unless `AutoStart` is false. The callbacks are deliberately thin: each formats a short message and both logs it and pushes it into a 64-slot event ring buffer, with important ones (breakpoint hit, exception with first/second-chance annotation, system breakpoint) also forwarded to MCP clients via `notifyEvent`.

**The HTTP server is hand-rolled Win32, not a framework.** `src/core/mcp_server.zig` spawns a background thread with `CreateThread`, initializes Winsock, binds with `SO_REUSEADDR` (default `127.0.0.1:9094` on x64, `9095` on x32, both configurable), and loops on `accept`, handing each connection to a fresh client thread. The handler parses the request line and headers with plain slice operations, answers CORS preflights, enforces the bearer token on everything (a mismatched `Authorization` header gets a 401 before any routing), then routes: `GET /sse` upgrades to an event stream that tracks up to 8 clients with five-second keepalives, `POST /messages` serves the legacy SSE reply channel keyed by session ID, and any other `POST` is treated as Streamable HTTP JSON-RPC. Four methods are implemented — `initialize`, `tools/list`, `tools/call`, and `ping` — with notifications acknowledged by `202 Accepted`.

**The tool registry is a data-driven contract with the agent.** In `src/mcp/tools.zig`, each of the 80 entries is a `ToolDef` struct: name, human-readable description, a `debug_only` flag, a `read_only` flag (exposed to clients as the MCP `readOnlyHint` annotation), a handler function, and a schema function that writes the tool's JSON Schema inline. That description field is doing real work — it tells the agent, for example, that `ExecuteDebuggerCommand` takes x64dbg-native commands only, never MCP tool names, and that `ReadMemory` caps at 4096 bytes per call. The dispatcher in `buildToolsCallResult` refuses `debug_only` tools when no session is active, writes results through the fixed-buffer `JsonWriter` from `src/mcp/json.zig` (which escapes strings and trims trailing commas without a single heap allocation), then drains queued `[event:*]` lines and appends the `[state]` snippet before returning.

**Configuration and trust are handled Win32-native.** `src/core/config.zig` builds the settings dialog from raw `CreateWindowExA` calls — bind IP, port, a read-only token field with Generate/Copy buttons, and an auto-start checkbox — and persists everything to `mcp_config.json` next to the x64dbg executable. On first run the token is generated with `SystemFunction036` (RtlGenRandom) as 32 hex characters. Saving the dialog applies the new settings immediately and restarts the server thread, so rotating a token never requires relaunching the debugger.

End to end, a breakpoint workflow looks like this: the agent's MCP client POSTs `tools/call SetBreakpoint` to port 9094; the listener thread authenticates the bearer token, `jsonrpc` dispatch finds the ToolDef, and the handler resolves the target through `DbgValFromString` and sets it via an x64dbg command; the client then calls `run`, which blocks until a breakpoint or exception fires (bounded by a configurable timeout up to ten minutes); the `CB_BREAKPOINT` callback fires inside the debugger, pushes an event, and the unblocking response carries the new `[state] PAUSED at 0x...` line plus the queued `[event:*]` entries; the agent now reads registers and memory knowing, not hoping, that the target is stopped.

## Advantages

- **True in-process integration.** The plugin resolves x64dbg's SDK at runtime and serves MCP from a thread inside the debugger, eliminating the desynchronization and polling fragility of out-of-process wrappers.
- **Complete, well-described tool surface.** Eighty registry entries in `src/mcp/tools.zig` cover breakpoints (including conditional, hardware, memory, and exception variants), stepping and tracing, memory read/write and patching, pattern scanning, xrefs, symbols, PE analysis, and module dumping — each with a description written to steer an LLM correctly.
- **Event-aware by design.** Twenty-two debugger callbacks feed both an SSE push channel and a polled queue, and every tool response carries `[state]` and `[event:*]` lines, so the agent's picture of the session is refreshed on every call.
- **Zero-dependency single binary.** Pure Zig against `kernel32`, `ws2_32`, and `user32`; `build.zig` cross-compiles both `.dp32` and `.dp64` plugins from any host OS with one command.
- **Dual-transport compatibility.** Streamable HTTP for modern MCP clients and SSE with a `/messages` reply channel for legacy ones, both speaking JSON-RPC 2.0 and protocol version 2024-11-05.
- **Deliberate guardrails.** Mandatory bearer-token auth, `debug_only` gating on session-dependent tools, `readOnlyHint` annotations, and a disclaimer steering use toward legitimate reverse engineering, security research, and education.

## Benefits

- **For learners of agentic architecture,** this is one of the clearest small codebases showing how an MCP server is embedded in a native host application — transport, dispatch, tool registry, and host bridge each isolated in one file.
- **For reverse engineering students,** an agent can shoulder the mechanical loop (break, dump, step, compare) while you focus on the interesting logic, turning training exercises into guided sessions rather than pointer-arithmetic tedium.
- **For malware analysts,**`FindPattern`, `GetStrings`, `GetReferences`, `DetectOEP`, and `DumpModule` automate the recon pass on packed or obfuscated samples in a lab environment you control.
- **For plugin developers,** the runtime `GetProcAddress` bridging in `src/core/bridge.zig` and the comptime callback-registration pattern in `src/main.zig` are reusable techniques for any closed-source host with a C API.
- **For tool builders,** the fixed-buffer `JsonWriter` and inline JSON-Schema functions demonstrate how to serve a structured protocol from an environment where you cannot afford heap allocation on every response.
- **For teams,** configuration is operational rather than surgical: bind address, port, and token live in one dialog and one `mcp_config.json`, with hot restart on save.

## Usage

Install from a release or build from source, then copy the deploy tree into your x64dbg root:

```console
zig build -Doptimize=ReleaseSafe --prefix dist
```

```text
dist/
├── x32/
│   └── plugins/
│       └── x64dbg-MCP-Server.dp32
└── x64/
    └── plugins/
        └── x64dbg-MCP-Server.dp64
```

Copy the contents of `dist/` into your x64dbg folder and launch x64dbg — the MCP server starts automatically on port 9094 (x64) or 9095 (x32). Point your MCP client at it (Streamable HTTP shown; use `http://localhost:9094/sse` for legacy SSE clients):

```json
{
  "mcpServers": {
    "x64dbg": {
      "type": "http",
      "url": "http://localhost:9094/",
      "headers": {
        "Authorization": "Bearer YOUR_TOKEN_HERE"
      }
    }
  }
}
```

The bearer token is auto-generated on first run and stored in `mcp_config.json`; view, copy, or rotate it under **Plugins > x64dbg-MCP Server > Configure MCP Server...**, which also controls the bind address (`0.0.0.0` for WSL or remote access, `127.0.0.1` for local-only) and auto-start. The same menu toggles the server, and the `StartMCPServer`/`StopMCPServer` commands work from x64dbg's command bar.

## Conclusion

x64dbg-mcp-server is a well-scoped piece of engineering: one Zig codebase that loads into a closed-source debugger, resolves its API without import libraries, serves a modern MCP protocol from a hand-rolled Win32 HTTP stack, and exposes a thoughtfully annotated 80-tool surface to AI agents — with state reporting and event push designed so an agent never acts on a stale picture of the session. The disclaimer is worth honoring: this is a tool for legitimate reverse engineering, security research, and education, it controls processes over plain HTTP, and it belongs on networks and targets you are authorized to touch. If you want to understand how agentic AI meets native tooling at the binary level, reading this source end to end is time well spent.

Links:

- GitHub repository: [https://github.com/duty1g/x64dbg-mcp-server](https://github.com/duty1g/x64dbg-mcp-server)
- Model Context Protocol: [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/)
- x64dbg: [https://x64dbg.com/](https://x64dbg.com/)
