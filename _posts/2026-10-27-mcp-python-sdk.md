---
layout: post
title: "Inside the MCP Python SDK: Sessions, Transports, and the Machinery Under Every MCP Server"
description: "A source tour of the official Model Context Protocol Python SDK: ClientSession, the shared JSON-RPC dispatcher, Streamable HTTP and stdio transports, and the MCPServer facade."
date: 2026-10-27
header-img: "img/post-bg.jpg"
permalink: /mcp-python-sdk/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/mcp-python-sdk/modelcontextprotocol-mcp-python-sdk-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: MCP, Model Context Protocol, Python SDK, ClientSession, MCPServer, FastMCP, JSON-RPC, Streamable HTTP, stdio transport, tool calling, LLM agents
author: "PyShine"
---

Every MCP server you have ever pointed an agent at is, underneath the marketing, a JSON-RPC peer. The [Model Context Protocol](https://modelcontextprotocol.io) defines the methods, the notification etiquette, and the capability negotiation; the Python SDK in the `modelcontextprotocol/python-sdk` repository is the most-deployed implementation of both halves of that conversation. When you write a twelve-line tool server or a production agent backend, this package is the machinery doing the talking. That makes it a rare artifact: a protocol reference implementation that real products are built on.

That underneath is worth a tour, because in 2026 the SDK quietly rebuilt itself. The repository now ships as a small workspace with two packages: `mcp`, the runtime, and `mcp-types`, a generated home for the Pydantic wire models that keeps protocol schemas versioned separately from session logic. The old `FastMCP` high-level server was renamed to `MCPServer` and moved to its own subpackage; the old `BaseSession` receive loop was replaced by a shared `JSONRPCDispatcher`; and a new per-request "modern" envelope arrived alongside the classic `initialize` handshake. If your mental model of this SDK comes from last year's tutorials, the map has changed.

This post walks the current `main` branch the way a contributor would: client facades, the shared dispatcher core, the two transports, and the three-layer server stack. All symbols and line numbers below were verified against the tree at commit `a525a3d9bece`. The two diagrams were generated directly from the source layout, so they double as a reading order.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/mcp-python-sdk/modelcontextprotocol-mcp-python-sdk-overview-architecture.svg" alt="Overview architecture of the MCP Python SDK showing client, transports, shared dispatcher core, server stack, and wire types" style="min-width:900px;"></div>

<p><em>The SDK at a glance: client and server stacks meet in a shared dispatcher core, while the mcp-types package holds the wire models both sides speak.</em></p>

Reading the overview from left to right:

- **`Client` is the front door.** The [Client, L268](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/client.py#L268) dataclass accepts a URL string, `StdioServerParameters`, a raw transport, or even an in-process `MCPServer` instance, and builds the whole stack for you inside an async context manager.
- **`stdio_client` launches a subprocess.** The [stdio_client, L115](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/stdio.py#L115) context manager spawns the server command described by [StdioServerParameters, L94](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/stdio.py#L94), pipes JSON-RPC over stdin and stdout, and owns a deliberately layered shutdown routine for process trees.
- **`StreamableHTTPTransport` carries the network half.** On the client side, [StreamableHTTPTransport, L111](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py#L111) wraps one `httpx` client and issues POSTs for requests while a long-lived GET streams server push; the `streamable_http_client` helper at [L713](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py#L713) is the ergonomic entry point.
- **`ClientSession` is the typed brain.** [ClientSession, L388](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/session.py#L388) owns the MCP layer: the [initialize, L651](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/session.py#L651) handshake, [call_tool, L993](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/session.py#L993) with output-schema validation, and callback hooks for sampling, elicitation, roots, and logging.
- **`JSONRPCDispatcher` correlates everything.** The [JSONRPCDispatcher, L264](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/shared/jsonrpc_dispatcher.py#L264) owns the receive loop, the pending-request table, cancellation, and error responses; [send_raw_request, L319](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/shared/jsonrpc_dispatcher.py#L319) is where an outbound request gets paired with its future response.
- **`SessionMessage` is the envelope.** Every message crossing a stream is a [SessionMessage, L59](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/shared/message.py#L59): a JSON-RPC payload plus client- or server-side metadata such as request contexts.
- **`ClientPeer` lets servers call back.** [ClientPeer, L65](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/shared/peer.py#L65) is the outbound surface a server session uses to ask the client's human or model for help, including [sample, L89](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/shared/peer.py#L89) for LLM completions.
- **`ServerSession` mirrors the client.** [ServerSession, L35](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/session.py#L35) is the per-connection server half, with capability-checked methods like [create_message, L183](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/session.py#L183) and [elicit, L329](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/session.py#L329).
- **`MCPServer` is the facade you actually write.** [MCPServer, L157](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/mcpserver/server.py#L157) hosts the `@tool`, `@resource`, and `@prompt` decorators; the [tool, L669](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/mcpserver/server.py#L669) decorator is where a plain Python function becomes an MCP tool.

## From Install to Your First Model

The package installs as `mcp` and requires Python 3.11 or newer. A server is three lines of decoration plus one line of registration; a client is one context manager away from a working call. The smallest honest example, adapted from the repository's own docs:

```python
from mcp.server.mcpserver import MCPServer

server = MCPServer("demo")

@server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

The [tool, L669](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/mcpserver/server.py#L669) decorator does more work than it appears to. Internally it hands your function to [ToolManager, L19](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/mcpserver/tools/tool_manager.py#L19), which calls [func_metadata, L287](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/mcpserver/utilities/func_metadata.py#L287) to build a Pydantic argument model from the type hints and emit the JSON Schema that clients will show the model. If your function annotates a `Context` parameter, the decorator injects it so the tool can log, report progress, or reach back into the request. On the client side, the symmetric experience is:

```python
from mcp import Client

async with Client("http://localhost:8000/mcp") as client:
    result = await client.call_tool("add", {"a": 1, "b": 2})
```

That [Client, L268](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/client.py#L268) facade is new in this generation, and it is the piece most tutorials have not caught up with. Passing a URL string selects Streamable HTTP; passing `StdioServerParameters` launches the subprocess; passing an `MCPServer` object connects in-process, which is why the SDK's test suite runs thousands of scenario stories without opening a single socket. Under the hood it still bottoms out in `ClientSession`; the facade caches resource listings, folds extensions into tool results, and only then hands you the typed calls. Tool results come back as a `CallToolResult` wire model — [CallToolResult, L1463](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp-types/mcp_types/_types.py#L1463) — carrying both human-readable content blocks and, when the tool declares one, structured content validated against the tool's `output_schema`. This is the same boundary that tool-calling models in [transformers](/transformers/) consume: the schema the SDK advertises is what the model learns to fill in.

## One Dispatcher to Correlate Them All

The most interesting refactoring in the current tree is what replaced the old `shared/session.py`. Both `ClientSession` and `ServerSession` now ride the same [JSONRPCDispatcher, L264](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/shared/jsonrpc_dispatcher.py#L264), which owns the raw conversation: reading `SessionMessage` items off a stream pair, matching responses to pending request ids, applying timeouts, converting peer errors into `MCPError` exceptions, and running request handlers as tasks in an anyio task group. The [send_raw_request, L319](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/shared/jsonrpc_dispatcher.py#L319) method documents the contract precisely: a `REQUEST_TIMEOUT` error if `opts["timeout"]` elapses, a `CONNECTION_CLOSED` error if the transport dies first.

Two design details repay attention. First, the dispatcher distinguishes "inline" methods — like `initialize` — that are awaited inside the read loop before the next message is dequeued, versus everything else that runs as a spawned task; the docstring warns that an inline handler that awaits its peer deadlocks the parked loop, a comment written from scar tissue. Second, cancellation is policy: `peer_cancel_mode` decides whether a cancelled request interrupts the handler or merely marks the result abandoned, and mid-handler requests made while servicing an inbound message are routed onto that inbound request's SSE stream via the `_related_request_id` channel. Above the dispatcher sit the type-specific sessions. `ClientSession` adds the constructor-injected callback surface — a client must answer `sampling/createMessage`, `elicitation/create`, and `roots/list` if the server asks — and absorbs tool listings so it can compile output-schema validators ahead of the first `call_tool`. `ServerSession` adds the mirrored server-to-client methods, all gated by [check_client_capability, L137](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/session.py#L137) so a server never assumes the far end can sample.

## Two Transports, One Contract

The transport layer is where the protocol's name promises live. `stdio` is the local case: `stdio_client` starts the process, deduplicates Windows command resolution, and — worth reading on its own — shuts down in ordered stages: close pipes, wait for exit, then escalate to terminating the whole process tree. The stream contract both sides honor is deliberately boring: write `SessionMessage` in, read `SessionMessage` or `Exception` out. That uniformity is what lets an in-process memory transport, a subprocess, and an HTTP peer be interchangeable to the dispatcher.

Streamable HTTP is the stateful network case, and the server side is the more instructive file. [StreamableHTTPServerTransport, L150](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/streamable_http.py#L150) is an ASGI application whose single entrypoint [handle_request, L474](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/streamable_http.py#L474) validates Accept headers and session ids before branching on the verb. A POST can be answered three ways: inline JSON for quick requests, an SSE stream scoped to one request id, or — for servers that support it — the standalone GET stream that carries server-initiated messages and notifications. Session state rides in the `mcp-session-id` header, and disconnects are survivable: if the server is constructed with an [EventStore, L116](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/streamable_http.py#L116), every SSE event is logged with an id and a reconnecting client presenting `Last-Event-ID` gets a replay of exactly what it missed. Deployment is delegated upward: the [StreamableHTTPSessionManager, L45](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/streamable_http_manager.py#L45) binds the transport to a running ASGI app and keeps per-session state across requests, so a server mounted behind uvicorn scales to many concurrent sessions.

## The Server Side, Three Layers Deep

Servers are stacked in three strata, and knowing which layer owns what explains most integration puzzles. The top layer is `MCPServer`, the decorator facade. It aggregates tools, resources, and prompts; [run_stdio_async, L1074](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/mcpserver/server.py#L1074) and its HTTP sibling are the one-liners that put it on a wire. The middle layer is the low-level [Server, L129](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/lowlevel/server.py#L129): a method registry where [add_request_handler, L484](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/lowlevel/server.py#L484) maps JSON-RPC method names to typed handlers, capabilities are computed from what you registered, and a default `server/discover` handler exposes the modern per-request envelope to clients that probe for it. The bottom layer is per-connection state: [Connection, L185](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/connection.py#L185) pairs a `ServerSession` with its request context and lifespan-scoped objects for the duration of one conversation.

The layering shows its value in the server-to-client direction. When your tool decides it needs the model — say, to summarize a 200-page PDF before extracting a table — it awaits `ctx.session.create_message`, which the session capability-checks, the dispatcher writes as a request, the client's `sampling_callback` receives, and the client's LLM answers. Elicitation is the same trip with a human on the other end: `elicit` sends a JSON-Schema form (or a URL, for out-of-band flows) and suspends until the user responds. Nothing in the high-level facade hides these round trips; the SDK just makes them one await away.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/mcp-python-sdk/modelcontextprotocol-mcp-python-sdk-architecture.svg" alt="Detailed architecture of the MCP Python SDK with client facade, transport internals, JSON-RPC dispatcher, server sessions, server facades, and wire types" style="min-width:900px;"></div>

<p><em>The detail map: one dispatcher core shared by both sides, transports that reduce to a SessionMessage stream contract, and a three-layer server stack ending in generated wire types.</em></p>

## Honest Limits

The protocol is young and the SDK says so in its own code. The version registry lists five revisions, and only `2025-11-25` and earlier are reachable through the classic `initialize` handshake; the newest `2026-07-28` revision — `LATEST_PROTOCOL_VERSION` at [L50](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp-types/mcp_types/version.py#L50) — uses a per-request envelope that clients must actively probe for. Interoperability across that boundary is exactly the kind of thing you should test against real peers rather than assume.

Second, the renames are churn you will pay for. `FastMCP` still exists as a module in name only: importing it raises a `ModuleNotFoundError` whose message exists purely to point you at the migration guide. Tutorials, agent frameworks, and Stack Overflow answers that say `from mcp.server.fastmcp import FastMCP` are all wrong on this branch. Pin `mcp<2` if you need the old surface.

Third, the abstractions have seams the docstrings admit to. The dispatcher's inline-method rule can deadlock a careless handler; the session keeps a bound-method reference on its dispatcher after exit, flagged in-source as a memory pin awaiting a rework; and stdio shutdown has four fallback stages because process trees do not die politely. None of this is hidden — it is commented — but the comfortable framing only holds until a connection hangs.

Where does this sit in the stack you already follow? On the serving side, the same tool-calling contracts are consumed by runtimes like [vllm-2](/vllm-2/) when a model must choose and fill a tool; on the agent side, servers in the style of [llama-stack](/llama-stack/) speak MCP so clients stay transport-agnostic. The SDK's job is to make both directions boring, and — read closely — it mostly succeeds.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Inside the MCP Python SDK: Sessions, Transports, and the Machinery Under Every MCP Server",
  "description": "A source tour of the official Model Context Protocol Python SDK covering ClientSession, the shared JSON-RPC dispatcher, Streamable HTTP and stdio transports, and the MCPServer facade.",
  "author": { "@type": "Person", "name": "PyShine" },
  "datePublished": "2026-10-27",
  "keywords": "MCP, Model Context Protocol, Python SDK, ClientSession, MCPServer, JSON-RPC, Streamable HTTP",
  "publisher": { "@type": "Organization", "name": "PyShine" },
  "mainEntityOfPage": "https://pyshine.com/mcp-python-sdk/"
}
</script>
