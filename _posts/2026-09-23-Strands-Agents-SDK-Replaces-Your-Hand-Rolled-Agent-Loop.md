---
layout: post
title: "Strands Agents: The SDK That Replaces Your Hand-Rolled Agent Loop"
description: "Strands Agents is an open-source SDK for building production AI agents in Python and TypeScript - a model-driven agent loop, lifecycle controls, tools, MCP, multi-agent patterns, memory, guardrails, and tracing, with a fully assembled harness on top. We tour the architecture of the 7,300-star monorepo."
date: 2026-09-23
header-img: "img/post-bg.jpg"
permalink: /Strands-Agents-SDK-Replaces-Your-Hand-Rolled-Agent-Loop/
tags:
  - AI
  - Agents
  - Python
  - TypeScript
  - Open Source
author: "PyShine"
---
# Strands Agents: The SDK That Replaces Your Hand-Rolled Agent Loop

Every team building agents eventually writes the same code: a while loop around a model call, a tool executor, retry logic, a token counter, and then - three months later - session persistence, guardrails, hooks for observability, and a small framework nobody wants to maintain. [Strands Agents](https://github.com/strands-agents/harness-sdk) is the open-source answer to that drift: an Apache-2.0 SDK, 7,330 stars and 1,151 forks, that gives you the agent loop and everything it grows into, in both Python and TypeScript, with no hosted control plane - it runs in your process. The project's pitch is a model-driven approach: give the model tools and a task, and the loop handles the rest, while lifecycle controls (turn limits, token budgets, cancellation, stop reasons) keep it accountable. Where most agent frameworks stop at the loop, Strands keeps going: MCP client and server support, multi-agent patterns, memory and sessions, streaming, guardrails, tracing by default, and even an evals SDK. And because a growing SDK can overwhelm newcomers, it ships a second layer on top - Strands harness, a fully assembled, benchmarked agent you get with one function call, then peel back as you need control. After covering [orchestration at cluster scale](https://pyshine.com/AX-Kubernetes-Thinking-For-AI-Agent-Workloads/) and [office runtimes for agents](https://pyshine.com/Univer-Open-Source-Office-Runtime-AI-Agents-Can-Drive/), this is the layer most teams will actually start on: the agent itself.

![Architecture overview of the Strands Agents repository showing the harness layer, SDK core, and capabilities](/assets/img/diagrams/harness-sdk/harness-sdk-overview-architecture.svg)

## Why You Need This

The honest case for Strands is the honest cost of not having it. A hand-rolled agent loop is easy for a demo and quietly expensive in production, because the loop is never just a loop. [The agent loop Strands ships](https://strandsagents.com/docs/user-guide/concepts/agents/agent-loop/) cycles model inference and tool execution, but also enforces turn limits and token budgets, produces explicit stop reasons, supports cancellation mid-run, and traces every decision by default - the difference between an agent that fails visibly and one that burns tokens silently at 3 a.m. Then there is the portability problem: your loop is probably written against one provider's SDK. Strands is model agnostic, with first-class providers for Amazon Bedrock, Anthropic, OpenAI, and Gemini, plus [more providers and custom ones](https://strandsagents.com/docs/user-guide/concepts/model-providers/) - swap backends when you scale and your agent code stays the same. Control is the third pillar: hooks let you intercept any step of the loop to log it, validate it, or redirect it, [guardrails catch mistakes before they run](https://strandsagents.com/docs/user-guide/safety-security/guardrails/), and steering handlers let the agent correct itself instead of failing silently. That last set of features matters because of what we saw in [OpenAI's own incident report](https://pyshine.com/OpenAI-Listed-Six-Ways-Its-Own-AI-Broke-the-Rules/): misbehaving agents rarely announce themselves, and the mitigation is infrastructure - interception points, budgets, isolation - not better prompts. Strands is that infrastructure, packaged, in your process, under an Apache-2.0 license.

## How It Works

The repository is a monorepo with a clear split: two SDKs that share a design philosophy across languages, two harnesses assembled on top of them, a CLI, an MCP server, and the documentation site.

![Detailed architecture diagram of the Strands Agents monorepo from the repository source](/assets/img/diagrams/harness-sdk/harness-sdk-architecture.svg)

At the core sits the loop. The Python SDK (`strands-py`) exposes an `Agent` class whose invocation runs the event loop in `strands-py/src/strands/event_loop`: alternate model inference and tool execution until the model produces a final answer or a limit trips. The TypeScript SDK (`strands-ts`) mirrors it in `strands-ts/src/agent`. Both SDKs implement the same concept set with enforced parity - the contributor guide requires identifiers to match across languages (re-cased to language idiom), hook event names to stay in sync, and vended plugin directories to translate mechanically (`vended_plugins` to `vended-plugins`). Around the loop sits the control plane. Hooks (`strands-ts/src/hooks`) emit events at every step you can subscribe to; interventions (`strands-py/src/strands/interventions`) inject steering so an agent can recover instead of dead-ending; middleware (`strands-py/src/strands/_middleware`) wraps invocations with composable behavior; and telemetry traces every decision by default. Capabilities hang off the loop. Model providers (`strands-py/src/strands/models`) normalize Bedrock, Anthropic, OpenAI, Gemini, Ollama, and custom backends behind one interface with bidirectional streaming. Tools (`strands-ts/src/tools`) turn decorated functions with schemas into callable capabilities, and the MCP client (`strands-ts/src/mcp`) connects external MCP servers as tool sources - the same protocol we saw [AX use to wire workspaces](https://pyshine.com/AX-Kubernetes-Thinking-For-AI-Agent-Workloads/). Memory (`strands-ts/src/memory`), sessions (`strands-ts/src/session`), and the context manager (`strands-ts/src/context-manager`) keep long-running agents coherent: conversation managers trim history inside token budgets, session persistence survives process restarts through pluggable storage, and a sandbox isolates untrusted tool execution. Above the SDKs sit two assembled harnesses: `create_harness()` in Python and `createHarness()` in TypeScript return an agent with benchmarked defaults for model, tools, memory, sessions, and context management, drawn from their own plugin and tool packages. The `strands` CLI wraps the TypeScript harness into a terminal chat - a full TUI with permission prompts, session management, and voice input. And the Strands MCP Server package turns agents themselves into MCP tools, so any MCP-speaking client can drive a Strands agent.

## Advantages

- **Two languages, one design.** Python and TypeScript SDKs with enforced cross-language parity - names, hook events, and wire formats stay in sync by policy, not luck.
- **Own the loop or rent it.** Drop to the SDK for full control of tools, providers, and memory; start with the harness when benchmarked defaults are enough.
- **Lifecycle controls built in.** Turn limits, token budgets, cancellation, and explicit stop reasons are loop primitives, not add-ons.
- **Model portability.** First-class Bedrock, Anthropic, OpenAI, and Gemini support, with streaming, plus custom providers.
- **Interception everywhere.** Hooks on every loop step, steering interventions, middleware, and default-on tracing make agents auditable.
- **Production surface included.** Guardrails, sandboxed execution, session persistence with pluggable storage, retry, and an evals SDK.

## Benefits

The compounding benefit is optionality. Teams start with the harness - [a one-call, batteries-included agent](https://strandsagents.com/docs/user-guide/harness/) - and ship in a day, then peel back layers as requirements harden, without a migration, because the harness was always just a curated composition of the same SDK underneath. The dual-language story pays off twice: server code in Python and product code in TypeScript can share concepts, tool names, and session semantics without translation, which is rare enough among agent frameworks to be a decision of its own. And because state, memory, and tool execution are pluggable, Strands composes with the rest of the stack - point it at [shared long-term memory across CLIs](https://pyshine.com/ai-memory-Rust-Long-Term-Memory-For-Coding-Agents/), embed it in [applications built around agents](https://pyshine.com/Agent-Native-BuilderIO-Framework-Builds-Apps-Around-Agents/), or drive it from a terminal the way the bundled CLI does. The MCP server closes the loop: your assembled agent becomes a tool that other agents - or your editor, or any MCP client - can call.

## Usage

The fastest path is the harness. Install it and invoke:

```bash
pip install strands-harness
```

```python
from strands_harness import create_harness

agent = create_harness()
agent("Find the slowest test in this repo and explain why it's slow")
```

The same shape in TypeScript:

```bash
npm install @strands-agents/harness
```

```typescript
import { createHarness } from '@strands-agents/harness'

const agent = await createHarness()
await agent.invoke("Find the slowest test in this repo and explain why it's slow")
```

When you want to own the loop, drop to the SDK - Python 3.10+ or Node.js 22+:

```bash
pip install strands-agents strands-agents-tools
```

```python
from strands import Agent
from strands_tools import calculator

agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

```typescript
import { Agent } from '@strands-agents/sdk'

const agent = new Agent()
const result = await agent.invoke('What is the square root of 1764?')
console.log(result)
```

Configure a provider with your keys (Bedrock, Anthropic, OpenAI, Gemini, Ollama), then register tools, hook the loop, and switch on sessions as you grow. To prototype from the terminal, install [the CLI](https://www.npmjs.com/package/@strands-agents/cli) and run `strands` for a chat against a harness agent. Worked examples live in the [samples repository](https://github.com/strands-agents/samples), and the [harness quickstart](https://strandsagents.com/docs/user-guide/harness/quickstart/) walks the assembled path end to end.

## Conclusion

Strands Agents takes the part of agent building everyone underestimates - the loop and its entourage of limits, hooks, memory, and guardrails - and makes it a library instead of a rite of passage. The monorepo is disciplined about it: two SDKs held to cross-language parity, two harnesses that prove the SDK's defaults, a CLI and MCP server that make agents usable from a terminal and from other agents. The [documentation](https://strandsagents.com) is deep, the license is permissive, and the dependency is only your model provider. If your next project is an agent, start with `create_harness()` and work down - and stop writing that while loop.
