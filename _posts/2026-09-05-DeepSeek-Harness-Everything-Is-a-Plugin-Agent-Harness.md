---
layout: post
title: "DeepSeek Harness: The Everything-Is-a-Plugin Agent Harness Powered by Cordis"
description: "DeepSeek Harness (dsh) is an open-source agent harness built on an everything-is-a-plugin architecture powered by Cordis, a meta-framework for spatiotemporal composability. Every part of the product — model adapters, tool registries, the agent loop itself — is a replaceable plugin."
date: 2026-09-05
header-img: "ai-coding-frameworks/ai-coding-frameworks"
permalink: /deepseek-harness-everything-is-a-plugin-agent-harness-cordis/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [DeepSeek, DeepSeek Harness, dsh, Cordis, Agent Harness, Plugin Architecture, TypeScript, Spatiotemporal Composability]
author: PyShine
---

# DeepSeek Harness: The Everything-Is-a-Plugin Agent Harness

DeepSeek AI has released **DeepSeek Harness** (`dsh`) — an open-source agent harness that has rocketed to the top of GitHub's trending chart with 62,300 weekly star gains. The core idea is radical in its simplicity: **everything is a plugin**, including the model adapter, the tool registry, the session log, and the agent loop itself. There is no privileged core to patch. You extend `dsh` by mounting a plugin beside the others, and every registration is a reversible effect that unwinds when its plugin unloads.

![Cordis Plugin Tree Architecture](/assets/img/diagrams/deepseek-harness/dsh-architecture.svg)

## The Foundation: Cordis and Spatiotemporal Composability

DeepSeek Harness is powered by [Cordis](https://github.com/cordiverse/cordis), a meta-framework whose design is described in the paper [*A Programming Paradigm for Spatiotemporal Composability*](https://arxiv.org/abs/2608.25512) (arXiv:2608.25512, submitted August 2026 by Yifan Shi, Wei Zhang, and Tianyi Cui from Peking University and DeepSeek-AI).

The paper identifies two orthogonal dimensions of dynamic composition:

| Dimension | Problem | Cordis Solution |
|-----------|---------|-----------------|
| **Temporal composability** | Reverting a component's side effects upon removal | **Revertible effects** — every context transformation carries an inverse that the runtime holds |
| **Spatial composability** | Declaring and managing inter-component dependencies | **Reactive coeffects** — every context change is classified against a component's coeffect specification to drive activation/deactivation |

These two dimensions are unified into a single **context type** — the **context paradigm** — through which every effect and coeffect is mediated. This means the effects of distinct components interleave without disturbing one another.

In practical terms: when a plugin registers a tool, a model adapter, or an event handler, that registration is an effect with an inverse. When the plugin unloads, the effect unwinds — the tool disappears from the registry, the adapter stops responding, the event handler stops firing. No manual cleanup required.

## Profiles and Bundles

A running `dsh` instance is a plugin tree composed at boot from ordered layers.

![Turn Flow](/assets/img/diagrams/deepseek-harness/dsh-turn-flow.svg)

### Profiles

A **profile** is a named composition stored in the Harness home. It lists the bundles it stacks, holds any out-of-tree plugins it installs, and keeps the user's own `cordis.patch.yml`. Five profiles ship as templates:

| Profile | Purpose | Reload |
|---------|---------|--------|
| `web` | Browser application at `http://127.0.0.1:3080` | Live reload |
| `headless` | One-shot runner with no server | Once at startup |
| `sdk` | SDK JSON-RPC server (TypeScript + Python) | Once at startup |
| `sdk-minimal` | Standalone bundle (does NOT apply `dsh-base`) | Once at startup |
| `acp` | Automation-only ACP server | Once at startup |

### Bundles

A **bundle** is a distribution format for Cordis config rows and the code they mount. Each declares itself in its `package.json` under a `dsh` field: `dsh.profile` lists a profile's bundles, and `dsh.bundle` points at a bundle's patch file.

**`dsh-base`** is the shared first layer of the `web`, `headless`, `sdk`, and `acp` profiles. It provides:
- Model adapters
- Tool registry
- Persistence
- Sandbox and approval policy
- Settings, credentials, telemetry

Profile-specific bundles then add their own capabilities on top:
- **`dsh-web-app`** adds the browser application
- **`dsh-headless`** adds a one-shot runner
- **`dsh-sdk-app`** adds the SDK JSON-RPC server
- **`dsh-acp-app`** adds the automation-only ACP server
- **`dsh-sdk-minimal`** is the exception — it owns its complete explicit SDK tree and does not apply `dsh-base`

### Patch Layers

Layers apply to an empty entry list in this order:

1. Each bundle in the profile's listed order
2. The profile's `cordis.patch.yml`
3. The home-level patch
4. Any `--patch` overlay

A patch targets a row by id and replaces its whole config, or inserts new rows. To see the tree your machine boots:

```bash
dsh --profile web --dump-config
```

Any row it prints can be replaced by a patch of your own.

## Core Packages

| Package | Owns | `ctx` Key |
|---------|------|-----------|
| `core/session` | Append-only `SessionEvent` log and in-memory store | `ctx.sessions` |
| `core/system-prompt` | Prompt-section and tool-schema assembly | `ctx.systemPrompt` |
| `core/tools` | Scoped tool registry and guarded execution pipeline | `ctx.tools` |
| `core/agent` | `Agent` interface, live registry, and `agent/*` events | `ctx.agents` |
| `core/agent-loop` | Default driver implementing the `Agent` interface | `ctx.agentLoop` |
| `core/scope` | Per-agent scoped-registration primitive | library, no key |
| `llm/llm` | Message and stream vocabulary plus the adapter seam | `ctx.llm` |
| `webhook/webhook` | Authenticated-delivery dispatch and Workspace Session creation | `ctx.webhookRuntime` |

## Agent Turn Flow

A **step** is one model request plus the tools it calls. A **turn** is zero or more steps: it opens before its first input is claimed and closes once nothing is owed.

The turn flow has three event domains:

1. **Session events** (durable) — `turn/*`, `step/*`, `user/message`, `assistant/*`, `tool/*` — survive reloads and are the source of model-visible context
2. **Agent events** (live) — `agent/*` — carry a live `Agent` for observing or intercepting work in flight
3. **Capability events** — `fs/*`, `tools/*`, `telemetry/*` — attach policy and adapters without importing the loop

Key waterfall events (listeners must call `next()` to delegate): `agent/pre-step`, `agent/request`, `llm/stream`, `tools/pre-execute`, `tools/post-execute`. The `agent/turn-stopping` event is serial and has no `next()`.

### The Model-Visible Invariant

**Model-visible means logged.** Anything that reaches a model request must be reconstructable from the session log, and a runtime invariant asserts this. This is why a new model-visible input requires a new session event: extend `SessionEventMap` and render from the log.

## Capability Seams: One Swap Changes Everything

![Capability Seams](/assets/img/diagrams/deepseek-harness/dsh-capability-seams.svg)

A **seam** is a swappable capability with three roles:

1. **Service Definition** — declares the interface (the contract)
2. **Service Provider** — implements the interface (the implementation)
3. **Consumer** — uses the capability (commonly a model-facing tool)

Seams are why one provider swap changes the whole product. Filesystem and subprocess providers share one execution world, so pointing them at a remote sandbox moves Bash, PTY, and LSP with them — no provider forks needed. Subagent providers vary just as widely behind one interface, from a fresh child agent to a delegated turn in another product.

### Experimental: Agent Teams

Agent Teams is a private opt-in coordination seam on `ctx.agentTeams`, with a durable roster, task board, and mailbox layered over continuable subagents.

## Extension Points Map

![Extension Points Map](/assets/img/diagrams/deepseek-harness/dsh-extension-points.svg)

New behavior attaches to a documented extension point. There is no privileged core to patch — you extend `dsh` by mounting a plugin beside the others.

| Goal | Mechanism |
|------|-----------|
| Add a model provider | Register its adapter on `ctx.llm` |
| Add a model-facing capability | Register on `ctx.tools`; schema joins prompt assembly |
| Add shell execution | Register a `ctx.shell` backend; local one spawns through `ctx.subprocess` |
| Add persistent terminal execution | Register a `ctx.terminals` backend plus `dsh-tool-terminal` |
| Add a human command | Register on `ctx.commands`; dispatches without a model turn |
| Add background work | Register on `ctx.jobs`; `job_*` tools collect or stop it |
| Start a Session from an external webhook | Register a trusted rule on `ctx.webhookRuntime` and mount a provider adapter |
| Add filesystem access or policy | Register a `ctx.fs` provider or listen to `fs/*` events |
| Confine spawned processes | Use a `ctx.sandbox` backend; consumers wrap argv before spawning |
| Intercept a request, tool, or turn | Use its `agent/*` or `tools/*` event; `agent/turn-stopping` stops a turn |
| Add model-facing context | Call `agent.inject()`; lands in the next admitted request |
| Add UI or editor integration | Drive `ctx.agents` and render from `session/event` |
| Add a Web Client Chat node | Register a `ConversationNodeDefinition` + keyed renderer |
| Add durable session state | Extend `SessionEventMap`; render and replay from the log |
| Store sessions in a new backend | Implement `SessionPersistence` (`create`/`open`/`stat`/`list`/`export`) |
| Fork a session at a turn boundary | `ctx.agents.create({ sessionId, seed, meta: { parentSession, seedLength } })` |

## Quick Start

### Run from npm

```bash
npx @deepseek-ai/dsh web
```

This starts the Web UI at `http://127.0.0.1:3080` by default. Pass `--no-open` to run the server without opening a browser.

### Run from Source

```bash
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm run build
pnpm dsh web
```

### Configure a Model

Open **Settings > Models** in the Web UI, enter your [DeepSeek API key](https://platform.deepseek.com/), and save. Model routing becomes available immediately — no server restart needed.

### Python SDK

The Python SDK packages the normal `dsh` CLI as `deepseek-harness-sdk-runtime-<platform>-<arch>`. The client launches `dsh --profile sdk` with an explicit Harness home by default. The minimal example selects the shipped `sdk-minimal` profile.

## Safety Notice

DeepSeek Harness is in **developer preview** and iterating rapidly. There will be compatibility-breaking changes. Review the [safety notice](https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.md) before running the project.

The repository enforces application entrypoint integrity via `verify-application-entrypoints`, which keeps every package bin, executable source, and root demo in an explicit class and rejects a Node application path that bypasses `dsh`.

## Key Design Decisions

**Why everything-is-a-plugin?** Traditional agent harnesses have a privileged core that you patch to add features. `dsh` eliminates the core: every capability is a plugin, so extension never requires modifying the framework. This makes customization safe and composable.

**Why revertible effects?** When a plugin unloads, its effects must completely unwind — otherwise the system accumulates stale registrations. Revertible effects make hot module replacement safe: you can load and unload plugins at runtime without restarting the server.

**Why the context paradigm?** Unifying effects and coeffects into a single context type means the runtime can mediate every interaction between components. This induces an observational equivalence under which the effects of distinct components interleave without disturbing one another — the formal foundation for plugin isolation.

**Why durable session events?** The session log is the source of truth for model-visible context. `deriveMessages()` projects model history from it, and raw `assistant/chunk` events preserve replay and UI fidelity. Fork, resume, transcripts, telemetry, and persistence all derive from this stream.

**Why waterfall events?** Waterfalls let multiple plugins observe and potentially modify a request before it proceeds. Each listener must call `next()` to delegate, giving plugins the power to intercept, rewrite, or reject model requests and tool executions in a composable chain.

## Further Reading

- [GitHub: deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) — MIT license
- [Documentation](https://deepseek-harness.github.io/deepseek-harness/)
- [Architecture Documentation](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)
- [Cordis Primer](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-primer.md)
- [Paper: A Programming Paradigm for Spatiotemporal Composability (arXiv:2608.25512)](https://arxiv.org/abs/2608.25512)
- [Agent Lifecycle Sequence Diagram](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/agent-lifecycle.md)
- [Tool Execution Pipeline](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/tool-execution-pipeline.md)
- [Extension Cookbook](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cookbook/extension-cookbook.md)
- [Capability Seams](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/capability-seams.md)
- [Discord Community](https://discord.gg/Ycq5dCaS4)
- [DeepSeek API Keys](https://platform.deepseek.com/)

## Summary

DeepSeek Harness is the #1 trending repository on GitHub, and for good reason. It rethinks the agent harness from first principles: instead of a monolithic core with extension points, every part of the product is a plugin — model adapters, tool registries, session logs, and the agent loop itself. The foundation is Cordis, a meta-framework for spatiotemporal composability whose formal theory (revertible effects + reactive coeffects = context paradigm) ensures that plugins can be loaded, unloaded, and interleaved without disturbing one another. With five profiles (web, headless, sdk, sdk-minimal, acp), a comprehensive extension points map, capability seams that make one provider swap change the whole product, and MIT licensing, `dsh` is both a production-ready agent harness and a research platform for the next generation of composable AI systems.
