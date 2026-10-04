---
layout: post
title: "OpenAgentCore: Run the OpenAI Agents API on Your Own Infrastructure - Inside MiniMax-AI/OpenAgentCore"
description: "A source tour of MiniMax-AI/OpenAgentCore, a self-hosted, protocol-first Go and TypeScript implementation of the OpenAI Agents API with swappable harnesses, sandbox providers, model providers, and Postgres-backed durable sessions."
date: 2026-10-04
header-img: "img/post-bg.jpg"
permalink: /OpenAgentCore-Run-the-OpenAI-Agents-API-On-Your-Own-Infrastructure/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/openagentcore/minimax-ai-openagentcore-architecture.svg
tags:
  - AI Agents
  - Go
  - OpenAI
  - Self-Hosted
categories: [AI, Open Source]
keywords: "OpenAgentCore, OpenAI Agents API, self-hosted agents, MiniMax, Codex harness, Claude Code harness, sandbox providers, Docker sandbox, E2B, microsandbox, durable sessions, Postgres, protocol-first design, Go monorepo, agents SDK"
author: "PyShine"
---

The OpenAI Agents API is a lovely thing to build against: create a Session, stream the events, poll the durable state. It is much less lovely to build on when every request leaves your data center, your harness choices are fixed, and your sandbox is somebody else's container fleet. MiniMax-AI's OpenAgentCore repository takes that whole contract and reimplements it on your own metal. It is an open-source, self-hosted implementation of the OpenAI Agents API, and the promise it makes is blunt: point the official OpenAI SDK, or plain HTTP, at your installation and nothing about your client code has to change.

The scope of the reimplementation is what makes the repository worth a source tour. This is a serious monorepo with roughly two thousand Go files and several hundred TypeScript files, wired as a Go workspace plus a pnpm workspace. The core service speaks three API surfaces: the public Agents API on `/v1`, an operator-facing Core API on `/core/v1`, and a machine connection API on `/api/v1` that nodes and daemons use to enroll and execute. Around those live a Postgres persistence layer, a sandbox layer with Docker, microsandbox and E2B providers, and an execution runtime that runs real harnesses, Codex, Claude Code, or MiniMax Code, inside the environment of your choice.

What elevates the code above a typical "OpenAI-compatible" clone is the discipline around its seams. The contributor handbook declares protocols at every boundary: exactly one code file and one document define each interface between Core, sandbox providers, runtimes, harnesses and model providers, and no component may join the system through a private side door. Reading this source teaches you how to build infrastructure that stays replaceable under real feature pressure, and that lesson is portable far beyond this project.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openagentcore/minimax-ai-openagentcore-overview-architecture.svg" alt="Architecture overview of the MiniMax-AI/OpenAgentCore repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the repository: clients enter through the Agents API, Core owns durable state and sandboxing, and execution flows through a gateway to the daemon that drives real harnesses against a model provider.*

Reading the overview from left to right: your application, still using the stock OpenAI SDK, talks to the route layer in `services/core/internal/api`, which serves both the public `/v1` surface and the operator `/core/v1` surface mounted by the server entrypoint in `services/core/cmd/server/main.go`. Durable truth lives in Postgres through the stores under `services/core/internal/persistence/postgres`, while the sandbox manager provisions an environment for each Session. On the right side, the runtime gateway accepts WebSocket connections from the execution daemon in `apps/daemon`, hands turn commands across the wire protocol in `internal/agentdaemon/proto`, and the harness registry in `apps/daemon/internal/agent` drives the selected coding agent, which in turn speaks to a model provider configured through `internal/modelprovider/config.go`.

## Why You Need This

The first problem is control. Managed agent platforms decide where your code runs, which harness executes it, and how long your workspace survives. OpenAgentCore flips every one of those decisions to you. Sandboxes run under Docker or microsandbox on your own nodes, or in E2B if you want a managed remote option, and the same Session can run on your own Linux, macOS or Windows machine as a self-hosted environment. Because every connection is a declared protocol, you can swap the sandbox backend without touching the execution flow.

The second problem is harness lock-in. A coding agent is more than a model call: it is a loop of tool executions, workspace reads and writes, steering messages and subagent turns. This repository does not rebuild that loop badly; it runs native harnesses, Codex, Claude Code and MiniMax Code, each through its maintained upstream SDK or protocol, and the project's own rules forbid building a second executor or a compatibility layer to fake parity. You get the genuine behavior of each harness, with the differences recorded in a coverage ledger rather than papered over.

The third problem is data gravity. Agent sessions contain your source files, your prompts, and your credentials. In OpenAgentCore, durable state lives in your Postgres, credentials sit in encrypted vaults, and the public API never interprets product payloads it does not own. The API contract is pinned to the official OpenAI Agents API through a recorded upstream baseline in `contracts/agents-api/upstream.json`, so compatibility is a checked property, not a marketing claim.

The fourth problem is operational blindness. The operator side answers it with a real web console under `apps/web`, backed by admin routes in `services/core/internal/api`, where you issue Project API keys, configure the default model, add execution capacity, and watch agent metrics. Installation is a single installer script that sets up the service, and the documentation is maintained in English and Simplified Chinese with automated checks that reject stale translations.

## How It Works

The elegant part of this codebase is that each layer knows exactly one neighbor, so a Session request becomes a chain of small, typed handoffs.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/openagentcore/minimax-ai-openagentcore-architecture.svg" alt="Detailed architecture of the MiniMax-AI/OpenAgentCore repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the API layer and its generated contracts, the domain stores, the sandbox manager with three interchangeable providers, the gateway-to-daemon wire, and the harness adapters that reach model providers.*

### Understanding the Architecture

**The entrypoint wires everything by hand.** `services/core/cmd/server/main.go` validates process configuration, connects to Postgres with pgx, applies schema migrations, and then assembles the dependency graph for the API layer. There is no magic container here; the ordering itself documents the system, and a `check-config` subcommand lets you fail fast on a bad environment before the listener opens.

**The API layer is one package with a strict route table.** Under `services/core/internal/api`, session creation and input handling in `session_creation_stream.go`, event streaming in `stream.go`, turn responses in `turns.go` and the admin surface in `admin_sessions_routes.go` share middleware that canonicalizes request paths and stamps every response with OpenAI-compatible headers and request IDs. The wire types it validates live in `contracts/agents-api/v1`, and the generator behind `make openapi` turns those annotations into the published OpenAPI documents, so the contract and the code cannot drift silently.

**Sandboxing is a protocol, not a feature.** The manager in `services/core/internal/api/sandbox_manager.go` talks to sandbox backends only through the contract declared in `services/core/internal/sandbox/sandbox_provider.go`. Docker, E2B and microsandbox each implement that contract in their own directories, deployments and placements are durable rows in Postgres, and a call fence keeps lifecycle operations from racing. Core never substitutes one provider for another; an unsupported combination is a typed error, which is precisely why the abstraction holds up.

**Execution crosses one wire, once.** Devices enroll and authenticate through the runtime gateway in `services/core/internal/runtimegateway`, which maintains a registry of live WebSocket sessions on the machine connection path. The daemon side, launched by `apps/daemon/cmd/oac-daemon/main.go`, receives typed envelopes from `internal/agentdaemon/proto`, covering startup, workspace reads and writes, tool calls, steering, subagents, token usage and suspension, and dispatches them in `apps/daemon/internal/dispatch`. A bootstrap handshake in `internal/runtimebootstrap/bootstrap.go` gets the runtime ready before the first turn.

**Harnesses are declared, not discovered.** The daemon's registry in `apps/daemon/internal/agent/registry.go` admits only harnesses that declare their capabilities through `internal/harnessconfig`, and the configuration check rejects combinations a harness never claimed to support. The Claude Code path is bridged by the adapter in `packages/claude-sdk-adapter/src/adapter.ts`, the MiniMax Code path runs through the worker in `packages/mcode-harness/worker.ts`, and Codex is configured natively. Whichever harness runs, its model calls are shaped by the provider protocol in `internal/modelprovider/config.go`, which is how a self-hosted Core can drive an OpenAI-compatible endpoint of your choosing.

**Persistence is boring on purpose.** Every domain store, sessions, projects, vaults, skills, model configuration, files, lands in Postgres through the generated access layer under `services/core/internal/persistence/postgres`, with schema changes shipped as migrations applied at startup. Sessions are durable first: the quickstart tells you to poll state rather than resubmit, and the API's semantics, pagination, error shapes and status transitions, mirror the official service closely enough to be checked automatically.

Follow one request end to end: your application calls `sessions.create` with an `openai_hosted` environment; the API layer validates the payload against the pinned contract, records the Session in Postgres, and asks the sandbox manager to prepare an environment; the chosen sandbox provider boots it while the runtime gateway waits for the enrolled daemon; the daemon receives the harness configuration over the wire protocol, starts the selected coding agent, and streams tool calls and model turns back through the gateway, where they are persisted as items and forwarded to your client as SSE events until the turn completes.

## Advantages

- **Drop-in compatibility.** The official OpenAI SDK works unchanged against your installation; even the response headers and error shapes are reproduced, which means existing tools keep working.
- **Protocol-first boundaries.** Each seam between Core, sandbox providers, runtimes, harnesses and model providers is one code file and one document, so replacement is a designed operation instead of an archaeology project.
- **Real harnesses, not a reimplementation.** Codex, Claude Code and MiniMax Code run through their maintained upstream SDKs, with differences tracked in a coverage ledger rather than hidden behind shims.
- **Choice of execution venue.** Docker, microsandbox and E2B sandboxes plus self-hosted execution on your own Linux, macOS or Windows machine are all first-class, selected per Session.
- **Durable by default.** Postgres-backed sessions, placements and credentials survive restarts, and the API guides clients toward polling durable state instead of speculative retries.
- **Operator tooling included.** A web console, admin API, metrics and a machine connection API ship in the same monorepo, so running the thing is part of the design, not an afterthought.

## Benefits

- **Your data stays yours.** Workspaces, transcripts and vaulted credentials live on infrastructure you control, which matters for regulated teams and anyone with real confidentiality needs.
- **Cost control through capacity ownership.** Nodes you already own become execution capacity; managed sandboxes become one option among several instead of the only meter running.
- **Vendor flexibility at the model layer.** Saved model configurations let you route harnesses to different OpenAI-compatible providers, so a model change is configuration rather than a migration.
- **A reference architecture for agent platforms.** The protocols-at-every-boundary style, with typed errors and declared capabilities, is a template you can lift into your own infrastructure projects.
- **Bilingual, generated documentation.** English and Chinese docs are checked against each other automatically, and the OpenAPI documents are generated from the same source as the code, reducing the usual doc rot.
- **Straightforward contribution surface.** New sandbox providers or harnesses arrive as adapter changes that touch no Core execution path, keeping reviews small and focused.

## Usage

Install on a Linux amd64 host with Docker and Python 3.9 or newer:

```sh
curl -fsSL https://github.com/MiniMax-AI/OpenAgentCore/releases/latest/download/install.sh | bash
```

Then sign in to the web console with the Core key the installer created, set a default model, issue a Project API key, and add execution capacity. Your application needs nothing exotic, because Core serves the official OpenAI Agents API:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install openai==3.13.0
export OPENAI_BASE_URL=https://core.example/v1
export OPENAI_API_KEY=your-project-key
```

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_BASE_URL and OPENAI_API_KEY
print(client.beta.agents.list().data)

session = client.beta.agents.sessions.create(
    environment={"type": "openai_hosted"},
    input="Create /workspace/hello.txt with a short greeting, then describe it.",
    extra_body={"agent": {"x_agents_core": {"harness": "codex"}}},
)
print(session.id)
```

The `x_agents_core` extension object is where Core additions live, including the harness choice and an optional model provider with its own protocol, base URL and key. After a Session is created, poll its durable state rather than resubmitting the request. If you want to develop on the monorepo itself, the Makefile drives the usual loop:

```sh
make build-core
make build-daemon
make check
make openapi
```

`make check` runs the Go, persistence and TypeScript checks against a configured test database, and `make openapi` regenerates the published API documents from the route annotations.

## Conclusion

OpenAgentCore is a rare kind of open-source project: an infrastructure clone that competes on engineering discipline rather than a checklist of endpoints. The OpenAI Agents API surface is reproduced faithfully enough to be machine-checked, the execution stack beneath it is genuinely swappable, and the protocol documents make the architecture legible in an afternoon. If you run agents in production and have been uneasy about where your sessions live, reading this source is a productive way to spend an evening, and self-hosting it may be the most direct upgrade you can make.

Links:

- Repository: https://github.com/MiniMax-AI/OpenAgentCore
- Documentation: https://minimax-ai.github.io/OpenAgentCore/
- Architecture guide: https://minimax-ai.github.io/OpenAgentCore/docs/architecture
