---
layout: post
title: "UFO Core: Team Agents on Your Own Infrastructure - Inside ufo-ai/ufo-core"
description: "A source tour of ufo-ai/ufo-core, an open-source runtime where a team hands work to AI agents in chat, with DBOS-durable turns, chat-native account grants, swappable sandbox carriers, and an extension system that covers tools, connectors, subagents, surfaces, and model providers."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /UFO-Core-Run-Team-Agents-On-Your-Own-Infrastructure/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ufo-core/ufo-ai-ufo-core-architecture.svg
tags:
  - AI Agents
  - Python
  - Rust
  - Self-Hosted
categories: [AI, Open Source]
keywords: "UFO core, agent runtime, self-hosted AI agents, DBOS durable workflows, sandbox carriers, Docker sandbox, E2B, connector grants, agent extensions, Slack agents, terminal agent client, SQLite to Postgres, agent operating system, ufoctl, workspace scoping"
author: "PyShine"
---

Most agent frameworks stop at the loop: a model, some tools, a while statement. UFO starts where those end. It is an open-source runtime for AI agents built for a team: members hand agents work in chat, agents read files and run commands to finish it, and every turn survives a crash. The same runtime powers the hosted service at ufo.ai, and this repository, ufo-ai/ufo-core, is the whole thing: the server, the Rust terminal client, the SDK, and the shipped extensions, licensed Apache-2.0.

The engineering shape of it is what earns a source tour. One command, `ufoctl serve`, runs the agent loop, the turn queue, the sandbox, the surfaces, and background jobs in a single process, on SQLite and local files for a laptop or Postgres, S3, and Redis for a fleet. Turns are DBOS workflows, which means the queue is durable by construction rather than by convention. Execution is a deliberate choice per conversation: in the default mode the connected client runs file reads, edits, and commands on your machine as your user; with `--remote` they run inside a configured sandbox carrier, a Seatbelt or Landlock-confined local carrier, Docker, or E2B.

And then there is the trust model, which is unlike anything else in this space. Agents never borrow the speaker's identity: a member grants an account to an agent through a connector grant made in chat, and the granting turn itself is the audit record. Every row in the database carries its workspace id, every query filters on it, and the Rust services that handle cache, egress, and preview hold no customer keys at all. Reading this code shows what it looks like to treat authorization as a first-class product feature rather than a warning in the docs.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ufo-core/ufo-ai-ufo-core-overview-architecture.svg" alt="Architecture overview of the ufo-ai/ufo-core repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the repository: the terminal client and chat surfaces feed one server process, which runs a durable turn queue and engine, drives the agent harness, and reaches sandboxes and model providers through extensions.*

Reading the overview from left to right: the Rust terminal client under `client/src` and the surface layer in `core/src/ufo/sdk/surfaces.py` both deliver messages into `core/src/ufo/serve.py`, the one-process server started by `ufoctl`. Turns are enqueued in `core/src/ufo/runtime/queue.py`, claimed and driven by the engine in `core/src/ufo/runtime/engine.py`, and executed by the agent harness in `core/src/ufo/harness/agent.py`. Memory and transcripts persist through `core/src/ufo/runtime/memory.py`, while tool execution reaches the sandbox carriers declared in `core/src/ufo/sdk/sandbox.py` and model providers arrive as extensions, like the OpenRouter adapter under `extensions/openrouter`.

## Why You Need This

The first problem is durability. A coding agent that dies mid-task because your laptop slept, your SSH dropped, or the process restarted is worse than no agent, because you cannot tell what it already did. UFO makes each turn a DBOS workflow, so the queue and the step state live in the database and a crashed turn resumes instead of vanishing. The quickstart's mental model is explicit: a session id confirms creation, not success, and you read durable state rather than resubmitting.

The second problem is execution boundaries. Sometimes you want the agent working on your actual machine with your actual permissions; sometimes you want it sealed in a box because it is reading untrusted input. UFO supports both as first-class modes of the same conversation, and a resumed conversation keeps the execution location it already has. The remote carriers are pluggable, the local carrier confines writes with Seatbelt on macOS or Landlock on Linux, and the documentation is honest about the limits: the default local carrier can read the whole host, so untrusted work belongs in Docker or E2B.

The third problem is shared access. A team agent needs to post to Slack, query a database, or open a pull request, and the usual answer, paste your tokens into the agent, is a security incident waiting to happen. UFO's answer is grants in chat: a member grants a specific account to an agent through a connector grant, the granting turn is the audit record, and workspace-owned accounts reach only the main agent until a member grants them onward. Authorization becomes a visible, reviewable conversation event.

The fourth problem is scale of surface area. A real deployment needs model providers, browser tools, memory, search, scheduled tasks, document handling, and connectors, and hard-wiring those turns a runtime into a product. UFO takes the opposite approach: everything is an extension, including tools, connectors, subagents, surfaces, model providers, and sandbox carriers. The repository ships dozens under `extensions/`, from browserbase and composio to mcp, rag, and redis_hub, and the default `assistant` pack simply names which ones a deployment activates.

## How It Works

The codebase is layered so that product concerns and agent mechanics never share a file, and each layer is small enough to read in one sitting.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ufo-core/ufo-ai-ufo-core-architecture.svg" alt="Detailed architecture of the ufo-ai/ufo-core repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: surfaces and the one-process server, the durable turn machinery, the harness internals, the extension SDK, and the storage and Rust services beneath.*

### Understanding the Architecture

**The server is intentionally boring.** `core/src/ufo/serve.py` and the `ufoctl` CLI in `core/src/ufo/cli.py` wire the process together: configuration, the database from `core/src/ufo/db.py`, the turn queue, surfaces, jobs, and the ingress that serves agent-built sites on subdomains of `ufo.localhost:8100`. There is one process to start, one configuration file, and the same bundle serves a laptop or a fleet, which is why the quickstart is five commands long.

**Turns are durable by construction.** The queue in `core/src/ufo/runtime/queue.py` hands work to the engine in `core/src/ufo/runtime/engine.py`, which claims turns as DBOS workflows, records progress in `core/src/ufo/runtime/steps.py`, appends events to `core/src/ufo/runtime/transcript.py`, and streams replies back through `core/src/ufo/runtime/delivery.py` to whichever surface the member is using. Background tasks and scheduled jobs in `core/src/ufo/runtime/background_tasks.py` and `core/src/ufo/runtime/jobs.py` flow through the same machinery, so a nightly research job and a chat message are the same kind of durable work.

**The harness is product-neutral.** Under `core/src/ufo/harness`, `agent.py` defines the message types and the agent loop, `rounds.py` runs one model round, `context.py` owns the window and compaction, and `durability.py` keeps turns crash-safe. The module `untrusted.py` exists purely to handle model output that must never be trusted, and `tools.py` defines how tool calls are issued and resolved. Because this layer has no product vocabulary, it can be read, tested, and reused without dragging the rest of the system along.

**Assembly happens per turn.** `core/src/ufo/host/assemble.py` builds the prompt and tool set for each turn from the environment in `core/src/ufo/host/environment.py` and whatever extensions the active pack declares. An extension is a Python package that imports only `ufo.sdk` and declares one manifest, exercised end to end by `extensions/sample`; the SDK pieces in `core/src/ufo/sdk/manifest.py`, `core/src/ufo/sdk/models.py`, and `core/src/ufo/sdk/sandbox.py` are the contract those manifests program against. `ufoctl ext search, install, remove` manages the installed set, so capability changes are deployment decisions, not code changes.

**Authorization is data, not ceremony.** The grant machinery in `core/src/ufo/sdk/grants.py` and `core/src/ufo/sdk/connectors.py` records which accounts an agent may use, with secrets held by `core/src/ufo/sdk/credentials.py` inside the runtime. Because a grant is created by a chat turn, the audit log and the authorization are the same object, and the engine consults grants before a tool call touches an external account.

**Storage scales with you, and Rust does the perimeter.** `core/src/ufo/schema` and `core/src/ufo/db.py` define the typed rows, workspace-scoped by default, served from SQLite and local files or from Postgres with S3 and Redis under it. The Rust services under `servers/`, cache, egress, and preview, sit at the edges where throughput matters, and they deliberately hold no customer keys; every secret stays inside the Python runtime. A session debugger and `ufoctl bundle`, which freezes a deploy into one artifact, round out the operations story.

Follow one request end to end: a member types a task in Slack or the terminal; the surface layer posts it into `ufoctl serve`, which enqueues a durable turn; the engine claims it, assembles the prompt and tools from the active pack, and the harness runs model rounds through a provider extension until the work is done; tool calls execute through the configured sandbox carrier, or on the member's machine in local mode; every step, transcript event, and memory update is written to the workspace-scoped database, and replies stream back to the surface while the turn is still running.

## Advantages

- **Crash-safe by design.** Turns are DBOS workflows with recorded steps, so a restart resumes work instead of losing it, and clients can leave and resume with `--resume`.
- **Two honest execution modes.** Local mode runs as you on your machine for trusted work; `--remote` moves execution into a sandbox carrier, with the trade-offs documented rather than hidden.
- **Chat-native authorization.** Connector grants made in chat delegate accounts to agents, with the granting turn as the audit record and no identity borrowing, ever.
- **Everything is an extension.** Tools, connectors, subagents, surfaces, model providers, and sandbox carriers all plug in through one manifest contract, and packs name what a deployment activates.
- **One process, laptop to fleet.** The same bundle serves SQLite-plus-files on a laptop and Postgres-plus-S3-plus-Redis in production, with `ufoctl bundle` producing a single deploy artifact.
- **A real operations story.** Ingress for agent-built sites, a session debugger for turns and memory, and Rust edge services that hold no customer keys.

## Benefits

- **Your team's context stays yours.** Workspaces, transcripts, memory, and credentials live on infrastructure you control, whether that is a laptop, an on-prem box, or your own fleet.
- **Delegation without credential sprawl.** Members grant exactly the accounts an agent needs, for exactly as long as the grant stands, instead of pasting tokens into prompts.
- **Untrusted work can be contained.** Docker and E2B carriers give you a real boundary for agent runs over untrusted input, with the local carrier's limits stated plainly in the security model.
- **Capability changes are configuration.** Installing an extension or switching model provider is a deployment decision managed by `ufoctl ext`, not a fork of the runtime.
- **The design is written down.** A substantial `spec.md` is the source of truth, so the architecture is arguable and reviewable rather than folkloric.
- **It is the hosted product.** The code that runs ufo.ai is the code you run, which means bug fixes and features land in the open repository first.

## Usage

To self-host on SQLite and local files, with no Docker, install [uv](https://docs.astral.sh/uv/), then:

```sh
make install
cp .env.template .env   # set the three model API keys
make build
make init EMAIL=email@work.com
make serve
```

`make init` creates the workspace with you as admin and writes a CLI token to `~/.ufoctl/token`; `make serve` listens on `http://localhost:8710`. In a second terminal, connect the client:

```sh
mkdir -p ~/.ufo && install -m 600 ~/.ufoctl/token ~/.ufo/credentials
echo http://localhost:8710 > ~/.ufo/workspace
./client/target/debug/ufo "what can you do?"
```

By default the client executes file reads, edits, and commands on your machine as your user. Start conversations with `ufo --remote` to run them in the workspace's configured sandbox instead, use `--wait SECONDS` to leave early, and `--resume ID` to read the rest of a finished turn. To use the hosted service instead, install the client with `curl -fsSL https://ufo.ai/ufo | sh` and sign in.

For a multi-instance deployment, point the configuration at Postgres and Redis:

```sh
make db
uv run python -c 'from ufo.cli import DEFAULT_CONFIG; print(DEFAULT_CONFIG, end="")' \
  | sed 's#sqlite+aiosqlite:///ufo.db#postgresql+asyncpg://ufo:ufo@127.0.0.1:5541/ufo#' > ufo.toml
```

Extensions are ordinary Python packages that declare one entry point:

```toml
[project.entry-points."ufo.extension"]
acme = "ufo_ext_acme.manifest:manifest"
```

`ufoctl ext search`, `ufoctl ext install`, and `ufoctl ext remove` manage the installed set, and `extensions/sample` exercises every manifest point if you want a template. For development, `make check` runs the static gates and `make test` the parallel suite, while `make test-integration` covers the serial and Docker passes.

## Conclusion

UFO core is one of the few agent projects where the interesting engineering is not in the model loop at all. The loop is competent and unremarkable on purpose; the ambition is in the durable turn machinery, the chat-native grant system, the honest execution boundaries, and an extension contract that keeps the product from calcifying. If you are building agents for a team rather than a demo, this repository is worth an afternoon of reading, and quite possibly a deployment.

Links:

- Repository: https://github.com/ufo-ai/ufo-core
- Documentation: https://ufo.ai/docs/
- Hosted service: https://ufo.ai
