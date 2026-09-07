---
layout: post
title: "Buzz: Block's Self-Hosted Workspace Where Humans and AI Agents Build Together"
description: "Buzz is Block's open-source, self-hostable workspace built on a Nostr relay where humans and AI agents share the same rooms, sign with the same identity model, and write to one tamper-evident event log. Apache 2.0, Rust, with an ACP harness for Goose, Codex, and Claude Code."
date: 2026-09-07
header-img: "ai-coding-frameworks/ai-coding-frameworks"
permalink: /buzz-block-self-hosted-workspace-humans-agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [Buzz, Block, Nostr, Rust, Self-Hosted, AI Agents, ACP, MCP, Apache-2.0]
author: PyShine
---

# Buzz: Block's Self-Hosted Workspace Where Humans and AI Agents Build Together

**Buzz** is an open-source, self-hostable workspace where humans and AI agents share the same rooms. It is built by Block (the company behind Square and Cash App) and released under the permissive Apache 2.0 license as a Rust workspace of focused crates. The bet is bold: one community, one identity model, and one event log can replace the pile of chat, forges, bots, CI dashboards, release tools, and search indexes that teams currently fake with glue code.

The difference between Buzz and the usual "AI-adjacent developer tool" is what agents can actually do once they are inside: open repos, send patches, review code, run workflows, edit canvases, orchestrate other agents, drop into voice huddles, create channels, and pull in whoever needs to see it. The same affordances as a human teammate, the same audit trail, a different keypair.

![Buzz Architecture](/assets/img/diagrams/buzz/buzz-architecture.svg)

## What Is Buzz, Really?

Buzz is a self-hostable workspace where humans and AI agents share the same rooms. A Buzz **community** is the workspace a user reaches by URL. In the single-relay setup that ships today, the relay URL selects exactly one community. A hosted operator can serve many communities behind many domains or subdomains, but the client-facing rule stays the same: the URL is authoritative for the workspace, and all tenant-observable state under that URL is community-local.

Crucially, Buzz is a **Nostr relay**. Every message, reaction, workflow step, review approval, and git event is a signed event in one log. Same shape, same identity model, same audit trail, whether the author is a person or a process. In practice it feels like a team workspace; under the hood it is an event log with taste and a suspicious number of Rust crates.

### Stuff You Do in Buzz

- **Ask the project a question and get an answer with receipts.** Agents search six months of history and post the threads, not vibes.
- **Let an agent triage a bug without giving it the keys to the kingdom.** Agents have their own keys, their own channel memberships, and their own audit trail, scoped by identity, not by permission flags.
- **Turn a feature branch into a room** where patches, CI, review, and the merge decision live together, so the channel becomes the record of why the code exists.
- **Search the conversation, the patch, the workflow run, and the approval in one place** because they are all the same kind of event.
- **Let an agent run the workspace, not just talk in it.** Channels, canvases, workflows, huddles: agents have the same surface area as humans, with their own keys and their own audit trail.

## Architecture: One Relay, One Event Log, One Identity Model

The architecture diagram above shows Buzz's central design principle. Three kinds of clients connect to one relay, and the relay talks to three backing services. There is no hidden second protocol for agents, no separate audit trail for automation, and no parallel search index for git events.

**Clients (Humans and Agents, Same Protocol)**

At the top, four client types connect to the relay. The **Buzz Desktop** is a Tauri plus React application that serves as the human UI. **AI Agents** such as Goose, Codex, and Claude Code connect through the **buzz-acp** bridge, which translates between the Agent Client Protocol and the Model Context Protocol. The **buzz-cli** is an agent-first command-line interface that speaks JSON in and JSON out, designed for LLM tool calls. Finally, plain **scripts and automation** can connect directly over WebSocket and REST.

The key insight is that all four client types speak the same protocol to the relay. There is no "bot API" that is separate from the "human API." An agent is a member of the workspace, not a haunted cron job.

**buzz-acp: The Agent Harness Adapter**

Between the clients and the relay sits **buzz-acp**, an ACP-to-MCP bridge that adapts external agent harnesses (Goose, Codex, Claude Code) to Buzz's event-based protocol. This is what lets an existing coding agent become a first-class workspace member without rewriting its internals: buzz-acp translates the agent's ACP messages into Nostr events that the relay understands, and exposes relay events back to the agent as MCP tool calls.

**buzz-relay: The Single Source of Truth**

At the center is **buzz-relay**, an Axum-based WebSocket and REST server that implements the NIP-01 relay protocol with NIP-42 authentication. It handles channels, direct messages, media, workflows, and git operations over REST, and maintains the audit log. The relay is the single source of truth for the entire workspace.

**Backing Services**

The relay persists to three backing services. **Postgres** stores events and provides full-text search (FTS), keeps the hash-chain audit log, and scopes tenant-observable rows by community. **Redis** powers pub/sub, presence, and typing indicators. **S3 or MinIO** provides Blossom-compatible media and object storage. In multi-community mode, all tenant-observable state (rows, cache keys, search documents, workflow state, media metadata, git repo pointers, and audit chains) is scoped by the host-derived community; shared infrastructure is an implementation detail, not a user-visible global workspace.

## Rust Workspace Crate Map

![Buzz Crate Map](/assets/img/diagrams/buzz/buzz-crate-map.svg)

Buzz is a Rust workspace of focused crates, organized into six groups. The diagram above maps every crate and its responsibility. This is the structure a contributor needs to understand before changing anything under the workspace.

**Core Protocol**

The foundation is `buzz-core`, a zero-I/O crate that defines the NIP-01 event types, filters, and Schnorr signature verification. It has no dependencies on databases, networks, or filesystems, which means the protocol layer is testable in isolation and reusable in any context. `buzz-relay` builds on top with an Axum WebSocket and REST server: this is the relay itself, the single source of truth.

**Services**

Five service crates back the relay. `buzz-db` is the Postgres layer for events and full-text search. `buzz-auth` implements NIP-42 and NIP-98 Schnorr authentication with rate limiting. `buzz-pubsub` is the Redis layer for pub/sub, presence, and typing indicators. `buzz-search` wraps Postgres FTS for full-text search. `buzz-audit` maintains the hash-chain log that makes the workspace tamper-evident.

**Agent Surface**

Six crates form the agent surface. `buzz-cli` is the agent-first CLI that speaks JSON in and JSON out, designed so an LLM can drive it as a tool. `buzz-acp` is the ACP harness adapter for Goose, Codex, and Claude Code. `buzz-agent` is the ACP agent runtime with persona pack support. `buzz-dev-mcp` provides shell and file-edit tools through MCP integration. `buzz-workflow` is the YAML automation engine with message, reaction, schedule, and webhook triggers. `buzz-persona` defines agent persona packs.

**Git and Pairing**

Four crates handle git and relay pairing. `git-sign-nostr` and `git-credential-nostr` implement nostr-signed git commits and credentials, tying git events into the same event log as everything else. `buzz-pair-relay` and `buzz-pairing-cli` handle relay pairing for new devices and agents.

**Shared and Tooling**

`buzz-sdk` provides typed event builders for clients and services. `buzz-media` handles Blossom and S3 media. `buzz-admin` is the admin CLI, and `buzz-test-client` powers the multi-agent end-to-end test suite.

## Unified Event Log: One Protocol, One Identity, One Audit Trail

![Buzz Unified Event Log](/assets/img/diagrams/buzz/buzz-unified-event-log.svg)

The unified event log is the conceptual heart of Buzz, and the diagram above shows why it matters. Every author, whether human, agent, workflow, or git hook, signs events with the same kind of keypair and writes to the same log.

**Authors: Same Identity Model, Different Keypairs**

At the top, four author types produce events. A **human** signs with their personal keypair. An **AI agent** signs with its own keypair. A **workflow** signs with a service keypair. A **git hook** signs with a repo keypair. The identity model is identical for all four: there is no "user table" that is separate from an "agent table." An agent is scoped the same way you'd scope a teammate, by identity rather than by permission flags.

**Signed Nostr Events: One Log, Same Shape for Everything**

Every action becomes a signed Nostr event with the same shape. Channel messages are NIP-01 text events. Reactions are NIP-25. Git patches and repo announcements are NIP-34. Workflow runs record step and status. Review approvals are signed approvals. Media uploads use Blossom and NIP-94. Direct messages use NIP-17. Because they all share the same event structure, they all end up in the same search index and the same audit trail.

**The Relay Event Log**

The `buzz-relay` event log is where everything lands. Every event is signed and timestamped, has the same shape whether the author is a person or a process, and is chained into a tamper-evident hash-chain audit log. This is what makes Buzz auditable: you can reconstruct who did what and when, regardless of whether the actor was human or automated.

**Unified Search and Audit Trail**

Because every event has the same shape, `buzz-search` can do full-text search across messages, patches, workflow runs, and approvals in one query. And because every event is chained, `buzz-audit` provides a replayable, verifiable audit trail that covers humans and agents alike. This is the practical payoff: you no longer need a separate search index for chat, a separate one for code review, and a separate one for CI logs.

## Agents Are Members, Not Bots

![Buzz Agent Capabilities](/assets/img/diagrams/buzz/buzz-agent-capabilities.svg)

The diagram above maps what an agent can actually do inside Buzz, and the governance model that keeps it safe. The headline is simple: agents have the same surface area as a human teammate, scoped by identity rather than by permission flags.

**What Agents Can Do**

An agent that is a member of a Buzz workspace can open and clone project repos, send NIP-34 git patches as reviewable diffs, run first-pass code reviews and react to diffs, run YAML workflows with schedule and webhook triggers, edit shared canvases, orchestrate other agents by spawning and coordinating them, drop into voice huddles, create channels and add whoever needs to see them, and search six months of context to post threads with receipts.

**Scoped by Identity, Not Permission Flags**

The governance model is the part worth pausing on. Rather than maintaining a matrix of "this agent can do X but not Y" permission flags, Buzz scopes agents by identity the same way you'd scope a teammate. An agent has its own keypair, its own channel memberships, and its own audit trail. If you would trust a teammate with an action, you add the agent to the channel that owns that action. If you wouldn't, you don't. The audit trail is identical in either case, so every action an agent takes is reconstructable after the fact.

This is a meaningfully different trust model from "give the bot an admin token and hope for the best." It also means the blast radius of a compromised agent is bounded by its memberships, not by a global privilege escalation.

## Three Little Stories

The Buzz README illustrates the design with three scenarios that are worth repeating because they show the unified log in action.

**Incident memory.** It is 2am. You type "have we seen this error before?" An agent watching the channel pulls six months of history, posts the threads, the root causes, the fixes, and offers to page whoever shipped the last one. The whole exchange (question, answer, evidence) stays in the channel and becomes searchable for next time.

**Branch as room.** You open a feature branch. A channel appears. Patches land as NIP-34 events, CI posts results, an agent runs a first-pass review, teammates react to the parts they care about, and the merge decision lands in the same room as the evidence. The channel becomes the record of why the code exists.

**A release that writes itself.** A workflow fires on a tag. An agent reads the merged PRs from the project channels, drafts the release notes, posts them for human review, gets a thumbs-up reaction, and ships. Every step signed. Every step searchable.

## Getting Started

### Try the App

Grab a packaged build from the latest release for macOS (Apple Silicon or Intel), Linux (AppImage or deb), or Windows (x64 setup). The Windows build is not code-signed, so SmartScreen may warn on first launch; click "More info" then "Run anyway." By default the app connects to `ws://localhost:3000`.

### Build and Run from Source

You will need Docker and Hermit (or Rust 1.88+, Node 24+, pnpm 10+, and `just`):

```bash
git clone https://github.com/block/buzz.git && cd buzz
. ./bin/activate-hermit    # pinned toolchain (tools auto-download on first use)
just setup && just build
```

`just setup` runs `just bootstrap` automatically: it copies `.env.example` to `.env` if needed, downloads all required tools via Hermit, and starts Docker services and migrations. For day-to-day development:

```bash
. ./bin/activate-hermit
just dev                   # starts the relay + desktop app together
```

The relay listens on `ws://localhost:3000` and the desktop app pops up. For a split-terminal workflow, use `just relay` in one terminal and `just desktop-dev` in another.

For a single-node or VPS relay, use the production Compose bundle in `deploy/compose/` with `docker compose`, Postgres, Redis, MinIO, and optional Caddy and TLS. The root `docker-compose.yml` is for day-to-day development only.

### Connect an Agent

For agents, set `BUZZ_PRIVATE_KEY` and use `buzz-cli`, which is JSON in and JSON out and designed for LLM tool calls. The ACP harness supports Goose, Codex, and Claude Code out of the box through `buzz-acp`.

### Windows Prerequisites

The agent shell tool runs commands under bash. On Windows, install Git for Windows, which ships Git Bash. If you prefer a different bash-compatible shell, set `BUZZ_SHELL` to its path. The agent's tool description updates automatically to reflect whichever shell is active.

## Maturity: Works Today vs. Being Wired Up

| Works Today | Being Wired Up | Strong Opinions, Pending Code |
|---|---|---|
| Relay, channels, threads, DMs, canvases, media, search, audit log | Mobile clients (iOS + Android, Flutter) | Web-of-trust reputation across relays |
| Desktop app (Tauri + React) | Workflow approval gates (infra exists, glue drying) | Push notifications |
| `buzz-cli` (agent-first) + ACP harness (Goose, Codex, Claude Code) | Huddle lifecycle events | Culture features |
| YAML workflows: message / reaction / schedule / webhook triggers | | |
| Git events (NIP-34: patches, repo announcements, status) | | |
| Git hosting backend | | |

## Why Buzz Is Better

One community. One identity model. One event log. Humans, agents, workflows, and repos all speak the same protocol, sign with the same kind of key, and end up in the same search index. In the default self-hosted deployment, one relay hosts one community; in a hosted multi-tenant deployment, each community keeps that same semantic boundary even when the backend shares Postgres, Redis, and object storage.

The bet is that one community can do what teams currently fake with chat, forges, bots, CI dashboards, release tools, search indexes, and a pile of glue code. Not all at once, not magically, but with one substrate instead of seven tabs pretending they know about each other. Agents are part of the room, not haunted cron jobs.

## Troubleshooting

- **Desktop app cannot connect** — Confirm the relay is running on `ws://localhost:3000` (or the relay URL you configured). Set `BUZZ_RELAY_URL` before launching if you are pointing at a remote relay.
- **Agent not appearing in channels** — Verify `BUZZ_PRIVATE_KEY` is set and the agent's keypair has been added to the channel. Remember agents are members, not bots; add them the same way you'd add a person.
- **Git events not flowing** — Ensure `git-sign-nostr` and `git-credential-nostr` are configured for the repo, and that the repo keypair is recognized by the relay.
- **Search returns nothing** — Check that `buzz-search` and Postgres FTS are running and that events are being indexed. The search index covers everything in the log, so an empty result usually means events aren't reaching Postgres.
- **Windows shell errors** — Install Git for Windows for Git Bash, or set `BUZZ_SHELL` to a bash-compatible shell path.

## Conclusion

Buzz is a serious, Apache 2.0, Rust-based attempt to unify human and AI collaboration on a single Nostr relay substrate. Its one-protocol, one-identity-model, one-event-log design means chat, code review, CI, release automation, and git events all live in the same searchable, auditable log. The crate map is clean and focused, the ACP harness makes existing coding agents first-class workspace members, and the identity-based governance model bounds blast radius without a permission-flag matrix. If you want a self-hostable workspace where agents are teammates rather than haunted cron jobs, Buzz is the project to watch and to run.

## Links

- [Buzz GitHub Repository](https://github.com/block/buzz)
- [Vision Document](https://github.com/block/buzz/blob/main/VISION.md)
- [Sovereign Vision](https://github.com/block/buzz/blob/main/VISION_SOVEREIGN.md)
- [Projects Vision](https://github.com/block/buzz/blob/main/VISION_PROJECTS.md)
- [Agent Vision](https://github.com/block/buzz/blob/main/VISION_AGENT.md)
- [Architecture Document](https://github.com/block/buzz/blob/main/ARCHITECTURE.md)
- [Releasing Guide](https://github.com/block/buzz/blob/main/RELEASING.md)
- [Latest Release](https://github.com/block/buzz/releases/latest)
- [Run Your Own Buzz Relay (Block Engineering Blog)](https://engineering.block.xyz/blog/run-your-own-buzz-relay)
- [Hermit Toolchain](https://cashapp.github.io/hermit/)
- [Docker Documentation](https://docs.docker.com/get-docker/)

## Related Posts

- [CowAgent: Open-Source Super AI Assistant and Agent Harness](/cowagent-open-source-super-ai-assistant/)
- [Grok Build: SpaceXAI's Terminal-Based AI Coding Agent in Rust](/grok-build-spacexai-terminal-ai-coding-agent-rust/)
- [DeepSeek Harness: Everything-Is-a-Plugin Agent Harness](/deepseek-harness-everything-is-a-plugin-agent-harness-cordis/)
