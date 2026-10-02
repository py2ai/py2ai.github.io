---
layout: post
title: "Foremerge: Catch Intent Conflicts Before Code Conflicts - Inside naw103/foremerge"
description: "Foremerge is an open-source coordination protocol for coding agents, built above Git. We tour the Rust source behind its deterministic conflict detector, verification-gated lifecycle, SQLite event ledger, and MCP server that lets parallel agents share intent instead of undoing each other's work."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /Foremerge-Catch-Intent-Conflicts-Before-Code-Conflicts/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/foremerge/naw103-foremerge-architecture.svg
tags:
  - Rust
  - Git
  - MCP
  - AI Agents
categories: [AI, Open Source]
keywords: "foremerge, coding agent coordination, git worktrees, conflict detection, MCP server, Rust CLI, SQLite event log, semantic scopes, verification gate, multi-agent development"
author: "PyShine"
---

Run two AI coding agents on the same repository at the same time and you will eventually meet the merge that should not have happened. Each agent works in its own copy of the code, each diff looks reasonable, Git merges both without complaint, and only later do you discover that one agent removed an extension point the other was quietly extending. Git compares text; it has no way to compare intent. That blind spot is precisely the problem [naw103/foremerge](https://github.com/naw103/foremerge) sets out to fix.

Foremerge is an open-source coordination protocol for coding agents, built above Git. Agents keep working in isolated worktrees, but before they touch code they announce what they are about to change: the semantic scopes they intend to modify, the operation they intend to perform, and the work they depend on. Every agent on the machine reads from one shared picture, so two colliding plans are caught while both worktrees are still clean. The project describes itself as a pre-1.0, local-first MVP, and version 0.5.0 already ships the full stack: a CLI, a JSON API, an MCP server, a SQLite store, a deterministic conflict detector, and a verification-gated lifecycle.

The source is worth a tour because it takes an unusually disciplined position on a fuzzy problem. Instead of asking a model to judge whether two plans conflict, Foremerge compares declared operations with deterministic rules, so the same inputs always produce the same verdict. Instead of locking files, it raises advisory warnings and leaves the human in charge. And instead of scattering state across config files, it keeps a hash-chained event ledger inside the repository's Git common directory, where every worktree on the machine sees the same truth. Let us walk through the architecture.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/foremerge/naw103-foremerge-overview-architecture.svg" alt="Architecture overview of the naw103/foremerge repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Foremerge codebase: three entry points (CLI, MCP, JSON API) converge on a single Foremerge service, which gates work through the conflict detector and verification registry while persisting projections and a hash-chained event log in SQLite beside Git.*

Reading the overview from left to right: agents working in isolated Git worktrees announce their intent through whichever surface they prefer, the CLI, the MCP stdio adapter, or the Axum JSON API. All three are thin transports that call the same Foremerge service, which is where every business rule lives. The service consults the deterministic conflict detector to compare declared scopes before code is written, consults the verification registry to decide whether work may be accepted, and persists everything through the SQLite store, with a ledger recovery module for operator safety. Git integration rounds out the picture by discovering repositories, taking snapshots, and creating the worktrees themselves.

## Why You Need This

The failure mode Foremerge targets is easy to reproduce and expensive to debug. Agent A decides to replace a class, agent B decides to extend it, and neither edit touches the same line. Git's three-way merge sees no overlap, both worktrees land cleanly, and the codebase is left with a stranded class and a half-finished migration. Text-based tooling cannot warn you about this because the collision is semantic, not lexical. The README's running example is a `PaymentService` being replaced by one agent while another adds PayPal support to it, and nothing in either diff would flag the clash.

The second problem is coordination overhead. Once you run more than a couple of agents, you become the integration point yourself: reading both agents' plans, spotting the overlap, and deciding who goes first. Foremerge turns that into a protocol. Each agent publishes its intent with semantic scopes, such as a symbol, an API surface, a schema, or a config file, and the store compares new intents against everything already published. When two declarations collide, you get a named advisory that explains both sides and suggests a split, such as coordinating on a stable abstraction, before either agent has written a line.

The third problem is trust. An agent saying "my work is done, tests pass" is a claim, not a fact. Foremerge makes acceptance verification-gated: the operator registers named checks, and the service runs the check itself rather than taking the agent's word. Work accepted without a runnable check is recorded as unverified, with the reason attached, so the audit trail never implies a check ran when none did.

Finally, there is the crash problem. Any coordination scheme that locks files turns one crashed agent into a stalled fleet. Foremerge deliberately never locks or blocks; its warnings are advisory and stale claims do not wedge the system. And because the conflict detector is deterministic, you can rerun a scenario tomorrow and get the same verdict, which is exactly what you want from infrastructure other agents act on.

## How It Works

Foremerge is a single Rust package whose module boundaries are the whole design, so the detailed diagram maps almost one-to-one onto files in `src/`.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/foremerge/naw103-foremerge-architecture.svg" alt="Detailed architecture of the naw103/foremerge repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of Foremerge: binaries and transports on the left, the coordination core in the middle, the SQLite store and Git boundary on the right, with tests, benchmark scenarios, and protocol documentation anchoring the edges.*

### Understanding the Architecture

**One service behind every door.** [src/service.rs](https://github.com/naw103/foremerge/blob/main/src/service.rs) exposes the Foremerge application service, and [src/main.rs](https://github.com/naw103/foremerge/blob/main/src/main.rs) plus [src/bin/fmg.rs](https://github.com/naw103/foremerge/blob/main/src/bin/fmg.rs) are deliberately thin wrappers over the shared command layer in [src/cli.rs](https://github.com/naw103/foremerge/blob/main/src/cli.rs). The MCP adapter in [src/mcp.rs](https://github.com/naw103/foremerge/blob/main/src/mcp.rs) and the Axum handlers in [src/api.rs](https://github.com/naw103/foremerge/blob/main/src/api.rs) likewise translate requests and hand them to the same service. Business rules never live in a transport, so an agent that connects over MCP sees exactly the behavior an HTTP client sees.

**Intent, declared before code.** [src/model.rs](https://github.com/naw103/foremerge/blob/main/src/model.rs) defines the domain types, including scopes that carry both a target, such as a symbol or a schema, and an operation, such as replace or extend. Because the operation is declared rather than inferred from a summary, two agents that phrase the same plan differently still reach the same verdict. The paraphrase probe in [tests/paraphrase_probe.rs](https://github.com/naw103/foremerge/blob/main/tests/paraphrase_probe.rs) exists precisely to pin that property down.

**Deterministic conflict rules.** [src/conflict.rs](https://github.com/naw103/foremerge/blob/main/src/conflict.rs) implements the intent analysis: it compares declared scopes and operations and raises advisories with explainable evidence, like suggesting a shared abstraction when one agent replaces a class another extends. There is no model call anywhere in this path, which keeps verdicts reproducible. The benchmark scenarios under [benchmarks/scenarios](https://github.com/naw103/foremerge/tree/main/benchmarks/scenarios) cover the classic shapes, including a payment provider conflict, a schema rename, and a negative control where two plans must not collide.

**A store with three kinds of memory.** [src/db.rs](https://github.com/naw103/foremerge/blob/main/src/db.rs) manages a SQLite database kept under the repository's Git common directory, so linked worktrees resolve the same coordination state. It maintains domain projections for direct queries, a materialized semantic graph relating agents, tasks, intents, scopes, changesets, tests, and decisions, and an append-only event journal where every mutation appends a row with a monotonic sequence, the previous hash, and a SHA-256 event hash, with SQLite triggers rejecting updates and deletes. [src/ledger.rs](https://github.com/naw103/foremerge/blob/main/src/ledger.rs) adds operator recovery, so a ledger written by a newer build can be set aside or restored without silently losing history.

**Verification-gated acceptance.** [src/checks.rs](https://github.com/naw103/foremerge/blob/main/src/checks.rs) holds the registry of named checks an operator trusts, such as a build or a typecheck, and acceptance runs the registered check rather than trusting the agent. [src/exclusions.rs](https://github.com/naw103/foremerge/blob/main/src/exclusions.rs) encodes the operator-owned exclusion policy for validation, and anything accepted without a runnable check is recorded as unverified with its reason.

**Setup as part of the protocol.** [src/integrations.rs](https://github.com/naw103/foremerge/blob/main/src/integrations.rs) installs the native skill and MCP entries for supported clients while preserving unrelated configuration, and [src/git.rs](https://github.com/naw103/foremerge/blob/main/src/git.rs) handles repository discovery, snapshots, and the worktree lifecycle that keeps each agent isolated. The end-to-end flow: an agent registers, publishes an intent with scopes, claims the scope, checks conflicts, starts work in its own worktree, publishes a changeset, passes the verification gate, and only then is the work accepted and committed, with every step appending a hash-chained event along the way.

## Advantages

- **Catches the merge Git cannot see.** Declared semantic scopes surface replace-versus-extend style clashes before either worktree changes, not after.
- **Deterministic by design.** Conflict verdicts come from fixed rules over declared operations, so identical inputs always yield identical answers, and paraphrase tests enforce it.
- **Advisory, never a lock.** Warnings name both agents and suggest a split, but nothing blocks, so one crashed agent cannot stall the fleet.
- **Verification over trust.** Acceptance runs operator-registered checks itself, and unverified acceptances are recorded with reasons rather than passing silently.
- **Auditable by construction.** An append-only, hash-chained event journal with SQLite triggers against mutation gives you an audit trail you can reason about.
- **One behavior, three transports.** CLI, MCP, and JSON API all delegate to the same service, eliminating drift between how humans and agents coordinate.

## Benefits

- **Fleet-friendly workflow.** Isolated worktrees plus a shared coordination store let several agents make progress on one repository without stepping on each other's files or plans.
- **Early, cheap conflict feedback.** Collisions are caught at intent time, when the fix is a conversation, instead of at merge time, when the fix is rework.
- **Client-agnostic setup.** One setup command wires skill files and MCP entries for supported coding agents while leaving unrelated configuration untouched.
- **Local-first operation.** State lives in your repository's Git common directory and a local SQLite file; there is no broker or external service in the loop.
- **Recoverable state.** Ledger recovery can set aside or restore the coordination store while refusing to run alongside another process that holds it open.
- **Documented protocol surface.** Architecture, protocol, state model, conflict detection, and MCP setup guides, plus an OpenAPI schema, live alongside the code in `docs/`.

## Usage

Install a prebuilt, checksum-verified release binary (macOS and Linux; Windows binaries are on the releases page), or build from source with Rust 1.85+:

```sh
curl -fsSL https://foremerge.com/install.sh | sh
```

```sh
cargo install --locked --git https://github.com/naw103/foremerge foremerge
```

The same binary answers to `foremerge` and its short alias `fmg`. Inside the repository you want to coordinate:

```sh
foremerge init
foremerge setup all
foremerge checks set test -- cargo test --all-targets
foremerge doctor --client all
```

`init` creates the local coordination state under the Git common directory without touching tracked files. `setup all` installs the native skill and MCP entry for each supported client, and `checks set` registers the named check the acceptance gate will run itself. If a repository has nothing meaningful to verify, switch the policy instead of registering a check that always passes:

```sh
foremerge checks policy advisory
```

Work accepted under that policy is recorded as unverified with the reason, keeping the audit trail honest. For HTTP-based tooling, the JSON API's contract is published in `docs/openapi.yaml`, and the MCP setup guide in `docs/mcp-setup.md` walks through wiring clients and upgrading without losing your configuration.

## Conclusion

Foremerge takes a problem that sounds like it needs machine intelligence, judging whether two agents' plans clash, and solves it with something better suited to infrastructure: declared operations, deterministic rules, and a verifiable ledger. The result is a coordination layer that treats Git as the durable repository and adds the missing shared awareness above it, without locking, without an LLM in the critical path, and without asking you to change where your code lives. For teams running multiple coding agents against one repository, it is one of the more thoughtfully engineered takes on the problem we have toured, and the protocol documentation makes the design decisions easy to evaluate for yourself.

Links:

- GitHub repository: [naw103/foremerge](https://github.com/naw103/foremerge)
- Architecture guide: [docs/architecture.md](https://github.com/naw103/foremerge/blob/main/docs/architecture.md)
- Protocol specification: [docs/protocol.md](https://github.com/naw103/foremerge/blob/main/docs/protocol.md)
