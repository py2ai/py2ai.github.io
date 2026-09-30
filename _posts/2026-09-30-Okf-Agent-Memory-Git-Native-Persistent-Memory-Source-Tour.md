---
layout: post
title: "okf-agent-memory: Git-Native Persistent Memory for AI Agents - Inside okf-memory/okf-agent-memory"
description: "A source tour of okf-agent-memory, a Go implementation of Google's Open Knowledge Format (OKF) v0.2 that turns a plain git-tracked knowledge/ folder into persistent memory for AI coding agents. We trace the zero-dependency core library, its in-memory BM25 search engine, the OKF v0.2 validator, and the Model Context Protocol server."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Okf-Agent-Memory-Git-Native-Persistent-Memory-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/okf-agent-memory/okf-memory-okf-agent-memory-architecture.svg
tags:
  - AI Agents
  - Agent Memory
  - Go
  - Open Source
categories: [AI, Open Source]
keywords: "okf-agent-memory, OKF v0.2, agent memory, AI coding agents, BM25 search, MCP server, git-native memory, AGENTS.md, knowledge management, dual-memory architecture, Go CLI, open knowledge format, Claude Code, Cursor"
author: "PyShine"
---

Every developer who has worked with an AI coding agent has felt the same frustration: the session ends, and the architectural decisions, domain discoveries, and hard-won operational facts evaporate with it. The common workarounds both fail in interesting ways. Stuffing everything into a monolithic `AGENTS.md` bloats the context window until the model starts ignoring the instructions that matter most. Pushing knowledge into a vector database fixes the bloat but creates a new blindspot, because agents rarely think to semantically search for operational constraints like formatting rules or security boundaries in the middle of a routine task.

**okf-agent-memory**, maintained under the okf-memory organization, takes a third path: it treats an ordinary, git-tracked `knowledge/` folder as the agent's long-term memory. The project is a Go implementation of the Open Knowledge Format (OKF) v0.2 — an open specification for agent knowledge that originated in Google's knowledge-catalog project — and it ships as a single static binary that can validate, search, mutate, and scaffold these knowledge bundles, as well as serve them over the Model Context Protocol (MCP) so agents like Claude Code or Cursor can query them as native tools.

In this source tour we will walk through the actual Go code: how the CLI dispatches commands, how the zero-dependency core library parses markdown concepts into an in-memory graph, how the BM25 scoring engine retrieves concepts in microseconds, and how the stdio MCP server exposes the whole thing to agents. The repository dogfoods its own design — the `knowledge/` directory in the repo root is itself a live OKF bundle — which makes the code unusually honest about what it can do.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/okf-agent-memory/okf-memory-okf-agent-memory-overview-architecture.svg" alt="Architecture overview of the okf-memory/okf-agent-memory repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the okf-agent-memory architecture: the CLI surface dispatches into the OKF core library, which loads the git-tracked knowledge bundle and serves it through search, validation, mutation, and MCP tooling, with an optional encrypted sync layer.*

Reading the overview from left to right: the `okf` binary starts at `cmd/okf/main.go`, hands arguments to the dispatcher in `internal/cli/root.go`, and resolves the requested subcommand through the registry in `internal/cli/registry.go`. Every command path converges on the OKF core library under `pkg/okf/`, where `bundle.go` loads the `knowledge/` directory into memory via the markdown and YAML parser in `parser.go`. On top of that shared foundation sit the BM25 search engine (`search.go`), the OKF v0.2 validator (`validator.go`), the mutation layer that maintains index and log bookkeeping (`mutate.go`), the bootstrap scaffolder (`bootstrap.go`), and the MCP server (`mcp.go`) that exposes the same operations as agent tools. Separately, the sync engine in `pkg/sync/engine.go` can push and pull the bundle to a remote hub as AES-256-GCM encrypted envelopes — but the memory itself never leaves plain markdown on your filesystem.

## Why You Need This

The first problem okf-agent-memory attacks is what the project calls the *prompt monolith*. When all domain knowledge lives in `AGENTS.md` or `CLAUDE.md`, every session pays the token cost up front, and the project's documentation in `docs/spec/DUAL_MEMORY_AGENT_ARCHITECTURE_RFC.md` argues that this causes attention drift — critical invariants get diluted by the volume of surrounding prose. The Dual-Memory Agent Architecture (DMAA) implemented here splits the problem in two: a compact normative codex stays in `AGENTS.md`, while the semantic domain knowledge lives in the `knowledge/` bundle and is pulled on demand. At session start, the pull layer costs zero tokens.

The second problem is the *RAG blindspot*. Behavioral rules — "never commit secrets", "always use this error-handling pattern" — are exactly the kind of content agents fail to retrieve semantically, because they never formulate a query that would match them. okf-agent-memory sidesteps this with **code-to-knowledge binding**: concepts can declare `code_refs` pointing at source paths or globs, and the `search --for-path` command (implemented as `SearchForPath` in `pkg/okf/search.go`) finds every concept that governs a file before you edit it. Results are ranked by governance level first — `hold` beats `constraint` beats `context` — so a frozen subsystem announces itself before the first line of code changes.

The third problem is trust in what agents write down. Because the memory is a set of plain markdown files, every concept carries OKF v0.2 metadata: `generated` records which agent authored a note and when, `verified` records human or process verification, `status` tracks draft/stable/deprecated lifecycle, and `stale_after` gives each fact an expiry date. The convention documented in `docs/spec/CONVENTION.md` explicitly forbids an agent from forging human verification. And because the whole memory is version-controlled text, reviewing what your agent "learned" this week is a `git diff` away — no dashboards, no opaque database exports.

Finally, there is the cost dimension. The retrieval layer is a purely local, in-memory BM25 index, so there are no embedding API calls and no vector database to operate. The benchmark suite under `benchmarks/` — with a runner in `cmd/okf-benchmark/main.go` — publishes the project's own measurements of sub-300-microsecond search latency against alternative memory runtimes; whatever your hardware yields, the structural point holds: retrieval is a function call over parsed markdown, not a network round trip.

## How It Works

The cleanest way to understand the system is to follow one bundle from disk to agent response, so let us open up the detailed architecture.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/okf-agent-memory/okf-memory-okf-agent-memory-architecture.svg" alt="Detailed architecture of the okf-memory/okf-agent-memory repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the okf-agent-memory codebase: the CLI command registry fans out to the OKF core library — loader, parser, types, BM25 search, filter, validator, mutations, MCP server, and bootstrap — while the vault and sync packages handle encrypted remote synchronization.*

### Understanding the Architecture

**The CLI shell is a thin, explicit dispatcher.** `cmd/okf/main.go` stamps version, commit, and date from Go build info and calls `cli.Execute`, which lives in `internal/cli/root.go`. There is no cobra or viper here: the dispatcher reads the first argument, resolves it against the registration table in `internal/cli/registry.go`, and delegates. Notably, if no bundle path is given, the helper `defaultBundle` falls back to a `knowledge/` directory in the current project, so `./bin/okf validate --strict` just works inside a DMAA-shaped repository. The registered commands cover the full lifecycle: `validate`, `search`, `show`, `create`, `update`, `relate`, `init`, `bootstrap`, `agents`, `mcp`, and `hub`.

**The bundle loader turns a directory into a knowledge graph.** `LoadBundle` in `pkg/okf/bundle.go` walks the bundle directory, treats `index.md` and `log.md` as reserved navigation files, and parses every other markdown file into a `Concept`. It then builds two maps — `Graph` for outbound links and `InboundGraph` for backlinks — and records `BrokenLinks` and `Orphans` along the way. The loader also implements `ensureWithinRoot`, which resolves symlinks and rejects any path that escapes the bundle root, a quiet but important security boundary for a tool that agents call autonomously.

**The parser is deliberately hand-rolled.** `pkg/okf/parser.go` splits frontmatter from body with `ExtractFrontmatter` and implements just the YAML subset OKF needs — scalars, quoted strings, flow mappings like `{ by: "agent", at: "..." }`, and lists — with zero external dependencies. The `Concept` type in `pkg/okf/types.go` carries the full OKF v0.2 metadata families (`sources`, `generated`, `verified`, `status`, `stale_after`, attestation), and unknown frontmatter keys are preserved in an `Extra` map so round-trips never destroy fields the tool does not understand — a hard requirement of the OKF spec.

**Search is field-weighted BM25 over the in-memory graph.** `SearchAdvanced` in `pkg/okf/search.go` tokenizes the query, computes document frequency per term, and scores candidates with the classic BM25-style IDF formula `log(1 + (N - df + 0.5) / (df + 0.5))` multiplied by field-weighted term frequencies: title matches count 4.0, tags 3.5, description 2.5, concept ID 2.0, and body matches 1.0 capped at five occurrences. Filters (`pkg/okf/filter.go`) apply frontmatter predicates, staleness horizons restrict results to concepts expiring within a duration, and the code-ref matcher supports exact paths, directory prefixes, and recursive `**` globs. The result is the sub-millisecond retrieval the project benchmarks, with matched-on fields telling the agent exactly why a concept surfaced.

**Validation gates mutations on conformance and graph health.** `Validate` in `pkg/okf/validator.go` returns a structured result covering OKF v0.2 conformance errors, connectivity warnings, broken links, orphaned concepts, and staleness counts. In `--strict` mode, trust gaps and orphans become hard errors, and `--drift` checks whether concept descriptions and `code_refs` still match reality — the mechanism behind the project's "validate before you exit" completion gates.

**The MCP server and mutation layer close the agent loop.** `pkg/okf/mcp.go` implements JSON-RPC 2.0 over stdio: the `initialize` handshake advertises the server as `okf-agent-memory`, and `tools/list` serves the schemas embedded from `pkg/okf/schemas/tools.json`. Six tools are exposed — `okf_search`, `okf_show`, `okf_validate`, `okf_create`, `okf_update`, and `okf_relate` — each resolving the bundle directory through the same confinement check before loading. Mutations route through `pkg/okf/mutate.go`, which writes concept files with symlink-safe replacement and automatically updates `index.md` and `log.md`, tagging the actor as `agent/mcp`. Bootstrap (`pkg/okf/bootstrap.go`) completes the picture by scaffolding a new project with the knowledge bundle, the embedded agent skill from `pkg/okf/assets/skill/`, and an `AGENTS.md` codex written in the compact Agent Action Grammar that `pkg/okf/aag/linter.go` can lint.

The end-to-end flow, then: an agent session starts with the compact `AGENTS.md` codex in context; before touching code, the agent calls `okf_search` (or `--for-path`) over MCP, receiving governed, scored concepts in microseconds; after making an architectural decision it calls `okf_create`, which writes the markdown concept and updates the index and log; before exiting it calls `okf_validate` under strict mode; and the human reviews the entire knowledge delta with an ordinary `git diff`, merging memory changes exactly like code changes.

## Advantages

- **Git-native by construction.** The memory store is a directory of markdown files. Inspect, audit, branch, and revert agent knowledge with standard git tooling — there is no external database anywhere in the design.
- **A genuinely zero-dependency core.** `go.mod` declares only `golang.org/x/crypto` for the optional sync layer; the parsing, search, and validation stack in `pkg/okf/` compiles into a single static binary with no runtime services to deploy.
- **Spec-true OKF v0.2 semantics.** Provenance (`sources`), trust tiers (`generated` versus `verified`), lifecycle (`status`, `stale_after`), and preserved unknown fields are implemented in the type system, and `docs/spec/OKF-COMPATIBILITY.md` maps every spec section to the implementation.
- **Governance-aware retrieval.** `hold`, `constraint`, and `context` levels plus `code_refs` binding mean a path-governed search surfaces freeze rules before edits, not after incidents.
- **Progressive disclosure instead of prompt stuffing.** Hierarchical `index.md` files and the link graph let agents load only the concepts a task needs, keeping the baseline prompt footprint at zero tokens for domain knowledge.
- **One memory, two interfaces.** The same operations are available interactively via the CLI and programmatically via the MCP server, so humans and agents share one source of truth.

## Benefits

- **Microsecond-scale retrieval for tool-call loops.** Because the bundle is parsed into memory and scored locally, the project's benchmarks in `benchmarks/` report sub-300-microsecond searches and roughly 4 ms full-bundle validation — fast enough for high-frequency agent tool calling.
- **No recurring retrieval costs.** Lexical BM25 replaces embedding APIs entirely, eliminating per-query token charges and network latency from the memory path.
- **Trustworthy agent-written knowledge.** Agents sign their writes as `generated` and cannot fabricate `verified` entries; humans review deltas through `log.md` and git history before trusting them.
- **Reduced context bloat and attention drift.** The DMAA split keeps `AGENTS.md` to a compact codex (the README cites roughly 100–150 tokens) while domain facts stay one `okf_search` call away.
- **Portability across agent platforms.** Anything that speaks MCP — Claude Code, Cursor, Codex and peers — can attach the same memory server; the bootstrap command wires the skill and instructions into any repository.
- **Privacy-preserving sync when you need it.** The optional hub path encrypts every blob client-side with AES-256-GCM envelopes (`pkg/vault/`), so the remote hub operates as a blind, content-addressed store that never sees plaintext knowledge.

## Usage

Build the standalone binary from a clone of the repository:

```bash
make build
```

This produces `bin/okf`. Validate, search, and inspect a knowledge bundle:

```bash
# Validate bundle conformance, graph connectivity, and description drift
./bin/okf validate knowledge --strict --drift

# Search concepts via in-memory BM25 scoring
./bin/okf search "architecture layers" knowledge

# Discover constraints and active holds governing a source file before editing
./bin/okf search --for-path pkg/okf/types.go knowledge

# Inspect a concept and its relationships (with --json support)
./bin/okf show architecture/layers knowledge --json
```

Create and mutate concepts with automatic `index.md` and `log.md` bookkeeping:

```bash
./bin/okf create decisions/auth-flow knowledge \
  --type Decision \
  --title "OAuth2 Authorization Flow" \
  --desc "Standardized on PKCE for client authentication."

./bin/okf update decisions/auth-flow knowledge \
  --desc "Updated OAuth2 PKCE token refresh interval."

# Bootstrap the full memory stack into any target project
./bin/okf bootstrap /path/to/project --name "My Project"
```

Run the memory as an MCP server for Claude Code, Cursor, or any MCP client:

```bash
./bin/okf mcp knowledge
```

Example MCP configuration for `claude_desktop_config.json` or Cursor:

```json
{
  "mcpServers": {
    "okf-memory": {
      "command": "/path/to/okf-agent-memory/bin/okf",
      "args": ["mcp", "/path/to/project/knowledge"]
    }
  }
}
```

## Conclusion

okf-agent-memory is a rare kind of infrastructure project: a memory system whose entire storage engine is `git`, whose retrieval engine is a few hundred lines of dependency-free Go, and whose trust model is encoded in frontmatter rather than enforced by a separate service. Reading the source, the design is consistently conservative — path confinement, symlink-safe writes, preserved unknown fields, no forged verification — which is exactly the temperament you want in a component that autonomous agents will write to. If your team is wrestling with prompt bloat or an agent that keeps relearning the same lessons, the codebase is compact enough to read in an afternoon and the DMAA convention it implements is documented well enough to adopt incrementally.

Links:

- GitHub repository: [okf-memory/okf-agent-memory](https://github.com/okf-memory/okf-agent-memory)
- Documentation index: [docs/README.md](https://github.com/okf-memory/okf-agent-memory/blob/main/docs/README.md)
- CLI and MCP reference: [docs/guides/CLI.md](https://github.com/okf-memory/okf-agent-memory/blob/main/docs/guides/CLI.md)
- OKF v0.2 specification: [GoogleCloudPlatform/knowledge-catalog — okf/SPEC.md](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
