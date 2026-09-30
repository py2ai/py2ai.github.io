---
layout: post
title: "Agent Memory: Plain Markdown as the Source of Truth for AI Agents - Inside tigerless-labs/agent-memory"
description: "A source tour of tigerless-labs/agent-memory, a local-first long-term memory runtime for AI agents that keeps Markdown files as the single source of truth, ranks retrieval locally with BM25 and optional vector fusion, and exposes everything through a CLI, host hooks, and an MCP server."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Agent-Memory-Markdown-Source-Of-Truth-Runtime-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/agent-memory/tigerless-labs-agent-memory-architecture.svg
tags:
  - AI Agents
  - Agent Memory
  - MCP
  - Python
categories: [AI, Open Source]
keywords: "agent memory, AI agent memory, long-term memory runtime, MCP server, markdown source of truth, BM25 retrieval, SQLite FTS5, local-first, Claude Code, Codex CLI, memory consolidation, sleep-time distillation, Python"
author: "PyShine"
---

An agent that closes its session forgets everything it learned in it. That single sentence, pulled straight from the README of tigerless-labs/agent-memory, describes the most persistent annoyance in everyday work with coding agents: every fresh conversation starts from zero, and the project decisions you explained yesterday have to be explained again today. agent-memory is a Python runtime built to end that loop, and it does so with an architecture choice that is refreshingly opinionated — your agent's memory lives as plain Markdown files on disk, not inside an opaque vector blob you cannot inspect.

The project positions itself as the long-term memory runtime for AI agents of any kind, not only coding ones. One store holds Markdown memories as the single source of truth, a SQLite index sits beside them as a rebuildable cache, and hosts as different as Claude Code, Codex CLI, and anything that can run a shell command share the same store. Retrieval is local and ranked, and — this is the part worth underlining — it answers with file paths rather than pasted text, so the agent opens each hit only as deep as the task requires.

The source is worth a tour precisely because the Markdown commitment is not a slogan; it is enforced in code. There is a documented invariant that removing the entire index directory loses zero knowledge, a single write path that every adapter must go through, a Manage layer that can never destroy a file on its own, and an MCP server that reuses exactly the same core calls as the CLI. Reading the modules shows a system designed so that the memory stays greppable, git-able, and portable — while still ranking like a retrieval engine.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/agent-memory/tigerless-labs-agent-memory-overview-architecture.svg" alt="Architecture overview of the tigerless-labs/agent-memory repository" style="max-width:100%;height:auto;" />
</div>

*Overview of tigerless-labs/agent-memory: three entry points (CLI, MCP server, host hooks) collapse onto one core Store, which writes Markdown truth, projects it into a rebuildable SQLite index, and serves ranked recall — with distillation and sleep-time Manage borrowing reasoning from an external executor.*

Reading the overview from left to right: an agent reaches memory through one of three adapters — the `mem` CLI in `packages/cli/src/agent_memory/cli/main.py`, the `mem-mcp` stdio server in `packages/mcp/src/agent_memory/mcp/server.py`, or the `mem-hook` entry point in `packages/adapters/src/agent_memory/adapters/hook_entry.py` that hosts fire at session boundaries. All of them call the same `Store` in `packages/core/src/agent_memory/core/store.py`, the single write path. The Store writes one file per memory into the Markdown store and reprojects the change through the Indexer into the FTS5 BM25 index and the optional vector index. Recall reads those indexes and answers with paths into the Markdown store. Below the main line, distillation and the sleep-time Manage layer trigger at boundaries and borrow their judgment from the executor package, which reasons through the host agent's own CLI.

## Why You Need This

The project's README frames the existing landscape as two architectural lines, and the framing matches what you see in the wild. One line builds a retrieval engine — embeddings, knowledge graphs, ranking pipelines — that finds the right thing but hands the agent an opaque chunk it cannot inspect, backed by a store it cannot easily migrate off. The other line hands the agent a filesystem — Markdown it can read directly, browsable with `ls` and `grep` — which is legible and free to run but does not rank, and stops scaling once the tree outgrows a directory listing.

agent-memory deliberately occupies both lines at once. In `packages/core/src/agent_memory/core/search_index.py`, the comment states that BM25 over abstract and body is "projection only: never a source of truth." The index exists to rank; the files exist to be true. That separation has practical consequences you can verify in code: the Indexer in `packages/core/src/agent_memory/core/indexer.py` performs incremental syncs keyed by content hash, and a full rebuild travels through the same code path, so `rm -rf .index/ && mem rebuild` loses nothing. The README explicitly notes this property is enforced by a test, not just promised in a document.

The second problem is write coverage. If remembering is left to the agent's own judgment mid-task, remembering is what gets skipped. agent-memory moves writes out of the agent's hands: hooks fire at conversation boundaries — the README describes SessionStart injecting, Stop and SessionEnd distilling, PreCompact evicting — and the full session trace is copied into the archive first, so a missed distillation never means lost knowledge. The hook entry point is written defensively too: every path in `packages/adapters/src/agent_memory/adapters/hook_entry.py` ends in exit code 0, because a hook that breaks its host is worse than a hook that never fires.

The third problem is governance of an unattended memory process. A system that rewrites your agent's knowledge overnight needs hard limits on what it may do unattended. agent-memory's Manage layer, implemented in `packages/core/src/agent_memory/core/manage.py`, separates deterministic tidy-up (T0: dates, weights, links, directories) from reasoned consolidation (T1: merge, split, supersede, delete proposals), caps each kind per sleep, files every proposal in a decision ledger, writes a dream report per pass, and never removes a file — physical removal is a human-run command the system cannot reach.

## How It Works

Everything funnels through one core so that CLI, MCP, and hooks cannot drift apart — the diagram below follows a memory from raw session material to ranked retrieval and sleep-time consolidation.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/agent-memory/tigerless-labs-agent-memory-architecture.svg" alt="Detailed architecture of the tigerless-labs/agent-memory repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of tigerless-labs/agent-memory: adapters, the single-write-path Store with its schema and placement rules, the content-hash-indexed FTS5 and vector projections, the read path with recall fusion and raw-evidence trace, and the distillation and Manage lifecycle over an external executor.*

### Understanding the Architecture

**The store is a directory of Markdown with strict placement.** `mem init` creates the layout the README documents: a `MEMORY.md` root index with one line per memory, a `config.toml` where an unknown knob is refused at load, a `schemas/` directory defining each memory type's fields, and memories stored at `<type>/<group>/<name>.md`. The `Store` class in `packages/core/src/agent_memory/core/store.py` is labeled the single write path — its docstring notes that every adapter, and Manage itself, enters there, and that nothing in it deletes a file. Deletion ends validity via an `invalid_at` timestamp, keeping replaced files on disk for `recall --as-of` queries.

**Indexing is a projection, incremental by content hash.** The Indexer in `packages/core/src/agent_memory/core/indexer.py` diffs a content-hash manifest against the files present, re-upserts only what changed, and records chunked abstract-plus-body text into two SQLite FTS5 surfaces — active and history — via `packages/core/src/agent_memory/core/search_index.py`. When the optional vector index is enabled, the same sync path upserts chunk embeddings through `packages/core/src/agent_memory/core/embeddings.py` (FastEmbed/ONNX, defaulting to the BAAI/bge-small-en-v1.5 model per the README) into `packages/core/src/agent_memory/core/vector_index.py`, which performs exact cosine search in SQLite.

**Recall applies eligibility before relevance, with zero model calls.** The `Recall` class in `packages/core/src/agent_memory/core/recall.py` first filters to the eligible surface — active files by default, the history surface only for `--as-of` — then ranks by relevance times weight times recency. Lexical candidates come from BM25; when vectors are enabled, `fuse_candidates` merges both lists with reciprocal-rank fusion (the fusion constant is 60 in the source) before lifecycle and scope rules apply. The result is the L0 list the README describes: one-line abstract, file path, anchor, and score — not pasted text.

**Reads are layered and leave truth untouched.** `mem read` opens a memory at three levels — abstract, outline, or full — defined in the same store module, and `mem context` combines recall with expanding the top hits in one call. Crucially, reads never mutate files: usage statistics go to the access log in `packages/core/src/agent_memory/core/access_log.py`, and weight is settled back into frontmatter later by Manage in batch. For auditability, `mem trace` in `packages/core/src/agent_memory/core/trace.py` resolves provenance pointers like `sessions/<session>#<start>-<end>` into the cited raw message ranges.

**Distillation runs at boundaries, not mid-task.** The pipeline in `packages/core/src/agent_memory/core/distill.py` describes itself as "one boundary, end to end: batch, render, reconcile, ask, apply, repair, advance." It batches the archived backlog, renders prompts via `packages/core/src/agent_memory/core/prompts.py`, and hands them to an ask callable supplied from outside the core — honoring the invariant that the library contains no LLM client. The executor package (`packages/executor/src/agent_memory/executor/distiller.py` and `reasoners.py`) borrows judgment from the host agent's own CLI by default or a configured model endpoint, which the README points out keeps every write visible in your transcript and installs zero API keys.

**Manage consolidates on its own clock, with authority tiers.** The sleep pass in `packages/core/src/agent_memory/core/manage.py` runs the deterministic T0 actions itself — normalizing dates, merging exact duplicates, adding links, settling weights — and for T1 it lets a reasoner choose only from a menu the core drafted, capped per kind per sleep, with every decision recorded in the ledger at `packages/core/src/agent_memory/core/ledger.py`. Each sleep leaves a report in `dream-reports/`, and supersede chains stay intact so updating never destroys.

End to end: a session starts and the hook injects `MEMORY.md`; the agent works, possibly calling `mem recall` or `memory_recall` over MCP; the session ends and the hook archives the trace, then launches distillation, which converts the backlog into schema-typed memory files through the single write path; the indexer projects the new files into FTS5 and optionally vectors; and the next `mem sleep` pass consolidates by value, filing anything destructive as a proposal you confirm with `mem decide <id> --accept`.

## Advantages

- **Plain Markdown is the source of truth.** One memory is one file with frontmatter — stable name, abstract, type fields, validity interval, links, weight, provenance — and a free-markdown body you can read, grep, and commit like any other file.
- **The index is disposable.** Every projection, from the content-hash manifest to FTS5 to vectors, is rebuildable from the files; deleting `.index/` and running `mem rebuild` is a documented, tested recovery path, not a leap of faith.
- **Retrieval answers with paths, not payloads.** The L0 list keeps context spending under your control — index line, then abstract, then outline, then full file, with the anchor that matched included for long documents.
- **Local and keyless reads.** The recall path calls no model and crosses no network; BM25 over FTS5 is the baseline, and the optional vector fusion runs as a local FastEmbed/ONNX embedder rather than a hosted embedding API.
- **One core, three entry points.** CLI, MCP, and hooks all collapse onto the same core calls — the packaging test suite even checks entry equivalence — so behavior does not depend on which door the agent walks through.
- **MCP without dependencies.** The server in `packages/mcp/src/agent_memory/mcp/server.py` implements JSON-RPC framing over stdio around the core, exposing nine tools from `memory_recall` to `memory_feedback` to any MCP-speaking client.

## Benefits

- **Session continuity for coding agents.** Decisions recorded once persist across sessions and even across different host agents, because the store is shared and host-agnostic — the README describes writer/reader interoperability tests across Claude Code, Codex CLI, and Hermes.
- **No more "forgot to save" failures.** Boundary-triggered distillation plus the append-only archive mean the raw material survives even when the distiller misses something, so coverage is the system's job rather than the agent's discipline.
- **Bounded, auditable autonomy.** Authority tiers, per-kind caps, the proposal ledger, and dream reports make the sleep-time consolidation process inspectable and reversible instead of a black box that edits your knowledge base overnight.
- **History instead of loss.** Supersede chains and validity intervals let `recall --as-of` answer as of a date, so correcting a memory never erases what the agent believed earlier — valuable for debugging why an agent acted as it did.
- **Vendor escape hatch.** Because truth is Markdown and the index is a cache, migrating off the system means copying a directory; the README calls this user sovereignty, and the code structure backs the claim.
- **A clean extension surface.** The core's no-LLM-client rule means the library is testable without a network, and the executor's reasoner abstraction lets you point distillation at the host CLI or your own model endpoint in `config.toml`.

## Usage

The project requires Python 3.12+ and `uv`, and is installed from a checkout rather than PyPI:

```bash
git clone https://github.com/tigerless-labs/agent-memory.git
cd agent-memory
uv sync --all-packages
```

That builds the `mem`, `mem-mcp`, and `mem-hook` executables into `.venv/bin`; expose them to your shell:

```bash
export PATH="$PWD/.venv/bin:$PATH"
```

Initialize the store (it defaults to `~/agent-memory-store`, relocated via the `AGENT_MEMORY_STORE` environment variable) and prove the round-trip — write one memory, recall it, then throw the index away and rebuild:

```bash
mem init

mem record --type decision --field project=agent-memory \
  --abstract "Markdown files are the single source of truth" \
  --body "Indexes are rebuildable caches."
mem --json recall "source of truth"
rm -rf ~/agent-memory-store/.index && mem rebuild
```

Everyday retrieval operates on levels, from the ranked L0 list to the full file:

```bash
mem recall "why files instead of a database" --limit 20
mem read <name> --level outline
mem context "why files instead of a database"
mem trace <name>
```

Wire it into a supported host or expose it over MCP:

```bash
mem setup --host claude-code   # or: --host codex
```

Sleep-time consolidation runs on demand and files anything destructive as a proposal:

```bash
mem sleep --reason host
mem proposals
mem decide <id> --accept
```

## Conclusion

tigerless-labs/agent-memory is a well-engineered answer to a question more teams are asking as agents become long-lived collaborators: where does the knowledge live, and who controls it? By making plain Markdown the source of truth, the SQLite index a rebuildable cache, and judgment something borrowed from the host agent rather than embedded in the library, it keeps memory legible to humans and portable between tools — while the code still delivers the part a folder of files cannot: ranked, scoped, lifecycle-aware retrieval with BM25 and optional vector fusion. The layered read path, the boundary-triggered write coverage, and the capped, proposal-based Manage layer make it a genuinely instructive codebase to read if you are building anything in the agent-memory space. It is early software — the README notes there is no PyPI release yet and that vector retrieval remains optional — but the invariants are real, tested, and worth stealing for your own systems.

Links:

- GitHub repository: [tigerless-labs/agent-memory](https://github.com/tigerless-labs/agent-memory)
- Design docs in the source tree: [docs/design/](https://github.com/tigerless-labs/agent-memory/tree/main/docs/design)
- Project invariants: [CLAUDE.md](https://github.com/tigerless-labs/agent-memory/blob/main/CLAUDE.md)
