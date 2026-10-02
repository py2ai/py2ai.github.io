---
layout: post
title: "Lemmalog: A Datalog Engine for LLM Agent Memory - Inside JordyZomer/lemmalog"
description: "Lemmalog is a Rust Datalog engine that turns an LLM agent's memory into a deductive database. We tour the source behind its stratified evaluation, bi-temporal facts, semiring provenance, why() proof trees, incremental maintenance, hybrid retrieval, and the MCP server that plugs it into coding agents."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /Lemmalog-A-Datalog-Engine-For-LLM-Agent-Memory/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/lemmalog/jordyzomer-lemmalog-architecture.svg
tags:
  - Rust
  - Datalog
  - AI Agents
  - Memory
categories: [AI, Open Source]
keywords: "lemmalog, datalog engine, LLM agent memory, deductive database, bi-temporal facts, provenance annotations, MCP server, Rust, seminaive evaluation, hybrid retrieval, LongMemEval, LoCoMo"
author: "PyShine"
---

Most agent memory systems are retrieval problems waiting to be re-solved: embed everything, vector-search, hope the right chunks surface. [JordyZomer/lemmalog](https://github.com/JordyZomer/lemmalog) starts from a different thesis entirely, that an agent's memory should be a deductive database. Instead of remembering better than a vector store, the agent builds a verifiable model of what it knows and mechanically reasons over how that knowledge changes. Base facts are asserted at the ingestion boundary, rules derive closures, temporal projections, and relevance diffusion, and every fact carries provenance back to the conversation it came from.

What makes the project immediately usable is that the thesis ships as a working artifact. Lemmalog is a Rust crate, plus an MCP server for Claude Code and Kimi CLI, plus a REPL, plus an agent skill, all backed by one engine. The repository is MIT-licensed, version 0.2.0, and it arrives with something rare in this space: a design document with an honest status log, a differential test harness that pits the engine against a brute-force oracle, and benchmark runs on standardized protocols with the measurement caveats printed right next to the numbers.

The source is worth a tour because it is a rare case of classic database machinery, stratified Datalog, seminaive evaluation, magic-sets rewriting, being applied to a genuinely modern problem, and being engineered with the paranoia those classics deserve. Let us look at how the pieces fit.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/lemmalog/jordyzomer-lemmalog-overview-architecture.svg" alt="Architecture overview of the JordyZomer/lemmalog repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Lemmalog codebase: three entry points (MCP server, REPL, headless CLI) feed the AgentMemory facade, which asserts facts into the Datalog engine and reads back through hybrid retrieval, while the parser, magic-sets rewriter, entity resolution, and semantic index all drive the engine core.*

Reading the overview from left to right: agents talk to Lemmalog through the MCP server, the REPL, or the scripted CLI, all converging on the AgentMemory facade in the agent layer. The facade observes conversation turns through an extractor, applies a deterministic update policy, and pushes assertions into the engine core, where the rule parser installs stratified programs and incremental evaluation derives the memory's derived views. Queries flow back out through the hybrid retrieval layer, which ranks distilled facts against the stored graph, while magic-sets rewriting answers point questions without materializing a full fixpoint and entity resolution keeps one canonical spelling per thing the agent knows.

## Why You Need This

The first problem is staleness. A conversational transcript records everything that was ever said, including facts that were later overwritten: "I work at Acme" followed three sessions later by "I moved to Gigant." A retrieval-first memory surfaces both and lets the model guess. Lemmalog treats knowledge update as a first-class operation, with an update policy that supersedes old values, closes their temporal intervals, and keeps the history queryable, so "what is true now" is a derived fact, not a lucky retrieval.

The second problem is trust. When an agent answers from memory, you want to ask not just "what do you know" but "how do you know it." Every fact in Lemmalog carries a semiring annotation, confidence that multiplies across rule bodies and provenance that unions, and the why() facility renders the proof tree of any fact back to its source episodes. An answer is no longer a paragraph with confident tone; it is a derivation you can inspect.

The third problem is cost. Stuffing an entire conversation history into context grows linearly with time and eventually overflows the window. The README's token economics section reports a constant per-question cost of roughly a couple thousand tokens regardless of history length, against full-context prompts that keep growing, with benchmark contexts measured far smaller than the full-transcript alternative. For an agent queried every turn across a long session, that difference compounds fast.

The fourth problem is correctness of the memory itself. Incremental derivation over retractions and negation is exactly where hand-rolled engines rot, and this repository treats that seriously: a differential harness generates hundreds of random stratified programs and compares the engine against a naive fixpoint oracle, plus a parser fuzz campaign, and the README credits that harness with catching real soundness bugs before they shipped.

## How It Works

The crate splits into four clean layers, and the detailed diagram maps almost every file to its role.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/lemmalog/jordyzomer-lemmalog-architecture.svg" alt="Detailed architecture of the JordyZomer/lemmalog repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of Lemmalog: entry points and setup on the left, the Datalog engine core in the middle, the agent memory layer to its right, and tests, benchmark tooling, and examples anchoring the outer ring.*

### Understanding the Architecture

**The engine core.** [src/eval.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/eval.rs) is the heart: a store of row vectors with per-position hash indexes, semiring annotations, stratification, and trail-backtracking seminaive evaluation. Each run() is an epoch; facts asserted since the last epoch seed the deltas, and rules fire only against those deltas until fixpoint, so an idle turn derives nothing and costs microseconds. Bi-temporal facts carry valid_from, valid_to, and asserted_at columns alongside a now() builtin, and negation is negation-as-absence with negative-cycle rejection at install time. Aggregated head arguments like count, min, max, and sum lower to a group-by fold with strict stratum ordering enforced like negation edges.

**Rules as data.** [src/ast.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/ast.rs) contains a hand-written parser for runtime-parsed Datalog over the symbol interner in [src/intern.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/intern.rs). Rules install as versioned batches that an agent can uninstall, and installing or removing a batch marks the program dirty so the next run backfills every rule against the existing store, which means rules installed mid-session fire against old facts, not just new ones.

**Demand-driven questions.** [src/magic.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/magic.rs) implements magic-sets rewriting for ask_deep: only the demand-relevant slice of the program is derived, the base store is untouched, and the engine reports how large the slice was. This is the README's argument against blindly materializing dense closures, point queries without a full fixpoint, with [examples/graph_queries.rs](https://github.com/JordyZomer/lemmalog/blob/main/examples/graph_queries.rs) demonstrating the trade-off on cyclic joins.

**Entity resolution.** The extractor proposes star-shaped alias edges, and [src/canonical.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/canonical.rs) derives the symmetric-transitive closure with Datalog rules, projects directional canonical views, and propagates confidence through the merge so weak two-hop aliases stay visibly low-confidence. Topology violations derive explicit conflict facts instead of silently merging identities, and retracting one alias edge collapses the closure and every downstream view in the same epoch.

**The agent layer.** [src/agent.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/agent.rs) hosts the AgentMemory facade and the extraction boundary: the LLM sits strictly at ingestion, asserting triples in a line protocol, while the fixpoint stays pure. A deterministic update policy decides between add, no-op with annotation merge, supersede, or escalate, and silent zero-fact ingestion is impossible because every dropped line is reported with its reason. [src/retrieval.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/retrieval.rs) then answers context requests with a three-signal ranker, in-crate BM25 over rendered facts and verbatim episodes, entity-match boosting from the graph, and a budget-aware positional assembler that puts distilled facts first and their provenance episodes last. [src/semantics.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/semantics.rs) adds the vector half, an embedder trait with entity seeds diffused through relevance rules, and [src/llm.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/llm.rs) wires real models through an optional feature flag.

**Surfaces and evaluation.** The MCP binary, REPL, and headless CLI all sit on [src/session.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/session.rs), and the end-to-end flow is: observe a turn, let the policy update the store, run the incremental fixpoint, then ask, ask deep, request context, or demand a why() proof. Evaluation is layered the same way: the differential harness in [tests/differential_test.rs](https://github.com/JordyZomer/lemmalog/blob/main/tests/differential_test.rs) validates the engine against an oracle, the scenario module generates long-horizon synthetic conversations with ground truth, and [src/longmemeval.rs](https://github.com/JordyZomer/lemmalog/blob/main/src/longmemeval.rs) runs the standardized memory benchmarks whose results, reported transparently with variance caveats in the README, put Lemmalog competitively against named systems on both LongMemEval and LoCoMo.

## Advantages

- **Memory you can audit.** Every fact, asserted or derived, carries confidence and provenance, and why() renders the proof tree back to source episodes.
- **Truth maintenance, not echo.** Supersessions close temporal intervals and trigger scoped recomputes, so stale values are retracted rather than retrieved alongside new ones.
- **Incremental by construction.** Per-epoch seminaive evaluation means idle turns cost essentially nothing and each turn's maintenance stays proportional to the change.
- **Deterministic where it matters.** The update policy, conflict detection, closure derivation, and aggregation are all mechanical; the LLM touches only the extraction boundary.
- **Demand queries over blind materialization.** Magic-sets rewriting answers point questions from a derived slice without exploding the store with dense closures.
- **Validated like a database.** A differential oracle harness, parser fuzzing, and a synthetic long-horizon eval with ground truth guard the engine's soundness.

## Benefits

- **Constant per-turn cost.** Context assembly is budgeted and selection-driven, so per-question token cost stays flat as history grows, unlike full-transcript prompting.
- **Plug-and-play with agents.** One installer registers the MCP server with supported CLIs and drops in a skill file that teaches the working-memory discipline.
- **Multi-surface access.** The same snapshot serves interactive REPL sessions, MCP tool calls, and scripted CLI commands from sub-agents or cron.
- **Revertable knowledge.** Rule batches install as versioned units an agent can uninstall, with automatic backfill against everything already in memory.
- **Honest benchmarking.** Standardized LongMemEval and LoCoMo runs, with loss analysis tooling and published caveats, make performance claims checkable.
- **Event-sourced persistence.** Snapshots store rules, episodes, and base facts; derived relations rebuild on load, so the durable state stays minimal.

## Usage

Build the MCP server and register it with your agent CLI:

```sh
cargo build --release --features mcp
claude mcp add lemmalog -- $(pwd)/target/release/lemmalog-mcp
```

Persistence lives at the snapshot path the server was given, defaulting to a memory file in your home directory, and you can point a session at a specific snapshot:

```sh
claude mcp add lemmalog --env LEMMALOG_MCP_PATH=/tmp/lemmalog.snapshot -- $(pwd)/target/release/lemmalog-mcp
```

The one-command installer builds, registers the MCP server with every supported CLI it finds, and installs the agent skill:

```sh
./scripts/install.sh               # install
./scripts/install.sh --uninstall   # remove registrations + skill
```

Sub-agents and scripts that cannot reach MCP can use the headless CLI against the same snapshot:

```sh
LEMMALOG_MCP_PATH=/tmp/lemmalog.snapshot lemmalog-cli observe --facts 'S --rel--> O'
LEMMALOG_MCP_PATH=/tmp/lemmalog.snapshot lemmalog-cli query --goal 'current("s", R, O)'
```

For development and exploration, the crate ships a REPL, a test suite, and a set of runnable examples:

```sh
cargo run --bin lemmalog          # interactive REPL
cargo test                        # engine, agent, aggregation, differential tests
cargo run --release --example investigation
cargo run --release --example perf
```

A typical agent session flows through the MCP tools: observe asserts facts with the line protocol, install_rules adds derived predicates, query binds variables, why prints the proof tree, what_if previews a hypothetical assertion with a byte-identical restore afterward, and canonicalize proposes alias edges for entity resolution.

## Conclusion

Lemmalog is a bet that the oldest ideas in databases, stratified negation, seminaive evaluation, provenance annotations, demand-driven rewriting, are exactly the right toolkit for the newest problem in agents, memory that changes. The source backs the bet with unusual rigor: a differential oracle harness that has already caught soundness bugs, a benchmark story told with its caveats attached, and an MCP surface designed so that malformed extractions are loud instead of silently lost. If your agents need memory that can answer "how do you know that" and "what changed since last turn" without re-reading the whole transcript, this repository is one of the most principled starting points we have toured.

Links:

- GitHub repository: [JordyZomer/lemmalog](https://github.com/JordyZomer/lemmalog)
- Design document: [datalog-context-engine-design.md](https://github.com/JordyZomer/lemmalog/blob/main/datalog-context-engine-design.md)
- Agent skill: [skills/lemmalog/SKILL.md](https://github.com/JordyZomer/lemmalog/blob/main/skills/lemmalog/SKILL.md)
