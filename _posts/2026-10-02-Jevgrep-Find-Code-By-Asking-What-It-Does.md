---
layout: post
title: "Jevgrep: Find Code by Asking What It Does - Inside dzhng/jevgrep"
description: "Jevgrep is an MIT-licensed CLI that gives coding agents semantic code search: ask a repository question and get relevant files, reading leads, and verbatim source excerpts in one stdout response. We tour the TypeScript source behind its hierarchical Jev-judged retrieval, Tree-sitter declaration parsing, provider abstraction, and measured SWE-bench cost savings."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /Jevgrep-Find-Code-By-Asking-What-It-Does/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/jevgrep/dzhng-jevgrep-architecture.svg
tags:
  - TypeScript
  - Code Search
  - AI Agents
  - CLI
categories: [AI, Open Source]
keywords: "jevgrep, semantic code search, coding agents, Jev, modeljudged retrieval, tree-sitter, TypeScript CLI, SWE-bench, verbatim source excerpts, agentic search, code retrieval, LLM tools"
author: "PyShine"
---

Coding agents spend a real slice of every unfamiliar task just finding the right files: grepping for guessed keywords, opening plausible folders, reading code that turns out to be irrelevant. [dzhng/jevgrep](https://github.com/dzhng/jevgrep) attacks that step directly. It is a CLI, `jg`, that takes a natural-language question about a repository, such as "How are database connections created, pooled, and closed?", and returns relevant files, reading leads, and verbatim source excerpts in a single stdout response. The judgment about what is relevant is made by Jev, the code-fluent model served through the Vercel AI Gateway, applied hierarchically across folders, files, and declarations.

The pitch is quantified on the box: same intelligence, about 30 percent lower coding-agent cost. In the repository's ten-task SWE-bench comparison, both Jevgrep-equipped and baseline agents solved the same 8 of 10 tasks, while the full agent cost fell from $7.62 to $5.44, a measured 28.6 percent reduction excluding Jev's own cost. The project is unusually honest about methodology: the eval protocols, per-task costs, and limitations live in the repository, and the README explicitly distinguishes single-run observations from guarantees.

The source is worth a tour because it is a clean design: the model classifies, the tool retrieves and presents evidence, and the calling agent keeps ownership of reasoning and implementation. Every layer, filesystem filtering, budgeted discovery, declaration-aware presentation, provider abstraction, is built to keep that contract. Let us walk through it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jevgrep/dzhng-jevgrep-overview-architecture.svg" alt="Architecture overview of the dzhng/jevgrep repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Jevgrep codebase: the jg CLI and its agent skill on the left, the retrieval engine in the middle where Jev judgments meet source selection, and the parsing and provider layers that ground the evidence in real code.*

Reading the overview from left to right: a question enters through the jg CLI, which parses flags, loads the saved provider key, and hands the request to the retrieval boundary. The retrieval engine explores the repository hierarchy, sending directory metadata and content previews to the Jev relevance judge, which reaches the model providers. Qualifying branches feed source selection, which uses the declaration parser, backed by Tree-sitter WASM grammars in a cancellable worker, to cut precise verbatim units. Results render to stdout, and memoized answers land in a local cache. The agent skill teaches coding agents how to invoke all of this.

## Why You Need This

The first problem is semantic search without an index. Traditional code search wants exact symbols; keyword grep fails when you know the behavior but not the vocabulary. Jevgrep inverts the flow: you ask what the code does, and the traversal asks relevance questions at each level, folders first, then files, then declarations, so a question like "which tests cover retry behavior when a request times out" finds candidates that share no keywords with the query.

The second problem is cost discipline in agent loops. The expensive failure mode is an agent reading dozens of full files to locate one function. Jevgrep's design pushes back with budgets everywhere: a navigation byte budget applies until source selection has confirmed useful code, preview requests are split into bounded chunks, provider requests get token-aware admission, and output is capped independently. The published SWE-bench comparison, 8 of 10 tasks solved both ways with roughly a third off the agent bill, is the measured result of that discipline.

The third problem is honest uncertainty. Many retrieval tools quietly truncate and never tell you. Jevgrep's retrieval boundary is explicit: unread descendants and failed classifications remain unknown, and reaching a budget reports incomplete discovery so the caller can narrow the root. Files that pass relevance but lack a confident excerpt still come back as reading leads rather than being dropped.

The fourth problem is setup friction. A retrieval tool that needs Python, ripgrep, and a pile of API configuration is a hard sell inside agent sandboxes. jg is a single npm package on Node 22+, needs no separate Python or ripgrep installation, talks to any of several TypeSafe-compatible providers, and ships an agent skill installer so your coding agent learns the workflow without you writing a custom prompt.

## How It Works

The repository is a Bun-workspace monorepo with an app for the CLI and a core package for retrieval, and the detailed diagram maps the modules.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jevgrep/dzhng-jevgrep-architecture.svg" alt="Detailed architecture of the dzhng/jevgrep repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of Jevgrep: the CLI surface and skill on the left, the retrieval engine and its budgets in the center, declaration parsing below, providers and snapshots on the right, with tests, the SWE-bench harness, and release tooling anchoring the ring.*

### Understanding the Architecture

**The retrieval boundary.** [packages/core/src/retrieve.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/retrieve.ts) owns discovery: hierarchical traversal with directory metadata and content previews, no upload of the whole tree, no fixed top-N. Weak navigation signals are suppressed unless strong evidence follows, and a bounded early source pass reuses the same judgments in later selection. The architecture notes in [docs/architecture.md](https://github.com/dzhng/jevgrep/blob/main/docs/architecture.md) state the contract plainly: retrieved source is data, never instructions, and the engine never executes repository code.

**Judgments and budgets.** [evaluator.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/evaluator.ts) frames relevance judgments across folder, file, and declaration levels, amortizing shared context through the call-context windows in [call-context.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/call-context.ts). Every model call passes through [requests.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/requests.ts) and the token-aware admission of [rate-budget.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/rate-budget.ts), and navigation estimates alone can never justify an unbounded search. Failed judgments do not erase evidence already obtained.

**Declaration-aware evidence.** [selection.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/selection.ts) chooses source units against the immutable snapshot in [source.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/source.ts), so line references always match the bytes that were classified. [parser.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/parser.ts) builds declarations and coordinates: Python, Go, and Rust use packaged Tree-sitter WASM grammars inside the shared cancellable worker in [parser-worker.mjs](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/parser-worker.mjs), while TypeScript and JavaScript use the TypeScript compiler parser; everything else falls back to bounded source chunks. Original bytes are preserved, never reconstructed from the syntax tree.

**Providers and auth.** [providers.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/providers.ts) abstracts the model endpoints: Vercel AI Gateway, TypeSafe, OpenRouter, OpenCode Zen, or a custom TypeSafe-compatible URL. The CLI side in [apps/cli/src/auth.ts](https://github.com/dzhng/jevgrep/blob/main/apps/cli/src/auth.ts) saves the key in an owner-only config file, and `jg doctor` verifies the setup with synthetic input.

**Local state and filtering.** [filesystem.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/filesystem.ts) implements the eligibility policy that respects ignore files and excludes hidden, dependency, build, binary, and obvious credential files, and `jg files` reports what a search would read without any network call. Answers are cached locally by default through [cache.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/cache.ts) over [storage.ts](https://github.com/dzhng/jevgrep/blob/main/packages/core/src/storage.ts), and output goes to stdout only; no report files are created.

**Agent workflow and validation.** The [skill installer](https://github.com/dzhng/jevgrep/blob/main/apps/cli/src/skill.ts) delegates to the skills CLI to teach Claude Code, Codex, OpenCode, and others the jg workflow, with the actual instructions in [skills/jevgrep/SKILL.md](https://github.com/dzhng/jevgrep/blob/main/skills/jevgrep/SKILL.md). Quality is layered: unit suites in [test/retrieval.test.ts](https://github.com/dzhng/jevgrep/blob/main/test/retrieval.test.ts) and [discovery tests](https://github.com/dzhng/jevgrep/blob/main/test/discovery.test.ts), Docker tests of the installed package, the SWE-bench harness in [evals/implementation/swebench/installed.py](https://github.com/dzhng/jevgrep/blob/main/evals/implementation/swebench/installed.py) whose outcomes are recorded in the [results records](https://github.com/dzhng/jevgrep/blob/main/evals/results/relevance-threshold-2026-09-27.md), and tag-triggered publication with validation of the exact public package. The end-to-end flow: a question in, budgeted hierarchical discovery, Jev judgments down the tree, declaration-grounded excerpts out, all in one stdout response the calling agent can act on.

## Advantages

- **Question-driven discovery.** Natural-language questions find code that keyword search misses, because relevance is judged across folders, files, and declarations.
- **Budgeted by construction.** Navigation byte budgets, chunked previews, token-aware request admission, and capped stdout keep every search bounded.
- **Honest partial results.** Incomplete discovery is reported as incomplete, unknown regions stay unknown, and reading leads survive without confident excerpts.
- **Verbatim, coordinate-accurate source.** Excerpts come from an immutable snapshot with Tree-sitter or compiler-backed declaration coordinates, never reconstructed text.
- **Provider-portable.** One auth flow covers Vercel AI Gateway, TypeSafe, OpenRouter, OpenCode Zen, and custom TypeSafe-compatible endpoints.
- **Agent-ready packaging.** A single npm binary on Node 22+, no Python or ripgrep dependency, plus a skill installer for the major coding agents.

## Benefits

- **Lower agent bills on unfamiliar code.** The published ten-task SWE-bench comparison shows the same task outcomes at a measurably lower agent cost.
- **Faster orientation.** One question returns a summary, a compact file list, and source excerpts, collapsing many manual grep-and-read cycles.
- **Predictable behavior in sandboxes.** stdout-only output, local-only caching, and `jg files` dry-run counts make the tool safe to reason about in restricted environments.
- **Transparent methodology.** Eval protocols, per-task costs, and limitations are committed to the repository, so the headline numbers can be checked.
- **Language-broad coverage.** Python, TypeScript/JavaScript, Go, and Rust get declaration parsing; any other text remains searchable through bounded chunks.
- **Maintainable core.** The app/core split, typed contracts, and layered test suites make the retrieval policy easy to follow and extend.

## Usage

Install the CLI globally and authenticate with your chosen provider:

```sh
npm install -g @dzhng/jevgrep
jg auth
jg doctor
```

Ask a repository question; the answer lands on stdout with a summary, source excerpts, and declaration locations:

```sh
jg skill
jg "How are telemetry events recorded and sent?" ./my-project
```

```sh
jg "Where is authentication checked before a request reaches a handler?" .
jg "How are database connections created, pooled, and closed?" ./src
jg "Which tests cover retry behavior when a request times out?" .
```

Install the agent skill so your coding agent learns the workflow; the installer detects Claude Code, Codex, OpenCode, and others:

```sh
jg skill
```

```sh
npx skills add dzhng/jevgrep --skill jevgrep
```

Count the files a search would read, without a provider key or network request, and exclude paths for one search:

```sh
jg files [root]
jg "Find the retry policy" . --exclude '**/*.test.ts' --exclude 'src/generated/'
```

For development from a checkout, the repository uses Bun workspaces and Turborepo:

```sh
bun install --frozen-lockfile
bun run dev --help
bun run verify
```

## Conclusion

Jevgrep is a focused tool with a clear contract: the model judges relevance, the tool gathers and presents evidence with honest bounds, and the calling agent stays in charge of reasoning and implementation. The engineering shows in the details that usually get skipped, budget accounting, snapshot-accurate coordinates, declared incompleteness, and a benchmark story committed alongside the code. For teams wiring coding agents over large unfamiliar repositories, it is a practical demonstration that a small, well-instrumented retrieval tool can pay for itself in both time and tokens.

Links:

- GitHub repository: [dzhng/jevgrep](https://github.com/dzhng/jevgrep)
- Architecture notes: [docs/architecture.md](https://github.com/dzhng/jevgrep/blob/main/docs/architecture.md)
- Agent skill: [skills/jevgrep/SKILL.md](https://github.com/dzhng/jevgrep/blob/main/skills/jevgrep/SKILL.md)
