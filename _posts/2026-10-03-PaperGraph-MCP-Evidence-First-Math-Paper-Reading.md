---
layout: post
title: "PaperGraph MCP: Read Math Papers With Evidence, Not Guesses - Inside lotchuazzz-crypto/papergraph-mcp"
description: "A source tour of PaperGraph MCP, a Python Model Context Protocol server that turns arXiv papers, local LaTeX and PDFs into theorem-centered workspaces with source-backed proof evidence, bounded reference expansion, dependency-aware reading paths and deterministic Markdown reading reports."
date: 2026-10-03
header-img: "img/post-bg.jpg"
permalink: /PaperGraph-MCP-Evidence-First-Math-Paper-Reading/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/papergraph-mcp/lotchuazzz-crypto-papergraph-mcp-architecture.svg
tags:
  - MCP
  - arXiv
  - Research
  - Python
categories: [AI, Open Source]
keywords: "MCP server, arXiv, LaTeX, theorem dependency graph, proof evidence, source slices, Crossref, OpenAlex, citation resolution, reading report, AI agents, mathematics research, reading sessions"
author: "PyShine"
---

Ask a language model to explain a proof and it will happily invent dependencies, conflate similar theorems, and present all of it with total confidence. For mathematics that failure mode is fatal, because the value of a paper is exactly in what implies what, and which claim came from where. PaperGraph MCP takes a different stance: the agent never guesses. Every result, proof, dependency and citation the tool reports carries a source span that can be sliced back out of the original paper, and empty results are explained as extraction limits rather than dressed up as mathematical facts.

[PaperGraph MCP](https://github.com/lotchuazzz-crypto/papergraph-mcp) (version 1.2.0, MIT) is a Model Context Protocol server - with a companion CLI - that turns arXiv papers, born-digital PDFs and local LaTeX projects into local, theorem-centered workspaces. It extracts theorem-like results and their proofs, builds a statement graph with evidence, generates dependency-aware reading paths, plans bounded imports for external references through Crossref, OpenAlex and arXiv metadata search, and exports deterministic Markdown Reading Reports and Cross-Paper Reading Plans that live happily in Git or a notes vault.

The source rewards a tour because of its discipline. A stable contract document pins what version 1 tools may return. A huge local workspace module keeps all reading state in SQLite - queues, sessions, checkpoints, notes, blocked targets, open questions - with no cloud anywhere. Expansion runs have explicit budgets and policies instead of unbounded crawling, and the project is refreshingly loud about what it does not do: it does not verify proofs, does not perform semantic theorem matching, and does not claim that similarly worded results are equivalent.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/papergraph-mcp/lotchuazzz-crypto-papergraph-mcp-overview-architecture.svg" alt="Architecture overview of the lotchuazzz-crypto/papergraph-mcp repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the PaperGraph MCP repository: the stdio server fronts a SQLite-backed workspace, which feeds LaTeX parsing and evidence extraction into a theorem graph, while bounded reference expansion consults scholarly metadata providers.*

Reading the overview from left to right: `src/papergraph/server.py` exposes the MCP tool surface and opens workspaces in `src/papergraph/workspace.py`, the large local store that owns all reading state. Papers arrive through `src/papergraph/arxiv.py` or the PDF importer, then flow into `src/papergraph/parser.py` whose spans feed `src/papergraph/evidence_extractors.py` and the theorem graph in `src/papergraph/graph.py`. When a reference points outside the workspace, `src/papergraph/reference_expansion.py` runs a bounded import plan against the metadata providers. Exits on the right are the deterministic artifacts: Reading Reports and Cross-Paper Reading Plans.

## Why You Need This

Reading a hard paper is not a linear activity. You encounter Theorem 3.1, need Lemma 2.4, which leans on a result from a different paper, and before long your context window is a graveyard of half-remembered statements. PaperGraph makes that structure explicit and inspectable: `workspace_get_paper_map` shows main-result candidates and result structure before you commit to reading, and `workspace_get_dependency_reading` produces a `bottom_up` prerequisite order built only from extracted local evidence - shared dependencies included, cycles reported as `cycle_blocked` rather than silently broken.

The second problem is trust in what an agent reads for you. PaperGraph's answer is evidence discipline. Proof-local references, cited stops, source slices and dependency diagnostics all carry provenance, and `workspace_get_source_slice` lets you pull the exact lines of the original paper behind any claim. Where an author writes an explicit correspondence such as "Theorem 1.1 (= Theorem 3.1)", the module `src/papergraph/author_correspondence.py` records it as `author_declared_correspondence` - the author's declaration, not verified equivalence - and keeps both statements separate.

The third problem is literature drift: one paper references five you should skim first, each of which references more. PaperGraph handles this with bounded expansion. You approve a policy (defaults: depth 2, ten new papers per run), then advance saved runs step by step. Only unique, strong, importable identities may be selected automatically; ambiguous references wait for review. Runs keep their budgets, evidence, decisions and recovery history, so an interrupted import is resumable rather than mysterious.

Finally, state. Research happens across days and devices, and a chat transcript is a terrible place to store it. The workspace keeps reading sessions, checkpoints and notes locally in SQLite, so the next session starts from known state - and `workspace_export_reading_session_summary` hands the whole context to another person or agent as a file.

## How It Works

The entry point is an MCP stdio server that wraps one large workspace class, and almost every tool the agent calls is a thin, validated window into that store.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/papergraph-mcp/lotchuazzz-crypto-papergraph-mcp-architecture.svg" alt="Detailed architecture of the lotchuazzz-crypto/papergraph-mcp repository" style="max-width:100%;height:auto;" />
</div>

*The detailed view: server and workspace, the extraction stack from LaTeX parsing to evidence triage, the reading and export modules, the reference resolution stack, its metadata providers, and the diagnostics layer.*

### Understanding the Architecture

**Papers enter through guarded loaders.** `src/papergraph/arxiv.py` fetches arXiv sources, `src/papergraph/pdf.py` imports legally obtained local PDFs through PyMuPDF, and version 1.2.0 adds DOI entry points - `discover_doi_paper` finds exact DOI metadata and public body candidates without downloading, while `workspace_import_doi_candidate` imports only after an explicit, user-confirmed selection. HTTPS PDF downloads are bounded to 50 MiB and preserve source, version and hash receipts beside the workspace, so you always know what body text came from where.

**Parsing and extraction produce evidence, not summaries.** `src/papergraph/parser.py` structures the LaTeX or PDF text, and `src/papergraph/evidence_extractors.py` identifies theorem-like results, proofs, dependencies and citations, each with source spans. The `src/papergraph/evidence.py` model defines the shared shapes, `src/papergraph/evidence_triage.py` helps interpret sparse extraction honestly, and `src/papergraph/graph.py` assembles the theorem and citation statement graph. `result_count`, `proof_count` and `evidence_counts` report what was actually extracted so the agent cannot overstate completeness.

**Reading paths respect mathematical direction.** `src/papergraph/dependency_reading.py` walks the graph to produce `top_down` exploration and `bottom_up` prerequisite orders, marks unknown and external prerequisites as visible risks, and refuses to emit a purportedly complete reading order when entries are ambiguous, missing, cyclic or deeper than eight hops. Own proofs take priority over imported ones, and discussion mentions never fabricate new results.

**External references become reviewable plans.** The resolution stack - `src/papergraph/reference_search.py` backed by `reference_identity.py`, `reference_query.py` and `reference_assessment.py` - searches scholarly metadata with conservative DOI and arXiv normalization, ranks candidates deterministically, and explains conflicts. Provider adapters for Crossref, OpenAlex and arXiv live under `src/papergraph/reference_providers/`. Approved identities feed `src/papergraph/reference_expansion.py` through its policy, store and reporting modules, each run bounded and resumable.

**State, artifacts and diagnostics close the loop.** `src/papergraph/reading.py` and `paper_map.py` track sessions and maps; `reading_report.py` and `cross_paper_reading_plan.py` export the Markdown artifacts. On the ops side, `diagnostics.py` powers the `doctor` command and `build_identity.py` records the build's source commit and tree hash, so what a server claims to be is checkable. `docs/reference/v1-core-contract.md` pins the stable tool surface that version 1 clients can rely on.

**End to end:** you open a workspace, load an arXiv paper, and the agent lists results with `workspace_list_results`, inspects a proof with `workspace_get_result_proof`, pulls dependencies with `workspace_get_proof_dependencies`, builds a queue from local evidence, reviews external import plans, and exports a Reading Report or Cross-Paper Reading Plan - with checkpoints and notes saved so the whole thread survives restarts.

## Advantages

- **Evidence-first by construction.** Every extracted claim carries a source span; `workspace_get_source_slice` returns the exact original text behind it.
- **Local and private.** Workspaces are SQLite files on your machine; nothing is uploaded, and private manuscripts stay put.
- **Bounded, reviewable crawling.** Expansion policies cap depth and imports, require confirmation for ambiguous identities, and keep full decision history.
- **Honest uncertainty.** Cycles, unknown prerequisites and sparse extraction are reported as such, never smoothed over into false completeness.
- **Deterministic exports.** Reading Reports, session summaries and cross-paper plans are Markdown you can commit, diff and hand off.
- **Agent-native setup.** A repository-local skill walks a coding agent through installation and client configuration, with verification steps before it trusts any instructions.

## Benefits

- **Stop re-reading from scratch.** Sessions, checkpoints and queues make a paper a place you return to, not a chat you scroll back through.
- **Catch fabricated dependencies.** Because links must be evidence-backed, a hallucinated "this follows from that" has nowhere to hide.
- **Plan reading like a project.** Paper maps show main-result candidates and external risks before you invest hours in the wrong paper.
- **Keep citations honest across papers.** Cross-paper plans separate selected-paper citation evidence from unresolved outside risks.
- **Upgrade gracefully.** The v1 core contract and release notes document what changed between versions, with explicit schema migration warnings.
- **Debug the tool itself.** The doctor command and build identity reporting make "which server am I actually talking to" a one-command answer.

## Usage

Verify the pinned version without cloning (requires uv):

```powershell
uvx --from git+https://github.com/lotchuazzz-crypto/papergraph-mcp.git@v1.2.0 papergraph-mcp --version
uvx --from git+https://github.com/lotchuazzz-crypto/papergraph-mcp.git@v1.2.0 papergraph-mcp doctor
```

Add the server to any MCP client with JSON-style stdio configuration:

```json
{
  "mcpServers": {
    "papergraph": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/lotchuazzz-crypto/papergraph-mcp.git@v1.2.0", "papergraph-mcp"]
    }
  }
}
```

Then drive it from your agent with the workspace tools - load, map, inspect, queue, resolve, export:

```
load_arxiv_request(input="...")            # validated entry point for bare IDs, URLs or prose
workspace_add_arxiv_paper(...)             # import into the workspace
workspace_get_paper_map(...)               # main-result candidates and structure
workspace_get_proof_dependencies(...)      # direct and recursive proof-local deps
workspace_export_paper_reading_report(...) # deterministic Markdown handoff
```

For a guided first run, follow the repository's own walkthrough: `docs/walkthroughs/first-workspace.md`, with bounded expansion covered in `docs/walkthroughs/bounded-reference-expansion.md`.

## Conclusion

PaperGraph MCP is a principled answer to a real problem: letting AI agents help with mathematical reading without letting them invent the mathematics. The evidence model, the bounded expansion policy and the loudly documented limits make it a trustworthy subcontractor for the tedious parts of scholarship - structure, provenance and state - while leaving judgment where it belongs.

Links:

- Repository: [https://github.com/lotchuazzz-crypto/papergraph-mcp](https://github.com/lotchuazzz-crypto/papergraph-mcp)
- v1 Core Contract: [https://github.com/lotchuazzz-crypto/papergraph-mcp/blob/main/docs/reference/v1-core-contract.md](https://github.com/lotchuazzz-crypto/papergraph-mcp/blob/main/docs/reference/v1-core-contract.md)
- First workspace walkthrough: [https://github.com/lotchuazzz-crypto/papergraph-mcp/blob/main/docs/walkthroughs/first-workspace.md](https://github.com/lotchuazzz-crypto/papergraph-mcp/blob/main/docs/walkthroughs/first-workspace.md)
- v1.2.0 release notes: [https://github.com/lotchuazzz-crypto/papergraph-mcp/blob/main/docs/reference/v1.2.0-release-notes.md](https://github.com/lotchuazzz-crypto/papergraph-mcp/blob/main/docs/reference/v1.2.0-release-notes.md)
