---
layout: post
title: "Seiso: A Convention And Linter For AI-Written Docs"
permalink: /Seiso-A-Convention-And-Linter-For-AI-Written-Docs/
image: https://pyshine.com/assets/img/diagrams/seiso/scarletkc-seiso-architecture.svg
tags: [Rust, Markdown, Documentation, Linter, AI]
---

Agent-written documentation has a peculiar failure mode: the text is fluent, tidy, and quietly wrong. A setup guide states the version of the moment as if it were a requirement. Three pages retell the same field list, and two of them are now stale. A how-to spends its second paragraph explaining what "you asked for" in a chat window nobody else can see. Each of these is a small lie that every future coding agent will ingest as context for its next change.

[seiso](https://github.com/scarletkc/seiso) by scarletkc attacks exactly this problem. It is a Markdown convention and linter for project documentation written by AI and read by humans and agents, released under MIT. Version 0.3.0 ships as a Rust crate on crates.io, a Python package on PyPI, and an npm package `@scarletkc/seiso` with prebuilt binaries for macOS on Apple silicon and Intel, Linux x64 and arm64 with glibc or musl, and Windows x64 and arm64. The pitch is the one rustfmt made a decade ago: stop negotiating documentation style per project, share one convention instead, and let tools check it.


## The convention comes first

The interesting design decision is that the tool is secondary. The repository carries a standalone [Seiso Convention Specification](https://github.com/scarletkc/seiso/blob/main/spec/convention.md) - currently version 0.2.0, deliberately versioned separately from the crate so the two can evolve on different clocks. A test in the source keeps the build's `SPECIFICATION_VERSION` in sync with the newest entry in the spec changelog. The spec is written in RFC 2119 style, every requirement carries a stable identifier like `KIND-1`, and it states plainly that a checker other than seiso may claim to check it. Before 1.0.0 the wording is provisional, and the changelog records each release.

The convention itself is short enough to hold in your head:

- Every document declares exactly one kind - `howto`, `reference`, `adr`, `readme`, `changelog` and a handful more - and holds only what that kind is for. A how-to gives steps; the reasoning belongs in an ADR.
- Each fact has one authoritative home. Other pages link to it rather than retelling it.
- Long-lived pages do not record values that change faster than the page: current versions, deployment status, counts. Those live in dated records or behind pointers.
- A pointer names a file or a symbol, so the reader does not have to search for what the sentence promised.
- The finished page does not address whoever asked for it and does not narrate how it was made.
- An exception to any requirement needs a written reason. An exception without one is itself a violation.

That last rule is the character of the whole thing. seiso does not guess whether prose sounds machine-written, and it leaves formatting and spelling to other tools. What it checks is structural honesty: who owns this fact, where the pointer leads, whether the page will still be true next month.

## One CLI, six commands

The binary is a clap-driven Rust CLI whose subcommands are `check`, `policy`, `index`, `rule`, `parse`, and `init`. `seiso init` writes a `seiso.toml` at the repository root with suggested exclusions and kind mappings - the repo dogfoods its own, mapping `README.md` to `readme`, `CONTRIBUTING.md` to `howto`, and the `docs/**` tree to `reference`, with release notes mapped to `changelog`.

`seiso check` runs only stable rules by default; preview rules need `--preview` and are kept out of CI gates by policy. Output formats cover plain text, concise, JSON, SARIF, and GitHub annotations, selected with `--output-format`. The `--fix` flag applies verified safe fixes, `--statistics` counts reported diagnostics and suppression states, and `--stdin-filename` checks a buffer in place of a named workspace file - the hook editor integrations use. `seiso rule <CODE>` explains any rule with examples, and `seiso parse` dumps the document model without running rules.

## How the checker is built

The crate splits into five load-bearing layers, and the architecture document in `docs/reference/architecture.md` describes them honestly. `src/workspace.rs` owns discovery, policy resolution, scoped reads, and parsing, producing a snapshot whose loading scope is independent of rule execution - `parse` reads only selected documents, while a `check` widens to workspace documents whenever an enabled rule needs the index. That subtlety matters: cross-file diagnostics can attach a related location to a file you selected, so the tool cannot infer its dependency scope from your command line alone.

The document model in `src/md/` is where precision lives. `src/md/parser.rs` builds a model of frontmatter, sections, blocks, sentences, and fragment kinds; every fragment retains a byte range into the original source, and `src/md/mapping.rs` keeps display columns counting Unicode characters, marking ambiguous synthesized mappings as inexact. `src/sections.rs` adds heuristic section roles with evidence, computed separately from the cached parse facts. `src/index/mod.rs` combines those parsed facts with current kind, language, domain, and path policy into the workspace index that cross-file rules consume.

Link checking lives in `src/paths.rs` and handles the details that usually make doc linters annoying. Local link targets are tried as written and, for documents with a `[[sites]]` entry, as the route of a documentation site. Percent decoding, scheme detection, workspace containment, and normalization are shared between file-existence and index rules. Letter case is compared with real entry names through `paths::Listings`, so a case-insensitive filesystem resolves the same targets Git would. GitHub-style heading slugs include duplicate suffixes, and HTML `id` and anchor `name` attributes are indexed. Existence and anchor knowledge stay distinct: a file outside the parsed sample can exist while its anchors remain unknown, and no missing-target diagnosis is issued for external URLs or template values.

## Twenty-four rules with receipts

The rule catalog in `docs/rules/` has twenty-four documents, each stating what the rule checks, its basis (normative or heuristic), why it matters, and a before-and-after example. A sample of the flavor:

- `DUP001` compares definition lists and tables across files in the same language and domain - a qualifying pair shares at least five distinct inline-code identifiers with identifier-set Jaccard similarity of at least 0.8, which is a precise way of saying "these two pages maintain the same facts and will drift."
- `STL001` flags a sentence that contains a current-state marker plus a version, hash, image tag, or count without a requirement or range marker - "the service currently uses v1.2.3" in a page that outlives releases.
- `VOX001` catches conversation remnants: phrases addressing the requester of an earlier conversation, like "as you requested" in an installation guide.
- `EVD001` flags comparative quality claims without a nearby source or measurement - "this implementation is faster" with nothing a reader could check.
- `LNK001` and `LNK002` define the diagnostic boundaries for broken targets and anchors.

The registry in `src/rules/mod.rs` owns each rule's execution phase, stability, applicable kinds, fragment requirements, and whether it needs the index; that last flag drives workspace dependency loading. `src/rules/normative.rs` and `src/rules/heuristic.rs` hold the single-document checks, `src/rules/cross_file.rs` the index-dependent ones, and `src/rules/links.rs` adapts filesystem listings and frozen Git inventories behind a `WorkspaceFiles` trait so rules never touch the disk directly.

Suppression is engineered rather than bolted on. `src/rules/suppression.rs` carries suppression codes as structured diagnostic data, so a wording change cannot silently change which codes a fix removes. `seiso policy --evaluate` records actual states - active, stale, incomplete - while plain `policy` only inspects declarations. The fix path in `src/rules/fixes.rs` is similarly defensive: when an edit plan exists, the tool reruns a fresh analysis, confirms the inputs and diagnostics, verifies source equality immediately before writing, and re-checks the result. Incomplete checks cannot authorize edits.

## Speed and caching

`src/cache/mod.rs` caches only content-derived parse data, keyed by content hash - modification times never determine freshness, so moving identical content still gets its new policy applied. Writes use temporary files with atomic persistence, concurrent processes do not share mutable parsed state, and once a day a cache-writing command prunes entries older than `MAX_ENTRY_AGE`. The cache directory carries its own `.gitignore` and `CACHEDIR.TAG`. The Cargo.toml forbids unsafe code outright (`unsafe_code = "forbid"`) and sets mimalloc as the global allocator, which fits a tool whose job is chewing through thousands of Markdown files on every save.

## Why it earns a spot in your repo

The honest comparison target is markdownlint, and seiso is deliberately not competing with it. markdownlint checks syntax; seiso checks the sociology of the document set - who owns which fact and whether the page will stay true. The evaluation records under `docs/evaluation/` are dated m0 through m3 reports with baselines and gates, and the promotion policy in `docs/evaluation/policy.md` states the evidence a rule needs before it stops being a preview. A linter that documents its own promotion criteria with the same rules it sells is making a testable claim, and that is exactly the kind of documentation the convention asks for.

Adoption cost is one command. `seiso init` writes the config, you review the kind mappings, and `seiso check` tells you how far your docs are from the convention - each diagnostic says where the problem is, how to fix it, and names the judgment call when a fix needs a human decision. For repositories where agents both write and read the docs, that loop closes the biggest trust gap in the modern workflow.
