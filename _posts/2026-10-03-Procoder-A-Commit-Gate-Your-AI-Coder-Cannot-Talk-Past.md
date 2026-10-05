---
layout: post
title: "Procoder: A Commit Gate Your AI Coder Cannot Talk Past"
permalink: /Procoder-A-Commit-Gate-Your-AI-Coder-Cannot-Talk-Past/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/procoder/azrtydxb-procoder-architecture.svg
tags: [Go, AI, Code Quality, Developer Tools, CI]
---

The weakest link in AI-assisted development is the moment the agent says "done". Nobody is standing behind it. The tests may not have run, the formatting may be off, a merge conflict marker may be sitting in a staged file, and the sentence "all checks pass" costs the agent nothing to say. [Procoder](https://github.com/azrtydxb/procoder) by azrtydxb is a Go binary that makes that sentence expensive. Version 3.7.0, Apache-2.0, is a harness for 20+ coding agents - Claude Code, Cursor, Windsurf, Cline, Kilo Code, Roo, Kiro, Codex CLI, Copilot CLI, Gemini, OpenCode, and anything that reads `AGENTS.md` - built on one principle the codebase calls P-CONTROL: the binary computes and reports, the agent acts, and nothing ever touches your code behind its back.

The engine is a single Go program - `cmd/procoder/main.go` wires up roughly fifty packages under `internal/` - and the hooks and skills that each agent adapter installs are thin callers into it. That matters for consistency: `check`, the git hook, and CI all run the same gate code, so they cannot disagree about what is clean.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/procoder/azrtydxb-procoder-overview-architecture.svg" alt="Architecture overview of the azrtydxb/procoder repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the repository: one Go binary fanning out from the gate to concurrent check legs, the workflow chain of specs, plans, backlogs, todos and releases that can refuse, the hook layer that fires on every agent file write, and the self-learning loop of lessons and adaptations.*

Reading the overview from left to right: `cmd/procoder/main.go` boots the command surface, which dispatches to `internal/gate/gate.go` for the commit gate - its legs (secrets, lint, the test suite, complexity, debt) run concurrently but report in a fixed order. The workflow chain in `internal/spec`, `internal/plan`, `internal/backlog` and `internal/todo` hands work to `internal/release` at the end, each stage able to refuse until its own acceptance criteria are met. `internal/hook/hook.go` plugs the same gate into every agent's PostToolUse event, and `internal/lessons` with `internal/learn` close the loop when a bug escapes anyway.

The commit gate, `procoder check`, covers formatting across the popular languages (Go, Python, JS/TS/HTML/CSS, PHP, Rust, C/C++, Java, Kotlin, Swift, Ruby, Dart, C#, shell - one canonical formatter each, with the project's own config always winning), git hygiene (conflict markers, junk files, oversized files, AI-attribution lines), secrets, lint, CI and infra hygiene, and documentation health. The implementation in `internal/gate/gate.go` runs its legs concurrently - the source comments measure it on this repository's 787 tracked files: gitleaks 41.2s, the suite 35.0s, semgrep 25.3s, lint 2.8s, osv 2.6s, complexity 1.1s, debt 0.2s - 108 seconds in a row, about 41 with the longest leg setting the pace. Crucially, the order of results is fixed regardless of which leg finishes first, because a report that reorders itself between runs would make "did my change cause this?" unanswerable.

The most interesting piece of the gate is `internal/gate/adoption.go`, which decides how much of procoder a repository is subject to. An **adopted** repo (one with a `.procoder/` directory, or an `AGENTS.md` naming procoder) gets the full set of house rules. Somebody else's repository only gets the checks that are true anywhere - secrets, oversized files, conflict markers, junk - and content-reading checks see only the lines your commit wrote. The decision is made from the repository itself, never the environment, with one escape hatch: `PROCODER_GATE_SCOPE` for a fork you are about to submit upstream where adding config would itself be a change you do not want. The comment explains the failure direction precisely: absence of evidence is not adoption, and it is better to say less about somebody else's code than more.

## Hooks that cannot be skipped

Self-serve commands - check, format, lint, scan, index, audit - are only half the story. The other half is `internal/hook/hook.go`, the PostToolUse handler the agent cannot avoid: every file write fires the gate over that file, and the findings come back as `additionalContext` in the same turn. The hook is defensive in the small details. Reading the host's payload runs in a goroutine with a hard 5-second wall, because a previous implementation's deadline only covered EAGAIN and a session-start hook was observed blocked for 31 minutes. Past roughly two kilobytes, the host inlines only a preview and a path, so unbounded payloads cannot wedge a session. `internal/hook/stop.go` handles session start, injecting engineering principles and stale-state warnings. And a tool that failed is never reported as clean - "unchecked" is its own count and counts as failing, said out loud in every summary line.

## The quality chain: refusals all the way down

Procoder's signature move is taking advice-shaped workflow tools and giving each one a controller that can refuse. `internal/spec/spec.go` runs a spec interview whose `spec check` blocks while sections are missing or questions stay open (`internal/spec/coverage.go` and `internal/spec/truth.go` do that accounting). `internal/plan/plan.go` blocks plans with placeholders, on the standard that a plan should be executable by an engineer with zero context. `internal/backlog/board.go` holds larger work as milestones, epics and user stories seeded from specs, worked in scope-boxed sprints (`internal/backlog/sprint.go` - one active sprint, explicit carry-over) with closes that refuse (`internal/backlog/close.go`). `internal/todo/todo.go` tracks standalone work, and `todo close` refuses until every acceptance criterion is checked, evidence is recorded, and the gate is clean.

The test domain feeds those refusals: `internal/testrun/testrun.go` runs the repository's real suite with each ecosystem's canonical runner - go test (parsed from its JSON stream by `internal/testrun/gojson.go`), cargo test, the package.json test script, pytest, gradle/maven - and reports PASS with counts, FAIL with failing tests named, or NOT run, which is never the same as green. Set `[test] policy = "block"` and a green suite becomes part of "done": the closes refuse while it is red or unverifiable.

Then `internal/release/release.go` is the last refusal before a tag: version in sync across every file you list, changelog entry present, tree clean, gate clean, suite green - every failure in one list, and on success the `git tag` command is printed for you to run. Procoder tags nothing itself.

## The self-learning loop

What happens to bugs that escape anyway? They become entries in a lessons ledger (`internal/lessons/lessons.go`), and the adaptation - a linter rule, a rubric line, a pinning test - must land before the work counts as done (`internal/learn/learn.go`). Even the fallback net's catch is harvested: `internal/lessons/copilot.go` implements `procoder copilot-leak`, which collects GitHub Copilot's auto-review findings, strips every trace of your code from them, and only after a terminal confirmation files them as issues recorded as unlearned until someone writes the adaptation that closes the class.

## The index, and the honesty details

`internal/codeindex/` is the agent's fast map - ctags plus SCIP under the hood, with find, refs, callers, impact, unused, and entrypoints (`graph.go`, `query.go`), plus a rename command that computes the diff and hands it to the agent rather than writing files. `internal/store/state.go` and `internal/store/atomic.go` persist all of this with atomic writes and locking. Onboarding an existing codebase is `procoder audit` (`internal/audit/audit.go`), a triaged scorecard.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/procoder/azrtydxb-procoder-architecture.svg" alt="Detailed architecture of the azrtydxb/procoder repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the same repository: the command binary with its config, principles and audit entry points, the commit gate split into adoption scoping and the secrets, formatting and hygiene legs, the test runner parsing go test JSON alongside the other ecosystems, the workflow chain from spec interview through plans, sprints, todos and closes to the release refusal, the lessons ledger with Copilot-leak harvesting feeding the adaptation loop, the ctags-plus-SCIP code index with graph, query and rename services over an atomic state store, and the portability layer that generates and drift-checks each agent's hooks and manifest.*

The portability layer is what makes one binary serve twenty agents: `internal/portability/portability.go` generates the per-agent adapters (hooks JSON, manifest, one instruction file), and `internal/portability/drift.go` checks that the generated files have not drifted from what the current version would produce. Config lives in `.procoder/` as plain editable files - `config.toml`, `PRINCIPLES.md`, the review rubric, the lessons ledger - where the repository's version always wins over the built-in default.

Procoder absorbed three earlier tools - superpowers (zero-context plans, evidence before done), ponytail (the build ladder and debt markers), serena (symbol navigation, now the index with no MCP server to run) - and ships a provenance map saying exactly where each replacement stops. A harness that enforces honesty in others, and documents its own borrowing, is rarer than it should be. If your AI coder keeps declaring victory early, this is the referee to install.
