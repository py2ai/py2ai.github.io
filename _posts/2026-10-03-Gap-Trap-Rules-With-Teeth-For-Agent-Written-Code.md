---
layout: post
title: "Gap-Trap: Rules With Teeth For Agent-Written Code"
permalink: /Gap-Trap-Rules-With-Teeth-For-Agent-Written-Code/
image: https://pyshine.com/assets/img/diagrams/gap-trap/pliablepixels-gap-trap-architecture.svg
tags: [AI, Developer Tools, CI, Code Quality, Testing]
---

Every team running coding agents has the same quiet problem: the rules exist, but nothing enforces them. The agent writes a helper that already exists, crosses a layer boundary because the shortcut compiled, and follows a rule in your instructions file for a week before forgetting it. The rules are text, and the only thing that reads them is the agent that breaks them. [gap-trap](https://github.com/pliablepixels/gap-trap) by pliablepixels closes that loop. It is an MIT-licensed agent skill that, run once in a repository, reads the code, writes the rules that fit that codebase, and attaches a gate to each rule - a check that fails the commit or the CI run when the rule breaks. The agent cannot skip a gate, because gates are just tests.

The origin story in `gap-trap/reference/framework.md` is concrete: one maintainer landed 2,313 commits in eight months - fourteen reverts - with agents writing the code and no one reading diffs. The framework is the distillation of what stopped the drift.

## Four kinds of instruction

The framework separates what agents read into four kinds with a clear precedence. **Rules** live in `AGENTS.md` with an ID, a statement, a one-clause why, and a gate name - IDs carry a tier letter (I for invariants that are never traded away, P for process, C for code, M for meta-rules governing the instruction files themselves). **Contracts** live in `AGENTS.project.md`, four lines per subsystem with one sanctioned path:

```
### HTTP
Owns: all network requests, including TLS handling.
Path: helpers in `src/lib/http.ts`.
Never: raw `fetch` or `axios`.
Gate: `tests/instruction-gate.test.ts` (no raw `fetch`/`axios`).
```

An agent reads that and learns, without opening source, who owns the concern, the one way to use it, that a working bypass is still a bug, and what fails if it bypasses. **Practices** are playbooks under `agents/` - advice backed by commit hashes, no IDs, losing to rules on conflict. **Facts** in `domain-context.md` record API quirks and failed approaches with hashes. Docs cite rule IDs rather than copying text, so a rule changes in exactly one place.

## The gates, cheapest first

`gap-trap/reference/gates.md` specs six kinds. A **grep gate** scans source for a forbidden pattern in milliseconds. The **instruction gate** is the clever one - a test over the instruction files themselves: every backticked token on a `Path:` or `Gate:` line must exist in the tree (a token with a slash is a file path, any other is a symbol resolved against the paths its own line names, and a symbol that survives only in tests does not count), the always-loaded files must stay under a word budget, every 8-hex commit hash cited in the knowledge files must exist (`git cat-file -e`), and no email or IP address may appear there. A contract with a guessed symbol name fails this gate immediately - the failure names the contract and the token.

The **ratchet** stores a count - lint backlog, files over 400 lines, existence-only assertions - that may fall or hold but never grow; raising it by hand needs a reason in the commit message. **Proven red** is the check most test suites lack: CI runs the tests a changed range touched against the code from before that range, in a worktree at the fork point, and fails when they pass there. A test that cannot fail proves nothing, and the gate distinguishes a real assertion failure from a mere missing symbol. The **mutation smoke** flips one branch in each risky module (auth, a parser, the API client) and requires that module's tests to fail - the one check proving existing tests can fail at all. And the **PR body check** requires the body to quote the issue's acceptance lines.

## One skill, two modes

The whole thing ships as a skill folder: `gap-trap/SKILL.md` plus reference docs and templates. `setup` builds the framework in a repo that has none; `refine` audits a repo that has it and proposes the next rules from what actually broke, through a self-improvement protocol that lands as one PR. Bare `gap-trap` means setup when there is no `AGENTS.project.md`, refine otherwise. Discovery (`gap-trap/reference/discovery.md`) produces a plan file where every claim is verified by a command that was run - stack from the manifest, test/lint/typecheck commands each confirmed by executing once and recording the output line that proves it, CI provider, branch protection, and a migrated-rules table for existing instruction files like `.cursorrules` or `copilot-instructions.md`, which get reduced to one-line pointers to `AGENTS.md` so no agent lands on a stale copy.

The setup sequence is unusually disciplined. Confirm once with at most six contracts. Install the sibling [slop-mop](https://github.com/pliablepixels/slop-mop) skill so docs, commit messages, and PR bodies read like a person wrote them. Then prove every gate red with a scratch violation before it lands, remove the violation, and commit one logical change per commit - instruction files with their gate first, then ratchet, proven red, mutation smoke, PR body check, CI last - because one commit holding the whole framework hides which gate was proven red against what and leaves nothing to revert on its own.

## A model gate, stated as text

The boldest design decision is at the top of `SKILL.md`: gap-trap refuses to run on a weaker model. Discovery decides which parts of a codebase get a contract, and a wrong contract costs every later session, so the skill specifies the exact message to print and the turn to end if the harness is running a cheaper tier - no reading, no writing, no dispatching stronger subagents from a weaker orchestrator. It is a quality gate implemented purely in prose, and the fact that it is even necessary says something about the era.

The templates cover the portability matrix pragmatically. Node repos get `instruction-gate.test.ts`, `proven-red.mjs`, and `ratchet.mjs` inside vitest or jest; Python gets `instruction_gate_test.py` and shell proven-red/ratchet around pytest; everything else - Go, Rust, Java, Ruby, .NET, PHP, Swift, C++ - gets shell versions needing only git, grep, awk, and the repo's test command, with `gap-trap/templates/gates/github-ci.yml` as the CI starting point and a per-stack table of unit-test file patterns for the proven-red classifier. The template implementations are themselves tested in `tests/` with Node's built-in runner, and `scripts/release.sh` cuts tagged releases with dry-run support while refusing to run off main or with uncommitted changes.

The project is honest about its limits: validate the generated rules and contracts, because that is the foundation everything later sessions build on; `Gate: review` is allowed only where no text search can settle a Never clause, and those unverified spots are listed explicitly in the post-setup check. If your agent-written codebase is drifting, this is the inner loop that bends it back.
