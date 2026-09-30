---
layout: post
title: "Intent-Router: Typed IntentSpec Contracts for AI Agents - Inside angel291592/Intent-Router"
description: "Intent-Router is an open-source intent compiler for AI agents that converges a vague request into a typed IntentSpec contract before any tool runs. A source-level tour of its three-pass engine, PROBE/ASK/HALT decision states, JSON Schema contract, and its seventeen-case evaluation harness."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Intent-Router-Typed-IntentSpec-Contracts-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/intent-router/angel291592-intent-router-architecture.svg
tags:
  - AI Agents
  - Prompt Engineering
  - Intent Routing
  - Open Source
categories: [AI, Open Source]
keywords: "intent router, IntentSpec, AI agents, intent compiler, agent skills, prompt engineering, Claude Code skills, intent routing, structured prompts, agent clarification, JSON Schema, OpenCode, MIT license, angel291592"
author: "PyShine"
---

Hand an AI coding agent a request like "add caching to the user API" and one of two things happens. Either the agent guesses — it picks a cache backend, a TTL policy, and a failure behavior you never asked for, and produces four hundred lines of confident work built on an assumption you never made. Or it interviews you — round after round of clarifying questions, many of them about things your own `package.json` or route definitions already answer. Both failures have the same root cause: nobody decided whether the missing information was worth asking a human for. A compiler does not guess the address of an undefined symbol; it goes and looks in the libraries it was given. Most agent harnesses never take that step.

Intent-Router, by GitHub user angel291592, is an intent compiler for AI agents that closes exactly this gap. It sits one layer before planning or coding: it takes an underspecified request and converges it into a typed, machine-readable contract called an `IntentSpec` — a normalized intent, the objects it acts on, evidence-backed constraints, and an explicit list of what is still unknown. Only then does it hand the contract off to your planner, agent, or workflow, and it refuses to emit anything at all while the intent remains underspecified. The project ships as an "Agent Skill": a `SKILL.md` procedure plus reference documents and a JSON Schema, installable with a single `npx skills add` command into Claude Code, Codex, Cursor, OpenCode, Gemini CLI, and many other harnesses.

The source is worth a tour because it is an unusually honest piece of engineering. The decision engine is a readable Markdown procedure rather than a black-box service; the contract it emits is pinned by a formal JSON Schema with a worked example for every outcome; and — rarest of all — the repo's claims are backed by a seventeen-case evaluation harness in Python that starts real agent sessions against fixture repositories, where a single invented evidence pointer fails the whole suite. Reading it is a masterclass in turning "the model should ask better questions" from a prompting vibe into a decidable state machine.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/intent-router/angel291592-intent-router-overview-architecture.svg" alt="Architecture overview of the angel291592/Intent-Router repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Intent-Router repository: the SKILL.md engine resolves unknowns through probe surfaces or a structured ask, emits a schema-validated IntentSpec that is saved under .intent/, and is held accountable by the Python evaluation harness.*

Reading the overview from left to right: the README installs the skill, whose engine in `skills/intent-router/SKILL.md` runs three compiler-style passes. During resolution it leans on the probe-surface catalog in `references/probe-surfaces.md` to look answers up, or on `references/ask-protocol.md` when a question genuinely belongs to a human. Everything it learns flows into the IntentSpec contract defined by `skills/intent-router/schema/intentspec.schema.json`, with worked examples for each decision state under `schema/examples/`, and every emitted spec is saved to a `.intent/` file following the convention documented in `references/intentspec.md`. Below it all, the evaluation layer — `evals/run.py`, `evals/cases.yaml`, and the fixture workspaces — scores the engine's real behavior and publishes the reports the README quotes.

## Why You Need This

The first problem is the guessing problem. When a request leaves decisions open — which objects, which approach, what happens on failure — an agent fills the gaps silently and keeps moving. Intent-Router's answer is structural: during parsing, anything the model had to supply itself is tagged `source: inferred`, so you can veto it in one glance, and an inferred value that touches an irreversible boundary blocks emission entirely. The skill would rather halt and name the open field than guess what you meant.

The second problem is the interview problem. Grilling the user is better than guessing, but it has a bill: many rounds of questions, most of which the workspace could answer. Intent-Router draws a hard line — its "iron law" states that if an objective answer exists anywhere it can reach, it must probe and never ask. Asking is reserved for answers that live in a person's head (preferences, priorities) or for decisions that cannot be walked back once shipped. In practice that collapses a forty-question interview into the one question no file can answer.

The third problem is that natively, two very different failures look identical. A probe that failed because a tool timed out and a request that is genuinely underspecified both surface as "I need more information." Intent-Router keeps them apart forever: a halt with cause `underspecified` is your move as the user, while cause `degraded` is an operations signal that something in the environment broke. Merge them, and a backend outage hides behind what looks like a clarifying question for a week.

Finally, there is the persistence problem. Answers given in conversation die with the session — compress the context, switch models, open a new window, and they are gone. Because every spec is written to `.intent/<intent>.intent.yaml` before the reply that carries it, the next agent, session, or teammate reads that file before probing anything: an answered question is never asked again, a settled contract is implemented as written, and the result of checking the delivered work is written back into the same file.

## How It Works

Intent-Router runs three passes in compiler order — Parse, Resolve, then Typecheck and emit — and resolves every open question through exactly one of four decision states: ROUTE, PROBE, ASK, or HALT.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/intent-router/angel291592-intent-router-architecture.svg" alt="Detailed architecture of the angel291592/Intent-Router repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the three passes inside SKILL.md, the on-demand reference documents they load, the IntentSpec schema and its state examples, and the evaluation harness that measures all of it.*

### Understanding the Architecture

**The engine is a document, not a daemon.** The entire decision procedure lives in `skills/intent-router/SKILL.md`, a single Markdown file with frontmatter (`name`, `description`, version 1.2.0, MIT license) that skill-aware harnesses load automatically. It deliberately names no tools — it asks only for "whatever file-reading, search, or shell capability your environment provides" — which is why the same file runs in Claude Code, Codex, Cursor, OpenCode, and dozens of other agents, with per-harness install paths cataloged in `references/harness-compat.md`.

**A silence check keeps a fully specified task untouched.** Before probing anything, the engine runs a four-part check against the request text alone: does it name objects precisely, name an approach, state a failure rule for every fallible step, and name what counts as done? If all four hold, one cheap premise check (at most two lookups) verifies no source contradicts the request — and if nothing does, the skill stays completely silent and the work proceeds as ordinary. A spec emitted after a passed silence check is treated as a false positive that costs more than the skill saves.

**Parse turns prose into a draft contract.** Pass 1 in SKILL.md records the user's own words verbatim (capped at 500 characters, never paraphrased), normalizes the action into a snake_case `intent` like `add_caching`, lists the objects, copies stated constraints in as `source: explicit`, and enumerates the unknowns across six categories from `scope` to `acceptance`. It also screens for three request defects that filling in cannot fix — `ambiguous` (two readings), `conflict` (two stated requirements that cannot both hold), and `premise` (a source that contradicts what the user stated) — and asks about those before anything else.

**Resolve applies the iron law: probe before ask.** Pass 2 classifies every decision-bearing unknown before saying a word to the user. Probes work an ordered surface list — dependency manifests first, then entry points and route definitions (the authoritative inventory of what exists), same-kind implementations, configuration, tests and CI, and finally version history — with a budget of three probe actions per unknown plus one reserved history query. `references/probe-surfaces.md` extends the catalog across ecosystems, and `references/domains.md` retargets the same machinery at ticket queues, policy archives, and research notes. When asking is unavoidable, `references/ask-protocol.md` fixes the shape: one question at a time, within a default budget of three, always with a "why you, not me" justification, a recommended option, and two or three concrete choices.

**Typecheck makes stopping computable.** Pass 3 evaluates a predicate — the spec is sufficient when the `unknown` list is empty and no constraint is both inferred and irreversible — and then emits exactly one of three outcomes. ROUTE hands off the complete spec with a target; ASK emits a snapshot plus the single pending question; HALT names either the `open_fields` (underspecified) or the failed lookups (degraded). The machine-checkable form of the contract is `skills/intent-router/schema/intentspec.schema.json`, JSON Schema draft 2020-12 with `additionalProperties: false`, and `schema/examples/` contains a worked spec for each state, including `route-support.yaml` for the non-code support-queue domain.

**The contract outlives the session and gets checked.** Every emitted spec is saved to `.intent/<intent>.intent.yaml` before the reply, with `references/intentspec.md` documenting the file convention and lifecycle statuses (`asked`, `routed`, `verified`, `superseded`). When the handed-off work is carried out in the same conversation, the skill rechecks the finished work against every constraint and reports one line each under an `Intent check` heading — `met` with a `path:line` pointer, `not met`, or `not checkable here` — and writes those results back into the same file.

**Claims are measured, not asserted.** `evals/run.py` is a Python runner (needing Python 3.13+ and `uv`) with cost-tiered modes from an offline `--selftest` up to the full suite, plus a `--delivery` mode that runs the same weak request with and without the skill and scores the resulting workspaces. The seventeen cases in `evals/cases.yaml` run against real fixture workspaces — a TypeScript `user-api` service and a `support-queue` of orders, tickets, and policies — and per `evals/README.md`, they assert the decision state, the counters, and that every evidence pointer names a file that actually exists; one invented citation fails the case. Results land as Markdown in `evals/reports/`.

The end-to-end flow is therefore: a vague request arrives; the silence check decides whether to engage at all; Parse drafts the spec and flags its unknowns; Resolve looks most of them up and asks you at most one well-formed question; Typecheck either routes a complete, schema-valid IntentSpec to your executor — saved under `.intent/` for the next session — or halts with a named cause; and after the work is done, the same contract is checked line by line against what was actually delivered.

## Advantages

- **Probe-first discipline.** The counters `resolved_by_probe / unknowns_found` are recorded in every spec, and a question whose answer was sitting in the workspace is defined as a defect, not a courtesy — you are interrupted only when a human judgment is genuinely required.
- **A decidable stopping condition.** Sufficiency is a predicate over required fields plus a hard ASK budget of three, so clarification runs terminate by computation instead of drifting on because it "felt" almost done.
- **Two failures, two signals.** `underspecified` and `degraded` halts are mutually exclusive by design — one tells you what to decide, the other pages an operator — instead of collapsing into a single fallback the way most clarify/route libraries do.
- **Evidence on everything.** Every probed or inferred constraint carries a single evidence token (`path:line`, `git:<short-sha>`, `record:system/id`, `doc:slug#section`), and the eval suite fails any run whose pointer names something that does not exist.
- **A typed, validated artifact.** The IntentSpec is pinned by a draft-2020-12 JSON Schema with fixed required fields and per-state examples, so downstream planners and auditors consume data, not prose.
- **Zero-install portability.** The default tier is prompt-only — no API key, no dependencies, no server — and the skill format loads across a long list of agent harnesses or pastes straight into a system prompt.

## Benefits

- **Senior-engineer interaction without prompt-engineering skills.** You say "add caching to the user API" the way you would to a capable teammate, and the skill writes the briefing a senior engineer would have written first.
- **Fewer, better interruptions.** Because lookups happen before questions, you face the one preference or irreversible trade-off that no file could answer, with options and a recommendation ready.
- **Continuity across sessions and teammates.** The `.intent/` contract means the next agent starts from what was actually decided — answered questions stay answered, and settled work is not redone or contradicted.
- **Safer irreversible choices.** The engine refuses to route when an inferred value would cross a boundary that clients or data cannot undo, naming the field instead — the one failure worse than a question is a silent irreversible guess.
- **Auditable runs.** Every spec carries a replayable `trace` of `parse`, `probe`, `ask`, `typecheck`, and `emit` steps, so "what was this change actually trying to do" has a file-backed answer instead of archaeology.
- **Domain-independent semantics.** The same three passes and stopping predicate work in a codebase, a support queue, or a research brief — only the probe surfaces change, and `references/domains.md` shows how to write your own.

## Usage

Install as a skill — the installer detects the agents you have and drops the skill in as `intent-router`:

```bash
npx skills add angel291592/Intent-Router
```

Manual install: copy the `skills/intent-router/` directory into `.claude/skills/` (Claude Code) or `.agents/skills/` (Cursor, Codex, OpenCode, Gemini CLI, and others) — see `references/harness-compat.md` for per-harness paths. Then just work; the skill is designed to fire when a request arrives underspecified. To force it:

```text
/intent-router refactor the auth module
$intent-router refactor the auth module
/skill:intent-router refactor the auth module
```

What you get back is a fenced YAML block — the whole spec, in field order, preceded by a few lines of plain language about what was looked up and decided:

```yaml
spec_version: "0.1"
request: add rate limiting to the public endpoints
intent: add_rate_limiting
objects:
  - POST /api/login
  - POST /api/signup
constraints:
  - source: explicit
    text: apply it to the public endpoints only
    category: scope
  - source: probed
    text: 'the entry point already registers a rate-limit middleware'
    evidence: src/app.ts:24
    category: approach
unknown: []
decision:
  state: ROUTE
  confidence: 0.82
  target: implement
resolution:
  unknowns_found: 2
  resolved_by_probe: 1
  asked: 1
  inferred: 0
  ask_budget: 3
trace:
  - step: parse
    detail: two decision-bearing unknowns
  - step: emit
    detail: unknown is empty and no inferred constraint is irreversible — sufficient; handing off
```

To run the evaluation suite yourself (requires Python 3.13+, `uv`, and a supported harness on PATH), the repo's cheapest-first tiers look like this:

```bash
uv run --with pyyaml --with jsonschema python evals/run.py --selftest
uv run --with pyyaml --with jsonschema python evals/run.py --smoke --harness opencode
uv run --with pyyaml --with jsonschema python evals/run.py --harness opencode --repeat 1 --jobs 4
```

## Conclusion

Intent-Router takes the fuzziest part of working with AI agents — what did the user actually mean? — and treats it like a compiler treats source code: parse it, resolve the unknown symbols by looking them up, typecheck the result, and refuse to link when something is still undefined. The result is a small repository with an outsized amount of discipline: a Markdown engine you can read in one sitting, a schema-validated contract that outlives the session that produced it, and an evaluation harness that fails the project's own claims the moment an evidence pointer points at nothing. If you build with coding agents and are tired of both the guessing and the forty-question interviews, the source is well worth your afternoon.

Links:

- GitHub repository: [angel291592/Intent-Router](https://github.com/angel291592/Intent-Router)
- Evaluation guide: [evals/README.md](https://github.com/angel291592/Intent-Router/blob/main/evals/README.md)
- Harness compatibility: [references/harness-compat.md](https://github.com/angel291592/Intent-Router/blob/main/skills/intent-router/references/harness-compat.md)
