---
layout: post
title: "AgentMeasure: Open Measurement Infrastructure for AI Agents - Inside roy-tong/AgentMeasure"
description: "AgentMeasure is an open measurement layer for AI agents: a zero-dependency Python CLI that audits Codex and Claude Code session logs for duplicate records, retry inflation, and token-accounting errors, plus a conformance harness that turns measurement assumptions into PASS / FAIL / UNPROVABLE CI checks. This source tour walks the adapters, check families, vector runners, and the evidence pipeline behind its published ecosystem audit."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /AgentMeasure-Open-Measurement-Infrastructure-Agents-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/agentmeasure/roy-tong-agentmeasure-architecture.svg
tags:
  - AgentMeasurement
  - AI Agents
  - Observability
  - Open Source
categories: [AI, Open Source]
keywords: "AgentMeasure, AI agent measurement, token accounting, billing audit, Codex rollout logs, Claude Code session logs, conformance testing, telemetry invariants, UNPROVABLE, settlement statement, AMS-1, usage analytics, retry amplification, open source Python"
author: "PyShine"
---

When an AI service starts charging by the outcome — per resolution, per task, per conversation — the interesting question stops being "how much" and becomes "what exactly did you count". Agents produce enormous amounts of runtime evidence in the form of rollout logs, session transcripts, and telemetry events, yet the numbers built on top of that evidence are usually computed by whichever tool happens to grep the logs first. Mixing execution facts with logical operations is how one user intent quietly becomes three billable operations, and how a dashboard inherits an error nobody ever decided to make.

AgentMeasure, maintained by roy-tong, is an open measurement infrastructure project that attacks this problem from both ends. At the tooling end sits `agentmeasure`, a PyPI-distributed Python CLI that reads the Codex rollout logs and Claude Code session logs your machine already writes and produces a check-up report with deterministic verdicts. At the standards end sits a set of normative documents under `standard/` (CORE, METRICS, QUALITY, SETTLEMENT), a machine-readable registry under `registry/`, and a conformance harness under `conformance/` that turns measurement assumptions into test vectors any implementation can be run against.

The source is worth a tour because its most distinctive rule — missing evidence is reported as UNPROVABLE, never silently zeroed — is not a slogan in the README. It is enforced structurally in the code: in the verdict enum of the check engine, in the fail-closed logic of the conformance pack, and in the whitelist sanitizer that decides what may leave your machine. Reading how those pieces fit together tells you a lot about how a measurement probe should be built, what an audit harness actually runs, and how findings get verified before they are published.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/agentmeasure/roy-tong-agentmeasure-overview-architecture.svg" alt="Architecture overview of the roy-tong/AgentMeasure repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the AgentMeasure repository: local probes in the healthcheck package feed deterministic checks; a conformance harness replays language-neutral vectors; outputs range from HTML reports to AMS-1 settlement statements, all anchored to a canonical observation schema.*

Reading the overview from left to right: the `agentmeasure` CLI is the front door; it auto-discovers runtime logs and hands them to the Codex or Claude Code adapter, both of which parse raw JSONL into the shared session model that the HC-01 through HC-06 checks consume. Verdicts and evidence flow into terminal and HTML reports, while the settlement path generates dispute-ready bundles. In parallel, the conformance side — the embedded pack and the repository-level vector runners — validates telemetry fixtures against the same invariant discipline, and the TypeScript provider SDK emits observations that conform to the canonical observation schema the standard defines.

## Why You Need This

The core problem AgentMeasure names is unit confusion. One assistant message in a Claude Code transcript is written as one JSONL line per content block; one Codex operation may surface as both a `response_item` record and an `item_completed` event; a retry chain is several executions of one logical operation. A tool that sums lines or events instead of canonical operations reports inflated usage, and a tool that treats a cached-input subset as an extra charge misprices the session. The project's own audit report, `campaigns/audit-report-2026-09.md`, headlines roughly 110 AI usage tools audited with 45-plus verified billing bugs across five recurring classes — per-line overcounting, re-emitted events, cache-pricing confusion, price-table drift, and resume/fork history loss. Those are their numbers and their filings, but the classes are real and reproducible with synthetic fixtures.

If you are on the buying side of outcome-based pricing, the problem is sharper: the rules are written by the seller and the bill is computed by the seller. AgentMeasure's answer is the recount-and-dispute path in `healthcheck/am_healthcheck/vendors.py` and `dispute.py`, which applies a vendor's own published counting rules to a counts-only export from your helpdesk, then builds a Dispute and Recovery Pack with both directions of the variance — billed-but-not-billable and billable-but-not-billed — plus a cannot-settle line for rows the evidence cannot decide.

If you build agent tooling, the problem is upstream of billing: your metrics need to mean what their labels claim. The conformance harness encodes semantics like "a retry is one logical operation, not two requests" and "a cache hit is not a new measurement" as invariant vectors, so a claim such as "operation success rate" can be checked rather than argued about. The embedded pack runs inside CI through the repository's GitHub Action (`action.yml`), turning telemetry assumptions into build-breaking checks.

Finally, if you share measurement results at all, the privacy posture matters. The healthcheck package contains no network code — a test enforces this by scanning every module for network imports — and the share flow is preview-then-export: a sanitized summary built from a fixed whitelist of aggregate counts, re-verified against the whitelist before anything is written, with a planted-string selftest confirming that prompts, paths, commands, and session ids cannot leak into it.

## How It Works

The repository separates probes, checks, and publication so cleanly that each layer can be read on its own; the detailed diagram below maps the modules we walked.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/agentmeasure/roy-tong-agentmeasure-architecture.svg" alt="Detailed architecture of the roy-tong/AgentMeasure repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of roy-tong/AgentMeasure: the am_healthcheck probe stack on the left, the conformance harness and its vector corpus in the middle, the reference collector and canonical schema on the standards side, and the published-findings artifacts at the right.*

### Understanding the Architecture

**The discovery and parsing probes.** Entry is the console script defined in `healthcheck/pyproject.toml` (`am_healthcheck.cli:main`), with a stable shim at `healthcheck/agentmeasure`. `discover.py` locates Codex rollout files by name pattern under `~/.codex/sessions/YYYY/MM/DD/` and Claude Code session files under `~/.claude/projects/`, applying date windows honestly — Claude files are named by session UUID rather than date, so the filter uses file mtime and says so. The two adapters, `codex.py` and `claude.py`, parse raw JSONL into the `SessionRecord` model defined in `model.py`, reading only envelope types, ids, exit codes, durations, and token counters — prompt and message content is never read into the analysis.

**The deterministic check families.** `checks.py` implements the three base checks — HC-01 duplicate records (byte-identical lines, repeated call/execution ids, and the rollout format's by-design dual-stream recording), HC-02 retry amplification (consecutive same-command failures that carry one logical operation), and HC-03 tool error runs (three or more back-to-back same-tool failures) — plus the audit-mode trio HC-04 operation-resolution coverage, HC-05 cache-accounting cross-check, and HC-06 token-accounting stability. Deduplication across split or resumed files happens in `_canonical_execs`, where a repeated execution id with an identical envelope collapses to one execution and a conflicting one is demoted to UNKNOWN rather than allowed to manufacture a success. Token totals use the last cumulative snapshot per session instead of summing per-event deltas, and any malformed snapshot fails the whole session's total closed.

**The conformance harness and what it actually runs.** `healthcheck/am_healthcheck/pack.py` validates a caller-supplied telemetry fixture line by line against the FMT-002 funnel-event schema (`healthcheck/am_healthcheck/lab/schemas/funnel-event.schema.json`), aggregates it through the lab analysis layer, and evaluates invariants loaded from `pack_invariants.json`. In data-only mode it checks internal consistency and computes reference values; with `--claims` it additionally compares observed claims against the computation and fails on mismatch; with `--require` a named invariant that ends up UNPROVABLE blocks the run. The same discipline exists repository-side as language-neutral vectors under `conformance/vectors/`, replayed by the runners `run_metrics.py` (selection-rate, execution-grain, and consumption vectors), `run_outcome_audit.py` (OUT-001 through OUT-005), `run_delegation.py` (DELEGATION-001 through 007), and `run_external_fixture.py`, which guards third-party fixtures under `conformance/vectors/external/` — schema validity, mutation rejection, and per-metric reproduction of the expected results. `verify_vectors.py` at the repository root covers receipt and correlation vectors against the reference collector, and its fixtures pin an explicit multi-year analysis window precisely so the vectors cannot age out with the wall clock.

**The reference collector and the provider SDK.** The standards side defines what an observation is: `reference/collector/ingest.py` is the single ingestion boundary that validates every canonical observation envelope against `schemas/observation.schema.json` and the per-type payload schemas under `schemas/payloads/`, including a caller-type versus identity-strength consistency rule, before flattening records for `normalizer/` and `aggregator/`. The TypeScript SDK at `sdk/src/index.ts` (`@agentmeasure/mcp`) emits exactly those envelopes from the provider side: emit is fail-open and non-blocking, buffering into a memory queue flushed to rotating spool files with explicit loss accounting, and content fields are unreachable by design.

**From findings to verified publication.** The publication path is deliberately mundane. The recount command in `vendors.py` reads vendor counting rules from `vendor-rules.json` and refuses to produce a number until `--inspect` has shown you the column mapping, so a wrong mapping cannot masquerade as a finding; `dispute.py` turns a recount plus optional effect-confirmed records into a negotiable pack; `settle.py` generates an AMS-1 settlement bundle and one-pager from effect-confirmed JSONL. Ecosystem findings themselves live as evidence cases under `conformance/evidence/` with pinned commits and explicit claim boundaries, as per-claim cards under `benchmark/claims/` carrying multi-axis evidence profiles, and as the audit report and measurement casebook under `campaigns/`.

End to end: logs on disk are discovered and parsed into a canonical session model, deterministic checks and conformance invariants produce verdicts that name their evidence, exports and snapshots carry versioned schemas so third parties can integrate without importing the package, and anything public-facing — a dispute pack, a settlement statement, an audit filing — traces back to a pinned fixture or a named rule with tests behind it.

## Advantages

- **Zero-dependency local probes.** The healthcheck package is Python 3.9+ standard library only, with no runtime dependencies and no network code — a test enforces both — so the probe runs anywhere your agent logs already exist.
- **UNPROVABLE as a first-class verdict.** When evidence is absent, every layer reports UNPROVABLE with a reason instead of a zero, which keeps silent undercounting out of the numbers.
- **Language-neutral conformance vectors.** The JSON fixtures under `conformance/vectors/` are contracts any Go, Rust, or TypeScript implementation can be validated against, not just the bundled Python runners.
- **Fail-closed by construction.** Corrupt lines, truncated files, conflicting duplicate ids, and malformed token snapshots all lower counts or void totals rather than producing plausible-looking garbage.
- **Versioned export schemas.** Reports, snapshots, and comparisons are JSON documents with versions (`report-v1`, `snapshot-v1`, `compare-v1`) validated on load, making the exports — not internal modules — the integration surface.
- **CI-ready conformance.** The composite GitHub Action in `action.yml` installs the PyPI package and runs the pack against your fixture, publishing the PASS / FAIL / UNPROVABLE report into the job summary.

## Benefits

- **A check-up for agent runs you already have.** Duplicate records, retry chains, and repeated tool failures are surfaced from existing Codex and Claude Code logs with file, line, and exit-code evidence and a concrete next step per finding.
- **A buyer-side recount of outcome bills.** Applying a vendor's own published rules to your export produces a delta both directions plus a cannot-settle line — the kind of artifact that keeps a renewal negotiation factual.
- **Honest trend tracking.** Snapshots and `compare` give a before/after preview for prompt or dependency changes, differencing token totals only when both sides are provable and disclosing window or version mismatches.
- **Safe sharing by default.** The preview-then-export share flow and whitelist sanitizer let you publish aggregate results without leaking prompts, paths, or session identifiers.
- **Cross-side corroboration.** Pairing provider-side SDK observations with runtime-side log analysis (`--sdk-events`) lets two independent surfaces check the same usage story.
- **Preregistered experimentation.** The Lab engine under `lab/` hashes hypothesis, primary metric, and guardrails before a run and replays seeds deterministically, so uplift claims meet the same evidence discipline as usage counts.

## Usage

The tool ships on PyPI, so the fastest path needs no checkout:

```bash
pipx run agentmeasure demo                 # synthetic example; no personal logs needed
pipx run agentmeasure check                # your local sessions, last 7 days
pipx run agentmeasure check --runtime claude   # force the Claude Code adapter
```

To install the command or run straight from a checkout:

```bash
pipx install "git+https://github.com/roy-tong/AgentMeasure#subdirectory=healthcheck"
python3 healthcheck/agentmeasure demo
```

A typical report-then-share workflow, using the preview-then-export flow:

```bash
agentmeasure check --json run.json          # the report export embeds the summary
agentmeasure share run.json                 # terminal PREVIEW — nothing is written
agentmeasure share run.json --out summary.md   # export after review
agentmeasure check --save-snapshot before.json
agentmeasure compare before.json after.json
```

The conformance pack runs on any telemetry fixture, locally or in CI:

```bash
agentmeasure conformance --fixture fixtures/telemetry.jsonl
```

```yaml
- uses: roy-tong/AgentMeasure@2cf476d6f7d0fc45401db5a822e1f12de009ac74
  with:
    fixture: fixtures/telemetry.jsonl
```

From a repository checkout, the full vector suite verifies the reference implementation:

```bash
python3 conformance/runners/run_metrics.py   # metric vectors
python3 verify_vectors.py                    # verification / correlation vectors
python3 registry/validate_entities.py        # validate the machine-readable registry
```

## Conclusion

AgentMeasure is unusual in that its artifact is a measurement standard and its code is the reference implementation of that standard, each holding the other honest. The probe stack shows how to structure measurement software that refuses to guess: discovery, adapters, and checks stay small and deterministic, verdicts name their evidence, and every export carries a version. The harness shows what a serious audit actually runs — schema-validated fixtures, invariant packs, guarded external vectors, and receipt-level correlation checks — rather than a pile of greps. And the publication path shows how findings earn trust: pinned fixtures, named rules, claim boundaries, and settlement statements that disclose what cannot be proven alongside what can. If your agents generate logs and someone, somewhere, turns those logs into a bill or a dashboard, reading this source will change how you ask "what did you count?"

Links:

- GitHub repository: [roy-tong/AgentMeasure](https://github.com/roy-tong/AgentMeasure)
- Project website: [roy-tong.github.io/AgentMeasure](https://roy-tong.github.io/AgentMeasure/)
- Core specification: [standard/CORE.md](https://github.com/roy-tong/AgentMeasure/blob/main/standard/CORE.md)
- Healthcheck quick start: [healthcheck/README.md](https://github.com/roy-tong/AgentMeasure/blob/main/healthcheck/README.md)
- Token-Accounting Bug Report: [campaigns/audit-report-2026-09.md](https://github.com/roy-tong/AgentMeasure/blob/main/campaigns/audit-report-2026-09.md)
