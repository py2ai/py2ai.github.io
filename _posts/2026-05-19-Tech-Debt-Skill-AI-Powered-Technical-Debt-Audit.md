---
layout: post
title: "Tech-Debt-Skill: AI-Powered Technical Debt Audit for Your Codebase"
date: 2026-05-19 00:00:00 +0800
categories: [ai, developer-tools, code-quality]
tags: [tech-debt, code-quality, claude-code, skill, audit, refactoring, ai]
seo:
  title: "Tech-Debt-Skill - AI-Powered Technical Debt Audit | PyShine"
  description: "Tech-Debt-Skill is a Claude Code skill that audits your codebase for technical debt, providing actionable insights and prioritized remediation plans."
  keywords: "tech debt, code quality, claude code skill, audit, refactoring, ai, technical debt"
featured-img: ai-coding-frameworks/ai-coding-frameworks
permalink: /Tech-Debt-Skill-AI-Powered-Technical-Debt-Audit/
---

Every codebase accumulates technical debt. The question isn't whether you have it — it's whether you know where it hides, how bad it really is, and what to fix first. Most "code review" tools produce generic best-practice checklists that feel comprehensive but lead to zero action. **Tech-Debt-Skill** takes a fundamentally different approach: it forces Claude Code to understand your codebase *before* judging it, cite specific file locations on every finding, and explicitly surface things that look bad but are actually fine.

> A finding without a citation is a vibe. Vibes don't get fixed.

## What is Tech-Debt-Skill?

Tech-Debt-Skill is a Claude Code skill that produces a thorough, citable tech debt audit of your entire codebase. Run a single command — `/tech-debt-audit` — and get back a `TECH_DEBT_AUDIT.md` with file-cited findings, severity ratings, effort estimates, and a ranked list of what to actually fix. It's not a generic best-practices checklist. It's an opinionated audit grounded in your actual code.

![Architecture](/assets/img/diagrams/tech-debt-skill/tech-debt-skill-architecture.svg)

## Key Features

- **Forced orientation before judgment** — The protocol requires reading the manifest, mapping directory structure, analyzing git churn, and building a mental model *before* forming any opinions. Phase 1 is not optional.
- **file:line citations on every finding** — Every concrete finding includes `path/to/file.ext:LINE`. Vague claims like "the code generally..." are rejected outright.
- **9 audit dimensions** — Architectural decay, consistency rot, type and contract debt, test debt, dependency and config debt, performance and resource hygiene, error handling and observability, security hygiene, and documentation drift.
- **Multi-stack tooling** — Automatically detects your stack and runs the right tools: `npm audit`, `knip`, `madge` for TS/JS; `pip-audit`, `ruff`, `vulture` for Python; `cargo audit`, `clippy` for Rust; `govulncheck`, `staticcheck` for Go.
- **Subagent dispatch for large repos** — For codebases over 50k LOC, the protocol parallelizes across modules so the main agent doesn't run out of context.
- **Repeat-run mode** — On subsequent runs, resolved findings are marked `RESOLVED`, stale ones are updated, and new findings are tagged `NEW`. The audit becomes a living document.
- **"Looks bad but is actually fine" section** — Forces the model to surface calls it considered making and chose not to. If this section is empty, the audit didn't look hard enough.
- **Persistent, committable artifact** — `TECH_DEBT_AUDIT.md` lives in your repo. You can commit it, review it in PRs, and link to specific findings.

![Features](/assets/img/diagrams/tech-debt-skill/tech-debt-skill-features.svg)

## How It Works

The audit follows a strict three-phase protocol designed to prevent the most common LLM audit failure modes:

### Phase 1: Orient

The model reads your README, package manifest, architecture docs, and `git log` for churn data. It identifies the top 20 largest files and the 20 most frequently modified files — their intersection is where debt usually hides. A mental model of the architecture is written *before* any findings are formed.

> Findings without context are vibes. Phase 1 isn't optional decoration.

### Phase 2: Audit

The model sweeps across nine dimensions using `rg`, `ast-grep`, and language-native tooling. Every finding must cite `file:line`. Stack-specific tools are detected and run in parallel when possible.

### Phase 3: Deliverable

The model writes `TECH_DEBT_AUDIT.md` with an executive summary, architectural mental model, findings table (ID, Category, File:Line, Severity, Effort, Description, Recommendation), top 5 priorities, quick wins checklist, the required "looks bad but is actually fine" section, and open questions for the maintainer.

## Getting Started

Install the skill globally (available across all your projects):

```bash
mkdir -p ~/.claude/skills/tech-debt-audit
```

```bash
curl -o ~/.claude/skills/tech-debt-audit/SKILL.md https://raw.githubusercontent.com/ksimback/tech-debt-skill/main/SKILL.md
```

Or install it project-only (just for this repo):

```bash
mkdir -p .claude/skills/tech-debt-audit && curl -o .claude/skills/tech-debt-audit/SKILL.md https://raw.githubusercontent.com/ksimback/tech-debt-skill/main/SKILL.md
```

Verify it's loaded:

```bash
claude --print "/skills" | grep tech-debt-audit
```

Run the audit in any repo:

```
/tech-debt-audit
```

Audit a specific subtree (useful for large monorepos):

```
/tech-debt-audit src/payments
```

## Why Tech-Debt-Skill Matters

Claude Code ships built-in skills like `/review`, `/simplify`, and `/debug`, but none of them do what a debt audit needs. `/review` is diff-scoped — useful before merging a branch, not when you've inherited 80k LOC and want to know what's rotten. `/simplify` is tactical, not architectural. `/debug` is reactive — you point it at a known problem, while an audit's job is to *find* the problems.

Tech-Debt-Skill fills this gap with three design choices that separate a real audit from checklist regurgitation:

1. **Forced orientation** prevents pattern-matching against generic heuristics without grounding in the actual code.
2. **Mandatory citations** make every finding falsifiable — you can verify it, dispute it, or act on it.
3. **The "looks bad but is actually fine" section** catches shallow analysis. Forcing the model to explain why it *didn't* flag something is what separates depth from surface-level scanning.

> If the "looks bad but is actually fine" section comes back empty, the audit didn't look hard enough.

The skill also explicitly forbids recommending rewrites and forbids padding categories with filler — both common LLM failure modes where rewriting is easier than diagnosing, and padding makes outputs feel thorough when they aren't.

## Customization

The skill is designed to be forked and adapted to your needs:

- **Add domain-specific dimensions** — Frontend repos can add accessibility; ML repos can add eval drift; LLM apps can add prompt versioning and tool-call cost.
- **Tune severity thresholds** — Adjust thresholds like "god files >500 LOC" to match your codebase baseline.
- **Per-project overrides** — A `.claude/skills/tech-debt-audit/SKILL.md` in a specific repo overrides the global one.
- **Split into supporting files** — Extract sections into sibling files that Claude Code lazy-loads on demand.

## Conclusion

Tech-Debt-Skill represents a shift in how AI assists with code quality — from generic checklist generation to grounded, citable, actionable audits. By forcing orientation before judgment, requiring file:line citations, and mandating a "looks bad but is actually fine" section, it produces findings that engineers actually act on. The persistent `TECH_DEBT_AUDIT.md` artifact and repeat-run mode turn it from a one-time scan into a living document you can track over time. If you've inherited a codebase and need to know what's really rotten underneath, this is the tool to reach for.

Check out the [GitHub repository](https://github.com/ksimback/tech-debt-skill) to get started.