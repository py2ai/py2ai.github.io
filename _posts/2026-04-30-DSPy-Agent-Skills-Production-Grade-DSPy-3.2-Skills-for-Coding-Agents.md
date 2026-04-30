---
layout: post
title: "DSPy Agent Skills: Production-Grade DSPy 3.2 Skills for Coding Agents"
description: "Learn how DSPy Agent Skills provides five spec-compliant skills that turn Claude Code and Codex CLI into DSPy experts, with progressive disclosure, runnable examples, and GEPA optimization pipelines."
date: 2026-04-30
header-img: "img/post-bg.jpg"
permalink: /DSPy-Agent-Skills-Production-Grade-DSPy-3.2-Skills-for-Coding-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Open Source, Developer Tools]
tags: [DSPy, agent skills, Claude Code, Codex CLI, GEPA optimizer, prompt optimization, LLM programming, DSPy 3.2, coding agents, open source]
keywords: "how to use DSPy agent skills, DSPy agent skills tutorial, DSPy 3.2 skills for Claude Code, GEPA optimizer DSPy, DSPy coding agent integration, Claude Code DSPy skills, prompt optimization DSPy, DSPy fundamentals module, DSPy evaluation harness, DSPy agent skills installation"
author: "PyShine"
---

# DSPy Agent Skills: Production-Grade DSPy 3.2 Skills for Coding Agents

DSPy Agent Skills is a synthesized, spec-compliant pack of five agent skills that turns Claude Code, Codex CLI, and any agentskills.io-compatible agent into a DSPy expert. Validated against DSPy 3.2.0 (the real API, not inferred from stale docs), it provides progressive disclosure from short references to deep documentation, runnable example scripts with offline dry-run mode, and committed baseline vs. GEPA-optimized performance numbers.

![Five Skills Overview](/assets/img/diagrams/dspy-agent-skills/dspy-agent-skills-overview.svg)

### Understanding the Five-Skill Pack

The overview diagram shows the five skills and how they relate to each other:

**dspy-fundamentals**
The foundation skill covers Signatures, Modules, Predict/ChainOfThought/ReAct, and save/load. It auto-invokes whenever you write any new DSPy code, providing the core API signatures and patterns you need to get started.

**dspy-evaluation-harness**
The evaluation skill covers writing metrics, splitting dev/val sets, and calling `dspy.Evaluate`. It auto-invokes when you need to measure DSPy program performance, providing the evaluation infrastructure that optimization depends on.

**dspy-gepa-optimizer**
The optimization skill covers optimizing and compiling DSPy programs with `dspy.GEPA`. It auto-invokes when you want to improve your DSPy program's performance, providing the GEPA optimizer that bootstraps few-shot examples and compiles optimized programs.

**dspy-rlm-module**
The long-context skill covers codebase QA, recursive exploration, and long-context reasoning via `dspy.RLM`. It auto-invokes when you need to work with large codebases or documents that exceed standard context windows.

**dspy-advanced-workflow**
The orchestration skill ties everything together. It auto-invokes for end-to-end builds, chaining the other four skills to produce a complete baseline-to-optimized pipeline. When you say "Build a DSPy sentiment classifier, optimize it with GEPA, and save the artifact," this skill coordinates the entire workflow.

## Auto-Invocation Flow

![Invocation Flow](/assets/img/diagrams/dspy-agent-skills/dspy-agent-skills-invocation.svg)

### Understanding the Auto-Invocation Flow

The invocation flow diagram shows how DSPy Agent Skills integrates with coding agents:

**User Prompt**
You describe what you want in natural language: "Build a DSPy sentiment classifier, optimize it with GEPA, and save the artifact." No special syntax or commands required.

**Coding Agent (Claude Code / Codex CLI)**
The coding agent receives your prompt and checks its loaded skills for matching triggers. Each skill defines when it should auto-invoke based on the content of your request.

**Skill Loader**
The skill loader matches your prompt against the trigger patterns defined in each `SKILL.md`. For the sentiment classifier example, it matches `dspy-advanced-workflow` (which orchestrates the other four skills).

**Skills Loaded**
Each skill provides three layers of information: the short `SKILL.md` (triggers, API signatures, quick-start examples), the deep `reference.md` (full API details, edge cases, version-specific notes), and runnable `example_*.py` scripts with offline `--dry-run` mode.

**Generated Code**
The agent uses the loaded skill knowledge to generate a complete pipeline: baseline definition, evaluation, GEPA optimization, and artifact export. No further prompting needed.

## Progressive Disclosure

![Progressive Disclosure](/assets/img/diagrams/dspy-agent-skills/dspy-agent-skills-disclosure.svg)

### Understanding the Progressive Disclosure Architecture

The progressive disclosure diagram shows the three-layer documentation architecture that each skill provides:

**SKILL.md -- Short Reference**
The `SKILL.md` file is the entry point. It contains triggers (when the skill auto-invokes), API signatures (the function and class names you need), and quick-start examples. This is what the agent reads first to understand the skill's scope and capabilities.

**reference.md -- Deep Documentation**
When the agent needs more detail, it reads the `reference.md` file. This contains full API documentation, edge cases, version-specific notes (especially important for DSPy 3.1.3 vs 3.2.0 differences), and detailed usage patterns. This layer prevents the agent from hallucinating API details that changed between versions.

**example_*.py -- Runnable Scripts**
Each skill includes runnable example scripts that can be executed in `--dry-run` mode (no API key needed) or with real LLM calls. These scripts serve as both documentation and validation: they demonstrate the correct API usage and can be run to verify that the generated code actually works.

**80 Validation Tests**
All three layers are validated by 80 tests that check frontmatter spec compliance, JSON schema correctness, Python AST validity, and skill-document correctness guards. This ensures that the skills remain accurate as DSPy evolves.

## BetterTogether Pipeline

![Pipeline](/assets/img/diagrams/dspy-agent-skills/dspy-agent-skills-pipeline.svg)

### Understanding the BetterTogether Pipeline

The pipeline diagram shows the four-stage workflow that `dspy-advanced-workflow` orchestrates:

**1. Baseline -- Define Signature + Module**
You start by defining a DSPy Signature (input/output specification) and wrapping it in a Module (Predict, ChainOfThought, or ReAct). This is your unoptimized baseline -- the raw LLM performance before any prompt engineering or few-shot optimization.

**2. Evaluate -- Define Metric + Split Data**
Next, you define a metric function that measures your program's performance, split your data into dev/val sets, and run `dspy.Evaluate` to establish baseline scores. This step is critical: you cannot improve what you cannot measure.

**3. Optimize -- GEPA Optimizer + Compile**
The GEPA optimizer takes your baseline program, metric, and training data, then compiles an optimized version. GEPA bootstraps few-shot examples, searches over prompt variations, and produces a program that consistently outperforms the baseline.

**4. Export -- Save Artifact + Reuse**
The optimized program is saved as an artifact that can be loaded with `.load()` for reuse in production. This creates a reproducible, version-controlled optimization result.

**Verified Results**
The pipeline includes committed baseline vs. optimized numbers from three real examples:
- RAG QA: 75.77 -> 100.00 (+24.23 points)
- Math Reasoning: 85.00 -> 93.33 (+8.33 points)
- Invoice Extraction: 0.833 -> 0.931 (+0.098 F1)

## Installation

### Claude Code (via marketplace)

```text
/plugin marketplace add intertwine/dspy-agent-skills
/plugin install dspy-agent-skills@dspy-agent-skills
```

### Agent Skills CLI (npx skills)

```bash
npx skills add intertwine/dspy-agent-skills --list
npx skills add intertwine/dspy-agent-skills --skill '*' -a codex -y
```

### Repo Checkout (both Claude Code and Codex CLI)

```bash
git clone https://github.com/intertwine/dspy-agent-skills
cd dspy-agent-skills
./scripts/install.sh           # symlinks into ~/.claude/skills/ and ~/.agents/skills/
```

Flags: `--claude-only`, `--codex-only`, `--copy` (copy instead of symlink), `--uninstall`, `--dry-run`.

### Manual

Drop `skills/*` into `~/.claude/skills/` (Claude Code) or `~/.agents/skills/` (Codex CLI).

## Key Features

| Feature | Description |
|---------|-------------|
| Five Spec-Compliant Skills | Fundamentals, Evaluation, GEPA Optimizer, RLM Module, Advanced Workflow |
| DSPy 3.2.0 Validated | Tested against the real API, not inferred from stale docs |
| Progressive Disclosure | Short SKILL.md + deep reference.md + runnable example scripts |
| Offline Dry-Run | All examples run with --dry-run (no API key needed) |
| BetterTogether Pipeline | End-to-end baseline -> evaluate -> optimize -> export workflow |
| 80 Validation Tests | Frontmatter spec, JSON schema, Python AST, skill-doc guards |
| Dual-Agent Support | Single source of truth for both Claude Code and Codex CLI |
| Plugin Manifest | Marketplace manifest for one-click install |
| Committed Results | Baseline vs. GEPA-optimized numbers from real LLM runs |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Skills not auto-invoking | Verify skills are installed in `~/.claude/skills/` or `~/.agents/skills/` |
| DSPy version mismatch | Use `env -u UV_EXCLUDE_NEWER uv run --with dspy==3.2.0` to force 3.2.0 |
| `--dry-run` fails | Ensure DSPy is installed: `pip install dspy==3.2.0` |
| GEPA optimization stalls | Check that your OPENAI_API_KEY is set and has sufficient quota |
| Validation tests fail | Run `uv run --with pytest python -m pytest tests/ -v` for details |
| Symlink issues on Windows | Use `--copy` flag instead of symlinks: `./scripts/install.sh --copy` |

## Conclusion

DSPy Agent Skills fills a critical gap in the coding agent ecosystem: giving agents deep, validated knowledge of a complex framework (DSPy 3.2.x) through a structured skill system. The five-skill pack covers the full DSPy lifecycle from fundamentals to advanced optimization, with progressive disclosure that lets agents start quickly and go deep when needed.

The most impressive aspect is the rigor: every API claim is validated against the real DSPy 3.2.0 API (not inferred from stale documentation), 80 validation tests ensure spec compliance, and committed baseline vs. optimized numbers provide ground-truth performance data. This is what production-grade agent skills look like -- not just prompts, but a complete, tested, version-aware knowledge system.

**Links:**

- GitHub: [https://github.com/intertwine/dspy-agent-skills](https://github.com/intertwine/dspy-agent-skills)
- DSPy Documentation: [https://dspy.ai/](https://dspy.ai/)