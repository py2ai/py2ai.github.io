---
layout: post
title: "AI Auto-Work: Dual-Model Agentic Coding Workflow with Claude and Codex Adversarial Review"
description: "Learn how AI Auto-Work orchestrates Claude Code and Codex in a dual-model adversarial workflow for autonomous full-cycle software development, from requirement research to atomic commits with mechanical quality gates."
date: 2026-04-30
header-img: "img/post-bg.jpg"
permalink: /AI-Auto-Work-Dual-Model-Agentic-Coding-Workflow/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, Software Engineering]
tags: [AI Auto-Work, agentic coding, Claude Code, Codex, dual-model review, adversarial review, autonomous development, quality gates, atomic commits, context repair]
keywords: "AI Auto-Work agentic coding workflow, how to use Claude Code and Codex together, dual-model adversarial code review, autonomous software development pipeline, AI coding quality gates, Claude Code Codex collaboration, agentic workflow automation, AI Auto-Work vs GitHub Copilot, automated code review with AI, context repair in AI coding agents"
author: "PyShine"
---

# AI Auto-Work: Dual-Model Agentic Coding Workflow with Claude and Codex Adversarial Review

AI Auto-Work is an agentic coding workflow system that orchestrates large language models to complete real-world software engineering tasks end-to-end. Using Claude as the executor and Codex as the adversarial reviewer, it forms a Triple-Check convergence loop that delivers production-quality code with minimal human intervention. The system automatically handles the full pipeline from requirement research through code commit, capable of autonomously managing medium to large-scale engineering projects.

## The Core Idea: Why Dual-Model Review Matters

Single-model self-review misses systematic issues due to shared cognitive biases. When Claude reviews its own code, it tends to overlook the same classes of problems it introduced. AI Auto-Work solves this by using two models with different training distributions: Claude builds the code, and Codex audits it independently.

| Issue Type | Claude (Executor) | Codex (Reviewer) |
|------------|-------------------|------------------|
| Concurrency races | May miss | Flags |
| Goroutine leaks | May miss | Flags |
| Resource exhaustion | May miss | Flags |
| Boundary conditions | May miss | Flags |
| Constitution compliance | Self-blind | Independent check |

What Claude overlooks, Codex tends to catch. This adversarial relationship produces higher-quality code than either model could achieve alone.

## The Full Pipeline

![Full Pipeline](/assets/img/diagrams/ai-auto-work/ai-auto-work-pipeline.svg)

### Understanding the Full Pipeline

The pipeline implements a complete software development lifecycle with mechanical quality gates at every stage. No code reaches the commit stage without passing both automated verification and adversarial review.

**Stage 1: Requirement Classification**

Every incoming requirement is classified by complexity: S (small), M (medium), or L (large). L-level tasks are automatically split into multiple M-level tasks that can be processed sequentially. This decomposition ensures that each task is small enough to be completed in a single context window without accumulation.

**Stage 2: Technical Research**

When the requirement involves technology selection or unfamiliar domains, the system launches a research loop. Three parallel agent shards perform web searches while another scans the project codebase. The result is a structured `research-result.md` covering problem definition, industry solutions, comparison tables, practical experience, project fit analysis, and a recommendation.

**Stage 3: Plan Generation**

The system iteratively generates a `plan.md` until quality converges. Odd rounds generate or fix the plan; even rounds submit it to Codex for review. Convergence is achieved when Critical issues equal zero and Important issues are at most two (maximum 20 rounds). The output includes data model design, API specs, implementation flow, test strategy, and risk assessment.

**Stage 4: Task Decomposition**

The plan is decomposed into atomic tasks, each captured in a `tasks/task-N.md` file. Each task is constrained to affect at most 3 files and 100 lines of code. This atomicity ensures that every commit is bisectable and every change is independently reviewable.

**Stage 5: Development Loop**

Each task goes through its own coding and review cycle: code implementation, compile check (fast-fail gate), Codex review (max 2 rounds), auto-fix failures (max 2 rounds), and a final gate of unit + integration + smoke tests.

**Stage 6: Quality Gate**

Before any review cycle begins, the code must pass mechanical quality gates: compilation succeeds and all tests pass. This prevents wasting review cycles on code that doesn't even compile.

**Stage 7: Adversarial Review**

Codex performs an independent review of the code, looking for issues that Claude's self-review missed. The convergence criterion is strict: Critical issues must equal zero and High issues must be at most two.

**Stage 8: Commit**

Only after convergence does the system create an atomic git commit with a standardized format: `<type>(<scope>): <description>`.

## Dual-Model Collaboration in Detail

![Dual-Model Collaboration](/assets/img/diagrams/ai-auto-work/ai-auto-work-dual-model.svg)

### Understanding the Dual-Model Architecture

The dual-model architecture is the key innovation of AI Auto-Work. Rather than relying on a single model to both write and review code, the system uses Claude and Codex in complementary roles.

**Claude as Executor**

Claude handles all code generation: writing implementations, fixing bugs, and creating tests. It operates within the constraints defined by the task decomposition, producing atomic changes that affect at most 3 files and 100 lines.

**Quality Gate: The Mechanical Filter**

Before any code reaches Codex for review, it must pass the quality gate: compilation succeeds and all automated tests pass. This mechanical filter catches syntax errors, type mismatches, and obvious logic failures without wasting the reviewer's capacity on trivial issues.

**Codex as Adversarial Reviewer**

Codex reviews the code independently, looking for classes of issues that Claude tends to miss: concurrency races, resource leaks, boundary conditions, and constitution compliance. Because Codex has a different training distribution, it surfaces blind spots that Claude's self-review would miss.

**Convergence Loop**

If Codex finds issues, Claude fixes them and the cycle repeats. The loop converges when Critical issues equal zero and High issues are at most two. If convergence is not achieved after maximum iterations, the system auto-escalates the complexity classification.

**Execution Modes**

The system supports two execution modes:

| Mode | Condition | Method | Review Quality |
|------|-----------|--------|----------------|
| ROUTE_MODE_A | Claude CLI + Codex CLI both available | Bash subprocess orchestration | Highest (true cross-model) |
| ROUTE_MODE_B | Agent tool only | Agent delegation | Good (proxy review) |

ROUTE_MODE_A provides the highest quality because it uses truly independent models. ROUTE_MODE_B is available when only Claude is installed, using agent delegation as a fallback.

## Three Workflow Modes

![Workflow Modes](/assets/img/diagrams/ai-auto-work/ai-auto-work-modes.svg)

### Understanding the Workflow Modes

AI Auto-Work provides three workflow modes optimized for different levels of complexity and human oversight.

**/fast-auto-work: Small Direct Changes**

Optimized for bug fixes, narrow extensions, and low-risk edits on top of existing patterns. It intentionally skips the research loop, formal planning artifacts, and acceptance reports. However, it preserves the compile gate, relevant test gate, and automatic escalation when changes exceed fast-path scope.

Use it when:
- The change is expected to stay within 3 or fewer implementation files
- The module already has an established implementation pattern
- You want code + build/test verification quickly

```bash
/fast-auto-work v0.2.0 fix-login-button align login button loading state
```

**/auto-work: Fully Automated Cycle**

Handles the complete cycle from raw requirement to committed code with no human intervention. All eight stages execute automatically: classification, research, planning, decomposition, development, acceptance, documentation, and commit.

```bash
/auto-work implement user avatar upload feature
```

**/manual-work: Human Checkpoints**

Same output as auto-work, but pauses at key milestones for human confirmation. The most critical checkpoint is Stage 2 (plan/architecture review), where the human can verify the approach before any code is written. Type `AUTOPILOT=true` at any checkpoint to switch to fully automated mode.

```bash
/manual-work refactor user authentication module
```

**Automatic Escalation**

The fast-auto-work mode includes automatic escalation: if it detects that the change exceeds its scope (touching routing, contracts, architecture files, or multiple top-level domains), it writes an escalation file and points back to `/auto-work` or `/manual-work` instead of guessing.

## Context Repair: Self-Improving Knowledge Base

![Context Repair](/assets/img/diagrams/ai-auto-work/ai-auto-work-context-repair.svg)

### Understanding the Context Repair Loop

Context repair is one of the most powerful features of AI Auto-Work. When Codex identifies a recurring class of issue (not a one-off mistake), the workflow writes the constraint into the `.ai/` shared knowledge base. This knowledge is loaded at the start of every future run, ensuring that the same class of error never recurs.

**How It Works**

1. During a workflow run, Codex reviews the code and identifies issues
2. If an issue is a one-off mistake, Claude fixes the code directly
3. If the issue represents a recurring class of error, the constraint is written to `.ai/constitution.md` or `.ai/constitution/` (module-specific rules)
4. On the next run, the `.ai/` knowledge base is loaded at session start
5. The same class of error is prevented by the updated constitution

**The .ai/ Directory Structure**

| Directory | Purpose |
|-----------|---------|
| `.ai/constitution.md` | Core engineering principles |
| `.ai/constitution/` | Module-specific rules (concurrency, testing, cross-module) |
| `.ai/context/project.md` | Project tech stack and constraints |

This is fundamentally different from simply fixing bugs. Context repair fixes the *process* that produced the bug, not just the bug itself. Over time, the `.ai/` knowledge base accumulates project-specific constraints that make the workflow increasingly accurate.

**Zero Context Pollution**

Each stage runs in an isolated process with a fresh context window. Handoff between stages happens exclusively through persisted files (`feature.md`, `plan.md`, `task-N.md`). This means context accumulation is never a limitation, even for large codebases. L-level tasks auto-split into sequential M-level tasks to handle arbitrary scope.

## Core Principles

| Principle | Rule |
|-----------|------|
| **Simplicity First** | Implement only what requirements explicitly ask |
| **Test-Driven** | New features and bug fixes start from failing tests |
| **Atomic Commits** | One task = one commit (at most 3 files / at most 100 lines) |
| **Mechanical Gates** | Compile + tests must pass before review |
| **Document-Driven Handoff** | `feature.md` then `plan.md` then `task-N.md` only |
| **Context Repair** | Systematic errors update `.ai/`, not just the code |

## Additional Workflow Commands

| Command | Purpose |
|---------|---------|
| `/feature:plan` | Iteratively generate `plan.md` until quality converges (max 20 rounds) |
| `/feature:develop` | Implement tasks sequentially with per-task coding + review loops |
| `/research:do` | Parallel web search (3 agent shards) + codebase scan for technology selection |
| `/bug:fix` | Root cause analysis, minimal fix, Codex review, experience consolidation |
| `/git:commit` | Standardized commit format, auto-skip sensitive files |
| `/git:push` | Blocks force-push to main, security scan before push |

## Usage Tips

- Prefer `/fast-auto-work` for direct small edits, but switch to `/auto-work` as soon as the change touches routing, contracts, architecture files, or multiple top-level domains
- Keep the workspace clean before running `/fast-auto-work`; it aborts by default on a dirty worktree
- Put stable requirement context in `Docs/Version/{version_id}/{feature_name}/idea.md`, then add only the delta in the command arguments
- Let the fast path fail closed: if it detects out-of-scope changes, it writes an escalation file and points back to `/auto-work` instead of guessing
- Use `FAST_AUTO_WORK_COMMIT=1` only when you explicitly want a local commit after gates pass

## Who Should Use AI Auto-Work

AI Auto-Work is designed for developers and teams who want to automate the full software development lifecycle:

- **Solo developers** using Claude Code who want autonomous end-to-end feature development with quality assurance
- **Teams** that need consistent code quality through mechanical gates and adversarial review
- **Projects with recurring error patterns** that benefit from context repair and self-improving knowledge bases
- **Anyone building medium to large features** where the full pipeline (research, planning, decomposition, development, review) adds value over simple code generation

## Conclusion

AI Auto-Work represents a significant step beyond simple AI code generation. By orchestrating Claude and Codex in a dual-model adversarial workflow with mechanical quality gates, atomic commits, and self-improving context repair, it delivers production-quality code autonomously. The three workflow modes (fast-auto-work, auto-work, manual-work) provide the right level of automation for every situation, from quick bug fixes to large feature development. The context repair mechanism ensures that the system gets better over time, accumulating project-specific knowledge that prevents recurring errors.

**Links:**

- GitHub Repository: [https://github.com/chaohong-ai/ai-auto-work](https://github.com/chaohong-ai/ai-auto-work)