---
layout: post
title: "Superpowers: The Agentic Skills Framework That Transforms AI Agents Into Systematic Engineers"
description: "Discover how Superpowers by obra transforms AI coding agents from code generators into disciplined engineers following TDD, proper planning, and quality workflows with 14 composable skills."
date: 2026-04-09
header-img: "img/post-bg.jpg"
permalink: /Superpowers-Agentic-Skills-Framework/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - Software Development
  - Tools
  - Test-Driven Development
  - Agentic Workflows
author: "PyShine"
---

# Superpowers: The Agentic Skills Framework That Transforms AI Agents Into Systematic Engineers

## Introduction

Superpowers is a revolutionary software development workflow framework designed specifically for AI coding agents. Created by Jesse Vincent at Prime Radiant, this open-source project has garnered an impressive 140,708 stars on GitHub, making it one of the most popular tools in the AI-assisted development space. Unlike traditional coding assistants that jump straight into writing code, Superpowers transforms AI agents into disciplined engineers who follow systematic workflows, proper planning, and rigorous testing methodologies.

The framework addresses a critical gap in AI-assisted development: while large language models excel at generating code, they often lack the systematic approach that experienced software engineers bring to complex projects. Superpowers fills this gap by providing a comprehensive set of composable "skills" that guide AI agents through every stage of the software development lifecycle, from initial brainstorming to final code review and deployment.

What makes Superpowers particularly valuable is its ability to enforce best practices automatically. When an AI agent equipped with Superpowers encounters a task, it doesn't just start writing code. Instead, it steps back, asks clarifying questions, develops specifications, creates implementation plans, and follows test-driven development practices. This systematic approach dramatically improves code quality and reduces the common pitfalls associated with AI-generated code.

## The Problem Superpowers Solves

AI coding agents, despite their impressive capabilities, suffer from several fundamental limitations that can lead to poor software quality:

**Jumping to Code Without Planning:** Most AI agents immediately start generating code when given a task, bypassing crucial planning and design phases. This often results in solutions that don't fully address requirements or require significant rework.

**Skipping Tests and Quality Checks:** Without explicit guidance, AI agents frequently produce code without corresponding tests, leading to fragile implementations that break unexpectedly. The absence of test-driven development practices means bugs often go undetected until production.

**Inconsistent Quality and Approach:** Different sessions with the same AI agent can produce vastly different code quality, depending on how the prompt is phrased. This inconsistency makes it difficult to maintain standards across a development team.

**Lack of Systematic Methodology:** AI agents don't naturally follow established software engineering practices like YAGNI (You Aren't Gonna Need It), DRY (Don't Repeat Yourself), or incremental development. This can lead to over-engineered solutions and technical debt.

Superpowers addresses these issues by providing a structured workflow that forces AI agents to follow proven software engineering practices, ensuring consistent, high-quality output regardless of the specific task or context.

## Core Philosophy

Superpowers is built on four foundational principles that guide every aspect of its design:

**Test-Driven Development (TDD):** At the heart of Superpowers lies an unwavering commitment to TDD. The framework enforces the RED-GREEN-REFACTOR cycle, ensuring that no production code is written without a failing test first. This approach catches bugs early, documents expected behavior, and creates a safety net for future refactoring. The iron law of Superpowers is clear: "NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST."

**Systematic Over Ad-Hoc:** Rather than relying on intuition or context-dependent decision making, Superpowers provides clear, repeatable processes for every development task. This systematic approach means that whether you're debugging an issue, planning a feature, or reviewing code, there's a defined workflow to follow. This consistency reduces cognitive load and ensures nothing important is missed.

**Complexity Reduction:** Superpowers treats simplicity as a primary goal. The framework actively fights against over-engineering by emphasizing YAGNI and encouraging developers to build only what's needed right now. Each skill is designed to help identify and eliminate unnecessary complexity before it becomes entrenched in the codebase.

**Evidence Over Claims:** Superpowers demands verification before declaring success. An implementation isn't complete until tests pass, code reviews are satisfied, and the original requirements are verified. This evidence-based approach prevents the common scenario where AI agents claim a task is complete but leave behind subtle bugs or incomplete implementations.

## Skills Library Overview

![Superpowers Skills Overview](/assets/img/diagrams/superpowers-overview.svg)

### Understanding the Skills Architecture

The Superpowers framework organizes its capabilities into a comprehensive skills library that covers every phase of software development. This architecture represents a paradigm shift in how AI agents approach development tasks, transforming them from simple code generators into systematic engineers who follow proven methodologies.

**Testing Skills (2 Skills)**

The testing category contains two critical skills that ensure code quality from the ground up:

**test-driven-development** is the cornerstone skill that enforces the RED-GREEN-REFACTOR cycle. When this skill activates, the AI agent must first write a failing test that defines the expected behavior. Only after watching the test fail can the agent write the minimal production code needed to make it pass. This discipline prevents the common anti-pattern of writing tests after the fact, which often leads to tests that verify implementation details rather than requirements. The skill includes a comprehensive testing anti-patterns reference that helps agents avoid common mistakes like testing private methods, mocking too much, or creating brittle tests that break with minor refactoring.

**verification-before-completion** ensures that declared success is backed by evidence. Before marking any task as complete, this skill requires the agent to verify that tests pass, the original requirements are met, and no regressions have been introduced. This prevents the frustrating scenario where an AI agent claims a fix is complete but the underlying issue persists or new bugs have been introduced.

**Debugging Skills (1 Skill)**

The debugging category provides a systematic approach to finding and fixing issues:

**systematic-debugging** implements a four-phase root cause analysis process that transforms debugging from a guessing game into a methodical investigation. Rather than randomly trying potential fixes, agents following this skill work through hypothesis formation, evidence gathering, targeted testing, and verified resolution. The skill includes advanced techniques like root-cause-tracing for identifying where problems originate, defense-in-depth strategies for preventing similar issues, and condition-based-waiting patterns for debugging timing-sensitive issues. This systematic approach dramatically reduces the time spent on debugging and ensures fixes address root causes rather than symptoms.

**Collaboration Skills (9 Skills)**

The collaboration category is the largest, reflecting Superpowers' emphasis on human-AI partnership:

**brainstorming** activates before any code is written, engaging the user in a Socratic dialogue to refine rough ideas into clear specifications. Rather than accepting vague requirements, this skill asks probing questions, explores alternatives, and presents design options in digestible sections for validation. The result is a saved design document that serves as the foundation for all subsequent work.

**writing-plans** takes an approved design and breaks it into bite-sized tasks, each taking 2-5 minutes to complete. Every task includes exact file paths, complete code snippets, and verification steps. This granularity ensures that implementation stays on track and makes it easy to resume work after interruptions.

**executing-plans** manages batch execution with human checkpoints, allowing for controlled progress through complex implementations. This skill provides the structure for systematic execution while maintaining human oversight at critical decision points.

**dispatching-parallel-agents** enables concurrent subagent workflows for tasks that can be parallelized. This skill coordinates multiple AI agents working simultaneously, managing dependencies and merging results efficiently.

**requesting-code-review** activates between tasks to review work against the plan, reporting issues by severity. Critical issues block progress, ensuring quality gates are maintained throughout development.

**receiving-code-review** provides a structured approach for responding to feedback, ensuring that review comments are addressed systematically rather than being dismissed or forgotten.

**using-git-worktrees** creates isolated workspaces on new branches, enabling parallel development without conflicts. This skill handles project setup and verifies clean test baselines before work begins.

**finishing-a-development-branch** manages the final stages of development, presenting options for merge, pull request, or branch management while ensuring all tests pass and cleanup is performed.

**subagent-driven-development** represents the most sophisticated workflow, launching fresh subagents for each task with a two-stage review process. The first stage checks spec compliance, ensuring the implementation matches requirements. The second stage evaluates code quality, catching issues like poor naming, missing error handling, or inefficient algorithms. This dual review approach has resulted in a 94% PR rejection rate for low-quality submissions, demonstrating its effectiveness at maintaining standards.

**Meta Skills (2 Skills)**

The meta category provides tools for extending and understanding the framework:

**writing-skills** guides the creation of new skills following established best practices. This skill includes testing methodology to ensure new skills integrate properly with the existing framework and maintain the same quality standards.

**using-superpowers** serves as an introduction to the skills system, helping new users understand how to leverage Superpowers effectively and how different skills interact with each other.

## Subagent-Driven Development

![Subagent-Driven Development Workflow](/assets/img/diagrams/superpowers-subagent-workflow.svg)

### Understanding the Subagent Architecture

The subagent-driven development workflow represents one of Superpowers' most innovative contributions to AI-assisted software development. This architecture addresses a fundamental challenge in AI coding: maintaining focus and quality over extended development sessions.

**Fresh Subagent Per Task**

Unlike traditional approaches where a single AI agent handles an entire project, subagent-driven development creates a fresh subagent for each discrete task. This approach offers several critical advantages:

First, it eliminates context pollution. When a single agent works through multiple tasks, it accumulates assumptions, shortcuts, and mental models that may not apply to subsequent tasks. Fresh subagents start with clean context, ensuring each task is approached with fresh perspective.

Second, it enables parallelization. Tasks that don't depend on each other can be dispatched to multiple subagents simultaneously, dramatically reducing overall development time. The controller agent manages dependencies and merges results appropriately.

Third, it provides natural quality gates. Each subagent completes its task and returns results to the controller, which can then evaluate the work before proceeding. This creates a rhythm of checkpoints that prevents compounding errors.

**Two-Stage Review Process**

The two-stage review process is what sets Superpowers apart from other AI development frameworks:

**Stage 1: Spec Compliance Review**

The first review stage verifies that the implementation matches the approved specification. This review asks: Does the code do what was requested? Are all requirements addressed? Are edge cases handled? This stage catches misunderstandings and incomplete implementations before they propagate.

**Stage 2: Code Quality Review**

The second review stage evaluates the quality of the implementation itself. This review asks: Is the code readable and maintainable? Are names clear and consistent? Is error handling comprehensive? Are there performance concerns? This stage catches issues that wouldn't cause test failures but would create technical debt.

**Controller Curates Context**

The controller agent plays a crucial role in managing context. Rather than dumping entire codebases into subagent prompts, the controller carefully curates what each subagent sees. This includes:

- Relevant files and their contents
- Applicable design decisions and constraints
- Previous task results that affect the current task
- Testing requirements and verification steps

This curated context ensures subagents have exactly what they need without being overwhelmed by irrelevant information. It also allows the controller to maintain a coherent vision across all tasks while subagents focus on their specific assignments.

The result is a development process where AI agents can work autonomously for hours at a time, following a plan that humans have approved, while maintaining the quality standards that professional software development demands.

## TDD with Iron Laws

![TDD Workflow](/assets/img/diagrams/superpowers-tdd-workflow.svg)

### Understanding the RED-GREEN-REFACTOR Cycle

Test-Driven Development forms the backbone of Superpowers' approach to software quality. The framework doesn't just encourage TDD; it enforces it through iron laws that cannot be bypassed. This strict adherence to TDD principles transforms how AI agents approach implementation tasks.

**The Iron Law: NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST**

This single rule fundamentally changes the development dynamic. When an AI agent encounters a feature request, it cannot immediately write implementation code. Instead, it must:

1. Write a test that defines the expected behavior
2. Run the test and watch it fail (proving the test is valid)
3. Write the minimal code needed to make the test pass
4. Run the test and watch it succeed
5. Refactor if needed, with tests protecting against regressions

This sequence ensures that every line of production code exists to make a test pass, eliminating speculative or unnecessary code.

**RED Phase: Writing Failing Tests**

The RED phase is where requirements become concrete. The AI agent must translate a specification into executable assertions that define success. This phase requires understanding not just what the code should do, but how to verify that behavior.

Key aspects of the RED phase include:
- Tests must be specific enough to fail for the right reasons
- Tests should be minimal, testing one behavior at a time
- Tests must be deterministic, producing the same result every time
- Tests should be independent, not depending on execution order

**GREEN Phase: Making Tests Pass**

The GREEN phase focuses on writing the minimum code necessary to make failing tests pass. This constraint prevents over-engineering and keeps implementations focused.

The agent must:
- Write only enough code to pass the current failing test
- Avoid adding functionality not required by tests
- Resist the temptation to "future-proof" code
- Keep implementations simple and direct

**REFACTOR Phase: Improving Design**

Once tests pass, the REFACTOR phase allows for code improvement without changing behavior. Tests provide a safety net that catches regressions during refactoring.

Common refactoring activities include:
- Extracting duplicated code into shared functions
- Improving naming for clarity
- Simplifying complex conditionals
- Removing dead code that accumulated during implementation

**Verification Steps**

Superpowers adds explicit verification steps at each phase transition:

- After RED: Verify the test fails for the right reason, not due to syntax errors or missing dependencies
- After GREEN: Verify all tests pass, not just the new one
- After REFACTOR: Verify behavior unchanged by running the full test suite

These verification steps prevent common TDD anti-patterns like writing tests that always pass or refactoring that accidentally changes behavior.

**Deleting Code Written Before Tests**

One of Superpowers' most controversial but effective rules is that code written before tests should be deleted. If an AI agent accidentally writes implementation code before tests, the framework requires starting over with tests first. This rule reinforces the TDD discipline and prevents the gradual erosion of testing practices that often occurs in real-world development.

## Brainstorming to Implementation Flow

![Brainstorming to Implementation Flow](/assets/img/diagrams/superpowers-brainstorming-flow.svg)

### Understanding the Complete Development Workflow

The brainstorming to implementation flow represents the complete journey from initial idea to finished feature. This workflow demonstrates how Superpowers transforms vague concepts into production-ready code through a series of well-defined stages.

**Stage 1: Brainstorming and Specification**

The workflow begins when a user expresses a need. Rather than immediately generating code, the brainstorming skill engages in Socratic dialogue:

- What problem are you trying to solve?
- Who are the users of this feature?
- What are the success criteria?
- What constraints exist?
- What alternatives have been considered?

This questioning continues until a clear specification emerges. The skill presents design options in sections short enough to digest, allowing for incremental approval and refinement. The final specification is saved as a design document that guides all subsequent work.

**Stage 2: Design Validation**

Before implementation begins, the design document is reviewed for completeness and feasibility. This stage catches issues early when they're cheapest to fix. The validation process asks:

- Are all requirements addressed?
- Are edge cases considered?
- Is the design testable?
- Are dependencies identified?
- Is the scope appropriate?

Only after design approval does the workflow proceed to planning.

**Stage 3: Implementation Planning**

The writing-plans skill transforms the approved design into a detailed implementation plan. Each task in the plan includes:

- Exact file paths to modify
- Complete code snippets for changes
- Verification steps to confirm success
- Estimated time (2-5 minutes per task)

This granularity serves several purposes:
- Tasks are small enough to complete without losing focus
- Progress is measurable and resumable after interruptions
- Each task has clear success criteria
- The plan can be reviewed before execution begins

**Stage 4: Git Worktree Setup**

The using-git-worktrees skill creates an isolated development environment:

- A new branch is created for the feature
- A git worktree provides a separate working directory
- Project setup runs to ensure dependencies are installed
- Tests run to establish a clean baseline

This isolation prevents conflicts with other work and ensures a known-good starting point.

**Stage 5: Subagent-Driven Development**

With the plan approved and environment prepared, subagent-driven development takes over:

- Each task is dispatched to a fresh subagent
- The subagent completes the task following TDD principles
- Results return to the controller for review
- Two-stage review ensures quality
- Progress continues until all tasks complete

**Stage 6: Code Review Integration**

Between tasks, the requesting-code-review skill ensures quality gates:

- Work is reviewed against the plan
- Issues are categorized by severity
- Critical issues block progress
- Non-critical issues are tracked for later resolution

**Stage 7: Finishing the Branch**

When all tasks complete, the finishing-a-development-branch skill manages the final steps:

- All tests are verified to pass
- Options are presented: merge, create PR, keep branch, or discard
- Cleanup is performed on worktrees and temporary files
- Documentation is updated if needed

This complete workflow ensures that ideas become implementations through a systematic, quality-focused process that maintains human oversight at critical decision points while allowing AI agents to work autonomously on well-defined tasks.

## Platform Support

Superpowers supports a wide range of AI coding platforms, making it accessible regardless of your preferred development environment:

**Claude Code:** Available through the official Claude plugin marketplace. Installation is straightforward using the plugin command: `/plugin install superpowers@claude-plugins-official`. This is the primary platform for which Superpowers was developed and offers the most integrated experience.

**Gemini CLI:** Google's Gemini CLI supports Superpowers through extensions. Install with: `gemini extensions install https://github.com/obra/superpowers`. Updates are managed through the same extension system.

**Cursor:** The popular AI-powered editor supports Superpowers through its plugin marketplace. Search for "superpowers" or use the command `/add-plugin superpowers` in Cursor Agent chat.

**Codex:** OpenAI's Codex requires manual setup following instructions from the Superpowers repository. Detailed documentation is available at `docs/README.codex.md`.

**OpenCode:** Similar to Codex, OpenCode requires manual setup with instructions available at `docs/README.opencode.md`.

**GitHub Copilot CLI:** Microsoft's command-line Copilot supports Superpowers through marketplace commands: `copilot plugin marketplace add obra/superpowers-marketplace` followed by `copilot plugin install superpowers@superpowers-marketplace`.

## Notable Features

**94% PR Rejection Rate:** The two-stage review process in subagent-driven development has achieved a 94% rejection rate for low-quality pull requests. This statistic demonstrates the effectiveness of systematic quality gates in preventing substandard code from entering codebases.

**Iron Laws and Rationalization Tables:** Superpowers uses "iron laws" - rules that cannot be bypassed - to enforce critical practices. When an AI agent attempts to rationalize skipping a step, rationalization tables provide counter-arguments that reinforce the importance of following the process.

**Red Flags Lists:** Each skill includes a list of red flags that indicate when something is going wrong. These lists help AI agents self-correct before problems compound, catching issues like premature optimization, insufficient testing, or scope creep.

**Skill TDD Methodology:** The framework applies its own TDD principles to skill development. New skills are tested using subagent-driven development, ensuring that the skills themselves meet quality standards before being added to the library.

## Getting Started

**Installation:**

For Claude Code users, installation is simple:

```bash
/plugin install superpowers@claude-plugins-official
```

For other platforms, follow the platform-specific instructions in the Platform Support section above.

**Basic Usage:**

After installation, Superpowers activates automatically when appropriate. Simply start a conversation with your AI coding agent about a task, and the relevant skills will trigger:

- "Help me plan this feature" triggers the brainstorming skill
- "Let's debug this issue" triggers systematic-debugging
- "I need to implement X" triggers the full planning and execution workflow

**Verification:**

To verify installation is working correctly, start a new session and ask for something that should trigger a skill. The agent should automatically invoke the relevant Superpowers skill rather than jumping straight into code generation.

**Learning More:**

- GitHub Repository: https://github.com/obra/superpowers
- Documentation: Available in the repository's README and docs folder
- Blog Post: [Superpowers for Claude Code](https://blog.fsck.com/2025/10/09/superpowers/)
- Discord Community: [Join us](https://discord.gg/35wsABTejz) for support and discussions

## Conclusion

Superpowers represents a significant advancement in AI-assisted software development. By providing a comprehensive framework of composable skills that enforce best practices, it transforms AI coding agents from code generators into disciplined engineers. The framework's emphasis on test-driven development, systematic workflows, and quality gates addresses the fundamental limitations of AI-generated code.

With 140,708 GitHub stars and active development by Jesse Vincent and the Prime Radiant team, Superpowers has proven its value to the developer community. Whether you're using Claude Code, Gemini CLI, Cursor, or another AI coding platform, Superpowers provides the structure needed to produce high-quality, maintainable code consistently.

The framework's innovative approach to subagent-driven development, combined with its two-stage review process and iron laws, ensures that AI agents follow the same rigorous practices that experienced software engineers have developed over decades. As AI continues to transform software development, Superpowers provides the guardrails needed to harness that power responsibly.
