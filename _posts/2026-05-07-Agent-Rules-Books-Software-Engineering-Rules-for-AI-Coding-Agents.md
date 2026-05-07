---
layout: post
title: "Agent Rules Books: Software Engineering Rules Distilled for AI Coding Agents"
description: "Learn how agent-rules-books distills 14 classic software engineering books into ready-to-use AGENTS.md rules for Claude Code, Codex, and Cursor with a three-tier compression system from full to nano."
date: 2026-05-07
header-img: "img/post-bg.jpg"
permalink: /Agent-Rules-Books-Software-Engineering-Rules-AI-Coding-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, Open Source]
tags: [AI coding agents, AGENTS.md, Claude Code, Codex, Cursor, software engineering rules, Clean Code, Domain-Driven Design, refactoring, code quality, AI agent rules, developer tools, open source, coding standards, AI workflow]
keywords: "how to use AGENTS.md rules for AI coding agents, agent-rules-books tutorial, Claude Code rules setup, Codex agent rules configuration, best AI coding agent rules, Clean Code rules for AI agents, Domain-Driven Design AGENTS.md, refactoring rules for coding agents, open source AI agent rules, AI coding agent configuration guide"
author: "PyShine"
---

# Agent Rules Books: Software Engineering Rules Distilled for AI Coding Agents

Agent-rules-books is an open-source project that distills 14 classic software engineering books into ready-to-use AGENTS.md rule sets for AI coding agents like Claude Code, Codex, and Cursor. Rather than letting AI agents rely solely on their training data, this project provides structured, book-derived rules that guide agents toward better architectural decisions, cleaner code, and safer refactoring -- all through a three-tier compression system that balances coverage with context budget constraints.

Created by Maciej Ciemborowicz and licensed under MIT, the project has quickly gained over 1,100 stars on GitHub by solving a practical problem: how do you give an AI coding agent the wisdom of books like *Clean Code*, *Domain-Driven Design*, and *Designing Data-Intensive Applications* without consuming your entire context window?

![Architecture Diagram](/assets/img/diagrams/agent-rules-books/agent-rules-books-architecture.svg)

### Understanding the Three-Tier Architecture

The architecture diagram above illustrates how agent-rules-books transforms raw book knowledge into actionable agent guidance. Let us break down each component:

**Source Books Layer (14 Books)**

The project draws from 14 well-known software engineering books spanning code quality, architecture, domain modeling, refactoring, production systems, and legacy code. Each book contributes its distinctive perspective -- Clean Code pushes for readability and small functions, Domain-Driven Design emphasizes ubiquitous language and bounded contexts, Release It! demands explicit failure semantics and circuit breakers.

**Full Canonical Source**

Each book is first distilled into a complete rule set (the `full` version) that preserves the book's structure, distinctive bias, and operational rules. These full versions range from 177 to 523 rules per book, expressed using MUST, SHOULD, and MUST NOT modal verbs to match the book's original intent.

**Rule Compression Process**

The compression process (documented in `PROCESS.md`) transforms full rule sets into decision-equivalent compressed versions. This is not mere summarization -- it preserves the rules that materially change an AI agent's decisions while removing rules that agents already follow by default.

**Mini and Nano Releases**

- **Mini** (28-47 rules): The recommended working layer for most tasks. Preserves the book's unique point of view while keeping only decision-changing rules.
- **Nano** (14-26 rules): The compact fallback for very tight always-on context budgets. Keeps only the rules that correct known model biases and prevent shortcuts in risky areas.

**AI Coding Agents**

The compressed rules integrate with Claude Code, Codex, and Cursor through AGENTS.md files, skills, scoped rules, or MCP/RAG patterns depending on the editor.

## The 14 Source Books

The project covers a broad spectrum of software engineering knowledge:

| Category | Books | Primary Focus |
|----------|-------|---------------|
| Code Quality | Clean Code, Code Complete | Readability, naming, small functions, defensive programming |
| Architecture | Clean Architecture, Patterns of Enterprise Application Architecture | Layer boundaries, dependency rule, enterprise patterns |
| Domain Modeling | Domain-Driven Design, DDD Distilled, Implementing DDD | Bounded contexts, ubiquitous language, aggregates |
| Refactoring | Refactoring, Refactoring.Guru, A Philosophy of Software Design | Small-step refactoring, code smells, complexity reduction |
| Production | Release It!, Designing Data-Intensive Applications | Circuit breakers, consistency, replication, failure semantics |
| General | The Pragmatic Programmer, Working Effectively with Legacy Code | DRY, orthogonality, seams, characterization tests |

> **Key Insight:** The full rule sets contain between 177 and 523 rules per book, but the compression process reduces these to 28-47 rules in mini and 14-26 in nano -- an 85-95% reduction while preserving decision-changing guidance.

## The Compression Process

The most innovative aspect of agent-rules-books is its rigorous compression process. Rather than simply summarizing books, the project classifies every rule before deciding whether to keep, merge, or drop it.

![Compression Process Diagram](/assets/img/diagrams/agent-rules-books/agent-rules-books-compression-process.svg)

### Understanding the Compression Pipeline

The compression process diagram above shows how rules flow from the full source through classification into either kept or dropped categories, then into the mini and nano releases.

**Rule Classification System**

Every rule in the full source is classified into one of eight categories before compression:

| Category | Description | Compression Decision |
|----------|-------------|---------------------|
| `book-thesis` | The book's central corrective bias | Always keep in mini and nano |
| `decision-changing` | Changes architecture, modeling, or error handling decisions | Always keep in mini |
| `micro-decision` | Changes repeated local choices (naming, function shape) | Keep if agents commonly miss it |
| `conflict-resolver` | Resolves tradeoffs between competing approaches | Always keep in mini |
| `trigger` | Activates only when touching a risky area | Keep if it prevents shortcuts |
| `checklist-only` | Useful for final scan, not as a main rule | Convert to checklist items |
| `framing` | Useful context, not operational | Drop from mini and nano |
| `default` | Agents already follow without prompting | Drop only with evidence |

**What Gets Preserved**

The compression process explicitly preserves:
- Every `book-thesis` rule, even when it acts through local code-shape discipline
- Every `decision-changing` and `conflict-resolver` rule
- Every `micro-decision` or `trigger` that target agents commonly miss
- Enough of the book's own bias and vocabulary that mini still feels like that book

**What Gets Dropped**

Rules are removed only when:
- They are verified defaults for target agents (with evidence, not assumption)
- They are truly redundant across sections
- They are framing context rather than operational instructions
- They are too situational for the compressed layer

> **Takeaway:** The compression process is decision-equivalent, not sentence-equivalent. The compressed versions preserve the rules that materially change an AI agent's design, architecture, refactoring, review, risk decisions, and repeated local implementation choices under context pressure.

### Example: Clean Code Mini

To see the compression in action, compare the structure of the Clean Code mini release:

```markdown
# OBEY Clean Code by Robert C. Martin

## When to use
Use when readability, local reasoning, and maintainable code
shape are the main concerns.

## Primary bias to correct
Working code is not automatically clean code.

## Decision rules
- Treat cleanliness as part of delivery.
- Write for local reasoning.
- Use precise names and one term per concept.
- Keep functions small, focused, and at one level of abstraction.
- Separate commands from queries.
- Expose behavior rather than raw representation.
...

## Trigger rules
- When a function mixes setup, validation, computation,
  and side effects, split the phases.
- When a comment explains control flow, simplify names
  or structure before keeping the comment.
...

## Final checklist
- Can a reader follow the change locally?
- Are names and APIs carrying the meaning without narration?
- Is mutation explicit and the happy path still clear?
...
```

Every mini release follows the same structure: when to use, primary bias to correct, decision rules, trigger rules, and a final checklist. This consistency makes it easy to switch between rule sets.

## Choosing the Right Rule Set

Not every project needs every rule set. The README provides clear guidance on which books to load for different tasks:

![Choosing Rules Diagram](/assets/img/diagrams/agent-rules-books/agent-rules-books-choosing-rules.svg)

### Understanding the Rule Selection Guide

The choosing rules diagram above maps common development tasks to their recommended rule sets. Here is a detailed breakdown:

**Everyday Code Quality**

For day-to-day coding and code review, load `clean-code`, `code-complete`, or `the-pragmatic-programmer`. These rule sets push for readability, small functions, precise naming, and disciplined implementation. Clean Code is the strongest default for most tasks.

**Architecture and Boundaries**

When designing system boundaries, load `clean-architecture`, `domain-driven-design`, or `patterns-of-enterprise-application-architecture`. These rule sets enforce the dependency rule, layer separation, and appropriate enterprise patterns.

**Domain Modeling**

For business-heavy domains, choose from `domain-driven-design`, `domain-driven-design-distilled`, or `implementing-domain-driven-design`. The distilled version is best when you want DDD benefits without excessive ceremony; the full DDD rule set is for complex domains where bounded contexts and ubiquitous language are critical.

**Refactoring and Legacy Code**

For refactoring tasks, load `refactoring`, `refactoring-guru`, or `a-philosophy-of-software-design`. For legacy code, add `working-effectively-with-legacy-code`. These rule sets emphasize small steps, behavior preservation, seams, and characterization tests.

**Production and Data Systems**

For services and data-intensive systems, load `release-it` and `designing-data-intensive-applications`. These rule sets demand explicit failure semantics, circuit breakers, consistency contracts, and safe schema evolution.

> **Important:** The compatibility matrix shows 78 complementary pairs, 11 overlapping pairs, and only 2 conflicting pairs across all 14 books. The two conflicts are between Domain-Driven Design and Patterns of Enterprise Application Architecture -- these books push different architectural decisions and should not be loaded together as equal active guidance.

## The Compatibility Matrix

One of the most valuable features is the book compatibility matrix. Since different books operate at different abstraction levels and sometimes encourage different tradeoffs, the project provides explicit compatibility ratings:

| Symbol | Meaning | Count |
|--------|---------|-------|
| Complementary | Can be combined as equal active guidance | 78 |
| Overlapping | Choose one; they apply similar pressure | 11 |
| Conflicting | Do not load together as equal active rule sets | 2 |

The two conflicting pairs are:
- **Domain-Driven Design vs. Patterns of Enterprise Application Architecture** -- DDD pushes for a rich domain model while PoEAA catalogues patterns that include both domain model and transaction script approaches, creating tension when loaded together.
- **Implementing DDD vs. Patterns of Enterprise Application Architecture** -- Same conflict as above, since IDDD extends DDD's implementation guidance.

Overlapping pairs include Clean Code vs. Code Complete (both push for code quality at the implementation level), and the three DDD books (DDD, DDD Distilled, IDDD) which cover similar territory at different depths.

## Installation and Usage

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ciembor/agent-rules-books.git

# Navigate to the project
cd agent-rules-books
```

### Using with Claude Code

For Claude Code, copy the desired mini rule file as your `AGENTS.md`:

```bash
# Copy the Clean Code mini rules as your project AGENTS.md
cp clean-code/clean-code.mini.md /your/project/AGENTS.md

# Or use nano for a compact always-on layer
cp clean-code/clean-code.nano.md /your/project/AGENTS.md
```

### Using with Cursor

For Cursor, add the rules to your `.cursor/rules/` directory:

```bash
# Create rules directory if it does not exist
mkdir -p /your/project/.cursor/rules

# Copy the desired rule set
cp domain-driven-design/domain-driven-design.mini.md /your/project/.cursor/rules/ddd.md
```

### Using with Codex

For Codex, reference the rules in your project configuration:

```bash
# Copy the rule file to your project root
cp release-it/release-it.mini.md /your/project/AGENTS.md
```

### Recommended Loading Strategy

The project recommends a progressive loading approach:

1. **Always-on layer**: Load one `nano` rule set as your default (e.g., `clean-code.nano.md`)
2. **On-demand layer**: Load `mini` rule sets when working on specific tasks (e.g., load `release-it.mini.md` when working on production paths)
3. **Reference layer**: Consult `full` rule sets only when you need the complete source material

> **Amazing:** The Domain-Driven Design full rule set contains 523 rules across 979 lines, but the nano version compresses this to just 21 rules in 39 lines -- a 96% reduction that still preserves the core DDD bias toward bounded contexts, ubiquitous language, and protecting the core domain.

## The Rule Workbench

Behind the released rule sets is a sophisticated workbench system in `_rule-workbench/` that manages the compression pipeline:

**Per-Book Workbench Structure:**

| File | Purpose |
|------|---------|
| `full.md` | Symlink to the canonical full source |
| `traceability.md` | Maps every retained rule back to source sections |
| `mini.md` | Compressed working version |
| `nano.md` | Compact fallback version |

**Traceability System:**

Every retained rule in mini gets an `M*` identifier, and every retained rule in nano gets an `N*` identifier. Each ID references the source section names and line ranges in the full source. For every omitted rule, the traceability file records one of:
- `covered by Mx` -- the rule's effect survived in a merged mini rule
- `covered by Nx` -- the rule's effect survived in a merged nano rule
- `intentionally lost` -- the rule was dropped with explicit justification

This traceability system makes the compression auditable. You can trace any compressed rule back to its source and understand why a particular rule was kept, merged, or dropped.

## Honest Criticism

The project includes a `CRITICISM.md` file that transparently documents recurring criticisms from the Reddit community, along with how each has been addressed. The most significant criticisms:

**1. No Clear Measurement of Improvement (9/10 validity)**

Without benchmarks, before/after comparisons, or defect-rate data, it is hard to know whether the rules actually improve code quality. The project acknowledges this is only weakly addressed (2/10 solved) -- the compression is auditable, but real coding outcome measurements are still missing.

**2. Token Burning and Context Pollution (9/10 validity)**

Loading too many rules at once can crowd out task-specific context. This was largely addressed (8/10) through the three-level release model and explicit guidance to use the smallest effective mechanism.

**3. Progressive Loading May Be Better (9/10 validity)**

Rules are more useful when loaded selectively. This was directly addressed (9/10) through the USAGE.md documentation that recommends progressive loading, scoped rules, skills, and retrieval-based patterns.

**4. Rules from Different Books May Conflict (8/10 validity)**

Different books push different architectural decisions. This was partially addressed (7/10) through the compatibility matrix and loading discipline recommendations.

## Adding a New Book

The project includes a documented workflow for adding new book-derived rule sets:

1. Ask the chatbot for the complete book outline: every chapter, section, and operational rule
2. Expand the extraction until nothing material is missing -- recover non-negotiable rules, tradeoff rules, trigger rules, anti-patterns, and review guidance
3. Produce a full `AGENTS.md` in the repository's standard, using MUST, SHOULD, and MUST NOT modal verbs
4. Review the generated file before importing -- check that local discipline was not flattened into generic advice
5. Move the approved file to `_rule-workbench/<book-name>/full.md`
6. Run the workflow from `_rule-workbench/PROCESS.md` for that book
7. Execute the release instructions from `_rule-workbench/RELEASE.md`

## Release Matrix

The current release (v0.5) includes these metrics for each book:

| Rule Set | Full Rules | Mini Rules | Nano Rules | Mini Size | Nano Size |
|----------|-----------:|-----------:|-----------:|----------:|----------:|
| A Philosophy of Software Design | 177 | 28 | 17 | 5.6 KB | 2.3 KB |
| Clean Architecture | 289 | 31 | 18 | 5.4 KB | 2.3 KB |
| Clean Code | 220 | 29 | 14 | 3.8 KB | 1.2 KB |
| Code Complete | 180 | 38 | 23 | 6.7 KB | 2.5 KB |
| Designing Data-Intensive Applications | 205 | 37 | 16 | 6.9 KB | 2.6 KB |
| Domain-Driven Design | 523 | 30 | 21 | 5.6 KB | 2.3 KB |
| DDD Distilled | 158 | 38 | 23 | 6.4 KB | 2.5 KB |
| Implementing DDD | 177 | 39 | 19 | 7.3 KB | 2.7 KB |
| Patterns of EA | 196 | 36 | 17 | 8.1 KB | 2.8 KB |
| Refactoring | 242 | 31 | 19 | 5.2 KB | 2.0 KB |
| Release It! | 204 | 30 | 20 | 6.4 KB | 2.2 KB |
| The Pragmatic Programmer | 179 | 47 | 26 | 7.2 KB | 2.3 KB |
| Working w/ Legacy Code | 193 | 32 | 17 | 5.7 KB | 1.8 KB |
| Refactoring.Guru | 478 | 46 | 23 | 6.3 KB | 2.6 KB |

## Conclusion

Agent-rules-books represents a thoughtful approach to giving AI coding agents structured engineering guidance. Rather than dumping entire books into context windows, the project applies a rigorous compression process that preserves decision-changing rules while respecting context budget constraints. The three-tier release system (full, mini, nano), the compatibility matrix, and the traceability system make this more than just a collection of rules -- it is a systematic framework for injecting software engineering wisdom into AI agent workflows.

The honest self-criticism in `CRITICISM.md` and the documented compression process in `PROCESS.md` demonstrate a maturity that sets this project apart from simpler rule collections. While the lack of empirical outcome measurements remains the strongest open criticism, the project's structural safeguards -- progressive loading, compatibility checks, and source-faithful compression -- provide a solid foundation for practical use.

**Links:**

- GitHub: [https://github.com/ciembor/agent-rules-books](https://github.com/ciembor/agent-rules-books)
- License: MIT