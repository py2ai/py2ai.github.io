---
layout: post
title: "Agent Rules Books: AGENTS.md Rules for AI Coding Agents from Classic Programming Books"
description: "Discover Agent Rules Books, a curated collection of AGENTS.md rules distilled from 14 classic programming books for Codex, Cursor, and Claude Code agents."
date: 2026-05-21
header-img: "img/post-bg.jpg"
permalink: /Agent-Rules-Books-AGENTS.md-Rules-for-AI-Coding-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Developer Tools, Open Source]
tags: [AGENTS.md, AI coding agents, Codex rules, Cursor rules, Claude Code rules, Clean Code, Clean Architecture, Domain-Driven Design, refactoring, code quality]
keywords: "AGENTS.md rules for AI coding agents, how to use Agent Rules Books with Codex, Cursor AI coding rules setup, Claude Code project rules tutorial, Clean Code rules for AI agents, Domain-Driven Design agent skills, AI coding agent best practices, open source agent rules library, programming books distilled for AI, refactoring rules for coding agents"
author: "PyShine"
---

# Agent Rules Books: AGENTS.md Rules for AI Coding Agents from Classic Programming Books

Agent Rules Books is an open-source project that distills timeless software engineering wisdom from 14 classic programming books into structured, ready-to-use AGENTS.md rules for modern AI coding agents. Whether you are using Codex, Cursor, Claude Code, or GitHub Copilot, this repository provides context-aware rule sets that help AI agents write cleaner, more maintainable, and architecturally sound code. By translating principles from books like Clean Code, Clean Architecture, Domain-Driven Design, and Refactoring into actionable agent instructions, the project bridges the gap between human software craftsmanship and AI-generated code.

> **Key Insight:** The project's validation experiment showed that using structured mini rules from A Philosophy of Software Design scored 74/100 versus only 46/100 when the agent was simply told to obey the book title. Concrete, enumerated rules dramatically outperform vague references.

![Architecture Diagram](/assets/img/diagrams/agent-rules-books/agent-rules-books-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates how Agent Rules Books transforms classic programming literature into actionable AI agent guidance. The system is organized into five conceptual layers that flow from source material to editor-specific delivery.

**Layer 1: Source Books**
At the foundation are 14 classic programming books spanning software design, architecture, refactoring, domain modeling, reliability, and data-intensive systems. These include Clean Code and Clean Architecture by Robert C. Martin, Refactoring by Martin Fowler, Domain-Driven Design by Eric Evans, Designing Data-Intensive Applications by Martin Kleppmann, Code Complete by Steve McConnell, The Pragmatic Programmer by Andrew Hunt and David Thomas, Release It! by Michael T. Nygard, Working Effectively with Legacy Code by Michael Feathers, Patterns of Enterprise Application Architecture by Martin Fowler, Implementing Domain-Driven Design and DDD Distilled by Vaughn Vernon, A Philosophy of Software Design by John Ousterhout, and Refactoring.Guru. Each book is distilled into a standalone rule set that captures its core decision pressure without reproducing copyrighted text.

**Layer 2: Version Tiers**
Every book rule set is released in three tool-agnostic Markdown versions. The Full version is the canonical complete source and reference, best used for audits and deep sessions. The Mini version is the recommended default for most real tasks, especially as a focused skill. The Nano version is the compact fallback for very tight context budgets or portable cross-tool baselines. This tiered approach ensures that agents receive the right amount of guidance without exceeding context limits.

**Layer 3: Delivery Patterns**
The project supports five delivery patterns. Skills or commands are best for refactoring passes, reviews, migrations, and domain modeling. Always-on project rules provide stable defaults that affect most tasks. Scoped rules target one directory, file type, or subsystem. On-demand rules are invoked only when the task matches. Retrieval, MCP, or RAG patterns handle large reference material that is too rarely needed for always-on context.

**Layer 4: Editor Targets**
The rule sets are designed for three major AI coding editors. Codex reads AGENTS.md from the repo root or nested directories, supports skills in .agents/skills/, and uses model_instructions_file for custom config. Claude Code uses CLAUDE.md or .claude/CLAUDE.md, supports scoped rules in .claude/rules/, and skills in .claude/skills/. Cursor uses .cursor/rules/*.mdc with rule types including Always, Auto Attached, Agent Requested, and Manual.

**Layer 5: Compatibility Matrix**
A comprehensive compatibility matrix compares all 91 book pairs, marking them as complementary, overlapping, or conflicting. This helps teams avoid loading conflicting guidance and choose complementary rule sets that strengthen each other. For example, Domain-Driven Design conflicts with Patterns of Enterprise Application Architecture, while Clean Code overlaps with Code Complete.

**Data Flow:**
The workflow begins when a developer selects a book rule set and version tier. The chosen rules are then delivered through an appropriate pattern to the target editor. The editor loads the rules into the agent's context, shaping code generation, review, and refactoring decisions. The compatibility matrix guides rule set selection to prevent conflicting guidance.

> **Takeaway:** Start with one primary Mini rule set as a skill, prefer scoped on-demand loading over global loading, and use the compatibility matrix to avoid pairing conflicting books like DDD with PoEAA.

![Features Diagram](/assets/img/diagrams/agent-rules-books/agent-rules-books-features.svg)

### Understanding the Features

The features diagram shows the six core capabilities of Agent Rules Books, each branching into specific sub-features that make the project practical for daily AI-assisted development.

**14 Book Rule Sets:**
Each rule set captures the decision pressure of its source book. Clean Code focuses on readability, naming, small functions, and separating commands from queries. Clean Architecture enforces inward dependencies, independent business rules, and replaceable details. Refactoring provides safe code improvement steps, smell detection, and behavior preservation. Domain-Driven Design introduces ubiquitous language, bounded contexts, and tactical patterns. Designing Data-Intensive Applications covers reliability, consistency, replication, and schema evolution. Code Complete addresses routine design, variables, control flow, and defensive programming. The Pragmatic Programmer emphasizes DRY at the knowledge level, orthogonality, and automation. Release It! focuses on circuit breakers, bulkheads, backpressure, and observability. Working Effectively with Legacy Code provides characterization tests, seams, and dependency breaking. Patterns of Enterprise Application Architecture catalogs layers, repositories, unit of work, and DTOs. Implementing DDD shows aggregates, domain events, and context integrations. DDD Distilled offers a lighter introduction to subdomains and context mapping. A Philosophy of Software Design fights complexity through deep modules and information hiding. Refactoring.Guru provides a practical smell catalog and treatment guide.

**3 Version Tiers:**
The Full version contains complete traceability back to the book's structure and bias. The Mini version contains enough decision pressure to change implementation choices without bringing in the full source. The Nano version provides the smallest reminder of a book's bias for extremely tight always-on budgets.

**5 Delivery Patterns:**
Skills activate for a kind of work rather than every message. Always-on rules shape most tasks. Scoped rules apply to specific paths. On-demand rules are invoked explicitly. Retrieval patterns use MCP or RAG for large or dynamic material.

**3 Editor Targets:**
Codex supports AGENTS.md, AGENTS.override.md, .codex/config.toml, skills, hooks, and MCP. Claude Code supports CLAUDE.md, .claude/rules/, .claude/skills/, subagents, and MCP. Cursor supports .cursor/rules/*.mdc with Always, Auto Attached, Agent Requested, and Manual rule types.

**Compatibility Matrix:**
The matrix contains 78 complementary pairs, 11 overlapping pairs, and 2 conflicting pairs. Complementary books can be combined as equal active guidance. Overlapping books apply similar pressure, so choose one. Conflicting books should not be loaded together.

**Validation and Documentation:**
The project includes an experimental validation comparing mini rules versus no rules, achieving a 74/100 versus 46/100 score. Traceability files map rules back to book chapters. Usage, compatibility, and adding-a-book guides help contributors extend the collection.

> **Amazing:** The repository contains over 972 commits and 14 complete rule sets with traceability, making it one of the most comprehensive open-source collections of structured AI coding agent guidance available today.

## How It Works

Agent Rules Books works by translating the principles, heuristics, and decision frameworks from classic software engineering literature into structured Markdown rule sets. Each rule set contains decision rules, trigger rules, and a final checklist. Decision rules guide the agent's choices during implementation. Trigger rules activate when specific code patterns or situations are detected. The final checklist ensures the agent verifies its work before completing a task.

For example, the Clean Code mini rule set instructs agents to treat cleanliness as part of delivery, write for local reasoning, use precise names, keep functions small and focused, separate commands from queries, and expose behavior rather than raw representation. The trigger rules tell the agent to split phases when a function mixes setup, validation, and computation; to simplify names before keeping explanatory comments; and to separate responsibilities when a function both mutates and answers.

The compatibility matrix helps teams compose rule sets intelligently. Loading complementary books like Clean Architecture and Release It! together strengthens guidance across different layers. Avoiding conflicting pairs like Domain-Driven Design and Patterns of Enterprise Application Architecture prevents contradictory advice.

> **Important:** These rules are inspired by the books but are not official materials from the authors or publishers. They are practical engineering instructions written for AI coding tools and should be used as lightweight working agreements, not as substitutes for reading the books.

## Installation

Agent Rules Books is a documentation repository, so installation is simply cloning and copying the desired rule sets into your project.

```bash
# Clone the repository
git clone https://github.com/ciembor/agent-rules-books.git
cd agent-rules-books

# Copy a mini rule set into your project as AGENTS.md
cp clean-code/clean-code.mini.md /path/to/your/project/AGENTS.md

# Or copy a skill into your Claude Code skills directory
cp clean-code/clean-code.mini.md /path/to/your/project/.claude/skills/clean-code/SKILL.md

# Or copy into Cursor rules
mkdir -p /path/to/your/project/.cursor/rules
cp clean-code/clean-code.mini.md /path/to/your/project/.cursor/rules/clean-code.mdc
```

For Codex, place the rule set at the repo root as AGENTS.md or reference it via model_instructions_file in .codex/config.toml. For Claude Code, import it from CLAUDE.md using @AGENTS.md or place it in .claude/skills/. For Cursor, translate the content into .cursor/rules/*.mdc files with appropriate rule types.

## Usage

The recommended workflow is to start with one primary Mini rule set and expand based on task needs.

**Step 1: Choose a primary rule set.**
For everyday coding and review, start with Clean Code. For architectural decisions, use Clean Architecture. For refactoring tasks, use Refactoring. For domain modeling, use Domain-Driven Design. For production reliability, use Release It!.

**Step 2: Select a delivery pattern.**
Use skills for task-specific workflows like refactoring passes or legacy changes. Use always-on rules for stable repo-wide defaults. Use scoped rules for subsystem-specific guidance. Use on-demand rules for one-off deep sessions.

**Step 3: Verify compatibility before combining.**
Check the compatibility matrix before loading multiple books. Prefer complementary pairs and avoid conflicting ones. When in doubt, keep one primary always-on rule set and move others to on-demand mechanisms.

**Step 4: Iterate and refine.**
Start small, observe how the rules affect agent output, and adjust the rule set or delivery pattern based on results. The project's validation experiment suggests that structured rules measurably improve architectural judgment, module depth, and responsibility boundaries.

## Features

| Feature | Description |
|---------|-------------|
| 14 Book Rule Sets | Complete rule sets distilled from classic programming literature |
| 3 Version Tiers | Full, Mini, and Nano versions for different context budgets |
| 5 Delivery Patterns | Skills, always-on, scoped, on-demand, and retrieval patterns |
| 3 Editor Targets | Native support for Codex, Claude Code, and Cursor |
| Compatibility Matrix | 91 book pair comparisons with complementary, overlapping, and conflicting verdicts |
| Traceability Files | Map rules back to source book chapters and sections |
| Experimental Validation | Quantitative evidence that structured rules outperform vague references |
| MIT License | Free to use, modify, and distribute |

## Conclusion

Agent Rules Books represents a significant step forward in making AI coding agents more reliable, consistent, and aligned with decades of software engineering best practices. By distilling 14 classic books into structured, context-aware rule sets, the project gives developers a practical way to elevate AI-generated code quality without manually prompting for every principle. The tiered versions, multiple delivery patterns, and comprehensive compatibility matrix make it adaptable to teams of any size and editors of any preference. If you are building with AI coding agents and want your codebase to reflect the wisdom of Clean Code, Clean Architecture, Domain-Driven Design, and more, Agent Rules Books is an essential addition to your toolkit.

**Links:**
- GitHub Repository: [https://github.com/ciembor/agent-rules-books](https://github.com/ciembor/agent-rules-books)
- Usage Guide: [docs/USAGE.md](https://github.com/ciembor/agent-rules-books/blob/main/docs/USAGE.md)
- Compatibility Matrix: [docs/COMPATIBILITY.md](https://github.com/ciembor/agent-rules-books/blob/main/docs/COMPATIBILITY.md)
