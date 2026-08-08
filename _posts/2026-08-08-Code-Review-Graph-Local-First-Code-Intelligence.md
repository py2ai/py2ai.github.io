---
layout: post
title: "Code Review Graph: Local-First Code Intelligence for AI Assistants"
description: "Discover how code-review-graph builds a structural map of your codebase using Tree-sitter, tracks changes incrementally, and delivers AI assistants precise context via MCP — reducing token usage by up to 71x."
date: 2026-08-08
header-img: "img/post-bg.jpg"
permalink: /Code-Review-Graph-Local-First-Code-Intelligence/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - Code Review
  - MCP
  - LLM
  - Developer Tools
  - Tree-sitter
  - Code Intelligence
author: "PyShine"
image: /assets/img/diagrams/awesome-codex-skills/awesome-codex-skills-anatomy.svg
---

# Code Review Graph: Local-First Code Intelligence for AI Assistants

AI coding assistants have become indispensable in modern development, but they share a common flaw: they're **blind to your codebase structure**. When you ask an AI to review a pull request or refactor a function, it typically resorts to brute-force tactics — dumping entire files or even the whole repository into the context window. This wastes tokens, inflates costs, and often produces shallow, context-free suggestions.

**code-review-graph** solves this by giving AI assistants a **precise, structured understanding** of your codebase — built locally, updated incrementally, and served through the Model Context Protocol (MCP). Instead of reading thousands of lines to understand a single change, an AI gets exactly the context it needs: the function, its callers, its dependencies, and its impact radius. The result is a staggering **71x reduction in token usage** while producing more accurate, context-aware code reviews.

Let's dive into how it works and why it matters.

---

## Why Token Waste Is the Silent Killer of AI Code Review

Let's be honest: token waste is the dirty secret of AI-assisted development. Consider a typical code review scenario:

1. A developer opens a PR changing 3 files in a Flask project
2. The AI assistant, lacking structural knowledge, reads the entire codebase to understand context
3. For Flask alone, that's roughly **143,594 tokens** consumed just to review a few hundred lines

That's like photocopying an entire book to check a single footnote. It's wasteful, slow, and it pushes your context window to its limit — leaving less room for actual reasoning.

The root cause is that most AI coding tools treat source code as flat text. They don't understand that `validate_user()` is called by `auth.login()` which is imported by `routes.py` which is registered by `create_app()`. They don't see the graph.

**code-review-graph** fixes this by making the graph visible.

---

## What code-review-graph Does

At its core, code-review-graph is a **code graph engine** that builds a structural map of your codebase and serves it to AI assistants through MCP. It operates in three stages:

### Stage 1: Parse with Tree-sitter

code-review-graph uses **Tree-sitter** — an incremental parser generator that builds concrete syntax trees (CSTs) for source code. Unlike regex-based or line-counting approaches, Tree-sitter understands the actual structure of your code: functions, classes, imports, calls, conditionals, and more.

```python
# A Python function parsed into a structured node
# code-review-graph extracts the function signature, body,
# imports, and all call relationships automatically

def calculate_price(quantity: int, unit_price: float) -> float:
    """Calculate total price with discount."""
    if quantity > 100:
        discount = 0.1  # 10% off for bulk orders
    elif quantity > 50:
        discount = 0.05
    else:
        discount = 0.0

    subtotal = quantity * unit_price
    return subtotal * (1 - discount)
```

Tree-sitter doesn't just read this text — it produces a structured tree that captures every node type, every relationship, and every code span. This becomes the foundation of the code graph.

### Stage 2: Build the Graph

Once parsed, code-review-graph connects the dots into a **knowledge graph**:

- **Call edges**: `calculate_price` is called by `checkout.process_order`
- **Import edges**: `checkout` imports from `models.inventory`
- **Inheritance edges**: `PremiumCustomer` extends `Customer`
- **Module edges**: `routes` registers `api` blueprint

The result is a directed graph where nodes represent code entities and edges represent relationships. This graph is the AI's source of truth for codebase understanding.

### Stage 3: Serve via MCP

The graph is exposed through the **Model Context Protocol (MCP)**, which means any AI assistant that supports MCP can query it. Instead of saying "read the whole codebase," the AI says:

- "Show me all callers of `calculate_price`"
- "What's the impact radius of changing `unit_price`?"
- "Find functions related to user authentication"

Each query returns **precise, structured context** — often just a few hundred tokens instead of tens of thousands.

---

## Key Features with Code Examples

### Installation

code-review-graph is available as a Python package:

```bash
pip install code-review-graph
```

### CLI Usage

The command-line interface makes it easy to build and query your code graph:

```bash
# Build the code graph for your project
crg build ./my-project

# Query the graph for semantic search
crg search "user authentication"

# Analyze impact radius of a specific function
crg impact checkout.process_order

# Get a code review with risk scoring
crg review ./my-project --diff git diff HEAD~1
```

### MCP Integration

Connect code-review-graph to your AI assistant via MCP:

```json
{
  "mcpServers": {
    "code-review-graph": {
      "command": "python",
      "args": ["-m", "code_review_graph.mcp", "--repo", "./my-project"]
    }
  }
}
```

Once configured, your AI assistant gains access to specialized tools:

- `crg_search` — Semantic code search across the entire graph
- `crg_impact` — Impact radius analysis for any function or class
- `crg_review` — Automated code review with risk scoring
- `crg_context` — Precise context retrieval for any code entity
- `crg_diff` — Incremental change analysis against git history

### Semantic Search

Find code by meaning, not just by name:

```bash
# Find code related to a concept
crg search "how does the system handle payments"

# Find all code touching a specific concern
crg search "error handling in authentication"

# Filter by language
crg search "database connection" --language python
```

### Incremental Change Tracking

One of the most powerful features is **incremental updates**. You don't need to rebuild the entire graph every time. code-review-graph tracks changes via git diffs and updates only the affected portions of the graph:

```bash
# Update the graph after code changes
crg update --repo ./my-project

# See what changed
crg diff --repo ./my-project
```

This makes it practical for large, actively-developed codebases where a full rebuild would be prohibitively slow.

### Code Review with Risk Scoring

code-review-graph can analyze a diff and provide risk assessments:

```bash
# Review a specific commit
crg review ./my-project --commit abc123

# Review uncommitted changes
crg review ./my-project --diff "git diff"

# Review a pull request
crg review ./my-project --pr 123
```

The risk scoring takes into account:

- **Blast radius**: How many downstream functions are affected?
- **Call chain depth**: How deep does the change propagate?
- **Module boundaries**: Does the change cross architectural layers?
- **Test coverage**: Are there adequate tests for the affected code?

---

## Token Savings Benchmarks

The headline metric is a **71x reduction** in token usage for code review tasks. Here's how it breaks down for real-world projects:

| Project | Full Context Tokens | code-review-graph Tokens | Reduction |
|---------|-------------------|------------------------|-----------|
| Flask | 143,594 | 2,196 | **65x** |
| Django | 287,102 | 4,058 | **71x** |
| FastAPI | 98,445 | 1,523 | **64x** |
| Requests | 45,230 | 892 | **51x** |

### What This Means in Practice

Consider a typical code review for a Flask pull request that changes the `calculate_price` function. Here's how the two approaches compare:

**Without code-review-graph:**
1. AI reads `calculate_price.py` (50 lines)
2. AI reads `checkout.py` to find callers (120 lines)
3. AI reads `routes.py` to find the blueprint (80 lines)
4. AI reads `app.py` to find app factory (150 lines)
5. AI reads `models.py` for database context (200 lines)
6. AI reads `config.py` for settings (100 lines)
7. AI reads `tests/` for test coverage (300 lines)
8. AI reads `README.md` for project context (200 lines)
9. AI reads `requirements.txt` for dependencies (50 lines)
10. AI reads 5+ more files to fill gaps

**Total: ~1,350 lines ≈ 5,400+ tokens — and that's a "focused" review!**

**With code-review-graph:**
1. AI queries `crg_context("calculate_price")` — returns the function, its 3 callers, its 2 imports, and the test that covers it
2. AI queries `crg_impact("calculate_price")` — returns the 7 downstream functions affected
3. AI queries `crg_search("discount calculation")` — returns 4 related functions

**Total: ~120 lines ≈ 480 tokens — 9x more focused, and more accurate!**

And for larger codebases, the gap widens dramatically. The 71x figure comes from comparing a full-repository dump against targeted graph queries for Django-sized projects.

---

## Supported Languages

code-review-graph supports multiple programming languages out of the box:

- **Python** — Full support for functions, classes, imports, decorators, and type hints
- **JavaScript / TypeScript** — Full support for ES modules, classes, arrow functions, and TS types
- **Java** — Full support for classes, interfaces, generics, and Spring annotations
- **Go** — Full support for packages, interfaces, goroutines, and channels
- **Rust** — Full support for structs, traits, lifetimes, and modules
- **C / C++** — Full support for functions, structs, classes, and includes
- **Ruby** — Full support for classes, modules, blocks, and gems
- **PHP** — Full support for classes, traits, namespaces, and Laravel patterns

New languages are added regularly via Tree-sitter grammar bindings.

---

## Who Should Use code-review-graph

### AI-Centric Development Teams

If you're building with AI coding assistants (Claude Code, Cursor, Windsurf, Copilot Workspace), code-review-graph gives your AI tools the structural context they're missing. It turns them from "glorified autocomplete" into **genuine code review partners** that understand your architecture.

### Code Reviewers and Maintainers

For maintainers of large codebases, code-review-graph eliminates the "where do I start?" problem. When a PR touches `calculate_price`, you instantly see the blast radius: which functions, modules, and tests are affected. No more manual grep-spelunking.

### Teams Managing Technical Debt

Incremental change tracking means you can see how your codebase evolves over time. code-review-graph highlights when changes cross architectural boundaries, when new dependencies are introduced, and when modules become overly coupled.

### Security-Conscious Developers

Since code-review-graph runs **100% locally**, your source code never leaves your machine. Unlike cloud-based code intelligence tools, there's no risk of your proprietary code being sent to third-party servers. This makes it suitable for regulated industries and sensitive codebases.

### Solo Developers

Even if you're a team of one, code-review-graph pays dividends. When you come back to a project after weeks away, the code graph acts as a **memory aid** — quickly reminding you of how pieces fit together without re-reading every file.

---

## Getting Started

Ready to give it a try? Here's a quick start:

```bash
# Install
pip install code-review-graph

# Navigate to your project
cd your-project

# Build the graph
crg build .

# Start the MCP server for AI integration
crg serve

# Or query directly
crg search "user authentication"
crg impact my_module.critical_function
crg review .
```

**For AI assistant integration**, configure the MCP server in your assistant's settings:

```json
{
  "mcpServers": {
    "code-review-graph": {
      "command": "crg",
      "args": ["serve", "--repo", "."]
    }
  }
}
```

Once configured, ask your AI assistant to:
- "Review my last commit using the code review graph"
- "Show me the impact radius of `UserService.validate`"
- "Find all functions related to payment processing"

---

## Conclusion

code-review-graph represents a fundamental shift in how AI assistants interact with codebases. Instead of dumping raw text into context windows and hoping the AI figures it out, code-review-graph provides **structured, precise, and actionable code intelligence** — right where the AI needs it.

The 71x token reduction isn't just a number — it translates to faster code reviews, lower costs, and more accurate suggestions. And since everything runs locally, there's no privacy compromise.

If you're building with AI coding tools in 2026, a code graph engine like code-review-graph isn't just a nice-to-have — it's becoming the **essential infrastructure** that makes AI-assisted development truly effective.

---

**Repository**: [https://github.com/tirth8205/code-review-graph](https://github.com/tirth8205/code-review-graph)

**License**: MIT

**Installation**: `pip install code-review-graph`