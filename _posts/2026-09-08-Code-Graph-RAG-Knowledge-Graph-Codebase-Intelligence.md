---
layout: post
title: "Code-Graph-RAG: Graph-Based Retrieval-Augmented Generation for Multi-Language Codebase Intelligence"
description: "Code-Graph-RAG parses a multi-language codebase with Tree-sitter, builds a knowledge graph of its structure in Memgraph, and lets you query, edit, and optimise that code in plain English. It works across a monorepo of mixed languages under one unified graph schema. The system has two components: a Tree-sitter based parser that reads the codebase and ingests functions, classes, methods, modules, and their relationships into Memgraph, and a RAG system that turns natural language into Cypher queries, retrieves matching code, and drives AI-powered editing and optimisation. Fully supported languages include Python, TypeScript, TSX, JavaScript, Rust, Go, Java, C, C++, C#, PHP, Lua, and Dart. MIT-licensed, with an MCP server so Claude Code and other MCP clients can query and edit your codebase directly."
date: 2026-09-08
header-img: "img/post-bg.jpg"
permalink: /Code-Graph-RAG-Knowledge-Graph-Codebase-Intelligence/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Code-Graph-RAG
  - RAG
  - Knowledge Graph
  - Memgraph
  - Tree-sitter
  - Code Intelligence
  - MCP
  - Open Source
  - Python
author: PyShine
---

## What is Code-Graph-RAG

Code-Graph-RAG parses a multi-language codebase with Tree-sitter, builds a knowledge graph of its structure in Memgraph, and lets you query, edit, and optimise that code in plain English. It works across a monorepo of mixed languages under one unified graph schema. The code is on GitHub at [vitali87/code-graph-rag](https://github.com/vitali87/code-graph-rag), the package is on [PyPI](https://pypi.org/project/code-graph-rag/), and the website is at [code-graph-rag.com](https://code-graph-rag.com).

## How It Works

The system has two components:

1. **Multi-language parser.** A Tree-sitter based parser reads the codebase and ingests functions, classes, methods, modules, and their relationships into Memgraph under a single language-agnostic schema.
2. **RAG system** (`codebase_rag/`). An interactive CLI that turns natural language into Cypher queries, retrieves matching code, and drives AI-powered editing and optimisation.

```
Source Code -> Tree-sitter Parser -> AST Analysis -> Memgraph Knowledge Graph
                                                             |
User Query -> AI Model (Cypher Gen) -> Cypher Query -> Graph Results -> Response
```

![Code-Graph-RAG end-to-end architecture](/assets/img/diagrams/code-graph-rag/cgr-architecture.svg)

Point Code-Graph-RAG at a repository and it reads every source file, extracts functions, classes, methods, modules, and the relationships between them, and stores the result as an interconnected graph. Once the graph exists you can ask questions about the codebase in natural language and get answers grounded in the real structure, retrieve the actual source of any function, class, or method by name or by intent, edit code through the agent with AST-based surgical patching and a diff preview before anything changes, optimise code against language best practices or your own coding standards, find dead code by walking call and reference edges from entry points, search and rewrite structurally by AST pattern with ast-grep, and overlay runtime behaviour by tracing a test run or pulling production eBPF profiles with `cgr trace`.

## Agent Capabilities

Once the knowledge graph is built, the cgr agent (built on pydantic-ai) can query it and perform a range of code intelligence tasks.

![Code-Graph-RAG agent capabilities](/assets/img/diagrams/code-graph-rag/cgr-capabilities.svg)

- **Code Editing:** AST-based surgical patching with a diff preview before anything changes, plus verification.
- **Code Optimisation:** optimise code against language best practices or your own coding standards, with an interactive approval workflow.
- **Dead Code Detection:** find dead code by walking call and reference edges from entry points.
- **Structural Search:** search and rewrite structurally by AST pattern with ast-grep.
- **Dynamic Call Tracing:** trace a test run (or pull production eBPF profiles) with `cgr trace` and merge the calls that actually happened into the graph, exposing dispatch that static analysis cannot see.

## Supported Languages

Python, TypeScript, TSX, JavaScript, Rust, Go, Java, C, C++, C#, PHP, Lua, and Dart are fully supported. Scala is in development, and Ruby, Kotlin, Swift, Elixir, Haskell, Solidity, Bash, and Nix have structural support (modules, functions, classes where the language has them, and imports) through the pluggable ast-grep tier.

![Code-Graph-RAG language support matrix](/assets/img/diagrams/code-graph-rag/cgr-language-matrix.svg)

## Real-Time Updates and MCP Integration

Code-Graph-RAG does not require a full re-index every time a file changes. The realtime_updater uses watchdog to monitor filesystem events and re-parses only the changed files, incrementally upserting the results into the Memgraph graph. This means the graph stays in sync with the codebase as you edit.

The system also runs as an [MCP](https://modelcontextprotocol.io) server so Claude Code and other MCP clients can query and edit your codebase directly. The `cgr daemon up` command starts the packaged Memgraph + Qdrant stack with no compose file needed.

![Code-Graph-RAG real-time updates and MCP integration](/assets/img/diagrams/code-graph-rag/cgr-integration.svg)

## Installation

`cgr` is published to PyPI. Install it system-wide with the `treesitter-full` (all languages) and `semantic` (vector search) extras:

```bash
# with uv (recommended)
uv tool install "code-graph-rag[treesitter-full,semantic]"

# or with pipx
pipx install "code-graph-rag[treesitter-full,semantic]"
```

To run code newer than the latest release, install from git:

```bash
uv tool install "code-graph-rag[treesitter-full,semantic] @ git+https://github.com/vitali87/code-graph-rag@main"
```

You also need Python 3.12+, Docker (for Memgraph), `cmake`, and `ripgrep`.

## Quick Start

```bash
# Start the packaged Memgraph + Qdrant stack (no compose file needed)
cgr daemon up

# Parse a repository into the graph, then query it
cgr start --repo-path /path/to/repo --update-graph
cgr start --repo-path /path/to/repo
```

Repeat the first command for each repository you want indexed; the graph is shared, and syncing one project leaves the others alone. To start over from an empty graph, add `--clean` - it deletes every project in the shared graph, not just this one, and asks for confirmation first when other projects would be destroyed.

## Key Dependencies

| Dependency | Purpose |
|---|---|
| `tree-sitter` | Language-agnostic AST parsing |
| `pymgclient` | Memgraph database adapter |
| `pydantic-ai` | Agent framework for LLM integration |
| `pydantic-settings` | Settings management |
| `mcp` | Model Context Protocol SDK |
| `typer` | CLI framework |
| `rich` | Terminal rendering |
| `prompt-toolkit` | Interactive command line |
| `diff-match-patch` | Code patching |
| `watchdog` | Filesystem events monitoring |
| `huggingface-hub` | UniXcoder model download |

## Conclusion

Code-Graph-RAG is a pragmatic answer to the problem of codebase understanding in large, multi-language monorepos. By parsing with Tree-sitter into a Memgraph knowledge graph and layering a natural-language RAG agent on top, it lets you query, edit, optimise, and trace code grounded in the real structure rather than in embeddings alone. The MCP server integration means Claude Code and other MCP clients can use the graph directly, and the real-time updater keeps the graph in sync as files change. The source is on GitHub at [vitali87/code-graph-rag](https://github.com/vitali87/code-graph-rag), the package is on [PyPI](https://pypi.org/project/code-graph-rag/), and the documentation covers installation, quick start, CLI reference, architecture, graph schema, language support, and the Python SDK.
