---
layout: post
title: "session-knowledge: Turning Claude Code History Into a Searchable Knowledge Base - Inside nameforjt-afk/session-knowledge"
description: "session-knowledge indexes every Claude Code session transcript on your machine into local SQLite databases and exposes 13 MCP tools so any new session can search past decisions, commands, implementations, and credentials. It is built entirely on the Python standard library with no external dependencies, and nothing ever leaves your machine. A source tour of nameforjt-afk/session-knowledge."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Session-Knowledge-Claude-Code-History-Knowledge-Base-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/session-knowledge/nameforjt-afk-session-knowledge-architecture.svg
tags:
  - Claude Code
  - MCP
  - Python
  - SQLite
categories: [AI, Open Source]
keywords: "session-knowledge, Claude Code, MCP server, session history search, SQLite FTS5, Python standard library, knowledge base, credential redaction, code capability tags, BM25 search, CJK bigram tokenization, local-first AI tools"
author: "PyShine"
---

If you have used Claude Code for more than a few weeks, you have felt the gap. Every session you have ever run is already on disk, under `~/.claude/projects/**/*.jsonl` — every decision, every command that worked, every connection string you pasted. But Claude Code's memory is scoped per project directory. Open a session somewhere else and none of that history is reachable. You re-ask questions you already answered, and you re-implement integrations you already wrote.

[nameforjt-afk/session-knowledge](https://github.com/nameforjt-afk/session-knowledge) closes that gap. It is a Python package called `sessionmcp` that performs a single pass over all of your session transcripts and writes them into three local SQLite databases: a full-text index, a credential registry, and a cross-project code index. It then registers a Model Context Protocol server that hands Claude thirteen tools over stdio, so any new session — in any directory — can search your entire history with plain questions. The whole thing is pure Python standard library: no third-party dependencies, no embeddings, no API calls, nothing leaves your machine.

The source is worth a tour because it is a masterclass in doing a lot with very little. Instead of an MCP SDK, there is a hand-written JSON-RPC loop in `sessionmcp/server.py`. Instead of a vector database, there is SQLite FTS5 with a custom CJK bigram pre-tokenizer in `sessionmcp/tokenize.py`. Instead of a secrets framework, there is a precise redaction layer in `sessionmcp/redact.py` that scrubs the index while deliberately preserving values in a permission-hardened vault. It is the kind of codebase you can read in one sitting and learn from for months.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/session-knowledge/nameforjt-afk-session-knowledge-overview-architecture.svg" alt="Architecture overview of the nameforjt-afk/session-knowledge repository" style="max-width:100%;height:auto;" />
</div>

*Overview of session-knowledge: a SessionStart hook drives incremental ingestion through the parser, everything is redacted before it reaches the FTS5 index, credentials take a separate hardened path into the vault, and both the MCP server and the CLI share one retrieval core.*

Reading the overview from left to right: the `refresh.sh` hook fires on every Claude Code start and triggers the CLI's incremental refresh; the CLI (`sessionmcp/cli.py`) runs a single pass that feeds both `sessionmcp/parse.py` and the store writers. Parsing and redaction are deliberately intertwined — `sessionmcp/redact.py` scrubs secrets from everything that will be searchable while extracting assignments for the vault. The three SQLite stores sit in the middle: `sessionmcp/indexer.py` for transcript text and tool calls, `sessionmcp/vault.py` for credentials, `sessionmcp/codeindex.py` for code symbols and capability tags. On the right, `sessionmcp/query.py` is the retrieval core shared by `sessionmcp/server.py`, which exposes the thirteen MCP tools, and the same CLI you use by hand.

## Why You Need This

The first problem is the memory gap itself. When you ask "have I already built this?", the answer usually exists in a transcript from another project — but nothing in your current session can reach it. session-knowledge indexes all of your history once and keeps it fresh, so the question becomes a search instead of a faint memory.

The second problem is that grep cannot answer "has anyone implemented this integration?". Search-by-name only works if you already know the string to look for — you need to know it is `tenant_access_token` before you can grep for Feishu auth. The code index in `sessionmcp/codeindex.py` solves this with capability tags: files are tagged by the external service they touch (Feishu, Discord, Stripe, Postgres, and so on, defined in `CODE_CAPABILITY_PATTERNS` in `sessionmcp/config.py`), so you can search by service name rather than by guessing internal identifiers.

The third problem is decision archaeology. The reasoning behind a choice is buried inside thousands of past instructions, and when a long session gets auto-compacted, the compact summary is often the only surviving record of the earlier conversation. `sessionmcp/parse.py` handles this carefully — the `isCompactSummary` flag is checked before the noise filters run, precisely so these summaries are never mistaken for junk. Tools like `get_timeline`, `synthesize_topic`, and `track_evolution` then reconstruct how a topic unfolded, which sessions it spanned, and how a standard changed month by month.

The fourth problem is credential recall with honesty. The same variable name routinely has several real values — multiple apps, dev versus prod, two different tables. Picking one silently is how incidents happen. The vault in `sessionmcp/vault.py` returns every candidate for a name, ranked by explicit evidence (non-placeholder, `.env` current value over session snapshot, non-local address, project match, recency), with the call site each one came from. You decide; the tool never does.

## How It Works

One pass over the transcripts fans out into three specialized stores, each with its own write path and its own guarantees, and two interfaces read them back.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/session-knowledge/nameforjt-afk-session-knowledge-architecture.svg" alt="Detailed architecture of the nameforjt-afk/session-knowledge repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of session-knowledge: the installer wires the MCP server and SessionStart hook into Claude Code configuration; parsing, redaction, and tokenization feed three SQLite databases; the shared retrieval core serves both the MCP server and the CLI, with a full test suite and CI on top.*

### Understanding the Architecture

**The single-pass ingestion.** `sessionmcp/cli.py` walks every top-level transcript in `~/.claude/projects` plus subagent transcripts under each project's `subagents/` directory, and calls `parse_session` in `sessionmcp/parse.py` exactly once per file. That one pass produces three things at once: normalized chunks in five kinds (`user_instruction`, `compact_summary`, `assistant_text`, `tool_call`, `tool_error`), a table of tool calls with their most useful target extracted (the Bash command itself, file paths, Grep patterns, or MCP parameters), and raw credential assignments pulled from both tool inputs and tool results. Files already indexed are detected by mtime and size in `IndexWriter.needs_reindex`, so the daily cost of a refresh is a re-scan of the few transcripts that actually changed.

**The redaction boundary.** `sessionmcp/redact.py` serves two opposite needs with one pattern set. On the index side, `redact()` finds standalone secret literals — Discord bot tokens, JWTs, `sk-` API keys, GitHub personal access tokens, AWS access key IDs, Bearer tokens, Feishu webhooks, basic-auth URLs — plus values assigned to secret-classified variable names, and replaces each with a `⟦SECRET:fingerprint⟧` token derived from a SHA-1 prefix of the value. Identifiers like app IDs and table IDs are deliberately kept, because they are valuable search anchors. On the vault side, `extract_assignments()` records the same matches as full `KEY=VALUE` observations. Because both directions share the pattern definitions, nothing the index misses can silently appear in the vault either.

**The FTS5 index with a hand-rolled tokenizer.** `sessionmcp/indexer.py` stores chunks in a regular table plus a separate FTS5 virtual table, `chunks_fts`, built with the `unicode61` tokenizer. The docstring explains the trade-off precisely: an external-content table would require re-sending original content on delete, leaving orphaned index entries whenever the tokenization logic changed, while the separate table lets a single session be rebuilt with one `DELETE ... WHERE session_id=?`. On top of that, `sessionmcp/tokenize.py` expands every CJK run into adjacent character pairs (bigrams) at both write and query time, because FTS5's built-in trigram cannot match two-character Chinese words and `unicode61` does not segment CJK at all. Each query term becomes a quoted phrase so its bigrams must appear adjacent and in order, preserving exact-match semantics and BM25 ranking; single-character Chinese queries, which bigrams cannot express, fall back to a `LIKE` filter in `sessionmcp/query.py`.

**The credential vault.** `sessionmcp/vault.py` keeps plaintext values in a separate `vault.db` with mode 0600 — and, as its `connect()` docstring notes, it deliberately avoids WAL journaling because the `-wal`/`-shm` sidecars would escape the chmod and leave a plaintext copy beside the database. Raw observations are stored per source file in `session_credential_observations`, then aggregated into the `credentials` view; deleting a transcript removes its observations on the next refresh, while credentials still seen in live transcripts remain available. A `blocked` table implements `creds forget` as delete-plus-blacklist, because the index re-scans daily and a merely deleted fake value would come right back. The retrieval side is equally deliberate: `list_credentials` never returns values, `get_credential` returns all ranked candidates, and `lookup_fingerprint` resolves a redaction token back to a variable name — still without the value.

**The code index and its knowledge file.** `sessionmcp/codeindex.py` scans active project directories — auto-derived in `sessionmcp/config.py` from working directories with at least two sessions in the last forty-five days, capped at thirty, overridable via `SESSION_KNOWLEDGE_PROJECT_DIRS` — and parses `.py` files with the `ast` module for accurate signatures and class methods, while JS/TS files go through regexes because of the language's many declaration variants. It records symbols, imports, and capability tags into `code.db`. Beyond lookup, it computes a duplication report (cross-project symbol definitions, filtered of scaffolding noise) and runs `detect_forks`, which flags pairs of "projects" that are really two copies of one codebase — a distinction that changes the remedy from "extract a shared library" to "archive one copy". It also renders a human-readable `where-things-are.md` knowledge map, honoring a hand-edited `canonical.json` that the generator reads but never overwrites.

**The interfaces.** `sessionmcp/server.py` implements the MCP stdio protocol by hand — `initialize`, `ping`, `tools/list`, `tools/call` — supporting three protocol versions, with the docstring arguing that the protocol is too small to justify an SDK dependency. stdout carries only protocol messages; all logging goes to stderr. The thirteen tools are declared as one `TOOLS` list whose schemas and handlers live side by side. The CLI in `sessionmcp/cli.py` shares `sessionmcp/query.py` with the server so the two interfaces cannot drift, and `install.sh` wires everything into `~/.claude.json` and `settings.json` using the atomic writes of `sessionmcp/configio.py`, while `refresh.sh` — the SessionStart hook — writes nothing to stdout, never exits nonzero, guards against stale locks, and throttles the `.env` and code indexes to once a day.

The end-to-end flow is then easy to trace: a session starts, the hook asynchronously refreshes the index; meanwhile your question prompts Claude to call `search_sessions`, which tokenizes the query, runs it against `chunks_fts` under BM25, merges duplicate hits from resumed sessions (counting the copies rather than hiding them), downweights subagent transcripts, and returns bounded snippets; if a `⟦SECRET:fp⟧` token appears in the snippet, `lookup_secret` names the variable, and only an explicit `get_credential` call ever brings a real value into context.

## Advantages

- **Genuinely zero dependencies.** The entire package imports nothing outside the Python standard library — no MCP SDK, no embedding service, no network calls — so a plain Python 3.10+ interpreter with FTS5-enabled SQLite is the only requirement, verified by `install.sh` before it touches anything.
- **Privacy engineered, not promised.** Secrets are redacted before they are ever written to the searchable index; the vault is mode 0600 with WAL disabled to avoid unprotected sidecars; every database artifact is tightened by `sessionmcp/dbio.py`; and `verify-redaction` scans the whole index and must report zero leaks.
- **Search that answers the real question.** Capability tags let you find "who implemented Stripe auth" without knowing a single function name, which is exactly the query grep cannot express.
- **Honest, evidence-ranked results.** Duplicate content from resumed sessions is merged with the copy count shown, subagent hits are downweighted rather than dropped, and credential candidates are all returned with their provenance instead of one being silently chosen.
- **CJK-aware full-text search.** The bigram pre-tokenizer gives exact matching for two-character Chinese terms on the same FTS5 index, keeping BM25 ranking with no fallback path for the common case.
- **Incremental by design.** mtime-and-size skipping, deletion pruning, daily throttling of slow indexes, and a lock-guarded async hook keep the index fresh in seconds without blocking your session.

## Benefits

- **Stop re-asking answered questions.** Decisions, conclusions, and rationale from every past session become one query away, in any project directory.
- **Recall the exact command that worked — or the one that failed.** `find_tool_call` with `errors_only` turns your history of failed commands into a trap-avoidance record.
- **Reuse before you rewrite.** Calling `find_implementation` before writing a new integration surfaces existing code by symbol or by service tag, and `list_duplication` doubles as a ready-made work list for extracting shared libraries.
- **Context budget respected at the source.** Retrieval interfaces return fixed-width snippets and paged transcripts, never whole documents, so a search cannot flood the model's context window.
- **Trustworthy credential bookkeeping.** The variable-name registry, fingerprint lookup, placeholder detection, and permanent blacklisting of fake values together keep the candidate list something you can actually act on.
- **A reversible, idempotent install.** Re-running `install.sh` is safe, the hook deduplicates itself, and `uninstall.sh` (with `--purge` for the index) removes everything cleanly.

## Usage

Install is two shell commands followed by a restart of Claude Code, as the README describes:

```bash
git clone https://github.com/nameforjt-afk/session-knowledge.git
cd session-knowledge
bash install.sh
```

After that, day-to-day use is just asking questions normally — a SessionStart hook refreshes the index incrementally on each Claude Code start. The CLI mirrors every capability directly:

```bash
python3 -m sessionmcp.cli index                     # incremental refresh
python3 -m sessionmcp.cli stats                     # index health check

python3 -m sessionmcp.cli search "deploy timeout"   # full text (multi-word = AND)
python3 -m sessionmcp.cli search "pricing" --kind user_instruction
python3 -m sessionmcp.cli tool "docker build"       # past commands and API calls
python3 -m sessionmcp.cli tool "stripe" --errors    # only the ones that failed
python3 -m sessionmcp.cli timeline "that migration" # reconstruct how it unfolded
python3 -m sessionmcp.cli synth "rate limiting"     # cross-session material pack
python3 -m sessionmcp.cli evolution "auth design"   # how a decision changed over time

python3 -m sessionmcp.cli code find --capability stripe   # who implemented payment auth
python3 -m sessionmcp.cli code find send_message          # by symbol name
python3 -m sessionmcp.cli code dup                        # duplicate-implementation report

python3 -m sessionmcp.cli creds list                 # variable names only, never values
python3 -m sessionmcp.cli creds get DATABASE_URL     # all candidates, ranked
python3 -m sessionmcp.cli verify-redaction           # confirm no plaintext in the index
```

When you uninstall, you choose how much to keep:

```bash
bash uninstall.sh            # remove MCP + hook, keep the index
bash uninstall.sh --purge    # remove the index too
```

## Conclusion

session-knowledge is a narrow tool aimed at a real gap, and its source shows what disciplined engineering looks like when the answer to every "should we add a dependency?" question is no. A hand-written JSON-RPC server, a custom FTS5 tokenizer for CJK text, a redaction layer that shares one pattern set between safety and utility, three purpose-shaped SQLite stores, and a test suite that covers the retention and permission semantics — all in a package you can read start to finish in an afternoon. If you live in Claude Code and your history has outgrown any single project directory, this repository is both a useful tool today and a reference implementation of local-first MCP server design.

One warning the project itself is refreshingly upfront about: the index directory contains plaintext credentials scraped from your sessions — that is the precondition for "which key did I use here" to work at all. Never commit it, never sync it, and run `verify-redaction` now and then.

Links:

- GitHub repository: [nameforjt-afk/session-knowledge](https://github.com/nameforjt-afk/session-knowledge)
- Documentation: the repository README and `README.zh-CN.md`, plus `CHANGELOG.md` and `SECURITY.md` in the repo root
