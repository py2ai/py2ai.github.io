---
layout: post
title: "Karpathy's LLM Wiki: Build a Compounding Knowledge Base With Your AI Agent"
description: "Learn how Andrej Karpathy's LLM Wiki pattern replaces traditional RAG with a persistent, compounding knowledge base where your LLM agent incrementally builds and maintains a structured wiki from your sources."
date: 2026-04-19
header-img: "img/post-bg.jpg"
permalink: /Karpathy-LLM-Wiki-Compounding-Knowledge-Base/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - AI Agent
  - Knowledge Management
  - Tutorial
author: "PyShine"
---

# Karpathy's LLM Wiki: Build a Compounding Knowledge Base With Your AI Agent

If you have ever used NotebookLM, ChatGPT file uploads, or any RAG system, you have experienced the same limitation: every query starts from scratch. You upload documents, the LLM retrieves relevant chunks, and it generates an answer. But nothing accumulates. Ask a subtle question that requires synthesizing five documents, and the LLM has to find and piece together the relevant fragments every single time. The knowledge does not compound.

**Andrej Karpathy**, former Tesla AI director and OpenAI founding member, has published a pattern he calls **LLM Wiki** -- a fundamentally different approach where your LLM agent incrementally builds and maintains a persistent, structured wiki from your sources. Instead of retrieving from raw documents at query time, the LLM compiles knowledge once and keeps it current. The wiki gets richer with every source you add and every question you ask.

The core insight is deceptively simple: **the wiki is a persistent, compounding artifact.** Cross-references are already there. Contradictions have already been flagged. The synthesis already reflects everything you have read. You never write the wiki yourself -- the LLM writes and maintains all of it. You curate sources, ask questions, and guide the analysis. The LLM does the rest.

![LLM Wiki Architecture](/assets/img/diagrams/llm-wiki/llm-wiki-architecture.svg)

### Understanding the Architecture

The architecture diagram above illustrates the three-layer system that powers the LLM Wiki pattern. Let us walk through each layer:

**Raw Sources (Top Row)**

Your curated collection of source documents forms the foundation. Articles, papers, images, data files, podcasts, and books all flow into the system. These sources are immutable -- the LLM reads from them but never modifies them. This is your source of truth. Whether you clip a web article with the Obsidian Web Clipper, download a research paper, or transcribe a podcast, the raw source stays pristine.

**The Wiki (Center)**

The wiki is a directory of LLM-generated markdown files. Summary pages, entity pages, concept pages, comparison pages, and a synthesis page all live here. The LLM owns this layer entirely. It creates pages, updates them when new sources arrive, maintains cross-references, and keeps everything consistent. A single source might touch 10-15 wiki pages as the LLM updates entity pages, revises topic summaries, and notes where new data contradicts old claims. You read the wiki; the LLM writes it.

**The Schema (Bottom Left)**

The schema is a document (such as CLAUDE.md for Claude Code or AGENTS.md for Codex) that tells the LLM how the wiki is structured, what the conventions are, and what workflows to follow when ingesting sources, answering questions, or maintaining the wiki. This is the key configuration file -- it is what makes the LLM a disciplined wiki maintainer rather than a generic chatbot. You and the LLM co-evolve this over time as you figure out what works for your domain.

**The LLM Agent (Center)**

The LLM agent sits between the raw sources and the wiki. It reads sources, extracts key information, integrates it into the existing wiki, updates entity pages, revises topic summaries, and flags contradictions. The human guides the process by curating sources, directing analysis, and asking questions. The LLM does all the grunt work -- the summarizing, cross-referencing, filing, and bookkeeping that makes a knowledge base actually useful over time.

## The Problem With Traditional RAG

Most people's experience with LLMs and documents looks like this: you upload a collection of files, the LLM retrieves relevant chunks at query time, and generates an answer. This works for simple lookups, but it breaks down for anything that requires synthesis across multiple sources.

The fundamental issue is that **there is no accumulation**. Every query is a fresh discovery. The LLM has to re-derive knowledge from scratch each time. If you ask a question that requires connecting information from five different documents, the LLM has to find and piece together all five fragments again. Nothing is built up.

NotebookLM, ChatGPT file uploads, and most RAG systems all share this limitation. They are stateless retrieval engines, not knowledge compilers.

![Compounding vs Traditional RAG](/assets/img/diagrams/llm-wiki/llm-wiki-compounding.svg)

### Understanding the Compounding Diagram

The compounding diagram contrasts the two approaches side by side:

**Traditional RAG (Top Path)**

The traditional RAG workflow is linear and disposable. You upload documents, the system retrieves chunks, it generates an answer, and then it discards everything. The next query starts from zero again. Each question requires re-discovering the same connections, re-synthesizing the same information, and re-deriving the same conclusions. The effort does not compound. It is like rewriting the same essay from scratch every time someone asks a question.

**LLM Wiki (Bottom Path)**

The LLM Wiki workflow is circular and compounding. You add a source, the LLM ingests and integrates it into the existing wiki, the wiki pages become cross-referenced and contradiction-flagged, and the whole knowledge base gets richer. The next source you add benefits from all the context that came before. The dashed arrow labeled "more context" shows the self-reinforcing loop: as the wiki grows, the LLM has more context to draw on when ingesting new sources, which produces better integrations, which makes the wiki even richer.

This is the key difference. In traditional RAG, the hundredth source is processed with the same empty context as the first. In the LLM Wiki pattern, the hundredth source is processed against the accumulated understanding of the previous ninety-nine. Knowledge compounds.

## Three Operations: Ingest, Query, Lint

The LLM Wiki pattern defines three core operations that keep the knowledge base alive and growing.

![Operations Diagram](/assets/img/diagrams/llm-wiki/llm-wiki-operations.svg)

### Understanding the Operations

The operations diagram shows the three core workflows that maintain and grow the wiki:

**Ingest (Left Column)**

When you drop a new source into the raw collection, the LLM reads it, extracts key information, and integrates it into the existing wiki. A single source might touch 10-15 wiki pages: updating entity pages, revising topic summaries, noting contradictions with earlier sources, and adding new cross-references. The LLM also updates `index.md` (the content catalog) and appends an entry to `log.md` (the chronological record). Karpathy recommends ingesting sources one at a time and staying involved -- reading the summaries, checking the updates, and guiding the LLM on what to emphasize. But you can also batch-ingest many sources at once with less supervision.

**Query (Center Column)**

When you ask a question against the wiki, the LLM searches `index.md` first to find relevant pages, then drills into them to synthesize an answer with citations. The critical insight is that **good answers can be filed back into the wiki as new pages.** A comparison you asked for, an analysis, a connection you discovered -- these are valuable and should not disappear into chat history. This way your explorations compound in the knowledge base just like ingested sources do. The dashed arrow from "File Answer as New Page" back to the wiki pages shows this feedback loop.

**Lint (Right Column)**

Periodically, you ask the LLM to health-check the wiki. It looks for contradictions between pages, stale claims that newer sources have superseded, orphan pages with no inbound links, important concepts mentioned but lacking their own page, missing cross-references, and data gaps that could be filled with a web search. The LLM is good at suggesting new questions to investigate and new sources to look for. This keeps the wiki healthy as it grows. The dashed arrows from Lint back to the wiki pages show how linting feeds back into maintaining the knowledge base.

## The Human + LLM Workflow

Karpathy describes his workflow like this: "I have the LLM agent open on one side and Obsidian open on the other. The LLM makes edits based on our conversation, and I browse the results in real time -- following links, checking the graph view, reading the updated pages. Obsidian is the IDE; the LLM is the programmer; the wiki is the codebase."

![Workflow Diagram](/assets/img/diagrams/llm-wiki/llm-wiki-workflow.svg)

### Understanding the Workflow

The workflow diagram shows how the human and LLM collaborate through the wiki:

**Human (Top Left)**

The human plays four roles: curating sources (finding and selecting the best material), exploring and asking questions (directing the LLM's analysis), guiding the analysis (telling the LLM what to emphasize), and reviewing wiki updates (reading the results in Obsidian and providing feedback). The human never writes wiki content directly -- that is the LLM's job.

**LLM Agent (Center)**

The LLM agent performs four types of work: summarizing sources (extracting key information from raw documents), cross-referencing and updating pages (connecting new information to existing knowledge), flagging contradictions (noting where new data challenges old claims), and linting/health-checking (keeping the wiki consistent and complete). All of this flows into the wiki directory.

**Wiki Directory (Bottom Center)**

The wiki is just a directory of markdown files with a git repository. You get version history, branching, and collaboration for free. The LLM writes to this directory; the human reads from it through Obsidian.

**Obsidian (Bottom Right)**

Obsidian serves as the reading interface. Its graph view shows what is connected to what, which pages are hubs, and which are orphans. The Dataview plugin can run queries over page frontmatter. The Marp plugin can generate slide decks from wiki content. The Obsidian Web Clipper browser extension converts web articles to markdown for quick source ingestion. The key insight at the bottom of the diagram captures the metaphor perfectly: "Obsidian = IDE, LLM = Programmer, Wiki = Codebase."

## Indexing and Logging

Two special files help the LLM (and you) navigate the wiki as it grows:

**index.md** is content-oriented. It is a catalog of everything in the wiki -- each page listed with a link, a one-line summary, and optionally metadata like date or source count. Organized by category (entities, concepts, sources, etc.), the LLM updates it on every ingest. When answering a query, the LLM reads the index first to find relevant pages, then drills into them. This works surprisingly well at moderate scale (around 100 sources and hundreds of pages) and avoids the need for embedding-based RAG infrastructure.

**log.md** is chronological. It is an append-only record of what happened and when -- ingests, queries, lint passes. If each entry starts with a consistent prefix (for example, `## [2026-04-02] ingest | Article Title`), the log becomes parseable with simple unix tools. Running `grep "^## \[" log.md | tail -5` gives you the last five entries. The log gives you a timeline of the wiki's evolution and helps the LLM understand what has been done recently.

## Use Cases

The LLM Wiki pattern applies to many contexts:

- **Personal**: Track your goals, health, psychology, and self-improvement by filing journal entries, articles, and podcast notes, building a structured picture of yourself over time.
- **Research**: Go deep on a topic over weeks or months by reading papers, articles, and reports, incrementally building a comprehensive wiki with an evolving thesis.
- **Reading a book**: File each chapter as you go, building out pages for characters, themes, and plot threads. By the end you have a rich companion wiki, like fan wikis such as Tolkien Gateway but built personally as you read.
- **Business/team**: Maintain an internal wiki fed by Slack threads, meeting transcripts, project documents, and customer calls, with humans in the loop reviewing updates.
- **Competitive analysis, due diligence, trip planning, course notes, hobby deep-dives** -- anything where you accumulate knowledge over time and want it organized rather than scattered.

## Optional Tooling

At some point you may want to build small tools that help the LLM operate on the wiki more efficiently. A search engine over the wiki pages is the most obvious one -- at small scale the index file is enough, but as the wiki grows you want proper search. [qmd](https://github.com/tobi/qmd) is a good option: it is a local search engine for markdown files with hybrid BM25/vector search and LLM re-ranking, all on-device. It has both a CLI (so the LLM can shell out to it) and an MCP server (so the LLM can use it as a native tool).

Other useful Obsidian integrations include the Web Clipper browser extension for quickly getting sources into your raw collection, local image downloading for offline reference, the graph view for visualizing connections, the Marp plugin for generating presentations from wiki content, and the Dataview plugin for running queries over page frontmatter.

## Why This Works

The tedious part of maintaining a knowledge base is not the reading or the thinking -- it is the bookkeeping. Updating cross-references, keeping summaries current, noting when new data contradicts old claims, maintaining consistency across dozens of pages. Humans abandon wikis because the maintenance burden grows faster than the value. LLMs do not get bored, do not forget to update a cross-reference, and can touch 15 files in one pass. The wiki stays maintained because the cost of maintenance is near zero.

The idea is related in spirit to Vannevar Bush's Memex (1945) -- a personal, curated knowledge store with associative trails between documents. Bush's vision was closer to this than to what the web became: private, actively curated, with the connections between documents as valuable as the documents themselves. The part he could not solve was who does the maintenance. The LLM handles that.

The human's job is to curate sources, direct the analysis, ask good questions, and think about what it all means. The LLM's job is everything else.

## Getting Started

The LLM Wiki pattern is intentionally abstract. It describes the idea, not a specific implementation. The exact directory structure, schema conventions, page formats, and tooling will depend on your domain, your preferences, and your LLM of choice. Everything is optional and modular -- pick what is useful, ignore what is not.

The right way to start is to copy Karpathy's [gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) to your LLM agent (Claude Code, Codex, OpenCode/Pi, or any other) and work together to instantiate a version that fits your needs. The document's only job is to communicate the pattern. Your LLM can figure out the rest.

The pattern works with any agent that can read and write files. Claude Code with its CLAUDE.md schema file, Codex with AGENTS.md, or any agent that supports custom instructions. The wiki is just a git repository of markdown files. You get version history, branching, and collaboration for free.