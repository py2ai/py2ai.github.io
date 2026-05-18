---
layout: post
title: "Book-to-Skill: Convert Book PDFs into Claude Code Skills"
date: 2026-05-19 00:00:00 +0800
categories: [ai, claude-code, productivity]
tags: [book-to-skill, claude-code, pdf, skill, knowledge-extraction, ai]
seo:
  title: "Book-to-Skill - Convert Book PDFs to Claude Code Skills | PyShine"
  description: "Book-to-Skill converts book PDFs into Claude Code skills, extracting knowledge from books and turning them into actionable AI agent instructions."
  keywords: "book to skill, claude code, pdf, skill extraction, knowledge, ai agent"
featured-img: ai-coding-frameworks/ai-coding-frameworks
permalink: /Book-to-Skill-Convert-Book-PDFs-to-Claude-Code-Skills/
---

You buy a great technical book. You read it once. Three months later, you can't remember what chapter 7 even covered. Searching the PDF gives you a list of pages, not answers. Asking Claude about the book leads to hallucinations or "I don't have that content." Taking notes produces a 200-line document you never open again.

**Book-to-Skill** solves this by turning any technical book — PDF or EPUB — into a structured Claude Code skill that loads on demand. Instead of dumping 200K tokens into every conversation, only the chapters relevant to your question load. The rest stays on disk until you need it.

![Architecture](/assets/img/diagrams/book-to-skill/book-to-skill-architecture.svg)

## How It Works

Book-to-Skill follows a 10-step pipeline that transforms raw book content into a structured, queryable skill:

1. **Validate Input** — Confirms the file is a valid PDF or EPUB
2. **Identify Book Type** — Asks whether the book is technical (code, tables, formulas) or text-heavy (prose), then selects the optimal extraction tool
3. **Extract Text** — Runs `extract.py` with the appropriate tool chain: Docling for technical books (preserves tables and code blocks as markdown), or pdftotext/PyPDF2/pdfminer for text-heavy books
4. **Cost Estimate** — Presents a pre-flight token cost estimate before any generation begins
5. **Analyze Structure** — Identifies title, author, chapters, table of contents, and core themes
6. **Determine Skill Name** — Proposes a slug based on author-concept or book title
7. **Generate Chapter Summaries** — Creates dense 800–1,200 token summaries per chapter, loaded on-demand
8. **Generate Supporting Files** — Builds glossary, patterns, and cheatsheet references
9. **Generate Master SKILL.md** — Creates the core skill file with frameworks, mental models, and chapter index (under 4,000 tokens)
10. **Cleanup** — Removes temporary files and reports the final skill structure

> The key insight: a 1,000-token summary beats a 10,000-token excerpt. Book-to-Skill extracts structure, not summaries — capturing named frameworks, exact formulations, and anti-patterns the author crystallized.

## Key Features

![Features](/assets/img/diagrams/book-to-skill/book-to-skill-features.svg)

- **Multi-Format Input** — Supports both PDF and EPUB with automatic format detection and magic-byte fallback
- **Smart Extraction Pipeline** — Docling for technical books (tables, code blocks, formulas preserved as markdown at ~1.5s/page), pdftotext for text-heavy books (instant), with PyPDF2 and pdfminer.six as fallbacks
- **Structured Skill Output** — Generates SKILL.md (core mental models + chapter index), per-chapter files, glossary.md, patterns.md, and cheatsheet.md
- **On-Demand Chapter Loading** — Chapter files only load when you ask about that topic, preserving your token budget
- **Cost-Aware Processing** — Pre-flight token estimate with price calculations before any generation begins; includes an analyze-only mode for previewing
- **Quality Principles** — Extracts structure not summaries, preserves author precision, front-loads the most important content, and never copies raw book text

## Getting Started

Install Book-to-Skill by copying this into your Claude Code session:

```
Install book-to-skill: https://raw.githubusercontent.com/virgiliojr94/book-to-skill/master/SKILL.md
```

Or install manually:

```bash
mkdir -p ~/.claude/skills/book-to-skill/scripts

curl -o ~/.claude/skills/book-to-skill/SKILL.md \
  https://raw.githubusercontent.com/virgiliojr94/book-to-skill/master/SKILL.md

curl -o ~/.claude/skills/book-to-skill/scripts/extract.py \
  https://raw.githubusercontent.com/virgiliojr94/book-to-skill/master/scripts/extract.py
```

Then convert any book:

```bash
# PDF - derive skill name from filename
/book-to-skill ~/Downloads/designing-data-intensive-applications.pdf

# EPUB - specify a custom slug
/book-to-skill ~/books/clean-code.epub clean-code

# Full path with explicit name
/book-to-skill /tmp/ddd-evans.pdf domain-driven-design
```

After the skill is created, use it like any other Claude Code skill:

```bash
/designing-data-intensive-apps                  # load core mental models
/designing-data-intensive-apps replication      # find and explain a topic
/designing-data-intensive-apps ch05             # dive into chapter 5
/designing-data-intensive-apps "what chapters do you have?"
```

### Requirements

At least one extraction tool must be installed:

**For PDF:**

| Book Type | Tool | Install | Speed |
|-----------|------|---------|-------|
| Text-heavy | `pdftotext` (poppler) | `sudo apt install poppler-utils` | Instant |
| Text-heavy fallback | `PyPDF2` | `pip3 install PyPDF2` | Instant |
| Text-heavy fallback | `pdfminer.six` | `pip3 install pdfminer.six` | Instant |
| Technical | `docling` | `pip3 install docling` | ~1.5s/page |

**For EPUB:**

| Tool | Install | Quality |
|------|---------|---------|
| `ebooklib` + `beautifulsoup4` | `pip3 install ebooklib beautifulsoup4` | Best |
| stdlib `zipfile` | Built-in | Always available |

## Why Book-to-Skill Matters

The fundamental problem Book-to-Skill addresses is the gap between *reading* and *using* knowledge. Traditional approaches fail in predictable ways:

- **Raw PDF injection** burns your entire token budget upfront — a 400-page book is ~200K tokens loaded into every conversation
- **RAG (Retrieval-Augmented Generation)** finds chunks similar to your query, but misses the author's structured frameworks and named methodologies
- **General Claude knowledge** about popular books is compressed, averaged across internet discussions, and prone to hallucination

> RAG answers: "here are chunks close to your query." A skill answers: "here are the 12 frameworks this author built, ready to reason with."

Book-to-Skill works at *compile time* — one deep analysis run extracts the author's actual frameworks, names them, describes when to use each, and captures the anti-patterns. The output is structure the author spent years building, not a similarity search over their sentences.

For searching across 50+ books, RAG wins. For going deep on one book and using its frameworks while you work, a skill wins. It's less "library search" and more "the author is sitting next to you while you work."

## Conclusion

Book-to-Skill bridges the gap between passive reading and active application. By converting books into Claude Code skills with on-demand chapter loading, structured frameworks, and token-efficient summaries, it transforms how developers interact with technical knowledge. Instead of books gathering digital dust on your hard drive, their frameworks become living tools embedded in your daily workflow. With 566 stars and growing, Book-to-Skill represents a compelling approach to making book knowledge actionable in the age of AI-assisted development.

Check out the [Book-to-Skill repository](https://github.com/virgiliojr94/book-to-skill) to get started.