---
layout: post
title: "Hister: Your Own Private Search Engine for Everything You Have Read"
description: "Hister by asciimoo, the creator of Searx, is a self-hosted search engine that indexes the pages you visit, the files you keep, and the bookmarks you save. Go, AGPLv3, 4.9k stars on GitHub."
date: 2026-09-19
header-img: "img/post-bg.jpg"
permalink: /Hister-Your-Own-Private-Search-Engine/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/hister/hister-architecture.svg
tags: [self-hosted, search engine, privacy, open source, Hister]
author: "PyShine"
---

You have experienced this a hundred times. You read the perfect answer to a problem last week, on some forum, in some doc, behind some tab you closed. Today the problem is back and the page is gone. Google will not find it for you because Google never indexed the comment thread you actually read, and your browser history is a landfill of URLs without content. Hister, the project lighting up GitHub's trending page right now with nearly five thousand stars, exists to end that loop: it is a private search engine that indexes the full text of the pages you visit, the files you keep, and the bookmarks you save.

The name behind it carries weight. Hister comes from asciimoo, the developer who created [Searx](https://github.com/asciimoo/searx), the meta search engine that powers countless privacy-conscious instances around the world. Where Searx protects you from search providers, Hister removes the need to lean on them in the first place for anything you have already seen. It is written in Go, ships as a single binary, and is licensed AGPLv3. There is a public [demo](https://demo.hister.org/) if you want to feel the interface before installing anything.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/hister/hister-architecture.svg" alt="Hister architecture: clients and surfaces, HTTP server layer, ingestion pipeline, search and storage, platform and config" style="max-width:100%;height:auto;" />
</div>

*The architecture map above is drawn from the real repository tree, so every box points at a real directory or file. Trace the arrows and you can follow a captured page from the browser extension all the way into the full-text index.*

## Capture everything, effortlessly

Hister's core trick is that indexing happens as a side effect of your ordinary habits. Install the browser extension for [Firefox](https://addons.mozilla.org/en-US/firefox/addon/hister/) or [Chrome](https://chromewebstore.google.com/detail/hister/cciilamhchpmbdnniabclekddabkifhb), and every page you visit is sent to your Hister server, where its full content is extracted and indexed. No tagging, no saving to a read-later list, no discipline required. The page you barely glanced at becomes searchable the moment you leave it.

For everything else there are deliberate paths in. The `hister import` command pulls in bookmarks and archives from the services self-hosters already run, including Raindrop, Linkwarden, Wallabag, Linkding, Karakeep, Readeck, and Shaarli. A crawler can index entire websites on demand, with a choice of a plain HTTP backend or a headless Chrome backend for JavaScript-heavy pages, and it respects robots.txt along the way. You can also point Hister at local directories and it will watch them, indexing PDFs, Word documents, and Markdown files as they change.

## What makes the search itself good

Under the hood sits [Bleve](https://github.com/blevesearch/bleve), a full-text indexing library for Go, and Hister leans on it well. The index is actually split per language, so a document in English and one in German do not fight over the same tokenization rules. The query language supports field filters, exact phrases, wildcards, negation, aliases, and result priorities, which puts it closer to a real search engine than to a naive substring match over a database.

There is also an optional semantic mode. If you configure an embeddings endpoint, Hister chunks your documents, embeds them, and stores the vectors in sqlite-vec or PostgreSQL. A search can then blend keyword and meaning matches, so "that article about db pool exhaustion" can surface a page that never uses those exact words. Crucially, this is opt-in and pointed at whatever endpoint you choose, which matters given the project's privacy stance: no telemetry, no mandatory cloud, your index stays on your server.

## Extractors: the secret weapon

Raw HTML is noisy, and generic parsing mangles the places people actually read. Hister's answer is a library of site-specific extractors, each one a small package that knows how to pull clean text and structure out of a particular platform. The registry covers Reddit, Hacker News, Mastodon, Bluesky, Twitter, GitHub, Stack Exchange, Lobsters, Wikipedia, Notion, and even shared ChatGPT conversations, with yt-dlp support for video subtitles. There is an SDK for writing your own. This is the same insight behind our earlier post on turning coding agents into [research agents](https://pyshine.com/OpenResearch-Turn-Coding-Agents-Into-Research-Agents/): the difference between mediocre and excellent retrieval is almost always the parsing layer.

## Search from anywhere, including your AI assistant

The server exposes one API and several ways to use it. The primary surface is a SvelteKit web interface compiled right into the binary, so `hister listen` is all it takes to get a full search app on port 4433. There is a slick terminal TUI for keyboard people, and the CLI covers crawling, importing, and maintenance.

The most forward-looking surface is the MCP endpoint. Hister implements the [Model Context Protocol](https://modelcontextprotocol.io) over Streamable HTTP and exposes three tools: search, get_preview, and get_history. That means Claude Desktop, Cursor, or any MCP-aware assistant can query your personal index directly, which turns it into private memory for your AI tools. We saw the same pattern in our write-up of [Octop](https://pyshine.com/Octop-Self-Hosted-Multi-Agent-AI-Assistant/): self-hosted software that exposes itself over MCP stops being a dashboard you visit and becomes a capability your assistant simply has. Hister even ships a trust marker on MCP results, warning the model that indexed content is untrusted source data, a small detail that shows the security thinking here is current.

Multi-user support comes standard, with per-user isolation of documents and results, session handling with CSRF protection, and optional OAuth login through GitHub, Google, or any OIDC provider. A Prometheus metrics endpoint covers anyone running it for a team. That team angle is worth pausing on: Hister started as a personal tool, but shared instances turn it into a searchable memory for a whole group, something asciimoo writes about on the project's [documentation site](https://hister.org/docs).

## Getting started

Grab a binary from the [releases page](https://github.com/asciimoo/hister/releases/latest), rename it to `hister`, and start the server:

```bash
./hister listen
```

Open `http://127.0.0.1:4433`, install the browser extension, and visit a page. Search for a phrase from that page and your first indexed result appears. Homebrew, Docker, and Nix installs are documented in the [quickstart](https://hister.org/docs/quickstart), and no configuration is required for a local setup.

The honest caveat is that Hister indexes what you feed it. It is not a proxy for the live web and it will not replace a general search engine for discovering things you have never seen. What it replaces is the slow erosion of "I know I read this somewhere," and that, for anyone who works with information for a living, is the search box that matters most. Given that this project pulls in hundreds of stars a day right now, a lot of people seem to agree.
