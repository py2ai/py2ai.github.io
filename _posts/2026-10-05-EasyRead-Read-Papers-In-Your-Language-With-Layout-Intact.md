---
layout: post
title: "EasyRead: Read Papers In Your Language With Layout Intact - Inside Edwardxlai/easyread"
description: "EasyRead translates academic PDFs page by page in the background while keeping equations, tables and layout intact, with margin-space Ask AI, highlights, notes and a local-first library. A source tour of its Python stdlib server, resumable translation queue and agent CLI."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /easyread-read-papers-in-your-language-with-layout-intact/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/easyread/edwardxlai-easyread-architecture.svg
tags: [AI Reading, Translation, PDF Tools, Open Source]
categories: [AI, Open Source]
keywords: EasyRead, paper translation, PDF reader, KaTeX, Claude Code, Codex CLI, Ollama, local-first
author: "PyShine"
---

Every researcher knows the drill: you find a paper that matters, it is forty pages of dense English with equations, two-column layouts and tables, and your reading speed drops to a crawl. Browser translators mangle the math, one-shot PDF translators destroy the layout, and cloud services want you to upload your library to someone else's server. EasyRead, an MIT-licensed project by Edwardxlai, takes a refreshingly grounded approach: drop in a PDF and it gets translated page by page in the background, with equations re-rendered in KaTeX, tables rebuilt as clean three-line tables, references left in the original, and the source PDF always one click away, following your reading position and boxing the paragraph you are on.

The feature set reads like a wish list assembled by someone who actually reads papers daily. A bilingual toggle shows the original under each paragraph. AI explanations live in the margin, strictly separated from the faithful translation of the main text, so you always know what the paper itself says. The Ask AI panel streams answers, accepts multi-paragraph quotes dragged into the input, understands references to your colored highlights, and supports multiple chats each bound to its own model: Claude Code, Codex, DeepSeek, Qwen, or a fully offline Ollama model. Four highlighter colors, underlines, notes and questions are collected in reading order and export to Markdown for Obsidian or Notion. Double-click any paragraph to edit the translation, change a term in the glossary and it is replaced everywhere. The library supports folders, pins, stars, reading progress, citation copying in GB/T 7714, APA and BibTeX, a restorable trash, and single-file offline HTML export for sharing.

Under the hood it is just as interesting. The backend is the Python standard library plus PDF libraries, with no frontend build step: the pages in `easyread/web/` are plain HTML, CSS and JavaScript you can edit and refresh. Translation runs as a resumable background job, split into parallel segments that each translate batches in order so terminology and context flow between batches. Everything is local: the server listens only on 127.0.0.1, papers live in ordinary folders named by the hash of the PDF, and you choose which model service, if any, sees your text.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/easyread/edwardxlai-easyread-overview-architecture.svg" alt="Architecture overview of the EasyRead repository" style="min-width:640px;width:100%;">
</div>

*Architecture overview of the EasyRead repository: the browser pages, the local server, the model backends and the per-paper library.*

Reading the overview from left to right: you open the reader page at `easyread/web/reader.html` or the library page at `easyread/web/library.html`, both of which talk to the local HTTP server in `easyread/server.py`. Import and translation requests land in the two task queues of `easyread/jobs.py`; whole-paper work flows into the batch translator at `easyread/translate.py`, which prepares pages with `easyread/pdfwork.py`, fills in metadata through `easyread/sources.py`, and runs model calls through the backends in `easyread/engines.py`. Every paper is one folder managed by `easyread/library.py`, whose state lives in JSON files written by `easyread/store.py`. A companion CLI at `easyread/cli.py`, documented for agents in `skill/paper-reading/SKILL.md`, lets Claude Code or Codex drive the whole workflow themselves.

## Why You Need This

- **Equations and tables survive.** Math is re-rendered with KaTeX and checked against the original page image, tables become clean three-line tables, and figures keep their place. You read typeset prose, not translator soup.
- **Faithful text, explained separately.** The main column is only the translation. AI commentary, answers and pinned notes go in the margin, so the paper's own claims are never blended with a model's interpretation.
- **Your subscription is the engine.** Claude Code and Codex run headlessly with your existing login, no API key needed; free-tier and local Ollama models work too. Translation costs whatever your current plan already allows.
- **Nothing is lost, ever.** Every batch is written to disk as it completes, so an interrupted job resumes where it stopped. Your paragraph edits are saved in the browser first and only cleared once the server confirms they are on disk.
- **Your library stays yours.** One folder per paper with the PDF, page images and JSON files inside; point the cloud library setting at any synced folder and your existing drive client handles backup.
- **Agents can read with you.** The bundled CLI and skill file let Claude Code or Codex import papers, watch progress, write margin answers next to your questions, and export results on their own.

## How It Works

EasyRead is a single Python package whose server hosts two browser pages and a small set of JSON APIs. There is no database and no build step: state is JSON on disk, pages are static files, and the heavy lifting happens in a background task system with two queues. The next diagram follows the code from the browser scripts down through the translation engine, the model backends, the PDF intake and the library storage.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/easyread/edwardxlai-easyread-architecture.svg" alt="Detailed architecture of the EasyRead codebase" style="min-width:720px;width:100%;">
</div>

*Detailed architecture of the EasyRead codebase, from the browser pages and server handlers to translation, backends and storage.*

### Understanding the Architecture

**The local server.** `easyread/server.py` is built on Python's `BaseHTTPRequestHandler` and binds only to 127.0.0.1. It serves the two HTML pages, injects interface translations at request time through `easyread/i18n.py`, streams live updates over the socket layer in `easyread/wsock.py`, and routes JSON APIs to dedicated modules: `easyread/library_api.py` for shelf operations, `easyread/translate_api.py` for imports and progress, and `easyread/settings_api.py` for model cards. Uploads are capped at 200 MB, enough for even the fattest journal PDFs.

**The library and the workspace.** `easyread/library.py` gives every paper its own directory keyed by the first twelve hex characters of the PDF's SHA-256, so the same paper imported twice is the same entry. Each workspace is a set of JSON documents managed by `easyread/store.py`: the paper's metadata and merged translation in `paper.json`, job state in `job.json`, plus notes and chats. Deleting a paper routes through `easyread/trash.py` so it can be restored, and `easyread/cloudlib.py` relocates the whole library to any folder you choose, which is how the cloud-library feature works with zero server code: your sync client does the syncing.

**Two task queues.** `easyread/jobs.py` separates whole-paper work from small tasks. Imports and full translations go on the bulk queue, which processes one paper at a time and persists state into `job.json` after every step, so a restart continues rather than restarts. Ask-AI answers and segment retranslations run on a separate small queue so you never wait for a forty-page job to finish just to get one answer. Cancellation is cooperative through per-job events, and the queue refuses to enqueue a paper that already has work queued or running.

**Segmented, resumable translation.** `easyread/translate.py` first prepares the PDF through `easyread/pdfwork.py`, which renders page images and extracts text. Translation then splits the pages into consecutive segments handled by `easyread/segments.py`: automatic mode opens up to four parallel lanes (manual up to eight), each lane translating batch after batch in order, with cuts placed where a page ends at a paragraph or heading boundary. Within a lane, each batch sees the glossary, section headings and the tail of the previous batch that were merged in before it, which is how terminology stays consistent across forty pages; the glossary machinery lives in `easyread/terms.py`. Prompts come from `easyread/prompts.py`. Every returned batch is validated by `easyread/checks.py` for malformed blocks and broken TeX before `easyread/figures.py` re-places figures and the result merges into `paper.json`. Failures retry, then are skipped and recorded for a one-click retry of just the failed pages; quota-style errors stop the job early instead of burning the rest of the batches against a dead key. Re-translating a paragraph you hand-edited produces a notice rather than a silent overwrite, guarded by `easyread/consistency.py`.

**Model backends.** `easyread/engines.py` speaks to three families of models. The `claude` engine runs a local Claude Code headlessly with your existing login and can look at the original page image to verify equations. The `codex` engine does the same through Codex CLI, sending page images as attachments, with the lean runner in `easyread/codex_lean.py`. The `openai` engine talks to any OpenAI-compatible endpoint, both Chat Completions and Responses formats, which covers Ollama and LM Studio locally plus DeepSeek, Zhipu, SiliconFlow, Gemini and friends remotely; the wire code is in `easyread/openai_api.py`. `easyread/usage.py` meters tokens for every call, and with a Claude subscription it also surfaces how much of the rolling quota is used and when it resets.

**Ask AI in the margin.** `easyread/chat.py` powers the streaming chat panel. Chats are stored per paper by `easyread/chat_store.py`, each bound to its own model, and the model registry that settings edits comes from `easyread/chat_models.py`. Selection-aware features fall out of the reader's highlight store: ask about the paragraphs you highlighted in red and the prompt includes exactly those.

**Reading view.** The reader page keeps the original PDF page image in a side panel that tracks scroll position and outlines the active paragraph, with cross-page alignment handled by `easyread/sentences.py` so a sentence split across a page boundary still highlights correctly on both sides.

**The agent door.** `easyread/cli.py` implements `easyread list, import, status, discuss, export, demo, merge` and more, and `skill/paper-reading/SKILL.md` teaches Claude Code or Codex to use them: an agent can import a paper, poll `easyread status` for your edits and open questions, then write answers into the margin with `easyread discuss` without touching the UI.

Follow one PDF end to end: the library page posts it to `easyread/translate_api.py`, which enqueues a bulk job in `easyread/jobs.py`; the translator prepares pages, resolves metadata from an arXiv ID or DOI through `easyread/sources.py` when the PDF came from the wild, splits work into lanes, and streams batches through the chosen engine with validation and figure handling on the way back; merges land in `paper.json`, the socket layer in `easyread/wsock.py` pushes progress to the reader page, and you start reading the finished pages while the rest are still translating.

## Advantages

- **Resumable by design.** Batch-level persistence means crashes, restarts and quota outages cost you at most one batch, not the whole paper.
- **Parallel without incoherence.** Segment lanes run concurrently but each batch still sees prior context and the shared glossary, so quality does not fall apart at lane boundaries.
- **Validated output.** Blocks and TeX are checked before merging, and failures are isolated to their pages with a one-click retry.
- **Local-first privacy.** The server binds to 127.0.0.1, the library is plain folders, and only the model you chose ever sees the text.
- **Zero-friction install paths.** A signed-checksum-free installer for each OS, a `start.cmd`/`start.sh` source path, or a plain `pip install` all land in the same home-folder layout.
- **Agent-native.** The CLI and bundled skill make the library a first-class workspace for coding agents, not just for humans.

## Benefits

- **Real reading speed on real papers.** Background page-by-page translation means you start reading the introduction while the appendix is still translating.
- **Trustworthy text.** Bilingual view plus the boxed original page next to the translation makes verifying a suspicious sentence a glance, not a chore.
- **Compound notes.** Highlights, notes and margin answers export to Markdown in reading order, so your reading session becomes a durable Obsidian or Notion document.
- **Budget visibility.** Token metering on every translation and answer, with Claude subscription quota tracking, means no surprise burn.
- **Shareable results.** Single-file offline HTML export carries the translation, equations, page images and your highlights to anyone with a browser; the demo variant even publishes to static hosting.
- **A clean stdlib codebase.** No framework, no build step: the whole backend is readable Python with unit tests and a Playwright end-to-end script.

## Usage

Install from the Releases page (Windows setup exe, macOS arm64 dmg, Linux AppImage), or run from source with Python 3.10 or newer:

```bash
git clone https://github.com/Edwardxlai/easyread.git
cd easyread
./start.sh        # on Windows, double-click start.cmd
```

or:

```bash
pip install git+https://github.com/Edwardxlai/easyread
easyread
```

Your browser opens `http://127.0.0.1:8765`. First stop is Settings, where detected engines appear automatically; add model cards, mark one as the translation engine, and use the built-in "Test one sentence" check. Then drag in a PDF or paste an arXiv ID, DOI, paper title or an OpenReview or journal page link; pick a target language (Chinese, Japanese, Korean, Spanish, French or German), optionally restrict to the body before the references or a page range such as 5 to 12, and let it run. Whole-paper jobs above sixty pages ask for confirmation first.

While reading: click a paragraph for its action bar, select text to highlight or note, drag quotes into Ask AI, and press `?` for the shortcut list. For agents:

```bash
easyread list
easyread import paper.pdf
easyread status ID
easyread discuss ID --from answers.json
easyread export ID
```

Development is straightforward: `python -m unittest discover tests` for unit tests, `node tests/e2e.cjs <paper-id>` for a browser end-to-end run with Playwright, and design notes in `docs/design.md` plus the data format in `docs/data-format.md`.

## Conclusion

EasyRead proves that a deeply useful AI reading tool needs neither a server farm nor a framework: a disciplined stdlib server, JSON-per-paper storage, a resumable two-queue job system, and model engines that borrow your existing subscriptions. The translation quality comes from small, honest engineering choices, batches that carry context forward, output that is validated before merging, edits that are never silently overwritten, and the result is a paper reader that respects both the source document and your attention. If you read literature in a second language, or you want to see how a multi-backend, multi-language document system fits in one readable Python package, this repository deserves an afternoon.

Links:

- Repository: [github.com/Edwardxlai/easyread](https://github.com/Edwardxlai/easyread)
- Releases: [github.com/Edwardxlai/easyread/releases](https://github.com/Edwardxlai/easyread/releases)
- Online demo: [edwardxlai.github.io/easyread/demo/](https://edwardxlai.github.io/easyread/demo/)
- Data format: [docs/data-format.md](https://github.com/Edwardxlai/easyread/blob/main/docs/data-format.md)
- License: [MIT](https://github.com/Edwardxlai/easyread/blob/main/LICENSE)
