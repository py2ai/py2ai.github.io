---
layout: post
title: "ModelScope Cookbook: A Hands-On Curriculum for Open-Source AI Apps - Inside modelscope/ms-cookbook"
description: "A source tour of modelscope/ms-cookbook, the ModelScope Purple Book (魔搭紫皮书): an 8-part, 35-chapter open curriculum that walks developers from model selection and inference to fine-tuning, evaluation, RAG, agents, and AIGC, complete with its own static reading site and FastAPI-powered community backend."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /ModelScope-Cookbook-Open-Source-AI-Curriculum/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ms-cookbook/modelscope-ms-cookbook-architecture.svg
tags:
  - ModelScope
  - AI Education
  - Fine-Tuning
  - Open Source
categories: [AI, Open Source]
keywords: "ModelScope Cookbook, purple book, mo da zi pi shu, open-source AI models, model fine-tuning, ms-swift, EvalScope, DiffSynth LoRA, RAG knowledge assistant, MCP tools, AI agents, Ollama local inference, model evaluation, AIGC tutorial, learning path"
author: "PyShine"
---

Most developers who stall with open-source models do not fail at the model itself. They fail one step earlier: which checkpoint fits the task, whether the hardware can hold it, how to turn messy business material into training data, and how to prove the fine-tune actually helped. Tutorials usually cover one of those steps in isolation; very few connect the whole chain into a single, ordered path you can actually follow.

That is exactly the gap the ModelScope Cookbook sets out to fill. Published by the ModelScope team under the nickname 魔搭紫皮书 ("ModelScope Purple Book"), it is an open-source, hands-on guide whose tagline reads "From open-source models to practical AI applications. Choose a model. Run it. Adapt it. Build with it." The book is organized into 8 parts and 35 chapters (34 of which are open for reading; one chapter is still marked pending in `content/manifest.json`), and it leans on real tooling from the ecosystem — EvalScope for evaluation, ms-swift for fine-tuning, DiffSynth for image customization, Ollama for local inference — alongside RAG and agent workflows.

What makes the repository worth a source tour is that it is more than a folder of Markdown files. The repo ships its own reading website, a build pipeline that turns canonical chapter sources into both a website data file and GitHub-readable Markdown, a FastAPI "reading community" backend with OAuth login, reader notes, and privacy-conscious analytics, plus validation scripts and tests. Reading the code tells you how a serious content project publishes, verifies, and maintains a curriculum at scale — and the chapters themselves are a curriculum you can work through today.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ms-cookbook/modelscope-ms-cookbook-overview-architecture.svg" alt="Architecture overview of the modelscope/ms-cookbook repository" style="max-width:100%;height:auto;" />
</div>

*The overview traces how the Purple Book flows from canonical chapter sources and a manifest through the content build pipeline into a zero-dependency reading site, with an optional FastAPI community backend and a quality-and-deployment track on the side.*

Reading the overview from left to right: the book's source of truth lives in `content/source-html/` (canonical HTML chapters) and `content/manifest.json` (part order, titles, and reading status). The builder at `scripts/build-content.py` consumes both and emits two artifacts: GitHub-readable Markdown chapters under `content/chapters/` and the generated `assets/content.js` book data that the reading interface at `index.html` loads and renders through its JavaScript layer. The same book data feeds the community API in `app.py`, which serves reader notes and analytics back to the page. Finally, the test suites under `tests/` validate both the generated book data and the API, and the deployment workflow at `.github/workflows/deploy-modelscope.yml` publishes site files to the public ModelScope Studio on every push to `main`.

## Why You Need This

If you have ever tried to learn open-source model development from scattered blog posts, you know the failure mode: each article assumes you already solved the previous stage. The Cookbook is deliberately sequenced to break that loop. Its parts progress from 认识开源模型 ("understanding open-source models"), through 从问题出发 ("starting from the problem"), to 跑得起 ("getting models to run"), 调得好 ("fine-tuning and evaluation"), and then full application systems — the same order real projects unfold in.

The second reason is decision support. Chapter 5, 要把业务问题转换成模型任务问题 ("turn a business problem into a model task"), teaches you to frame inputs, outputs, and evaluation criteria before touching any weights. Chapter 6, 先评再选：用 EvalScope 形成开源模型的第一份报告 ("evaluate first, then choose: your first report with EvalScope"), makes model selection a measured exercise instead of a vibes check. Later chapters on server sizing, quantization trade-offs, and running inference on a laptop with Ollama address the resource questions that stop most side projects before they start.

The third reason is the application chapters. Part 5, 场景篇 ("scenario chapters"), builds complete systems rather than code snippets: an AI fitness coach that compares body keypoints against exercise video (chapter 16), customer-service call transcription and quality analysis (chapter 17), a speak-and-listen voice assistant (chapter 18), and an enterprise knowledge Q&A assistant grounded in retrieval (chapter 19). Each one names the workflow it explores up front.

Finally, the agent and AIGC parts are unusually current for a curriculum. Part 6 walks through generative-image use cases, DiffSynth-based LoRA customization, and product-marketing image editing; Part 7 moves from "what is an agent" through MCP tool connections to quick-starts on Claude Code, PI, and DeepSeek Harness, closing with a production-line inspection agent built with Penguin Harness.

## How It Works

Underneath the prose, ms-cookbook is a small but disciplined publishing system: canonical sources, a deterministic build, a static reader, and an optional community server — each piece verified by scripts and tests.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ms-cookbook/modelscope-ms-cookbook-architecture.svg" alt="Detailed architecture of the modelscope/ms-cookbook repository" style="max-width:100%;height:auto;" />
</div>

*The detailed view expands each subsystem: the content build pipeline, the static reader and its search layer, the FastAPI community service, and the validation and deployment tooling.*

### Understanding the Architecture

**The content pipeline.** The single source of truth is `content/source-html/`, where each chapter is a canonical HTML file preserving paragraphs, headings, tables, code, formulas, and images in order. `scripts/build-content.py` reads those files plus `content/manifest.json` — which fixes the 8-part order, chapter titles, and per-chapter reading status — and generates two things: reader-friendly Markdown under `content/chapters/` (via the converter in `scripts/chapter_markdown.py`) and `assets/content.js`, a single generated data file that ships the entire book to the browser. An import mode can also pull a Feishu document snapshot, localizing media into `assets/manuscript-20260914/` and recording revision evidence in `content/sync-report.json`.

**The reading website.** The site needs no build step and no dependencies to read: `index.html` loads the generated `assets/content.js` and renders chapters through `assets/paper.js`, with hash-based navigation from `assets/reader-routes.js` and reader conveniences in `assets/reading.js`. Full-text keyword search across all chapter titles and bodies is implemented client-side in `assets/book-search.js`, and mathematical formulas render through a bundled KaTeX copy in `assets/katex/`. That is why the README can promise local reading with just Python 3 and a browser — the entire book is one data file plus static assets.

**The reading community backend.** `app.py` is a FastAPI application that turns the book into a community space. It implements ModelScope OAuth login through Authlib, then exposes per-chapter note endpoints under `/api/chapters/{chapter}/entries` supporting highlights, annotations, and comments, plus a "wishes" endpoint (`/api/wishes`) where readers request missing content. Sessions, CSRF tokens, origin checks, and simple rate limits protect the write paths, and everything is stored in SQLite with WAL mode.

**Privacy-first analytics.** The same server records page views through `/api/analytics/*`, but it never stores raw visitor identifiers: each visitor string is HMAC-hashed with the session secret before it touches the database. Daily and hourly SQLite tables track views and unique visitors with explicit retention windows, a snapshot mechanism copies consistent database images to persistent storage, and a repair path quarantines corrupted databases and restores the last good snapshot.

**Validation and tests.** Quality gates live in `scripts/check-site.mjs`, which asserts that the generated book data and `content/manifest.json` agree on chapter counts and order, that discussion IDs are unique, that every local link and image resolves inside the repo, and that no unsafe URLs sneak into chapter HTML. Node-based tests cover the search dialog and reader routes, Python tests cover the discussion and analytics APIs, and `tests/discussion-browser.cjs` exercises the community UI end to end.

**Deployment.** Publishing is automated in `.github/workflows/deploy-modelscope.yml`: pushes to `main` that touch site files trigger `scripts/deploy-modelscope.mjs`, which synchronizes committed files to the public Studio `ms-cookbook-team/ms-cookbook` using a stored API key and verifies the live page and deployed revision before reporting success. GitHub remains the source of truth; the Studio is the reading surface. For self-hosting the community features, a slim `Dockerfile` installs `requirements.txt` (FastAPI, Authlib, httpx, itsdangerous, uvicorn) and launches the app with uvicorn on port 7860.

The end-to-end flow is therefore a straight line: an author edits a chapter in `content/source-html/`, runs the builder, validation scripts confirm nothing broke, the commit lands on `main`, the workflow publishes the update to the ModelScope Studio, and readers get the new chapter — with highlights and comments still attached to the right chapter via stable discussion IDs.

## Advantages

- **A complete, ordered curriculum.** Eight parts and thirty-five chapters cover the full arc from model discovery and downloads to fine-tuning, evaluation, RAG, agents, and AIGC — not isolated tricks.
- **Decision-first teaching.** Chapters on framing model tasks, building an EvalScope baseline, sizing servers, and quantization force the planning habits that make projects succeed before any code runs.
- **Real ecosystem tooling.** Every recipe names its tools — ms-swift, EvalScope, DiffSynth, Ollama — so what you learn maps directly to what you will run.
- **Zero-friction reading.** The book renders from static files with client-side search and bundled KaTeX; no install, build, API key, or GPU is needed just to read.
- **Disciplined content engineering.** Canonical HTML sources, generated Markdown, a manifest, and strict site validation keep 35 chapters consistent as they evolve.
- **A community layer done right.** OAuth-gated notes, a wish wall, and privacy-preserving hashed analytics live in one readable `app.py` you can audit or self-host.

## Benefits

- **Faster time to first result.** The getting-started path (chapters 1 → 5 → 7) is explicitly designed to get you from "what is an open model" to a first successful inference quickly.
- **Confident model selection.** Instead of guessing, you learn to produce a baseline evaluation report and reason about size, memory, and quantization trade-offs against your own hardware.
- **Practical fine-tuning literacy.** The adapt path (chapters 12 → 13 → 15) teaches data preparation from business materials, lightweight fine-tuning with ms-swift, and baseline-versus-tuned comparison — the loop that actually improves products.
- **Reusable application patterns.** Fitness coaching, customer-service quality analysis, voice assistants, and enterprise RAG are worked end to end, so the workflows transfer to your own domain.
- **Current agent knowledge.** MCP, skills, and multiple agent harness quick-starts give you a grounded map of the agent toolchain.
- **A maintainable publishing model to copy.** If you run your own docs or course repo, the source-to-generated-data pipeline with validation gates is a proven pattern worth borrowing.

## Usage

Reading locally requires only Python 3 and a browser — the README's recommended way to browse offline is:

```bash
git clone https://github.com/modelscope/ms-cookbook.git
cd ms-cookbook
python3 -m http.server 4173 --bind 127.0.0.1
```

Then open `http://127.0.0.1:4173/#home` in a modern browser (use `4174` if the port is occupied; stop the server with `Ctrl+C`). For online reading, no installation is needed at all — the book is served on ModelScope at `modelscope.cn/studios/ms-cookbook-team/ms-cookbook` with full-text search, guided reading paths, and rendered formulas.

If you contribute a chapter edit, the README's maintenance workflow rebuilds and validates the generated artifacts:

```bash
python3 scripts/build-content.py
node scripts/build-content.mjs --check
node scripts/check-site.mjs
```

And if you want to self-host the reading community API rather than just the static pages, the `Dockerfile` shows the runtime contract: install `requirements.txt` and serve the app with uvicorn on port 7860:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Individual exercises inside chapters may require model downloads, credentials, or compute — the chapters state their own environment requirements.

## Conclusion

The ModelScope Cookbook is a rare combination: a genuinely sequenced curriculum for open-source model work, and a cleanly engineered publishing platform that treats its own content with the same rigor a codebase gets. The chapters walk you from choosing and running a model through fine-tuning, evaluation, RAG, agents, and generative AI; the repository around them shows how to build, validate, and deploy that knowledge as a product. If you are starting with open-source models — or maintaining learning content of your own — both layers are worth your time. The project is Apache 2.0 licensed and welcomes contributions on GitHub.

Links:

- GitHub repository: <https://github.com/modelscope/ms-cookbook>
- Read online on ModelScope: <https://modelscope.cn/studios/ms-cookbook-team/ms-cookbook>
- Contributing guide: <https://github.com/modelscope/ms-cookbook/blob/main/CONTRIBUTING.md>
