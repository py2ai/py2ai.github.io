---
layout: post
title: "Coursebook: One LaTeX Tree, Three Textbook Formats - Inside cs341-illinois/coursebook"
description: "How cs341-illinois/coursebook compiles a tree of LaTeX chapters into PDF, EPUB, and GitHub-wiki editions using a Makefile, pandoc filters, and a three-way GitHub Actions deploy matrix. A source tour of the docs-as-code pipeline behind UIUC's open systems programming textbook."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /Coursebook-LaTeX-Textbook-Pipeline/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/coursebook/cs341-illinois-coursebook-architecture.svg
tags:
  - LaTeX
  - Open Source
  - Systems Programming
  - Documentation
categories: [AI, Open Source]
keywords: "cs341 coursebook, systems programming textbook, LaTeX textbook pipeline, pandoc filters, gen_wiki.py, lualatex, latexmk, GitHub Actions deploy, open source textbook, UIUC CS 341, EPUB generation, GitHub wiki publishing, docs as code, C programming, University of Illinois"
author: "PyShine"
---

Most textbooks live in a word processor, exported by hand whenever someone remembers to. The Coursebook is the opposite: an entire systems programming textbook that lives in a Git repository, where a single `git push` compiles the prose into a PDF, an EPUB, a GitHub wiki, and a course website without anyone pressing an export button. It is textbook publishing treated exactly like software: versioned, reviewed in pull requests, built by CI, and deployed automatically.

The repository, maintained by the course staff of CS 341: System Programming at the University of Illinois Urbana-Champaign, houses an open-source introductory systems programming textbook. All instruction and code is in C — the project describes C as the de-facto language of the Linux kernel — and the book assumes readers already know a programming language and some assembly. The project explicitly positions itself as a standardization and continuation of Lawrence Angrave's crowd-sourced SystemProgramming wikibook experiment, with stated goals of improving rigour, adding citations, footnotes, extended reading, and a glossary, and exporting to PDF, Markdown, and HTML.

What makes the source worth a tour is not just the prose, which is genuinely good, but the machinery around it. This is a compact, complete docs-as-code system: a data file controls the chapter order, a Python script farms pandoc conversions across CPU cores, custom pandoc filters harvest metadata and enforce accessibility rules, and a GitHub Actions matrix publishes three different formats to three different targets. Anyone who maintains documentation — for a course, a library, or a company — will recognize their own problems in this tree and can borrow its answers.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/coursebook/cs341-illinois-coursebook-overview-architecture.svg" alt="Architecture overview of the cs341-illinois/coursebook repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the coursebook architecture: content sources feed a LaTeX spine and a pandoc-based markdown factory, both driven by the Makefile and dispatched by CI.*

Reading the overview from left to right: the content sources on the far left — the chapter `.tex` files, the `order.yaml` chapter list, and `glossary.tex` — flow into the LaTeX assembly layer, where `_scripts/gen_order.py` turns `order.yaml` into the generated `order.tex` that `main.tex` includes, and `prelude.tex` supplies the shared packages and styles. The build toolchain in the middle fans this out: the `Makefile` runs latexmk with lualatex to produce the book PDF, while `_scripts/gen_wiki.py` reads the same order file and converts the same chapters to GitHub-flavored Markdown through pandoc filters. On the right, the CI layer closes the loop — the deploy workflow runs `_scripts/script.sh`, which dispatches either the make targets or the wiki generator, and hands the finished artifacts to `_scripts/deploy.sh` for publishing.

## Why You Need This

If you have ever maintained a living document read by a large audience, you know the maintenance trap. The coursebook's own contributing guide notes that its work will be read by hundreds of students every semester, which means every typo, stale explanation, and broken link matters at a scale most documentation never reaches. Storing the book in LaTeX source inside Git means a fix made once propagates to every format on the next build — there is no "which copy is current" problem, because there is only one copy.

The second problem is drift, the quiet killer of wikis. The original Angrave wikibook succeeded because anyone could edit it, but wiki pages accumulate in no particular order, with inconsistent depth and no citations. The coursebook keeps the openness — anyone can open a pull request — but adds the discipline of a fixed chapter list in `order.yaml`, per-chapter BibTeX bibliographies, a shared glossary in `glossary.tex`, and a written style guide in `CONTRIBUTING.md` that standardizes everything from pronoun usage to which LaTeX macros highlight system calls.

The third problem is multi-format publishing. Producing a polished PDF, a clean EPUB, and web-friendly Markdown from the same source is tedious manual labor in most projects, and it is exactly the kind of repetitive, error-prone work that should be automated. Here, every push to master triggers builds for all three formats, so contributors "can focus on writing," as the README's goals put it.

Finally, if you are an educator or technical author, you need a starting template that respects the separation of content from presentation. This repository is that template: chapters are self-contained folders, ordering is data rather than code, formatting lives in one prelude file, and the entire publish pipeline is inspectable shell and Python you can adapt to your own book.

## How It Works

The pipeline is best understood as three cooperating layers — content, assembly, and delivery — wired together by the Makefile and GitHub Actions.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/coursebook/cs341-illinois-coursebook-architecture.svg" alt="Detailed architecture of the cs341-illinois/coursebook repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the coursebook: every node is a real file in the repository, from the chapter sources and pandoc filters to the deploy scripts that push the wiki and the deploy branches.*

### Understanding the Architecture

**The content tree is data-driven.** `order.yaml` at the repository root is the single source of truth for the book's table of contents, listing eighteen chapters from `introduction/introduction` through `post_mortems/post_mortems`. Each chapter is a self-contained folder holding the `.tex` source, a matching `.bib` bibliography (see `processes/processes.tex` and `processes/processes.bib`), and a `drawings/` directory of `.eps` figures. Nothing about chapter ordering is hard-coded into the LaTeX — it is all in that one YAML file.

**The LaTeX spine assembles the book.** `main.tex` declares a 10pt `book` document, inputs `prelude.tex` (which carries the font setup, code-listing styles, and bibliography packages including `natbib` with `chapterbib`), loads `glossary.tex`, and then simply runs `\input{order.tex}`. That `order.tex` is not committed — the `Makefile` generates it by invoking `_scripts/gen_order.py`, which converts each `order.yaml` entry into a `\include{}` line. The `Makefile` builds `main.pdf` through latexmk with lualatex, and can also build a single chapter in isolation (for example `make introc/introc.pdf`) using an `\includeonly` trick that reuses the full `main.tex` preamble.

**The markdown factory is a parallel pandoc driver.** `_scripts/gen_wiki.py` reads the same `order.yaml`, then for each chapter builds a standalone document by concatenating `prelude.tex`, the GitHub compatibility shims from `github_redefinitions.tex`, and the chapter source, and converts it with pandoc targeting `gfm+raw_html` so the output works on GitHub and Jekyll. Conversions run across a `multiprocessing.Pool` sized to one less than the CPU count, and Jinja2 templates render the wiki's `Home.md` index and `_Sidebar.md` from the harvested chapter metadata.

**Custom filters do quality control and rewriting.** Before conversion, `_scripts/pandoc_header_filter.py` walks each document and emits YAML metadata — chapter name, level-two subsections, and bibliography file — that drives the wiki index. It also enforces two quiet quality gates: every image must carry a real alt tag or the build raises, and every external link is validated with a HEAD request under a 15-second timeout, cached for 30 days so builds stay fast. After citeproc, `_scripts/pandoc_wiki_filter.py` rewrites relative image URLs to absolute `raw.githubusercontent.com` links, swaps `.eps` extensions for `.png`, converts links to raw HTML anchors to avoid GitHub-versus-Jekyll parsing ambiguity, and wraps math in `$$` blocks. `_scripts/pandoc_epub_filter.py` performs the same alt-text and image-rewriting duties for the EPUB target.

**Delivery is a three-way CI matrix.** `.github/workflows/deploy.yaml` runs a matrix over `WIKI`, `EPUB`, and `PDF` with `fail-fast: false`, so a wiki failure cannot leave a healthy PDF undeployed, and a concurrency group collapses bursts of pushes instead of canceling a deploy halfway. `_scripts/script.sh` dispatches on the build focus — wiki goes to `gen_wiki.py`, the other two to `make pdf` and `make epub`. `_scripts/install.sh` pins pandoc and installs a curated list of texlive packages rather than the multi-gigabyte `texlive-full`. Publishing then splits: `_scripts/deploy.sh` force-pushes orphan branches, `pdf_deploy` for the PDFs and `epub_deploy` for the EPUB, while `_scripts/push_to_wiki.sh` clones the repository's wiki, copies the generated Markdown in, and pushes it. Finally `_scripts/site_deploy.sh` clones the course site repository, points its `_coursebook` submodule at the fresh wiki commit, and pushes an empty commit whose only job is to trigger the site's rebuild. A sibling workflow, `.github/workflows/build.yaml`, runs the same matrix on pull requests with aggressive cancellation so fixup pushes do not queue behind nine-minute builds.

**End to end:** a contributor edits one sentence in `processes/processes.tex` and opens a PR; `build.yaml` proves all three formats still compile; on merge, `deploy.yaml` converts the chapter to Markdown through the prelude, shims, citeproc, and wiki filter, pushes the updated wiki and site, recompiles the book with lualatex onto `pdf_deploy`, regenerates the EPUB onto `epub_deploy`, and the course website picks up the new wiki commit — one sentence edited, four surfaces updated, zero manual steps.

## Advantages

- **True single-source publishing.** One `.tex` tree yields PDF, EPUB, wiki Markdown, and the HTML site; the pipeline in `Makefile` and `_scripts/` guarantees the formats cannot drift apart.
- **Order as data.** Chapter sequence lives in `order.yaml` and is compiled to `order.tex` by `_scripts/gen_order.py`, so restructuring the book is a one-file diff, not a hunt through include statements.
- **Builds that police themselves.** The filters in `_scripts/pandoc_header_filter.py` fail the build on missing image alt text and flag unreachable external links, with a 30-day cache keeping the checks cheap.
- **Parallel and resilient CI.** The WIKI/EPUB/PDF matrix uses `fail-fast: false` so one failing leg cannot stale the others, while concurrency settings collapse redundant runs.
- **Fast inner loop for writers.** `rebuilder.sh` watches the tree with inotify and re-runs make automatically, and per-chapter make targets rebuild a single chapter instead of the whole book.
- **Reproducible toolchain.** `_scripts/install.sh` documents and installs exactly the dependencies each build focus needs, from the pinned pandoc to the curated texlive package list.

## Benefits

- **Free and open, permanently.** The book is released under the University of Illinois/NCSA Open Source License, and the stated philosophy in `CONTRIBUTING.md` is an accessible, first-class systems textbook "free to use forever."
- **Rigor built in.** Per-chapter bibliographies, citations processed by pandoc-citeproc, a shared glossary, and a strict writing style guide raise the factual bar above typical course notes.
- **Read it your way.** Students can use the full PDF, the EPUB for e-readers, the GitHub wiki for quick browsing, or the HTML site — same content, four presentations.
- **A proven docs-as-code template.** The whole design — data-driven ordering, parallel pandoc conversion, filter-based quality gates, matrix CI — transfers directly to any book or documentation project.
- **Low-friction contribution.** The contributing guide walks newcomers through forking, editing, and PRs, and CI gives every contributor confidence their change compiles in all formats before a human ever reviews it.
- **Semester-aware versioning.** Releases follow a term-year-plus-increment scheme described in `CONTRIBUTING.md`, so each cohort can pin the edition of the book it was taught with.

## Usage

To build the wiki (Markdown) version, set up Python dependencies and run the generator as described in `CONTRIBUTING.md`:

```bash
virtualenv -p python3 env
source env/bin/activate
python -m pip install -r requirements.txt

mkdir out
python _scripts/gen_wiki.py order.yaml out
```

For the PDF, install a TeX distribution and use the Makefile:

```bash
sudo apt install texlive-full
make main.pdf
```

Individual chapters can be built alone, which keeps iteration fast:

```bash
make introc/introc.pdf
```

Running plain `make` builds both the PDF and the EPUB targets defined in the `Makefile`. For an automatic rebuild-as-you-type loop, the contributing guide suggests the optional watcher (requires `inotify-tools`):

```bash
sudo apt install inotify-tools
./rebuilder.sh
```

## Conclusion

The coursebook repository is a reminder that the interesting engineering in a "textbook repo" is rarely the prose alone. With little more than a Makefile, a few hundred lines of Python, and GitHub Actions, the CS 341 staff turned a folder of LaTeX into a self-publishing system: `order.yaml` decides what the book is, `main.tex` and the pandoc filters decide what it becomes, and the deploy scripts decide where it lives. The result is a textbook that is genuinely open — anyone can fix a typo and watch it reach every format — and a pipeline that any documentation team can study, borrow from, or outright reuse. If you teach, write, or maintain docs, reading this source is an afternoon well spent.

Links:

- [GitHub repository: cs341-illinois/coursebook](https://github.com/cs341-illinois/coursebook)
- [Current book PDF (pdf_deploy branch)](https://github.com/cs341-illinois/coursebook/blob/pdf_deploy/main.pdf)
- [Current book EPUB (epub_deploy branch)](https://github.com/cs341-illinois/coursebook/blob/epub_deploy/main.epub)
- [CS 341 course site](http://cs341.cs.illinois.edu/)
- [Original Angrave SystemProgramming wikibook](https://github.com/angrave/SystemProgramming/wiki)
