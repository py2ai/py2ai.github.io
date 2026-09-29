---
layout: post
title: "HowToLiveBetter: 630 Evidence-Graded Life Decisions, Ranked by Cost-Benefit - Inside eternity4719/HowToLiveBetter"
description: "A source tour of eternity4719/HowToLiveBetter, a Chinese-language evidence-ranked life guide with 630 graded entries across longevity, first aid, personal finance, legal red lines and parenting. We map how the book is organized, how its A/B/C evidence grading and cost-benefit value tiers work, and how a static search page, EPUB/PDF builders and an AI skill are all generated from the same Markdown corpus."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /HowToLiveBetter-Cost-Benefit-Life-Guide/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/howtolivebetter/eternity4719-howtolivebetter-architecture.svg
tags:
  - Evidence-Based
  - Open Source
  - Personal Finance
  - Decision Making
categories: [AI, Open Source]
keywords: "HowToLiveBetter, evidence-based life guide, cost benefit ranking, evidence grading, personal finance, longevity, first aid, legal red lines, static search page, GitHub Pages, EPUB generator, AI skill, Markdown, open source"
author: "PyShine"
---

Most life-advice content on the internet fails at the two questions that actually matter: what will this cost me, and how good is the evidence behind the promised payoff. eternity4719/HowToLiveBetter — a Chinese-language repository running at roughly 24,500 GitHub stars — is a deliberately engineered answer to both. It is a full book, "High Cost-Performance Life Guide" (高性价比人生指南), containing 630 pieces of advice, and every single entry carries the same skeleton: what it costs in money, time and willpower; what it buys you; an evidence grade; and a source that points only at journal papers with DOIs or official documents. With roughly 24,500 stars, it has clearly struck a nerve in the Chinese-language open-source community.

The subject matter is unusually wide for a "guide" — 34 chapters span avoiding early death, quitting smoking and alcohol, personal finance and insurance, unemployment benefits, legal red lines that turn ordinary people into criminal defendants, first aid, renting, chronic illness, eldercare, childbirth, studying abroad, disability, and home medicine safety. The repo explicitly frames its four "currencies" as lifespan, time and energy, money, and personal freedom — and insists those four are never cross-compared against each other. A stat that lowers all-cause mortality by 12% and one that saves 500 yuan a year are, by design, not on the same ruler.

What makes the source worth a tour is that the "architecture" here is not software architecture in the usual sense — it is an editorial methodology that has been turned into a build system. The Markdown corpus is parsed by a client-side search page, linted by machine checks for reference integrity and plain-language style, compiled by CI into EPUB, PDF and a single-file offline HTML, and queried by an AI-assistant skill that is instructed to search the book before answering and to refuse when it cannot. Content repos rarely show this level of pipeline discipline, so let us walk through how the pieces actually fit.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/howtolivebetter/eternity4719-howtolivebetter-overview-architecture.svg" alt="Architecture overview of the eternity4719/HowToLiveBetter repository" style="max-width:100%;height:auto;" />
</div>

*Overview of eternity4719/HowToLiveBetter: a Markdown book (README plus 34 section files and supporting docs) feeding three reading interfaces — a static search page, an AI decision skill, and CI-built EPUB/PDF/offline editions.*

Reading the overview from left to right: the book group is anchored by `README.md`, which catalogs the 34 section files under `book/` and links the long-form essays under `docs/`. The reading interfaces group contains `index.html`, a dependency-free search page that fetches and parses the README and the section files at runtime, and the `skills/life-decision-guide` skill, which reads the README's section map and greps the same corpus. The build-and-release group holds the Node.js tooling: `tools/lib` parses the README structure once for every builder, the EPUB, PDF and offline-HTML builders all import it, and the `.github/workflows/book.yml` workflow drives the three builds on every content push. Nothing in the diagram requires a backend — the entire system runs from a static file tree.

## Why You Need This

The core problem HowToLiveBetter attacks is decision quality under information asymmetry. Health, finance and legal advice is a market flooded with secondhand retellings, and the repo's authoring rules (spelled out in `CLAUDE.md`) ban them outright: sources must be original literature — journal articles with DOI or PubMed links, or reports from bodies like WHO, CDC and national statistics offices — never Zhihu posts, WeChat public accounts or content farms. Numbers that cannot be verified are not written at all; they are marked "TODO, to be verified." The README badge counts 1,341 such source links across the book.

The second problem is that trustworthy information is still not actionable without a cost axis. An intervention can be evidence grade A and still be a bad deal for you — the repo's own example is the shingles vaccine, backed by a phase-3 randomized trial with 97.2% efficacy yet costing thousands of yuan for two doses against a disease that rarely kills. So every entry additionally carries machine-readable cost tags in an HTML comment (`钱=0|少|多 时间=少|中|多 毅力=否|些|是 收益=大|中|小 口径=死亡率|金钱|时间|自由` — money, time, willpower, benefit magnitude, and which currency the benefit is paid in), letting readers filter by what they can actually afford to spend.

The third problem is ranking. Entries inside each section are ordered by cost-performance, not by category: near-zero-cost, high-benefit actions come first. The README states the resulting distribution — of 630 entries, 108 (17%) land in the top "extreme value" tier, 288 (46%) in "high" and 234 (37%) in "average" — and is refreshingly honest that this tier is the author's judgment, which by the book's own standard only qualifies as grade C evidence. The guide is framed as a menu, not a task list: pick one or two items and you have gotten your money's worth.

Finally, it solves the retrieval problem in four different reading situations. Browse and filter online, carry the whole book as an EPUB or PDF, double-click a self-contained offline HTML file that works with no server and no network, or delegate the lookup to an AI assistant wired to quote section-and-entry numbers. The book even anticipates the "who benefits" question, splitting every payoff into four beneficiary tiers from yourself down to strangers, with the explicit rule that stranger-facing advice must state the scam-and-retaliation risk alongside the upside.

## How It Works

The repository is a content pipeline in which one Markdown corpus is parsed, linted, compiled and served by small purpose-built tools; the detailed diagram below traces every stage from authoring rules to release artifacts.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/howtolivebetter/eternity4719-howtolivebetter-architecture.svg" alt="Detailed architecture of the eternity4719/HowToLiveBetter repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the book corpus and its evidence apparatus, the four internal stages of the index.html search page, the authoring automation that keeps stats and cross-references honest, the CI build pipeline, and the AI skill that reuses the same corpus and ranking math.*

### Understanding the Architecture

**The book corpus.** The heart is `README.md` plus 34 numbered files under `book/`, from `book/01-不要早死.md` ("Don't Die Early") to `book/34-家里的常备药别吃出事.md` ("Don't Get Hurt by Home Medicine"). The split is not aesthetic — the README explains the book outgrew GitHub's 512 KB single-file Markdown rendering cap, so later chapters were being cut off. Every entry follows one fixed template: a verb-first title, an invisible cost-tag comment, then cost, a plain-language line (说人话), the raw benefit figures, an A/B/C evidence grade, sources, and notes that must open with "争议" (disputed) when counter-evidence exists. `README.md` itself is the machine-readable hub: the question-to-chapter table, the grading rules, the value-tier definitions and a 41-term statistics glossary all live there, and the build tooling parses it to discover the file list.

**The evidence apparatus.** Under `docs/`, a folder named `docs/核实记录` ("verification records") holds 97 files documenting how individual sources were checked, while `docs/引用对照.md` is a generated cross-reference table mapping every "see entry X" citation to the entry title it actually lands on. The latter exists because entry numbers are positional: insert one entry and every later number shifts silently. The table is committed to the repo precisely so that a git diff exposes any citation that drifted — a genuinely clever use of version control as an editorial safety net.

**The search page.** `index.html` is a single-file, dependency-free application with four internal stages. A retrying fetcher (`readText`) pulls the README and all 34 section files with a 20-second timeout, exponential backoff and a corpus cache, because a flaky mobile connection was killing mid-book reads. A parser turns the Markdown into entries and also extracts the glossary from the README, decorating every statistics term on the page with hover tooltips. A ranking stage applies the `COST_W` weights — money, time and willpower each scored 0–2 — and the `e.ratio` rule (large benefit with all-zero costs is "extreme value"; large benefit with cost score ≤ 2, or medium benefit with zero costs, is "high"; everything else is "average"). The shell then renders filterable cards: by keyword, chapter, evidence grade, the three cost dimensions, or benefit magnitude, with dashed cross-reference popups that show the cited entry inline and a modal that renders the `docs/` long essays in place.

**The authoring automation.** Writing rules are enforced by machine, not vibes. `tools/check-plain.mjs` lints every plain-language line for length (120 characters), banned research jargon (HR, RR, CI, cohort, meta-analysis), numbers that do not appear in the entry's own benefit field, and a list of vague AI-ish filler phrases — the file header records that a reader complaint (issue #42) triggered 173 fixes in a single day. `tools/check-refs.mjs` validates the 533-baseline cross-references, including anchor-word matching so each citation carries a content hint, not a bare number. `tools/sync-stats.ps1` recomputes the entry and grade counts and rewrites them into the README, `index.html` and `tools/og.html` in one shot, then regenerates the `og.png` social card. Issue templates under `.github/ISSUE_TEMPLATE` route correction and new-content reports into the same discipline.

**The build and release pipeline.** `tools/lib/book.mjs` is the shared parser whose `readBook()` function extracts the description, chapter list and essay list purely from README structure — no hardcoded file names, so new sections are picked up automatically. `tools/epub/build.mjs` produces an EPUB 3 with `marked` as its only dependency and a hand-rolled ZIP writer (because EPUB requires the uncompressed `mimetype` entry first). `tools/pdf/build.mjs` routes Markdown through pandoc into typst, with the page layout in `tools/pdf/template.typ`. `tools/offline/build.mjs` inlines the entire corpus into a `window.__CORPUS__` variable inside a copy of `index.html`, strips the analytics snippet so the offline file makes zero outbound requests, and data-URIs the sidebar image. `.github/workflows/book.yml` runs the two checks as separate jobs — deliberately not chained to the build, so a broken anchor cannot block a release — then builds all three editions, runs epubcheck, and republishes them to the fixed `epub-latest` release with permanently stable download links.

**The AI layer.** `skills/life-decision-guide/SKILL.md` is a protocol that turns Claude Code or Codex into a book-literate advisor: locate the right chapters via the README table, grep out whole entries (reading the notes field, not just titles), sort them by replicating the exact `COST_W`/`e.ratio` math from `index.html` rather than from memory, and cite every claim as "section X, entry Y." Its refusal rules are the interesting part — if the corpus cannot be fetched, the skill must say so instead of reconstructing the book from memory, and anything not in the book must be labeled as common sense rather than book content. The same file is mounted at `.claude/skills/life-decision-guide/SKILL.md` and surfaced for Codex through the root `AGENTS.md`, with `skills/life-decision-guide/README.md` carrying the install commands.

The end-to-end flow closes cleanly: an edit to any file under `book/` or `docs/` triggers CI, which lints cross-references and plain language, rebuilds the three e-book editions from the README-parsed structure, and republishes them; the live search page picks up the same files on its next load; and the AI skill greps whatever corpus it can reach. One source of truth, four synchronized reading surfaces, and no database anywhere.

## Advantages

- **Evidence grading is structural, not decorative.** Every one of the 630 entries carries an A/B/C grade with defined meanings — A means quantified results from meta-analyses, large cohorts or randomized trials — and disputed A/B entries must list the opposing evidence, with the tally (A 420, B 159, C 51, 58 disputed) synced by script.
- **Cost is a first-class, machine-readable dimension.** The HTML-comment cost tags make money, time and willpower filterable in the search page, which is what enables the "108 entries that cost nothing, take no time and need no willpower" query.
- **Zero-backend simplicity.** The online search page is one static HTML file; the whole deployment story is any static server or GitHub Pages, with no dependencies to install and no data leaving the reader's browser.
- **Self-policing content quality.** The cross-reference table committed to git, the plain-language linter, and the single-command stat sync mean content drift — the classic failure mode of long-lived docs repos — is caught by CI instead of by readers.
- **Honest refusal in the AI layer.** The skill's hard rule that unfetchable corpora and unbook-derived claims must be declared, not improvised, is a pattern worth copying for any retrieval-grounded assistant.
- **Fixed-link, auto-refreshing artifacts.** The `epub-latest` release URL never changes, yet its EPUB, PDF and offline HTML are rebuilt on every content push, so shared copies stay link-stable while the online edition stays current.

## Benefits

- **For readers:** a ranked, filterable menu of 630 life decisions where you can isolate, say, the grade-A entries with quantified numbers, or the items that cost literally nothing — instead of trusting an influencer's top-ten list.
- **For risky situations:** dedicated chapters on legal red lines for ordinary people and for programmers (`book/09`, `book/11`), emergency first-actions (`book/13`), and what to do in the first weeks after bereavement, job loss or a serious diagnosis (`book/29`) — the kind of content that is hard to think clearly about in the moment.
- **For offline and low-bandwidth users:** one double-clickable HTML file containing the entire book, search and filters, shareable through messaging apps — built for the reality that many readers are on phones in poor signal.
- **For non-Chinese readers:** community-maintained English, Russian and Spanish translations hosted separately, plus a third-party checklist web app, with the README candidly flagging that translations may lag the Chinese original.
- **For contributors:** machine-enforced entry format, anchored citations and plain-language rules mean a correction PR is checked against the same standards the author uses, and the diff-able citation table makes breakage reviewable.
- **For tool builders:** the repo is a working template for "content as a build pipeline" — corpus, linters, multi-format compilation and an AI retrieval layer — licensed under the Unlicense, so every piece is public-domain and reusable.

## Usage

Most people need no deployment at all — the online search page is live, and the README recommends the offline single-file HTML for offline use. To run your own copy locally, serve the static tree (note that `index.html` must be opened over HTTP, not by double-clicking the file):

```bash
git clone https://github.com/eternity4719/HowToLiveBetter.git
cd HowToLiveBetter
python -m http.server 8000
```

Then open `http://localhost:8000/`. To generate the three e-book editions yourself (the release artifacts are built automatically by CI, so this is optional):

```bash
cd tools/epub && npm ci && npm run build   # EPUB
node tools/offline/build.mjs               # offline single-file HTML
node tools/pdf/build.mjs                   # PDF, also needs pandoc >= 3.1 and typst >= 0.13
```

All three outputs land in `dist/`. To install the AI decision skill for Claude Code (usable from any directory):

```bash
mkdir -p ~/.claude/skills/life-decision-guide && curl -fsSL -o ~/.claude/skills/life-decision-guide/SKILL.md "https://raw.githubusercontent.com/eternity4719/HowToLiveBetter/main/skills/life-decision-guide/SKILL.md"
```

For Codex, the equivalent is:

```bash
mkdir -p ~/.codex/prompts && curl -fsSL -o ~/.codex/prompts/life-decision-guide.md "https://raw.githubusercontent.com/eternity4719/HowToLiveBetter/main/skills/life-decision-guide/SKILL.md"
```

## Conclusion

HowToLiveBetter is proof that a content project can have real architecture. The topic structure — 34 chapters mapped to life problems, entries ranked within each by cost-performance — and the evidence-first methodology — A/B/C grading, four non-comparable benefit currencies, machine-readable cost tags, original-sources-only citation — are encoded not just in prose but in parsers, linters, CI jobs and an AI protocol. The result is a guide whose trustworthiness does not depend on trusting the author: every number is traceable to a DOI or an official document, and every internal claim is cross-checked by tooling that treats a drifted reference the way a compiler treats a type error.

A natural caveat, which the repo itself states plainly: the book gives general guidance, not personal medical, legal or tax advice, and the value tiers are the author's judgment rather than evidence. Treat it as the best-curated menu we have seen in this genre — then do your own math on the entries you pick.

Links:

- GitHub repository: https://github.com/eternity4719/HowToLiveBetter
- Online search page: https://eternity4719.github.io/HowToLiveBetter/
- E-book downloads (EPUB / PDF / offline HTML): https://github.com/eternity4719/HowToLiveBetter/releases/tag/epub-latest
- English translation mirror: https://dlgrv.github.io/HowToLiveBetter/en/
