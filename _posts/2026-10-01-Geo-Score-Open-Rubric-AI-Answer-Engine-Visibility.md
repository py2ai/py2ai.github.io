---
layout: post
title: "geo-score: An Open Rubric For Whether AI Answer Engines Can Cite Your Site - Inside jianruntech/geo-score"
description: "geo-score is a zero-dependency Python CLI that scores any website 0-100 for AI answer-engine visibility against a published, versioned rubric, then optionally checks actual citations through the AI engines' APIs with your own keys. We tour the repository: the 21 tiered checks, the gate rules, the 317-site public benchmark, and the MCP server that exposes it all to agents."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /Geo-Score-Open-Rubric-AI-Answer-Engine-Visibility/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/geo-score/jianruntech-geo-score-architecture.svg
tags:
  - GEO
  - AI Search
  - SEO
  - Python
categories: [AI, Open Source]
keywords: "geo-score, generative engine optimization, AI visibility, answer engine optimization, llms.txt, structured data, rubric, MCP server, Python CLI, ChatGPT citations, Perplexity, open source"
author: "PyShine"
---

A strange kind of invisibility has arrived for websites: your pages can rank well on Google and still never be quoted by ChatGPT, Perplexity, Gemini, or Copilot, because answer engines do not rank results — they retrieve passages, decide which sources deserve trust, and cite a handful of them. [geo-score](https://github.com/jianruntech/geo-score) is an open-source instrument for that new question. It reads a site the way a retrieval crawler would and prints a 0 to 100 readiness score against a rubric that is published, versioned, and open to challenge.

The tool's pitch is refreshingly concrete: one command, no installation, no API key, about twenty seconds, and every check shows its evidence and what the next tier requires. It is a single Python file built entirely on the standard library, so there is nothing to break. Beyond the free score, two paid levels — using your own keys — ask the AI engines a real question and watch a question set week over week, measuring whether you are actually cited rather than merely citable. The project is explicit that these are two different numbers that are never added together.

The source rewards a tour because it treats measurement as an honest engineering problem. The rubric ships as both prose and JSON, three checks act as gates that cap a hopeless site's score, the benchmark of well-known sites publishes every raw per-site report, and the repository even measures its own reproducibility and validity against hand-scored audits — a level of self-skepticism rare in the SEO tooling world.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/geo-score/jianruntech-geo-score-overview-architecture.svg" alt="Architecture overview of the jianruntech/geo-score repository" style="max-width:100%;height:auto;" />
</div>

*High-level architecture overview of the jianruntech/geo-score repository, from the versioned rubric through the scoring engine, interfaces, tests, and public benchmark.*

Reading the overview from left to right: the machine-readable rubric loads its checks into the single-file scoring engine, which pairs with the citation watcher for the paid levels; the same engine is invoked by a Claude Code skill, a GitHub Action, and a local MCP server documented for every agent client; unit and end-to-end tests pin the engine's behavior against fixture mini-sites and a golden report; and a benchmark runner drives the engine across hundreds of public sites into published statistics.

## Why You Need This

The failure modes of answer-engine visibility are mechanical, invisible, and cheap to fix — which is exactly why a scored checklist beats intuition. A robots.txt line can forbid every AI crawler at once. A template can omit a date so freshness is unverifiable. A homepage can render its body only after JavaScript runs, handing a crawler an empty shell while a browser sees a beautiful page. The project's own benchmark found that a quarter of 317 well-known sites are unreadable to AI crawlers — 47 of them serve JavaScript-only bodies, and the report notes that group almost certainly did not choose it.

The second reason is the difference between the two questions the tool separates. Classic SEO asks where you rank; answer engines ask whether a passage is worth quoting. A site can sit at position three on Google and never be quoted, while an unlinked page with clean, self-contained paragraphs gets cited daily. The rubric weights Content Citability highest on purpose — answer engines retrieve passages, not domains — and keeps the citation outcome out of the score, because a site can score high and still lose every answer to a competitor with more third-party coverage.

The third reason is calibration. Vague advice like "add structured data" is worthless without knowing which schema, which pages, and how many points it is worth. Every check in geo-score is tiered against a count out of eight sampled pages, so two people scoring the same site agree on the arithmetic, and the output ranks your biggest gaps by points to gain. You get a prioritized punch list, not a lecture.

## How It Works

The repository is a deliberately small core — a single-file engine and its rubric — surrounded by four interfaces, a test suite with real fixture sites, and a public benchmark with reproducibility machinery.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/geo-score/jianruntech-geo-score-architecture.svg" alt="Detailed architecture of the jianruntech/geo-score repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the jianruntech/geo-score repository, tracing rubric specification, the scoring engine, agent interfaces, tests, and the benchmark.*

### Understanding the Architecture

**The rubric is data, not folklore.** rubric/v1.1.md specifies 21 tiered checks totalling 100 points across five pillars — Reachable 15, Understandable 22, Content Citability 35, Brand Credibility 18, Answer Fit 10 — plus 4 bonus checks worth up to 6 points outside the denominator. rubric/v1.1.json encodes the same rules for the engine to load, and companion documents record the calibration reasoning, the evidence each check requires, and the open questions the maintainers still debate. The machine rubric feeds directly into cli/geo_score.py, the scoring engine.

**Three checks are gates, and that changes strategy.** Score zero on crawler access, live reachability, or server-rendered content and the result caps at 40, because until a retrieval crawler can reach the content nothing else matters. This single rule reorders the punch list for most sites: fix the gates before decorating the schema.

**The engine fetches like a retrieval crawler.** The CLI samples pages from the target site, checks robots.txt against a list of AI user-agents, tests reachability, inspects structured data and llms.txt, and reads the actual text for answer-shaped passages, sourced statistics, and bylines. Four checks require off-site search or human judgement, and rather than guessing, the CLI drops them from the denominator and names them — you can see exactly what was and was not measured.

**Interfaces meet you where you work.** action.yml runs the score on every push with a fail-under threshold, rendering its summary through action/summary.py; SKILL.md turns the tool into a Claude Code skill that also performs the four judgement checks; and the same file exposes an MCP server entry point — seven tools from score_site to diff, plus the rubric as a resource — with a --read-only flag that guarantees the server cannot spend your API keys. guide/mcp.md documents setup for Claude Code, Cursor, VS Code, Codex, Gemini CLI and others.

**Levels two and three measure outcomes.** cli/geo_watch.py adds the paid half: ask one question of OpenAI, Perplexity, Gemini, Anthropic, or OpenRouter models with your own keys, and track a fixed question set week over week, with runs, diffs, and noise thresholds documented in guide/watch-methodology.md. The README's warning is worth repeating: a desktop agent client does not see your shell's exported keys, so config must point at absolute paths.

**The benchmark is part of the design.** benchmark/run.py drives the engine across benchmark/sites.json and writes every raw report into the repository; benchmark/stats.py computes the medians and confidence intervals; benchmark/validity.py compares CLI tiers against the five hand-scored audits in examples/audits/v1.1; and benchmark/REPRODUCIBILITY.md publishes the rerun experiment — the project reports 96 percent of sites land within 5 points of their first score. The headline findings: a median of 56 across 317 sites, a quarter of sites failing a gate check, and China-market sites scoring roughly 20 points lower on average.

End to end: the versioned rubric defines the checks, the engine gathers evidence from a site the way a retrieval crawler would, the interfaces deliver the same measurement in a terminal, CI, or an agent conversation, and the benchmark proves the whole instrument works — including where it disagrees with humans.

## Advantages

- **Zero dependencies, one file.** Python 3.8 standard library only; the free level runs from a single curl-piped command with nothing to install or break.
- **A published, versioned rubric.** Every check, tier, and point value is public prose plus machine-readable JSON, with calibration notes and open questions — no black box scoring.
- **Gates encode real causality.** Crawler access, reachability, and server-rendered content cap the score at 40, matching the fact that unreachable content cannot be cited regardless of quality.
- **Honest about uncertainty.** Unmeasurable checks leave the denominator instead of being guessed, and the reproducibility and validity studies are published alongside the benchmark.
- **Same engine everywhere.** CLI, GitHub Action, Claude Code skill, and MCP server all drive one implementation, so a score means the same thing in every context.
- **Spending is fenced.** The paid levels require your keys, the watch config uses absolute paths, and a --read-only MCP flag removes spending capability entirely.

## Benefits

- **A prioritized fix list in twenty seconds.** The output names the biggest gaps with their point values — headings that match questions, verifiable authorship, freshness signals — so effort goes where points are.
- **Citation truth, not just readiness.** Levels two and three answer whether the engines actually cite you for questions you care about, tracked week over week with diffs.
- **Portable across agent ecosystems.** Claude Code skill, Claude plugin, Gemini CLI extension, and a standards-shaped MCP server mean the tool follows whichever agent you use.
- **A replicable benchmark methodology.** Raw per-site reports, statistics, validity hand audits, and rerun experiments form a template for anyone publishing measurements, not just claims.
- **Benchmark findings you can act on.** The public data — a quarter of major sites unreadable to crawlers, JS-only bodies as the largest invisible failure — is immediately useful content for your own team's argument.
- **MIT licensed and self-verifying.** Unit, end-to-end, and watch test suites with fixture mini-sites and a golden report pin the engine's behavior for contributors.

## Usage

The fastest path is the one-liner from the README (Python 3.8 or newer, nothing to install):

```bash
curl -sL https://raw.githubusercontent.com/jianruntech/geo-score/v1.4.0/cli/geo_score.py \
  | python3 - stripe.com --brief
```

From a clone, the CLI's useful flags cover evidence, JSON, comparison, and badges:

```bash
git clone https://github.com/jianruntech/geo-score.git
python3 cli/geo_score.py example.com            # human-readable report
python3 cli/geo_score.py example.com --explain  # evidence behind every check
python3 cli/geo_score.py example.com --json     # schema-conforming report
python3 cli/geo_score.py example.com --compare competitor.com
python3 cli/geo_score.py example.com --badge aiv-badge.svg
```

As a GitHub Action that fails the build when readiness regresses:

```yaml
- uses: jianruntech/geo-score@v1
  with:
    url: https://example.com
    fail-under: 40
```

As a Claude Code skill that also performs the four judgement checks:

```bash
git clone https://github.com/jianruntech/geo-score ~/.claude/skills/geo-score
# then: /geo-score audit https://example.com
```

And as an MCP server for any agent client:

```bash
claude mcp add --scope user geo-score -- uvx --from git+https://github.com/jianruntech/geo-score@v1.4.0 geo-score-mcp
```

Levels two and three — asking the engines and weekly watch — add cli/geo_watch.py next to the CLI, an API key from any supported provider, and a geo-score-watch.json config passed by absolute path.

## Conclusion

geo-score does for answer-engine visibility what lighthouse-style tools did for page performance: it takes a fuzzy, contested topic and turns it into versioned checks, reproducible arithmetic, and published evidence — then refuses to hide its own measurement error. The rubric's bets are legible (passages beat domain authority; gates come first; citations are a separate metric), the benchmark is refreshingly embarrassing for the big sites that fail its gates, and the whole thing runs from one dependency-free file. If your site's future includes being quoted by machines, this repository tells you, in points, what stands in the way.

Links:

- GitHub repository: https://github.com/jianruntech/geo-score
- Rubric v1.1: https://github.com/jianruntech/geo-score/blob/main/rubric/v1.1.md
- Public leaderboard: https://www.jianruntech.com/leaderboard
- MCP setup guide: https://github.com/jianruntech/geo-score/blob/main/guide/mcp.md
- Reproducibility report: https://github.com/jianruntech/geo-score/blob/main/benchmark/REPRODUCIBILITY.md
