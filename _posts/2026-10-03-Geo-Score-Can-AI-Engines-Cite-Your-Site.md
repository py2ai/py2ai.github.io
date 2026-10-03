---
layout: post
title: "geo-score: Can AI Engines Cite Your Site? - Inside jianruntech/geo-score"
description: "A source tour of geo-score, a zero-dependency Python toolkit that scores any website 0-100 on AI answer-engine visibility against a published rubric, then tracks whether ChatGPT, Perplexity, Gemini and Claude actually cite it - with an MCP server, a GitHub Action and a 324-site public benchmark."
date: 2026-10-03
header-img: "img/post-bg.jpg"
permalink: /Geo-Score-Can-AI-Engines-Cite-Your-Site/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/geo-score/jianruntech-geo-score-architecture.svg
tags:
  - GEO
  - SEO
  - MCP
  - Python
categories: [AI, Open Source]
keywords: "generative engine optimization, AI visibility, answer engine optimization, llms.txt, AI crawlers, MCP server, citation tracking, site audit, Python CLI, GitHub Action, structured data, content citability"
author: "PyShine"
---

Search is quietly splitting in two. One half ranks pages the way it always has. The other half answers: ChatGPT, Perplexity, Google AI Overviews, Gemini and Copilot retrieve passages, judge whether a source is worth quoting, and either cite you or silently skip you. A site can sit comfortably at position three on Google and never appear in a single AI answer, while a page nobody links to gets quoted daily because its paragraphs are clean and self-contained. The practices behind that outcome have a name now - Generative Engine Optimization, GEO for short - and until recently there was no open, inspectable way to measure where you stand.

That is the gap [jianruntech/geo-score](https://github.com/jianruntech/geo-score) fills. It is a free, standard-library-only Python toolkit that scores any public site 0-100 on whether AI engines can reach, parse, trust and cite it - against a published, versioned rubric rather than a black box. One command, no API key, nothing to install: `python3 cli/geo_score.py example.com` samples your pages and prints a tiered breakdown of every check. Two paid, opt-in levels add the question the rubric deliberately keeps out of the number: do the engines actually cite you, and is that changing week over week?

The source is worth a tour because it is unusually honest engineering. Every check is documented in `rubric/v1.1.md` with evidence and calibration notes. The fetch-and-score engine lives in one readable file, `cli/geo_score.py`. Output conforms to versioned JSON schemas under `schema/`. And the authors publish their own benchmark - 324 well-known sites scored in public, with reproducibility data - so you can see how the tool behaves before trusting it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/geo-score/jianruntech-geo-score-overview-architecture.svg" alt="Architecture overview of the jianruntech/geo-score repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the geo-score repository: the scorer CLI and watch CLI form the core, the rubric and schemas form the published contract, and delivery surfaces - the GitHub Action, MCP bundle and Claude plugin - wrap the same engine.*

Reading the overview from left to right: everything funnels through `cli/geo_score.py`, the scorer CLI that loads the versioned rubric from `rubric/v1.1.json` and emits reports conforming to `schema/report.v2.json`. For citation measurement it lazily loads `cli/geo_watch.py`, the watch CLI that talks to the AI engines' APIs and keeps statistics honest. Delivery surfaces - the composite GitHub Action in `action.yml`, the MCP bundle manifest in `mcpb/manifest.json`, and the Claude Code plugin under `.claude-plugin/` - all wrap that same engine rather than reimplementing it. The benchmark directory proves the tool against 324 real sites, and three test suites cover scorer, watcher and end-to-end behavior.

## Why You Need This

The uncomfortable finding in the repository's own benchmark is that a quarter of 324 well-known sites are effectively unreadable to AI crawlers. Seventy-three sites have a gate check at zero: a retrieval crawler cannot get the content, so there is nothing for an engine to quote. Twenty of them block AI crawlers by name in `robots.txt` - an editorial choice, and the report says so - but forty-one serve pages whose body only exists after JavaScript runs. That last group almost certainly did not choose it. The content is there; a browser sees it; the crawler gets an empty shell.

These are exactly the failures a traditional SEO dashboard never surfaces, because ranking and retrievability are different questions. Answer engines do not rank - they retrieve passages and decide whether to quote them. The failure modes are mechanical and mostly cheap to fix: a `robots.txt` line, a JSON-LD block, a date in a template, a paragraph rewritten so it stands on its own. The hard part has always been knowing which of them you are missing and what each is worth. geo-score turns that unknown into a numbered, tiered checklist with points attached.

There is also a trust problem with any score, and the project attacks it head-on. The rubric is published and versioned, with three scoring gates that cap the total at 40 when a crawler cannot even reach the content - because nothing else you fix matters until that is true. The four checks a static fetch cannot observe (third-party listings, independent mentions, question coverage, Chinese-engine readiness) are left out of the denominator rather than guessed. The CLI's agreement with five hand-scored audits is published in `benchmark/VALIDITY.md`, check by check, including where it disagrees.

Finally, the two-halves design matters for your budget. The score is free and offline-cheap: it reads public URLs only. Citation measurement requires paid API calls, so it is a separate, explicit step that uses your keys and is reported next to the 100-point score, never summed into it. A site can score 90 and still lose every answer to a competitor with more third-party coverage - the tool is built so you see both truths.

## How It Works

One intro sentence does a lot of work here: the entire level-1 engine is a single dependency-free Python file, and everything else in the repository is either published contract, delivery wrapper, or proof.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/geo-score/jianruntech-geo-score-architecture.svg" alt="Detailed architecture of the jianruntech/geo-score repository" style="max-width:100%;height:auto;" />
</div>

*The detailed view: the scoring engine and its fetch layer, the watch engine with engine adapters and statistics, the rubric and schema contract, delivery surfaces, and the benchmark-plus-tests that keep both honest.*

### Understanding the Architecture

**The fetch layer is defensive by design.** Inside `cli/geo_score.py`, a guarded HTTP stack (`_GuardedHTTP`, `_GuardedHTTPS`, `fetch`, `fetch_steady`) wraps `urllib` with timeouts, a bounded body reader, gzip inflation limits, redirect handling and a `FetchMemo` cache so one run never re-fetches the same URL. A `challenge_kind` helper recognizes bot-challenge pages, which matters because gate checks can flip when protection software answers a crawler differently - the reproducibility study in `benchmark/REPRODUCIBILITY.md` documents exactly that instability.

**Page understanding is hand-rolled HTML analysis.** Functions like `ld_blocks`, `jsonld` and `resolve_refs` extract and merge JSON-LD structured data; `visible_text`, `text_blocks`, `headings` and `paragraphs` strip chrome and reduce the page to analyzable text; `stat_reading` looks for statistics and whether they carry a source; `robots_groups` and `robots_rules` parse crawler access per user-agent token. Language detection (`page_lang`, `audience_language`) lets the same heuristics read Chinese pages in their own language, added in the 1.5.0 line after the benchmark exposed a measurement gap.

**Scoring runs 21 tiered checks over a sampled crawl.** The `run` function discovers around eight representative pages (`discover` walks sitemaps and links), applies every check in the five pillars - Reachable 15 points, Understandable 22, Content Citability 35, Brand Credibility 18, Answer Fit 10 - and `score` normalizes the result into the 0-100 scale with band names from Not started to Leading. Each check has two to four tiers with named point counts, so two people scoring the same site agree on the arithmetic. Output writers follow: human report, `--explain` evidence, `--json` against `schema/report.v2.json`, SARIF, JUnit XML, badges, and a `diff` subcommand backed by `schema/diff.v1.json`.

**The watch engine measures the outcome, not the readiness.** `cli/geo_watch.py` carries one adapter per engine - `call_openai`, `call_perplexity`, `call_gemini`, `call_anthropic`, `call_openrouter` - each reading its key from the server environment via `key_for` and redacting secrets in logs. Runs are configured by a `geo-score-watch.json` validated against `schema/watch-config.v1.json`, write run files under `.geo-score/watch/runs/` conforming to `schema/watch.v1.json`, and the statistics module is properly serious: Wilson intervals, McNemar and sign-flip tests, Newcombe paired intervals and Holm correction, so week-over-week changes come with honest uncertainty rather than noise-chasing.

**Delivery surfaces wrap, never fork.** The composite action in `action.yml` runs the CLI once per job, then `action/summary.py` turns the same report into a job summary, annotations and optional SARIF, with inputs like `fail-under`, `fail-on-gate`, `baseline` and per-check `assert` expressions. The MCP server (`mcp_main`, also the `geo-score-mcp` entry point) exposes `score_site`, `ask`, `run`, `list_runs`, `report`, `diff` and `status` over stdio, plus the rubric as a resource and five prompts including `audit_site` and `weekly_watch`. The Claude Code plugin under `.claude-plugin/` adds the four judgement-based checks a static fetch cannot do.

**End to end:** you run `geo_score.py stripe.com`; the fetch layer samples and retrieves pages defensively, the HTML analysis layer reduces each page to structured signals, the 21 checks tier every signal against the published rubric, and the score plus its biggest gaps land in your terminal, a JSON file, a SARIF upload or a badge. When you want outcomes, `watch run` asks your tracked questions across five engines on a schedule, stores versioned run files, and `diff` tells you which engines changed and whether the change is outside the noise.

## Advantages

- **Zero dependencies.** Python 3.8+ standard library only - the CLI is genuinely one file, so the `curl | python3` one-liner works and there is no supply chain to audit.
- **Published rubric, not a black box.** Every check, tier and point value is specified in `rubric/v1.1.md` with evidence and calibration notes, and versioned JSON under `rubric/` keeps it machine-readable.
- **Honest about uncertainty.** Unobservable checks leave the denominator, gates cap the score at 40, and the benchmark publishes validity and reproducibility studies - including the tool's own disagreement with hand audits.
- **Three levels with clean separation.** Free readiness scoring, paid citation checks, and scheduled weekly watching are separate surfaces with the citation half never mixed into the 100-point score.
- **CI-native.** The composite action supports baselines, per-check assertions, `fail-on-drop` semantics, SARIF and JUnit output, so regression gates in pull requests are a few YAML lines.
- **Agent-ready.** A stdio MCP server, an MCP bundle manifest and a Claude Code skill mean your assistant can score, ask and watch on your behalf with read-only modes available.

## Benefits

- **Find the silent failures first.** The gate checks catch robots blocks, unreachable pages and JavaScript-only bodies - the failures where no other fix matters.
- **Prioritize by points, not vibes.** The report names the biggest gaps and the points each is worth toward the next band, so the fix list is ordered by return.
- **Prove improvement over time.** Baseline comparison, check-by-check diffs and weekly watch runs turn GEO from a one-off audit into a tracked discipline.
- **Keep budgets explicit.** Nothing spends money without your keys, a `--read-only` MCP mode exists, and dry runs report planned calls and caps before any spend.
- **Benchmark yourself against reality.** The public 324-site leaderboard, raw results and per-site reports in `benchmark/` give you a distribution to compare against, not just your own number.
- **Adopt it anywhere.** Run it as a CLI, a GitHub Action, an MCP server in Cursor or Claude, or a scheduled workflow - the same engine and the same report schema everywhere.

## Usage

Score any site with zero setup (Python 3.8+):

```bash
curl -sL https://raw.githubusercontent.com/jianruntech/geo-score/v1.6.0/cli/geo_score.py \
  | python3 - stripe.com --brief
```

Common CLI patterns from the README:

```bash
python3 cli/geo_score.py example.com --explain          # evidence behind every check
python3 cli/geo_score.py example.com --json             # schema/report.v2.json output
python3 cli/geo_score.py example.com --compare competitor.com
python3 cli/geo_score.py example.com --badge aiv-badge.svg
python3 cli/geo_score.py diff before.json after.json    # check-by-check changes
python3 cli/geo_score.py example.com --baseline before.json --fail-on-drop
```

Run the MCP server for agents (via uvx, no install):

```bash
claude mcp add --scope user geo-score -- uvx geo-score@1.6.0 mcp
```

```json
{
  "mcpServers": {
    "geo-score": {
      "command": "uvx",
      "args": ["geo-score@1.6.0", "mcp"]
    }
  }
}
```

Gate your builds with the composite action:

```yaml
- uses: jianruntech/geo-score@v1
  with:
    url: https://example.com
    fail-under: 40
    fail-on-gate: true
```

Install the Claude Code skill for the four judgement-based checks:

```bash
git clone https://github.com/jianruntech/geo-score ~/.claude/skills/geo-score
# then: /geo-score audit https://example.com
```

## Conclusion

geo-score is a rare kind of measurement tool: it tells you a number, shows exactly how the number is computed, publishes the evidence for every check, and reports its own error bars. The one-file, zero-dependency core makes it trivially adoptable, while the rubric, schemas, action, MCP server and benchmark make it serious enough to build a practice on. If AI answers are becoming a traffic surface you care about, this repository is the most inspectable starting point we have seen.

Links:

- Repository: [https://github.com/jianruntech/geo-score](https://github.com/jianruntech/geo-score)
- Rubric v1.1: [https://github.com/jianruntech/geo-score/blob/main/rubric/v1.1.md](https://github.com/jianruntech/geo-score/blob/main/rubric/v1.1.md)
- Troubleshooting guide: [https://github.com/jianruntech/geo-score/blob/main/guide/troubleshooting.md](https://github.com/jianruntech/geo-score/blob/main/guide/troubleshooting.md)
- MCP setup guide: [https://github.com/jianruntech/geo-score/blob/main/guide/mcp.md](https://github.com/jianruntech/geo-score/blob/main/guide/mcp.md)
