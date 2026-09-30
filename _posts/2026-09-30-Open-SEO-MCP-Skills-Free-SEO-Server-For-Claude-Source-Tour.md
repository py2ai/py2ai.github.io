---
layout: post
title: "Open SEO MCP Skills: Free SEO Server + GEO Skills for Claude - Inside Ryze-AI-Adgent/open-seo-mcp-skills"
description: "A source tour of Ryze-AI-Adgent/open-seo-mcp-skills: eight MIT-licensed SEO and GEO skill packs for Claude that run on a free MCP connector wiring Google Search Console, GA4, Google Ads and DataForSEO into keyword research, rank tracking, site audits, backlinks, competitor gaps and AI visibility workflows."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Open-SEO-MCP-Skills-Free-SEO-Server-For-Claude-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/open-seo-mcp/ryze-ai-adgent-open-seo-mcp-architecture.svg
tags:
  - SEO
  - MCP
  - Claude
  - Open Source
categories: [AI, Open Source]
keywords: "SEO MCP server, Claude skills, open source SEO, Google Search Console, GA4, DataForSEO, keyword research, rank tracking, backlink check, AI visibility, GEO, Claude Code plugin, competitor gap, content brief, SEO audit"
author: "PyShine"
---

Ask an "open-source SEO tool" where its numbers come from and the answer is usually the same: a paid SERP API behind the scenes, with the free part being little more than a dashboard on top. Ryze-AI-Adgent/open-seo-mcp-skills takes a different route entirely. It is a collection of eight Markdown-encoded SEO and GEO skill packs for Claude, shipped as a Claude Code plugin and wired to a free MCP connector that exposes Google Search Console, GA4, Google Ads, and DataForSEO as callable tools. The skills are MIT-licensed; the interesting part is not a UI but the workflows themselves, written out step by step for a model to execute.

What makes the repository worth a source tour is its economy. There is no server code to run, no database, no frontend. The whole project is two JSON manifests, one short Shell installer, and eight SKILL.md files under `skills/`, each of which turns a handful of MCP tool calls into a complete professional routine: full site audits, keyword research with real Google Ads volumes, rank tracking without a tracker subscription, competitor keyword gaps, backlink profiles, AI-engine visibility, SERP-driven content briefs, and an ads-versus-organic waste analysis. Reading the sources is a lesson in how much of a paid SaaS product can be expressed as a well-structured prompt plus the right data plumbing.

It is also a clean example of the skills pattern for anyone building on the Model Context Protocol. Every skill follows the same contract: it states what it requires, enumerates an ordered workflow with the exact namespaced tool calls (`google_search_console__runRawSearchAnalytics`, `google_ads__generateKeywordIdeas`, and friends), prescribes an output format, and names its fallbacks when a provider is not connected. That discipline is why this small repository punches above its file count.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/open-seo-mcp/ryze-ai-adgent-open-seo-mcp-overview-architecture.svg" alt="Architecture overview of the Ryze-AI-Adgent/open-seo-mcp-skills repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the open-seo-mcp-skills repository: the plugin packaging layer on the left, the eight skill packs on the right.*

Reading the overview from left to right: the packaging layer begins with `.claude-plugin/marketplace.json`, which lists the plugin as an installable entry in a marketplace named `ryze`; `.claude-plugin/plugin.json` is the heart of that layer, declaring version 0.2.0, the path to the skills folder, and — critically — an MCP server entry of type `http` pointing at `https://connector.get-ryze.ai/mcp`. The `install.sh` script is the manual route, copying each pack under `skills/` into `~/.claude/skills` so Claude Code picks them up globally. The right side shows the product itself: `skills/` contains eight SKILL.md packs, and each one is a self-contained workflow that reaches back through the declared MCP server to pull its data at runtime.

## Why You Need This

The core problem this repository attacks is the gap between "open source" and "free data". Plenty of SEO projects publish their code but leave you paying per request to a SERP scraping API, and the numbers you get back — rankings and traffic — are estimates reconstructed from scraped result pages. The skill packs here never treat estimates as the primary source for your own site. Your positions come from Search Console, which records the position Google actually served; your traffic and conversions come from GA4; your keyword volumes come from Google Ads keyword planner data, which is the same source commercial tools resell.

The second problem is workflow, not data. Raw API access is not an SEO service: knowing that a `runRawSearchAnalytics` endpoint exists does not tell you to compare the last 90 days against the previous 90, split dimensions into query and page passes, and cross-check landing-page conversions before declaring a page decayed. Each SKILL.md encodes exactly that kind of professional judgment — thresholds, bucketing rules, prioritization logic, even the instruction to sort losers by clicks lost rather than position delta, because a small drop on a money query matters more than a big drop on a page nobody visits.

Third, there is the newer GEO question: do AI engines send you traffic, and what do they cite? Most "AI visibility trackers" sample prompts and extrapolate. The `ai-visibility` skill starts from what GA4 actually records — sessions referred by ChatGPT, Perplexity, Gemini, Claude and Copilot — then reads the pages that earn those clicks to extract a "citation recipe" you can apply elsewhere on the site. Measuring the real thing before diagnosing is a refreshingly honest ordering.

Finally, this solves the tooling fragmentation problem. Because the connector exposes Search Console, GA4, Google Ads, and DataForSEO under one MCP server with namespaced tools, a single sentence in Claude — "run an SEO audit on mysite.com" — fans out across four data sources that would otherwise live in four dashboards and three exports.

## How It Works

The entire system is a thin, disciplined layer of Markdown workflows over a hosted MCP connector.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/open-seo-mcp/ryze-ai-adgent-open-seo-mcp-architecture.svg" alt="Detailed architecture of the Ryze-AI-Adgent/open-seo-mcp-skills repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: each skill pack declares which provider tools it calls through the ryze MCP server, while the README, installer and MIT license round out the package.*

### Understanding the Architecture

**The plugin manifest is the wiring.** `.claude-plugin/plugin.json` does three jobs at once: it names and describes the plugin (`open-seo-mcp-skills`, version 0.2.0), it points the `skills` key at `./skills`, and it declares an `mcpServers` block that registers the Ryze connector as an HTTP MCP server at `https://connector.get-ryze.ai/mcp`. That last part is why the README says adding the connector manually is optional in Claude Code: installing the plugin bundles the server declaration, so the data connection arrives with the skills. `.claude-plugin/marketplace.json` wraps this in a marketplace entry named `ryze`.

**Each skill is a numbered workflow, not a prompt fragment.** Open any pack, say `skills/seo-audit/SKILL.md`, and you get a strict contract: a frontmatter name and trigger description, a "Requires" section, a "Workflow" of numbered steps naming exact tools — `google_search_console__listSites` to pick the property, `getIndexationSummary` for indexed-versus-excluded counts, two `runRawSearchAnalytics` calls per period over `query` and `page` dimensions to build the performance baseline, then classification into decaying pages, CTR anomalies (position at or better than 5 but CTR under 2 percent), striking-distance queries at positions 8 through 20, and cannibalization. Spot checks use `inspectUrl`, and traffic cross-checks come from `google_analytics__runRawReport` plus `google_analytics__getAIReferrals`.

**Tool discovery is built in.** Ryze MCP tools are namespaced as `provider__tool`, and when a request shape is unclear the skills call `native__get_provider_docs` with a provider name such as `dataforseo` to retrieve the official API documentation at runtime. This is a pragmatic answer to the fact that tool availability can vary by workspace: every skill states its fallback explicitly. `skills/backlink-check/SKILL.md`, for example, prefers DataForSEO backlinks endpoints (`summary`, `referring_domains`, `anchors`), falls back to a connected `ahrefs` or `semrush` provider, and otherwise stops and says which connection is missing — never fabricating link counts.

**The data-source split is deliberate.** The packs treat your own properties and third-party indexes differently. Rank tracking and audits run exclusively on Search Console because it records served positions rather than scraped guesses. Competitor analysis inverts this: `skills/competitor-gap/SKILL.md` pulls the rival's estimated keywords from DataForSEO Labs (`ranked_keywords`, `domain_rank_overview`) and diffs them against your real GSC impressions, explicitly labeling which numbers are estimates and which are actuals in the output.

**Cross-source joins create the unique reports.** The most interesting pack, `skills/seo-vs-ads/SKILL.md`, joins a Google Ads search-terms report pulled via `google_ads__runRawGaql` against GSC organic queries over the same window, then buckets every term: double-paying (ranking top 3 organically while still spending, especially on brand terms), defensible (competitors are bidding), paid-only winners (converting terms with no organic ranking — a content roadmap pre-validated by money), and organic-only converters. A similar join powers `skills/content-brief/SKILL.md`, which combines the live DataForSEO SERP, Google Ads keyword metrics, and your site's existing GSC foothold into a brief with table stakes versus a differentiating angle.

**The end-to-end flow** is uniform across all eight packs. A user asks a question in Claude; the matching skill's trigger description activates; the workflow's first step picks or verifies the right property or account; the middle steps fire namespaced MCP calls through the connector declared in `plugin.json`, enriching with DataForSEO where available and skipping silently where not; the final step renders a prescribed output — verdict first, then tables, then a prioritized action list. `install.sh` is the only executable code in the repository: a short loop that copies each `skills/` subdirectory into `~/.claude/skills`, with the README noting the plugin install route as the primary path.

## Advantages

- **Real data instead of estimates.** Rankings come from Search Console's served positions and traffic from GA4, including AI-referral traffic — not from SERP scraping.
- **Zero-cost, MIT-licensed skills.** The LICENSE file is plain MIT (Copyright 2026, Ryze AI), so the packs can be copied, modified and redistributed; a file copy is the whole install.
- **No infrastructure to run.** There is no server, database or build step; the connector is a hosted HTTP MCP endpoint, and every artifact in the repo is Markdown or JSON.
- **Explicit fallback discipline.** Each skill names what happens when a provider is missing — check `ahrefs`/`semrush`, else stop and say which connection is missing — which keeps Claude from inventing numbers.
- **Tool self-documentation.** `native__get_provider_docs` returns official API docs for any provider on demand, so request shapes never need to be hardcoded in the skills.
- **Client portability.** Beyond Claude Code, the same connector URL works in claude.ai, Claude Desktop or Cursor, and the skills are plain Markdown that can be pasted into any client.

## Benefits

- **Audits that read like a consultant's.** `seo-audit` ends with a verdict paragraph, sections for indexation, losers and quick wins ranked by impressions, and a prioritized action list.
- **Keyword plans grounded in planner data.** `keyword-research` expands a seed to roughly 200 ideas, attaches volume, competition and CPC from Google Ads historical metrics, and clusters into pillar-plus-supporting plans.
- **Rank tracking without a subscription.** `rank-tracking` diffs two GSC windows into winners, losers, new and lost queries, and suggests scheduling it as a recurring task instead of paying a tracker.
- **Actionable competitive intelligence.** `competitor-gap` separates keywords you are behind on from true gaps and your moat, prioritized by volume and winnability.
- **A measurable answer on GEO.** `ai-visibility` reports which engines send traffic, which pages they cite, and which strong organic pages get zero AI referrals.
- **Money-saving joins.** `seo-vs-ads` quantifies spend on queries you already rank for organically and lists exact negatives to add and pages to build.

## Usage

Connect the free MCP connector (in Claude Code, one command; in claude.ai, Desktop or Cursor, add a custom connector with the same URL):

```
claude mcp add ryze --transport http https://connector.get-ryze.ai/mcp
```

Install the skills as a Claude Code plugin from the marketplace:

```
claude plugin marketplace add Ryze-AI-Adgent/open-seo-mcp-skills
claude plugin install open-seo-mcp-skills@ryze
```

Or install the packs manually with the included Shell script's logic — clone and copy:

```bash
git clone https://github.com/Ryze-AI-Adgent/open-seo-mcp-skills && cp -r open-seo-mcp-skills/skills/* ~/.claude/skills/
```

For the desktop and web clients, the README's manual connector setup is two fields: Name `Ryze AI`, URL `https://connector.get-ryze.ai/mcp`. After connecting your Search Console, GA4 and ads accounts once, just ask: "run an SEO audit on mysite.com", "what keywords does competitor.com rank for that I don't?", or "how much am I paying for clicks I'd get free?"

## Conclusion

Ryze-AI-Adgent/open-seo-mcp-skills is a small repository with a clear thesis: the valuable part of an SEO tool is the workflow and the data source, not the wrapper. Its eight skill packs encode real professional practice — thresholds, bucketing rules, fallbacks — on top of an MCP connector serving your own Search Console, GA4 and Google Ads data alongside DataForSEO's indexes. If you work in Claude Code and want SEO and GEO routines that cite your actual numbers, the source tour takes ten minutes and the install takes two commands.

Links:

- GitHub repository: https://github.com/Ryze-AI-Adgent/open-seo-mcp-skills
- Connector setup guide: https://www.get-ryze.ai/how-to-connect-claude-to-google-meta-ads-mcp
- License: MIT (see LICENSE in the repository)
