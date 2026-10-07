---
layout: post
title: "Limelit Open: Self-Hosted AI Visibility Tracking - Inside limelit-co/open"
description: "A source-level tour of Limelit Open, the Apache-2.0 Go binary that asks ChatGPT, Claude, Perplexity, Gemini, Google AI Overview, Google AI Mode and Bing Copilot about your brand, finds mentions by text search, stores everything in a SQLite file you own, and serves every number with the evidence behind it."
date: 2026-10-07
header-img: "img/post-bg.jpg"
permalink: /limelit-open-ai-visibility/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/limelit-open/limelit-co-limelit-open-architecture.svg
tags: [AI Visibility, AEO, Go, MCP]
categories: [AI, Open Source]
keywords: Limelit Open, AI visibility, AEO, GEO, share of voice, citation analysis, self-hosted analytics, SQLite, Go, MCP, ChatGPT, Perplexity, Google AI Overview
author: "PyShine"
---

Ask ChatGPT which GPU cloud to pick, or Perplexity which tool leads your category, and the answer names brands and cites sources within seconds. Whether your brand is among them has become a boardroom question, and Limelit Open is an open-source answer to it. Published by Limelit under the Apache-2.0 license, it is a single Go binary that tracks how ChatGPT, Claude, Perplexity, Gemini, Google AI Overview, Google AI Mode and Bing Copilot mention and cite your brand against your competitors. The binary creates and migrates its own SQLite database on first start and serves a dashboard at localhost:1515 - no Docker requirement, no Node build, no Postgres.

The practice answers to four names at once - AEO (Answer Engine Optimization), GEO (Generative Engine Optimization), AIO (AI Search Optimization) and LLMO (LLM Optimization) - and they all ask the same question: when someone asks an AI assistant about your category, does your brand come up? Most commercial platforms in this category are closed SaaS: you send prompts to their cloud, they send back a number, and how it was computed is theirs. Limelit Open takes the opposite side of that trade: prompts, competitors and every answer live in a SQLite file you own, mentions are found by text search rather than by a model deciding what it saw, and every metric is derived from stored rows you can recompute with a tool that ships in the repository.

The project is pre-release, with v0.1 being built in the open, and what works today is verified against a live instance: demo.limelit.co runs this repository unmodified, read-only, tracking a GPU cloud vendor against 21 competitors across six engines, with a banner naming the commit it runs. This tour walks the source tree with two diagrams drawn from the repository layout, from the CLI in cmd/limelit down to the dashboard templates in internal/ui.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/limelit-open/limelit-co-limelit-open-overview-architecture.svg" alt="Architecture overview of the Limelit Open repository, showing the limelit CLI, the setup wizard, the engine registry, the evaluation runner, the answer analyzer, the metrics engine and the SQLite-backed dashboard" style="max-width:100%;">
</div>
<p><em>Architecture overview of the Limelit Open repository, from the CLI to the dashboard.</em></p>

Reading the overview from left to right:

- The **limelit CLI** in [cmd/limelit/main.go](https://github.com/limelit-co/open/blob/main/cmd/limelit/main.go) owns the verbs - serve, mcp, run, export, login, upgrade, version - and [cmd/limelit/login.go](https://github.com/limelit-co/open/blob/main/cmd/limelit/login.go) fetches and saves a free Limelit Cloud key.
- The **setup wizard** in [internal/ui/wizard.go](https://github.com/limelit-co/open/blob/main/internal/ui/wizard.go) asks for your domain, reads your brand's name from your own site through [internal/siteinfo/siteinfo.go](https://github.com/limelit-co/open/blob/main/internal/siteinfo/siteinfo.go), takes up to five competitors, and fills starter prompts from [internal/promptpack/promptpack.go](https://github.com/limelit-co/open/blob/main/internal/promptpack/promptpack.go); [internal/target/target.go](https://github.com/limelit-co/open/blob/main/internal/target/target.go) writes what to track as engine:provider[:model][:online].
- The **engine registry** in [internal/provider/registry.go](https://github.com/limelit-co/open/blob/main/internal/provider/registry.go) resolves targets onto adapters: vendor APIs for ChatGPT, Claude, Perplexity and Gemini, and scrapers for the Google AI surfaces and Bing Copilot, which have no API at all.
- The **run pass** in [internal/runner/runner.go](https://github.com/limelit-co/open/blob/main/internal/runner/runner.go) asks every active prompt of every enabled target, enforces the runs_per_day ceiling before anything is spent, and writes each answer into the SQLite store in [internal/store/store.go](https://github.com/limelit-co/open/blob/main/internal/store/store.go).
- The **answer analyzer** in [internal/runner/analyzer.go](https://github.com/limelit-co/open/blob/main/internal/runner/analyzer.go) scans stored answers through [internal/mentions/mentions.go](https://github.com/limelit-co/open/blob/main/internal/mentions/mentions.go) for brand mentions with the rank of the list item they appear in, and classifies every cited source in [internal/citations/citations.go](https://github.com/limelit-co/open/blob/main/internal/citations/citations.go) as your own, a competitor's, social, informational or other.
- The **metrics engine** in [internal/metrics/metrics.go](https://github.com/limelit-co/open/blob/main/internal/metrics/metrics.go) turns those rows into visibility, share of voice, average position and citation share, and the dashboard in [internal/ui/app.go](https://github.com/limelit-co/open/blob/main/internal/ui/app.go) renders the numbers.

## Why You Need This

The first reason is that the numbers are auditable by construction. Mentions are found by text search over stored text, not by a model deciding what it saw, so nothing can be invented and nothing is missed because a model stayed quiet. Every metric is derived from stored rows and can be recomputed; the formulas and exclusions are published in [docs/methodology.md](https://github.com/limelit-co/open/blob/main/docs/methodology.md), and the code that enforces them is right here. A number you are going to put in a board deck should survive an audit, and this is the rare codebase in its category where you can run that audit yourself.

The second reason is ownership: your infrastructure, your keys, your bill. There is no pricing, no credits and no markup anywhere in the project, and no per-seat tax on looking at your own data. You bring your own provider keys and pay the engines directly, or start with a free Limelit Cloud allowance. Usage counters track calls and tokens per target per day, a hard runs_per_day ceiling (default 200) is checked before any spend, and because mention detection is a text search there is no per-answer model call - the only thing you pay for is the answer itself.

The third reason is honesty about what is measured. An answer from a vendor API with web search on is not the answer a person sees in ChatGPT; they are different surfaces, and every number carries which one it came from, labeled api or scraped and never averaged. The denominator rules are written down because this is where visibility metrics usually go wrong: prompts that name your own brand are tagged branded and left out of the headline number, and an answer surface that did not render is excluded rather than counted as a miss. Every result carries n, the number of answers it rests on, so a figure from four runs never gets read as a trend.

## How It Works

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/limelit-open/limelit-co-limelit-open-architecture.svg" alt="Detailed architecture of the Limelit Open repository, showing the CLI, configuration, engine providers, evaluation runner, analysis layers, metrics, HTTP and MCP surfaces, SQLite storage and the deployment tooling" style="max-width:100%;">
</div>
<p><em>Detailed architecture of the Limelit Open repository, including the provider adapters and the operations tooling.</em></p>

### Understanding the Architecture

**The CLI is the front door.** The binary in [cmd/limelit/main.go](https://github.com/limelit-co/open/blob/main/cmd/limelit/main.go) dispatches the verbs. What to track lives in limelit.yaml, read by [internal/config/config.go](https://github.com/limelit-co/open/blob/main/internal/config/config.go) and meant to be committed and diffed, so it never holds a key; credentials come from the environment through [internal/credentials/credentials.go](https://github.com/limelit-co/open/blob/main/internal/credentials/credentials.go), and anything pasted into the dashboard is encrypted at rest by [internal/secrets/secrets.go](https://github.com/limelit-co/open/blob/main/internal/secrets/secrets.go). The one-command move to the hosted product is [internal/upgrade/upgrade.go](https://github.com/limelit-co/open/blob/main/internal/upgrade/upgrade.go), which uploads everything, deletes nothing here, never sends a provider key, and re-runs safely after a dropped connection.

**Setup is a wizard, not a config file hunt.** The wizard in [internal/ui/wizard.go](https://github.com/limelit-co/open/blob/main/internal/ui/wizard.go) asks for your domain and reads your brand's name from your own site, so answers write "Kindle to PDF" and not "kindletopdf.com", then your category and up to five competitors, a starter prompt pack from [internal/promptpack/promptpack.go](https://github.com/limelit-co/open/blob/main/internal/promptpack/promptpack.go), and one key. The free route is limelit login, and that key reaches ChatGPT, Gemini, Perplexity and Google's AI surfaces inside a monthly allowance.

**Seven engines, six of eleven providers, one interface.** The registry in [internal/provider/registry.go](https://github.com/limelit-co/open/blob/main/internal/provider/registry.go) maps targets onto adapters: the API side covers ChatGPT through [internal/provider/openai.go](https://github.com/limelit-co/open/blob/main/internal/provider/openai.go), Claude through [internal/provider/anthropic.go](https://github.com/limelit-co/open/blob/main/internal/provider/anthropic.go), Gemini through [internal/provider/google.go](https://github.com/limelit-co/open/blob/main/internal/provider/google.go) and Perplexity through [internal/provider/perplexity.go](https://github.com/limelit-co/open/blob/main/internal/provider/perplexity.go), all with web search on; the scraped side covers the consumer surfaces through [internal/provider/dataforseo.go](https://github.com/limelit-co/open/blob/main/internal/provider/dataforseo.go) and [internal/provider/searchapi.go](https://github.com/limelit-co/open/blob/main/internal/provider/searchapi.go), because Google AI Overview, Google AI Mode and Bing Copilot have no API at all. [internal/provider/limelit.go](https://github.com/limelit-co/open/blob/main/internal/provider/limelit.go) is the managed-key adapter for the free allowance, and [internal/provider/catalog.go](https://github.com/limelit-co/open/blob/main/internal/provider/catalog.go) lists the providers still to build.

**The runner is where money is guarded.** The evaluation runner in [internal/runner/runner.go](https://github.com/limelit-co/open/blob/main/internal/runner/runner.go) runs every active prompt against every enabled target, on demand with limelit run or on a daily or hourly schedule inside limelit serve, with a hard runs_per_day ceiling checked before any spend and usage counters per target per day. When the matching rules change, [internal/runner/reanalyze.go](https://github.com/limelit-co/open/blob/main/internal/runner/reanalyze.go) reruns the analysis over stored answers without paying the engines again.

**Analysis is deterministic.** The analyzer in [internal/runner/analyzer.go](https://github.com/limelit-co/open/blob/main/internal/runner/analyzer.go) feeds stored answers to the matcher in [internal/mentions/mentions.go](https://github.com/limelit-co/open/blob/main/internal/mentions/mentions.go), which searches for the names you gave in the text that was stored, takes the rank from the enclosing list item when the answer is a ranked list, and resolves aliases through the phrase rules in [internal/mentions/phrase.go](https://github.com/limelit-co/open/blob/main/internal/mentions/phrase.go). Every cited URL is normalized and classified by host in [internal/citations/citations.go](https://github.com/limelit-co/open/blob/main/internal/citations/citations.go) as your own site, a tracked competitor, social, informational or other.

**Metrics are aggregations, nothing more.** The engine in [internal/metrics/metrics.go](https://github.com/limelit-co/open/blob/main/internal/metrics/metrics.go) defines visibility as the share of answers that mention the brand, share of voice as the brand's mentions over all tracked brands' mentions in the same answers, position as the mean rank where the brand appears in a ranked list, and citation share as the share of cited sources that are the brand's own site. The breakdowns in [internal/metrics/breakdowns.go](https://github.com/limelit-co/open/blob/main/internal/metrics/breakdowns.go) split visibility by question type and build the prompt by engine grid, and [internal/metrics/chats.go](https://github.com/limelit-co/open/blob/main/internal/metrics/chats.go) feeds the per-answer evidence views. Every figure carries n, and under 20 the interface says so.

**Serving is one process.** The HTTP surface in [internal/httpx/server.go](https://github.com/limelit-co/open/blob/main/internal/httpx/server.go) hosts the dashboard, the JSON API and MCP over streamable HTTP with a bearer token generated in Settings. The dashboard is embedded in the binary: templates, CSS and handlers under [internal/ui/app.go](https://github.com/limelit-co/open/blob/main/internal/ui/app.go), with [internal/ui/measure.go](https://github.com/limelit-co/open/blob/main/internal/ui/measure.go) rendering every answer in full with your brand marked at the matcher's own offsets, the searches the engine ran, the brands named with their ranks, and every source it cited, and [internal/ui/settings.go](https://github.com/limelit-co/open/blob/main/internal/ui/settings.go) managing keys, engines and the schedule. The MCP server in [internal/mcpserver/server.go](https://github.com/limelit-co/open/blob/main/internal/mcpserver/server.go) exposes fourteen read tools on Limelit Cloud's names, so a conversation written against this server keeps working after an upgrade. The exporter in [internal/export/export.go](https://github.com/limelit-co/open/blob/main/internal/export/export.go) streams the whole instance as one JSON document or a directory of CSVs.

**Storage stays boring.** The store in [internal/store/store.go](https://github.com/limelit-co/open/blob/main/internal/store/store.go) opens a pure-Go SQLite database with embedded migrations from [internal/store/migrations/](https://github.com/limelit-co/open/tree/main/internal/store/migrations); the query layer lives in [internal/store/repo.go](https://github.com/limelit-co/open/blob/main/internal/store/repo.go) and pass rows are recorded by [internal/store/runs.go](https://github.com/limelit-co/open/blob/main/internal/store/runs.go). LIMELIT_DATA_DIR sets where the file lives (default ./data). [deploy/litestream.yml](https://github.com/limelit-co/open/blob/main/deploy/litestream.yml) replicates the database, and [tools/recompute/main.go](https://github.com/limelit-co/open/blob/main/tools/recompute/main.go) re-derives every figure from the stored rows.

**Operations are the repository's quiet half.** The [Dockerfile](https://github.com/limelit-co/open/blob/main/Dockerfile) builds the one-container image, and setting LIMELIT_DEMO=1 turns the same binary into a read-only public instance where every button that would change or spend anything refuses with an explanation. The engines being measured are enumerated in [internal/engines/engines.go](https://github.com/limelit-co/open/blob/main/internal/engines/engines.go). The plugins/ directory ships a Claude plugin marketplace: limelit-open connects this install over stdio with the limelit-open-visibility-report skill, and limelit-audit carries ai-crawler-check and llms-txt-check, two skills that audit a robots.txt and an llms.txt.

Follow one answer end to end and the layers connect like this: the CLI resolves targets through the registry, the runner spends under the ceiling and stores each answer in SQLite, the analyzer turns that text into mention and citation rows, the metrics engine aggregates the rows into the four numbers plus their breakdowns, and the results surface in three places at once - the dashboard on localhost:1515, the JSON API, and the MCP tools an assistant reads. Export and upgrade hang off the same store, so moving data out or up is a command, not a migration project.

## Advantages

- **One binary, zero ceremony.** No Docker, no Node, no Postgres, no cgo; it creates and migrates its own SQLite database on first start and serves the dashboard from the same process.
- **Auditable numbers.** Mentions by text search, metrics from stored rows, formulas and exclusions published, figures recomputable with a tool that ships in the repository.
- **Your keys, your bill.** Bring your own provider keys or start on the free Limelit Cloud allowance; no markup, no credits, no per-seat tax on your own data.
- **Hybrid measurement, labeled.** Vendor APIs and consumer-surface scrapers behind one interface, marked api or scraped on every metric and never averaged together.
- **MCP-first access.** Fourteen read tools on stable Cloud names over stdio or streamable HTTP.
- **Honest denominators.** Branded prompts excluded, non-rendered answer surfaces excluded, and every figure carries the sample it rests on.

## Benefits

- **A number you can put in a board deck.** Visibility, share of voice, average position and citation share, plus the competitor ranking, the trend and the prompt by engine grid.
- **Evidence behind every point.** Every answer readable in full with your brand marked in the text, the searches the engine ran, the brands named with ranks, and every source cited and classified.
- **Data you own.** One SQLite file you back up by copying; limelit export streams everything as JSON or CSVs whenever you want out.
- **Spend you control.** Usage counters per target per day and a hard ceiling refused before anything is spent, with no per-answer model call inflating the bill.
- **A free on-ramp.** A free Cloud key inside a monthly allowance reaches ChatGPT, Gemini, Perplexity and Google's AI surfaces.
- **A path up without lock-in.** limelit upgrade uploads everything to Limelit Cloud, deletes nothing, sends no provider keys, and re-running imports nothing twice.

## Usage

Quick start, nothing to install first:

```bash
mkdir -p ~/limelit && cd ~/limelit
curl -fsSL https://github.com/limelit-co/open/releases/latest/download/limelit_$(uname -s | tr A-Z a-z)_$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/').tar.gz | tar xz limelit
./limelit serve
```

Then open http://localhost:1515. The same binary two other ways:

```bash
# With Go 1.25 or newer
go install github.com/limelit-co/open/cmd/limelit@latest
limelit serve

# With Docker; the database lives in the limelit volume
docker run -p 1515:1515 -v limelit:/data ghcr.io/limelit-co/open
```

limelit login saves a free Cloud key; the wizard takes your domain, brand, competitors and one key, then you press Run and answers appear on the Overview as each engine replies.

Connect Claude Code and ask in plain language:

```bash
claude mcp add limelit -s user -- sh -c 'cd ~/limelit && exec ./limelit mcp'
```

Ask "Get started with Limelit" or "How visible is my brand in AI answers?" - the tools are the same ones the dashboard reads, so the assistant's number is the screen's number.

limelit.yaml holds what to track - property, competitors, targets like chatgpt:openai:gpt-5.5:online, limits such as runs_per_day: 200, and a schedule of daily, hourly or off - and it is meant to be committed and diffed. The full verb list:

```text
limelit serve     dashboard, JSON API, MCP over HTTP, and the scheduler
limelit mcp       MCP over stdio, for Claude Desktop and Claude Code
limelit run       one evaluation pass, then exit
limelit export    write everything this instance knows to stdout
limelit login     get a free Limelit Cloud key and save it here
limelit upgrade   move this instance to Limelit Cloud
limelit version   version and build info
```

One target on its own, and the Claude plugin marketplace:

```bash
limelit run --target chatgpt:openai:online
```

```text
/plugin marketplace add limelit-co/open
/plugin install limelit-open@limelit
```

Development needs Go 1.25 or newer and no other toolchain:

```bash
make build   # bin/limelit
make test    # go test ./...
make lint    # gofmt + go vet
make run     # build, then serve on :1515
```

## Conclusion

Limelit Open occupies an unusual position in its category: the README's own comparison table lists the commercial platforms it sits next to, and on the axes that matter to an engineer - open source, self-hostable, auditable metrics, data ownership, bring-your-own keys - it is the only one in the table that answers yes across the row. It is honest about the trade: several commercial tools offer prompt volume estimates, sentiment analysis and on-page optimization that this project deliberately does not. If what you want is a defensible, recomputable number about how AI assistants see your brand, with the evidence one click behind every point, this is the codebase to read.

Links:

- Repository: [github.com/limelit-co/open](https://github.com/limelit-co/open)
- Live demo: [demo.limelit.co](https://demo.limelit.co)
- Tool catalog: [docs/tools.md](https://github.com/limelit-co/open/blob/main/docs/tools.md)
- Methodology: [docs/methodology.md](https://github.com/limelit-co/open/blob/main/docs/methodology.md)
- Provider guide: [docs/providers.md](https://github.com/limelit-co/open/blob/main/docs/providers.md)
- License: Apache-2.0 ([LICENSE](https://github.com/limelit-co/open/blob/main/LICENSE), [NOTICE](https://github.com/limelit-co/open/blob/main/NOTICE))
