---
layout: post
title: "Agent Console: Local-First Observability for AI Coding Agents - Inside LockedinLabs-AI/agent-console"
description: "A source tour of LockedinLabs-AI/agent-console, a zero-dependency Node.js dashboard that turns Claude Code and Codex JSONL transcripts into live token, cache and cost telemetry. We walk the collector, the salted-hash privacy projection, the append-only hub store and the browser UI that tie it together."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Agent-Console-Local-First-Agent-Observability-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/agent-console/lockedinlabs-ai-agent-console-architecture.svg
tags:
  - AI Coding Agents
  - Observability
  - Claude Code
  - Open Source
categories: [AI, Open Source]
keywords: "agent console, LockedinLabs-AI, Claude Code, Codex, token usage tracking, cost tracking, local-first observability, JSONL transcript parsing, AI coding agents, Node.js dashboard, cache read write tokens, self-hosted telemetry"
author: "PyShine"
---

If you have been running Claude Code or Codex for a few months, your machine already holds a small mountain of JSONL transcripts, and almost nobody opens them. The usage blocks repeat, the files are long, and the answers you actually want — what did yesterday cost, which session is burning cache writes right now, which machine on the team is silent — are scattered across thousands of lines. Agent Console, from LockedinLabs-AI, is a purpose-built answer: a dashboard that reads the transcripts the agents already wrote and turns them into live lanes, token class breakdowns and list-price cost estimates.

The project calls itself open-source, local-first observability for AI coding agents, and the code backs that up in an unusual way. It is all JavaScript with zero npm dependencies — `package.json` ships an empty `dependencies` object and requires Node.js 22 or newer — so there is no build step and nothing to install beyond Node itself. One command against a checkout opens the console in your browser, normally at `http://127.0.0.1:6787`, already reading this machine's history, and other computers can join a self-hosted hub so every machine's usage lands in one view.

The source is worth a tour because it solves the hard parts honestly: parsing two vendors' streaming transcript formats without double-counting a message spread over several lines, pricing models that have no verified rate, and merging reports when the same transcript exists on two machines. The answers live in a small set of well-commented modules.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/agent-console/lockedinlabs-ai-agent-console-overview-architecture.svg" alt="Architecture overview of the LockedinLabs-AI/agent-console repository" style="max-width:100%;height:auto;" />
</div>

*Overview of Agent Console: transcript collection, the reporting transport, and the hub that stores and aggregates usage for the browser dashboard.*

Reading the overview from left to right: the thin CLI entry `bin/agent-console.mjs` either starts the console server from `server.js` or dispatches the reporter commands in `lib/reporter.js`. The collection group — `lib/collector` — walks the local `.jsonl` transcripts, parses each line into allowlisted usage records in `lib/collector/parsers.js`, and prices them against the offline table in `lib/collector/pricing.js`. A machine that is not the hub runs the reporter, posting record batches through `lib/collector/transport.js` to the hub's reporting listener over pinned TLS. On the hub side, `lib/hub/routes.js` ingests everything into the append-only store `lib/hub/store.js`, whose minute buckets feed the aggregation in `lib/hub/aggregate.js` and, finally, the browser UI in `public/console.js`.

## Why You Need This

The first problem is simple visibility. Claude Code and Codex each keep history in their own format — roughly `~/.claude/projects` and `~/.codex/sessions` — and neither answers "what did we spend today?". Agent Console reads both roots on every machine it watches and projects them into one shape: four disjoint token classes (fresh input, output, cache write, cache read) plus the model id and minute of each event, rendered as headline totals, burn rates, per-model and per-machine tables, and one lane per live session.

The second problem is trust in the numbers. Dashboards that silently drop what they cannot measure are worse than none, and this codebase is obsessive on that point. The store's header comment in `lib/hub/store.js` is literally titled "UNKNOWN IS NOT ZERO": a record missing a token class makes the displayed total a floor, and a model with no verified list price is excluded from the dollar figure rather than priced at zero. Silent machines show when they were last heard from, and uncountable records are tallied by reason beside the figures.

The third problem is privacy. Every record that leaves a reporting machine passes through one allowlisted projection — `projectRecord` in `lib/collector/collector.js` — so only token counts, a model id, a minute timestamp and salted hashes of session and project cross the wire; never prompts, replies, tool arguments, file paths or contents. The project proves it in code: `test/hub-e2e.test.js` pushes transcripts full of planted canaries through a real reporter and a real hub, records every byte on the wire, and checks that no canary escapes.

The fourth problem is multi-machine reality. Usage fragments across a laptop, a build box and a server, each keeping its own transcripts. Agent Console's hub-and-reporter model sends only counts to the hub, with record ids derived from transcript content rather than file location — so a copied transcript is counted once, not twice.

## How It Works

The pipeline in one sentence: transcripts are scanned incrementally, parsed into salted, allowlisted metadata records, delivered to a hub that keeps append-only daily files plus in-memory minute buckets, and the browser polls an aggregation of those buckets every two seconds.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/agent-console/lockedinlabs-ai-agent-console-architecture.svg" alt="Detailed architecture of the LockedinLabs-AI/agent-console repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of LockedinLabs-AI/agent-console, from the CLI and collector through the transport, hub store, analysis core, policy controls and browser scripts.*

### Understanding the Architecture

**The collector and its parsers.** `lib/collector/collector.js` walks the transcript roots, keeps a private cursor per file, and spools records before delivery so a crash can replay but never lose one. `lib/collector/parsers.js` handles the vendors' quirks differently: Claude's repeated usage blocks become per-message deltas, Codex's cumulative counters become per-event deltas, and copied parent history is excluded. Each record carries the four token classes (plus the 5-minute and 1-hour cache-write split when reported), a validated model id, and salted hashes instead of anything readable.

**The incremental scanner.** Re-reading tens of thousands of transcripts every two seconds would pin a core, so `lib/collector/scanner.js` does a full walk once a minute and, between walks, looks only where something can be happening: recently changed files are re-stat-ed, folders whose modification time moved are re-listed, and the sweep is sliced into a small per-pass time budget so the console keeps answering.

**The two-listener hub.** `server.js` creates the store and registry and mounts the two HTTP surfaces from `lib/hub/routes.js`. The console listener is loopback-only (default port 6787) and requires a signed-in browser via `lib/hub/admin.js`; the reporting listener (default port 6788) serves only the join page, the join exchange and token-checked ingestion. Join links carry the SHA-256 fingerprint of the hub's certificate from `lib/hub/tls.js`, and the reporter accepts that certificate only.

**The honest store.** `lib/hub/store.js` writes one append-only NDJSON file per UTC day under the state directory (default `~/.agent-console/hub`, retention 8 days, range 1–90) and keeps minute buckets in memory rather than a database, because records arrive in batches, are never updated, and every question is a sum over a retention-length window. Each record is priced once on arrival via `lib/collector/pricing.js` against the dated table in `lib/collector/prices.json`, and a daily rollup outlives the minute detail so 30-day views stay answerable.

**The aggregation and analysis core.** `lib/hub/aggregate.js` computes the whole screen in one pass over the buckets, with documented rules against each misleading figure it refuses to show. The pure analysis functions — context and cache health in `lib/analysis/context.js`, the orchestrator/subagent tree in `lib/analysis/agent-tree.js`, spend ratios in `lib/analysis/cost-outcome.js` — are exported as a versioned subpath (`@lockedinlabs/agent-console/analysis`). Live alerts in `lib/hub/alerts.js` reuse the same detection rules: repeated identical tool calls, burn spikes, and spending without an observed successful tool result.

**The dashboard and the extras.** `public/index.html` loads `public/console.js` — roughly 280 KB of dependency-free UI that polls every two seconds and renders lanes, charts, team and project views, with a presenting mode that replaces real names with stand-ins. `--interop` adds local Prometheus `/metrics` and OTLP or gateway ingest behind scoped credentials from `lib/hub/metrics-token.js`, and the policy compiler (`lib/policy/cli.js`, `lib/policy/hook.mjs`, `lib/policy/classify.mjs`) installs reversible Claude Code hooks that classify selected tool calls on a worker thread.

End to end: an agent writes a transcript line; the scanner notices the file changed; the parser emits zero or more salted, allowlisted records; on the hub's own machine they go straight to the store through `lib/hub/local.js`, while on any other machine the reporter spools them and posts batches of at most 500 records over pinned TLS, advancing its cursor only after a complete receipt; the store prices and appends each record, folds it into its minute bucket, and the next two-second poll serves the aggregated view to the browser.

## Advantages

- **Zero dependencies, zero build.** The server and UI run on the Node.js standard library; no `npm install`, no bundler, no account, and a tiny supply chain surface.
- **Local-first by construction.** Transcripts stay on their source machine; the wire carries only counts, model ids, minutes and salted hashes, verified by a canary test in `test/hub-e2e.test.js`.
- **Honest numbers everywhere.** Unpriced models, missing token classes, silent machines and dropped records are labelled as such instead of being folded into the totals.
- **Copied transcripts count once.** Record ids derive from transcript content and a shared salt, so duplicates deduplicate in the store's first-writer-wins logic.
- **Scales to real histories.** The incremental scanner keeps idle CPU low, and a joining reporter catches up its backlog in paced, resumable batches.
- **Team and fleet aware.** Machines join with single-use, certificate-pinned links and are revoked from the Team view; daily rollups keep 30-day views answerable.

## Benefits

- **See spend before it surprises you.** Burn per minute, dollars per hour, and a spend spectrum that surfaces models whose cost share dwarfs their token share.
- **Keep cache economics honest.** Cache read and write are disjoint classes with lifetime splits, and the context drill-down flags possible cache breaks and idle gaps via `lib/analysis/context.js`.
- **Watch agents, not just totals.** The agent tree shows an orchestrator and its subagents in place; live alerts flag repeated tool calls, burn spikes and stalled spending.
- **Present without leaking.** Presenting mode swaps every project, machine and person for stand-in names, backed by a multi-process privacy test.
- **Fit it into existing tooling.** A Prometheus `/metrics` endpoint, an included Grafana dashboard, and the versioned analysis subpath feed your own monitoring and scripts.
- **Own your data.** Everything lives under the hub's state directory, one file per day, deleted whole past retention — no third party sees your agents' metadata.

## Usage

Run from a source checkout (needs Node.js 22 or newer):

```sh
git clone https://github.com/LockedinLabs-AI/agent-console.git
cd agent-console
node bin/agent-console.mjs --open
```

To look around first without reading anything of yours, add `--demo`:

```sh
node bin/agent-console.mjs --demo --open
```

Install the published v0.4.1 package from a GitHub release archive instead of the npm registry:

```sh
npm install --global --ignore-scripts https://github.com/LockedinLabs-AI/agent-console/releases/download/v0.4.1/lockedinlabs-agent-console-0.4.1.tgz
agent-console --open
```

Open the reporting port to your network, then add another computer from the console's **Add a machine** button and run this on that machine:

```sh
node bin/agent-console.mjs --listen 0.0.0.0 --open
node bin/agent-console.mjs join '<join link>'
node bin/agent-console.mjs report
node bin/agent-console.mjs leave
```

Useful options include `--port` (6787), `--retention-days` (8), `--poll-ms` (2000), and `--claude-root` / `--codex-root` if your transcripts live elsewhere; `--help` prints the full list:

```sh
node bin/agent-console.mjs --help
node bin/agent-console.mjs join --help
```

## Conclusion

Agent Console is a small idea executed with unusual care: a collector that reduces two transcript formats to a strict, salted projection, a store that refuses to turn unknowns into zeros, and a privacy contract enforced by a test that inspects every byte on the wire. If you run AI coding agents on more than one machine, it is one of the few observability tools you can deploy without sending your metadata anywhere.

Links:

- GitHub repository: [LockedinLabs-AI/agent-console](https://github.com/LockedinLabs-AI/agent-console)
- Architecture documentation: [docs/ARCHITECTURE.md](https://github.com/LockedinLabs-AI/agent-console/blob/main/docs/ARCHITECTURE.md)
- Collector contract: [docs/COLLECTOR-CONTRACT.md](https://github.com/LockedinLabs-AI/agent-console/blob/main/docs/COLLECTOR-CONTRACT.md)
- Measurement definitions: [docs/MEASUREMENTS.md](https://github.com/LockedinLabs-AI/agent-console/blob/main/docs/MEASUREMENTS.md)
