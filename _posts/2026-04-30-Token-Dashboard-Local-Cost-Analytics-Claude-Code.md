---
layout: post
title: "Token Dashboard: Local Cost Analytics for Claude Code Sessions"
description: "Learn how Token Dashboard reads Claude Code JSONL transcripts to provide per-prompt cost analytics, tool heatmaps, subagent attribution, cache analysis, and rule-based tips -- all running locally with zero telemetry."
date: 2026-04-30
header-img: "img/post-bg.jpg"
permalink: /Token-Dashboard-Local-Cost-Analytics-Claude-Code/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [Developer Tools, Open Source, AI Agents]
tags: [Token Dashboard, Claude Code, token analytics, cost tracking, AI agent monitoring, local dashboard, privacy-first, developer tools, open source, LLM costs]
keywords: "how to use Token Dashboard, Token Dashboard tutorial, Claude Code token tracking, AI agent cost analytics, local LLM token dashboard, Token Dashboard vs alternatives, Claude Code cost monitoring, token usage optimization, privacy-first AI analytics, Token Dashboard installation guide"
author: "PyShine"
---

# Token Dashboard: Local Cost Analytics for Claude Code Sessions

Token Dashboard is a local, privacy-first dashboard that reads the JSONL transcripts Claude Code writes to `~/.claude/projects/` and turns them into per-prompt cost analytics, tool and file heatmaps, subagent attribution, cache analytics, project comparisons, and a rule-based tips engine. Everything runs locally -- no data leaves your machine, no telemetry, no API calls for your data, and no login required.

![Data Flow](/assets/img/diagrams/token-dashboard/token-dashboard-data-flow.svg)

### Understanding the Data Flow

The data flow diagram above shows how Token Dashboard processes your Claude Code session data from start to finish:

**Claude Code JSONL Sessions**
Every time you use Claude Code, it writes one JSONL file per session to `~/.claude/projects/<project-slug>/<session-id>.jsonl`. These files contain the complete transcript of your conversation, including input tokens, output tokens, cache hits, tool calls, and tool results. The dashboard never modifies these files -- it only reads them.

**Scanner (cli.py -> scanner.py)**
The scanner reads all JSONL files, deduplicates streaming snapshots by `message.id` (Claude Code writes each assistant response 2-3 times during streaming), and populates a local SQLite cache. This deduplication ensures the final token tally matches what the API actually billed, not an inflated count from streaming intermediaries.

**SQLite Cache (token-dashboard.db)**
All processed data is stored in a local SQLite database at `~/.claude/token-dashboard.db`. This cache enables fast queries without re-scanning JSONL files every time. You can delete it and rebuild from scratch with `python3 cli.py scan` at any time.

**HTTP Server (server.py)**
The server exposes `/api/*` JSON routes that the browser dashboard queries. It also serves the static web UI from the `web/` directory. Server-Sent Events (SSE) push live updates to the browser every 30 seconds as the scanner re-scans for new sessions.

**Browser Dashboard**
The browser UI uses vanilla JavaScript with ECharts for charts, a dark theme, and a hash-based router for the 7 tabs. All JS, CSS, and fonts are served locally -- no external CDN calls.

## The 7 Tabs

![Seven Tabs](/assets/img/diagrams/token-dashboard/token-dashboard-seven-tabs.svg)

### Understanding the Seven Tabs

The dashboard organizes your token analytics into seven focused tabs, each backed by its own JSON API endpoint:

**Overview -- The Landing Tab**
The Overview tab provides all-time input/output/cache token totals, session counts, turn counts, and estimated cost on your chosen plan (API, Pro, Max, or Max-20x). Daily work charts show token usage over time, while cache-read charts reveal how much you are saving through prompt caching. The tokens-by-project breakdown and token share by model give you an instant snapshot of where your usage concentrates. A built-in "What do these numbers mean?" panel explains input/output/cache tokens in plain English.

**Prompts -- Most Expensive User Prompts**
The Prompts tab ranks your user prompts by token count, making it immediately clear which interactions are costing the most. Click any row to see the assistant response, tool calls made, and the size of each tool result. This is where you discover that a single tool call returning 80k tokens is responsible for a disproportionate share of your costs.

**Sessions -- Turn-by-Turn Analysis**
The Sessions tab provides a turn-by-turn view of any single session, showing per-turn tokens and tool calls. This granular view helps you understand exactly where tokens accumulate within a conversation and identify patterns like repeated file reads or unnecessarily large context windows.

**Projects -- Cross-Project Comparison**
The Projects tab compares token usage across all your projects, showing which projects consume the most tokens, have the most sessions, and which files were touched most frequently. This helps you understand whether your heaviest project is the one you expect, or if a side project is quietly burning through your token budget.

**Skills -- Skill Invocation Analytics**
The Skills tab shows which skills you invoke most often and, where measurable, their token cost. This helps you optimize your workflow by identifying skills that are expensive to run and whether they deliver proportional value.

**Tips -- Rule-Based Suggestions**
The Tips tab provides actionable suggestions for reducing token usage: repeated file reads, oversized tool results, low cache-hit rates, and other wasteful patterns. These are generated by a rule-based engine that analyzes your session data.

**Settings -- Plan Pricing**
The Settings tab lets you switch pricing between API / Pro / Max / Max-20x so cost figures everywhere else in the dashboard reflect your actual plan. Pricing data lives in `pricing.json`, which you can edit directly if model prices change or to add a new plan.

## Privacy Architecture

![Privacy Architecture](/assets/img/diagrams/token-dashboard/token-dashboard-privacy.svg)

### Understanding the Privacy Architecture

The privacy architecture diagram emphasizes Token Dashboard's core design principle: everything stays on your machine.

**No External Calls**
The dashboard makes zero external network calls. No telemetry, no analytics, no login servers, no API calls for your data. The browser fetches its JSON from `127.0.0.1`, and all JS/CSS/fonts are served from that same local server. ECharts is vendored into `web/`, and the UI falls back to system fonts rather than pulling from a font CDN. You can verify this yourself: `grep -r "https://" token_dashboard/ web/` returns nothing.

**Local-Only Storage**
All data stays in two places: the original JSONL files that Claude Code already writes, and the SQLite cache at `~/.claude/token-dashboard.db`. There is no cloud sync, no remote backup, and no data sharing. Delete the SQLite file and re-scan to start fresh.

**Security Considerations**
The default bind address is `127.0.0.1`, which means the dashboard is only accessible from your local machine. The README explicitly warns against setting `HOST=0.0.0.0` on networks you do not fully control (no coffee-shop Wi-Fi, no coworking spaces), as that would expose your entire prompt history to anyone on the same network.

## Tips Engine

![Tips Engine](/assets/img/diagrams/token-dashboard/token-dashboard-tips-engine.svg)

### Understanding the Tips Engine

The tips engine analyzes your session data through a set of rule-based checks and surfaces actionable suggestions for reducing token waste:

**Repeated File Reads**
If the same file is read 20+ times in a single session, the tips engine flags it. This pattern often indicates that the agent is re-reading context it already has, suggesting an opportunity to reduce context window size or improve prompt instructions to reference previously loaded content.

**Oversized Tool Results**
When a tool call returns 80k+ tokens, the tips engine highlights it. Large tool results are the most common cause of unexpectedly expensive prompts. The fix might be to limit the scope of tool calls, use pagination, or filter results before returning them to the agent.

**Low Cache-Hit Rate**
If your cache-hit rate is consistently low, the tips engine suggests strategies to improve it: reordering prompts to keep stable context at the beginning, reducing unnecessary context churn, or structuring conversations to maximize prompt caching benefits.

**Subagent Attribution**
When subagents are responsible for a significant portion of token costs, the tips engine surfaces this information so you can decide whether the subagent overhead is justified or whether the workflow should be restructured to reduce delegation costs.

## Installation

```bash
# Clone the repository
git clone https://github.com/nateherkai/token-dashboard.git
cd token-dashboard

# Start the dashboard (no pip install, no Node.js, no build step)
python3 cli.py dashboard
```

On Windows, if `python3` is not on your PATH, substitute `py -3`:

```bash
py -3 cli.py dashboard
```

The command will:
1. Scan `~/.claude/projects/` (first run can take 20-60 seconds on a heavy user's machine)
2. Start a local server at `http://127.0.0.1:8080`
3. Open your default browser to that URL

Leave it running; it re-scans every 30 seconds and pushes updates live. Stop with `Ctrl+C`.

### CLI Reference

```bash
python3 cli.py scan          # populate / refresh the local DB, then exit
python3 cli.py today         # today's totals (terminal)
python3 cli.py stats         # all-time totals (terminal)
python3 cli.py tips          # active suggestions (terminal)
python3 cli.py dashboard     # scan + serve the UI at http://localhost:8080

# dashboard flags
python3 cli.py dashboard --no-open   # don't auto-open the browser
python3 cli.py dashboard --no-scan   # skip the initial scan (use cached DB only)
```

Change the port: `PORT=9000 python3 cli.py dashboard`

### Custom Data Location

```bash
python3 cli.py dashboard --projects-dir /path/to/projects --db /path/to/cache.db
```

## Key Features

| Feature | Description |
|---------|-------------|
| Per-Prompt Cost Analytics | See which prompts are expensive, ranked by token count with tool result breakdowns |
| Tool and File Heatmaps | Visualize which tools and files are accessed most across all sessions |
| Subagent Attribution | Trace token costs back to specific subagent calls |
| Cache Analytics | Understand cache-hit rates and how much prompt caching saves you |
| Project Comparisons | Compare token usage, session counts, and file touch frequency across projects |
| Rule-Based Tips | Actionable suggestions for reducing waste (repeated reads, oversized results, low cache hits) |
| Plan Pricing Switch | Switch between API / Pro / Max / Max-20x pricing to match your actual plan |
| 100% Local | No data leaves your machine -- no telemetry, no remote API calls, no login |
| Zero Dependencies | Python stdlib only, no pip install, no Node.js, no build step |
| Live Refresh | Server-Sent Events push updates every 30 seconds as new sessions are detected |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No data" or empty charts | Run `python3 cli.py scan` once to populate the DB, then reload |
| Port 8080 already in use | `PORT=9000 python3 cli.py dashboard` |
| Numbers look wrong or stuck | Delete `~/.claude/token-dashboard.db` and re-run `python3 cli.py scan` |
| Running dashboard twice | Do not -- both processes fight over the SQLite DB. Stop all instances first |
| First scan is slow | Heavy users may have thousands of JSONL files; 20-60 seconds is normal for the first run |

## Conclusion

Token Dashboard fills a critical gap in the Claude Code ecosystem: visibility into how much your AI coding sessions actually cost. By reading the JSONL transcripts that Claude Code already writes and presenting them through 7 focused tabs with per-prompt analytics, tool heatmaps, subagent attribution, and actionable tips, it gives you the information you need to optimize your token usage -- all without sending a single byte of data off your machine.

Whether you are on the API plan paying per token, or on Pro/Max trying to confirm you are getting your money's worth, Token Dashboard provides the analytics to make informed decisions about your AI coding workflow.

**Links:**

- GitHub: [https://github.com/nateherkai/token-dashboard](https://github.com/nateherkai/token-dashboard)