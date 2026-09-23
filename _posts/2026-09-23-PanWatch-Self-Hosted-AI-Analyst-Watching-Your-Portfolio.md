---
layout: post
title: "PanWatch: A Self-Hosted AI Analyst Watching Your Portfolio"
description: "PanWatch is an open-source, self-hosted AI market watcher for A-shares, Hong Kong, and US stocks - scheduled agent runs, TradingAgents nine-agent deep analysis with bull-bear debate, paper trading, multi-channel alerts, and a PWA dashboard. We tour the architecture of the 1,300-star Python monorepo."
date: 2026-09-23
header-img: "img/post-bg.jpg"
permalink: /PanWatch-Self-Hosted-AI-Analyst-Watching-Your-Portfolio/
tags:
  - AI
  - Agents
  - Python
  - Fintech
  - Open Source
author: "PyShine"
---
# PanWatch: A Self-Hosted AI Analyst Watching Your Portfolio

The market does not care that you have a job. Pre-open moves happen on another continent's clock, intraday signals fire while you are in a meeting, and by the time the daily recap matters you have forgotten what you wanted to check. [PanWatch](https://github.com/TNT-Likely/PanWatch) - 1,389 stars and 289 forks, MIT licensed, written in Python - is a self-hosted answer to that problem: an AI watcher that runs on your own hardware, monitors A-share, Hong Kong, and US markets in real time, and turns price movement into analysis rather than noise. It is not an indicator dashboard with a chat box bolted on. The project integrates the [TradingAgents](https://github.com/TauricResearch/TradingAgents) multi-agent decision framework directly: click one button on a holding and a nine-agent research team takes over - four analyst roles covering technicals, sentiment, news, and fundamentals, followed by a structured bull-versus-bear debate, a risk review, and a portfolio-manager decision memo. The full reasoning chain lands in three to five minutes, and the conclusion is pushed to your messaging app of choice. Around that centerpiece sit the unglamorous features that make it usable daily: scheduled agent runs, paper trading with equity curves, multi-account portfolio tracking, and condition-based alerting. It is also a very good study in how to build a private, budget-metered agent system - the same concerns we saw at [cluster scale in AX](https://pyshine.com/AX-Kubernetes-Thinking-For-AI-Agent-Workloads/), here scaled down to a single Docker container.

![Architecture overview of the PanWatch repository showing the app surface, agent core, and platform services](/assets/img/diagrams/panwatch/panwatch-overview-architecture.svg)

## Why You Need This

Two problems keep retail traders from using AI seriously: privacy and cost. Cloud portfolio analyzers want your holdings on their servers, and open-ended LLM chats burn tokens without discipline. PanWatch attacks both by design. Self-hosted means your positions, your broker accounts, and your watchlists never leave your machine - the README's first selling point is exactly that. Cost is handled structurally, not aspirationally: every agent run passes through a dedicated token-meter package that tracks usage and budget, the default model is deepseek-chat at roughly five cents per full nine-agent analysis by the project's own accounting, and the platform layer keeps LLM calls behind one interface so you can swap providers. The second reason is the analyst flow itself. A single prompt asking a model whether a stock looks good produces confident mush. TradingAgents produces something auditable: analysts work their specialties first, the bull and bear cases are argued against each other before anything reaches a conclusion, a risk review gates the output, and the portfolio manager synthesizes a memo. That shape - debate before decision - mirrors what we covered in [OpenAI's own postmortem of agent failures](https://pyshine.com/OpenAI-Listed-Six-Ways-Its-Own-AI-Broke-the-Rules/): unreviewed single-shot output is where agents go wrong, and structured contention is a working mitigation. PanWatch wraps that flow in scheduling and delivery so it runs whether or not you remember to ask.

## How It Works

The repository is a clean FastAPI monorepo: one server entry, domain modules, a shared platform layer, and four packages under `packages/`.

![Detailed architecture diagram of the PanWatch monorepo from the repository source](/assets/img/diagrams/panwatch/panwatch-architecture.svg)

The entry point is `server.py`, a FastAPI application that boots through `src/bootstrap` - wiring domain modules, platform services, and packages together - and serves both the API and the installable PWA frontend in `frontend/src`. The frontend covers the daily surfaces: portfolio aggregation across broker accounts, an opportunities page scored by AI, price alerts, technical indicator views, stock detail pages, and a simulated trading view with equity curves. The agent layer is the interesting half. `packages/pan-agent-runtime` holds the execution engine: `runtime.py` drives agent runs, `contracts.py` types the interfaces between agents and services, `registry.py` registers agents and tools, and `policy.py` bounds execution - the same loop-controls philosophy (limits, budgets, structured stop) we walked through in [Strands Agents](https://pyshine.com/Strands-Agents-SDK-Replaces-Your-Hand-Rolled-Agent-Loop/), here specialized for market workflows. Two companion packages feed it: `pan-agent-tool-research` supplies the tool pack research agents call into, and `pan-agent-token-meter` meters every run. Prompt engineering lives in a versioned `prompts/` directory rather than scattered strings, with templates for each analyst role, the debate, the risk review, and the PM memo. On top of that runtime sit the domain modules. `research` drives the TradingAgents sequence on demand from a holding page. `automation` runs the scheduled fleet - a pre-open agent that synthesizes overnight moves and news into a strategy brief, an intraday monitor that watches for RSI, KDJ, and MACD resonance during trading hours, a post-market agent that writes the daily review, and a news digest that scrapes and filters financial headlines against your holdings. `market` handles quotes, indicators, and opportunity scoring through `packages/marketdata`, whose adapters cover A-shares, Hong Kong, and US exchanges. `portfolio` and `paper_trading` track real and simulated accounts, marked to market continuously. Underneath everything, the platform layer does the quiet work: timezone-aware `scheduling` so the pre-open agent fires at your market's open, an internal `events` bus, `persistence` under the container's data volume, `notifications` that route to Telegram, WeCom, DingTalk, Feishu, Bark, or any webhook - with per-rule channel selection on alerts - and `observability` with structured logs plus optional OpenTelemetry export when you set an OTLP endpoint.

## Advantages

- **Self-hosted and private.** Holdings, accounts, and watchlists stay on your hardware; nothing transits a third-party analyzer.
- **Structured multi-agent analysis.** Nine roles from four analyst specialties through bull-bear debate and risk review to a PM memo - auditable reasoning, not vibes.
- **Budget discipline.** A dedicated token meter, a cheap default model, and one swappable AI interface keep costs visible and bounded.
- **Always on.** Scheduled pre-market, intraday, post-market, and news agents run on timezone-aware triggers without human nudging.
- **Serious alerting.** Combined AND/OR conditions on price, change, turnover, and volume ratio, with trading-hours windows, cooldowns, daily caps, and per-rule channels.
- **Real engineering hygiene.** Typed contracts between agents and services, versioned prompts, structured logging, and optional OpenTelemetry traces.

## Benefits

The practical benefit is a market workflow that runs itself and leaves a trail you can inspect. Every conclusion arrives with its full reasoning chain - analyst outputs, both sides of the debate, the risk gate, and the decision memo - so you judge the process, not just the verdict, and push notifications mean the analysis finds you instead of the reverse. Because positions are tracked alongside AI scoring, the system's suggestions stay grounded in what you actually hold, per account and per strategy style; and because paper trading shares the same market data and agent flow, you can validate an idea against simulated equity curves before committing money. Privacy and cost stay engineered rather than hoped for: one container, one data volume, a token meter on every run. And because the agent layer is a clean runtime with contracts, registries, and policy rather than ad-hoc prompts in request handlers, extending it - a new analyst role, another market, a different model provider - is a code change, not a fork.

## Usage

The quickest path is Docker:

```bash
docker run -d \
  --name panwatch \
  -p 8000:8000 \
  -v panwatch_data:/app/data \
  sunxiao0721/panwatch:latest
```

Open `http://localhost:8000`, set an admin username and password on first visit, and the dashboard is live. The image pulls from [Docker Hub](https://hub.docker.com/r/sunxiao0721/panwatch), and a compose file in the README adds restart policy and volume management in a few lines. Configuration is environment-driven: `AUTH_USERNAME` and `AUTH_PASSWORD` preseed login credentials, `JWT_SECRET` signs sessions, `DATA_DIR` relocates storage, `TZ` aligns scheduled agents to your market's clock, `LOG_LEVEL` exposes collection and scheduling internals when debugging, and `HTTP_PROXY` routes outbound calls. Browser-driven features like screenshots install Chromium into the data volume on first boot; setting `PLAYWRIGHT_SKIP_BROWSER_INSTALL=1` skips that download if you do not need it. Then wire in your model key, add accounts or a watchlist, and the agents take their shifts: the pre-open brief arrives before your market opens, intraday resonance alerts fire during sessions, the post-market review lands after the close, and the nine-agent deep analysis is one click away on any holding.

## Conclusion

PanWatch is a reminder that the most useful agent products are rarely flashy: a scheduling system, a meter, a debate chain, and a push notification. By combining the [TradingAgents](https://github.com/TauricResearch/TradingAgents) research framework with solid platform engineering - typed contracts, versioned prompts, timezone-aware scheduling, multi-channel delivery - it delivers something no hosted dashboard will: a private AI analyst that works your market hours, shows its reasoning, and costs about a nickel per deep dive. The [repository](https://github.com/TNT-Likely/PanWatch) is young and moving fast, the MIT license invites forking, and the architecture invites extension. If you have been waiting for AI tooling that respects both your data and your budget, run the container and let the agents take the night shift.
