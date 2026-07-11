---
layout: post
title: "AI Berkshire: A Value-Investing Multi-Agent Research Framework for Claude Code"
description: "AI Berkshire turns one person plus Claude Code or Codex into an investment research team by encoding the methodologies of Buffett, Munger, Duan Yongping, and Li Lu into 19 AI agent skills with forced conclusions and decimal-precise financial math."
date: 2026-07-11
header-img: "img/post-bg.jpg"
permalink: /AI-Berkshire-Value-Investing-Multi-Agent-Research-Framework/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI Berkshire
  - Multi-Agent
  - Claude Code
  - Value Investing
  - Open Source
  - AI Agents
author: "PyShine"
---

# AI Berkshire: A Value-Investing Multi-Agent Research Framework for Claude Code

Value investing has always been a craft passed down through apprenticeship. You read Buffett's letters, study Munger's mental models, absorb Duan Yongping's business judgment, and learn Li Lu's long-horizon thinking — then spend decades practicing. **AI Berkshire** compresses that apprenticeship into software. It is a research framework built for Claude Code and OpenAI Codex that encodes the methodologies of four investing masters into AI agent skills, so a single person with a coding agent can function as an entire investment research team.

With over 12,700 stars on GitHub and an MIT license, the project's tagline says it plainly: "One person + Claude Code / Codex = an investment research team." Let us look at how it turns investing philosophy into reproducible agent workflows.

## Architecture Overview

AI Berkshire is built as three layers: a Skill layer of entry points, an Agent layer that runs parallel master-perspective agents, and a Tool layer that does the financial math.

![AI Berkshire Architecture](/assets/img/diagrams/ai-berkshire/ai-berkshire-architecture.svg)

The three layers work together:

- **Skill Layer**: 19 distinct slash-command entry points for different research scenarios — deep research, earnings analysis, industry screening, portfolio management, and thinking tools
- **Agent Layer**: team-type skills (like `/investment-team` and `/earnings-team`) use a Team Lead that dispatches four parallel master-perspective agents. Each agent independently searches the web, cross-validates data, and reaches its own conclusion before the Team Lead synthesizes them. Lightweight skills bypass this layer and connect straight to tools.
- **Tool Layer**: `tools/financial_rigor.py` provides precise financial calculations using Python `decimal.Decimal` rather than `float`, with multi-source cross-validation, Benford's Law detection, three-scenario valuation, and market-cap verification.

The key design decision is that the four masters are not a division of labor. They are designed to challenge each other. When Buffett says a stock is cheap, Li Lu asks whether it will still exist in ten years. That tension is the point.

## The Four Master Methodologies

Every analysis in AI Berkshire runs through four distinct perspectives, each modeled on a real investor's thinking.

![AI Berkshire Masters](/assets/img/diagrams/ai-berkshire/ai-berkshire-masters.svg)

- **Duan Yongping (段永平)** — "The right business." Focuses on business-model essence: is this fundamentally a good business? This serves as the starting point for all other perspectives.
- **Warren Buffett (巴菲特)** — Moat, margin of safety, and management quality. The financial-valuation lens.
- **Charlie Munger (芒格)** — Inverse thinking and cognitive-bias self-checks. Asks not "why will this succeed?" but "how could this company die?" with explicit risk checklists.
- **Li Lu (李录)** — Civilizational trends and paradigm shifts. Looks for long-term certainty over 10+ year horizons rather than quarterly performance.

The arrows between them are not decoration. They represent the challenges the framework forces: Buffett's cheapness is tested by Li Lu's survival question, Munger reverse-tests Buffett, and Duan Yongping's business-quality judgment gates the rest.

## The 19 Skills

AI Berkshire exposes 19 slash commands organized into five categories. Each is a focused entry point designed for a specific research scenario.

![AI Berkshire Skills](/assets/img/diagrams/ai-berkshire/ai-berkshire-skills.svg)

**Deep Research (5 skills):** `/investment-research` runs a full four-master analysis of a listed company; `/investment-team` dispatches four parallel agents on one company; `/management-deep-dive` assesses management; `/private-company-research` handles unlisted companies like SpaceX or Ant Group; `/deep-company-series` produces an 8-article deep series on a single company at roughly 120,000 words.

**Earnings Analysis (2 skills):** `/earnings-review` reads original financial reports only, refusing second-hand research notes; `/earnings-team` runs parallel earnings reading plus a WeChat-article publishing pipeline.

**Industry Screening (5 skills):** `/industry-research` scans a full industry chain; `/industry-funnel` runs a market-wide funnel from roughly 60 companies down to 10 and then 3 final picks; `/quality-screen` applies 7 hard metrics to quickly eliminate sub-par companies; `/bottleneck-hunter` finds supply-chain bottlenecks for super-trends; `/investment-checklist` runs Buffett's 6-gate pre-buy checklist for a 10-minute decision.

**Portfolio Management (4 skills):** `/portfolio-review` handles position sizing, concentration, and rebalancing; `/thesis-tracker` tracks whether investment theses are being invalidated over time; `/thesis-drift` compares two reports to detect drift in your own thinking; `/news-pulse` attributes stock-price movements in 10 minutes.

**Thinking Tools (3 skills):** `/dyp-ask` reasons through any question in Duan Yongping's style; `/financial-data` enforces financial-data cross-validation standards (2+ sources, any discrepancy above 1% triggers an alert); `/wechat-article` uses a 3-agent collaboration of author, editor, and reader to produce publishable articles.

## How Parallel Multi-Agent Research Works

The `/investment-team` skill is the clearest expression of the framework's thesis. It launches four independent AI agents, each embodying one master's perspective, all researching the same company simultaneously.

![AI Berkshire Workflow](/assets/img/diagrams/ai-berkshire/ai-berkshire-workflow.svg)

The flow has five steps:

1. **Trigger**: you run `/investment-team <company>` in Claude Code
2. **Team Lead dispatches** four parallel agents — Duan Yongping, Buffett, Munger, and Li Lu
3. **Each agent independently searches the web** and cross-validates its data, then scores the company from its own perspective
4. **The Team Lead synthesizes** all four independent conclusions into a unified scoring table
5. **A forced verdict** is produced with tiered price targets — aggressive, stable, and conservative

The repo includes real worked examples. For PDD (Pinduoduo), the Duan Yongping agent scored the C2M business model 3.7/5, the Buffett agent scored the cash-adjusted PE of 6.3x at 4.4/5, the Munger agent scored it 3.5/5 while flagging Douyin's 4T GMV in 3 years as a threat, and the Li Lu agent scored it 2.0/5 on management-culture concerns. Four different scores from four different lenses, synthesized into one decision.

## Forced Conclusions and Anti-Bias Mechanisms

The most opinionated part of AI Berkshire is what it refuses to do. It will not produce "on the other hand" hedging. Every output gives a pass, fail, or gray-zone verdict with a specific price range. That pressure toward a decision is the whole value proposition — an investment tool that never concludes is not a tool.

To support forced conclusions without deceiving itself, the framework layers in several anti-bias mechanisms:

- **Information richness grading** (A/B/C) so you know how much weight a conclusion can bear
- **Munger-style reverse testing** as a required step, not an option
- **An 8-rule fast-veto checklist** to kill bad ideas quickly
- **Anti-consensus checks** that explicitly question what everyone agrees on
- **A "leave blank" principle** — when the data is not there, the framework says so rather than inventing an answer
- **A "mirror test"** — you must be able to explain a buy rationale in five sentences, or you do not buy

The `/news-pulse` skill even outputs "true cause unknown" as a valid conclusion when a stock moves without a clear reason — which, as the README notes, may itself be a signal of insider trading worth respecting as data.

## Financial Precision at the Tool Layer

All calculations in `financial_rigor.py` use `decimal.Decimal` rather than `float`. That is not pedantry — floating-point errors compound across thousands of financial figures and quietly corrupt valuations. Critical data is verified from 2+ independent sources, with Benford's Law applied to detect fabricated numbers, and three-scenario valuation (aggressive / stable / conservative) produced for every target. Market-cap verification is a separate step. The result is that the same inputs always produce structurally consistent outputs, which is what makes cross-company comparison possible at all.

## Installation

AI Berkshire installs as a set of slash commands for either Claude Code or OpenAI Codex.

For Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
git clone https://github.com/xbtlin/ai-berkshire.git
cd ai-berkshire
./scripts/install-claude-commands.sh
```

For Codex:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
git clone https://github.com/xbtlin/ai-berkshire.git
cd ai-berkshire
./scripts/install-codex-skills.sh
```

Then run a skill in Claude Code, for example `/investment-research 腾讯` or `/investment-team 美团`. The framework maintains three parallel entry points: `skills/*.md` for Claude Code, `codex-skills/*/SKILL.md` for Codex (auto-generated via `scripts/sync-codex-skills.py`), and `codex-prompts/*.md` as an optional Codex slash-prompt compatibility layer.

## A Note on the Track Record

The repository claims real — not backtested — portfolio returns of +69.29% in 2024 versus the S&P 500's +23.31%, and +66.38% in 2025 versus +16.39%. Treat any such claim with appropriate skepticism and do your own diligence; the point of covering the framework is its architecture and methodology, not an endorsement of its returns.

## Why It Matters

AI Berkshire is interesting well beyond investing. It is one of the most explicit examples of encoding a human reasoning tradition — one that took decades to learn — into a set of AI agent skills with forced outputs, anti-bias checks, and decimal-precise math. The pattern of multiple adversarial agents that must challenge each other before a synthesis layer produces a verdict is generalizable to any domain where you want rigor rather than a single confident-sounding answer.

If you use Claude Code or Codex and want to see what serious multi-agent methodology looks like in practice — or if you actually want to research stocks with it — AI Berkshire is worth a careful read.

**Links:**

- GitHub: [https://github.com/xbtlin/ai-berkshire](https://github.com/xbtlin/ai-berkshire)
- Install (Claude Code): `./scripts/install-claude-commands.sh`
- Install (Codex): `./scripts/install-codex-skills.sh`
- License: MIT