---
layout: post
title: "CloddsBot: The Open Source AI Trading Agent That Works Across 1000+ Markets"
description: "CloddsBot is an open source autonomous AI trading terminal built in TypeScript and powered by Anthropic's Claude. It monitors and executes trades across more than 1000 financial venues spanning 10 prediction markets (Polymarket, Kalshi, Betfair, Smarkets, Drift, Manifold, Metaculus, PredictIt, Opinion.xyz, Predict.fun), 7 perpetual futures exchanges (Binance 125x, Bybit 100x, MEXC 200x, Hyperliquid 50x, Drift 20x, Percolator, Lighter 50x), Solana DeFi (Jupiter, Raydium, Orca, Meteora, Kamino, MarginFi, Solend, Pump.fun, Bags.fm), and 5 EVM chains (ETH, ARB, OP, Base, Polygon via Uniswap V3, 1inch, PancakeSwap, Virtuals Protocol). Chat via 21 messaging platforms (Telegram, Discord, WhatsApp, Slack, Teams, Signal, Matrix, iMessage, LINE, Nostr, and more). 118+ trading strategies including momentum, mean reversion, expiry fade, DCA bots, whale tracking, copy trading, and cross-platform arbitrage detection (arXiv:2508.03474). Unified risk engine with circuit breaker, VaR/CVaR, Kelly sizing, daily loss limits, kill switch. Trade ledger with SHA-256 integrity hashing and onchain anchoring. x402 protocol for machine-to-machine USDC payments. Agent forum, agent marketplace, compute API, token launch, Bittensor mining. 121 MCP skills. MIT license, 463 commits, built for Colosseum Agent Hackathon on Solana."
date: 2026-09-12
header-img: "img/post-bg.jpg"
permalink: /CloddsBot-Open-Source-AI-Trading-Agent-1000-Markets/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - CloddsBot
  - AI Trading
  - Open Source
  - Claude
  - Prediction Markets
  - Solana
  - DeFi
  - MCP
author: PyShine
---

Imagine typing "buy 50 USDC of BTC up on Polymarket" into a chat window, and an AI agent figures out which platform to use, checks the risk, executes the trade, logs the decision with a SHA-256 hash, and anchors it to Solana for tamper-proof proof. All without you touching a single API.

That is CloddsBot.

## What is CloddsBot

[CloddsBot](https://github.com/alsk1992/CloddsBot) is an open source autonomous AI trading terminal built in TypeScript and powered by Anthropic's Claude. The name says it all: **Claude + Odds = Clodds**. It was built for the [Colosseum Agent Hackathon](https://colosseum.com/) on Solana in 12 days as a fully-featured autonomous trading agent, and it has since grown into a serious project with 463 commits, MIT license, and active community development.

What makes CloddsBot different from every other trading bot you have seen: you talk to it in plain language through any of 21 messaging platforms, and it handles everything from market analysis to order execution to risk management to post-trade auditing. It is not just a bot that places orders. It is an agent that reasons about trades.

## System Architecture

CloddsBot is structured as an AI core that drives a TypeScript gateway, connecting to messaging channels, trading venues, and a massive skills system.

![CloddsBot architecture](/assets/img/diagrams/cloddsbot/cloddsbot-architecture.svg)

### Understanding the Architecture

**AI Core (Claude-powered)**

The brain of CloddsBot is Claude, but it supports 8 LLM providers: Claude (primary), GPT-4, Gemini, Groq, Together, Fireworks, AWS Bedrock, and Ollama. It runs 4 specialized agents (Main, Trading, Research, Alerts) with access to 18 tools including Browser, Docker, Exec, Files, Git, Email, SMS, Webhooks, SQL, and Vision. Memory is semantic: LanceDB embeddings plus BM25 hybrid search, with persistent user profiles.

**Gateway (TypeScript/Node.js)**

The gateway is the orchestration layer. CLI commands include `onboard` (interactive setup wizard), `start` (start gateway), `repl` (interactive REPL), `doctor` (system diagnostics), `secure` (harden security), `locale` (change language), and `mcp` (start MCP server). A built-in WebChat interface opens at `localhost:18789` with a Claude-style sidebar, unlimited history, and context compacting. Data persistence uses SQLite (local trades), LanceDB (semantic memory), and PostgreSQL (analytics).

**21 Messaging Channels**

You can chat with CloddsBot through Telegram, Discord, WhatsApp, Slack, Teams, Matrix, Signal, iMessage, LINE, Nostr, Twitch, WebChat, and more. All channels support real-time sync, rich media, and offline queuing. This means you can trade from whatever platform you already live in.

**17 Trading Venues**

CloddsBot connects to 10 prediction markets and 7 perpetual futures exchanges. The prediction markets include Polymarket (crypto USDC), Kalshi (US regulated), Betfair (sports exchange), Smarkets (sports), Drift (Solana DEX), Manifold (play money data), Metaculus (forecasting data), PredictIt (US politics data), Opinion.xyz (BNB Chain), and Predict.fun (BNB Chain).

The perpetual futures exchanges span Binance (125x leverage), Bybit (100x), MEXC (200x), Hyperliquid (50x, no KYC), Drift (20x, Solana DEX), Percolator (on-chain Solana perps), and Lighter (50x, Arbitrum DEX). Long/short, cross/isolated margin, TP/SL, liquidation alerts, and funding tracking are all supported.

**121 MCP Skills**

CloddsBot bundles 121 skills that are also exposed as MCP (Model Context Protocol) tools for Claude Desktop and Claude Code. Skills cover trading (16 platforms), analysis (arbitrage, whale tracking, token security), automation (cron jobs, triggers, bots, webhooks), and AI (memory, embeddings, multi-agent routing). Skills are lazy-loaded, so missing dependencies do not crash the app.

## Risk Management and Trade Ledger

Trading without risk management is gambling. CloddsBot treats risk as a first-class concern with a unified risk engine and a full decision audit trail.

![CloddsBot risk engine and trade ledger](/assets/img/diagrams/cloddsbot/cloddsbot-risk-ledger.svg)

### Unified Risk Engine

The risk engine sits between every trade decision and execution. It includes:

- **Circuit breaker**: Auto-stops trading when losses hit a threshold
- **VaR/CVaR**: Value at Risk and Conditional Value at Risk calculations
- **Volatility regime detection**: Adapts to market conditions
- **Stress testing**: Scenario analysis for extreme events
- **Kelly sizing**: Optimal position sizing based on edge and bankroll
- **Daily loss limits**: Hard stop on daily drawdown
- **Kill switch**: Total stop for emergencies

### Pre-Trade Validation

Before any trade is executed, CloddsBot runs token security audits powered by GoPlus: honeypot detection, rug-pull analysis, holder concentration checks, and risk scoring. A Security Shield with 75 code-scanning rules and a scam database of 70+ addresses provides multi-chain address checking and pre-trade transaction validation.

### Trade Ledger

Every trade decision is logged with its reasoning in a tamper-proof audit trail:

- **Decision capture**: Every trade, copy, and risk decision logged with reasoning
- **Confidence calibration**: Track AI prediction accuracy vs confidence levels
- **SHA-256 integrity hashing**: Tamper-proof records
- **Onchain anchoring**: Anchor hashes to Solana, Polygon, or Base for immutable proof
- **Statistics**: Win rates, P&L, block reasons, accuracy by confidence bucket

```bash
clodds ledger stats           # Show decision statistics
clodds ledger calibration     # Confidence vs accuracy analysis
clodds ledger verify <id>     # Verify record integrity
clodds ledger anchor <id>     # Anchor hash to Solana
```

### Safety Defaults

CloddsBot defaults to simulation mode. Shell commands need approval (sandboxed execution). Credentials are encrypted with AES-256-GCM. All trades have audit logging. The recommendation is to use isolated sub-accounts and start with small amounts you can afford to lose.

## 118+ Trading Strategies and Arbitrage Detection

CloddsBot ships with 118+ strategies across multiple categories, plus a sophisticated arbitrage detection system based on [arXiv:2508.03474](https://arxiv.org/abs/2508.03474).

![CloddsBot strategies and arbitrage](/assets/img/diagrams/cloddsbot/cloddsbot-strategies-arbitrage.svg)

### Strategy Categories

| Category | Examples |
|----------|----------|
| Momentum | Trend following, breakout detection, volume confirmation, moving average crossover |
| Mean Reversion | Bollinger bands, RSI oversold/overbought, Z-score reversion, penny clipper |
| Expiry Fade | Polymarket round-based (5min, 15min, 1h, 4h, daily) with timing gates and round-based discovery |
| DCA Bots | Dollar-cost averaging with configurable sizing, SL/TP, smart routing |
| Whale Tracking | Multi-chain monitoring (Solana, ETH, Polygon, ARB, Base, OP) with copy trading |
| Copy Trading | Mirror successful wallets with sizing controls and SL/TP |
| Swarm Trading | Coordinated multi-wallet Pump.fun trading (20 wallets, Jito bundles) |
| Smart Routing | Best price, liquidity, or fees across platforms |

### Arbitrage Detection

CloddsBot detects three types of arbitrage:

**Internal arbitrage**: Buy both sides of a prediction market when YES + NO costs less than $1.

```
YES: 45c + NO: 52c = 97c -> Buy both -> 3c profit
```

**Cross-platform arbitrage**: Same market on different platforms at different prices.

```
Polymarket @ 52c vs Kalshi @ 55c -> 3c spread
```

**Combinatorial arbitrage**: Multi-leg semantic matching with combinatorial analysis and real-time scanning.

Arbitrage defaults to dry-run mode. Cross-platform arbitrage has currency and settlement complexity that must be considered.

### External Data Sources

CloddsBot pulls from external data sources for edge detection: FedWatch (rate predictions), FiveThirtyEight (polls), Silver Bulletin (forecasts), RealClearPolitics (polls), and Odds API (sports odds).

## Agent Commerce Protocol

This is where CloddsBot gets genuinely futuristic. It is not just a trading bot. It is a node in a machine-to-machine economy.

![CloddsBot agent commerce](/assets/img/diagrams/cloddsbot/cloddsbot-agent-commerce.svg)

### x402 Payment Protocol

CloddsBot implements the x402 protocol for machine-to-machine USDC payments on Base and Solana. Agents pay other agents for services: compute, data, strategies, APIs. This is not a future roadmap item. It is in the code today.

### Agent Forum

There is an agent-only discussion platform at cloddsbot.com/forum where agents share market insights, share strategies, and vote. Bots do not just trade. They collaborate.

### Agent Marketplace

A peer-to-peer marketplace where AI agents buy and sell strategies, APIs, and datasets with USDC escrow on Solana. Agents trade with agents.

### Compute API

A pay-per-use compute API where agents can purchase LLM inference, code execution, web scraping, data, storage, and trade execution with USDC micropayments.

### Token Launch

One API call launches a Solana token via Meteora Dynamic Bonding Curves, with a 90/10 creator fee split, anti-sniper protection, auto AMM graduation, and agent-gated access.

### Bittensor Mining

CloddsBot can mine TAO on Bittensor subnets with wallet management, Chutes SN64 GPU compute, and earnings tracking.

### Cross-Chain Bridging

Wormhole protocol enables cross-chain transfers: ETH to Solana, Polygon to Base, and more.

## Quick Start

**Requirement:** Node.js 22 or newer.

### Install and Onboard

```bash
npm install -g https://github.com/alsk1992/CloddsBot/releases/latest/download/clodds.tgz --loglevel=error
clodds onboard
```

The setup wizard walks you through API key, messaging channel, and starts the gateway. WebChat opens at `http://localhost:18789/webchat`.

### From Source

```bash
git clone https://github.com/alsk1992/CloddsBot.git && cd CloddsBot
npm install && cp .env.example .env
# Add ANTHROPIC_API_KEY to .env
npm run build && npm start
```

### CLI Commands

```bash
clodds onboard          # Interactive setup wizard
clodds start            # Start the gateway
clodds repl             # Interactive REPL
clodds doctor           # System diagnostics
clodds secure           # Harden security
clodds locale set zh    # Change language
clodds mcp              # Start MCP server (for Claude Desktop/Code)
clodds mcp install      # Auto-configure Claude Desktop/Code
```

### Trading Commands

```bash
# Futures
/futures long BTCUSDT 0.1 10x
/futures sl BTCUSDT 95000

# Percolator (on-chain Solana perps)
/percolator status
/percolator long 100
/percolator short 50

# Bittensor mining
/tao status
/tao earnings daily
```

## Key Features

| Feature | Description |
|---------|-------------|
| 21 messaging channels | Telegram, Discord, WhatsApp, Slack, Teams, Signal, Matrix, iMessage, LINE, Nostr, Twitch, WebChat, and more |
| 10 prediction markets | Polymarket, Kalshi, Betfair, Smarkets, Drift, Manifold, Metaculus, PredictIt, Opinion.xyz, Predict.fun |
| 7 futures exchanges | Binance 125x, Bybit 100x, MEXC 200x, Hyperliquid 50x, Drift 20x, Percolator, Lighter 50x |
| 118+ strategies | Momentum, mean reversion, expiry fade, DCA, whale tracking, copy trading, smart routing |
| 121 MCP skills | Exposed as MCP tools for Claude Desktop and Claude Code |
| Unified risk engine | Circuit breaker, VaR/CVaR, Kelly sizing, daily loss limits, kill switch |
| Trade ledger | SHA-256 hashing, onchain anchoring, confidence calibration |
| Solana DeFi | Jupiter, Raydium, Orca, Meteora, Kamino, MarginFi, Solend, Pump.fun, Bags.fm |
| EVM DeFi (5 chains) | Uniswap V3, 1inch, PancakeSwap, Virtuals Protocol on ETH, ARB, OP, Base, Polygon |
| Agent commerce | x402 M2M USDC payments, agent forum, agent marketplace, compute API |
| Token security | GoPlus audits, honeypot detection, rug-pull analysis, risk scoring |
| 10 languages | EN, ZH, ES, JA, KO, DE, FR, PT, RU, AR |
| Bittensor mining | TAO mining with wallet management and earnings tracking |
| Token launch | Solana launches via Meteora DBC, anti-sniper, auto AMM graduation |

## The Honest Caveats

CloddsBot is impressive engineering, but it is important to be clear about what it is and is not:

- **Built in 12 days for a hackathon**: While the community is maintaining it, this is not a battle-tested institutional trading system. Stability under extreme market conditions has not been validated at scale.
- **200x leverage is available**: Even with the risk engine, a flash crash or liquidation cascade can wipe out an account. The risk engine helps, but it cannot predict black swans.
- **Arbitrage is not free money**: Platform fees, withdrawal fees, settlement rules, and slippage can eat profits. The "1000+ markets" and "arbitrage opportunities" are the project's claims, not guaranteed profit.
- **AI trading has tail risks**: Model hallucination, API latency, smart contract exploits, and slippage in illiquid pools are all real dangers.
- **Start in simulation mode**: CloddsBot defaults to simulation. Keep it that way until you understand the system. If you go live, use isolated sub-accounts with money you can afford to lose.

## Why CloddsBot Matters

CloddsBot is a preview of what autonomous economic agents look like. It is not just a trading bot. It is an agent that reasons about trades, manages risk, audits its own decisions, pays other agents for services, shares insights on a forum, and can mine cryptocurrency. The x402 payment protocol and agent marketplace point toward a future where agents are economic actors, not just tools.

The MCP integration is particularly significant. By exposing all 121 skills as MCP tools, CloddsBot becomes a trading skill pack for Claude Desktop and Claude Code. You can ask Claude to check your portfolio, analyze a market, or place a trade, and it uses CloddsBot's skills to do it. This is the composable agent economy in action.

And it is all open source under MIT, running on your own machine, with your own keys. That is the open future of autonomous trading.

## Related Posts

- [AI Trader: Fully Automated Agent Native Trading](/AI-Trader-Fully-Automated-Agent-Native-Trading/)
- [HyperFrames: Write HTML, Render Video, Built for AI Agents](/HyperFrames-Write-HTML-Render-Video-Built-for-Agents/)
- [Grok Build: SpaceXAI Terminal AI Coding Agent in Rust](/grok-build-spacexai-terminal-ai-coding-agent-rust/)
