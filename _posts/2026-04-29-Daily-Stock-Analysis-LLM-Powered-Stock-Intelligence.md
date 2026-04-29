---
layout: post
title: "Daily Stock Analysis: LLM-Powered Stock Intelligence for A-Share, HK, and US Markets"
description: "Discover how Daily Stock Analysis uses LLM agents to deliver automated technical, sentiment, and fundamental analysis with decision dashboards pushed to WeChat, Feishu, Telegram, Discord, and Slack."
date: 2026-04-29
header-img: "img/post-bg.jpg"
permalink: /Daily-Stock-Analysis-LLM-Powered-Stock-Intelligence/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Agents, Finance, Open Source]
tags: [Daily Stock Analysis, LLM stock analysis, AI trading dashboard, stock market automation, GitHub Actions finance, multi-model LLM routing, YAML trading strategies, stock notification push, A-share analysis, open source]
keywords: "how to use daily stock analysis, LLM stock analysis setup guide, AI stock market dashboard tutorial, daily stock analysis vs alternatives, open source stock analysis tool, GitHub Actions stock analysis, multi-model LLM trading, YAML trading strategies, stock push notification WeChat Feishu, daily stock analysis installation"
author: "PyShine"
---

# Daily Stock Analysis: LLM-Powered Stock Intelligence for A-Share, HK, and US Markets

Daily Stock Analysis (DSA) is an open-source system that leverages large language models to deliver automated, multi-dimensional stock analysis for Chinese A-shares, Hong Kong stocks, and US equities. With 32,000+ GitHub stars, it generates a daily "Decision Dashboard" -- a one-glance summary with sentiment scores, buy/sell sniper points, risk alerts, and an action checklist -- and pushes it to your preferred channel: WeChat Work, Feishu, Telegram, Discord, Slack, or email.

![DSA Architecture Overview](/assets/img/diagrams/daily-stock-analysis/dsa-architecture.svg)

### Understanding the DSA Architecture

The system is organized as a multi-stage pipeline that moves data from market sources through analysis to notification delivery:

**1. Data Fetcher -- Multi-Source Market Data**

DSA aggregates real-time and historical data from multiple providers: AkShare, Tushare, Pytdx, Baostock, YFinance, Longbridge, and TickFlow. The fetcher layer abstracts away provider differences, automatically falling back when a source is unavailable. This resilience ensures analysis runs reliably even when individual data APIs experience outages.

**2. Multi-Dimensional Analysis Engine**

Three parallel analysis tracks run on each stock:

- **Technical Analysis**: Moving average alignment (MA5/MA10/MA20/MA60), MACD golden cross/death cross detection, RSI overbought/oversold signals, volume-price divergence, and chip distribution analysis.
- **Sentiment and News Analysis**: Real-time news aggregation from SerpAPI, Tavily, Anspire, SearXNG, and Brave. Social sentiment from Reddit, X, and Polymarket (US stocks). Negative news triggers a one-vote veto on buy signals.
- **Fundamental Analysis**: Financial ratios, capital flow, sector rankings, and earnings data aggregated from market data providers.

**3. LLM Analysis Engine**

The core intelligence layer feeds all analysis results into an LLM (via LiteLLM routing) which synthesizes a coherent narrative: trend prediction, operation advice, risk warnings, and sniper-level buy/sell/stop-loss/take-profit points. The LLM acts as the final decision maker, weighing technical signals against sentiment and fundamentals.

**4. Report Generator and Decision Dashboard**

The output is a structured report with a "Decision Dashboard" format: one-sentence core conclusion, sentiment score (0-100), ideal buy/secondary buy/stop-loss/take-profit levels, risk alerts, and an action checklist. Reports support Markdown, WeChat-formatted, and brief formats.

**5. Multi-Channel Push Notifications**

The notification layer supports 10+ channels: WeChat Work webhook, Feishu webhook, Telegram Bot, Discord Bot, Slack webhook, Email SMTP, Pushover, PushPlus, ServerChan3, AstrBot, and custom webhooks. Channel detection is automatic based on configuration.

![DSA LLM Routing](/assets/img/diagrams/daily-stock-analysis/dsa-llm-routing.svg)

### Multi-Model LLM Routing

DSA uses LiteLLM as its model router, supporting three configuration sources:

- **Environment Variables**: Simple setup for single-model deployments (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`)
- **litellm_config.yaml**: Full LiteLLM configuration with model list, API bases, and parameters
- **LLM Channels Config**: Advanced multi-channel routing with primary/fallback model chains

The router supports six major providers out of the box: Google Gemini, OpenAI GPT, DeepSeek, Anthropic Claude, Ollama (local models), and AIHubMix (multi-provider aggregator). When the primary model fails or times out, the system automatically falls back to the next model in the chain, ensuring analysis completes even during API outages.

This design is particularly valuable for users in regions where certain APIs may be unreliable -- you can configure a local Ollama model as the ultimate fallback, guaranteeing that your daily analysis never skips a day.

![DSA Strategy System](/assets/img/diagrams/daily-stock-analysis/dsa-strategy-system.svg)

### YAML-Based Strategy System

One of DSA's most innovative features is its no-code strategy system. Strategies are defined as YAML files with natural language instructions -- no Python code required. The system ships with 11 built-in strategies:

| Strategy | Category | Description |
|----------|----------|-------------|
| MA Golden Cross | Trend | MA5 crosses above MA10/MA20 with volume confirmation |
| Chan Theory (Chan Lun) | Pattern | Chinese candlestick fractal analysis |
| Wave Theory (Elliott) | Pattern | Elliott wave counting and impulse detection |
| Emotion Cycle | Framework | Market sentiment cycle identification |
| Volume Breakout | Trend | Volume surge with price breakout confirmation |
| Bull Trend Following | Trend | Strong uptrend continuation with MA alignment |
| Bottom Volume | Reversal | Volume accumulation at price bottoms |
| Dragon Head | Trend | Leading stock identification in sector rotation |
| Box Oscillation | Pattern | Range-bound trading with support/resistance |
| Shrink Pullback | Trend | Low-volume pullback to moving average support |
| One Yang Three Yin | Reversal | Bullish reversal candlestick pattern |

Each strategy YAML specifies its name, category, required tools (e.g., `get_daily_history`, `analyze_trend`), core trading rules it references, and natural language instructions. The Agent Orchestrator loads these strategies and uses them to guide the LLM's analysis, ensuring that AI-generated recommendations align with proven trading principles.

Custom strategies can be added by creating a YAML file in the strategies directory or pointing the `AGENT_SKILL_DIR` environment variable to a custom directory. Custom strategies override built-in ones with the same name.

### Technical Analysis Deep Dive

The `StockTrendAnalyzer` class implements a comprehensive 100-point scoring system:

| Factor | Weight | Criteria |
|--------|--------|----------|
| Trend | 30 pts | MA alignment: Strong Bull (30) > Bull (26) > Weak Bull (18) > Consolidation (12) > Weak Bear (8) > Bear (4) > Strong Bear (0) |
| Bias | 20 pts | Price proximity to MA5: pullback (20) > near MA5 (18) > slight above (14) > high bias (4) |
| Volume | 15 pts | Shrink pullback (15) > heavy volume up (12) > normal (10) > shrink up (6) > heavy volume down (0) |
| Support | 10 pts | MA5 support (5) + MA10 support (5) |
| MACD | 15 pts | Golden cross above zero (15) > golden cross (12) > crossing up (10) > bullish (8) > bearish (2) > death cross (0) |
| RSI | 10 pts | Oversold (10) > strong buy (8) > neutral (5) > weak (3) > overbought (0) |

The scoring system enforces a "strict entry" philosophy: high bias (price far above MA5) is penalized heavily, shrink-volume pullbacks are preferred over aggressive breakouts, and negative news triggers a one-vote veto regardless of technical signals.

### Agent Q&A and Backtesting

Beyond daily reports, DSA provides an interactive "Agent Ask" feature where users can query stocks using natural language with any of the 11 built-in strategies. The agent orchestrator selects the appropriate strategy, calls the required tools, and returns a strategy-specific analysis.

The backtesting system validates past analysis against actual price movements, calculating directional accuracy and simulated returns. This creates a feedback loop: strategies that consistently underperform can be refined or disabled.

![DSA Deployment Options](/assets/img/diagrams/daily-stock-analysis/dsa-deployment-options.svg)

### Deployment Options

DSA supports four deployment modes, each triggered by different mechanisms:

**1. GitHub Actions (Recommended for Beginners)**

Zero cost, no server required. Fork the repository, configure API keys as GitHub Secrets, and the workflow runs on a cron schedule. This is the fastest path from zero to daily analysis -- most users are operational within 5 minutes.

**2. Docker (Containerized)**

Pull the official Docker image and run with environment variables. Ideal for homelab setups and users who want full control over scheduling and data persistence.

**3. Local Cron (Scheduled)**

Run directly on your machine with cron or Task Scheduler. Best for users who already have a always-on server and want minimal overhead.

**4. FastAPI Service Mode**

Run as a long-lived service with REST API endpoints. Supports manual analysis triggers from the Web UI, on-demand API calls, and webhook integrations. The Web UI provides a dual-theme workbench with configuration management, task progress tracking, history browsing, and portfolio management.

### Web UI Features

The web interface provides:

- Manual stock analysis with real-time progress tracking
- Configuration management for models, data sources, and notification channels
- History report browsing with full Markdown rendering
- Portfolio import from images, CSV/Excel, or clipboard (supports code/name/pinyin autocomplete)
- Backtesting dashboard with accuracy metrics
- Agent Q&A interface for strategy-based stock queries

### Getting Started with GitHub Actions

The fastest way to start using DSA:

```bash
# 1. Fork the repository on GitHub
# 2. Configure Secrets in Settings > Secrets and variables > Actions
# 3. Add at least one AI model API key:
#    - AIHUBMIX_KEY (recommended: one key for all models)
#    - GEMINI_API_KEY
#    - OPENAI_API_KEY
#    - ANTHROPIC_API_KEY
# 4. Add at least one notification channel:
#    - WECHAT_WEBHOOK_URL
#    - FEISHU_WEBHOOK_URL
#    - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
#    - DISCORD_WEBHOOK_URL
# 5. Add your stock watchlist:
#    - STOCK_CODES: "600519,000858,AAPL,00700"
# 6. Enable GitHub Actions workflow
```

The system will automatically run analysis on trading days and push the Decision Dashboard to your configured channels.

### Ecosystem: AlphaSift and AlphaEvo

DSA focuses on daily analysis reports. Two companion projects extend the ecosystem:

- **AlphaSift**: Multi-factor stock screening and full-market scanning for extracting candidate stocks from the universe
- **AlphaEvo**: Strategy backtesting and self-evolution for validating strategy rules and iteratively exploring parameter combinations

These projects are independently maintained but are being explored for integration with DSA's candidate stock import, backtesting verification, and report linking.

### Key Takeaways

Daily Stock Analysis stands out in the AI-finance space for three reasons:

1. **No-code strategy extensibility**: YAML-based strategies let traders encode their philosophy without writing Python, while the LLM ensures analysis follows those rules consistently.

2. **Multi-model resilience**: LiteLLM routing with automatic fallback means your daily analysis never fails due to a single API outage. Local Ollama models provide the ultimate safety net.

3. **Zero-cost deployment**: GitHub Actions deployment eliminates the need for servers, making institutional-grade AI analysis accessible to individual investors at zero infrastructure cost.

The project is MIT-licensed, actively maintained, and has a comprehensive test suite with 100+ test files covering analysis logic, notification delivery, API contracts, and pipeline resilience.

> **Disclaimer**: Daily Stock Analysis is for educational and research purposes only. It does not constitute investment advice. Stock markets carry risk; invest carefully.