---
layout: post
title: "OpenStock: An Open-Source Alternative to Expensive Market Platforms"
description: "OpenStock is a free, open-source stock market app built with Next.js 15, Better Auth, MongoDB, Finnhub, TradingView widgets, and Inngest - with AI-personalized emails, watchlists, and 30+ international exchanges."
date: 2026-09-20
header-img: "img/post-bg.jpg"
permalink: /OpenStock-Open-Source-Alternative-to-Expensive-Market-Platforms/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - TypeScript
  - Next.js
  - MongoDB
  - Finnhub
  - TradingView
  - Finance
  - Stock Market
author: "PyShine"
---
# OpenStock: An Open-Source Alternative to Expensive Market Platforms

Real-time stock prices, personalized alerts, and company insights are usually sold as subscriptions, and the best-known market platforms charge handsomely for all three. [OpenStock](https://github.com/Open-Dev-Society/OpenStock) is an open-source alternative built by the Open Dev Society with a simple promise: track real-time prices, set personalized alerts, and explore detailed company insights, built openly, for everyone, forever free. The project is written almost entirely in TypeScript on Next.js 15 and has climbed past sixteen thousand stars, earning a spot as a GitHub trending repository of the day along the way. This post walks through why it exists, how the application is architected, and how to run your own instance.

![High-level architecture overview of the OpenStock repository](/assets/img/diagrams/openstock/openstock-overview-architecture.svg)

## Why You Need This

Most market platforms lock the interesting features behind paywalls. Real-time quotes cost extra, alerts are a premium tier, and the underlying data belongs to the vendor. OpenStock flips that arrangement by giving you the whole application, under the AGPL-3.0 license, to self-host and modify as you like. If you deploy a modified version, the license asks you to share your source in return, which keeps the ecosystem open.

There is also a learning argument. OpenStock is a complete, modern full-stack application: server-side rendering with the Next.js App Router, authentication, database modeling, third-party API integration, background jobs, and AI-generated content, all in one approachable codebase. The project is community-built and credits [Adrian Hajdin's stock market app tutorial](https://github.com/adrianhajdin) as its starting point, then extends the idea into a genuinely useful product with onboarding, sentiment insights, and an automation layer.

Finally, there is the manifesto. The Open Dev Society states plainly that knowledge should not hide behind paywalls and that the project will never charge for access. OpenStock is not a brokerage and does not give financial advice; it is a tracking and insight tool. But as a demonstration that a polished market app can live entirely outside the subscription economy, it is a compelling piece of evidence.

## How It Works

The diagram below maps the main components of the repository and how they connect.

![Detailed architecture of the OpenStock repository](/assets/img/diagrams/openstock/openstock-architecture.svg)

**Frontend routes and components.** The app uses the Next.js App Router with two route groups. The [authentication routes](https://nextjs.org/docs/app) cover sign-in, sign-up, forgot-password, and reset-password flows. The main app group contains the dashboard home, the dynamic stock detail page, the watchlist page, and supporting pages. On top of these sit shared components: a command palette for global stock search bound to Cmd or Ctrl plus K, a wrapper that embeds [TradingView widgets](https://www.tradingview.com/widget/) for charts, heatmaps, quotes, and timelines, and the watchlist button that appears on every stock page. The interface is built from [shadcn/ui](https://ui.shadcn.com) and [Radix UI](https://www.radix-ui.com) primitives styled with [Tailwind CSS](https://tailwindcss.com), dark theme by default.

**Server layer.** Every request passes through middleware that validates the session and protects all routes except sign-in, sign-up, and static assets. Data access happens through server actions rather than a separate API: Finnhub actions fetch symbol search results, company profiles, quotes, and market news; watchlist actions add or remove symbols; user actions store onboarding answers such as country, investment goals, risk tolerance, and preferred industry; and alert actions manage personalized price alerts. An optional set of sentiment actions calls the Adanos API for structured sentiment snapshots across Reddit, X.com, news, and Polymarket, used only for the stock detail sentiment card.

**Authentication and data.** [Better Auth](https://www.better-auth.com) handles email and password authentication with a MongoDB adapter, and [Mongoose](https://mongoosejs.com) provides the connection layer and schemas. The database currently models user watchlists, with a unique symbol constraint per user, and price alerts. Because MongoDB is the only stateful dependency, the whole stack is two containers in Docker Compose: the app and the database.

**Automation, AI, and email.** An [Inngest](https://www.inngest.com) endpoint receives events and cron triggers. When a user signs up, a workflow fires that generates a personalized welcome email, calling an AI provider abstraction that defaults to [Google Gemini](https://ai.google.dev/gemini-api/docs) with MiniMax and Siray as configured alternatives. A daily cron job runs at noon and builds a news summary per user by pulling fresh stories for the symbols in that user's watchlist, then sends it through [Nodemailer](https://nodemailer.com) over Gmail transport. Prompt templates live in their own module, so the AI-generated content is easy to inspect and adjust.

## Advantages

- **Truly free and open.** The entire application is available under AGPL-3.0, with no feature gates and no vendor lock-in; you host it, you own it.
- **A modern, coherent stack.** Next.js 15, React 19, TypeScript strict mode, Tailwind v4, and shadcn/ui make the codebase pleasant to read and extend.
- **Server actions over boilerplate.** Data mutations go through typed server actions, which keeps the client thin and the data layer auditable in one folder.
- **Automation as a first-class feature.** Inngest workflows and cron jobs, rather than ad hoc timers, drive welcome emails and daily summaries, and they run identically in development and production.
- **Pluggable AI.** The provider abstraction means the AI behind welcome emails is a configuration choice, not a hard dependency.
- **Optional extras stay optional.** Sentiment insights degrade gracefully; without an Adanos key the rest of the app is unaffected.

## Benefits

The obvious benefit is money: quotes, watchlists, alerts, charts, and news summaries that usually sit behind premium tiers run on your own hardware against free data tiers. Market coverage spans more than thirty international exchanges through Finnhub, with the honest caveat, documented in the repository, that free-tier quotes for non-US markets can be delayed and some TradingView widgets restrict emerging markets.

You also gain control. Your watchlists and profiles live in your own MongoDB instance, not in a vendor's data store feeding someone else's product decisions. Self-hosting with Docker Compose takes two services and one configuration file, and because the app is stateless outside the database, moving hosts is trivial.

For developers, the benefit is a reference implementation. Want to learn how middleware guards a Next.js app, how server actions replace REST endpoints, or how event-driven email automation fits together? Every layer is small enough to read in an afternoon and honest enough to copy from.

## Usage

Getting started takes a few minutes. Clone the repository, install dependencies with pnpm or npm, and create a `.env` file with a MongoDB URI (local via Docker Compose or hosted on Atlas), a Better Auth secret, a free [Finnhub](https://finnhub.io) API key, Gmail credentials for email, and optionally a Gemini key for AI-personalized content. Verify the database connection with the included test script, then start the development server:

```bash
pnpm dev
```

For the full experience, run the Inngest development server in a second terminal so workflows and the daily news cron are active:

```bash
npx inngest-cli@latest dev
```

Or skip configuration entirely with Docker Compose, which starts the app and a MongoDB instance with persistent storage:

```bash
docker compose up -d mongodb
docker compose up -d --build
```

Once running at localhost:3000, sign up to trigger your own welcome email, complete the onboarding questions, and press Cmd or Ctrl plus K to search symbols from Finnhub's [symbol lookup](https://finnhub.io/docs/api/symbol-lookup), with coverage for [more than thirty exchanges](https://github.com/Open-Dev-Society/OpenStock) documented in the repository. Add stocks to your watchlist, open a symbol page to see TradingView charts, company profile, financials, and optional sentiment, and set alerts that reflect your own risk tolerance. If you enjoy pairing market tools with AI, our earlier coverage of [FinceptTerminal](https://pyshine.com/FinceptTerminal-Open-Source-Financial-Intelligence-Platform/) and the [AI Hedge Fund multi-agent system](https://pyshine.com/AI-Hedge-Fund-Multi-Agent-Investment-System/) makes good companion reading. And remember the project's own disclaimer: OpenStock is community-built and not a brokerage, and nothing it shows is financial advice.

## Conclusion

OpenStock proves that the features market platforms charge for, real-time tracking, alerts, curated news, and personalized insights, can be delivered by an open-source application running on free data tiers. Its architecture is a model of modern full-stack practice: protected routes behind middleware, typed server actions, Mongoose models, event-driven automation, and pluggable AI, all in a codebase a single afternoon can cover. Clone it, host it, extend it, and if you improve it, the license asks one thing in return: share what you built.
