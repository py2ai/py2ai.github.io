---
layout: post
title: "BankMCP: Your AI Now Reads Your Bank - Inside noskillish/bankmcp"
description: "A source tour of BankMCP, a self-hosted, read-only Model Context Protocol server that connects AI assistants to 2,700+ European banks through the Enable Banking PSD2 API, with a single-user OAuth 2.1 edge, background balance watches with webhook alerts, and a strict no-payments, no-storage security model."
date: 2026-10-03
header-img: "img/post-bg.jpg"
permalink: /BankMCP-Your-AI-Now-Reads-Your-Bank/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/bankmcp/noskillish-bankmcp-architecture.svg
tags:
  - MCP
  - Open Banking
  - TypeScript
  - Security
categories: [AI, Open Source]
keywords: "MCP server, open banking, PSD2, Enable Banking, read-only banking, OAuth 2.1, Claude connector, self-hosted, balance watches, webhook notifications, TypeScript, personal finance AI"
author: "PyShine"
---

The questions are simple. Has the invoice from Acme been paid? What did we spend on groceries in August? Which subscriptions am I paying for, and what do they cost per year? Yet the answers live behind bank logins, per-bank apps and CSV exports, so most people either log in manually or hand their credentials to a screen-scraping service. BankMCP, by [noskillish](https://github.com/noskillish/bankmcp), takes the regulated third road: a small open-source MCP server you host yourself, wired through a licensed PSD2 aggregator, strictly read-only, with no payments and no third party holding your data.

BankMCP (npm package `bankmcp`, version 0.1.15, MIT, Node 24+) speaks the Model Context Protocol, so Claude, ChatGPT, Cursor, Mistral or a local Ollama model can ask about your accounts the way they ask about any other tool. Under the hood it connects to Enable Banking, which wraps 2,700+ European banks in one PSD2 API: your assistant talks OAuth to your server, your server talks JWT to Enable Banking, Enable Banking talks PSD2 to your bank. You log in once at your bank's own site; balances and transactions are fetched on demand and never written to disk.

The source is worth a tour because it is a working reference for a hard genre: an MCP server that touches real money data. Every `/mcp` request passes bearer authentication before any tool runs. The authorization story is a complete OAuth 2.1 server with exactly one user. Watches poll no more often than PSD2 allows. And the threat model - transaction descriptions written by strangers, webhook destinations fixed by the operator, a kill switch that logs every client out - is written down in the README and enforced in `src/`.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/bankmcp/noskillish-bankmcp-overview-architecture.svg" alt="Architecture overview of the noskillish/bankmcp repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the BankMCP repository: the HTTP edge authenticates and serves pages, the MCP surface exposes read-only tools and prompts, the Enable Banking client reaches your banks, and the state store backs background watches.*

Reading the overview from left to right: `src/app.ts` is the HTTP edge - it mounts the MCP endpoint, applies `requireBearerAuth` and serves the setup and login pages from `src/pages.ts`. Authenticated requests reach the MCP server in `src/mcp.ts`, which exposes the read-only tools in `src/tools.ts` and the finance prompts in `src/prompts.ts`. Tools resolve data through `src/enablebanking.ts`, the client for the PSD2 aggregator, with mapping in `src/data.ts`. `src/store.ts` keeps the single JSON state file - consents, account ids, watches, tokens - and `src/watcher.ts` runs the background rules that notify a webhook when something fires.

## Why You Need This

Open banking solved the plumbing problem years ago, but nobody solved the conversation problem. Your bank data sits behind a PSD2 API that answers machines, not questions, and the only easy interfaces are either the bank's own app or a fintech that wants to hold your data. A self-hosted MCP server flips the default: the data path runs from your bank through a licensed aggregator to *your* server, and the only reader is whichever assistant you point at it. Delete the state folder and the server forgets everything.

The read-only discipline is the second reason. There are no payment tools, deliberately - payments need a licensed PISP and a different security model entirely. What you get instead is the question space that matters day to day: booked and available balances per account, paginated transaction histories with signed amounts and one counterparty per line, human labels for accounts ("Everyday", "Joint expenses") that every tool accepts instead of raw ids, and consent management with expiry warnings, since PSD2 consents last up to 180 days and can be renewed from the same conversation.

The third reason is proactive alerting without a fintech subscription. Watches are server-side rules - balance below or above a threshold, a single debit over an amount, an incoming or outgoing payment matching a name, or "tell me if this payment has not arrived by this date". They are checked at most four times a day, which is precisely the PSD2 limit for unattended access, and notifications go to one operator-configured webhook as a Slack message or JSON POST.

Finally, the security posture is unusually legible for this category. Five wrong passwords lock an address out for fifteen minutes; twenty from all addresses lock sign-in entirely. Redirect URIs are restricted to known MCP client domains so a phishing link cannot route your sign-in elsewhere. Tokens are stored hashed. Every successful sign-in is logged and can ping your webhook, so a sign-in you did not make is your alarm - and changing the admin password hash is the kill switch that logs every client out.

## How It Works

BankMCP runs in two modes: a local stdio server launched by `npx -y bankmcp` with no OAuth at all, and a hosted HTTP server where the OAuth edge earns its keep.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/bankmcp/noskillish-bankmcp-architecture.svg" alt="Detailed architecture of the noskillish/bankmcp repository" style="max-width:100%;height:auto;" />
</div>

*The detailed view: entry points and config, the HTTP and OAuth edge with its setup flow, the MCP tool surface, the Enable Banking registration and data path, the state store and watch runner, and the delivery surfaces from Docker to the Claude plugin.*

### Understanding the Architecture

**Entry points fork early by deployment mode.** `bin/bankmcp.js` starts `src/local.ts`, which bridges stdio for desktop clients; `src/server.ts` boots the HTTP app for hosted deployments, with `src/config.ts` reading environment variables such as `DATA_DIR`, `BASE_URL`, `BIND_HOST`, `NOTIFY_WEBHOOK_URL` and `TLS_CERT_PATH`/`TLS_KEY_PATH`. `src/cli.ts` carries the operator commands: `check` to verify config and the Enable Banking application, `hash-password`, and `watch --force` to run all watches once and print what fired.

**The setup flow is a state machine served by the app itself.** A fresh server shows pages built by `src/pages.ts` and orchestrated by `src/setup.ts`: it can email a sign-in link through Enable Banking's control panel via `src/controlpanel.ts`, or list the exact values to register an application by hand. Before anything is stored, the application id and key are verified against Enable Banking. The server generates its own localhost TLS certificate because the bank redirect requires https; mkcert is supported for a trusted warning-free alternative.

**The OAuth edge is a complete single-user authorization server.** `src/auth.ts` implements `SingleUserProvider` on top of the MCP SDK's OAuth support: discovery, dynamic client registration and PKCE, tokens issued only after the password check in `completeLogin`, stored hashed. `src/app.ts` guards every `/mcp` request with `requireBearerAuth` before any tool executes. In local mode none of this applies - the client on your machine is the trust boundary.

**Tools are thin, read-only windows over the aggregator.** `src/tools.ts` exposes `list_banks`, `start_consent`, `consent_status`, `disconnect_bank`, `list_accounts`, `set_account_label`, `get_balances`, `get_transactions`, and the watch quartet `create_watch`, `list_watches`, `delete_watch`, `check_watches`. `src/enablebanking.ts` handles the JWT-signed API calls and consent flows; `src/data.ts` normalizes responses into signed amounts and one-counterparty descriptions so a model reads clean rows rather than raw bank payloads.

**Watches turn the server into a small scheduler.** `src/watcher.ts` evaluates the rules at most four times per day per account - the PSD2 unattended-access limit - tracks which transactions already fired so alerts do not repeat, and posts matched transactions to the single `NOTIFY_WEBHOOK_URL`. Because a watch cannot name its own webhook, an assistant that has been talked into something cannot exfiltrate transaction data elsewhere.

**Delivery surfaces meet users where they are.** `mcpb/manifest.json` packs the desktop extension via `scripts/build-mcpb.sh`; `compose.yaml`, `fly.toml` and a GitHub Actions workflow producing the `ghcr.io/noskillish/bankmcp` image cover hosting; the root marketplace manifest plus `plugin/.claude-plugin/plugin.json` deliver the Claude Code skills `/bank:setup`, `/bank:deploy` and the `bank` data-work skill, which keeps account maps and categorization rules on your side of the fence.

**End to end:** a hosted user opens the server URL, completes setup, adds `https://YOUR-HOST/mcp` as a connector, signs in with the admin password once, and says "connect my bank" - the `connect-bank` prompt returns a bank link, consent is approved at the bank, and from then on questions like "what did we spend on groceries in August" resolve through `get_transactions` and the `monthly-summary` or `build-budget` prompts.

## Advantages

- **Regulated plumbing, not screen scraping.** Account access runs through Enable Banking's licensed PSD2 API; you authenticate at your bank's own site and credentials never touch the server.
- **Read-only by construction.** No payment tools exist to abuse; every tool is a query, and the server never writes balances or transactions to disk.
- **Single-user OAuth 2.1.** Discovery, dynamic registration, PKCE, hashed tokens and lockout thresholds come standard - a rarity in self-hosted MCP servers.
- **Webhook exfiltration is structurally hard.** One operator-set https destination, no per-watch webhook, and matched-transaction deduplication.
- **Runs anywhere Node runs.** Local stdio for desktop clients, a container for hosted use, with Railway, Fly.io and Render one-click paths.
- **Model-agnostic.** Works with Claude, ChatGPT, Mistral, Cursor or a local Ollama model over stdio, so no AI vendor needs to see a transaction.

## Benefits

- **Ask instead of export.** Natural-language questions about balances, subscriptions and cash flow replace CSV archaeology.
- **Stay ahead of consent expiry.** Consent status is a tool, and renewal happens from the same conversation that notices the warning.
- **Catch the payment that did not arrive.** Expected-payment watches close the loop on invoices and refunds without a finance app.
- **Own your data path.** State is one JSON file you can back up or delete; no aggregator account stores your transactions.
- **Trust but verify the server.** Security notes name the exact files and functions where each check lives, and the test suite covers auth, tokens, store and watcher.
- **Adopt incrementally.** Start local with `npx bankmcp`, then move to a hosted connector when phone and claude.ai access matter.

## Usage

Local, no deployment (Node 24+):

```bash
claude mcp add bankmcp -- npx -y bankmcp
```

Or any MCP client with the same stdio command:

```json
{ "mcpServers": { "bankmcp": { "command": "npx", "args": ["-y", "bankmcp"] } } }
```

Hosted on your own box:

```bash
docker run -d --name bankmcp --restart unless-stopped \
  -p 8080:8080 -v bankmcp-data:/data ghcr.io/noskillish/bankmcp:latest
```

Add the connector and sign in:

```bash
claude mcp add --transport http bank https://YOUR-HOST/mcp
```

Operator commands:

```bash
npm start              # http server
npm run check          # verify config and the Enable Banking application
npm run hash-password  # produce ADMIN_PASSWORD_HASH
npm run watch -- --force   # run all watches once, print what fired
npm test               # unit tests (node:test)
```

Install the Claude Code plugin for guided setup and data-work rules:

```bash
export OPENBANK_URL=https://YOUR-HOST/mcp
/plugin marketplace add noskillish/bankmcp
/plugin install bank@bank
```

## Conclusion

BankMCP is a careful answer to a question more people are asking: how do I let my AI assistant see my finances without handing my data to yet another company? By combining a licensed PSD2 aggregator, a self-hosted read-only MCP server, and a genuinely thought-through single-user OAuth implementation, it gives assistants useful eyes on your accounts while keeping the data path, the storage and the alerts under your control.

Links:

- Repository: [https://github.com/noskillish/bankmcp](https://github.com/noskillish/bankmcp)
- Website: [https://bankmcp.dk/](https://bankmcp.dk/)
- Enable Banking: [https://enablebanking.com](https://enablebanking.com)
- Bank skill: [https://github.com/noskillish/bankmcp/blob/main/plugin/skills/bank/SKILL.md](https://github.com/noskillish/bankmcp/blob/main/plugin/skills/bank/SKILL.md)
