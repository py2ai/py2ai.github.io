---
layout: post
title: "Treg: OpenRouter, But for Agent Tools - Inside superdesigndev/treg"
description: "A source tour of superdesigndev/treg, the open-source tools-registry that gives AI agents one token and one base URL for thousands of third-party APIs. We walk the credential-injecting proxy, the Fernet-encrypted secret store, the faithful-relay contract, and the layered FastAPI architecture behind it."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /Treg-OpenRouter-For-Agent-Tools/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/treg/superdesigndev-treg-architecture.svg
tags:
  - AI Agents
  - Python
  - API Gateway
  - Open Source
categories: [AI, Open Source]
keywords: "treg, tools-registry, superdesigndev, OpenRouter for tools, agent tools, MCP server, credential injection proxy, API aggregation, FastAPI, SQLModel, self-hosted registry, AI agents, tool catalog, Fernet encryption, Apache 2.0"
author: "PyShine"
---

Every agent framework eventually hits the same wall: the model is ready, the plan is written, and then the agent needs a work email, a backlink report, or a TikTok profile fetch — and that capability lives behind a subscription, a signup wall, or no public API at all. Model routing got its OpenRouter moment years ago; tool routing did not. The repo we are touring today, [superdesigndev/treg](https://github.com/superdesigndev/treg), is an attempt to give tools the same treatment: one catalog, one token, one base URL, priced per call.

Treg (published on PyPI as `tools-registry`) describes itself as "a remote registry that turns team skills into shareable, callable tools via a credential-injecting proxy." It ships as two things in one codebase: a light, pure-Python CLI that agents drive, and a self-hostable FastAPI server that holds the credentials, meters the calls, and streams bytes to upstream providers. There is a hosted instance at treg.to, but the README is explicit that anyone can run their own registry, and the repository carries everything needed to do it — server code, dashboard assets, a test suite of some 168 modules, and design documents for every subsystem.

What makes the source worth a tour is not the product pitch but the engineering discipline underneath it. Credential handling is the kind of problem where almost every shortcut is a security incident, and treg's authors clearly know it: secrets are Fernet-encrypted at rest, decrypted only at call time, injected through a pluggable seam, and guarded by a call-time SSRF check. The proxy core deliberately refuses to be clever. Reading how they drew those lines — and kept roughly 170 test files honest against them — is the real payoff of this repository.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/treg/superdesigndev-treg-overview-architecture.svg" alt="Architecture overview of the superdesigndev/treg repository" style="max-width:100%;height:auto;" />
</div>

*Overview of treg's architecture: client surfaces converge on the FastAPI app, the call application resolves a tool against the YAML catalog, and the streaming proxy injects decrypted credentials before relaying upstream.*

Reading the overview from left to right: agents and teammates arrive through one of three doors — the `treg` CLI (`src/treg/cli.py`), the mounted MCP front door (`src/treg/mcp.py`), or the dashboard assets under `src/treg/web` — and all of them talk to a single FastAPI application (`src/treg/api.py`). Requests that actually do work land on the `/call` adapter (`src/treg/routers/call.py`), which hands off to the call orchestration layer (`src/treg/application/call/service.py`). That layer resolves which tool you mean using the catalog store (`src/treg/domain/catalog/store.py`), which in turn reads the provider catalog — a database-shaped directory of 129 YAML files under `src/treg/catalog` describing vendors from Hunter to DataForSEO. Once a tool is resolved and authorized, the call is streamed through the faithful relay (`src/treg/infra/upstream/relay.py`), which applies credential bindings via the injector registry (`src/treg/infra/upstream/injectors.py`), decrypting each secret through the Fernet module (`src/treg/crypto.py`) and pulling tool and secret rows from the SQLModel tables (`src/treg/models.py`) only at the last moment.

## Why You Need This

The problem treg attacks is economic before it is technical. SEO and enrichment data — the exact inputs an agent needs for real marketing or sales work — sit behind subscriptions priced for daily professional use: the README cites Semrush at $139/month, Moz at $99/month, Crunchbase at $99/month, Apollo at $59/seat. Nobody buys those for a single agent run, and most providers offer no per-call tier at all. Treg's answer is to carry the accounts on the server side and bill fractions of a cent per call from a prepaid team balance, so an agent can spend two cents finding a work email instead of two hundred dollars a month renting the firehose.

The second problem is coordination. Even when a team does hold the right API keys, those keys end up pasted into `.env` files on every laptop that runs an agent. Treg inverts that: credentials are registered once — via `treg secret add`, a bulk `treg upload` from a `.env`, or an OAuth connect flow — and from then on every teammate's agent calls through the registry without ever holding the key. The token identifies the caller (`X-Treg-Token` on every request); the credential never leaves the server. The same mechanism covers vendor CLIs (`treg run stripe -- get /v1/balance` executes the real Stripe CLI with the org's credential injected) and whole skills, where a `SKILL.md` recipe, its secrets, and its tools are registered together as a bundle.

The third problem is honesty about what you are calling. Aggregators in this space tend to hide the vendor, silently fail over between providers, and blur who billed what. Treg takes the opposite stance, and it is refreshingly opinionated: `treg catalog search` shows competing providers side by side with their prices, and the README states plainly that choosing is yours — the registry does not silently pick or fail over between providers for you. A credential ladder governs each call in a fixed order: your team's own registered tool first, then a stored secret through a virtual tool, then a verified public route that needs no provider key, and only then treg's own key billed to your balance. An endpoint with no published price is refused rather than served free.

Finally, there is the self-hosting story. Everything the hosted service does is in the repository — the server extra (`pip install "tools-registry[server]"`) pulls FastAPI, SQLAlchemy, and the cryptography stack, and a one-command dev environment (`scripts/dev-local.sh up`) stands up a hot-reloading server on SQLite with email OTP dev mode. If your threat model says "our credentials never touch a third party's disk," the code lets you act on it.

## How It Works

The detailed graph below maps the modules that make a single proxied call happen, from the CLI to the upstream wire.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/treg/superdesigndev-treg-architecture.svg" alt="Detailed architecture of the superdesigndev/treg repository" style="max-width:100%;height:auto;" />
</div>

*The full call path: FastAPI assembly and client surfaces on top, the call application's resolve-authorize-relay-settle pipeline in the middle, with the streaming proxy, identity and trust modules, and the data layer underneath.*

### Understanding the Architecture

**The composition root.** The server is not assembled where you might expect. `src/treg/__main__.py` simply runs `uvicorn` against `treg.api:app`, and at the bottom of `src/treg/api.py` you find the real construction: `from .bootstrap import create_app; app = create_app()`. `src/treg/bootstrap.py` is the composition root — it builds the FastAPI app, wires in the routers for auth, orgs, billing, catalog, and calls, and mounts the MCP surfaces at `/mcp` and `/mcp/v2` (the latter being the curated Claude-connector catalog). This split keeps `src/treg/api.py` as the router registrations while lifecycle, middleware, and mounts live in bootstrap.

**The credential ladder and the resolver.** Every call enters through `src/treg/routers/call.py`, whose `call_tool` handler builds a `CallInput` from the raw request — deliberately using the raw, still-percent-encoded path so an encoded slash cannot change the upstream route — and then delegates to `create_call_context` and `execute_call` in `src/treg/application/call/service.py`. Resolution (`src/treg/application/call/resolve.py`) first tries your team's own tools, then falls down the catalog ladder using the catalog store's host-plus-longest-prefix matching against the 129 provider YAMLs. Authorization (`src/treg/application/call/authorize.py`) checks org roles, deny rules, and daily caps; billing (`src/treg/application/billing.py`) guards the prepaid balance against the `CreditBlock` and `LedgerEntry` tables.

**The faithful relay.** The heart of the system is `relay()` in `src/treg/infra/upstream/relay.py` — which the README calls "the whole product in one function." Its contract is unusual and strict: the proxy alters only three things on the way upstream — hop-by-hop transport headers, treg's own control headers and cookies, and the injected credentials — and everything else passes verbatim. The code shows the scar tissue of real-world relaying: `Accept-Encoding` is pinned to `identity` for metered calls so the settlement path can parse the provider's own reported charge (the comments recount a real bug where gzip made DataForSEO bill from an estimate instead of its reported amount), and query parameters are merged onto the URL manually because httpx would otherwise strip a catalog endpoint's own query string.

**The injector seam.** Credential injection is where many registries grow spaghetti; treg isolates it completely in `src/treg/infra/upstream/injectors.py`. A tool carries a list of bindings — plain dicts declaring a secret id, an injector name, a location (`header`, `query`, or `json`), a target name, and a format template. The relay never branches on auth shape; it just calls `INJECTORS[binding["injector"]]` per binding. Four shapes ship: `env` for plain API keys, `secret_file` for JSON token files, `oauth` for refreshable tokens, and `cli_auth` for material lifted from a CLI's keychain. Adding a fifth auth shape touches one file and never the proxy core — a textbook seam.

**Trust at the edges.** Secrets live Fernet-encrypted in the database (`src/treg/crypto.py`), decrypted only in memory at call time, and the module doubles as the token store, hashing API tokens with plain SHA-256 because a high-entropy random token needs lookup speed, not password stretching. OAuth tokens stay fresh through single-flight refresh (`src/treg/infra/oauth_refresh.py`), the connect flow mints the first token via browser consent (`src/treg/application/connect.py`), and `src/treg/health.py` can probe every registered credential and webhook the owner of anything broken. The relay even defends against DNS rebinding: the SSRF guard in `src/treg/infra/upstream/ssrf.py` re-resolves the upstream host at call time, refusing targets that now answer on an internal address.

**The audit trail and the platform layer.** Every call — including refusals and unexpected 500s — funnels through `record_call` in `src/treg/audit.py`, and the out-of-balance path is machine-readable by design: an HTTP 402 carrying `balance_micro`, `estimated_cost_micro`, and a `topup_url`, so an agent can act on it without parsing prose. Around the core, `src/treg/worker.py` runs the scheduled maintenance (`treg-worker`: capacity sweeps, overflow verification, async task settlement, folding audit rows into per-endpoint reliability stats), while `src/treg/convert.py` scaffolds a skill directory into a registerable bundle manifest and `src/treg/skills.py` backs the workflow skills the CLI installs.

End to end, a call reads like this: an agent sends `GET /call/https://api.intercom.io/conversations` with only `X-Treg-Token`; the router captures identity and raw path, the service resolves the tool against the catalog or team registry, authorization and balance gates pass, the relay decrypts the bound secret, injects it, streams the bytes upstream and the response back, and a deferred audit row lands with the call id returned in `X-Treg-Call-Id` — the caller never saw a credential, and the upstream never saw the token.

## Advantages

- **One credential surface for every tool.** The same `X-Treg-Token` works across the catalog, team-registered endpoints, vendor CLIs, and skills — there is no per-provider account plumbing for the agent to manage.
- **Credentials that never reach the client.** Secrets are Fernet-encrypted at rest, decrypted only inside the relay at call time, and injected server-side; callers hold a token, never a key.
- **A deliberately dumb proxy core.** The faithful-relay contract in `src/treg/infra/upstream/relay.py` plus the injector seam keeps auth shapes pluggable without ever teaching the proxy about a specific vendor.
- **Honest economics.** Per-call pricing with a visible credential ladder, side-by-side provider comparison, refusal instead of silent free service, and a structured 402 that agents can act on.
- **Real security engineering.** Call-time SSRF checks against DNS rebinding, session-cookie scrubbing, sandbox CSP headers on relayed responses, and idempotency handling that releases claims on failure.
- **Genuinely self-hostable.** The server extra, the Alembic schema, the dashboard assets, and a one-command tmux dev environment are all in the repository, with SQLite for development and Postgres for production.

## Benefits

- **For agent builders:** a single integration point — CLI, MCP, or plain HTTP — that turns "find me a work email" into a one-line call, with `treg catalog get` documenting parameters and price before you spend anything.
- **For teams:** shared tools and skills maintained in one place, org-scoped access with owner/admin/member/viewer roles, and per-member tool access control.
- **For security reviewers:** the properties you would demand are structurally enforced — no plaintext secrets, no token leakage upstream, no reflected XSS through the relay, audit rows for every refusal.
- **For operators:** the worker profile covers the boring reliability work — credential health probes, capacity sweeps, async settlement — so the registry does not silently rot.
- **For learners:** the layering is exemplary — routers are thin HTTP adapters, the application layer holds the use cases, `src/treg/domain/` and `src/treg/infra/` keep rules and adapters apart, and `docs/context/` documents each subsystem against its sources.
- **For the cautious:** the light CLI install (`pip install tools-registry`) pulls only three dependencies, and the local-proxy extra is optional — your machine does not carry the server stack to talk to a registry.

## Usage

Install the CLI and get productive immediately, from the repository's README:

```bash
# 1. install the CLI — also points it at the registry
curl -fsSL https://treg.to/install.sh | sh

# 2. sign in (GitHub default · --email for a one-time code · --token for agents/CI)
treg login

# 3. do something useful immediately — no key, nothing registered
treg catalog search "backlinks for a domain"
treg call hunter.people.email.find --query domain=reddit.com --query full_name="Alexis Ohanian"
treg balance
```

Register your own team's credentials and tools:

```bash
treg secret add STRIPE_KEY --value sk_live_123
treg add stripe --base-url https://api.stripe.com --secret STRIPE_KEY
treg upload env --select openai,stripe,resend   # or straight from the .env
treg scan     # read-only preview of what an upload would register
treg run stripe -- get /v1/balance              # vendor CLI with the org's credential injected
```

Run your own registry from source (needs `tmux` and `uv`):

```bash
scripts/dev-local.sh up          # server on http://localhost:18790, dev-safe settings
# or, directly:
bash scripts/build-dashboard.sh  # Node 22.12+ and npm; build the Dashboard
uv sync                          # create the venv from uv.lock
uv run python -m treg upgrade    # prepare schema + idempotent release tasks
uv run python -m treg            # serve on 0.0.0.0:18790 (add --reload for dev)
```

For a server install rather than from source, the base package is CLI-only — add the server extra:

```bash
pip install "tools-registry[server]"   # FastAPI, DB drivers, encryption
```

## Conclusion

Treg is one of the more thoughtfully engineered pieces of agent infrastructure you will find on GitHub right now. The idea — OpenRouter, but for tools — is easy to state, but the value is in the execution: a relay that refuses to model the upstream, an injector seam that keeps auth shapes out of the proxy core, Fernet-wrapped secrets with a real SSRF story, and a billing model that treats agents as first-class economic actors. The Apache 2.0 license carries one additional term worth noting: self-hosting and internal commercial use are explicitly encouraged, but redistributing the code as a competing hosted registry service requires written permission. If your agents keep stumbling over signup walls, or your team's API keys are sprinkled across `.env` files, reading this source — or running it — is time well spent.

Links:

- [GitHub repository: superdesigndev/treg](https://github.com/superdesigndev/treg)
- [Hosted service and dashboard: treg.to](https://treg.to)
- [Full CLI reference: USAGE.md](https://github.com/superdesigndev/treg/blob/main/USAGE.md)
- [Per-subsystem design docs: docs/context](https://github.com/superdesigndev/treg/tree/main/docs/context)
