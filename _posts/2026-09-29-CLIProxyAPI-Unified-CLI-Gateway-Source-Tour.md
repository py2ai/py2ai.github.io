---
layout: post
title: "CLIProxyAPI: One Local Gateway for Your CLI AI Subscriptions - Inside router-for-me/CLIProxyAPI"
description: "A source tour of router-for-me/CLIProxyAPI, the Go proxy that wraps CLI-based AI coding backends like Claude Code, Codex, and Gemini CLI behind unified OpenAI, Gemini, and Claude-compatible API endpoints. We walk the proxy core, provider executors, OAuth token management, and the request translation layer."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /CLIProxyAPI-Unified-CLI-Gateway-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/cliproxyapi/router-for-me-cliproxyapi-architecture.svg
tags:
  - AI Proxy
  - Go
  - API Gateway
  - Claude Code
categories: [AI, Open Source]
keywords: "CLIProxyAPI, router-for-me, Claude Code proxy, Codex proxy, Gemini CLI proxy, OpenAI compatible API, Go proxy server, OAuth token management, multi-account load balancing, request translation layer, local AI gateway, anthropic messages API, gemini generateContent, self-hosted AI gateway"
author: "PyShine"
---

AI coding assistants have quietly built their own walled gardens. Claude Code speaks to Anthropic with your Claude subscription, Codex CLI talks to OpenAI through your ChatGPT plan, and Gemini CLI carries your Google account. Each tool works beautifully in its own terminal — until you want to use any of those subscriptions from something else: a script, an IDE extension, your own agent harness. The credentials exist, the quota exists, but there is no API endpoint you can point at.

CLIProxyAPI, from the router-for-me organization, is the missing adapter. It is a self-hosted proxy server, written in Go, that sits between your clients and those CLI-based providers and exposes unified OpenAI (including Responses), Gemini (including Interactions), and Claude-compatible endpoints. Your existing tools keep talking their native protocol; behind the scenes the proxy translates requests, selects one of your logged-in CLI accounts, and forwards the call upstream.

The repository is worth more than a quick skim. Under a modest entry point sits a genuinely layered system: a pairwise protocol-translation registry, a credential "conductor" with cooldowns and weighted scheduling, a family of per-provider executors, pluggable token persistence, and a hot-reloading configuration watcher. Reading it is a compact lesson in how to build a protocol-multiplexing gateway that survives real-world token expiry and rate limits. Let's walk the source.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/cliproxyapi/router-for-me-cliproxyapi-overview-architecture.svg" alt="Architecture overview of the router-for-me/CLIProxyAPI repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the CLIProxyAPI architecture: the Gin HTTP surface, the translator registry, the credential conductor with its OAuth flows and token store, the provider executors, and the platform services that keep configuration live.*

Reading the overview from left to right: the server binary boots the SDK-level `Service` in `sdk/cliproxy/service.go`, which hosts the Gin HTTP server defined in `internal/api`. Inbound requests reach the protocol handlers in `sdk/api/handlers`, which lean on the translator registry in `sdk/translator` to reshape payloads between wire formats, and on the auth conductor in `sdk/cliproxy/auth` to pick a credential — the conductor draws on the OAuth flows in `internal/auth` and persists tokens through the backends in `internal/store`. Execution bottoms out in the provider executors under `internal/runtime/executor`, all of which implement the executor contract declared in `sdk/cliproxy/executor/types.go`, while `internal/config` and `internal/watcher` keep the whole thing reconfigurable without a restart.

## Why You Need This

The first problem is access, not capability. Subscriptions attached to Claude Code, Codex, Gemini CLI, Kimi, or Grok's Build tooling are consumed through CLI-specific OAuth logins, not through public API keys. If your monthly plan includes generous quota, that quota is effectively stranded inside one terminal tool. CLIProxyAPI turns those OAuth logins into a local HTTP service: you run `--claude-login` or `--codex-login` once, and from then on any OpenAI-, Gemini-, or Claude-compatible client can spend that quota through a normal endpoint.

The second problem is protocol fragmentation. A Claude client POSTs to `/v1/messages` with Anthropic's message schema; a Gemini client calls `models/*/generateContent`; everyone else expects `/v1/chat/completions`. Upstream providers accept only one of those shapes each. Writing a client against a specific backend locks you in, and writing a translation layer yourself is a multi-week detour into streaming semantics, tool-call encodings, and thinking-block quirks.

The third problem is the single-account ceiling. Any one login eventually hits its rate window — Anthropic's five-hour windows, Codex quotas, Gemini limits. If you have several accounts, manually swapping them is untenable. CLIProxyAPI's auth layer implements multi-account pools with round-robin scheduling, weighted and prioritized selection, per-credential cooldowns driven by quota signals like `429` retry hints, and automatic token refresh before credentials go stale.

Finally, there is the operational grind: where do OAuth tokens live, how do they survive restarts, how do you rotate config without downtime? The project answers each concretely — token records behind a `Store` interface with file, Git, S3-compatible object, and Postgres backends, plus an fsnotify-based watcher that applies configuration and account changes to a running server.

## How It Works

The cleanest way into this codebase is to follow one request from socket to upstream and back, with the detailed diagram as a map.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/cliproxyapi/router-for-me-cliproxyapi-architecture.svg" alt="Detailed architecture of the router-for-me/CLIProxyAPI repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of CLIProxyAPI: route table and handler packages, the pairwise translator registry fed by per-provider translator packages, the conductor with selector, scheduler, cooldowns, refresh loop and token store, the executor family, and the persistence and watcher platform services.*

### Understanding the Architecture

**The entry point and service core.** `cmd/server/main.go` parses a surprisingly rich CLI surface: besides server mode it offers per-provider login flags (`--claude-login`, `--codex-login`, `--antigravity-login`, `--kimi-login`, `--xai-login`, `--devin-login`, `--meta-login`), Vertex service-account import, LAN discovery via DNS-SD, and a bubbletea-based `--tui` management mode. It hands everything to `Service` in `sdk/cliproxy/service.go`, the lifecycle owner that assembles the HTTP server, the legacy and core auth managers, the access manager, the config watcher, the pprof server, and the mDNS advertiser. Because `Service` is exported from the `sdk` tree, other Go programs can embed the entire proxy — the docs in `docs/sdk-usage.md` show how.

**The protocol translation layer.** The heart of interoperability is a registry, not an if-else chain. In `sdk/translator/registry.go`, `Register(from, to Format, request RequestTransform, response ResponseTransform)` records directional converters between `Format` identifiers (plain strings naming schemas like OpenAI chat-completions, OpenAI Responses, Claude messages, Gemini, Interactions, Codex, and Antigravity). The blank import in `internal/translator/init.go` is the wiring manifest: it pulls in packages such as `internal/translator/claude/openai/chat-completions` or `internal/translator/gemini/claude`, each of which registers its pair at init time. Adding a new wire format means adding a package and one import line, not touching the request path.

**The credential conductor.** `sdk/cliproxy/auth/conductor.go` defines the `ProviderExecutor` contract (`Execute`, `ExecuteStream`, `Refresh`, `CountTokens`, `HttpRequest`) and the `Manager` that orchestrates it. Around it lives an unusually mature scheduling stack: `selector.go` and `scheduler.go` choose among candidate credentials with round-robin, priority, and weight support (`weight.go`, `priority.go`); `cooldown_state.go` and `conductor_cooldown.go` park credentials that report quota exhaustion; `quota_signals.go` feeds those decisions from upstream responses; and `auto_refresh_loop.go` refreshes OAuth tokens in the background. Results flow back through a typed `Result` struct that distinguishes request-scoped failures from credential-scoped ones, so a malformed prompt never cools down a healthy account.

**The provider executors.** Each upstream gets a dedicated executor in `internal/runtime/executor`: `claude_executor.go` for Anthropic's messages API, `codex_executor.go` plus the `codex_websockets_*` family for OpenAI Codex including its duplex WebSocket mode, `gemini_executor.go` and `gemini_vertex_executor.go` and `aistudio_executor.go` for the Google flavors, `antigravity_executor.go`, `kimi_executor.go`, `xai_executor.go`, `devin_executor.go`, `meta_executor.go`, and `openai_compat_executor.go` for generic OpenAI-compatible upstreams configured in YAML. Executors receive a translated `Request` (model, payload bytes, format, metadata) plus `Options` carrying the original request bytes and stream flags, and they return either a full `Response` or a `StreamResult` whose `Chunks` channel is translated back to the client's format downstream.

**The HTTP surface and management plane.** `internal/api/server_routes.go` mounts the protocol facade: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages` and `/v1/messages/count_tokens`, `/v1beta/models/*action` and `/v1beta/interactions` for Gemini, plus realtime WebSocket routes under `/v1/realtime` and Codex-CLI-friendly aliases under `/backend-api/codex`. An `AuthMiddleware` validates client API keys through the access manager before any handler runs. A separate management API (legacy `/v0/management` and the v8 layout) guarded by its own hashed secret drives account import, credential inspection, and configuration edits, with a bundled web panel served at `/management.html`.

**The platform services.** Tokens and runtime state persist through implementations of the `Store` interface in `sdk/cliproxy/auth/store.go`: `internal/store` ships Git-backed, S3-compatible object, and Postgres backends (`gitstore.go`, `objectstore.go`, `postgresstore.go`, `postgres_cooldown_store.go`) on top of the default local auth-file directory. `internal/watcher` watches configuration and credential files with fsnotify and streams updates into the service, so adding an account or editing a route takes effect without a restart. A plugin host (`internal/pluginhost` with the `sdk/pluginapi` contract) lets external code add executors or schedulers, and `examples/custom-provider` demonstrates the full recipe.

End to end, a call looks like this: a client POSTs its native payload to a versioned route; `AuthMiddleware` checks the client key; the matching handler in `sdk/api/handlers` resolves the requested model, asks the translator registry to convert the body into the target upstream schema, and hands the result to the auth manager. The conductor selects a cooled-in, weighted credential for that provider, applies any after-auth interceptors, and invokes the provider executor, which signs and sends the upstream request — streaming or not. Chunks or the full response come back through the reverse translation into the client's original format, while the conductor records success or quota signals for the next scheduling decision.

## Advantages

- **One endpoint, every dialect.** OpenAI chat-completions and Responses, Anthropic messages, and Gemini generateContent/Interactions are all first-class inbound routes, so mixed client fleets can share one gateway.
- **Translation as data, not branching.** The pairwise `Register(from, to, ...)` registry in `sdk/translator` keeps format support modular and auditable — every arrow is a small, testable package.
- **Production-grade credential pooling.** Round-robin, weighted, and prioritized selection with cooldowns, quota signals, session affinity, and automatic refresh turns a pile of OAuth logins into a resilient account pool.
- **Hot-reconfigurable.** The fsnotify watcher in `internal/watcher` applies config and account changes live, and the Management API plus web panel and TUI give you several ways to operate it.
- **Embeddable by design.** The SDK packages (`sdk/cliproxy`, `sdk/translator`, `sdk/pluginapi`) are documented for embedding in `docs/sdk-usage.md` and `docs/sdk-advanced.md`, with a working custom-provider example.
- **Pluggable durability.** Token and cooldown state can live in local files, a Git repo, S3-compatible object storage, or Postgres, matching anything from a laptop to a clustered deployment.

## Benefits

- **Spend subscriptions, not new budget.** OAuth logins for Claude Code, Codex, Gemini CLI, Kimi, Grok, Antigravity, and friends become ordinary API endpoints — no additional API keys to purchase.
- **Zero client rewrites.** Point an existing OpenAI- or Claude-compatible tool at the local port and it works; the proxy absorbs the protocol differences, streaming semantics, and tool-call encodings.
- **Fewer rate-limit interruptions.** Multi-account rotation with cooldowns and retry-aware scheduling keeps a workload flowing when any single account hits its window.
- **Runs where you run.** A single Go binary or the provided Docker Compose file deploys locally; credentials stay on your machine rather than in a third-party relay.
- **Operable without ceremony.** Health checks at `/healthz`, structured logging, a management control panel, optional TLS, and LAN discovery make it behave like a real service rather than a script.
- **A readable blueprint.** For Go developers, the repository doubles as a reference implementation of translator registries, executor interfaces, and credential state machines you can imitate in your own gateways.

## Usage

Copy the example configuration and adjust the listener and client keys (the default port is `8317`):

```bash
cp config.example.yaml config.yaml
```

The quickest start is Docker Compose, which ships in the repository with the `eceasy/cli-proxy-api` image and mounts your `config.yaml` and auth directory:

```bash
docker compose up -d
docker compose logs -f
```

Alternatively, build from source — the Dockerfile's build step shows the canonical target:

```bash
go build -o ./CLIProxyAPI ./cmd/server/
./CLIProxyAPI --config config.yaml
```

Authorize your CLI accounts once with the provider login flags:

```bash
./CLIProxyAPI --claude-login
./CLIProxyAPI --codex-login
./CLIProxyAPI --antigravity-login
```

Then call it with any compatible client — the route table exposes `POST /v1/chat/completions`, `/v1/messages`, `/v1/responses`, and the Gemini `/v1beta` routes:

```bash
curl http://127.0.0.1:8317/v1/chat/completions \
  -H "Authorization: Bearer YOUR_CONFIGURED_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"Hello"}]}'
```

## Conclusion

CLIProxyAPI is a focused tool with an unglamorous job — make CLI-native AI subscriptions reachable from any client — and it does that job with unusually clean layering. The translator registry, the executor contract, and the credential conductor are three independent ideas that compose into one gateway, and each is worth studying in its own right. If you run AI coding tools across more than one provider, or you are building a gateway of your own, the source repays the tour.

Links:

- GitHub repository: [router-for-me/CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)
- Documentation and guides: [help.router-for.me](https://help.router-for.me/)
- SDK usage docs: [docs/sdk-usage.md](https://github.com/router-for-me/CLIProxyAPI/blob/main/docs/sdk-usage.md)
