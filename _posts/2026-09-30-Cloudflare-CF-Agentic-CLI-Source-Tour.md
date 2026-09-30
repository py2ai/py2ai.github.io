---
layout: post
title: "cf: The Agentic CLI for the Entire Cloudflare API - Inside cloudflare/cf"
description: "A source tour of cloudflare/cf, Cloudflare's official agentic CLI that maps the whole Cloudflare API onto agent-friendly commands. We walk the OpenAPI code generator, JSON-first output, OAuth profiles, dry-run safety gates, and the local Miniflare runtime that make it work."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Cloudflare-CF-Agentic-CLI-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/cf-cli/cloudflare-cf-cli-architecture.svg
tags:
  - Cloudflare
  - CLI
  - AI Agents
  - TypeScript
categories: [AI, Open Source]
keywords: "cloudflare cf, agentic CLI, cloudflare api, ai agents, developer tools, typescript cli, openapi code generation, mcp tools, oauth cli, json output, miniflare, cloudflare workers, cli architecture, source code tour, pyshine"
author: "PyShine"
---

Cloudflare's API surface is famously enormous — DNS, Workers, R2, KV, D1, Queues, Zero Trust, WAF, AI, and dozens of other products, each with its own endpoints and schema. Agents that try to drive that surface today have to guess at REST calls, juggle tokens, and hold API reference pages in their context window. The `cloudflare/cf` repository is Cloudflare's answer to that problem: an official, open-beta CLI named `cf` that is explicitly built for "the next generation of software development," where an AI agent, not a human, is often the one typing the commands. Reading its README, the pitch is blunt — agents should be able to find the command they need through bespoke search and steering, JSON should be the default interface, and a TypeScript `cloudflare.config.ts` should bring type safety to the whole platform.

The repository, currently published as an open beta (`cf` 1.0.0-beta.5), is a pnpm monorepo whose heart is `packages/cli`. What makes it worth a tour is that it is not a thin wrapper around `fetch`. It contains a code generator that turns Cloudflare's OpenAPI specification into a yargs command tree and a typed TypeScript SDK, an agent-detection layer that rewrites its own help output when a harness is detected, a MiniSearch-powered intent search, MCP tool definitions, and a set of safety gates around every mutating call.

In this post we walk the actual source: the entry pipeline in `packages/cli/src/index.ts`, the generator in `packages/cli/generate.ts`, the generated command pattern, the auth stack under `packages/cli/src/lib/oauth/`, and the dry-run and local-mode machinery that keeps both humans and agents out of trouble. Every path we cite exists in the tree, and the architecture diagrams below were compiled directly from the repository's real file structure.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/cf-cli/cloudflare-cf-cli-overview-architecture.svg" alt="Architecture overview of the cloudflare/cf repository" style="max-width:100%;height:auto;" />
</div>

*Overview of cloudflare/cf: the bin launcher and yargs root dispatch into a lazily loaded command tree — a generated registry covering the Cloudflare API plus a small set of hand-written roots — while an agent-facing surface (search, schema, tools) and the auth/output platform layer round out the system.*

Reading the overview from left to right: the `bin/cf` launcher checks the Node version and loads the bundle into the yargs root at `packages/cli/src/index.ts`, which registers commands lazily through `packages/cli/src/lib/lazy-command.ts`. The generated command registry at `packages/cli/src/commands/_generated/index.ts` is produced offline by the OpenAPI codegen in `packages/cli/generate.ts`. Alongside it, an agent surface — `cf cli search`, `cf schema`, and the hidden `cf tools` — gives harnesses discovery and introspection primitives, while `packages/cli/src/lib/auth.ts` builds an authenticated API client and `packages/cli/src/lib/output.ts` guarantees that every result comes back as clean JSON.

## Why You Need This

The first problem is discoverability at scale. The source of `packages/cli/src/index.ts` notes, in the comment that justifies the bare-`cf` splash screen, that the CLI ships roughly 6,400 generated commands, organized as 177 generated product directories under `packages/cli/src/commands/_generated/`. No human memorizes that, and no agent should burn context exploring nested `--help` calls. The CLI's own help output says it outright: when an agent harness is detected, `packages/cli/src/lib/agent-context.ts` flips a flag and the help text is prefixed with an "AGENT COMMAND DISCOVERY" block that tells the agent to stop chaining help calls and use `cf cli search` instead. That search, implemented in `packages/cli/src/commands/cli/search.ts`, builds a MiniSearch index over command metadata with fuzzy matching (0.2 tolerance), prefix matching, and field boosts weighted toward the command path, then returns five compact JSON matches.

The second problem is context economy. Agents pay for tokens, and pretty-printed prose responses are expensive. `cf` makes JSON the default interface: `packages/cli/src/lib/output.ts` emits pretty-printed, syntax-highlighted JSON on TTYs, stays silent on stdout when a mutation returns no payload (so scripts piping to `jq` never see the literal string `null`), and writes only a one-line confirmation to stderr for humans. The README describes this as JSON "pretty printed for humans and condensed for agents for maximum context savings." Notably, the `--json`, `--ndjson`, and `--pretty` flags are deliberately reserved but not implemented — the source comments in `index.ts` state that output is all-JSON today and the names are claimed to prevent future collisions.

The third problem is safety. An agent that can create, update, and delete production DNS records is one hallucinated flag away from an outage. Rather than trusting prompts, `cf` builds safety into the command schema itself: every generated mutating command accepts `--dry-run`, which validates the request and prints the exact URL, HTTP method, resolved path parameters, query string, and body without sending anything (see `packages/cli/src/lib/dry-run.ts`). Interactive confirmations go through `packages/cli/src/lib/dialog.ts`, whose clack-based `confirm` fails closed — returning the fallback value, defaulting to `false`, in non-interactive or CI environments.

The fourth problem is credential hygiene. Agents should not be handed long-lived API tokens embedded in shell histories. The CLI ships a full OAuth flow with multi-profile support under `packages/cli/src/lib/oauth/index.ts`, built on `@cloudflare/workers-auth/cf`, with env-token support for CI via `getAuthFromEnv` in `packages/cli/src/lib/auth.ts`, and telemetry that is sanitized to keep resource IDs, domains, and search queries out of event payloads.

## How It Works

The whole system is a pipeline from one OpenAPI document to thousands of typed, lazy-loaded commands, with an agent-aware dispatch layer on top.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/cf-cli/cloudflare-cf-cli-architecture.svg" alt="Detailed architecture of the cloudflare/cf CLI showing code generation, command surface, auth, safety gates, and local runtime" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of cloudflare/cf: the forge-based OpenAPI transformer emits builders, handlers, the command registry, and the TypeScript SDK; generated leaves flow through auth, body parsing, and dry-run gates into JSON output; hand-written roots and the Miniflare-backed local runtime extend the surface.*

### Understanding the Architecture

**The launcher stays deliberately thin.** `packages/cli/bin/cf` refuses to run on Node older than 22, suppresses a noisy `punycode` deprecation warning, and — before the large main bundle is ever imported — checks whether a project-local `cf` install should take over, using a small standalone chunk emitted from `packages/cli/src/lib/delegate.ts`. Only when staying local does it import the real bundle. This mirrors the wrangler-2-style delegation pattern and keeps global-invocation startup fast.

**The commands are generated, not hand-written.** `packages/cli/generate.ts` downloads a pinned `openapi.forge.json` asset from a Cloudflare forge release, feeds it to `initFromOpenApi` from the vendored `@cloudflare/forge` package, filters the operations for CLI audience, and rewrites two tracked trees: the yargs command sources in `packages/cli/src/commands/_generated/` and the typed SDK in `packages/cli/src/sdk/`. The per-command emitters live in `packages/cli/generator/emit/` — `builder.ts` emits the yargs flags, conflicts, and implies rules, while the handler emitter writes the request-assembly and output logic. Because both emitters are derived from the same intermediate representation, the flags a builder writes and the handler reads cannot drift apart.

**Every generated leaf follows one auditable pattern.** Take the example leaf `packages/cli/src/commands/_generated/dns/records/create.ts`: its builder registers typed flags plus `--dry-run` and a raw `--body` bypass; its handler, wrapped in `runWithTelemetry`, either prints the dry-run plan (method, URL, path params, query, body kind, body) through `formatDryRun`, or builds a client with `createCommandClient`, resolves the zone with `getZoneId`, calls `client.dns.records.create` on the generated SDK, and prints the result through `formatOutput`. JSON in, JSON out, with the same shape for every product from Access to Zero Trust.

**Agent support is a first-class subsystem, not a bolt-on.** `packages/cli/src/lib/agent-context.ts` detects harnesses from environment signals and reports whether the invocation is agentic. Three commands serve agents directly: `cf cli search` for intent-based discovery, `cf schema` (which, per its own docstring in `packages/cli/src/commands/schema.ts`, "always outputs JSON" — operation IDs, HTTP methods, paths, parameter tables, and request-body fields), and the hidden `cf tools`, which converts the command metadata in `packages/cli/src/lib/metadata.ts` into Model Context Protocol tool definitions, complete with JSON input schemas, so any MCP-compatible harness can mount the CLI as tools.

**Auth is layered and profile-aware.** `packages/cli/src/lib/auth.ts` composes the request stack: an OAuth token from `packages/cli/src/lib/oauth/index.ts` (which delegates identity, scopes, refresh, and profile storage to `@cloudflare/workers-auth/cf`), environment-token fallback for CI, default headers from `packages/cli/src/lib/request-headers.ts`, and account/zone resolution from `packages/cli/src/lib/context.ts`, all feeding a `CloudflareApiClient` with a 30-second default timeout. Global `--profile` and `--zone` flags let one shell work across accounts.

**Local development is a real API emulation, not a mock flag.** Passing `--local` reroutes fetches through `packages/cli/src/lib/local.ts` into `packages/cli/src/lib/local-runtime.ts`, which spawns a Miniflare instance over the persisted state directory and dispatches to a local-explorer worker; endpoints without a local equivalent fail loudly with a "no local equivalent" error instead of silently doing nothing. Meanwhile, `cf migrate` wraps the Wrangler-to-cf transition (`packages/cli/src/lib/wrangler-migration.ts`), `cf init` scaffolds projects with confirm prompts that fail closed in CI, and `cf deploy` uploads Build Output artifacts.

End to end, then: `bin/cf` delegates or loads the bundle; the yargs root lazily resolves the command; a generated leaf parses flags, assembles a typed request, and either prints a dry-run plan or authenticates through the OAuth/env stack and calls the generated SDK; results return as theme-highlighted JSON; and telemetry, sanitized of identifying values, flushes in the background with a one-second cap so it never delays exit.

## Advantages

- **Total API coverage from one spec.** Because commands are generated from Cloudflare's OpenAPI document via `@cloudflare/forge`, new API products appear as commands without hand-written plumbing, and the SDK in `packages/cli/src/sdk/` stays in lockstep with the CLI.
- **Agent-native discovery.** `cf cli search` with MiniSearch fuzzy matching, an `AGENT COMMAND DISCOVERY` block injected into help when a harness is detected, and hidden `cf tools` MCP definitions mean an agent can go from intent to the right command in one step.
- **JSON-first output everywhere.** Clean, pretty-printed JSON on stdout, human-only confirmations on stderr, and silent handling of empty results make the CLI composable with `jq` and trivial for agents to parse.
- **Fail-safe interactivity.** The clack-based dialogs in `packages/cli/src/lib/dialog.ts` return safe fallbacks in CI instead of hanging, and non-interactive runs cannot accidentally confirm destructive prompts.
- **Dry-run on every mutating command.** `--dry-run` prints the exact request — method, URL, path params, query, and body — before anything is sent, giving both humans and agents a rehearsal step.
- **Fast startup through laziness.** Lazy command shells, a lightweight delegation chunk, and deferred imports of OAuth, metadata, and Miniflare keep `cf --help` and single-command paths from paying for the full tree.

## Benefits

- **Lowered token costs for agent workflows.** Condensed JSON output and top-five search results are designed, per the README, "for maximum context savings" — directly reducing what an agent must read to act.
- **Safer automation by default.** Dry-run gates, fail-closed confirms, strict yargs parsing (`strictCommands` and `strictOptions` are set in the root), and sanitized telemetry reduce the blast radius of automated mistakes.
- **One mental model across Cloudflare.** Global `--zone`, `--profile`, `--local`, and `--quiet` flags plus a uniform command grammar replace dozens of product-specific tools and dashboards.
- **Type safety reaches configuration.** The `cloudflare.config.ts` format, re-exported from `packages/cli/src/config.ts` via `@cloudflare/config`, gives editors and language servers typed views of project configuration — the README calls this bringing "the safety and accuracy of TypeScript" to your agent through the LSP.
- **Reproducible local testing.** `--local` with `--persist-to` runs commands against an on-disk Miniflare state, so agent-driven workflows can be rehearsed without touching production.
- **A maintained open-source foundation.** The repo is dual-licensed under MIT and Apache-2.0, tested with vitest and MSW (including an imported wrangler test corpus in `packages/wrangler-tests`), and documented for both humans and agents via extensive `AGENTS.md` guidance at the root.

## Usage

Install the open beta globally (requires Node.js 22 or newer, per `packages/cli/bin/cf`):

```bash
npm i -g cf
```

Authenticate and explore:

```bash
cf auth login
cf --help
cf complete bash
```

Discover commands by intent, then inspect the underlying API schema — the exact loop the CLI prescribes for agents:

```bash
cf cli search "create a dns record for a zone"
cf schema dns records create
```

Rehearse a mutation before sending it, then execute it for real with a JSON body:

```bash
cf dns records create --zone example.com --dry-run --body @record.json
cf dns records create --zone example.com --body @record.json
```

Work against local Miniflare state instead of the production API:

```bash
cf --local --persist-to ./state kv namespace list
```

## Conclusion

`cloudflare/cf` is a compelling template for what "agent-ready infrastructure tooling" looks like when the platform vendor builds it themselves: a generated, exhaustive command surface; JSON as the contract; discovery, schema, and MCP introspection built for machines; and safety gates — dry-run, fail-closed prompts, sanitized telemetry — that assume automation is the primary caller. The source is clean, heavily commented TypeScript that explains its own design decisions, which makes it as useful to read as it is to run. If you are building CLI tooling that AI agents will drive, this repository is worth studying in detail.

Links:

- GitHub repository: [https://github.com/cloudflare/cf](https://github.com/cloudflare/cf)
- Cloudflare developer docs: [https://developers.cloudflare.com/](https://developers.cloudflare.com/)
