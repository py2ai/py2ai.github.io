---
layout: post
title: "Codex with ChatGPT: ChatGPT Thinks, Codex Works - Inside XiaoDuoYa/codex-with-chatgpt"
description: "A source tour of XiaoDuoYa/codex-with-chatgpt, a TypeScript bridge that turns the ChatGPT web app into the planning and review brain for Codex coding sessions. Explore the read-only MCP data plane, the tiny C2C control protocol, OAuth 2.1 pairing, and the model-routing seam that keeps execution local."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Codex-With-ChatGPT-ChatGPT-Thinks-Codex-Works/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/codex-with-chatgpt/xiaoduoya-codex-with-chatgpt-architecture.svg
tags:
  - MCP
  - Codex
  - ChatGPT
  - AI Agents
categories: [AI, Open Source]
keywords: "codex-with-chatgpt, ChatGPT MCP bridge, Codex harness, AI coding agent, model routing, planner executor split, read-only MCP tools, OAuth 2.1 PKCE, Cloudflare tunnel, MCP server, TypeScript, OpenAI Codex, ChatGPT connector, source tour, pyshine"
author: "PyShine"
---

If you pay for a ChatGPT Plus or Pro subscription and also run an agentic coding tool, you have probably felt the same irony the author of this project did: the generous web-app quota sits mostly idle, while the coding agent grinds through scarce API tokens doing the two things that burn the most reasoning budget — planning what to build and reviewing what was built. The expensive thinking happens on the metered pipe; the prepaid pipe goes unused.

**Codex with ChatGPT** (the repo's own tagline: "ChatGPT thinks. Codex works.") is a TypeScript project that flips that arrangement. It turns the ChatGPT web app into the planning and review brain for Codex coding sessions, while the Codex harness keeps full ownership of execution: editing files, running shells, applying git, and running tests. There are no API keys to configure and no reverse proxy to deploy — the bridge uses the official ChatGPT web UI with a user-created connector, backed by a local OAuth-protected server that ChatGPT reaches through a Cloudflare tunnel.

The source is worth a tour because the interesting engineering is not in prompts but in the seam between two models. The repo implements model routing as an explicit, inspectable protocol: one plane carries tiny control messages between Codex and ChatGPT, the other lets ChatGPT pull exactly the code it needs from your disk through nine read-only MCP tools. Every security decision — path containment, token scoping, log sanitization — is visible in a few hundred lines of TypeScript, and it is a genuinely different design from the browser-wrapper style of ChatGPT-to-Codex integrations.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/codex-with-chatgpt/xiaoduoya-codex-with-chatgpt-overview-architecture.svg" alt="Architecture overview of the XiaoDuoYa/codex-with-chatgpt repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the codex-with-chatgpt architecture: the Codex Skill drives the c2c CLI, which spawns the loopback-only C2C Bridge; the bridge exposes nine read-only MCP tools to ChatGPT through an OAuth-guarded Cloudflare tunnel, while execution records flow back for review.*

Reading the overview from left to right: the planning brain lives in `skill/SKILL.md` and `docs/protocol.md`, which define the C2C state machine and drive the `c2c` CLI in `src/cli`. The CLI, via `src/process`, spawns the Express-based bridge in `src/bridge/server.ts`, which mounts the MCP tool server from `src/mcp/server.ts` behind the OAuth and pairing machinery in `src/auth` and `src/pairing/manager.ts`. The bridge publishes itself through the Cloudflare tunnel providers in `src/tunnel`. On the data side, every MCP read funnels through `src/workspace` — path containment, sensitive-file policy, git, and search — while `src/execution` holds the evidence trail that ChatGPT reviews after each Codex iteration.

## Why You Need This

The first problem is economic. Planning and review are the most reasoning-heavy phases of an agentic coding loop, and if your agent performs them on metered API or Codex tokens, you are paying twice — once for the subscription you forgot to use and once for the tokens you keep buying. This project routes those phases to the ChatGPT web app you already fund, and leaves the cheap, mechanical execution loop to the Codex harness.

The second problem is trust. Most agents grade their own homework: they claim the tests pass and move on. Here, after Codex reports an iteration finished, ChatGPT is instructed — in the boot prompt encoded in `docs/protocol.md` — to independently inspect the actual git diff and execution records through MCP, and the protocol states plainly: "Do not assume an implementation succeeded just because Codex says so." Review becomes evidence-based rather than claim-based.

The third problem is privacy. The usual pattern for giving a web model context is uploading code, zips, or pasted files into the chat. This project refuses that trade: your repository is never uploaded. ChatGPT reads exactly the lines it needs, on demand, through a read-only MCP connection whose server literally does not implement any write, shell, or commit tool. Sensitive files — `.env` files, private keys, SSH directories, credentials — are denied by default at the `src/workspace` layer, before a single byte leaves your machine.

Finally, there is the durability problem that kills most hobby bridges: the tunnel URL rotates, ports collide, tokens expire, and the whole contraption silently rots. This repo treats repair as a first-class feature. The `c2c doctor` command in `src/cli/index.ts` diagnoses the bridge, MCP, OAuth, and tunnel, restarts what is broken, and — when the public address has changed — hands the Codex Skill exact instructions to delete and recreate that one workspace's ChatGPT connector, so recovery is automated rather than left to the user.

## How It Works

The system is best understood as two strictly separated planes stitched together by a small local server, with the Codex Skill acting as the conductor.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/codex-with-chatgpt/xiaoduoya-codex-with-chatgpt-architecture.svg" alt="Detailed architecture of the XiaoDuoYa/codex-with-chatgpt repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of codex-with-chatgpt: from the c2c CLI and daemon through the Express bridge, MCP tool server, OAuth 2.1 authorization server, tunnel providers, and the workspace/execution data layer, down to the vitest suite that covers the security-critical paths.*

### Understanding the Architecture

**The two-plane split is the model router.** Routing in this project is not a switch that picks between API endpoints; it is a protocol-level division of labor. The control plane is Computer Use: Codex and ChatGPT exchange tiny structured `[C2C]` messages — `INIT → PLAN → EXECUTED → REVIEW → DONE | BLOCKED` — that carry state, goals, and rationale but never file bodies, diffs, or logs, and are kept under 1 KB. The data plane is MCP: ChatGPT pulls whatever content it needs itself. Because the control messages carry no content, whichever model handles reasoning (ChatGPT web) and whichever handles execution (Codex) can be swapped at this seam without the protocol caring.

**The bridge is a paranoid loopback server.** `src/bridge/server.ts` assembles an Express app and refuses to bind anywhere but `127.0.0.1`, `::1`, or `localhost` — public exposure goes through the tunnel, period. It prefers port 48765 (from `src/config/paths.ts`) and falls back to an ephemeral port on conflict. The admin API the CLI uses is triple-guarded: the request must arrive on loopback, must not carry proxy headers like `cf-connecting-ip` (defense against reaching admin endpoints through the tunnel), and must present a per-process random admin token — anything else gets a blank 404 so the surface is not even advertised. A companion daemon in `src/process/daemon.ts` spawns the bridge detached, waits up to twenty seconds for health, and reuses an already-healthy instance instead of starting a second one.

**The auth chain is real OAuth, not a shared secret.** `src/auth/oauth.ts` implements an OAuth 2.1 authorization server: discovery metadata plus Protected Resource Metadata, dynamic client registration (RFC 7591), authorization code flow with PKCE — S256 only — refresh token rotation, and revocation, with opaque tokens stored as SHA-256 hashes. `src/auth/middleware.ts` guards `/mcp`: a missing or invalid token draws a 401 with a `WWW-Authenticate` challenge pointing at the resource metadata, and a token minted for a different workspace draws a 403, which is what makes "one workspace = one boundary" enforceable rather than aspirational. The only secret that ever touches a browser is a one-time pairing code from `src/pairing/manager.ts` — CSPRNG-generated, short-lived, rate-limited, and destroyed on use.

**Every read passes a gauntlet.** All nine tools in `src/mcp/server.ts` — `workspace_info`, `list_directory`, `read_file`, `search_workspace`, `git_status`, `git_diff`, `test_status`, `execution_summary`, and `execution_output` — are annotated read-only and check a scope (`workspace.read`, `workspace.search`, `git.read`, `execution.read`) per call, and each tool description embeds an untrusted-data note warning the model that file contents are data, never instructions. `src/workspace/manager.ts` resolves untrusted paths by canonicalizing the deepest existing ancestor with `realpath` — which defeats symlink escapes even for not-yet-existing leaves — then verifies containment against the case-normalized workspace root, whose identity is the first twelve hex characters of a SHA-256 of the real path. `src/workspace/ignore.ts` hard-denies `.env` files, keys, `.ssh/`, `.aws/`, `.npmrc`, and similar patterns (`.env.example` explicitly allowed), layers user rules from `.c2cignore` on top, and filters noisy directories; reads are paginated, and `src/workspace/search.ts` runs ripgrep when available with a plain Node fallback.

**Review evidence is recorded, then gated.** When Codex finishes an iteration, the Skill calls `c2c record` (in `src/cli/index.ts`), which appends a JSONL execution record via `src/execution/records.ts` — task id, iteration, changed files, test summary, exit status. If a test/build/lint command ran, Codex may nominate its output through `src/execution/output.ts`, but nomination is not permission: `src/execution/sanitize.ts` is a deterministic gate that withholds anything containing a private key entirely, redacts GitHub PATs, `sk-` style keys, Slack tokens, AWS `AKIA` ids, Google `AIza` keys, and `apikey=` assignments, scrubs home-directory paths, and caps output at 64 KB and 200 lines. Restricted items appear in listings with metadata but no body, so ChatGPT knows the output exists and reviews from `git_diff` instead.

**The tunnel layer absorbs impermanence.** `src/tunnel` defines a `TunnelProvider` interface with two Cloudflare implementations: a Quick Tunnel that needs no account but produces a new `trycloudflare.com` URL on every start (the provider even health-checks that the URL actually fronts this bridge), and a Named Tunnel that gives a stable `c2c-<project>.your-domain.com` hostname after a one-time Cloudflare login. When the Quick URL rotates, `c2c doctor` detects the change and the Skill recreates the connector; if named provisioning fails, the code falls back to the Quick Tunnel rather than failing the session.

The end-to-end flow ties it together: Codex sends `INIT` with the user's goal; ChatGPT calls `workspace_info`, `read_file`, and `search_workspace` through the tunnel to understand the code, then replies with a finite, executable `PLAN`; Codex executes with its own harness, records the iteration, and sends a content-free `EXECUTED`; ChatGPT reviews the real diff and test records via `git_diff`, `test_status`, and `execution_output`, then answers `DONE`, another `PLAN`, or `BLOCKED`. Checkpoints in `src/session/state.ts` (`PLAN_RECEIVED`, `EXECUTED_SENT`, and friends) let a restarted Codex resume without re-sending anything, a `HANDOFF` message moves an in-flight task to a fresh conversation, and `maxIterations` — twelve by default, configurable in `.c2c.json` — stops runaway loops.

## Advantages

- **Subscription-first economics.** Planning and review run on the ChatGPT web quota you already pay for; no API key is ever configured, and the README is explicit that this is official web UI plus a read-only MCP bridge — no reverse engineering, no proxying.
- **Read-only by construction.** Write, delete, shell, and commit tools simply do not exist on the MCP server, so no prompt injection can enable what is not implemented; scope checks and `readOnlyHint` annotations back that up per tool.
- **A real security boundary per workspace.** Tokens are audience-bound to a single workspace (403 on mismatch), paths are canonicalized and contained, sensitive files are denied by default, and the full threat model is documented in `docs/security.md`.
- **Independent, evidence-based review.** ChatGPT inspects actual git diffs and recorded execution metadata instead of trusting "all tests passed," which catches the classic agent failure mode of optimistic self-reporting.
- **Self-repairing operations.** Port fallback, daemon reuse, tunnel health checks, `c2c doctor --fix`, and scripted connector recovery mean the happy path keeps working after restarts without user intervention.
- **Tested where it matters.** A vitest suite the README counts at 150 tests exercises path security, OAuth, pairing, tunnel behavior, and MCP end-to-end flows — the exact code that a hostile input would hit first.

## Benefits

- **You stop paying twice for thinking.** The reasoning-heavy phases move to the prepaid web subscription, stretching metered Codex/API budget for the execution work it is actually good at.
- **Your repository stays on your disk.** Nothing is uploaded; ChatGPT reads the few lines it needs through a token-scoped, path-contained, sensitive-file-filtered connection, and credentials never live in the project.
- **No more context-stuffing pastes.** Because ChatGPT pulls its own data via MCP, the `[C2C]` control messages stay under 1 KB, chats stay readable, and the model always sees current code instead of stale pasted snapshots.
- **Sessions survive interruptions.** Local checkpoints and the HANDOFF protocol let a restarted Codex or a lost ChatGPT conversation resume the task from goal, progress, and next step — never from pasted logs.
- **One skill, many projects.** Each workspace gets its own connector and Project binding, so the same installed Skill handles any repo you open without reconfiguration.
- **Honest visibility into runs.** Test and build summaries flow through execution records, and opt-in command output passes a sanitizer that strips secrets before ChatGPT can ever read it.

## Usage

The repo is a pnpm workspace targeting Node.js 20+. From the README's developer section:

```bash
pnpm install
pnpm build          # -> dist/, exposes the `c2c` bin
pnpm test           # vitest: 150 tests (path security, OAuth, pairing, MCP e2e)

c2c setup           # bridge + tunnel + pairing code, all in one
c2c sandbox-allow   # whitelist the settings dir in Codex (macOS + Windows)
c2c status / doctor / pair / unpair / logs / stop
```

For the intended zero-touch flow, you install the Codex Skill and let Codex do the work — copy `skill/` to `~/.codex/skills/codex-with-chatgpt/`, then just tell Codex:

```text
"Set up Codex with ChatGPT."
"Use Codex with ChatGPT to implement XXX."
```

Requirements are Node.js >= 20, git, and `cloudflared` for the public connection (auto-detected; the Skill installs it). If QUIC is blocked on your network, the README suggests setting `C2C_TUNNEL_PROTOCOL=http2` and restarting the bridge.

## Conclusion

codex-with-chatgpt is a rare kind of integration project: instead of wrapping a model's API or scraping a UI, it draws a clean protocol line between thinking and doing, then engineers both sides of that line — a hardened read-only data plane and a terse control plane — so that each model does what it is best at with the least exposure of your code. The TypeScript source is small enough to read in an afternoon and dense with decisions worth stealing: the loopback-only bridge with its silent 404 admin surface, the realpath-based path containment, the deterministic output sanitizer, and the doctor-driven repair loop. If you are building any agent-to-agent or model-routing bridge of your own, this repository is a masterclass in doing it defensively. Note that it is an unofficial community project, not affiliated with or endorsed by OpenAI.

Links:

- GitHub repository: [https://github.com/XiaoDuoYa/codex-with-chatgpt](https://github.com/XiaoDuoYa/codex-with-chatgpt)
- Architecture docs: [docs/architecture.md](https://github.com/XiaoDuoYa/codex-with-chatgpt/blob/main/docs/architecture.md)
- C2C protocol: [docs/protocol.md](https://github.com/XiaoDuoYa/codex-with-chatgpt/blob/main/docs/protocol.md)
- Security threat model: [docs/security.md](https://github.com/XiaoDuoYa/codex-with-chatgpt/blob/main/docs/security.md)
