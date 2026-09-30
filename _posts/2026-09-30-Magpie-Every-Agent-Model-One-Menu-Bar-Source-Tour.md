---
layout: post
title: "Magpie: Every Agent's Model, One Menu Bar - Inside yetone/magpie"
description: "A source tour of yetone/magpie, the Go menu-bar app that lets any coding agent - Claude Code, Codex, Gemini CLI, OpenCode and more - run on any model through one local gateway. We walk the real code: protocol translation, key rotation, failover, and surgical config editing."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Magpie-Every-Agent-Model-One-Menu-Bar-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/magpie/yetone-magpie-architecture.svg
tags:
  - Go
  - AI Agents
  - Local Proxy
  - Open Source
categories: [AI, Open Source]
keywords: "magpie, yetone, Claude Code, Codex, model router, LLM proxy, local gateway, Go open source, AI coding agents, DeepSeek, Kimi, Qwen, menu bar app, Anthropic Messages API, OpenAI Responses API"
author: "PyShine"
---

Every coding agent ships with an opinion about who should serve its intelligence. Claude Code speaks the Anthropic Messages API and wants Anthropic keys, Codex wants OpenAI, Gemini CLI wants Google, and each one keeps its preference in a different file format in a different corner of your home directory. If you want Codex running on DeepSeek, or Claude Code on Kimi, you are normally left hand-editing TOML and JSON, translating between API dialects in your head, and hoping nothing else rewrites the file behind your back.

`magpie` by yetone is a small Go program that refuses all of that. It sits in the menu bar (the tray, on Windows and Linux), shows one screen listing every coding agent on the machine and the model each is set to - Claude Code, Codex, Gemini CLI, OpenCode, Pi, Goose, Cursor CLI, Copilot CLI, Crush, and more, twenty-seven agent integrations in all per the README's table - and you click a value and pick a model. That is the whole interface. The same screen opens as a normal window, runs in a terminal as `magpie tui`, and even serves itself into a browser with `magpie web` for WSL and SSH boxes.

The real reason to tour this source, though, is underneath the panel: a local gateway on `127.0.0.1:3425` that every agent points at, which speaks OpenAI Chat Completions, OpenAI Responses, Anthropic Messages and the Gemini API, and forwards each call to whichever vendor actually serves the model - translating between the APIs when it must, streaming and tool calls included. Getting that right is a genuinely hard distributed-systems problem compressed into a single binary, and the code in `internal/gateway` solves it with a clarity that repays reading.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/magpie/yetone-magpie-overview-architecture.svg" alt="Architecture overview of the yetone/magpie repository" style="max-width:100%;height:auto;" />
</div>

*Overview of magpie: three user surfaces on the left feed the gateway core, which routes every request through the rotation and failover engine out to providers, while the agent-wiring layer keeps each tool's config pointing at the loopback URL.*

Reading the overview from left to right: the entry points in `main.go` open the Wails-based menu bar app (`internal/gui/app.go`) or the bubbletea terminal UI (`internal/tui/tui.go`), and both surfaces drive the same agent registry (`internal/agent/agent.go`) that rewrites config files through the surgical editors. Every agent's requests land on the gateway server (`internal/gateway/gateway.go`), which parses each call into a protocol-neutral intermediate form (`internal/gateway/ir.go`), consults the key-and-account rotation engine (`internal/gateway/routing.go`) and the failover logic (`internal/gateway/fallback.go`), resolves the requested `provider/model` against the provider registry (`internal/provider/provider.go`) and the models.dev catalog (`internal/catalog/catalog.go`), and forwards to the vendor that speaks the model.

## Why You Need This

The first problem is vendor lock-in at the tool level. Your subscriptions are real assets: a Claude Code plan, a ChatGPT plan, a Copilot seat - each one usable only by the tool it came with, while the open-weight and pay-per-token models you actually want to try sit behind separate keys. magpie dissolves that wall in two directions. Any agent can pick any model you have configured, spelled `provider/model` and served by the gateway. And a subscription you have signed in to appears as a provider itself: the README describes how a Claude Code OAuth login, a ChatGPT login in `~/.codex/auth.json`, or a Copilot login shows up as *signed in as ...*, so every other agent can use its models through the gateway with nothing copied and no key pasted.

The second problem is configuration fragility. These config files are shared real estate - the agent writes them, its installer writes them, other switchers write them. Naive tooling that rewrites the whole file destroys comments and ordering, and a switcher that silently desyncs leaves an agent claiming one model while asking another vendor for it. magpie's answer is in `internal/edit`: purpose-built JSON/JSONC, TOML, YAML and dotenv editors that touch only the one key you changed and write atomically, plus `internal/agent/applied.go`, which remembers what magpie last set on each agent and warns you - Drift, in the code's vocabulary - when something else has broken that wiring.

The third problem is quota arithmetic. Once you have several keys and subscriptions per vendor, you are back to manually spreading requests, watching for 429s, and mentally tracking which allowance renews when. magpie's routing engine does that arithmetic per request: it knows the difference between out-of-credit, out-of-quota, rate-limited and briefly overloaded, and it rests a key or account for exactly as long as its failure says before trying it again.

And the fourth is vendor outages and model churn. When one provider is down or spent, a hard-wired agent stops. With magpie, the request moves to the next candidate that can take it, and when a vendor releases a model this morning, the picker shows it on the next refresh because the model lists are fetched live from the vendors - nothing is compiled in - with the models.dev catalog filling in names and reasoning levels.

## How It Works

One HTTP server, one intermediate representation, one routing engine, and a layer of per-agent config writers make the whole thing move.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/magpie/yetone-magpie-architecture.svg" alt="Detailed architecture of the yetone/magpie repository" style="max-width:100%;height:auto;" />
</div>

*The detailed tour: the gateway server and its wire codecs, the rotation and failover engine, the subscription bridges, the provider and catalog stores, and the per-agent wiring that rewrites each tool's config through the surgical editors.*

### Understanding the Architecture

**The gateway server.** `internal/gateway/gateway.go` defines `Server`, an `net/http` mux listening on loopback at `127.0.0.1:3425` by default (`MAGPIE_ADDR` changes it). Its `Handler()` registers the endpoints agents expect: `POST /v1/chat/completions`, `POST /v1/responses`, `POST /v1/messages`, token counting, image generation, and a Gemini-shaped `POST /v1beta/models/{model}:generateContent`. The bearer token is literally `magpie` - the code comments that agents insist on having a key, and since the gateway only listens on loopback it accepts anything. Each handled call becomes a `Call` record (agent, model, provider, protocol pair, status, milliseconds, time-to-first-token) kept in a mutex-guarded list of the last forty, which powers the Gateway tab's live routing view.

**The translation layer.** `internal/gateway/ir.go` declares the shape the client APIs have in common: messages of typed parts - text, image, file, tool call, tool result, thinking, even web-search hits. Requests are parsed into that IR and replies are produced from it, so a Codex request aimed at an Anthropic-only vendor, or a Claude Code request aimed at an OpenAI-compatible one, is a matter of re-encoding: `chat.go`, `anthropic.go` and `responses.go` each carry their wire codec, and `sse.go` handles the streaming side. One asymmetry is deliberate, as `internal/provider/provider.go` states: the Gemini protocol is served to clients but never spoken upstream - magpie translates Gemini calls into one of the three protocols vendors actually implement.

**The routing engine.** `internal/gateway/routing.go` spreads a provider's requests over its keys and accounts in four modes - `smart` (the default: of the subscriptions with quota to spare, the one whose allowance renews soonest first), `order`, `rotate`, and `usage` (least used first, with a token counter that decays with a one-hour half-life). Failure handling is where the code gets wonderfully concrete: constants distinguish out-of-credit (a thirty-minute rest), out-of-quota (fifteen minutes when the vendor says nothing), rate limits, and a vendor's own "try again at" (trusted up to an hour), with repeated failures escalating up to ten minutes. `internal/gateway/affinity.go` then decides how long a conversation stays with the key or account that answered it - because the vendor's prompt cache of that conversation is worth real money - with `auto`, `session`, `turn` and `off` settings.

**Routing groups and rules.** `internal/provider/group.go` lets several models, from one provider or many, be picked as one id, `group/<id>`, with the gateway routing each request over every member's keys and accounts. Groups nest, magpie derives a group automatically when two of your providers serve the same model name, and rules can send matching requests to one member first. When a rule needs to understand the user's message, `internal/gateway/decide.go` asks a classifier model which intent the turn matches - or, for a group with effort set to `auto`, how hard the turn is to think about - so cheap turns get cheap reasoning and hard turns get more.

**Your subscriptions as providers.** The boldest part of the design lives in `internal/gateway/claude_subscription.go` and `internal/claudebridge/mcp.go`: because Anthropic classifies another agent's system prompt as third-party traffic, magpie does not fake Claude Code's API for a Claude subscription - it drives the genuine local `claude` binary for every generation, bridging the caller's tools into that live turn over MCP and resuming the same process for tool results. The ChatGPT path goes the other way: as the README notes, the ChatGPT backend only streams and rejects a few parameters, so magpie translates non-streaming requests and drops what it would refuse. Background goroutines started in `ListenAndServe` - keeping logins alive, moving Codex and Claude Code to their next account when one is spent, keeping quota windows warm - show how thoroughly the concurrency model is used to keep this state fresh without blocking requests.

**Surgical config writes.** `internal/agent/agents.go` describes each of the twenty-seven integrations: where its config lives, which fields matter, and how to detect it. Applying a pick goes through `internal/edit` - `json.go`, `toml.go`, `yaml.go`, `env.go` - which touch only the key you changed, preserve comments and ordering, and write atomically; values magpie replaced are kept in a stash file so switch-back restores whatever was there. Claude Code gets its environment block in `~/.claude/settings.json`, Codex gets a `[model_providers.magpie]` table in `~/.codex/config.toml`, and picking a native model again removes all of it.

The end-to-end flow is short enough to say in one breath: you run `magpie claude deepseek/deepseek-chat`; `internal/agent/claude.go` surgically edits `settings.json` to point `ANTHROPIC_BASE_URL` at the loopback gateway with the model named; Claude Code posts a Messages-API request to `127.0.0.1:3425`; the gateway parses it into the IR, resolves `deepseek/deepseek-chat` through the provider registry, lets the routing engine pick a healthy key, re-encodes the request as an OpenAI Chat Completions call, forwards it, and streams the reply back translated into Anthropic SSE events - and the routing record lands in the Gateway tab before the first token finishes arriving.

## Advantages

- **One endpoint for every wire API.** OpenAI Chat Completions, OpenAI Responses, Anthropic Messages and a Gemini-shaped endpoint on one loopback port, so any tool with a base-URL setting can join without plugins or forks.
- **Translation that respects the hard parts.** Streaming, tool calls, reasoning content, images and token counting all survive the cross-API hop, because the IR in `internal/gateway/ir.go` models them explicitly rather than flattening them.
- **Routing with financial literacy.** The engine distinguishes credit, quota, rate-limit and outage failures, rests each candidate for exactly as long as its failure warrants, and orders subscriptions by which allowance renews soonest.
- **Prompt-cache-aware affinity.** Conversations stay pinned to the account that answered them for as long as the vendor's cache is worth keeping, which the code ties to what the vendor reported reading from its cache.
- **Non-destructive config editing.** Only the changed key is touched, writes are atomic, replaced values are stashed for switch-back, and drift is detected and reported rather than silently clobbered.
- **A small, honest binary.** The README sizes it under 15 MB with the desktop app (it uses the system webview through Wails, nothing bundled) and 7 MB for the terminal-only build - all Go, no runtime to install.

## Benefits

- **Model freedom without retraining your fingers.** Click a value in the menu-bar panel, or type `magpie codex deepseek/deepseek-chat` - the same catalog, names, and reasoning levels in every agent's picker.
- **Resilience you stop thinking about.** Fallback chains and routing groups move a request to the next healthy candidate when a provider is spent, rate-limited or down, and the account that is resting says so in the trace.
- **Your subscriptions, finally portable.** A Claude, ChatGPT, Copilot or Google sign-in becomes a provider every other agent can use - no keys copied, tokens refreshed the way the agent itself does it, nothing stored beyond your model picks.
- **Whole-machine profiles.** `magpie save work` snapshots every agent's settings and `magpie use work` puts them all back in one move, with encrypted backup and WebDAV sync for moving between machines.
- **A privacy posture you can read in the source.** Provider keys live in `~/.config/magpie/providers.json` at mode 0600, are never read from environment variables, and the gateway only listens on loopback unless you explicitly share it; the single daily usage event is opt-out in the settings and absent from source builds.
- **Every surface you might need.** Menu bar panel, window, terminal TUI, browser UI over SSH or WSL, a plain `magpie serve` for servers, and a distroless Docker image for the same gateway.

## Usage

Install the app from the project's site, or from a terminal:

```sh
curl -fsSL https://usemagpie.ai/install.sh | sh
```

From source:

```sh
go install github.com/yetone/magpie@latest
# or build locally:
make build    # ./magpie with the desktop app (needs cgo + the platform webview)
make cli      # terminal-only build, no cgo
```

Run it and wire up a provider:

```sh
magpie              # open the app: a window plus the menu bar icon
magpie tray         # menu bar icon only
magpie tui          # the same thing, in the terminal
magpie serve        # run the gateway alone

magpie presets                        # the vendors magpie knows
magpie provider add deepseek sk-...   # a preset needs only the key
magpie provider add ollama            # local servers need none
magpie providers                      # host, key, exposed models, who uses what
```

Point an agent at a model, or build a routing group:

```sh
magpie claude deepseek/deepseek-chat     # Claude Code on DeepSeek
magpie codex moonshot/kimi-k2.5          # Codex on Kimi
magpie group add "Opus anywhere" models=claude/claude-opus-5-5,copilot/claude-opus-5.5 routing=order stays=session
magpie claude group/opus-anywhere        # use the group as one model
```

Anything with a base-URL setting can use the gateway directly - `OPENAI_BASE_URL=http://127.0.0.1:3425/v1` with `OPENAI_API_KEY=magpie` for OpenAI-speaking tools, or `ANTHROPIC_BASE_URL=http://127.0.0.1:3425` for Anthropic-speaking ones.

## Conclusion

magpie is one of those projects whose small surface hides a serious engine. The menu-bar panel is the visible half; the source reveals the other half: a protocol-translation gateway, a quota-aware routing engine, subscription bridges that drive real local binaries, and config editors careful enough to be trusted with files other programs also write. For anyone running more than one coding agent - or more than one model provider - it collapses a pile of manual plumbing into a single click, and the Go source is a fine study in how to build that kind of infrastructure without ceremony.

Links:

- GitHub repository: [yetone/magpie](https://github.com/yetone/magpie)
- Project site: [usemagpie.ai](https://usemagpie.ai)
- Import-link guide: [usemagpie.ai/docs/import](https://usemagpie.ai/docs/import)
