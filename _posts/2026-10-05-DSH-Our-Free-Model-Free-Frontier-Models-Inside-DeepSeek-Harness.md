---
layout: post
title: "dsh-our-free-model: Free Frontier Models Inside DeepSeek Harness - Inside zouyuxuan122/dsh-our-free-model"
description: "A MIT-licensed, zero-dependency plugin that wires free frontier models from a public key-free lane into DeepSeek Harness - with honest availability probing, real thinking budgets, a local OpenAI-compatible port, and signed in-app upgrades."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /dsh-our-free-model-free-frontier-models-inside-deepseek-harness/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/dsh-our-free-model/zouyuxuan122-dsh-our-free-model-architecture.svg
tags: [AI, DeepSeek, Local Tools, Open Source]
categories: [AI, Open Source]
keywords: [dsh-our-free-model, DeepSeek Harness, free models, OpenAI compatible, Koishi plugin]
author: "PyShine"
---

Free access to frontier models usually comes wrapped in sign-up forms, API keys, quota dashboards, and a reseller sitting between you and the inference endpoint. The [dsh-our-free-model](https://github.com/zouyuxuan122/dsh-our-free-model) plugin takes the opposite path: install it into DeepSeek Harness (dsh), restart once, and models including Muse Spark 1.3 and MiMo V2.6 simply appear in your picker - no login, no key, no other step. The upstream is named plainly in the repository: OpenCode's Zen gateway, with your conversation flowing from your machine straight there and no third party in the middle.

What makes this repository remarkable is not the free lane itself but the engineering discipline wrapped around it. This is a plugin that measured its own gateway's lying `Content-Type` headers and rewired itself to read responses by body shape. It turned a `reasoning_effort` string that the gateway silently ignores into hard output-token budgets that demonstrably bind. It repairs the tool-pairing corruption that would otherwise poison an entire session from one interrupted call. Every claim in the README carries a verification note, and twenty-plus offline test suites run without spending a single token of quota.

The cost of entry is equally plain: zero dependencies, no build step, pure JavaScript source, MIT licensed. One adapter covers both the 0.1.5 and 0.1.7 kernel lines because it never pins itself to one. And when the free lane hiccups, the plugin degrades honestly - a model the gateway refuses is withdrawn from the picker with its refusal recorded, while a model merely rate-limited or 5xx'd stays right where it is.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/dsh-our-free-model/zouyuxuan122-dsh-our-free-model-overview-architecture.svg" alt="dsh-our-free-model overview architecture" style="min-width:720px;width:100%;">
</div>

*The overview: one host half registers a structural adapter, probes availability, and serves a local forward port, while the browser half renders the settings dashboard and the trust chain signs every upgrade.*

Reading the overview from left to right: `index.js` is the host half - it registers the adapter, refreshes the model roster from `src/catalog.js`, schedules availability probes through `src/probe.js`, owns the settings and stats stores, mounts the webServer API routes, and runs the forward lifecycle. The adapter in `src/adapter.js` is deliberately structural: it duck-types the harness's LlmAdapter interface and reaches the kernel only through `adapter/kernel.js`, the single module allowed to import the official packages. On the wire, `src/messages.js` converts harness messages into the shapes each of the three upstream protocols expects, and `src/stream.js` decodes Chat Completions, Messages, and Responses frames back into harness StreamChunks. Everything talks to the gateway identity in `src/upstream.js`. Alongside, `src/forward.js` exposes the same models to any local tool through an OpenAI-compatible loopback listener. The browser half, `client.js`, is a hand-written bundle with no build step that renders the six-section settings page and consumes the SSE push channel. Finally, `src/updater.js` and the optional `worker/worker.js` gateway form the trust chain - signed manifests and a signed relay that never hands your plugin the real key.

## Why You Need This

If you run dsh, the case is simple: this is the difference between configuring nothing and assembling accounts, keys, and proxy scripts for the same capability. But the deeper reasons are architectural, and they apply to anyone building on someone else's gateway.

The first is the honesty of the picker. Most integrations trust the model list and discover breakage mid-turn. Here, a model the gateway names but refuses to route - a 404 for that id - leaves the dropdown and stays visible on the settings page with its refusal and probe time recorded, returning automatically when a probe gets through. Everything that is not a statement about the model keeps it reachable: a 5xx, a 429 quota window, a timeout, or a dropped connection never demotes a working model. Region-gated ones move to their own `region-limited` group, and toggling a VPN plus a re-probe reclassifies them.

The second is that the thinking-effort control is real. The author sampled repeated requests at different nominal `reasoning_effort` values and found the gateway statistically indifferent - so shipping that control would have been shipping a placebo. Instead, Light, Balanced, and Deep map to output ceilings of 2,048, 8,192, and the model's full capacity, enforced on the request, and models that cannot switch thinking off (MiMo V2.6 among them) get the ladder doubled to 4,096 and 16,384 because their thinking and their answer share one budget. Every model card prints the number it will actually send.

The third is a lesson in defensive parsing that reads like a war story. Under load, this gateway answers 200 with a JSON `Content-Type` over a perfectly ordinary SSE frame stream. Trusting the header meant the stream was swallowed into a failed `JSON.parse`, the turn was lost, and - because the error carried status 200 - the availability probe demoted a perfectly working model. The fix in `src/http.js` sniffs the first bytes, classifies the body by shape, and replays them into the stream, so nothing is buffered and no token arrives late.

## How It Works

The adapter layer is worth pausing on. The kernel never checks `instanceof`, so `src/adapter.js` implements `providerInfo`, `listModels`, `resolveModel`, `prepareCall`, `stream`, and `providerRetryPolicy` as a structural match - which is what lets one codebase run unchanged on both kernel versions. Three kernel behaviors are documented as hard-won warnings for any provider-plugin author: the retry policy must be returned flat and already-resolved (nested values produce `NaN` backoff delays that the durable session log rejects, aborting recoverable turns); one interrupted tool call poisons the whole session with a replayed 400, so `src/messages.js` repairs pairing before sending on all three wires; and services must be awaited through `ctx.inject(deps, callback)` rather than read at load time, because plugins load before the browser half publishes the web server.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/dsh-our-free-model/zouyuxuan122-dsh-our-free-model-architecture.svg" alt="dsh-our-free-model detailed architecture" style="min-width:900px;width:100%;">
</div>

*The detailed view: the host half and wire decoders, the local data and push channels, the signed trust chain with the optional EAC gateway, and the offline test suites that keep it all verifiable.*

### Understanding the Architecture

**The wire layer normalizes three protocols.** Upstream calls go to the Zen gateway's chat completions, messages, or responses endpoints depending on the model, always with `Authorization: Bearer public` - a public, key-free allowance the plugin holds no secret of. `src/stream.js` decodes each wire into harness StreamChunks with disjoint token accounting, recognizing thinking under `reasoning`, `reasoning_content`, `reasoning_text`, and the `reasoning_details` array, and counting a thought repeated under two of them exactly once. Outbound, `src/messages.js` maps harness messages to wire shapes and merges tool results keyed by call id, so parallel calls each reach their answer.

**The forward port makes the lane portable.** Enable it on the settings page and `src/forward.js` serves `GET /v1/models`, plus streaming and non-streaming `POST /v1/chat/completions` and `POST /v1/responses`, authenticated by a key minted with the crypto module, compared with `timingSafeEqual`, and stored in a `0600` file. The listener binds loopback by resolution, not by spelling - `localhost` is resolved with `dns.lookup` and every answer must be loopback, so a hosts-file trick cannot hand your quota to the subnet. Widening it is refused with a 400 that says why; reaching other machines takes an explicit second door with its own key, three whitelisted paths, and a 508 refusal for relay-to-relay loops. While a lane is thinking, the port emits periodic SSE comment frames so a client's idle timeout cannot mistake "still thinking" for "socket dead".

**Recovery turns a cut stream into a completed answer.** When a first stream delivers nonempty reasoning but reaches EOF with no answer text - the shape the host would report as an empty response - `src/recovery.js` sends exactly one continuation request carrying the original input and the received reasoning as a checkpoint, asking for the conclusion first. The guardrails are explicit: tools disabled, at most two physical requests per logical turn, a 480-second total deadline with at most 180 seconds for recovery, an 8,192-token recovery output cap that still respects the user's ceiling, and a checkpoint capped at 131,072 characters. The dashboard counts physical requests and logical turns separately, so a cut plus a successful continuation reads as two requests, one recovered turn.

**The trust chain assumes the network is hostile.** Plugin routes sit behind the request fence in `src/trust.js`, which defers to the composition's connection admission per request and otherwise enforces loopback Host, same-origin, and a fail-closed refusal of missing Host headers. The in-app updater in `src/updater.js` verifies an Ed25519 signature against a public key pinned in the plugin, then validates the manifest, checks per-file SHA-256 and size on download, verifies staging and the installed copy by reading them back, and restores the backup on any failure - so even a poisoned mirror, jsDelivr included, produces a failed upgrade rather than executed code. The optional EAC gateway in `worker/worker.js` inverts the key problem entirely: the plugin sends HMAC-SHA256-signed requests with a timestamp, the worker verifies the signature against a clock-skew window, whitelists the two allowed routes, and injects the real relay key that lives only in its environment.

**The dashboards measure honestly.** The usage board keeps everything local under `DSH_HOME/our-free-model/` - a 17-week token heatmap, cumulative curves, and per-call time-to-first-token. After the author caught an early build publishing 2,941 tok/s for a lane really doing about 40, `windowTokens()` now removes reasoning tokens that never streamed from the numerator, `decodeWindow()` rejects windows too short to time, and the panel sums tokens over summed seconds instead of averaging per-call ratios. A model whose answer lands in two frames reports no speed at all rather than publishing its thinking time as decoding speed.

Tracing one turn end to end: you pick a model under Our Free Model, the adapter resolves it through the catalog, `src/effort.js` translates your effort rung into a real output ceiling, `src/messages.js` shapes the request and repairs any dangling tool pairs, `src/http.js` sniffs the response body by shape, `src/stream.js` decodes the frames, and the turn either completes, fails with a durable-log-safe error code, or hands off to recovery - while the store records the request, the tokens, and the latency for the heatmap.

## Advantages

- **Zero dependencies, zero build step.** Plain JavaScript source loads directly as a local plugin; anyone can audit the gateway logic, and the README treats that as a property, not a problem.
- **The picker only advertises what answers.** Probes from your own egress decide admission, refusals are recorded with timestamps, and quota errors never hide a working model.
- **Effort rungs bind.** The output ceiling is enforced on the wire and printed on every model card, replacing a placebo string with a measurable control.
- **Self-upgrading with a real trust root.** Ed25519-signed manifests, byte-level verification, atomic replace, hot reload, and automatic rollback - no reinstall, no restart.
- **An OpenAI-compatible local port included.** Any tool that speaks the Chat Completions format can use the same models, on a listener that refuses to bind anywhere but loopback.
- **Failure modes are documented like features.** The kernel gotchas, the lying header, the fake tok/s - each is explained with the measurement that caught it.

## Benefits

- **Free frontier models with no account theater.** Install, restart, pick a model - the roster tracks upstream on every refresh instead of freezing into the plugin.
- **Your usage data stays home.** Settings, stats, and keys live in one local JSON directory; the only outbound destinations are the gateway, the announcement feed, and an egress-IP lookup, each enumerated in the README.
- **Cut reasoning streams recover automatically.** The bounded continuation request converts the most common silent failure of thinking models into a finished answer.
- **The dashboard tells the truth.** Speed panels decline to publish numbers their windows cannot support, and quota exhaustion surfaces as a state, not a disappearance.
- **Announcements arrive without a scraper.** The owner edits one JSON file and pushes; every installation picks it up within a poll cycle, rendered through a strict allowlist that an XSS corpus test keeps honest.
- **The test suite never spends your quota.** Twenty-plus offline suites, a live self-test you run by hand, and evidence probes under `scripts/probes/` separate what is proven from what is assumed.

## Usage

Install into a dsh profile and restart once:

```bash
dsh plugin --profile web add /absolute/path/to/dsh-our-free-model
```

Requires Node `^22.19.0 || >=24.0.0`. Desktop hosts want a real directory rather than a symlink - the README documents the profile gate that rejects junctions and walks through the manual install into `node_modules/dsh-our-free-model/` step by step.

Pick a model from the Our Free Model group in the composer selector; change thinking depth with the Effort menu's Light, Balanced, and Deep rungs. The settings page exposes six sections: the model roster with per-model ceilings and a single-call benchmark, the announcement center, the usage board with its 17-week heatmap, the local forward panel, plugin settings, and the upgrade panel.

Serve other local tools through the forward port:

```bash
# after enabling Local forward and generating a key
curl http://127.0.0.1:18899/v1/models \
  -H "Authorization: Bearer <your-forward-key>"

curl http://127.0.0.1:18899/v1/chat/completions \
  -H "Authorization: Bearer <your-forward-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Re-check geography after switching networks with the re-probe button; receive announcements automatically; upgrade in one click with the updater.

Run the verification suite yourself:

```bash
npm test                            # every offline suite, no network, no quota
node scripts/client-lint.mjs        # browser bundle checks
node scripts/host-selftest.mjs      # live upstream end-to-end (real network)
node scripts/build-manifest.mjs     # regenerate and sign the release manifest
```

Deploy your own signed relay with the Cloudflare Worker or a self-hosted node - the `worker/` directory documents the HMAC signing contract, the route whitelist, the SSE pre-flush that survives proxy read timeouts, and an admin dashboard with per-IP rate-limit statistics.

## Conclusion

dsh-our-free-model is a masterclass in treating someone else's free tier as a production system. It measures before it advertises, repairs before it retries, signs before it executes, and declines to publish numbers it cannot defend. The result feels less like a plugin and more like a small operating layer around a free model lane - with a picker that tells the truth, budgets that bind, a forward port that respects the loopback, and an upgrade path that assumes the mirror is compromised. If you run DeepSeek Harness, it is the fastest route to useful free models; if you build provider integrations for a living, its README alone is worth the read.

- Repository: [github.com/zouyuxuan122/dsh-our-free-model](https://github.com/zouyuxuan122/dsh-our-free-model)
- English README: [README_EN.md](https://github.com/zouyuxuan122/dsh-our-free-model/blob/main/README_EN.md)
- Recovery record: [docs/issue-12-recovery.md](https://github.com/zouyuxuan122/dsh-our-free-model/blob/main/docs/issue-12-recovery.md)
- License: [MIT](https://github.com/zouyuxuan122/dsh-our-free-model/blob/main/LICENSE)
