---
layout: post
title: "Codenotch: macOS usage limits at the screen edge - Inside vinzdg/codenotch"
description: "A source tour of Codenotch, an open-source Swift app for macOS that pins a notch to the screen edge and shows how much of each AI coding assistant's usage limit you have burned. We walk through how it reads usage from Claude Code, Cursor, Codex and a long list of providers, how UsageStore aggregates the readings, and how SwiftUI draws the edge widget."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Codenotch-MacOS-Usage-Limits-At-Screen-Edge-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/codenotch/vinzdg-codenotch-architecture.svg
tags:
  - macOS
  - Swift
  - SwiftUI
  - AI Coding Tools
categories: [AI, Open Source]
keywords: "Codenotch, vinzdg codenotch, macOS usage limits, Claude Code usage monitor, Cursor usage tracker, Codex limit widget, SwiftUI notch app, AI coding assistant limits, token usage dashboard, Swift open source macOS, developer productivity tools, usage provider architecture"
author: "PyShine"
---

Every AI coding assistant meters you differently. Claude Code counts a rolling five-hour session window plus a weekly cap, Cursor keeps its plan state in a local SQLite database, Codex reads its own `auth.json`, and none of them publishes a clean "here is your percentage" API. The result is a tab-hopping ritual: open one dashboard, then another, then squint at a CLI output, all to answer one question — can I keep working, or am I about to hit the wall?

Codenotch, from GitHub user vinzdg, is a macOS app that answers that question permanently. It pins a small black notch to any screen edge and fills one ring per provider account with how much of the limit you have burned, alongside a live count of which agent sessions are running, busy, or waiting on you. When an agent stops working — or stops to ask you something — the notch opens itself for five seconds and sounds an alert. There is a Windows port built on Rust and Tauri in the same repository, but the original and the bulk of the code is Swift, and that is the side this tour walks through.

The source is worth reading because it is an unusually honest answer to a hard integration problem. With no official usage APIs to call, each adapter in `Sources/Providers/` reads whatever the owning tool itself reads from — a Chromium HTTP cache, a signed-in SQLite store, a keychain item, a language server's RPC — and then declares how much it trusts its own numbers. The aggregation layer refuses to invent a percentage when a source goes dark, and the rendering layer has to draw all of this in a one-dimensional strip of screen that maps onto four different edges. Those three problems — unreliable sources, honest aggregation, edge-constrained drawing — are solved in clearly separated layers, which is exactly what makes the repository a good tour.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/codenotch/vinzdg-codenotch-overview-architecture.svg" alt="Architecture overview of the vinzdg/codenotch repository" style="max-width:100%;height:auto;" />
</div>

*Overview of Codenotch's architecture: SwiftUI boots an AppKit delegate that installs provider adapters behind one protocol, a store polls and aggregates them into snapshots, session monitors report live activity, and a per-screen fleet of SwiftUI panels draws the notch.*

Reading the overview from left to right: the `CodenotchMain` entry point in `Sources/App/CodenotchMain.swift` is nearly empty on purpose — it exists only because SwiftUI's `App` protocol demands a scene, and hands everything to `AppDelegate` in `Sources/App/AppDelegate.swift`. The delegate constructs every provider adapter, each implementing the `UsageProvider` protocol from `Sources/Providers/UsageProvider.swift`, and assembles them into `UsageStore` (`Sources/Model/UsageStore.swift`), which polls on a schedule and publishes `ProviderSnapshot` values from `Sources/Model/UsageModel.swift`. Session activity flows in parallel through the `AgentActivityMonitor` family in `Sources/Sessions/`, marking the store busy so the polling schedule can react. Finally the store's published snapshots reach `NotchFleet` (`Sources/Notch/NotchFleet.swift`), which keeps one panel alive per screen and renders them through the SwiftUI view in `Sources/Notch/NotchRootView.swift`.

## Why You Need This

The first problem is simply visibility. Each assistant's usage figure lives in a different place — Claude's panel, Cursor's settings, Codex's CLI — and none of them is visible while you are actually working in a terminal or an editor. A percentage pinned to the screen edge removes the whole ritual. Codenotch draws one ring per account, and the ring's headline is the window the provider itself leads with: for Claude that is the current session window, chosen deliberately so the notch and Claude's own `/usage` output can never disagree, a decision documented directly in `ProviderSnapshot.headline` in `Sources/Model/UsageModel.swift`.

The second problem is trust. A third-party tool that shows "83% used" is making a claim about your account, and if that claim is a guess dressed up as a measurement, it is worse than no number at all. Codenotch makes every adapter declare a `Fidelity` — `.official`, `.derived`, or `.manual` — and derived figures are prefixed with a tilde on screen, so a worked-out estimate never reads as a vendor-published figure. The same honesty applies to failure: a source that cannot answer degrades to a visible status (`stale`, `needsAuth`, `error`) rather than a made-up percentage, with distinct error cases in `UsageProviderError` for "you are signed out", "macOS refused the keychain read", and "the owning app emptied its own credential".

The third problem is attention. Usage limits do not only matter as percentages; they matter as moments. Codenotch watches live sessions and, when an agent finishes or blocks waiting for your input, opens the notch and plays a sound — both of which can be switched off separately. Threshold crossings get their own treatment: when a provider's headline limit crosses 80% or reaches 100%, `ThresholdNotifier` in `Sources/Model/ThresholdNotifier.swift` raises a system notification once per crossing, and stays quiet while the limit remains crossed.

The fourth is scale. The provider table in the README lists Claude Code, Cursor, Codex, Antigravity, GitHub Copilot, Kimi, Kiro, Amp, Grok, OpenCode, GLM, MiniMax, DeepSeek, Apify, Kilo, Command Code, Ollama and LM Studio — and multiple Claude or Codex accounts each get their own ring. Nobody wants to poll eighteen endpoints hard all day, both because it is wasteful and because providers rate-limit. Codenotch's schedule spends its polling budget where the number actually changes: roughly every 30 seconds while a session is working, dropping to every 5 minutes when nothing is running, with `UsageStore` marking itself busy or idle from the live session monitors.

## How It Works

Everything hangs off one protocol: implement `UsageProvider`, and the store knows how to poll you, order you, and draw you.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/codenotch/vinzdg-codenotch-architecture.svg" alt="Detailed architecture of the vinzdg/codenotch repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of Codenotch: Claude's three fallback sources, editor and CLI credential readers, hosted and web-session providers, local runtime monitors, the aggregation store with its archive and threshold alerts, and the per-screen SwiftUI notch rendering chain.*

### Understanding the Architecture

**The provider protocol is the whole contract.** `Sources/Providers/UsageProvider.swift` defines `UsageProvider`: an id, a display name, a glyph, a `fetchSnapshot()` that returns a `ProviderSnapshot`, and an account/sign-in surface so Settings can say which door to knock on. It also defines `UsageFreshness` — `.standard`, `.live`, `.fromSource` — which lets the caller say how fresh a reading must be. A hover on a ring asks for `.live`; "Refresh now" demands `.fromSource`, refusing every cached reading. The comments in this file are remarkable in themselves: they record why each requirement lives on the protocol rather than in an extension, because a method resolved statically through `any UsageProvider` once silently reported every account as absent.

**Claude gets three fallback sources.** `ClaudeOAuthProvider` in `Sources/Providers/ClaudeOAuthProvider.swift` tries, in order: Claude Desktop's own cached usage response, read straight from the Chromium HTTP cache under `~/Library/Application Support/Claude/Cache/Cache_Data` by `ClaudeDesktopUsageCache.swift` — zstd-decoded with a decode-only vendored build of Zstandard in `Sources/Vendor/zstd`, and matched only on entries whose cached URL is *this account's* `/api/organizations/<id>/usage`; then Claude Code's own `/usage`, asked of the installed `claude` binary by `ClaudeUsageCLI.swift`; then the OAuth token in the login keychain via `ClaudeCredentials.swift`, with `ClaudeTokenRefresher.swift` re-minting expired tokens. Multiple `~/.claude-<slug>` profile directories are discovered at launch by `ClaudeProfile.swift`, so a work account and a personal account become two independent rings.

**Editor and CLI adapters borrow existing sign-ins.** `CursorLocalProvider` gets the editor's session through `CursorCredentials.swift`, which opens Cursor's `state.vscdb` SQLite store via the thin `SQLiteStore.swift` wrapper and falls back to the `cursor-agent` keychain login. `CodexLocalProvider` reads each profile's `auth.json` through `CodexCredentials.swift`, never writing or refreshing the credential itself. The same borrowed-session pattern repeats across the folder: Kimi reads `~/.kimi-code/credentials/kimi-code.json`, Grok reads `~/.grok/auth.json` and renews an expired session in memory only, Amp reads `~/.local/share/amp/secrets.json`, and GitHub Copilot authenticates with the GitHub CLI session already on the machine. Providers with no local credential to borrow — DeepSeek, MiniMax, QianwenAI — get an explicit in-app WKWebView sign-in through `WebSessionProvider.swift`.

**Local runtimes are monitored, not metered.** `OllamaLocalProvider` and `LMStudioLocalProvider` produce model cells rather than quota rings. LM Studio's side is the more interesting read: `LMStudioMetrics` in `Sources/Sessions/LMStudioMetrics.swift` watches the SDK socket on LM Studio's own port to see what each model is doing, while `LMStudioServerLog.swift` parses the server log files under `~/.lmstudio/server-logs` — one JSON line per request, carrying token counts and tokens-per-second — to build speed and context numbers. Only counts and timings are read from those files, never prompts or replies.

**UsageStore is the honest aggregator.** `Sources/Model/UsageStore.swift` is a `@MainActor` `ObservableObject` holding every provider. It ticks four times a minute, but ticking is not fetching: each tick decides whether a fetch is owed, based on whether any session is busy, whether a limit window has rolled over, or whether someone is actually looking at a ring. It publishes `snapshots` and `notchSnapshots`, tracks which providers are refreshing, refused, or awaiting renewal, persists the last good reading across launches through `UsageArchive.swift`, and hands crossings to `ThresholdNotifier` for the 80%/100% alerts. The `ProviderSnapshot` and `LimitWindow` types in `Sources/Model/UsageModel.swift` carry the numbers, with the headline window declared by each provider rather than guessed by position — a missing declared window shows a blank ring rather than silently promoting a different limit into its place.

**The notch renders in stack space.** The rendering half lives in `Sources/Notch/`. `NotchFleet.swift` keeps one `NotchWindowController` per attached screen, reconciling against `NSScreen.screens` as displays come and go. Each controller owns an `NSPanel` hosting `NotchRootView` through a `NotchHostingView`, with `NotchViewModel` as the `ObservableObject` the SwiftUI tree observes. The elegant trick is `NotchPlacement.swift`: every view works in a two-coordinate "stack space" — `along` the provider stack, `across` measured inward from the bezel — and `NotchPlacement` is the single place that maps those onto real screen coordinates for whichever of the four edges the notch currently occupies. `NotchLayout.swift` holds every measurement, quoted from the design frame image in the docs folder so layout can be checked against the design directly.

The end-to-end flow, then: `AppDelegate` builds the provider fleet and the monitors at launch; `UsageStore` polls each provider on its adaptive schedule and turns responses into snapshots; `ThresholdNotifier` and the session watchers turn those snapshots and session events into notifications, sounds, and the five-second peek; and `NotchFleet` pushes the final ordered snapshots into each screen's SwiftUI panel, where rings, arcs and tooltips draw flush against the physical screen edge.

## Advantages

- **No separate sign-ins for most providers.** The adapters borrow the credential or session a tool already holds — Cursor's SQLite store, Codex's `auth.json`, the GitHub CLI session, the Kimi CLI credentials — so switching Codenotch on adds no accounts to manage.
- **Honest numbers by construction.** Every snapshot carries a declared `Fidelity`, derived figures are visually prefixed, and every failure mode is a named, visible status rather than a plausible-looking zero.
- **Adaptive polling that respects rate limits.** The schedule fetches every 30 seconds while work is running and every 5 minutes while nothing is, treats a 429 as a back-off floor, and spaces repeated hovers into a single fetch.
- **Multi-account done properly.** Extra `~/.claude-<slug>` and `~/.codex-<slug>` profile directories are discovered at launch, each becoming its own ring with its own limits, sessions and Settings row.
- **One code path for four screen edges.** The stack-space abstraction in `NotchPlacement` means edge changes, Option-drag repositioning, and the hardware-notch shape are all handled in one place instead of being duplicated per orientation.
- **A real test net.** The `Tests/` directory pins each adapter's response shape, which matters when the underlying endpoints are internal and can change without notice.

## Benefits

- **You stop babysitting dashboards.** The ring on the screen edge answers "how much do I have left" with a glance, and the hover card shows every limit window and exactly when each resets.
- **You find out when an agent needs you.** The finished-or-blocked peek plus a per-event sound means a stuck session is a notification, not a discovery twenty minutes later.
- **You get warned before the wall.** Threshold alerts at 80% and 100% fire once per crossing, per provider, and can be muted per row in Settings.
- **Local models are first-class citizens.** Ollama and LM Studio models each get a cell with speed, context use and today's token counts, read from runtimes' own logs without touching your prompts.
- **Your readings survive restarts.** `UsageArchive` keeps the last good reading across launches, so relaunching shows aged-but-real numbers instead of empty rings.
- **It stays out of your way otherwise.** The app has no Dock icon by default, updates itself via Sparkle with EdDSA-signed feeds, and logs diagnostics to the unified logging system.

## Usage

The simplest install is the signed disk image from the repository's releases — the download button links straight to the latest `Codenotch.dmg`, and the app updates itself from then on. If you grab a preview build instead (unsigned, so macOS quarantines it), clear the flag once after dragging the app to Applications:

```sh
xattr -dr com.apple.quarantine /Applications/Codenotch.app
```

To build and run from source, the README's flow uses `xcodegen` and `make`, with no signing identity required for a Debug build:

```sh
brew install xcodegen create-dmg   # once
make run                # generate, build, launch a Debug build
make test               # unit tests
```

To watch what the app is doing while it runs — it has no window of its own — everything worth diagnosing goes to the unified log:

```sh
/usr/bin/log stream --predicate 'subsystem == "com.vinz.codenotch"' --level debug
```

And to see fixed sample data instead of live readings, launch with the demo environment variable: `CODENOTCH_DEMO=1`. Enabling Ollama speed and thinking measurement routes your local generations through Codenotch's relay, for example:

```sh
OLLAMA_HOST=http://127.0.0.1:11435 ollama run gemma4:e4b --think
```

## Conclusion

Codenotch is a good example of a small app with a genuinely hard problem at its core, solved with layers you can name: adapters that read whatever each tool itself reads from and declare their own trustworthiness, a store that aggregates on an adaptive schedule and degrades visibly instead of inventing numbers, and a rendering layer that treats the screen edge as a one-dimensional stack. The code is current, heavily commented with the reasoning behind each decision, and backed by unit tests that pin every adapter's response shape. If you run AI coding assistants on a Mac, it is useful today; if you write Swift, the provider-adapter pattern and the stack-space layout trick are worth the read on their own.

Links:

- GitHub repository: [vinzdg/codenotch](https://github.com/vinzdg/codenotch)
- README and provider table: [github.com/vinzdg/codenotch#what-it-reads](https://github.com/vinzdg/codenotch#what-it-reads)
- Windows port: [windows/ directory](https://github.com/vinzdg/codenotch/tree/main/windows)
