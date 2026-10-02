---
layout: post
title: "Coucou: A Tiny Friend in Your Notch That Watches Your AI Coding Agents - Inside Louis-CFM/coucou"
description: "Coucou is an open-source notch companion that lives in your Mac's notch, or the top of your screen on Windows and Linux, and surfaces AI coding agent sessions in real time. We tour the Swift and Rust source behind its SwiftUI-drawn character, Claude Code hook bridge, permission approvals, and integration pollers."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /Coucou-A-Notch-Companion-For-Your-AI-Coding-Agents/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/coucou/louis-cfm-coucou-architecture.svg
tags:
  - Swift
  - Rust
  - macOS
  - AI Agents
categories: [AI, Open Source]
keywords: "coucou, notch companion, Claude Code hooks, SwiftUI animation, Tauri 2 app, agent session monitor, permission approvals, macOS menu bar, Mochi character, open source desktop app, developer tools"
author: "PyShine"
---

AI coding agents work in terminals while you work somewhere else, which means the moment they need you, a permission prompt, a clarifying question, a finished build, you are usually looking the other way. [Louis-CFM/coucou](https://github.com/Louis-CFM/coucou) solves that with charm and engineering at the same time. Coucou is a tiny friend that lives in your Mac's notch, or at the top of your screen on Windows and Linux, and keeps an eye on your AI coding agent sessions: it shows every step the agent takes, raises permission requests with Allow and Deny buttons, and celebrates with a happy little jump when the work is done.

The character is Mochi, a soft squircle with big eyes that follows your cursor, blinks, yawns, gets annoyed when you poke it, and turns dizzy if you insist. The project positions itself as the open version of the notch-companion concepts design studios have teased: every line of code, every animation, and every one of the 28 handcrafted sounds is free to read, fork, and remix. Version 0.1.1 shipped with multi-agent pills, Gemini CLI and OpenAI chat support, and a first Linux beta.

The source is worth a tour because behind the cuteness sits a genuinely well-considered system: a zero-dependency native SwiftUI app on macOS, a Tauri 2 port for Windows and Linux that reuses the same shapes and timings, a hook bridge that can never block your agent, and a design specification committed right next to the code. Let us look inside.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/coucou/louis-cfm-coucou-overview-architecture.svg" alt="Architecture overview of the Louis-CFM/coucou repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Coucou codebase: the native macOS app on the left with its island state machine and character engine, the Tauri-based Windows and Linux port on the right, and the coucou-hook binary that bridges agent events into both.*

Reading the overview from left to right: the macOS app owns a borderless panel hugging the notch, driven by the island state machine that decides when Mochi hides, peeks, goes compact, or expands into a full view. The hook server receives agent events and injects them into that state machine, the chat service handles conversations with Claude, and the pill catalog declares every tool and integration that can appear in the island. On Windows and Linux the same experience is rebuilt in a Tauri 2 app: a Rust core handles windows, hooks, and secrets, while a TypeScript webview redraws Mochi in Canvas 2D with the same shapes and timings. The coucou-hook binary is the shared entry point for agent payloads.

## Why You Need This

The first problem is attention fragmentation. Agents pause on permission requests and questions, and every minute you spend not noticing them is idle compute. Coucou makes the island open itself on the alert view, even when you are away, and stays open until you answer, with a queue handling multiple alerts one at a time. Approvals happen with one click right in the notch, whether your session lives in a terminal, VS Code, or Cursor, and on macOS you can even jump straight to the exact terminal window of a session.

The second problem is trust in the bridge. Giving a third-party app hooks into your coding agent sounds risky, so the design is defensive on purpose. The hook forwards events over a local Unix socket on macOS, a named pipe on Windows, or a user-scoped socket on Linux, and if Coucou is not running, the hook exits immediately, so Claude Code is never blocked. During setup, the app backs up your settings file, merges its hooks, and shows you the diff before writing anything. Version 0.1.1 tightened this further with user-account-scoped sockets, size and time limits, and trimmed logs.

The third problem is context switching for small interactions. Need to ask Claude about a file? Drop the file onto the notch and Mochi turns into a box and swallows it. Want an open window as context? Drag Mochi onto it. Chat supports Anthropic models plus Google AI and OpenAI with your own keys, with the model list coming from your API account, all without leaving the island.

The fourth problem is visibility across services. Beyond coding agents, Coucou ships lightweight pollers for Stripe payments, n8n workflow runs, GitHub events, Vercel deployments, Resend emails, Notion, and Cal.com, each rendered as its own colored mini-Mochi. The pollers pause when nothing is watching, which keeps the app quiet on your battery.

## How It Works

The repository holds two implementations of one experience, a native macOS app and a Tauri port, plus the hook binary and a committed design specification, and the detailed diagram maps them.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/coucou/louis-cfm-coucou-architecture.svg" alt="Detailed architecture of the Louis-CFM/coucou repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of Coucou: the macOS app layers on the left, service pollers beside them, the Tauri Rust core and webview TypeScript on the right, with the hook binary and the specification and tests anchoring the bottom.*

### Understanding the Architecture

**The island and its state machine.** [NotchBuddy/Sources/App/IslandWindowController.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/IslandWindowController.swift) manages a borderless panel above the menu bar, with click-through handled by toggling mouse-event transparency at frame rate so the transparent area never blocks clicks. [IslandStateMachine.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/IslandStateMachine.swift) implements the behavior rules from the specification: hidden when nothing runs, peek on hover, compact with eye-tracking while agents work, expanded on intent, with an inactivity countdown and Escape to close. [IslandScreenGeometry.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/IslandScreenGeometry.swift) computes the real notch metrics and falls back to a compact top bar on screens without a notch, covered by unit tests in [tests/IslandScreenGeometryTests.swift](https://github.com/Louis-CFM/coucou/blob/main/tests/IslandScreenGeometryTests.swift).

**A hand-drawn character.** Mochi is drawn in SwiftUI Canvas and TimelineView at 60 fps in [BotEngine.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/BotEngine.swift) and [BotCanvasView.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/BotCanvasView.swift): a squircle body, eyes projected on a sphere that track your cursor, spring animations, emotes, and a wake-and-wave greeting on launch. There are no images, no Rive, no Lottie, and the macOS app has zero third-party dependencies. Sounds play through preloaded AVAudioPlayers via [SoundEngine.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/SoundEngine.swift).

**The agent bridge.** [HookServer.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/HookServer.swift) receives hook events that agents emit through a small helper script and forwards them over a Unix socket, feeding the state machine. For approval requests the helper waits for your click and then answers the hook, which is how Allow and Deny flow back into the agent session. [docs/AGENTS.md](https://github.com/Louis-CFM/coucou/blob/main/docs/AGENTS.md) documents how any agent can get its own pill by tagging its hook payload, and Claude Code, Gemini CLI, and Antigravity hooks install from [SettingsView.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/SettingsView.swift) with a backup-and-diff flow.

**The Tauri port.** The Windows and Linux app pairs a Rust core, [windows/src-tauri/src/lib.rs](https://github.com/Louis-CFM/coucou/blob/main/windows/src-tauri/src/lib.rs), with a TypeScript webview. [hooks.rs](https://github.com/Louis-CFM/coucou/blob/main/windows/src-tauri/src/hooks.rs) and [pipe.rs](https://github.com/Louis-CFM/coucou/blob/main/windows/src-tauri/src/pipe.rs) implement the intake and named-pipe transport for the coucou-hook binary in [windows/hook/src/main.rs](https://github.com/Louis-CFM/coucou/blob/main/windows/hook/src/main.rs), [secrets.rs](https://github.com/Louis-CFM/coucou/blob/main/windows/src-tauri/src/secrets.rs) stores keys in Windows Credential Manager or the Linux Secret Service, and [platform/mod.rs](https://github.com/Louis-CFM/coucou/blob/main/windows/src-tauri/src/platform/mod.rs) abstracts the per-OS differences, including the gtk-layer-shell overlay that anchors the island to the top edge on Wayland. The webview side mirrors the Mac experience: [windows/src/island/fsm.ts](https://github.com/Louis-CFM/coucou/blob/main/windows/src/island/fsm.ts) runs the same state machine, [mochi/engine.ts](https://github.com/Louis-CFM/coucou/blob/main/windows/src/mochi/engine.ts) redraws the character in Canvas 2D, and [views/chat.ts](https://github.com/Louis-CFM/coucou/blob/main/windows/src/views/chat.ts) hosts the chat.

**Declarations and privacy.** [PillCatalog.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/PillCatalog.swift) is the single source of truth for every pill, coding tools, agents, AI providers, services, each with its identity, color, and category, while the pollers like [GithubPoller.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/GithubPoller.swift), [StripePoller.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/StripePoller.swift), and [N8nPoller.swift](https://github.com/Louis-CFM/coucou/blob/main/NotchBuddy/Sources/App/N8nPoller.swift) push service events into shared application state. The whole design is specified, measured, and behavioral-rule numbered in [docs/SPEC.md](https://github.com/Louis-CFM/coucou/blob/main/docs/SPEC.md). The end-to-end flow: an agent fires a hook, the bridge validates and forwards it, the state machine decides whether to hide, peek, or open, Mochi animates the moment, and your click flows back through the bridge to the waiting agent.

## Advantages

- **Never blocks your agent.** If the app is closed, the hook exits immediately; approvals wait for your click without stalling the CLI.
- **Zero-dependency native app on macOS.** Swift 6, SwiftUI, and AppKit only, with no third-party packages in the critical path.
- **One experience, three platforms.** The Tauri port reuses the same shapes, timings, and sounds, with a Rust core and a Canvas 2D webview.
- **A committed design specification.** Window measures, animation curves, and numbered behavior rules live in the repository, making the character reproducible.
- **Privacy-first by construction.** No telemetry and no account; keys stay in Keychain, Windows Credential Manager, or the Linux Secret Service.
- **Extensible pill system.** One catalog file declares every tool, agent, and service pill, so new integrations are additive.

## Benefits

- **Faster agent loops.** Permission requests and questions surface instantly in your peripheral vision, cutting the idle time between agent turns.
- **Fewer context switches.** Approvals, file drops, window context, and chat happen in the island instead of a separate app window.
- **Multi-agent visibility.** Each running session gets its own mini character, and any agent can claim a pill with a payload tag.
- **Service awareness.** Payments, deployments, emails, workflows, and repository events appear as colored companions alongside your agents.
- **Open and remixable.** MIT-licensed code, committed spec, and per-platform build instructions make forks and ports practical.
- **Polished human interface.** Idle breathing, blinking, eye tracking, emotes, and a full sound set turn a utility into something people keep around.

## Usage

Install on macOS from the releases, or build from source with XcodeGen:

```bash
brew install xcodegen
git clone https://github.com/Louis-CFM/coucou.git
cd coucou/NotchBuddy
xcodegen
open NotchBuddy.xcodeproj   # then build and run
```

On Windows or Linux, build the Tauri app with Rust and Node 20+:

```powershell
cd coucou/windows
npm install
npm run pack                # installer lands in windows/release/
```

For Linux distributions, install the WebKitGTK, layer-shell, and appindicator development packages listed in the README first, then the same `npm run pack` produces AppImage, deb, and rpm artifacts. After launching Coucou from the menu bar or system tray, open Settings and:

```text
1. Install Claude Code hooks     -> backs up ~/.claude/settings.json,
                                    merges the hooks, shows the diff
2. Add your Anthropic API key    -> chat and file questions
3. Pick your active pills        -> VS Code, Cursor, Gemini CLI,
                                    Anthropic, Google AI, OpenAI
4. Optionally connect services   -> Stripe, n8n, GitHub, Vercel,
                                    Resend, Notion, Cal.com
```

Then interact with the island: hover the notch to make Mochi peek and wave, click to open, drag a file onto it to attach it to a question, click the model name above the chat box to switch provider, and let it sit hidden the rest of the time. Any agent can join by tagging its hook payload with the agent tag documented in the AGENTS guide.

## Conclusion

Coucou is proof that agent tooling does not have to look like infrastructure. It takes a real workflow problem, agents that need your attention at unpredictable moments, and wraps the solution in a character with genuine personality, while the underlying engineering stays conservative: a hook that never blocks, keys in the OS keychain, a state machine specified to the animation curve, and a port strategy that reuses the design instead of reinventing it. For anyone running Claude Code or similar agents daily, it is one of the most delightful ways to keep an eye on the machine's new coworker.

Links:

- GitHub repository: [Louis-CFM/coucou](https://github.com/Louis-CFM/coucou)
- Design specification: [docs/SPEC.md](https://github.com/Louis-CFM/coucou/blob/main/docs/SPEC.md)
- Agent integration guide: [docs/AGENTS.md](https://github.com/Louis-CFM/coucou/blob/main/docs/AGENTS.md)
