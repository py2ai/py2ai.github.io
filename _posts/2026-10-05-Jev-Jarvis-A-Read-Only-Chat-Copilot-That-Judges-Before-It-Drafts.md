---
layout: post
title: "Jev Jarvis: A Read-Only Chat Copilot That Judges Before It Drafts - Inside jev-chat/jev-chat-jarvis"
description: "jev-chat/jev-chat-jarvis is an MIT-licensed Android copilot that reads the visible chat through the accessibility service, asks a judge model seven typed questions about intent, danger and timing, drafts three candidate replies, and fills the one you pick into the input box - never sending by itself. A source tour of the capture adapters, the OCR fallback, the local knowledge base, and the fill-not-send guard."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /Jev-Jarvis-A-Read-Only-Chat-Copilot-That-Judges-Before-It-Drafts/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/jarvis/jev-chat-jev-chat-jarvis-architecture.svg
tags:
  - Android
  - AI Assistants
  - On-Device AI
  - Privacy
categories: [AI, Open Source]
keywords: "Jev Jarvis, chat copilot, Android accessibility service, typed decisions, intent classification, danger level scoring, ML Kit offline OCR, reply suggestions, QQ assistant, Feishu assistant, on-device privacy, jev-chat"
author: "PyShine"
---

Most assistant tools that promise to help with chatting make the same move: they grab the conversation, throw it at a generative model, and hand you a paragraph to send. The paragraph usually reads fine and is often wrong for the moment - because the hard part of a delicate conversation is not composing sentences, it is understanding what the other person actually wants right now, how close the thread is to a fight, and whether this is the minute to say anything at all. Jev 聊天助手, an MIT-licensed Android app published under the jev-chat organization, is built around that inversion: it judges first and drafts second, and it never touches the send button.

The project - repository jev-chat/jev-chat-jarvis, version 1.4, Kotlin, package com.jev.probe - works on QQ, X direct messages, and Feishu, with a manual screenshot-and-recognize path for other apps. Its data boundary is unusually strict for this category: no hooking, no repackaging, no calls to the chat apps' own interfaces or accounts, no database reads. It reads what the accessibility service exposes on screen, OCRs screenshots locally when the accessibility tree has no text, keeps your keys and knowledge base in the app's private storage, and - the part the README repeats like a mantra - only ever fills the input box. Sending is your keystroke, and transfers or red packets are out of scope entirely.

The source is worth a tour because it solves, in plain Kotlin, three problems that every screen-aware assistant stumbles over: how to extract a conversation from wildly different app UIs without touching them, how to keep the judgment structured enough that a floating panel can show calibrated confidence instead of prose, and how to fill text into someone else's input box safely, with retries and verification, while guaranteeing that nothing is ever sent. The whole thing is about three dozen Kotlin files plus a small Python calibration toolkit, and nearly every file has one legible job.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jarvis/jev-chat-jev-chat-jarvis-overview-architecture.svg" alt="Architecture overview of the jev-chat/jev-chat-jarvis repository" style="max-width:100%;height:auto;" />
</div>

*The loop from on-device capture through per-app adapters and an offline OCR fallback, into the judgment and drafting clients, out to the floating panel, and back to a guarded input writer that fills but never sends.*

Reading the overview from left to right: the accessibility service in app/src/main/java/com/jev/probe/capture/ChatCaptureService.kt watches the foreground window and dispatches by package name to the adapters in ChatAppAdapter.kt, which turn the window into a title plus a message list; when a tree has no readable text, the path detours through ocr/ScreenCapture.kt and ocr/MlKitOcr.kt, which recognize text on-device. The snapshot and any knowledge hits from core/kb/ContextBuilder.kt flow into the judgment client in jev/JudgeClient.kt, which sends the seven typed questions of JevQuestions.kt in one call; the drafting client in jev/ReplyClient.kt produces three candidates that the same judgment route ranks. Results land on the floating panel in overlay/OverlayController.kt, and the tap you make there ends at capture/GuardedInputWriter.kt, which fills the input box under its own verification. SettingsActivity.kt configures the three independent model routes, and the local knowledge store in core/kb/KbStore.kt feeds the context assembly.

## Why You Need This

The first problem is that chat interfaces are minefields of implied meaning. A short "fine then" can be acceptance, sarcasm, or an ultimatum depending on the previous ten messages, and a generative model asked to "write a reply" will happily paper over that ambiguity with pleasant words. Jev Jarvis instead asks a judge model a fixed battery of typed questions about the thread - is there subtext, what is the person's true intent, how close is this to a rupture, do they need an apology or proof you remember something - and each answer comes back with a probability, not as free text. The overlay shows the danger level, the intent, and the ranked options, so you see the model's read of the situation before you see a single suggested sentence.

The second problem is access. Screen-level integration with chat apps usually means one of three ugly things: injection frameworks that can get accounts banned, unofficial API wrappers that break every release, or screen scraping that uploads your conversations to someone's server. This project takes a fourth path: the accessibility service reads only what is displayed, exactly as a screen reader would; the OCR fallback runs entirely on-device with ML Kit's offline Chinese model; and the analysis request goes to whatever model endpoint you configure yourself, with no relay server operated by the author. The app never logs into anything and never touches your chat database.

The third problem is the last mile. A suggestion is worthless if using it requires manual copy-paste gymnastics, but an assistant that can send messages on your behalf is a liability in a tool designed for emotionally loaded conversations. The repository's answer is a carefully guarded middle ground: the app writes your chosen reply into the input box using the accessibility set-text action, verifies the text landed, retries with focus and clipboard paste if it did not, and stops there. The send key stays under your finger. That single design decision - fill, never send - is threaded through the code from the overlay controller down to the input writer, and it is what makes the tool safe to keep installed.

The fourth problem is memory. Replies that contradict what you know about a person are worse than no replies, so the app ships a local knowledge base: pinned notes that always ride along, tagged notes that ride along when the conversation title or the last six messages mention them, and contact profiles with aliases, relationships, and remarks matched against the conversation title. Optional local history - off by default - adds the last thirty messages, deduplicated and minus what is already visible on screen. The panel tells you exactly what was attached: a line showing how many knowledge entries and how many history messages went into the analysis.

## How It Works

The app is a loop around one snapshot object, and the detailed map below places every module in that loop.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jarvis/jev-chat-jev-chat-jarvis-architecture.svg" alt="Detailed source architecture of the jev-chat/jev-chat-jarvis repository" style="max-width:100%;height:auto;" />
</div>

*The full source map: the capture stack with its adapters and OCR fallback, the judgment and drafting clients behind one HTTP transport, the knowledge modules, the Android surfaces, the unit tests, and the Python calibration toolkit.*

### Understanding the Architecture

**The capture service.** app/src/main/java/com/jev/probe/capture/ChatCaptureService.kt is an AccessibilityService that reacts to window changes, resolves the current target into a ConversationSession token (ConversationSession.kt), and submits analysis work to a single-thread executor so that stale snapshots cannot race a newer one. It owns the OCR fallback path: when an adapter reports a chat window whose tree carries no text, it captures the screen (ocr/ScreenCapture.kt), crops each bubble rectangle, recognizes them with the offline engine (ocr/MlKitOcr.kt behind the OcrEngine.kt interface), groups the lines into messages, and merges the result into the snapshot. Capture is rate-limited with failure backoff, the recognition avoids the app's own overlay window, and a manual "recognize once" action covers apps that have no adapter.

**The adapter contract.** capture/ChatAppAdapter.kt defines the whole integration surface in one interface: a package name and an extract function that turns the accessibility tree into a ChatSnapshot - a title plus a list of Msg objects, each marked me or other - or null when the foreground window is not a conversation. Four adapters live in that file: QQ (com.tencent.mobileqq) reads the message body and title by view id and assigns sides by which avatar a bubble sits next to; X (com.twitter.android) parses the content-desc of Compose nodes, where the sender is embedded as text; Feishu (com.ss.android.lark) gets bubble rectangles and read state from the tree and leaves the words to OCR; and a WeChat adapter exists but is deliberately inert, since the project has dropped WeChat collection entirely. The README documents the contract precisely: returning null means not a chat window, returning an empty list means a chat window with no readable text, and only the latter triggers OCR.

**The typed judgment set.** jev/JevQuestions.kt holds seven questions sent in one request, ported verbatim from the Python calibration set in tools/jev/questions.py so that the wording is exactly what passed calibration: a yes/no on whether the latest message is purely literal, a six-way choice on true intent (testing whether you care, venting anger, requesting action, seeking an explanation, casual chat, or peacefully closing the topic), a ten-bin score for danger level whose bins are written as concrete scenes rather than adjectives, a yes/no on whether the next message should carry substance now, a seven-way choice on the best action type (check history first, apologize, give a commitment, explain, acknowledge, say less, make a plan), a five-way choice on what the other person needs, and a yes/no on whether the tension is already resolved. Instructions are in English while the chat text stays in its original language, and every question appends a note that knowledge-base background is given context, not an off-topic digression to be penalized.

**The clients and the transport.** jev/JudgeClient.kt posts the state - conversation plus optional background and history fields - with the question set to the configured judgment endpoint (jev/JevClient.kt and jev/HttpJson.kt carry the transport and per-provider headers). The parser rebuilds typed answers: choices with probabilities, scores with confidence, and a ranked list over the reply candidates keyed reply_a, reply_b, reply_c. One defensive detail stands out: because the enriched state fields were not yet verified against every production endpoint, a request that comes back with a 4xx status is retried once without the knowledge fields, so a new feature can degrade an analysis but never break it. jev/ReplyClient.kt handles drafting through the generation route - the default reply model is a DeepSeek chat model - and jev/VisionClient.kt gives the vision route its own configuration for screenshot-based questions.

**The knowledge layer.** core/kb/KbStore.kt persists notes and contacts in the app's private directory; core/kb/KbModels.kt defines them (notes with title, content, tags, and a pinned flag; contacts with name, aliases, relation, and remarks); core/kb/ContextBuilder.kt does the matching - conversation title against names and aliases, tags and titles against the title and the last six messages, pinned notes always included, at most five hits - and core/kb/KbSelfCheck.kt validates the store so a corrupted edit fails loudly instead of silently. Optional history is stored only locally, deduplicated by speaker and text, and the build state is assembled into the background and history fields the judge request carries.

**The surfaces and the guard.** overlay/OverlayController.kt renders the floating panel - danger level, intent, needs, and the three ranked candidates with their probabilities - and routes taps either to the clipboard or to the input writer. capture/GuardedInputWriter.kt is the safety core: every step re-resolves the live session through a callback that validates the token, so a panel tap from a conversation you have already left cannot write into a different input box. It sets the text, checks after 150 milliseconds, refocuses and retries at 300, and falls back through clipboard and the paste action, verifying at each checkpoint that the box now contains the text. SettingsActivity.kt exposes the three routes - judgment, reply, vision - each with its own endpoint, key, and model plus a connectivity test, with presets for OpenRouter, a Bocha-hosted judgment service, TypeSafe-compatible endpoints, a Vercel gateway, and OpenCode Zen; a single key can be inherited by the other routes. MainActivity.kt walks the permission wizard, KnowledgeActivity.kt edits the knowledge base, and unit tests cover the session token logic and the input writer's full fallback sequence.

End to end: a window change becomes a snapshot, the snapshot becomes one judgment call plus one drafting call, the answers become a panel you can read in a glance, and your tap becomes a verified fill - with the send button left, at every point in the flow, untouched.

## Advantages

- **Judgment before prose.** Seven typed questions with probabilities give you the situation first - intent, danger, timing - instead of a confident paragraph that hides its assumptions.
- **A capture layer that touches nothing.** Accessibility reading, on-device OCR, no hooks, no repackaging, no unofficial APIs, no database access - the integration surface is exactly what a screen reader sees.
- **Fill, never send, enforced in code.** The guarded input writer revalidates the live conversation at every step of the fill, and nothing in the code path can press send for you.
- **One small file per job.** Adapters, OCR, questions, clients, knowledge, overlay, and the guard are each isolated modules; a new chat app is one adapter function plus one registry line.
- **Your keys, your endpoints.** Three independent routes with per-route connectivity tests, portable presets, and no server operated by the author in the middle.
- **Calibration lives with the code.** The Python toolkit in tools/jev/ holds the question wording and calibration harness the shipped Kotlin set was ported from.

## Benefits

- **You stop answering the wrong conversation.** Intent and danger scoring catch the loyalty-test and ultimatum cases where a fluent generated reply would do real damage.
- **It works where other tools are banned or blind.** No injection means no ban risk from modified clients, and the OCR fallback covers apps whose interfaces hide text from the accessibility tree.
- **Replies stay consistent with what you know.** Pinned notes, tagged notes, and contact aliases ride along automatically, with a visible count of what was attached.
- **Nothing leaves the phone unless you route it.** Screenshots are recognized on-device, history stays local and off by default, and analysis text goes only to the endpoint you configured.
- **The send button remains yours.** For a tool aimed at emotionally loaded conversations, that boundary is the difference between an assistant and a liability.
- **Honest limits, written down.** The README documents background-freezer quirks on some vendor ROMs, the read-state heuristic for Feishu sides, group-chat blind spots, and the substring-only knowledge matching.

## Usage

Build from source with JDK 17 and the Android SDK (platform 35), or install the signed release APK shipped in the repository:

```bash
adb install -r apk/jev-assistant-v1.4-release.apk
```

On first launch, the wizard asks for the three permissions that matter - the accessibility service, the overlay, and battery exemptions so vendor ROMs do not freeze the background service. Then configure the judgment route (the other two inherit its key if left empty): the simplest start is an OpenRouter key, and the judgment presets also include TypeSafe-compatible endpoints such as a Bocha-hosted service at https://jev.bocha.cn with model bocha-jev-v1, a Vercel AI Gateway entry, and OpenCode Zen with model jev-1.13, all speaking the same typed-decisions request body of model, state, and questions.

After that the app is passive until you open a supported chat: the floating panel appears, the judgment runs in about a second, three ranked candidates appear, and tapping one fills the input box for you to review and send. To adapt another chat app, implement the adapter interface - package name plus an extract function over the accessibility tree returning the title and the message list - register it in the capture service, and nothing downstream changes; the README suggests prototyping the extraction with a UI dump of the target app first.

## Conclusion

Jev 聊天助手 is a quiet argument about what an assistant should be: not a ghostwriter, but a level-headed friend who reads the room, tells you what they see, drafts three options, and hands the decision back to you. The repository earns that argument in code - a capture layer that never crosses the line, a judgment set precise enough to show its probabilities, a knowledge base that keeps replies honest, and an input writer paranoid enough to verify every fill. If you are building any tool that assists humans inside other apps, this source is a masterclass in doing it without crossing the lines that get users hurt.

Links:

- GitHub repository: https://github.com/jev-chat/jev-chat-jarvis
- Guides: https://chatjevs.com/guides/android-setup.html and https://chatjevs.com/guides/review-ai-replies.html
- Privacy policy: PRIVACY.md in the repository
