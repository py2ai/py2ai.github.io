---
layout: post
title: "AIKON: Today's AI on a 2007 Nokia - Inside emir/AIKON"
description: "A source-level tour of AIKON, the Java ME chat client for Nokia Series 40 and Symbian S60 phones plus the small Go server that carries 2007-era handsets to Claude, OpenAI, Gemini and Grok over TLS 1.0 with a private certificate authority."
date: 2026-10-07
header-img: "img/post-bg.jpg"
permalink: /aikon-nokia-ai-gateway/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/aikon/emir-aikon-architecture.svg
tags: [Java ME, Retro Tech, Go, LLM]
categories: [AI, Open Source]
keywords: AIKON, Nokia Series 40, Symbian S60, Java ME, MIDP 2.0, TLS 1.0, private CA, Go gateway, Claude, Gemini, OpenAI, Grok
author: "PyShine"
---

There is a drawer in many homes holding a Nokia that still turns on, and until recently that phone had no path to modern AI. AIKON, published by Emir Karşıyakalı under the MIT license, builds that path. It is an AI chat client for Nokia Series 40 and Symbian S60 phones, written in Java ME against CLDC 1.1 and MIDP 2.0, paired with a small Go server. Pick a model per chat from Claude, OpenAI, Gemini or Grok, type on the keypad, and read the answer on a 240x320 screen, complete with today's news, weather and exchange rates from web search. The project was tested on a Nokia 6300 and a Nokia E63, which means the TLS handshake, the certificate store, and every screen layout were verified on real 2007 hardware.

The project was called Claude S40 while it only talked to Claude; since it now speaks to OpenAI, Gemini and Grok as well, it became AIKON, Nokia spelled backwards. The repository moved from emir/claude-s40 to emir/AIKON, and the Java package and server names keep the old name as a quiet piece of history. What makes the repository remarkable is not nostalgia but engineering discipline: the server speaks TLS 1.0 to the phone with a certificate from your own private CA, because no publicly obtainable certificate chains to a root a 6300 trusts, and it speaks modern HTTPS to the model providers on the same port, split by SNI.

The feature list on the phone side is long enough to be a product in its own right: per-chat model selection, paged reading mode, message actions like shorten and translate, pinned and searchable chat history, voice dictation, photo questions, calendar and to-do integration, twenty quick prompts, and a UI in eight languages. This tour walks both halves of the codebase, the MIDlet in Java and the gateway in Go, and shows how a phone from 2007 ends up chatting with models from this decade.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/aikon/emir-aikon-overview-architecture.svg" alt="Architecture overview of the AIKON repository, showing the Java ME phone app, the Go gateway with its phone API, chat service and SQLite store, the provider clients, and the private CA tooling" style="max-width:100%;">
</div>
<p><em>Architecture overview of the AIKON repository, from the Nokia keypad to the model providers.</em></p>

Reading the overview from left to right:

- The **MIDlet entry** in [app/src/io/github/emir/claudes40/ClaudeS40MIDlet.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/ClaudeS40MIDlet.java) owns every screen, with the chat canvas in [app/src/io/github/emir/claudes40/ChatCanvas.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/ChatCanvas.java) drawing bubbles, day headings, and a typing indicator that counts the seconds.
- The **chat session** in [app/src/io/github/emir/claudes40/ChatSession.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/ChatSession.java) tracks messages and the selected model from [app/src/io/github/emir/claudes40/Models.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/Models.java), and sends everything through the networking layer in [app/src/io/github/emir/claudes40/Net.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/Net.java).
- The **gateway server** in [server/main.go](https://github.com/emir/AIKON/blob/main/server/main.go) terminates TLS for the phone, routes the phone API in [server/protocol.go](https://github.com/emir/AIKON/blob/main/server/protocol.go), and feeds the chat service in [server/chat.go](https://github.com/emir/AIKON/blob/main/server/chat.go), which persists conversations in the SQLite store in [server/store.go](https://github.com/emir/AIKON/blob/main/server/store.go).
- The **provider clients** start with the Claude adapter in [server/anthropic.go](https://github.com/emir/AIKON/blob/main/server/anthropic.go) and continue to OpenAI, Grok and Gemini.
- The **web mux** in [server/web.go](https://github.com/emir/AIKON/blob/main/server/web.go) splits one TLS port between modern browsers and the phone, and the private CA tool in [server/scripts/pki.sh](https://github.com/emir/AIKON/blob/main/server/scripts/pki.sh) issues the certificate the phone trusts.

## Why You Need This

The first reason is that this is a masterclass in constrained-client engineering. Every screen except text entry is drawn by the app itself on a Canvas, with line icons, two-line list rows, settings switches that save at once, light, dark or automatic themes, and three text sizes. Replies keep their paragraphs, lists and hanging indents; long answers page through a reading mode that survives a text-size change. The keypad drives everything, with a Shortcuts screen listing every key, and a retry after an error is one keypress that never charges twice. If you have ever wondered what product polish looks like when the screen is 240x320 and the input is a 12-key pad, this codebase is the reference.

The second reason is the TLS story, which is documented with unusual honesty in [docs/ARCHITECTURE.md](https://github.com/emir/AIKON/blob/main/docs/ARCHITECTURE.md). A Nokia 6300 sends a ClientHello with TLS 1.0 only, no SNI, no extensions, and cipher suites from another era; its certificate store holds only late-1990s and 2000s VeriSign, Thawte, Equifax and GeoTrust roots, which are expired or distrusted today. No public CA issues certificates under those roots, and CDN front ends fail the handshake before a certificate is even sent. The project's answer is a private root CA and an RSA-2048 server certificate that the user saves on the phone once, with the server negotiating TLS 1.0-1.3, AES-CBC suites for the phone, and ECDHE for modern clients, and no RC4, no 3DES, and no plain HTTP listener.

The third reason is responsible gateway design. The server keeps per-device access tokens issued through a 6-digit pairing code, enforces daily request, token, web-search, voice-message and photo limits, and never retries a paid model call automatically. Chats live in SQLite for 30 days, pinned ones until unpinned, the admin API listens on localhost only, and the logs never contain message text, replies, tokens, keys or client IPs. Anyone running a small paid AI service can borrow this posture directly.

## How It Works

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/aikon/emir-aikon-architecture.svg" alt="Detailed architecture of the AIKON repository, showing the Java ME app internals, the Go gateway services, the four provider clients, and the deployment and build tooling" style="max-width:100%;">
</div>
<p><em>Detailed architecture of the AIKON repository, including the provider clients and the deployment tooling.</em></p>

### Understanding the Architecture

**The phone app is a hand-drawn product.** The MIDlet in [app/src/io/github/emir/claudes40/ClaudeS40MIDlet.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/ClaudeS40MIDlet.java) composes the splash, home, chat and settings canvases. The chat canvas in [app/src/io/github/emir/claudes40/ChatCanvas.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/ChatCanvas.java) renders your messages in bubbles and replies full width, with day headings and a typing indicator; the chat list in [app/src/io/github/emir/claudes40/ChatList.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/ChatList.java) brings earlier chats from the server with pinning, deletion and ASCII-tolerant search, and [app/src/io/github/emir/claudes40/ChatStore.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/ChatStore.java) keeps an optional offline copy of the last chat. Every label resolves through [app/src/io/github/emir/claudes40/LangPack.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/LangPack.java).

**Messages do more than display.** The session in [app/src/io/github/emir/claudes40/ChatSession.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/ChatSession.java) feeds the message actions in [app/src/io/github/emir/claudes40/Text.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/Text.java), where shorten, explain more simply, translate and ask-about-it only fill the editor until you press Send. Voice messages recorded by [app/src/io/github/emir/claudes40/Dictation.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/Dictation.java) travel through [app/src/io/github/emir/claudes40/Net.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/Net.java) and come back as text you can fix before sending, and photos from [app/src/io/github/emir/claudes40/Photo.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/Photo.java) stay visible to follow-up questions in the same chat. Calendar requests produce a prefilled form in [app/src/io/github/emir/claudes40/CalendarForm.java](https://github.com/emir/AIKON/blob/main/app/src/io/github/emir/claudes40/CalendarForm.java), and only your Save writes to the phone's own calendar or to-do list through the PIM APIs stubbed under [app/stubs/jsr75-pim/](https://github.com/emir/AIKON/tree/main/app/stubs/jsr75-pim).

**The gateway is one Go binary with three listeners.** The public TLS listener in [server/main.go](https://github.com/emir/AIKON/blob/main/server/main.go) serves the phone API in [server/protocol.go](https://github.com/emir/AIKON/blob/main/server/protocol.go), including the pairing claim route; the admin listener is plain HTTP on localhost only; and an optional plain-HTTP landing listener exists solely for the first step on a phone that does not trust the root yet, showing the root's fingerprint and the ca.cer file. One TLS port serves both audiences: a client that names a public host in SNI gets a Let's Encrypt certificate and the web mux in [server/web.go](https://github.com/emir/AIKON/blob/main/server/web.go) with HSTS and a strict CSP, while the Nokia, which sends no SNI, gets the private-CA certificate and the phone API. A Host header that does not match the side chosen in the handshake gets a 421, so the phone API is never served under a public name.

**The chat service turns one endpoint into four providers.** The service in [server/chat.go](https://github.com/emir/AIKON/blob/main/server/chat.go) resolves the model from the catalog built by [server/model.go](https://github.com/emir/AIKON/blob/main/server/model.go) from the single MODELS setting with a DEFAULT_MODEL, checks the daily limits in [server/meter.go](https://github.com/emir/AIKON/blob/main/server/meter.go), and dispatches to the Claude client in [server/anthropic.go](https://github.com/emir/AIKON/blob/main/server/anthropic.go), the OpenAI and xAI Grok Responses-API client in [server/responses.go](https://github.com/emir/AIKON/blob/main/server/responses.go), or the Gemini client in [server/gemini.go](https://github.com/emir/AIKON/blob/main/server/gemini.go). Web search runs on the server with each provider's own search tool, and sources are listed under the reply. Replies are fitted to the small screen by [server/sanitize.go](https://github.com/emir/AIKON/blob/main/server/sanitize.go), photos are downscaled by [server/image.go](https://github.com/emir/AIKON/blob/main/server/image.go), and voice recordings in the phone's AMR format are transcribed by [server/transcribe.go](https://github.com/emir/AIKON/blob/main/server/transcribe.go) with a minimal ffmpeg in the image. Every conversation lands in [server/store.go](https://github.com/emir/AIKON/blob/main/server/store.go), and the eight language packs embedded from [server/lang/](https://github.com/emir/AIKON/tree/main/server/lang) are served by [server/lang.go](https://github.com/emir/AIKON/blob/main/server/lang.go) so the JAR itself only carries English.

**Deployment treats reproducibility as a feature.** The private CA is created and inspected with [server/scripts/pki.sh](https://github.com/emir/AIKON/blob/main/server/scripts/pki.sh), which defaults to RSA-2048 with SHA-1 signatures, the combination verified to work on a 6300, with a 10-year root and a 2-year server certificate. The DigitalOcean helper in [server/deploy/do-create.sh](https://github.com/emir/AIKON/blob/main/server/deploy/do-create.sh) prints a plan and creates nothing without an explicit flag; [server/deploy/push.sh](https://github.com/emir/AIKON/blob/main/server/deploy/push.sh) builds the image locally and starts the compose stack from [server/compose.yaml](https://github.com/emir/AIKON/blob/main/server/compose.yaml) on the server; and [server/deploy/smoke.sh](https://github.com/emir/AIKON/blob/main/server/deploy/smoke.sh) probes health and pairing in test mode, where replies are fake and cost nothing. On the phone side, [app/tools/package.py](https://github.com/emir/AIKON/blob/main/app/tools/package.py) produces AIKON.jad, AIKON.jar and SHA256SUMS with reproducible rebuilds.

End to end, a message travels like this: you type on the keypad, the chat session hands the request to the networking layer, which opens TLS to the gateway with the private-CA certificate. The phone API authenticates the device token, the chat service checks the daily meter, assembles the conversation with your saved notes, and calls the selected provider, with web search attached when the question needs it. The reply comes back sanitized and paged if long, labelled with its model, and stored in SQLite where a later chat can pick it up.

## Advantages

- **Runs on hardware everyone already owns.** A CLDC 1.1 / MIDP 2.0 app tested on the Nokia 6300 and E63, with a download path through the phone's own browser from the server's download page.
- **One gateway, four providers.** Claude, OpenAI, Gemini and Grok behind a single MODELS setting with a default, switchable per chat, so the handset outlives any single provider's product decisions.
- **A rigorous TLS story.** A private CA, a documented cipher policy, SNI-based host splitting on one port, and a landing listener that exists only to hand the phone the root certificate.
- **Abuse-resistant by design.** Per-device tokens from 6-digit pairing, daily request, token, web-search, voice and photo limits, and no automatic retries of paid calls.
- **Real product polish.** Reading mode, message actions, pinned chats, offline copy, saved replies as .txt files, calendar and to-do forms, data-usage stats, and eight UI languages.
- **Reproducible builds on both ends.** Checksummed JAD and JAR packaging on the phone side, and a compose-based server deploy with a smoke test that runs in free test mode first.

## Benefits

- **Second life for drawer phones.** A device that could not reach a modern API now carries news, weather, translation and conversation, with nothing discarded but its browser's dignity.
- **A reference for legacy-TLS servers.** The handshake analysis in the architecture doc, from ClientHello quirks to the expiry of 2000s roots, is directly reusable when supporting any old embedded client.
- **Safer than a shared proxy.** Chats expire after 30 days unless pinned, and the admin API is localhost-only, with logs that carry status and latency but never message text or keys.
- **Cost visibility.** The meter's daily limits and the phone's own data-usage screen keep both bandwidth and API spend observable.
- **Localization done right.** Server-embedded language packs mean new languages ship without a new JAR, and the build fails loudly if two English strings share a hash code.
- **Testable without spending.** Test mode returns fake replies, so the whole setup, pairing and deploy flow can be rehearsed before the first real model call.

## Usage

Create your private certificate authority and server certificate, then deploy:

```bash
server/scripts/pki.sh ca     ~/.config/claude-s40/pki
server/scripts/pki.sh server ~/.config/claude-s40/pki $IP
server/scripts/pki.sh show   ~/.config/claude-s40/pki   # note the root's fingerprints
```

```bash
cd server
make test                          # go vet + tests
deploy/push.sh $SERVER             # shows the plan
deploy/push.sh $SERVER --execute   # build here, send image, start on :443
```

The server starts in test mode with fake replies that cost nothing; probe it, then list and revoke the smoke device:

```bash
deploy/smoke.sh $IP 443 ~/.config/claude-s40/pki/ca.pem
deploy/admin.sh $SERVER devices    # then: deploy/admin.sh $SERVER revoke <smoke device id>
```

Build the phone app and install it on the handset:

```bash
echo "GATEWAY_URL=https://$IP" > app/app.local.properties
make -C app                        # build + package checks + reproducible rebuild
ls app/dist                        # AIKON.jad, AIKON.jar, SHA256SUMS
```

On first start the phone runs a setup wizard: enter the server address, run the connection test, and pair with the 6-digit code after saving the root certificate. Then pick a provider and model, and start typing; the keypad drives every screen, and the Shortcuts screen lists them all.

## Conclusion

AIKON is that rare project where the constraint is the point. By taking a 2007 keypad phone seriously, its TLS handshake, its memory, its screen, and its 12-key input, the repository produces a cleaner small-service design than most modern chat front ends: one binary, one port, strict host splitting, metered access, and a reproducible deploy. It is also simply delightful to read, from the hand-drawn canvases to the translation checker that fails the build on hash collisions. If you care about software that runs for decades, this is a codebase to study and a phone worth rescuing from the drawer.

Links:

- Repository: [https://github.com/emir/AIKON](https://github.com/emir/AIKON)
- Architecture notes: [docs/ARCHITECTURE.md](https://github.com/emir/AIKON/blob/main/docs/ARCHITECTURE.md)
- Setup guide: [docs/SETUP.md](https://github.com/emir/AIKON/blob/main/docs/SETUP.md)
- License: MIT
