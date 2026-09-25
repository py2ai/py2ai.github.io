---
layout: post
title: "SimpleX Chat: The Messenger With No User IDs, Running on Relay-Only Servers"
description: "SimpleX Chat is a Haskell messaging platform that assigns users no identifiers of any kind. Servers act as disposable relays for unidirectional message queues, private keys stay on your device, and file transfer rides on XFTP. A look at the architecture of simplex-chat/simplex-chat, from the SMP agent layer to the mobile, desktop and terminal clients."
date: 2026-09-25
header-img: "img/post-bg.jpg"
permalink: /SimpleX-Chat-Private-Messaging-Without-User-IDs/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/simplex-chat/simplex-chat-architecture.svg
tags:
  - Privacy
  - Messaging
  - Haskell
  - Open Source
  - Security
categories: [Privacy, Open Source]
keywords: "SimpleX Chat, simplex-chat simplex-chat, private messenger no user ID, SMP protocol, XFTP file transfer, Haskell messenger, metadata privacy, double ratchet post-quantum, relay servers no accounts, private message routing"
author: "PyShine"
---

Every mainstream messenger asks you to identify yourself first. Signal and WhatsApp want a phone number, Telegram wants a username, XMPP and Matrix are built on domain-based addresses, and even the P2P systems that skip the phone book anchor your identity to a public key or another globally unique ID. That identifier becomes the thread an attacker pulls: it ties your profile to your contacts, and your contacts to the times you talk.

[SimpleX Chat](https://github.com/simplex-chat/simplex-chat) takes the opposite position: the platform assigns users no identifiers of any kind - not even random numbers. There is nothing to leak, because there is nothing to attach. Connections are made by sharing one-time invitation links or QR codes, and delivery works through unidirectional "simplex" message queues hosted on relay servers that never learn who you are. The project is written in Haskell, the protocols are open and public domain, and the whole stack - mobile apps, desktop and terminal clients, bots, and language SDKs - lives in this one repository.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/simplex-chat/simplex-chat-overview-architecture.svg" alt="Architecture overview of the SimpleX repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview of the major layers: clients on user devices, the Haskell chat core, the messaging agent, and the relay network.*

## Why You Need This

If you have ever felt uneasy that a single company holds the graph of everyone you know, you already understand the problem. Metadata is often more revealing than message content: who you talk to, how often, from which network, at what time. End-to-end encryption protects what you say; it does nothing to hide the pattern of your relationships from the operator in the middle.

Before platforms like SimpleX, your options were limited. You could trust a centralized provider not to mine your social graph. You could run a federated setup and accept that every participating server can see addressing metadata. Or you could go peer-to-peer and inherit the availability and attack-surface problems of P2P networks - the SimpleX documentation devotes a detailed comparison to why that trade is usually a bad one.

SimpleX Chat is for people and teams who want strong content encryption and metadata privacy without running a heavy federation. If you are a privacy-conscious individual, a journalist protecting sources, a business that cannot afford to leak its contact network, or a developer who wants to build chat services on a platform that cannot profile its users, the design is worth understanding - and using.

## How It Works

The diagram below maps the real subsystems of the repository and how they connect.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/simplex-chat/simplex-chat-architecture.svg" alt="Detailed architecture of the SimpleX messaging system" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the clients, the chat core library, the agent and network layer, local state, and the integration surface.*

### Understanding the Architecture

**Client applications.** The repository builds several clients from one core. The terminal CLI starts at `apps/simplex-chat/Main.hs` and runs on Linux, macOS and Windows. The Android and iOS apps live under `apps/multiplatform`, written with Kotlin Multiplatform; the desktop app shares that code under `apps/multiplatform/desktop`. Both mobile and desktop clients load the Haskell core through a C-level bridge implemented in `src/Simplex/Chat/Mobile.hs`, which exposes the chat controller to Kotlin and Swift. There is also a WebSocket-based chat server (`apps/simplex-chat/Server.hs`, default port 5225) that turns a running client into an automation endpoint.

**Chat core library.** Everything user-facing flows through `src/Simplex/Chat/Core.hs`. Commands arrive from any client surface, get parsed against the chat protocol definitions in `src/Simplex/Chat/Messages.hs`, and are executed against shared state held in `src/Simplex/Chat/Controller.hs` - the `ChatController` owns the current user, the agent client handle, and the output queues. Responses are serialized as UI events by `src/Simplex/Chat/View.hs` and rendered by whichever client is attached. On top of this loop sit the feature subsystems: `src/Simplex/Chat/Delivery.hs` schedules and retries message sends, `src/Simplex/Chat/Call.hs` negotiates end-to-end encrypted WebRTC audio and video calls by carrying ICE candidates and session data inside chat messages, `src/Simplex/Chat/Files.hs` handles attachment flow, and `src/Simplex/Chat/Remote.hs` implements the remote-control sessions that let a mobile app drive a desktop terminal as its "remote host".

**Agent and network layer.** The chat core does not speak to servers itself. It delegates to the SMP agent, a separate library maintained in the [simplexmq](https://github.com/simplex-chat/simplexmq) repository. The agent manages the unidirectional message queues that make SimpleX unique: each connection between two users consists of two queues (one per direction), addressed by pairwise per-queue identifiers instead of user IDs. The agent opens these queues on SMP relay servers, which require authorization for messages sent to a queue but perform no user authentication at all - they have no user records, do not talk to each other, and drop messages once delivered. The same agent handles file transfer through XFTP servers, which store encrypted file chunks temporarily, and `src/Simplex/Chat/Operators.hs` holds the server operator presets and configuration used to choose which relays the app connects to.

**Local state.** All user data lives on the device. `src/Simplex/Chat/Store.hs` opens a pair of databases - chat store and agent store - implemented over SQLite with encryption on mobile and desktop (`src/Simplex/Chat/Store/SQLite`), with a Postgres variant (`src/Simplex/Chat/Store/Postgres`) for server-side deployments. Contacts, groups, and message history are only ever persisted where the user controls the keys.

**Bots and integrations.** Because the core is a library, bots embed it directly - the squaring-bot example at `apps/simplex-bot/Main.hs` builds a complete chat bot in a page of Haskell. The same core is wrapped as Node.js native bindings (`packages/simplex-chat-nodejs`) and Python bindings (`packages/simplex-chat-python`), and a TypeScript client package (`packages/simplex-chat-client`) connects over the WebSocket API for building custom interfaces.

**Data flow.** A typical message follows this path: the client sends a chat command through the FFI bridge, WebSocket, or terminal; the core parses it and records it in the local database; the delivery layer hands the encrypted payload to the SMP agent; the agent pushes it into the recipient's queue on an SMP relay; the recipient's agent picks it up, decrypts it through the double-ratchet session, and the recipient's core emits a response event to their UI. At no point in that chain does any server hold a user identifier or learn the full graph of who talks to whom.

## Advantages

- **No user identifiers, by design.** For `n` users the network can hold up to `n * (n-1)` message queues, each addressed by pairwise identifiers. Observing the application-level network graph becomes structurally difficult, not just policy-restricted.
- **Servers that know nothing.** Relay servers keep no user accounts, do not communicate with each other, and use in-memory message storage - they cannot correlate sent and received traffic because an additional NaCl cryptobox layer ensures there is no ciphertext in common between the two directions.
- **Layered cryptography.** Each queue is end-to-end encrypted with NaCl cryptobox on top of TLS 1.2/1.3; each conversation runs the double ratchet algorithm with post-quantum resistant key exchange on every ratchet step; message metadata - including server receive times rounded to a second - travels inside the encrypted envelope.
- **Private message routing by default.** Modern clients route traffic through additional relays so that unknown messaging servers cannot see your IP address, with Tor support available on top.
- **A real implementation, reviewed.** The platform is written in Haskell with a strict type discipline, and both the implementation (2022) and the cryptographic design (2024) have undergone external security review by Trail of Bits, with reports published on the project blog.
- **One core, many surfaces.** Terminal, Android, iOS, desktop, bots, and Node.js/Python/TypeScript SDKs all share the same audited core rather than reimplementing protocol logic per platform.

## Benefits

- **Identity freedom.** You can be contacted only if you choose to share an invitation link or an optional, revocable address. Spam has nothing to latch onto, and you can delete an address without losing existing connections.
- **Data you actually own.** Your contacts, groups, and history live in an encrypted local database under your control - exportable and portable, not held hostage by an account.
- **Resilience without federation.** Relays are cheap, stateless, and disposable; you can run your own servers alongside the pre-configured ones, and conversations can be moved between relays manually by rotating queues.
- **A platform for builders.** The bot API, WebSocket server, and language bindings let you ship a chat service in any language without inventing a transport - the boring parts of messaging are already solved.
- **Open and verifiable.** AGPLv3-licensed, protocol specifications public, reproducible server builds supported. You are not asked to trust; you are invited to check.

## Usage

The quickest way to try SimpleX Chat is the terminal client on Linux, macOS or Windows:

```bash
curl -o- https://raw.githubusercontent.com/simplex-chat/simplex-chat/stable/install.sh | bash
```

Once installed, start it from your terminal:

```bash
simplex-chat
```

You create a local encrypted profile on first run, then connect with someone by exchanging a one-time invitation link or QR code - the channel you share it on does not need to be secure, as long as you can confirm who sent it.

For developers, the same CLI runs as a local WebSocket server that any language can drive. Passing `--chat-server-port` starts the socket:

```bash
simplex-chat --chat-server-port 5225
```

A chat bot is a program that connects to that socket, sends chat commands, and reacts to contact messages.
The full bot API reference and a TypeScript client with a squaring-bot example are included in the repository under the `bots/` and `packages/` directories, alongside Node.js and Python bindings for embedding the core library natively. If you prefer the full experience, the Android and iOS apps and the desktop app are built from the same core and are available through the project's download pages.

## Conclusion

SimpleX Chat is a rare thing in messaging: a design that removes the identifier instead of encrypting around it, backed by a complete, working implementation in one open repository. The Haskell core, the relay-only server model, the SMP and XFTP protocols, and the family of clients built on a single library make it both a practical messenger today and a solid foundation for building private services. Explore the repository, read the protocol docs, and consider what your stack would look like if it simply had no user IDs to protect.

**Links:**

- Repository: [https://github.com/simplex-chat/simplex-chat](https://github.com/simplex-chat/simplex-chat)
- Website: [https://simplex.chat](https://simplex.chat)
- Protocol and agent library: [https://github.com/simplex-chat/simplexmq](https://github.com/simplex-chat/simplexmq)
- XFTP file transfer announcement: [https://simplex.chat/blog/20230301-simplex-file-transfer-protocol.html](https://simplex.chat/blog/20230301-simplex-file-transfer-protocol.html)
- Terminal CLI documentation: [https://github.com/simplex-chat/simplex-chat/blob/stable/docs/CLI.md](https://github.com/simplex-chat/simplex-chat/blob/stable/docs/CLI.md)
