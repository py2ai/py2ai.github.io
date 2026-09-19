---
layout: post
title: "quiche: Cloudflare's Rust Implementation of QUIC and HTTP/3"
description: "quiche is Cloudflare's open-source Rust implementation of the QUIC transport protocol and HTTP/3, powering the Cloudflare edge, Android's DNS resolver, and curl. A tour of its workspace, architecture, and usage."
date: 2026-09-20
header-img: "img/post-bg.jpg"
permalink: /quiche-Cloudflares-Rust-Implementation-of-QUIC-and-HTTP3/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/quiche/quiche-architecture.svg
tags:
  - quiche
  - QUIC
  - HTTP3
  - Rust
  - Open Source
author: "PyShine"
---

Every time you load a modern website, chances are the bytes arrive over QUIC — the UDP-based transport protocol that replaced TCP-plus-TLS with something faster to set up, harder to bottleneck, and designed for the mobile era. Implementing QUIC correctly is famously difficult: it merges transport and cryptography into one interleaved handshake, multiplexes streams without head-of-line blocking, and rethinks congestion control. [quiche](https://github.com/cloudflare/quiche) is Cloudflare's answer, a Rust implementation of the QUIC transport protocol and HTTP/3 as specified by the [IETF working group](https://quicwg.org/), and it is not a toy: the same code answers HTTP/3 requests on the Cloudflare edge network, resolves DNS inside Android, and gives curl its HTTP/3 support. The project sits near 12,000 stars under a permissive BSD-2-Clause license, and Cloudflare's [design retrospective](https://blog.cloudflare.com/enjoy-a-slice-of-quic-and-rust/) tells the story of how it came about.

What makes quiche special is its shape. Rather than shipping a full server that hides the protocol from you, quiche exposes a low-level API for processing QUIC packets and handling connection state, while your application stays in charge of sockets and the event loop. That design is what lets it live everywhere from a CDN to a phone. The overview below shows how the pieces fit.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/quiche/quiche-overview-architecture.svg" alt="High-level architecture overview of the quiche repository" style="min-width:900px;width:100%;">
</div>

*High-level overview: applications reach quiche through command-line tools or the C API, the core drives QUIC with HTTP/3 and congestion control on top, and an async wrapper plus qlog tooling round out the workspace.*

## Why You Need This

If you build anything networked, QUIC is arriving in your future whether you invite it or not. Browsers already prefer it, the [HTTP/3 standard](https://www.rfc-editor.org/rfc/rfc9114.html) is ratified, and entire ecosystems — proxies, VPNs, censorship-resistant tools — now assume it. The alternatives are a black-box library that owns your event loop, or weeks of reading RFCs. quiche offers the third path: a production-proven implementation you can embed on your own terms, with the socket and timer policies left to you.

The proof of fitness is in the users. Cloudflare runs quiche on its edge, which means it terminates a meaningful fraction of the world's HTTP/3 traffic every day. Android's DNS resolver uses it for DNS over HTTP/3, a demanding environment where connection setup latency directly affects every lookup. And curl integrates quiche for HTTP/3, which makes it one of the most widely exercised QUIC stacks on earth — an enormous amount of real-world edge-case testing happens for free.

There is also a learning payoff. QUIC is documented in long, dense RFCs, but quiche is a readable Rust codebase where each protocol concept maps to a module. Reading it alongside the specification is the fastest way I know to genuinely understand the protocol.

## How It Works

The repository is a Cargo workspace of eleven crates, and the detailed diagram below maps the main ones.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/quiche/quiche-architecture.svg" alt="Detailed architecture of the quiche repository" style="min-width:900px;width:100%;">
</div>

*Detailed architecture: the core crates for connection state, HTTP/3, recovery, and TLS sit at the center, wrapped by the tokio-quiche async layer, with primitives and observability crates around them.*

At the center is the core `quiche` crate, and specifically the `Connection` struct in the nine-thousand-line library root. You create a `Config` — version, ALPN identifiers, flow control limits, idle timeout — and then either `connect()` as a client or `accept()` as a server. From there the API is a disciplined loop: hand incoming packets to `conn.recv()`, pull outgoing packets out of `conn.send()`, and honor the timer returned by `conn.timeout()`. Stream data flows through `stream_send()` and `stream_recv()`, with `readable()` telling you which streams have something to offer, and each outgoing batch even carries a pacing hint for when it should hit the network.

Three supporting modules complete the core. The HTTP/3 module layers request-and-response semantics over QUIC streams with its own connection type. The recovery module implements loss detection and congestion control in two flavors — the classic algorithm and a newer BBR2-based implementation. And the TLS layer implements QUIC's cryptographic handshake on BoringSSL, linked in through the boring-sys crate at build time.

Around the core, the workspace adds the machinery that makes it pleasant to use. `tokio-quiche` is the async integration: an `IoWorker` runs a per-connection I/O loop with batching socket tricks, an `H3Driver` event loop pushes HTTP/3 progress, and the `ApplicationOverQuic` trait is the extension point where you plug in your own service. The primitives crates handle the sharp edges — zero-copy buffers, a fast UDP socket abstraction, and an async cancellation token. The observability trio emits and consumes the qlog format: a schema crate, a Chrome netlog parser, and a visualizer, so you can watch a connection frame by frame. A thin C API compiles the whole thing into a static library for C and C++ applications.

## Advantages

- **Battle-tested.** Runs the Cloudflare edge and Android's DNS resolver; the edge cases found at that scale never reach your issue tracker.
- **Your event loop, your rules.** The library never owns sockets or timers, so it fits runtimes from bare-metal loops to Tokio.
- **Complete workspace.** Client and server tools, an interactive HTTP/3 debugger, async drivers, and qlog visualization ship together.
- **C and C++ friendly.** The FFI feature produces a standalone static library for non-Rust codebases.
- **Permissively licensed.** BSD-2-Clause keeps commercial embedding simple.

## Benefits

The immediate benefit is latency where it is most visible: QUIC establishes connections in one round trip and resumes with zero, which transforms first-request performance on mobile networks. Multiplexed streams without head-of-line blocking keep page loads smooth even when a packet drops. Because quiche is embeddable, those gains arrive in your application — not just in a browser you do not control.

The second benefit is architectural freedom. Since the library separates protocol state from I/O, you can route QUIC through your own socket handling, add your own telemetry, or build a custom protocol over the same transport. That is how the ecosystem around quiche grew: proxies, test harnesses, and specialized servers all reuse the core. And when something misbehaves, the qlog tooling turns an opaque network failure into a timeline you can actually read.

## Usage

Getting started takes minutes:

1. Install Rust 1.88 or later via rustup, plus cmake for the BoringSSL build (NASM on Windows).
2. Clone the repository and try the client: `cargo run --bin quiche-client -- https://cloudflare-quic.com/` fetches a real HTTP/3 site.
3. Run the server: `cargo run --bin quiche-server` with a test certificate, then point the client at it.
4. Embed the API: create a `Config`, call `connect()` or `accept()`, and wire `recv()`, `send()`, and `timeout()` into your event loop — the [API documentation](https://docs.quic.tech/quiche/) and its [docs.rs mirror](https://docs.rs/quiche) walk through each step.
5. Go async by depending on tokio-quiche and implementing the application trait, or use the [C API header](https://github.com/cloudflare/quiche/blob/master/quiche/include/quiche.h) when your host language is C or C++.

The example applications are demonstrations rather than hardened production servers — for production you embed the library. If you want to see QUIC pushed in another direction, our [Hysteria overview](/Hysteria-QUIC-Censorship-Resistant-Proxy/) shows a proxy built on the protocol, and the [Iroh networking stack](/Iroh-Modular-Networking-Stack-Rust/) explores Rust-based peer-to-peer tunnels.

## Conclusion

quiche is what happens when a company that serves a large share of internet traffic decides to open the machinery. It takes the hardest protocol on the modern web and hands it to you as a clean, modular Rust workspace: a rigorous core, an ergonomic async layer, honest tooling for debugging, and an API shape that respects your architecture instead of replacing it. Whether you need HTTP/3 in a product, QUIC under a custom protocol, or simply the best possible companion while reading the RFCs, quiche is the implementation to study — and the easiest slice of the internet's future you will ever bake into your own stack.

