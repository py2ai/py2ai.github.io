---
layout: post
title: "Tailscale's Open-Source Client: A WireGuard Mesh VPN with the Data Plane Fully Open"
description: "Tailscale builds a zero-config WireGuard mesh VPN where devices connect peer-to-peer across NAT and firewalls, authenticated by SSO + 2FA. With 33.8k stars and a BSD-3-Clause license, the open-source client repo (tailscaled daemon + tailscale CLI, written in Go) implements the entire data plane — WireGuard engine, DERP relays, NAT traversal, Tailnet Key Authority, embeddable tsnet, and Tailscale SSH — while only the coordination control server stays proprietary."
date: 2026-07-11
header-img: "img/post-bg.jpg"
permalink: /Tailscale-Open-Source-Client-WireGuard-Mesh-VPN-in-Go/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Tailscale
  - WireGuard
  - VPN
  - Go
  - Networking
  - Open Source
author: "PyShine"
---

# Tailscale's Open-Source Client: A WireGuard Mesh VPN with the Data Plane Fully Open

Most VPN products are a single binary you trust opaquely. **Tailscale** is different in one important way: the part that touches your packets — the entire data plane — is open source. The `tailscale/tailscale` repository (33.8k stars, BSD-3-Clause, written in Go) contains the `tailscaled` daemon and the `tailscale` CLI that run on Linux, Windows, macOS, FreeBSD, and OpenBSD, and the core networking code that powers the iOS and Android apps. What stays proprietary is only the coordination server — the thing that hands out keys and ACLs but never sees your traffic.

The tagline is "The easiest, most secure way to use WireGuard and 2FA." Let us look at how the open-source client is built.

## Architecture: Control Plane vs Data Plane

The defining design decision in Tailscale is the clean split between a proprietary control plane and an open-source data plane.

![Tailscale Architecture](/assets/img/diagrams/tailscale/tailscale-architecture.svg)

**Control plane (proprietary, hosted by Tailscale)** — the coordination server authenticates users through SSO/OAuth plus 2FA, enforces ACLs, and distributes WireGuard public keys and network maps to nodes. Critically, it does **not** see user traffic. Its only role is connection setup: handing out keys and the list of peers.

**Data plane (open source, this repo)** — all actual packet forwarding happens directly between nodes using WireGuard tunnels. The `tailscaled` daemon on each device handles this. After the coordination server hands out keys, ongoing data flows peer-to-peer and encrypted end-to-end; the control server is no longer in the path.

This split is the whole trust story. You can read every line of the code that touches your packets. The only opaque piece is the key-distribution server, and a compromised coordination server cannot decrypt your traffic — it only has public keys and ACLs. The `tka/` (Tailnet Key Authority) directory goes further: a blockchain-inspired network-level key authority that lets a tailnet detect coordination-server compromise by making tamper-evident state that the server cannot silently rewrite.

## The Mesh and DERP Fallback

Tailscale is a mesh, not a hub-and-spoke VPN. Nodes connect peer-to-peer rather than through a central VPN server. The hard part is establishing those direct connections across NAT and restrictive firewalls, which is what the `disco/` protocol handles.

![Tailscale Mesh](/assets/img/diagrams/tailscale/tailscale-mesh.svg)

The connectivity logic:

1. **NAT traversal** via the `disco/` discovery and signaling protocol attempts to establish a direct peer-to-peer connection
2. **Decision: is direct P2P possible?** — if yes, a direct WireGuard tunnel is established (the fastest path)
3. **DERP fallback** when direct traversal fails — for example, symmetric NAT on both sides. DERP (Designated Encrypted Relay for Packets) servers relay already-encrypted WireGuard packets between peers. A DERP relay **cannot decrypt traffic** — it only forwards encrypted packets. The privacy property is preserved even when the fast path is not available.

This is why Tailscale "just works" from a hotel Wi-Fi behind symmetric NAT: you do not get a direct tunnel, but you get a relayed one that is still end-to-end encrypted, and the user does not have to know or care which path was taken.

## Repository Components

The repo is organized by subsystem, and the directory layout is a good map of how a mesh VPN is actually built.

![Tailscale Components](/assets/img/diagrams/tailscale/tailscale-components.svg)

A few of the notable pieces:

- **`wgengine/`** — the WireGuard engine, the core data plane
- **`cmd/`** — the `tailscale` and `tailscaled` entry points
- **`control/`** — client-side communication with the coordination server; `tailcfg/` holds the configuration protocol types
- **`tka/`** — Tailnet Key Authority, the tamper-evident network-level key management
- **`derp/`** — DERP relay client and server implementation
- **`disco/`** — discovery and NAT-traversal protocol
- **`ipn/`** — "IP Notifications," the local node configuration and state management
- **`tsnet/`** — an embeddable Tailscale library for Go applications, so you can build Tailscale into a program rather than installing it on a host
- **`k8s-operator/`** and **`kube/`** — first-class Kubernetes integration
- **`ssh/tailssh/`** — the Tailscale SSH server
- **`drive/`** — Taildrop and shared-filesystem features
- **`sessionrecording/`**, **`posture/`**, **`prober/`**, **`health/`** — session recording, device-posture attestation, active connectivity probing, and health tracking

The breadth here is the point. This is not a thin client wrapper around WireGuard; it is a complete networking stack — NAT traversal, relay, key authority, embeddable library, Kubernetes operator, SSH, file sharing, posture, and health — all open and auditable.

## What Is Open and What Is Not

Worth being precise about, because the split is the trust story:

**Open source (this repo):** the `tailscaled` daemon (data plane), the `tailscale` CLI, the DERP relay implementation, NAT traversal / discovery, the WireGuard engine, all client-side code for Linux/Windows/macOS/FreeBSD/OpenBSD, and the mobile core networking code.

**Proprietary / not in this repo:** the coordination/control server (Tailscale's hosted control plane), the GUI wrappers on macOS, iOS, and Windows (the Android GUI lives in a separate `tailscale/tailscale-android` repo), and Synology/QNAP/Chocolatey packaging.

The philosophy, stated in the repo: the data plane that handles all user traffic is fully open and auditable, while the coordination server that only handles key distribution and ACLs remains proprietary. You can read more at `tailscale.com/opensource/`.

## Installation and Connection

The build path is a standard Go install:

```
go install tailscale.com/cmd/tailscale{,d}
```

For distribution packaging, `build_dist.sh` embeds commit IDs and version info:

```
./build_dist.sh tailscale.com/cmd/tailscale
./build_dist.sh tailscale.com/cmd/tailscaled
```

The project tracks the latest Go release (Go 1.26 at time of writing) and pre-built packages live at `pkgs.tailscale.com`.

![Tailscale Workflow](/assets/img/diagrams/tailscale/tailscale-workflow.svg)

The connection flow:

1. **Install** the `tailscale` and `tailscaled` binaries
2. **Start the daemon** with `tailscaled`
3. **Authenticate** with `tailscale up` — SSO + 2FA in the browser
4. **Coordination server** issues WireGuard keys and a network map
5. **NAT traversal** runs via the disco protocol to establish connectivity with peers
6. **Path selection** — direct WireGuard peer-to-peer tunnel if possible, otherwise DERP relay (encrypted, forward-only)

## Why It Matters

Tailscale's open-source client is worth studying for three reasons.

First, **the control-plane / data-plane split is the correct trust architecture for a VPN.** By keeping the opaque part limited to key distribution — a part that, by construction, cannot decrypt your traffic — and open-sourcing everything that touches packets, Tailscale earns a kind of auditability that "trust us" VPNs cannot. The Tailnet Key Authority then makes even the coordination server's state tamper-evident, so a compromised control plane is detectable. That is a layered trust model worth copying.

Second, **the mesh-plus-DERP-fallback design is why it actually works in the real world.** Hub-and-spoke VPNs are simple but have a single point of failure and a bandwidth bottleneck at the hub. Pure mesh is fast but breaks behind symmetric NAT. DERP relays give you a reliable fallback that preserves the end-to-end encryption property — the user gets connectivity without ever having to know whether they are on the fast path or the relay.

Third, **`tsnet` turns Tailscale into a library, not just a product.** Embedding a full mesh VPN into a Go application as a library — without a host-level install — is a building block for a lot of things beyond "connect my laptop to my server." Combined with the `k8s-operator`, the SSH server, and the posture and health subsystems, the repo reads less like a VPN client and more like a programmable secure-connectivity platform.

If you care about networking, trust models, or Go systems programming, the `tailscale/tailscale` repository is one of the better codebases to read. It is working, production-grade, and almost entirely open.

**Links:**

- GitHub: [https://github.com/tailscale/tailscale](https://github.com/tailscale/tailscale)
- Install: `go install tailscale.com/cmd/tailscale{,d}`
- Pre-built packages: [https://pkgs.tailscale.com](https://pkgs.tailscale.com)
- Open-source philosophy: [https://tailscale.com/opensource/](https://tailscale.com/opensource/)
- License: BSD-3-Clause (plus a separate PATENTS grant)