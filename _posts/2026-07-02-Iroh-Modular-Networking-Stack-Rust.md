---
layout: post
title: "Iroh: Modular Networking Stack in Rust"
description: "IP addresses break, dial keys instead. A Rust library for establishing direct connections between endpoints using dial-by-public-key instead of IP addresses."
date: 2026-07-02
header-img: "img/post-bg.jpg"
permalink: /Iroh-Modular-Networking-Stack-Rust/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Rust
  - Networking
  - P2P
  - QUIC
  - Hole-Punching
  - Relay
author: "PyShine"
---

## Introduction

IP addresses are fundamentally unreliable. They change, they get blocked, they require complex NAT traversal, and they create security vulnerabilities. The traditional networking model that relies on IP addresses and ports breaks down in modern distributed systems, peer-to-peer applications, and real-time collaboration tools. When two devices want to connect, they shouldn't need to know each other's IP addresses or configure firewalls. They should just need to know each other's identities.

Iroh solves this problem by introducing a revolutionary dial-by-public-key model for networking. Instead of connecting to IP addresses, you connect to cryptographic identities. Iroh handles all the complexity of NAT traversal, hole punching, relay connections, and connection optimization automatically. It finds the fastest route between endpoints, whether that's a direct connection, a relay server, or a combination of both. The result is a networking stack that is resilient, secure, and incredibly easy to use.

This blog post explores Iroh's architecture, its key features, and how it transforms the way we think about building distributed systems in Rust.

## What is Iroh?

Iroh is a Rust library that provides a modular networking stack for establishing direct connections between endpoints using dial-by-public-key instead of IP addresses. The project slogan "less net work for networks" captures its core philosophy: simplify networking by abstracting away the complexity of IP addresses, NAT traversal, and connection management.

At its core, Iroh gives you an API for dialing by public key. You say "connect to that phone," and Iroh will find and maintain the fastest connection for you, regardless of where it is. This abstraction layer sits on top of QUIC, providing authenticated encryption, concurrent streams with stream priorities, datagram transport, and automatic avoidance of head-of-line blocking.

Iroh is designed for modern distributed systems where reliability and ease of use are paramount. It's particularly well-suited for peer-to-peer applications, real-time collaboration tools, distributed storage systems, and any application that needs to establish direct connections between devices without requiring users to configure firewalls or understand network topology.

The project is developed by n0, a team focused on building decentralized infrastructure. Iroh is part of a larger ecosystem of crates that build on top of the core networking functionality, including iroh-blobs for content-addressed blob transfer, iroh-gossip for publish-subscribe overlay networks, and iroh-docs for eventually-consistent key-value stores.

## The Dial-by-Public-Key Model

The dial-by-public-key model is the heart of Iroh's innovation. In traditional networking, you connect to an IP address and port, like `192.168.1.100:8080`. This requires the remote endpoint to be reachable, properly configured, and not blocked by firewalls or NAT devices. When devices are behind NAT or in restrictive network environments, establishing connections becomes increasingly difficult.

Iroh replaces this model with cryptographic identities. Each Iroh endpoint has a unique public key that serves as its network address. To connect to another endpoint, you simply provide its public key. Iroh handles all the complexity of finding the endpoint, establishing a connection, and maintaining it.

The process begins when you create an Iroh endpoint using `Endpoint::bind()`. This generates a unique public key and binds the endpoint to listen for incoming connections. The endpoint can be configured with various presets that optimize for different use cases, such as `presets::N0` which is the default configuration.

To connect to another endpoint, you use `endpoint.connect()`, passing the remote endpoint's public key and an application-layer protocol negotiation (ALPN) string. The ALPN string identifies which protocol you want to use for communication, allowing multiple protocols to coexist on the same connection.

Once connected, Iroh automatically attempts to establish a direct connection between the endpoints. This is where the magic happens. Iroh uses hole punching techniques to traverse NAT devices and firewalls. If direct connection fails, Iroh falls back to its ecosystem of public relay servers. These relays temporarily route encrypted traffic until a direct connection is established, at which point the relay steps back and data flows directly between endpoints.

The connection management is fully automatic. Iroh continuously measures connection quality and switches between direct connections, relays, and different relay servers to ensure optimal performance. This means your application gets the best possible connection without any manual intervention.

## Architecture Deep Dive

Iroh's architecture is designed for modularity, performance, and ease of use. The project is organized as a Rust workspace with multiple crates that can be used independently or together. Understanding this architecture is key to leveraging Iroh effectively in your applications.

### Core Components

The main components of Iroh's architecture include:

**iroh**: The core library that provides the primary API for hole punching and communicating with relays. This is the main crate that most users will interact with. It handles endpoint creation, connection management, protocol routing, and automatic connection optimization.

**iroh-relay**: The relay client and server implementation. This crate provides the infrastructure for public relay servers that Iroh uses as a fallback when direct connections aren't possible. The relay server runs in production for the public relays and can also be deployed by users who want to run their own relays. The relay supports multiple access control modes, including allowlists, denylists, shared tokens, and HTTP callouts to external authentication services.

**iroh-base**: Common types like `EndpointId` and `RelayUrl` that are used across the Iroh ecosystem. This crate provides foundational data structures and utilities that are shared between the core library and protocol crates.

**iroh-dns-server**: A DNS server implementation that powers the DNS/Pkarr address lookup for EndpointIds. This server runs at `dns.iroh.link` and provides DNS-over-HTTPS and DNS-over-TCP/UDP services for resolving EndpointIds to their public keys. It also functions as a pkarr relay, serving signed packets over DNS.

### Layered Architecture

Iroh follows a layered architecture that separates concerns and enables composability:

**Transport Layer**: Built on top of QUIC via the noq library. QUIC provides authenticated encryption, concurrent streams with stream priorities, datagram transport, and automatic avoidance of head-of-line blocking. This gives Iroh strong security guarantees and efficient data transfer.

**Connection Layer**: Handles the establishment and management of connections between endpoints. This includes hole punching, relay fallback, connection monitoring, and automatic reconnection. The connection layer maintains a pool of connections to each remote endpoint and selects the best one based on performance metrics.

**Protocol Layer**: Provides a framework for defining and implementing application protocols. Each protocol is identified by an ALPN string and implements the `ProtocolHandler` trait. The protocol layer routes incoming connections to the appropriate handler and manages protocol-specific state.

**Routing Layer**: The Router component manages protocol handlers and routes incoming connections to the correct protocol. It also handles protocol registration and spawning of protocol tasks.

### Protocol Composition

One of Iroh's strengths is its support for protocol composition. Instead of writing networking code from scratch, you can use pre-existing protocols built on Iroh:

**iroh-blobs**: A BLAKE3-based content-addressed blob transfer system that scales from kilobytes to terabytes. Blobs are identified by their cryptographic hash, making them tamper-evident and efficient to transfer. The protocol handles chunking, deduplication, and efficient transfer of large files.

**iroh-gossip**: A publish-subscribe overlay network that scales to thousands of nodes while requiring only resources that an average phone can handle. This protocol enables efficient distribution of updates across a network of peers, making it ideal for real-time collaboration, distributed state updates, and content distribution.

**iroh-docs**: An eventually-consistent key-value store of iroh-blobs blobs. This provides a simple API for storing and retrieving data, with automatic synchronization across peers. The protocol handles conflict resolution, replication, and consistency guarantees.

This composability means you can build complex distributed systems by combining these protocols without implementing any networking code yourself.

## Key Features

Iroh provides a comprehensive set of features that make it a powerful networking stack for Rust applications. Let's explore the six key features in detail.

### 1. Dial-by-Public-Key

The dial-by-public-key model is the foundation of Iroh's networking. Instead of IP addresses, you connect to cryptographic identities. Each endpoint has a unique public key that serves as its network address. This model eliminates many of the problems associated with IP-based networking:

- **No IP address configuration**: Users don't need to know or configure IP addresses
- **NAT traversal**: Iroh automatically handles NAT traversal through hole punching
- **Firewall-friendly**: Connections work through firewalls without port forwarding
- **Security**: Public keys provide strong authentication and encryption
- **Reliability**: Connections are maintained even when IP addresses change

The API is simple and intuitive. To create an endpoint, you use `Endpoint::bind()`. To connect to another endpoint, you use `endpoint.connect()`, passing the remote endpoint's public key. That's it. Iroh handles all the complexity.

### 2. Automatic Hole Punching

Iroh automatically attempts to establish direct connections between endpoints using hole punching techniques. When two endpoints want to connect, Iroh exchanges information about their network addresses and attempts to punch holes through NAT devices and firewalls.

The hole punching process works by having both endpoints contact a rendezvous server (which can be a relay server or a dedicated rendezvous service). The rendezvous server exchanges information about each endpoint's network addresses. Once both endpoints have this information, they can attempt to establish a direct connection.

If the direct connection succeeds, data flows directly between the endpoints. If it fails, Iroh falls back to relay servers. This hybrid approach ensures that connections are established reliably while still preferring direct connections when possible.

The hole punching implementation is robust and handles various network conditions, including symmetric NATs, port-restricted NATs, and other challenging network topologies.

### 3. Relay Fallback

When direct connections aren't possible, Iroh seamlessly falls back to its ecosystem of public relay servers. These relays temporarily route encrypted traffic until a direct connection is established. Once a direct connection is possible, the relay steps back and data flows directly between endpoints.

The relay system is designed for performance and reliability. Iroh continuously measures connection quality and switches between relays to ensure optimal performance. The relay servers are deployed globally and can handle high volumes of traffic.

The relay also supports multiple access control modes, allowing you to configure who can use your relay server. You can allow everyone, use allowlists or denylists of endpoint IDs, require shared tokens, or integrate with external authentication services via HTTP callouts.

### 4. Connection Optimization

Iroh continuously monitors connection quality and optimizes connections in real-time. It measures metrics like latency, bandwidth, and packet loss to determine the best connection for each use case.

The connection optimizer automatically switches between direct connections, relays, and different relay servers based on performance. It also manages connection pools, keeping multiple connections open to the same endpoint and selecting the best one for each operation.

This optimization happens transparently in the background. Your application doesn't need to be aware of connection changes or manually switch between connections. Iroh ensures that you always get the best possible connection.

### 5. Protocol Composition

Iroh's protocol composition feature allows you to build complex distributed systems by combining pre-existing protocols. Instead of implementing networking code from scratch, you can use protocols like iroh-blobs, iroh-gossip, and iroh-docs.

Each protocol is identified by an ALPN string and implements the `ProtocolHandler` trait. The protocol layer routes incoming connections to the appropriate handler and manages protocol-specific state. This makes it easy to add new protocols or use existing ones.

The protocol composition model is particularly powerful for building distributed systems. You can combine multiple protocols on the same connection, use protocols as building blocks for more complex protocols, and create protocol hierarchies that leverage the strengths of each protocol.

### 6. Rust Ecosystem Integration

Iroh is deeply integrated with the Rust ecosystem. It uses modern Rust features like async/await, tokio for async runtime, and provides comprehensive documentation and examples.

The crate is well-documented with inline documentation, examples, and a comprehensive documentation site. The examples directory contains a variety of examples that demonstrate different use cases, from simple echo protocols to complex transfer protocols.

Iroh also provides FFI bindings for use from other languages through the iroh-ffi repository. This allows you to use Iroh from languages like Python, JavaScript, and C, expanding its reach beyond the Rust ecosystem.

## Technology Stack and Ecosystem

Iroh's technology stack is built on modern, well-maintained Rust crates and protocols. Understanding this stack helps you make informed decisions about when and how to use Iroh in your projects.

### Core Rust Crates

**iroh**: The main library crate that provides the primary API for networking. This is the crate you'll use most frequently. It includes endpoint creation, connection management, protocol routing, and automatic connection optimization.

**iroh-relay**: The relay client and server implementation. This crate provides the infrastructure for public relay servers and can also be used to run your own relays. It includes support for multiple access control modes and can be configured via TOML configuration files.

**iroh-base**: Common types and utilities used across the Iroh ecosystem. This includes `EndpointId`, `RelayUrl`, and other foundational data structures. The base crate is designed to be minimal and focused, providing only the types and utilities that are shared between components.

**iroh-dns-server**: A DNS server implementation that provides DNS-over-HTTPS and DNS-over-TCP/UDP services for resolving EndpointIds. It also functions as a pkarr relay, serving signed packets over DNS. This server runs at `dns.iroh.link` and is essential for the public DNS/Pkarr address lookup service.

### Protocol Crates

**iroh-blobs**: A BLAKE3-based content-addressed blob transfer system. Blobs are identified by their cryptographic hash, making them tamper-evident and efficient to transfer. The protocol handles chunking, deduplication, and efficient transfer of large files. It's designed to scale from kilobytes to terabytes and can be used for file sharing, content distribution, and distributed storage.

**iroh-gossip**: A publish-subscribe overlay network that scales to thousands of nodes while requiring only resources that an average phone can handle. This protocol enables efficient distribution of updates across a network of peers. It's ideal for real-time collaboration, distributed state updates, and content distribution. The protocol uses efficient gossip algorithms to propagate updates with minimal bandwidth.

**iroh-docs**: An eventually-consistent key-value store of iroh-blobs blobs. This provides a simple API for storing and retrieving data, with automatic synchronization across peers. The protocol handles conflict resolution, replication, and consistency guarantees. It's designed for applications that need distributed storage without requiring strong consistency guarantees.

### Infrastructure and Tools

**iroh-docs**: The documentation repository that contains the official documentation for Iroh. This includes user guides, API documentation, tutorials, and examples. The documentation is comprehensive and well-organized, making it easy to get started with Iroh.

**iroh-ffi**: The FFI bindings repository that provides bindings for using Iroh from other languages. This includes bindings for Python, JavaScript, and C, allowing you to integrate Iroh into applications written in these languages. The FFI bindings provide a C-compatible API that can be called from any language with FFI support.

**iroh-experiments**: An experimental repository that contains experimental features and prototypes. This is where new ideas are tested and validated before being integrated into the main Iroh codebase. If you're interested in contributing to Iroh or exploring cutting-edge features, this repository is worth exploring.

### Protocol Dependencies

**BLAKE3**: Iroh uses BLAKE3 for cryptographic hashing and content addressing. BLAKE3 is a fast, secure, and efficient hash function that's well-suited for cryptographic applications. It provides a hash function, a keyed hash function, and a MAC (message authentication code). The BLAKE3 specification is available at https://github.com/BLAKE3-team/BLAKE3.

**QUIC**: Iroh uses QUIC as its transport protocol. QUIC is a modern transport protocol that provides authenticated encryption, concurrent streams with stream priorities, and automatic avoidance of head-of-line blocking. QUIC is built on UDP and is designed to be more efficient and reliable than TCP. The QUIC protocol is documented at https://en.wikipedia.org/wiki/QUIC.

**noq**: Iroh uses the noq library to establish QUIC connections. noq is a Rust implementation of QUIC that provides a clean, idiomatic API for working with QUIC. It handles the complexity of QUIC protocol implementation and provides a simple interface for creating and managing connections.

### Infrastructure Services

**dns.iroh.link**: The DNS server that provides DNS-over-HTTPS and DNS-over-TCP/UDP services for resolving EndpointIds. This service is essential for the public DNS/Pkarr address lookup system. It allows clients to resolve EndpointIds to their public keys without needing to know the underlying network topology.

**perf.iroh.computer**: The performance dashboard that provides real-time metrics on Iroh's relay servers and connection quality. This dashboard shows metrics like connection latency, bandwidth, and packet loss, helping users understand how Iroh is performing in different network conditions.

**iroh.computer**: The official website for Iroh, providing documentation, examples, and resources. The website includes links to the documentation, examples, and other resources that help users get started with Iroh.

**n0.computer**: The website for n0, the team that develops Iroh. This website provides information about the team, their projects, and their philosophy. It's a good resource for learning more about the people behind Iroh and their vision for decentralized infrastructure.

## Getting Started

Getting started with Iroh is straightforward. The library is well-documented, and there are plenty of examples to help you get up and running quickly.

### Installation

To use Iroh in your Rust project, add it to your `Cargo.toml` file:

```toml
[dependencies]
iroh = "0.95"
```

You can also enable the `test-utils` feature for integration testing:

```toml
[dependencies]
iroh = { version = "0.95", features = ["test-utils"] }
```

### Basic Echo Example

The simplest way to understand Iroh is to look at the echo example. This example implements a simple protocol that echoes any data sent to it back to the sender.

On the accepting side, you create an endpoint and register a protocol handler:

```rust
use iroh::{
    Endpoint,
    endpoint::{Connection, presets},
    protocol::{AcceptError, ProtocolHandler, Router},
};

const ALPN: &[u8] = b"iroh-example/echo/0";

#[tokio::main]
async fn main() -> Result<()> {
    let endpoint = Endpoint::bind(presets::N0).await?;
    let router = Router::builder(endpoint)
        .accept(ALPN.to_vec(), Arc::new(Echo))
        .spawn()
        .await?;

    router.endpoint().online().await;
    router.shutdown().await.anyerr()?;
    Ok(())
}

#[derive(Debug, Clone)]
struct Echo;

impl ProtocolHandler for Echo {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        let (mut send, mut recv) = connection.accept_bi().await?;
        let bytes_sent = tokio::io::copy(&mut recv, &mut send).await?;
        send.finish()?;
        connection.closed().await;
        Ok(())
    }
}
```

On the connecting side, you create an endpoint and connect to the accepting endpoint:

```rust
use iroh::{Endpoint, EndpointAddr, endpoint::presets};

const ALPN: &[u8] = b"iroh-example/echo/0";

#[tokio::main]
async fn main() -> Result<()> {
    let endpoint = Endpoint::bind(presets::N0).await?;
    let conn = endpoint.connect(addr, ALPN).await?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(b"Hello, world!").await?;
    send.finish()?;
    let response = recv.read_to_end(1000).await?;
    assert_eq!(&response, b"Hello, world!");
    conn.close(0u32.into(), b"bye!");
    endpoint.close().await;
    Ok(())
}
```

This example demonstrates the core concepts of Iroh: creating endpoints, connecting by public key, opening streams, and handling data transfer. The full example code with more comments can be found in the [`echo.rs`](https://github.com/n0-computer/iroh-examples) example.

### Documentation

Comprehensive documentation is available at https://docs.rs/iroh. The documentation includes API reference, guides, and examples. The official documentation site at https://iroh.computer/docs provides additional resources, tutorials, and examples.

### Examples

The Iroh repository includes a variety of examples in the `iroh/examples` directory. These examples demonstrate different use cases, including:

- [`echo.rs`](https://github.com/n0-computer/iroh-examples): A simple echo protocol
- [`connect.rs`](https://github.com/n0-computer/iroh-examples): Basic connection example
- [`listen.rs`](https://github.com/n0-computer/iroh-examples): Listening for connections
- [`transfer.rs`](https://github.com/n0-computer/iroh-examples): File transfer example
- [`remote-info.rs`](https://github.com/n0-computer/iroh-examples): Getting information about remote endpoints

These examples are a great way to learn how to use Iroh and can serve as templates for your own applications.

## Use Cases and Applications

Iroh's dial-by-public-key model and automatic connection management make it ideal for a wide range of applications and use cases. Let's explore some of the most common applications.

### Distributed Systems

Iroh is well-suited for building distributed systems that require reliable, direct connections between nodes. Traditional distributed systems often rely on IP addresses and complex networking code to establish connections. Iroh simplifies this by providing a clean, high-level API for connection management.

Applications like distributed databases, distributed file systems, and peer-to-peer networks can benefit from Iroh's automatic NAT traversal and relay fallback. These systems need to maintain connections between nodes in different network environments, and Iroh handles this complexity transparently.

The protocol composition feature is particularly useful for distributed systems. You can combine protocols like iroh-blobs and iroh-docs to build complex distributed systems without implementing any networking code yourself.

### Real-Time Collaboration

Real-time collaboration tools like code editors, design tools, and document editors require reliable, low-latency connections between users. Traditional approaches often require users to configure firewalls or use centralized servers for signaling.

Iroh enables real-time collaboration by providing direct connections between users. When two users want to collaborate, they connect using their public keys, and Iroh establishes a direct connection. This connection can be used to synchronize changes, share cursors, and exchange real-time updates.

The hole punching and relay fallback ensure that connections are established reliably even in restrictive network environments. Users don't need to configure anything or understand network topology.

### P2P Networks

Peer-to-peer networks are a natural fit for Iroh. P2P networks require direct connections between peers, and Iroh's dial-by-public-key model is perfect for this use case.

Applications like file sharing networks, content distribution networks, and decentralized social networks can benefit from Iroh's automatic connection management. Peers can connect to each other using their public keys, and Iroh handles all the complexity of NAT traversal and relay fallback.

The iroh-blobs protocol is particularly useful for P2P file sharing. It provides content-addressed blob transfer that scales from kilobytes to terabytes, making it ideal for sharing large files and content.

### IoT Applications

Internet of Things applications often face challenges with network connectivity. Devices are often behind NAT, have limited resources, and need to communicate with each other reliably.

Iroh's automatic NAT traversal and relay fallback make it ideal for IoT applications. Devices can connect to each other using their public keys, and Iroh handles all the complexity of establishing connections.

The protocol composition feature allows you to build complex IoT applications by combining protocols like iroh-gossip for distributed updates and iroh-docs for distributed state management.

### Decentralized Applications

Decentralized applications (dApps) often require peer-to-peer communication without relying on centralized servers. Iroh provides the networking infrastructure for these applications.

Applications like decentralized social networks, decentralized storage, and decentralized marketplaces can benefit from Iroh's direct connections and automatic connection management. Users connect to each other using their public keys, and Iroh ensures reliable communication.

The relay fallback ensures that connections are established even when direct connections aren't possible. This makes decentralized applications more resilient and accessible.

## Conclusion

Iroh represents a significant advancement in networking for Rust applications. By introducing a dial-by-public-key model, it eliminates many of the complexities associated with IP-based networking. The automatic hole punching, relay fallback, and connection optimization make it incredibly easy to build reliable, distributed systems.

The modular architecture and protocol composition feature make Iroh flexible and extensible. You can use the core library for basic networking, or combine it with protocols like iroh-blobs, iroh-gossip, and iroh-docs to build complex distributed systems without implementing any networking code yourself.

Iroh is built on modern, well-maintained Rust crates and protocols, including QUIC, BLAKE3, and noq. This ensures that it's performant, secure, and future-proof. The comprehensive documentation and examples make it easy to get started, and the active development community ensures that it will continue to evolve and improve.

Looking ahead, Iroh has significant potential for growth and adoption. As more developers discover its benefits, we can expect to see more applications built on top of it. The protocol composition feature will enable new use cases and applications that were previously difficult or impossible to build.

If you're building a distributed system, real-time collaboration tool, P2P network, or any application that requires reliable connections between endpoints, Iroh is worth considering. Its simplicity, reliability, and performance make it a compelling choice for modern networking in Rust.

To learn more about Iroh, visit the official website at https://iroh.computer, read the documentation at https://docs.rs/iroh, or check out the examples at https://github.com/n0-computer/iroh-examples. Join the community on Discord at https://discord.com/invite/DpmJgtU7cW or follow the YouTube channel at https://www.youtube.com/@n0computer for updates and tutorials.

![Iroh Architecture](/assets/img/diagrams/iroh/iroh-architecture.svg)

The architecture diagram above illustrates the layered design of Iroh, showing how the core components work together to provide a modular networking stack.

![Dial-by-Public-Key Model](/assets/img/diagrams/iroh/iroh-key-dialing-model.svg)

This diagram demonstrates the dial-by-public-key model, showing how endpoints connect using cryptographic identities instead of IP addresses.

![Key Features](/assets/img/diagrams/iroh/iroh-features.svg)

The features diagram highlights the six key features of Iroh: dial-by-public-key, automatic hole punching, relay fallback, connection optimization, protocol composition, and Rust ecosystem integration.

![Technology Stack](/assets/img/diagrams/iroh/iroh-tech-stack.svg)

This diagram shows the technology stack and ecosystem of Iroh, including core crates, protocol crates, infrastructure services, and dependencies.
