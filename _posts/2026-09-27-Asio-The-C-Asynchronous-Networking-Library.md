---
layout: post
title: "Asio: The C++ Asynchronous Networking Library and How It Works"
description: "Asio is the header-only C++ networking library behind Boost.Asio. A source-level tour of chriskohlhoff/asio: the I/O objects, the completion and coroutine model, executors and strands, the scheduler, the per-platform demultiplexers from epoll to IOCP to io_uring, and the proactor design that ties them together."
date: 2026-09-27
header-img: "img/post-bg.jpg"
permalink: /Asio-The-C-Asynchronous-Networking-Library/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/asio/chriskohlhoff-asio-architecture.svg
tags:
  - C++
  - Networking
  - Asio
  - Concurrency
  - Open Source
categories: [C++, Open Source]
keywords: "asio, chriskohlhoff asio, C++ networking library, Boost.Asio, asio architecture, io_context, proactor model, asio executors, strand, asio coroutines, awaitable, co_spawn, use_awaitable, async_result, epoll reactor, kqueue, IOCP, io_uring, header-only C++ library, asynchronous sockets, strand serialization"
author: "PyShine"
---

Writing a networked application in C++ used to mean choosing a poison. Block on `recv()` and one thread per connection caps you at a few thousand sockets. Wrap `epoll` on Linux and your server now needs a rewrite to run on macOS or Windows. Sprinkle callbacks everywhere and the happy path disappears into a cascade of handler functions. [Asio](https://github.com/chriskohlhoff/asio), the C++ library Christopher M. Kohlhoff started in 2003, is the answer the ecosystem converged on: a cross-platform library for network and low-level I/O that gives you one consistent asynchronous model, and does it as a collection of headers you drop into almost any C++11 codebase.

The library that ships as Asio 1.38.2 is also, via an automated conversion, the `boost::asio` that ships inside Boost - both come from this same repository. That makes the source worth reading regardless of which flavor you use: it is the reference implementation of the proactor style that C++ networking has been converging on for two decades. This post walks through that machinery as it is actually implemented in the repository - the I/O objects, the completion and coroutine machinery, the executor framework, the scheduler, and the platform demultiplexers that range from `epoll` to Windows IOCP to `io_uring`.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/asio/chriskohlhoff-asio-overview-architecture.svg" alt="Architecture overview of the Asio repository" style="max-width:100%;height:auto;" />
</div>

*High-level overview: the master `asio.hpp` header exposes the user-facing I/O objects; their asynchronous operations complete through the coroutine and completion-token model, run on executors and strands, which queue handlers on the scheduler and the per-platform reactors underneath.*

Reading the overview from top to bottom mirrors how a program actually flows. You include one header, construct I/O objects such as sockets and timers, and launch coroutines that `co_await` on asynchronous operations. Those operations are executed by executors, which hand completed handlers to a scheduler - the internal completion queue that every `io_context` owns. At the bottom, the scheduler talks to a reactor matched to your operating system, and that reactor performs the real non-blocking I/O through portable syscall wrappers. The sections below zoom into each of those layers using the real files in the repository.

## Why You Need This

The problem Asio solves is that the operating systems we program against expose completely different ways to wait for I/O efficiently. Linux has `epoll` (and now `io_uring`), BSD and macOS have `kqueue`, Windows has I/O completion ports, and every one of them has a different API shape, a different way of reporting errors, and a different set of edge cases. On top of that sits the harder conceptual problem: asynchronous code needs somewhere to *come back to* when an operation finishes, and naive answers - one thread per connection, or deeply nested callbacks - either collapse under load or collapse under complexity.

Asio's answer has three parts, each visible in the repository as a concrete subsystem. First, a portable object model: you work with `ip::tcp::socket`, `steady_timer`, `ssl::stream`, and friends, and the platform differences are absorbed deep in the `detail/` headers so the same code compiles and performs well on Linux, macOS, Windows, and FreeBSD. Second, a completion model: every operation starts with `async_*`, and you choose how to receive the result - a callback, a `std::future`, or, since C++20, an `awaitable` coroutine that lets you write asynchronous code that reads like synchronous code. Third, an executor framework that decides *where* and *by whom* each completion handler runs, with `strand` as the tool for serializing access to shared state without locks.

It helps to know where the boundary of the project sits. This repository is standalone Asio: everything lives in namespace `asio::`, the master header is `asio.hpp`, and for C++11 and later it needs no Boost libraries at all. The official site keeps a page explaining the [differences from Boost.Asio](https://think-async.com/Asio/AsioAndBoostAsio.html) - essentially namespace, header names, and macro prefixes - and the conversion from this tree into Boost format is automated by the `boostify.pl` script you can find in the repository root. If your project already uses Boost, you get the same engine; if you want zero dependencies beyond the standard library, this tree is the one to read.

## How It Works

The diagram below maps the real subsystems of the repository and how they connect, from the public headers a programmer touches down to the syscall wrappers and the platform demultiplexers.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/asio/chriskohlhoff-asio-architecture.svg" alt="Detailed architecture of Asio, from public headers to platform demultiplexers" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the public API surface, the completion and coroutine model, executors and execution contexts, the I/O objects, the detail layer of services and the scheduler, the per-platform demultiplexers, and the examples, tests, and documentation that surround them.*

### Understanding the Architecture

**The public API surface.** Everything starts at `include/asio.hpp`, the master header that pulls in the entire library - a deliberate convenience modeled on how Boost distributes Asio. The header tree under `include/asio/` is organized by concern: `ip/` for TCP, UDP, ICMP, and name resolution; `ssl/` for TLS built on OpenSSL; `posix/` and `windows/` for platform-specific descriptors and handles; `local/` for Unix domain sockets; `generic/` for protocol-agnostic raw sockets; and a `ts/` directory that provides the flavor of the library matching the C++ Networking Technical Specification. Most of the library is templates, which is exactly why it can stay header-only: there is little to precompile, and your compiler instantiates only what you use.

**I/O objects.** The classes you construct directly - `ip::tcp::socket`, `basic_socket_acceptor`, `basic_waitable_timer`, `basic_stream_file`, `signal_set` - are deliberately thin. Each is a front end that aggregates its per-context service through the service registry in `detail/service_registry.hpp`. The socket you hold is a lightweight handle; the machinery that knows how to talk to the operating system lives in services like `detail/reactive_socket_service.hpp`. That separation is why an `io_context` can host thousands of sockets cheaply, and why the library can swap the entire I/O strategy underneath without changing the object model.

**Buffers.** Read and write operations never copy through the library. Instead you pass a `mutable_buffer` or `const_buffer` view - created by the `asio::buffer()` function from `include/asio/buffer.hpp` - or a sequence of them for scatter-gather I/O. The diagram shows the buffer header feeding the socket layer, because every read and write call ultimately resolves to these views plus a syscall. This design is one of the quiet reasons for Asio's performance: zero-copy in the common path, with optional registered buffers for hot loops.

**The completion and coroutine model.** The trait at the center is `async_result` in `include/asio/async_result.hpp`: an asynchronous operation does not hardcode how its result is delivered - it forwards to whatever completion token you passed. That single customization point is what lets the same `async_read_some` accept a lambda, `asio::use_future` (yielding a `std::future`), or `asio::use_awaitable` (yielding an object you can `co_await`). The coroutine side is `awaitable` plus `co_spawn`, which launches a coroutine onto an executor; `include/asio/experimental/` extends this with channels, promises, and parallel groups for structured concurrency. Cancellation is a first-class citizen too: `cancellation_signal` and its associated slot let in-flight operations be aborted cooperatively. For older codebases there is also the stackful `spawn()` interface, which needs the optional Boost.Coroutine dependency.

**Executors and execution contexts.** Two execution contexts exist in the library: `io_context` and `thread_pool` (both in `include/asio/`), and both drive their own scheduler instance. An executor is a lightweight, copyable token that says "run completions here" - and Asio implements the concepts of the proposed standard executors library in `include/asio/execution/`, with `any_io_executor` as the type-erased executor you get by default from any I/O object. The `strand` adapter wraps another executor and guarantees that handlers submitted through it never run concurrently, which converts the classic multithreaded-server problem of shared-state locking into a simple choice of executor. Notice in the diagram that `co_spawn`, strands, and the I/O objects all converge on this layer: the executor is the glue between application logic and the machinery below.

**The detail layer: services and the scheduler.** Every `io_context` owns a `detail/service_registry` - a collection of typed services that implement I/O strategies - and a `detail/scheduler`, the beating heart of the library. The scheduler is the completion queue: operations that are ready to run are queued as reference-counted handler objects (allocated through `detail/executor_op.hpp`-style machinery), and your calls to `io_context.run()` pop and invoke them. It distinguishes `dispatch` (run now if you are already on the context), `post` (queue for later), and `defer` (queue, but opportunistically run inline) - the three functions the library itself uses to move work around. The scheduler also polls timer queues, so expired deadlines become ordinary completions.

**Platform demultiplexers.** Below the scheduler sit the reactors: `detail/epoll_reactor.hpp` on Linux, `detail/kqueue_reactor.hpp` on BSD and macOS, a `select_reactor` fallback, and `detail/io_uring_service.hpp` for Linux's modern asynchronous interface. On Windows there is something subtly different: `detail/win_iocp_io_context.hpp` - IOCP is a *native* operating-system proactor, so on Windows the completion queue and the I/O dispatch are the same mechanism, while on reactor-based systems Asio emulates proactor semantics by performing non-blocking operations and queuing the results itself. All of the platforms share `detail/socket_ops.hpp`, the portable layer that wraps `send`, `recv`, `getaddrinfo` and friends and normalizes error codes. This is the layer you never see but always depend on - it is why the same socket code produces correct behavior for edge-triggered epoll, kqueue readiness lists, and overlapped Windows I/O alike.

**A completion in flight.** Follow one `co_await socket.async_read_some(...)` from the repository's own echo example. The coroutine suspends and returns control to `co_spawn`; the operation object is allocated and armed with the reactor - on Linux, the socket's descriptor is registered with `epoll` for readability. Your thread returns into `io_context.run()`, which may be doing nothing more than blocking in `epoll_wait`. Data arrives; the reactor marks the operation ready; the scheduler queues its completion handler; the handler resumes the coroutine at the `co_await` point, and `use_awaitable`'s machinery makes the resumed value the read result. Nothing blocked, no extra thread was created for the connection, and the entire round trip went through the boxes in the diagram exactly once.

## Advantages

- **One model, every platform.** The reactor layer means your networking code is genuinely portable without being the lowest common denominator - you get epoll-grade efficiency on Linux, kqueue on macOS, IOCP on Windows, and io_uring where available, all behind one API.
- **The proactor scale story.** Completions are queued to executors rather than threads being blocked per connection, so a single thread can service thousands of sockets, and adding threads to `io_context::run()` scales the handler execution rather than the connection count.
- **Composability through completion tokens.** Because `async_result` decouples operations from result delivery, the same operation supports callbacks, futures, coroutines, and library-level compositions like `experimental::co_composed` - you adopt coroutines incrementally instead of rewriting.
- **Concurrency control as a type, not a convention.** `strand` gives you serialized execution of handlers with no locks in your code, and executors let you pin specific work to specific threads - designs that are enforced by the library rather than by code review.
- **Zero-dependency pragmatism.** Standalone Asio is header-only and needs nothing but a C++11 compiler for most uses; OpenSSL is only required if you want TLS, and Boost only if you want the stackful `spawn()` interface.
- **Two decades of battle-testing.** Development has run continuously since 2003, the codebase is the upstream of Boost.Asio, and the repository carries an extensive unit and functional test suite in `src/tests/` plus recipes from simple echo servers to full HTTP clients in `src/examples/`.

## Benefits

- **Dramatically less code for servers.** The coroutine interface turns connection handling into plain functions; the repository's own C++20 echo server fits in a few dozen lines, including graceful shutdown via a `signal_set`.
- **Debuggability that asynchronous code rarely gets.** Compile with `ASIO_ENABLE_HANDLER_TRACKING` and Asio logs every operation's lifecycle; the `src/tools/` directory ships Perl scripts (`handlerviz.pl`, `handlertree.pl`, `handlerlive.pl`) that turn those logs into visual graphs of your handler relationships.
- **Skills that transfer.** Learn the Asio object and executor model once and it applies across platforms and across both library flavors - and because a subset of it shaped the C++ Networking TS, the concepts themselves are portable to the standard library's future.
- **Permissive licensing.** The whole tree is distributed under the Boost Software License, so it can be used in commercial, closed-source products without obligations beyond keeping the copyright notice.
- **Proven at scale.** The project's own list of [applications using Asio](https://think-async.com/Asio/WhoIsUsingAsio.html) ranges from WebSocket libraries like WebSocket++ to embedded REST frameworks like Restbed and high-throughput log aggregation systems.
- **Faster release cadence than Boost.** Standalone Asio ships new features and fixes on its own schedule; Boost.Asio picks them up in subsequent Boost releases.

## Usage

The fastest way in is the header-only path. Clone the repository and compile one of its own examples - the C++20 coroutine echo server lives at `src/examples/cpp20/coroutines/echo_server.cpp` - pointing the include path at the repository's `include/` directory:

```bash
git clone https://github.com/chriskohlhoff/asio.git asio
cd asio
g++ -std=c++20 -I include src/examples/cpp20/coroutines/echo_server.cpp -o echo_server -lpthread
./echo_server
```

The heart of that example shows how small idiomatic Asio has become (condensed from the file above):

```cpp
awaitable<void> echo(tcp::socket socket)
{
  char data[1024];
  for (;;)
  {
    std::size_t n = co_await socket.async_read_some(asio::buffer(data), use_awaitable);
    co_await async_write(socket, asio::buffer(data, n), use_awaitable);
  }
}
```

A `listener` coroutine accepts connections in a loop and hands each one to `co_spawn`, and `main` constructs an `asio::io_context` with a single thread, waits on a `signal_set` for SIGINT/SIGTERM to stop cleanly, and calls `io_context.run()`. No callbacks, no state machines - just functions that `co_await`.

To build the repository's full test and example suite from a git checkout on Linux or macOS, use the autotools files at the repository root (`autogen.sh` requires GNU autoconf and automake):

```bash
./autogen.sh
./configure
make
make check
```

`./configure` defaults to standalone mode - no Boost required - and accepts `--with-boost=DIR` if you want the Boost-dependent paths; `make check` builds and runs the unit tests. On Windows with MSVC, the repository ships `src/Makefile.msc`; set `BOOSTDIR` if needed, then run `nmake -f Makefile.msc` from the `src` directory.

For daily development, most projects just add `include/` to their include path and compile. Two options are worth knowing. If you prefer shorter link times or want to ship a compiled library, define `ASIO_SEPARATE_COMPILATION` project-wide and put `#include <asio/impl/src.hpp>` in exactly one translation unit (add `#include <asio/ssl/impl/src.hpp>` as well if you use TLS). And if you use TLS, download and link OpenSSL, then include `asio/ssl.hpp` - the `ssl::stream` template wraps any stream object, sockets included, which is why the diagram shows it plugging into the socket layer.

## Conclusion

Asio endures because its layering is exactly right: a thin, expressive object model on top; a completion machinery that has absorbed every style C++ has thrown at it, from callbacks to futures to coroutines; an executor framework that makes concurrency policy explicit; and underneath it all, a per-platform engine that quietly does the epoll, kqueue, IOCP, and io_uring work that nobody wants to maintain by hand. Reading the repository makes that layering concrete - one master header, a forest of focused templates, and a `detail/` directory where the operating systems meet the proactor. If you write networked C++, clone it, compile the echo server, and trace one completion through the architecture above; the source remains the best documentation of why modern C++ networking looks the way it does.

**Links:**

- Repository: [https://github.com/chriskohlhoff/asio](https://github.com/chriskohlhoff/asio)
- Website: [https://think-async.com](https://think-async.com)
- Documentation: [https://think-async.com/Asio/Documentation.html](https://think-async.com/Asio/Documentation.html)
- Download: [https://think-async.com/Asio/Download.html](https://think-async.com/Asio/Download.html)
- Asio and Boost.Asio: [https://think-async.com/Asio/AsioAndBoostAsio.html](https://think-async.com/Asio/AsioAndBoostAsio.html)
- Applications using Asio: [https://think-async.com/Asio/WhoIsUsingAsio.html](https://think-async.com/Asio/WhoIsUsingAsio.html)
