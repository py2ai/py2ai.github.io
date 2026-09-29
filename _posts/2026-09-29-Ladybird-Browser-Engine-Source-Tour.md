---
layout: post
title: "Ladybird: A Source Tour of the Independent Web Engine - Inside LadybirdBrowser/ladybird"
description: "A guided source-tour of the LadybirdBrowser/ladybird repository, walking the real code behind the only truly independent web browser engine: the LibWeb content layer, the LibJS bytecode interpreter with its Rust parser and native Flap interpreter, the Rust layout and painting pipeline, and the sandboxed multi-process architecture. Learn how a browser engine is built from scratch by reading the actual directories and files that power it."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Ladybird-Browser-Engine-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ladybird/ladybirdbrowser-ladybird-architecture.svg
tags:
  - Ladybird
  - Web Engine
  - C++
  - Rust
categories: [AI, Open Source]
keywords: "ladybird browser, ladybird source code, independent browser engine, LibWeb engine, LibJS javascript engine, web browser architecture, multi-process browser, browser rendering pipeline, rust layout engine, C++ browser engine, display list rendering, web standards implementation, browser internals, open source browser, source tour"
author: "PyShine"
---

Nearly everything you read on the web reaches you through one of three engines — Chromium's Blink, WebKit, or Gecko — and all three trace their lineage back decades. [LadybirdBrowser/ladybird](https://github.com/LadybirdBrowser/ladybird) is the exception: a truly independent web browser implementing the modern web from first principles, with its own rendering engine and its own JavaScript engine, and with more than 60,000 GitHub stars watching it do something most people assumed was no longer possible. The README states the ambition plainly: a complete, usable browser for the modern web, built on a novel engine grounded in web standards.

The repository is a C++23 monorepo with a rapidly growing Rust core. `Libraries/` holds the engine libraries the README names — LibWeb (web rendering), LibJS (JavaScript), LibWasm, LibGfx (2D graphics and image decoding), LibCrypto/LibTLS, LibUnicode, LibMedia, LibCore, and LibIPC — while `Services/` holds the out-of-process helpers that give Ladybird its security posture: a `WebContent` renderer per tab, a `RequestServer` for all networking, an `ImageDecoder`, and a `Compositor`. The UI layer lives in `UI/` (Qt everywhere, AppKit on macOS, a native Android UI), and `Documentation/` is unusually good — `ProcessArchitecture.md` and `LibWebFromLoadingToPainting.md` are the maps this tour leans on. Everything is BSD 2-clause licensed.

The source is worth a tour because Ladybird is currently the only place where you can read a complete web engine written for modern hardware, with spec-mirroring code you can actually hold in your head. Blink and Gecko teach you archaeology; LibWeb teaches you the web platform. The project is also at a fascinating inflection point: layout, painting, and parts of the JS front end are being ported to Rust, so the tree itself documents how an engine migrates languages without stopping. Every claim below points at a real path in the master branch, so you can follow along.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ladybird/ladybirdbrowser-ladybird-overview-architecture.svg" alt="Architecture overview of the LadybirdBrowser/ladybird repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Ladybird source layout: the Qt/AppKit UI and LibWebView form the browser application, each tab gets a WebContent renderer process hosting the LibWeb engine, LibJS executes scripts, the Rust ports drive layout and painting into LibGfx, and sandboxed helper services handle networking, image decoding, and frame composition over LibIPC.*

Reading the overview from left to right: the browser application in `UI/Qt` sits on `Libraries/LibWebView`, which spawns one `Services/WebContent` process per tab and speaks to it through `Libraries/LibIPC`. Inside WebContent, the `Libraries/LibWeb` engine owns the DOM, CSS, and rendering, calling into `Libraries/LibJS` for scripts and into the Rust layout-and-paint core under `Libraries/LibWeb/Rust/src` before handing pixels to `Libraries/LibGfx`. On the right, WebContent delegates everything dangerous outward: `Services/RequestServer` for the network, `Services/ImageDecoder` for untrusted encoded images, and `Services/Compositor` for presenting frames.

## Why You Need This

The first problem Ladybird solves for you is comprehension. Browser engines are the most complex software most developers interact with daily, yet the big three are effectively unreadable at whole-system scale. LibWeb mirrors the specifications it implements: the HTML parser in `Libraries/LibWeb/HTML/Parser/HTMLParser.cpp` follows the spec's tokenizer-parser dance, the fetch algorithm in `Libraries/LibWeb/Fetch/Fetching/Fetching.cpp` follows the fetch standard, and the event loop in `Libraries/LibWeb/HTML/EventLoop/EventLoop.cpp` follows the HTML event loop model. If you have ever wanted to actually understand what happens between "typing a URL" and "pixels on screen", this is the codebase where that journey is short enough to complete.

The second problem is web standards literacy. Reading specs is dry; reading executable code that implements them is not. `Documentation/LibWebFromLoadingToPainting.md` walks the pipeline from resource loading through HTML parsing, CSS cascade, layout, and painting, and each stage lands in a specific, readable file — style computation in `Libraries/LibWeb/CSS/StyleComputer.cpp`, the user-agent stylesheet in `Libraries/LibWeb/CSS/Default.css`, selector matching with its rightmost-first evaluation and per-class rule buckets. For engineers who build on the web platform, this is a ground-truth model of why the platform behaves the way it does.

The third problem is architectural: how do you isolate hostile web content in a desktop application? The README and `Documentation/ProcessArchitecture.md` describe the answer, and `Services/` implements it — each tab gets its own WebContent renderer, image decoding happens in a dedicated `ImageDecoder` process precisely because decoders parse malicious input, and all networking is confined to `RequestServer` so the renderer itself cannot open sockets. The sandbox setup in `Services/RendererSandbox.h` shows isolation as a first-class design constraint rather than an afterthought. These patterns transfer directly to any application that handles untrusted content.

Finally, the source matters if you follow programming-language engineering. The tree is a live experiment in evolving a huge C++ system toward Rust: formatting contexts live in `Libraries/LibWeb/Rust/src/layout/`, paintables and stacking contexts in `Libraries/LibWeb/Rust/src/painting/`, and the C++ side calls over bridges such as `Libraries/LibWeb/Layout/LayoutRustBridge.cpp`. LibJS similarly hosts a Rust parser in `Libraries/LibJS/Rust/src/parser.rs` and a native interpreter-assembly compiler called Flap in `Libraries/LibJS/Flap/`. Watching a real codebase make this migration incrementally, with bridges rather than rewrites, is a masterclass in large-system evolution.

## How It Works

Every page load funnels through the same chain: the UI process spawns a WebContent renderer, the renderer runs LibWeb, and LibWeb parses, styles, lays out, and paints the document while delegating network, image decoding, and compositing to sibling processes over IPC.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ladybird/ladybirdbrowser-ladybird-architecture.svg" alt="Detailed architecture of the LadybirdBrowser/ladybird repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of Ladybird: the Qt app connects over LibIPC to WebContent's ConnectionFromClient and PageHost, LibWeb builds the DOM with the HTML parser, computes style in C++ with StyleComputer, builds boxes and paints through the Rust modules, executes scripts with the LibJS interpreter over bytecode, and delegates to the RequestServer, ImageDecoder, and Compositor services.*

### Understanding the Architecture

**The browser app is a thin shell over a service mesh.** `UI/Qt/main.cpp` boots the application into `UI/Qt/BrowserWindow.cpp`, where each tab owns a `WebContentView` (`UI/Qt/WebContentView.cpp`) — a widget backed not by local state but by a remote renderer process. The client side of that relationship is `Libraries/LibWebView/BrowserProcess.cpp` and the view implementations in `Libraries/LibWebView`; `.ipc` files such as `Services/WebContent/WebContentClient.ipc` define the wire contract that `Libraries/LibIPC` turns into C++ stubs. Across the socket, `Services/WebContent/ConnectionFromClient.cpp` demarshals requests into `Services/WebContent/PageHost.cpp`, which owns the `PageClient` instances hosting the engine's `Web::Page` objects from `Libraries/LibWeb/Page/Page.cpp`.

**The DOM is built by a spec-exact HTML parser.** `Libraries/LibWeb/HTML/Parser/HTMLParser.cpp` drives the state machines of `HTMLTokenizer.cpp` and `HTMLEncodingDetection.cpp`, producing the tree that `Libraries/LibWeb/DOM/Document.cpp` owns. Subresources are requested through `Libraries/LibWeb/Loader/ResourceLoader.cpp`, which routes fetches through `Libraries/LibWeb/Fetch/Fetching/Fetching.cpp` and out to RequestServer. The HTML event loop (`Libraries/LibWeb/HTML/EventLoop/EventLoop.cpp`) sequences tasks, microtasks, and rendering updates, while input and hit-testing live in `Libraries/LibWeb/Page/EventHandler.cpp`.

**Style is computed in C++; layout runs in Rust.** The cascade, selector matching, and custom-property resolution live in `Libraries/LibWeb/CSS/StyleComputer.cpp`, fed by the parser in `Libraries/LibWeb/CSS/Parser/Parser.cpp` and the user-agent rules of `Libraries/LibWeb/CSS/Default.css`. The styled tree then crosses into Rust via `Libraries/LibWeb/Layout/LayoutRustBridge.cpp`, where `Libraries/LibWeb/Rust/src/layout/tree_builder.rs` builds the box tree and the formatting-context engines — `block_formatting_context.rs`, `inline_formatting_context.rs`, plus flex, grid, table, and SVG siblings — resolve used values and build line boxes exactly as the documentation describes.

**Painting is a display-list pipeline ending in a dedicated compositor.** Layout results become paintables in `Libraries/LibWeb/Rust/src/painting/paintable_build.rs`, z-ordering is resolved by `Libraries/LibWeb/Rust/src/painting/stacking_context/mod.rs`, and `Libraries/LibWeb/Painting/PaintingRustBridge.cpp` records the display list that `Libraries/LibGfx/Painter.cpp` rasterizes. Frames then travel to `Services/Compositor`, whose `BackingStoreManager.cpp`, `OpenGLContext.cpp`, and `VSyncScheduler.cpp` present content in step with the display — so a hung renderer cannot freeze the whole window.

**The JS engine is a bytecode interpreter with a Rust front end and native handlers.** Script text arrives as `ClassicScript` (`Libraries/LibWeb/HTML/Scripting/ClassicScript.cpp`) and is parsed by the Rust parser in `Libraries/LibJS/Rust/src/parser.rs`, whose bytecode generator in `LibJS/Rust/src/bytecode/` lowers the AST to an executable carried by `Libraries/LibJS/Bytecode/Executable.cpp`. The main loop in `Libraries/LibJS/Interpreter/Interpreter.cpp` dispatches those instructions through a native entry point that Flap-compiled handlers implement, while `Libraries/LibGC/Heap.cpp` runs the garbage collector behind everything with conservative stack rooting over a block allocator. Flap (`Libraries/LibJS/Flap/src/lib.rs`) is the remarkable piece: a build-time compiler that lowers typed interpreter handler definitions through SSA construction, machine lowering, and register allocation into real dispatch assembly for x86_64 and aarch64.

**Isolation shapes every boundary.** `Services/ImageDecoder/ConnectionFromClient.cpp` decodes untrusted encoded images out-of-process; `Services/RequestServer/ConnectionFromClient.cpp` confines all socket access, with actual transfers handled by libcurl (see `Services/RequestServer/CURL.cpp`); and the WebDriver service (`Services/WebDriver/Client.cpp`) plus the Firefox-protocol DevTools server in `Libraries/LibDevTools` expose the engine to automation without weakening the sandbox. Even WebAssembly compilation of untrusted code is offloaded to `Services/WasmCompiler`.

The end-to-end flow in one breath: you type a URL, `WebContentView` sends a navigate message over LibIPC to the tab's WebContent process, `PageHost` attaches it to a `Page`, `ResourceLoader` fetches the document through RequestServer, the HTML parser grows the DOM, StyleComputer cascades CSS onto every element, the Rust layout engine turns that into boxes and line boxes, the Rust painting engine records a display list, and the Compositor process rasterizes and presents it — while every script runs as Flap-accelerated bytecode inside LibJS's GC-managed heap.

## Advantages

- **A truly independent engine.** No Chromium, WebKit, or Gecko code anywhere in the tree — LibWeb, LibJS, and every support library in `Libraries/` are written from scratch against web standards.
- **Spec-mirroring readability.** Files correspond to concepts you already know — `HTMLParser.cpp`, `StyleComputer.cpp`, `Fetching.cpp`, `EventLoop.cpp` — so the learning curve is architectural, not archaeological.
- **Serious multi-process security.** Per-tab renderers, out-of-process image decoding and networking, sandboxed helpers, and IPC-only communication, all readable end to end.
- **A modern dual-language core.** C++23 for the DOM and platform integration, Rust where memory safety pays most — layout, painting, and the JS parser — joined by explicit FFI bridges.
- **Complete platform UIs.** Qt on Linux, Windows, and *Nixes, native AppKit on macOS, and a native Android UI, all driven through `Libraries/LibWebView`.
- **First-class docs.** `Documentation/ProcessArchitecture.md` and `LibWebFromLoadingToPainting.md` are real maps of the code, written by its authors.

## Benefits

- **You finally understand the browser.** Following one page load — parse, style, layout, paint, composite — builds a mental model that transfers to performance work in any engine.
- **Web standards become concrete.** The HTML tokenizer states, the CSS cascade, fetch, and the event loop are implemented step by step; reading the source doubles as reading the specs with working reference code.
- **Engine-grade patterns to reuse.** The GC in `Libraries/LibGC`, the bytecode pipeline in LibJS, display-list rendering, and the `.ipc` protocols are clean, reusable designs far beyond browsers.
- **Practical tooling.** The `js` REPL exercises the JS engine standalone, the WebDriver service automates real pages, and `Meta/ladybird.py` builds, runs, and debugs the whole stack with one command.
- **A front-row seat to an engine's evolution.** The in-progress Rust migration and the Flap native interpreter are visible in the tree — a rare look at how a mature codebase changes languages incrementally.
- **Permissive licensing.** The 2-clause BSD license lets you read, reuse, and build on the code with minimal friction.

## Usage

Install the prerequisites (Debian/Ubuntu example from `Documentation/BuildInstructionsLadybird.md`; Qt 6.9+, a C++23 compiler, a Rust toolchain, and CMake 3.30+ are required):

```bash
sudo apt install autoconf autoconf-archive automake build-essential ccache cmake curl fonts-liberation2 git glslang-tools libdrm-dev libgl1-mesa-dev libncurses-dev libpulse-dev libtool nasm ninja-build pkg-config python3-venv qt6-base-private-dev qt6-positioning-dev qt6-tools-dev-tools qt6-wayland tar unzip zip
```

Clone and run — the simplest path is the `Meta/ladybird.py` script, which configures the vcpkg dependencies, builds, and launches:

```bash
git clone https://github.com/LadybirdBrowser/ladybird.git
cd ladybird
./Meta/ladybird.py run
```

A debug build, a manual CMake build, and the standalone JavaScript REPL:

```bash
BUILD_PRESET=Debug ./Meta/ladybird.py run

cmake --preset Release -B MyBuildDir
cmake --build --preset Release MyBuildDir
ninja -C MyBuildDir run-ladybird

./Meta/ladybird.py run --no-build js --evaluate 'console.log(1 + 1)'
```

## Conclusion

Ladybird is the rare project whose value is self-evident from its directory listing: a complete, independent web engine where every layer — parser, cascade, layout, paint, JavaScript, IPC, sandboxing — is implemented from scratch in readable, spec-faithful code. It is pre-alpha and honest about it, which is exactly why the source is such a good teacher: no legacy to excavate, only architecture to understand. Clone the repository, open `Documentation/LibWebFromLoadingToPainting.md`, and follow one page load from `HTMLParser.cpp` to `stacking_context/mod.rs`.

Links:

- [LadybirdBrowser/ladybird on GitHub](https://github.com/LadybirdBrowser/ladybird)
- [Ladybird project website](https://ladybird.org)
- [Build instructions](https://github.com/LadybirdBrowser/ladybird/blob/master/Documentation/BuildInstructionsLadybird.md)
- [Process architecture documentation](https://github.com/LadybirdBrowser/ladybird/blob/master/Documentation/ProcessArchitecture.md)
- [From loading to painting](https://github.com/LadybirdBrowser/ladybird/blob/master/Documentation/LibWebFromLoadingToPainting.md)
