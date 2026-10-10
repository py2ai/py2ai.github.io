---
layout: post
title: "VectorCraft: Exact Curve Booleans and 27 ms Renders - Inside storytold/vectorcraft"
description: "A source tour of storytold/vectorcraft: how a pure-Rust clean-room reimplementation of Illustrator gets exact path booleans, 20,000 shapes at 27 ms, journaled commands and an MCP server for agents."
date: 2026-10-10
header-img: "img/post-bg.jpg"
permalink: /vectorcraft/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/vectorcraft/storytold-vectorcraft-overview-architecture.svg
tags: [Rust, Open Source]
categories: [AI, Open Source]
keywords: vectorcraft, rust, illustrator alternative, vector editor, pathfinder, bezier booleans, mcp server, open source
author: "PyShine"
---

Vector editors are where precision goes to die. Ask any tool to subtract one blob from another and half the time you get the infamous "cannot perform operation" dialog, a snapped anchor you never moved, or a file that renders differently the moment you export it. VectorCraft, from the storytold org, is a clean-room reimplementation of the Adobe Illustrator workflow built in pure Rust, and its entire architecture is organized around refusing those failure modes. Version 0.8.0 spans 21 crates under an MIT OR Apache-2.0 license, runs natively on macOS, Windows, Linux and FreeBSD, and compiles the same interface to the browser through WebAssembly. The headline numbers are bold: 20,000 shapes render in about 27 ms at full retina resolution while the interface holds 120 fps.

What makes the repository worth a source tour is not just the speed claim. The layout, tools and panels deliberately mirror Illustrator, so the Pen, Direct Selection, Pathfinder, Smart Guides, Appearance and Layers all behave the way muscle memory expects. Underneath, every tool gesture reduces to a journaled command, every committed document state passes an invariant guard, and every menu item is reachable over an MCP server that lets Claude or another agent draw, edit and export the way a person does. The gallery posters in the repository were built entirely through that command interface, which is the strongest possible proof that the automation layer is not an afterthought.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/vectorcraft/storytold-vectorcraft-overview-architecture.svg" alt="VectorCraft overview architecture: desktop, CLI and web frontends feeding the engine, which mutates the document model, calls exact booleans, and drives rendering, effects, plug-ins and file formats" style="max-width:100%;"></div>

<p><em>Overview: three frontends, one engine, and the crates that do the heavy math, rendering and file work.</em></p>

Reading the overview from left to right: the three frontends contain no editor logic.
[The desktop app](https://github.com/storytold/vectorcraft/blob/main/apps/vectorcraft/src)
hosts the interface and every panel, [the CLI](https://github.com/storytold/vectorcraft/blob/main/apps/vectorcraft-cli/src)
runs documents, batches and benchmarks headlessly, and [the web app](https://github.com/storytold/vectorcraft/blob/main/apps/vectorcraft-web)
reuses the same interface in the browser. They all talk to
[the engine](https://github.com/storytold/vectorcraft/blob/main/crates/engine/src), a Session facade whose command set is the only way anything changes. The engine mutates
[the document model](https://github.com/storytold/vectorcraft/blob/main/crates/doc/src),
calls [pathops](https://github.com/storytold/vectorcraft/blob/main/crates/pathops/src) for exact curve booleans, and asks
[the render crate](https://github.com/storytold/vectorcraft/blob/main/crates/render/src) to rasterize frames, with
[effects](https://github.com/storytold/vectorcraft/blob/main/crates/effects/src) evaluated live in the appearance stack. File work lives in
[svg](https://github.com/storytold/vectorcraft/blob/main/crates/svg/src) and
[pdf](https://github.com/storytold/vectorcraft/blob/main/crates/pdf/src), sandboxed extensions run through
[the plugins crate](https://github.com/storytold/vectorcraft/blob/main/crates/plugins/src), and
[the mcp crate](https://github.com/storytold/vectorcraft/blob/main/crates/mcp/src) exposes the same commands to agents over JSON-RPC. The dashed edge from the MCP server into the desktop app is the Remote backend: an agent can attach to a running editor on a control channel and drive it like a person at the keyboard.

## Why You Need This

The first reason is boolean honesty. Most editors approximate curve intersections with flattening or sampling, which is exactly why their Pathfinder operations fail or produce kinks. VectorCraft builds its booleans on a sweep-line designed for cubic Beziers, preserving curves through the intersection instead of reducing them to polylines, and the result either succeeds with mathematical exactness or reports why it could not. A crescent moon cut from two circles comes out with smooth handles on both sides, and property tests hammer the operations with random inputs to keep it that way. If you have ever lost an afternoon to a boolean that silently mangled your artwork, this is the crate to read.

The second reason is speed that holds when the file gets ugly. The renderer is multithreaded SIMD work running off the UI thread, built on sparse strip rasterization and culled by bounds so untouched regions cost nothing. The practical consequence is that a 20,000-shape poster with live glows, blends and gradients stays interactive at 120 fps instead of degrading into a beach ball. Because rendering reads the document rather than a cached bitmap, what you see on screen and what you export are the same pixels by construction, and the CLI's bench subcommand lets you verify the timing claim on your own hardware instead of trusting a README.

The third reason is that this is the most agent-native creative codebase you can clone today. Every menu item, panel, dialog and tool gesture is available through the same command API, exposed over stdio as a hand-written JSON-RPC MCP server or over a remote control channel to a running editor. Agents built every example poster in the repository through that interface, which means the automation path is a first-class product surface rather than a bolted-on scripting hole. If you are exploring how AI agents can operate professional creative software, this repository is a working reference, not a prototype.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/vectorcraft/storytold-vectorcraft-architecture.svg" alt="VectorCraft detail architecture: frontends, engine core with commands, tooling and guard, document model with geom and color, geometry services, rendering, file format crates and the MCP server" style="max-width:100%;"></div>

<p><em>Detail: the engine core, the model it protects, the geometry services it calls, and the format crates on the way out.</em></p>

Start at [the desktop entry point](https://github.com/storytold/vectorcraft/blob/main/apps/vectorcraft/src/main.rs). It loads a document, hosts the engine, and hands the interface to
[the ui-egui crate](https://github.com/storytold/vectorcraft/blob/main/crates/ui-egui/src), which builds the panels: Layers, Appearance, Swatches, Pathfinder and the toolbar. The panels never mutate the document directly; they issue commands. The same split holds for
[the CLI](https://github.com/storytold/vectorcraft/blob/main/apps/vectorcraft-cli/src/main.rs), whose run subcommand executes a script against the engine headlessly, and for
[the web app](https://github.com/storytold/vectorcraft/blob/main/apps/vectorcraft-web), which reuses the identical UI crate under WebAssembly.

Inside the engine, [the Session](https://github.com/storytold/vectorcraft/blob/main/crates/engine/src/lib.rs) is the one entry point every frontend calls, and
[the cmd directory](https://github.com/storytold/vectorcraft/blob/main/crates/engine/src/cmd) holds one implementation per mutation: grouping, selection, rectangle creation, appearance changes, you name it.
[The tooling module](https://github.com/storytold/vectorcraft/blob/main/crates/engine/src/tooling.rs) hosts tool sessions built on
[the tools crate](https://github.com/storytold/vectorcraft/blob/main/crates/tools/src), and the crucial design decision is that tools reduce to commands. A drag with the Pen is not a special mutation path; it decomposes into the same journaled commands the menus use, which is why every gesture is undoable, replayable and visible to agents.
[The guard module](https://github.com/storytold/vectorcraft/blob/main/crates/engine/src/guard.rs) runs invariant checks before a state is committed, so a document that fails validation never becomes something you can save or undo into. The engine also carries an enormous test suite alongside it, including journal replay and recovery tests, which is rare discipline for a creative tool.

The model layer is [doc](https://github.com/storytold/vectorcraft/blob/main/crates/doc/src) holding layers, objects, appearance stacks and styles, with
[geom](https://github.com/storytold/vectorcraft/blob/main/crates/geom/src) supplying kurbo-based curves and hit testing and
[color](https://github.com/storytold/vectorcraft/blob/main/crates/color/src) covering paints, gradients and colour spaces. Structural sharing is what makes undo unlimited: committing a change shares unchanged subtrees instead of deep-copying the document, so a long history costs memory proportional to what actually changed. Down in the geometry services,
[pathops](https://github.com/storytold/vectorcraft/blob/main/crates/pathops/src) implements unite, subtract and intersect over a sweep-line that keeps cubic Beziers intact, refitting curve segments where intersections split them;
[brush](https://github.com/storytold/vectorcraft/blob/main/crates/brush/src) builds variable-width stroke outlines,
[trace](https://github.com/storytold/vectorcraft/blob/main/crates/trace/src) converts raster images into vector paths, and
[text](https://github.com/storytold/vectorcraft/blob/main/crates/text/src) does harfrust shaping and type on a path.

The rendering side is [the render crate](https://github.com/storytold/vectorcraft/blob/main/crates/render/src), which evaluates the scene on vello_cpu's sparse strips with SIMD, culls by bounds, and walks appearance stacks, clip groups and gradients so live effects stay live.
[The effects crate](https://github.com/storytold/vectorcraft/blob/main/crates/effects/src) provides drop shadows, glows and blurs, while
[plugins](https://github.com/storytold/vectorcraft/blob/main/crates/plugins/src) runs third-party extensions inside wasmi, a pure-Rust WebAssembly interpreter: no host imports, a fuel budget per instruction count, memory and recursion caps, and a fresh instance per run so a plug-in cannot corrupt your session. The format crates round out the picture:
[svg](https://github.com/storytold/vectorcraft/blob/main/crates/svg/src) via usvg,
[pdf](https://github.com/storytold/vectorcraft/blob/main/crates/pdf/src) on krilla with PDF-compatible .ai import, plus EPS import that recovers the editing copy embedded in the file, CAD/DXF, EMF/WMF, and a genuinely unusual one:
[the affinity crate](https://github.com/storytold/vectorcraft/blob/main/crates/affinity/src) opens native .afdesign, .afphoto and .afpub files.
[The format crate](https://github.com/storytold/vectorcraft/blob/main/crates/format/src) defines the .vectorcraft JSON native format, documented and covered by property-tested round trips.

Finally, [the mcp crate](https://github.com/storytold/vectorcraft/blob/main/crates/mcp/src) is a hand-written JSON-RPC 2.0 server over stdio with two backends: Headless, which runs commands against an in-memory document, and Remote, which attaches to a running desktop editor over the control channel on port 7979. Because the MCP surface is the same command set the UI uses, an agent gets no superpowers a human lacks, and the human gets a complete automation interface for free.

## From Install to First Export

Clone the repository and run the desktop app with `cargo run --release -p vectorcraft`, optionally opening one of the example posters. For headless work, the CLI is the tool: `vectorcraft-cli run --in examples/dusk-poster.vectorcraft --export out.pdf` produces a PDF without opening a window, and `bench` prints render timing for any document. To wire an agent in, start the MCP server with `cargo run --release -p vectorcraft-cli -- mcp` and register it with your client using `claude mcp add vectorcraft`; the documentation in docs/mcp.md lists the tools. To attach to a live editor instead, launch the app with `--control 7979` and use the Remote backend. The web build is a `trunk build --release` inside apps/vectorcraft-web, and Japanese text ships via the optional craft-fonts build input, falling back to system fonts otherwise.

Honest limits: the project measures itself at roughly 69 to 75 percent of Illustrator's feature surface, with the power-user core around 40 to 55 percent away from indistinguishable, so expect missing niches rather than missing fundamentals. Plug-ins pay interpreter overhead inside wasmi, the CLI and web builds share the engine but not every desktop affordance, and the web assembly download is not trivial. None of that blunts the core achievement: a vector editor where the math is exact, the history is total, and the agent interface is the same door a human walks through.
