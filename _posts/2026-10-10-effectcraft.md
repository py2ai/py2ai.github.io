---
layout: post
title: "EffectCraft: 306 Effects and Zero FFmpeg - Inside storytold/effectcraft"
description: "A source tour of storytold/effectcraft: how a nine-day-old pure-Rust clean-room reimplementation of After Effects ships 306 effects, a wgpu compositor, deterministic JavaScript expressions and an MCP server without a line of FFmpeg."
date: 2026-10-10
header-img: "img/post-bg.jpg"
permalink: /effectcraft/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/effectcraft/storytold-effectcraft-overview-architecture.svg
tags: [Rust, Open Source]
categories: [AI, Open Source]
keywords: effectcraft, rust, after effects alternative, motion graphics, vfx, compositor, lottie, mcp server, open source
author: "PyShine"
---

Every motion designer knows the deal: the compositor that defines the industry asks for a subscription, a login, and a machine that eats RAM for breakfast, and its project files are a binary blob you cannot even diff. EffectCraft, from the storytold org, is the answer this blog series keeps meeting: a clean-room reimplementation of the Adobe After Effects workflow in pure Rust, dual-licensed MIT OR Apache-2.0, version 0.7.0, native on macOS, Windows and Linux and running the full editor in the browser. Here is the part that made me open the source: the repository's first commit is dated 1 October 2026. This codebase is nine days old and it already ships 306 effects, a 3D camera system, a render queue, and expressions.

The feature arithmetic is the hook. EffectCraft implements every one of After Effects' 298 effects and adds more, across the categories you know: blur and sharpen, colour correction with Curves and Lumetri, distort with Turbulent Displace, generate with Fractal Noise, keying, simulation with CC Particle World and Shatter, time effects like Echo and Timewarp, and audio effects from Reverb to Parametric EQ. All nine layer styles are there with Global Light. Keyframes behave the AE way with speed and influence numbers, the Graph Editor draws value and speed graphs, and the puppet tools deform a mesh with Position, Bend, Starch and Overlap pins. None of it is a wrapper around FFmpeg: every decoder and encoder is pure Rust.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/effectcraft/storytold-effectcraft-overview-architecture.svg" alt="EffectCraft overview architecture: desktop, CLI and web frontends feeding the engine, which edits the project model copy-on-write, and drives CPU and GPU compositing, effects, expressions and the render queue" style="max-width:100%;"></div>

<p><em>Overview: three frontends, one command registry, and the compositor split between CPU reference and GPU acceleration.</em></p>

Reading the overview from left to right: the frontends contain no editor logic.
[The desktop app](https://github.com/storytold/effectcraft/blob/main/apps/effectcraft/src)
docks the familiar panels, [the CLI](https://github.com/storytold/effectcraft/blob/main/apps/effectcraft-cli/src)
runs one-shot commands and headless renders, and [the web app](https://github.com/storytold/effectcraft/blob/main/apps/effectcraft-web)
compiles the same interface to WebAssembly. All three talk to
[the engine](https://github.com/storytold/effectcraft/blob/main/crates/engine/src), whose Session::execute is the single entry point for every menu item, drag and property edit, dispatching commands with stable ids like `layer.newSolid` and `keys.easyEase`. The engine edits
[the project model](https://github.com/storytold/effectcraft/blob/main/crates/project/src) copy-on-write and stores animated values through
[keyframe](https://github.com/storytold/effectcraft/blob/main/crates/keyframe/src) with AE's exact interpolation semantics. Rendering splits in two:
[the render crate](https://github.com/storytold/effectcraft/blob/main/crates/render/src) is the CPU reference compositor that caches each layer's content, and
[the gpu crate](https://github.com/storytold/effectcraft/blob/main/crates/gpu/src) composites those caches on wgpu compute shaders, the counterpart to Mercury GPU Acceleration.
[Effects](https://github.com/storytold/effectcraft/blob/main/crates/effects/src)
run as ordinary property groups, [expressions](https://github.com/storytold/effectcraft/blob/main/crates/expr/src)
evaluate in JavaScript, [the export crate](https://github.com/storytold/effectcraft/blob/main/crates/export/src)
writes the render queue's output, and [the automation crate](https://github.com/storytold/effectcraft/blob/main/crates/automation/src)
exposes every command to agents over MCP, with a dashed bridge edge into the running desktop app.

## Why You Need This

The first reason is that the project file is finally text. Compositions save as `.ecproj`, versioned JSON with a schema number, so a motion graphics project diffs cleanly in version control and two artists can actually review what changed in a title animation. Undo works the same way at the architecture level: the engine holds an `Arc<Project>` and edits copy-on-write, so undo snapshots share every composition that did not change and the memory cost of a long history tracks what you actually touched. For teams that live in git, this alone is worth the clone.

The second reason is the export stack, because "no FFmpeg" is a design decision with consequences you can read. The render queue writes H.264 MP4 and ProRes MOV from FilmCraft's pure-Rust codecs, HEVC and AV1 MP4 from EffectCraft's own encoders, WebM with VP9 inter frames and alpha plus Opus audio from its own VP9 and Opus encoders, 32-bit float EXR sequences, animated GIF, and WAV or AIFF. Field rendering with 3:2 pulldown, effect and solo overrides, crop, region of interest, resize, storage overflow and render logs all behave like the After Effects dialogs. The same queue runs from the command line, which makes unattended rendering a script instead of a workstation.

The third reason is that this is the most aggressively agent-drivable compositor in existence, and it says so in the crate docs. Every user-visible action is a command with an id and JSON parameters, so the MCP server over stdio, the CLI's one-shot subcommands, the web build and the desktop panels all stand on the same registry. You can set a property with `effectcraft-cli set Main '#1' transform/position '[100,360]' --time 0 main.ecproj --save`, bridge the MCP server to a running app on port 9877 and let an agent take screenshots and click widgets by automation id, or ship a headless agent a repository that includes a ready `.mcp.json`. A compositor you can drive from a text stream is a different product category.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/effectcraft/storytold-effectcraft-architecture.svg" alt="EffectCraft detail architecture: frontends, engine core with render queue and rotoscope tools, project model with keyframes and time, expressions, CPU and GPU compositing, media, export and Lottie crates" style="max-width:100%;"></div>

<p><em>Detail: the engine core, the copy-on-write model, the two compositors, and the pure-Rust media stack.</em></p>

Start at [the desktop entry point](https://github.com/storytold/effectcraft/blob/main/apps/effectcraft/src/main.rs). It opens the windows, starts audio output for synced preview, and brings up
[the control server](https://github.com/storytold/effectcraft/blob/main/apps/effectcraft/src/control_server.rs), the JSON-lines channel on port 9877 that accepts commands, inspects and clicks any widget by automation id, and takes screenshots. The panels come from
[the ui-egui crate](https://github.com/storytold/effectcraft/blob/main/crates/ui-egui/src): Project, Composition, Timeline, Effect Controls, Effects & Presets, Character and Paragraph, docked the way After Effects docks them, and like every frontend they only issue commands.
[The CLI](https://github.com/storytold/effectcraft/blob/main/apps/effectcraft-cli/src/main.rs) adds set, render and exec subcommands with JSON output, and
[the web app](https://github.com/storytold/effectcraft/blob/main/apps/effectcraft-web) proves the layering by running the identical stack in a tab.

Inside the engine, [the Session](https://github.com/storytold/effectcraft/blob/main/crates/engine/src/lib.rs) is the facade every frontend shares, and its module list reads like a feature index: autosave, render queue, preview, roto, tracking, shortcuts, templates, media cache.
[The render_queue module](https://github.com/storytold/effectcraft/blob/main/crates/engine/src/render_queue.rs) holds Render Settings and Output Module templates with defaults and post-render actions, feeding
[the export crate](https://github.com/storytold/effectcraft/blob/main/crates/export/src), whose lib.rs opens with a table of every format, container, codec, alpha and audio combination it writes.
[The roto module](https://github.com/storytold/effectcraft/blob/main/crates/engine/src/roto.rs) carries Roto Brush and tracking, optionally wired to MobileSAM and Google's MediaPipe Face Landmarker as small on-demand model downloads.

The model layer is [project](https://github.com/storytold/effectcraft/blob/main/crates/project/src), a tree of items, compositions, layers and properties with the schema version stamped into the file, hung on
[keyframe](https://github.com/storytold/effectcraft/blob/main/crates/keyframe/src), which implements the subtle parts of AE animation: eases stored as speed and influence exactly like the Keyframe Velocity dialog shows, spatial properties moving along Bezier motion paths through an arc-length table, and auto-Bezier tangents derived from neighbours.
[The time crate](https://github.com/storytold/effectcraft/blob/main/crates/time/src) underlies both with ticks and frame rates including 29.97 drop-frame, so timing is frame-accurate everywhere.

Expressions deserve their own paragraph because they are the part most clones fake.
[The expr crate](https://github.com/storytold/effectcraft/blob/main/crates/expr/src) runs JavaScript on the boa engine with a prelude implementing the AE object model: `thisComp`, `effect()()`, `sourceRectAtTime`, `wiggle`, `loopOut`, `toComp`, vector maths on arrays, and the `linear` and `ease` helpers. The craft detail is determinism: `wiggle` and `random` are seeded by layer index and property uid, so every thread and every platform renders identical frames, and one boa context per thread caches compiled scripts by text.

The compositing side is where CPU and GPU meet by design.
[The render crate](https://github.com/storytold/effectcraft/blob/main/crates/render/src) walks each layer bottom to top through source, masks, effects, transform with motion-blur sub-samples, track matte and blend, using
[raster](https://github.com/storytold/effectcraft/blob/main/crates/raster/src) for tiny-skia SIMD pixel work, and its per-layer cache is exactly what
[the gpu crate](https://github.com/storytold/effectcraft/blob/main/crates/gpu/src) uploads once and composites with all 38 blend modes on Metal, Vulkan, Direct3D 12 and WebGPU. Advanced 3D comps take a separate WGSL route with depth buffers, PBR lighting and shadow maps. GPU effect families keep the CPU effect's exact steps and fall back to the CPU when a parameter cannot be matched, which is the honest way to keep both paths identical.
[Effects](https://github.com/storytold/effectcraft/blob/main/crates/effects/src) register as specs with stable ids and typed parameters, and
[the plugin crate](https://github.com/storytold/effectcraft/blob/main/crates/plugin/src) loads third-party `.wasm` effects into the same registry: no imports, so no files, network or clock, a fuel limit per frame, and deterministic floats, so a plug-in cannot hang the app or drift between machines.

The media stack closes the loop: [host](https://github.com/storytold/effectcraft/blob/main/crates/host/src) wires a fully loaded Session by connecting
[media](https://github.com/storytold/effectcraft/blob/main/crates/media/src) for footage decoding, the expression engine, JavaScript scripting and the export queue, while
[the lottie crate](https://github.com/storytold/effectcraft/blob/main/crates/lottie/src) maps compositions to and from Lottie JSON, listing anything Lottie cannot express instead of silently dropping it. Finally,
[the automation crate](https://github.com/storytold/effectcraft/blob/main/crates/automation/src) is the agent surface: a hand-rolled JSON-RPC 2.0 MCP server with no async runtime, a tool catalogue shared with the CLI, and the bridge client that forwards to a live desktop session.

## From Install to First Render

Grab an installer from the releases page or build it: `cargo run --release -p effectcraft` opens the app, and `-- --demo` loads the demo project so you have something to poke. Render it headless immediately with `cargo run --release -p effectcraft-cli -- render --out intro.mp4`, which exercises the whole queue without a window. Drive properties from the shell with `effectcraft-cli set`, register the MCP server with your agent client (the repository ships a `.mcp.json`), or attach it to the running editor with `--bridge 9877` after starting the app with `--control 9877`. The browser build is `cargo xtask web --serve 8765`, and the docs cover footage import, the control protocol and the web build.

Honest limits: the project's own README is admirably blunt. After Effects projects cannot be opened and AE plug-ins do not run, behaviour has not yet been measured against AE's renders automatically, macOS is the most tested platform while Linux and Windows users have hit basic interaction bugs, and the Roto Brush and face tracking models are too new to have benchmarks. For a nine-day-old codebase, that is a to-do list, not a eulogy, and the architecture underneath is already the most transparent compositor source you can read.
