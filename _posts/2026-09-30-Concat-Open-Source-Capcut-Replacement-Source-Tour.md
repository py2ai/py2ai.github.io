---
layout: post
title: "Concat: The Truly Free, Open-Source CapCut Replacement - Inside jub0t/Concat"
description: "A source tour of jub0t/Concat, the free and open-source cross-platform CapCut replacement written in Rust. We walk its actual code: the arena-based timeline model, the wgpu GPU rendering pipeline, the FFmpeg media layer, and the JSON-RPC, gRPC and MCP API that lets scripts and AI agents edit video."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Concat-Open-Source-Capcut-Replacement-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/concat/jub0t-concat-architecture.svg
tags:
  - Rust
  - Video Editing
  - Open Source
  - AI Agents
categories: [AI, Open Source]
keywords: "Concat video editor, open source CapCut alternative, Rust video editor, jub0t Concat, GPU compositor, wgpu, MCP API, JSON-RPC video editing, whisper auto captions, text to speech, background removal, free video editor no watermark, cross-platform video editor, Slint UI, AGPL video editor"
author: "PyShine"
---

Everyone who cuts short-form video knows the deal: the editor is free until you export, and then the watermark appears, or the 4K button asks for a subscription, or your footage quietly travels to somebody else's cloud. Concat, at roughly 3,900 GitHub stars at the time of writing, is the loudest "no" to that deal we have seen in a while. It bills itself as the truly free, open-source cross-platform CapCut replacement, and unlike most projects that wave at that ambition, it ships a full editor: multi-track cutting, auto-captions, text-to-speech, background removal, keyframes, 4K export.

Its README states the terms plainly: no watermark, no account, no subscription, no upload. Everything runs locally on a native Rust engine with a GPU compositor. The AI models for captions, voices and cutout download once from Settings and work offline after that. Your footage never leaves your disk.

Why is the source worth a tour? Because Concat is one of the rare video editors laid out as an honest, dependency-ordered Rust workspace you can actually read. The repository documents its own architecture in ARCHITECTURE.md, each of the sixteen crates under src/crates opens with a doc comment saying what it is for, and the dependency arrows point one way: nothing lower knows the window exists, and the engine's core knows nothing about FFmpeg. For anyone studying how a real editor works, or looking for an AI-controllable video backend, this codebase is a gift.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/concat/jub0t-concat-overview-architecture.svg" alt="Architecture overview of the jub0t/Concat repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Concat architecture: the document model on the left, the engine core in the middle, the Slint window and host services, the automation doors for scripts and agents, and the local AI crates on the right.*

Reading the overview from left to right: the document group holds the edit itself, where every change is a validated command (`src/crates/concat-project/src/commands`) applied to the project model (`src/crates/concat-project/src/model.rs`) and kept undoable by snapshot in `editor.rs`. The engine core turns that document into pixels: the document flattens into a `core::Timeline` built on exact rational time (`src/crates/concat-core/src/timeline.rs`), the frame planner decides what is on screen at any instant (`src/crates/concat-render/src/plan.rs`), and the wgpu compositor draws it (`src/crates/concat-render/src/gpu.rs`) with frames decoded and cached by the FFmpeg layer (`src/crates/concat-media/src/decode.rs`). The app group is what you click: the Slint window's Studio controller (`src/crates/concat/src/studio.rs`) edits through host sessions (`src/crates/concat-host/src/session.rs`). The automation doors expose the same verbs over JSON-RPC and gRPC (`src/crates/concat-api/src/lib.rs`, `src/crates/concat-server/src/lib.rs`) — exactly where scripts, plugins, MCP servers and AI agents plug in. Finally, the local AI crates do captions, voices and cutouts entirely on your machine (`src/crates/concat-speech/src/lib.rs`, `src/crates/concat-vision/src/lib.rs`).

## Why You Need This

If you have ever opened CapCut for a TikTok, a Reel or a YouTube tutorial, you already know the feature set people actually need: auto-captions, quick text-to-speech, background removal, keyframe animation, effects and titles, multi-track cutting, and a clean 4K export. Concat covers precisely that list. Captions land on the timeline already styled, voices can be cloned from a few seconds of a recording, background removal handles people and objects (or you can paint the mask yourself), and exports go out as H.264, HEVC or AV1, up to 4K at 60 fps in 10-bit colour.

The second problem is trust and longevity. A subscription editor owns your workflow; when the pricing changes or the service shuts down, your muscle memory and your project files are hostage. Concat is AGPL-3.0-or-later (with a plugin exception in LICENSE-EXCEPTIONS.md), the project file is a documented JSON document, and the engine compiles from source on macOS, Windows, Linux and Android, with iOS sideloaded. Version 0.2.5 is beta and the README says so plainly, but the code behind it is not a toy: there is an end-to-end export test suite (`src/crates/concat-host/tests/export.rs`) and a GPU-vs-CPU parity suite holding rendered frames to a structural similarity above 0.99 (`src/crates/concat-render/src/gpu/tests.rs`).

The third reason is the one our readers care most about: it is built for machines as well as people. The README says it directly, "Also for machines: a JSON-RPC, gRPC and MCP API, so scripts and AI agents can cut video with it too." We verified in the source that this is not marketing: `src/crates/concat-api/src/lib.rs` opens with "The command line, a daemon on a socket, an MCP server, a plugin: each is a transport," and every edit the window can make is one an API caller can make, with the same clamps and refusals. If you are building automated video pipelines, agent-driven editing or batch captioning, this is a rare editor where that door is a first-class citizen.

## How It Works

The best one-sentence summary comes from the repository's own src/README.md: the engine is a set of crates with no window in them, the window is a Slint application over that engine, and a command-line tool and a socket server drive the same engine through the same API the window uses — one document format, one command set, one renderer, three ways in.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/concat/jub0t-concat-architecture.svg" alt="Detailed architecture of the jub0t/Concat repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of Concat: the document and undo loop, the core vocabulary of exact time and arena handles, the render and effects stack, the media and export machinery, host services, local AI, and the transports that carry the API.*

### Understanding the Architecture

**The document and its command loop.** The edit lives in `src/crates/concat-project/src/model.rs` as a project of timelines, tracks and clips, with times stored as seconds and every keyframe expressed as a fraction of its clip's length, so splits and trims never orphan keys. Every change goes through a typed `Command` from `src/crates/concat-project/src/commands/` (grouped by clip, keys, audio, tracks, timelines and media), and each command funnels through `Clip::tidy`, the single place clamps live. The `Editor` (`src/crates/concat-project/src/editor.rs`) keeps undo as snapshots that share whatever a command did not touch, and the file format (`src/crates/concat-project/src/doc.rs`) is a versioned `concat.json` that migrates one step at a time and keeps unknown fields.

**Exact time and handle-based graphs.** Two conventions from `src/crates/concat-core` make the rest of the engine tractable. First, all timestamps are `concat_core::time::Rational` seconds, never `f64` — frame-accurate editing and floating point do not mix, as the code's own conventions note. Second, tracks and clips live in generational arenas (`src/crates/concat-core/src/arena.rs`) and are addressed by copyable `TrackId` and `ClipId` handles (`src/crates/concat-core/src/timeline.rs`), with track order explicit and bottom-first: track zero composites first and everything after draws on top. A clip's `speed` is a Rational and the only definition of retiming, so the frame plan, the export decoder rate and the audio graph cannot disagree about what a sped-up clip means.

**The three shapes of a clip on its way to a pixel.** `src/crates/concat-export/src/flatten.rs` turns the document into one flat list of export clips, `resolve.rs` builds the engine's `core::Timeline` quantised to the frame grid with per-clip facts the model has no field for, and `src/crates/concat-render/src/plan.rs` produces a `FramePlan`: a pure description of what is visible at instant t — media, exact source time, placement, opacity, blend — with no IO and no pixels. Because geometry (crop, fit, scale) and weighing (fades, wipes, masks) are computed once in the plan, the export renderer and the on-screen monitor cannot drift apart.

**One compositor in linear light.** `src/crates/concat-render/src/gpu.rs` implements the wgpu compositor that draws every frame, the monitor's and the export's alike, falling back to a software adapter on machines without a GPU. The working space is linear light on extended Rec. 709 primaries in half floats (what Windows calls scRGB), so blending, opacity and fades happen in light the way professional colour tools do them; HDR sources in HLG or PQ are decoded deep and conformed with BT.2390 roll-off. Effects are GPU packages: 123 folders under `src/crates/concat-effects/packages/`, each a `effect.toml` manifest plus a WGSL shader, parsed and validated at load by `src/crates/concat-effects/src/shader.rs` — the README counts 170+ effects, filters, transitions and text animations in total. A shader that binds something the host did not declare, or loops without a break, is simply refused.

**Decoding, caching and scheduling.** `src/crates/concat-media` is the only crate that knows FFmpeg exists. `decode.rs` prefers platform hardware decode (`hardware.rs`: VideoToolbox on macOS, D3D11VA on Windows, MediaCodec on Android) and falls back to software once, `pool.rs` caches decoded frames keyed by file, decode level and frame index, and `prefetch.rs` schedules monitor frames first, then filmstrips, artwork and proxies, on a few threads with one always kept clear of background work. Large footage gets a quarter-size H.264 proxy (`src/crates/concat-host/src/proxy.rs`), and playback audio is decoded to memory-mapped WAVs in the project cache (`src/crates/concat-host/src/playback.rs`).

**Local AI, and the doors for machines.** Captions and voices are two crates: `src/crates/concat-speech/src/transcribe.rs` runs whisper.cpp in-process, `tts.rs` drives Kokoro through sherpa-onnx, both one-at-a-time and both downloading models on demand rather than bundling them. Cutouts and enhancement come from `src/crates/concat-vision` (segmentation masks and an ONNX restoration model). Then there are the doors: `src/crates/concat-api/src/lib.rs` is the one dispatcher whose verbs are JSON in and responses and events out, and `src/crates/concat-server/src/lib.rs` carries the same API over JSON-RPC lines on TCP or a Unix socket, or gRPC behind a feature flag from `src/crates/concat-server/proto/concat.proto`. Every connection presents a token first, compared in constant time (`src/crates/concat-server/src/token.rs`), the server binds loopback by default, and an MCP server is just another transport over the same dispatcher — so an agent that speaks JSON-RPC gets the full editing vocabulary, including exports that arrive as jobs with progress events.

Follow one export end to end and the whole design clicks: the window (or a script) sends a command; the document validates and snapshots it; the export crate flattens and resolves the document into a core timeline; for each frame the planner emits a FramePlan; the decoder fills it; the compositor blends it in linear light; and the encoder writes H.264, HEVC or AV1 to disk — locally, the entire way.

## Advantages

- **Genuinely free and open.** AGPL-3.0-or-later, no watermark, no account, no paid tier, and the licensing terms including the plugin exception are spelled out in LICENSE-EXCEPTIONS.md.
- **Fully local by design.** FFmpeg is linked, whisper.cpp and sherpa-onnx are compiled in, and nothing uploads; the AI models download once and then work offline.
- **A real engine, not a wrapper.** A GPU compositor on wgpu in linear light, hardware decode on every platform, frame caching and a scheduler — 4K scrubs stay smooth.
- **Frame-accurate fundamentals.** Rational-number time and arena-handle timelines mean the timeline model, the renderer and the audio graph all agree on what a frame is.
- **Machine-friendly API.** The same JSON-RPC dispatcher serves the CLI, sockets, gRPC and MCP transports, with token authentication, capability discovery via a `version` call, and exports as jobs with progress events.
- **Cross-platform for real.** One Rust codebase builds the editor for macOS, Windows, Linux and Android (iOS sideloaded), with a dedicated phone shell in the UI and wasm builds kept green for the pure-Rust engine crates.

## Benefits

- **For short-form creators:** auto-captions, text-to-speech, voice cloning, background removal and a magnetic multi-track timeline cover the whole CapCut muscle memory without the watermark tax.
- **For privacy-conscious editors:** footage, models and exports all stay on disk; there is no account to create and no telemetry funnel in the design.
- **For teams and long-term projects:** the human-readable `concat.json` project format with versioned migration means your edit is data, not a black box.
- **For automation engineers:** batch caption a folder, render a nightly cut, or drive exports from CI with the same commands the window uses.
- **For AI agent builders:** an MCP-able, JSON-RPC-first editing backend with typed commands, predictable error codes and job-based exports is exactly the primitive most "AI video" stacks are missing.
- **For Rust learners and researchers:** strict crate boundaries, a documented architecture, and parity-tested rendering make this one of the best real-world codebases to study engine design in.

## Usage

Concat is a native application: grab the right build from the website or GitHub Releases for Windows (`.msi`), macOS, Linux (`.deb`, `.rpm`, `.AppImage`) or Android. To build and drive it from source, the repository's src/README.md is the guide; a build needs the FFmpeg 7+ development libraries, plus cmake and a C++ toolchain for whisper.cpp:

```sh
cargo build
cargo run -p concat                    # the editor window
cargo run -p concat-cli -- probe some-video.mp4
cargo run -p concat-cli -- render some-video.mp4 out.mp4 --frames 120
cargo run -p concat-cli -- serve       # the API on 127.0.0.1:7420
cargo run -p concat-cli --features grpc -- serve --grpc 127.0.0.1:7421
```

Once a server is running, every call is a JSON-RPC 2.0 line, one object per line, authenticated with a token:

```jsonl
{"jsonrpc":"2.0","id":1,"method":"project.open","params":{"path":"/edits/Reel"}}
{"jsonrpc":"2.0","id":2,"method":"edit.apply","params":{"path":"/edits/Reel","command":{"op":"addTextClip","start":1.5}}}
{"jsonrpc":"2.0","id":3,"method":"export.run","params":{"path":"/edits/Reel","output":"/edits/reel.mp4"}}
```

An `edit.apply` carries a `concat-project` command exactly as the window would send it, so anything the editor can do — captions, cutouts, keyframes, effects — is available to your script. The full method reference, including a page per transport, lives in the developer docs on the project's website.

## Conclusion

Concat is what it claims to be: a complete, cross-platform, no-strings video editor whose source you can actually read. Its architecture is unusually disciplined — exact rational time, handle-based timelines, a pure frame plan, one GPU compositor in linear light, and a single API dispatcher behind every way of driving it. If you want a CapCut replacement that respects your footage, or a scriptable, agent-ready video engine to build on, clone it and start at `src/README.md`; the map is already drawn for you.

Links:

- GitHub repository: [github.com/jub0t/Concat](https://github.com/jub0t/Concat)
- Website and downloads: [concatenate.pages.dev](https://concatenate.pages.dev/)
- Developer docs (Concat API): [concatenate.pages.dev/docs](https://concatenate.pages.dev/docs)
