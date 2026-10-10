---
layout: post
title: "Knuth-Plass Type and Print-Grade PDF - Inside storytold/designcraft"
description: "A source tour of storytold/designcraft: how a pure-Rust clean-room reimplementation of InDesign gets a Knuth-Plass paragraph composer, IDML interchange, spot-colour PDF/A export and an MCP server with enforced layering."
date: 2026-10-10
header-img: "img/post-bg.jpg"
permalink: /designcraft/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/designcraft/storytold-designcraft-overview-architecture.svg
tags: [Rust, Open Source]
categories: [AI, Open Source]
keywords: designcraft, rust, indesign alternative, page layout, publishing, knuth-plass, idml, pdf export, mcp server, open source
author: "PyShine"
---

Page layout is the most conservative craft in software. A magazine does not care how modern your toolkit is; it cares that the rag is even, the hyphenation is defensible, the folios land on the parent page, and the PDF that reaches the printer has the right trim box. DesignCraft, from the storytold org, is a clean-room reimplementation of the Adobe InDesign workflow built in pure Rust, dual-licensed MIT OR Apache-2.0, version 0.5.0, native on macOS, Windows and Linux and running in the browser over WebGPU with a WebGL2 fallback. Spreads and parent pages, frames and threaded stories, paragraph and character styles, swatches and text wrap all behave the way InDesign muscle memory expects.

The claim that separates it from hobbyist layout tools is typographic: a Knuth-Plass paragraph composer that considers the whole paragraph when choosing line breaks, dictionary hyphenation from the public-domain Moby word list plus the team's own trained patterns, word, letter and glyph-scaling justification, keeps, optical margin alignment, columns, a baseline grid, tabs, rules and shading. The detail that proves the architecture: line breaks are identical on screen and in the exported PDF, because both paths read the same composition cache. This is the seventh and final Crafting App in this series, and it closes the loop on a full open-source creative suite.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/designcraft/storytold-designcraft-overview-architecture.svg" alt="DesignCraft overview architecture: desktop, CLI and web frontends feeding the engine, which edits the document model, composes stories through the text engine and fonts, and drives rendering, print PDF, IDML and the native format" style="max-width:100%;"></div>

<p><em>Overview: three frontends, one command registry, a document model, the type engine, and the output crates.</em></p>

Reading the overview from left to right: the frontends hold no layout logic.
[The desktop app](https://github.com/storytold/designcraft/blob/main/apps/designcraft/src)
docks the InDesign-style panels, [the CLI](https://github.com/storytold/designcraft/blob/main/apps/designcraft-cli/src)
runs documents headlessly and renders every page to PNG, and [the web app](https://github.com/storytold/designcraft/blob/main/apps/designcraft-web)
is the same codebase in a tab. All three talk to
[the engine](https://github.com/storytold/designcraft/blob/main/crates/engine/src), whose Session::execute is the only way anything changes, with commands like `frame.create`, `text.insert` and `layout.pages.insert`. The engine edits
[the document model](https://github.com/storytold/designcraft/blob/main/crates/doc/src) copy-on-write, composes stories through
[the compose crate](https://github.com/storytold/designcraft/blob/main/crates/compose/src), which shapes text with
[the fonts crate](https://github.com/storytold/designcraft/blob/main/crates/fonts/src), and renders pages through
[the render crate](https://github.com/storytold/designcraft/blob/main/crates/render/src) on multithreaded SIMD rasterization. Output branches three ways:
[the pdf crate](https://github.com/storytold/designcraft/blob/main/crates/pdf/src) writes print PDF,
[idml](https://github.com/storytold/designcraft/blob/main/crates/idml/src) handles InDesign interchange in both directions, and
[the format crate](https://github.com/storytold/designcraft/blob/main/crates/format/src) owns the native `.designcraft` file.
[The mcp crate](https://github.com/storytold/designcraft/blob/main/crates/mcp/src)
exposes every command to agents, with a dashed Remote edge into the running desktop app's control channel.

## Why You Need This

The first reason is the composer. Most free layout tools break lines greedily, one line at a time, which is why their justified columns develop rivers and their hyphenation looks arbitrary. DesignCraft implements Knuth-Plass total fit, the same algorithm behind TeX, evaluating the whole paragraph so the third line can accept a worse fit to make the fifth line better, alongside a single-line greedy composer for when you ask for it. Because shaping, breaking and placement live in one crate with a cache, the line breaks you approve on screen are byte-identical to what the PDF embeds, and the sample magazine in the repository's screenshots was laid out by the program itself from code, not by hand in another tool.

The second reason is honest file formats. The native `.designcraft` document is a zip whose first entry is an uncompressed `mimetype`, followed by `document.json`, pretty-printed and documented by the serde model, with placed assets embedded byte-for-byte, so a layout diffs in version control like source code. For InDesign shops, the IDML crate implements import and export clean-room from the public IDML specification: swatches including spot colours and gradients, paragraph, character and object styles, parent spreads, threaded stories, text wrap and transparency all survive the round trip, with an explicit TODO list of what does not rather than silent loss. And when the file must go to paper, the PDF export writes real print boxes, spot inks as separation colour spaces, and text as real, selectable, searchable glyphs with subsetted fonts.

The third reason is that layout becomes scriptable without losing its guardrails. Every menu item, tool gesture, panel control and dialog is a command with an id and JSON parameters through Session::execute, listable with `designcraft-cli commands`, reachable over a loopback JSON control channel on port 7979, and exposed over a hand-written MCP server so agents can lay out documents like a designer. The layering that makes this safe is enforced, not aspirational: `cargo xtask layers` checks that the UI sits on top and nothing underneath depends on it, so the same engine that drives the panels drives a headless render farm.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/designcraft/storytold-designcraft-architecture.svg" alt="DesignCraft detail architecture: frontends, engine core with commands, tooling, guard and the sample generator, the document model with geom and color, the compose text engine, rendering with placed art, and the PDF, IDML, EPUB, text import and native format crates" style="max-width:100%;"></div>

<p><em>Detail: the engine core, the Arc-shared document model, the Knuth-Plass engine, and the output crates around them.</em></p>

Start at [the desktop entry point](https://github.com/storytold/designcraft/blob/main/apps/designcraft/src/main.rs), which shows the start screen, opens the sample magazine with `--sample`, and starts the control channel on port 7979. The panels live in
[the ui-egui crate](https://github.com/storytold/designcraft/blob/main/crates/ui-egui/src): the Control panel showing measurements in picas, the Layers and Pages panels, and the non-printing chrome such as guides, frame edges, thread ports and selection, which the UI draws as vector overlays so they stay crisp at any zoom while the renderer draws document content only.
[The CLI](https://github.com/storytold/designcraft/blob/main/apps/designcraft-cli/src/main.rs) runs a document and writes every page out as PNG, and
[the web app](https://github.com/storytold/designcraft/blob/main/apps/designcraft-web) uses the browser's file picker for Open and Place, downloading on save and export.

Inside the engine, [the Session](https://github.com/storytold/designcraft/blob/main/crates/engine/src/lib.rs) is the facade every frontend shares, and
[the cmd directory](https://github.com/storytold/designcraft/blob/main/crates/engine/src/cmd) implements each mutation, from frame creation to page insertion.
[The tooling module](https://github.com/storytold/designcraft/blob/main/crates/engine/src/tooling.rs) hosts tool sessions built on
[the tools crate](https://github.com/storytold/designcraft/blob/main/crates/tools/src), which turns pointer events into commands and draws the overlays, so every gesture is journaled and replayable like a menu command.
[The guard](https://github.com/storytold/designcraft/blob/main/crates/engine/src/guard.rs) catches a panicking command and keeps the document as it was, turning a bug into a reportable error instead of lost work, and
[sample.rs](https://github.com/storytold/designcraft/blob/main/crates/engine/src/sample.rs) generates the four-page sample magazine from code, which is both a demo and a test fixture.

The model layer is [doc](https://github.com/storytold/designcraft/blob/main/crates/doc/src): spreads of pages, parent spreads, layers, stories, styles, swatches, sections and embedded assets, all Arc-shared so an edit clones only what it touches and undo snapshots are O(1). Coordinates are points with y down, each spread owning its own space. It leans on
[geom](https://github.com/storytold/designcraft/blob/main/crates/geom/src) for kurbo paths, corner options and the measurement parser that understands picas and millimetres, and on
[color](https://github.com/storytold/designcraft/blob/main/crates/color/src) for CMYK, RGB and Lab, tints and gradients.

The type engine is the heart: [compose](https://github.com/storytold/designcraft/blob/main/crates/compose/src) walks a story through style resolution, shaping, the breakers, and placement in columns and frames with first-baseline offsets, leading, space before and after, the baseline grid, text wrap, column and frame breaks, and vertical justification, ending with overset detection. It also carries the harder edges of real publishing: footnotes, ruby, tables, bidirectional text via unicode-bidi and unicode-script, cross-references and text variables. Shaping comes from
[the fonts crate](https://github.com/storytold/designcraft/blob/main/crates/fonts/src), a font database with harfrust shaping and skrifa outlines.

The output side starts with [render](https://github.com/storytold/designcraft/blob/main/crates/render/src), which turns spreads into premultiplied RGBA on vello_cpu with multithreading and a damage-aware cache, reading
[images](https://github.com/storytold/designcraft/blob/main/crates/images/src) for placed art: rasters through the image crate, Photoshop merged composites, EPS with bounding-box proxies, and SVG parsed with usvg, rasterized for screen but drawn as vectors in PDF.
[The pdf crate](https://github.com/storytold/designcraft/blob/main/crates/pdf/src) mirrors the renderer's walk to write MediaBox, TrimBox and BleedBox, gradient and solid fills, spot swatches as separation spaces, embedded subsetted fonts with Unicode mapping from the story text, crop and bleed marks, PDF/A-2b through krilla's validator, and even booklet imposition.
[idml](https://github.com/storytold/designcraft/blob/main/crates/idml/src) maps between the two coordinate systems and the two style models,
[epub](https://github.com/storytold/designcraft/blob/main/crates/epub/src) exports reflowable EPUB 3 with paragraph styles as CSS classes,
[textimport](https://github.com/storytold/designcraft/blob/main/crates/textimport/src) Places plain text, Word documents with styles by name, footnotes and tables, plus xlsx data merge capped at 100,000 rows, and
[the format crate](https://github.com/storytold/designcraft/blob/main/crates/format/src) round-trips the native zip, still opening older single-file JSON documents.

Finally, [the mcp crate](https://github.com/storytold/designcraft/blob/main/crates/mcp/src) is a hand-written JSON-RPC 2.0 server over stdio with no async runtime, two backends: Remote, forwarding to a running desktop app's control channel, and Headless, hosting an in-process Session that renders pages itself, so an agent can build a layout and look at it without a window ever opening.

## From Install to First Export

Clone and run: `cargo run --release -p designcraft` shows the start screen, `-- --sample` opens the sample magazine, and adding `--control 7979` lets you drive the running app with JSON lines. Headless, `cargo run --release -p designcraft-cli -- run --sample --all-pages out/` renders every page to PNG without a window, and `designcraft-cli commands` prints the full command registry. The browser build is a `trunk build --release` inside apps/designcraft-web, and Japanese text comes from the optional craft-fonts build input, falling back to system fonts when absent. Layout something, export IDML, reopen it, and watch the round trip.

Honest limits: the README still lists PDF export as on the roadmap while the pdf crate has quietly shipped print PDF with PDF/A-2b, so the docs lag the code, and PDF/X-4 lacks its output intent because krilla does not write one yet. IDML import ignores tables, footnotes and anchored objects with the gaps published rather than hidden, tagged PDF and bookmarks are not written, and there is no `.indd` import, only the interchange format. For a young tool, the foundation is the right one: an enforced layering, a real composer, and file formats you can read.
