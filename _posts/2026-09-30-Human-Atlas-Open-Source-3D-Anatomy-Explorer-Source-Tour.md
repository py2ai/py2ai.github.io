---
layout: post
title: "Human Atlas: An Interactive 3D Anatomy Explorer in the Browser - Inside ashemag/human-atlas"
description: "A source-level tour of ashemag/human-atlas, a TypeScript and Three.js application that renders 2,234 selectable BodyParts3D meshes with merged geometry batches, GPU state textures, a packed exploded view, and concept-level search. We trace the rendering layer, the anatomy data model, and the validation pipeline that keeps it honest."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Human-Atlas-Open-Source-3D-Anatomy-Explorer-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/human-atlas/ashemag-human-atlas-architecture.svg
tags:
  - Three.js
  - WebGL
  - TypeScript
  - Anatomy
categories: [AI, Open Source]
keywords: "human atlas, 3d anatomy explorer, bodyparts3d, three.js, webgl, react, typescript, exploded view, anatomical systems, mesh rendering, gpu state textures, open source, source code tour, meshoptimizer, vite"
author: "PyShine"
---

Interactive anatomy on the web tends to live at two extremes: heavyweight clinical platforms behind subscriptions, or static plates you can look at but never take apart. Somewhere in between sits a genuinely hard engineering problem — how do you render thousands of individually selectable meshes in a browser, keep the frame rate civilized, and make every structure clickable, hideable, and separable, without a backend? The `ashemag/human-atlas` repository answers that question with a small, unusually disciplined TypeScript codebase.

Human Atlas is an open-source 3D anatomy explorer built with React, Three.js, and shadcn/ui. It packages the BodyParts3D 4.0 adult male reference anatomy into **2,234 individually selectable meshes** organized across **15 anatomical systems**, with **3,432 named FMA concepts** searchable from the interface. The whole model carries 2,288,268 triangles and downloads roughly 33 MB of compressed geometry. You can orbit and zoom a lit figure on a turntable stage, toggle individual systems or jump to skeleton and organ presets, drag an "explode" slider that spreads the visible body into a spaced inventory of every piece, search anatomical names, and isolate a selected structure to read its details.

The source is worth a tour precisely because it solves the hard version of the problem. Rather than keeping 2,234 separate meshes and 2,234 draw calls, it merges geometry into per-system batches and encodes per-structure translation, visibility, and selection in GPU textures that its custom shaders read at render time. The repository also ships the entire data pipeline — OBJ conversion, meshoptimizer simplification, gzip compression — plus two validation scripts that assert the numbers above against the actual binary buffers. It is a complete, reproducible case study in shipping a large scientific dataset to the browser.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/human-atlas/ashemag-human-atlas-overview-architecture.svg" alt="Architecture overview of the ashemag/human-atlas repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the Human Atlas architecture: the Vite entry shell boots the React explorer UI, which hosts a Three.js rendering engine fed by the anatomy data model, the atlas manifest, and binary geometry chunks — all produced by the geometry build pipeline.*

Reading the overview from left to right: the Entry and Shell group is just two files, `web/index.html` and `web/main.tsx`, which mount the React root; the Explorer Interface group centers on `app/page.tsx`, the single "Home" component that owns all UI state and also registers optional WebMCP agent tools. The 3D Engine group is where the interesting work happens — `app/scene.tsx` drives the WebGL renderer and pulls in the exploded-view packer, the tap-versus-orbit filter, and the chunk decoder. The Anatomy Data group holds the typed data model in `app/anatomy.ts` plus the `public/models/atlas.json` manifest and the `body-*.bin(.gz)` geometry chunks the engine streams in. Finally, the Geometry Pipeline group contains the offline scripts that build both data artifacts in the first place.

## Why You Need This

If you have ever wanted a real anatomy reference in a classroom, a study session, or a blog post, the usual options are frustrating. Native 3D atlas applications cost money, demand installs, and lock their content. Static diagram collections are free but frozen — you cannot peel away muscles to reach the heart. Human Atlas needs nothing but a URL and a WebGL-capable browser: no API keys, no accounts, and no backend. The README explicitly notes that the included `vercel.json` configures `npm ci`, `npm run build`, and the `dist` output directory, so the app deploys to Vercel as a Vite project or any other static host.

For developers, the codebase is a rare, complete example of large-scene WebGL done with restraint. Most Three.js tutorials stop at a few meshes; the moment your scene contains thousands of pickable objects, the naive approach collapses. This repository shows the production pattern — merged geometry batches, per-part state encoded in `DataTexture` samplers, custom shader injection through `onBeforeCompile`, invisible picker meshes for accurate raycasting — in about seven application files under `app/` and `web/`. Everything is small enough to read in one sitting, and every claim the README makes is checkable against the code.

For educators and content builders, the anatomy itself is handled responsibly. The data comes from BodyParts3D 4.0, © The Database Center for Life Science, licensed CC BY 4.0, and `public/ATTRIBUTION.md` documents the source archive, the license links, and every adaptation made (axes converted from millimeters/Z-up to meters/Y-up, geometry simplified with a 0.2% relative error limit per structure, normals quantized to signed 16-bit, and system groupings curated for display). The interface states plainly that this is an educational reference, not a diagnostic or surgical tool, and that the male reference does not represent every human structure or variation.

Finally, if you build interactive 3D for touch devices, this repo is a quiet masterclass. `app/pointer-tap.ts` implements a small state machine that distinguishes a tap from an orbit, a pinch, a pan, or a canceled touch sequence, using a 5-pixel threshold for mouse and 12 pixels for touch and blocking selection whenever a second pointer joins. The scene caps device pixel ratio at 1.5 on small screens and 2 on desktop, and `scripts/validate-interactions.mjs` re-verifies tap handling, packing, and search contracts on every run.

## How It Works

The entire experience flows through one React component that owns the state and one Three.js scene that renders it, connected by a typed manifest that describes every mesh in the body.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/human-atlas/ashemag-human-atlas-architecture.svg" alt="Detailed architecture of the ashemag/human-atlas repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of ashemag/human-atlas, from the entry shell and shadcn/ui interface layer through the Three.js engine and the validated BodyParts3D data artifacts, back to the build-and-validation pipeline that produces them.*

### Understanding the Architecture

**The manifest-driven data model.** Everything starts with `public/models/atlas.json`, fetched by `app/page.tsx` on mount. Its shape is defined in `app/anatomy.ts`: an `Atlas` carries `version`, `sex`, `source`, `parts`, `concepts`, `chunks`, and `triangles`; each `Part` records an `id`, an English `name`, a `conceptId`, its display `system`, which binary `chunk` holds its geometry, byte offsets for `positions`, `normals`, and `indices`, vertex and index counts, and a bounding box. Each `Concept` groups one or more mesh ids under a named FMA concept — which is why 2,234 source meshes map onto 3,432 named concepts. The same file defines the 15 entries of the `SYSTEMS` array (skeleton, muscles, arteries, veins, nerves, and so on, each with a display color) and nine curated `EXPLANATIONS` for major organs.

**Merged batch rendering with GPU-encoded state.** `app/scene.tsx` creates a `WebGLRenderer` with antialiasing, ACES filmic tone mapping, and an sRGB output color space, lights the figure with a hemisphere light plus two directional lights over a PMREM-processed `RoomEnvironment`, and builds the stage: a ground disc, a platform cylinder, and two guide rings. Geometry arrives in 15 chunks (`body-0.bin` through `body-14.bin`), fetched by three concurrent workers; each chunk's `ArrayBuffer` is sliced directly into `Float32Array` positions, normalized `Int16Array` normals — the code comments that these "keep the complete atlas compact in memory" — and `Uint32Array` indices. Then comes the key move: geometries belonging to the same system are merged with `mergeGeometries` from Three's `BufferGeometryUtils`, so the entire body renders as one mesh per system rather than thousands. Per-part state lives in two textures: a float `DataTexture` holding an XYZ translation plus a visibility bit per part, and a byte texture holding selection. `materialFor` injects a `partIndex` attribute and shader patches via `onBeforeCompile`, so each vertex reads its own part's row from the textures, translates itself for the exploded view, and `discard`s itself when hidden — meaning visibility, selection tinting, and explosion animation cost only a texture update, not a scene graph rebuild.

**Picking and pointer semantics.** Alongside each merged batch, `app/scene.tsx` keeps an invisible picker mesh per part with `matrixAutoUpdate` disabled. On pointer-up, the `PointerTap` filter (from `app/pointer-tap.ts`) first decides whether the gesture was a tap at all; if so, a raycaster pretests each visible part's translated bounding box before running exact triangle intersection on the picker meshes. When the body is exploded, the renderer also projects every visible part's bounding-box corners to screen space and builds a target list, so taps (and hover tooltips) can fall back to fast screen-space hit tests against projected rectangles.

**The exploded view as a packing problem.** `app/explosion-layout.ts` implements `createExplosionLayout`, a shelf packer that sorts the visible parts by projected height, computes a target width from total card area and the camera aspect ratio, and assigns every part a non-overlapping cell. The scene animates explosion in two phases: below 45% of the slider, pieces drift outward along an angle derived from their system index; beyond that, they interpolate into their packed cells, and the camera auto-refits so the inventory fits the viewport. Once explosion passes 75%, a `Points` cloud of per-part markers appears, with a shader patch that discards fragments outside a circular point sprite. Isolating a structure is handled with `camera.setViewOffset`, which shifts the projection so the isolated piece frames itself in the space beside the open detail panel.

**Search and inspection.** `app/page.tsx` implements search as a simple, effective filter: concept names and ids are matched by lowercase substring, sorted by name length so specific structures surface first, and capped at 80 results in a shadcn/ui combobox (with `/` as the keyboard shortcut). Selecting a concept selects all its member meshes, and the detail sheet shows the system color accent, an explanation drawn from the nine curated organ entries or the system description, the atlas reference id, the piece count, and a member list linking to individual meshes. The isolate button then focuses the camera on exactly those pieces. There is also an agent-facing surface: `app/agent-tools.ts` exposes `find_anatomy` and `inspect_anatomical_structure` tools through the browser's optional `document.modelContext` WebMCP registration, and degrades silently when the browser does not support it — the visible interface never depends on it.

**A reproducible data pipeline with teeth.** The `scripts/` directory rebuilds the data from the official BodyParts3D OBJ archive: `convert-anatomy.py` streams the OBJ meshes, converts millimeter/Z-up coordinates to meter/Y-up, quantizes normals to signed 16-bit, and packs parts into chunked binaries plus the manifest; `optimize-anatomy.mjs` runs meshoptimizer's quadric simplifier with a 0.2% relative error bound per structure (preserving every named mesh) and re-chunks the results into the shipped `body-*.bin` files; `compress-models.mjs` gzips each chunk at level 9 and records the gzip URLs and sizes in the manifest. Crucially, `scripts/validate-atlas.mjs` asserts exactly 2,234 parts and 3,432 concepts with unique ids, walks every byte buffer, verifies index bounds and finite positions, sums the triangles, and checks that every concept references existing mesh ids — while `scripts/validate-interactions.mjs` proves the exploded layout never overlaps cells at desktop and mobile aspect ratios and replays the tap state machine's edge cases.

End to end: you open the page, React fetches the manifest, `AnatomyScene` mounts, three workers stream and decode 15 gzip chunks, merged system meshes appear on the stage, and every later interaction — toggling a system, dragging the explode slider, tapping a bone — writes rows into a pair of GPU textures that the patched shaders apply on the next dirty frame. Selection raycasts against real component geometry, the detail sheet explains what you picked, and isolation reframes the camera. No draw call is ever created or destroyed for any of it.

## Advantages

- **No backend, no keys, no accounts.** The app is pure static files after `vite build` — a JSON manifest and 15 binary chunks — so it deploys to Vercel (config included) or any static host and works entirely client-side.
- **2,234 pickable meshes without a draw-call explosion.** Per-system merged geometry plus texture-encoded per-part state means visibility, selection, and explosion never multiply draw calls, which is the pattern most Three.js projects eventually need and rarely find demonstrated.
- **Accurate whole-body picking.** Invisible per-part picker meshes with bounding-box pretests give exact triangle-level selection, and a projected screen-space fallback keeps hover and tap working when the body is exploded into a grid.
- **Touch-first interaction discipline.** A dedicated tap-state machine separates taps from orbits and pinches, mobile gets capped pixel ratios, compact panels, and validated exploded layouts, and WebGL context loss is handled with a graceful reload message.
- **Validated, self-checking data.** Two node scripts assert mesh counts, concept membership, buffer integrity, triangle totals, non-overlapping packing, and tool contracts, so the dataset cannot silently rot.
- **A clean licensing story.** Application code is MIT; the BodyParts3D data is CC BY 4.0 with the full attribution, source links, and adaptation history preserved in `public/ATTRIBUTION.md`.

## Benefits

- **A serious anatomy reference for learners and educators.** Fifteen toggleable system layers, skeleton and organ presets, view buttons, auto-rotation, and per-structure explanations make it usable in a lesson or a study session without any setup.
- **A worked example of modern WebGL state management.** Reading `app/scene.tsx` teaches shader injection, data textures, geometry merging, and dirty-frame rendering in one coherent, real codebase rather than in isolated snippets.
- **A template for shipping large scientific datasets.** Chunked binaries with gzip fallbacks, magic-byte sniffing for double-decompression, and exact byte-length validation form a reusable recipe for any mesh collection, not just anatomy.
- **Anatomy your agents can query.** The optional WebMCP tools expose search and inspection programmatically, opening the door to LLM-driven exploration of the atlas in compatible browsers.
- **A reproducible pipeline for future data.** When a new BodyParts3D release lands, the convert, optimize, compress, and validate scripts regenerate the entire dataset with the same guarantees.
- **A small, honest codebase.** Roughly seven application files carry the whole product, the README's numbers match the validators' assertions, and its limitations are stated in plain language right in the interface.

## Usage

Getting the explorer running locally takes two commands. The README requires Node.js 22.13 or newer, and no API keys or accounts are needed:

```sh
npm ci
npm run dev
```

Open `http://localhost:3016` to explore the atlas. To produce a static build for deployment, run the build and find the output in `dist/`:

```sh
npm run build
```

The repository also includes its own validation suite, which checks mesh buffers, names and concept membership, exploded-layout packing at desktop and mobile aspect ratios, and the search, inspection, and tap-handling contracts:

```sh
npm run check
node scripts/validate-atlas.mjs
node scripts/validate-interactions.mjs
npm run build
```

Rebuilding the geometry from the official BodyParts3D archive is optional — the repository already ships browser-ready chunks — but the full path is documented: obtain the official BodyParts3D OBJ archive and English metadata tables, prepare the joined concepts and display-system mappings, run `scripts/convert-anatomy.py`, then `node scripts/optimize-anatomy.mjs` and `node scripts/compress-models.mjs`.

## Conclusion

Human Atlas is a reminder that the most instructive open-source projects are often the most focused ones. It takes one dataset, one rendering strategy, and one interaction model, implements them in a handful of readable TypeScript files, and then validates every claim it makes in its README. The rendering layer — merged batches, GPU state textures, shader injection, bounding-box picking — is the part worth studying and stealing for any large-mesh Three.js project; the data pipeline and validators are the part worth imitating for any scientific dataset you ship. Whether you arrive as a student of anatomy, a teacher assembling a lesson, or a graphics engineer chasing the trick to 2,234 pickable meshes, the tour through `ashemag/human-atlas` pays for itself.

Links:

- GitHub repository: [https://github.com/ashemag/human-atlas](https://github.com/ashemag/human-atlas)
- Live demo: [https://human-atlas-seven.vercel.app](https://human-atlas-seven.vercel.app)
- Anatomy data source and license: [BodyParts3D, The Database Center for Life Science](https://lifesciencedb.jp/bp3d/)
