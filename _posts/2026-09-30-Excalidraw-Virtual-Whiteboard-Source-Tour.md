---
layout: post
title: "Excalidraw: Virtual Whiteboard Source Tour - Inside excalidraw/excalidraw"
description: "A guided source-tour of excalidraw/excalidraw, the open-source virtual whiteboard with a hand-drawn look. We walk the element model, the dual-canvas render loop, delta-based undo/redo, the end-to-end encrypted realtime layer, and the PNG/SVG export pipeline."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Excalidraw-Virtual-Whiteboard-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/excalidraw/excalidraw-excalidraw-architecture.svg
tags:
  - Excalidraw
  - Whiteboard
  - Canvas
  - TypeScript
categories: [AI, Open Source]
keywords: "excalidraw, virtual whiteboard, hand-drawn diagrams, canvas rendering, React, TypeScript, realtime collaboration, end-to-end encryption, undo redo, fractional indexing, roughjs, open source, architecture, source code tour"
author: "PyShine"
---

Sketching a system diagram with your team usually means one of two things: a heavyweight design tool that fights your freehand instincts, or a throwaway screenshot nobody can edit afterwards. Excalidraw takes a third path — an infinite, canvas-based whiteboard where every rectangle and arrow deliberately looks like it was drawn by hand, yet the whole scene is a structured, versioned data model that any machine can read. The project describes itself simply as "an open source virtual hand-drawn style whiteboard, collaborative and end-to-end encrypted," and that one sentence hides a great deal of careful engineering.

What makes the repository unusually interesting is that it is not one app but a yarn-workspaces monorepo. The `packages/excalidraw` workspace is the npm-published `@excalidraw/excalidraw` package — literally "Excalidraw as a React component" — while `excalidraw-app` is the production website at excalidraw.com built on top of it. Around them sit small, self-contained packages: `@excalidraw/element` for the scene data model, `@excalidraw/math` for geometry, `@excalidraw/common` for shared utilities, plus `fractional-indexing` and `laser-pointer` as standalone libraries. Everything is written in TypeScript with React on the view side, and the whole tree ships under the MIT license.

That split is exactly why the source is worth a tour. If you have ever wondered how a canvas editor reconciles multiplayer edits, how undo/redo survives distributed clients, or how a whiteboard exports pixel-perfect SVG from the same code path that paints the screen, this codebase answers each question in a clearly separated module. In the rest of this post we walk the repository from the element model outward: rendering, history, collaboration, and export.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/excalidraw/excalidraw-excalidraw-overview-architecture.svg" alt="Architecture overview of the excalidraw/excalidraw repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the excalidraw/excalidraw architecture: the app shell, the editor core, the rendering passes, the data and export layer, and the realtime collaboration stack.*

Reading the overview from left to right: the `excalidraw-app` shell boots the React `App` component and the collaboration controller, then wraps the editor package's own `App` component from `packages/excalidraw/components/App.tsx`. That component is the beating heart — it dispatches user intent into the `actions` registry, mutates the element model in `packages/element/src`, commits changes through the store, and asks the `scene`/`renderer` modules to repaint. On the lower path, the `data` layer in `packages/excalidraw/data` restores and serializes scenes and drives exports, while the collab controller reconciles remote edits and pushes encrypted state into the `excalidraw-app/data` storage backends.

## Why You Need This

The first problem Excalidraw solves is friction. Design tools optimize for precision, which makes quick communication sketches slow and intimidating. Excalidraw inverts that: the hand-drawn aesthetic comes from rendering every shape through roughjs with a per-element random `seed`, so the wobble is stable across renders but the tool never pretends to be a CAD package. An infinite, zoomable canvas with grid, panning, arrow-binding, and a wide tool set (rectangle, diamond, line, free-draw, eraser, frames, and more) keeps the interaction model as light as a napkin.

The second problem is data ownership. Scenes are not saved in a proprietary blob — the `.excalidraw` file format is an open JSON document whose schema is even documented in the repository's dev-docs (`dev-docs/docs/codebase/json-schema.mdx`). Because elements are plain typed objects, other tools can generate, consume, and diff them; the repo itself ships `mermaid.ts` plus the `@excalidraw/mermaid-to-excalidraw` dependency to convert Mermaid diagrams into editable elements. Your drawings outlive any particular vendor relationship.

The third problem is collaboration without surveillance. The hosted app offers realtime rooms, but the server never learns what you drew: everything that leaves the browser — socket frames and the persisted scene snapshot alike — is encrypted with AES-GCM under a key the server does not hold. For teams, that means a shared scratchpad with confidentiality guarantees that are enforced in the client code you can read, not just in a privacy policy.

Finally, it solves the integration problem. Because the editor is an npm React component, embedding a diagramming surface into documentation portals, note apps, or internal tools is a package install away — the README lists integrations and users ranging from VS Code extensions to Google Cloud, Obsidian, Notion, and Replit. The same components power all of them.

## How It Works

The cleanest way to understand the codebase is to follow one element from creation to paint to sync. The detailed graph below lays out the modules involved.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/excalidraw/excalidraw-excalidraw-architecture.svg" alt="Detailed architecture of the excalidraw/excalidraw repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of excalidraw/excalidraw: the element model and store feed the React editor, which paints through scene renderers, persists via the data layer, and syncs through the encrypted collaboration stack.*

### Understanding the Architecture

**The element model is flat, immutable, and versioned.** Every shape on canvas is an `ExcalidrawElement` whose base type lives in `packages/element/src/types.ts`: geometry and style fields (`x`, `y`, `width`, `angle`, `strokeColor`, `roughness`, ...), plus the bookkeeping that makes multiplayer and undo possible. `version` is a monotonically incremented integer bumped on each change, `versionNonce` is a random tie-breaker regenerated per change, `index` is a fractional-index string (of the form used by the rocicorp fractional-indexing library) that encodes z-order, and `isDeleted` implements soft deletion so history and sync can reason about tombstones. Element factories in `packages/element/src/newElement.ts` assign the roughjs `seed` via `randomInteger()` so the hand-drawn jitter is per-element and deterministic. Mutations never edit in place — `mutateElement.ts` and `newElementWith` produce new element objects and notify subscribers.

**The scene is a thin observable index over those elements.** The `Scene` class in `packages/element/src/Scene.ts` keeps the non-deleted element array plus a lookup map, exposes selection queries, and — crucially for multiplayer — keeps array order and fractional indices in lockstep with `syncMovedIndices`/`syncInvalidIndices` (with a throttled validator that runs in dev and test). The editor component in `packages/excalidraw/components/App.tsx` instantiates `Scene` and pairs it with the `Renderer` from `packages/excalidraw/scene/Renderer.ts`, which computes the visible element set from the viewport with memoized culling so a huge board never repaints off-screen content.

**Painting is split across two canvases and throttled to the animation frame.** The React tree mounts `StaticCanvas` and `InteractiveCanvas` from `packages/excalidraw/components/canvases/`; each is memoized on a `canvasNonce` and relevant app-state props, so only meaningful changes reach the drawing code. A render then flows into `packages/excalidraw/renderer/staticScene.ts` (or `interactiveScene.ts` for selection boxes, handles, and other ephemera), where `renderStaticSceneThrottled` wraps the pass in the `throttleRAF` helper from `packages/common/src/utils.ts` — a `requestAnimationFrame`-based throttle. Actual element drawing goes through `packages/element/src/renderElement.ts`, which delegates to roughjs for the sketchy strokes. There is no single imperative game loop; the "render loop" is a composition of React re-renders, memoization nonces, and one rAF-throttled paint per canvas.

**History is a stack of deltas, not full snapshots.** All changes funnel through the `Store` in `packages/element/src/store.ts`, which snapshots scene state, computes `StoreDelta` objects via `packages/element/src/delta.ts`, and lets callers declare intent with `CaptureUpdateAction.IMMEDIATELY`, `NEVER`, or `EVENTUALLY` (so drags stay ephemeral until they commit). The `History` class in `packages/excalidraw/history.ts` keeps `undoStack`/`redoStack` arrays of `HistoryDelta` objects; applying a delta re-inflates the inverse patch while deliberately excluding `version` and `versionNonce`, so every undo/redo becomes a fresh user action that collaborates correctly with remote peers.

**The realtime layer is encryption-first, reconciliation-second.** In `excalidraw-app`, `collab/Collab.tsx` owns the session and `collab/Portal.tsx` owns the socket.io connection. `Portal._broadcastSocketData` serializes every outgoing message and encrypts it with `encryptData` from `packages/excalidraw/data/encryption.ts` — an AES-GCM envelope from the Web Crypto API with a fresh 12-byte IV. Incoming remote elements run through `reconcile.ts` (`packages/excalidraw/data/reconcile.ts`): an element being actively edited locally wins, otherwise the higher `version` wins, and identical versions are broken deterministically by the lowest `versionNonce`. The reconciled set is re-ordered by fractional index, which is exactly why z-order survives concurrent edits from different clients.

**Persistence and export share the same data plumbing.** Locally, `excalidraw-app/data/LocalData.ts` splits state between `localStorage` and IndexedDB (via `idb-keyval`, with an image store named `files-db`) and prunes stale binary blobs by their `lastRetrieved` timestamp. Remotely, `excalidraw-app/data/firebase.ts` stores a room as `{ sceneVersion, iv, ciphertext }` in Firestore — ciphertext only, matching the socket encryption. Export is the mirror image of restore: `packages/excalidraw/data/index.ts` prepares the selection (with frame-aware element collection in `prepareElementsForExport`), `packages/excalidraw/scene/export.ts` drives `exportToCanvas`/`exportToSvg` reusing the static renderers including `packages/excalidraw/renderer/staticSvgScene.ts` for the SVG backend, and `data/json.ts` serializes the open `.excalidraw` format that `filesystem.ts` hands to the browser's file-save dialog.

Put together, the end-to-end flow reads: a pointer event creates or mutates an element through the action handlers and `mutateElement`; the scene notifies subscribers and the store commits a delta; React re-renders the memoized canvas components; the rAF-throttled render pass repaints only the visible set; and the collab layer broadcasts the encrypted version bump while persistence quietly autosaves. Every arrow in the detailed diagram is one link in that chain.

## Advantages

- **Dual-canvas rendering with viewport culling.** The `Renderer` in `packages/excalidraw/scene/Renderer.ts` memoizes the visible set, and static versus interactive content paint on separate canvases, keeping interaction latency independent of scene size.
- **Delta-based undo/redo that plays well with multiplayer.** Because `HistoryDelta` patches exclude version metadata, undoing locally still produces a correctly versioned change for other clients.
- **Deterministic conflict resolution.** The `version`/`versionNonce` tie-break and fractional-index ordering in `data/reconcile.ts` mean concurrent edits converge to the same scene on every participant's machine.
- **End-to-end encryption by construction.** AES-GCM envelopes from `data/encryption.ts` wrap both socket traffic and the Firestore snapshot; the coordination server only ever relays ciphertext.
- **Local-first storage.** IndexedDB-backed autosave in `LocalData.ts` keeps drawings available offline and survives reloads without an account.
- **A reusable component, not just a website.** The published package targets React 17, 18, and 19 as peer dependencies, so the editor drops into virtually any modern React app.

## Benefits

- **Open, documented file format.** The `.excalidraw` JSON schema is versioned and documented in-repo, making scenes diffable, scriptable, and portable.
- **MIT license throughout.** Both the editor package and the app shell carry permissive licensing, so embedding and self-hosting carry no legal friction.
- **Modular monorepo.** `@excalidraw/element`, `@excalidraw/math`, `@excalidraw/common`, `laser-pointer`, and `fractional-indexing` are independently buildable packages (`yarn build:packages`), so you can reuse the geometry or indexing layers without the whole editor.
- **Proven at scale.** The same code that ships on excalidraw.com — PWA support, realtime rooms, shareable read-only links — is the code you read, with the app features documented directly in the README.
- **Mermaid-to-whiteboard pipeline.** The bundled `@excalidraw/mermaid-to-excalidraw` dependency converts text-based diagrams into native, editable elements.
- **Self-hostable app.** The repository includes a `Dockerfile` and `docker-compose.yml` for running the app shell yourself, keeping collaboration data under your control.

## Usage

To embed the editor in your own React application, install it with npm or yarn (React itself is a peer dependency):

```bash
npm install react react-dom @excalidraw/excalidraw
# or
yarn add react react-dom @excalidraw/excalidraw
```

To run the repository itself for development, the monorepo root exposes workspace scripts (`package.json`), with the app dev server living in `excalidraw-app`:

```bash
yarn          # install workspace dependencies
yarn start    # run the excalidraw-app dev server
yarn build    # production build of the app
```

Full development setup notes — including the required Node version (18 or newer) and environment configuration — are maintained in the project's Development Guide at docs.excalidraw.com.

## Conclusion

Excalidraw is a rare example of a playful product built on a serious architectural spine: an immutable, versioned element model; observable scenes feeding memoized, rAF-throttled canvas renderers; delta-based history; and an encrypted collaboration layer whose server never sees plaintext. Reading `packages/element/src/types.ts` alongside `packages/excalidraw/renderer/staticScene.ts` and `excalidraw-app/collab/Portal.tsx` is a compact masterclass in how client-heavy, privacy-respecting realtime apps are actually structured. If your next project needs a whiteboard, or you just want to study well-factored canvas application code, this repository rewards the climb.

Links:

- GitHub repository: [excalidraw/excalidraw](https://github.com/excalidraw/excalidraw)
- Editor: [excalidraw.com](https://excalidraw.com)
- Documentation: [docs.excalidraw.com](https://docs.excalidraw.com)
