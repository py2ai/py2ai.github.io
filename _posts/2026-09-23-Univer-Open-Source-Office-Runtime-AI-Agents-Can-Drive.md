---
layout: post
title: "Univer: The Open-Source Office Runtime AI Agents Can Drive"
description: "Univer is an open-source, isomorphic office SDK for spreadsheets, documents, and presentations - Canvas rendering, a formula engine, a plugin for every feature, and a headless Node.js mode built for agent infrastructure. We tour the architecture of the 15,000-star TypeScript monorepo and show how to embed it."
date: 2026-09-23
header-img: "img/post-bg.jpg"
permalink: /Univer-Open-Source-Office-Runtime-AI-Agents-Can-Drive/
tags:
  - AI
  - Agents
  - TypeScript
  - Open Source
  - SDK
author: "PyShine"
---
# Univer: The Open-Source Office Runtime AI Agents Can Drive

Spreadsheets won the office. Every serious application eventually grows a grid, and every grid eventually grows formulas, formatting, and a user base that will not tolerate a plain HTML table. [Univer](https://github.com/dream-num/univer) - 15,340 stars and 1,367 forks, Apache-2.0 licensed, TypeScript end to end - is an open-source SDK that gives you the whole office surface, not just the grid: spreadsheets, documents, presentations, Canvas-drawn images, relational tables, and PDF, composable inside your own product. The project has been building since 2022, but its current tagline is the tell: an office harness for AI agents. That is not marketing gloss. The architecture genuinely splits into a UI you can embed and a headless runtime you can run in Node.js, built on the same code - which means an agent can create workbooks, run formulas, and produce documents with the exact engine your users see in the browser. In this tour we walk the real architecture of the monorepo, then show the fastest path from install to a working embedded app. It slots neatly next to the rest of the agent stack we have covered recently: [orchestrators that run agent workloads at cluster scale](https://pyshine.com/AX-Kubernetes-Thinking-For-AI-Agent-Workloads/), [frameworks that build applications around agents](https://pyshine.com/Agent-Native-BuilderIO-Framework-Builds-Apps-Around-Agents/), and [rendering engines that let models paint UI](https://pyshine.com/json-render-Generative-UI-Framework-Keeps-AI-on-a-Leash/).

![Architecture overview of the Univer repository showing the integration surface, core platform, and capability plugins](/assets/img/diagrams/univer/univer-overview-architecture.svg)

## Why You Need This

Embedding office capability usually means one of two bad options: iframe a hosted app you do not control, or spend a quarter wiring a heavyweight component suite into your product. Univer offers a third path, and the repository's structure is the argument. It is a pnpm monorepo with roughly fifty-six packages under `packages/`, and every capability - filtering, conditional formatting, comments, hyperlinks, drawing, number formats, data validation - ships as its own plugin, usually with a separate UI twin. You compose exactly what you need, lazy-load what is expensive, and replace what does not fit. Three integration modes match three postures. Preset Mode hands you curated plugin bundles so a working spreadsheet app is about ten lines of code. Plugin Mode gives you manual control over package selection, registration order, and locale merging when bundles are too coarse. Headless Mode runs the same architecture in Node.js with no UI at all, which is the mode that matters for AI infrastructure: workbook processing, formula calculation, and document generation become server-side calls an agent can make. And because the whole thing renders to Canvas with a dedicated formula engine underneath, complex workbooks stay responsive at sizes where DOM-based grids collapse. For teams building AI products that need to produce, edit, or analyze office documents, this is the layer that turns a language model's output into something a finance team can actually open.

## How It Works

The [architecture guide](https://docs.univer.ai/guides/recipes/architecture/univer) describes the mental model, and the source tree backs it: one core platform, two engines, and a fleet of domain plugins hanging off a plugin registry.

![Detailed architecture diagram of the Univer monorepo from the repository source](/assets/img/diagrams/univer/univer-architecture.svg)

The foundation is `@univerjs/core`. A `Univer` instance hosts a dependency-injected plugin system (`packages/core/src/services/plugin`), and each plugin registers commands and mutations - the command service in `packages/core/src/services/command` is the transaction layer every operation flows through. Around it sit the operational services: an instance service that manages units, an undo-redo service that records command history, a permission service that gates operations on permission points, plus locale, lifecycle, and resource loading. On top of the foundation sit two engines. The render engine (`packages/engine-render`) draws everything - render managers, shapes, components, floating objects - to Canvas, shared across document types. The formula engine (`packages/engine-formula`) owns calculation: a dependency graph, a function library registered into the engine, and the controllers that decide when a dirty cell recalculates. The domain plugins then give you the actual applications. `@univerjs/sheets` carries the spreadsheet models, commands, controllers, and the skeleton generation that maps data to the render engine; `@univerjs/docs` and `@univerjs/slides` do the same for word processing and presentations. Feature plugins extend them - `sheets-formula` wires the formula engine into worksheets, `drawing` shares images and shapes across Sheets and Docs, and dozens more follow the same pattern. Integration happens through three doors. The [Facade API](https://docs.univer.ai/reference/classes/univer) - `FUniver` and per-package facades - is the unified surface for workbooks, ranges, formulas, commands, and events, identical in browser and Node.js. The UI shell (`@univerjs/ui`, built on the `@univerjs/design` component library) provides menus, toolbars, and panels, with Vue 3 and Web Component adapters for frameworks outside React. And the server side is real, not an afterthought: `rpc` and `rpc-node` packages provide an isomorphic remote procedure channel, with a shared `network` package underneath, which is what collaborative and headless flows ride on.

## Advantages

- **Isomorphic by design.** The same architecture runs the browser UI and headless Node.js processing, so an agent and a user operate the same document model.
- **Plugin-shaped.** Every capability is a composable plugin that can be added, removed, replaced, or lazy-loaded - you ship what your product needs, nothing more.
- **Canvas rendering at scale.** Large editable surfaces stay responsive because rendering never goes through the DOM.
- **A real formula engine.** Dependency tracking and a function library live in their own engine, not bolted onto the grid.
- **Framework-flexible UI.** React by default, with Vue 3 and Web Component adapters, all fed by one design system.
- **Fast on-ramp.** Presets turn a working Sheets, Docs, or Node setup into a few imports; the [showcase](https://docs.univer.ai/showcase) proves the ceiling.

## Benefits

The payoff compounds as your product grows. Starting with [preset bundles](https://github.com/dream-num/univer-presets) means day one is a working app, not a build system; moving to Plugin Mode later is a re-composition, not a rewrite, because everything was already plugin-shaped. Keeping all `@univerjs/*` packages on one version keeps that story simple. For AI products specifically, headless mode is the unlock: [server-side workbook processing](https://docs.univer.ai/guides/sheets/getting-started/node) lets an agent read a spreadsheet, calculate with the real engine, write results, and export - without a browser, a screenshot, or a fragile macro recorder. And because the agent's headless calls and your user's browser edits hit the same command layer, what the agent did is always visible, auditable, and undoable in the UI your team already knows. That closes the loop between autonomous work and human review - the same separation [PI-Desktop draws on the desktop](https://pyshine.com/PI-Desktop-Open-Source-Desktop-Where-AI-Agents-Get-Their-Own-Workspace/), here drawn across the client-server boundary.

## Usage

For most products, start with Preset Mode, following the [installation guide](https://docs.univer.ai/guides/sheets/getting-started/installation):

```bash
pnpm add @univerjs/presets @univerjs/preset-sheets-core
```

```ts
import { UniverSheetsCorePreset } from '@univerjs/preset-sheets-core'
import UniverPresetSheetsCoreEnUS from '@univerjs/preset-sheets-core/locales/en-US'
import { createUniver, LocaleType, mergeLocales } from '@univerjs/presets'

import '@univerjs/preset-sheets-core/lib/index.css'

const { univerAPI } = createUniver({
  locale: LocaleType.EN_US,
  locales: {
    [LocaleType.EN_US]: mergeLocales(UniverPresetSheetsCoreEnUS),
  },
  presets: [
    UniverSheetsCorePreset({
      container: 'app',
    }),
  ],
})

univerAPI.createWorkbook({})
```

Give the page a container and the spreadsheet is live:

```html
<div id="app" style="height: 100vh"></div>
```

From there, the Facade API is the working surface - `univerAPI` creates workbooks, reads and writes ranges, registers formulas, and listens to events through one consistent API. When presets are too coarse, switch to Plugin Mode: install the individual `@univerjs/*` packages you need and call `registerPlugin` for each, controlling order and lazy loading yourself - the repository's `examples/` directory shows complete compositions. For headless use in Node.js, skip the UI packages entirely and drive the same Facade from a script or an agent. One rule keeps every path sane: keep all `@univerjs/*` packages on the same version.

## Conclusion

Univer is what happens when a team spends years building an office suite and then points it at the agents that will operate it: one isomorphic runtime, Canvas rendering, a real formula engine, and a plugin for everything - with the headless mode to prove the architecture was never just about pixels. The [repository](https://github.com/dream-num/univer) is active, the [documentation](https://docs.univer.ai) is thorough, and the integration ladder from preset to plugin to headless means you never outgrow it. If your product needs an office surface, or your agents need one to work in, this is the open-source foundation worth building on.
