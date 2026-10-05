---
layout: post
title: "M3E Canvas: Sketch Material 3 Screens Into AI Prompts - Inside lnkiai/m3e-canvas"
description: "M3E Canvas is a browser editor for Material 3 Expressive screens: drag and drop 36 real M3 parts, link and preview flows, theme everything, then export the design as a natural-language prompt or a share link for AI coding tools. A source tour of the React architecture behind it."
date: 2026-10-05
header-img: "img/post-bg.jpg"
permalink: /M3E-Canvas-Sketch-Material-3-Screens-Into-AI-Prompts/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/m3e-canvas/lnkiai-m3e-canvas-architecture.svg
tags: [Material Design, React, Next.js, AI Tools]
categories: [AI, Open Source]
keywords: M3E Canvas, Material 3 Expressive, UI prototyping, prompt engineering, Next.js, React, vibe coding, design to code
author: "PyShine"
---

Every AI coding tool is only as good as the brief you give it. Describe a screen in prose and you get a generic
guess; show it exactly which components go where, with real Material 3 Expressive styling, and the result is far
closer to what you imagined. M3E Canvas, an MIT-licensed project with more than 8,000 stars, lives exactly in
that gap: it is a browser editor where you sketch Android and web screens from real Material 3 Expressive parts,
connect them into flows, tap through a live preview, and then copy the whole design as a concise natural-language
prompt for Claude Code, Codex, Gemini CLI, Cursor or any other AI coding tool.

What makes the project unusual is its discipline. The parts are not approximations - the shape-morphing loading
indicator is ported from Google's own Android implementation, the color system generates a full Material 3
scheme from a single seed color, and buttons interpolate correctly between the five named M3 sizes instead of
snapping. The app is a static Next.js export with no backend at all: your design lives in localStorage, travels
in a URL fragment, and validates itself every time it is opened.

In this tour we walk the real source tree: the data model that treats a design as one validated JSON document,
the renderer that draws 36 component kinds to spec, the prompt builder that turns geometry back into words, and
the agent workflow that lets a coding tool sketch the design for you. Every path cited below exists in the
repository today.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/m3e-canvas/lnkiai-m3e-canvas-overview-architecture.svg" alt="Architecture overview of the M3E Canvas repository, from the Next.js app shell through the editing panels and data model to the prompt, share and AI outputs" style="min-width:720px;width:100%;max-width:1100px;" />
</div>
*Architecture overview of the repository: the editor shell hosts the editing panels, every panel reads and writes the shared design document, and the document flows out as a prompt, a share link or an AI-assisted rewrite.*

Reading the overview from left to right:

- **App shell.** `app/page.tsx` mounts the editor in `app/Editor.tsx`, which owns the canvas, undo and redo,
  keyboard shortcuts and the panel layout.
- **Canvas and editing.** You add parts from `components/PartsPalette.tsx`, the renderer in
  `components/M3Node.tsx` draws each one to Material 3 geometry, and `components/Inspector.tsx` edits whatever
  is selected. `components/Layers.tsx` manages z-order and groups, and `components/Preview.tsx` lets you tap
  through the screens you have linked.
- **Design data model.** `lib/tokens.ts` defines the whole document - frames, groups, items, 36 part kinds -
  plus the geometry tables every component consults. `lib/theme.ts` carries the active theme and fonts.
- **Prompt, share and AI.** When the sketch is done, `lib/prompt.ts` turns the document into a natural-language
  brief, `lib/share.ts` packs it into a URL fragment, and `lib/ai.ts` optionally polishes behavior notes with
  your own model key.

## Why You Need This

Handing an AI tool a vague description is the slowest way to build a UI. The model invents a layout, you correct
it, it invents another, and the loop repeats. M3E Canvas short-circuits that loop by letting you compose the
exact screen first - with correct Material 3 Expressive components, not boxes - and then speak to the AI in its
own most reliable format: a precise, unambiguous description of what is on screen.

The prompt output is engineered, not dumped. It describes overlaps and side-by-side rows explicitly so the
generated layout keeps them, writes your per-part behavior notes into the brief, and targets either Android
(the default) or the web, asking for the matching stack. Same-named screens in different sizes are written as
one screen at two widths, which is exactly how a responsive design should be described.

The second use case is collaboration and agency. Because a design is one JSON document that packs into a link,
you can send a live sketch to anyone - or to an AI coding agent, which reads the project's agent brief, writes
the document itself, and replies with a link that opens on your canvas. Design review stops being a screenshot
exchange and becomes a URL.

## How It Works

The detail view below follows the document through the editor: how it is validated, rendered, themed, tidied and
finally serialized out.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/m3e-canvas/lnkiai-m3e-canvas-architecture.svg" alt="Detailed architecture of M3E Canvas: app shell, editing panels, renderer, design data model, and the prompt, share and AI outputs" style="min-width:760px;width:100%;max-width:1200px;" />
</div>
*Detailed architecture of the application: the editor shell wires the panels to a validated document model, the renderer and preview share one part tree, and three outputs - prompt, share link and AI helper - consume the same data.*

### Understanding the Architecture

**One document is the whole truth.** `lib/tokens.ts` defines `Doc`: frames (screens), groups and items, with a
catalog of 36 part kinds - from buttons, FABs and chips through app bars, navigation rails, dialogs and text
fields to carousels, date pickers and the loading indicator. Every item carries its label, icon, variant
(filled, tonal, elevated, outlined or text), tabs, selection state and an optional behavior note. Validation
lives in `lib/project.ts`, whose header states the philosophy plainly: a project file is the Doc as JSON,
nothing more - reading one back only checks the shape the editor relies on, and the same migrations that run on
a saved document bring older files up to date. That same validation is what makes share links and agent-written
documents safe to open.

**Geometry is tables, not magic numbers.** The M3 Expressive button scale is encoded in `lib/tokens.ts` as five
named sizes from 32 to 136 dp, and `buttonMetrics` interpolates every measure - height, padding, gap, icon and
font - proportionally between neighboring stops, so a slider keeps buttons M3-shaped the whole way. Phone
frames are 412 by 892, desktop frames 1280 by 800, and `lib/rail.ts` plus `lib/railView.ts` convert a bottom
navigation bar into a side rail and back, re-laying-out the parts beside it. `components/M3Node.tsx` consumes
these tables to render each part; the preview in `components/Preview.tsx` reuses the identical part tree, with
`components/TapStage.tsx` playing tap and swipe transitions and `components/CardStage.tsx` handling carousels.

**The theme is generated, not picked from a screenshot.** `lib/color.ts` implements `schemeFromSeed`: one seed
color becomes a full Material 3 scheme you can fine-tune, with light and dark modes and three contrast levels.
`lib/theme.ts` provides the theme context and loads Google fonts - Roboto, Roboto Flex, Roboto Serif or the
system font - on first use, refreshing measured widths once the face is ready. The loading indicator is the
crown jewel: `lib/shapes.ts` holds SVG path data for seven official shapes that morph in the sequence soft
burst, cookie nine, pentagon, pill, sunny, cookie four, oval - 180 sampled points per shape - mirroring the
animation model of Google's Android implementation, and `components/Loading.tsx` renders it.

**Tidying is rule-based, deliberately.** `lib/tidy.ts` states in its header that nothing there is guessed by a
model: parts of one connectable family that touch fuse into a connected run the same way the magnetic drop
works, bars snap to the edges they belong to, the FAB takes the bottom-right corner, dialogs center, and
everything else stacks on the 16 dp layout margins. Press the button again and it undoes. This is why the
magnetic connections in the canvas feel predictable - they are the same join rules the tidy pass applies.

**Three outputs share one model.** The prompt builder in `lib/prompt.ts` walks the document and produces a
concise brief in Japanese, English, Chinese or Korean via `lib/i18n.ts` - every user-facing string in the app
exists in all four languages, and a parity test enforces it. `lib/share.ts` serializes the design as JSON,
deflates it and base64url-encodes it after a fragment marker (plain JSON for tools that cannot compress), so
the static site stays static - the fragment never reaches a server. The shareable form deliberately strips
picked images and AI rewrite history. And `lib/ai.ts` is the optional helper: you bring a key for OpenAI,
Claude, Gemini or DeepSeek, it is stored only in your browser, requests go straight to the provider, every
action uses a fixed prompt with a fixed JSON answer shape, the result is applied only after you have reviewed
it, and the model never touches coordinates.

**The agent loop is documented, not improvised.** `public/agent.md` is a brief for coding agents: it explains
that a design is one JSON document, instructs the agent to reply with a share link (a short Node script in the
brief deflates the file), and tells it not to bother verifying its own output - the app validates on open and
reports problems to the person. It is a thoughtful pattern for any tool that wants AI agents to produce
artifacts for it.

End to end: you drag parts from the palette, `M3Node` draws them from the geometry tables, the inspector writes
changes back into the document, the tidy pass keeps the layout honest, and when you copy the prompt, the builder
walks the same document and hands your AI coding tool a brief that reads like a senior designer's spec.

## Advantages

- **Real Material 3 Expressive, not a lookalike.** Component geometry, the loading indicator morph and the
  color system follow Google's specification, ported where necessary from the Android source.
- **The prompt is structured for code generation.** Overlaps, rows, behavior notes and screen-size pairs are
  spelled out, which is precisely what coding models get wrong when left to guess.
- **No backend and no lock-in.** The app is a static export; designs live in localStorage and travel in URL
  fragments as plain JSON anyone can inspect.
- **Validated everywhere.** The same document validator guards localStorage saves, opened files, share links
  and agent-written sketches, with migrations for older files.
- **Optional AI that respects the author.** Bring your own key, fixed prompts, reviewable results, and the
  layout coordinates are never handed to the model.
- **Four languages throughout.** Every string exists in English, Japanese, Chinese and Korean, enforced by a
  parity test - rare discipline for an open-source UI tool.

## Benefits

- **Faster loop from idea to working screen.** Sketching with correct components beats correcting an AI's
  invented layout, and the resulting prompt lands far closer on the first try.
- **Works with every AI coding tool.** The output is a plain text prompt or a link - no plugin, no integration,
  no vendor coupling.
- **Runs anywhere a browser runs.** The static site is hosted on GitHub Pages and can be self-hosted under any
  sub-path with a single build-time variable.
- **Design and review in one artifact.** Share links open the live editor on the recipient's machine, so
  feedback happens on the real thing instead of a picture.
- **A clean React codebase to learn from.** The split between tokens, validation, rendering and prompt
  generation is a textbook example of keeping a design tool's data model independent of its UI.
- **Tested where it matters.** The lib layer carries a suite of tests covering validation, prompts, color,
  shapes, rail conversion, tidy rules and string parity.

## Usage

The fastest way in is the hosted demo at `https://lnkiai.github.io/m3e-canvas/` - open it, drag parts, link two
screens, press P and tap through the flow. To run it locally:

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # static export to ./out
```

The build is a static Next.js export; to host it under a sub-path such as a GitHub Pages project site, set
`NEXT_PUBLIC_BASE_PATH=/your-repo` at build time, which is what the repository's deploy workflow does.

The daily workflow is three steps. First, sketch: pick parts from the palette, group them, set a theme from a
seed color, and connect buttons or list items to target screens with a transition. Second, verify: press P and
tap through the preview, checking that back navigation plays transitions in reverse. Third, export: open the
prompt panel, choose the language and the Android or web target, and copy the brief into Claude Code, Codex,
Gemini CLI or Cursor.

To hand a design to someone - or to an agent - use the share menu: copy a link that opens the design on their
canvas, or copy the agent instruction set that tells a coding tool how to write the document itself and reply
with a link. Keyboard workers get the essentials: V and H switch select and hand tools, plus, minus and zero
zoom, Control-D duplicates, arrows nudge, and Control-Z undoes.

## Conclusion

M3E Canvas understands something important about AI-assisted development: the highest-leverage place to spend
effort is the description of what you want, not the correction of what you got. By building a proper editor on
top of real Material 3 Expressive components - with a validated JSON document at its core and three well-designed
exits (prompt, link, agent brief) - the project turns visual design into a first-class input for coding agents.

It is also simply a well-made tool: fast, offline-capable, four languages, no account, no backend. If you build
Android or web UIs with AI tools, put it in your bookmarks; if you build design tools, put the source on your
reading list.

**Links:**

- Repository: <https://github.com/lnkiai/m3e-canvas>
- Live demo: <https://lnkiai.github.io/m3e-canvas/>
- License: <https://github.com/lnkiai/m3e-canvas/blob/main/LICENSE>
