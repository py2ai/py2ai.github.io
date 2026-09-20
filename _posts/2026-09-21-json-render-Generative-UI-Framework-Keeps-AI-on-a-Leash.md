---
layout: post
title: "json-render: Vercel's Generative UI Framework That Keeps AI on a Leash"
description: "json-render from Vercel Labs is a generative UI framework where AI writes only the JSON spec and your pre-approved components do the rendering - a deep dive into its catalog, SpecStream, renderers, and MCP integration."
date: 2026-09-21
header-img: "img/post-bg.jpg"
permalink: /json-render-Generative-UI-Framework-Keeps-AI-on-a-Leash/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - TypeScript
  - React
  - AI
  - Generative UI
  - Vercel
author: "PyShine"
---
# json-render: Vercel's Generative UI Framework That Keeps AI on a Leash

Letting an AI model draw your application's interface sounds like a recipe for chaos: hallucinated components, invalid markup, and output that changes shape on every request. [json-render](https://github.com/vercel-labs/json-render) from [Vercel Labs](https://vercel.com/labs) takes a different bet - the model never draws anything. Instead, you define a catalog of approved components and actions with typed props, the model generates a JSON specification constrained to that catalog, and your own battle-tested components do the actual rendering. The result is generative UI with the reliability of hand-written code: the AI is creative only inside the guardrails you set. With over 17,000 stars, an Apache-2.0 license, and 33 published packages, it is one of the most complete takes on this pattern we have covered - and a natural successor to the ideas in our earlier post on [OpenGenerativeUI](https://pyshine.com/CopilotKit-OpenGenerativeUI-Open-Source-Generative-UI-Framework/). This post digs into how it actually works.

![High-level architecture overview of the json-render repository](/assets/img/diagrams/json-render/json-render-overview-architecture.svg)

## Why You Need This

Two extremes dominate AI-built interfaces today. The first is freeform generation: the model writes raw HTML or JSX, which is flexible but fragile - invalid output, XSS-shaped risks, and layouts that break on the second render. The second is rigid templating: fixed dashboards where the model only fills in numbers, which is safe but barely generative. json-render occupies the pragmatic middle ground we keep encountering in production-grade AI tooling: the model composes freely, but from a vocabulary you control.

The practical wins are immediate. Because component props are validated against [Zod](https://zod.dev) schemas, malformed output fails loudly instead of rendering garbage. Because the spec is plain data, it can be streamed, diffed, logged, replayed, and audited. Because the same catalog drives renderers for React, Vue, Svelte, Solid, React Native, terminals, PDFs, video, and email, one AI integration covers every surface your product ships on. And because the model is constrained, latency drops too - generating a compact element tree is far cheaper than generating a full page of code, a principle that also powers our [React and Next.js tutorial](https://pyshine.com/Learn-React-Next-js-in-One-Post-Complete-Tutorial-Components-Hooks-Server-Components-Quick-Start/) approach of separating what renders from what describes.

## How It Works

The diagram below maps the repository's main components and how they connect.

![Detailed architecture of the json-render repository](/assets/img/diagrams/json-render/json-render-architecture.svg)

**The core engine.** Everything starts with `defineCatalog`, where you declare components - a Card, a Metric, a Button - each with Zod-validated props and a description the model will read, plus named actions like export_report. The catalog compiler in `packages/core/src/prompt.ts` turns that definition into instructions for the model, so your catalog is simultaneously your API contract and your prompt. The model responds with a spec: a flat tree of elements, each carrying a type, props, and children. As tokens arrive, the SpecStream diff engine in `packages/core/src/diff.ts` turns the partial output into patches, and the spec validator in `packages/core/src/spec-validator.ts` checks every element against the catalog before it can render - unknown component types simply cannot pass. A concrete spec reads like a dictionary of elements: an id pointing at a root Card, whose children reference a Button by id, each entry carrying its type, props, and children list. That flat shape is what makes streaming, diffing, and partial rendering possible - the renderer can paint the card before the model has finished describing the button.

**Renderers for every surface.** The React renderer resolves each validated element against your registry, where real implementations of Card and Button live, and paints progressively as the stream lands. Sibling packages carry the same contract elsewhere: Vue, Svelte 5 with runes, SolidJS, React Native for mobile, Ink for interactive terminal UIs, and a [Next.js](https://nextjs.org) integration that turns specs into full apps with routes, layouts, and server rendering. The batteries-included group is remarkable: 36 pre-built shadcn/ui components, plus renderers that emit PDF documents with [React PDF](https://react-pdf.org), [Remotion](https://www.remotion.dev) videos, and HTML [email](https://react.email) from the same spec shape. A React Three Fiber renderer even generates 3D scenes, including gaussian splatting.

**State, actions, and tooling.** During rendering, dynamic props are evaluated through a state store contract with adapters for Redux, Zustand, Jotai, and XState, while user interactions flow back through the action system into an action observer that tracks outcomes. Around the core sit developer tools: framework-agnostic devtools with stream taps, a YAML wire format with a streaming parser for those who prefer indentation to braces, template directives for formatting and i18n, and an MCP Apps integration that lets json-render surfaces run inside Claude, ChatGPT, Cursor, and VS Code. The repository even documents its own AI workflow in an AGENTS.md, with per-package agent skills and dependency sources vendored for deep analysis.

## Advantages

The design compounds several advantages. Safety is structural rather than aspirational - a model physically cannot emit a component outside the catalog, which removes an entire class of prompt-injection and markup bugs. Performance benefits twice over: compact specs stream cheaply, and rendering happens in your existing components with no runtime code generation. Portability is real, not marketing - the identical catalog produces a web dashboard, a mobile screen, and a terminal view. The spec format is inspectable data, so debugging means reading JSON with devtools attached rather than reverse-engineering generated code. And the package boundaries are unusually disciplined: core has no React dependency, adapters are opt-in, and every public package shares one version synced from the core package. Smaller members round out the family: a TanStack Start renderer, an image renderer for social cards via Satori, and codegen utilities that convert specs back into source code.

## Benefits

For teams, that translates into faster AI features with fewer review gates. Product dashboards, onboarding flows, email campaigns, and support tools can accept natural-language input without surrendering design consistency, because the brand lives in your registry, not in the model's taste. Legal and security reviews get easier when the worst possible output is a validation error. Costs shrink since the model writes descriptions, not DOM. The TypeScript experience is first-class - catalogs produce typed registries, so a typo in a component name is a compile error. And the MCP integration puts your guardrailed UI inside the AI assistants people already use, which turns every chat client into a potential surface for your product without exposing your data layer.

## Usage

Install the core and the renderer for your framework - `npm install @json-render/core @json-render/react` - and follow the three-step loop from the README: define a catalog with Zod schemas, bind real components with `defineRegistry`, and render any spec with the Renderer component. Pair it with the [Vercel AI SDK](https://github.com/vercel/ai) for streaming from any provider, or wire the MCP package into Claude or Cursor for assistant-native surfaces. The [documentation site](https://json-render.dev) carries guides, API references, and per-package skills, while the repository's examples folder demonstrates dashboards, chat apps, a game engine, an email builder, and a full Next.js website builder. Each example is a runnable project, so the fastest way to internalize the catalog pattern is to open the dashboard example and extend it with one component of your own. Most renderers are separate installs, so you only ship what your product needs.

## Conclusion

json-render is the most considered answer yet to a question every AI product team faces: how much freedom should the model have? Its answer - all of the composition, none of the markup - turns generative UI from a demo trick into an engineering practice. The catalog-as-contract idea, the streaming spec architecture, and the multi-surface renderer family are all patterns worth studying even outside this framework. If your product needs an interface that adapts to users and still passes code review, this repository is where that reconciliation has been done most thoroughly.
