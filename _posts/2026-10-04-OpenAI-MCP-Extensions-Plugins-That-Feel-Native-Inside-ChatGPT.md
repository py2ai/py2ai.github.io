---
layout: post
title: "OpenAI MCP Extensions: Plugins That Feel Native Inside ChatGPT - Inside openai/mcp-extensions"
description: "A source tour of openai/mcp-extensions, the official spec and SDK pair that lets an MCP server register sidebar entrypoints, structured settings, composer mentions, file handlers, resource subscriptions, and thumbnail-driven form elicitation inside ChatGPT and Codex."
date: 2026-10-04
header-img: "img/post-bg.jpg"
permalink: /OpenAI-MCP-Extensions-Plugins-That-Feel-Native-Inside-ChatGPT/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/mcp-extensions/openai-mcp-extensions-architecture.svg
tags:
  - MCP
  - ChatGPT
  - OpenAI
  - SDK
categories: [AI, Open Source]
keywords: "MCP extensions, ChatGPT plugins, MCP Apps, OpenAI SDK, TypeScript SDK, Python SDK, form elicitation, structured settings, composer mentions, deep links, resource subscriptions, Model Context Protocol, Codex plugins, sidebar entrypoints"
author: "PyShine"
---

If you have built an MCP server, you know the ceiling of the base specification: tools, resources, prompts, and a stdio or HTTP wire. It works everywhere, which means it feels native nowhere. The model can call your tool, but the user cannot find your app in a sidebar, cannot toggle your settings in a native panel, and cannot @-mention your catalog from the composer. OpenAI's `mcp-extensions` repository exists to break through that ceiling. It is the official companion to a written specification that extends MCP and MCP Apps with ChatGPT-specific capabilities, shipped together with TypeScript and Python SDKs that implement every extension for you.

The repository is young but finished in a way that matters: both SDKs carry version 0.1.0, released on September 29, 2026, under the Apache License 2.0, and the whole thing is wired as a pnpm workspace with a zero-warning ESLint gate, Prettier, and publint packaging checks. The TypeScript package `@openai/mcp-extensions` serves MCP servers and MCP Apps with two entry points (`/server` and `/app`), while the Python package `openai-mcp-extensions` mirrors the server-side surface plus a dedicated `openai_mcp_form_protocol` package that owns form schemas and answer validation. Everything targets Node 22 or later and Python 3.10 or later.

What earns the source tour is how the code translates a prose spec into small, honest modules. Nothing here is a framework you embed; each file is a zod-validated contract plus a thin factory function that registers one tool, one capability, or one request method. Reading it teaches you exactly which bytes go in `_meta["openai/ui"]`, which client capability gates each app-side extension, and how optimistic concurrency on resource writes is negotiated with nothing more exotic than an etag.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/mcp-extensions/openai-mcp-extensions-overview-architecture.svg" alt="Architecture overview of the openai/mcp-extensions repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the repository: one written specification feeding three implementations, with the example plugin consuming the two TypeScript surfaces.*

Reading the overview from left to right: the specification document at `docs/spec.md` is the single source of truth, and it is implemented three times. The TypeScript server half (`typescript/src/server/extensions.ts`) registers the settings, mentions, and form-elicitation tools your MCP server exposes. The TypeScript app half (`typescript/src/app/extensions.ts`) gives an MCP App capability-gated clients for resources, messages, and model-context updates. The Python SDK mirrors the server surface for teams whose backends are Python. At the bottom, the Bits & Bolts example plugin proves the whole stack end to end by organizing CAD parts through every extension the spec defines.

## Why You Need This

The first problem is discoverability. Under plain MCP, your app is invisible until the model decides to invoke it. The entrypoints extension fixes that by letting one MCP App register up to three static entrypoints: a global one in the primary sidebar, a thread one as a content tab inside a conversation, and a file one that turns your app into a viewer for matching file extensions. Each is just a tool annotated with `_meta["openai/ui"]["entrypoints"]`, but the effect is that users open your Parts Library, dashboard, or editor deliberately, the way they open any native feature.

The second problem is configuration. Servers traditionally beg the model to ask the user questions, which is slow and lossy. Structured settings invert the flow: your server declares a capability naming a read tool and an update tool, returns a JSON Schema plus a layout of groups, properties, and buttons, and ChatGPT renders real native controls. The validation in `typescript/src/server/settings.ts` is strict where it matters, rejecting layout entries that reference unknown or duplicate property keys before a single screen is drawn.

The third problem is data entry. Legacy elicitation gives you a raw JSON form, which is fine for text and useless for picking a part from a catalog. The OpenAI form elicitation extension adds suggested values, thumbnails, and previews on options, plus a resource-input field that lets users pick files with visual confirmation. Your server sends `openai/elicitation/create` with a typed schema and receives validated answers back, with every constraint enforced twice, once by the host UI and once again in your SDK.

The fourth problem is state. A plugin that shows files, carts, or selections needs to stay in sync with the conversation and the filesystem. The app-side resource extension gives you reads with text or blob representation, writes guarded by etag and `ifMatch`, subscriptions that push change notifications, and a `ui/update-model-context` channel so the model always sees the user's current selections without re-reading anything.

## How It Works

The codebase is two SDKs orbiting one specification, and the detailed diagram shows how each SDK decomposes into exports, factories, and shared validation helpers.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/mcp-extensions/openai-mcp-extensions-architecture.svg" alt="Detailed architecture of the openai/mcp-extensions repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view: server-side factories register tools and capabilities; app-side factories wrap host requests; shared schemas keep both honest.*

### Understanding the Architecture

**The server surface is three factories in a class.** `typescript/src/server/extensions.ts` defines a four-line constructor: `createElicitInput`, `createMentions`, and `createSettings` each take the wrapped MCP server and return one focused API. `typescript/src/server/index.ts` re-exports every zod schema and TypeScript type so plugin authors can use the same contracts the SDK enforces internally. There is no hidden state, no plugin registry, no lifecycle hooks.

**Structured settings are a read-update pair with teeth.** The capability declares two tool names, `readTool` and `updateTool`. The read result carries `schema`, `layout`, and `values`; layout groups mix property references and tool buttons, where a button invokes a same-server tool accepting empty arguments. Updates arrive as a nonempty `set` record, and the SDK's schema notes openly that custom tools must still validate values against their declared settings schema. That division of labor, envelope validation in the SDK and value validation in your handler, is documented rather than buried.

**Form elicitation validates answers twice.** `typescript/src/server/forms/elicitation.ts` first checks that the client advertises the `openai/elicitation` form capability, then sends the request method and parses the result with `OpenAIFormResultSchema`. On accept, `createOpenAIFormContentSchema` in `typescript/src/server/forms/schema.ts` runs the answers through a JSON Schema validator (the `@cfworker/json-schema` package, draft 2020-12), maps each error back onto a field path, attaches a "Required field" issue to missing controls, and rejects file selections that do not match the picker's constraints. A quiet detail worth admiration: the record parser rebuilds objects from entry tuples so field names like `__proto__` survive parsing intact instead of hitting prototype inheritance.

**The app surface gates everything on host capabilities.** `typescript/src/app/extensions.ts` exposes `resources`, `message`, `modelContext`, and `files` as getters that return the underlying API only when the host advertised the matching experimental capability key (`openai/resource`, `openai/message`, `openai/modelContext`, `openai/files`). Deep links are always available because they ride host context. The constructor also applies one piece of presentation: it maps the host's `openai/interactionCursor` preference onto a `--cursor-interaction` CSS custom property, wrapping `app.connect` so the initial host context is applied too.

**Resource writes negotiate with an etag.** `typescript/src/app/resources.ts` defines the app-to-host `openai/resources/write` method. A read can request text or blob representation through `_meta["openai/resource"]`; returned content carries `openaiMetadata` with `etag` and `writable`. A write passes `ifMatch` with the last-seen etag and gets back `saved` (with the new etag), `conflict`, or `too-large` with a `maxBytes`. Subscription is equally tidy: `addUpdateHandler` registers the notification handler once, fans updates out to a set of callbacks, and returns a dispose function.

**The small files carry the platform honesty.** `typescript/src/app/deep-link.ts` normalizes older host payloads that expressed a deep link as separate `path` and `query` arrays into a single URL string, with a comment admitting the fallback exists because installed apps and hosts update independently. `typescript/src/server/resources.ts` exposes `getResourcePath` so a server can read files relative to an opened file while confining access with `realpath` and a `isWithin` check, and the README's example ends its catch block with the right instinct: never expose the host filesystem path to the app.

The end-to-end flow ties together: a user installs Bits & Bolts, clicks the sidebar entrypoint, and the host invokes the annotated tool with empty arguments; the app renders from the initial tool result, reads CAD resources with etags, saves a reoriented STL through `openai/resources/write`, picks review references through a thumbnail form, and pushes cart updates through `ui/update-model-context` so the model follows along without tool calls.

## Advantages

- **Spec and SDK ship together.** `docs/spec.md` defines every `_meta` key and platform limitation, and the SDKs implement exactly that, so you never reverse-engineer payloads from screenshots.
- **Capability-gated by design.** App-side extensions return `undefined` instead of throwing when a host lacks support, which makes cross-platform degradation explicit at the type level.
- **Validation is layered, not decorative.** Layout references, form answers, file selections, and write envelopes are each validated with named schemas and precise error paths you can surface to users.
- **Prototype-safe by construction.** Form records parse through entry tuples, so hostile field names cannot smuggle prototype pollution into your server.
- **Concurrency without a database.** Resource writes use etag plus `ifMatch` to detect conflicts, a pattern any server can back with a file hash.
- **A working reference implementation.** The Bits & Bolts plugin exercises every extension, including file handlers for STL, 3MF, STEP, and STP files, and its build script produces a distributable plugin directory with no runtime dependency installation.

## Benefits

- **Reach with fidelity.** One MCP server now presents sidebar entrypoints, native settings, and composer mentions across ChatGPT Desktop, Work web, iOS, and Android, with the spec's support table telling you exactly what degrades where.
- **Less prompt begging.** Structured settings and forms move data collection out of model turns and into host-rendered UI, which is faster, cheaper, and cannot hallucinate a field.
- **Safer filesystem integration.** The relative-path helpers confine reads to the opened file's directory and refuse to leak host paths in errors.
- **Consistent look for free.** The shipped `styles.css` gives cards, form controls, and buttons that match ChatGPT, including the pointer-versus-default cursor preference.
- **Two language ecosystems, one contract.** Python servers get settings, mentions, UI metadata, and the full form protocol through pydantic models with strict, forbid-extra validation.
- **Boring tooling you can trust.** Prettier, a zero-warning ESLint gate, publint, and release-please keep the 0.1.0 baseline reproducible.

## Usage

Install the TypeScript SDK next to your MCP server or app:

```sh
pnpm add @openai/mcp-extensions
```

Wrap a server and register structured settings:

```ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { OpenAIExtensions } from "@openai/mcp-extensions/server";
import { z } from "zod";

const server = new McpServer({ name: "viewer", version: "1.0.0" });
const extensions = new OpenAIExtensions(server);

extensions.settings?.register({
  fields: {
    units: { schema: z.enum(["mm", "in"]), title: "Measurement units" },
  },
  read: (extra) => loadPreferences(extra.authInfo),
  update: (set, extra) => updatePreferences(set, extra.authInfo),
});
```

Wrap an MCP App and subscribe to resource changes:

```ts
import { App } from "@modelcontextprotocol/ext-apps";
import { OpenAIExtensions } from "@openai/mcp-extensions/app";

const app = new App({ name: "my-app", version: "1.0.0" });
const openaiExtensions = new OpenAIExtensions(app);
app.ontoolresult = (result) => render(result.structuredContent);
await app.connect();

const dispose = openaiExtensions.resources?.addUpdateHandler(async (n) => {
  if (n.params.uri === resourceUri) await reloadFile(resourceUri);
});
await openaiExtensions.resources?.subscribe({ uri: resourceUri });
```

The Python SDK installs from its package directory with `pip install openai-mcp-extensions`, and the Bits & Bolts example builds into a self-contained plugin directory from the repository root:

```sh
pnpm install --frozen-lockfile
pnpm build
node plugins/bits-and-bolts/scripts/build.mjs --plugin-dir /tmp/bits-and-bolts
```

## Conclusion

The extension points that make software feel native are rarely glamorous; they are annotation keys, capability flags, and validated envelopes. `openai/mcp-extensions` does the unglamorous work in the open: a specification with per-platform support tables, two SDKs that implement it with zod and pydantic contracts, and an example plugin that touches every feature before you have to. If your roadmap includes ChatGPT plugins, reading this repository will save you a month of protocol archaeology.

Links:

- GitHub repository: [openai/mcp-extensions](https://github.com/openai/mcp-extensions)
- Specification: [docs/spec.md](https://github.com/openai/mcp-extensions/blob/main/docs/spec.md)
- TypeScript SDK: [@openai/mcp-extensions on npm](https://www.npmjs.com/package/@openai/mcp-extensions)
