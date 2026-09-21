---
layout: post
title: "Agent-Native: The BuilderIO Framework That Builds Apps Around Agents, Not Buttons"
description: "Agent-Native is BuilderIO's open-source TypeScript framework for building agentic applications. Define each capability once as a shared action, and the agent uses it as a tool while the UI calls it from code. We dig into the architecture, the AgentKit protocol, and how to ship your first agent-native app."
date: 2026-09-21
header-img: "img/post-bg.jpg"
permalink: /Agent-Native-BuilderIO-Framework-Builds-Apps-Around-Agents/
tags:
  - AI
  - Agents
  - TypeScript
  - Open Source
  - Frameworks
author: "PyShine"
---
# Agent-Native: The BuilderIO Framework That Builds Apps Around Agents, Not Buttons

Every agent interface today is a chat box bolted onto an app. The agent talks in one panel, the application works in another, and a fragile translation layer of screen readers, browser extensions, and computer-use models tries to make one drive the other. [Agent-Native](https://github.com/BuilderIO/agent-native), an open-source TypeScript framework from [BuilderIO](https://www.builder.io), attacks that problem at the framework level: you define each capability of your app exactly once as a shared action, the agent consumes it as a tool, and the UI calls the very same function from code. One definition, identical validation and permissions on both paths, zero clicking through interfaces. The repository launched in March 2026 and has already collected 5,111 stars and 480 forks on GitHub, and its companion gallery ships full open-source agents for meetings, design, slides, analytics, calendar, and mail that you can run today. We went through the actual source tree, and this post is the architecture tour: what the framework is, how the action layer and the agent engine really connect, and how to ship your first agent-native app in an afternoon. If you read our earlier piece on [generative UI and keeping AI on a leash](https://pyshine.com/json-render-Generative-UI-Framework-Keeps-AI-on-a-Leash/), this is the other half of that story - not rendering UI for the model, but letting the model and the UI share one nervous system.

![Architecture overview of the Agent-Native repository showing app surfaces, the core runtime, and the state layer](/assets/img/diagrams/agent-native/agent-native-overview-architecture.svg)

## Why You Need This

Consider what happens when you bolt a chat assistant onto a traditional app. The assistant needs to change a record, so it either calls private APIs you never designed for it, or it drives the UI like a human would - slowly, brittlely, and with the failure modes we documented in our [OpenAI six-ways incident postmortem analysis](https://pyshine.com/OpenAI-Listed-Six-Ways-Its-Own-AI-Broke-the-Rules/). You end up maintaining two applications: the one your users click and the one your agent can actually reach. Agent-Native removes the second application entirely. The framework's core claim is that the agent should never click through the UI - both the agent and the interface work through the same action layer, with the same [Zod](https://zod.dev) schemas, the same permission checks, and the same implementation. When the agent finishes work, the results appear in the UI because both read the same database. When the user selects a record, the agent sees that context because both share application state. That inversion - the agent as a first-class citizen of the application rather than a guest in it - is what "agent-native" actually means, and once you have built one app this way, the bolted-on-chat-box approach feels as dated as it is.

## How It Works

The framework lives in a pnpm monorepo whose heart is the `@agent-native/core` package on [npm](https://www.npmjs.com/package/@agent-native/core). Its source splits into a handful of subsystems that map cleanly onto the diagram below.

![Detailed architecture diagram of the Agent-Native framework from the repository source](/assets/img/diagrams/agent-native/agent-native-architecture.svg)

The entry point is [the action](https://agent-native.com/docs/actions-overview): a single `defineAction` call in a file such as `actions/hello.ts` that carries a description, a Zod schema, an optional HTTP binding, and a `run` function. From that one definition, the framework generates every surface. The server's action discovery indexes your action files, and the action routes expose them over HTTP on a [Nitro](https://nitro.build) server. The React client calls the same action with hooks like `useActionQuery("hello", { name: "Alex" })`. The [MCP](https://agent-native.com/docs/getting-started) server turns actions into tools an external coding agent can call, the A2A module speaks a cross-agent protocol so other agents can delegate work to your app, and the CLI invokes actions from a terminal. One capability, five doors, all guarded by the same authorization layer.

Inside the agent runtime, the engine is what actually runs a model against your app: it receives your actions as its tool list, wrapped by the harness that manages sessions, and equipped by skills and memory from the resources module. A distinctive piece is observational memory, which lets a long-running agent keep persistent context across sessions without you wiring up a vector store. The [AgentKit](https://agent-native.com/docs/getting-started) package underneath provides a provider-neutral protocol, headless client, and transports, so the same conversation streams to the React UI, and the same code path works whether you bring OpenAI, Anthropic, or a local model. Everything lands in state built on [Drizzle ORM](https://orm.drizzle.team): PostgreSQL in production, [PGlite](https://pglite.dev) for local development, with chat threads, artifacts, and checkpoints as first-class tables rather than logs.

The surrounding packages round out the platform story. The `frame` package is a local development shell that shows agent chat and a CLI sidebar beside your app in an iframe. The `dispatch` package is a workspace control plane that adds a vault, integrations, destinations, scheduled jobs, and cross-app delegation in a single drop-in module - it is how one agent-native app hands work to another. The `toolkit` package carries reusable app-building UI, `pinpoint` gives agents visual feedback and annotation on what they built, and the `embedding` package lets you drop pickers and agents from one app into another.

## Advantages

The advantages are architectural, and they compound:

- **One definition per capability.** The action you write for the UI is the tool the agent gets. No drift between what the interface can do and what the model can do, because there is no second interface.
- **Permissions are not an afterthought.** Because both paths flow through the same action routes, the authorization layer applies identically to a user click and an agent tool call. Auditing one surface audits both.
- **Bring your own everything.** Your LLM, your SQL database, your tools, your infrastructure. The framework does not meter usage or phone home, and the MIT license means everything you build stays yours.
- **Protocol coverage out of the box.** HTTP for browsers, MCP for coding agents, A2A for agent-to-agent delegation, CLI for scripts. Most frameworks pick one; Agent-Native ships all four from the same action.
- **Real state, not screenshots.** The agent reads and writes the same tables the UI reads and writes, so its work is inspectable, correctable, and durable - a contrast with computer-use approaches that operate on pixels.

## Benefits

Who wins, and how. Product teams benefit first: the gallery of open-source agents - [Clips](https://agent-native.com/apps) for meetings and voice notes, Design for interactive mockups, Slides for on-brand presentations, Analytics for dashboards, Calendar, Mail, Assets, Content, and Plans - doubles as a curriculum. Each is a full app whose source you can read, fork, and reshape, which is a far better starting point than a blank repository. Engineering teams benefit from the boring-but-critical parts being done: authentication, permissions, [automations](https://agent-native.com/docs/automations) that run agent work on schedules or events, and [agent teams](https://agent-native.com/docs/agent-teams) that delegate to specialists in the same workspace. Startups benefit from the deployment story: any Nitro-compatible host works, so a prototype on PGlite migrates to production PostgreSQL without rewrites. And the ecosystem benefits from the license - MIT, no usage caps, no platform lock-in - which is why the repository grew to 5,111 stars within its first six months.

## Usage

Getting productive takes minutes, not days. Scaffold a complete app with a chat surface included:

```bash
npx --yes @agent-native/core@latest create my-agent --standalone --template chat
```

Then add a capability by creating `actions/hello.ts`:

```typescript
import { defineAction } from "@agent-native/core/action";
import { z } from "zod";

export default defineAction({
  description: "Return a friendly greeting.",
  schema: z.object({
    name: z.string().default("world").describe("Name to greet"),
  }),
  http: { method: "GET" },
  run: async ({ name }) => {
    return { message: `Hello, ${name}!` };
  },
});
```

That is the entire ceremony. The agent immediately has `hello` as a tool; React calls it with `useActionQuery`; an HTTP GET reaches it; the MCP server lists it; the CLI runs it. From there, the natural path is the [getting started guide](https://agent-native.com/docs/getting-started), then a closer look at the [server and database](https://agent-native.com/docs/server-database) docs before you move past local PGlite. Run the full app gallery from the repository to see how production-shaped agents structure their actions, and skim `DEVELOPMENT.md` if you plan to work on the framework itself - it documents the workspace layout and the guard scripts that keep contributions consistent.

## Conclusion

Agent-Native is the most complete answer yet to a question the whole industry is asking: what does an application look like when the agent is a primary user? BuilderIO's answer is disciplined rather than exotic - a shared action layer, honest database-backed state, and protocol doors instead of screen-scraping. The monorepo is young, the star count is climbing, and the gallery proves the pattern works for real knowledge work. If your roadmap includes any agent feature this year, building it agent-natively is the difference between bolting a demo onto your product and giving the agent a real job in it.
