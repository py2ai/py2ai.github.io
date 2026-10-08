---
layout: post
title: "Gemini CLI: Google's Open Source Terminal Agent - Inside google-gemini/gemini-cli"
description: "A source-code tour of Gemini CLI, Google's Apache-2.0 terminal agent: an npm monorepo with an Ink-powered UI, a tool scheduler with policy-driven confirmations, MCP integration, and a 1M token context window."
date: 2026-10-15
header-img: "img/post-bg.jpg"
permalink: /gemini-cli/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/gemini-cli/google-gemini-gemini-cli-overview-architecture.svg
tags: [AI Agent, CLI, Gemini, Open Source]
categories: [AI, Open Source]
keywords: "gemini cli, google gemini, coding agent, mcp, terminal, open source, architecture"
author: "PyShine"
---

Gemini CLI is Google's open source AI agent that lives in your terminal, and it arrives with a proposition that is hard to ignore: run `npx @google/gemini-cli` and you get the Gemini models, with a free tier of 60 requests per minute and 1,000 requests per day for personal Google accounts, plus a context window measured in a million tokens. Under the Apache-2.0 license the entire system is readable: an npm monorepo whose packages cover the CLI itself, the core agent engine, a TypeScript SDK, an agent-to-agent server, and a VS Code companion extension.

What separates this codebase from a thin API wrapper is the machinery behind the prompt. The core package owns a tool scheduler that orders and executes the model's calls, a policy engine and confirmation bus that decide what needs human approval, sandboxing for shell commands, and an MCP layer that folds external servers into the same tool registry as the built-ins. Tools range from file editing and ripgrep-powered search to Google Search grounding, and the terminal experience itself is a React application rendered with Ink, complete with slash commands, themes, and conversation checkpointing.

As always in this series, this is an educational tour of published source code. Gemini CLI reads your files and runs shell commands with model guidance, and the project pairs that power with policy rules, confirmation dialogs, and sandbox modes for a reason. Use it on projects you own, read the approval prompts it shows you, and study how its architecture keeps an ambitious agent on a leash, because that design is a lesson in itself.

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/gemini-cli/google-gemini-gemini-cli-overview-architecture.svg" alt="Gemini CLI overview architecture diagram" style="max-width:100%;"></div>
<p><em>Gemini CLI at a glance: one entrypoint branches into interactive and scripted runs, feeding a core engine whose tool registry unifies built-ins and MCP servers.</em></p>

Reading the overview from left to right:

- The binary starts at [packages/cli/src/gemini.tsx](https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/gemini.tsx), which loads config and branches between interactive and scripted modes.
- Interactive sessions run through [packages/cli/src/interactiveCli.tsx](https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/interactiveCli.tsx), rendering the React-in-the-terminal UI under [packages/cli/src/ui](https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/ui).
- Scripted usage flows through [packages/cli/src/nonInteractiveCli.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/cli/src/nonInteractiveCli.ts) for one-shot runs in CI and automation.
- The agent engine lives under [packages/core/src/core](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/core) with a [packages/core/src/scheduler](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/scheduler) that sequences tool calls.
- Built-in tools are defined under [packages/core/src/tools](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools) and registered by [packages/core/src/tools/tool-registry.ts](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/tool-registry.ts).
- External capabilities enter through the MCP layer in [packages/core/src/mcp](https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/mcp) and land in the same registry.
- Project behavior is tailored by [GEMINI.md](https://github.com/google-gemini/gemini-cli/blob/main/GEMINI.md) context files loaded into the prompt.
- Other surfaces build on the same engine: the [packages/sdk](https://github.com/google-gemini/gemini-cli/blob/main/packages/sdk) library and the [packages/a2a-server](https://github.com/google-gemini/gemini-cli/blob/main/packages/a2a-server) for agent-to-agent hosting.

## Why You Need This

The first reason is free, direct access to frontier models. Sign in with a personal Google account and the CLI grants a daily allowance of requests that covers real work, with the option to swap in an API key or Vertex AI for enterprise needs. That matters twice over: it makes the tool genuinely useful from minute zero, and it makes the repository's patterns testable by anyone. You can read the code and immediately run it against real quotas instead of admiring it from a distance.

The second reason is the tool architecture. The tool registry is the center of gravity: built-in tools for shell, file editing, search, web fetch, and todo planning register alongside MCP tools wrapped from external servers, and everything passes through the same scheduler, policy engine, and confirmation bus. Many agent codebases bolt MCP on as a side door; this one routes every capability through one governed flow. If you are designing a tool system for an agent, the symmetry here, one schema, one approval path, one execution lane, is the pattern to steal.

The third reason is the terminal UX engineering. Building a coding agent's interface with React and Ink is a distinctive choice, and the result is a rich experience with streaming output, diff rendering, theming, and slash commands that still ships as a plain npm package. Alongside it, the monorepo shows how to expose the same engine as a programmatic SDK and as an A2A server so other agents can delegate work. Few open source projects let you compare a TUI, an SDK, and a server protocol over one shared core this cleanly.

## How It Works

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/gemini-cli/google-gemini-gemini-cli-architecture.svg" alt="Gemini CLI detailed architecture diagram" style="max-width:100%;"></div>
<p><em>Inside Gemini CLI: CLI internals, the core engine with its scheduler and agents, the built-in tool set, the MCP layer, and the policy stack guarding every call.</em></p>

### Understanding the Architecture

**An entrypoint with two personalities.** The gemini.tsx entrypoint parses arguments and authentication choices, then hands off to interactiveCli.tsx for the full-screen experience or nonInteractiveCli.ts for scripted runs. The interactive path wires up the Ink UI, slash commands under packages/cli/src/commands, and an ACP adapter for editor integration; the non-interactive path validates auth, runs a single agent session, and prints results for automation. Both paths converge on the same core engine, so behavior differs only in presentation.

**A core with real structure.** The core package splits its responsibilities into visible directories: the core module coordinates turns, an agent module runs the model loop, an agents module hosts subagents for delegation, a scheduler sequences tool invocations, and a context module assembles what the model sees. Config handling, prompt templates, and fallback logic between providers each get their own modules. The layout makes the flow traceable: a turn enters the agent, context is assembled from files and GEMINI.md instructions, the model responds, and the scheduler executes whatever tools were requested.

**A registry full of practical tools.** Tool files are individually small and single-purpose: shell and its background variants run commands, edit and write-file modify code with diff rendering, read-file and read-many-files pull content in, grep delegates to bundled ripgrep, glob and ls enumerate, web-fetch and web-search reach the outside world, write-todos maintains a task plan, and activate-skill loads packaged expertise. Each tool declares a schema and a confirmation profile, and the registry exposes them all to the model in one uniform catalog.

**MCP as a first-class citizen.** The mcp directory discovers and connects to servers declared in config, the client manager maintains those connections, and mcp-tool wraps each remote capability in the same invocation interface as built-ins before registering it. That means approval policies, telemetry, and error handling behave identically whether a tool lives in the repo or on some other machine. The README highlights MCP-based media generation with Imagen, Veo, and Lyria as example integrations.

**A policy stack around every call.** The policy engine evaluates rules about which tools are allowed and when, the confirmation bus routes approval requests to the UI where the human decides, and the sandbox module contains shell execution. This trio is what turns a powerful tool set into a responsible one, and its separation from the tools themselves means you can run stricter modes in CI than in a local terminal without forking the tool code.

**Surfaces beyond the terminal.** The sdk package exposes the engine programmatically so TypeScript applications can embed the agent, and the a2a-server hosts it behind the agent-to-agent protocol for inter-agent workflows. A vscode-ide-companion extension and the core ide module stream editor context into sessions, and the README documents a GitHub Action built on the CLI for pull request reviews and issue triage. One engine, many shells around it.

**End to end.** A prompt arrives at an entrypoint, the core assembles context from GEMINI.md and the workspace, the model replies with tool calls, the scheduler orders them, the policy stack gates them, and results stream back into the Ink UI or your script's stdout. Sessions can be checkpointed and resumed, tools can come from anywhere via MCP, and every path leads back to the same core. That coherence is the project's defining trait.

## Advantages

- **Generous free tier.** 60 requests per minute and 1,000 per day with a personal Google account, no card required.
- **A million tokens of context.** Large-window Gemini models make whole-repository questions practical.
- **One governed tool path.** Built-ins and MCP servers share a registry, scheduler, policy engine, and confirmation flow.
- **React-powered terminal UI.** Streaming diffs, themes, and slash commands rendered with Ink, shipped as a normal npm package.
- **Embeddable and federable.** The SDK and A2A server expose the same engine to applications and other agents.
- **Automation-ready.** Non-interactive mode, checkpointing, and an official GitHub Action cover scripted and team workflows.

## Benefits

- **Learn tool registry design.** The uniform schema, approval profile, and registration flow is a blueprint for any agent's tool system.
- **See MCP done symmetrically.** Remote tools get identical treatment to built-ins, which is the cleanest integration shape available.
- **Study terminal UI architecture.** React with Ink over a streaming agent protocol is a transferable pattern for CLI products.
- **Copy the confirmation stack.** Policy rules plus a confirmation bus separate what an agent can do from what it may do.
- **Understand context assembly.** GEMINI.md loading and the context module show how project instructions become model input.
- **Bridge human and machine workflows.** The SDK, A2A server, and GitHub Action demonstrate one core serving very different consumers.

## Usage

Run it instantly without installing, or install globally:

```bash
npx @google/gemini-cli
npm install -g @google/gemini-cli
brew install gemini-cli
```

Start the interactive agent and sign in with your Google account:

```bash
gemini
```

Or authenticate with an API key from AI Studio:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
gemini
```

Release channels are tagged for preview, latest, and nightly:

```bash
npm install -g @google/gemini-cli@preview
npm install -g @google/gemini-cli@nightly
```

For organizational usage with Code Assist, point the CLI at your project with `export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"` before launching.

## Conclusion

Gemini CLI shows what happens when a first-party model team treats the terminal as a product surface. The engine is structured for auditability, the tools are unified behind one governed registry, MCP servers are peers rather than plugins, and the free tier makes all of it immediately runnable. Read the scheduler and confirmation bus for agent architecture, borrow the registry pattern for your own tools, and drive it daily with the confidence that its safety design was built by the same people who built the model.

Links:

- [Gemini CLI on GitHub](https://github.com/google-gemini/gemini-cli)
- [Gemini CLI documentation](https://geminicli.com/docs/)
- [Gemini CLI GitHub Action](https://github.com/google-github-actions/run-gemini-cli)
