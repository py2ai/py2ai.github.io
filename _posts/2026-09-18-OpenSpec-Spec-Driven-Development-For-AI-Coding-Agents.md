---
layout: post
title: "OpenSpec: Spec-Driven Development That Keeps AI Coding Agents Honest"
description: "OpenSpec by Fission-AI adds a plain-Markdown specification layer between you and your AI coding agent: propose, apply, archive. MIT licensed, 69k stars on GitHub."
date: 2026-09-18
header-img: "img/post-bg.jpg"
permalink: /OpenSpec-Spec-Driven-Development-For-AI-Coding-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/openspec/openspec-architecture.svg
tags: [AI coding agents, spec-driven development, OpenSpec, open source, developer tools]
author: "PyShine"
---

Ask an AI coding agent to add a feature and it will happily invent half the requirements, guess at your data model, and call it done. The code compiles, the demo works, and two weeks later nobody remembers what was actually specified. Fission-AI's OpenSpec attacks exactly this gap, and with roughly 69,000 GitHub stars it has become one of the most popular open-source answers to a question every team is asking right now: how do we keep AI agents on a leash we can actually see?

The tool's answer is deceptively simple. Before any code gets written, the change you want is written down as a small set of plain Markdown specifications. The agent plans against those specs, implements against them, and the archive step keeps a versioned paper trail of what was agreed. Specs are the contract; the agent is the contractor.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/openspec/openspec-architecture.svg" alt="OpenSpec architecture: CLI and delivery, planning workflow, agent tooling and state, documentation site" style="max-width:100%;height:auto;" />
</div>

*The architecture map above was generated with the open-source [GitDiagram](https://pyshine.com/GitDiagram-Turn-Any-GitHub-Repo-Into-An-Interactive-Architecture-Diagram/) pipeline, the same prompts and deterministic Mermaid compiler the service uses, so every path you see is a real file in the repository. Click through and you can trace each arrow to source.*

## Why specifications beat vibes

Prompt-driven development has a failure mode the industry has started calling vibe coding: the agent produces something plausible, but the intent only ever lived in a chat bubble. OpenSpec's README puts its philosophy in four contrasts worth repeating. Fluid, not rigid. Iterative, not waterfall. Easy, not complex. And brownfield-friendly, meaning it is built for the messy codebases most of us actually live in, not just greenfield demos.

The workflow is deliberately small. There is no heavyweight phase-gate ceremony and no special markup language. A spec is just Markdown with requirements and WHEN/THEN scenarios, which means it diffs cleanly in git, reviews cleanly in a pull request, and reads cleanly six months later. If your project already has documentation, OpenSpec layers on top instead of demanding a rewrite.

## The propose, apply, archive loop

Under the hood every change lives in its own folder, typically something like `openspec/changes/<change-id>/`. Each folder holds a `proposal.md` that states what is being changed and why, a `specs/` directory containing the requirement deltas with their scenarios, an optional `design.md` for technical decisions, and a `tasks.md` checklist that the agent ticks off as it implements. Nothing here is magic; it is a folder of Markdown files that both humans and agents can read.

The new artifact-guided workflow exposes this as four slash commands that the agent understands: `/opsx:explore` to think through an idea before committing, `/opsx:propose` to draft the change and its spec deltas, `/opsx:apply` to implement against the plan, and `/opsx:archive` to fold the finished change back into the main specs. An expanded profile adds commands like `/opsx:new`, `/opsx:continue` and `/opsx:ff` for teams that want more machinery, and the whole loop is documented in the project's [opsx guide](https://github.com/Fission-AI/OpenSpec/blob/main/docs/opsx.md).

What makes this click is the resolution step. The artifact graph knows what exists, what is in progress, and what should come next, so the agent is never improvising the process itself. It resolves the next artifact, derives progress, and validates the result against the schema before anything touches your code. Vague instructions get caught at the proposal stage, where they cost minutes, not at the review stage, where they cost days.

## One spec layer, thirty tools

OpenSpec does not care which agent you wake up in the morning. During `openspec init` the CLI asks which tools you use and installs the right instruction files for each. The adapter registry covers more than thirty of them, from Claude Code, Cursor, Gemini CLI and GitHub Copilot to Kilo Code, OpenCode, Trae and Windsurf, each getting its native command spelling. The full list lives in the [supported tools table](https://github.com/Fission-AI/OpenSpec/blob/main/docs/supported-tools.md), and adding a new adapter is a matter of registering one more configuration.

Since everything is Markdown, the specs stay yours. OpenSpec never becomes a runtime dependency of your product; it is scaffolding that produces and maintains files you could hand to a human contractor unchanged. You can check them in, branch them, and argue about them in code review like any other artifact.

## Stores, for teams that outgrow one repo

The most interesting recent addition is stores, currently in beta. A store is a Git-backed repository of shared planning artifacts: one place where requirements live, separate from any single implementation repo. Multiple projects can reference the same source-of-truth specs, which is how spec-driven development scales past the solo-project stage. The [stores user guide](https://github.com/Fission-AI/OpenSpec/blob/main/docs/stores-beta/user-guide.md) walks through creating one, linking repos to it, and keeping everything in sync through ordinary git push and pull.

## How it compares

GitHub's [Spec Kit](https://github.com/github/spec-kit) popularized spec-driven development but takes a heavier, more prescriptive route with explicit phase gates. Amazon's [Kiro](https://kiro.dev) bakes the idea into an IDE. OpenSpec's bet is different: keep the surface tiny, keep the artifacts boring, and let any agent in any editor follow the same plain-Markdown contract. If you want ceremony, the expanded profile has room for it; if you do not, the default loop is three commands long. We looked at a different point on this spectrum in an earlier post on [the Pi agent harness](https://pyshine.com/Pi-Agent-Harness-Self-Extensible-Coding-Agent/), and OpenSpec is the option that asks the least of you up front.

## Getting started

The CLI ships as [an npm package](https://www.npmjs.com/package/@fission-ai/openspec) and runs on Node.js 20.19 or newer:

```bash
npm install -g @fission-ai/openspec
cd your-project
openspec init
```

The init wizard asks which agents you use, scaffolds the `openspec/` directory, and installs the slash commands. The [getting started guide](https://github.com/Fission-AI/OpenSpec/blob/main/docs/getting-started.md) then walks through your first proposal, and from there the loop is explore, propose, apply, archive. Anonymous telemetry is on by default but can be disabled with one flag.

OpenSpec is MIT licensed and, in a nice bit of dogfooding, the project uses its own workflow to develop itself, with its specs and changes visible right inside the repository. If your AI pair has been making decisions you never signed off on, this is the cheapest fix we have seen yet: make the agent read the spec before it touches the code, and make it show you the diff when it does.
