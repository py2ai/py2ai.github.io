---
layout: post
title: "OpenAI Plugins: A Curated Library of Codex Plugin Examples"
description: "OpenAI Plugins is the official curated collection of Codex plugin examples - skills, MCP servers, and app surfaces - with a marketplace index and authoring tooling for building your own."
date: 2026-09-19
header-img: "img/post-bg.jpg"
permalink: /OpenAI-Plugins-A-Curated-Library-of-Codex-Plugin-Examples/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/openai-plugins/oaip-architecture.svg
tags:
  - OpenAI
  - Codex
  - AI Agents
  - Plugins
  - Open Source
author: "PyShine"
---

If you have ever wanted your coding agent to actually know Figma, talk to Notion, drive Stripe, or scaffold an Expo app without you pasting instructions every session, the answer is not a smarter model. The answer is packaging: playbooks, tool wiring, and conventions the agent can load on demand. [OpenAI Plugins](https://github.com/openai/plugins) is the official, curated collection of exactly that packaging for [Codex](https://github.com/openai/codex) — the command-line and IDE coding agent from OpenAI. The repository bundles more than sixty working plugin examples, the marketplace indexes that make them discoverable, and the authoring tooling you need to build your own. It has grown fast, sitting near 7,000 stars and over 900 forks, and it doubles as the de facto specification for what a Codex plugin is.

One honesty note first: the repository does not ship a license file, so treat the content as reference material to study and adapt rather than code you blindly redistribute. As a map of how plugins are meant to be structured, though, nothing else comes close. The overview below shows the three big pieces: the agent on one side, the marketplace indexes in the middle, and the plugin bundles themselves on the other.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/openai-plugins/oaip-overview-architecture.svg" alt="High-level architecture overview of the openai/plugins repository" style="min-width:900px;width:100%;">
</div>

*High-level overview: Codex clients install from the marketplace indexes, which point at plugin bundles carrying skills, manifests, and tool wiring.*

## Why You Need This

Every agent harness eventually hits the same wall: the base assistant is impressive, but real work lives in external tools. Your designs live in Figma, your specs live in Notion, your payments run through Stripe, your deploys run through Vercel or Cloudflare. Wiring each of those by hand — tool descriptions, retry logic, prompting conventions — is exactly the kind of undifferentiated work you should never do twice.

This repository is the "never do it twice" answer, straight from the source. Instead of guessing at what a plugin manifest should contain or how an MCP server should be declared, you read sixty-plus working examples: [Figma](https://github.com/openai/plugins/tree/main/plugins/figma) for design-to-code workflows, [Notion](https://github.com/openai/plugins/tree/main/plugins/notion) for planning and knowledge capture, plus bundles for GitHub, Stripe, Supabase, Expo, Vercel, Cloudflare, Sentry, Slack, Zoom, and dozens more. Each one demonstrates a slightly different surface, so between them they cover nearly every integration pattern you will ever need.

There is also a consistency argument. Teams that adopt agents informally end up with a patchwork of personal prompt snippets and tribal knowledge. A plugin format fixes that: the knowledge becomes a file tree, versioned in git, reviewable in pull requests, and installable by anyone on the team with one command.

## How It Works

The repository has three distinct layers, and understanding the split makes everything else obvious.

**The marketplace layer.** At the top level, `.agents/plugins/marketplace.json` is the default index — it points at the standard `plugins/` directory and tells Codex what is available. A second index, `.agents/plugins/api_marketplace.json`, serves users who log in with API keys. The indexes are structured catalogs that reference local paths under `plugins/`, which is why installation stays fast and works against a plain directory tree rather than a registry service.

**The anatomy of a plugin.** Each plugin lives in `plugins/<name>/` and requires exactly one thing: a `.codex-plugin/plugin.json` manifest declaring its name, description, and surfaces. Everything else is optional:

- `skills/` — directories containing a `SKILL.md` playbook, an optional `agents/openai.yaml` for model-facing hints, and supporting `assets/`
- `.app.json` — an application surface for plugins that render or interact with an app
- `.mcp.json` — MCP server declarations that let the agent spawn and talk to external tool servers
- `agents/`, `commands/`, `hooks.json` — plugin-level agents, slash commands, and lifecycle hooks

The skill is the star of the show. A `SKILL.md` is a markdown playbook the agent reads when the relevant task appears: conventions, step-by-step procedures, guardrails, and example invocations. The Figma plugin bundles skills for design-system rules and code-to-canvas workflows; the Notion plugin bundles skills for meeting capture and research.

**The authoring tooling.** The repository eats its own dog food: the `.agents/skills/plugin-creator/` directory is itself a skill that walks the agent through building a new plugin. It ships a formal reference for the manifest format and a scaffolding script that generates a valid skeleton on demand. You describe what you want, the skill reads the spec, runs the scaffolder, and you get a conforming plugin directory to fill in.

The detailed diagram below puts all three layers together, from the Codex clients at the top to the external services reached through MCP at the bottom.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/openai-plugins/oaip-architecture.svg" alt="Detailed architecture of the openai/plugins repository" style="min-width:900px;width:100%;">
</div>

*Detailed architecture: marketplace indexes, the required manifest and optional surfaces inside a plugin, and the plugin-creator tooling used to author new bundles.*

## Advantages

- **Source of truth.** These are the official examples. When the format evolves, it shows up here first.
- **Conventional over clever.** Every plugin follows the same skeleton, so reading one teaches you the pattern for all sixty.
- **Skills plus MCP in one bundle.** Playbooks tell the agent how to think about a domain; MCP wiring gives it hands to act. Shipping both together is what makes these plugins feel native.
- **Zero infrastructure.** The marketplace is a JSON file and the plugins are directories. No server, no registry.
- **Self-bootstrapping.** The tooling for making plugins is itself a plugin, so the repository documents and extends itself with the same mechanism it ships.

## Benefits

The most immediate benefit is speed: what used to be a weekend of reverse-engineering a plugin format becomes an afternoon of running the scaffolder and copying the closest example. The second benefit is quality, because the examples encode OpenAI's own conventions for scoping a skill, choosing surfaces, and wiring tools — your plugins inherit that discipline instead of accumulating personal hacks.

For teams, the benefit is institutional. Skills and manifests live in version control, so agent behavior finally gets the same review process as code. A senior engineer can codify a deployment procedure once, and every teammate's agent executes it the same way. For the ecosystem, the repository lowers the barrier far enough that publishing a well-made integration becomes reasonable for any SaaS vendor.

## Usage

Getting started takes minutes:

1. Have Codex installed and working on your machine. If you have not tried the agent itself yet, start at the [Codex CLI repository](https://github.com/openai/codex) first.
2. Browse the [plugin directory](https://github.com/openai/plugins/tree/main/plugins) and read the [README](https://github.com/openai/plugins/blob/main/README.md) to see the highlighted bundles.
3. Install a plugin through Codex's plugin workflow — it reads the marketplace index and wires the chosen bundle into your sessions. Try Figma or Notion first; they are the richest examples.
4. To build your own, invoke the plugin-creator skill in Codex and describe your integration. It reads the [manifest specification](https://github.com/openai/plugins/blob/main/.agents/skills/plugin-creator/references/plugin-json-spec.md), runs the [scaffolding script](https://github.com/openai/plugins/blob/main/.agents/skills/plugin-creator/scripts/create_basic_plugin.py), and hands you a valid skeleton to complete.
5. Iterate: add a skill for each recurring workflow, an MCP config where tools are needed, and keep each `SKILL.md` focused on one domain.

The structure mirrors ideas we have covered before: [spec-driven development for coding agents](/OpenSpec-Spec-Driven-Development-For-AI-Coding-Agents/) pushes the same "write the playbook down" philosophy, and the [plugin-based agent harness design](/deepseek-harness-everything-is-a-plugin-agent-harness-cordis/) shows how far an everything-is-a-plugin architecture can go.

## Conclusion

OpenAI Plugins is a small repository with an outsized role: it is simultaneously a library, a specification, and a tutorial. The format it demonstrates — manifests, markdown playbooks, MCP wiring, marketplace indexes — is becoming the standard way capability gets attached to coding agents, and the sixty-plus examples here are the best study material available. Install a couple of bundles today, read the rest as blueprints, and steal the structure for your own integrations. The playbook era of agent extensions is here, and this repository wrote the book.

