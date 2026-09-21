---
layout: post
title: "ai-memory: The Rust Memory Server That Lets Your Coding Agents Trade Notes"
description: "ai-memory is a self-contained Rust server that gives every coding agent - Claude Code, Codex, Cursor, Gemini CLI and 20+ more - one shared long-term memory. We tour its hook-capture flow, git-backed markdown wiki, typed claim-once handoffs, and how to install it in minutes."
date: 2026-09-22
header-img: "img/post-bg.jpg"
permalink: /ai-memory-Rust-Long-Term-Memory-For-Coding-Agents/
tags:
  - AI
  - Agents
  - Rust
  - Open Source
  - Developer Tools
author: "PyShine"
---
# ai-memory: The Rust Memory Server That Lets Your Coding Agents Trade Notes

Every coding agent you use is memory-impaired by design. Claude Code takes its own notes, Cursor remembers a few things, and every platform is adding its own memory layer - but all of them share the same walls: the notes live on one machine, belong to one agent, and vanish the moment you switch tools. [ai-memory](https://github.com/akitaonrails/ai-memory) is what is on the other side of those walls. It is a self-contained Rust server, MIT-licensed, that has pulled 7,635 stars and 519 forks since May 2026, and its pitch is the scene every developer knows: quit Claude Code mid-task, start OpenAI Codex in the same directory, and continue without re-explaining the architecture, the failed approaches, or the open questions. Twenty-plus agent harnesses - Claude Code, Codex, Cursor, Gemini CLI, OpenCode, Grok, Devin, Kimi, Kiro, and more - feed one shared memory through lifecycle hooks, and the next agent picks up a real handoff: where you left off, what failed, what is still open. We went through the source tree, and the design is refreshingly contrarian: the source of truth is a git-backed wiki of plain markdown files, the default path makes zero LLM calls, and handoffs are a typed protocol that can be claimed exactly once. This post is the architecture tour, plus how to get it running on your machine today. If you have been following our agent-infrastructure series - from [frameworks that build apps around agents](https://pyshine.com/Agent-Native-BuilderIO-Framework-Builds-Apps-Around-Agents/) to [the six ways agents break the rules](https://pyshine.com/OpenAI-Listed-Six-Ways-Its-Own-AI-Broke-the-Rules/) - this is the piece of the stack that makes agents replaceable instead of loyal.

![Architecture overview of the ai-memory repository showing agent surfaces, the memory flow, and storage](/assets/img/diagrams/ai-memory/ai-memory-overview-architecture.svg)

## Why You Need This

Agent memory today is a silo by default. Your notes live inside one vendor's config directory, formatted for that vendor, on the machine where the work happened. Switch to a different agent tomorrow and you are back to re-explaining your codebase like it is the first day. Run the same project at home and at the office and the memory does not follow you. Work on a team, and what your sessions learned stays in your head even though your agents did the learning. ai-memory solves all four silos at once: one server holds the memory for every agent that speaks [MCP](https://github.com/akitaonrails/ai-memory), on your laptop, a homelab box, or anywhere in between; knowledge is shared per project while personal handoffs stay personal; and the whole thing is a single binary with multi-user auth, per-person attribution, and an audit log built in. The part that makes it stick is the format choice: your memory is plain markdown - grep it, open it in [Obsidian](https://obsidian.md), edit it by hand, rsync it to a backup. The database is a derived index that can always be rebuilt from the files. No vector store to babysit, nothing held hostage in a binary blob. For a tool whose entire job is to be trustworthy infrastructure, boring in exactly this way is the feature.

## How It Works

The repository is a Rust workspace of eleven crates, and the flow it implements is captured in four words: capture, consolidate, recall, handoff.

![Detailed architecture diagram of the ai-memory crate workspace from the repository source](/assets/img/diagrams/ai-memory/ai-memory-architecture.svg)

Capture is silent. The hooks directory ships first-party adapters for more than a dozen harnesses - Claude Code, Codex, Cursor, Gemini CLI, OpenCode, Grok, Kimi, Kiro, Devin, and others - and as you work, they emit observations: prompts, tool calls, session boundaries. The hook router in `ai-memory-hooks` passes everything through a typed privacy boundary that sanitizes before anything is stored, then writes it into the store. No "remember this" ceremony, and no LLM bill: the default capture path makes zero model calls. The `ai-memory-workstream` crate feeds managed sessions through the same router, and the `ai-memory-importer` companion loads past sessions from agents you used before adopting the server.

Consolidation happens at session end. The `ai-memory-consolidate` crate reads accumulated observations and compiles them into coherent markdown pages in the project's wiki - optionally LLM-written through the `ai-memory-llm` crate, which speaks to providers including Anthropic and OpenAI-auth flows, but perfectly useful without. The wiki lives in `ai-memory-wiki` as ordinary git-backed `.md` files with atomic writes, backups, and an admission ledger. From those files, the derived index in `ai-memory-store` is rebuilt: full-text search, entity links, and optional vectors, fused into one ranking. The measured write ceiling is about 700 writes per second - a published number instead of a guessed one, which is very much the spirit of the project.

Recall is where the agents come in. The `ai-memory-mcp` crate runs an MCP server with authentication, admin, and per-actor attribution, and every agent - new session, new machine, different vendor - gets a bounded brief injected and can search everything the team has ever captured. The `ai-memory-cli` handles setup, queries, and server control, and the `ai-memory-web` crate mounts a browser UI over the same store. Handoff is the crown jewel: `handoff.rs` in the core crate implements handoffs as a typed, owned protocol where the next agent claims the baton exactly once - a claim-once semantic, not a convention buried in a prompt template. The full invariants that keep multi-user, multi-session use safe are documented in the [architecture notes](https://github.com/akitaonrails/ai-memory/blob/main/docs/ARCHITECTURE.md).

## Advantages

- **Cross-agent by construction.** Twenty-plus harnesses are first-class integrations kept honest by CI, with a [published support matrix](https://github.com/akitaonrails/ai-memory/blob/main/docs/comparison.md) that marks exactly what each one gets - hooks, MCP, or both.
- **Files you own.** The git-backed markdown wiki is the source of truth; the database is disposable and rebuildable. Delete the server and your knowledge is still a folder of text files.
- **Zero-LLM default.** Capture, search, and handoffs all work with no API key. LLM consolidation and vector search are opt-in, never required.
- **One self-contained binary.** Rust, no runtime dependencies, a measured write ceiling, purge commands that say exactly what "deleted" means, and an audit log of every mutation.
- **Team-aware security model.** Multi-user auth, per-person attribution, project-shared knowledge with private personal handoffs, and loopback-only binding by default - expose it to your LAN with a one-line bearer token.

## Benefits

For solo developers, the benefit is continuity: the project you left on the desktop is the project you resume on the laptop, with the same open questions and the same scars from failed approaches. For teams, it is leverage: what one person's sessions learn, everyone's agents can retrieve, so the junior developer's Claude Code starts the morning knowing what the senior developer's Codex learned overnight. For the pragmatists, there is lineage worth trusting: the project describes itself as the Rust successor to the agentmemory project, builds on the compile-not-retrieve philosophy of Andrej Karpathy's [LLM wiki experiment](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), and borrows the Zettelkasten-style atomic notes of the [A-MEM paper](https://arxiv.org/abs/2502.12110) - then replaces their hosted or LLM-bound assumptions with a binary you can run airgapped. And because the wiki is markdown, the memory outlives the tool: even if ai-memory disappeared tomorrow, your agents' knowledge is a git repository you can read with your eyes.

## Usage

The quickest durable install on Arch Linux is the AUR package, which ships the binary, hook sources, and systemd units:

```bash
yay -S ai-memory-bin    # prebuilt Linux x86_64/aarch64 binary
```

A single-user workstation setup then takes four commands:

```bash
ai-memory --data-dir ~/.local/share/ai-memory \
  --config ~/.config/ai-memory/config.toml init
systemctl --user enable --now ai-memory.service
ai-memory install-mcp --client claude-code --apply
ai-memory install-hooks --agent claude-code --apply
```

That registers the MCP server with your agent and installs the lifecycle hooks that make capture automatic - repeat the two install commands for every other agent CLI you use, from Codex to Gemini CLI, and they all start feeding the same memory. On macOS, Windows via WSL2, or anywhere with Docker, the release page publishes a [self-contained binary](https://github.com/akitaonrails/ai-memory/releases/latest) plus a Docker image with amd64 and arm64 variants; the container's default quick-start binds to loopback with no auth, which is the right posture for a single-user laptop. From there the workflow is invisible by design: work with your agent as usual, and at session end the consolidator compiles the session into wiki pages you can open, edit, or grep. The [comparison document](https://github.com/akitaonrails/ai-memory/blob/main/docs/comparison.md) includes a fair rundown against Mem0, Zep, basic-memory, and the built-in memory of Claude Code if you want the full feature debate.

## Conclusion

ai-memory is a rare kind of project: it takes the flashiest idea in AI - agents that remember - and implements it with the least flashy stack imaginable, a single Rust binary writing markdown files to disk. That contrarianism is exactly why it works. Memory that lives in files you own, shared across every agent you use and every machine you touch, captured without ceremony, and handed off as a typed protocol rather than a hopeful prompt - this is what makes an agent ecosystem feel like infrastructure instead of a collection of walled gardens. Install it once, point your agents at it, and the question "wait, which tool did we try that in?" quietly disappears.
