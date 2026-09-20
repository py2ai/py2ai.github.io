---
layout: post
title: "Firstmate: Talk to One Agent, Ship With a Crew"
description: "Firstmate is an open-source agent distro that turns one coding agent into a supervised crew - tmux-visible crewmates, treehouse git worktrees, a zero-token bash watcher, secondmates, and finished PRs."
date: 2026-09-20
header-img: "img/post-bg.jpg"
permalink: /Firstmate-Talk-to-One-Agent-Ship-With-a-Crew/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - Shell
  - AI Agents
  - Multi-Agent
  - Developer Tools
  - Automation
author: "PyShine"
---
# Firstmate: Talk to One Agent, Ship With a Crew

Running one coding agent is easy. Running four at once turns you into a tab juggler: babysitting terminal sessions, copy-pasting context between repos, and forgetting which window held the failing test. [Firstmate](https://github.com/kunchenguid/firstmate), an MIT-licensed project with over 6,700 stars written almost entirely in Shell, flips that model with a simple promise: talk to one agent, ship with a crew. The clever part is what it is not. It is not a model, not a harness, not an MCP server, and not an app to install. It calls itself an agent distro - a portable directory of instructions, skills, tooling, and state conventions that turns a general-purpose agent into a specialized one. You clone the repository, launch a supported harness inside it, and the cloned repo itself becomes the brain of your first mate. This post walks through how that works.

![High-level architecture overview of the Firstmate repository](/assets/img/diagrams/firstmate/firstmate-overview-architecture.svg)

## Why You Need This

The default workflow for parallel agent work is manual. You open one terminal per task, re-explain the project in each, track which agent touched what, and resolve collisions when two edits land in the same branch. Context lives in your head, coordination lives in your memory, and nothing survives a closed laptop.

Firstmate attacks each of these pains with a structural rule rather than a suggestion. You speak only to a single liaison agent, the first mate, and it dispatches crewmates for everything else. Each crewmate gets its own visible session and a clean git worktree carved from the treehouse tool, so parallel work on one repository never collides - a pattern we explored in our earlier post on [Worktrunk, which makes git worktrees easy for parallel agents](https://pyshine.com/Worktrunk-Git-Worktrees-Made-Easy-for-Parallel-AI-Agents/). Tasks come in two shapes: ship tasks deliver authorized changes as pull requests or local merges, while scout tasks leave standalone investigation reports. And every piece of state lives on disk, so killing the session loses nothing; the next session reconciles and carries on. If you have read our coverage of [Octop, a self-hosted multi-agent assistant](https://pyshine.com/Octop-Self-Hosted-Multi-Agent-AI-Assistant/), this is the same itch scratched from a different direction: less infrastructure, more discipline.

## How It Works

The diagram below maps the repository's main components and how they connect.

![Detailed architecture of the Firstmate repository](/assets/img/diagrams/firstmate/firstmate-architecture.svg)

**The distro core.** When you launch a verified harness inside the clone, [AGENTS.md](https://github.com/kunchenguid/firstmate/blob/main/AGENTS.md) takes over as the identity and prime directives of your first mate. The repository carries around 190 helper scripts under `bin/`, plus bundled skills such as the public `skills/stow` knowledge-sweep skill. Your role in this arrangement is formalized too: you become the captain, and the first mate is your direct report.

**Dispatch and delivery.** Asking for work - say, fix a flaky login test and add dark mode - makes the first mate check its toolchain, clone the target project, and spawn one autonomous crewmate per task through `bin/fm-spawn.sh`. Each crewmate works in its own tmux window, Herdr tab, or an experimental Zellij, cmux, or Orca workspace you can watch or type into, with tmux as the hard default. Delivery depends on the project mode: `no-mistakes`, `direct-PR`, or `local-only`, with an optional `+yolo` flag for merge autonomy. Ship tasks end as pull requests on [GitHub](https://docs.github.com/en/pull-requests) through `bin/fm-pr-poll.sh` and `bin/fm-pr-merge.sh`, approved local merges through `bin/fm-merge-local.sh`, or scout reports filed under the task's data directory. The first mate is read-only over your projects except for a narrow set of guarded, captain-approved operations; crewmates make every other change behind the configured merge authority.

**Zero-token supervision.** The most interesting design decision is that supervision costs no tokens. A bash watcher, `bin/fm-watch.sh`, sleeps on the fleet and wakes the first mate only when something genuinely needs attention. Wakes are classified and queued durably on disk, so a wake promised survives a restart. Verified primary harnesses also get a turn-end backstop that blocks or follows up on a blind stop while work is under way. Wedge detection escalates on staleness with configurable timers, so a wedged crewmate surfaces instead of quietly burning minutes.

**Secondmates and Relay.** For larger fleets you can opt in to secondmates: persistent second mates that run from their own isolated `FM_HOME` on your machine or an SSH-reachable host, dispatched through `bin/fm-remote-secondmate-control.sh` with recovery that never silently substitutes a local route for an unavailable remote one. There is also an opt-in Relay: after pairing with a token in a local `.env`, the same fleet can answer your public mentions on [X](https://x.com/kunchenguid) and Discord through a mention lifecycle identical to chat requests, with durable promised replies reconciled from disk and a dry-run preview before go-live. The project maintains a community [Discord](https://discord.gg/Wsy2NpnZDu) for discussion.

## Advantages

The architecture buys several concrete advantages. Supervision is event-driven and token-free: the watcher is plain bash, so watching costs nothing and the expensive agent only wakes on real events. Isolation is structural: worktrees per task mean no shared working directory, no branch collisions, and disposable teardown after delivery. Visibility is total: every crewmate lives in a pane you can watch or type into, rather than hidden behind a daemon. Restart-proofness is absolute: session state, wakes, and holds all live on disk and in the active backend, so you can kill anything and reconcile later. And because the distro is just a directory of text, it works with any of the verified harnesses - Claude Code, Grok, Pi, Oh My Pi, [Codex](https://github.com/openai/codex), OpenCode, and Cursor Agent CLI - including [Claude Code](https://github.com/anthropics/claude-code), one of the three equal co-primary recommendations alongside Grok and Pi.

## Benefits

In daily terms, those advantages compound. You stop context-switching between terminals because the first mate summarizes fleet state on demand through skills like `/ahoy` and `/bearings`, and it escalates only real decisions - merge or not, accept this risk or not. You save tokens twice over: once in supervision, and again in away mode, where the `/afk` skill self-handles routine notifications in bash and batches what matters into digests. You gain honest scaling: two or three crewmates need nothing extra, while a larger fleet can add secondmates without turning your laptop into a port-forwarding puzzle. You also gain durability of intent - a promised reply in a Relay thread or a held captain decision survives restarts because it is written down, not remembered.

## Usage

Getting started takes three commands. First, authenticate the [GitHub CLI](https://github.com/cli/cli) with `gh auth login`. Second, clone the distro with `git clone https://github.com/kunchenguid/firstmate` and change into it. Third, launch one of the verified harnesses inside the clone - `claude`, `grok --trust`, `pi`, or `omp` - and AGENTS.md takes over from there. Requirements are modest: macOS or Linux, git, the GitHub CLI, and the CLI for your chosen backend such as tmux. Then just talk to it: ask it to look at a project, fix a test, and add a feature, and minutes later it returns a review-ready pull request with a risk note and green CI. Setup guides for every backend, including the reference [tmux](https://github.com/tmux/tmux) backend and the experimental [Zellij](https://github.com/zellij-org/zellij) one, live in the repository's documentation alongside deep dives into configuration, remote secondmates, and the voice relay.

## Conclusion

Firstmate is a quietly radical take on multi-agent work: no platform, no server, no vendor - just a disciplined directory of instructions and shell scripts that turns one agent into a responsible manager of many. Its best ideas - zero-token event supervision, worktree isolation, on-disk state, and a single accountable liaison - are all reproducible patterns worth studying even if you never run a crew. With an active maintainer, clear documentation, and an MIT license, it is easy to clone and hard to outgrow. Give it a terminal and a task list, and let the crew row.
