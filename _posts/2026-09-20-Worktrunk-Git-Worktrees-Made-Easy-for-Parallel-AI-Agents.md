---
layout: post
title: "Worktrunk: Git Worktrees Made Easy for Parallel AI Agents"
description: "Worktrunk is a Rust CLI that makes git worktrees as easy as branches, designed for running AI coding agents in parallel - with hooks, LLM commit messages, CI-aware listings, and a one-command merge workflow."
date: 2026-09-20
header-img: "img/post-bg.jpg"
permalink: /Worktrunk-Git-Worktrees-Made-Easy-for-Parallel-AI-Agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - Open Source
  - Rust
  - Git
  - Worktrees
  - CLI
  - AI Agents
  - Developer Tools
author: "PyShine"
---
# Worktrunk: Git Worktrees Made Easy for Parallel AI Agents

AI coding agents such as Claude Code and Codex can now handle long tasks without supervision, which makes it practical to run five or ten of them side by side. Git's built-in worktree feature is the natural foundation for this: it gives every agent its own working directory so they never step on each other's changes. The problem is that the worktree interface itself was designed for a single developer doing one thing at a time, and it shows. [Worktrunk](https://github.com/max-sixty/worktrunk) is a command-line tool for git worktree management, written in Rust and designed from the start for running AI agents in parallel. Released at the start of 2026 under a dual MIT OR Apache-2.0 license, it has already collected more than eight thousand stars and has, by its author's account, quickly become the most popular git worktree manager. This post looks at why worktrees matter for agent workflows, how the tool is built, and how to put it to work on your own machine.

![High-level architecture overview of the Worktrunk repository](/assets/img/diagrams/worktrunk/worktrunk-overview-architecture.svg)

## Why You Need This

Git worktrees let you check out multiple branches of the same repository into separate directories at the same time. Each worktree has its own files and its own checked-out branch, while all of them share the underlying object database. That is exactly what a fleet of agents needs: each one works in isolation, yet they all see the same history and can push branches to the same remotes.

The catch is the interface. Creating a worktree for a new feature means typing the branch name three times: once to create the branch, once to name the worktree directory, and once more to change into it. A minimal sequence looks like `git worktree add -b feat ../repo.feat` followed by `cd ../repo.feat`, and the [official git documentation](https://git-scm.com/docs/git-worktree) offers many more flags than most people want to memorize. Cleanup is just as manual.

When a human juggles two or three branches, that friction is annoying. When five to ten agents each need their own checkout, it becomes a real bottleneck. Add the surrounding work that multi-checkout development demands, such as starting a dev server per worktree, checking CI status per branch, and squashing and rebasing before a merge, and the overhead can eat the very productivity that parallel agents were supposed to deliver. Worktrunk exists to remove exactly that overhead.

## How It Works

The diagram below maps the main components of the repository and how they connect.

![Detailed architecture of the Worktrunk repository](/assets/img/diagrams/worktrunk/worktrunk-architecture.svg)

**CLI entry and shell integration.** Everything starts at the `wt` binary. The entry point parses arguments into typed commands, and a shell integration layer, installed with `wt config shell install`, is what allows commands like `wt switch` to change the directory of your actual terminal session rather than a child process. On Windows, the installer registers the binary as `git-wt` so it does not collide with Windows Terminal's own `wt.exe`.

**Core commands.** The command layer implements the verbs that cover the whole worktree lifecycle. `wt switch` creates or selects a worktree and moves you into it; `wt list` renders a status table of every worktree with uncommitted changes, divergence from main, remote sync state, and commit summaries; `wt merge` folds a finished branch back with squash, rebase, merge, and cleanup in one step; `wt remove` deletes a worktree and its branch together; `wt step` runs configured build or test steps; and `wt config` manages settings, aliases, and shell integration. An interactive picker lets you browse worktrees with live CI status, diffs, logs, and pull request previews before jumping to one.

**Automation with an approval gate.** Worktrunk can run [hooks](https://worktrunk.dev/hook/): user-defined commands that fire on worktree creation, before a merge, after a merge, and at other lifecycle points. Because hooks are arbitrary shell commands, their source matters; cloning a repository should never mean executing that repository's code. The automation layer therefore separates hook definitions, hook plans, and an approvals module: a plan is only executed after it has been approved, and that gate is deliberately the only thing standing between a fresh clone and remote code execution. Alongside hooks, an LLM module generates [commit messages from diffs](https://worktrunk.dev/llm-commits/) and can produce per-branch summaries for the list view.

**The git layer and infrastructure.** All git operations go through a dedicated repository module that models worktrees, diffs, remote references, and [CI platform](https://worktrunk.dev/list/) queries, so every command speaks to git through one consistent interface. Underneath, a caching layer keeps repeated repository queries fast, a structured shell-execution module runs every subprocess the same way, and a trace module captures what happened during a command for diagnostics.

## Advantages

- **Worktrees addressed by branch name.** Paths come from a configurable template, and any command that takes a branch also accepts the path of the worktree where it is checked out. You think in branches; the tool handles directories.
- **A single, fast binary.** Worktrunk is one statically built Rust executable with no runtime dependencies, available for macOS, Linux, and Windows.
- **Safety-first automation.** Hooks are powerful, and the approval gate makes them safe to define in shared configuration without handing remote repositories execution rights on your machine.
- **CI and pull request awareness.** The list view and the interactive picker surface CI status, logs, and pull request previews, and `wt switch pr:123` jumps straight to a pull request's branch.
- **LLM assistance built in.** Commit messages and per-branch summaries are generated from your actual diffs, without leaving the terminal.
- **One-command merge.** Squash, rebase, merge, and cleanup happen together, so finished work never leaves debris behind.

## Benefits

Parallel agent development becomes genuinely practical: each agent gets an isolated checkout with one command, and you can start several in the time it used to take to set up one. Context switching gets cheaper, because moving between tasks never dirties your working tree or forces a stash.

The automations you define keep paying off. Hooks codify per-project setup, such as installing dependencies or starting services, and run the same way for you and for every agent you launch. Because they are approval-gated, you can adopt them without worrying about what a cloned repository might try to run. Visibility improves as well: one table answers what is running, what changed, what is merged, and what is stuck in CI. And because the merge command cleans up after itself, a finished feature leaves one trace, a merged branch, instead of a graveyard of stale directories.

## Usage

Installation is a one-liner on the main platforms. On macOS and Linux:

```bash
brew install worktrunk && wt config shell install
```

Or via Rust's package manager:

```bash
cargo install worktrunk && wt config shell install
```

On Windows, use `winget install max-sixty.worktrunk` and then run `git-wt config shell install`; a community-maintained [conda-forge feedstock](https://github.com/conda-forge/worktrunk-feedstock) covers conda and pixi users. Full documentation lives at [worktrunk.dev](https://worktrunk.dev), including a dedicated guide for [using Worktrunk with Claude Code](https://worktrunk.dev/claude-code/).

Creating a worktree for a new feature is one command:

```console
$ wt switch --create feature-auth
```

This creates the branch, creates the worktree, and switches your shell into it. To hand the worktree to an agent immediately, use the execution flag, which starts Claude Code right in the new checkout:

```console
$ wt switch -c -x claude feature-auth
```

Repeat that for each task and you have a fleet of agents, each in its own directory, all sharing one repository; [Codex](https://github.com/openai/codex) works the same way. While they run, `wt list --full` adds CI status and AI-generated summaries per branch. When an agent finishes, `wt merge` squashes, rebases, merges, and cleans up in one go.

Everyday niceties add up quickly: `wt switch pr:123` checks out a pull request by number, the `hash_port` template filter gives each worktree a unique dev server port, and aliases let you define your own `wt <name>` commands. The [tips and patterns](https://worktrunk.dev/tips-patterns/) page collects recipes for dev servers, databases, and agent handoffs, and there is a short [video walkthrough](https://youtu.be/WBQiqr6LevQ) of the whole workflow if you prefer watching to reading. If you are exploring multi-agent setups, our earlier coverage of [OpenAI's curated plugin examples for Codex](https://pyshine.com/OpenAI-Plugins-A-Curated-Library-of-Codex-Plugin-Examples/) and [Octop, a self-hosted multi-agent assistant](https://pyshine.com/Octop-Self-Hosted-Multi-Agent-AI-Assistant/) pairs well with Worktrunk's switch command.

## Conclusion

Git worktrees were always the right primitive for parallel work; they just needed an interface that matches how developers and agents actually operate. Worktrunk turns them into something you can use as casually as branches: one command to create and enter, one table to see everything, one command to finish. With approval-gated hooks, CI-aware listings, LLM-generated commit messages, and a merge command that cleans up after itself, it removes the last excuses for running agents in a single cramped checkout. If you plan to run more than one AI agent this year, install it, run `wt config shell install`, and give your next task its own worktree.
