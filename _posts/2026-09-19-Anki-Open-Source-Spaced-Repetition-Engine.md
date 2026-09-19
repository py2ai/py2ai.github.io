---
layout: post
title: "Anki: Inside the Open-Source Spaced Repetition Engine That Makes Memory a Choice"
description: "A tour of Anki's architecture: a Rust core with the FSRS scheduling algorithm, a PyQt desktop shell, a Svelte web frontend, and a self-hostable sync server, all glued together by protobuf contracts."
date: 2026-09-19
header-img: "img/post-bg.jpg"
permalink: /Anki-Open-Source-Spaced-Repetition-Engine/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/anki/anki-architecture.svg
tags:
  - Anki
  - Rust
  - Spaced Repetition
  - Open Source
  - Architecture
author: "PyShine"
---

Ask anyone who has learned a language, passed a medical exam, or memorized a thousand chess openings, and the same tool keeps coming up: [Anki](https://github.com/ankitects/anki). It is a spaced repetition flashcard program with decades of history, tens of thousands of GitHub stars, and a loyal following that borders on religious. What fewer people have seen is what sits under the hood: a serious piece of systems engineering, and this post takes you inside it.

The project is licensed under AGPL-3.0 (with some contributed portions under BSD-3), which means every line that decides when you will next see a card is public, auditable, and yours to modify. Even better, the developers keep an unusually clean, layered architecture that is genuinely pleasant to study. We rendered it with our local GitDiagram pipeline (see our [GitDiagram introduction](/GitDiagram-Turn-Any-GitHub-Repo-Into-An-Interactive-Architecture-Diagram/) for how that works), and the overview below is the honest shape of the repository.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/anki/anki-overview-architecture.svg" alt="Anki high-level overview architecture" style="min-width:900px;width:100%;">
</div>

*Anki overview, generated from the real repository tree with the GitDiagram pipeline. Mobile clients are separate codebases that sync over HTTPS.*

## Why You Need This

If you have ever crammed for an exam and forgotten everything a week later, you already know the problem: human memory forgets on a predictable curve, and most study methods ignore that curve entirely. Spaced repetition flips it around. Each fact you review is scheduled at the moment you are just about to forget it, which is precisely when review produces the strongest memory. Common experience shows that ten minutes of well-timed review beats an hour of frantic cramming.

But not all schedulers are equal. Older flashcard apps use crude intervals that either waste your time on easy cards or drown you in hard ones. Anki ships the FSRS algorithm as a first-class citizen, a modern scheduler that models your actual memory and predicts recall probability per card. If you care about learning efficiently rather than just diligently, the scheduler is the product, and Anki's scheduler is state of the art.

You also need it if you care about ownership. Your collection is a local SQLite database on your own disk, not rows in somebody's cloud. There is no subscription, no lock-in, and no streak expiring because a server said so. Students, language learners, and medical students all fit the same pattern: a private, permanent, algorithmically optimized memory.

## How It Works

The remarkable thing about Anki's codebase is that it spans four languages and still stays coherent. The secret is a contract layer: 25 protobuf service definitions in the proto directory describe every operation the core can perform, and each language talks to the same contract.

<div style="overflow-x:auto;">
<img src="/assets/img/diagrams/anki/anki-architecture.svg" alt="Anki detailed architecture" style="min-width:1100px;width:100%;">
</div>

*Detailed Anki architecture from the GitDiagram pipeline: the full request path from UI to SQLite, plus the sync system.*

Follow the request path. The desktop program you install is a PyQt shell in `qt/aqt`. Its windows embed web views, and a small local HTTP server (`mediasrv.py`) serves the Svelte and TypeScript frontend from `ts/` into those views. When you answer a card, the reviewer UI sends HTTP POST requests through that local server into the Python layer. `pylib/anki` is the public Python library; its `_backend.py` turns every call into a protobuf RPC, which crosses a PyO3 bridge (`pylib/rsbridge`) straight into Rust.

Inside `rslib`, the Rust core, a backend dispatch layer routes each service call to its domain: collection operations, the scheduler, search parsing, card rendering, media handling, and package import or export. The scheduler is the crown jewel. Its queue builder gathers, buries, sorts, and intersperses cards for a session, while the FSRS module maintains per-card memory state, target retention, and a simulator for forecasting your workload. Everything the scheduler decides is persisted through typed storage modules onto the SQLite collection.

The fourth layer is sync. The client protocol lives in `rslib/src/sync`, and Anki ships a complete, self-hostable sync server written in the same Rust workspace at `rslib/src/sync/http_server`. Point the desktop app at your own server, and your collection and media replicate over HTTPS exactly as they would against the hosted AnkiWeb service, which appears in the diagram as an external dependency for good reason: it is optional, not required.

## Advantages

The first advantage is the protobuf contract layer. Most multi-language projects rot at their boundaries; Anki generates type-safe bindings from the same contract definitions into Python and TypeScript, so a change to a service definition ripples correctly into every layer. The generated Python wrappers and the generated TS client you see in the diagram are build artifacts, not hand-maintained glue.

The second is Rust where it matters. Scheduling, database access, rendering, and sync are all in a memory-safe, fast, natively compiled core. The GUI languages stay where GUIs are productive, and the heavy lifting never leaves the core. It is the same architecture instinct we saw when we covered [Hister, the private search engine](/Hister-Your-Own-Private-Search-Engine/): put the durable logic in one honest core and keep every client thin.

Third, the FSRS scheduler is not a plugin bolted on later; it has dedicated modules for memory state, parameter optimization, desired retention, and load simulation. And fourth, because the sync server lives in the main workspace, self-hosting is a supported path with shared types, not a community afterthought.

## Benefits

The practical payoff starts with your time. FSRS schedules each card at the interval that maximizes recall for the effort spent, and users consistently report remembering more material with fewer daily reviews than legacy intervals. Multiplied over a multi-year learning project, that difference compounds into hundreds of saved hours.

Then there is longevity. Your knowledge base outlives any single device or vendor. The collection is a portable SQLite file, export and import handle the standard deck package format, and the AGPL license guarantees the program itself cannot be taken away from you. Twenty years of continuous development backs that promise.

Finally, there is the ecosystem benefit: because the core is a real library (the same `pylib` that powers the GUI), other developers build on it. Add-ons extend the desktop app, alternate frontends reuse the Python API, and the self-hosted sync server lets families, schools, or companies run their own sync infrastructure at zero licensing cost.

## Usage

Getting started takes minutes. Install the desktop build for your platform from the [official site](https://apps.ankiweb.net), create a collection, and either write your first cards or download a shared deck. The [user manual](https://docs.ankiweb.net) covers deck options, filtered decks, and add-ons.

The workflow most learners settle into is simple and worth copying:

1. Create a deck and add cards, one fact per card, with the answer as short as possible.
2. Study daily; the queue builder interleaves new, learning, and review cards automatically.
3. Enable FSRS in deck options and let the optimizer tune parameters from your own review history.
4. Create a free AnkiWeb account for hosted sync, or run your own sync server from the repository and configure the custom sync URL in preferences.
5. Review on your phone with AnkiDroid or AnkiMobile; both clients sync against the same server.

Developers who want to build rather than study should start with the [development documentation](https://dev-docs.ankiweb.net) and the contributor guide in the repository. The project uses a `just`-based build system: clone the repo, run `just run` to launch a development build, and `just check` before submitting changes. Rust changes are checked with `cargo check`, and the web frontend has its own live-reload workflow.

## Conclusion

Anki earns its reputation twice over: once as a learning tool that quietly runs one of the best scheduling algorithms available, and once as an open-source codebase that shows how a four-language system can stay coherent through a single contract layer. A Rust core, a thin Python bridge, a Svelte frontend, and a self-hostable sync server, all described by protobuf, all on your own disk. If you want your memory to be an asset you own rather than a subscription you rent, the repository is waiting.

Links:

- Repository: [github.com/ankitects/anki](https://github.com/ankitects/anki)
- Website and downloads: [apps.ankiweb.net](https://apps.ankiweb.net)
- User manual: [docs.ankiweb.net](https://docs.ankiweb.net)
- Development docs: [dev-docs.ankiweb.net](https://dev-docs.ankiweb.net)
