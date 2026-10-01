---
layout: post
title: "second-brain-os: An AI-Maintained Knowledge Vault - Inside undefined-ui/second-brain-os"
description: "second-brain-os turns Obsidian plus a coding agent into a knowledge base that files, links, and maintains itself. We tour the repository: the vault template, 18 skills, 72 slash commands, 6 subagents, and the dependency-free Python tooling behind it."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /Second-Brain-OS-An-Ai-Maintained-Knowledge-Vault/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/second-brain-os/undefined-ui-second-brain-os-architecture.svg
tags:
  - AI Agents
  - Obsidian
  - Knowledge Management
  - Open Source
categories: [AI, Open Source]
keywords: "second brain, AI knowledge base, Obsidian agent, Claude Code skills, personal wiki, note linking, GraphRAG, markdown vault, agent skills, open source"
author: "PyShine"
---

Every knowledge system you have ever used has the same failure mode: it works for two weeks while you are enthusiastic, and then the filing falls behind and the pile wins. Bookmarks accumulate without being read, read-later queues become guilt archives, and half-filled note apps hold fragments that never connect to anything. The filing was never the hard part intellectually — it was just boring enough that humans reliably stop doing it.

second-brain-os, from undefined-ui, hands that boring work to an agent. The repository is a complete operating system for an AI-maintained knowledge base: an Obsidian vault template, eighteen agent skills, seventy-two slash commands, six subagents, and a set of dependency-free Python scripts. You drop articles, transcripts, PDFs, and chat exports into a raw inbox; the agent reads them, splits them into concepts and entities, writes linked wiki pages, and connects the new material to everything already there. Everything lives in plain markdown files you own.

What repays a source tour is how much of the system's quality lives in the instructions rather than in code. The vault's root `CLAUDE.md` is a genuinely interesting artifact: a contract that tells the agent which folders it owns, which it may never touch, and what it must record before deleting anything. Around it, each skill encodes a workflow with opinions — for example, that an ingest which only writes a summary page produces a vault that grows without getting smarter.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/second-brain-os/undefined-ui-second-brain-os-overview-architecture.svg" alt="Architecture overview of the undefined-ui/second-brain-os repository" style="max-width:100%;height:auto;" />
</div>

*Architecture overview of the undefined-ui/second-brain-os repository: a capture inbox feeds agent skills that write a linked wiki, alongside a project layer, with slash commands, subagents, and Python tooling in support.*

Reading the overview from left to right: material lands in the raw inbox and feeds the ingest skill through the slash-command layer. The agent skills write the wiki — one page per idea, densely linked — which feeds a parallel projects layer and receives from it in return. A root `CLAUDE.md` contract governs the whole arrangement, subagents carry delegated work, the Python tooling lints and measures the vault, and the repository's guide and course tie every workflow to a page you can follow.

## Why You Need This

The first reason is ownership. Hosted note-taking services keep your knowledge in their format, on their servers, behind their pricing changes. second-brain-os produces a folder of markdown files — the universal interchange format of thinking — opened by Obsidian, which is free, and maintained by an agent you control. If the project vanished tomorrow, your vault would keep working exactly as it is.

The second reason is compounding. The core rule printed at the top of the ingest skill is "nothing is ingested until it is linked" — every run ends with new pages connected to existing pages in both directions. That single constraint is the difference between a vault that gets better as it grows and a pile that just gets bigger. A source that arrives is checked against what already exists; if it contradicts a concept page, both positions are recorded rather than the old one silently overwritten.

The third reason is retrieval you can trust. Because source pages record claims as claims — this source says X, not X is true — asking the vault a question produces an answer with provenance. The repository's guide devotes entire sections to query patterns, context budget, and what it calls the graph: typed links between pages that let you find bridges between distant ideas, spot hubs, and locate the gaps where a topic is underdeveloped.

The fourth reason is that it doubles as an agents course. Alongside the guide sit seven modules that take you from a single prompt to a production agent — context, decision-making, orchestration, tools, and evaluation — with the repository's own plugins installable in two commands as the course's tools. If you have wanted to learn agent architecture and ended up with a useful second brain as a side effect, that is a rare two-for-one.

## How It Works

The system separates an archive nobody reads from a wiki everybody reads, and makes the agent responsible for the translation between them.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/second-brain-os/undefined-ui-second-brain-os-architecture.svg" alt="Detailed architecture of the undefined-ui/second-brain-os repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the undefined-ui/second-brain-os repository, from the raw inbox through skills, subagents, the wiki's four page types, and the Python tooling.*

### Understanding the Architecture

**The vault is the contract.** `vault-template/CLAUDE.md` assigns ownership up front: the wiki is the agent's to write and keep correct, `raw/` is never edited after material lands, and no wiki page is deleted without a record in `log.md` first — a deletion you cannot trace is the one failure the owner cannot recover from by reading a diff. The wiki itself splits into four page types under `vault-template/wiki/`: `sources/` for one page per ingested item, `entities/` for people, organisations, products and tools, `concepts/` for ideas and frameworks, and `synthesis/` for comparisons and open questions that no single source covered.

**Ingest is the engine.** The skill at `skills/second-brain-ingest/` runs a seven-step workflow: read the source completely before writing anything, search the wiki for existing entities and concepts, write the source page, extract concepts and entities one page per idea, link everything in both directions, update the index and append to the log in the same run, then report what changed. The skill's own commentary is blunt about the stakes — a page that lands unlinked is invisible within a week.

**Templates keep pages honest.** `vault-template/templates/` holds frontmatter contracts for each page type — title, type, created and updated dates, aliases, and a two-or-three-tag vocabulary drawn from what already exists. Concept pages must explain where an idea came from, what supports it, what argues against it, and what is still unclear, so a well-written concept page is readable by someone who has never seen the sources.

**Subagents carry the delegated work.** The six files under `agents/` define specialists the workflows can delegate to — `agents/ingestor.md` among them — with four of the six deliberately read-only, so an agent asked a question cannot restructure your vault on the way to answering it. Slash commands under `commands/` provide seventy-two scoped entry points, from `/ingest` and `/ask` to `/contradictions`, `/gaps`, and `/dedupe`, each documented with what it does and what it refuses to do.

**The project layer is deliberately separate.** `vault-template/projects/example-project/` carries its own `CLAUDE.md` plus an `Inputs / Process / Outputs / Feedback` pipeline, so what you are doing lives in one folder per project while what you know lives in the wiki — and the two feed each other through links rather than by merging. Finished work that must leave the vault goes to `vault-template/output/`.

**Python scripts audit what the agent writes.** `scripts/link_check.py` verifies that links resolve, `scripts/vault_stats.py` powers the `/metrics` and `/health` commands, and `scripts/graph_export.py` backs `/graph-export` — all dependency-free Python, so the tooling runs anywhere Python does. On Windows the guide has a practical warning: use `python` where it says `python3`, since the `python3` name usually resolves to the Microsoft Store stub and does nothing.

The end-to-end flow: clip or drop material into `raw/`, invoke `/ingest`, and the agent writes a source page, extracts and updates concepts and entities, links everything both ways, and logs the run. When you later ask a question or request a report, the agent queries a wiki where every claim carries its provenance and every idea sits on a page that says how it connects.

## Advantages

- **Plain markdown, zero lock-in.** The vault is a folder of files you own; Obsidian renders it, git versions it, any tool can read it.
- **The linking rule compounds.** Nothing enters the wiki unlinked, so the system gets smarter as it grows instead of just bigger.
- **Contradictions survive.** Disagreeing sources are both recorded on the page rather than the latest one overwriting the truth.
- **Read-only subagents by design.** Four of the six subagents cannot write, which caps the blast radius of a question.
- **Dependency-free tooling.** The Python scripts need no packages, so link checks and metrics run on any machine.
- **Seventy-two scoped entry points.** Slash commands give predictable, documented invocations instead of hoping the agent guesses your intent.

## Benefits

- **Your filing actually happens.** The work humans abandon after two weeks is the work the agent never gets tired of.
- **Answers with provenance.** Source pages keep claims attributable, so vault answers come with their receipts attached.
- **A guide that is followed, not skimmed.** Ten sections of pages written to be walked through step by step, with the failures everyone hits collected in the troubleshooting section.
- **A second brain and an agent education.** The seven-module course builds agent literacy on the same repository you are already using.
- **Privacy you control.** The vault is local; the maintenance and privacy skills address what should and should not leave it.
- **Maintenance on a schedule.** The guide's scheduled-maintenance pattern means you can wake up to a vault that filed itself overnight.

## Usage

Clone the repository and copy the starter vault and agent layer:

```bash
git clone https://github.com/undefined-ui/second-brain-os.git
cp -r second-brain-os/vault-template ~/brain

mkdir -p ~/brain/.claude
cp -r second-brain-os/skills   ~/brain/.claude/skills
cp -r second-brain-os/commands ~/brain/.claude/commands
cp -r second-brain-os/agents   ~/brain/.claude/agents
cp -r second-brain-os/scripts  ~/brain/scripts

# the folder READMEs are for reading on GitHub, not for the agent
rm ~/brain/.claude/*/README.md ~/brain/scripts/README.md

cd ~/brain && claude
```

Open the `~/brain` folder in Obsidian with "Open folder as vault". Then feed it material: clip an article with the Obsidian Web Clipper into `raw/`, and run:

```
/ingest
```

Check the vault's pulse with `/metrics` and `/health`, export the link graph with `/graph-export`, and put the whole thing on a schedule so maintenance happens while you sleep.

## Conclusion

second-brain-os is a rare kind of repository: it is simultaneously a finished product you can adopt tonight, a set of well-engineered agent skills worth studying for their craft, and a course on building agents with the theory included. Its central insight — that an agent should hold a maintenance contract to your knowledge base, with explicit ownership, unbreakable linking rules, and auditable tooling — generalises well beyond note-taking. If your saved reading has been quietly mocking you from a read-later queue, this is the most direct way to end the relationship on better terms.

Links:

- Repository: https://github.com/undefined-ui/second-brain-os
- Web guide: https://undefined-ui.github.io/second-brain-os/
- Getting the most from the graph: https://github.com/undefined-ui/second-brain-os/blob/main/docs/05-graphs/README.md
- Agents course: https://github.com/undefined-ui/second-brain-os/blob/main/docs/course-0-map/README.md
