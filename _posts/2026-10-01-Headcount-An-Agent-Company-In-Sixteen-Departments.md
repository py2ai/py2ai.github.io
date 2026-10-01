---
layout: post
title: "headcount: An Agent Company In Sixteen Departments - Inside cbrock84/headcount"
description: "headcount packages a full business org chart into installable AI agent departments: a chief executive over sixteen plugin departments and 172 skills, running identically in Claude Code and ChatGPT. We tour the repository that builds, validates, and governs it."
date: 2026-10-01
header-img: "img/post-bg.jpg"
permalink: /Headcount-An-Agent-Company-In-Sixteen-Departments/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/headcount/cbrock84-headcount-architecture.svg
tags:
  - AI Agents
  - Claude Code
  - Prompt Engineering
  - Open Source
categories: [AI, Open Source]
keywords: "headcount, AI agent organization, Claude Code plugins, agent skills, ChatGPT Codex skills, department plugins, agent governance, SKILL.md, source citation, open source AI"
author: "PyShine"
---

Most agent libraries are piles of prompts. You install a mega-pack, your context fills with instructions you will never need this session, and when two skills disagree there is nothing in the repo that says who wins. The prompt pack grows; the discipline around it does not. It is the software equivalent of hiring two hundred people and giving them one shared desk and no reporting lines.

headcount, by Chris Brock, takes the opposite approach: it is an agent organization structured as a company. A chief executive sits over sixteen departments — from Technology and Security to Finance, People, and Corporate Strategy — and together they carry 172 skills. Every department is an independently installable plugin, so a project loads only the functions it needs. Skills are addressed as `department:skill` — `security:threat-modeling`, `finance:unit-economics` — so names never collide, and the same tree installs unchanged in Claude Code, ChatGPT, and Codex.

What makes the source worth a tour is not the skill count but the machinery underneath it. The repository treats its own content the way a serious engineering team treats code: generated files are never hand-edited, every tracked path has exactly one owner, every skill's claims trace to cited sources, and a single script runs every check CI runs. The result is a repo that teaches two things at once — how to package professional expertise for agents, and how to keep a large agent-facing content base from rotting.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/headcount/cbrock84-headcount-overview-architecture.svg" alt="Architecture overview of the cbrock84/headcount repository" style="max-width:100%;height:auto;" />
</div>

*Architecture overview of the cbrock84/headcount repository: agent hosts install department plugins, skills load on request match, and a build-and-check layer keeps the whole organization coherent.*

Reading the overview from left to right: two agent hosts, Claude Code and ChatGPT/Codex, discover the same department tree through their respective marketplace manifests. The departments ship the skills themselves plus agent charters that let a whole department be delegated as a subagent. Source catalogs feed citations into the skills, a surface map assigns one owner to every path in the repository, and underneath everything a build layer of Python scripts validates the organization and emits standalone vertical variants.

## Why You Need This

The first problem headcount solves is selection. Sixteen departments of expertise would be enormous to load at once — and in ChatGPT the repository's own getting-started guide notes that skill descriptions share a context budget of roughly eight thousand characters, which all sixteen departments together far exceed. Because each department is its own plugin with its own manifest, you install the three or four that match your week: IT operations gets `it-operations`, `security`, and `operations`; a founder gets `executive`, `finance`, and `operations`. A smaller installed surface also produces sharper triggering, so the right specialist engages rather than a near miss.

The second problem is authority. Anyone can write a prompt that sounds like a CFO; headcount grounds its skills in cited sources instead. The `sources/` directory holds catalogs that map outside authorities to the skills whose questions they settle, and the build emits each skill's reference list into its own `references/sources.md`. A license field on each source decides whether an agent may quote it or only cite it — a quiet but important distinction, since most of what a professional must cite is not open. There are 184 cited sources across the organization.

The third problem is overreach. Most agent packs will happily generate an opinion on anything; headcount gives two departments — Security and Legal & Risk — reviewer class, meaning their blocking findings are not overrulable by the department under review. When a security review says a design does not ship, the marketing department does not get a vote. That structural check is exactly what organizations have human review boards for, and it is rare to see it encoded into an agent library.

Finally, it solves portability. The skills are identical files in Claude Code and ChatGPT/Codex; only the manifests differ, and both manifest sets are generated from the same tree. A skill fixed once is fixed for both audiences, which is the difference between a library and a fork farm.

## How It Works

The repository is organized so that human-authored content — skills, sources, department manifests — lives in a canonical tree, and everything agents or hosts read downstream is generated and verified against it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/headcount/cbrock84-headcount-architecture.svg" alt="Detailed architecture of the cbrock84/headcount repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the cbrock84/headcount repository, from agent hosts through the department tree, build scripts, and validation checks.*

### Understanding the Architecture

**The department tree is the product.** Each department lives in `plugins/<department>/` — sixteen of them, from `plugins/executive/` (Office of the CEO) to `plugins/security/`. A department carries three things: its skills under `skills/<skill>/SKILL.md`, a Claude manifest at `.claude-plugin/plugin.json`, and a Codex manifest at `.codex-plugin/plugin.json`. The Claude-facing marketplace that ties all sixteen together is `.claude-plugin/marketplace.json`, while ChatGPT reads `.agents/plugins/marketplace.json`. Both marketplaces describe the same skills; only the manifests differ.

**A skill is a file with a trigger.** Each `SKILL.md` carries YAML frontmatter with `name` and `description` plus a body of working method — the threat-modeling skill, for example, walks the four questions of design-time threat analysis, from drawing the real data flow to prioritizing by attacker effort against impact. Skills load themselves when a request matches the description; asking why a landing page does not convert pulls in `demand-generation:landing-page-cro-expert` without you naming it. Next to the skill sits `references/sources.md`, the per-skill citation list the build emits.

**Porting is generation, not duplication.** `scripts/build-port.py` derives each Codex manifest from its Claude counterpart and regenerates `AGENTS.md`, the cross-tool repository context file. Because the manifests are generated, a `--check` flag fails the build when one drifts from its Claude twin. The design note in the script explains what was rejected: copying the whole tree into a separate Codex directory, which had already gone a month stale once before review. Sixteen small generated manifests cannot drift.

**Generated artifacts are verified, never edited.** `scripts/build-readme.py` emits the README, `scripts/build-org-chart.py` produces the searchable org chart served at `docs/org-chart.html`, and `scripts/build-sources.py` distributes the `sources/*.toml` catalogs into each skill's `references/sources.md`. All of them support `--check`, so CI can prove the committed artifacts match the tree. `scripts/build-vertical.py` goes further: it composes `verticals/<slug>/` with the core into a standalone repository, verifies it in a temporary directory, and never commits the output.

**One owner per path is enforced, not suggested.** `docs/AGENT-SURFACES.md` maps every tracked path in the repository to exactly one agent, and `plugins/executive/skills/agent-hierarchy/scripts/agent-guard.mjs` enforces the map — the surface guard reads `git ls-files`, which is why unstaged files are invisible to it locally and fail in CI instead. Each department also ships an agent charter under `.claude/agents/`, so a department can be delegated to as a subagent with its own exclusive write surface.

**A single check script is the whole gate.** `scripts/check-all.sh` runs every validation the repository has: `scripts/validate-skills.py` for frontmatter, `scripts/check-provenance.py` for third-party license text, `scripts/check-skill-refs.py` to confirm skill references resolve, `scripts/check-us-english.py` for spelling by exact word form, `scripts/check-never-blocks.py` for the internal consistency of reviewer-class blocks, and `scripts/check-sources.py` for the source catalog. The CI workflow invokes this same file, so local checks and CI cannot drift apart.

The end-to-end flow: you add the marketplace in your host, install the departments that match your work, and ask a question in a department's territory. The matching `SKILL.md` loads, its cited sources come with it, and if the answer belongs to a reviewer-class department, its findings can stop the work outright rather than decorate it.

## Advantages

- **Install what you use.** Department-level packaging keeps the loaded surface small, which sharpens triggering and respects the context budget of hosts like ChatGPT.
- **One tree, two ecosystems.** The identical skills install in Claude Code and ChatGPT/Codex, with generated manifests on both sides that a single fix updates together.
- **Names that never collide.** The `department:skill` addressing scheme makes `finance:unit-economics` unambiguous no matter how many departments you install.
- **Citations built in.** Source catalogs flow into per-skill reference files automatically, with a license field controlling quote versus cite.
- **Reviewer class stops work.** Security and Legal & Risk findings are structurally overruling, encoding the veto power organizations rely on.
- **A repo that checks itself.** Generated README, org chart, manifests, and source lists are all verified against the tree on every run.

## Benefits

- **Faster specialist access.** You get vetted working method — threat modeling, unit economics, incident response — the moment a request matches, without hunting for the right prompt.
- **Lower drift risk.** Because documents about the organization are generated from it, they cannot quietly contradict it; a stale artifact fails the build.
- **Clean delegation.** Department charters turn a whole function into a subagent with an exclusive write surface, so parallel agent work does not trample files.
- **Honest grounding.** The citation discipline gives agents something to point at beyond their own confidence, which matters when the output drives real decisions.
- **Vertical reach.** The verticals mechanism composes the core with domain material into standalone repositories, verified without committing generated output.
- **Contributor safety.** One script reproduces the entire CI gate locally, so a contributor knows before pushing whether the change lands.

## Usage

Add the marketplace in Claude Code once:

```
/plugin marketplace add cbrock84/headcount
```

Install the departments you need:

```
/plugin install it-operations@headcount
/plugin install security@headcount
```

In ChatGPT or Codex, add the same repository as a plugin marketplace — the manifest it reads is `.agents/plugins/marketplace.json` — or drop the department you want into `.agents/skills/` in your own project.

Then just ask, and the matching skill loads itself. To force a specific lens in Claude Code, name it:

```
/finance:cost-accounting
```

Elsewhere, name it in the sentence — "use the cost accounting skill" — since the slash form is Claude Code's convention. To work on the repository itself, contributors run every check locally:

```
./scripts/check-all.sh
```

## Conclusion

headcount is interesting twice over: as a library of 172 professional skills you can install department by department today, and as a pattern for maintaining large agent-facing content bases with engineering discipline. The governance layer — one owner per path, generated-and-verified artifacts, reviewer-class authority, cited sources — is the part most prompt collections never attempt, and it is what keeps a company-sized organization of agents coherent. If you are building your own agent library, the way this repository builds and polices itself is worth studying as much as the skills it ships.

Links:

- Repository: https://github.com/cbrock84/headcount
- Interactive org chart: https://cbrock84.github.io/headcount/org-chart.html
- Getting started guide: https://github.com/cbrock84/headcount/blob/main/docs/GETTING-STARTED.md
- Agent surfaces map: https://github.com/cbrock84/headcount/blob/main/docs/AGENT-SURFACES.md
