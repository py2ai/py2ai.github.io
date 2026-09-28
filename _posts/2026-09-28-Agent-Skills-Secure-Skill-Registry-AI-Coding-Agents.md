---
layout: post
title: "Agent Skills: A Hardened Skill Registry for AI Coding Agents - Inside tech-leads-club/agent-skills"
description: "A source tour of tech-leads-club/agent-skills, an open-source TypeScript monorepo that ships a curated, security-scanned catalog of 92 skills for AI coding agents, an interactive installer CLI, and an MCP server built around progressive disclosure. We walk the real architecture: catalog, registry, CDN delivery, hexagonal core, Ink-based CLI, and Next.js marketplace."
date: 2026-09-28
header-img: "img/post-bg.jpg"
permalink: /Agent-Skills-Secure-Skill-Registry-AI-Coding-Agents/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/agent-skills/tech-leads-club-agent-skills-architecture.svg
tags:
  - AI Agents
  - Claude Code
  - Developer Tools
  - Open Source
categories: [AI, Open Source]
keywords: "agent skills, tech-leads-club, AI coding agents, Claude Code, Cursor, MCP server, skill registry, npm CLI, SKILL.md, agentic workflows, developer tools, open source"
author: "PyShine"
---

Every developer using an AI coding agent eventually hits the same wall: the agent is only as capable as the instructions it carries. A fresh Claude Code or Cursor session knows how to write code, but not how *your* team plans features, reviews security, or structures a Rails service. The obvious fix — paste more instructions — collides with a second wall: the "skills" ecosystem is young, fragmented across nineteen different tools, and, as the Snyk report cited in the project's README notes, a significant share of marketplace skills contain critical issues. Installing someone else's prompt bundle means running unvetted instructions inside a tool that can read your files and your environment.

**Agent Skills** from Tech Leads Club (`tech-leads-club/agent-skills`) is a direct answer to both walls. It is an open-source, TypeScript monorepo that curates a catalog of 92 skills across 14 categories — from `(development)` and `(security)` to `(cloud)`, `(design)`, and `(web-automation)` — and wraps that catalog in the tooling to consume it safely: an interactive CLI that installs skills into 19 different AI coding agents, an MCP server that lets agents consult the catalog on demand during a session, and a static marketplace site for browsing everything in a browser.

The source is worth a tour even if you never install a single skill, because it is one of the most complete reference implementations of "trustworthy agent tooling" currently on GitHub. The installer layers sanitization, path isolation, symlink guards, and an append-only audit trail. The core library is a textbook hexagonal architecture with ports and adapters. The MCP server is engineered around token economics with a three-step progressive-disclosure workflow. Reading how these pieces fit together teaches you what production-grade agent infrastructure actually requires.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/agent-skills/tech-leads-club-agent-skills-overview-architecture.svg" alt="Architecture overview of the tech-leads-club/agent-skills repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the agent-skills system: skills are authored and validated in the catalog, indexed into a registry that ships as an npm package, and consumed by three client surfaces — the CLI, the MCP server, and the marketplace site — which install into or serve nineteen agent environments.*

Reading the overview from left to right: the **Skill Authoring & QA** group is where content originates — the `packages/skills-catalog/skills` tree plus a scaffolding generator (`tools/skill-plugin`) and a contract validator (`tools/validate-skills.ts`). The **Registry & CDN Delivery** group is the distribution spine: `generate-registry.ts` compiles the catalog into `skills-registry.json`, which ships inside the `@tech-leads-club/skills-catalog` npm package that every client fetches over a CDN. The **Client Tooling** group contains the three consumers — the shared `libs/core` services powering both the interactive CLI (`packages/cli`) and the MCP server (`packages/mcp`), plus the Next.js marketplace (`packages/marketplace`). On the right, the **Agent Environments** group is where work lands: 19 agent integrations defined in `libs/core/src/lib/services/agents.service.ts`, tracked by an atomic, content-hashed lockfile.

## Why You Need This

The first problem is fragmentation. Cursor wants skills in `.cursor/skills`, Claude Code in `.claude/skills`, GitHub Copilot in `.github/skills`, Windsurf, Cline, Aider, Gemini CLI, TRAE, Kilo Code, and a dozen others each have their own location — and the list keeps growing. Without tooling, "install a skill" means manually copying `SKILL.md` folders into the right dot-directory for every agent you use, then repeating the process for updates. The `agents.service.ts` file in `libs/core` normalizes all of this into a single registry of target definitions, each mapping an agent to its skills directory, so one install command can fan out to any combination of agents, per-project or globally.

The second problem is trust. A skill is an instruction file that your agent will follow — which makes a malicious skill a supply-chain attack on your development environment. The project's README and `SECURITY.md` lean on a Snyk Agent Scan report finding that over 13% of marketplace skills contain critical vulnerabilities, and they structure the entire repository as a counterargument: 100% open source with no binaries, static analysis in CI, immutable integrity via lockfiles and SHA-256 content hashing, human curation of every prompt, and a Snyk scan gate wired into the release scripts so nothing publishes unscanned.

The third problem is context economics. An agent that loads a whole catalog of instructions to answer one question burns tokens on noise. This is where the MCP server earns its keep: it exposes the same catalog through a search-first workflow so the agent reads only the one skill it needs — and only the reference files that skill's instructions actually call for. The design goal, stated plainly in `packages/mcp/README.md`, is that each level of disclosure pays only for itself.

Finally, there is the team problem: you want every developer on the team — and every agent they happen to prefer — working from the same versioned skill set. The lockfile that records exactly which skills are installed, from which source, with which content hash, turns "what instructions is my agent running?" into a question with a precise, auditable answer.

## How It Works

The repository is an Nx-managed npm-workspaces monorepo where four packages and one shared library each own one responsibility, and the whole pipeline — from authoring a skill to it appearing in your agent — is visible in the code.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/agent-skills/tech-leads-club-agent-skills-architecture.svg" alt="Detailed architecture of the tech-leads-club/agent-skills repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture: the authoring and registry-build pipeline on the left feeds the npm/CDN package; the hexagonal core serves the CLI and MCP clients; the Next.js marketplace reads generated site data; release governance ties the validator, scanner, and allowlist together.*

### Understanding the Architecture

**The catalog.** Content lives in `packages/skills-catalog/skills`, organized as parenthesized category folders — `(development)`, `(cloud)`, `(security)`, `(design)`, and ten more — with each skill being a folder containing a `SKILL.md` plus optional `references/`, `scripts/`, and `assets/` subdirectories. A representative example is `(development)/tlc-spec-driven`, whose `SKILL.md` drives a four-phase Specify → Design → Tasks → Implement workflow and whose `references/` and Python `scripts/` provide on-demand depth. Contribution quality is enforced by `tools/validate-skills.ts`, which checks kebab-case folder names, exact `SKILL.md` casing, valid YAML frontmatter, a description capped at 1024 characters, and — tellingly — the absence of any `README.md`, on the principle that "skills are for agents, not humans."

**The registry and CDN spine.** Two build scripts turn that file tree into something distributable: `packages/skills-catalog/src/generate-registry.ts` writes `skills-registry.json`, and `scan-skills.ts` extracts searchable trigger keywords into it. The registry ships inside the `@tech-leads-club/skills-catalog` npm package, which becomes the CDN payload: `libs/core/src/lib/services/registry.service.ts` constructs a jsDelivr URL for the package (with an unpkg fallback), using the `SKILLS_CDN_REF` environment variable to pick the ref. `libs/core/src/lib/constants.ts` tunes the delivery behavior — a 24-hour registry cache TTL, a 15-second fetch timeout, three retries with exponential backoff, and up to ten concurrent file downloads per skill.

**The hexagonal core.** `libs/core` is the reusable engine, split into `ports/` (interfaces for filesystem, HTTP, environment, logging, paths, shell, and package resolution), `adapters/` (Node implementations of each port), and `services/` (the domain logic). The security-critical one is `installer.service.ts`: it sanitizes every skill name, verifies every resolved path stays inside the allowed base directory, treats symlinks as untrusted via `lstat` with target validation and loop detection (using Windows junctions on Windows), and delegates bookkeeping to `lockfile.service.ts`, which parses `.agents/.skill-lock.json` through a strict Zod schema and writes it atomically — backup, temp file, rename — with a SHA-256 content hash per skill. `agents.service.ts` holds the 19 agent definitions, and `audit-log.service.ts` appends every install, update, and remove as a JSON Lines entry to `~/.agent-skills/audit.log`.

**The CLI.** `packages/cli` is a surprisingly polished terminal application built with Ink (React for the terminal) on top of Commander. `src/index.ts` registers the command handlers in `src/cli/` — install, update, remove, cache, audit — which call straight into the core services, while `src/app.tsx`, `src/views/` (skill browser, install wizard, agent selector), and `src/hooks/` render the interactive browse-filter-select-install experience. Each step offers a back option, and the whole thing is targeted by property-based and unit tests.

**The MCP server.** `packages/mcp` exposes the catalog to any MCP-compatible client with a deliberately three-step workflow: `search_skills` performs weighted Fuse.js fuzzy matching over name, extracted triggers, description, and category, drops "weak" matches below a noise floor, and returns up to five ranked candidates; `read_skill` returns the skill's `SKILL.md` with frontmatter stripped, plus its reference file list; and `fetch_skill_files` or `prepare_skill_files` retrieve only the declared files — the former capping responses at 50,000 characters, the latter staging scripts on disk and handing back `file://` links. Before anything is returned, `src/integrity.ts` verifies content hashes against the registry, so a tampered CDN payload fails closed.

**The marketplace site.** `packages/marketplace` is a Next.js static site: a generate-data step parses every `SKILL.md` into `src/data/skills.json`, the routes in `src/app` render listing, category, and skill-detail pages with search and dark mode, and a GitHub workflow deploys the build to `agent-skills.techleads.club`.

End to end: a developer runs `npx @tech-leads-club/agent-skills`, the wizard fetches the cached registry from the CDN, the developer filters by category and picks skills and target agents, and `installer.service.ts` downloads each skill, verifies it, copies or symlinks it into each agent's skills directory, records everything in the lockfile, and appends to the audit log — after which the agent picks the skill up from its own configuration on the next session. The MCP route skips installation entirely: the agent searches, reads, and fetches exactly what the current task demands.

## Advantages

- **One catalog, nineteen agents.** A single `agent-skills install` command reaches Cursor, Claude Code, GitHub Copilot, Windsurf, Cline, Aider, Gemini CLI, TRAE, and more, per-project or globally — no per-agent copying rituals.
- **Security as a pipeline, not a promise.** Input sanitization, resolved-path isolation, symlink guards, Zod-validated atomic lockfile writes, content hashing, and a Snyk Agent Scan gate on release are all in the code path, not the marketing page.
- **Progressive disclosure by design.** The MCP server's search → read → fetch workflow, weighted fuzzy search, and 50k-character response budget keep token spend proportional to the task.
- **Reproducible installs.** The lockfile plus SHA-256 content hashes make it possible to answer exactly what is installed, from where, and whether anything changed on disk.
- **A genuinely testable core.** The ports-and-adapters split in `libs/core` means the same services power the CLI and MCP server and are unit-tested against fake ports — a pattern worth stealing for your own tooling.
- **Full operational observability.** An append-only JSON Lines audit log, cache inspection and clearing commands, and a credits command round out the operational surface.

## Benefits

- **Faster capability onboarding.** Instead of writing an agent workflow from scratch, you start from a curated skill — spec-driven development, AWS architecture review, Playwright automation, security review — and adapt it.
- **Lower risk than open marketplaces.** Human curation, static analysis, and the no-binaries policy meaningfully shrink the attack surface that the Snyk report shows is real.
- **Lower token costs.** Installing only the skills you use — or serving them through the MCP's on-demand path — keeps prompts lean compared to stuffing a monolithic instruction file into context.
- **Team-wide consistency.** A shared, versioned skill set with lockfile-pinned installs means code review standards and planning rituals survive across editors, agents, and machines.
- **Offline-friendly.** Downloaded skills are cached under `~/.cache/agent-skills/`, so repeat installs and updates do not re-fetch what has not changed.
- **A paved contribution road.** `npm run generate:skill` scaffolds a new skill from a template, `tools/validate-skills.ts` checks the contract locally, and the registry regeneration is one Nx target away.

## Usage

Install skills in your project with the interactive wizard:

```bash
npx @tech-leads-club/agent-skills
```

Or drive the CLI directly (works the same after `npm install -g @tech-leads-club/agent-skills`, using the `agent-skills` binary):

```bash
# List available skills
agent-skills list

# Install one skill
agent-skills install -s tlc-spec-driven

# Install multiple skills to specific agents
agent-skills install -s aws-advisor nx-workspace -a cursor windsurf cline

# Install globally (to ~/.claude, ~/.cursor, etc.)
agent-skills install -s my-skill -g

# Update all installed skills
agent-skills update

# Remove skills
agent-skills remove -s my-skill

# Inspect the audit trail
agent-skills audit -n 20
```

Alternatively, expose the catalog to your agent live through the MCP server — no installation step at all:

```json
{
  "mcpServers": {
    "agent-skills": {
      "command": "npx",
      "args": ["-y", "@tech-leads-club/agent-skills-mcp"]
    }
  }
}
```

## Conclusion

`tech-leads-club/agent-skills` is what the skills ecosystem needs more of: a catalog that treats trust as an engineering problem and solves it with validation, scanning, hashing, and audit trails — while keeping the developer experience light enough that `npx` is the only command you remember. Whether you adopt the skills, the CLI, the MCP server, or just borrow the hexagonal-core and progressive-disclosure patterns for your own agent tooling, the source rewards the read.

Links:

- GitHub repository: [tech-leads-club/agent-skills](https://github.com/tech-leads-club/agent-skills)
- Marketplace site: [agent-skills.techleads.club](https://agent-skills.techleads.club/)
- Docs: [tech-leads-club.github.io/agent-skills](https://tech-leads-club.github.io/agent-skills/)
