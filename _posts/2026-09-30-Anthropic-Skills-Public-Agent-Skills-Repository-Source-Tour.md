---
layout: post
title: "Anthropic Skills: How SKILL.md Folders Teach Claude Specialized Work - Inside anthropics/skills"
description: "A source tour of anthropics/skills, Anthropic's official repository of Agent Skills for Claude. We map the repository from SKILL.md anatomy and bundled Python scripts to the .claude-plugin marketplace manifest, with two architecture diagrams showing how nineteen example skills are organized, packaged, and loaded."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Anthropic-Skills-Public-Agent-Skills-Repository-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/anthropic-skills/anthropics-anthropic-skills-architecture.svg
tags:
  - Agent Skills
  - Anthropic
  - Claude
  - Open Source
categories: [AI, Open Source]
keywords: "Anthropic skills, Agent Skills, SKILL.md, anthropics/skills, Claude Code plugins, document skills, docx automation, pptx skill, xlsx skill, pdf skill, MCP builder, skill creator, Claude API, AI agent framework, open source AI"
author: "PyShine"
---

Every conversation with an AI assistant starts from zero. The model may know what a Word document is, but it does not know your company's brand guidelines, your preferred redlining workflow, or the dozen gotchas that make Office XML corrupt files when you get it slightly wrong. Anthropic's answer to that amnesia is a deceptively simple idea called Agent Skills, and the company ships its own public collection of them in a repository that has drawn roughly 179,000 GitHub stars: [anthropics/skills](https://github.com/anthropics/skills).

The repository is Anthropic's implementation of skills for Claude. As the README puts it, skills are "folders of instructions, scripts, and resources that Claude loads dynamically to improve performance on specialized tasks." Each skill lives in its own folder, anchored by a `SKILL.md` file whose YAML frontmatter and markdown body are what Claude actually reads. Around that core, the repo ships nineteen example skill folders under `skills/`, a minimal `template/SKILL.md`, a pointer to the Agent Skills specification in `spec/`, and a `.claude-plugin/marketplace.json` manifest that packages selected skills into installable Claude Code plugins.

The source is worth a tour because this is not a demo graveyard. Several of the skills here — the `docx`, `pdf`, `pptx`, and `xlsx` folders — are the same document skills that power Claude's document capabilities in production, shared as a reference for what a serious, script-backed skill looks like. Reading the repository end to end teaches you the anatomy of a skill, the patterns for bundling scripts and references, and the mechanics of distribution, all from working code rather than a whitepaper.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/anthropic-skills/anthropics-anthropic-skills-overview-architecture.svg" alt="Architecture overview of the anthropics/skills repository" style="max-width:100%;height:auto;" />
</div>

*Overview of anthropics/skills: the documentation and template foundation, the plugin marketplace manifest, and the major skill families under `skills/`.*

Reading the overview from left to right: the foundation group contains `README.md`, the `spec/agent-skills-spec.md` pointer, `template/SKILL.md`, and the `skills/` directory that holds all nineteen example skills. The distribution group is a single file, `.claude-plugin/marketplace.json`, which the README tells you to register with one slash command; from there the manifest bundles skill folders into plugins such as `document-skills` and `example-skills`. The remaining groups fan out into the document skills like `skills/docx`, the development-oriented skills like `skill-creator`, `mcp-builder`, and `webapp-testing`, and the creative and communication skills like `algorithmic-art`, `canvas-design`, and `internal-comms`.

## Why You Need This

The first problem skills solve is repetition. If you have ever re-explained the same workflow to a coding agent for the third time — how your docs should be structured, how to run your test suite, which conventions your PowerPoint deck follows — you have felt the gap between a model's general competence and your specific requirements. A skill captures that workflow once, as text and scripts in a folder, and makes it available every time the relevant situation appears.

The second problem is context economy. Stuffing every manual into the system prompt would bloat every request and degrade behavior on everything else. Skills flip this around: Claude sees only the compact `name` and `description` fields from each skill's frontmatter, and loads the full `SKILL.md` body — plus any bundled scripts — only when the description matches the task at hand. The `skills/skill-creator/SKILL.md` file is explicit about this, calling the description "the primary triggering mechanism" and advising authors to include both what the skill does and the specific contexts in which to use it.

The third problem is packaging and distribution. A useful workflow is only valuable if your teammates, or the wider community, can actually install it. The repository solves this with `.claude-plugin/marketplace.json`, which declares a Claude Code plugin marketplace named `anthropic-agent-skills` containing five plugins — `document-skills`, `example-skills`, `claude-api`, `academy-guide`, and `discernment-nudge` — each mapping to concrete skill folders like `./skills/xlsx` or `./skills/mcp-builder`. Registering the repository turns a pile of folders into a browsable, one-command install.

Finally, the repo is a pattern library. Its skills span a real range of shapes: pure instruction skills like `brand-guidelines` and `internal-comms` (which carries worked examples in `skills/internal-comms/examples/`), asset-backed skills like `theme-factory` with its ten prewritten themes in `skills/theme-factory/themes/`, and full toolkits like `docx` that ship Python scripts, OOXML schemas, and validation logic. Studying the spectrum teaches you when a skill needs executable muscle and when prose is enough.

## How It Works

Underneath the marketing terms, the whole system is a convention over folders — here is how the pieces connect.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/anthropic-skills/anthropics-anthropic-skills-architecture.svg" alt="Detailed architecture of the anthropics/skills repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of anthropics/skills: the metadata and distribution layer, the source-available document skills with their Python tooling, the development toolkit skills, the creative and communication skills, and the knowledge skills.*

### Understanding the Architecture

**The anatomy of a skill.** Every skill is a folder whose required file is `SKILL.md`. The file `template/SKILL.md` is the minimal form: YAML frontmatter with just `name` and `description`, followed by markdown instructions. The README shows the same recipe inline, and the `skills/skill-creator/SKILL.md` guide spells out the contract — the frontmatter fields trigger the skill, and everything below them instructs Claude once it is active. Optional bundled resources sit beside the file: `scripts/`, `references/`, `templates/`, `examples/`, and `agents/` all appear in this repository.

**The document skills are the heavyweight reference.** The `skills/docx` folder opens its instructions with a crucial reframing: a `.docx` is a ZIP archive of XML files. From there the skill routes each task — create with the `docx` npm library, edit by unzipping and patching `word/document.xml`, read via `pandoc`. The bundled `skills/docx/scripts/merge_runs.py` coalesces fragmented text runs so content is findable, `scripts/accept_changes.py` produces clean copies from tracked changes, and `scripts/office/validate.py` XSD-validates output against the OOXML schemas stored under `skills/docx/scripts/office/schemas/`. When output needs eyes, `scripts/office/soffice.py` drives LibreOffice to convert documents to PDF for rendering. These four skills are shared as source-available code — the `docx` frontmatter states its license is proprietary, with complete terms in `LICENSE.txt` — precisely so developers can study a production-grade skill.

**A skill that builds skills.** `skills/skill-creator` is a meta-skill with real tooling: `scripts/run_eval.py` runs test prompts against the skill under development, subagent personas live in `skills/skill-creator/agents/` (`analyzer.md`, `comparator.md`, `grader.md`), `eval-viewer/generate_review.py` renders results for human review, and `scripts/improve_description.py` optimizes the frontmatter description for better triggering. The loop — draft, evaluate, rewrite, package with `scripts/package_skill.py` — is the repo's own quality process, exposed as a skill.

**Toolkits for the wider agent ecosystem.** `skills/mcp-builder` guides the creation of Model Context Protocol servers, backed by best-practice references in `skills/mcp-builder/reference/` and helper scripts like `scripts/connections.py` and `scripts/evaluation.py`. `skills/webapp-testing` is a Playwright-based toolkit whose single helper, `scripts/with_server.py`, manages dev-server lifecycles so the agent's automation script only contains browser logic. Both skills embed an operational discipline in their instructions: run scripts with `--help` first and treat them as black boxes, rather than reading large sources into the context window.

**The distribution layer.** `.claude-plugin/marketplace.json` is what turns this folder collection into something installable. It declares the `anthropic-agent-skills` marketplace and lists five plugins, each with a description and a `skills` array pointing at folders — the four document skills under `document-skills`, twelve example skills under `example-skills`, and one each under the `claude-api`, `academy-guide`, and `discernment-nudge` plugins. The Claude Code slash commands in the README operate entirely against this manifest.

**Knowledge as a skill.** The remaining folders show that reference material itself can be packaged this way. `skills/claude-api` pairs a `SKILL.md` with per-SDK markdown documentation in subfolders for Go, Java, PHP, C#, and more, while `skills/academy-guide` and `skills/discernment-nudge` adjust assistant behavior with instructions alone.

End to end, the flow looks like this: a user asks Claude to extract form fields from a PDF; the agent compares the request against the short descriptions it knows from the installed skills; `skills/pdf` matches, so its `SKILL.md` body is loaded; the instructions direct the agent to the bundled scripts and verification steps; and the agent executes, checks, and returns a result — the skill having shaped every step without any of its content being present at the start of the conversation.

## Advantages

- **Minimal, standardized format.** A skill requires only a folder and a `SKILL.md` with two frontmatter fields, a contract documented in `template/SKILL.md` and the Agent Skills specification at agentskills.io.
- **Targeted loading instead of prompt bloat.** Descriptions are scanned cheaply; full instructions and scripts load only on match, keeping unrelated conversations clean.
- **Executable verification, not just prose.** Skills like `docx` ship validators (`scripts/office/validate.py` against real OOXML XSD schemas) so the agent can check its own output rather than hoping for the best.
- **Production provenance.** The document skills backing Claude's own document capabilities live in this repo, giving skill authors an honest reference for complexity that actually ships.
- **Batteries-included distribution.** The `.claude-plugin/marketplace.json` manifest plus Claude Code's plugin commands make the whole collection installable in two commands, and the same mechanism works for private team skills.
- **Honest licensing boundaries.** Most example skills carry Apache 2.0 `LICENSE.txt` files, while the document skills are clearly marked as source-available rather than open source — no ambiguity about what you may reuse.

## Benefits

- **Faster authoring of your own skills.** The template plus the `skill-creator` toolkit — including evaluation, description optimization, and packaging scripts — turn skill creation into a guided loop instead of guesswork.
- **More reliable document automation.** Redlining, comment handling, and XSD validation in the `docx` skill address the classic failure modes of Office XML editing, which is exactly the kind of knowledge no model carries by default.
- **Consistent enterprise communication.** `brand-guidelines`, `internal-comms` with its worked examples, and `theme-factory` with its ten preset themes keep output on-brand without re-explaining style rules each session.
- **Better MCP servers.** `mcp-builder`'s references on tool naming, API coverage, and error messages encode hard-won integration lessons you can apply to any LLM tool integration, not just Claude.
- **Disciplined webapp testing.** `webapp-testing`'s `with_server.py` removes the fiddly server-bootstrapping half of UI automation, leaving the agent to focus on actual browser assertions via Playwright.
- **A free education in agent design.** The entire collection is inspectable — the Apache 2.0 skills can be studied, forked, and adapted — making the repo a self-guided course on teaching agents specialized work.

## Usage

Register the repository as a Claude Code plugin marketplace and install a plugin (commands from the README):

```bash
/plugin marketplace add anthropics/skills
```

Then either browse and install via `Browse and install plugins`, or install a plugin directly:

```bash
/plugin install document-skills@anthropic-agent-skills
/plugin install example-skills@anthropic-agent-skills
```

After installing, just mention the skill in a request — for example, "Use the PDF skill to extract the form fields from `path/to/some-file.pdf`".

The same skills are available to paid plans in Claude.ai, and Anthropic's pre-built skills plus custom uploads are usable through the Claude API via the Skills API quickstart.

Creating your own basic skill needs nothing but a folder and a `SKILL.md` (the minimal shape from the README):

```markdown
---
name: my-skill-name
description: A clear description of what this skill does and when to use it
---

# My Skill Name

[Add your instructions here that Claude will follow when this skill is active]
```

The frontmatter requires only `name` (lowercase, hyphens for spaces) and `description`; everything below is the instruction body Claude follows once the skill triggers.

## Conclusion

anthropics/skills is that rare official repository that doubles as documentation: the concept of an Agent Skill is defined not by a long standard document but by a template file, a marketplace manifest, and nineteen working examples ranging from two-line instruction files to a document-editing toolkit with Python scripts and OOXML schemas. If you are building agents — on Claude or anywhere else — an afternoon reading `skills/docx/SKILL.md`, `skills/skill-creator/`, and `.claude-plugin/marketplace.json` will teach you more about practical agent ergonomics than most long essays on the subject.

Links:

- GitHub repository: [https://github.com/anthropics/skills](https://github.com/anthropics/skills)
- Agent Skills specification: [https://agentskills.io/specification](https://agentskills.io/specification)
- Anthropic engineering post on Agent Skills: [https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- What are skills? (Claude support): [https://support.claude.com/en/articles/12512176-what-are-skills](https://support.claude.com/en/articles/12512176-what-are-skills)
