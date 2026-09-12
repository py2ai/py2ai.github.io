---
layout: post
title: "TeamAI: Make Every Team AI Native with Git-Native Skill Sync"
description: "TeamAI is an open source CLI from Tencent that manages your team's skills, rules, MCP, and knowledge across Claude Code, Codex, Cursor, CodeBuddy, OpenCode, OpenClaw, and other AI coding agents. It uses a shared Git repo as the single source of truth, distributing harness resources through a push-review-merge-pull workflow with SessionStart hooks for automatic sync. The three-layer architecture covers Team Execution (skills, rules, agents, hooks, MCP, env), Team Context (friction-scored learnings, BM25 + graph-boosted recall, codebase knowledge graph with WASM tree-sitter AST extraction), and Team Improvement (weekly digest, session analytics, web dashboard, KB maintenance). Built with TypeScript, Node 20+, commander, simple-git, and web-tree-sitter. MIT licensed, 2.9k stars, trending on GitHub. This post breaks down the architecture, the push-pull workflow, the codebase knowledge graph pipeline, and the agent support matrix."
date: 2026-09-12
header-img: "img/post-bg.jpg"
permalink: /TeamAI-Make-Every-Team-AI-Native/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - TeamAI
  - Tencent
  - AI Agents
  - Claude Code
  - Skills
  - Knowledge Graph
  - Open Source
  - Developer Tools
author: PyShine
---

## What is TeamAI

TeamAI is an open source CLI from Tencent that manages your team's skills, rules, MCP, and knowledge across Claude Code, Codex, Cursor, CodeBuddy, OpenCode, OpenClaw, and other AI coding agents. The code is on GitHub at [Tencent/teamai-cli](https://github.com/Tencent/teamai-cli), MIT licensed, with 2.9k stars and trending on GitHub.

The core idea is simple but powerful: a single shared Git repo acts as the source of truth for your team's entire AI harness. Skills, rules, docs, agents, hooks, MCP configurations, environment variables, culture files, and CLAUDE.md templates all live in that repo. When an admin pushes changes through a Merge Request and they get merged, every team member's local AI sessions automatically pull the latest resources on startup via a SessionStart hook. No manual sync needed.

TeamAI supports six Git providers (GitHub, GitLab, GitCode, CNB, TGit, and private Git services) and eleven AI agents. The npm package is [teamai-cli](https://www.npmjs.com/package/teamai-cli) (v0.22.0), installable globally with `npm install -g teamai-cli`.

## System Architecture

TeamAI is built with TypeScript, Node 20+, tsup (ESM bundler), and Vitest for testing. It uses commander for CLI parsing, simple-git for Git operations, gray-matter for frontmatter parsing, web-tree-sitter for AST-based code analysis, and zod for schema validation.

![TeamAI three-layer architecture](/assets/img/diagrams/teamai-cli/teamai-architecture.svg)

### Understanding the Architecture

The architecture is organized into three layers that build on each other. The shared team repo sits at the center, acting as the hub that every agent and every layer connects through.

**Layer 1: Team Execution**

Team Execution is the foundation. It manages the harness resources that define how every AI agent in your team operates. The resources include:

- **Skills** stored as `skills/<name>/SKILL.md` files, each defining a discrete capability the AI agent can invoke
- **Rules** stored as `rules/*.md` files that constrain agent behavior and coding standards
- **Docs** in `docs/` for foundational project documentation with progressive disclosure
- **Agents** as `agents/<name>.yaml` defining subagent configurations
- **Hooks** in `hooks/hooks.yaml` for lifecycle event handlers (SessionStart, Stop, etc.)
- **MCP** in `mcp/mcp.yaml` for Model Context Protocol server configurations
- **Env** in `env/` for shared team-level environment variables and switches
- **Culture** as `culture.md` defining team mission, values, and working principles injected into each agent's CLAUDE.md

The key design decision is that all these resources live in a Git repo, not a database. This means the entire team's AI configuration is version-controlled, reviewable through Merge Requests, and diffable. An admin changes a skill, pushes a branch, opens an MR, a reviewer approves it, and once merged, every team member gets the update automatically on their next AI session.

**Layer 2: Team Context (beta)**

Team Context makes every agent understand the team. Beyond just distributing the harness, TeamAI organizes accumulated team experience and code structure into a searchable knowledge base. This includes:

- **Shared Learnings**: When a session ends, the Stop hook scores it by friction signals (interruptions, denied tool calls, retries). If the friction score is high enough, the AI suggests sharing what was learned.
- **Team Knowledge Recall**: BM25 keyword search combined with graph-boosted re-ranking. A dedicated subagent (`teamai-recall`) extracts keywords from the user's task, runs the search, reads matched files, and returns a structured summary.
- **Codebase Knowledge Graph**: Source repositories are parsed into a structured graph of components, interfaces, configs, and cross-repo import edges. This enables structurally-aware retrieval.
- **TeamWiki**: Deep enrichment and reconciliation generate knowledge docs from extracted code evidence.

**Layer 3: Team Improvement (beta)**

Team Improvement turns session data into actionable insights. The weekly digest shows 7-day success rates, prompt counts, active time, estimated costs, cache utilization, and correction trends. Session analytics provide privacy-scrubbed summaries of tool sequences and interventions. A web dashboard shows live session status and KB health metrics. KB maintenance archives low-confidence learnings and flags stale skills for cleanup.

**Git Providers**

The architecture supports six Git providers, making it adaptable to different organizational environments. GitHub and GitLab are the most common for open source teams. GitCode and CNB serve Chinese enterprise environments. TGit is Tencent's internal Git service. Private Git services are supported through a generic provider. Each provider implements the same interface: clone, push, open MR, fetch MR, and org-level operations.

## Push / Review / Merge / Pull Workflow

The distribution flow is the heartbeat of TeamAI. It ensures that changes to the team's harness go through proper review before reaching every member's local tools.

![TeamAI push-pull workflow](/assets/img/diagrams/teamai-cli/teamai-push-pull-flow.svg)

### Understanding the Workflow

The diagram shows the complete lifecycle from a resource change to team-wide distribution, including the feedback loop that makes every execution improve the team.

**Step 1: teamai push**

When a team member creates or modifies a skill, rule, or any harness resource locally, they run `teamai push`. This command creates a feature branch, commits the changes, pushes to the team repo, and opens a Merge Request on the configured Git provider. The MR includes coauthor tracking and path-filtered push guidance.

**Step 2: Team Review**

The MR goes through standard code review. Reviewers can comment, request changes, and track what was modified. TeamAI supports MR comments with structured hints. The reviewer approves and merges the MR to the main branch.

**Step 3: SessionStart Hook Auto-Pull**

Once merged, the change does not require manual distribution. Every team member has a SessionStart hook installed by `teamai init`. When they start a new AI session (in Claude Code, Codex, Cursor, or any supported agent), the hook fires `teamai pull` automatically. This pulls the latest resources from the team repo and injects them into the local AI tool's configuration directories.

The pull process respects distribution controls:
- **Roles**: Each member syncs only the skills mapped to their role namespace
- **Tags**: Members subscribe to specific tags and only receive tagged resources
- **Exclude**: Members can exclude specific skills they do not need locally
- **Sources**: Subscribed external repos sync automatically during pull

**Step 4: Session Runs with Shared Resources**

The AI session now runs with the full team harness: skills, rules, hooks, MCP servers, culture file, and environment variables all synced. Every agent in the team operates from the same configuration.

**Step 5: Stop Hook and Friction Scoring**

When the session ends, the Stop hook scores it by friction. Friction signals include: the user interrupted the AI, the user denied a tool call, the AI had to retry failing tools. A long-but-routine session with many tool calls but no friction does not trigger. A short session where the user fought a real problem does. If the friction score exceeds the threshold, the AI suggests running `/teamai-share-learnings`.

**Step 6: Feedback Loop**

The `/teamai-share-learnings` skill summarizes the session, extracts what was learned, and pushes a learning document to the team repo. This creates a feedback loop: executions improve the team's knowledge base, which in turn improves future executions. The dashed purple arrow in the diagram represents this iterative improvement cycle.

## Codebase Knowledge Graph and Recall Pipeline

The codebase knowledge graph is what makes TeamAI's recall structurally aware. Instead of relying solely on keyword matching, it builds a graph of code relationships and uses those edges to boost relevant results.

![TeamAI knowledge graph and recall pipeline](/assets/img/diagrams/teamai-cli/teamai-knowledge-graph.svg)

### Understanding the Knowledge Graph

The diagram shows two main pipelines: the extraction pipeline (top) that builds the knowledge graph, and the recall pipeline (bottom) that queries it.

**Extraction Pipeline: Two Tracks**

The extraction pipeline runs two tracks in parallel, with AST results taking precedence on overlap:

- **AST Track**: Uses a WASM-based tree-sitter parser to resolve precise code relationships. For TypeScript/JavaScript, it handles `import`/`require` statements, call sites, and TypeScript `implements` clauses. For Python, it parses import statements and function calls. For Go, it resolves package imports. The AST track produces three edge types: `DEPENDS_ON` (file A imports from file B), `REFERENCES` (file A calls a function defined in file B), and `IMPLEMENTS` (file A implements an interface defined in file B). Each edge is tagged `code-ast` with a confidence weight.

- **Heuristic Track**: Uses regex-based extraction for all languages, including Java and Rust that the AST track does not cover. Edges are tagged `code-heuristic`. This track also serves as a fallback when the AST parser fails to load. If that happens, an `AST_UNAVAILABLE` gap is recorded so teams know which edges may be less precise.

The WASM parser is a pure-JavaScript dependency, requiring no native toolchain. This is a deliberate design choice for portability. Teams can force heuristic-only extraction by setting `TEAMAI_SKIP_AST=1`.

**Edge Merge and Storage**

Edges from both tracks are merged into a unified graph stored under the `teamwiki/` directory. The graph stores components, interfaces, configs, and cross-repo import edges. When AST and heuristic edges overlap for the same file pair, the AST result takes precedence because it is structurally precise.

**Deep Enrichment and Reconciliation**

After extraction, two post-processing steps improve the graph:
- **Deep Enrich** (`teamai codebase --deep-enrich`): Generates detailed knowledge documents from the extracted evidence, turning raw graph data into readable docs.
- **Reconcile** (`teamai codebase --reconcile`): Maps product documentation pages to their corresponding code pages, ensuring docs and code stay aligned. A lint command checks the graph for health issues.

**Recall Pipeline: BM25 + Graph-Boosted Re-ranking**

The recall pipeline is triggered when an AI agent needs team knowledge before a task:

1. **Query**: The user's task or query enters the system via `teamai recall <query>`
2. **Subagent**: The `teamai-recall` subagent (deployed into each AI tool's `agents/` directory by `teamai pull`) extracts keywords from the task, runs the search, reads matched source files, and returns a structured summary
3. **Relevance Precheck**: Before running the full search, the subagent runs `teamai recall --check` to determine if the task is even related to team knowledge. If not, retrieval is skipped entirely, saving tokens and time
4. **BM25 Search**: Keyword relevance ranking across learnings and codebase documents using a BM25 index
5. **Graph-Boosted Re-ranking**: The codebase knowledge graph edges boost results that are structurally related to the query. If a recall hit comes from a codebase page, the result includes a `Sources:` line listing the relevant source file paths
6. **Ranked Results**: Results include score, tags, author, and source file paths, giving agents a direct starting point for code changes

## Agent Support Matrix

TeamAI supports eleven AI agents with varying levels of integration. The matrix shows which capabilities each agent supports.

![TeamAI agent support matrix](/assets/img/diagrams/teamai-cli/teamai-agent-matrix.svg)

### Understanding the Agent Matrix

The diagram visualizes the support matrix across three layers and the distribution controls that admins configure once.

**Full Support Agents**

Claude Code, Codex, Cursor, CodeBuddy, and Qoder have full support across all three layers. They support all seven Team Execution resources (skills, rules, docs, env, agents, hooks, MCP), all three Team Context capabilities (learnings, codebase, teamwiki), and all three Team Improvement features (usage, sessions, dashboard). These agents have the deepest integration with TeamAI.

**Partial Support Agents**

WorkBuddy lacks agents support but has everything else. OpenCode lacks teamwiki, usage, sessions, and dashboard (the entire Team Improvement layer and teamwiki from Team Context). OpenClaw lacks agents, hooks, and MCP but retains skills, rules, docs, env plus the full Team Context layer. Hermes has skills, docs, env, and context but lacks rules, agents, hooks, and MCP. DeepSeek Harness has the most limited support with only skills, docs, and context. ZCode lacks rules support but has the rest of execution plus context and improvement.

**Distribution Controls**

Three team-wide settings that an admin configures once and delivers to every member on `teamai pull`:

- **Roles** (`teamai roles`): Define role-to-namespace mappings so each member syncs only the skills for their role. A backend developer gets backend skills; a frontend developer gets frontend skills.
- **Tags** (`teamai tags`): Tag skills and rules so members subscribe to just the tags they need. A member working on deployment subscribes to the `deploy` tag and gets only deployment-related skills.
- **Sources** (`teamai source`): Subscribe to additional skill repos from other teams or shared/public repos within your own org. Subscribed skills sync automatically on pull, enabling cross-team skill sharing without manual copying.

## Installation

Install TeamAI globally via npm:

```bash
npm install -g teamai-cli
```

### Team Admin Setup

Create a shared-experience repo on your Git host (GitHub, GitLab, GitCode, CNB, TGit, or a private Git service), grant write access to team members, then initialize:

```bash
# Project-scope init (default, resources installed under the project directory)
cd /path/to/my-project
teamai init https://github.com/yourorg/yourrepo

# Or, user-scope init (resources installed under ~/)
teamai init https://github.com/yourorg/yourrepo --scope user
```

If you do not have a team repo yet, browse the [teamai-hub](https://github.com/teamai-hub) org on GitHub, click "Use this template" on a pre-loaded repo, then run `teamai init` against your new repo.

### Team Member Setup

Once the admin has set up the team repo, members simply run:

```bash
cd /path/to/my-project
teamai init https://github.com/yourorg/yourrepo
```

After initialization, every AI session automatically pulls the latest skills, rules, and other harness updates published by admins. No manual sync needed.

## Usage

### Core Commands

| Command | Description |
|---------|-------------|
| `teamai init` | Initialize: OAuth login, link repo, register member, inject hooks |
| `teamai pull` | Pull team resources and inject into local AI tools |
| `teamai push` | Push local resources to a branch and open a Merge Request |
| `teamai status` | Show local vs team repo diff |
| `teamai packages [install] [target]` | Install declared npm packages and Claude plugins |
| `teamai members` | List team members |
| `teamai roles` | Manage team roles and namespaces |
| `teamai tags` | Manage tag-based skill/rule filtering |
| `teamai source` | Manage skill subscription sources |
| `teamai doctor` | Diagnose configuration issues |
| `teamai uninstall` | Remove all teamai resources and hooks |

### Knowledge and Codebase Commands

| Command | Description |
|---------|-------------|
| `teamai recall <query>` | Search the team knowledge base (BM25 + graph-boost) |
| `teamai recall enable/disable/status` | Toggle or check recall state |
| `teamai recall promote [learningId]` | Promote a high-confidence learning to formal knowledge |
| `teamai recall maintenance` | Prune low-confidence learnings, flag stale entries |
| `teamai import` | Import knowledge (--dir, --from-repo, --from-org, --from-mr) |
| `teamai codebase --extract [path]` | Extract code facts and build the local graph |
| `teamai codebase --deep-enrich` | Generate deep knowledge docs from extracted evidence |
| `teamai codebase --reconcile` | Reconcile product docs with extracted code knowledge |
| `teamai codebase --lint` | Knowledge graph health check |

### Analytics Commands

| Command | Description |
|---------|-------------|
| `teamai digest` | Generate weekly team usage digest |
| `teamai session save` | Record a privacy-scrubbed session summary |
| `teamai dashboard` | Launch web dashboard with live sessions and KB health |
| `teamai contribute` | Share session experience to team repo |

### Enabling Team Knowledge Recall

Recall is off by default. Enable it explicitly:

```bash
# Enable: deploy the teamai-recall subagent + inject guidance rules
teamai recall enable

# Disable: remove the subagent and rules
teamai recall disable

# Show effective state (team default + user override)
teamai recall status
```

### Importing Codebase Knowledge

Build the codebase knowledge graph from your source repositories:

```bash
# Import from a single repo
teamai import --from-repo https://github.com/org/repo

# Batch import all repos from an org
teamai import --from-org myorg

# Local extract into teamwiki/
teamai codebase --extract /path/to/repo

# Generate deep knowledge docs
teamai codebase --deep-enrich --project my-service --output /path/to/repo

# Reconcile product docs with code
teamai codebase --reconcile --output /path/to/repo

# Check graph health
teamai codebase --lint --output /path/to/repo
```

## Key Features

| Feature | Description |
|---------|-------------|
| Git-native | All team harness resources live in a Git repo, version-controlled and reviewable through MRs |
| Multi-agent | Supports 11 AI agents across Claude Code, Codex, Cursor, CodeBuddy, WorkBuddy, OpenCode, OpenClaw, Hermes, DeepSeek Harness, Qoder, and ZCode |
| Multi-provider | Works with GitHub, GitLab, GitCode, CNB, TGit, and private Git services |
| Auto-sync | SessionStart hook pulls latest resources automatically on every AI session |
| Friction scoring | Stop hook scores sessions by friction signals to identify valuable learnings |
| Knowledge graph | WASM tree-sitter AST extraction builds a codebase graph for structurally-aware recall |
| BM25 + graph-boost | Recall combines keyword search with graph-boosted re-ranking for relevant results |
| Distribution controls | Roles, tags, and sources filter what each member receives |
| Analytics | Weekly digest, session analytics, web dashboard, and KB health monitoring |
| Self-improving | Feedback loop: executions produce learnings that improve future executions |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Resources not syncing | SessionStart hook not installed | Run `teamai init` to reinstall hooks, or `teamai doctor` to diagnose |
| Recall returns no results | Recall not enabled or empty knowledge base | Run `teamai recall enable`, then `teamai import --from-repo` to populate |
| AST extraction fails for Java/Rust | AST track only covers TS/JS/Python/Go | Falls back to heuristic track automatically; `AST_UNAVAILABLE` gap recorded |
| Push opens MR on wrong provider | Git provider misconfigured | Check `teamai.yaml` provider config; run `teamai doctor` |
| Agent not receiving all resources | Role or tag filter excluding them | Check `teamai roles` and `teamai tags` configuration |
| Dashboard not showing data | Sessions not saved | Run `teamai session save --push` to record and feed the digest |
| npm install fails globally | Node version too old | Requires Node 20+; check with `node --version` |

## Conclusion

TeamAI solves a real problem in the age of multi-agent AI coding: how do you keep an entire team's AI harness consistent, reviewable, and self-improving? The answer is elegantly Git-native. By storing skills, rules, hooks, MCP, and knowledge in a shared repo with a push-review-merge-pull workflow, TeamAI brings the same discipline to AI configuration that teams already apply to application code. The friction-based learning sharing and codebase knowledge graph with tree-sitter AST extraction make it more than a config sync tool, it is a system that gets smarter with every session.

The project is actively developed by Tencent, with v0.22.0 currently on npm, 47 published versions, comprehensive test coverage (over 200 test files including e2e tests for multiple Git providers), and active discussions on GitHub. The three-layer architecture (Execution, Context, Improvement) provides a clear roadmap from basic harness distribution to full team intelligence.

## Related Posts

- [CowAgent: Open Source Super AI Assistant](/cowagent-open-source-super-ai-assistant/)
- [WeKnora: Tencent Open Source Knowledge Framework](/WeKnora-Tencent-Open-Source-Knowledge-Framework-RAG-Agents-Wiki/)
- [Everything Claude Code: AI Agent Harness](/Everything-Claude-Code-AI-Agent-Harness/)
