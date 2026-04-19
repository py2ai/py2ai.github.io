---
layout: post
title: "GBrain - Self-Wiring Knowledge Graph For AI Agents"
description: "GBrain gives AI agents a brain. Built by YC President Garry Tan, it powers his real deployment: 17,888 pages, 4,383 people, 723 companies, 21 cron jobs. Self-wiring knowledge graph, hybrid search, Minions job queue, 26 skills."
date: 2026-04-20
header-img: "assets/img/ai-coding-frameworks/ai-coding-frameworks"
permalink: /gbrain-self-wiring-knowledge-graph-ai-agents/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags: [AI, Knowledge-Graph, Agent-Framework, Open-Source, Y-Combinator, Postgres]
author: "PyShine"
---

Your AI agent is smart but forgetful. Every conversation starts from scratch. Every meeting context is lost. Every person you mention is a stranger. **GBrain** fixes this by giving your agent a persistent, self-wiring knowledge graph that compounds daily.

Built by Garry Tan -- President and CEO of Y Combinator -- to run his actual AI agents, GBrain is the production brain powering his OpenClaw and Hermes deployments: **17,888 pages, 4,383 people, 723 companies**, 21 cron jobs running autonomously, built in 12 days. The agent ingests meetings, emails, tweets, voice calls, and original ideas while you sleep. It enriches every person and company it encounters. It fixes its own citations and consolidates memory overnight. You wake up and the brain is smarter than when you went to bed.

## Architecture Overview

![GBrain Architecture](/assets/img/diagrams/gbrain/gbrain-architecture.svg)

GBrain follows a three-pillar architecture that separates concerns cleanly while maintaining bidirectional data flow. The Brain Repo (git) serves as the source of truth -- plain markdown files that humans can always read and edit directly. GBrain itself is the retrieval layer, powered by Postgres with pgvector for hybrid search combining vector similarity, keyword matching, and Reciprocal Rank Fusion. The AI Agent reads and writes through both layers, with 26 skills defining how to use the brain and RESOLVER.md routing intent to the appropriate skill.

The engine layer is pluggable: PGLite (embedded Postgres 17.5) runs locally with zero configuration in `~/.gbrain/brain.pglite`, while PostgresEngine connects to Supabase Pro for production deployments at scale. Bidirectional migration between engines means you can start local and move to Supabase when your brain outgrows a single machine. The `gbrain sync` command picks up any manual edits to markdown files, ensuring the human always wins -- edit any file and the brain stays in sync.

## The 26 Skills

![Skills Overview](/assets/img/diagrams/gbrain/gbrain-skills-overview.svg)

GBrain ships 26 skills organized by RESOLVER.md, which acts as the skill dispatcher telling your agent which skill to read for any given task. The philosophy is "thin harness, fat skills" -- the intelligence lives in the skills, not the runtime. Skills are fat markdown documents that encode entire workflows: when to fire, what to check, how to chain with other skills, and what quality bar to enforce. The agent reads the skill and executes it, while deterministic TypeScript code handles the parts that should not be left to LLM judgment.

The Always-On skills run on every message. Signal-detector fires a cheap model in parallel to capture original thinking and entity mentions, making the brain compound on autopilot. Brain-ops implements the brain-first lookup pattern -- checking the brain before any external API call, creating the read-enrich-write loop that makes every response smarter. Content Ingestion skills handle different input types: idea-ingest for links and articles, media-ingest for video, audio, PDF, and GitHub repos, and meeting-ingestion for transcripts with attendee enrichment. Brain Operations skills cover the full lifecycle: enrich with tiered person/company pages, query with 3-layer search, maintain with periodic health checks, and citation-fixer for format consistency. Operational skills manage tasks, cron scheduling, reports, cross-modal quality gates, webhook transforms, and the Minion orchestrator for background jobs.

## The Self-Wiring Knowledge Graph

Pages in GBrain are not just text. Every mention of a person, company, or concept becomes a typed link in a structured graph. The brain wires itself with zero LLM calls. When you write a meeting page mentioning Alice and Acme AI, the auto-link extraction post-hook parses entity references from content using regex patterns, infers relationship types (attended, works_at, invested_in, founded, advises), reconciles stale links when content changes, and uses backlinks to rank well-connected entities higher in search.

The typed inference cascade works through pattern matching: "CEO of X" maps to works_at, "invested in" maps to invested_in, "advises" or "advisor" maps to advises, "founded" or "co-founded" maps to founded. Code-fence stripping prevents false positives from slugs inside code blocks. Within-page dedup collapses multiple references to the same target into a single link. Multi-type link constraints allow the same person to have both works_at and advises relationships simultaneously.

This graph powers questions that vector search alone cannot answer: "who works at Acme AI?", "what has Bob invested in?", "find the connection between Alice and Carol." Graph traversal uses recursive CTEs with cycle prevention, type-filtered edges, direction control, and depth caps for DoS prevention. Backfill an existing brain in one command with `gbrain extract links --source db`.

## Hybrid Search Pipeline

![Search Pipeline](/assets/img/diagrams/gbrain/gbrain-search-pipeline.svg)

The search pipeline layers approximately 20 deterministic techniques together, where no single technique is magic but the win comes from stacking them so each layer covers what the others miss. The pipeline starts with an intent classifier that categorizes queries as entity, temporal, event, or general, then routes accordingly. Multi-query expansion uses Claude Haiku to rephrase the question three ways, capturing different phrasings of the same information need.

The parallel search paths run vector search (HNSW cosine over OpenAI embeddings) and keyword search (Postgres tsvector with websearch_to_tsquery) simultaneously. Reciprocal Rank Fusion merges both result sets using the formula `score = sum(1/(60 + rank))`, which balances the strengths of each approach. Vector search catches conceptual matches that keyword search misses; keyword search catches exact slug references that vector search misses. RRF picks the best of both.

After fusion, cosine re-scoring re-ranks chunks against the actual query embedding, compiled-truth boost ensures assessments outrank timeline noise, and backlink boost ranks well-connected entities higher. A 4-layer dedup with compiled truth guarantee ensures one clean chunk per page in the final results. For relational queries, the intent classifier routes to graph traversal first (high-precision typed answers) with grep fallback when the graph returns nothing.

The benchmarks are striking. On the BrainBench v1 corpus (240 rich-prose pages), Recall@5 jumps from 83% to 95%, Precision@5 from 39% to 45%, yielding 30 more correct answers in the agent's top-5 reads. Graph-only F1 hits 86.6% compared to grep's 57.8% -- a 28.8 percentage point improvement. All deterministic, all in concert, all measured.

## Minions: Durable Background Jobs

![Minions vs Sub-agents](/assets/img/diagrams/gbrain/gbrain-minions-vs-subagents.svg)

Minions is a durable, Postgres-native job queue built into the brain that solves the six daily pains of using sub-agents for deterministic work: spawn storms, agents that stop responding, forgotten dispatches, gateway crashes mid-run, runaway grandchildren, and debugging soup. The routing rule is simple: deterministic work (same input produces same steps and same output) goes to Minions; judgment work (input requires assessment or decision) goes to sub-agents.

The production numbers tell the story. Under a 19-cron load on Garry's personal OpenClaw deployment, pulling a month of social posts and ingesting them end-to-end: Minions completed in 753ms with $0.00 token cost and 100% success rate. Sub-agents could not even spawn within the 10-second gateway timeout, resulting in 0% success. At scale, 19,240 posts across 36 months processed via Minions in approximately 15 minutes total for $0.00, while sub-agents would take approximately 9 minutes best case, cost approximately $1.08 in tokens, and fail approximately 40% of the time.

Minions achieves this through `max_children` caps, `timeout_ms` with AbortSignal, `child_done` inbox for result collection, full `parent_job_id`/depth/transcript per job, Postgres durability with stall detection, cascade cancel via recursive CTE, idempotency keys, attachment validation, and `removeOnComplete` cleanup. The `gbrain jobs smoke` command proves the install in half a second.

## The Knowledge Model

Every page follows the compiled truth plus timeline pattern. Above a `---` separator sits the compiled truth: your current best understanding, which gets rewritten when new evidence changes the picture. Below sits the timeline: an append-only evidence trail that is never edited, only added to. This separation means the brain maintains both a clean, current summary and a complete audit trail of how that summary evolved.

Entity enrichment auto-escalates based on mention frequency. A person mentioned once gets a stub page (Tier 3). After 3 mentions across different sources, they get web and social enrichment (Tier 2). After a meeting or 8+ mentions, the full pipeline activates (Tier 1). The brain learns who matters without being told. Deterministic classifiers improve over time via a fail-improve loop that logs every LLM fallback and generates better regex patterns from the failures. `gbrain doctor` shows the trajectory: "intent classifier: 87% deterministic, up from 40% in week 1."

## Skillify: Keeping the Skills Tree Honest

Hermes and similar agent frameworks auto-create skills as a background behavior, which works until you do not know what the agent shipped. Skillify turns raw code into a properly-skilled feature through a 10-item checklist: SKILL.md, deterministic script, unit tests, integration tests, LLM evals, resolver trigger, resolver trigger eval, E2E smoke, and brain filing. Every item is required.

`gbrain check-resolvable` walks the whole skills tree checking reachability, MECE overlap, DRY violations, gap detection, and orphaned skills. It exits non-zero if anything is off. In practice this produces zero orphaned skills with every feature having tests, evals, resolver triggers, and evals of the triggers. Compounding quality instead of compounding entropy.

## Getting Data In

GBrain ships integration recipes that your agent sets up for you. Each recipe tells the agent what credentials to ask for, how to validate, and what cron to register. Available recipes include Public Tunnel (ngrok for MCP and voice), Credential Gateway (Gmail and Calendar access), Voice-to-Brain (phone calls to brain pages via Twilio and OpenAI Realtime), Email-to-Brain (Gmail to entity pages), X-to-Brain (Twitter timeline and mentions), Calendar-to-Brain (Google Calendar to searchable daily pages), and Meeting Sync (Circleback transcripts to brain pages with attendees).

Data research recipes extract structured data from email into tracked brain pages, with built-in recipes for investor updates (MRR, ARR, runway, headcount), expense tracking, and company metrics. Create your own with `gbrain research init`.

## Installation

**On an agent platform (recommended):**

Paste this into your OpenClaw or Hermes agent:

```
Retrieve and follow the instructions at:
https://raw.githubusercontent.com/garrytan/gbrain/master/INSTALL_FOR_AGENTS.md
```

The agent clones the repo, installs GBrain, sets up the brain, loads 26 skills, and configures recurring jobs. You answer a few questions about API keys. Approximately 30 minutes.

**Standalone CLI:**

```bash
git clone https://github.com/garrytan/gbrain.git && cd gbrain && bun install && bun link
gbrain init                     # local brain, ready in 2 seconds
gbrain import ~/notes/          # index your markdown
gbrain query "what themes show up across my notes?"
```

**MCP server (Claude Code, Cursor, Windsurf):**

```json
{
  "mcpServers": {
    "gbrain": { "command": "gbrain", "args": ["serve"] }
  }
}
```

## GBrain + GStack

GStack is the coding engine (70,000+ stars, 30,000 developers per day). GBrain is the everything-else mod. The bridge file `hosts/gbrain.ts` tells GStack's coding skills to check the brain before coding. Together they give your agent both coding intelligence and persistent memory.

## The Origin

Garry Tan was setting up his OpenClaw agent and started a markdown brain repo. One page per person, one page per company, compiled truth on top, timeline on the bottom. Within a week: 10,000+ files, 3,000+ people, 13 years of calendar data, 280+ meeting transcripts, 300+ captured ideas. The agent runs while he sleeps. The dream cycle scans every conversation, enriches missing entities, fixes broken citations, consolidates memory. What took 11 days to build by hand ships as a mod you install in 30 minutes.

**Repository:** [github.com/garrytan/gbrain](https://github.com/garrytan/gbrain)  
**License:** MIT  
**Stars:** 9,400+