---
layout: post
title: "ksor: One Governed Knowledge Record for Humans and AI Agents - Inside panaversity/ksor"
description: "A source-level tour of panaversity/ksor, the open-source TypeScript SDK that compiles governed Markdown into a documentation site for people and an MCP server for AI agents, with citations, provenance, and measured abstention. We walk the record checker, the governance policy engine, ingest generations, and the fail-closed agent gateway."
date: 2026-09-30
header-img: "img/post-bg.jpg"
permalink: /Ksor-Knowledge-System-Of-Record-SDK-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/ksor/panaversity-ksor-architecture.svg
tags:
  - KSoR
  - MCP
  - Knowledge Management
  - AI Agents
categories: [AI, Open Source]
keywords: "ksor, knowledge system of record, panaversity, MCP, model context protocol, AI agents, knowledge governance, Open Knowledge Format, OKF, pgvector, abstention, citations, provenance, TypeScript SDK, llms.txt"
author: "PyShine"
---

Ask two AI agents in the same company a policy question and you can get two different answers — not because the models are weak, but because nobody ever defined which knowledge wins. The answers get assembled from an old wiki page, a current policy PDF, a Slack thread, a stale RAG index, and whatever the model remembers from training. A retrieval system can rank those sources by relevance; it cannot manufacture organizational authority, because the institution itself never declared it.

**ksor** — short for Knowledge System of Record — is the open-source SDK from Panaversity that attacks exactly that gap. It gives an organization one governed, authoritative knowledge record written in plain Markdown, and it publishes that record to every audience that needs it: a documentation site for people, an `llms.txt` discovery file for AI systems, an MCP server for agents, and portable OKF exchange bundles for other knowledge systems. The record lives in your repository, under your control, with no vendor owning it.

The source is worth a tour because the project's central claim — governed, bounded, traceable knowledge — is not marketing copy. It is enforced in code, and sometimes enforced with unusual care: a build that fails writes nothing, a server that would be unauthenticated refuses to boot, an uncalibrated record refuses to answer rather than guessing. Reading `panaversity/ksor` is a lesson in what "governance" means when it is mechanical rather than descriptive.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ksor/panaversity-ksor-overview-architecture.svg" alt="Architecture overview of the panaversity/ksor repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the ksor monorepo: a governed Markdown record on the left, a governance boundary that checks it, the build-and-publish pipeline, and the surfaces that serve it.*

Reading the overview from left to right: authors write the record as ordinary Markdown under `knowledge/`, with an `instance.md` declaring what the record is authoritative for. The governance boundary — `.ksor/governance.yaml` feeding the record checker — is the root of authority every publication decision is validated against. The `ksor` CLI drives the pipeline: `ksor build` checks the whole record and writes its lock, and `ksor ingest` chunks the corpus into an immutable generation inside a Postgres/pgvector store. From there the serving surfaces fan out — a Fumadocs site for humans, and the content kernel behind an MCP gateway that serves agents floor-gated, cited passages.

## Why You Need This

The problem ksor solves has a precise shape. Enterprise and educational knowledge is fragmented across wikis, decks, PDFs, prompts, RAG indexes, and human memory, and none of those artifacts answers the question an agent actually needs answered: *which of these is authoritative?* A traditional system of record — an accounting ledger, a CRM — settles what is operationally true. Nothing in the classic stack settles what the organization knows and how it says agents should operate from it. ksor exists to be that missing layer: it records policies, procedures, rules, standards, definitions, decision criteria, and — in an education setting — curriculum, learning objectives, and assessment rules.

Retrieval does not fix this, and the project is refreshingly explicit that it does not try to replace retrieval with better vibes. A model can rank what looks relevant; it cannot reliably invent authority the institution never defined. The interesting design move in ksor is making that distinction *executable*: the governance policy in `.ksor/governance.yaml` names audience registries, knowledge owners, approval authorities, and takedown authorities, and the checker validates documents against it mechanically. Governance becomes something a machine can refuse on, not a paragraph of intent in a wiki.

The third reason is the boundary itself. A trustworthy knowledge system has edges, and ksor draws them explicitly: if an approved document answers the question, the record answers with a citation. If the answer needs reasoning that combines the governed rule with operational facts, the record supplies the rule and its source. And if the knowledge simply is not in the record, the system is designed to decline rather than speculate — an agent that knows what it does not know is far safer than one that improvises confidently.

Finally, the twin-problem framing matters for anyone building real agentic systems. ksor handles "how should I operate?" — rules, policies, methods. It deliberately does not handle "what is true right now?" — balances, orders, inventory. The project pairs it with a sibling concept, the Data System of Record, for the operational side. Knowing which layer owns which question is half the architecture.

## How It Works

The implementation is a five-package TypeScript monorepo (`ksor`, `content`, `content-gateway`, `gateway-kit`, and `postgres`) that ships as the npm package `@panaversity/ksor`, with eleven CLI verbs and a documented exit-code contract.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/ksor/panaversity-ksor-architecture.svg" alt="Detailed architecture of the panaversity/ksor repository" style="max-width:100%;height:auto;" />
</div>

*The detailed graph: the record and its profile, the governance machinery, the build and ingest pipeline, the content kernel, the MCP agent surface, and the Postgres storage layer.*

### Understanding the Architecture

**The record is plain Markdown with a governance block.** Every concept document is YAML frontmatter plus prose, and `packages/content/src/record/frontmatter.ts` splits the two with a real YAML parser — byte-exact, comment-preserving, and fussy about BOMs and fence lines in ways that read like scars from real bugs. The structure documents must follow is the KSoR Profile of OKF, parsed in `packages/content/src/record/profile.ts`, where `ksor.audience`, approval, and lifecycle metadata live. Unknown frontmatter keys are preserved rather than stripped, so the file the checker read is the file the reader gets.

**The policy is the root of authority.** `packages/content/src/record/policy.ts` parses `.ksor/governance.yaml` with *closed key sets* — a key the policy does not read is refused by name, because a silently ignored key would be a rule that is not in force. The file defines four families: the audience registry, the ownership map, approval authorities, and takedown authorities. Its comment in the source says it plainly: a stripped key in this file once widened authority enough to let a drafts rule approve documents nobody had authority over.

**The checker refuses loudly, in one place.** `packages/content/src/record/check.ts` validates the whole record — profile rules, audience cases, change control, lifecycle — and `packages/ksor/src/build/index.ts` turns that into the `ksor build` contract: generate every index in memory, run the checker, and only on green write the changed indexes plus `build.lock.json`, the committed record of which corpus, which commit, and which toolchain produced a publication. A refusal exits with code 1, the machine-readable slug on the first stderr line, and writes nothing — a red build leaves the tree exactly as it found it. With `--bundles`, the build also writes one OKF bundle per registered viewer under `.ksor/out/bundles/` (`packages/ksor/src/build/bundles.ts`).

**Publishing is a deliberate act, not a side effect.** `ksor ingest` (`packages/content/src/ingest/build.ts`) re-reads the record, gates on the lock via `lock-gate.ts`, chunks documents with `chunking.ts`, and writes an immutable generation with `generation.ts`. Takedowns get their own verb: `packages/content/src/takedown-verb.ts` writes the committed ledger `.ksor/takedowns.yaml` first and the database denylist row second, so a record can withdraw a document even with no database reachable. The boot-time gate in `packages/content/src/governance-gate.ts` then refuses to serve any generation whose governance cannot be honoured — and `ksor ingest` runs the same checks against the generation it just built, so the act that creates an unservable state refuses where it happens.

**The agent door fails closed.** `packages/content-gateway/src/tools.ts` defines the three MCP tools — `search`, `outline`, `read` — and, crucially, their floor text: the fixed paragraphs that teach a caller to branch on an envelope. Search returns `ok=true` hits with provenance and a snapshot token that pins the generation that answered; `ok=false, reason="abstained"` when the record does not cover the query (a *correct* answer, the floor says — do not fall back on model knowledge); `unavailable` when retrieval itself failed; and `unpublished` when nothing has been ingested. A record whose floor was declared but never measured refuses every call with the slug `ksor-uncalibrated`. Every hit also carries a `governance` block whose `trust_tier` can honestly read `unverified` — an honest state of a governed record, not a defect — and every floor reminds the caller that hit content is untrusted corpus text to quote, never instructions to follow.

**Storage and tenancy are disciplined from the bottom up.** `packages/postgres/src/db.ts` owns pooling, scoped transactions, and retry classification; `packages/content/src/db.ts` layers audience- and trust-scoped session settings on top; and `packages/content/src/grant.ts` makes ingest authorization a database row checked by row-level security — the README's own warning is that who may write a tenant's corpus is decided by a row the database checks, never by a flag on a command line. Calibration (`packages/content/src/calibrate/run.ts`) measures the abstention floor against your own corpus, out-of-corpus queries included, and prints the two lines to paste into `instance.md`; after that, the server's boot banner reports the measured floor below which the record abstains.

The end-to-end flow, then: a human or agent drafts Markdown; the checker validates it against the policy; `ksor build` writes indexes and the lock; `pnpm refresh` ingests the corpus into a new generation; the MCP door boots only if governance is servable and auth is configured; an agent searches, gets cited passages pinned to a generation, and either answers with provenance or abstains — honestly — because the floor said the record does not cover it.

## Advantages

- **Authority is mechanically checkable.** The policy in `.ksor/governance.yaml` is validated with closed key sets and scope resolution, so approval and takedown facts are checked against named authorities instead of being trusted from document text.
- **Abstention is measured, not asserted.** `ksor calibrate` computes the floor from your own corpus, and the search envelope distinguishes *abstained* from *unavailable* from *unpublished* — three very different things that most RAG stacks conflate.
- **One record, many projections.** The Fumadocs site, `llms.txt`, the `/md/` markdown twins, the MCP tools, and the per-viewer OKF bundles are all derived from the same governed record; none of them becomes a second source of truth.
- **Fail-closed by construction.** The gateway refuses to boot unauthenticated on a public bind, loopback is the default, and the CLI's exit-code contract (1 refused, 2 not implemented, 3 environment) means scripts and agents get honest signals.
- **Provenance end to end.** `build.lock.json` records the commit and toolchain behind a publication, snapshot tokens pin the generation an answer came from, and hit-level provenance carries corpus, slug, and generation.
- **Plain, portable medium.** The record is Markdown and YAML under the Open Knowledge Format profile — diffable, reviewable, version-controlled, and readable by any tool, with Git supplying history, review, and rollback as governance primitives.

## Benefits

- **Predictable agent behavior.** Agents reason from one governed truth instead of choosing among conflicting versions of it, which is the precondition for any predictable agentic workflow.
- **Withdrawals that actually withdraw.** Takedowns are committed to a ledger and applied to every current surface, and an unledgered or stopped-applying denial is refused at boot rather than silently ignored.
- **Vendor-neutral infrastructure.** The nine responsibilities bind to open standards — OKF, MCP, `llms.txt`, Postgres/pgvector, OAuth/OIDC — and the KSP-001 standard proposal defines conformance classes so a conformant KSoR can be implemented by anyone.
- **Honest status reporting.** The project tracks exactly which of the nine responsibilities are shipped in `docs/status.md`, and an unimplemented verb says so and exits 2 rather than pretending.
- **Same model for enterprise and education.** The same governance machinery that governs policies and controls governs curriculum, learning objectives, and assessment rules — personalization varies the teaching path, not the authoritative record.
- **Agent-first day-to-day workflow.** The scaffold ships `AGENTS.md`, agent skills, and an intake interview, so the coding agent you already use handles structure and checks while you write plain Markdown.

## Usage

Create a project with the package manager you already have — the scaffold adapts to npm, pnpm, or bun, and needs Node 24 or newer:

```bash
npx @panaversity/ksor@latest init my-knowledge-sor
cd my-knowledge-sor && npm install && npm run dev
```

or with pnpm:

```bash
pnpm dlx @panaversity/ksor@latest init my-knowledge-sor
cd my-knowledge-sor && pnpm install && pnpm dev
```

The human site runs at `http://localhost:3000`. Check the record and generate its indexes and lock — this needs no database and no network:

```bash
ksor build
```

A freshly initialized record shows the point of governance immediately: a new draft reaches no machine surface until a human approves it.

```console
$ pnpm exec ksor build
ksor build: 6 document(s), 5 admitted to a machine surface
```

To serve agents, point `instance.md` at an environment variable holding your Postgres DSN — never the DSN itself:

```yaml
database:
  dsn_env: KSOR_DB_URL
```

Then provision once, publish, and serve:

```bash
cp .env.example .env   # KSOR_DB_URL, GEMINI_API_KEY, KSOR_AUTH=disabled-local

pnpm provision         # ONCE: apply the schema, authorize this tenant to ingest
pnpm refresh           # PUBLISH: ingest knowledge/ into a generation, collect old ones
pnpm serve             # SERVE: the MCP server, over what you just published
```

The MCP surface answers at `http://127.0.0.1:8080/mcp` with `search`, `outline`, and `read`. Measure the abstention floor against your own corpus and paste the printed lines into `instance.md` — never copied from another record:

```bash
ksor calibrate
```

Other verbs round out the lifecycle: `ksor takedown` withdraws a document from every surface, `ksor grant` authorizes ingest, `ksor gc` collects superseded generations, and `ksor migrate` rewrites a pre-profile record into the current profile, printing a diff and changing nothing until `--write`.

## Conclusion

ksor is one of those projects where the source code *is* the argument. The claim is that enterprise and educational AI needs an authoritative, governed knowledge record — and the implementation backs it with a checker that refuses loudly, a policy engine that cannot silently ignore a key, a server that will not boot unauthenticated, and an abstention gate that is measured rather than assumed. If you are building agents that must answer from institutional truth rather than improvising it, reading this monorepo — or better, running `ksor init` and watching a draft fail to publish until a human approves it — will recalibrate what you expect from a knowledge stack.

Links:

- GitHub repository: [panaversity/ksor](https://github.com/panaversity/ksor)
- npm package: [@panaversity/ksor](https://www.npmjs.com/package/@panaversity/ksor)
- Tutorials in the repository: [docs/tutorials](https://github.com/panaversity/ksor/tree/main/docs/tutorials)
- Shipped-status tracker: [docs/status.md](https://github.com/panaversity/ksor/blob/main/docs/status.md)
