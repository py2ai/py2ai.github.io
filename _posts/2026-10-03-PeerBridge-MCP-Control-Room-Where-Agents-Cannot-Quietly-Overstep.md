---
layout: post
title: "PeerBridge MCP: A Control Room Where Agents Cannot Quietly Overstep"
permalink: /PeerBridge-MCP-Control-Room-Where-Agents-Cannot-Quietly-Overstep/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/peerbridge-mcp/hoylon-peerbridge-mcp-architecture.svg
tags: [Python, MCP, Multi-Agent, AI, Security]
---

Running several coding agents at once is easy. Knowing what they are allowed to do, what they actually did, and whether the record of it is true - that is the hard part, and it is where most multi-agent setups quietly fall apart. [PeerBridge MCP](https://github.com/Hoylon/peerbridge-mcp) by Hoylon is a local-first, auditable multi-agent control room for coding, review, evidence, and private remote work. Alpha 6 (v0.1.0a6), Apache-2.0, Python 3.11+, and remarkably its runtime dependency list is empty - the core ships on the standard library alone, with cryptography as an optional extra. Codex, Claude Code, Grok, Kimi, provider APIs, OpenAI-compatible endpoints, and local models join as one governed engineering team across desktop and phone, without flattening their native capabilities.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/peerbridge-mcp/hoylon-peerbridge-mcp-overview-architecture.svg" alt="Architecture overview of the Hoylon/peerbridge-mcp repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the repository: the MCP layer of stdio entries and the tool server, the coordination core around the bridge database and the round supervisor, governance with the approval broker, execution policy, operation queue and credential vault, a provider runner, SHA-linked evidence, and the operator surfaces reading the same store.*

Reading the overview from left to right: `src/peerbridge_mcp/cli.py` starts one stdio process into `src/peerbridge_mcp/server.py`, which dispatches tool calls into the bridge core in `src/peerbridge_mcp/bridge.py`. Write requests pass through the writer leases of `src/peerbridge_mcp/operation_queue.py` to the approval broker, gated by the policy in `src/peerbridge_mcp/execution_governance.py`, while `src/peerbridge_mcp/credentials.py` holds raw keys away from every caller. The round supervisor advances discussions and places provider calls through the OpenAI-compatible runner, and the monitor plus remote pages read the same store with scoped reads while the bridge links evidence into the proof bundle.

## Ten layers, one SQLite file

The architecture document is refreshingly blunt: ten small layers. `protocol.py` formats MCP and JSON-RPC responses; `server.py` exposes the stdio tool catalog and dispatches calls; `bridge.py` - the 400-kilobyte heart - implements validation, rooms, reusable agent identities, coordination rules, SQLite transactions, and hashes; `monitor.py` reads the shared store and provides the operator UI; `credentials.py` keeps raw provider secrets in Windows Credential Manager; `ccswitch.py` drives CC Switch's public CLI without reading its database; `remote.py` serves a loopback-only page meant for an authenticated Tailscale Serve proxy; `openai_compatible_runner.py` adapts relay and local model APIs with a least-privilege tool loop; `attachments.py` stages content-addressed evidence copies without persisting the original path; and `tailcat_runtime.py` manages the pinned, allow-listed companion runtime.

The deployment model follows from one decision: there is no central network daemon. Every MCP client starts its own independent stdio process, and they all point at the same project-local SQLite database. The database is the coordination plane; the processes are disposable.

## Credentials never ride the MCP path

The security posture is the standout. When the operator onboards a provider, the raw endpoint and key go into Windows Credential Manager; the MCP and SQLite layers see only redacted IDs and a SHA-256. A saved route is selected as agent, then provider connection, then model, then a reasoning setting that exact model supports - and broadcast messages deliberately bypass model routing, because multiple recipients cannot share one verifiable runtime identity. A message may bind a route request, which stays `requested` until the recipient's observed identity matches every requested field; only then is a separate route receipt written. The remote phone path is equally narrow: Tailscale Serve is the only supported proxy, an allowlist gate sits in front, reads are scoped in SQL, and a human write converts into a short-lived stdio `send_message` call. Provider secrets and private endpoints never enter that path.

## The claim-to-proof loop

The coordination model is where the design gets serious. `claim_task(paths, policy)` runs under `BEGIN IMMEDIATE` with a conflict check - a task declares read and write path prefixes, read/read overlap is fine, any overlap with a live write lease conflicts - and returns a random capability token once, storing only its SHA-256. The writer then records proof (changed paths, before and after hashes, tests, evidence paths), peers request and submit reviews whose verdicts count distinct approved identities, and `complete_task` rehashes the files and fails if they drifted, then closes the lease atomically. Every state transition appends an audit event whose chain hash includes its payload hash, the previous chain hash, scope, actor, task, type, ID, and timestamp - the verification engine recalculates the whole chain.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/peerbridge-mcp/hoylon-peerbridge-mcp-architecture.svg" alt="Detailed architecture of the Hoylon/peerbridge-mcp repository" style="max-width:100%;height:auto;" />
</div>

*Detailed view of the same repository: the MCP layer with protocol formatting and session contracts, the coordination core's agent identities, authorized sessions, discussion tracking and continuity snapshots, governance adding secret scanning to the lease-and-broker chain, five provider runners replying into the bridge, the evidence stack of proof bundles, the verification engine, provider and collaboration receipts and the trust timeline, the four operator surfaces, and the project documents and tests that cover it.*

The demo proves the pattern without any provider at all: `python examples/demo_workflow.py --workspace demo-workspace --scope demo` produces a public receipt showing an overlapping second writer was rejected, two independent reviewers satisfied quorum, completion rehashed the synthetic artifact, and the audit chain verified with zero writes.

## Rooms that stop on purpose

Room automation is bounded by design. Every room owns one SHA-bound policy: automation off, one parallel response round, or bounded parallel discussion, with state tracking rounds, message counts, limits, a stagnation digest, status, and stop reason. `post_room_message` is the only path that wakes other seats; `request_review` merely appends a source-bound request - it does not invoke a provider - so review requests render on a Review surface instead of polluting chat. Provider replies are always child messages, never new root posts, which structurally prevents reply cascades. The single supervisor claims only top-level coordinator prompts, runs each round's provider calls through a bounded parallel executor, and then advances state once in one immediate transaction - stopping for consensus, blockers, dispatch failure, repeated content, or a round or message limit. If a seat's credential is missing at dispatch, the result is a terminal `credential_unavailable` and the room enters `waiting_human` instead of pretending the provider is online.

## Shared memory with a promotion ceremony

The memory ledger is provider-neutral by schema rather than by pretending model contexts are compatible. A fact has three visibility levels: `private` to one agent in one room, `room` for active members, or `project` - and the jump to project requires explicit human promotion, a source message or artifact, and a stored row binding the live source SHA-256. Revocation marks a record inactive and appends a revocation hash; it never rewrites or deletes the original. Provider-neutral runners get read-only memory tools by default, and receipts record tool names and hashes, never memory bodies.

## An interface for trust, not just chat

On top of the core sit two full frontends - the dense Pixel-style control room and the Modern Workbench (a sizeable `workbench/app.js`), plus `desktop_cockpit.py` and a localization layer carrying the whole UI in English, Simplified and Traditional Chinese. The `trust_timeline.py` surface renders agent activity, mutual scores, audits, token usage, and stale-proof warnings, so the operator reviews decisions rather than vibes. There is a threat model document, a telemetry policy, an open-core boundary document, and a continuity manifest that snapshots workspace state. Receipt CLIs - `peerbridge-provider-receipt` with its `verify-` twin, plus MCP-client and ACP-client variants - make the capture/verify pattern a first-class command-line citizen.

The honest label is on the tin: alpha, with the coordination and audit core tested and the public API still moving. But as a demonstration that multi-agent orchestration can be built around leases, quorum, rehashing, and hash-chained audit events instead of around optimistic chat, PeerBridge is one of the more complete open-source statements out there.
