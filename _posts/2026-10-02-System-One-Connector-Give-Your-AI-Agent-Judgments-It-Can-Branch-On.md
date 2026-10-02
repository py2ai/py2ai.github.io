---
layout: post
title: "system-one-connector: Give Your AI Agent Judgments It Can Branch On - Inside itsmostafa/system-one-connector"
description: "system-one-connector is an MIT-licensed Go binary that plugs TypeSafe's Jev, CLM, Laya and Liquid AI's d1 into Claude Code, Codex, Claude Desktop, Hermes and pi as a single MCP tool. We tour the source behind its typed noul, choice and score questions, probability answers, batched items fan-out, multi-route model selection, profiles, and checksum-verified self-update."
date: 2026-10-02
header-img: "img/post-bg.jpg"
permalink: /System-One-Connector-Give-Your-AI-Agent-Judgments-It-Can-Branch-On/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/system-one-connector/itsmostafa-system-one-connector-architecture.svg
tags:
  - Go
  - MCP
  - AI Agents
  - LLM
categories: [AI, Open Source]
keywords: "system-one-connector, evaluate, MCP server, Jev, TypeSafe, System One models, noul, choice, score, probabilities, Claude Code, Codex, Claude Desktop, Hermes, pi, OpenRouter, CLM, Liquid AI d1, Go, agent tooling"
author: "PyShine"
---

When an AI agent needs a quick judgment call, the usual move is to ask a reasoning model, read a paragraph back, and guess what it meant. "This seems fairly urgent" gives the agent nothing to put in an `if` statement, and it cannot tell a confident answer from a coin flip. [itsmostafa/system-one-connector](https://github.com/itsmostafa/system-one-connector) attacks exactly that gap. It ships a single Go binary, `evaluate`, that connects Claude Code, Claude Desktop, Codex, Hermes and pi to TypeSafe's Jev model, a System One model built for judgments rather than text generation. The agent names a question and the possible answers, and gets back a typed result with real probabilities: 0.95 that a ticket is urgent, "billing" at 86 percent with "technical" at 14, a position of 3.87 on five ordered severity levels.

The connector is not tied to one provider. The same tool also runs other System One models, whether you host them yourself, such as the open CLM and Laya models, or they are hosted for you, such as Liquid AI's d1. And because it speaks MCP over stdio, one static binary with only two direct Go dependencies serves all of them, with a setup command that registers itself with every supported client it finds.

The source is worth a tour because it is disciplined about the two things that make this pattern work: keeping answers structured and comparable, and keeping the agent's own conclusions out of the judgment. Every layer, from the request path to the reply shaper to the guidance text sent at connect time, is built around that contract. Let us walk through it.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/system-one-connector/itsmostafa-system-one-connector-overview-architecture.svg" alt="Architecture overview of the itsmostafa/system-one-connector repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the system-one-connector codebase: agent clients on the left, the evaluate tool's processing stages in the middle, the three model endpoint routes on the right, and the setup, profile and self-update operations underneath.*

The walkthrough starts at the clients. `evaluate setup mcp` registers the binary with Claude Code, Codex and Hermes through their own CLIs and edits Claude Desktop's config file directly, while `evaluate setup pi` renders a native extension for pi, which has no MCP client of its own. Every client starts `evaluate mcp` over stdio, and the server exposes exactly one tool, also named `evaluate`.

## Why You Need This

The first reason is that agents need data they can branch on, not prose to interpret. The `evaluate` tool accepts three question types that cover most judgment calls. A `noul` question returns the probability that a yes-or-no condition holds. A `choice` question returns one option from a criteria map, with a probability for each option. A `score` question returns a probability-weighted position on ordered levels, from zero-indexed floor to ceiling, with a legend mapping each index back to its label. Every answer comes back under the question ID you chose, in JSON your code can compare against a threshold.

The second reason is confidence you can act on. Choice and score answers carry a confidence value derived from how concentrated the probabilities are, not from the chosen option's probability. A noul near 0.5 means uncertain, not medium intensity, and the connector will say so rather than dress the answer up. Better still, you can set a `min_confidence` threshold on any noul or choice question; when an answer falls below it, the tool marks the answer uncertain, and a choice becomes the reserved `__uncertain__` option, so the agent can escalate the case instead of acting on a shaky read.

The third reason is that the whole thing is one command to wire up. `evaluate setup mcp` detects the client CLIs on your PATH, restores entries when an install goes wrong, and bakes the relevant environment variables into each client's config, because MCP clients start servers without your shell environment. There is no Node runtime and no Python environment to babysit.

## How It Works

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/system-one-connector/itsmostafa-system-one-connector-architecture.svg" alt="Detailed architecture of the itsmostafa/system-one-connector repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of the evaluate MCP server: the tool's processing path from raw-argument decoding through validation, request building and the items fan-out, the HTTP transport with its backoff and body caps, the three endpoint routes with profile-based selection, and the operations layer for setup and self-update.*

**The request path preserves your bytes on purpose.** Incoming tool arguments are re-decoded with JSON numbers kept as their original digit strings rather than floating-point values, so two distinct identifiers past the safe integer threshold cannot silently collapse into one. Criteria travel upstream as the caller's raw bytes, which means a choice's options reach the model in the order you wrote them instead of an alphabetized shuffle. On the way back, the reply shaper reorders each answer's probability map into criteria order, or level order for scores, so two batches of the same question present their options identically, and it re-encodes without HTML escaping so angle brackets in API strings survive intact.

**Validation catches what the API cannot.** Before anything is sent, the validator checks that criteria have the shape each question type requires, and rejects two inputs the upstream API is known to mishandle: an unknown question type, which would otherwise come back as a bare "Invalid request.", and noul criteria keys other than true and false, which the API silently drops. Errors name the exact field path you sent, in bracket-quoted form so an ID containing a dot cannot be misread as nesting.

**The items mode turns one call into a parallel sweep.** Pass up to 500 records as items and the tool asks the same questions of each one, at most eight requests in flight, each record judged independently with its own state plus shared context. One failed item lands in the errors map without cancelling its siblings; only a total failure fails the call. Results, errors and one combined metadata block come back in a single response, bounded by a 16 MiB ceiling so half a thousand replies cannot pile up gigabytes.

**The transport is defensive by design.** The HTTP client imposes a sixty-second timeout, retries rate-limit and capacity responses up to three times with exponential backoff, rejects any response larger than the cap rather than truncating it, and refuses a non-JSON success body, so a proxy's HTML error page can never be mistaken for an answer.

**Model selection is explicit and shadow-proof.** The route resolver checks, in order: a profile named by the `TYPESAFE_PROFILE` variable or marked active in the profiles file; the TypeSafe API when `TYPESAFE_API_KEY` is set, honoring `TYPESAFE_BASE_URL` for custom hosts and `TYPESAFE_MODEL` for models like CLM or Liquid AI's d1; and finally OpenRouter's Decisions endpoint when only an OpenRouter key is present. TypeSafe wins when both keys are set, so a stray OpenRouter key left in the shell cannot silently reroute and re-bill an existing setup. Profiles live in an owner-only config file written atomically, and a selected profile replaces the environment keys wholly, so switching models is one command.

**Self-maintenance is checksum-gated.** `evaluate update` fetches the latest release, verifies the platform archive against the published SHA-256 checksums, stages the new binary beside the old one, and swaps them with an atomic rename; the installer script follows the same discipline. At startup a tightly-budgeted release check appends an update notice to the client's guidance, so your agent can remind you once to upgrade.

The end-to-end flow reads cleanly. A client starts `evaluate mcp` over stdio; the resolver picks TypeSafe, a self-hosted CLM server, Liquid AI's d1, or OpenRouter; the agent sends state and typed questions; the server validates, builds, and fans them out; the transport retries and caps; the shaper orders the probabilities and applies abstention thresholds; and the agent gets JSON it can branch on, guided by instructions served at connect time that teach it to send evidence rather than conclusions.

## Advantages

- **Typed answers, not prose**: noul, choice and score questions return probabilities your code compares and branches on directly.
- **Honest confidence semantics**: confidence measures how concentrated the probabilities are, `min_confidence` turns weak answers into explicit abstentions, and a near-0.5 noul is reported as uncertainty rather than medium intensity.
- **Batch judgment at scale**: up to 500 items per call with independent per-item requests, bounded concurrency, per-item error isolation and one aggregated metadata block.
- **Provider freedom**: TypeSafe, OpenRouter, or any host speaking the same System One endpoint, including self-hosted CLM and Laya and Liquid AI's d1, switchable through named profiles.
- **Gentle setup**: one command registers the server with Claude Code, Claude Desktop, Codex, Hermes and pi, with backup-and-restore around each client's config and variables baked in for you.
- **Self-sufficient binary**: one static Go executable with two direct dependencies, owner-only profile storage, checksum-verified updates, and strict response caps against oversized or non-JSON replies.

## Benefits

- **Fewer token-hungry detours**: judgments that would otherwise cost a full reasoning-model round trip return in a fraction of a second, and the tool's guidance steers agents toward narrow questions the model answers reliably.
- **Escalation built in**: uncertain answers are flagged and resolvable to `__uncertain__`, so low-confidence cases can flow to a human or a reasoning model instead of being acted on silently.
- **Deterministic presentations**: probability maps keep the caller's option order across every item in a batch, which matters when the same question is asked hundreds of times.
- **No silent re-billing**: the route priority keeps a leftover OpenRouter key from moving an existing setup onto a different account.
- **Safe to automate**: the tool is read-only, rate-limit responses back off automatically, and every response size is capped before it can exhaust memory.
- **Kept current without effort**: the binary checks for newer releases within a tight time budget and tells your agent, once, to run the update command.

## Usage

Install on macOS or Linux, then register the server with your agents:

```sh
curl -fsSL https://raw.githubusercontent.com/itsmostafa/system-one-connector/main/install.sh | sh
TYPESAFE_API_KEY=your-key evaluate setup mcp
```

With an OpenRouter account instead, set `OPENROUTER_API_KEY` before running setup. Point the connector at a self-hosted model by giving the custom host route:

```sh
TYPESAFE_API_KEY=local TYPESAFE_BASE_URL=http://127.0.0.1:8700 TYPESAFE_MODEL=clm-latest evaluate setup mcp
```

Liquid AI's hosted d1 works the same way through its own base URL and model name. Save each combination as a profile and switch without re-running setup:

```sh
evaluate profile add typesafe-ai --model jev-latest --api-key your-key
evaluate profile add liquid --model d1:free --base-url https://api.liquid.ai/decisions --api-key your-liquid-key
evaluate profile use liquid
evaluate profile list
```

For pi, which has no MCP client, install the native extension instead:

```sh
evaluate setup pi
```

Then ask your agent, in its own words: "Use evaluate to decide whether this ticket is urgent and which team should own it." The agent sends a state object with the observed evidence, one question per judgment, and gets probabilities back under each question ID. Cap the per-call item count with `TYPESAFE_MAX_ITEMS` if you want a tighter ceiling, update in place with `evaluate update`, and build from source with `task build` if you prefer.

## Conclusion

system-one-connector is a small repository with a sharp thesis: the judgment calls inside an agent workflow deserve their own model primitive, and the plumbing between that primitive and your agent should be boring, safe, and set up in one command. The Go source follows through. Numbers survive the trip to the model as written, criteria keep their order, malformed questions fail locally with useful errors, uncertain answers abstain loudly, and the binary can replace itself without breaking anything. If you have ever wanted a quick "is this urgent?" answered as 0.95 rather than a paragraph, this connector is the straightforward way to get it.

Links:

- [itsmostafa/system-one-connector on GitHub](https://github.com/itsmostafa/system-one-connector)
- [Tool reference](https://github.com/itsmostafa/system-one-connector/blob/main/docs/tool-reference.md)
- [Configuration guide](https://github.com/itsmostafa/system-one-connector/blob/main/docs/configuration.md)
- [TypeSafe website](https://typesafe.ai)
- [TypeSafe API docs](https://docs.typesafe.ai/api)
- [CLM-v0.1-8B on Hugging Face](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
- [Liquid AI](https://liquid.ai)
