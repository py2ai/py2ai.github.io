---
layout: post
title: "Cloudflare's Security Audit Skill: Turn Your Coding Agent Into a Security Auditor"
description: "Cloudflare open-sourced the skill that seeded its own vulnerability discovery harness. Six phases, adversarial validation, and findings your team can actually trust."
date: 2026-09-18
header-img: "img/post-bg.jpg"
permalink: /Cloudflare-Security-Audit-Skill-Coding-Agent-Security-Auditor/
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: https://pyshine.com/assets/img/diagrams/security-audit-skill/security-audit-pipeline.svg
tags:
  - Open Source
  - Security
  - AI Agents
  - Cloudflare
author: "PyShine"
---

Ask your coding agent to "find security vulnerabilities" and you will usually get back a confident list of guesses: a missing header here, a hardcoded secret there, half of it wrong, none of it reproducible. The problem is not that agents lack security knowledge. It is that a one-shot prompt has no structure, no coverage accounting, and no mechanism to catch its own mistakes. Cloudflare decided to fix that by open-sourcing the very skill they use internally. [security-audit-skill](https://github.com/cloudflare/security-audit-skill) is a coding-agent skill that turns any capable agent into a disciplined security auditor, and it has already collected more than 10,000 GitHub stars since landing in June 2026.

![Security Audit Skill Pipeline](https://pyshine.com/assets/img/diagrams/security-audit-skill/security-audit-pipeline.svg)

### Understanding the Six-Phase Pipeline

The pipeline diagram above shows the full journey from a single prompt to independently verified findings. Let's walk through each phase, because the structure is where all the value lives.

**The parent agent.** Everything starts with a parent agent that coordinates the run. It is the only writer of the shared run files: `run-metadata.json`, `architecture.md`, `coverage-ledger.json`, `findings.json`, and the final reports. This single-writer rule eliminates an entire class of coordination bugs where two sub-agents race to update the same state. The parent plans coverage, delegates work, and owns the ledger.

**Phase 1, reconnaissance.** Before anyone hunts for bugs, the parent maps the target: architecture, trust boundaries, input surfaces, and any prior audit evidence. The output lands in `architecture.md` and a machine-readable `coverage-ledger.json` that becomes the backbone of the whole audit. Every future check is tied to a unit in this ledger, which is what makes "what did we actually cover?" an answerable question instead of a shrug.

**Phase 2, coverage-led hunting.** The parent assigns isolated hunter agents to ledger units. Each hunter works in its own directory, records what it checked, and coverage critics sweep in afterward to find the gaps. If a hunter skipped an input surface, a critic notices and waves it back in.

**Phase 3, candidate validation.** Here is the twist that separates this skill from prompt-and-pray auditing: every unique candidate goes to a fresh verifier agent whose only job is to disprove it. The agent that found the bug never grades it.

**Phases 4 and 5, structured output and independent verification.** Verdicts are written as `confirmed`, `needs_validation`, or `rejected` records in `findings.json`, validated against a JSON schema by a zero-dependency Node.js validator. Then another set of fresh agents verifies the final source claims. Material replacements get yet another independent verifier.

**Phase 6, target-neutral reporting.** The skill derives `REPORT.md`, `FINDINGS-DETAIL.md`, and `NEEDS-VALIDATION.md` from the verified records. The report tells you the priority, the smallest effective fix, and exactly which facts still need human eyes.

This skill is the single-repo seed of something much bigger: it is the starting point that grew into [Cloudflare's fleet-wide vulnerability discovery harness](https://blog.cloudflare.com/build-your-own-vulnerability-harness). What you install today is the same methodology, scoped to one repo and one agent.

### Two Modes, Not One

Load the skill and it does nothing by default, and that is deliberate. The skill ships with two operating modes:

- **Guidance mode** activates for security questions, focused reviews, and triage work. The agent uses only the relevant parts of the methodology, launches no audit pipeline, and writes no files.
- **Full audit mode** activates only when you explicitly ask for an audit, a pen-test, or report artifacts. That is when all six phases run.

If your request is ambiguous, the agent asks one focused question before creating anything. It is a small design decision that prevents the classic failure of skills that hijack every conversation.

### The Sandbox Rules Are Not Optional

![Write Isolation and Sandbox Model](https://pyshine.com/assets/img/diagrams/security-audit-skill/security-audit-isolation.svg)

### Understanding Write Isolation

The isolation diagram above is the part most similar tools skip entirely, and it is the part that makes the findings trustworthy.

**Why isolation matters.** An audit requires running untrusted code: builds, tests, fuzzers, fixtures. If that code runs with your agent's permissions, the audit itself becomes the attack surface. The skill requires an OS-enforced sandbox with four non-negotiable controls: no external network (loopback only for local client/server checks), an empty environment rebuilt from an explicit allowlist, a read-only target and toolchain, and hard limits on CPU, memory, disk, and wall-clock time.

**Scratch versus artifacts.** Every hunter and verifier gets a unique `agents/<agent-id>/` directory with separate `scratch/` and `artifacts/` folders. Target-controlled processes may write only to `scratch/`. The `artifacts/` directory is parent-owned and never exposed to the sandbox at all.

**The promotion procedure.** When the sandbox exits, trusted parent-side code promotes allowlisted files from scratch to artifacts through an eleven-step procedure: validating every path component, walking directories with no-follow operations, verifying file identity and size with `fstat`, and creating the destination exclusively. Symlinks, FIFOs, sockets, and oversized files are rejected outright. If any check fails, the evidence is demoted to `needs_validation` with the exact blocker recorded.

If the environment cannot enforce these controls, the skill does not execute target code at all. It keeps the lead as `needs_validation` and hands you a safe validation plan instead. That honesty is a feature.

### Coverage Across Every Attack Surface

![Attack-Class Companion Files](https://pyshine.com/assets/img/diagrams/security-audit-skill/security-audit-coverage.svg)

### Understanding the Attack-Class Library

The coverage diagram above shows the hunting playbook. Hunters do not improvise; they consult eleven companion files, each a curated set of attack prompts for one target family:

| Companion File | Target Family |
|---|---|
| `ATTACK-CLASSES.md` | Core, wildcard, and obvious-things prompts |
| `AI-AND-LLM.md` | Prompt injection and agent/tool abuse |
| `WEB-PROTOCOL-AND-AUTH.md` | Request framing, cache deception, auth flaws |
| `CLIENT-SIDE.md` | DOM injection, messaging trust, prototype pollution |
| `SUPPLY-CHAIN-AND-RELEASE.md` | Dependencies, CI, signing, update channels |
| `CLOUD-AND-DEPLOYMENT.md` | IAM, IaC, containers, serverless |
| `PROTOCOLS-RPC-AND-MESSAGING.md` | Serialization, queues, webhooks, streaming |
| `RESOURCE-EXHAUSTION-AND-AVAILABILITY.md` | Quotas, workers, operator spend |
| `DATA-ISOLATION-AND-LIFECYCLE.md` | Tenant isolation, cache, export, backup, deletion |
| `DESKTOP-MOBILE-AND-LOCAL-IPC.md` | Deep links, webviews, helpers, local IPC |
| `MEMORY-SAFETY-AND-BINARY.md` | Memory safety, binary, and kernel hunting |

Notice the `AI-AND-LLM.md` file. Auditing a codebase that itself calls LLMs raises questions traditional tools never ask: what happens when model output flows into a shell command, and who can inject instructions into your agent's context. Cloudflare ships prompts for exactly these hunts. The `SUPPLY-CHAIN-AND-RELEASE.md` class pairs naturally with the dependency-hardening approach we covered in the [Pi agent harness](/Pi-Agent-Harness-Self-Extensible-Coding-Agent/) post.

### Three Verdicts, No Wishful Thinking

![Adversarial Validation and Verdicts](https://pyshine.com/assets/img/diagrams/security-audit-skill/security-audit-verdicts.svg)

### Understanding the Verdict System

The verdict diagram above captures the design principles that keep the output honest:

- **`confirmed`** requires a complete source trace plus a bounded observed result. Severity is computed as likelihood times impact, not as a deviation from a checklist.
- **`needs_validation`** records one exact unresolved fact and carries no severity. It is a promise to be precise, not a soft confirm.
- **`rejected`** records the disproved candidate so future runs do not chase it again.

Two principles deserve a moment. First, "defense-in-depth gaps are not vulnerabilities": if Layer A prevents the attack, the absence of Layer B is a hardening note, not a finding. This single rule kills most of the noise that plagues automated scanners. Second, runs are additive: Cloudflare's own testing showed a single run finds roughly half of the vulnerabilities that repeated runs find in total, so the skill reuses prior ledgers to target gaps and revalidate changed source on every run.

### Try It On Your Own Codebase

Requirements are modest: an agent with tool use and parallel sub-agents, Node.js for the validators, and an OS-enforced sandbox. Installation is one command through the [Skills CLI](https://www.skills.sh/):

```bash
npx skills add https://github.com/cloudflare/security-audit-skill \
  --skill security-audit
```

Then start your agent in the codebase and ask:

```text
security audit this codebase
```

Output defaults to `~/security-audit-skill/<repo-name>/run-<N>` outside your repository, so audits never pollute your working tree. Everything is MIT licensed, and the [repository](https://github.com/cloudflare/security-audit-skill) includes the full schema and validator test suites if you want to extend it.

The deeper lesson is bigger than one skill. The same pattern, structured phases, write isolation, and adversarial validation, is what separates agent workflows you can trust from demos that merely look impressive. We saw the trust question from a different angle in [PentAGI](/PentAGI-AI-Agent-That-Hacks-So-You-Dont-Have-To/), where a multi-agent system automates offensive testing, and in [OpenResearch](/OpenResearch-Turn-Coding-Agents-Into-Research-Agents/), which imposes evidence gates on research claims. Cloudflare's contribution is the defensive mirror image: agents that find flaws, then spend more effort trying to disprove their own findings than to confirm them. That inversion is the whole trick, and it is now one `npx` command away.

## Related Posts

- [PentAGI: The Open Source AI Agent That Hacks So You Don't Have To](/PentAGI-AI-Agent-That-Hacks-So-You-Dont-Have-To/)
- [Pi: The Agent Harness Where the Coding Agent Extends Itself](/Pi-Agent-Harness-Self-Extensible-Coding-Agent/)
- [OpenResearch: Turn Your Coding Agent Into a Research Agent](/OpenResearch-Turn-Coding-Agents-Into-Research-Agents/)
