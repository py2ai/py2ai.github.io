---
layout: post
title: "Jev Ultrafast: A Browser Agent That Chooses Instead of Generates - Inside browser-use/jev-ultrafast"
description: "A source tour of browser-use/jev-ultrafast, a Python browser agent that picks an operation and a target from an indexed action space in one TypeSafe request, completes a Google Flights search in about seven seconds, and keeps model output away from selectors and executable code. We walk the agent loop, the atomic DOM snapshot, the speculative target heads, and the freshness guards that make the speed possible."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /Jev-Ultrafast-Browser-Agent-Source-Tour/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/jev-ultrafast/browser-use-jev-ultrafast-architecture.svg
tags:
  - Browser Agents
  - Web Automation
  - Python
  - LLM
categories: [AI, Open Source]
keywords: "jev-ultrafast, browser-use, browser agent, TypeSafe, web automation, DOM snapshot, Chrome DevTools Protocol, CDP, Python agent, LLM browser automation, indexed action space, fast web agent, AI agent loop, open source, Google Flights automation"
author: "PyShine"
---

Most browser agents are slow for structural reasons. They screenshot the page, ship pixels to a large multimodal model, wait for a text answer, translate it into a selector, and only then touch the browser — and every round trip is paid again on each action. Jev Ultrafast, published by the browser-use organization, takes the opposite route: its headline measurement is a Zürich → London search on Google Flights completed in **7.1 seconds**, one natural-language goal, actual generated text, and page-loading waits included.

The project describes itself as **a browser agent with a dynamic, indexed action space**. Every page observation produces a numbered table of visible controls, and a policy service — TypeSafe's Jev — picks an operation together with the element it applies to. A small, separate LLM writes text only when the operation is `TYPE_TEXT`; everything else is a choice among finite options, never generated prose. The full operation set is `CLICK`, `TYPE_TEXT`, `SELECT`, `SCROLL_UP`, `SCROLL_DOWN`, `WAIT`, `DONE`, and `BLOCKED`, and only supported operations and compatible targets are ever offered.

The repository is worth a source tour because it is small enough to read end to end, and every speed claim in its README maps to a specific mechanism in the code. The complete loop lives in `jev_ultrafast/agent.py` at 164 lines; the DOM reader is a single JavaScript file, `jev_ultrafast/snapshot.js`, evaluated in one browser call; `jev_ultrafast/browser.py` and `jev_ultrafast/model.py` round out the picture at under 200 lines each. Where the seconds actually go in a browser agent, this codebase answers mechanism by mechanism.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jev-ultrafast/browser-use-jev-ultrafast-overview-architecture.svg" alt="Architecture overview of the browser-use/jev-ultrafast repository" style="max-width:100%;height:auto;" />
</div>

*Overview of browser-use/jev-ultrafast: the agent core consumes a TypeSafe policy and drives the browser through one atomic DOM snapshot, while the inspector, examples, and tests surround the same small package.*

Reading the overview from left to right: everything funnels through `jev_ultrafast/__init__.py`, which exports just `Agent` and `Browser`; the loop in `jev_ultrafast/agent.py` asks `jev_ultrafast/model.py` for a decision and asks `jev_ultrafast/browser.py` for observations and execution; the driver evaluates `jev_ultrafast/snapshot.js` exactly once per observation; the prompt rules in `jev_ultrafast/questions.py` feed both policy and text helper; and the inspector in `jev_ultrafast/demo.py`, the two runnable examples, and the offline suite in `tests/test_agent.py` all sit outside the hot path, consuming the same package a production script would.

## Why You Need This

The first problem is latency. A traditional agent loop turns every action into at least one slow model call, and screenshot-based loops multiply the cost. Jev Ultrafast's default loop contains **no screenshots at all** — the policy consumes structured state, screenshots are an explicit opt-in used by the inspector and the recording scripts, and the demo video comes from a separate continuous screencast. The repo's matched comparison of six alternating runs reports a median task time drop from 9.450 s to 7.092 s, with median browser protocol calls falling from 1,092 to 101. The authors note this is three repeats of one task on one browser profile, not a general benchmark — but the protocol-call count explains the direction.

The second problem is fragility. Hand-written automation scripts hard-code selectors and field values; they shatter when a site redesigns a button. Jev Ultrafast has no site-specific action scripts or prepared field strings anywhere in the policy — the Flights example supplies only a goal, and the DOM reader re-derives the whole action space on every observation, so a moved element simply appears under a new index.

The third problem is trust. Agent outputs that become selectors, coordinates, or JavaScript are an injection surface, and this codebase refuses that design: model output never becomes selectors, coordinates, shell commands, or executable JavaScript, and `jev_ultrafast/browser.py` resolves every executed target from a node observed and stored by the code itself. A JSON validity gate in `jev_ultrafast/model.py` bounds what the text helper can contribute.

The fourth problem is verification. An agent that says "done" is not evidence of done. `examples/flights.py` independently checks the resulting page — URL, one-way setting, origin and destination values, the encoded departure date, and matching flight options — and its `verify()` function, not the model's `DONE` choice, decides whether the run passed.

## How It Works

The whole system is a loop of observe, decide, and act, where the interesting engineering is in how few bytes and how few round trips each phase needs.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/jev-ultrafast/browser-use-jev-ultrafast-architecture.svg" alt="Detailed architecture of the browser-use/jev-ultrafast repository" style="max-width:100%;height:auto;" />
</div>

*Detailed architecture of browser-use/jev-ultrafast: the agent loop and its policy, the CDP browser layer with its atomic snapshot, the loopback inspector with its static UI, and the surrounding scripts, tests, and reports.*

### Understanding the Architecture

**The indexed action space.** Each observation runs `jev_ultrafast/snapshot.js` once inside a single `Runtime.evaluate` call — one browser round trip per snapshot, not hundreds. The script walks common HTML and ARIA controls, computes roles, resolves accessible names, and records current values. A `WeakMap` assigns each actual DOM node a code-owned integer identity, and a `Map` keeps the live references used later for execution; replaced elements get new identities and disconnected ones are pruned. Visible text is collected up to 6,000 characters, action candidates are capped at 250, and scroll and `wait` pseudo-actions are appended — one index per node even when it supports both clicking and typing.

**One request, two decisions.** `choose()` in `jev_ultrafast/model.py` builds one TypeSafe request containing an operation question plus a speculative target head for each available operation, and posts it once. Operation and target are answered together — two decisions in one network round trip — but the target heads are speculative: if the operation resolves to `CLICK`, only the `click_target` answer can execute. `validate_choice()` enforces a strict contract before anything moves: the chosen id must be among the offered ids, all probabilities must be finite numbers in [0, 1], they must sum to 1 within 0.02, and the choice must be the argmax.

**A tiny text helper, tightly sandboxed.** Only `TYPE_TEXT` invokes a second model. `field_text()` in `jev_ultrafast/model.py` sends the goal, the selected field, 6,000 characters of page context, and recent actions to any OpenAI-compatible endpoint — the example uses `inception/mercury-2.5` via OpenRouter with reasoning disabled — and demands a JSON object with exactly one key, `text`, at most 2,000 characters. Nothing else is typed, and nothing is guessed by the executor. If a stale page forces a retry, `jev_ultrafast/agent.py` reuses the generated value only when the entire helper input is unchanged. The repo's performance notes record that generating "Zurich" took 581 ms and "London" 346 ms, with both calls billed at $0.00006272.

**Freshness without counting mutations.** `jev_ultrafast/browser.py` compares semantic state instead of counting DOM mutations, so an animation does not invalidate a decision. `snapshot.js` produces a marker — page origin, URL, scroll, viewport, title, visible text, and the semantic action list — plus a `page_key` and per-node guard tuples covering form values, checkbox state, and nearby form/dialog/row context. `fresh()` compares the marker for coarse checks and the guards for clicks and selects; the SHA-256 `fingerprint()` over URL, text, actions, and scroll gives the inspector a cheap "did the page change" bit. Scoped guards deliberately allow unrelated visible content to change; the design notes call this a practical heuristic, not proof.

**Execution that never guesses.** `act()` rechecks freshness immediately before input, including after text generation. The executor re-resolves the node's current geometry, verifies it is visible, enabled, and not read-only, and hit-tests that the center point belongs to the target before dispatching real CDP mouse events; typing uses a browser select-all command followed by `Input.insertText`. Mutations are never retried by transport recovery, execution is logged before the next observation — so a navigation interrupt cannot erase what already happened — and an interrupted native-select stops rather than risking a double change event.

**A bounded, observable loop.** `jev_ultrafast/agent.py` exposes `tick`, `predict`, and `act` commands with a `run()` generator yielding a snapshot per step. A `StalePage` exception clears the decision, re-observes, and tries again without double-clicking, because the pending decision is consumed before any mutation. Runs are bounded by `MAX_STEPS = 60` actions and a doubled decision budget from `jev_ultrafast/questions.py`, and the loop blocks itself after three consecutive non-wait actions that changed nothing. The connection is one CDP session through the Browser Harness daemon — no per-step subprocess — with focus emulation keeping the owned background tab rendering.

End to end, one cycle looks like this: the driver evaluates the snapshot script once and hands the agent an element table with guards; the agent posts one request whose operation and target heads answer together; `TYPE_TEXT` additionally triggers the sandboxed helper; the executor re-verifies geometry and occlusion and dispatches real input events; the action is logged before the next observation, which waits at most two animation frames or 50 ms — up to 200 ms for autocomplete — before repeating.

## Advantages

- **One request per decision cycle.** Operation and target heads share the same observation — two decisions, one network round trip.
- **No screenshots in the hot path.** The policy consumes structured state; the recorded run's median policy latency was 178 ms.
- **One browser call per snapshot.** The snapshot script reads controls, values, and text atomically, keeping live DOM references for execution.
- **Targets validated before input.** Geometry, visibility, enabled state, and occlusion are re-resolved before every interaction.
- **Model output cannot become code.** Choices are validated against offered ids; text must parse as a single-key JSON object before typing.
- **Bounded and observable.** Hard action and decision budgets, a repetition detector, and per-step snapshots keep runs finite and inspectable.

## Benefits

- **Seconds, not minutes, on real sites.** A verified 7.073-second Google Flights run, a Wikipedia article in 2.798 s, a local hotel task in 1.896 s.
- **Cheap to operate.** The two text-helper calls in the recorded run cost $0.00006272, and the default loop avoids screenshot traffic.
- **Readable in an afternoon.** A handful of short files, with `docs/design.md` explaining every runtime decision and its limits.
- **Safe by construction.** Credentials stay server-side, the inspector is loopback-only with token and origin checks, mutations are never blindly retried.
- **Independent verification built in.** `examples/flights.py` checks the final page instead of trusting the model's `DONE`.
- **A clean library surface.** Import `Agent`, pass a URL and a goal, iterate `run()` — the demo's policy drives your script.

## Usage

The README's quick start clones the repo, syncs dependencies with `uv`, and starts the local inspector:

```bash
git clone https://github.com/browser-use/jev-ultrafast.git
cd jev-ultrafast
uv sync
cp .env.example .env
# Add TYPESAFE_API_KEY and TEXT_MODEL_API_KEY.
uv run jev
```

Open **http://127.0.0.1:8766** and click **Start demo → Run automatically** to watch numbered elements, operation and target probabilities, and executed actions; **Choose next** pauses before execution. Chrome connects through [Browser Harness](https://github.com/browser-use/browser-harness), installed by `uv sync`; run `uv run browser-harness --doctor` if it needs connecting.

To use it as a library:

```python
from jev_ultrafast import Agent

with Agent(
    "https://www.google.com/travel/flights?hl=en",
    "Find one-way flights from Zurich to London on September 20, 2026, "
    "for one adult in economy. Stop when matching flight options are visible.",
) as agent:
    for state in agent.run():
        print(state["elapsed_ms"], state["status"])
```

The same policy runs any task you can phrase as a goal:

```bash
uv run --env-file .env python examples/run.py \
  --url https://en.wikipedia.org/wiki/Main_Page \
  --goal 'Find and open the Wikipedia article about Gödel’s incompleteness theorems.'
```

For development, the repo's own checks are offline:

```bash
uv run ruff check .
uv run pytest
```

## Conclusion

Jev Ultrafast is a focused argument, written in code: the fastest browser agent is not the one with the biggest model, but the one whose loop touches the network least, whose observation is one atomic read, and whose model output is confined to choices and short strings. The package requires Python 3.12 or newer, depends only on `browser-harness` and `httpx`, and is MIT licensed. The authors document their limits honestly — no shadow roots, frames, canvas, uploads, or pop-up tabs in this MVP, and two websites do not establish broad reliability — which is exactly the boundary drawing you want in a source you are about to learn from. Read `jev_ultrafast/agent.py` first, then `snapshot.js`; together they are the whole speed story.

Links:

- GitHub repository: [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast)
- Performance measurements: [docs/performance.md](https://github.com/browser-use/jev-ultrafast/blob/main/docs/performance.md)
- Design notes: [docs/design.md](https://github.com/browser-use/jev-ultrafast/blob/main/docs/design.md)
- TypeSafe documentation: [docs.typesafe.ai](https://docs.typesafe.ai/introduction)
- Browser Harness: [browser-use/browser-harness](https://github.com/browser-use/browser-harness)
