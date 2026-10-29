---
layout: post
title: "Instructor Source Tour: How the Structured-Outputs Library Turns LLM JSON Into Pydantic Models"
description: "A guided source tour of 567-labs/instructor: the lazy-import facade, the patch() wrapper, the (Provider, Mode) handler registry, and the tenacity-driven validation loop that reasks the model until the JSON fits."
date: 2026-10-29
header-img: "img/post-bg.jpg"
permalink: /instructor/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/instructor/567-labs-instructor-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: instructor, structured outputs, LLM, Pydantic, OpenAI, Anthropic, function calling, retry logic, source code, Python
author: "PyShine"
---

If you have ever extracted structured data from an LLM, you have probably met *Instructor*: define a Pydantic model, hand it to `client.chat.completions.create(response_model=...)`, and get back a validated, typed object instead of a string of JSON you have to scrape out of markdown fences yourself. The library sits on top of the OpenAI SDK and, through adapter functions, on top of Anthropic, Gemini, Cohere and a dozen other providers. Its README is disarmingly short, but the source has an architecture that rewards a closer look — especially now that the codebase has gone through a visible reorganization.

This post is a source tour of [567-labs/instructor](https://github.com/567-labs/instructor) at version 1.18.0 (the `main` branch as of this writing). The headline structural fact: the package you import, `instructor`, is now a thin facade. The historical modules under `instructor/core/` have shrunk to three-line shims, and the real implementation lives in a `v2` subtree — `instructor/v2/core/` for the client and dispatch machinery, `instructor/v2/providers/` for per-vendor handlers, and `instructor/v2/dsl/` for the response-model helpers. Everything below links to the exact lines, verified against the tree.

The first diagram is the overview: the facade on the left, factories and the patch-and-dispatch core in the middle, the validation-and-retry loop on the right.

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/instructor/567-labs-instructor-overview-architecture.svg" alt="Overview architecture of the Instructor library: facade, client wrappers, provider factories, mode registry and retry loop" /></div>

<p><em>Overview of Instructor's request flow: the lazy-import facade feeds factory functions that build <code>Instructor</code>/<code>AsyncInstructor</code> wrappers around a patched <code>create</code> method; the patch layer consults the mode registry and delegates every call to the retry chain, where raw completions are parsed and validated into Pydantic models.</em></p>

Reading the overview from left to right:

- [`instructor/__init__.py`, L111](https://github.com/567-labs/instructor/blob/main/instructor/__init__.py#L111) — the `_LAZY_IMPORTS` table. Importing `instructor` costs almost nothing; each public name (`Instructor`, `from_openai`, `Mode`, the DSL helpers) is resolved through a module-level `__getattr__` the first time you touch it. Optional exports like `from_anthropic` are only added when the corresponding SDK is installed.
- [`Instructor`, L385](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/client.py#L385) — the sync client wrapper. It stores the patched `create_fn`, the active `Mode`, the `Provider`, default kwargs and a `Hooks` registry, and exposes `create`, `create_iterable`, `create_partial` and `create_with_completion`.
- [`AsyncInstructor`, L758](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/client.py#L758) — the async twin, overriding every create method with awaitable or async-generator variants.
- [`from_openai`, L119](https://github.com/567-labs/instructor/blob/main/instructor/v2/providers/openai/client.py#L119) — the factory that takes an `openai.OpenAI` (or compatible) client, patches its `chat.completions.create`, and returns an `Instructor`. The same file also powers `from_anyscale`, `from_together`, `from_databricks` and `from_deepseek`, because they are all OpenAI-wire-compatible.
- [`from_provider`, L92](https://github.com/567-labs/instructor/blob/main/instructor/v2/auto_client.py#L92) — the string-based router: `"openai/gpt-4o"` or `"anthropic/claude-3-5-sonnet"` is split on the slash, the right SDK is constructed, and the matching factory is called.
- [`patch`, L147](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/patch.py#L147) — the wrapper installer. `patch_v2` (just above it, at [L83](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/patch.py#L83)) validates that the `(provider, mode)` pair is registered, then wraps the underlying callable in a sync or async shell.
- [`ModeRegistry`, L59](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/registry.py#L59) — the dispatch table. A dictionary keyed by `(Provider, Mode)` maps to a `ModeHandlers` bundle; lookups are lazily loaded under a lock, and the module-level singleton lives at [L368](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/registry.py#L368).
- [`retry_sync_v2`, L138](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/retry.py#L138) — the validation loop. Every structured call passes through here: call the API, parse with the registry's `response_parser`, and on a `ValidationError` or JSON error, hand the request to the registry's `reask_handler` and try again.
- [`process_response`, L256](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/response.py#L256) — the parsing dispatcher that turns a raw completion into a validated model and attaches the original response as `_raw_response`.

## A Facade That Imports Nothing Until You Ask

The most interesting organizational decision in the current tree is how aggressively the package defers work. The top-level `instructor/__init__.py` declares `__all__` and a `_LAZY_IMPORTS` dictionary — [see L111](https://github.com/567-labs/instructor/blob/main/instructor/__init__.py#L111) — and nothing else runs at import time except two small helpers. When you write `instructor.from_provider(...)`, the `__getattr__` hook imports `.auto_client`, which itself is a shim re-exporting `instructor.v2.auto_client`. The same pattern applies to `Mode` (a one-line re-export of `instructor.v2.core.mode`) and to the legacy `instructor.core.client` path, which is now a star re-export of the v2 implementation.

This buys two things. Cold-start time: importing `instructor` no longer drags in provider SDKs — Anthropic or Gemini are only imported when their factory is resolved, and only if the SDK is installed, as the `_add_optional_export` calls at the bottom of `__init__.py` show. Backward compatibility: older code doing `from instructor.core.patch import patch` keeps working through the shims. The cost is indirection — a stack-trace frame pointing at `instructor/core/retry.py` is really executing code in `instructor/v2/core/retry.py`.

## From Install to Your First Model

Installation is a single package; there is no separate "provider pack" to install for the common case:

```bash
pip install instructor
# or: uv add instructor / poetry add instructor
```

The core dependencies, per `pyproject.toml`, are `openai`, `pydantic` (2.8+), `tenacity`, `httpx` and `jinja2`, and the library supports Python 3.9 through 3.x (anything below 4.0). If you plan to use Anthropic or Gemini factories, you install those SDKs yourself; Instructor detects their presence at import time.

The canonical first program, straight from the README:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4o-mini")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)
print(user)  # User(name='John', age=25)
```

What actually happens on that last call: `from_provider` constructs the OpenAI client and calls [`from_openai`, L119](https://github.com/567-labs/instructor/blob/main/instructor/v2/providers/openai/client.py#L119), which returns an `Instructor` whose `create_fn` is the patched `chat.completions.create`. The `chat` and `completions` attributes on the wrapper are literally `self` — [properties at L474–484 of the client](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/client.py#L474) — so the call site reads like vanilla OpenAI while the patched function does the schema work. If you want the raw completion too (token counts, finish reason), call `create_with_completion` instead; the wrapper returns a `(model, raw_response)` tuple, implemented at [L708](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/client.py#L708). Retry behavior is controlled by `max_retries` (default 3) on the `create` signature at [L538](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/client.py#L538).

## patch() and the Mode Registry

The heart of the library is the pair `patch` / `patch_v2` in [`instructor/v2/core/patch.py`](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/patch.py#L147). `patch` accepts either an OpenAI-style client or a bare `create` callable, resolves the underlying method, and hands it to `patch_v2`, which consults the registry and returns a wrapper. The sync wrapper — [`_create_sync_wrapper`, L185](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/patch.py#L185) — is where the per-request choreography happens:

1. Reject response models containing async validators (the sync path cannot honor them).
2. Validate any `token_budget` against the request via the budget helpers in [`instructor/v2/core/budget.py`](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/budget.py).
3. Pop cache kwargs — the wrapper supports transparent response caching when a cache adapter is passed.
4. Look up the `ModeHandlers` bundle from the registry and call its `request_handler` to inject provider-specific fields — for `Mode.TOOLS`, the `tools` array built from your model's `openai_schema`; for JSON modes, the schema folded into the system message. The schema itself is generated by [`generate_openai_schema`, L26](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/schema.py#L26) from your Pydantic model (docstring included as the description).
5. Run message templating (`handle_templating`) so Jinja-style placeholders in messages can be filled from a `context` dict.
6. Check the cache for a hit, then enter the retry chain with an isolated copy of the messages so reask mutations cannot leak into caller state.

The `Mode` enum in [`instructor/v2/core/mode.py`, L10](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/mode.py#L10) enumerates roughly forty strategies: `TOOLS`, `TOOLS_STRICT`, `JSON_SCHEMA`, `RESPONSES_TOOLS`, and vendor-specific entries like `ANTHROPIC_TOOLS`, `GEMINI_JSON`, `CEREBRAS_TOOLS`. Each has an entry in the registry whose bundle is typed by [`ModeHandlers`, L47](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/registry.py#L47): a `request_handler`, a `reask_handler`, a `response_parser`, and optional streaming extractors and message converters. The contract those handlers implement is the abstract [`ModeHandler`, L17](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/handler.py#L17), whose `prepare_request` and `handle_reask` methods define how a request is dressed up and how a failed attempt is rebuilt.

## The Validation Loop: Parse, Reask, Repeat

Once the prepared request reaches [`retry_sync_v2`, L138](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/retry.py#L138), the code builds a tenacity `Retrying` policy — `stop_after_attempt(max_retries + 1)`, optionally combined with `stop_after_delay` when a timeout is set — and enters the loop. On each attempt it emits the `completion:kwargs` hook, calls the real API, emits `completion:response`, and accumulates token usage into a running total via [`update_total_usage`, L117](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/usage.py#L117), which understands both OpenAI's `CompletionUsage` and Anthropic's `Usage` objects. That total later shows up on your model instance as `_total_usage`.

If the model parses and validates, the loop returns immediately, attaching `_raw_response` (and, for list results, a `ListResponse` wrapper — [`instructor/v2/dsl/response_list.py`, L15](https://github.com/567-labs/instructor/blob/main/instructor/v2/dsl/response_list.py#L15)). If instead the parser raises a `ValidationError`, a `json.JSONDecodeError`, an `AsyncValidationError` or a `ResponseParsingError` — the tuple of retryable errors is defined at the top of the retry module — the failed attempt is recorded as a `FailedAttempt`, the `parse:error` hook fires, and the registry's `reask_handler` rebuilds the request. For tools mode that job belongs to [`reask_tools`, L161](https://github.com/567-labs/instructor/blob/main/instructor/v2/providers/openai/handlers.py#L161) and its siblings: they append the validation error to the conversation, then raise to trigger the next attempt. When attempts run out, the loop raises [`InstructorRetryException`, L188](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/errors.py#L188), carrying the last completion, the attempt count, the accumulated usage and the full `FailedAttempt` history for debugging.

The detail diagram traces this machinery end to end, including the pieces the overview compresses:

<div style="overflow-x:auto;"><img src="/assets/img/diagrams/instructor/567-labs-instructor-architecture.svg" alt="Detail architecture of Instructor: facade shims, client layer, patching layer, mode registry, retry loop, response DSL and provider integrations" /></div>

<p><em>Detail view: facade shims, the client layer with hooks, patching-layer request preparation (templating, budget, caching), the (Provider, Mode) registry contract, the retry chain with usage accumulation and reask handlers, and the DSL types process_response understands.</em></p>

## The Response DSL

Not every response is a single flat model, so `instructor/v2/dsl/` provides a small family of wrappers. `Partial[User]` — the [`Partial` factory, L599](https://github.com/567-labs/instructor/blob/main/instructor/v2/dsl/partial.py#L599) — builds an all-Optional mirror of your model so streamed chunks can be parsed as they arrive; the README's streaming example prints a `User` whose fields fill in one by one. `IterableModel` — defined at [L204](https://github.com/567-labs/instructor/blob/main/instructor/v2/dsl/iterable.py#L204) of `iterable.py` — wraps a `tasks` list schema so one completion can yield many validated objects, with `IterableBase` supplying the chunk-splitting machinery. [`ModelAdapter`, L32](https://github.com/567-labs/instructor/blob/main/instructor/v2/dsl/simple_type.py#L32) lets you pass a bare Python type like `str` or `list[int]` as the response model, and `CitationMixin` ([L5](https://github.com/567-labs/instructor/blob/main/instructor/v2/dsl/citation.py#L5)) adds quote-anchored citations to any model. Under the hood, tool-call models ride on [`ResponseSchema`, L119](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/function_calls.py#L119), whose `openai_schema` class property (at [L124](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/function_calls.py#L124)) produces the function-calling schema and whose `from_response` delegates straight back into the registry parser.

If you build tool-facing systems, the schema-generation half of this story pairs naturally with the [MCP Python SDK](/mcp-python-sdk/), where tools also declare typed input schemas; for model-quality measurement rather than structure, see [lm-eval-harness](/lm-eval-harness/). For agentic retrieval flows that overlap with Instructor's extraction niche, see our [LlamaIndex](/llama-index/) source tour.

## Honest Limits

Staying within what the project itself claims: the README positions Instructor as a structured-extraction tool and explicitly says so — "Use Instructor for fast extraction, reach for PydanticAI when you need agents" — and its comparison section argues it is "focused on one thing — structured extraction" relative to agent frameworks. Popularity claims on the README (3M+ monthly downloads, 10K+ GitHub stars) are the maintainers' own figures, not ours. We did not run any benchmarks for this post and none are implied.

A few limits are visible in the code itself. The sync `create` path rejects async validators rather than honoring them, by design. `apatch` is a deprecated alias that emits a `DeprecationWarning` ([patch.py, L171](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/patch.py#L171)), and `Mode.FUNCTIONS` prints a deprecation warning when selected ([client.py, L407](https://github.com/567-labs/instructor/blob/main/instructor/v2/core/client.py#L407)) — the library is steering users toward tools modes. The v1-to-v2 reorganization means some documentation and code paths disagree about module locations; reading `instructor/core/*.py` shims before `instructor/v2/core/` is the fastest way to avoid confusion. Finally, retries are only as good as the model's willingness to fix its own output: the reask handlers feed validation errors back verbatim, which works well for schema slips and helps less with models that persistently misunderstand the task.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Instructor Source Tour: How the Structured-Outputs Library Turns LLM JSON Into Pydantic Models",
  "description": "A guided source tour of 567-labs/instructor: the lazy-import facade, the patch() wrapper, the (Provider, Mode) handler registry, and the tenacity-driven validation loop that reasks the model until the JSON fits.",
  "datePublished": "2026-10-29",
  "dateModified": "2026-10-29",
  "author": { "@type": "Person", "name": "PyShine" },
  "publisher": { "@type": "Organization", "name": "PyShine" },
  "keywords": "instructor, structured outputs, LLM, Pydantic, OpenAI, Anthropic, function calling, retry logic, source code, Python",
  "about": { "@type": "SoftwareSourceCode", "name": "instructor", "programmingLanguage": "Python", "codeRepository": "https://github.com/567-labs/instructor" }
}
</script>
