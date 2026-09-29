---
layout: post
title: "LiteLLM: One OpenAI-Format API for 100+ LLM Providers - Inside BerriAI/litellm"
description: "A source-level tour of BerriAI/litellm, the open-source AI gateway and Python SDK that exposes 100+ LLM providers behind a single OpenAI-format API. We walk through the proxy server, the completion routing layer, provider adapter transforms, cost tracking, and the caching architecture that ties it all together."
date: 2026-09-29
header-img: "img/post-bg.jpg"
permalink: /LiteLLM-One-Gateway-For-100-Plus-LLM-Providers/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/litellm/berriai-litellm-architecture.svg
tags:
  - AI
  - LLM
  - API Gateway
  - Python
categories: [AI, Open Source]
keywords: "LiteLLM, AI gateway, LLM proxy, OpenAI format, LLM API translation, load balancing, model routing, BerriAI, Python SDK, spend tracking, prompt caching, FastAPI, provider adapters, router strategies"
author: "PyShine"
---

Every serious LLM application eventually hits the same wall: the code you wrote against OpenAI's SDK now needs to call Anthropic, Bedrock, Gemini, Azure, and a dozen self-hosted models — each with its own authentication scheme, request shape, streaming protocol, and error vocabulary. Rewriting per vendor is miserable, and lock-in is expensive. LiteLLM, an open-source project from BerriAI, attacks that problem at the API-translation layer: you keep writing OpenAI-format calls, and LiteLLM does the talking to everything else.

LiteLLM ships in two shapes that share one codebase. It is a **Python SDK** (`pip install litellm`) that gives you `completion()` and `acompletion()` functions covering 100+ providers, and it is a self-hosted **AI Gateway** — a FastAPI proxy server that centralizes authentication, rate limiting, budgets, load balancing, and spend tracking for a whole team. The gateway is not a separate product; per the repository's own ARCHITECTURE.md, the proxy uses the SDK internally for every LLM call, so the same translation code serves both.

What makes the repository genuinely worth a source tour is that the architecture is unusually legible once you know where to look. There is a deliberate seam between the gateway (`litellm/proxy/`), the router (`litellm/router.py`), the call core (`litellm/main.py`), and a huge family of provider adapters under `litellm/llms/` — 139 provider subpackages at the time of writing, from OpenAI and Anthropic to Ollama and Vertex AI. In this post we walk through those layers in source order and show how a single `/v1/chat/completions` request becomes a provider HTTP call, comes back translated, and gets priced.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/litellm/berriai-litellm-overview-architecture.svg" alt="Architecture overview of the BerriAI/litellm repository" style="max-width:100%;height:auto;" />
</div>

*Overview of the LiteLLM codebase: the FastAPI gateway authenticates and hooks each request, hands it to the Router for deployment selection, and delegates the actual provider call to the SDK core and its per-provider transform adapters, with cost and logging wired in as cross-cutting concerns.*

Reading the overview from left to right: a request lands on the proxy server (`litellm/proxy/proxy_server.py`), which first verifies the caller via API-key authentication (`litellm/proxy/auth/user_api_key_auth.py`) backed by a shared cache, and runs pre-call proxy hooks (`litellm/proxy/hooks/`). The routing module (`litellm/proxy/route_llm_request.py`) then dispatches the call to the `Router` (`litellm/router.py`), which picks a concrete deployment using pluggable balancing strategies (`litellm/router_strategy/`) and consults the DualCache for TPM/RPM counters and cooldown state. The SDK entry points in `litellm/main.py` take over, driving the central HTTP handler (`litellm/llms/custom_httpx/llm_http_handler.py`), which applies a provider-specific `BaseConfig` transform before sending anything over the wire. On the way back, the logging object (`litellm/litellm_core_utils/litellm_logging.py`) fans out success events and the cost calculator (`litellm/cost_calculator.py`) attributes spend.

## Why You Need This

The first problem LiteLLM solves is plain **API fragmentation**. Every provider disagrees about message roles, tool-calling JSON, image parts, thinking budgets, and streaming chunk formats. LiteLLM normalizes all of it: `get_llm_provider()` (implemented in `litellm/litellm_core_utils/get_llm_provider_logic.py` and re-exported through `litellm/utils.py`) parses a model string like `anthropic/claude-sonnet-4-20250514` into provider, model, and credentials, and from there the OpenAI-shaped payload you supplied is translated in both directions. Your application code never learns a second SDK.

The second problem is **operational resilience**. A single provider is not a single endpoint: you may run several deployments of the same model across regions or accounts, and any of them can 429 or 500 at any moment. The `Router` in `litellm/router.py` handles retries, fallbacks across model groups, and cooldowns — unhealthy deployments are evicted via `litellm/router_utils/cooldown_handlers.py` and traffic shifts to healthy ones. That logic is painful to hand-roll, and it sits at the heart of why teams run the gateway rather than calling providers directly.

Third, there is the **governance problem**. Once more than a handful of developers and services touch LLMs, you need virtual API keys, per-key and per-team budgets, rate limits, and a single audit trail of spend. That is exactly what the proxy layer adds: `litellm/proxy/auth/user_api_key_auth.py` validates keys, `litellm/proxy/hooks/parallel_request_limiter_v3.py` enforces TPM/RPM limits, and every response's cost is computed and persisted so budgets are real rather than aspirational.

Finally, LiteLLM solves the **experimentation problem**. Because the entire surface speaks OpenAI format, evaluating a cheaper model means changing a string in a config file, not refactoring call sites. The repository's model price map (`model_prices_and_context_window.json`) plus `litellm/cost_calculator.py` make before/after cost comparisons mechanical, and the router's strategy modules — lowest latency, lowest cost, least busy — let you optimize for whichever dimension matters to you this quarter.

## How It Works

The cleanest way to understand LiteLLM is to follow one request through the layers, so we have drawn the detailed module graph of the gateway, router, and adapter pipeline below.

<div style="overflow-x:auto;">
<img src="https://pyshine.com/assets/img/diagrams/litellm/berriai-litellm-architecture.svg" alt="Detailed architecture of the BerriAI/litellm repository" style="max-width:100%;height:auto;" />
</div>

*Detailed source map of BerriAI/litellm: the FastAPI gateway with its auth, hooks, and spend-writing machinery; the Router with its four balancing strategies and cooldown handlers; the SDK core with provider resolution and streaming; the per-provider transform adapters; the caching layer; and the logging and cost pipeline fed by the model price map.*

### Understanding the Architecture

**The gateway is a FastAPI application, not a monolith.** The `app = FastAPI(...)` instance in `litellm/proxy/proxy_server.py` mounts the familiar OpenAI routes — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` — plus native endpoints such as `/v1/messages` (`litellm/proxy/anthropic_endpoints/`), the Responses API (`litellm/proxy/response_api_endpoints/`), and provider passthrough routes (`litellm/proxy/pass_through_endpoints/`). The `litellm` console command is wired in `pyproject.toml` to `litellm:run_server`, which boots this app via `litellm/proxy/proxy_cli.py`. Every LLM route depends on `user_api_key_auth` (`litellm/proxy/auth/user_api_key_auth.py`), which resolves the virtual key against the in-memory-plus-Redis usage cache and falls back to the PostgreSQL database described by `litellm/proxy/schema.prisma`.

**Routing is a separate, reusable engine.** `litellm/proxy/route_llm_request.py` maps each endpoint to a router method (`acompletion`, `aembedding`, and so on in its `ROUTE_ENDPOINT_MAPPING`) and hands the request to the `Router` class in `litellm/router.py`. The Router owns the model list, applies retry and fallback policies, tracks TPM/RPM budgets in its `DualCache`, and consults cooldown handlers in `litellm/router_utils/cooldown_handlers.py` before choosing a deployment. Selection itself is delegated to strategy modules in `litellm/router_strategy/`: `simple_shuffle.py` for weighted random, `lowest_latency.py`, `lowest_cost.py`, and `least_busy.py` for the corresponding policies, with tag-based and budget-limiter variants alongside.

**Provider support is a transform, not a fork.** Each provider gets a subpackage under `litellm/llms/` whose chat config class inherits from `BaseConfig` in `litellm/llms/base_llm/chat/transformation.py` and implements `transform_request()` and `transform_response()`. The central orchestrator, `BaseLLMHTTPHandler` in `litellm/llms/custom_httpx/llm_http_handler.py`, resolves the right config and calls those two methods around a raw HTTP exchange performed by `HTTPHandler`/`AsyncHTTPHandler` (`litellm/llms/custom_httpx/http_handler.py`). Adding a provider means writing one transformation file — Anthropic's lives at `litellm/llms/anthropic/chat/transformation.py`, Bedrock Converse's at `litellm/llms/bedrock/chat/converse_transformation.py` — and the handler never changes.

**The SDK core owns the call lifecycle.** `litellm/main.py` hosts the public `completion()`/`acompletion()` entry points: after provider resolution, it builds the logging object, checks the LLM response cache in `litellm/caching/caching.py` (with in-memory, disk, S3, Azure Blob, GCS, and Redis backends, plus a `DualCache` in `litellm/caching/dual_cache.py` that layers Redis over local memory), and either returns a `ModelResponse` or wraps streaming chunks in `CustomStreamWrapper` from `litellm/litellm_core_utils/streaming_handler.py`. Provider errors are normalized to OpenAI-style exceptions by `litellm/exceptions.py` and `litellm_core_utils/exception_mapping_utils.py`, so your `except` blocks stay provider-agnostic.

**Cost tracking is a first-class pipeline.** The logging object created per call (`litellm/litellm_core_utils/litellm_logging.py`) fans success and failure events out to integrations in `litellm/integrations/` — Langfuse, Datadog, and many more — and computes spend via `completion_cost()` in `litellm/cost_calculator.py`, which reads token prices from `model_prices_and_context_window.json`. In gateway mode, `litellm/proxy/common_request_processing.py` surfaces that cost on the response, and the `DBSpendUpdateWriter` (`litellm/proxy/db/db_spend_update_writer.py`) batches spend increments into PostgreSQL, with background jobs refreshing budgets and syncing deployments on a schedule.

**Everything converges on one end-to-end flow.** A client POSTs OpenAI-format JSON to the gateway; auth and hooks validate and throttle it; `route_request()` selects a healthy deployment through the Router and a strategy; `acompletion()` resolves the provider and invokes `BaseLLMHTTPHandler`; the provider's config translates the request to native wire format and translates the response back; the streaming wrapper or response object returns through the same path while the logging object fires callbacks, computes cost, and queues spend for the database. One file owns each step, which is why the ARCHITECTURE.md debugging recipes — "compare how each translation handles cache_control in transform_request()" — actually work in practice.

## Advantages

- **One API, 139 provider subpackages.** The `litellm/llms/` tree gives OpenAI, Anthropic, Bedrock, Gemini, Vertex AI, Azure, Ollama, vLLM, and over a hundred more the same `transform_request()`/`transform_response()` contract, so coverage is structural rather than incidental.
- **Battle-tested routing built in.** `litellm/router.py` composes retries, fallbacks, cooldown eviction (`litellm/router_utils/cooldown_handlers.py`), and pluggable strategies — lowest latency, lowest cost, least busy, weighted shuffle — instead of leaving you to reinvent them.
- **Drop-in OpenAI compatibility.** The gateway exposes `/v1/chat/completions` and friends, so any OpenAI SDK client works by changing only `base_url`, and Anthropic-native `/v1/messages` is also mounted for clients that want it.
- **Provider errors normalized once.** `litellm/exceptions.py` plus `litellm_core_utils/exception_mapping_utils.py` map every vendor's 429s and 500s into a consistent exception hierarchy, which makes retry code portable across providers.
- **Cost attribution without instrumentation.** `litellm/cost_calculator.py` and the maintained `model_prices_and_context_window.json` price map attach a computed cost to every response, in the SDK and in the gateway alike.
- **Streaming done properly.** `litellm_core_utils/streaming_handler.py` converts each provider's chunk dialect into OpenAI-style stream deltas, including usage aggregation, behind one `CustomStreamWrapper` type.

## Benefits

- **No vendor lock-in.** Switching models or providers is a config change; the router can even fall back across model groups automatically when a primary provider degrades.
- **Team-scale governance.** Virtual keys, per-key and per-team budgets, TPM/RPM limiting (`litellm/proxy/hooks/parallel_request_limiter_v3.py`), and audit-ready spend logs come with the self-hosted gateway rather than as paid add-ons.
- **Centralized observability.** The `litellm/integrations/` callback family streams call data to Langfuse, Datadog, and similar backends from one code path, so tracing coverage does not depend on each service remembering to add it.
- **Performance that scales sideways.** The DualCache pattern (`litellm/caching/dual_cache.py`) keeps hot state in-process and shared state in Redis, which is how the gateway stays fast under concurrent rate limiting and cooldown checks; the README cites 8ms P95 latency at 1k RPS in its published benchmarks.
- **A codebase you can actually extend.** The ARCHITECTURE.md documents exactly which file to touch for a new provider, a new hook, or a new proxy endpoint, and translations are unit-testable without any API calls.
- **Self-hosted and MIT-licensed core.** The SDK and gateway source under the repo root are MIT licensed (the `enterprise/` directory carries its own license), so you can run the whole gateway on your own infrastructure.

## Usage

Install the Python SDK and call two providers through the same function (from the README):

```shell
uv add litellm
```

```python
from litellm import completion
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# OpenAI
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hello!"}])

# Anthropic
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=[{"role": "user", "content": "Hello!"}])
```

Start the AI Gateway with a single model:

```shell
uv tool install 'litellm[proxy]'
litellm --model gpt-4o
```

Point any OpenAI client at it:

```python
import openai

client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

Or call the gateway directly over HTTP:

```bash
curl -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
  -H 'Authorization: Bearer <your-master-key>' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

For the full deployment — gateway, PostgreSQL for keys and spend logs, and Prometheus — the repository's `docker-compose.yml` builds the `docker.litellm.ai/berriai/litellm:main-stable` image and brings up the database with `docker-compose up`.

## Conclusion

LiteLLM's source rewards the kind of tour we have taken here: the gateway, the router, the SDK core, and the provider transforms are genuinely separate layers with narrow contracts between them. `BaseConfig`'s two-method transform contract explains how 100+ providers stay maintainable; the Router's strategy and cooldown modules explain how the gateway keeps traffic flowing through failures; and the logging-to-cost-calculator pipeline explains how spend tracking happens without touching application code. If you are evaluating an AI gateway or just want a masterclass in API-translation design, cloning this repository and reading along `ARCHITECTURE.md` is time well spent.

Links:

- GitHub repository: [https://github.com/BerriAI/litellm](https://github.com/BerriAI/litellm)
- Documentation: [https://docs.litellm.ai](https://docs.litellm.ai)
