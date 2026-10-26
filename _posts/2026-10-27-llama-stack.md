---
layout: post
title: "Llama Stack Is Now OGX: A Source Tour of Meta's Agentic API Server"
description: "A guided source tour of meta-llama/llama-stack — now renamed OGX — covering the FastAPI server, unified inference APIs, provider plugins, routing tables, and the request flow from HTTP call to model."
date: 2026-10-27
header-img: "img/post-bg.jpg"
permalink: /llama-stack/
featured-img: "ai-coding-frameworks/ai-coding-frameworks"
image: https://pyshine.com/assets/img/diagrams/llama-stack/meta-llama-llama-stack-overview-architecture.svg
tags: [LLM, Python, Open Source]
categories: [AI Inference]
keywords: llama stack, meta, inference api, ogx, openai compatible api, provider plugins, fastapi server, source tour
author: "PyShine"
---

If you have been following the LLM serving ecosystem for the last couple of years, you have almost certainly bumped into Llama Stack, Meta's batteries-included attempt to standardize how applications talk to Llama models: one server, one set of provider plugins, one client SDK, and a provider routing layer that hides the difference between a model on your laptop and a model behind a hosted endpoint. In late 2026 the project took a step that surprised many people watching the repo: the README now opens with the line "Llama Stack is now OGX." The name changed, and so did the stated mission — the codebase is no longer framed as a Llama-only stack but as a model-agnostic, multi-SDK, OpenAI-compatible agentic API server. The GitHub repository we are touring today is still `meta-llama/llama-stack`, the default branch is still `main`, and the architecture documents inside it describe the project as a server that "exposes a unified API for AI capabilities: inference, responses orchestration, vector storage, tool execution, evaluation, and more."

Why is the source worth a tour rather than a skim of the docs? Because the interesting design decisions live in the file tree, not on the landing page. The codebase is split into two Python packages with a deliberately asymmetric relationship: `ogx_api` holds the API contracts — Python `Protocol` classes, Pydantic data types, and provider spec definitions — with no server code at all, while `ogx` holds the server implementation: provider resolution, routing, storage, a CLI, and every built-in provider adapter. Third-party providers are expected to depend only on the contract package. On top of that sits an OpenAI-compatible HTTP surface, an Anthropic-compatible `/v1/messages` dialect, and a Google GenAI-style interactions route, all served by the same FastAPI application. That is a lot of moving parts, and seeing how they connect is the fastest way to understand what the project actually promises.

In this post we walk the repository in the same spirit as our looks at [Llama 2](/llama2/) and [TGI](/tgi/), though this codebase is considerably broader than a single-model server. We generated two diagrams directly from the real file tree at HEAD (about 1,230 tracked files), map the major symbols with line-anchored links, trace one request end to end, and finish with an honest list of the repo's rough edges.

## The Big Picture

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/llama-stack/meta-llama-llama-stack-overview-architecture.svg" alt="Llama Stack architecture overview" style="max-width:100%;"></div>

<p><em>The overview layers of the repo: a CLI entry pair, the FastAPI server shell, the ogx_api contract package, the routing core, the provider registry, and a distribution template that turns all of it into one runnable stack.</em></p>

Reading the overview from left to right:

- **Two CLI entry points.** [`StackLetsGo`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/cli/stack/lets_go.py#L1007) backs the `ogx go` command: it probes your environment for usable providers via [`_autodetect_providers`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/cli/stack/lets_go.py#L583), writes a config, and boots the server. [`StackRun`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/cli/stack/run.py#L278) is the more explicit `ogx run`, whose [`run_stack_cmd`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/cli/stack/run.py#L74) validates a config file and hands off to uvicorn.
- **The FastAPI factory.** [`create_app`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/server/server.py#L470) reads the `OGX_CONFIG` environment variable, resolves it to a concrete config, and builds a [`StackApp`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/server/server.py#L139) instance — the FastAPI subclass every request ultimately lands on.
- **A middleware stack with real opinions.** [`AuthenticationMiddleware`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/server/auth.py#L33), [`RouteAuthorizationMiddleware`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/server/auth.py#L215), and [`TenancyMiddleware`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/server/auth.py#L277) handle tokens, per-route access policies, and single versus multi-tenant mode before any handler runs.
- **Router auto-discovery.** [`_discover_router_factories`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/server/fastapi_router_registry.py#L39) finds router factories contributed by each API implementation, and [`build_fastapi_router`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/server/fastapi_router_registry.py#L91) mounts exactly one FastAPI router per implemented API — which is why adding an API does not require editing the server.
- **Contracts in their own package.** The [`Inference`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx_api/inference/api.py#L140) protocol extends [`InferenceProvider`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx_api/inference/api.py#L48) and declares methods like `openai_chat_completion` as plain Python `Protocol` members. Anything that satisfies the protocol can serve inference; nothing in `ogx_api` imports server internals.
- **HTTP routes bound to the protocol.** [`create_router`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx_api/inference/fastapi_routes.py#L100) takes a concrete implementation and returns an `APIRouter` exposing the OpenAI-shaped endpoints, including SSE streaming.
- **One router to bind them.** [`InferenceRouter`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routers/inference.py#L107) is the server-side brain for inference: it resolves the requested model, applies access control, and delegates to whichever provider owns that model.
- **Routing tables and the registry.** [`CommonRoutingTableImpl`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routing_tables/common.py#L88) answers the central question "which provider handles this identifier?" through [`get_provider_impl`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routing_tables/common.py#L130), backed by the KVStore-backed [`DistributionRegistry`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/store/registry.py#L23). The provider lineup itself is declared in [`available_providers`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/providers/registry/inference.py#L16), and the starter distribution assembled by [`get_distribution_template`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/distributions/starter/starter.py#L125) picks a sensible default subset of it.

## From Install to Your First Model

The quickest path from nothing to a working server is deliberately short. The README offers a one-line shell installer, or you can go straight through `uv pip install ogx` and then run `uv run ogx go`. That single command is where `StackLetsGo` earns its name: instead of forcing you to hand-write a provider config, it detects which inference backends are actually reachable from your machine — an Ollama daemon on localhost, an OpenAI-shaped endpoint behind an environment variable, an Anthropic key — registers the models it can see, and starts serving. If you prefer determinism, `ogx run` takes an explicit YAML config, and `create_app` is happy with either: it funnels both through [`resolve_config_or_distro`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/server/server.py#L485), which accepts a path to a config file or the name of a bundled distribution such as `starter`.

Once the server is up on port 8321, you talk to it with a completely ordinary OpenAI client, which is the whole point of the rebrand:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8321/v1", api_key="fake")
response = client.chat.completions.create(
    model="llama-3.3-70b",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Nothing in that snippet is OGX-specific — any OpenAI SDK, any agentic framework that speaks the OpenAI wire format, works unchanged. The same server also answers Anthropic-style `/v1/messages` calls and Google GenAI-style interactions calls, so teams with mixed client libraries can point all of them at one process. Under the hood, the exact HTTP contract for the stable and experimental APIs is written down in the repo's own [`openapi.yml`](https://github.com/meta-llama/llama-stack/blob/main/client-sdks/spec/openapi.yml#L3), which doubles as the input for generating the separately-published client SDKs.

## Inside the Request Flow

<div style="overflow-x:auto;"><img src="https://pyshine.com/assets/img/diagrams/llama-stack/meta-llama-llama-stack-architecture.svg" alt="Llama Stack detailed request flow" style="max-width:100%;"></div>

<p><em>The detail view: one chat completion travelling from the CLI through the server shell and the contract package, across the routing tables and registry, and out to a provider adapter, with storage and access control in supporting roles.</em></p>

Let us follow `POST /v1/chat/completions` all the way through. The route handler produced by `create_router` does protocol-level validation, then dispatches to [`InferenceRouter.openai_chat_completion`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routers/inference.py#L256). The router first resolves `params.model` — a logical name like `ollama/llama3.2:3b` — into the provider-specific resource id. That resolution is exactly what [`get_provider_impl`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routing_tables/common.py#L130) is for: it asks the registry whether this model object exists and which provider id claims it, then returns the live implementation from its in-memory map. The registry itself is a [`CachedDiskDistributionRegistry`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/store/registry.py#L138), a disk-persisted registry with an in-memory cache and TTL refresh, which is how multiple uvicorn workers stay coherent when one of them registers a new model.

With the provider implementation in hand, the request crosses into provider land. Remote inference providers overwhelmingly inherit [`OpenAIMixin`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/providers/utils/inference/openai_mixin.py#L61), a shared adapter that owns an `AsyncOpenAI` client pointed at whatever base URL the provider declares, and adds opt-in behaviors like response-id rewriting, image downloading, and embedding metadata. The router resolves the model to the provider's own identifier before the adapter is called, so adapter code never re-parses logical names — a small decision that keeps provider implementations short. Streaming responses bubble back up as SSE events through the same chain, which is why the route layer carries explicit helpers for formatting SSE error events.

The same pattern generalizes beyond chat. [`VectorIORouter`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routers/vector_io.py#L69) routes vector-store operations, [`ToolRuntimeRouter`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routers/tool_runtime.py#L29) routes tool invocations such as web search and MCP servers, and each resource family gets its own routing table — [`ModelsRoutingTable`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routing_tables/models.py#L43) and [`VectorStoresRoutingTable`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/routing_tables/vector_stores.py#L45) are thin subclasses of the common implementation. Authorization is not bolted on afterward: the policy engine's [`matches_conditions`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/core/access_control/access_control.py#L92) evaluates structured rules against the resource being touched, and the server's route table maps every registered method-plus-path pair to its implementation so policies can be expressed per route.

## Providers: The Pluggable Core

The provider system is the reason to study this repo even if you never run the server. Every provider is described by a declarative spec — an `InlineProviderSpec` for implementations that run in-process, or a `RemoteProviderSpec` that adapts an external service — carrying the API it implements, its provider type string like `remote::ollama`, its Python module, and a Pydantic config class. The registry module collects those specs, the distribution templates bundle selections of them into runnable configs, and the routing layer instantiates only what the active config names. Adding a new backend is therefore mostly writing one adapter class plus one spec, not touching the server at all.

The remote inference family shows how cheap that is. The Ollama adapter, [`OllamaInferenceAdapter`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/providers/remote/inference/ollama/ollama.py#L50), inherits from `AnthropicMixin` and `OpenAIMixin` because Ollama speaks both dialects — which is what makes our earlier [Ollama](/ollama-3/) walkthrough a five-minute setup. The vLLM adapter, [`VLLMInferenceAdapter`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/providers/remote/inference/vllm/vllm.py#L79), is nearly the same story for production GPU serving, and pairs naturally with our [vLLM](/vllm-2/) notes. Dozens of other providers — Fireworks, Groq, Bedrock, Gemini, SambaNova, Cerebras — follow the identical template.

Inline providers cover the pieces you want running next to the server rather than behind it: a FAISS-backed vector store ([`FaissVectorIOAdapter`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/providers/inline/vector_io/faiss/faiss.py#L369) keeps its metadata in a SQLite-backed SQL store), local file search over those stores, and the built-in Responses implementation, [`BuiltinResponsesImpl`](https://github.com/meta-llama/llama-stack/blob/main/src/ogx/providers/inline/responses/builtin/impl.py#L99), which orchestrates multi-step agentic turns — tool calls, file search, follow-up completions — server-side so a single API call can do real work. Stateful conversations get their own [`Conversations` protocol](https://github.com/meta-llama/llama-stack/blob/main/src/ogx_api/conversations/api.py#L26), with the Pydantic types for the Responses API living in the contract package where SDK generators can reach them.

## Honest Limits

No source tour is complete without the caveats, and this repo has some visible seams. First, the name churn is real and it leaks into the code: `src/ogx/models/llama/` is today an empty package whose `__init__.py` contains only a copyright header. The Llama-specific inference implementations, tokenizers, and chat-format code that older material describes at `llama_stack/models/llama/` are simply not there anymore — the project is explicitly model-agnostic now, and the Llama model plumbing has moved out of this tree. Anyone following two-year-old tutorials will find the paths have moved twice: first `llama_stack/distribution/*` to `llama_stack/core/*` in earlier restructurings, and now into the `src/ogx` and `src/ogx_api` packages with entirely new import roots.

Second, the client SDKs are no longer in this repository. `client-sdks/` contains only the OpenAPI specification and its build tooling; the Python and TypeScript clients live in separate repositories and are generated from that spec. The tradeoff: a source dive into this repo will not show you client-side streaming internals. Third, breadth cuts both ways: the provider matrix is enormous, and coverage quality is uneven — the CI file even mentions disabling the watsonx and vertexai test lanes while their accounts are sorted out, a candid reminder that "supported provider" and "exercised provider" are not the same thing. Finally, if your need is narrow — serve one model, fast, with a tiny operational footprint — a focused server like [TGI](/tgi/) remains easier to reason about than a platform whose starter config wires up batches, files, vector stores, skills, and an admin surface you may never use.

## Why This Codebase Is Worth Reading

Strip the branding away and what remains is one of the cleanest large-scale demonstrations of two ideas. The first is dependency inversion done for real: `ogx_api` is pure contract — Protocol classes and Pydantic types with zero server imports — so an ecosystem of third-party providers can compile against a package that will never drag FastAPI or a database into their dependency tree. The second is the registry-plus-routing-table pattern: every resource family (models, vector stores, tool groups, conversations) flows through the same `get_provider_impl` discipline, which is why the server can grow a new API surface without growing new routing concepts to match.

Practically, the unified-API bet has paid off well enough that Meta renamed the whole project around it: one process, OpenAI and Anthropic and Google dialects on the same port, providers swappable by config. Read the server entry point first, then one routing table, then one adapter built on `OpenAIMixin` — those three files contain most of the architecture. The rest is plugins, and now you know exactly where the plugin holes are drilled.

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "datePublished": "2026-10-27",
  "headline": "Llama Stack Is Now OGX: A Source Tour of Meta's Agentic API Server",
  "description": "A guided source tour of meta-llama/llama-stack — now renamed OGX — covering the FastAPI server, unified inference APIs, provider plugins, routing tables, and the request flow from HTTP call to model.",
  "author": "PyShine",
  "keywords": "llama stack, meta, inference api, ogx, openai compatible api, provider plugins, fastapi server"
}
</script>
