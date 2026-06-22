---
layout: post
title: "FreeLLMAPI: OpenAI-Compatible Proxy Stacking 16 Free LLM Providers"
description: "Learn how FreeLLMAPI provides an OpenAI-compatible API proxy that stacks 16+ free LLM providers for zero-cost AI access with smart routing and automatic fallover."
date: 2026-06-22
header-img: "img/post-bg.jpg"
permalink: /FreeLLMAPI-OpenAI-Compatible-Proxy-16-Free-LLM-Providers/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - LLM
  - Open Source
  - TypeScript
  - API
author: "PyShine"
---

# FreeLLMAPI: OpenAI-Compatible Proxy Stacking 16 Free LLM Providers

Every major AI lab now offers a free tier of their LLM API -- Google Gemini gives millions of tokens per month, Groq provides ultra-fast inference at no cost, and Cerebras, NVIDIA, and Mistral all offer zero-cost access to capable models. Alone these tiers are modest, but combined they provide roughly **1.7 billion tokens per month** across 100+ models.

The problem has always been managing the integration. Seventeen different SDKs, seventeen sets of rate limits, seventeen authentication flows -- that complexity killed most free-tier projects before they started. [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) solves exactly this, providing a single self-hosted, OpenAI-compatible proxy that transparently routes across your configured free-tier providers with smart routing, automatic fallover on errors, per-key rate limiting, encrypted key storage, and a management dashboard.

**At a glance:**

| Metric | Value |
|--------|-------|
| Free LLM Providers | 16+ (with a custom endpoint option) |
| Combined Free Tokens | ~1.7 Billion / month |
| OpenAI-compatible Endpoint | POST /v1/chat/completions, GET /v1/models |
| Anthropic-compatible Endpoint | POST /v1/messages (Claude SDK support) |
| Key Security | AES-256-GCM at-rest encryption |
| License | MIT |
| Language | TypeScript / Node.js 20+ |

![FreeLLMAPI Architecture](/assets/img/diagrams/freellmapi/freellmapi-architecture.svg)

### Understanding the Architecture

The architecture diagram above shows the multi-layered design of FreeLLMAPI. Requests flow from any OpenAI-compatible client through a unified proxy port at :3001. The server supports not only the OpenAI Chat Completions API but also the Anthropic Messages API (making **Claude Code** directly compatible), the OpenAI Responses API, embedding endpoints, image generation, and text-to-speech endpoints.

**Client Layer (Green):** Any application that can speak the OpenAI protocol connects directly. This ranges from the Python `openai` package and `curl` on the command line, to **Claude Code** pointed at `ANTHROPIC_BASE_URL=http://localhost:3001`, to LangChain agents, LlamaIndex pipelines, IDE extensions, and automation workflows.

**Express Proxy Layer (Blue, :3001):** At the heart, the Express server handles four simultaneous API personalities: the OpenAI chat completions response format at `/v1/chat/completions`, the more recent Responses endpoint at `/v1/responses` used by current Codex CLI versions, the anthropic-formatted messages endpoint at `/v1/messages`, and the multi-modal endpoints `/v1/images/generations`, `/v1/audio/speech`, and `/v1/embeddings`. Each incoming request carries a single `freellmapi-` prefixed API key instead of seventeen different upstream bearer tokens.

**Smart Router (Orange Decision Node):** The router evaluates every request through Bandit-pro algorithm based intelligence + speed ranking: picking highest priority currently available models that have healthy and non rate-exhausted keys present, and routing there initially. It automatically handles context-aware scoring with reliability (based on historic availability and 5xx-rate tracking data to learn which keys and models on which upstream provide a faster response), speed scores, intelligence ranking preferences, headroom metrics on token-buckets still active per-provider limit window, rate-limiter-allowances, all composed into a single scoring priority order that your client code never concerns itself with (except the order configured on the chain priority in the Admin UI which determines the default scoring preferences priorization override preferences via Fallback UI chain view page you get access).

---

## Supported Providers

FreeLLMAPI currently integrates these free LLM providers organized into authentication categories.

### Providers That Require API Keys

| Provider | Free Tier Highlights | Auth Method |
|----------|---------------------|-------------|
| Google Gemini | ~1M tokens/day; Gemini 2.5 Flash, 3.x previews | API key from ai.google.dev |
| Groq | Llama 4 Maverick/Scout, Qwen3; ~600 req/day via LPU inference | api.groq.com keys free |
| Cerebras | Qwen3 235B; ultra-fast inference | api.cerebras.ai keys free |
| NVIDIA NIM | 40 RPM free (eval-only ToS) | build.nvidia.com keys free |
| Mistral | Large 3, Medium 3.5, Codestral, Devstral | api.mistral.ai keys free |
| OpenRouter | 21 free-tier models routed across providers | openrouter.ai keys free |
| GitHub Models | GPT-4.1, GPT-4o free tier | GitHub personal access token |
| Cohere | Command R+, Command-A (trial) | api.cohere.com keys free |
| Cloudflare | Kimi K2, GLM-4.7, GPT-OSS, Granite 4 on Workers AI | account_id:token from Cloudflare |
| Z.ai (Zhipu) | GLM-4.5, GLM-4.7 Flash | open.bigmodel.cn keys free |
| HuggingFace | Router to DeepSeek V4, Kimi K2.6, Qwen3 | hf.co Inference API keys free |
| SiliconFlow | FLUX.1-schnell image gen, CosyVoice2 TTS | siliconflow.com keys free |
| Reka | Reka Flash 3 (text), Reka Edge 2603 (multimodal) | platform.reka.ai keys free |
| Agnes AI | Agnes models at $0/token (promotional) | platform.agnes-ai.com keys free |
| OpenCode Zen | DeepSeek V4 Flash, Nemotron (promo) | opencode.ai/auth keys free |

### Providers That Work Without API Keys (Anonymous Access)

| Provider | Free Tier Highlights | Access |
|----------|---------------------|--------|
| Ollama Cloud | GLM-4.7, Kimi K2, Qwen3; 1 concurrent, 5h session caps | Anonymous (no key needed) |
| Kilo Gateway | :free routes; 200 req/hr per IP | Anonymous (keyless) |
| Pollinations | GPT-OSS 20B; 1 concurrent request per IP | Anonymous (keyless) |
| LLM7 | GPT-OSS, Llama 3.1, GLM; 100 req/hr | Anonymous or keyed |
| OVH AI Endpoints | Qwen3.5 397B, GPT-OSS, Llama 3.3; 2 req/min per IP per model | Anonymous (keyless) |

Plus a **custom** provider option -- point at any OpenAI-compatible endpoint (llama.cpp, LM Studio, vLLM, local Ollama, or a remote gateway) from the Keys page.

---

## Smart Routing and Automatic Fallback

![Smart Router and Fallback](/assets/img/diagrams/freellmapi/freellmapi-router-fallback.svg)

The diagram above illustrates how FreeLLMAPI routes each incoming request through its intelligent fallback system. When a client sends a request with `model: "auto"`, the proxy does not simply forward it to a random provider. Instead, the Bandit-pro router evaluates every available model and key combination against multiple scoring dimensions to pick the best option for that specific request.

**Request Flow:** Every request enters through the unified Express proxy on port 3001, which accepts a single `freellmapi-...` bearer token. The proxy first validates this token against its internal key registry. If the token is invalid, the request is rejected immediately with a 401 error. Valid requests proceed to the session lookup phase.

**Session Lookup:** The router checks whether the client has an active sticky session (identified by the `X-Session-Id` header or a SHA-1 hash of the first user message). If a session exists and the previously used model is still healthy and under its rate limits, the router reuses that model for continuity. Sticky sessions last 30 minutes, preventing the hallucination spikes that occur when a conversation switches models mid-stream.

**Bandit-pro Scoring:** When no sticky session applies, the router scores every candidate model across four dimensions: health status (healthy, rate_limited, invalid, or error), per-key rate limit headroom (RPM, RPD, TPM, TPD counters), intelligence ranking (model capability tier), and speed scores (latency tracking from historical 5xx rates). These dimensions compose into a single priority order that determines which model the router tries first.

**Automatic Fallback:** If the chosen provider returns a 429 (rate limit), 5xx (server error), or times out, the router immediately places that key on a cooldown, skips it, and retries on the next model in the fallback chain. This continues for up to 20 attempts across all configured providers. Only if every key is exhausted does the proxy return a 503 error to the client.

**Context Handoff:** When a session falls over to a different model mid-conversation, FreeLLMAPI can optionally inject a compact system message that tells the new model it is continuing an existing task. This feature is disabled by default and enabled with `FREELLMAPI_CONTEXT_HANDOFF=on_model_switch` in `.env`. The handoff message includes a brief session summary so the new model can pick up where the previous one left off without re-asking questions or discarding prior tool results.

---

## Security Features

![Security Flow](/assets/img/diagrams/freellmapi/freellmapi-security-flow.svg)

The security diagram above shows how FreeLLMAPI protects your provider API keys and controls access to the proxy. Security is built in layers: encrypted storage at rest, unified authentication at the edge, and per-key rate limiting to keep you under every provider's free-tier caps.

**AES-256-GCM Key Encryption:** All upstream provider API keys are encrypted with AES-256-GCM before being written to the SQLite database. Decryption happens in-memory only at the moment a request needs the key, and the plaintext never touches disk or logs. The encryption key is generated on first run via `openssl rand -hex 32` and stored in the `.env` file. This means that even if someone gains access to the SQLite database file, they cannot extract your provider keys without the encryption key.

**Unified Authentication:** Clients authenticate to the proxy with a single `freellmapi-...` bearer token instead of seventeen different upstream API keys. This token is generated by the dashboard and can be rotated at any time. Your applications never see the real provider keys -- they only know the unified token. The `/v1` proxy routes use this unified-key auth, while the admin dashboard and all `/api/*` routes are gated behind an email + password account with scrypt-hashed passwords and session-token auth.

**Per-Key Rate Limiting:** The router maintains in-memory counters for RPM (requests per minute), RPD (requests per day), TPM (tokens per minute), and TPD (tokens per day) for every `(platform, model, key)` combination. When a provider returns a 429 rate-limit response, the affected key is placed on a cooldown and the router automatically falls over to the next available key. This ensures you never waste a request on a key that has already hit its cap.

**Health Probes:** A background health-check service periodically tests each key by making lightweight requests to the upstream provider. Keys are marked as `healthy`, `rate_limited`, `invalid`, or `error` so the router can skip dead keys automatically. This means that if a provider revokes your key or changes their free-tier limits, the router learns about it and routes around the problem without any manual intervention.

---

## Installation

### Docker (Recommended)

The fastest way to get started is with the one-liner install script:

```bash
curl -fsSL https://freellmapi.co/install.sh | bash
```

This sets up `~/freellmapi`, generates an encryption key, pulls the Docker image, and starts the container. Re-running it is safe -- your `.env` and encryption key are preserved.

Or manually with Docker Compose:

```bash
git clone https://github.com/tashfeenahmed/freellmapi.git
cd freellmapi

# Generate an encryption key for at-rest key storage
ENCRYPTION_KEY="$(openssl rand -hex 32)"
printf "ENCRYPTION_KEY=%s\nPORT=3001\n" "$ENCRYPTION_KEY" > .env

docker compose up -d
```

Open http://localhost:3001, add your provider keys on the Keys page, reorder the Fallback Chain, and grab your unified API key.

### Local Development

```bash
git clone https://github.com/tashfeenahmed/freellmapi.git
cd freellmapi
npm install
cp .env.example .env
ENCRYPTION_KEY="$(node -e 'console.log(require("crypto").randomBytes(32).toString("hex"))')"
printf "ENCRYPTION_KEY=%s\nPORT=3001\n" "$ENCRYPTION_KEY" > .env
npm run dev
```

Open http://localhost:5173 for the Vite dev UI. For production builds: `npm run build && node server/dist/index.js`.

### Desktop App

Download the macOS `.dmg` or Windows `.exe` installer from the [latest release](https://github.com/tashfeenahmed/freellmapi/releases/latest). The desktop app runs the entire router and dashboard from your system tray.

---

## Usage

### Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3001/v1",
    api_key="freellmapi-your-unified-key",
)

resp = client.chat.completions.create(
    model="auto",  # let the router pick; or specify e.g. "gemini-2.5-flash"
    messages=[{"role": "user", "content": "Summarise the fall of Rome in one sentence."}],
)
print(resp.choices[0].message.content)
print("Routed via:", resp.headers.get("x-routed-via"))
```

### Claude Code

Point Claude Code at your free pool by setting two environment variables:

```bash
export ANTHROPIC_BASE_URL=http://localhost:3001
export ANTHROPIC_AUTH_TOKEN=freellmapi-your-unified-key   # NOT ANTHROPIC_API_KEY
claude
```

Use `ANTHROPIC_AUTH_TOKEN` (sent as a Bearer token), not `ANTHROPIC_API_KEY` -- Claude Code treats a set `ANTHROPIC_API_KEY` as a conflicting first-party credential and refuses to start.

### curl

```bash
curl http://localhost:3001/v1/chat/completions \
  -H "Authorization: Bearer freellmapi-your-unified-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

Every response carries an `X-Routed-Via: <platform>/<model>` header so you can see which provider actually served each call. If a request fell over between providers, you will also see `X-Fallback-Attempts: N`.

---

## Admin Dashboard

The built-in admin dashboard is a React + Vite + shadcn/ui application served at the root URL. It provides:

- **Keys page** -- Add, remove, and health-check your upstream provider API keys. Each key shows a status dot (green/yellow/red) and when it was last probed. Grab your unified `freellmapi-...` key from the header.
- **Fallback Chain page** -- Drag-and-drop reorder the priority chain that the router follows when picking models. Models higher in the chain are tried first (subject to health and rate-limit checks).
- **Models page** -- Browse all available models organized by provider, with toggle switches to enable or disable individual models. Separate tabs for Chat, Embeddings, Image, and Audio models.
- **Playground page** -- Send a chat completion through the router and see which provider served it, with the model ID and latency printed on the response.
- **Analytics page** -- Request volume, success rate, tokens in and out, average latency, and per-provider breakdowns over 24h / 7d / 30d windows.

---

## Key Features

| Feature | Description |
|---------|-------------|
| OpenAI-compatible | POST /v1/chat/completions and GET /v1/models work with official OpenAI SDKs |
| Anthropic Messages API | POST /v1/messages for Claude Code and Anthropic SDKs |
| Responses API | POST /v1/responses for Codex CLI wire format |
| Embeddings | /v1/embeddings with family-based routing (failover never crosses models) |
| Image generation | POST /v1/images/generations routes across media-capable providers |
| Text-to-speech | POST /v1/audio/speech routes across TTS-capable providers |
| Streaming | Server-Sent Events for stream:true, JSON response otherwise |
| Tool calling | OpenAI-style tools/tool_choice passed through; multi-step flows work |
| Automatic fallover | 429/5xx/timeout triggers cooldown and retry on next provider (up to 20 attempts) |
| Per-key rate tracking | RPM, RPD, TPM, TPD counters per (platform, model, key) |
| Sticky sessions | 30-minute session affinity to avoid mid-conversation model switches |
| Context handoff | Optional system message injection on model switch |
| Encrypted key storage | AES-256-GCM at-rest encryption for all provider keys |
| Unified API key | Single freellmapi-... bearer token instead of 17 upstream keys |
| Health checks | Periodic probes mark keys as healthy/rate_limited/invalid/error |
| Admin dashboard | React + Vite + shadcn/ui for key management, chain reordering, analytics |
| Vision support | Image input routes to vision-capable models only |
| Google Search grounding | Request google_search tool for Gemini models |
| Desktop app | macOS .dmg and Windows .exe tray app |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| 401 Unauthorized | Check that your unified key starts with `freellmapi-` and matches the one shown on the Keys page |
| 429 Too Many Requests | The router should auto-fallback; if all keys are exhausted, wait for cooldown or add more provider keys |
| 503 All Keys Exhausted | Every configured key is on cooldown. Add keys for additional providers or wait for rate-limit windows to reset |
| Model not found in /v1/models | Enable the model on the dashboard Models page; check that the provider key is valid and healthy |
| Slow responses | Cerebras and Groq are fastest; later in the day the router may fall back to slower providers. Reorder the Fallback Chain |
| Encryption key error | Ensure ENCRYPTION_KEY in .env matches the one used when keys were added. Changing it makes existing keys undecryptable |
| Docker port not reachable from LAN | Start with `HOST_BIND=0.0.0.0 docker compose up -d` to publish on all interfaces |
| Claude Code refuses to start | Use `ANTHROPIC_AUTH_TOKEN` not `ANTHROPIC_API_KEY` -- Claude Code treats the latter as a conflicting credential |

---

## Conclusion

FreeLLMAPI turns the fragmented landscape of free LLM tiers into a single, reliable, OpenAI-compatible endpoint. With 16+ providers contributing roughly 1.7 billion tokens per month, smart routing that automatically falls over on errors, per-key rate limiting that keeps you under every cap, and AES-256-GCM encryption for your keys, it is the easiest way to access production-quality LLM inference at zero cost. The self-hosted architecture means your prompts and completions never leave your machine -- the proxy only talks to upstream providers when routing requests, and the catalog server never sees your data.

Whether you are prototyping an AI feature, running Claude Code on a budget, or just experimenting with different models, FreeLLMAPI gives you a single endpoint that just works. Clone the repo, add your free-tier keys, and start making requests in under five minutes.

---

## Links

- [GitHub Repository](https://github.com/tashfeenahmed/freellmapi)
- [Live Model Catalog](https://freellmapi.co)
- [Docker Image](https://github.com/tashfeenahmed/freellmapi/pkgs/container/freellmapi)
- [Desktop App Downloads](https://github.com/tashfeenahmed/freellmapi/releases/latest)
- [Contributing Guide](https://github.com/tashfeenahmed/freellmapi/blob/main/CONTRIBUTING.md)

---

## Related Posts

- [Harbor: The Open Source Container Registry](/Harbor-The-Open-Source-Container-Registry/)
- [LiteRT-LM: On-Device AI for Everyone](/LiteRT-LM-On-Device-AI-for-Everyone/)
- [Claude Code: AI-Powered Development in Your Terminal](/Claude-Code-AI-Powered-Development-in-Your-Terminal/)