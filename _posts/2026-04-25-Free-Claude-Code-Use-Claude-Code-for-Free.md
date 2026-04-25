---
layout: post
title: "Free Claude Code: Use Claude Code CLI and VSCode for Free with NVIDIA NIM, OpenRouter, and Local Models"
description: "Learn how Free Claude Code lets you use Claude Code CLI and VSCode for free by routing Anthropic API calls to NVIDIA NIM, OpenRouter, DeepSeek, LM Studio, or llama.cpp. No Anthropic API key required."
date: 2026-04-25
header-img: "img/post-bg.jpg"
permalink: /Free-Claude-Code-Use-Claude-Code-for-Free/
featured-img: ai-coding-frameworks/ai-coding-frameworks
categories: [AI Tools, Developer Tools, Open Source]
tags: [Free Claude Code, Claude Code, NVIDIA NIM, OpenRouter, DeepSeek, LM Studio, llama.cpp, AI coding assistant, free Claude Code alternative, proxy server, Discord bot, local LLM]
keywords: "how to use Claude Code for free, Free Claude Code setup tutorial, Claude Code without Anthropic API key, Free Claude Code vs official Claude Code, NVIDIA NIM free tier Claude Code, OpenRouter free models Claude Code, run Claude Code locally with LM Studio, llama.cpp Claude Code proxy setup, Free Claude Code Discord bot tutorial, free AI coding assistant alternatives"
author: "PyShine"
---

# Free Claude Code: Use Claude Code CLI and VSCode for Free with NVIDIA NIM, OpenRouter, and Local Models

Free Claude Code is a lightweight proxy that routes Claude Code's Anthropic API calls to free or low-cost alternatives including NVIDIA NIM (40 requests per minute free), OpenRouter (hundreds of free and paid models), DeepSeek (direct API), LM Studio (fully local), and llama.cpp (local inference). With 9,833+ stars and growing, this open-source project enables developers to use Claude Code's powerful CLI and VSCode extension without requiring an Anthropic API key or paid subscription.

![Free Claude Code Architecture](/assets/img/diagrams/free-claude-code/free-claude-code-architecture.svg)

### Understanding the Free Claude Code Architecture

The architecture diagram above shows how Free Claude Code acts as a transparent proxy between Claude Code and various LLM providers. Let's examine each component:

**Component 1: Claude Code Client**
The Claude Code CLI or VSCode extension sends standard Anthropic API requests in Server-Sent Events (SSE) format. The client believes it is communicating directly with Anthropic's servers, but the requests are intercepted by the proxy.

**Component 2: Free Claude Code Proxy**
The proxy server runs on localhost (default port 8082) and implements Claude-compatible API endpoints including GET /v1/models, POST /v1/messages, POST /v1/messages/count_tokens, plus HEAD and OPTIONS support for common probe endpoints. The proxy handles request detection, model routing, format translation, and response streaming.

**Component 3: LLM Providers**
The proxy supports five provider backends:
- **NVIDIA NIM**: 40 requests per minute free tier, recommended for daily use
- **OpenRouter**: Access to hundreds of models including free tiers from various providers
- **DeepSeek**: Direct API access to DeepSeek chat and reasoner models
- **LM Studio**: Fully local inference with no API key required
- **llama.cpp**: Lightweight local inference engine via llama-server

**Data Flow:**
Claude Code sends Anthropic-format requests to the proxy. The proxy detects trivial requests (quota probes, title generation, prefix detection, suggestions, filepath extraction) and responds locally without consuming API quota. Non-trivial requests are routed to the appropriate provider based on model mapping (Opus, Sonnet, Haiku, or fallback). The provider response is translated back to Anthropic format and streamed to Claude Code.

## Request Flow: How Requests Are Processed

![Free Claude Code Request Flow](/assets/img/diagrams/free-claude-code/free-claude-code-request-flow.svg)

### Step 1: Request Detection
When Claude Code sends a request, the proxy first checks if it is a trivial request that can be handled locally. Five categories of trivial requests are intercepted:
- Quota probes and health checks
- Title generation requests
- Prefix detection requests
- Suggestion mode requests
- Filepath extraction requests

These local responses save API quota and reduce latency for common operations.

### Step 2: Model Routing
For non-trivial requests, the proxy determines which model to use based on the request type:
- **Opus requests** route to MODEL_OPUS
- **Sonnet requests** route to MODEL_SONNET
- **Haiku requests** route to MODEL_HAIKU
- **Unrecognized models** fall back to MODEL

Each model variable uses the format `provider_prefix/model/name`, allowing different providers for different model tiers.

### Step 3: Format Translation
The proxy translates between Anthropic Messages format and OpenAI chat format depending on the provider:
- **Native Anthropic providers**: LM Studio and llama.cpp use native Anthropic Messages endpoints
- **OpenAI-compatible providers**: NVIDIA NIM and DeepSeek use shared OpenAI chat translation
- **OpenRouter**: Supports both formats depending on the selected model

### Step 4: Response Streaming
Provider responses are translated back to Anthropic SSE format and streamed to Claude Code in real-time. When `ENABLE_THINKING=true`, thinking tokens from `reasoning_content` fields and ` <think> ` tags are converted into native Claude thinking blocks.

## Provider Comparison

![Free Claude Code Providers](/assets/img/diagrams/free-claude-code/free-claude-code-providers.svg)

### NVIDIA NIM (Recommended)
NVIDIA NIM offers a generous free tier with 40 requests per minute, making it ideal for daily development work. Popular models include MiniMax-M2.5, Qwen3.5, GLM-5, Kimi-K2.5, and Step-3.5-Flash. No credit card required for the free tier.

### OpenRouter
OpenRouter provides access to hundreds of models from various providers, including free tiers. This is useful when you need model variety or fallback options. Free models include Arcee Trinity, Step-3.5-Flash, DeepSeek-R1, and GPT-OSS-120B.

### DeepSeek
DeepSeek offers direct API access to their chat and reasoner models. This is ideal if you specifically want DeepSeek's capabilities or prefer their pricing model over other providers.

### LM Studio (Fully Local)
LM Studio enables completely local inference with no API key required and no rate limits. Load models like LiquidAI LFM2, MiniMax-M2.5, GLM-4.7-Flash, or Qwen3.5 in GGUF format. Best for privacy-sensitive work or offline development.

### llama.cpp (Lightweight Local)
llama.cpp provides a lightweight local inference engine via `llama-server`. Ensure you have a tool-capable GGUF model loaded. This option is ideal for resource-constrained environments or when you want minimal overhead.

## Key Features

### Zero-Cost Operation
With NVIDIA NIM's 40 req/min free tier and OpenRouter's free models, you can use Claude Code for daily development without spending anything. Local options (LM Studio, llama.cpp) require only your own hardware.

### Drop-in Replacement
Free Claude Code requires only two environment variables:
- `ANTHROPIC_BASE_URL`: Point to the proxy (e.g., `http://localhost:8082`)
- `ANTHROPIC_AUTH_TOKEN`: Optional authentication token

No modifications to Claude Code CLI or VSCode extension are needed.

### Per-Model Mapping
Route Opus, Sonnet, and Haiku requests to different models and providers. Mix providers freely - for example, use NVIDIA NIM for Opus, OpenRouter for Sonnet, and LM Studio for Haiku.

### Thinking Token Support
The proxy parses `reasoning_content` fields and ` <think> ` tags from provider responses and converts them into native Claude thinking blocks when `ENABLE_THINKING=true`.

### Heuristic Tool Parser
Models that output tool calls as text are automatically parsed into structured tool use, enabling tool-capable models that don't natively support Anthropic's tool format.

### Smart Rate Limiting
Proactive rolling-window throttling plus reactive 429 exponential backoff with optional concurrency cap (`PROVIDER_MAX_CONCURRENCY`) prevents rate limit violations.

### Subagent Control
Task tool interception forces `run_in_background=False`, preventing runaway subagents from consuming excessive resources.

## Installation and Setup

### Prerequisites

1. Get an API key for your chosen provider:
   - **NVIDIA NIM**: [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
   - **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys)
   - **DeepSeek**: [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)
   - **LM Studio**: No API key needed - download from [lmstudio.ai](https://lmstudio.ai)
   - **llama.cpp**: No API key needed - run `llama-server` locally

2. Install Claude Code from [Anthropic's repository](https://github.com/anthropics/claude-code)

### Quick Install

```bash
# Install uv package manager
pip install uv

# Clone the repository
git clone https://github.com/Alishahryar1/free-claude-code.git
cd free-claude-code

# Copy environment template
cp .env.example .env
```

### Configure Your Provider

Edit `.env` with your chosen provider:

**NVIDIA NIM (recommended):**
```dotenv
NVIDIA_NIM_API_KEY="nvapi-your-key-here"
MODEL="nvidia_nim/z-ai/glm4.7"
ENABLE_THINKING=true
```

**OpenRouter:**
```dotenv
OPENROUTER_API_KEY="sk-or-your-key-here"
MODEL_OPUS="open_router/deepseek/deepseek-r1-0528:free"
MODEL_SONNET="open_router/openai/gpt-oss-120b:free"
MODEL_HAIKU="open_router/stepfun/step-3.5-flash:free"
MODEL="open_router/stepfun/step-3.5-flash:free"
```

**LM Studio (local):**
```dotenv
MODEL_OPUS="lmstudio/unsloth/MiniMax-M2.5-GGUF"
MODEL_SONNET="lmstudio/unsloth/Qwen3.5-35B-A3B-GGUF"
MODEL_HAIKU="lmstudio/unsloth/GLM-4.7-Flash-GGUF"
MODEL="lmstudio/unsloth/GLM-4.7-Flash-GGUF"
```

**Mix providers:**
```dotenv
NVIDIA_NIM_API_KEY="nvapi-your-key-here"
OPENROUTER_API_KEY="sk-or-your-key-here"
MODEL_OPUS="nvidia_nim/moonshotai/kimi-k2.5"
MODEL_SONNET="open_router/deepseek/deepseek-r1-0528:free"
MODEL_HAIKU="lmstudio/unsloth/GLM-4.7-Flash-GGUF"
MODEL="nvidia_nim/z-ai/glm4.7"
```

### Run the Proxy

**Terminal 1 - Start the proxy server:**
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2 - Run Claude Code:**

PowerShell:
```powershell
$env:ANTHROPIC_AUTH_TOKEN="freecc"; $env:ANTHROPIC_BASE_URL="http://localhost:8082"; claude
```

Bash:
```bash
ANTHROPIC_AUTH_TOKEN="freecc" ANTHROPIC_BASE_URL="http://localhost:8082" claude
```

### VSCode Extension Setup

1. Start the proxy server
2. Open VSCode Settings (Ctrl + ,) and search for `claude-code.environmentVariables`
3. Click **Edit in settings.json** and add:

```json
"claudeCode.environmentVariables": [
  { "name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082" },
  { "name": "ANTHROPIC_AUTH_TOKEN", "value": "freecc" }
]
```

4. Reload extensions
5. If you see the login screen, click **Anthropic Console**, then authorize. The extension will start working.

## Discord and Telegram Bot

![Free Claude Code Discord Bot](/assets/img/diagrams/free-claude-code/free-claude-code-discord-bot.svg)

Free Claude Code includes a messaging platform integration that enables remote autonomous coding via Discord or Telegram.

### Capabilities

- **Tree-based message threading**: Reply to a message to fork the conversation
- **Session persistence**: Sessions survive server restarts
- **Live streaming**: Real-time thinking tokens, tool calls, and results
- **Unlimited concurrent sessions**: Controlled by `PROVIDER_MAX_CONCURRENCY`
- **Voice notes**: Send voice messages that are transcribed and processed as prompts
- **Commands**: `/stop` (cancel task), `/clear` (reset sessions), `/stats`

### Discord Setup

1. Create a bot at the [Discord Developer Portal](https://discord.com/developers/applications)
2. Enable **Message Content Intent** under Bot settings
3. Add to `.env`:

```dotenv
MESSAGING_PLATFORM="discord"
DISCORD_BOT_TOKEN="your_discord_bot_token"
ALLOWED_DISCORD_CHANNELS="123456789,987654321"
CLAUDE_WORKSPACE="./agent_workspace"
ALLOWED_DIR="C:/Users/yourname/projects"
```

4. Start the server and invite the bot with OAuth2 URL Generator (scopes: `bot`, permissions: Read Messages, Send Messages, Manage Messages, Read Message History)

### Voice Notes

Voice messages are transcribed using Hugging Face Whisper (default, free and offline) or NVIDIA NIM. Install voice extras:

```bash
# Local Whisper
uv sync --extra voice_local

# NVIDIA NIM voice
uv sync --extra voice
```

## Configuration Reference

### Core Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL` | Fallback model (`provider/model/name`) | `nvidia_nim/z-ai/glm4.7` |
| `MODEL_OPUS` | Model for Claude Opus requests | empty (falls back to MODEL) |
| `MODEL_SONNET` | Model for Claude Sonnet requests | empty (falls back to MODEL) |
| `MODEL_HAIKU` | Model for Claude Haiku requests | empty (falls back to MODEL) |
| `ENABLE_THINKING` | Enable reasoning/thinking blocks | `true` |

### Rate Limiting

| Variable | Description | Default |
|----------|-------------|---------|
| `PROVIDER_RATE_LIMIT` | Requests per window | `40` |
| `PROVIDER_RATE_WINDOW` | Window in seconds | `60` |
| `PROVIDER_MAX_CONCURRENCY` | Max simultaneous streams | `5` |

### Request Optimization (enabled by default)

| Variable | Description |
|----------|-------------|
| `FAST_PREFIX_DETECTION` | Enable fast prefix detection |
| `ENABLE_NETWORK_PROBE_MOCK` | Mock network probe requests |
| `ENABLE_TITLE_GENERATION_SKIP` | Skip title generation |
| `ENABLE_SUGGESTION_MODE_SKIP` | Skip suggestion mode |
| `ENABLE_FILEPATH_EXTRACTION_MOCK` | Mock filepath extraction |

## Extending Free Claude Code

### Adding a New Provider

Extend `OpenAIChatTransport` for OpenAI-compatible providers:

```python
from providers.openai_compat import OpenAIChatTransport
from providers.base import ProviderConfig

class MyProvider(OpenAIChatTransport):
    def __init__(self, config: ProviderConfig):
        super().__init__(config, provider_name="MYPROVIDER",
                         base_url="https://api.example.com/v1", api_key=config.api_key)
```

### Adding a Messaging Platform

Extend `MessagingPlatform` and implement:
- `start()`: Initialize the platform connection
- `stop()`: Clean up resources
- `send_message()`: Send a message to a channel
- `edit_message()`: Edit an existing message
- `on_message()`: Handle incoming messages

## Project Structure

```
free-claude-code/
├── server.py              # Entry point
├── api/                   # FastAPI routes, model routing, optimizations
├── core/                  # Anthropic protocol helpers, SSE, parsers
├── providers/             # Provider registry, transports
├── messaging/             # Discord/Telegram bots, voice, sessions
├── config/                # Settings, logging
├── cli/                   # CLI session management
└── tests/                 # Pytest test suite
```

## Conclusion

Free Claude Code democratizes access to Claude Code's powerful interface by enabling free and local alternatives to Anthropic's API. Whether you choose NVIDIA NIM's generous free tier, OpenRouter's model variety, DeepSeek's direct API, or fully local inference with LM Studio or llama.cpp, you can enjoy Claude Code's agentic coding capabilities without subscription costs. The Discord/Telegram bot integration extends this accessibility to remote and collaborative workflows, making Free Claude Code a versatile addition to any developer's toolkit.

## Links

- [Free Claude Code GitHub Repository](https://github.com/Alishahryar1/free-claude-code)
- [NVIDIA NIM](https://build.nvidia.com/explore/discover)
- [OpenRouter Models](https://openrouter.ai/models)
- [LM Studio](https://lmstudio.ai)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Claude Code Official Repository](https://github.com/anthropics/claude-code)

## Related Posts

- [Claude Code Best Practices](/Claude-Code-Best-Practices/)
- [Claude HowTo: Mastering Claude Code](/Claude-HowTo-Mastering-Claude-Code/)
- [RTK: Reduce LLM Token Consumption by 60-90%](/RTK-Rust-Token-Killer-LLM-Optimization/)
