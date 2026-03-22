---
layout: post
title: "PS Smart Agent - Getting Started with API Providers"
date: 2026-03-22
categories: [AI, VS Code, Tutorial]
featured-img: ps-smart-agent/api-providers
description: "Learn how to connect PS Smart Agent to various AI providers including Ollama, OpenAI, Anthropic, DeepSeek, and more."
keywords:
- PS Smart Agent
- API provider
- Ollama
- OpenAI
- Anthropic
- DeepSeek
- configuration
---

# Getting Started with API Providers

PS Smart Agent supports multiple AI providers, giving you flexibility to choose the best model for your needs. This guide will help you connect to your preferred provider.

## Supported Providers

| Provider | Type | Best For |
|----------|------|----------|
| **Ollama** | Local | Privacy, offline use, free |
| **OpenAI** | Cloud | GPT-4, GPT-3.5 |
| **Anthropic** | Cloud | Claude 3.5 Sonnet, Claude 3 Opus |
| **DeepSeek** | Cloud | Cost-effective coding |
| **OpenRouter** | Cloud | 100+ models via one API |
| **LM Studio** | Local | Local model hosting |
| **VS Code LM** | Built-in | Copilot integration |

## Setting Up Ollama (Local - Free)

Ollama allows you to run AI models completely locally on your machine.

### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows - Download from https://ollama.ai/download
```

### 2. Pull a Model

```bash
# Pull Llama 3.2
ollama pull llama3.2

# Pull Qwen 2.5
ollama pull qwen2.5

# Pull Mistral
ollama pull mistral
```

### 3. Configure PS Smart Agent

1. Open PS Smart Agent in VS Code
2. Click the settings icon
3. Select "Ollama" as provider
4. Enter base URL: `http://localhost:11434`
5. Select your model from the dropdown

## Setting Up OpenAI

1. Get your API key from [OpenAI Platform](https://platform.openai.com)
2. In PS Smart Agent settings, select "OpenAI"
3. Enter your API key
4. Select your preferred model (GPT-4, GPT-3.5-turbo, etc.)

## Setting Up Anthropic Claude

1. Get your API key from [Anthropic Console](https://console.anthropic.com)
2. In PS Smart Agent settings, select "Anthropic"
3. Enter your API key
4. Select Claude 3.5 Sonnet or Claude 3 Opus

## Setting Up DeepSeek

DeepSeek offers cost-effective models optimized for coding.

1. Get your API key from [DeepSeek Platform](https://platform.deepseek.com)
2. In PS Smart Agent settings, select "DeepSeek"
3. Enter your API key
4. Base URL: `https://api.deepseek.com/v1`

## Setting Up OpenRouter

OpenRouter provides access to 100+ models through a single API.

1. Get your API key from [OpenRouter](https://openrouter.ai)
2. In PS Smart Agent settings, select "OpenRouter"
3. Enter your API key
4. Select from available models

## Testing Your Connection

After configuring your provider:

1. Click "Test Connection" in settings
2. If successful, you'll see a green checkmark
3. If failed, check your API key and base URL

## Troubleshooting

### Connection Failed
- Verify your API key is correct
- Check the base URL includes `/v1` for OpenAI-compatible APIs
- Ensure you have available API credits

### Ollama Not Found
- Make sure Ollama is running: `ollama serve`
- Check the base URL is `http://localhost:11434`

### Model Not Available
- For Ollama: `ollama pull <model-name>`
- For cloud providers: Check your subscription tier

---

*Need help? Visit [pyshine.com](https://pyshine.com) for more tutorials.*
