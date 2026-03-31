---
layout: post
title: "PS Smart Agent - Ollama Local and Remote Models Guide"
date: 2026-03-22
categories: [AI, VS Code, Tutorial, Ollama]
featured-img: ai-coding-frameworks/ai-coding-frameworks
description: "Learn how to use local and remote Ollama models with PS Smart Agent, including tool support detection for agentic coding workflows."
keywords:
- PS Smart Agent
- Ollama
- local LLM
- remote Ollama
- tool support
- agentic coding
- AI models
---

## Using Ollama Local and Remote Models with Tool Support

PS Smart Agent v3.0.2 introduces enhanced Ollama support with automatic tool detection, remote server connectivity, and a clear visual indication of which models support agentic coding workflows.

## Why Tool Support Matters

For agentic coding workflows, the AI model needs to be able to use tools like:
- **Read files** - Access your codebase
- **Write files** - Create and modify code
- **Execute commands** - Run terminal commands
- **Search code** - Find relevant code patterns

Not all Ollama models support these tool capabilities. PS Smart Agent now automatically detects and displays this information.

## Local Ollama Setup

### 1. Install Ollama

```bash
# On macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows, download from https://ollama.ai/download
```

### 2. Pull a Model with Tool Support

```bash
# Models known to support tools
ollama pull llama3.2
ollama pull qwen2.5
ollama pull mistral

# Check model capabilities
ollama show llama3.2 --modelfile
```

### 3. Configure PS Smart Agent

1. Open VS Code Settings (`Ctrl+,`)
2. Search for "PS Smart Agent"
3. Select **Ollama** as the API Provider
4. The default URL `http://localhost:11434` works for local installations
5. Click **Test Connection** to verify connectivity
6. Select a model with tool support (shown with green "tools" badge)

## Remote Ollama Server

### Setting Up a Remote Server

If you have Ollama running on another machine (e.g., a powerful GPU server), you can connect to it remotely.

#### On the Server Machine:

```bash
# Start Ollama with host binding
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Or set environment variable permanently
export OLLAMA_HOST=0.0.0.0:11434
```

#### On Your Development Machine:

1. Open PS Smart Agent settings
2. Select **Ollama** as the API Provider
3. Enter the remote URL: `http://YOUR_SERVER_IP:11434`
   - Example: `http://192.168.1.100:11434`
4. Click **Test Connection** to verify
5. Models will load automatically from the remote server

### Network Requirements

- Both machines must be on the same network (or have proper routing)
- Port 11434 must be accessible
- Firewall rules may need adjustment

```bash
# Check if remote Ollama is accessible
curl http://192.168.1.100:11434/api/tags
```

## Tool Support Indicators

When you select Ollama as your provider, PS Smart Agent shows clear visual indicators:

### Models WITH Tool Support
- ✅ **Green "tools" badge**
- ✅ **Selectable** - Click to use the model
- ✅ **Full agentic capabilities** - Can read, write, execute

### Models WITHOUT Tool Support
- ⚠️ **Red "no tools" badge**
- ⚠️ **Disabled (grayed out)** - Cannot be selected
- ⚠️ **Chat only** - Can answer questions but cannot modify code

### Models Known to Support Tools

| Model | Tool Support | Notes |
|-------|-------------|-------|
| llama3.2 | ✅ Yes | Recommended for most tasks |
| qwen2.5 | ✅ Yes | Good for code generation |
| mistral | ✅ Yes | Fast and capable |
| codellama | ✅ Yes | Specialized for code |
| deepseek-coder | ✅ Yes | Excellent for coding tasks |
| llama2 | ❌ No | Chat only |
| gemma | ❌ No | Chat only |
| phi3 | ❌ No | Chat only |

## Test Connection Feature

The **Test Connection** button helps diagnose connectivity issues:

### Success States
- ✅ **Connected** - Server is reachable
- ✅ **Models loaded** - Available models are displayed

### Error States
- ❌ **Connection refused** - Ollama not running or wrong URL
- ❌ **No models found** - Server running but no models installed
- ❌ **Timeout** - Network issues or firewall blocking

## Troubleshooting

### Models Not Showing

1. **Check Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **For remote servers, verify connectivity:**
   ```bash
   curl http://YOUR_SERVER_IP:11434/api/tags
   ```

3. **Check firewall settings:**
   ```bash
   # Linux
   sudo ufw allow 11434
   
   # Windows - Add firewall rule for port 11434
   ```

### Connection Errors

| Error | Solution |
|-------|----------|
| Connection refused | Start Ollama: `ollama serve` |
| No models found | Pull a model: `ollama pull llama3.2` |
| Timeout | Check firewall and network |
| Wrong URL | Verify URL format: `http://IP:PORT` |

### Model Tool Support Not Detected

If a model you know supports tools shows "no tools":
1. Update Ollama to the latest version
2. Re-pull the model: `ollama pull MODEL_NAME`
3. Restart Ollama: `ollama serve`

## Best Practices

### For Local Development
- Use models with tool support for coding tasks
- Keep frequently used models pulled locally
- Monitor GPU memory usage with large models

### For Remote Servers
- Use a wired connection for stability
- Consider network latency when selecting models
- Run multiple models on a powerful server, access from lightweight machines

### Model Selection Tips
- **Coding tasks**: Use models with tool support (green badge)
- **Quick questions**: Any model works
- **Large codebases**: Prefer models with larger context windows

## Example Workflow

1. **Start Ollama server** (local or remote)
2. **Open PS Smart Agent** settings
3. **Select Ollama** provider
4. **Enter URL** (localhost or remote IP)
5. **Click Test Connection**
6. **Select a model** with tool support (green badge)
7. **Start coding** with full agentic capabilities!

## Conclusion

PS Smart Agent's Ollama integration makes it easy to use both local and remote models for agentic coding. The tool support indicators help you choose the right model for the task, ensuring you have full capabilities when you need them.

For more information, visit [pyshine.com](https://pyshine.com) or check the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=PyShine.smart-agent).
