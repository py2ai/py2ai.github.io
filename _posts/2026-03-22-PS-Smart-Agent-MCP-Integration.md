---
layout: post
title: "PS Smart Agent - MCP Integration Guide"
date: 2026-03-22
categories: [AI, VS Code, Tutorial]
featured-img: ai-coding-frameworks/ai-coding-frameworks
description: "Learn how to extend PS Smart Agent with MCP (Model Context Protocol) servers for database connections, API integrations, and custom tools."
keywords:
- PS Smart Agent
- MCP
- Model Context Protocol
- integration
- tools
- database
---

# MCP Integration in PS Smart Agent

The Model Context Protocol (MCP) allows you to extend PS Smart Agent with external tools, databases, and APIs. This guide shows you how to set up and use MCP servers.

## What is MCP?

MCP is a protocol that enables AI assistants to:
- Connect to external data sources
- Execute custom tools
- Access databases
- Integrate with APIs
- Extend functionality

## Setting Up MCP Servers

### 1. Open MCP Settings

1. Click the MCP icon in the toolbar
2. Or go to Settings > MCP

### 2. Add a Server

Click "Add Server" and configure:

```json
{
  "name": "my-server",
  "command": "node",
  "args": ["path/to/server.js"],
  "env": {
    "API_KEY": "your-key"
  }
}
```

## Popular MCP Servers

### Filesystem Server
Access files outside your workspace:

```json
{
  "name": "filesystem",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
}
```

### PostgreSQL Server
Connect to PostgreSQL databases:

```json
{
  "name": "postgres",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-postgres"],
  "env": {
    "POSTGRES_URL": "postgresql://user:pass@localhost/db"
  }
}
```

### GitHub Server
Interact with GitHub:

```json
{
  "name": "github",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_TOKEN": "your-token"
  }
}
```

### Brave Search Server
Web search capabilities:

```json
{
  "name": "brave-search",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-brave-search"],
  "env": {
    "BRAVE_API_KEY": "your-key"
  }
}
```

## Using MCP Tools

Once configured, MCP tools appear in PS Smart Agent:

1. Type `@` in the chat
2. Select the MCP tool
3. Provide the required parameters

## Creating Custom MCP Servers

### Basic Server Structure

```javascript
import { Server } from '@modelcontextprotocol/sdk';

const server = new Server({
  name: 'my-custom-server',
  version: '1.0.0'
});

// Add tools
server.addTool({
  name: 'my_tool',
  description: 'Does something useful',
  parameters: {
    type: 'object',
    properties: {
      input: { type: 'string' }
    }
  },
  handler: async (params) => {
    return { result: `Processed: ${params.input}` };
  }
});

server.start();
```

## Troubleshooting MCP

### Server Not Starting
- Check the command path
- Verify environment variables
- Check server logs in MCP view

### Tools Not Appearing
- Restart the MCP server
- Check server configuration
- Verify tool definitions

---

*Learn more at [pyshine.com](https://pyshine.com)*
