---
layout: post
title: "Claude Code MCP Guide - Connect to External Tools and APIs"
date: 2026-03-31
categories: [AI, Tools, Tutorial, Claude Code]
featured-img: ai-coding-frameworks/ai-coding-frameworks
description: "Learn to use MCP (Model Context Protocol) with Claude Code. Connect to GitHub, databases, Slack, and other external services."
keywords:
- Claude Code
- MCP
- Model Context Protocol
- external tools
- API integration
---

# Claude Code MCP Guide - Connect to External Tools and APIs

MCP (Model Context Protocol) is a standardized way for Claude to access external tools, APIs, and real-time data sources. It extends Claude's capabilities beyond your local files.

## What is MCP?

MCP provides:

- **Real-time access** to external services
- **Live data synchronization**
- **Extensible architecture**
- **Secure authentication**
- **Tool-based interactions**

## MCP Architecture

```
Claude Code → MCP Server → External Service
     ↑            ↓
     └────────────┘
```

Claude communicates with MCP servers, which in turn interact with external services like GitHub, databases, Slack, and more.

## Installation Methods

### HTTP Transport (Recommended)

```bash
# Basic HTTP connection
claude mcp add --transport http notion https://mcp.notion.com/mcp

# With authentication header
claude mcp add --transport http api https://api.example.com/mcp \
  --header "Authorization: Bearer your-token"
```

### Stdio Transport (Local)

```bash
# Local Node.js server
claude mcp add --transport stdio myserver -- npx @myorg/mcp-server

# With environment variables
claude mcp add --transport stdio myserver --env KEY=value -- npx server
```

### WebSocket Transport

```bash
claude mcp add --transport ws realtime wss://example.com/mcp
```

## OAuth 2.0 Authentication

Claude Code supports OAuth 2.0 for MCP servers that require it:

```bash
# Interactive OAuth flow
claude mcp add --transport http my-service https://my-service.example.com/mcp

# Pre-configured OAuth
claude mcp add --transport http my-service https://my-service.example.com/mcp \
  --client-id "your-client-id" \
  --client-secret "your-client-secret"
```

## Popular MCP Servers

### GitHub MCP

```bash
claude mcp add --transport http github https://mcp.github.com/mcp
```

**Available Tools:**
- Create/update issues
- Create pull requests
- Review code
- Manage repositories

### Filesystem MCP

```bash
claude mcp add --transport stdio filesystem -- npx @anthropic/mcp-server-filesystem
```

**Available Tools:**
- Read/write files
- Create directories
- Search files
- Manage permissions

### Database MCP

```bash
claude mcp add --transport stdio postgres -- npx @anthropic/mcp-server-postgres \
  --env DATABASE_URL=postgresql://...
```

**Available Tools:**
- Execute queries
- Inspect schema
- Manage tables

### Slack MCP

```bash
claude mcp add --transport http slack https://mcp.slack.com/mcp
```

**Available Tools:**
- Send messages
- Read channels
- Manage threads

## MCP Configuration File

Create `.mcp.json` in your project:

```json
{
  "mcpServers": {
    "github": {
      "type": "http",
      "url": "https://mcp.github.com/mcp"
    },
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["@anthropic/mcp-server-filesystem"]
    },
    "postgres": {
      "type": "stdio",
      "command": "npx",
      "args": ["@anthropic/mcp-server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost/db"
      }
    }
  }
}
```

## Using MCP Tools

### Via Commands

MCP tools appear as slash commands:

```bash
/mcp__github__list_issues
/mcp__github__create_pr
/mcp__postgres__query "SELECT * FROM users"
```

### In Conversation

```
> List all open issues in the repository
> Create a pull request with these changes
> Send a message to #dev-team about the deployment
```

## MCP Prompts as Commands

MCP servers can expose prompts as slash commands:

```
/mcp__<server-name>__<prompt-name> [arguments]
```

**Examples:**
```bash
/mcp__github__list_prs
/mcp__github__pr_review 456
/mcp__jira__create_issue "Bug title" high
```

## Managing MCP Servers

### List Servers

```bash
/mcp
```

### Remove Server

```bash
claude mcp remove server-name
```

### Update Server

```bash
claude mcp update server-name --url https://new-url/mcp
```

## Security Considerations

1. **Review permissions**: Understand what each MCP server can access
2. **Use OAuth**: Prefer OAuth over static tokens when possible
3. **Limit scope**: Grant minimum necessary permissions
4. **Audit regularly**: Review connected servers periodically

## MCP Permission Syntax

Control MCP server access in permissions:

- `mcp__github` - Access entire GitHub MCP server
- `mcp__github__*` - Wildcard access to all tools
- `mcp__github__get_issue` - Specific tool access

## Troubleshooting

### Connection Failed

```bash
# Check server status
claude mcp list

# Test connection
claude mcp test server-name
```

### Authentication Issues

```bash
# Re-authenticate
claude mcp auth server-name

# Clear stored credentials
claude mcp clear-auth server-name
```

## Best Practices

1. **Start with HTTP**: Prefer HTTP transport for simplicity
2. **Use OAuth**: More secure than static tokens
3. **Document servers**: Keep track of what each server does
4. **Regular audits**: Review and remove unused servers
5. **Test first**: Verify MCP servers in isolation

## Related Guides

- [Claude Code Complete Guide](/claude-code-complete-guide/)
- [Claude Code Subagents Guide](/claude-code-subagents-guide/)
- [Claude Code Skills Guide](/claude-code-skills-guide/)
