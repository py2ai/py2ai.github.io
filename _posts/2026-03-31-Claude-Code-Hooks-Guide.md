---
layout: post
title: "Claude Code Hooks - Event-Driven Automation"
date: 2026-03-31
categories: [AI, Tools, Tutorial, Claude Code]
featured-img: ai-coding-frameworks/ai-coding-frameworks
description: "Learn to use Claude Code hooks for event-driven automation. Automate workflows with PreToolUse, PostToolUse, and Stop hooks."
keywords:
- Claude Code
- hooks
- automation
- event-driven
- workflow automation
---

# Claude Code Hooks - Event-Driven Automation

Hooks are event-driven scripts that run automatically before or after Claude performs actions. They enable powerful automation and safety checks.

## What Are Hooks?

Hooks allow you to:

- **Validate actions** before they happen
- **Log activities** for auditing
- **Transform outputs** automatically
- **Enforce policies** across your team
- **Integrate with external tools**

## Hook Types

| Hook | When It Runs | Use Case |
|------|-------------|----------|
| `PreToolUse` | Before tool execution | Validation, safety checks |
| `PostToolUse` | After tool execution | Logging, notifications |
| `Stop` | When Claude stops | Cleanup, summaries |

## Hook Configuration

### File Location

Hooks are configured in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/validate-bash.sh"
          }
        ]
      }
    ]
  }
}
```

### Hook Structure

```json
{
  "matcher": "ToolName",
  "hooks": [
    {
      "type": "command",
      "command": "/path/to/script.sh"
    }
  ]
}
```

## PreToolUse Hooks

Run before a tool is executed. Can block the action by returning an error.

### Example: Block Dangerous Commands

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/check-dangerous-commands.sh"
          }
        ]
      }
    ]
  }
}
```

**Script:** `scripts/check-dangerous-commands.sh`

```bash
#!/bin/bash

# Read the command from stdin
read -r command

# Block dangerous commands
if echo "$command" | grep -qE "rm -rf|DROP TABLE|DELETE FROM"; then
  echo "Error: Dangerous command blocked"
  exit 2
fi

# Allow the command
exit 0
```

### Example: Validate File Writes

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/validate-write.sh"
          }
        ]
      }
    ]
  }
}
```

## PostToolUse Hooks

Run after a tool completes. Useful for logging and notifications.

### Example: Log All Actions

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/log-action.sh"
          }
        ]
      }
    ]
  }
}
```

**Script:** `scripts/log-action.sh`

```bash
#!/bin/bash

# Read tool name and result from stdin
read -r tool_name
read -r result

# Log to file
echo "$(date): $tool_name - $result" >> ~/.claude/activity.log
```

### Example: Notify on Errors

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/notify-error.sh"
          }
        ]
      }
    ]
  }
}
```

## Stop Hooks

Run when Claude stops (completes or errors).

### Example: Generate Summary

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/generate-summary.sh"
          }
        ]
      }
    ]
  }
}
```

## Matcher Patterns

### Exact Match

```json
"matcher": "Bash"
```

### Wildcard

```json
"matcher": "*"
```

### Regex

```json
"matcher": "Bash(npm:*)"
```

### Multiple Tools

```json
"matcher": "Bash|Write|Edit"
```

## Hook Input/Output

### Input (stdin)

Hooks receive JSON input via stdin:

```json
{
  "tool": "Bash",
  "input": {
    "command": "npm test"
  },
  "result": {
    "output": "All tests passed",
    "exitCode": 0
  }
}
```

### Output (stdout)

- **Exit 0**: Allow the action
- **Exit 2**: Block with error message
- **Other**: Unexpected error

## Skill-Scoped Hooks

Define hooks within a skill:

```yaml
---
name: safe-deploy
description: Deploy with safety checks
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/pre-deploy-check.sh"
---

Deploy the application with safety checks...
```

## Practical Examples

### Example 1: Secret Detection

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/detect-secrets.sh"
          }
        ]
      }
    ]
  }
}
```

### Example 2: Auto-Format Code

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "prettier --write $FILE_PATH"
          }
        ]
      }
    ]
  }
}
```

### Example 3: Slack Notifications

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash(git push*)",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/notify-slack.sh"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

1. **Keep hooks fast**: They run synchronously
2. **Handle errors gracefully**: Don't break the workflow
3. **Log everything**: For debugging and auditing
4. **Test thoroughly**: Verify hooks work as expected
5. **Document clearly**: Explain what each hook does

## Debugging Hooks

### Enable Debug Logging

```bash
CLAUDE_DEBUG_HOOKS=1 claude
```

### Test Hook Script

```bash
# Test manually
echo '{"tool":"Bash","input":{"command":"test"}}' | ./scripts/my-hook.sh
echo $?
```

## Related Guides

- [Claude Code Complete Guide](/claude-code-complete-guide/)
- [Claude Code Skills Guide](/claude-code-skills-guide/)
- [Claude Code MCP Guide](/claude-code-mcp-guide/)
