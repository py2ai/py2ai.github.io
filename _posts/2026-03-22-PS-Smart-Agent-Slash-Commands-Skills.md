---
layout: post
title: "PS Smart Agent - Slash Commands and Skills"
date: 2026-03-22
categories: [AI, VS Code, Tutorial]
featured-img: ps-smart-agent/slash-commands
description: "Learn how to use slash commands and skills in PS Smart Agent to automate common tasks and extend functionality."
keywords:
- PS Smart Agent
- slash commands
- skills
- automation
- shortcuts
---

# Slash Commands and Skills in PS Smart Agent

Slash commands and skills provide quick shortcuts for common tasks. This guide shows you how to use and create them.

## Using Slash Commands

Slash commands are shortcuts that trigger specific actions.

### Built-in Commands

| Command | Description |
|---------|-------------|
| `/explain` | Explain selected code |
| `/fix` | Fix issues in code |
| `/improve` | Improve code quality |
| `/test` | Generate tests |
| `/doc` | Add documentation |

### How to Use

1. Type `/` in the chat input
2. Select a command from the list
3. Provide additional context if needed

## Creating Custom Slash Commands

### 1. Open Settings

Settings > Slash Commands > Add New

### 2. Define the Command

```yaml
name: /review
description: Review code for best practices
prompt: |
  Review the following code for:
  - Best practices
  - Potential bugs
  - Performance issues
  - Security concerns
  
  Provide actionable suggestions.
```

### Example Commands

**Code Review**
```yaml
name: /review
description: Comprehensive code review
prompt: |
  Perform a thorough code review focusing on:
  1. Code quality and readability
  2. Design patterns
  3. Error handling
  4. Test coverage
```

**Refactor**
```yaml
name: /refactor
description: Refactor for cleaner code
prompt: |
  Refactor this code to:
  - Improve readability
  - Apply design patterns
  - Reduce complexity
  - Follow DRY principles
```

## Skills

Skills are reusable workflows that combine multiple tools.

### Built-in Skills

| Skill | Description |
|-------|-------------|
| Git Operations | Commit, branch, merge |
| File Operations | Create, move, delete |
| Testing | Run tests, generate coverage |
| Deployment | Build and deploy |

### Creating Custom Skills

1. Settings > Skills > Create New
2. Define the workflow:

```yaml
name: Deploy to Staging
description: Deploy current branch to staging
steps:
  - action: run_tests
    command: npm test
  - action: build
    command: npm run build
  - action: deploy
    command: ./deploy.sh staging
```

## Combining Commands and Skills

Use commands within skills for powerful workflows:

```yaml
name: Full PR Review
steps:
  - command: /review
  - command: /test
  - action: create_pr
```

## Best Practices

1. **Keep commands focused** - One task per command
2. **Use descriptive names** - `/security-check` vs `/sc`
3. **Document your commands** - Help others understand usage
4. **Share with team** - Export and import commands

---

*Learn more at [pyshine.com](https://pyshine.com)*
