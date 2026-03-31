---
layout: post
title: "PS Smart Agent - Using Modes for Different Tasks"
date: 2026-03-22
categories: [AI, VS Code, Tutorial]
featured-img: ai-coding-frameworks/ai-coding-frameworks
description: "Learn how to use different modes in PS Smart Agent for coding, architecture planning, debugging, and more."
keywords:
- PS Smart Agent
- modes
- code mode
- architect mode
- debug mode
- custom modes
---

## Using Modes in PS Smart Agent

PS Smart Agent offers specialized modes for different development tasks. Each mode is optimized for specific workflows, helping you get better results.

## Available Modes

### Code Mode
**Best for:** Everyday coding, file operations, refactoring

Code mode is your go-to for daily development tasks:
- Writing and editing code
- Creating new files
- Refactoring existing code
- Adding features
- Fixing bugs

### Architect Mode
**Best for:** Planning, system design, documentation

Architect mode thinks before acting:
- Designing system architecture
- Planning migrations
- Creating technical specifications
- Documenting APIs
- Code reviews

### Ask Mode
**Best for:** Quick questions, explanations, learning

Ask mode provides fast answers:
- Code explanations
- Documentation lookups
- Quick questions
- Learning new concepts
- Best practices advice

### Debug Mode
**Best for:** Troubleshooting, error analysis

Debug mode helps you find and fix issues:
- Analyzing error messages
- Tracing bugs
- Adding logging
- Performance analysis
- Memory leak detection

### Custom Modes
**Best for:** Specialized workflows

Create your own modes for specific tasks:
- Test-driven development
- Documentation writing
- Security auditing
- Performance optimization

## Switching Modes

1. Click the mode dropdown in the chat input area
2. Select your desired mode
3. The mode indicator shows your current selection

## Mode-Specific Behavior

Each mode has different default behaviors:

| Behavior | Code | Architect | Ask | Debug |
|----------|------|-----------|-----|-------|
| Reads files | ✓ | ✓ | ✓ | ✓ |
| Edits files | ✓ | Plan only | ✗ | ✓ |
| Runs commands | ✓ | ✗ | ✗ | ✓ |
| Creates checkpoints | ✓ | ✗ | ✗ | ✓ |
| Deep analysis | ✓ | ✓ | ✓ | ✓ |

## Creating Custom Modes

1. Open Settings > Modes
2. Click "Create Custom Mode"
3. Define:
   - Name and description
   - System prompt
   - Allowed tools
   - Auto-approve settings

### Example Custom Mode: TDD

```yaml
name: TDD
description: Test-Driven Development mode
prompt: |
  Always write tests before implementation.
  Follow the red-green-refactor cycle.
tools:
  - read_file
  - write_to_file
  - execute_command
autoApprove:
  - read_operations
```

## Tips for Using Modes

1. **Start with Ask mode** for understanding code
2. **Switch to Code mode** for making changes
3. **Use Architect mode** for planning large features
4. **Use Debug mode** when something breaks

---

*Learn more at [pyshine.com](https://pyshine.com)*
