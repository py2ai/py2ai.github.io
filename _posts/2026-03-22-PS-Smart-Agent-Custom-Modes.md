---
layout: post
title: "PS Smart Agent - Custom Modes Guide"
date: 2026-03-22
categories: [AI, VS Code, Tutorial]
featured-img: ps-smart-agent/custom-modes
description: "Create custom modes in PS Smart Agent for specialized workflows like TDD, security auditing, and documentation writing."
keywords:
- PS Smart Agent
- custom modes
- TDD
- security
- documentation
- workflow
---

# Creating Custom Modes in PS Smart Agent

Custom modes allow you to tailor PS Smart Agent for specific workflows. This guide shows you how to create and use custom modes.

## Why Custom Modes?

Custom modes let you:
- Enforce specific workflows
- Limit available tools
- Set custom system prompts
- Create team standards

## Creating a Custom Mode

### 1. Open Mode Settings

1. Click the mode dropdown
2. Select "Create Custom Mode"

### 2. Configure the Mode

```yaml
name: Security Audit
description: Reviews code for security vulnerabilities
prompt: |
  You are a security expert. Analyze code for:
  - SQL injection vulnerabilities
  - XSS vulnerabilities
  - Authentication issues
  - Sensitive data exposure
  - Insecure configurations
  
  Always provide severity ratings and remediation steps.
tools:
  - read_file
  - search_files
  - list_files
autoApprove: []
```

## Example Custom Modes

### Test-Driven Development

```yaml
name: TDD
description: Enforces test-driven development workflow
prompt: |
  Follow TDD principles:
  1. Write a failing test first
  2. Write minimal code to pass
  3. Refactor while keeping tests green
  
  Never write implementation before tests.
tools:
  - read_file
  - write_to_file
  - execute_command
  - list_tests
autoApprove:
  - read_operations
```

### Documentation Writer

```yaml
name: Doc Writer
description: Generates comprehensive documentation
prompt: |
  Write clear, comprehensive documentation:
  - Use JSDoc/docstring format
  - Include examples
  - Document parameters and return values
  - Note edge cases
  
  Never modify code, only add comments.
tools:
  - read_file
  - write_to_file
autoApprove:
  - read_operations
```

### Performance Optimizer

```yaml
name: Performance
description: Analyzes and optimizes code performance
prompt: |
  Analyze code for performance issues:
  - Time complexity analysis
  - Memory usage
  - Database query optimization
  - Caching opportunities
  
  Provide before/after comparisons.
tools:
  - read_file
  - write_to_file
  - execute_command
autoApprove: []
```

## Sharing Custom Modes

Export your custom modes:
1. Settings > Modes > Export
2. Share the JSON file with your team

Import custom modes:
1. Settings > Modes > Import
2. Select the JSON file

## Mode Best Practices

1. **Be specific in prompts** - Clear instructions get better results
2. **Limit tools appropriately** - Don't give unnecessary access
3. **Test thoroughly** - Verify mode behavior before sharing
4. **Document your modes** - Help others understand the purpose

---

*Learn more at [pyshine.com](https://pyshine.com)*
