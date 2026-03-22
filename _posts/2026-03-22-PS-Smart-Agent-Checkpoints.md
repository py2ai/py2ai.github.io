---
layout: post
title: "PS Smart Agent - Checkpoints and Version Control"
date: 2026-03-22
categories: [AI, VS Code, Tutorial]
featured-img: ai-coding-frameworks/ai-coding-frameworks
description: "Learn how to use PS Smart Agent's checkpoint system to safely experiment with code changes and restore when needed."
keywords:
- PS Smart Agent
- checkpoints
- version control
- git
- restore
- safety
---

# Checkpoints in PS Smart Agent

Checkpoints provide a safety net for your coding sessions. They allow you to experiment freely and restore your code if something goes wrong.

## How Checkpoints Work

PS Smart Agent automatically creates checkpoints during task execution:
- Before making file changes
- At key decision points
- When you request them

## Viewing Checkpoints

1. Click the checkpoint icon in the chat
2. See a list of all checkpoints
3. Each checkpoint shows:
   - Timestamp
   - Files changed
   - Description

## Restoring from Checkpoints

### Restore Files Only
Reverts files to the checkpoint state:
1. Select a checkpoint
2. Click "Restore Files"
3. Confirm the restore

### Restore Files & Task
Reverts files and continues from that point:
1. Select a checkpoint
2. Click "Restore Files & Task"
3. PS Smart Agent continues from the checkpoint

## Viewing Diffs

Before restoring, you can see what changed:
1. Click on a checkpoint
2. Select "View Diff"
3. Review all file changes

## Manual Checkpoints

Create checkpoints manually:
1. Click the checkpoint icon
2. Select "Create Checkpoint"
3. Add a description

## Best Practices

1. **Create checkpoints before major changes**
2. **Review diffs before restoring**
3. **Use descriptive checkpoint names**
4. **Don't rely solely on checkpoints - use git too**

## Checkpoint vs Git

| Feature | Checkpoints | Git |
|---------|-------------|-----|
| Automatic | ✓ | ✗ |
| Granular | Per-task | Per-commit |
| Fast restore | ✓ | ✓ |
| Branching | ✗ | ✓ |
| Remote backup | ✗ | ✓ |

---

*Learn more at [pyshine.com](https://pyshine.com)*
