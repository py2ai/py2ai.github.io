---
layout: post
title: "PS Smart Agent - Codebase Indexing for Fast Search"
date: 2026-03-22
categories: [AI, VS Code, Tutorial]
featured-img: ai-coding-frameworks/ai-coding-frameworks
description: "Learn how to use PS Smart Agent's codebase indexing feature for fast semantic search across your project."
keywords:
- PS Smart Agent
- codebase indexing
- semantic search
- code search
- vector search
---

## Codebase Indexing in PS Smart Agent

Codebase indexing enables fast semantic search across your project. Instead of searching by text, you can search by meaning.

## What is Codebase Indexing?

Codebase indexing:
- Creates embeddings for your code
- Enables semantic search
- Helps AI understand your codebase
- Speeds up context gathering

## Enabling Codebase Indexing

### 1. Open Settings

1. Click the settings icon
2. Navigate to "Codebase Indexing"

### 2. Configure Indexing

```json
{
  "codeIndexing.enabled": true,
  "codeIndexing.maxFiles": 10000,
  "codeIndexing.embeddingBatchSize": 60
}
```

### 3. Start Indexing

Click "Index Codebase" to begin the process.

## Using Indexed Search

### In Chat

Use `@` to search your codebase:
```
@ Where is the authentication logic?
```

### Direct Search

1. Click the search icon
2. Enter your query
3. Results show relevant code snippets

## Indexing Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `maxFiles` | Maximum files to index | 10,000 |
| `embeddingBatchSize` | Batch size for embeddings | 60 |
| `excludePatterns` | Files to exclude | `node_modules`, `.git` |

## Best Practices

### 1. Exclude Large Directories

```json
{
  "codeIndexing.excludePatterns": [
    "node_modules/**",
    "dist/**",
    "build/**",
    "*.min.js",
    "*.lock"
  ]
}
```

### 2. Use Meaningful Queries

Instead of: `function name`
Try: `handles user login validation`

### 3. Keep Index Updated

- Re-index after major changes
- Use auto-reindex on file changes

## Performance Tips

1. **Limit file count** - Don't index generated files
2. **Use appropriate batch size** - Lower for rate-limited APIs
3. **Clear old indexes** - Remove outdated embeddings

## Troubleshooting

### Index Not Building
- Check API key has embedding access
- Verify file permissions
- Check available disk space

### Slow Search
- Reduce max files
- Clear and rebuild index
- Check network connection

---

*Learn more at [pyshine.com](https://pyshine.com)*
