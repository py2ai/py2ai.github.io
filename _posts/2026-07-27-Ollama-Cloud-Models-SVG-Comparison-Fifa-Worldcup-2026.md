---
layout: post
title: "Which Ollama Cloud Model is Best? FIFA World Cup 2026 SVG Comparison (13 Models)"
description: "Compare 13 Ollama cloud models on a trending prompt: FIFA Football World Cup 2026. Find the best LLM for complex branded SVG scenes. You decide the winner."
date: 2026-07-27
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Fifa-Worldcup-2026/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - Ollama
  - SVG
  - LLM
  - Comparison
  - Benchmark
  - Best LLM
  - SVG generation
  - FIFA
  - World Cup
  - Football
  - Soccer
  - Sports
  - 2026
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama cloud model comparison, FIFA World Cup 2026 SVG, AI football drawing, LLM SVG benchmark, AI image generation comparison, deepseek vs glm vs qwen, which Ollama model is best, Ollama cloud models 2026, AI creative art, soccer SVG art, FIFA trophy SVG, AI art comparison, complex SVG scene, World Cup host nations, USA Canada Mexico SVG, sports SVG, tournament poster AI"
---

# Which Ollama Cloud Model is Best? FIFA World Cup 2026 SVG Comparison (13 Models)

The FIFA World Cup 2026 is the biggest sporting event of the year -- hosted across three nations (USA, Canada, and Mexico) for the first time in history. So we asked 13 Ollama cloud models to draw **"FIFA Football World Cup 2026"** as an SVG. This prompt tests whether models can combine text rendering (FIFA, World Cup, 2026), brand recognition (the iconic trophy), sports imagery (soccer ball), and national identity (flags) into a single cohesive poster.

The prompt was: `Make svg image of Fifa football worldcup 2026`

This is the eighth in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/), [duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/), [baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/), and [octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/).

**Why the FIFA World Cup 2026?** This prompt is uniquely challenging because it combines: (1) **Text rendering** -- models must render "FIFA", "World Cup", and "2026" as actual `<text>` elements, which many LLMs struggle with (text in SVG is harder than shapes), (2) **Brand recognition** -- the FIFA World Cup trophy has a very specific silhouette (a golden globe held by two athletes) that the model must recall from training data, (3) **Sports imagery** -- a soccer ball with its iconic black-and-white pentagon/hexagon pattern, (4) **National identity** -- the 2026 World Cup is hosted by USA, Canada, and Mexico, so flags and host nation references test the model's knowledge of current events, (5) **Poster composition** -- a World Cup poster needs a central focal point (trophy or ball), supporting text, and decorative elements (stars, flags, gradient backgrounds), (6) **Trending relevance** -- the 2026 World Cup is happening right now, so models trained on recent data should know the official slogan "We Are 26" and the host nations.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Branded SVG Posters

The FIFA World Cup 2026 prompt rewards different things than previous prompts. Here are the criteria to use:

- **Text accuracy**: Does the SVG render "FIFA", "World Cup", and "2026" as readable text? Text rendering in SVG is hard for LLMs because it requires `<text>` elements with proper positioning, fonts, and sizing. A model that draws a great shark may fail here because it cannot spell "FIFA".
- **Trophy recognition**: Does the SVG include a recognizable FIFA World Cup trophy? The trophy has a specific shape (golden globe on a base, held by two athletes). A model with good brand knowledge will reproduce this silhouette.
- **Soccer ball**: Does the SVG include a soccer ball with the iconic black-and-white pentagon/hexagon pattern? Or is it just a generic circle?
- **Host nations**: Does the SVG reference USA, Canada, and Mexico (the 2026 hosts)? This tests whether the model knows current events. Look for flags, country names, or colors.
- **Poster composition**: Is there a clear focal point (trophy, ball, or "2026" text)? Are supporting elements balanced around it? A good poster has visual hierarchy.
- **Color palette**: Does the SVG use a vibrant, celebratory palette (gold for the trophy, green for the field, national colors for flags)? Or is it dull and monochrome?
- **SVG code quality**: Does it use `<defs>`, `<symbol>`, `<use>`, and gradients? Better code is easier to tweak (e.g., to recolor for a different team or add a sponsor logo).

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with minimal post-processing (adding `width="100%" height="auto"` for responsive embedding and fixing XML errors so the SVG renders in browsers).

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `deepseek-v4-flash:cloud` | 11177 | 70 | 16 | Very high | Richest scene |
| 2 | `deepseek-v4-pro:cloud` | 7767 | 28 | 6 | High | Detailed |
| 3 | `gemma4:31b-cloud` | 1942 | 11 | 11 | Compact | Compact |
| 4 | `gemma4:cloud` | 2238 | 11 | 6 | Compact | Compact |
| 5 | `glm-5.1:cloud` | 21973 | 99 | 17 | Very high | Richest scene |
| 6 | `glm-5.2:cloud` | 5841 | 39 | 14 | High | Detailed |
| 7 | `gpt-oss:120b-cloud` | 2624 | 6 | 5 | Compact | Balanced |
| 8 | `kimi-k2.6:cloud` | 7048 | 19 | 8 | High | Detailed |
| 9 | `minimax-m2.7:cloud` | 4992 | 17 | 6 | Medium | Balanced |
| 10 | `minimax-m3:cloud` | 7812 | 58 | 17 | High | Detailed |
| 11 | `nemotron-3-super:cloud` | 1775 | 9 | 2 | Compact | Compact |
| 12 | `nemotron-3-ultra:cloud` | 7733 | 24 | 12 | High | Detailed |
| 13 | `qwen3.5:397b-cloud` | 3391 | 7 | 6 | Medium | Balanced |
| 14 | `bjoernb/claude-opus-4-5:latest` | - | - | - | - | Retired (410) |
| 15 | `deepseek-v3.1:671b-cloud` | - | - | - | - | Retired (410) |
| 16 | `glm-5:cloud` | - | - | - | - | Retired (410) |
| 17 | `qwen3-vl:235b-cloud` | - | - | - | - | Retired (410) |

**13 out of 17** active models produced a valid SVG. The 4 retired models returned HTTP 410 Gone (removed from Ollama Cloud on 2026-07-15).

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most detailed World Cup poster SVG**: pick models labeled "Very high" complexity in the table above
- **You want the fastest response**: look at the per-model sections below for the elapsed time
- **You want the cleanest, most reusable SVG code**: pick models that use `<defs>`, `<symbol>`, `<use>`, and transforms (see raw source below each SVG)
- **You want a small, efficient SVG for web embedding**: pick models with "Compact" verdict
- **You want accurate text rendering ("FIFA", "2026")**: check the per-model analysis -- text rendering is the hardest part of this prompt
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"
- **You want to compare within a model family**: pick `deepseek-v4-pro` vs `deepseek-v4-flash`, or `glm-5.1` vs `glm-5.2`, or `minimax-m2.7` vs `minimax-m3`

Now read on for the full per-model breakdown and judge for yourself.

## 1. `deepseek-v4-flash:cloud`

**SVG size:** 11177 characters  
**Complexity:** Very high  
**Shape elements:** 70  
**Distinct colors:** 16  
**Raw response:** 11643 characters  
**Generation time:** 34.8s

![deepseek-v4-flash:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/deepseek-v4-flash_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Uses `<use>` references to reuse defined elements -- ideal for repeating flag patterns or stars
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Defines reusable `<symbol>` elements -- shows mature SVG authoring for repeating motifs like flags
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders the FIFA World Cup trophy -- the model captured the iconic symbol of the tournament
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Renders flags -- the model captured the international/national element of the World Cup
- Renders a stadium -- the model captured the venue setting
- References "We Are 26" -- the model captured the official 2026 tournament slogan
- Includes stars -- the model added celebratory/tournament decoration

With 11,177 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <defs>
    <!-- Background Gradient -->
    <radialGradient id="bgGradient" cx="50%" cy="50%" r="70%" fx="50%" fy="50%">
      <stop offset="0%" stop-color="#1a1a2e" />
      <stop offset="60%" stop-color="#16213e" />
      <stop offset="100%" stop-color="#0f0f1a" />
    </radialGradient>
    <!-- Gold Gradient for Trophy -->
    <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#F3E5AB" />
      <stop offset="30%" stop-color="#D4AF37" />
      <stop offset="70%" stop-color="#AA7700" />
      <stop offset="100%" stop-color="#5C4300" />
    </linearGradient>
    <!-- Vibrant Gradient for 26 Logo -->
    <linearGradient id="vibrantGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FF3366" />
      <stop offset="30%" stop-color="#FF9933" />
      <stop offset="70%" stop-color="#33CCFF" />
      <stop offset="100%" stop-color="#9933FF" />
    </linearGradient>
    <!-- Flame Gradient -->
    <linearGradient id="flameGradient" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FF0000" />
      <stop offset="50%" stop-color="#FF6600" />
      <stop offset="100%" stop-color="#FFD700" />
    </linearGradient>
    <!-- Soccer Ball Gradient -->
    <radialGradient id="ballGradient" cx="40%" cy="40%" r="60%">
      <stop offset="0%" stop-color="#ffffff" />
      <stop offset="80%" stop-color="#e0e0e0" />
      <stop offset="100%" stop-color="#a0a0a0" />
    </radialGradient>
    <!-- Glow Filter -->
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="8" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <!-- Drop Shadow -->
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="#000000" flood-opacity="0.6" />
    </filter>
    <!-- USA Flag Pattern -->
    <symbol id="flag-usa" viewBox="0 0 60 40">
      <rect width="60" height="40" fill="#ffffff" />
      <!-- Stripes -->
      <rect y="0" width="60" height="3.07" fill="#B22234" />
      <rect y="6.15" width="60" height="3.07" fill="#B22234" />
      <rect y="12.3" width="60" height="3.07" fill="#B22234" />
      <rect y="18.45" width="60" height="3.07" fill="#B22234" />
      <rect y="24.6" width="60" height="3.07" fill="#B22234" />
      <rect y="30.75" width="60" height="3.07" fill="#B22234" />
      <rect y="36.9" width="60" height="3.07" fill="#B22234" />
      <!-- Canton -->
      <rect width="24" height="21.5" fill="#3C3B6E" />
      <!-- Stars -->
      <g fill="#ffffff" transform="scale(0.8) translate(2, 2)">
        <polygon points="3,0 4,2 6,2 4.5,3.5 5,5.5 3,4.5 1,5.5 1.5,3.5 0,2 2,2" />
        <polygon points="9,0 10,2 12,2 10.5,3.5 11,5.5 9,4.5 7,5.5 7.5,3.5 6,2 8,2" />
        <polygon points="15,0 16,2 18,2 16.5,3.5 17,5.5 15,4.5 13,5.5 13.5,3.5 12,2 14,2" />
        <polygon points="21,0 22,2 24,2 22.5,3.5 23,5.5 21,4.5 19,5.5 19.5,3.5 18,2 20,2" />
        <polygon points="27,0 28,2 30,2 28.5,3.5 29,5.5 27,4.5 25,5.5 25.5,3.5 24,2 26,2" />
        <polygon points="3,6 4,8 6,8 4.5,9.5 5,11.5 3,10.5 1,11.5 1.5,9.5 0,8 2,8" />
        <polygon points="9,6 10,8 12,8 10.5,9.5 11,11.5 9,10.5 7,11.5 7.5,9.5 6,8 8,8" />
        <polygon points="15,6 16,8 18,8 16.5,9.5 17,11.5 15,10.5 13,11.5 13.5,9.5 12,8 14,8" />
        <polygon points="21,6 22,8 24,8 22.5,9.5 23,11.5 21,10.5 19,11.5 19.5,9.5 18,8 20,8" />
        <polygon points="27,6 28,8 30,8 28.5,9.5 29,11.5 27,10.5 25,11.5 25.5,9.5 24,8 26,8" />
      </g>
    </symbol>
    <!-- Canada Flag Pattern -->
    <symbol id="flag-canada" viewBox="0 0 60 40">
      <rect width="60" height="40" fill="#ffffff" />
      <rect width="15" height="40" fill="#FF0000" />
      <rect x="45" width="15" height="40" fill="#FF0000" />
      <!-- Simplified Maple Leaf -->
      <path d="M 30 5 L 32 12 L 38 12 L 34 16 L 36 22 L 30 18 L 24 22 L 26 16 L 22 12 L 28 12 Z" fill="#FF0000" />
      <rect x="28" y="22" width="4" height="8" fill="#FF0000" />
      <rect x="26" y="28" width="8" height="4" fill="#FF0000" />
    </symbol>
    <!-- Mexico Flag Pattern -->
    <symbol id="flag-mexico" viewBox="0 0 60 40">
      <rect width="20" height="40" fill="#006847" />
      <rect x="20" width="20" height="40" fill="#ffffff" />
      <rect x="40" width="20" height="40" fill="#CE1126" />
      <!-- Simplified Eagle -->
      <path d="M 30 10 L 32 15 L 35 15 L 33 18 L 34 22 L 30 20 L 26 22 L 27 18 L 25 15 L 28 15 Z" fill="#000000" />
      <circle cx="30" cy="12" r="1.5" fill="#000000" />
    </symbol>
  </defs>
  <!-- Background -->
  <rect width="800" height="600" fill="url(#bgGradient)" />
  <!-- Abstract Background Elements (Stadium Lights / Glowing Orbs) -->
  <g opacity="0.15">
    <circle cx="100" cy="100" r="150" fill="#FF3366" filter="url(#glow)" />
    <circle cx="700" cy="500" r="200" fill="#33CCFF" filter="url(#glow)" />
    <circle cx="400" cy="300" r="250" fill="#9933FF" filter="url(#glow)" opacity="0.5" />
  </g>
  <!-- Geometric Background Pattern -->
  <g stroke="#ffffff" stroke-width="1" opacity="0.05" fill="none">
    <polygon points="0,600 200,0 400,600" />
    <polygon points="200,600 400,0 600,600" />
    <polygon points="400,600 600,0 800,600" />
    <polygon points="100,600 300,0 500,600" />
    <polygon points="300,600 500,0 700,600" />
  </g>
  <!-- Central Glowing Soccer Ball (Background) -->
  <g transform="translate(400, 300)" opacity="0.15">
    <circle cx="0" cy="0" r="180" fill="url(#ballGradient)" />
    <!-- Pentagons and Hexagons -->
    <path d="M 0 -180 L 30 -150 L 10 -120 L -10 -120 L -30 -150 Z" fill="#ffffff" stroke="#000000" stroke-width="2" />
    <path d="M 0 -180 L -30 -150 L -60 -160 L -50 -190 Z" fill="#ffffff" stroke="#000000" stroke-width="2" />
    <path d="M 0 -180 L 30 -150 L 60 -160 L 50 -190 Z" fill="#ffffff" stroke="#000000" stroke-width="2" />
    <path d="M -30 -150 L -10 -120 L -40 -90 L -70 -110 Z" fill="#ffffff" stroke="#000000" stroke-width="2" />
    <path d="M 30 -150 L 10 -120 L 40 -90 L 70 -110 Z" fill="#ffffff" stroke="#000000" stroke-width="2" />
  </g>
  <!-- 26 Logo (Background/Hero) -->
  <g transform="translate(400, 240)" filter="url(#glow)">
    <!-- Stylized 2 with Flame -->
    <path d="M -120 -40 C -120 -100, -60 -120, -20 -80 C 10 -50, 0 -20, -20 0 C -40 20, -80 20, -100 0 L -80 -20 C -70 -10, -50 -10, -40 -20 C -30 -30, -30 -60, -10 -80 C 10 -100, 40 -80, 40 -50 C 40 -20, 20 0, 0 20 L -120 20 Z" fill="url(#vibrantGradient)" />
    <!-- Stylized 6 with Soccer Ball Pattern -->
    <path d="M 20 -80 C 80 -80, 120 -40, 120 20 C 120 80, 80 120, 20 120 C -20 120, -40 100, -40 60 C -40 20, -20 0, 20 0 C 60 0, 80 20, 80 60 C 80 80, 60 100, 40 100 C 20 100, 10 90, 10 70 C 10 50, 20 40, 40 40 C 50 40, 60 50, 60 60 C 60 70, 50 80, 40 80 C 30 80, 30 70, 40 60 M 20 -80 L 20 20" fill="url(#vibrantGradient)" />
  </g>
  <!-- FIFA World Cup Trophy (Center) -->
  <g transform="translate(400, 320)" filter="url(#shadow)">
    <!-- Malachite Base -->
    <rect x="-45" y="150" width="90" height="15" rx="3" fill="#2E8B57" />
    <rect x="-40" y="145" width="80" height="5" fill="#3CB371" />
    <!-- Gold Base Plate -->
    <rect x="-35" y="130" width="70" height="15" rx="2" fill="url(#goldGradient)" />
    <rect x="-30" y="125" width="60" height="5" fill="url(#goldGradient)" />
    <!-- Stem -->
    <path d="M -15 125 L 15 125 L 20 60 L -20 60 Z" fill="url(#goldGradient)" />
    <rect x="-25" y="55" width="50" height="10" rx="2" fill="url(#goldGradient)" />
    <!-- Globe / Cup -->
    <path d="M -40 55 C -40 -10, 40 -10, 40 55 C 40 75, -40 75, -40 55 Z" fill="url(#goldGradient)" />
    <!-- Globe Details (Continents) -->
    <path d="M -20 55 C -20 20, 0 10, 10 30 C 20 50, 10 60, 0 55" fill="none" stroke="#5C4300" stroke-width="2" />
    <path d="M 20 55 C 20 30, 30 20, 35 40" fill="none" stroke="#5C4300" stroke-width="2" />
    <path d="M -30 40 C -20 30, -10 40, -15 50" fill="none" stroke="#5C4300" stroke-width="2" />
    <!-- Figures (Stylized Athletes holding the globe) -->
    <!-- Left Figure -->
    <path d="M -35 55 C -50 30, -40 10, -25 20 L -20 40 Z" fill="url(#goldGradient)" />
    <path d="M -25 20 C -15 10, -5 20, -10 35" fill="none" stroke="#5C4300" stroke-width="2" />
    <!-- Right Figure -->
    <path d="M 35 55 C 50 30, 40 10, 25 20 L 20 40 Z" fill="url(#goldGradient)" />
    <path d="M 25 20 C 15 10, 5 20, 10 35" fill="none" stroke="#5C4300" stroke-width="2" />
    <!-- Globe Highlight -->
    <path d="M -30 30 C -20 10, 20 10, 30 30" fill="none" stroke="#F3E5AB" stroke-width="4" opacity="0.6" />
  </g>
  <!-- Text Elements -->
  <g text-anchor="middle" font-family="Arial, Helvetica, sans-serif">
    <!-- FIFA WORLD CUP (Top) -->
    <text x="400" y="80" font-size="36" font-weight="900" letter-spacing="8" fill="#ffffff" filter="url(#glow)">
      FIFA WORLD CUP
    </text>
    <!-- 2026 (Center Top) -->
    <text x="400" y="140" font-size="60" font-weight="900" letter-spacing="4" fill="url(#vibrantGradient)" filter="url(#glow)">
      2026
    </text>
    <!-- USA - CANADA - MEXICO (Bottom) -->
    <text x="400" y="530" font-size="24" font-weight="bold" letter-spacing="6" fill="#ffffff" opacity="0.9">
      USA - CANADA - MEXICO
    </text>
    <!-- We Are 26 (Subtext) -->
    <text x="400" y="560" font-size="14" font-weight="normal" letter-spacing="4" fill="#a0a0a0">
      WE ARE 26
    </text>
  </g>
  <!-- Host Nation Flags -->
  <g transform="translate(400, 480)">
    <!-- USA Flag -->
    <use href="#flag-usa" x="-160" y="-20" width="60" height="40" />
    <text x="-130" y="30" font-family="Arial, sans-serif" font-size="10" fill="#ffffff" text-anchor="middle" letter-spacing="1">USA</text>
    <!-- Canada Flag -->
    <use href="#flag-canada" x="-30" y="-20" width="60" height="40" />
    <text x="0" y="30" font-family="Arial, sans-serif" font-size="10" fill="#ffffff" text-anchor="middle" letter-spacing="1">CANADA</text>
    <!-- Mexico Flag -->
    <use href="#flag-mexico" x="100" y="-20" width="60" height="40" />
    <text x="130" y="30" font-family="Arial, sans-serif" font-size="10" fill="#ffffff" text-anchor="middle" letter-spacing="1">MEXICO</text>
  </g>
  <!-- Decorative Lines -->
  <g stroke="url(#vibrantGradient)" stroke-width="2" opacity="0.6">
    <line x1="100" y1="95" x2="300" y2="95" />
    <line x1="500" y1="95" x2="700" y2="95" />
    <line x1="150" y1="545" x2="300" y2="545" />
    <line x1="500" y1="545" x2="650" y2="545" />
  </g>
  <!-- Stars Decoration -->
  <g fill="#ffffff" opacity="0.3">
    <polygon points="100,200 102,205 107,205 103,208 105,213 100,210 95,213 97,208 93,205 98,205" />
    <polygon points="700,150 702,155 707,155 703,158 705,163 700,160 695,163 697,158 693,155 698,155" />
    <polygon points="150,400 152,405 157,405 153,408 155,413 150,410 145,413 147,408 143,405 148,405" />
    <polygon points="650,450 652,455 657,455 653,458 655,463 650,460 645,463 647,458 643,455 648,455" />
  </g>
</svg>
```

</details>

## 2. `deepseek-v4-pro:cloud`

**SVG size:** 7767 characters  
**Complexity:** High  
**Shape elements:** 28  
**Distinct colors:** 6  
**Raw response:** 8206 characters  
**Generation time:** 77.8s

![deepseek-v4-pro:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/deepseek-v4-pro_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Uses `<use>` references to reuse defined elements -- ideal for repeating flag patterns or stars
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders the FIFA World Cup trophy -- the model captured the iconic symbol of the tournament
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Includes stars -- the model added celebratory/tournament decoration

With 7,767 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <defs>
    <!-- Background Gradient -->
    <radialGradient id="bgGrad" cx="50%" cy="50%" r="70%">
      <stop offset="0%" stop-color="#1A237E"/>
      <stop offset="100%" stop-color="#0D0D0D"/>
    </radialGradient>
    <!-- Gold Gradient for Trophy -->
    <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFD700"/>
      <stop offset="50%" stop-color="#FFA500"/>
      <stop offset="100%" stop-color="#FF8C00"/>
    </linearGradient>
    <!-- Malachite Green Gradient for Base -->
    <linearGradient id="malachiteGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1E5631"/>
      <stop offset="50%" stop-color="#4C9A2A"/>
      <stop offset="100%" stop-color="#1E5631"/>
    </linearGradient>
    <!-- Globe Gradient -->
    <radialGradient id="globeGrad" cx="30%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#4FC3F7"/>
      <stop offset="100%" stop-color="#0277BD"/>
    </radialGradient>
    <!-- Text Gradient for 2026 -->
    <linearGradient id="textGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FFD700"/>
      <stop offset="50%" stop-color="#FFA500"/>
      <stop offset="100%" stop-color="#FFD700"/>
    </linearGradient>
    <!-- Ribbon Gradients -->
    <linearGradient id="ribbon1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FF0000"/>
      <stop offset="100%" stop-color="#FFA500"/>
    </linearGradient>
    <linearGradient id="ribbon2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00BFFF"/>
      <stop offset="100%" stop-color="#0000FF"/>
    </linearGradient>
    <linearGradient id="ribbon3" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#32CD32"/>
      <stop offset="100%" stop-color="#008000"/>
    </linearGradient>
    <linearGradient id="ribbon4" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FF00FF"/>
      <stop offset="100%" stop-color="#800080"/>
    </linearGradient>
    <linearGradient id="ribbon5" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FFD700"/>
      <stop offset="100%" stop-color="#FF8C00"/>
    </linearGradient>
    <!-- Glow Filter -->
    <filter id="blur" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="15"/>
    </filter>
    <!-- Star Symbol -->
    <g id="star">
      <path d="M 0,-15 L 4,-4 L 15,-4 L 6,3 L 10,15 L 0,8 L -10,15 L -6,3 L -15,-4 L -4,-4 Z" fill="#FFD700"/>
    </g>
    <!-- Curved Path for FIFA Text -->
    <path id="fifaPath" d="M 150 150 Q 400 80, 650 150" fill="none"/>
  </defs>
  <!-- Background -->
  <rect width="800" height="600" fill="url(#bgGrad)"/>
  <!-- Glow Behind Trophy -->
  <circle cx="400" cy="300" r="150" fill="#FFFFFF" opacity="0.1" filter="url(#blur)"/>
  <!-- Dynamic Ribbons -->
  <g>
    <path d="M 50 150 Q 200 50, 400 150 T 750 150" fill="none" stroke="url(#ribbon1)" stroke-width="15" stroke-linecap="round"/>
    <path d="M 100 250 Q 250 150, 400 250 T 700 250" fill="none" stroke="url(#ribbon2)" stroke-width="15" stroke-linecap="round"/>
    <path d="M 50 350 Q 200 250, 400 350 T 750 350" fill="none" stroke="url(#ribbon3)" stroke-width="15" stroke-linecap="round"/>
    <path d="M 100 450 Q 250 350, 400 450 T 700 450" fill="none" stroke="url(#ribbon4)" stroke-width="15" stroke-linecap="round"/>
    <path d="M 150 500 Q 300 400, 400 500 T 650 500" fill="none" stroke="url(#ribbon5)" stroke-width="15" stroke-linecap="round"/>
  </g>
  <!-- Stars / Confetti -->
  <g>
    <g transform="translate(150, 100) scale(0.8)"><use href="#star"/></g>
    <g transform="translate(650, 150) scale(1)"><use href="#star"/></g>
    <g transform="translate(200, 500) scale(0.6)"><use href="#star"/></g>
    <g transform="translate(600, 450) scale(0.9)"><use href="#star"/></g>
    <g transform="translate(100, 300) scale(0.7)"><use href="#star"/></g>
    <g transform="translate(700, 350) scale(0.8)"><use href="#star"/></g>
    <g transform="translate(250, 200) scale(0.5)"><use href="#star"/></g>
    <g transform="translate(550, 200) scale(0.5)"><use href="#star"/></g>
    <g transform="translate(120, 450) scale(0.4)"><use href="#star"/></g>
    <g transform="translate(680, 250) scale(0.6)"><use href="#star"/></g>
  </g>
  <!-- FIFA World Cup Trophy -->
  <g>
    <!-- Trophy Shadow -->
    <ellipse cx="400" cy="480" rx="80" ry="15" fill="#000000" opacity="0.6"/>
    <!-- Trophy Base -->
    <rect x="330" y="450" width="140" height="20" rx="4" fill="url(#malachiteGrad)"/>
    <rect x="340" y="440" width="120" height="10" rx="2" fill="url(#goldGrad)"/>
    <rect x="350" y="430" width="100" height="10" rx="2" fill="url(#malachiteGrad)"/>
    <!-- Trophy Stem -->
    <rect x="390" y="410" width="20" height="20" fill="url(#goldGrad)"/>
    <!-- Trophy Figures (Stylized) -->
    <!-- Left Figure -->
    <path d="M 370 410 C 370 380, 360 340, 380 300 C 390 280, 400 280, 400 280" fill="none" stroke="url(#goldGrad)" stroke-width="12" stroke-linecap="round"/>
    <!-- Right Figure -->
    <path d="M 430 410 C 430 380, 440 340, 420 300 C 410 280, 400 280, 400 280" fill="none" stroke="url(#goldGrad)" stroke-width="12" stroke-linecap="round"/>
    <!-- Necks -->
    <rect x="370" y="300" width="10" height="15" fill="url(#goldGrad)"/>
    <rect x="420" y="300" width="10" height="15" fill="url(#goldGrad)"/>
    <!-- Heads -->
    <circle cx="375" cy="290" r="12" fill="url(#goldGrad)"/>
    <circle cx="425" cy="290" r="12" fill="url(#goldGrad)"/>
    <!-- Arms Holding the Globe -->
    <path d="M 375 320 C 360 300, 370 260, 385 250" stroke="url(#goldGrad)" stroke-width="8" fill="none" stroke-linecap="round"/>
    <path d="M 425 320 C 440 300, 430 260, 415 250" stroke="url(#goldGrad)" stroke-width="8" fill="none" stroke-linecap="round"/>
    <!-- Globe -->
    <circle cx="400" cy="260" r="35" fill="url(#globeGrad)"/>
    <circle cx="400" cy="260" r="35" fill="none" stroke="#FFFFFF" stroke-width="2" opacity="0.5"/>
    <!-- Stylized Continents on Globe -->
    <path d="M 380 250 Q 390 240, 400 250 Q 410 260, 400 270 Q 390 270, 380 260 Z" fill="#81C784"/>
    <path d="M 415 250 Q 425 250, 425 260 Q 425 270, 415 270 Q 410 260, 415 250 Z" fill="#81C784"/>
    <path d="M 390 280 Q 400 280, 400 290 Q 390 300, 380 290 Z" fill="#81C784"/>
  </g>
  <!-- Text Elements -->
  <g>
    <!-- FIFA WORLD CUP Curved Text -->
    <text font-family="'Arial Black', Impact, sans-serif" font-size="36" font-weight="900" fill="#FFFFFF" letter-spacing="6">
      <textPath href="#fifaPath" startOffset="50%" text-anchor="middle">FIFA WORLD CUP</textPath>
    </text>
    <!-- Curved Underline -->
    <path d="M 180 170 Q 400 100, 620 170" fill="none" stroke="#FFD700" stroke-width="4" stroke-linecap="round"/>
    <!-- 2026 Text with Glow and Stroke -->
    <text x="400" y="520" font-family="'Arial Black', Impact, sans-serif" font-size="90" font-weight="900" text-anchor="middle" fill="#FFD700" opacity="0.3" filter="url(#blur)" letter-spacing="8">2026</text>
    <text x="400" y="520" font-family="'Arial Black', Impact, sans-serif" font-size="90" font-weight="900" text-anchor="middle" fill="none" stroke="#FFFFFF" stroke-width="6" letter-spacing="8">2026</text>
    <text x="400" y="520" font-family="'Arial Black', Impact, sans-serif" font-size="90" font-weight="900" text-anchor="middle" fill="url(#textGrad)" letter-spacing="8">2026</text>
    <!-- Host Nations Text -->
    <text x="400" y="570" font-family="'Arial', sans-serif" font-size="18" font-weight="bold" text-anchor="middle" fill="#B0BEC5" letter-spacing="6">UNITED STATES · CANADA · MEXICO</text>
  </g>
</svg>
```

</details>

## 3. `gemma4:31b-cloud`

**SVG size:** 1942 characters  
**Complexity:** Compact  
**Shape elements:** 11  
**Distinct colors:** 11  
**Raw response:** 3115 characters  
**Generation time:** 9.0s

![gemma4:31b-cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/gemma4_31b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Includes stars -- the model added celebratory/tournament decoration

With 1,942 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
    <!-- Background Circle for Depth -->
    <circle cx="250" cy="220" r="160" fill="#f0f0f0" stroke="#ccc" stroke-width="2"/>
    <!-- The Football (Stylized) -->
    <g id="football">
        <circle cx="250" cy="220" r="120" fill="#ffffff" stroke="#000" stroke-width="4"/>
        <!-- Hexagon patterns for the ball -->
        <path d="M250 100 L280 120 L280 160 L250 180 L220 160 L220 120 Z" fill="#1a1a1a" stroke="#fff" stroke-width="2"/>
        <path d="M280 120 L320 140 L320 180 L280 200 L250 180" fill="#555" stroke="#fff" stroke-width="2"/>
        <path d="M220 120 L180 140 L180 180 L220 200 L250 180" fill="#555" stroke="#fff" stroke-width="2"/>
        <path d="M250 180 L280 200 L300 240 L250 260 L200 240 L220 200 Z" fill="#1a1a1a" stroke="#fff" stroke-width="2"/>
        <path d="M220 200 L180 200 L160 240 L200 260 L220 240" fill="#555" stroke="#fff" stroke-width="2"/>
    </g>
    <!-- Three Stars (Representing USA, Mexico, Canada) -->
    <g fill="#FFD700" stroke="#B8860B" stroke-width="2">
        <!-- Star 1 -->
        <polygon points="150,140 158,152 170,152 161,161 164,173 150,166 136,173 139,161 130,152 142,152" />
        <!-- Star 2 -->
        <polygon points="250,100 258,112 270,112 261,121 264,133 250,126 236,133 239,121 230,112 242,112" />
        <!-- Star 3 -->
        <polygon points="350,140 358,152 370,152 361,161 364,173 350,166 336,173 339,161 330,152 342,152" />
    </g>
    <!-- Text Elements -->
    <text x="250" y="380" text-anchor="middle" font-family="Arial, sans-serif" font-weight="bold" font-size="42" fill="#003366">WORLD CUP</text>
    <text x="250" y="430" text-anchor="middle" font-family="Arial, sans-serif" font-weight="900" font-size="60" fill="#CC0000">2026</text>
    <!-- Bottom Decorative Line -->
    <rect x="150" y="395" width="200" height="5" rx="2" fill="#003366" />
</svg>
```

</details>

## 4. `gemma4:cloud`

**SVG size:** 2238 characters  
**Complexity:** Compact  
**Shape elements:** 11  
**Distinct colors:** 6  
**Raw response:** 3298 characters  
**Generation time:** 10.4s

![gemma4:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/gemma4_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts

With 2,238 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background Gradient -->
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#fdfbfb;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#ebedee;stop-opacity:1" />
    </linearGradient>
    <!-- Shadow for the ball -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="5" />
      <feOffset dx="2" dy="5" result="offsetblur" />
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.3" />
      </feComponentTransfer>
      <feMerge>
        <feMergeNode />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>
  <!-- Canvas Background -->
  <rect width="500" height="300" rx="20" fill="url(#grad1)" />
  <!-- Host Nation Color Accents (USA, Mexico, Canada) -->
  <rect x="0" y="270" width="166" height="30" fill="#B22234" /> <!-- Red (Canada/USA) -->
  <rect x="166" y="270" width="168" height="30" fill="#006847" /> <!-- Green (Mexico) -->
  <rect x="334" y="270" width="166" height="30" fill="#002868" /> <!-- Blue (USA) -->
  <!-- Stylized Football -->
  <g filter="url(#shadow)">
    <circle cx="250" cy="130" r="60" fill="#fff" stroke="#333" stroke-width="2" />
    <!-- Football Hexagons (Simplified) -->
    <path d="M250 70 L270 85 L260 110 L230 110 L220 85 Z" fill="#333" />
    <path d="M250 130 L280 140 L290 110 L260 100 Z" fill="#333" />
    <path d="M220 130 L230 150 L210 160 L190 130 Z" fill="#333" />
    <path d="M250 130 L230 150 L210 120 L230 110 Z" fill="#333" />
    <path d="M270 150 L290 150 L300 120 L280 110 Z" fill="#333" />
  </g>
  <!-- Text Elements -->
  <text x="250" y="210" font-family="Arial, Helvetica, sans-serif" font-weight="bold" font-size="52" text-anchor="middle" fill="#333">
    FIFA WORLD CUP
  </text>
  <text x="250" y="250" font-family="Arial, Helvetica, sans-serif" font-weight="900" font-size="70" text-anchor="middle" fill="#002868">
    2026
  </text>
  <!-- Decorative Arc -->
  <path d="M150 210 Q250 240 350 210" stroke="#B22234" stroke-width="4" fill="none" stroke-linecap="round" />
</svg>
```

</details>

## 5. `glm-5.1:cloud`

**SVG size:** 21973 characters  
**Complexity:** Very high  
**Shape elements:** 99  
**Distinct colors:** 17  
**Raw response:** 26157 characters  
**Generation time:** 70.1s

![glm-5.1:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/glm-5.1_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders the FIFA World Cup trophy -- the model captured the iconic symbol of the tournament
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Renders flags -- the model captured the international/national element of the World Cup
- Includes stars -- the model added celebratory/tournament decoration

With over 21,973 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering the trophy, text, flags, and decorative elements.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 900 1050" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
            <defs>
                <!-- Gradients -->
                <linearGradient id="goldGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#f5d96e"/>
                    <stop offset="30%" stop-color="#d4a833"/>
                    <stop offset="60%" stop-color="#f0c94b"/>
                    <stop offset="100%" stop-color="#a87d1a"/>
                </linearGradient>
                <linearGradient id="goldGrad2" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stop-color="#ffe680"/>
                    <stop offset="50%" stop-color="#d4a833"/>
                    <stop offset="100%" stop-color="#8b6914"/>
                </linearGradient>
                <linearGradient id="trophyBody" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#ffeaa0"/>
                    <stop offset="25%" stop-color="#f0c94b"/>
                    <stop offset="55%" stop-color="#d4a833"/>
                    <stop offset="80%" stop-color="#b8912a"/>
                    <stop offset="100%" stop-color="#8b6914"/>
                </linearGradient>
                <linearGradient id="trophyBase" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#d4a833"/>
                    <stop offset="100%" stop-color="#6b4f10"/>
                </linearGradient>
                <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#0d1b3e"/>
                    <stop offset="50%" stop-color="#091228"/>
                    <stop offset="100%" stop-color="#050a18"/>
                </linearGradient>
                <radialGradient id="spotlight" cx="0.5" cy="0.35" r="0.5">
                    <stop offset="0%" stop-color="rgba(212,168,51,0.12)"/>
                    <stop offset="100%" stop-color="rgba(212,168,51,0)"/>
                </radialGradient>
                <linearGradient id="flagStripeUSA1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#B22234"/><stop offset="100%" stop-color="#B22234"/>
                </linearGradient>
                <linearGradient id="pitchGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#1a6e2e"/>
                    <stop offset="100%" stop-color="#0e4420"/>
                </linearGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="softGlow">
                    <feGaussianBlur stdDeviation="6" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="strongGlow">
                    <feGaussianBlur stdDeviation="8" result="blur"/>
                    <feComposite in="blur" in2="SourceGraphic" operator="over"/>
                </filter>
                <clipPath id="svgClip">
                    <rect x="0" y="0" width="900" height="1050" rx="20"/>
                </clipPath>
            </defs>
            <!-- Background -->
            <g clip-path="url(#svgClip)">
                <rect width="900" height="1050" fill="url(#bgGrad)"/>
                <rect width="900" height="1050" fill="url(#spotlight)"/>
                <!-- Subtle pitch lines background -->
                <rect x="150" y="780" width="600" height="220" rx="4" fill="none" stroke="rgba(26,110,46,0.2)" stroke-width="2"/>
                <circle cx="450" cy="890" r="50" fill="none" stroke="rgba(26,110,46,0.2)" stroke-width="2"/>
                <line x1="150" y1="890" x2="750" y2="890" stroke="rgba(26,110,46,0.15)" stroke-width="1"/>
                <!-- Decorative hex pattern background -->
                <g opacity="0.04" stroke="#d4a833" stroke-width="1" fill="none">
                    <circle cx="150" cy="200" r="60"/>
                    <circle cx="750" cy="300" r="80"/>
                    <circle cx="100" cy="700" r="50"/>
                    <circle cx="800" cy="800" r="70"/>
                    <polygon points="450,50 470,80 450,110 430,80" />
                    <polygon points="820,500 840,530 820,560 800,530" />
                    <polygon points="80,450 100,480 80,510 60,480" />
                </g>
                <!-- Top decorative arc -->
                <path d="M100,120 Q450,30 800,120" fill="none" stroke="rgba(212,168,51,0.15)" stroke-width="1"/>
                <!-- FIFA text at top -->
                <text x="450" y="85" text-anchor="middle" font-family="'Impact','Arial Black',sans-serif" font-size="38" font-weight="900" letter-spacing="18" fill="url(#goldGrad)" filter="url(#glow)">FIFA</text>
                <!-- WORLD CUP text -->
                <text x="450" y="130" text-anchor="middle" font-family="'Trebuchet MS','Arial',sans-serif" font-size="20" letter-spacing="12" fill="rgba(255,234,160,0.7)">WORLD CUP</text>
                <!-- Decorative line under title -->
                <line x1="250" y1="148" x2="650" y2="148" stroke="url(#goldGrad)" stroke-width="1.5" opacity="0.5"/>
                <circle cx="250" cy="148" r="3" fill="#d4a833" opacity="0.5"/>
                <circle cx="650" cy="148" r="3" fill="#d4a833" opacity="0.5"/>
                <circle cx="450" cy="148" r="4" fill="#f0c94b" opacity="0.7"/>
                <!-- Sparkle stars around trophy -->
                <g class="star-sparkle" filter="url(#softGlow)">
                    <polygon points="300,240 304,250 314,250 306,256 309,266 300,260 291,266 294,256 286,250 296,250" fill="#ffe680" opacity="0.8"/>
                </g>
                <g class="star-sparkle" filter="url(#softGlow)">
                    <polygon points="600,260 603,268 611,268 605,273 607,281 600,276 593,281 595,273 589,268 597,268" fill="#ffe680" opacity="0.7"/>
                </g>
                <g class="star-sparkle" filter="url(#softGlow)">
                    <polygon points="260,380 263,388 271,388 265,393 267,401 260,396 253,401 255,393 249,388 257,388" fill="#ffe680" opacity="0.6"/>
                </g>
                <g class="star-sparkle" filter="url(#softGlow)">
                    <polygon points="640,370 643,378 651,378 645,383 647,391 640,386 633,391 635,383 629,378 637,378" fill="#ffe680" opacity="0.6"/>
                </g>
                <g class="star-sparkle" filter="url(#softGlow)">
                    <polygon points="450,200 454,212 466,212 457,219 460,231 450,224 440,231 443,219 434,212 446,212" fill="#ffe680" opacity="0.9"/>
                </g>
                <!-- Trophy group with float animation -->
                <g class="trophy-group">
                    <!-- Trophy outer glow -->
                    <ellipse cx="450" cy="500" rx="120" ry="200" fill="rgba(212,168,51,0.05)" filter="url(#strongGlow)"/>
                    <!-- TROPHY -->
                    <!-- Base bottom -->
                    <rect x="390" y="680" width="120" height="16" rx="3" fill="url(#trophyBase)" stroke="#8b6914" stroke-width="1"/>
                    <rect x="380" y="696" width="140" height="14" rx="4" fill="url(#goldGrad2)" stroke="#8b6914" stroke-width="1"/>
                    <!-- Base nameplate -->
                    <rect x="395" y="683" width="110" height="10" rx="2" fill="#1a1a2e" stroke="#d4a833" stroke-width="0.5"/>
                    <text x="450" y="691" text-anchor="middle" font-family="serif" font-size="6" fill="#d4a833" letter-spacing="2">WORLD CUP</text>
                    <!-- Base middle column -->
                    <rect x="415" y="660" width="70" height="22" rx="2" fill="url(#trophyBase)"/>
                    <rect x="425" y="648" width="50" height="14" rx="2" fill="url(#goldGrad2)"/>
                    <!-- Trophy stem -->
                    <path d="M435,648 L435,580 Q435,570 440,565 L460,565 Q465,570 465,580 L465,648" fill="url(#trophyBase)" stroke="#8b6914" stroke-width="0.5"/>
                    <!-- Trophy stem gold band -->
                    <rect x="432" y="600" width="36" height="8" rx="2" fill="#f0c94b" stroke="#b8912a" stroke-width="0.5"/>
                    <rect x="432" y="625" width="36" height="6" rx="1.5" fill="#d4a833" stroke="#b8912a" stroke-width="0.5"/>
                    <!-- Trophy body - the main cup -->
                    <path d="M380,400 
                             Q380,395 390,390 
                             L410,385 
                             Q430,380 450,378 
                             Q470,380 490,385 
                             L510,390 
                             Q520,395 520,400 
                             L520,420 
                             Q520,480 500,520 
                             Q485,550 470,560 
                             L460,565 
                             L440,565 
                             L430,560 
                             Q415,550 400,520 
                             Q380,480 380,420 
                             Z" 
                          fill="url(#trophyBody)" stroke="#b8912a" stroke-width="1.5"/>
                    <!-- Trophy inner shadow -->
                    <path d="M395,405 
                             Q395,400 400,398 
                             L450,388 
                             L500,398 
                             Q505,400 505,405 
                             L505,430 
                             Q505,480 490,510 
                             Q478,540 465,550 
                             L435,550 
                             Q422,540 410,510 
                             Q395,480 395,430 
                             Z" 
                          fill="rgba(139,105,20,0.3)"/>
                    <!-- Trophy highlight left -->
                    <path d="M390,400 Q388,430 395,470 Q400,490 410,510" 
                          fill="none" stroke="rgba(255,234,160,0.4)" stroke-width="3" stroke-linecap="round"/>
                    <!-- Trophy highlight right -->
                    <path d="M510,400 Q512,430 505,470 Q500,490 490,510" 
                          fill="none" stroke="rgba(255,234,160,0.25)" stroke-width="2" stroke-linecap="round"/>
                    <!-- Trophy rim -->
                    <ellipse cx="450" cy="395" rx="70" ry="14" fill="url(#goldGrad)" stroke="#b8912a" stroke-width="1"/>
                    <ellipse cx="450" cy="393" rx="65" ry="11" fill="none" stroke="rgba(255,234,160,0.3)" stroke-width="1"/>
                    <!-- Handles - Left -->
                    <path d="M380,415 
                             Q340,415 325,440 
                             Q310,470 320,500 
                             Q330,530 355,540 
                             Q370,545 380,535" 
                          fill="none" stroke="url(#goldGrad)" stroke-width="12" stroke-linecap="round"/>
                    <path d="M380,415 
                             Q345,418 332,440 
                             Q318,468 326,498 
                             Q335,525 355,535 
                             Q368,540 380,530" 
                          fill="none" stroke="rgba(255,234,160,0.3)" stroke-width="3" stroke-linecap="round"/>
                    <!-- Handles - Right -->
                    <path d="M520,415 
                             Q560,415 575,440 
                             Q590,470 580,500 
                             Q570,530 545,540 
                             Q530,545 520,535" 
                          fill="none" stroke="url(#goldGrad)" stroke-width="12" stroke-linecap="round"/>
                    <path d="M520,415 
                             Q555,418 568,440 
                             Q582,468 574,498 
                             Q565,525 545,535 
                             Q532,540 520,530" 
                          fill="none" stroke="rgba(255,234,160,0.3)" stroke-width="3" stroke-linecap="round"/>
                    <!-- Figure on trophy -->
                    <!-- Body -->
                    <path d="M450,405 Q455,420 453,440 L450,450 L447,440 Q445,420 450,405" fill="#d4a833" stroke="#b8912a" stroke-width="0.5"/>
                    <!-- Head -->
                    <circle cx="450" cy="400" r="7" fill="#f0c94b" stroke="#b8912a" stroke-width="0.5"/>
                    <!-- Arms raised -->
                    <path d="M443,420 Q430,410 420,395" fill="none" stroke="#d4a833" stroke-width="2.5" stroke-linecap="round"/>
                    <path d="M457,420 Q470,410 480,395" fill="none" stroke="#d4a833" stroke-width="2.5" stroke-linecap="round"/>
                    <!-- Legs -->
                    <path d="M450,450 Q445,465 440,475" fill="none" stroke="#d4a833" stroke-width="2" stroke-linecap="round"/>
                    <path d="M450,450 Q455,465 460,475" fill="none" stroke="#d4a833" stroke-width="2" stroke-linecap="round"/>
                    <!-- Small ball under figure's foot -->
                    <circle cx="440" cy="477" r="4" fill="#f0c94b" stroke="#b8912a" stroke-width="0.5"/>
                    <circle cx="460" cy="477" r="4" fill="#f0c94b" stroke="#b8912a" stroke-width="0.5"/>
                </g>
                <!-- "2026" large text -->
                <text x="450" y="790" text-anchor="middle" font-family="'Impact','Arial Black',sans-serif" font-size="110" font-weight="900" fill="url(#goldGrad)" filter="url(#glow)" letter-spacing="8">2026</text>
                <!-- Subtle text shadow for 2026 -->
                <text x="451" y="791" text-anchor="middle" font-family="'Impact','Arial Black',sans-serif" font-size="110" font-weight="900" fill="rgba(0,0,0,0.3)" letter-spacing="8" style="pointer-events:none">2026</text>
                <!-- Host countries section -->
                <!-- USA Flag -->
                <g transform="translate(195, 845)">
                    <rect x="0" y="0" width="70" height="46" rx="4" fill="#B22234" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
                    <!-- Stripes -->
                    <rect x="0" y="4.2" width="70" height="3.5" fill="#fff"/>
                    <rect x="0" y="11.2" width="70" height="3.5" fill="#fff"/>
                    <rect x="0" y="18.2" width="70" height="3.5" fill="#fff"/>
                    <rect x="0" y="25.2" width="70" height="3.5" fill="#fff"/>
                    <rect x="0" y="32.2" width="70" height="3.5" fill="#fff"/>
                    <rect x="0" y="39.2" width="70" height="3.5" fill="#fff" rx="0"/>
                    <!-- Blue canton -->
                    <rect x="0" y="0" width="28" height="24.5" rx="3" fill="#3C3B6E"/>
                    <!-- Stars simplified -->
                    <circle cx="7" cy="6" r="1.5" fill="#fff"/>
                    <circle cx="14" cy="6" r="1.5" fill="#fff"/>
                    <circle cx="21" cy="6" r="1.5" fill="#fff"/>
                    <circle cx="7" cy="12" r="1.5" fill="#fff"/>
                    <circle cx="14" cy="12" r="1.5" fill="#fff"/>
                    <circle cx="21" cy="12" r="1.5" fill="#fff"/>
                    <circle cx="7" cy="18" r="1.5" fill="#fff"/>
                    <circle cx="14" cy="18" r="1.5" fill="#fff"/>
                    <circle cx="21" cy="18" r="1.5" fill="#fff"/>
                    <text x="35" y="62" text-anchor="middle" font-family="'Segoe UI',sans-serif" font-size="13" fill="rgba(255,234,160,0.8)" font-weight="600">USA</text>
                </g>
                <!-- Canada Flag -->
                <g transform="translate(365, 845)">
                    <rect x="0" y="0" width="70" height="46" rx="4" fill="#fff" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
                    <rect x="0" y="0" width="14" height="46" rx="4" fill="#FF0000"/>
                    <rect x="56" y="0" width="14" height="46" rx="4" fill="#FF0000"/>
                    <!-- Maple leaf -->
                    <path d="M35,8 L37,16 L43,14 L39,19 L45,22 L39,23 L40,30 L35,26 L30,30 L31,23 L25,22 L31,19 L27,14 L33,16 Z" fill="#FF0000"/>
                    <text x="35" y="62" text-anchor="middle" font-family="'Segoe UI',sans-serif" font-size="13" fill="rgba(255,234,160,0.8)" font-weight="600">CANADA</text>
                </g>
                <!-- Mexico Flag -->
                <g transform="translate(535, 845)">
                    <rect x="0" y="0" width="70" height="46" rx="4" fill="#fff" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
                    <rect x="0" y="0" width="23.3" height="46" rx="4" fill="#006847"/>
                    <rect x="23.3" y="0" width="23.4" height="46" fill="#F4EDED"/>
                    <rect x="46.7" y="0" width="23.3" height="46" rx="4" fill="#CE1126"/>
                    <!-- Simplified coat of arms -->
                    <circle cx="35" cy="23" r="7" fill="none" stroke="#006847" stroke-width="1"/>
                    <path d="M32,20 L35,16 L38,20 L37,24 L33,24 Z" fill="#006847"/>
                    <text x="35" y="62" text-anchor="middle" font-family="'Segoe UI',sans-serif" font-size="13" fill="rgba(255,234,160,0.8)" font-weight="600">MEXICO</text>
                </g>
                <!-- Connecting plus signs between flags -->
                <text x="310" y="873" text-anchor="middle" font-family="'Segoe UI',sans-serif" font-size="22" fill="rgba(212,168,51,0.6)" font-weight="300">×</text>
                <text x="480" y="873" text-anchor="middle" font-family="'Segoe UI',sans-serif" font-size="22" fill="rgba(212,168,51,0.6)" font-weight="300">×</text>
                <!-- Decorative divider -->
                <line x1="200" y1="925" x2="700" y2="925" stroke="rgba(212,168,51,0.2)" stroke-width="1"/>
                <circle cx="450" cy="925" r="3" fill="#d4a833" opacity="0.5"/>
                <circle cx="325" cy="925" r="2" fill="#d4a833" opacity="0.3"/>
                <circle cx="575" cy="925" r="2" fill="#d4a833" opacity="0.3"/>
                <!-- "UNITED BY PASSION" tagline -->
                <text x="450" y="960" text-anchor="middle" font-family="'Trebuchet MS','Segoe UI',sans-serif" font-size="14" letter-spacing="8" fill="rgba(212,168,51,0.45)" font-weight="400">UNITED BY PASSION</text>
                <!-- 48 teams marker -->
                <text x="450" y="990" text-anchor="middle" font-family="'Segoe UI',sans-serif" font-size="11" fill="rgba(255,234,160,0.3)" letter-spacing="4">48 TEAMS  •  104 MATCHES</text>
                <!-- Football decorative elements -->
                <!-- Left football -->
                <g transform="translate(115, 420) scale(0.6)" opacity="0.15">
                    <circle cx="0" cy="0" r="30" fill="none" stroke="#d4a833" stroke-width="2"/>
                    <polygon points="0,-20 6,-8 18,-8 8,2 12,15 0,8 -12,15 -8,2 -18,-8 -6,-8" fill="none" stroke="#d4a833" stroke-width="1.5"/>
                </g>
                <!-- Right football -->
                <g transform="translate(785, 480) scale(0.6)" opacity="0.15">
                    <circle cx="0" cy="0" r="30" fill="none" stroke="#d4a833" stroke-width="2"/>
                    <polygon points="0,-20 6,-8 18,-8 8,2 12,15 0,8 -12,15 -8,2 -18,-8 -6,-8" fill="none" stroke="#d4a833" stroke-width="1.5"/>
                </g>
                <!-- Confetti / decorative particles -->
                <g class="confetti">
                    <rect x="170" y="300" width="6" height="6" rx="1" fill="#ce1126" opacity="0.5" transform="rotate(30 173 303)"/>
                </g>
                <g class="confetti">
                    <rect x="720" y="350" width="5" height="5" rx="1" fill="#1a47a8" opacity="0.5" transform="rotate(45 722 352)"/>
                </g>
                <g class="confetti">
                    <rect x="200" y="600" width="4" height="4" rx="1" fill="#f0c94b" opacity="0.4" transform="rotate(15 202 602)"/>
                </g>
                <g class="confetti">
                    <rect x="700" y="620" width="5" height="5" rx="1" fill="#ce1126" opacity="0.4" transform="rotate(60 702 622)"/>
                </g>
                <g class="confetti">
                    <circle cx="160" cy="500" r="2.5" fill="#1a6e2e" opacity="0.5"/>
                </g>
                <g class="confetti">
                    <circle cx="740" cy="450" r="2" fill="#d4a833" opacity="0.5"/>
                </g>
                <g class="confetti">
                    <rect x="680" y="260" width="4" height="4" rx="1" fill="#fff" opacity="0.2" transform="rotate(20 682 262)"/>
                </g>
                <g class="confetti">
                    <circle cx="220" cy="750" r="1.5" fill="#f0c94b" opacity="0.4"/>
                </g>
                <!-- Animated ring around trophy area -->
                <g class="ring-outer">
                    <ellipse cx="450" cy="520" rx="150" ry="180" fill="none" stroke="rgba(212,168,51,0.06)" stroke-width="1.5" stroke-dasharray="8 12"/>
                </g>
                <!-- Corner decorations -->
                <!-- Top left -->
                <path d="M30,30 L80,30 M30,30 L30,80" fill="none" stroke="rgba(212,168,51,0.2)" stroke-width="2"/>
                <!-- Top right -->
                <path d="M870,30 L820,30 M870,30 L870,80" fill="none" stroke="rgba(212,168,51,0.2)" stroke-width="2"/>
                <!-- Bottom left -->
                <path d="M30,1020 L80,1020 M30,1020 L30,970" fill="none" stroke="rgba(212,168,51,0.2)" stroke-width="2"/>
                <!-- Bottom right -->
                <path d="M870,1020 L820,1020 M870,1020 L870,970" fill="none" stroke="rgba(212,168,51,0.2)" stroke-width="2"/>
                <!-- Bottom center football -->
                <g transform="translate(450, 1020)">
                    <circle cx="0" cy="0" r="0" fill="none" stroke="none"/>
                </g>
            </g>
        </svg>
```

</details>

## 6. `glm-5.2:cloud`

**SVG size:** 5841 characters  
**Complexity:** High  
**Shape elements:** 39  
**Distinct colors:** 14  
**Raw response:** 5907 characters  
**Generation time:** 40.0s

![glm-5.2:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/glm-5.2_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders the FIFA World Cup trophy -- the model captured the iconic symbol of the tournament
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Renders flags -- the model captured the international/national element of the World Cup
- Renders a stadium -- the model captured the venue setting
- Includes stars -- the model added celebratory/tournament decoration

With 5,841 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
    <defs>
        <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0a1128"/>
            <stop offset="50%" stop-color="#1a3a52"/>
            <stop offset="100%" stop-color="#0c4a3e"/>
        </linearGradient>
        <linearGradient id="trophyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#ffe653"/>
            <stop offset="50%" stop-color="#f6a623"/>
            <stop offset="100%" stop-color="#d48110"/>
        </linearGradient>
        <linearGradient id="textGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#ffffff"/>
            <stop offset="100%" stop-color="#a0a0a0"/>
        </linearGradient>
        <linearGradient id="textGold" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#ffd700"/>
            <stop offset="50%" stop-color="#f6a623"/>
            <stop offset="100%" stop-color="#d48110"/>
        </linearGradient>
        <radialGradient id="ballGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#ffffff" stop-opacity="0.3"/>
            <stop offset="70%" stop-color="#ffffff" stop-opacity="0"/>
        </radialGradient>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="5" stdDeviation="5" flood-color="#000000" flood-opacity="0.5"/>
        </filter>
    </defs>
    <!-- Background -->
    <rect width="800" height="600" fill="url(#bgGrad)"/>
    <!-- Abstract Stadium Top -->
    <path d="M -100 250 Q 400 100 900 250 L 900 0 L -100 0 Z" fill="#112244" opacity="0.5"/>
    <path d="M -100 280 Q 400 130 900 280 L 900 0 L -100 0 Z" fill="#0c1832" opacity="0.8"/>
    <!-- Light Beams -->
    <polygon points="400,180 0,0 250,0" fill="#ffffff" opacity="0.03"/>
    <polygon points="400,180 800,0 550,0" fill="#ffffff" opacity="0.03"/>
    <polygon points="400,180 0,250 0,100" fill="#ffffff" opacity="0.03"/>
    <polygon points="400,180 800,250 800,100" fill="#ffffff" opacity="0.03"/>
    <!-- Abstract Pitch Bottom -->
    <path d="M -100 500 Q 400 620 900 500 L 900 600 L -100 600 Z" fill="#0a3d2e" opacity="0.8"/>
    <path d="M -100 540 Q 400 660 900 540 L 900 600 L -100 600 Z" fill="#06281b"/>
    <!-- Glow Behind Trophy -->
    <circle cx="400" cy="220" r="150" fill="url(#ballGlow)"/>
    <!-- Stylized World Cup Trophy -->
    <g transform="translate(400, 220)" filter="url(#shadow)">
        <!-- Base -->
        <path d="M -30 80 L 30 80 L 20 60 L -20 60 Z" fill="url(#trophyGrad)"/>
        <!-- Stem -->
        <path d="M -10 60 L 10 60 L 10 20 L -10 20 Z" fill="url(#trophyGrad)"/>
        <!-- Earth/Globe on top -->
        <circle cx="0" cy="0" r="40" fill="url(#trophyGrad)"/>
        <circle cx="0" cy="0" r="40" fill="none" stroke="#d48110" stroke-width="3"/>
        <!-- Curved lines for globe -->
        <path d="M -40 0 Q 0 -15 40 0" fill="none" stroke="#d48110" stroke-width="2"/>
        <path d="M -40 0 Q 0 15 40 0" fill="none" stroke="#d48110" stroke-width="2"/>
        <path d="M 0 -40 Q -15 0 0 40" fill="none" stroke="#d48110" stroke-width="2"/>
        <path d="M 0 -40 Q 15 0 0 40" fill="none" stroke="#d48110" stroke-width="2"/>
        <!-- Swirls holding it up -->
        <path d="M -30 -10 C -60 -40, -60 -80, -10 -60" fill="none" stroke="url(#trophyGrad)" stroke-width="6" stroke-linecap="round"/>
        <path d="M 30 -10 C 60 -40, 60 -80, 10 -60" fill="none" stroke="url(#trophyGrad)" stroke-width="6" stroke-linecap="round"/>
    </g>
    <!-- Confetti / Stars -->
    <g fill="#ffd700" opacity="0.7">
        <circle cx="150" cy="100" r="3"/>
        <circle cx="250" cy="250" r="2"/>
        <circle cx="650" cy="150" r="4"/>
        <circle cx="550" cy="350" r="3"/>
        <circle cx="100" cy="450" r="2"/>
        <circle cx="700" cy="500" r="3"/>
        <circle cx="300" cy="80" r="2"/>
        <circle cx="500" cy="120" r="4"/>
        <circle cx="600" cy="280" r="2"/>
        <circle cx="200" cy="400" r="3"/>
    </g>
    <!-- Text: FIFA WORLD CUP -->
    <text x="400" y="370" font-family="'Arial Black', Gadget, sans-serif" font-size="36" font-weight="bold" fill="url(#textGrad)" letter-spacing="3" text-anchor="middle" opacity="0.9">FIFA WORLD CUP</text>
    <!-- Text: 2026 -->
    <text x="400" y="470" font-family="'Arial Black', Gadget, sans-serif" font-size="110" font-weight="900" fill="url(#textGold)" letter-spacing="-2" text-anchor="middle" filter="url(#shadow)">2026</text>
    <!-- Text: USA • CANADA • MEXICO -->
    <text x="400" y="510" font-family="'Arial', sans-serif" font-size="18" font-weight="bold" fill="#cccccc" letter-spacing="4" text-anchor="middle">USA &#8226; CANADA &#8226; MEXICO</text>
    <!-- Abstract Host Flag Colors -->
    <g transform="translate(400, 540)">
        <!-- USA Abstract -->
        <g transform="translate(-90, 0)">
            <rect x="-12" y="-8" width="8" height="16" fill="#b22234"/>
            <rect x="-4" y="-8" width="8" height="16" fill="#ffffff"/>
            <rect x="4" y="-8" width="8" height="16" fill="#3c3b6e"/>
        </g>
        <!-- Canada Abstract -->
        <g transform="translate(0, 0)">
            <rect x="-12" y="-8" width="8" height="16" fill="#ff0000"/>
            <rect x="-4" y="-8" width="8" height="16" fill="#ffffff"/>
            <rect x="4" y="-8" width="8" height="16" fill="#ff0000"/>
        </g>
        <!-- Mexico Abstract -->
        <g transform="translate(90, 0)">
            <rect x="-12" y="-8" width="8" height="16" fill="#006847"/>
            <rect x="-4" y="-8" width="8" height="16" fill="#ffffff"/>
            <rect x="4" y="-8" width="8" height="16" fill="#ce1126"/>
        </g>
    </g>
</svg>
```

</details>

## 7. `gpt-oss:120b-cloud`

**SVG size:** 2624 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 5  
**Raw response:** 4267 characters  
**Generation time:** 17.8s

![gpt-oss:120b-cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/gpt-oss_120b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Uses `<use>` references to reuse defined elements -- ideal for repeating flag patterns or stars
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Includes stars -- the model added celebratory/tournament decoration

With 2,624 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg
     viewBox="0 0 800 600"
     xmlns="http://www.w3.org/2000/svg"
     role="img"
     aria-labelledby="title desc" width="100%" height="auto">
  <title id="title">FIFA World Cup 2026</title>
  <desc id="desc">A stylised illustration featuring a soccer ball surrounded by 48 stars and the text “FIFA WORLD CUP 2026”.</desc>
  <!-- Background gradient (blue → teal) -->
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0d47a1"/>
      <stop offset="100%" stop-color="#00695c"/>
    </linearGradient>
    <!-- Simple star shape (5‑point) -->
    <polygon id="star" points="0,-10 2.94,-3.09 9.51,-3.09 4.28,1.18 6.18,8.09 0,4  -6.18,8.09 -4.28,1.18 -9.51,-3.09 -2.94,-3.09"
             fill="#ffd700"/>
  </defs>
  <rect width="800" height="600" fill="url(#bgGrad)"/>
  <!-- Central soccer ball (simplified) -->
  <g transform="translate(400,260) scale(1.8)">
    <!-- Outer circle -->
    <circle r="70" fill="#fff" stroke="#000" stroke-width="4"/>
    <!-- Hexagon pattern – just a few to give the look -->
    <g stroke="#000" stroke-width="2" fill="none">
      <polygon points="0,-60 52,-30 52,30 0,60 -52,30 -52,-30"/>
      <polygon points="0,-30 45,0 0,30 -45,0"/>
      <polygon points="-30,-52 30,-52 30,52 -30,52"/>
    </g>
  </g>
  <!-- Ring of 48 stars (8‑pointed circles of 6 stars each) -->
  <g id="starRing" transform="translate(400,260)">
    <!-- radius of the ring -->
    <g id="stars" transform="rotate(0) translate(0,-140)">
      <use href="#star"/>
    </g>
    <!-- Duplicate the star 48 times, rotating each copy -->
    <script type="application/ecmascript"><![CDATA[
      const svgns = "http://www.w3.org/2000/svg";
      const ring = document.currentScript.parentNode;
      const base = ring.querySelector('#stars');
      for (let i = 1; i < 48; i++) {
        const copy = base.cloneNode(true);
        const angle = (360/48) * i;
        copy.setAttribute('transform', `rotate(${angle}) translate(0,-140)`);
        ring.appendChild(copy);
      }
    ]]></script>
  </g>
  <!-- Tournament text -->
  <text x="400" y="540"
        font-family="Arial, Helvetica, sans-serif"
        font-size="48"
        font-weight="bold"
        fill="#fff"
        text-anchor="middle"
        letter-spacing="2">
    FIFA WORLD CUP 2026
  </text>
  <!-- Small sub‑text for host region (just decorative, not official) -->
  <text x="400" y="580"
        font-family="Arial, Helvetica, sans-serif"
        font-size="24"
        fill="#e0e0e0"
        text-anchor="middle">
    North America (USA · Canada · Mexico)
  </text>
</svg>
```

</details>

## 8. `kimi-k2.6:cloud`

**SVG size:** 7048 characters  
**Complexity:** High  
**Shape elements:** 19  
**Distinct colors:** 8  
**Raw response:** 10560 characters  
**Generation time:** 24.3s

![kimi-k2.6:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/kimi-k2.6_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders a soccer ball -- the model captured the sport's central object
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Includes stars -- the model added celebratory/tournament decoration

With 7,048 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet" width="100%" height="auto">
            <!-- DEFINITIONS: Gradients, Filters, and Patterns -->
            <defs>
                <!-- Gold Gradient for Outer Rim -->
                <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#bf953f" />
                    <stop offset="25%" stop-color="#fcf6ba" />
                    <stop offset="50%" stop-color="#b38728" />
                    <stop offset="75%" stop-color="#fbf5b7" />
                    <stop offset="100%" stop-color="#aa771c" />
                </linearGradient>
                <!-- Metallic Sheen for Inner Shield -->
                <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="#ffffff" />
                    <stop offset="100%" stop-color="#cbd5e1" />
                </linearGradient>
                <!-- Dark Blue Background Gradient -->
                <radialGradient id="darkBlueGrad" cx="50%" cy="50%" r="70%">
                    <stop offset="0%" stop-color="#1e40af" />
                    <stop offset="100%" stop-color="#0f172a" />
                </radialGradient>
                <!-- 3D Ball Effect -->
                <radialGradient id="ballGrad" cx="35%" cy="35%" r="60%">
                    <stop offset="0%" stop-color="#ffffff" />
                    <stop offset="50%" stop-color="#e2e8f0" />
                    <stop offset="100%" stop-color="#64748b" />
                </radialGradient>
                <!-- Filter for Gold Bevel/Emboss -->
                <filter id="bevel" x="-20%" y="-20%" width="140%" height="140%">
                    <feGaussianBlur in="SourceAlpha" stdDeviation="2" result="blur"/>
                    <feOffset in="blur" dx="1" dy="1" result="offsetBlur"/>
                    <feSpecularLighting in="blur" surfaceScale="5" specularConstant=".75" specularExponent="20" lighting-color="#ffffff" result="specOut">
                        <fePointLight x="-5000" y="-10000" z="20000"/>
                    </feSpecularLighting>
                    <feComposite in="specOut" in2="SourceAlpha" operator="in" result="specOut"/>
                    <feComposite in="SourceGraphic" in2="specOut" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="litPaint"/>
                    <feMerge>
                        <feMergeNode in="offsetBlur"/>
                        <feMergeNode in="litPaint"/>
                    </feMerge>
                </filter>
                <!-- Drop Shadow -->
                <filter id="dropShadow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur in="SourceAlpha" stdDeviation="4"/>
                    <feOffset dx="2" dy="4" result="offsetblur"/>
                    <feComponentTransfer>
                        <feFuncA type="linear" slope="0.5"/>
                    </feComponentTransfer>
                    <feMerge> 
                        <feMergeNode/>
                        <feMergeNode in="SourceGraphic"/> 
                    </feMerge>
                </filter>
            </defs>
            <!-- 1. OUTER RING (The Frame) -->
            <circle cx="300" cy="300" r="280" fill="url(#goldGrad)" stroke="#785c18" stroke-width="2" filter="url(#bevel)" />
            <!-- 2. INNER SHIELD BACKGROUND -->
            <circle cx="300" cy="300" r="265" fill="url(#darkBlueGrad)" stroke="#fff" stroke-width="1" />
            <!-- 3. DECORATIVE ACCENTS (Representing 3 Hosts/US, Mexico, Canada) -->
            <g opacity="0.6">
                <!-- Top Star -->
                <path d="M300 40 L308 65 L335 65 L313 82 L321 107 L300 92 L279 107 L287 82 L265 65 L292 65 Z" fill="#ef4444" />
                <!-- Bottom Left Star -->
                <path d="M60 440 L68 465 L95 465 L73 482 L81 507 L60 492 L39 507 L47 482 L25 465 L52 465 Z" fill="#ffffff" />
                <!-- Bottom Right Star -->
                <path d="M540 440 L548 465 L575 465 L553 482 L561 507 L540 492 L519 507 L527 482 L505 465 L532 465 Z" fill="#3b82f6" />
            </g>
            <!-- 4. TEXT: YEAR -->
            <path id="curveTop" d="M 130,250 A 170,170 0 0,1 470,250" fill="transparent" />
            <text font-size="90" font-weight="900" fill="url(#goldGrad)" text-anchor="middle" letter-spacing="10" filter="url(#dropShadow)">
                <textPath href="#curveTop" startOffset="50%">
                    2026
                </textPath>
            </text>
            <!-- 5. TEXT: FIFA WORLD CUP -->
            <path id="curveBottom" d="M 140,350 A 160,160 0 0,0 460,350" fill="transparent" />
            <text font-size="32" font-weight="bold" fill="#ffffff" text-anchor="middle" letter-spacing="4" font-family="sans-serif">
                <textPath href="#curveBottom" startOffset="50%">
                    FIFA WORLD CUP
                </textPath>
            </text>
            <!-- 6. CENTERPIECE: THE FOOTBALL -->
            <g transform="translate(300, 300) scale(0.9)">
                <!-- Ball Shadow -->
                <circle cx="0" cy="10" r="105" fill="#000" opacity="0.4" filter="url(#blur)" />
                <!-- Ball Base -->
                <circle cx="0" cy="0" r="100" fill="url(#ballGrad)" />
                <!-- Classic Football Pattern (Pentagons/Hexagons) -->
                <!-- Central Pentagon -->
                <path d="M-20 -35 L20 -35 L35 10 L0 40 L-35 10 Z" fill="#0f172a" stroke="#fff" stroke-width="2" />
                <!-- Connecting Strips -->
                <path d="M-20 -35 L-60 -60 L-20 -85 L20 -85 L60 -60 L20 -35 Z" fill="#0f172a" stroke="#fff" stroke-width="2" transform="translate(0,0)" />
                <path d="M35 10 L70 40 L35 80 L0 40 Z" fill="#0f172a" stroke="#fff" stroke-width="2" />
                <path d="M-35 10 L-70 40 L-35 80 L0 40 Z" fill="#0f172a" stroke="#fff" stroke-width="2" />
                <!-- Additional Hexagons for fullness -->
                <path d="M-20 -85 L-70 -60 L-60 0 L-35 10 Z" fill="#0f172a" stroke="#fff" stroke-width="2" />
                <path d="M20 -85 L70 -60 L60 0 L35 10 Z" fill="#0f172a" stroke="#fff" stroke-width="2" />
                <path d="M-70 40 L-100 80 L-50 95 L-35 80 Z" fill="#0f172a" stroke="#fff" stroke-width="2" />
                <path d="M70 40 L100 80 L50 95 L35 80 Z" fill="#0f172a" stroke="#fff" stroke-width="2" />
                <!-- Highlight sheen on ball -->
                <ellipse cx="-40" cy="-40" rx="30" ry="20" fill="#fff" opacity="0.2" transform="rotate(-45 -40 -40)" />
            </g>
            <!-- 7. BOTTOM BANNER (Optional, adds official look) -->
            <path d="M 170 480 Q 300 520 430 480 L 430 510 Q 300 550 170 510 Z" fill="url(#goldGrad)" filter="url(#bevel)" />
            <text x="300" y="508" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle" letter-spacing="2">UNITED • CANADA • MEXICO</text>
        </svg>
```

</details>

## 9. `minimax-m2.7:cloud`

**SVG size:** 4992 characters  
**Complexity:** Medium  
**Shape elements:** 17  
**Distinct colors:** 6  
**Raw response:** 5388 characters  
**Generation time:** 41.6s

![minimax-m2.7:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/minimax-m2.7_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders the FIFA World Cup trophy -- the model captured the iconic symbol of the tournament
- Includes stars -- the model added celebratory/tournament decoration

With 4,992 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Definitions for Gradients and Shadows -->
  <defs>
    <!-- Background Gradient: Dark Night Sky -->
    <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a0b2e;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0d0421;stop-opacity:1" />
    </linearGradient>
    <!-- Gold Gradient for the Trophy -->
    <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#ffe082;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#ffca28;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#ff6f00;stop-opacity:1" />
    </linearGradient>
    <!-- Highlight for Gold Shine -->
    <linearGradient id="goldShine" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#ffffff;stop-opacity:0.4" />
      <stop offset="100%" style="stop-color:#ffffff;stop-opacity:0" />
    </linearGradient>
    <!-- 2026 Branding Colors (Ribbon Gradient) -->
    <linearGradient id="brandGradient1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#00c6ff;stop-opacity:1" /> <!-- Cyan -->
      <stop offset="100%" style="stop-color:#0072ff;stop-opacity:1" /> <!-- Blue -->
    </linearGradient>
    <linearGradient id="brandGradient2" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f72585;stop-opacity:1" /> <!-- Pink -->
      <stop offset="100%" style="stop-color:#7209b7;stop-opacity:1" /> <!-- Purple -->
    </linearGradient>
    <linearGradient id="brandGradient3" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#4cc9f0;stop-opacity:1" /> <!-- Light Blue -->
      <stop offset="100%" style="stop-color:#4361ee;stop-opacity:1" /> <!-- Royal Blue -->
    </linearGradient>
    <!-- Drop Shadow -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
      <feOffset dx="2" dy="4" result="offsetblur"/>
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.3"/>
      </feComponentTransfer>
      <feMerge> 
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/> 
      </feMerge>
    </filter>
  </defs>
  <!-- Background Rectangle -->
  <rect width="100%" height="100%" fill="url(#bgGradient)" />
  <!-- Abstract Ribbon Shapes (Stylized Logo Motif) -->
  <g opacity="0.8">
    <path d="M-50,300 Q150,50 300,300 T650,300" fill="none" stroke="url(#brandGradient2)" stroke-width="40" stroke-linecap="round" />
    <path d="M-50,350 Q150,100 300,350 T650,350" fill="none" stroke="url(#brandGradient3)" stroke-width="30" stroke-linecap="round" />
    <path d="M-50,320 Q150,80 300,320 T650,320" fill="none" stroke="url(#brandGradient1)" stroke-width="20" stroke-linecap="round" />
  </g>
  <!-- Text: Title -->
  <text x="300" y="60" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-weight="bold" font-size="24" fill="#ffffff" letter-spacing="4" opacity="0.8">FIFA WORLD CUP</text>
  <!-- Text: Year -->
  <text x="300" y="360" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-weight="bold" font-size="80" fill="url(#brandGradient1)" style="text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">2026</text>
  <!-- The Golden Trophy -->
  <g transform="translate(300, 200)" filter="url(#shadow)">
    <!-- Base -->
    <polygon points="-40,100 40,100 30,80 -30,80" fill="#bdbdbd" />
    <rect x="-30" y="80" width="60" height="10" fill="#424242" />
    <polygon points="-35,90 35,90 45,100 -45,100" fill="#757575" />
    <!-- Stem -->
    <path d="M-15,80 L-10,40 L10,40 L15,80 Z" fill="url(#goldGradient)" />
    <!-- Cup Body -->
    <path d="M-55,0 C-55,40 55,40 55,0 C55,-45 -55,-45 -55,0 Z" fill="url(#goldGradient)" />
    <!-- Cup Rim -->
    <ellipse cx="0" cy="-45" rx="55" ry="10" fill="#d32f2f" /> <!-- Red inside rim -->
    <ellipse cx="0" cy="-45" rx="55" ry="10" fill="none" stroke="url(#goldGradient)" stroke-width="5" />
    <!-- Handles -->
    <path d="M-55,-5 C-90,-5 -90,35 -55,25" fill="none" stroke="url(#goldGradient)" stroke-width="10" stroke-linecap="round" />
    <path d="M55,-5 C90,-5 90,35 55,25" fill="none" stroke="url(#goldGradient)" stroke-width="10" stroke-linecap="round" />
    <!-- Shine/Highlight on Cup -->
    <path d="M-40,0 C-40,30 0,30 0,0" fill="none" stroke="url(#goldShine)" stroke-width="15" />
  </g>
  <!-- Decorative Stars -->
  <g fill="#ffffff" opacity="0.6">
    <polygon points="100,50 105,65 120,65 108,75 112,90 100,80 88,90 92,75 80,65 95,65" transform="scale(0.6) translate(100, 50)" />
    <polygon points="100,50 105,65 120,65 108,75 112,90 100,80 88,90 92,75 80,65 95,65" transform="scale(0.8) translate(650, 400)" />
    <polygon points="100,50 105,65 120,65 108,75 112,90 100,80 88,90 92,75 80,65 95,65" transform="scale(0.5) translate(900, 200)" />
  </g>
</svg>
```

</details>

## 10. `minimax-m3:cloud`

**SVG size:** 7812 characters  
**Complexity:** High  
**Shape elements:** 58  
**Distinct colors:** 17  
**Raw response:** 8655 characters  
**Generation time:** 24.8s

![minimax-m3:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/minimax-m3_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders the FIFA World Cup trophy -- the model captured the iconic symbol of the tournament
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Renders flags -- the model captured the international/national element of the World Cup
- Includes stars -- the model added celebratory/tournament decoration

With 7,812 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <!-- Background gradient -->
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a2980;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#26d0ce;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1a2980;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="trophyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FFD700;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#FFA500;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#B8860B;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="ballGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#ffffff;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#cccccc;stop-opacity:1" />
    </linearGradient>
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#FFD700;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#FFD700;stop-opacity:0" />
    </radialGradient>
  </defs>
  <!-- Background -->
  <rect width="800" height="600" fill="url(#bgGrad)"/>
  <!-- Background stars/decorative elements -->
  <g opacity="0.3">
    <circle cx="100" cy="80" r="2" fill="white"/>
    <circle cx="200" cy="150" r="1.5" fill="white"/>
    <circle cx="300" cy="50" r="2" fill="white"/>
    <circle cx="500" cy="100" r="1.5" fill="white"/>
    <circle cx="650" cy="60" r="2" fill="white"/>
    <circle cx="750" cy="130" r="1.5" fill="white"/>
    <circle cx="80" cy="450" r="2" fill="white"/>
    <circle cx="180" cy="500" r="1.5" fill="white"/>
    <circle cx="720" cy="480" r="2" fill="white"/>
    <circle cx="600" cy="520" r="1.5" fill="white"/>
  </g>
  <!-- Glow effect behind trophy -->
  <ellipse cx="400" cy="320" rx="200" ry="180" fill="url(#glow)"/>
  <!-- FIFA WORLD CUP Text -->
  <text x="400" y="60" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="white" text-anchor="middle" letter-spacing="4">FIFA WORLD CUP</text>
  <!-- 2026 Big Text -->
  <text x="400" y="520" font-family="Arial Black, sans-serif" font-size="72" font-weight="bold" fill="url(#trophyGrad)" text-anchor="middle" letter-spacing="6" stroke="#8B6914" stroke-width="1">2026</text>
  <!-- Trophy - Base/Stand -->
  <rect x="350" y="440" width="100" height="15" fill="#8B6914" stroke="#5C4A0A" stroke-width="1"/>
  <rect x="360" y="455" width="80" height="25" fill="#A0821C" stroke="#5C4A0A" stroke-width="1"/>
  <!-- Trophy - Stem -->
  <path d="M 380 455 Q 400 445 420 455 L 415 410 L 385 410 Z" fill="url(#trophyGrad)" stroke="#8B6914" stroke-width="1"/>
  <!-- Trophy - Globe (bottom sphere) -->
  <ellipse cx="400" cy="395" rx="45" ry="35" fill="url(#trophyGrad)" stroke="#8B6914" stroke-width="2"/>
  <!-- Trophy - Globe lines (continent patterns) -->
  <g stroke="#8B6914" stroke-width="1" fill="none" opacity="0.6">
    <ellipse cx="400" cy="395" rx="45" ry="15"/>
    <ellipse cx="400" cy="395" rx="20" ry="35"/>
    <line x1="355" y1="395" x2="445" y2="395"/>
    <path d="M 370 380 Q 385 385 400 380 Q 415 385 430 380"/>
    <path d="M 365 410 Q 385 415 400 410 Q 415 415 435 410"/>
  </g>
  <!-- Trophy - Bowl/Top portion -->
  <path d="M 355 360 Q 355 290 400 270 Q 445 290 445 360 Z" fill="url(#trophyGrad)" stroke="#8B6914" stroke-width="2"/>
  <!-- Trophy - Handles (two curved arms) -->
  <path d="M 355 295 Q 320 295 320 330 Q 320 355 345 360" fill="none" stroke="url(#trophyGrad)" stroke-width="8" stroke-linecap="round"/>
  <path d="M 445 295 Q 480 295 480 330 Q 480 355 455 360" fill="none" stroke="url(#trophyGrad)" stroke-width="8" stroke-linecap="round"/>
  <!-- Trophy - Top figures (decorative) -->
  <path d="M 385 270 Q 400 250 415 270" fill="none" stroke="#FFD700" stroke-width="3"/>
  <circle cx="400" cy="255" r="5" fill="#FFD700"/>
  <!-- Football/Soccer Ball on left -->
  <g transform="translate(150, 380)">
    <circle cx="0" cy="0" r="60" fill="url(#ballGrad)" stroke="#333" stroke-width="2"/>
    <!-- Pentagon pattern -->
    <polygon points="0,-25 23,-8 14,22 -14,22 -23,-8" fill="#333" stroke="#222" stroke-width="1"/>
    <!-- Surrounding lines -->
    <path d="M 0 -25 L -23 -8 L -45 -20" fill="none" stroke="#333" stroke-width="2"/>
    <path d="M 0 -25 L 23 -8 L 45 -20" fill="none" stroke="#333" stroke-width="2"/>
    <path d="M 14 22 L 35 35" fill="none" stroke="#333" stroke-width="2"/>
    <path d="M -14 22 L -35 35" fill="none" stroke="#333" stroke-width="2"/>
    <!-- Highlight -->
    <ellipse cx="-15" cy="-20" rx="15" ry="10" fill="white" opacity="0.5"/>
  </g>
  <!-- Football/Soccer Ball on right -->
  <g transform="translate(650, 380)">
    <circle cx="0" cy="0" r="50" fill="url(#ballGrad)" stroke="#333" stroke-width="2"/>
    <polygon points="0,-20 19,-6 11,17 -11,17 -19,-6" fill="#333" stroke="#222" stroke-width="1"/>
    <path d="M 0 -20 L -19 -6 L -37 -16" fill="none" stroke="#333" stroke-width="2"/>
    <path d="M 0 -20 L 19 -6 L 37 -16" fill="none" stroke="#333" stroke-width="2"/>
    <ellipse cx="-12" cy="-15" rx="12" ry="8" fill="white" opacity="0.5"/>
  </g>
  <!-- Host Countries flags/banners -->
  <g transform="translate(400, 540)">
    <!-- USA Flag -->
    <rect x="-150" y="20" width="40" height="25" fill="#B22234" stroke="white" stroke-width="1"/>
    <rect x="-150" y="23" width="40" height="2" fill="white"/>
    <rect x="-150" y="28" width="40" height="2" fill="white"/>
    <rect x="-150" y="33" width="40" height="2" fill="white"/>
    <rect x="-150" y="38" width="40" height="2" fill="white"/>
    <rect x="-150" y="20" width="18" height="13" fill="#3C3B6E"/>
    <!-- Canada Flag -->
    <rect x="-50" y="20" width="40" height="25" fill="white" stroke="#ccc" stroke-width="1"/>
    <rect x="-50" y="20" width="10" height="25" fill="#FF0000"/>
    <rect x="-20" y="20" width="10" height="25" fill="#FF0000"/>
    <path d="M -35 28 L -32 33 L -28 30 L -31 36 L -28 38 L -33 39 L -30 42 L -35 40 L -37 42 L -38 38 L -42 38 L -39 34 L -42 30 L -38 33 Z" fill="#FF0000"/>
    <!-- Mexico Flag -->
    <rect x="50" y="20" width="40" height="25" stroke="#ccc" stroke-width="1"/>
    <rect x="50" y="20" width="13" height="25" fill="#006847"/>
    <rect x="63" y="20" width="14" height="25" fill="white"/>
    <rect x="77" y="20" width="13" height="25" fill="#CE1126"/>
    <ellipse cx="70" cy="32" rx="4" ry="3" fill="#8B4513" opacity="0.6"/>
  </g>
  <!-- Host Countries Labels -->
  <text x="280" y="575" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white" text-anchor="middle">USA</text>
  <text x="380" y="575" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white" text-anchor="middle">CANADA</text>
  <text x="480" y="575" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="white" text-anchor="middle">MEXICO</text>
  <!-- Connecting lines between flags -->
  <line x1="300" y1="530" x2="460" y2="530" stroke="white" stroke-width="1" opacity="0.5" stroke-dasharray="3,3"/>
  <!-- Decorative laurel/leaf elements -->
  <g transform="translate(400, 320)">
    <path d="M -120 0 Q -140 -30 -130 -60 Q -110 -50 -100 -20 Z" fill="#2E7D32" opacity="0.7"/>
    <path d="M 120 0 Q 140 -30 130 -60 Q 110 -50 100 -20 Z" fill="#2E7D32" opacity="0.7"/>
    <path d="M -110 10 Q -125 -10 -115 -35 Q -100 -25 -95 0 Z" fill="#388E3C" opacity="0.7"/>
    <path d="M 110 10 Q 125 -10 115 -35 Q 100 -25 95 0 Z" fill="#388E3C" opacity="0.7"/>
  </g>
  <!-- Bottom tagline -->
  <text x="400" y="590" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle" letter-spacing="2" opacity="0.8">THE GREATEST SHOW ON EARTH</text>
</svg>
```

</details>

## 11. `nemotron-3-super:cloud`

**SVG size:** 1775 characters  
**Complexity:** Compact  
**Shape elements:** 9  
**Distinct colors:** 2  
**Raw response:** 4137 characters  
**Generation time:** 31.8s

![nemotron-3-super:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/nemotron-3-super_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts

With 1,775 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background (transparent) -->
  <!-- Central Symbol: Three interlocking arcs representing USA, Canada, Mexico unity -->
  <path d="M150 50 
           C100 0, 50 100, 150 150 
           C250 100, 200 0, 150 50Z"
        fill="none" stroke="#002B5C" stroke-width="12" stroke-linecap="round"/>
  <path d="M150 250 
           C100 300, 50 200, 150 150 
           C250 200, 200 300, 150 250Z"
        fill="none" stroke="#002B5C" stroke-width="12" stroke-linecap="round"/>
  <path d="M50 150 
           C0 100, 100 50, 150 150 
           C200 50, 300 100, 250 150Z"
        fill="none" stroke="#002B5C" stroke-width="12" stroke-linecap="round"/>
  <!-- Football texture hint (subtle pentagons) -->
  <g fill="#002B5C" opacity="0.08">
    <polygon points="140,110 150,95 160,110 155,120 145,1205,120"/>
    <polygon points="140,190 150,205 160,190 155,180 145,180"/>
    <polygon points="95,140 110,150 95,160 80,150"/>
    <polygon points="205,140 190,150 205,160 220,150"/>
  </g>
  <!-- "2026" Text -->
  <text x="150" y="210" 
        font-family="Arial, Helvetica, sans-serif" 
        font-size="48" 
        font-weight="bold" 
        fill="#002B5C" 
        text-anchor="middle">2026</text>
  <!-- Subtle Laurel Wreath Hint (modern, minimal) -->
  <path d="M80 180 
           Q70 160, 80 140 
           Q90 120, 110 120 
           Q130 120, 140 140 
           Q150 160, 140 180"
        fill="none" stroke="#002B5C" stroke-width="3" opacity="0.3"/>
  <path d="M220 180 
           Q230 160, 220 140 
           Q210 120, 190 120 
           Q170 120, 180 140 
           Q170 160, 180 180"
        fill="none" stroke="#002B5C" stroke-width="3" opacity="0.3"/>
</svg>
```

</details>

## 12. `nemotron-3-ultra:cloud`

**SVG size:** 7733 characters  
**Complexity:** High  
**Shape elements:** 24  
**Distinct colors:** 12  
**Raw response:** 9603 characters  
**Generation time:** 46.0s

![nemotron-3-ultra:cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/nemotron-3-ultra_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders "2026" -- the model captured the year, critical for this specific World Cup
- Renders the FIFA World Cup trophy -- the model captured the iconic symbol of the tournament
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts
- Renders flags -- the model captured the international/national element of the World Cup
- References "We Are 26" -- the model captured the official 2026 tournament slogan

With 7,733 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 450" width="100%" height="auto">
  <!-- Definitions: Gradients, Patterns, Filters -->
  <defs>
    <!-- Main Gold Gradient (Trophy/Accent) -->
    <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#FFD700"/>
      <stop offset="50%" style="stop-color:#F5C300"/>
      <stop offset="100%" style="stop-color:#C59600"/>
    </linearGradient>
    <!-- Dark Metallic Gold (Trophy Depth) -->
    <linearGradient id="darkGoldGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#E8C547"/>
      <stop offset="100%" style="stop-color:#B8860B"/>
    </linearGradient>
    <!-- Host Nation Gradient Background (Red/White/Blue/Green hints) -->
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#002B5C"/>  <!-- Deep Blue (US/Canada) -->
      <stop offset="50%" style="stop-color:#001a3a"/>
      <stop offset="100%" style="stop-color:#003366"/>
    </linearGradient>
    <!-- Red Accent Gradient (Canada/US/Mexico Red) -->
    <linearGradient id="redGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#FF0000"/>
      <stop offset="100%" style="stop-color:#CC0000"/>
    </linearGradient>
    <!-- Green Accent (Mexico) -->
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#006847"/>
      <stop offset="100%" style="stop-color:#004d33"/>
    </linearGradient>
    <!-- Glow Filter for Trophy -->
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <!-- Subtle Texture Pattern -->
    <pattern id="noisePattern" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
      <image href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIj48ZmlsdGVyIGlkPSJub2lzZSIgeD0iMCIgeT0iMCI+PGZlVHVyYnVsZW5jZSBiYXNlRnJlcXVlbmN5PSIwLjkiIG51bU9jdGF2ZXM9IjQiIHN0aXRjaE1ldGhvZD0ic3RpdGNoIiB0eXBlPSJmcmFjdGFsTm9pc2UiLz48ZmVDb2xvck1hdHJpeCB0eXBlPSJzYXR1cmF0ZSIgdmFsdWVzPSIwIDAgMCAwIDAgMCAwIDAgMCAwIDAgMCAwIDEyOCAwIi8+PC9maWx0ZXI+PHJlY3QgeD0iMCIgeT0iMCIgd2lkdGg9IjEwMCIgaGVpZ2h0PSIxMDAiIGZpbHRlcj0idXJsKCNub2lzZSkiIG9wYWNpdHk9IjAuMDMiLz48L3N2Zz4=" width="100" height="100"/>
    </pattern>
  </defs>
  <!-- Background Layer -->
  <rect width="800" height="450" fill="url(#bgGrad)"/>
  <rect width="800" height="450" fill="url(#noisePattern)"/>
  <!-- Decorative Geometric Accents (Host Nation Colors) -->
  <!-- Top Left Red Bar -->
  <rect x="0" y="0" width="12" height="450" fill="url(#redGrad)" opacity="0.15"/>
  <!-- Bottom Green Bar -->
  <rect x="788" y="0" width="12" height="450" fill="url(#greenGrad)" opacity="0.15"/>
  <!-- Central Diamond/Shard Accents (Dynamic Football Motion) -->
  <g opacity="0.08" fill="url(#goldGrad)">
    <polygon points="150,100 180,50 210,100 180,150" transform="rotate(-15 180 100)"/>
    <polygon points="600,350 640,300 680,350 640,400" transform="rotate(20 640 350)"/>
    <polygon points="50,300 80,260 110,300 80,340" transform="rotate(10 80 300)"/>
    <polygon points="720,80 750,40 780,80 750,120" transform="rotate(-5 750 80)"/>
  </g>
  <!-- Main Logo Group: "26" with Trophy -->
  <g transform="translate(120, 60)" filter="url(#glow)">
    <!-- DIGIT '2' -->
    <!-- Custom geometric '2' matching FIFA 2026 sharp style -->
    <path d="M0,0 
             h120 
             a40,40 0 0 1 0,80 
             h-80 
             a40,40 0 0 0 0,80 
             h120 
             v20 
             h-120 
             a60,60 0 0 1 0,-120 
             h80 
             a60,60 0 0 0 0,-120 
             h-120 
             z" 
          fill="url(#goldGrad)" stroke="#FFF" stroke-width="1.5" opacity="0.95"/>
    <!-- Highlight on '2' -->
    <path d="M10,10 h100 a30,30 0 0 1 0,60 h-60" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="3" stroke-linecap="round"/>
    <!-- DIGIT '6' WITH TROPHY INTEGRATED -->
    <g transform="translate(160, 0)">
      <!-- The '6' Bowl Shape (Base of digit) -->
      <path d="M0,100 
               a60,60 0 0 1 120,0 
               v10 
               a70,70 0 0 0 -120,0 
               z" 
            fill="url(#darkGoldGrad)" stroke="#C59600" stroke-width="2"/>
      <!-- The '6' Inner Counter (White/Negative Space) -->
      <ellipse cx="60" cy="140" rx="35" ry="45" fill="url(#bgGrad)" stroke="rgba(0,0,0,0.2)" stroke-width="1"/>
      <!-- TROPHY SILHOUETTE (Replaces the top loop of the '6') -->
      <!-- Trophy Base (sits inside the top of the 6 bowl) -->
      <rect x="45" y="20" width="30" height="15" rx="2" fill="url(#goldGrad)" stroke="#B8860B" stroke-width="1"/>
      <rect x="40" y="35" width="40" height="8" rx="1" fill="#C59600"/>
      <!-- Trophy Cup Body (The iconic two handles) -->
      <path d="M60,35 
               C 30,35  10,90  30,130 
               L 35,135 
               C 50,150  70,150  90,135 
               L 95,130 
               C 115,90  95,35  60,35 Z" 
            fill="url(#goldGrad)" stroke="#B8860B" stroke-width="1.5"/>
      <!-- Trophy Handles (Cutouts/Negative space suggestion) -->
      <path d="M35,60 C 15,70 15,100 35,110" fill="none" stroke="url(#darkGoldGrad)" stroke-width="6" stroke-linecap="round" opacity="0.5"/>
      <path d="M90,60 C 110,70 110,100 90,110" fill="none" stroke="url(#darkGoldGrad)" stroke-width="6" stroke-linecap="round" opacity="0.5"/>
      <!-- Trophy Top Globe/Detail -->
      <ellipse cx="60" cy="28" rx="18" ry="6" fill="url(#goldGrad)" stroke="#E8C547" stroke-width="1"/>
      <circle cx="60" cy="26" r="5" fill="#FFD700" opacity="0.6"/>
      <!-- Highlight on Trophy -->
      <path d="M50,40 C 45,60 45,100 55,120" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="4" stroke-linecap="round"/>
    </g>
  </g>
  <!-- Text: FIFA WORLD CUP 26 -->
  <g transform="translate(120, 320)" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" fill="#FFFFFF">
    <!-- FIFA -->
    <text x="0" y="0" font-size="48" font-weight="700" letter-spacing="8" fill="url(#goldGrad)" paint-order="stroke fill" stroke="#000" stroke-width="0.5">FIFA</text>
    <!-- WORLD CUP -->
    <text x="0" y="60" font-size="36" font-weight="300" letter-spacing="6" fill="#E0E0E0" opacity="0.9">WORLD CUP</text>
    <!-- Host Line -->
    <g transform="translate(0, 110)" font-size="16" font-weight="400" letter-spacing="2" fill="rgba(255,255,255,0.7)">
      <text x="0" y="0">CANADA • MEXICO • USA</text>
      <!-- Small Flag Color Dots -->
      <circle cx="185" cy="-10" r="4" fill="#FF0000"/> <!-- Red for Canada/US -->
      <circle cx="192" cy="-10" r="4" fill="#002B5C"/> <!-- Blue for US -->
      <circle cx="199" cy="-10" r="4" fill="#006847"/> <!-- Green for Mexico -->
      <circle cx="206" cy="-10" r="4" fill="#FFFFFF"/> <!-- White -->
    </g>
    <!-- Year 26 (Small, below text) -->
    <text x="0" y="150" font-size="24" font-weight="700" letter-spacing="4" fill="url(#goldGrad)">2026</text>
  </g>
  <!-- Official Tagline / Slogan Area -->
  <text x="400" y="420" text-anchor="middle" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif" 
        font-size="14" font-weight="300" letter-spacing="3" fill="rgba(255,255,255,0.5)">
    WE ARE 26
  </text>
  <!-- Watermark/Credit -->
  <text x="790" y="440" text-anchor="end" font-family="sans-serif" font-size="9" fill="rgba(255,255,255,0.15)">
    Inspired by FIFA Brand Identity • Generated SVG
  </text>
</svg>
```

</details>

## 13. `qwen3.5:397b-cloud`

**SVG size:** 3391 characters  
**Complexity:** Medium  
**Shape elements:** 7  
**Distinct colors:** 6  
**Raw response:** 3744 characters  
**Generation time:** 37.0s

![qwen3.5:397b-cloud FIFA World Cup 2026 SVG](/assets/img/diagrams/ollama-svg-fifa/qwen3.5_397b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the gold trophy and vibrant backgrounds)
- Applies SVG filters (e.g., drop shadows, glows) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the trophy, ball, and flags
- Renders actual text (FIFA, World Cup, 2026, host nation names) -- a strong sign the model understood the prompt
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders the "FIFA" text -- the model captured the tournament brand
- Renders "World Cup" text -- the model captured the event name
- Renders a soccer ball -- the model captured the sport's central object
- References USA (host nation) -- the model knew the 2026 hosts
- References Canada (host nation) -- the model knew the 2026 hosts
- References Mexico (host nation) -- the model knew the 2026 hosts

With 3,391 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <defs>
    <!-- Gradients matching the official branding vibe -->
    <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#002D72;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#3E64FF;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="grad6" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#DA291C;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#F7941D;stop-opacity:1" />
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="2" dy="4" stdDeviation="3" flood-color="#000" flood-opacity="0.3"/>
    </filter>
  </defs>
  <!-- Background -->
  <rect width="600" height="600" fill="#ffffff" rx="20" />
  <!-- Subtle Map/Abstract Background Pattern -->
  <path d="M50 50 Q 150 150 50 250 T 50 450" fill="none" stroke="#f0f0f0" stroke-width="20" stroke-linecap="round"/>
  <path d="M550 550 Q 450 450 550 350 T 550 150" fill="none" stroke="#f0f0f0" stroke-width="20" stroke-linecap="round"/>
  <!-- The "2" Shape (Stylized Map/Wave) -->
  <path d="M 180 180 
           C 280 180, 320 220, 320 280 
           C 320 360, 240 380, 180 380 
           C 140 380, 120 360, 120 340
           L 160 340
           C 160 350, 180 360, 220 360
           C 280 360, 300 320, 300 280
           C 300 230, 260 200, 180 200
           Z" 
        fill="url(#grad2)" filter="url(#shadow)" />
  <!-- The "6" Shape (Stylized Ball/Loop) -->
  <path d="M 420 280 
           C 420 220, 380 180, 320 180 
           C 260 180, 220 220, 220 280 
           C 220 360, 280 420, 340 420 
           C 400 420, 440 380, 440 340
           C 440 300, 400 280, 360 280
           C 320 280, 300 300, 300 320
           C 300 340, 320 360, 340 360
           C 380 360, 400 340, 400 320
           L 360 320
           C 360 330, 350 340, 340 340
           C 330 340, 320 330, 320 320
           C 320 310, 330 300, 360 300
           C 420 300, 460 320, 460 360
           C 460 420, 400 460, 340 460
           C 260 460, 200 380, 200 280
           C 200 200, 280 140, 360 140
           C 440 140, 500 200, 500 280
           Z" 
        fill="url(#grad6)" filter="url(#shadow)" transform="translate(20, 0)"/>
  <!-- Text: FIFA WORLD CUP -->
  <text x="300" y="500" font-family="Arial, Helvetica, sans-serif" font-weight="900" font-size="24" text-anchor="middle" fill="#002D72" letter-spacing="2">
    FIFA WORLD CUP
  </text>
  <!-- Text: 26 (Large) -->
  <text x="300" y="535" font-family="Arial, Helvetica, sans-serif" font-weight="900" font-size="40" text-anchor="middle" fill="#DA291C" letter-spacing="5">
    26
  </text>
  <!-- Host Nations -->
  <text x="300" y="565" font-family="Arial, Helvetica, sans-serif" font-weight="bold" font-size="14" text-anchor="middle" fill="#555" letter-spacing="1">
    USA • CANADA • MEXICO
  </text>
  <!-- Decorative Soccer Ball Hexagon Pattern (Bottom Right) -->
  <g transform="translate(480, 480) scale(0.4)" opacity="0.1">
     <path d="M50 0 L93.3 25 L93.3 75 L50 100 L6.7 75 L6.7 25 Z" fill="#002D72" stroke="#002D72" stroke-width="2"/>
     <path d="M143.3 25 L186.6 50 L186.6 100 L143.3 125 L100 100 L100 50 Z" fill="#DA291C" stroke="#DA291C" stroke-width="2"/>
  </g>
</svg>
```

</details>

## Conclusion

That's all 13 FIFA World Cup 2026 SVGs from the active Ollama cloud models. This trending-event prompt tested a different skill set than our previous prompts: text rendering (FIFA, World Cup, 2026), brand recognition (the trophy), sports imagery (soccer ball), national identity (flags of USA, Canada, Mexico), and poster composition.

**No single model is best at everything.** A model that produces a stunning octopus may produce a basic World Cup poster, and vice versa. The best model for you depends on your specific use case:

- For **maximum detail**: look at the "Very high" complexity models
- For **speed**: look at the generation times in each section
- For **text accuracy**: check which models actually rendered "FIFA", "World Cup", and "2026" as readable text
- For **brand knowledge**: check which models rendered a recognizable FIFA trophy and knew the 2026 host nations
- For **code quality**: look at the raw SVG source -- clean, well-structured code is easier to customize

Try the same prompt yourself with [Ollama Cloud](https://ollama.com/cloud) and see if you agree with our analysis. And check out the other posts in our SVG benchmark series:

- [Duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/)
- [Duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/)
- [Duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/)
- [Cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/)
- [Duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/)
- [Baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/)
- [Octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/)
