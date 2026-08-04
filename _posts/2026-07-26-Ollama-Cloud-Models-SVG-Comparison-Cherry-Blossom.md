---
layout: post
title: "Which Ollama Cloud Model is Best? Cherry Blossom Trees SVG Comparison (14 Models)"
description: "Compare 14 Ollama cloud models on a nature / scenery prompt: drawing cherry blossom trees with flowers. Find the best LLM for SVG art. You decide the winner."
date: 2026-07-26
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/
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
  - Cherry Blossom
  - Nature Art
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama cloud model comparison, deepseek vs glm vs qwen, LLM SVG benchmark, AI image generation comparison, cherry blossom SVG, sakura SVG, which Ollama model is best, Ollama cloud models 2026, AI nature art, LLM drawing benchmark"
---

# Which Ollama Cloud Model is Best? Cherry Blossom Trees SVG Comparison (14 Models)

After testing LLMs on ducks and vehicles, we wanted to know: **can today's top models draw nature?** This time we asked 14 Ollama cloud models to draw **cherry blossom trees with flowers** -- a softer, more organic prompt that tests color palettes, repeated organic shapes (petals), and scene composition.

The prompt was: `Make an svg image about cherry blossom trees with flowers`.

This is the fourth in our SVG benchmark series. See also:
[duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), and [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/).

**Why cherry blossoms?** Unlike the duck prompts, this scene has no central character. It tests a different skill set: gradient skies, repeating petal shapes, branch structures, and a calm nature palette (pinks, greens, blues). A model that drew a great duck may struggle here, and vice versa -- which is exactly why we run multiple prompts.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Nature SVGs

Nature prompts reward different things than character prompts. Here are the criteria to use:

- **Color palette**: Cherry blossoms demand pinks, soft greens, and sky blues. Does the model use a tasteful palette, or does it dump random colors?
- **Organic shapes**: Look for curved petals, branching trunks, and natural-looking foliage. Models that only draw rectangles and circles struggle here.
- **Repetition handling**: Blossoms require many similar-but-not-identical petals. Models that use `<use>` with transforms scale better than those that hand-draw each petal.
- **Depth and layering**: Does the SVG have a foreground/background, sky gradient, and ground? Layering is what makes nature scenes feel real.
- **SVG code quality**: Does it use `<defs>`, `<use>`, gradients, and filters? Cleaner code is easier to tweak (e.g., to recolor for autumn).
- **Output size**: Very small SVGs miss the detail that makes blossoms beautiful. Very large SVGs can be slow but are usually worth it for nature scenes.

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with zero post-processing.

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary Table: Compare All 14 Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `deepseek-v4-flash_cloud` | 16296 | 52 | 14 | Very high | Rich detail, fast |
| 2 | `deepseek-v4-pro_cloud` | 13680 | 49 | 15 | High | Most detailed |
| 3 | `gemma4_31b-cloud` | 3109 | 29 | 6 | Low | Compact |
| 4 | `gemma4_cloud` | 2649 | 28 | 9 | Low | Minimalist |
| 5 | `glm-5.1_cloud` | 19810 | 69 | 21 | Very high | Most detailed |
| 6 | `glm-5.2_cloud` | 15956 | 52 | 27 | Very high | Highly detailed |
| 7 | `gpt-oss_120b-cloud` | 3493 | 9 | 6 | Low | Compact |
| 8 | `kimi-k2.6_cloud` | 1678 | 3 | 7 | Low | Minimalist |
| 9 | `minimax-m2.7_cloud` | 3319 | 20 | 11 | Low | Compact |
| 10 | `minimax-m3_cloud` | 15741 | 171 | 15 | Very high | Highly detailed |
| 11 | `nemotron-3-super_cloud` | 2648 | 42 | 3 | Low | Compact |
| 12 | `nemotron-3-ultra_cloud` | 16411 | 32 | 20 | Very high | Richest scene |
| 13 | `qwen3.5_397b-cloud` | 4458 | 25 | 8 | Low | Balanced |
| 14 | `deepseek-v4-flash_0731-cloud` | 15165 | 152 | 9 | Very high | Richest scene |
| 15 | `bjoernb/claude-opus-4-5:latest` | - | - | - | - | Retired (410) |
| 16 | `deepseek-v3.1:671b-cloud` | - | - | - | - | Retired (410) |
| 17 | `glm-5:cloud` | - | - | - | - | Retired (410) |
| 18 | `qwen3-vl:235b-cloud` | - | - | - | - | Retired (410) |

**14 out of 18** active models produced a valid SVG. The 4 retired models returned HTTP 410 Gone (removed from Ollama Cloud on 2026-07-15).

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most detailed, painterly cherry blossom SVG**: pick `glm-5.1:cloud`, `glm-5.2:cloud`, `deepseek-v4-flash:cloud`, or `nemotron-3-ultra:cloud`
- **You want the fastest response**: pick `gpt-oss:120b-cloud` (~11s) or `nemotron-3-super:cloud` (~17s)
- **You want the cleanest, most reusable SVG code**: pick `deepseek-v4-pro:cloud` (uses `<defs>`, `<use>`, transforms)
- **You want a small, efficient SVG for web embedding**: pick `gemma4:cloud` or `kimi-k2.6:cloud`
- **You want a balance of detail and speed**: pick `qwen3.5:397b-cloud` or `minimax-m3:cloud`
- **You want to compare within a model family**: pick `deepseek-v4-pro` vs `deepseek-v4-flash`, or `glm-5.1` vs `glm-5.2`, or `minimax-m2.7` vs `minimax-m3`

Now read on for the full per-model breakdown and judge for yourself.

## 1. deepseek-v4-flash_cloud

**SVG size:** 16296 characters  
**Complexity:** Very high  
**Shape elements:** 52  
**Distinct colors:** 14  
**Raw response:** 16702 characters

![deepseek-v4-flash_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/deepseek-v4-flash_cloud.svg)

### Analysis

This SVG contains approximately **52 shape elements** and uses **14 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating petals
- Includes gradient fills for richer visual depth (great for skies and petals)
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

With over 15,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering individual petals, layered branches, and atmospheric backgrounds. Best for users who want a painterly cherry blossom scene.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
    <defs>
        <!-- Background Gradient -->
        <linearGradient id="bg-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#f0f8ff" />
            <stop offset="100%" stop-color="#ffe4e1" />
        </linearGradient>

        <!-- Sun Glow Gradient -->
        <radialGradient id="sun-glow" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0%" stop-color="#fff9c4" stop-opacity="0.8" />
            <stop offset="50%" stop-color="#fff9c4" stop-opacity="0.3" />
            <stop offset="100%" stop-color="#fff9c4" stop-opacity="0" />
        </radialGradient>

        <!-- Petal Base -->
        <g id="petal">
            <path d="M 0,0 C -15,-20 -20,-40 0,-50 C 20,-40 15,-20 0,0 Z" fill="currentColor" />
        </g>

        <!-- Blossom Type 1 (Deep Pink) -->
        <g id="blossom1">
            <use href="#petal" color="#f06292" />
            <use href="#petal" color="#f06292" transform="rotate(72)" />
            <use href="#petal" color="#f06292" transform="rotate(144)" />
            <use href="#petal" color="#f06292" transform="rotate(216)" />
            <use href="#petal" color="#f06292" transform="rotate(288)" />
            <circle cx="0" cy="0" r="6" fill="#fce4ec" />
            <circle cx="0" cy="-2" r="2" fill="#ffeb3b" />
            <circle cx="2" cy="1.5" r="2" fill="#ffeb3b" />
            <circle cx="-2" cy="1.5" r="2" fill="#ffeb3b" />
        </g>

        <!-- Blossom Type 2 (Light Pink) -->
        <g id="blossom2">
            <use href="#petal" color="#f8bbd0" />
            <use href="#petal" color="#f8bbd0" transform="rotate(72)" />
            <use href="#petal" color="#f8bbd0" transform="rotate(144)" />
            <use href="#petal" color="#f8bbd0" transform="rotate(216)" />
            <use href="#petal" color="#f8bbd0" transform="rotate(288)" />
            <circle cx="0" cy="0" r="6" fill="#f06292" />
            <circle cx="0" cy="-2" r="2" fill="#ffeb3b" />
            <circle cx="2" cy="1.5" r="2" fill="#ffeb3b" />
            <circle cx="-2" cy="1.5" r="2" fill="#ffeb3b" />
        </g>

        <!-- Blossom Type 3 (Pale Pink) -->
        <g id="blossom3">
            <use href="#petal" color="#fff0f5" />
            <use href="#petal" color="#fff0f5" transform="rotate(72)" />
            <use href="#petal" color="#fff0f5" transform="rotate(144)" />
            <use href="#petal" color="#fff0f5" transform="rotate(216)" />
            <use href="#petal" color="#fff0f5" transform="rotate(288)" />
            <circle cx="0" cy="0" r="6" fill="#ec407a" />
            <circle cx="0" cy="-2" r="2" fill="#ffeb3b" />
            <circle cx="2" cy="1.5" r="2" fill="#ffeb3b" />
            <circle cx="-2" cy="1.5" r="2" fill="#ffeb3b" />
        </g>

        <!-- Bud -->
        <g id="bud">
            <path d="M 0,0 C -10,-15 -15,-25 0,-30 C 15,-25 10,-15 0,0 Z" fill="currentColor" />
            <path d="M 0,0 C -5,-10 -8,-20 0,-25 C 8,-20 5,-10 0,0 Z" fill="#fce4ec" transform="rotate(30)" />
            <path d="M 0,0 C -5,-10 -8,-20 0,-25 C 8,-20 5,-10 0,0 Z" fill="#fce4ec" transform="rotate(-30)" />
        </g>

        <!-- Leaf -->
        <g id="leaf">
            <path d="M 0,0 C 10,-10 20,-15 30,-5 C 20,5 10,10 0,0 Z" fill="#8d6e63" />
            <path d="M 0,0 C 10,-10 20,-15 30,-5" stroke="#4e342e" stroke-width="1" fill="none" />
        </g>

        <!-- Blossom Cluster 1 -->
        <g id="c1">
            <use href="#blossom1" x="0" y="0" />
            <use href="#blossom2" x="20" y="-15" transform="scale(0.8) rotate(30)" />
            <use href="#blossom3" x="-15" y="-20" transform="scale(0.9) rotate(-45)" />
            <use href="#bud" x="-25" y="5" color="#f06292" transform="rotate(-20)" />
            <use href="#bud" x="25" y="10" color="#f8bbd0" transform="rotate(20)" />
            <use href="#leaf" x="10" y="20" transform="rotate(30)" />
        </g>

        <!-- Blossom Cluster 2 -->
        <g id="c2">
            <use href="#blossom2" x="0" y="0" />
            <use href="#blossom1" x="-20" y="-10" transform="scale(0.7) rotate(15)" />
            <use href="#blossom3" x="15" y="-25" transform="scale(1.1) rotate(-60)" />
            <use href="#bud" x="0" y="-35" color="#ec407a" transform="scale(0.8) rotate(45)" />
            <use href="#leaf" x="-15" y="15" transform="rotate(-45)" />
        </g>

        <!-- Blossom Cluster 3 -->
        <g id="c3">
            <use href="#blossom3" x="0" y="0" />
            <use href="#blossom1" x="25" y="-5" transform="scale(0.9) rotate(80)" />
            <use href="#blossom2" x="-20" y="15" transform="scale(0.8) rotate(-15)" />
            <use href="#bud" x="10" y="25" color="#f06292" transform="rotate(90)" />
            <use href="#bud" x="-10" y="-25" color="#f8bbd0" transform="scale(0.9) rotate(-90)" />
            <use href="#leaf" x="20" y="-10" transform="rotate(120)" />
        </g>
    </defs>

    <!-- Background -->
    <rect width="800" height="600" fill="url(#bg-grad)" />

    <!-- Sun Glow -->
    <circle cx="400" cy="300" r="250" fill="url(#sun-glow)" />

    <!-- Ground -->
    <g>
        <path d="M 0,550 Q 200,520 400,550 T 800,550 L 800,600 L 0,600 Z" fill="#aed581" opacity="0.6" />
        <path d="M 0,570 Q 300,540 600,580 T 800,580 L 800,600 L 0,600 Z" fill="#81c784" opacity="0.8" />
    </g>

    <!-- Tree Branches -->
    <g fill="none" stroke="#4e342e" stroke-linecap="round" stroke-linejoin="round">
        <!-- Trunk -->
        <path d="M 400,600 C 380,500 390,400 400,300" stroke-width="28" />
        <!-- Trunk Texture -->
        <path d="M 390,550 C 380,500 385,450 395,400" stroke="#6d4c41" stroke-width="4" />
        <path d="M 405,520 C 400,480 405,430 410,380" stroke="#6d4c41" stroke-width="3" />
        
        <!-- Main Left Branch -->
        <path d="M 390,450 C 320,400 250,350 200,250" stroke-width="18" />
        <path d="M 320,400 C 250,380 180,350 120,280" stroke-width="12" />
        <path d="M 250,350 C 200,320 150,280 100,200" stroke-width="10" />
        <path d="M 200,250 C 150,200 120,150 80,100" stroke-width="8" />
        <path d="M 120,280 C 80,250 50,200 40,150" stroke-width="6" />
        
        <!-- Weeping Left Branches -->
        <path d="M 200,250 C 180,300 150,350 120,400" stroke-width="6" />
        <path d="M 150,280 C 130,320 100,360 80,420" stroke-width="4" />
        <path d="M 100,200 C 80,250 60,300 40,350" stroke-width="4" />
        
        <!-- Main Right Branch -->
        <path d="M 400,450 C 480,400 550,350 600,250" stroke-width="18" />
        <path d="M 480,400 C 550,380 620,350 680,280" stroke-width="12" />
        <path d="M 550,350 C 600,320 650,280 700,200" stroke-width="10" />
        <path d="M 600,250 C 650,200 680,150 720,100" stroke-width="8" />
        <path d="M 680,280 C 720,250 750,200 760,150" stroke-width="6" />
        
        <!-- Weeping Right Branches -->
        <path d="M 600,250 C 620,300 650,350 680,400" stroke-width="6" />
        <path d="M 650,280 C 670,320 700,360 720,420" stroke-width="4" />
        <path d="M 700,200 C 720,250 740,300 760,350" stroke-width="4" />
        
        <!-- Center Branches -->
        <path d="M 400,300 C 400,200 400,150 400,80" stroke-width="14" />
        <path d="M 400,300 C 350,250 300,200 250,120" stroke-width="10" />
        <path d="M 400,300 C 450,250 500,200 550,120" stroke-width="10" />
        <path d="M 400,200 C 380,250 360,300 340,350" stroke-width="6" />
        <path d="M 400,200 C 420,250 440,300 460,350" stroke-width="6" />
        
        <!-- Additional Twigs -->
        <path d="M 250,120 C 220,80 200,60 180,40" stroke-width="4" />
        <path d="M 550,120 C 580,80 600,60 620,40" stroke-width="4" />
        <path d="M 400,80 C 380,50 360,30 340,20" stroke-width="4" />
        <path d="M 400,80 C 420,50 440,30 460,20" stroke-width="4" />
        <path d="M 300,200 C 280,160 260,140 240,120" stroke-width="4" />
        <path d="M 500,200 C 520,160 540,140 560,120" stroke-width="4" />
    </g>

    <!-- Blossoms and Clusters -->
    <g>
        <!-- Left Side Clusters -->
        <use href="#c1" x="200" y="250" transform="scale(0.8) rotate(15)" />
        <use href="#c2" x="180" y="280" transform="scale(0.9) rotate(-20)" />
        <use href="#c3" x="150" y="300" transform="scale(1) rotate(45)" />
        <use href="#c1" x="120" y="280" transform="scale(0.7) rotate(-10)" />
        <use href="#c2" x="100" y="200" transform="scale(0.9) rotate(30)" />
        <use href="#c3" x="80" y="100" transform="scale(0.8) rotate(-40)" />
        <use href="#c1" x="120" y="400" transform="scale(0.6) rotate(60)" />
        <use href="#c2" x="100" y="350" transform="scale(0.7) rotate(-15)" />
        <use href="#c3" x="250" y="120" transform="scale(0.9) rotate(80)" />
        <use href="#c1" x="220" y="80" transform="scale(0.7) rotate(-30)" />
        <use href="#c2" x="40" y="150" transform="scale(0.6) rotate(10)" />
        <use href="#c3" x="80" y="420" transform="scale(0.5) rotate(-60)" />
        <use href="#c1" x="40" y="350" transform="scale(0.5) rotate(25)" />

        <!-- Right Side Clusters -->
        <use href="#c1" x="600" y="250" transform="scale(0.8) rotate(-15)" />
        <use href="#c2" x="620" y="280" transform="scale(0.9) rotate(20)" />
        <use href="#c3" x="650" y="300" transform="scale(1) rotate(-45)" />
        <use href="#c1" x="680" y="280" transform="scale(0.7) rotate(10)" />
        <use href="#c2" x="700" y="200" transform="scale(0.9) rotate(-30)" />
        <use href="#c3" x="720" y="100" transform="scale(0.8) rotate(40)" />
        <use href="#c1" x="680" y="400" transform="scale(0.6) rotate(-60)" />
        <use href="#c2" x="700" y="350" transform="scale(0.7) rotate(15)" />
        <use href="#c3" x="550" y="120" transform="scale(0.9) rotate(-80)" />
        <use href="#c1" x="580" y="80" transform="scale(0.7) rotate(30)" />
        <use href="#c2" x="760" y="150" transform="scale(0.6) rotate(-10)" />
        <use href="#c3" x="720" y="420" transform="scale(0.5) rotate(60)" />
        <use href="#c1" x="760" y="350" transform="scale(0.5) rotate(-25)" />

        <!-- Center Clusters -->
        <use href="#c1" x="400" y="300" transform="scale(1) rotate(5)" />
        <use href="#c2" x="400" y="200" transform="scale(0.9) rotate(-15)" />
        <use href="#c3" x="400" y="150" transform="scale(0.8) rotate(35)" />
        <use href="#c1" x="400" y="80" transform="scale(0.7) rotate(-55)" />
        <use href="#c2" x="350" y="250" transform="scale(0.8) rotate(25)" />
        <use href="#c3" x="300" y="200" transform="scale(0.9) rotate(-35)" />
        <use href="#c1" x="250" y="120" transform="scale(0.7) rotate(65)" />
        <use href="#c2" x="450" y="250" transform="scale(0.8) rotate(-25)" />
        <use href="#c3" x="500" y="200" transform="scale(0.9) rotate(35)" />
        <use href="#c1" x="550" y="120" transform="scale(0.7) rotate(-65)" />
        <use href="#c2" x="380" y="250" transform="scale(0.6) rotate(15)" />
        <use href="#c3" x="420" y="250" transform="scale(0.6) rotate(-15)" />
        <use href="#c1" x="340" y="350" transform="scale(0.5) rotate(45)" />
        <use href="#c2" x="460" y="350" transform="scale(0.5) rotate(-45)" />
        <use href="#c3" x="180" y="40" transform="scale(0.5) rotate(20)" />
        <use href="#c1" x="620" y="40" transform="scale(0.5) rotate(-20)" />
        <use href="#c2" x="340" y="20" transform="scale(0.4) rotate(10)" />
        <use href="#c3" x="460" y="20" transform="scale(0.4) rotate(-10)" />

        <!-- Individual Blossoms and Buds to fill gaps -->
        <use href="#blossom1" x="160" y="260" transform="scale(0.8)" />
        <use href="#blossom2" x="140" y="290" transform="scale(0.7) rotate(45)" />
        <use href="#blossom3" x="110" y="240" transform="scale(0.9) rotate(-30)" />
        <use href="#bud" x="90" y="180" color="#f06292" transform="scale(0.8) rotate(15)" />
        <use href="#blossom1" x="60" y="120" transform="scale(0.6) rotate(60)" />
        <use href="#blossom2" x="130" y="380" transform="scale(0.5) rotate(-20)" />
        <use href="#bud" x="110" y="360" color="#f8bbd0" transform="scale(0.7) rotate(40)" />
        
        <use href="#blossom1" x="640" y="260" transform="scale(0.8)" />
        <use href="#blossom2" x="660" y="290" transform="scale(0.7) rotate(-45)" />
        <use href="#blossom3" x="690" y="240" transform="scale(0.9) rotate(30)" />
        <use href="#bud" x="710" y="180" color="#ec407a" transform="scale(0.8) rotate(-15)" />
        <use href="#blossom1" x="740" y="120" transform="scale(0.6) rotate(-60)" />
        <use href="#blossom2" x="670" y="380" transform="scale(0.5) rotate(20)" />
        <use href="#bud" x="690" y="360" color="#f06292" transform="scale(0.7) rotate(-40)" />
        
        <use href="#blossom3" x="380" y="180" transform="scale(0.8) rotate(15)" />
        <use href="#blossom1" x="420" y="180" transform="scale(0.8) rotate(-15)" />
        <use href="#bud" x="360" y="220" color="#f8bbd0" transform="scale(0.9) rotate(30)" />
        <use href="#bud" x="440" y="220" color="#ec407a" transform="scale(0.9) rotate(-30)" />
        <use href="#blossom2" x="320" y="160" transform="scale(0.7) rotate(45)" />
        <use href="#blossom3" x="480" y="160" transform="scale(0.7) rotate(-45)" />
        <use href="#blossom1" x="280" y="100" transform="scale(0.6) rotate(10)" />
        <use href="#blossom2" x="520" y="100" transform="scale(0.6) rotate(-10)" />
    </g>

    <!-- Falling Petals -->
    <g>
        <use href="#petal" color="#f06292" transform="translate(100, 150) rotate(45) scale(0.8)" />
        <use href="#petal" color="#f8bbd0" transform="translate(200, 100) rotate(-30) scale(0.7)" />
        <use href="#petal" color="#fff0f5" transform="translate(300, 50) rotate(15) scale(0.9)" />
        <use href="#petal" color="#f06292" transform="translate(500, 80) rotate(-60) scale(0.6)" />
        <use href="#petal" color="#f8bbd0" transform="translate(600, 120) rotate(75) scale(0.8)" />
        <use href="#petal" color="#fff0f5" transform="translate(700, 60) rotate(-15) scale(0.7)" />
        
        <use href="#petal" color="#f06292" transform="translate(50, 300) rotate(120) scale(0.9)" />
        <use href="#petal" color="#f8bbd0" transform="translate(150, 450) rotate(-45) scale(0.6)" />
        <use href="#petal" color="#fff0f5" transform="translate(250, 500) rotate(30) scale(0.8)" />
        <use href="#petal" color="#f06292" transform="translate(350, 400) rotate(-90) scale(0.7)" />
        <use href="#petal" color="#f8bbd0" transform="translate(450, 480) rotate(15) scale(0.9)" />
        <use href="#petal" color="#fff0f5" transform="translate(550, 520) rotate(-75) scale(0.6)" />
        <use href="#petal" color="#f06292" transform="translate(650, 450) rotate(60) scale(0.8)" />
        <use href="#petal" color="#f8bbd0" transform="translate(750, 300) rotate(-120) scale(0.7)" />
        
        <use href="#petal" color="#fff0f5" transform="translate(80, 200) rotate(10) scale(0.5)" />
        <use href="#petal" color="#f06292" transform="translate(180, 350) rotate(80) scale(0.6)" />
        <use href="#petal" color="#f8bbd0" transform="translate(280, 250) rotate(-20) scale(0.7)" />
        <use href="#petal" color="#fff0f5" transform="translate(480, 350) rotate(40) scale(0.5)" />
        <use href="#petal" color="#f06292" transform="translate(580, 250) rotate(-80) scale(0.6)" />
        <use href="#petal" color="#f8bbd0" transform="translate(680, 150) rotate(20) scale(0.7)" />
        
        <use href="#petal" color="#f06292" transform="translate(120, 550) rotate(30) scale(0.8)" />
        <use href="#petal" color="#f8bbd0" transform="translate(220, 580) rotate(-45) scale(0.9)" />
        <use href="#petal" color="#fff0f5" transform="translate(320, 560) rotate(15) scale(0.7)" />
        <use href="#petal" color="#f06292" transform="translate(520, 570) rotate(75) scale(0.8)" />
        <use href="#petal" color="#f8bbd0" transform="translate(620, 590) rotate(-20) scale(0.9)" />
        <use href="#petal" color="#fff0f5" transform="translate(720, 560) rotate(60) scale(0.7)" />
    </g>
</svg>
```

</details>

## 2. deepseek-v4-pro_cloud

**SVG size:** 13680 characters  
**Complexity:** High  
**Shape elements:** 49  
**Distinct colors:** 15  
**Raw response:** 14073 characters

![deepseek-v4-pro_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/deepseek-v4-pro_cloud.svg)

### Analysis

This SVG contains approximately **49 shape elements** and uses **15 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating petals
- Includes gradient fills for richer visual depth (great for skies and petals)
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

With 10,000-15,000 characters, this is a highly detailed SVG. The model added many petals and scene elements while keeping the markup structured. A strong choice if you want detail without overwhelming complexity.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <!-- Gradients -->
        <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#87CEEB"/>
            <stop offset="60%" stop-color="#FFE4E1"/>
            <stop offset="100%" stop-color="#FFF0F5"/>
        </linearGradient>
        
        <linearGradient id="trunkGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#1a0f0a"/>
            <stop offset="50%" stop-color="#3b2b1a"/>
            <stop offset="100%" stop-color="#1a0f0a"/>
        </linearGradient>

        <!-- Symbols -->
        <g id="flower">
            <!-- Petals -->
            <circle cx="0" cy="-6" r="8" fill="#ffb7c5" opacity="0.9"/>
            <circle cx="6" cy="-2" r="8" fill="#ffb7c5" opacity="0.9"/>
            <circle cx="4" cy="7" r="8" fill="#ffb7c5" opacity="0.9"/>
            <circle cx="-4" cy="7" r="8" fill="#ffb7c5" opacity="0.9"/>
            <circle cx="-6" cy="-2" r="8" fill="#ffb7c5" opacity="0.9"/>
            <!-- Center -->
            <circle cx="0" cy="0" r="5" fill="#ff5e7e" opacity="0.9"/>
            <circle cx="0" cy="0" r="2" fill="#ff0000" opacity="0.8"/>
        </g>

        <g id="blossom-cluster">
            <use href="#flower" transform="translate(0, 0) scale(1)"/>
            <use href="#flower" transform="translate(15, -10) scale(0.8)"/>
            <use href="#flower" transform="translate(-15, -5) scale(0.9)"/>
            <use href="#flower" transform="translate(10, 15) scale(0.7)"/>
            <use href="#flower" transform="translate(-10, 10) scale(0.85)"/>
            <use href="#flower" transform="translate(25, 5) scale(0.6)"/>
            <use href="#flower" transform="translate(-25, -15) scale(0.75)"/>
            <use href="#flower" transform="translate(0, -20) scale(0.8)"/>
            <use href="#flower" transform="translate(20, 20) scale(0.7)"/>
            <use href="#flower" transform="translate(-20, 20) scale(0.7)"/>
            <use href="#flower" transform="translate(5, -25) scale(0.65)"/>
            <use href="#flower" transform="translate(-5, 25) scale(0.65)"/>
        </g>

        <g id="petal">
            <ellipse cx="0" cy="0" rx="6" ry="3" fill="#ffb7c5" opacity="0.8"/>
        </g>

        <g id="leaf">
            <path d="M 0 0 Q 5 -10 10 0 Q 5 10 0 0" fill="#8fbc8f" opacity="0.8"/>
        </g>

        <g id="bird">
            <path d="M 0 0 Q 5 -5 10 0 Q 15 -5 20 0 Q 15 -2 10 2 Q 5 -2 0 0" fill="#555"/>
        </g>
    </defs>

    <!-- Background Sky -->
    <rect fill="url(#skyGrad)"/>

    <!-- Sun -->
    <circle cx="100" cy="100" r="50" fill="#fff" opacity="0.3"/>
    <circle cx="100" cy="100" r="30" fill="#fff" opacity="0.5"/>
    <circle cx="100" cy="100" r="15" fill="#fff" opacity="0.8"/>

    <!-- Distant Hills -->
    <path d="M 0 500 Q 100 400 200 450 T 400 420 T 600 460 T 800 430 L 800 600 L 0 600 Z" fill="#c2d6a6" opacity="0.5"/>
    <path d="M 0 550 Q 150 480 300 520 T 600 500 T 800 530 L 800 600 L 0 600 Z" fill="#a3c97c" opacity="0.6"/>

    <!-- Birds -->
    <use href="#bird" transform="translate(200, 100) scale(1.5)"/>
    <use href="#bird" transform="translate(600, 80) scale(1.2)"/>
    <use href="#bird" transform="translate(500, 150) scale(1)"/>
    <use href="#bird" transform="translate(300, 60) scale(0.8)"/>
    <use href="#bird" transform="translate(700, 120) scale(1.1)"/>

    <!-- Tree: Background Branches -->
    <g fill="none" stroke="#1a0f0a" stroke-linecap="round" stroke-linejoin="round">
        <path d="M 400 500 Q 250 450 150 400 L 160 390 Q 260 440 410 490 Z" fill="#1a0f0a" stroke="none"/>
        <path d="M 400 450 Q 550 400 650 350 L 660 360 Q 560 410 410 460 Z" fill="#1a0f0a" stroke="none"/>
        <path d="M 400 400 Q 250 300 150 250 L 160 240 Q 260 290 410 390 Z" fill="#1a0f0a" stroke="none"/>
        <path d="M 400 350 Q 550 250 650 200 L 660 210 Q 560 260 410 360 Z" fill="#1a0f0a" stroke="none"/>
    </g>

    <!-- Tree: Main Trunk -->
    <g fill="none" stroke="url(#trunkGrad)" stroke-linecap="round" stroke-linejoin="round">
        <path d="M 370 600 Q 380 450 400 300 L 430 300 Q 420 450 440 600 Z" fill="url(#trunkGrad)" stroke="none"/>
    </g>

    <!-- Tree: Foreground Branches -->
    <g fill="none" stroke="#2a1b14" stroke-linecap="round" stroke-linejoin="round">
        <path d="M 400 550 Q 300 500 200 450 L 210 440 Q 310 490 410 540 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 400 500 Q 500 450 600 400 L 610 410 Q 510 460 410 510 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 400 450 Q 300 350 200 300 L 210 290 Q 310 340 410 440 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 400 400 Q 500 300 600 250 L 610 260 Q 510 310 410 410 Z" fill="#2a1b14" stroke="none"/>
        
        <!-- Sub-branches -->
        <path d="M 250 380 Q 200 350 150 300 L 160 290 Q 210 340 260 370 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 550 330 Q 600 300 650 250 L 660 260 Q 610 310 560 340 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 300 280 Q 250 230 200 180 L 210 170 Q 260 220 310 270 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 500 250 Q 550 200 600 150 L 610 160 Q 560 210 510 260 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 400 200 Q 350 150 300 100 L 310 90 Q 360 140 410 190 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 400 200 Q 450 150 500 100 L 510 110 Q 460 160 410 210 Z" fill="#2a1b14" stroke="none"/>
        <path d="M 400 150 Q 400 100 400 50 L 410 50 Q 410 100 410 150 Z" fill="#2a1b14" stroke="none"/>
    </g>

    <!-- Tree Texture Lines -->
    <g fill="none" stroke="#1a0f0a" stroke-width="3" stroke-linecap="round">
        <path d="M 380 600 Q 390 450 400 300"/>
        <path d="M 420 600 Q 410 450 400 300"/>
        <path d="M 400 450 Q 300 400 200 350"/>
        <path d="M 400 400 Q 500 350 600 300"/>
        <path d="M 400 350 Q 300 300 200 250"/>
        <path d="M 400 300 Q 500 250 600 200"/>
    </g>

    <!-- Blossom Clusters -->
    <!-- Left Side -->
    <use href="#blossom-cluster" transform="translate(150, 300) scale(1.5)"/>
    <use href="#blossom-cluster" transform="translate(200, 180) scale(1.2)"/>
    <use href="#blossom-cluster" transform="translate(300, 100) scale(1.3)"/>
    <use href="#blossom-cluster" transform="translate(250, 380) scale(1.4)"/>
    <use href="#blossom-cluster" transform="translate(180, 250) scale(1.1)"/>
    <use href="#blossom-cluster" transform="translate(250, 200) scale(1.4)"/>
    <use href="#blossom-cluster" transform="translate(200, 120) scale(1.2)"/>
    
    <!-- Right Side -->
    <use href="#blossom-cluster" transform="translate(650, 250) scale(1.5)"/>
    <use href="#blossom-cluster" transform="translate(600, 150) scale(1.2)"/>
    <use href="#blossom-cluster" transform="translate(500, 100) scale(1.3)"/>
    <use href="#blossom-cluster" transform="translate(550, 330) scale(1.4)"/>
    <use href="#blossom-cluster" transform="translate(620, 200) scale(1.1)"/>
    <use href="#blossom-cluster" transform="translate(550, 200) scale(1.4)"/>
    <use href="#blossom-cluster" transform="translate(600, 120) scale(1.2)"/>
    
    <!-- Top & Center -->
    <use href="#blossom-cluster" transform="translate(400, 50) scale(1.6)"/>
    <use href="#blossom-cluster" transform="translate(350, 120) scale(1.2)"/>
    <use href="#blossom-cluster" transform="translate(450, 120) scale(1.2)"/>
    <use href="#blossom-cluster" transform="translate(400, 180) scale(1.4)"/>
    <use href="#blossom-cluster" transform="translate(300, 200) scale(1.1)"/>
    <use href="#blossom-cluster" transform="translate(500, 200) scale(1.1)"/>
    <use href="#blossom-cluster" transform="translate(400, 250) scale(1.3)"/>
    <use href="#blossom-cluster" transform="translate(250, 300) scale(1.2)"/>
    <use href="#blossom-cluster" transform="translate(550, 300) scale(1.2)"/>
    <use href="#blossom-cluster" transform="translate(350, 80) scale(1.4)"/>
    <use href="#blossom-cluster" transform="translate(450, 80) scale(1.4)"/>
    <use href="#blossom-cluster" transform="translate(300, 80) scale(1.3)"/>
    <use href="#blossom-cluster" transform="translate(500, 80) scale(1.3)"/>
    <use href="#blossom-cluster" transform="translate(400, 150) scale(1.2)"/>
    <use href="#blossom-cluster" transform="translate(350, 200) scale(1.1)"/>
    <use href="#blossom-cluster" transform="translate(450, 200) scale(1.1)"/>
    <use href="#blossom-cluster" transform="translate(300, 250) scale(1.3)"/>
    <use href="#blossom-cluster" transform="translate(500, 250) scale(1.3)"/>

    <!-- Individual Flowers for Detail -->
    <use href="#flower" transform="translate(200, 200) scale(1.5)"/>
    <use href="#flower" transform="translate(600, 200) scale(1.5)"/>
    <use href="#flower" transform="translate(400, 100) scale(1.8)"/>
    <use href="#flower" transform="translate(300, 150) scale(1.4)"/>
    <use href="#flower" transform="translate(500, 150) scale(1.4)"/>
    <use href="#flower" transform="translate(250, 300) scale(1.6)"/>
    <use href="#flower" transform="translate(550, 300) scale(1.6)"/>
    <use href="#flower" transform="translate(150, 250) scale(1.3)"/>
    <use href="#flower" transform="translate(650, 250) scale(1.3)"/>
    <use href="#flower" transform="translate(350, 80) scale(1.7)"/>
    <use href="#flower" transform="translate(450, 80) scale(1.7)"/>
    <use href="#flower" transform="translate(400, 200) scale(1.6)"/>
    <use href="#flower" transform="translate(300, 250) scale(1.5)"/>
    <use href="#flower" transform="translate(500, 250) scale(1.5)"/>

    <!-- Leaves -->
    <use href="#leaf" transform="translate(200, 200) rotate(30) scale(1.5)"/>
    <use href="#leaf" transform="translate(600, 200) rotate(-45) scale(1.2)"/>
    <use href="#leaf" transform="translate(400, 100) rotate(15) scale(1.8)"/>
    <use href="#leaf" transform="translate(300, 150) rotate(-20) scale(1.4)"/>
    <use href="#leaf" transform="translate(500, 150) rotate(60) scale(1.6)"/>
    <use href="#leaf" transform="translate(250, 300) rotate(-15) scale(1.3)"/>
    <use href="#leaf" transform="translate(550, 300) rotate(45) scale(1.5)"/>
    <use href="#leaf" transform="translate(150, 250) rotate(10) scale(1.2)"/>
    <use href="#leaf" transform="translate(650, 250) rotate(-30) scale(1.4)"/>

    <!-- Falling Petals -->
    <use href="#petal" transform="translate(100, 200) rotate(30) scale(1.5)"/>
    <use href="#petal" transform="translate(200, 400) rotate(-45) scale(1.2)"/>
    <use href="#petal" transform="translate(300, 500) rotate(15) scale(1.8)"/>
    <use href="#petal" transform="translate(500, 450) rotate(-20) scale(1.4)"/>
    <use href="#petal" transform="translate(600, 350) rotate(60) scale(1.6)"/>
    <use href="#petal" transform="translate(700, 250) rotate(-15) scale(1.3)"/>
    <use href="#petal" transform="translate(150, 150) rotate(45) scale(1.1)"/>
    <use href="#petal" transform="translate(650, 100) rotate(-60) scale(1.5)"/>
    <use href="#petal" transform="translate(450, 250) rotate(10) scale(1.2)"/>
    <use href="#petal" transform="translate(350, 350) rotate(-30) scale(1.4)"/>
    <use href="#petal" transform="translate(250, 250) rotate(80) scale(1.1)"/>
    <use href="#petal" transform="translate(550, 200) rotate(-40) scale(1.3)"/>
    <use href="#petal" transform="translate(50, 300) rotate(20) scale(1.6)"/>
    <use href="#petal" transform="translate(750, 400) rotate(-70) scale(1.4)"/>
    <use href="#petal" transform="translate(400, 550) rotate(5) scale(1.7)"/>
    <use href="#petal" transform="translate(100, 500) rotate(-10) scale(1.5)"/>
    <use href="#petal" transform="translate(700, 500) rotate(40) scale(1.2)"/>
    <use href="#petal" transform="translate(120, 180) rotate(45) scale(1.2)"/>
    <use href="#petal" transform="translate(680, 180) rotate(-30) scale(1.4)"/>
    <use href="#petal" transform="translate(300, 400) rotate(15) scale(1.5)"/>
    <use href="#petal" transform="translate(500, 350) rotate(-60) scale(1.3)"/>
    <use href="#petal" transform="translate(400, 500) rotate(25) scale(1.6)"/>
    <use href="#petal" transform="translate(200, 500) rotate(-15) scale(1.4)"/>
    <use href="#petal" transform="translate(600, 450) rotate(50) scale(1.5)"/>
    <use href="#petal" transform="translate(100, 400) rotate(-40) scale(1.3)"/>
    <use href="#petal" transform="translate(700, 350) rotate(10) scale(1.2)"/>
    <use href="#petal" transform="translate(350, 250) rotate(-80) scale(1.1)"/>
    <use href="#petal" transform="translate(450, 200) rotate(35) scale(1.4)"/>
    <use href="#petal" transform="translate(250, 150) rotate(-25) scale(1.2)"/>
    <use href="#petal" transform="translate(550, 100) rotate(70) scale(1.5)"/>
    <use href="#petal" transform="translate(50, 250) rotate(-10) scale(1.6)"/>
    <use href="#petal" transform="translate(750, 300) rotate(40) scale(1.4)"/>

    <!-- Foreground Ground -->
    <ellipse cx="400" cy="600" rx="450" ry="120" fill="#d4e6b3" opacity="0.8"/>
    <ellipse cx="400" cy="620" rx="500" ry="100" fill="#a3c97c" opacity="0.6"/>

    <!-- Grass Details -->
    <g fill="none" stroke="#8fbc8f" stroke-width="2" stroke-linecap="round">
        <path d="M 100 580 Q 105 570 110 580"/>
        <path d="M 200 590 Q 205 580 210 590"/>
        <path d="M 300 570 Q 305 560 310 570"/>
        <path d="M 500 580 Q 505 570 510 580"/>
        <path d="M 600 590 Q 605 580 610 590"/>
        <path d="M 700 570 Q 705 560 710 570"/>
        <path d="M 150 600 Q 155 590 160 600"/>
        <path d="M 650 600 Q 655 590 660 600"/>
        <path d="M 400 610 Q 405 600 410 610"/>
    </g>
</svg>
```

</details>

## 3. gemma4_31b-cloud

**SVG size:** 3109 characters  
**Complexity:** Low  
**Shape elements:** 29  
**Distinct colors:** 6  
**Raw response:** 4323 characters

![gemma4_31b-cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/gemma4_31b-cloud.svg)

### Analysis

This SVG contains approximately **29 shape elements** and uses **6 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for skies and petals)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate petal detail. Best for embedding in web pages where small file size matters more than visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
  <!-- Background Gradient -->
  <defs>
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#e0f7fa;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#fff9c4;stop-opacity:1" />
    </linearGradient>
    
    <!-- Filter for a soft glow on flowers -->
    <filter id="glow">
      <feGaussianBlur stdDeviation="2" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>

  <!-- Sky -->
  <rect fill="url(#skyGradient)" />

  <!-- Grass/Ground -->
  <ellipse cx="400" cy="520" rx="600" ry="150" fill="#dcedc8" />

  <!-- Tree 1 (Left) -->
  <g id="tree-left" transform="translate(200, 400)">
    <!-- Trunk -->
    <path d="M0,0 Q-20,-50 0,-120 Q20,-180 0,-250" stroke="#5d4037" stroke-width="15" fill="none" stroke-linecap="round" />
    <path d="M0,-120 Q-30,-140 -60,-160" stroke="#5d4037" stroke-width="8" fill="none" stroke-linecap="round" />
    <path d="M0,-180 Q30,-200 60,-220" stroke="#5d4037" stroke-width="8" fill="none" stroke-linecap="round" />
    
    <!-- Blossoms (Clusters of circles) -->
    <g fill="#ffc1e3" filter="url(#glow)">
      <circle cx="-40" cy="-180" r="35" />
      <circle cx="-70" cy="-160" r="30" />
      <circle cx="-20" cy="-200" r="40" />
      <circle cx="20" cy="-230" r="35" />
      <circle cx="50" cy="-210" r="30" />
      <circle cx="0" cy="-260" r="45" />
    </g>
    <!-- Highlights -->
    <g fill="#f8bbd0">
      <circle cx="-30" cy="-190" r="15" />
      <circle cx="10" cy="-240" r="20" />
    </g>
  </g>

  <!-- Tree 2 (Right/Center) -->
  <g id="tree-right" transform="translate(550, 420) scale(1.2)">
    <!-- Trunk -->
    <path d="M0,0 Q20,-60 0,-150 Q-20,-210 0,-280" stroke="#5d4037" stroke-width="18" fill="none" stroke-linecap="round" />
    <path d="M0,-150 Q30,-170 60,-190" stroke="#5d4037" stroke-width="10" fill="none" stroke-linecap="round" />
    <path d="M0,-200 Q-30,-220 -60,-240" stroke="#5d4037" stroke-width="10" fill="none" stroke-linecap="round" />
    
    <!-- Blossoms -->
    <g fill="#ffc1e3" filter="url(#glow)">
      <circle cx="30" cy="-210" r="40" />
      <circle cx="60" cy="-180" r="35" />
      <circle cx="0" cy="-270" r="50" />
      <circle cx="-30" cy="-250" r="40" />
      <circle cx="-60" cy="-220" r="30" />
    </g>
    <!-- Highlights -->
    <g fill="#f8bbd0">
      <circle cx="20" cy="-220" r="20" />
      <circle cx="-20" cy="-260" r="15" />
    </g>
  </g>

  <!-- Falling Petals -->
  <g fill="#ffc1e3" opacity="0.8">
    <ellipse cx="150" cy="200" rx="4" ry="2" transform="rotate(45 150 200)" />
    <ellipse cx="300" cy="150" rx="4" ry="2" transform="rotate(-20 300 150)" />
    <ellipse cx="450" cy="280" rx="4" ry="2" transform="rotate(110 450 280)" />
    <ellipse cx="600" cy="120" rx="4" ry="2" transform="rotate(10 600 120)" />
    <ellipse cx="700" cy="300" rx="4" ry="2" transform="rotate(70 700 300)" />
    <ellipse cx="100" cy="350" rx="4" ry="2" transform="rotate(-40 100 350)" />
  </g>
</svg>
```

</details>

## 4. gemma4_cloud

**SVG size:** 2649 characters  
**Complexity:** Low  
**Shape elements:** 28  
**Distinct colors:** 9  
**Raw response:** 3684 characters

![gemma4_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/gemma4_cloud.svg)

### Analysis

This SVG contains approximately **28 shape elements** and uses **9 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate petal detail. Best for embedding in web pages where small file size matters more than visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
  <!-- Background: Soft Sky -->
  <rect width="800" height="600" fill="#e0f4ff" />
  
  <!-- Distant Hills -->
  <path d="M0 600 Q 200 450 400 550 T 800 500 V 600 H 0 Z" fill="#d1e8d1" />
  <path d="M0 600 Q 300 500 600 580 T 800 550 V 600 H 0 Z" fill="#b8d8b8" />

  <!-- Tree 1 (Right Foreground) -->
  <g transform="translate(550, 450)">
    <!-- Trunk -->
    <path d="M0 0 Q -20 -100 0 -200 Q 20 -100 0 0" fill="#5d4037" />
    <path d="M0 -120 Q -40 -150 -60 -180" stroke="#5d4037" stroke-width="8" fill="none" stroke-linecap="round" />
    <path d="M0 -150 Q 40 -180 60 -210" stroke="#5d4037" stroke-width="6" fill="none" stroke-linecap="round" />
    
    <!-- Blossom Clusters -->
    <circle cx="-40" cy="-200" r="40" fill="#ffc0cb" opacity="0.8" />
    <circle cx="-10" cy="-230" r="45" fill="#ffb6c1" opacity="0.9" />
    <circle cx="30" cy="-210" r="40" fill="#ffc0cb" opacity="0.8" />
    <circle cx="60" cy="-240" r="35" fill="#ffb6c1" opacity="0.9" />
    <circle cx="20" cy="-260" r="40" fill="#ffc0cb" opacity="0.8" />
    <circle cx="-30" cy="-250" r="35" fill="#ffb6c1" opacity="0.9" />
    
    <!-- Individual Petals/Flowers -->
    <circle cx="10" cy="-270" r="5" fill="#fff" />
    <circle cx="40" cy="-230" r="4" fill="#fff" />
    <circle cx="-50" cy="-210" r="5" fill="#fff" />
  </g>

  <!-- Tree 2 (Left Middle Ground) -->
  <g transform="translate(200, 500) scale(0.8)">
    <!-- Trunk -->
    <path d="M0 0 Q -10 -80 0 -160 Q 10 -80 0 0" fill="#6d4c41" />
    <path d="M0 -80 Q -30 -110 -50 -140" stroke="#6d4c41" stroke-width="6" fill="none" stroke-linecap="round" />
    
    <!-- Blossom Clusters -->
    <circle cx="-30" cy="-160" r="35" fill="#ffc0cb" opacity="0.8" />
    <circle cx="10" cy="-190" r="40" fill="#ffb6c1" opacity="0.9" />
    <circle cx="40" cy="-150" r="35" fill="#ffc0cb" opacity="0.8" />
    <circle cx="0" cy="-220" r="35" fill="#ffb6c1" opacity="0.9" />
  </g>

  <!-- Falling Petals -->
  <ellipse cx="150" cy="200" rx="4" ry="2" fill="#ffc0cb" transform="rotate(45 150 200)" />
  <ellipse cx="300" cy="350" rx="4" ry="2" fill="#ffb6c1" transform="rotate(110 300 350)" />
  <ellipse cx="600" cy="300" rx="4" ry="2" fill="#ffc0cb" transform="rotate(20 600 300)" />
  <ellipse cx="450" cy="420" rx="4" ry="2" fill="#fff" transform="rotate(160 450 420)" />
  <ellipse cx="700" cy="480" rx="4" ry="2" fill="#ffb6c1" transform="rotate(300 700 480)" />
  <ellipse cx="100" cy="400" rx="4" ry="2" fill="#ffc0cb" transform="rotate(70 100 400)" />

  <!-- Ground Grass -->
  <rect width="800" height="50" y="550" fill="#8fb88f" />
</svg>
```

</details>

## 5. glm-5.1_cloud

**SVG size:** 19810 characters  
**Complexity:** Very high  
**Shape elements:** 69  
**Distinct colors:** 21  
**Raw response:** 22858 characters

![glm-5.1_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/glm-5.1_cloud.svg)

### Analysis

This SVG contains approximately **69 shape elements** and uses **21 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating petals
- Includes gradient fills for richer visual depth (great for skies and petals)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

With over 15,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering individual petals, layered branches, and atmospheric backgrounds. Best for users who want a painterly cherry blossom scene.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 700">
  <defs>
    <!-- Sky gradient -->
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fce4ec"/>
      <stop offset="40%" stop-color="#f8bbd0"/>
      <stop offset="100%" stop-color="#e1bee7"/>
    </linearGradient>

    <!-- Ground gradient -->
    <linearGradient id="groundGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#81c784"/>
      <stop offset="100%" stop-color="#4a7c59"/>
    </linearGradient>

    <!-- Trunk gradient -->
    <linearGradient id="trunkGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#5d4037"/>
      <stop offset="50%" stop-color="#795548"/>
      <stop offset="100%" stop-color="#4e342e"/>
    </linearGradient>

    <!-- Blossom gradient pink -->
    <radialGradient id="blossomPink" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fff"/>
      <stop offset="40%" stop-color="#f8bbd0"/>
      <stop offset="100%" stop-color="#f48fb1"/>
    </radialGradient>

    <!-- Blossom gradient white -->
    <radialGradient id="blossomWhite" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fff"/>
      <stop offset="50%" stop-color="#fce4ec"/>
      <stop offset="100%" stop-color="#f8bbd0"/>
    </radialGradient>

    <!-- Blossom gradient deep -->
    <radialGradient id="blossomDeep" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#f8bbd0"/>
      <stop offset="100%" stop-color="#ec407a"/>
    </radialGradient>

    <!-- Sun glow -->
    <radialGradient id="sunGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fff9c4"/>
      <stop offset="60%" stop-color="#fff59d"/>
      <stop offset="100%" stop-color="#ffee5858"/>
    </radialGradient>

    <!-- Petal shape -->
    <path id="petal" d="M0,-6 C3,-5 5,-2 5,0 C5,2 3,5 0,6 C-3,5 -5,2 -5,0 C-5,-2 -3,-5 0,-6Z" />

    <!-- Individual flower cluster -->
    <g id="flower5">
      <circle cx="0" cy="0" r="8" fill="url(#blossomPink)" opacity="0.9"/>
      <circle cx="7" cy="-4" r="7" fill="url(#blossomWhite)" opacity="0.85"/>
      <circle cx="-7" cy="-4" r="7" fill="url(#blossomPink)" opacity="0.85"/>
      <circle cx="4" cy="6" r="7" fill="url(#blossomWhite)" opacity="0.8"/>
      <circle cx="-4" cy="6" r="7" fill="url(#blossomPink)" opacity="0.85"/>
      <circle cx="0" cy="0" r="3" fill="#e91e63" opacity="0.7"/>
    </g>

    <g id="flower4">
      <circle cx="0" cy="-5" r="7" fill="url(#blossomWhite)" opacity="0.85"/>
      <circle cx="5" cy="3" r="7" fill="url(#blossomPink)" opacity="0.85"/>
      <circle cx="-5" cy="3" r="7" fill="url(#blossomDeep)" opacity="0.75"/>
      <circle cx="0" cy="0" r="2.5" fill="#c2185b" opacity="0.6"/>
    </g>

    <g id="flower3">
      <circle cx="0" cy="-4" r="6" fill="url(#blossomPink)" opacity="0.9"/>
      <circle cx="4" cy="3" r="6" fill="url(#blossomWhite)" opacity="0.85"/>
      <circle cx="-4" cy="3" r="6" fill="url(#blossomPink)" opacity="0.85"/>
      <circle cx="0" cy="0" r="2" fill="#ad1457" opacity="0.6"/>
    </g>

    <!-- Soft bloom filter -->
    <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <filter id="strongGlow" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="4" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <!-- Water ripple filter -->
    <filter id="waterRipple">
      <feTurbulence type="fractalNoise" baseFrequency="0.02" numOctaves="2"/>
      <feDisplacementMap in="SourceGraphic" scale="3"/>
    </filter>
  </defs>

  <!-- ===== SKY ===== -->
  <rect width="1000" height="700" fill="url(#skyGrad)"/>

  <!-- Sun -->
  <circle cx="800" cy="120" r="80" fill="url(#sunGlow)" class="glow-pulse"/>
  <circle cx="800" cy="120" r="40" fill="#fffde7" opacity="0.7"/>

  <!-- Distant mountains -->
  <path d="M0,500 Q100,420 200,460 Q300,400 400,440 Q500,380 600,430 Q700,390 800,450 Q900,410 1000,460 L1000,520 L0,520Z" fill="#a5d6a7" opacity="0.4"/>

  <!-- ===== LEFT TREE (large, foreground) ===== -->
  <g class="tree-sway">
    <!-- Trunk -->
    <path d="M180,680 Q175,600 160,520 Q145,460 155,400 Q162,360 150,320 Q140,290 155,260" 
          stroke="url(#trunkGrad)" stroke-width="22" fill="none" stroke-linecap="round"/>
    <!-- Trunk texture lines -->
    <path d="M180,680 Q175,600 160,520 Q145,460 155,400" 
          stroke="#4e342e" stroke-width="1" fill="none" opacity="0.3"/>
    
    <!-- Main branches -->
    <path d="M155,260 Q130,230 90,200 Q60,180 40,160" 
          stroke="#5d4037" stroke-width="10" fill="none" stroke-linecap="round"/>
    <path d="M155,260 Q170,220 200,190 Q230,170 260,150" 
          stroke="#5d4037" stroke-width="10" fill="none" stroke-linecap="round"/>
    <path d="M155,260 Q155,240 140,210 Q125,185 110,170" 
          stroke="#6d4c41" stroke-width="7" fill="none" stroke-linecap="round"/>
    <path d="M160,350 Q200,310 250,290 Q280,280 310,270" 
          stroke="#5d4037" stroke-width="8" fill="none" stroke-linecap="round"/>
    <path d="M150,400 Q100,370 70,350" 
          stroke="#6d4c41" stroke-width="7" fill="none" stroke-linecap="round"/>

    <!-- Sub branches -->
    <path d="M90,200 Q70,180 50,200 Q30,220 20,250" 
          stroke="#795548" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M90,200 Q110,175 130,160" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M200,190 Q230,170 250,190" 
          stroke="#795548" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M260,150 Q290,140 310,155" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M260,150 Q270,120 280,100" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M40,160 Q20,140 10,120" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M40,160 Q30,170 15,175" 
          stroke="#795548" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M250,290 Q280,275 300,285" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M250,290 Q260,310 280,320" 
          stroke="#795548" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M70,350 Q50,340 30,350" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M70,350 Q60,370 45,380" 
          stroke="#795548" stroke-width="3" fill="none" stroke-linecap="round"/>

    <!-- ===== Blossom clusters on left tree ===== -->
    <g filter="url(#softGlow)">
      <!-- Top area blossoms -->
      <use href="#flower5" x="40" y="150" transform="translate(40,150) scale(1.2)"/>
      <use href="#flower4" x="20" y="110" transform="translate(20,110) scale(1)"/>
      <use href="#flower5" x="10" y="140" transform="translate(10,140) scale(0.9)"/>
      <use href="#flower3" x="50" y="125" transform="translate(50,125) scale(0.8)"/>
      <use href="#flower5" x="90" y="185" transform="translate(90,185) scale(1.1)"/>
      <use href="#flower4" x="110" y="155" transform="translate(110,155) scale(1)"/>
      <use href="#flower5" x="130" y="145" transform="translate(130,145) scale(1.3)"/>
      <use href="#flower3" x="75" y="165" transform="translate(75,165) scale(0.9)"/>

      <!-- Upper right blossoms -->
      <use href="#flower5" x="200" y="180" transform="translate(200,180) scale(1.1)"/>
      <use href="#flower4" x="230" y="165" transform="translate(230,165) scale(1)"/>
      <use href="#flower5" x="255" y="140" transform="translate(255,140) scale(1.2)"/>
      <use href="#flower3" x="280" y="95" transform="translate(280,95) scale(0.9)"/>
      <use href="#flower5" x="310" y="145" transform="translate(310,145) scale(0.8)"/>
      <use href="#flower4" x="240" y="185" transform="translate(240,185) scale(1.1)"/>
      <use href="#flower3" x="265" y="100" transform="translate(265,100) scale(0.7)"/>

      <!-- Mid area blossoms -->
      <use href="#flower5" x="150" y="250" transform="translate(150,250) scale(1)"/>
      <use href="#flower4" x="170" y="235" transform="translate(170,235) scale(0.9)"/>
      <use href="#flower5" x="140" y="205" transform="translate(140,205) scale(1.1)"/>
      <use href="#flower3" x="120" y="220" transform="translate(120,220) scale(0.8)"/>

      <!-- Right branch blossoms -->
      <use href="#flower5" x="295" y="270" transform="translate(295,270) scale(1)"/>
      <use href="#flower4" x="310" y="280" transform="translate(310,280) scale(0.8)"/>
      <use href="#flower3" x="280" y="260" transform="translate(280,260) scale(0.9)"/>
      <use href="#flower5" x="275" y="315" transform="translate(275,315) scale(0.7)"/>

      <!-- Left branch blossoms -->
      <use href="#flower5" x="60" y="340" transform="translate(60,340) scale(1)"/>
      <use href="#flower4" x="30" y="345" transform="translate(30,345) scale(0.8)"/>
      <use href="#flower3" x="40" y="375" transform="translate(40,375) scale(0.7)"/>

      <!-- Extra fill blossoms -->
      <use href="#flower5" x="160" y="190" transform="translate(160,190) scale(0.7)"/>
      <use href="#flower4" x="100" y="200" transform="translate(100,200) scale(0.6)"/>
      <use href="#flower3" x="180" y="170" transform="translate(180,170) scale(0.8)"/>
      <use href="#flower5" x="220" y="155" transform="translate(220,155) scale(0.7)"/>
    </g>
  </g>

  <!-- ===== RIGHT TREE (medium, slightly behind) ===== -->
  <g class="tree-sway" style="animation-delay: -1.5s;">
    <!-- Trunk -->
    <path d="M750,680 Q755,620 760,560 Q765,510 758,460 Q752,420 760,380 Q765,350 755,320" 
          stroke="url(#trunkGrad)" stroke-width="18" fill="none" stroke-linecap="round"/>
    <path d="M750,680 Q755,620 760,560" 
          stroke="#4e342e" stroke-width="1" fill="none" opacity="0.3"/>

    <!-- Branches -->
    <path d="M755,320 Q730,280 700,260 Q670,245 640,240" 
          stroke="#5d4037" stroke-width="9" fill="none" stroke-linecap="round"/>
    <path d="M755,320 Q780,285 810,265 Q840,250 870,240" 
          stroke="#5d4037" stroke-width="9" fill="none" stroke-linecap="round"/>
    <path d="M755,320 Q745,300 730,285" 
          stroke="#6d4c41" stroke-width="6" fill="none" stroke-linecap="round"/>
    <path d="M758,460 Q800,435 840,420 Q870,410 890,405" 
          stroke="#5d4037" stroke-width="7" fill="none" stroke-linecap="round"/>
    <path d="M758,460 Q720,440 690,430" 
          stroke="#6d4c41" stroke-width="6" fill="none" stroke-linecap="round"/>

    <!-- Sub branches -->
    <path d="M700,260 Q680,250 660,270" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M640,240 Q620,230 600,240" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M810,265 Q830,250 845,265" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M870,240 Q890,230 910,240" 
          stroke="#795548" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M870,240 Q880,215 895,200" 
          stroke="#795548" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M730,285 Q720,270 710,260" 
          stroke="#795548" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M840,420 Q860,400 875,395" 
          stroke="#795548" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M690,430 Q670,420 655,430" 
          stroke="#795548" stroke-width="3" fill="none" stroke-linecap="round"/>

    <!-- ===== Blossom clusters on right tree ===== -->
    <g filter="url(#softGlow)">
      <use href="#flower5" x="700" y="245" transform="translate(700,245) scale(1.1)"/>
      <use href="#flower4" x="660" y="235" transform="translate(660,235) scale(1)"/>
      <use href="#flower5" x="640" y="230" transform="translate(640,230) scale(0.9)"/>
      <use href="#flower3" x="600" y="235" transform="translate(600,235) scale(0.8)"/>
      <use href="#flower5" x="660" y="265" transform="translate(660,265) scale(0.8)"/>
      <use href="#flower4" x="730" y="270" transform="translate(730,270) scale(0.9)"/>

      <use href="#flower5" x="810" y="250" transform="translate(810,250) scale(1.1)"/>
      <use href="#flower4" x="845" y="258" transform="translate(845,258) scale(1)"/>
      <use href="#flower5" x="870" y="230" transform="translate(870,230) scale(1.2)"/>
      <use href="#flower3" x="910" y="235" transform="translate(910,235) scale(0.8)"/>
      <use href="#flower5" x="895" y="195" transform="translate(895,195) scale(0.9)"/>

      <use href="#flower5" x="755" y="310" transform="translate(755,310) scale(1)"/>
      <use href="#flower4" x="780" y="280" transform="translate(780,280) scale(0.9)"/>

      <use href="#flower5" x="875" y="395" transform="translate(875,395) scale(0.8)"/>
      <use href="#flower4" x="840" y="405" transform="translate(840,405) scale(0.7)"/>
      <use href="#flower3" x="690" y="420" transform="translate(690,420) scale(0.8)"/>
      <use href="#flower3" x="655" y="425" transform="translate(655,425) scale(0.7)"/>

      <!-- Extra fills -->
      <use href="#flower5" x="770" y="260" transform="translate(770,260) scale(0.7)"/>
      <use href="#flower3" x="720" y="255" transform="translate(720,255) scale(0.6)"/>
    </g>
  </g>

  <!-- ===== SMALL TREE (background, left-center) ===== -->
  <g class="tree-sway" style="animation-delay: -3s;">
    <path d="M460,680 Q462,640 465,600 Q468,570 463,540 Q458,520 463,495" 
          stroke="url(#trunkGrad)" stroke-width="10" fill="none" stroke-linecap="round"/>
    <path d="M463,495 Q440,470 415,455" 
          stroke="#6d4c41" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M463,495 Q480,465 505,450" 
          stroke="#6d4c41" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M463,495 Q458,480 450,470" 
          stroke="#795548" stroke-width="3" fill="none" stroke-linecap="round"/>

    <g filter="url(#softGlow)" opacity="0.85">
      <use href="#flower5" x="415" y="445" transform="translate(415,445) scale(0.9)"/>
      <use href="#flower4" x="440" y="465" transform="translate(440,465) scale(0.8)"/>
      <use href="#flower5" x="505" y="440" transform="translate(505,440) scale(0.9)"/>
      <use href="#flower3" x="480" y="455" transform="translate(480,455) scale(0.8)"/>
      <use href="#flower4" x="463" y="485" transform="translate(463,485) scale(0.7)"/>
      <use href="#flower5" x="450" y="460" transform="translate(450,460) scale(0.6)"/>
    </g>
  </g>

  <!-- ===== GROUND ===== -->
  <path d="M0,600 Q200,580 400,590 Q600,600 800,585 Q900,580 1000,590 L1000,700 L0,700Z" fill="url(#groundGrad)"/>
  <!-- Grass texture -->
  <path d="M0,610 Q150,595 300,605 Q500,615 700,600 Q850,592 1000,600 L1000,700 L0,700Z" fill="#66bb6a" opacity="0.5"/>

  <!-- Fallen petals on ground -->
  <g opacity="0.7">
    <use href="#petal" x="100" y="620" fill="#f8bbd0" transform="translate(100,620) rotate(30) scale(1.5)"/>
    <use href="#petal" x="200" y="610" fill="#f48fb1" transform="translate(200,610) rotate(-20) scale(1.2)"/>
    <use href="#petal" x="350" y="618" fill="#fce4ec" transform="translate(350,618) rotate(60) scale(1)"/>
    <use href="#petal" x="500" y="605" fill="#f8bbd0" transform="translate(500,605) rotate(-45) scale(1.3)"/>
    <use href="#petal" x="650" y="600" fill="#f48fb1" transform="translate(650,600) rotate(15) scale(1.1)"/>
    <use href="#petal" x="780" y="595" fill="#fce4ec" transform="translate(780,595) rotate(-60) scale(1.4)"/>
    <use href="#petal" x="880" y="590" fill="#f8bbd0" transform="translate(880,590) rotate(80) scale(1)"/>
    <use href="#petal" x="420" y="608" fill="#f48fb1" transform="translate(420,608) rotate(10) scale(0.8)"/>
    <use href="#petal" x="560" y="602" fill="#fce4ec" transform="translate(560,602) rotate(-30) scale(1.2)"/>
    <use href="#petal" x="720" y="598" fill="#f8bbd0" transform="translate(720,598) rotate(50) scale(0.9)"/>
  </g>

  <!-- ===== FALLING PETALS (animated) ===== -->
  <g>
    <!-- Petal 1 -->
    <g class="petal1" style="animation-delay: 0s;">
      <use href="#petal" fill="#f8bbd0" transform="translate(120, 100) scale(2)" opacity="0.9"/>
    </g>
    <g class="petal2" style="animation-delay: 1.2s;">
      <use href="#petal" fill="#f48fb1" transform="translate(300, 80) scale(1.8)" opacity="0.85"/>
    </g>
    <g class="petal3" style="animation-delay: 2.5s;">
      <use href="#petal" fill="#fce4ec" transform="translate(500, 120) scale(2.2)" opacity="0.9"/>
    </g>
    <g class="petal4" style="animation-delay: 0.8s;">
      <use href="#petal" fill="#f8bbd0" transform="translate(700, 90) scale(1.6)" opacity="0.85"/>
    </g>
    <g class="petal1" style="animation-delay: 3.2s;">
      <use href="#petal" fill="#f48fb1" transform="translate(850, 150) scale(1.9)" opacity="0.8"/>
    </g>
    <g class="petal2" style="animation-delay: 1.8s;">
      <use href="#petal" fill="#fce4ec" transform="translate(180, 200) scale(1.5)" opacity="0.85"/>
    </g>
    <g class="petal3" style="animation-delay: 4s;">
      <use href="#petal" fill="#f8bbd0" transform="translate(400, 180) scale(2)" opacity="0.9"/>
    </g>
    <g class="petal4" style="animation-delay: 2.2s;">
      <use href="#petal" fill="#f48fb1" transform="translate(600, 250) scale(1.7)" opacity="0.8"/>
    </g>
    <g class="petal1" style="animation-delay: 0.5s;">
      <use href="#petal" fill="#fce4ec" transform="translate(80, 300) scale(1.4)" opacity="0.85"/>
    </g>
    <g class="petal2" style="animation-delay: 3.5s;">
      <use href="#petal" fill="#f8bbd0" transform="translate(250, 280) scale(2.1)" opacity="0.8"/>
    </g>
    <g class="petal3" style="animation-delay: 1s;">
      <use href="#petal" fill="#f48fb1" transform="translate(780, 300) scale(1.6)" opacity="0.85"/>
    </g>
    <g class="petal4" style="animation-delay: 2.8s;">
      <use href="#petal" fill="#fce4ec" transform="translate(920, 200) scale(1.3)" opacity="0.9"/>
    </g>
    <g class="petal1" style="animation-delay: 4.5s;">
      <use href="#petal" fill="#f8bbd0" transform="translate(470, 150) scale(1.8)" opacity="0.85"/>
    </g>
    <g class="petal3" style="animation-delay: 3.8s;">
      <use href="#petal" fill="#f48fb1" transform="translate(640, 350) scale(1.5)" opacity="0.8"/>
    </g>
    <g class="petal2" style="animation-delay: 0.3s;">
      <use href="#petal" fill="#fce4ec" transform="translate(340, 330) scale(2)" opacity="0.85"/>
    </g>
  </g>

  <!-- ===== ATMOSPHERIC PARTICLES ===== -->
  <g filter="url(#strongGlow)" class="glow-pulse" style="animation-delay: -1s;">
    <circle cx="150" cy="350" r="2" fill="#fff" opacity="0.4"/>
    <circle cx="350" cy="280" r="1.5" fill="#fff" opacity="0.3"/>
    <circle cx="600" cy="320" r="2" fill="#fff" opacity="0.35"/>
    <circle cx="800" cy="380" r="1.5" fill="#fff" opacity="0.3"/>
    <circle cx="900" cy="300" r="2" fill="#fff" opacity="0.4"/>
  </g>

  <!-- Subtle light rays from the sun -->
  <g opacity="0.08">
    <polygon points="800,120 650,700 700,700" fill="#fff9c4"/>
    <polygon points="800,120 780,700 830,700" fill="#fff9c4"/>
    <polygon points="800,120 900,700 950,700" fill="#fff9c4"/>
  </g>

  <!-- ===== SMALL BIRDS ===== -->
  <g fill="none" stroke="#5d4037" stroke-width="1.5" stroke-linecap="round" opacity="0.4">
    <path d="M600,150 Q605,145 610,148 Q615,145 620,150"/>
    <path d="M580,140 Q584,136 588,139 Q592,136 596,140"/>
    <path d="M620,130 Q623,127 626,129 Q629,127 632,130"/>
  </g>
</svg>
```

</details>

## 6. glm-5.2_cloud

**SVG size:** 15956 characters  
**Complexity:** Very high  
**Shape elements:** 52  
**Distinct colors:** 27  
**Raw response:** 17459 characters

![glm-5.2_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/glm-5.2_cloud.svg)

### Analysis

This SVG contains approximately **52 shape elements** and uses **27 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating petals
- Includes gradient fills for richer visual depth (great for skies and petals)
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Includes animation elements -- the model attempted motion (e.g., falling petals)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

With over 15,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering individual petals, layered branches, and atmospheric backgrounds. Best for users who want a painterly cherry blossom scene.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" text-rendering="geometricPrecision">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#f9d9e1"/>
      <stop offset="0.45" stop-color="#f1c5d4"/>
      <stop offset="0.75" stop-color="#d3b3c6"/>
      <stop offset="1" stop-color="#aeb8d0"/>
    </linearGradient>
    <radialGradient id="sun" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#fff2dc" stop-opacity="0.95"/>
      <stop offset="0.45" stop-color="#ffd4bc" stop-opacity="0.45"/>
      <stop offset="1" stop-color="#ffd4bc" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="hillfar" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#9a8aa8"/>
      <stop offset="1" stop-color="#776a86"/>
    </linearGradient>
    <linearGradient id="hillmid" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#7d7490"/>
      <stop offset="1" stop-color="#5f5878"/>
    </linearGradient>
    <linearGradient id="ground" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#6e7d5a"/>
      <stop offset="1" stop-color="#3f4a32"/>
    </linearGradient>
    <linearGradient id="trunk" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#3a261c"/>
      <stop offset="0.5" stop-color="#5c4131"/>
      <stop offset="1" stop-color="#3a261c"/>
    </linearGradient>

    <!-- three blossom tints: light, medium, deep -->
    <symbol id="bl" viewBox="-10 -10 20 20" overflow="visible">
      <g>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffe2ec"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffe2ec" transform="rotate(72)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffe2ec" transform="rotate(144)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffe2ec" transform="rotate(216)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffe2ec" transform="rotate(288)"/>
        <circle r="1.1" fill="#d96ea0"/>
      </g>
    </symbol>
    <symbol id="bm" viewBox="-10 -10 20 20" overflow="visible">
      <g>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffc0d8"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffc0d8" transform="rotate(72)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffc0d8" transform="rotate(144)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffc0d8" transform="rotate(216)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ffc0d8" transform="rotate(288)"/>
        <circle r="1.1" fill="#c25e8e"/>
      </g>
    </symbol>
    <symbol id="bd" viewBox="-10 -10 20 20" overflow="visible">
      <g>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ff9ec2"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ff9ec2" transform="rotate(72)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ff9ec2" transform="rotate(144)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ff9ec2" transform="rotate(216)"/>
        <ellipse cx="0" cy="-5" rx="3.2" ry="4.8" fill="#ff9ec2" transform="rotate(288)"/>
        <circle r="1.1" fill="#a13f74"/>
      </g>
    </symbol>

    <symbol id="petal" viewBox="-4 -3 8 6" overflow="visible">
      <path d="M -3 0 Q -2 -2.6 0 -2.6 Q 2 -2.6 3 0 Q 2 2.2 0 2.6 Q -2 2.2 -3 0 Z" fill="#ffc6da" opacity="0.9"/>
    </symbol>

    <style>
      @media (prefers-reduced-motion: reduce) {
        animate, animateTransform, animateMotion { display: none }
        * { animation: none !important }
      }
    </style>
  </defs>

  <!-- Sky -->
  <rect fill="url(#sky)"/>

  <!-- Sun glow -->
  <circle cx="560" cy="190" r="200" fill="url(#sun)"/>
  <circle cx="560" cy="190" r="34" fill="#fff4e0" opacity="0.85">
    <animate attributeName="opacity" values="0.7;0.95;0.7" dur="6s" repeatCount="indefinite"/>
  </circle>

  <!-- Distant mountains -->
  <path d="M 0 380 L 80 320 L 160 350 L 240 300 L 320 340 L 400 308 L 480 340 L 560 288 L 640 328 L 720 312 L 800 348 L 800 430 L 0 430 Z" fill="url(#hillfar)" opacity="0.65"/>

  <!-- Mid hills -->
  <path d="M 0 425 Q 200 392 400 418 T 800 408 L 800 488 L 0 488 Z" fill="url(#hillmid)" opacity="0.7"/>

  <!-- Foreground ground -->
  <path d="M 0 488 Q 200 470 400 480 T 800 476 L 800 600 L 0 600 Z" fill="url(#ground)"/>

  <!-- Soft grass tufts -->
  <g stroke="#4a5a3a" stroke-width="1.4" fill="none" stroke-linecap="round" opacity="0.6">
    <path d="M 60 540 q 4 -12 8 -14 M 70 540 q 0 -10 3 -14 M 80 540 q -3 -10 -1 -14"/>
    <path d="M 320 555 q 4 -12 8 -14 M 330 555 q 0 -10 3 -14 M 340 555 q -3 -10 -1 -14"/>
    <path d="M 520 545 q 4 -12 8 -14 M 530 545 q 0 -10 3 -14 M 540 545 q -3 -10 -1 -14"/>
    <path d="M 720 558 q 4 -12 8 -14 M 730 558 q 0 -10 3 -14 M 740 558 q -3 -10 -1 -14"/>
  </g>

  <!-- Ground shadows -->
  <ellipse cx="200" cy="528" rx="125" ry="14" fill="#1f2a18" opacity="0.4"/>
  <ellipse cx="600" cy="528" rx="105" ry="12" fill="#1f2a18" opacity="0.4"/>
  <ellipse cx="410" cy="492" rx="55" ry="8" fill="#1f2a18" opacity="0.35"/>

  <!-- Back small tree -->
  <g>
    <animateTransform attributeName="transform" type="rotate" values="-1.2 410 488; 1.2 410 488; -1.2 410 488" dur="7s" repeatCount="indefinite"/>
    <path d="M 410 492 Q 408 465 414 442 Q 418 426 410 410" stroke="#4a3528" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M 414 440 Q 428 432 440 422" stroke="#4a3528" stroke-width="3.5" fill="none" stroke-linecap="round"/>
    <path d="M 410 425 Q 395 420 384 412" stroke="#4a3528" stroke-width="3.5" fill="none" stroke-linecap="round"/>
    <use href="#bl" x="410" y="405" width="14" height="14"/>
    <use href="#bm" x="392" y="410" width="12" height="12"/>
    <use href="#bl" x="432" y="422" width="13" height="13"/>
    <use href="#bm" x="420" y="430" width="11" height="11"/>
    <use href="#bd" x="384" y="412" width="10" height="10"/>
    <use href="#bl" x="405" y="430" width="12" height="12"/>
    <use href="#bl" x="445" y="416" width="11" height="11"/>
  </g>

  <!-- Left main tree -->
  <g>
    <animateTransform attributeName="transform" type="rotate" values="-0.9 200 528; 0.9 200 528; -0.9 200 528" dur="8s" repeatCount="indefinite"/>

    <path d="M 200 528 Q 194 482 206 442 Q 216 402 200 360" stroke="url(#trunk)" stroke-width="22" fill="none" stroke-linecap="round"/>
    <path d="M 200 360 Q 180 340 148 322 Q 128 312 108 308" stroke="url(#trunk)" stroke-width="10" fill="none" stroke-linecap="round"/>
    <path d="M 200 360 Q 232 340 270 326 Q 300 318 322 322" stroke="url(#trunk)" stroke-width="11" fill="none" stroke-linecap="round"/>
    <path d="M 205 412 Q 188 396 162 386 Q 142 380 126 382" stroke="url(#trunk)" stroke-width="9" fill="none" stroke-linecap="round"/>
    <path d="M 205 412 Q 226 396 256 388 Q 282 384 298 390" stroke="url(#trunk)" stroke-width="9" fill="none" stroke-linecap="round"/>
    <path d="M 148 322 Q 134 306 122 296" stroke="#4a3528" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M 270 326 Q 286 308 296 294" stroke="#4a3528" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M 108 308 Q 92 302 82 298" stroke="#4a3528" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M 322 322 Q 342 316 358 320" stroke="#4a3528" stroke-width="3" fill="none" stroke-linecap="round"/>

    <!-- blossom clusters -->
    <use href="#bl" x="108" y="308" width="22" height="22"/>
    <use href="#bm" x="88" y="298" width="20" height="20"/>
    <use href="#bl" x="124" y="292" width="18" height="18"/>
    <use href="#bd" x="98" y="286" width="16" height="16"/>
    <use href="#bl" x="140" y="316" width="20" height="20"/>
    <use href="#bm" x="116" y="320" width="17" height="17"/>
    <use href="#bl" x="78" y="306" width="15" height="15"/>
    <use href="#bd" x="136" y="296" width="14" height="14"/>
    <use href="#bl" x="118" y="300" width="13" height="13"/>

    <use href="#bm" x="290" y="322" width="22" height="22"/>
    <use href="#bl" x="312" y="316" width="20" height="20"/>
    <use href="#bl" x="276" y="306" width="18" height="18"/>
    <use href="#bd" x="296" y="296" width="16" height="16"/>
    <use href="#bl" x="322" y="306" width="18" height="18"/>
    <use href="#bm" x="338" y="322" width="15" height="15"/>
    <use href="#bl" x="262" y="318" width="16" height="16"/>
    <use href="#bd" x="306" y="332" width="15" height="15"/>
    <use href="#bl" x="284" y="312" width="13" height="13"/>

    <use href="#bl" x="196" y="356" width="22" height="22"/>
    <use href="#bm" x="176" y="366" width="18" height="18"/>
    <use href="#bl" x="216" y="366" width="20" height="20"/>
    <use href="#bd" x="202" y="346" width="16" height="16"/>
    <use href="#bl" x="226" y="352" width="17" height="17"/>
    <use href="#bm" x="186" y="346" width="14" height="14"/>

    <use href="#bm" x="164" y="386" width="20" height="20"/>
    <use href="#bl" x="150" y="380" width="16" height="16"/>
    <use href="#bl" x="180" y="396" width="17" height="17"/>
    <use href="#bd" x="134" y="386" width="15" height="15"/>
    <use href="#bl" x="156" y="396" width="14" height="14"/>
    <use href="#bl" x="142" y="392" width="12" height="12"/>

    <use href="#bl" x="256" y="388" width="20" height="20"/>
    <use href="#bm" x="270" y="380" width="16" height="16"/>
    <use href="#bl" x="240" y="382" width="17" height="17"/>
    <use href="#bd" x="286" y="386" width="15" height="15"/>
    <use href="#bl" x="266" y="396" width="14" height="14"/>
    <use href="#bl" x="250" y="372" width="13" height="13"/>
  </g>

  <!-- Right tree -->
  <g>
    <animateTransform attributeName="transform" type="rotate" values="0.9 600 528; -0.9 600 528; 0.9 600 528" dur="9s" repeatCount="indefinite"/>

    <path d="M 600 528 Q 606 482 594 442 Q 584 402 600 372" stroke="url(#trunk)" stroke-width="18" fill="none" stroke-linecap="round"/>
    <path d="M 600 372 Q 580 352 544 336 Q 520 326 500 326" stroke="url(#trunk)" stroke-width="9" fill="none" stroke-linecap="round"/>
    <path d="M 600 372 Q 626 352 660 336 Q 686 326 706 332" stroke="url(#trunk)" stroke-width="9" fill="none" stroke-linecap="round"/>
    <path d="M 600 422 Q 584 406 558 396 Q 538 392 522 394" stroke="url(#trunk)" stroke-width="7" fill="none" stroke-linecap="round"/>
    <path d="M 600 422 Q 616 406 642 398 Q 662 396 678 400" stroke="url(#trunk)" stroke-width="7" fill="none" stroke-linecap="round"/>
    <path d="M 544 336 Q 528 320 520 312" stroke="#4a3528" stroke-width="3" fill="none" stroke-linecap="round"/>
    <path d="M 660 336 Q 676 320 686 314" stroke="#4a3528" stroke-width="3" fill="none" stroke-linecap="round"/>

    <use href="#bl" x="500" y="326" width="20" height="20"/>
    <use href="#bm" x="484" y="320" width="17" height="17"/>
    <use href="#bl" x="516" y="316" width="18" height="18"/>
    <use href="#bd" x="526" y="332" width="15" height="15"/>
    <use href="#bl" x="496" y="308" width="14" height="14"/>
    <use href="#bm" x="510" y="304" width="13" height="13"/>
    <use href="#bl" x="478" y="332" width="13" height="13"/>

    <use href="#bm" x="700" y="332" width="20" height="20"/>
    <use href="#bl" x="684" y="322" width="17" height="17"/>
    <use href="#bl" x="716" y="322" width="18" height="18"/>
    <use href="#bd" x="676" y="326" width="14" height="14"/>
    <use href="#bl" x="690" y="310" width="14" height="14"/>
    <use href="#bl" x="706" y="344" width="13" height="13"/>

    <use href="#bl" x="596" y="368" width="20" height="20"/>
    <use href="#bm" x="580" y="378" width="16" height="16"/>
    <use href="#bl" x="610" y="378" width="17" height="17"/>
    <use href="#bd" x="602" y="358" width="15" height="15"/>
    <use href="#bl" x="588" y="362" width="13" height="13"/>

    <use href="#bl" x="524" y="394" width="18" height="18"/>
    <use href="#bm" x="540" y="386" width="15" height="15"/>
    <use href="#bl" x="510" y="386" width="14" height="14"/>
    <use href="#bd" x="534" y="398" width="13" height="13"/>

    <use href="#bl" x="676" y="400" width="18" height="18"/>
    <use href="#bm" x="660" y="392" width="15" height="15"/>
    <use href="#bl" x="690" y="392" width="14" height="14"/>
    <use href="#bd" x="670" y="408" width="13" height="13"/>
  </g>

  <!-- Drifting petals -->
  <g>
    <g>
      <use href="#petal" width="9" height="7" x="-4.5" y="-3.5">
        <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="3.2s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 140,-30 Q 120,200 160,400 Q 130,560 110,650" dur="11s" repeatCount="indefinite"/>
    </g>
    <g>
      <use href="#petal" width="11" height="8" x="-5.5" y="-4">
        <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="4s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 260,-40 Q 290,180 240,420 Q 280,580 250,650" dur="13s" repeatCount="indefinite" begin="-3s"/>
    </g>
    <g>
      <use href="#petal" width="8" height="6" x="-4" y="-3">
        <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="2.6s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 380,-20 Q 360,220 400,420 Q 370,580 390,650" dur="12s" repeatCount="indefinite" begin="-6s"/>
    </g>
    <g>
      <use href="#petal" width="10" height="7" x="-5" y="-3.5">
        <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="3.6s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 480,-30 Q 510,180 470,400 Q 500,580 480,650" dur="14s" repeatCount="indefinite" begin="-2s"/>
    </g>
    <g>
      <use href="#petal" width="9" height="7" x="-4.5" y="-3.5">
        <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="3s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 600,-40 Q 580,200 620,400 Q 590,560 610,650" dur="12.5s" repeatCount="indefinite" begin="-5s"/>
    </g>
    <g>
      <use href="#petal" width="12" height="9" x="-6" y="-4.5">
        <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="4.4s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 700,-25 Q 670,220 710,420 Q 680,580 700,650" dur="13.5s" repeatCount="indefinite" begin="-8s"/>
    </g>
    <g>
      <use href="#petal" width="8" height="6" x="-4" y="-3">
        <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="2.8s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 80,-30 Q 110,180 70,400 Q 100,560 80,650" dur="10.5s" repeatCount="indefinite" begin="-1s"/>
    </g>
    <g>
      <use href="#petal" width="10" height="7" x="-5" y="-3.5">
        <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="3.4s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 200,-50 Q 170,200 210,420 Q 180,580 200,650" dur="12s" repeatCount="indefinite" begin="-4s"/>
    </g>
    <g>
      <use href="#petal" width="9" height="7" x="-4.5" y="-3.5">
        <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="3.1s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 540,-30 Q 560,200 520,420 Q 550,580 530,650" dur="11.5s" repeatCount="indefinite" begin="-7s"/>
    </g>
    <g>
      <use href="#petal" width="11" height="8" x="-5.5" y="-4">
        <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="3.8s" repeatCount="indefinite"/>
      </use>
      <animateMotion path="M 340,-40 Q 310,200 350,420 Q 320,580 340,650" dur="13s" repeatCount="indefinite" begin="-9s"/>
    </g>
  </g>

  <!-- Subtle warm vignette -->
  <radialGradient id="vig" cx="0.5" cy="0.55" r="0.75">
    <stop offset="0.6" stop-color="#000" stop-opacity="0"/>
    <stop offset="1" stop-color="#3a2030" stop-opacity="0.35"/>
  </radialGradient>
  <rect width="800" height="600" fill="url(#vig)"/>
</svg>
```

</details>

## 7. gpt-oss_120b-cloud

**SVG size:** 3493 characters  
**Complexity:** Low  
**Shape elements:** 9  
**Distinct colors:** 6  
**Raw response:** 4335 characters

![gpt-oss_120b-cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/gpt-oss_120b-cloud.svg)

### Analysis

This SVG contains approximately **9 shape elements** and uses **6 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating petals
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate petal detail. Best for embedding in web pages where small file size matters more than visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 400 300"
   
   
    role="img"
    aria-labelledby="title desc">

  <title id="title">Cherry blossom tree</title>
  <desc id="desc">A stylized cherry‑blossom tree with pink flowers against a light sky.</desc>

  <!-- Background -->
  <rect width="100%" height="100%" fill="#e0f7ff"/>

  <!-- Trunk -->
  <path d="M200 260 C190 250, 190 210, 200 170
           C210 210, 210 250, 200 260"
        fill="#8b5a2b" stroke="#654321" stroke-width="2"/>

  <!-- Branches (simple curved lines) -->
  <path d="M200 180 C150 150, 130 100, 160 70"
        stroke="#8b5a2b" stroke-width="8" fill="none" stroke-linecap="round"/>
  <path d="M200 180 C250 150, 270 100, 240 70"
        stroke="#8b5a2b" stroke-width="8" fill="none" stroke-linecap="round"/>
  <path d="M160 70 C150 50, 165 30, 180 40"
        stroke="#8b5a2b" stroke-width="6" fill="none" stroke-linecap="round"/>
  <path d="M240 70 C250 50, 235 30, 220 40"
        stroke="#8b5a2b" stroke-width="6" fill="none" stroke-linecap="round"/>

  <!-- Blossom cluster definitions (re-used with <use>) -->
  <defs>
    <!-- Single blossom petal group -->
    <g id="petal">
      <ellipse cx="0" cy="0" rx="4" ry="6" fill="#f8c8dc"/>
    </g>

    <!-- Full flower (5 petals + tiny yellow center) -->
    <g id="flower">
      <use href="#petal" transform="rotate(0) translate(0,-8)"/>
      <use href="#petal" transform="rotate(72) translate(0,-8)"/>
      <use href="#petal" transform="rotate(144) translate(0,-8)"/>
      <use href="#petal" transform="rotate(216) translate(0,-8)"/>
      <use href="#petal" transform="rotate(288) translate(0,-8)"/>
      <circle cx="0" cy="0" r="2" fill="#ffedab"/>
    </g>
  </defs>

  <!-- Flowers placed on branches -->
  <g fill-opacity="0.9">
    <!-- Left large branch -->
    <g transform="translate(160,70)">
      <use href="#flower" transform="translate(-30,-15) rotate(-10) scale(1.2)"/>
      <use href="#flower" transform="translate(-10,-30) rotate(15)"/>
      <use href="#flower" transform="translate(5,-10) rotate(-5) scale(0.9)"/>
      <use href="#flower" transform="translate(25,-25) rotate(20) scale(1.1)"/>
      <use href="#flower" transform="translate(40,-5) rotate(-8) scale(0.8)"/>
      <use href="#flower" transform="translate(20,-15) rotate(5) scale(0.95)"/>
    </g>

    <!-- Right large branch -->
    <g transform="translate(240,70)">
      <use href="#flower" transform="translate(-35,-20) rotate(12) scale(1.0)"/>
      <use href="#flower" transform="translate(-20,-5) rotate(-6) scale(0.85)"/>
      <use href="#flower" transform="translate(-5,-25) rotate(18) scale(1.15)"/>
      <use href="#flower" transform="translate(15,-10) rotate(-2) scale(0.9)"/>
      <use href="#flower" transform="translate(30,-20) rotate(10) scale(1.05)"/>
      <use href="#flower" transform="translate(5,-30) rotate(-15) scale(0.8)"/>
    </g>

    <!-- Smaller twigs near the top -->
    <g transform="translate(180,40)">
      <use href="#flower" transform="translate(-8,-12) rotate(5) scale(0.8)"/>
      <use href="#flower" transform="translate(4,-9) rotate(-12) scale(0.75)"/>
    </g>

    <g transform="translate(220,40)">
      <use href="#flower" transform="translate(-7,-13) rotate(8) scale(0.78)"/>
      <use href="#flower" transform="translate(6,-11) rotate(-10) scale(0.70)"/>
    </g>
  </g>

  <!-- Optional ground line -->
  <line x1="0" y1="260" x2="400" y2="260"
        stroke="#6b8e23" stroke-width="3"/>

</svg>
```

</details>

## 8. kimi-k2.6_cloud

**SVG size:** 1678 characters  
**Complexity:** Low  
**Shape elements:** 3  
**Distinct colors:** 7  
**Raw response:** 16993 characters

![kimi-k2.6_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/kimi-k2.6_cloud.svg)

### Analysis

This SVG contains approximately **3 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for skies and petals)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate petal detail. Best for embedding in web pages where small file size matters more than visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" id="sakuraCanvas" preserveAspectRatio="xMidYMid slice">
        <defs>
            <!-- Gradient for Background -->
            <linearGradient id="bgGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#2b32b2;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#1488cc;stop-opacity:1" />
            </linearGradient>

            <!-- Gradient for Tree Trunk -->
            <linearGradient id="trunkGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#4e342e;stop-opacity:1" />
                <stop offset="40%" style="stop-color:#3e2723;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#5d4037;stop-opacity:1" />
            </linearGradient>

            <!-- Filter for soft glow -->
            <filter id="softGlow" x="-75%" y="-75%">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>

        <!-- Background Rect -->
        <rect id="sky" width="100%" height="100%" fill="url(#bgGradient)" />
        
        <!-- Moon/Sun -->
        <circle cx="80%" cy="20%" r="60" fill="#ffffff" fill-opacity="0.1" filter="url(#softGlow)" />

        <!-- Ground -->
        <path id="ground" d="" fill="#1a237e" fill-opacity="0.2" />

        <!-- Tree Container -->
        <g id="treeGroup"></g>

        <!-- Falling Petals Container -->
        <g id="petalGroup"></g>
    </svg>
```

</details>

## 9. minimax-m2.7_cloud

**SVG size:** 3319 characters  
**Complexity:** Low  
**Shape elements:** 20  
**Distinct colors:** 11  
**Raw response:** 3627 characters

![minimax-m2.7_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/minimax-m2.7_cloud.svg)

### Analysis

This SVG contains approximately **20 shape elements** and uses **11 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating petals
- Includes gradient fills for richer visual depth (great for skies and petals)
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate petal detail. Best for embedding in web pages where small file size matters more than visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Gradient for the Sky -->
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#E0F7FA;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#B2EBF2;stop-opacity:1" />
    </linearGradient>

    <!-- Gradient for the Hill -->
    <linearGradient id="hillGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#AED581;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#8BC34A;stop-opacity:1" />
    </linearGradient>

    <!-- Reusable Blossom Cluster -->
    <g id="blossomCluster">
      <circle cx="0" cy="0" r="15" fill="#F48FB1" opacity="0.9"/>
      <circle cx="12" cy="-6" r="12" fill="#F8BBD0" opacity="0.9"/>
      <circle cx="-12" cy="-6" r="12" fill="#F8BBD0" opacity="0.9"/>
      <circle cx="6" cy="12" r="12" fill="#F8BBD0" opacity="0.9"/>
      <circle cx="-6" cy="12" r="12" fill="#F06292" opacity="0.8"/>
      <circle cx="0" cy="0" r="5" fill="#FFF" opacity="0.6"/> <!-- Center Glow -->
    </g>
  </defs>

  <!-- Background: Sky -->
  <rect width="100%" height="100%" fill="url(#skyGradient)" />
  
  <!-- Sun -->
  <circle cx="500" cy="80" r="40" fill="#FFF59D" opacity="0.8" />

  <!-- Background: Hills -->
  <path d="M0 520 Q 200 450 400 500 T 600 480 V 600 H 0 Z" fill="url(#hillGradient)" opacity="0.8"/>
  <path d="M0 580 Q 300 530 600 580 V 600 H 0 Z" fill="#689F38" opacity="0.9"/>

  <!-- Tree Trunk and Branches -->
  <g stroke="#5D4037" stroke-linecap="round" fill="none">
    <!-- Main Trunk -->
    <path d="M300 550 Q 290 450 290 350 Q 290 250 210 200" stroke-width="20" />
    <!-- Main Branches -->
    <path d="M290 350 Q 360 300 410 230" stroke-width="14" />
    <path d="M290 250 Q 260 200 210 160" stroke-width="12" />
    <!-- Sub Branches -->
    <path d="M210 200 Q 180 180 150 150" stroke-width="6" />
    <path d="M410 230 Q 440 210 460 180" stroke-width="6" />
    <path d="M290 300 Q 250 280 220 250" stroke-width="6" />
  </g>

  <!-- Blossoms (Using the defined clusters) -->
  <!-- Top Canopy -->
  <use href="#blossomCluster" x="210" y="160" transform="scale(2.5)" />
  <use href="#blossomCluster" x="150" y="150" transform="scale(1.8)" />
  <use href="#blossomCluster" x="280" y="130" transform="scale(1.5)" />

  <!-- Right Branch Canopy -->
  <use href="#blossomCluster" x="410" y="230" transform="scale(2.2)" />
  <use href="#blossomCluster" x="460" y="180" transform="scale(1.8)" />
  <use href="#blossomCluster" x="380" y="180" transform="scale(1.5)" />

  <!-- Middle Canopy Fill -->
  <use href="#blossomCluster" x="250" y="240" transform="scale(1.8)" />
  <use href="#blossomCluster" x="290" y="350" transform="scale(1.5)" />
  <use href="#blossomCluster" x="320" y="300" transform="scale(1.8)" />
  <use href="#blossomCluster" x="250" y="300" transform="scale(1.2)" />
  
  <!-- Falling Petals -->
  <g fill="#F48FB1" opacity="0.7">
    <ellipse cx="100" cy="350" rx="6" ry="4" transform="rotate(30 100 350)" />
    <ellipse cx="480" cy="400" rx="6" ry="4" transform="rotate(-15 480 400)" />
    <ellipse cx="350" cy="450" rx="5" ry="3" transform="rotate(45 350 450)" />
    <ellipse cx="200" cy="420" rx="6" ry="4" transform="rotate(-30 200 420)" />
  </g>
</svg>
```

</details>

## 10. minimax-m3_cloud

**SVG size:** 15741 characters  
**Complexity:** Very high  
**Shape elements:** 171  
**Distinct colors:** 15  
**Raw response:** 16614 characters

![minimax-m3_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/minimax-m3_cloud.svg)

### Analysis

This SVG contains approximately **171 shape elements** and uses **15 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for skies and petals)
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

With over 15,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering individual petals, layered branches, and atmospheric backgrounds. Best for users who want a painterly cherry blossom scene.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <!-- Sky gradient background -->
  <defs>
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FFE5F0;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#FFC0D0;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFB6C1;stop-opacity:1" />
    </linearGradient>
    <radialGradient id="sunGlow" cx="80%" cy="20%" r="40%">
      <stop offset="0%" style="stop-color:#FFF8DC;stop-opacity:0.8" />
      <stop offset="100%" style="stop-color:#FFF8DC;stop-opacity:0" />
    </radialGradient>
    <linearGradient id="trunkGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#4A2C1A" />
      <stop offset="50%" style="stop-color:#6B4423" />
      <stop offset="100%" style="stop-color:#4A2C1A" />
    </linearGradient>
  </defs>
  
  <!-- Sky background -->
  <rect width="800" height="600" fill="url(#skyGradient)"/>
  
  <!-- Sun glow -->
  <circle cx="650" cy="120" r="80" fill="url(#sunGlow)"/>
  <circle cx="650" cy="120" r="35" fill="#FFF8DC" opacity="0.9"/>
  
  <!-- Distant hills -->
  <path d="M0 450 Q200 380 400 420 Q600 450 800 400 L800 600 L0 600 Z" fill="#E8B4C8" opacity="0.4"/>
  <path d="M0 480 Q150 440 350 460 Q550 480 800 450 L800 600 L0 600 Z" fill="#D4A0B8" opacity="0.5"/>
  
  <!-- Ground/grass -->
  <rect x="0" y="500" width="800" height="100" fill="#90B88C"/>
  <path d="M0 500 Q400 490 800 500 L800 600 L0 600 Z" fill="#7AA078"/>
  
  <!-- Falling petals background -->
  <g opacity="0.6">
    <ellipse cx="50" cy="100" rx="4" ry="3" fill="#FFB6C1" transform="rotate(20 50 100)"/>
    <ellipse cx="150" cy="180" rx="3" ry="2" fill="#FF99B0" transform="rotate(-30 150 180)"/>
    <ellipse cx="250" cy="80" rx="4" ry="3" fill="#FFC0D0" transform="rotate(45 250 80)"/>
    <ellipse cx="350" cy="200" rx="3" ry="2" fill="#FFB6C1" transform="rotate(15 350 200)"/>
    <ellipse cx="500" cy="150" rx="4" ry="3" fill="#FF99B0" transform="rotate(-25 500 150)"/>
    <ellipse cx="600" cy="250" rx="3" ry="2" fill="#FFC0D0" transform="rotate(60 600 250)"/>
    <ellipse cx="700" cy="180" rx="4" ry="3" fill="#FFB6C1" transform="rotate(-40 700 180)"/>
    <ellipse cx="100" cy="300" rx="3" ry="2" fill="#FF99B0" transform="rotate(30 100 300)"/>
    <ellipse cx="200" cy="380" rx="4" ry="3" fill="#FFC0D0" transform="rotate(-15 200 380)"/>
    <ellipse cx="450" cy="350" rx="3" ry="2" fill="#FFB6C1" transform="rotate(50 450 350)"/>
    <ellipse cx="750" cy="320" rx="4" ry="3" fill="#FF99B0" transform="rotate(-35 750 320)"/>
  </g>
  
  <!-- Large cherry blossom tree (left) -->
  <g>
    <!-- Trunk -->
    <path d="M120 500 Q115 400 125 300 Q130 250 135 200" stroke="url(#trunkGradient)" stroke-width="28" fill="none" stroke-linecap="round"/>
    <!-- Main branches -->
    <path d="M125 300 Q90 280 60 250" stroke="#5C3A1E" stroke-width="10" fill="none" stroke-linecap="round"/>
    <path d="M125 280 Q160 260 200 240" stroke="#5C3A1E" stroke-width="10" fill="none" stroke-linecap="round"/>
    <path d="M128 250 Q100 220 70 180" stroke="#5C3A1E" stroke-width="8" fill="none" stroke-linecap="round"/>
    <path d="M130 230 Q160 200 195 170" stroke="#5C3A1E" stroke-width="8" fill="none" stroke-linecap="round"/>
    <path d="M133 210 Q110 170 85 130" stroke="#5C3A1E" stroke-width="6" fill="none" stroke-linecap="round"/>
    <path d="M135 200 Q165 170 195 140" stroke="#5C3A1E" stroke-width="6" fill="none" stroke-linecap="round"/>
    <!-- Small twigs -->
    <path d="M60 250 Q40 230 30 210" stroke="#5C3A1E" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M60 250 Q70 230 75 215" stroke="#5C3A1E" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M200 240 Q220 220 230 200" stroke="#5C3A1E" stroke-width="4" fill="none" stroke-linecap="round"/>
    <path d="M200 240 Q210 215 215 195" stroke="#5C3A1E" stroke-width="4" fill="none" stroke-linecap="round"/>
  </g>
  
  <!-- Cherry blossom clusters for large tree -->
  <g>
    <!-- Cluster 1 - top -->
    <g transform="translate(135, 180)">
      <circle cx="0" cy="0" r="35" fill="#FFB6C1" opacity="0.9"/>
      <circle cx="-15" cy="-10" r="25" fill="#FFC0CB" opacity="0.9"/>
      <circle cx="15" cy="-5" r="28" fill="#FFB6C1" opacity="0.9"/>
      <circle cx="5" cy="15" r="22" fill="#FFC0D0" opacity="0.9"/>
    </g>
    <!-- Cluster 2 - left -->
    <g transform="translate(60, 240)">
      <circle cx="0" cy="0" r="30" fill="#FFC0CB" opacity="0.9"/>
      <circle cx="-12" cy="8" r="22" fill="#FFB6C1" opacity="0.9"/>
      <circle cx="10" cy="10" r="20" fill="#FFC0D0" opacity="0.9"/>
    </g>
    <!-- Cluster 3 - right -->
    <g transform="translate(210, 230)">
      <circle cx="0" cy="0" r="32" fill="#FFB6C1" opacity="0.9"/>
      <circle cx="-10" cy="-8" r="24" fill="#FFC0CB" opacity="0.9"/>
      <circle cx="12" cy="5" r="26" fill="#FFC0D0" opacity="0.9"/>
    </g>
    <!-- Cluster 4 -->
    <g transform="translate(80, 170)">
      <circle cx="0" cy="0" r="28" fill="#FFC0CB" opacity="0.9"/>
      <circle cx="10" cy="-5" r="22" fill="#FFB6C1" opacity="0.9"/>
    </g>
    <!-- Cluster 5 -->
    <g transform="translate(195, 155)">
      <circle cx="0" cy="0" r="28" fill="#FFC0D0" opacity="0.9"/>
      <circle cx="-8" cy="8" r="22" fill="#FFB6C1" opacity="0.9"/>
    </g>
  </g>
  
  <!-- Individual blossoms on large tree -->
  <g>
    <!-- Blossom 1 -->
    <g transform="translate(140, 170)">
      <circle cx="0" cy="-5" r="6" fill="#FFC0CB"/>
      <circle cx="4" cy="-3" r="6" fill="#FFC0CB"/>
      <circle cx="4" cy="3" r="6" fill="#FFB6C1"/>
      <circle cx="0" cy="5" r="6" fill="#FFC0CB"/>
      <circle cx="-4" cy="3" r="6" fill="#FFB6C1"/>
      <circle cx="0" cy="0" r="3" fill="#FF69B4"/>
    </g>
    <!-- Blossom 2 -->
    <g transform="translate(70, 200)">
      <circle cx="0" cy="-5" r="5" fill="#FFC0D0"/>
      <circle cx="4" cy="-2" r="5" fill="#FFB6C1"/>
      <circle cx="3" cy="4" r="5" fill="#FFC0CB"/>
      <circle cx="-3" cy="4" r="5" fill="#FFB6C1"/>
      <circle cx="-4" cy="-2" r="5" fill="#FFC0D0"/>
      <circle cx="0" cy="0" r="2" fill="#FF1493"/>
    </g>
    <!-- Blossom 3 -->
    <g transform="translate(220, 195)">
      <circle cx="0" cy="-5" r="5" fill="#FFB6C1"/>
      <circle cx="4" cy="-2" r="5" fill="#FFC0D0"/>
      <circle cx="3" cy="4" r="5" fill="#FFB6C1"/>
      <circle cx="-3" cy="4" r="5" fill="#FFC0CB"/>
      <circle cx="-4" cy="-2" r="5" fill="#FFB6C1"/>
      <circle cx="0" cy="0" r="2" fill="#FF69B4"/>
    </g>
    <!-- More blossoms scattered -->
    <g transform="translate(110, 150)">
      <circle cx="0" cy="-4" r="4" fill="#FFC0CB"/>
      <circle cx="3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="2" cy="3" r="4" fill="#FFC0D0"/>
      <circle cx="-2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFC0CB"/>
      <circle cx="0" cy="0" r="2" fill="#FF1493"/>
    </g>
    <g transform="translate(170, 130)">
      <circle cx="0" cy="-4" r="4" fill="#FFB6C1"/>
      <circle cx="3" cy="-2" r="4" fill="#FFC0D0"/>
      <circle cx="2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-2" cy="3" r="4" fill="#FFC0CB"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="0" cy="0" r="2" fill="#FF69B4"/>
    </g>
    <g transform="translate(50, 215)">
      <circle cx="0" cy="-4" r="4" fill="#FFC0D0"/>
      <circle cx="3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="2" cy="3" r="4" fill="#FFC0CB"/>
      <circle cx="-2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFC0D0"/>
      <circle cx="0" cy="0" r="2" fill="#FF1493"/>
    </g>
    <g transform="translate(240, 210)">
      <circle cx="0" cy="-4" r="4" fill="#FFB6C1"/>
      <circle cx="3" cy="-2" r="4" fill="#FFC0CB"/>
      <circle cx="2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-2" cy="3" r="4" fill="#FFC0D0"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="0" cy="0" r="2" fill="#FF69B4"/>
    </g>
    <g transform="translate(90, 240)">
      <circle cx="0" cy="-4" r="4" fill="#FFC0CB"/>
      <circle cx="3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="2" cy="3" r="4" fill="#FFC0D0"/>
      <circle cx="-2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFC0CB"/>
      <circle cx="0" cy="0" r="2" fill="#FF1493"/>
    </g>
  </g>
  
  <!-- Medium cherry blossom tree (center-right) -->
  <g>
    <!-- Trunk -->
    <path d="M450 500 Q445 420 455 350 Q460 310 465 280" stroke="url(#trunkGradient)" stroke-width="22" fill="none" stroke-linecap="round"/>
    <!-- Branches -->
    <path d="M455 350 Q425 330 395 310" stroke="#5C3A1E" stroke-width="8" fill="none" stroke-linecap="round"/>
    <path d="M455 340 Q485 320 515 305" stroke="#5C3A1E" stroke-width="8" fill="none" stroke-linecap="round"/>
    <path d="M460 310 Q435 290 410 270" stroke="#5C3A1E" stroke-width="6" fill="none" stroke-linecap="round"/>
    <path d="M462 300 Q485 275 510 260" stroke="#5C3A1E" stroke-width="6" fill="none" stroke-linecap="round"/>
    <path d="M465 280 Q445 250 425 220" stroke="#5C3A1E" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M465 270 Q490 245 510 215" stroke="#5C3A1E" stroke-width="5" fill="none" stroke-linecap="round"/>
  </g>
  
  <!-- Blossoms for medium tree -->
  <g>
    <g transform="translate(465, 260)">
      <circle cx="0" cy="0" r="30" fill="#FFC0CB" opacity="0.9"/>
      <circle cx="-12" cy="-8" r="22" fill="#FFB6C1" opacity="0.9"/>
      <circle cx="12" cy="-5" r="24" fill="#FFC0D0" opacity="0.9"/>
      <circle cx="5" cy="12" r="20" fill="#FFB6C1" opacity="0.9"/>
    </g>
    <g transform="translate(400, 300)">
      <circle cx="0" cy="0" r="25" fill="#FFC0D0" opacity="0.9"/>
      <circle cx="-10" cy="5" r="20" fill="#FFB6C1" opacity="0.9"/>
    </g>
    <g transform="translate(515, 295)">
      <circle cx="0" cy="0" r="26" fill="#FFB6C1" opacity="0.9"/>
      <circle cx="10" cy="6" r="20" fill="#FFC0CB" opacity="0.9"/>
    </g>
    <g transform="translate(420, 250)">
      <circle cx="0" cy="0" r="22" fill="#FFC0CB" opacity="0.9"/>
      <circle cx="8" cy="-5" r="18" fill="#FFB6C1" opacity="0.9"/>
    </g>
    <g transform="translate(510, 245)">
      <circle cx="0" cy="0" r="22" fill="#FFC0D0" opacity="0.9"/>
      <circle cx="-8" cy="5" r="18" fill="#FFB6C1" opacity="0.9"/>
    </g>
    <!-- Individual blossoms -->
    <g transform="translate(440, 230)">
      <circle cx="0" cy="-4" r="4" fill="#FFB6C1"/>
      <circle cx="3" cy="-2" r="4" fill="#FFC0D0"/>
      <circle cx="2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-2" cy="3" r="4" fill="#FFC0CB"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="0" cy="0" r="2" fill="#FF69B4"/>
    </g>
    <g transform="translate(490, 220)">
      <circle cx="0" cy="-4" r="4" fill="#FFC0CB"/>
      <circle cx="3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="2" cy="3" r="4" fill="#FFC0D0"/>
      <circle cx="-2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFC0CB"/>
      <circle cx="0" cy="0" r="2" fill="#FF1493"/>
    </g>
  </g>
  
  <!-- Small cherry blossom tree (right) -->
  <g>
    <!-- Trunk -->
    <path d="M680 500 Q678 440 685 380 Q688 350 690 330" stroke="url(#trunkGradient)" stroke-width="18" fill="none" stroke-linecap="round"/>
    <!-- Branches -->
    <path d="M685 380 Q660 365 635 350" stroke="#5C3A1E" stroke-width="6" fill="none" stroke-linecap="round"/>
    <path d="M685 370 Q710 355 730 345" stroke="#5C3A1E" stroke-width="6" fill="none" stroke-linecap="round"/>
    <path d="M688 350 Q670 330 650 310" stroke="#5C3A1E" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M690 340 Q710 320 725 305" stroke="#5C3A1E" stroke-width="5" fill="none" stroke-linecap="round"/>
  </g>
  
  <!-- Blossoms for small tree -->
  <g>
    <g transform="translate(690, 310)">
      <circle cx="0" cy="0" r="25" fill="#FFC0CB" opacity="0.9"/>
      <circle cx="-10" cy="-5" r="20" fill="#FFB6C1" opacity="0.9"/>
      <circle cx="10" cy="0" r="22" fill="#FFC0D0" opacity="0.9"/>
    </g>
    <g transform="translate(640, 340)">
      <circle cx="0" cy="0" r="20" fill="#FFB6C1" opacity="0.9"/>
      <circle cx="8" cy="5" r="16" fill="#FFC0CB" opacity="0.9"/>
    </g>
    <g transform="translate(730, 335)">
      <circle cx="0" cy="0" r="20" fill="#FFC0D0" opacity="0.9"/>
      <circle cx="-8" cy="5" r="16" fill="#FFB6C1" opacity="0.9"/>
    </g>
    <g transform="translate(660, 300)">
      <circle cx="0" cy="0" r="18" fill="#FFC0CB" opacity="0.9"/>
    </g>
    <g transform="translate(720, 295)">
      <circle cx="0" cy="0" r="18" fill="#FFB6C1" opacity="0.9"/>
    </g>
    <!-- Individual blossoms -->
    <g transform="translate(675, 280)">
      <circle cx="0" cy="-4" r="4" fill="#FFB6C1"/>
      <circle cx="3" cy="-2" r="4" fill="#FFC0D0"/>
      <circle cx="2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-2" cy="3" r="4" fill="#FFC0CB"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="0" cy="0" r="2" fill="#FF69B4"/>
    </g>
    <g transform="translate(705, 275)">
      <circle cx="0" cy="-4" r="4" fill="#FFC0CB"/>
      <circle cx="3" cy="-2" r="4" fill="#FFB6C1"/>
      <circle cx="2" cy="3" r="4" fill="#FFC0D0"/>
      <circle cx="-2" cy="3" r="4" fill="#FFB6C1"/>
      <circle cx="-3" cy="-2" r="4" fill="#FFC0CB"/>
      <circle cx="0" cy="0" r="2" fill="#FF1493"/>
    </g>
  </g>
  
  <!-- More falling petals in foreground -->
  <g>
    <ellipse cx="80" cy="450" rx="5" ry="3" fill="#FFB6C1" transform="rotate(25 80 450)"/>
    <ellipse cx="180" cy="480" rx="4" ry="3" fill="#FF99B0" transform="rotate(-20 180 480)"/>
    <ellipse cx="300" cy="460" rx="5" ry="3" fill="#FFC0D0" transform="rotate(40 300 460)"/>
    <ellipse cx="380" cy="490" rx="4" ry="3" fill="#FFB6C1" transform="rotate(-15 380 490)"/>
    <ellipse cx="520" cy="470" rx="5" ry="3" fill="#FF99B0" transform="rotate(30 520 470)"/>
    <ellipse cx="600" cy="450" rx="4" ry="3" fill="#FFC0D0" transform="rotate(-35 600 450)"/>
    <ellipse cx="720" cy="480" rx="5" ry="3" fill="#FFB6C1" transform="rotate(20 720 480)"/>
    <ellipse cx="250" cy="430" rx="4" ry="3" fill="#FF99B0" transform="rotate(45 250 430)"/>
    <ellipse cx="420" cy="440" rx="5" ry="3" fill="#FFC0D0" transform="rotate(-25 420 440)"/>
    <ellipse cx="560" cy="425" rx="4" ry="3" fill="#FFB6C1" transform="rotate(50 560 425)"/>
  </g>
  
  <!-- Petals on the ground -->
  <g opacity="0.7">
    <ellipse cx="100" cy="540" rx="5" ry="3" fill="#FFB6C1" transform="rotate(20 100 540)"/>
    <ellipse cx="200" cy="555" rx="4" ry="3" fill="#FF99B0" transform="rotate(-30 200 555)"/>
    <ellipse cx="350" cy="545" rx="5" ry="3" fill="#FFC0D0" transform="rotate(45 350 545)"/>
    <ellipse cx="480" cy="560" rx="4" ry="3" fill="#FFB6C1" transform="rotate(15 480 560)"/>
    <ellipse cx="620" cy="550" rx="5" ry="3" fill="#FF99B0" transform="rotate(-25 620 550)"/>
    <ellipse cx="750" cy="545" rx="4" ry="3" fill="#FFC0D0" transform="rotate(40 750 545)"/>
    <ellipse cx="150" cy="570" rx="4" ry="3" fill="#FFB6C1" transform="rotate(-10 150 570)"/>
    <ellipse cx="400" cy="575" rx="5" ry="3" fill="#FF99B0" transform="rotate(35 400 575)"/>
    <ellipse cx="550" cy="580" rx="4" ry="3" fill="#FFC0D0" transform="rotate(-40 550 580)"/>
  </g>
  
  <!-- Small flowers/grass on ground -->
  <g>
    <ellipse cx="50" cy="510" rx="3" ry="6" fill="#7AA078"/>
    <ellipse cx="250" cy="515" rx="3" ry="6" fill="#7AA078"/>
    <ellipse cx="450" cy="512" rx="3" ry="6" fill="#7AA078"/>
    <ellipse cx="650" cy="518" rx="3" ry="6" fill="#7AA078"/>
  </g>
</svg>
```

</details>

## 11. nemotron-3-super_cloud

**SVG size:** 2648 characters  
**Complexity:** Low  
**Shape elements:** 42  
**Distinct colors:** 3  
**Raw response:** 4190 characters

![nemotron-3-super_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/nemotron-3-super_cloud.svg)

### Analysis

This SVG contains approximately **42 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are not prominent -- the model may have used a generic green/brown palette

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate petal detail. Best for embedding in web pages where small file size matters more than visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="100%" height="100%" fill="white"/>
  
  <!-- Tree Trunk & Branches -->
  <path d="M100,250 
           C90,200 110,180 100,150 
           C90,120 110,100 100,70"
        stroke="#8B4513" 
        stroke-width="8" 
        fill="none"
        stroke-linecap="round"/>
  
  <!-- Secondary Branches -->
  <path d="M100,150 
           C80,130 60,140 50,120"
        stroke="#8B4513" 
        stroke-width="4" 
        fill="none"/>
  <path d="M100,150 
           C120,130 140,140 150,120"
        stroke="#8B4513" 
        stroke-width="4" 
        fill="none"/>
  <path d="M50,120 
           C40,100 30,90 20,80"
        stroke="#8B4513" 
        stroke-width="2" 
        fill="none"/>
  <path d="M150,120 
           C160,100 170,90 180,80"
        stroke="#8B4513" 
        stroke-width="2" 
        fill="none"/>
  
  <!-- Cherry Blossom Clusters -->
  <!-- Left Branch Blossoms -->
  <g fill="#FFB6C1" opacity="0.9">
    <circle cx="20" cy="80" r="3"/>
    <circle cx="18" cy="75" r="2.5"/>
    <circle cx="22" cy="78" r="2.5"/>
    <circle cx="40" cy="100" r="3"/>
    <circle cx="38" cy="95" r="2.5"/>
    <circle cx="42" cy="98" r="2.5"/>
    <circle cx="50" cy="120" r="3"/>
    <circle cx="48" cy="115" r="2.5"/>
    <circle cx="52" cy="118" r="2.5"/>
  </g>
  
  <!-- Right Branch Blossoms -->
  <g fill="#FFB6C1" opacity="0.9">
    <circle cx="180" cy="80" r="3"/>
    <circle cx="178" cy="75" r="2.5"/>
    <circle cx="182" cy="78" r="2.5"/>
    <circle cx="160" cy="100" r="3"/>
    <circle cx="158" cy="95" r="2.5"/>
    <circle cx="162" cy="98" r="2.5"/>
    <circle cx="150" cy="120" r="3"/>
    <circle cx="148" cy="115" r="2.5"/>
    <circle cx="152" cy="118" r="2.5"/>
  </g>
  
  <!-- Upper Branch Blossoms -->
  <g fill="#FFB6C1" opacity="0.85">
    <circle cx="80" cy="110" r="2.5"/>
    <circle cx="78" cy="105" r="2"/>
    <circle cx="82" cy="108" r="2"/>
    <circle cx="120" cy="110" r="2.5"/>
    <circle cx="118" cy="105" r="2"/>
    <circle cx="122" cy="108" r="2"/>
    <circle cx="100" cy="90" r="3"/>
    <circle cx="98" cy="85" r="2.5"/>
    <circle cx="102" cy="88" r="2.5"/>
  </g>
  
  <!-- Blossom Centers (for depth) -->
  <g fill="#FF69B4" opacity="0.7">
    <circle cx="20" cy="80" r="1"/>
    <circle cx="40" cy="100" r="1"/>
    <circle cx="50" cy="120" r="1"/>
    <circle cx="180" cy="80" r="1"/>
    <circle cx="160" cy="100" r="1"/>
    <circle cx="150" cy="120" r="1"/>
    <circle cx="80" cy="110" r="0.8"/>
    <circle cx="120" cy="110" r="0.8"/>
    <circle cx="100" cy="90" r="1"/>
  </g>
</svg>
```

</details>

## 12. nemotron-3-ultra_cloud

**SVG size:** 16411 characters  
**Complexity:** Very high  
**Shape elements:** 32  
**Distinct colors:** 20  
**Raw response:** 17911 characters

![nemotron-3-ultra_cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/nemotron-3-ultra_cloud.svg)

### Analysis

This SVG contains approximately **32 shape elements** and uses **20 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating petals
- Includes gradient fills for richer visual depth (great for skies and petals)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Includes animation elements -- the model attempted motion (e.g., falling petals)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

With over 15,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering individual petals, layered branches, and atmospheric backgrounds. Best for users who want a painterly cherry blossom scene.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <!-- ============================================================
       DEFINITIONS: Gradients, Filters & Reusable Components
       ============================================================ -->
  <defs>
    <!-- Sky Gradient (Dawn/Dusk) -->
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2c3e50" />
      <stop offset="40%" stop-color="#3498db" />
      <stop offset="70%" stop-color="#f8c8dc" />
      <stop offset="100%" stop-color="#fff0f5" />
    </linearGradient>

    <!-- Ground Gradient -->
    <linearGradient id="groundGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2e8b57" />
      <stop offset="100%" stop-color="#1a5c33" />
    </linearGradient>

    <!-- Trunk Gradient (Bark Texture) -->
    <linearGradient id="trunkGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#4a3728" />
      <stop offset="30%" stop-color="#5d4037" />
      <stop offset="50%" stop-color "#3e2723" />
      <stop offset="70%" stop-color="#5d4037" />
      <stop offset="100%" stop-color="#4a3728" />
    </linearGradient>

    <!-- Petal Gradient (Pink to White) -->
    <radialGradient id="petalGradient" cx="50%" cy="30%" r="60%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.9"/>
      <stop offset="60%" stop-color="#ffb7c5"/>
      <stop offset="100%" stop-color="#ff8fab" stop-opacity="0.8"/>
    </radialGradient>

    <!-- Petal Gradient (Darker Pink for depth) -->
    <radialGradient id="petalGradientDark" cx="50%" cy="30%" r="60%">
      <stop offset="0%" stop-color="#ffd1dc"/>
      <stop offset="60%" stop-color="#ff6b8a"/>
      <stop offset="100%" stop-color="#e84d6e" stop-opacity="0.9"/>
    </radialGradient>

    <!-- Flower Center Gradient -->
    <radialGradient id="centerGradient" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fff9c4"/>
      <stop offset="100%" stop-color="#fdd835"/>
    </radialGradient>

    <!-- Soft Glow Filter for Blossoms -->
    <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="1.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <!-- Falling Petal Symbol (Reusable) -->
    <symbol id="fallingPetal" viewBox="-10 -10 20 20">
      <path d="M0,-10 C10,-10 10,10 0,10 C-10,10 -10,-10 0,-10 Z" fill="url(#petalGradient)" opacity="0.85"/>
    </symbol>

    <!-- Blossom Cluster Symbol (Reusable) -->
    <symbol id="blossomCluster" viewBox="-30 -30 60 60">
      <!-- Back Layer Petals -->
      <g fill="url(#petalGradientDark)" opacity="0.6">
        <path d="M0,-25 C-8,-30 -20,-20 -15,-10 C-10,0 0,5 0,0 Z" transform="rotate(0)"/>
        <path d="M0,-25 C8,-30 20,-20 15,-10 C10,0 0,5 0,0 Z" transform="rotate(0)"/>
        <path d="M0,-25 C-8,-30 -20,-20 -15,-10 C-10,0 0,5 0,0 Z" transform="rotate(72)"/>
        <path d="M0,-25 C8,-30 20,-20 15,-10 C10,0 0,5 0,0 Z" transform="rotate(72)"/>
        <path d="M0,-25 C-8,-30 -20,-20 -15,-10 C-10,0 0,5 0,0 Z" transform="rotate(144)"/>
        <path d="M0,-25 C8,-30 20,-20 15,-10 C10,0 0,5 0,0 Z" transform="rotate(144)"/>
        <path d="M0,-25 C-8,-30 -20,-20 -15,-10 C-10,0 0,5 0,0 Z" transform="rotate(216)"/>
        <path d="M0,-25 C8,-30 20,-20 15,-10 C10,0 0,5 0,0 Z" transform="rotate(216)"/>
        <path d="M0,-25 C-8,-30 -20,-20 -15,-10 C-10,0 0,5 0,0 Z" transform="rotate(288)"/>
        <path d="M0,-25 C8,-30 20,-20 15,-10 C10,0 0,5 0,0 Z" transform="rotate(288)"/>
      </g>
      <!-- Front Layer Petals -->
      <g fill="url(#petalGradient)" filter="url(#softGlow)">
        <path d="M0,-20 C-6,-24 -16,-16 -12,-8 C-8,0 0,3 0,0 Z" transform="rotate(36)"/>
        <path d="M0,-20 C6,-24 16,-16 12,-8 C8,0 0,3 0,0 Z" transform="rotate(36)"/>
        <path d="M0,-20 C-6,-24 -16,-16 -12,-8 C-8,0 0,3 0,0 Z" transform="rotate(108)"/>
        <path d="M0,-20 C6,-24 16,-16 12,-8 C8,0 0,3 0,0 Z" transform="rotate(108)"/>
        <path d="M0,-20 C-6,-24 -16,-16 -12,-8 C-8,0 0,3 0,0 Z" transform="rotate(180)"/>
        <path d="M0,-20 C6,-24 16,-16 12,-8 C8,0 0,3 0,0 Z" transform="rotate(180)"/>
        <path d="M0,-20 C-6,-24 -16,-16 -12,-8 C-8,0 0,3 0,0 Z" transform="rotate(252)"/>
        <path d="M0,-20 C6,-24 16,-16 12,-8 C8,0 0,3 0,0 Z" transform="rotate(252)"/>
        <path d="M0,-20 C-6,-24 -16,-16 -12,-8 C-8,0 0,3 0,0 Z" transform="rotate(324)"/>
        <path d="M0,-20 C6,-24 16,-16 12,-8 C8,0 0,3 0,0 Z" transform="rotate(324)"/>
      </g>
      <!-- Center -->
      <circle cx="0" cy="0" r="5" fill="url(#centerGradient)">
        <animate attributeName="r" values="4;6;4" dur="3s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="1;0.7;1" dur="3s" repeatCount="indefinite" />
      </circle>
    </symbol>

    <!-- Branch Path Definition for Tree 1 (Left) -->
    <path id="branchPath1" fill="none" stroke="url(#trunkGradient)" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" d="
      M 150, 550 
      C 150, 450  180, 380  200, 320
      M 180, 420 
      C 140, 380  120, 320  100, 280
      M 190, 380 
      C 230, 340  260, 300  280, 250
      M 210, 300 
      C 250, 260  290, 220  300, 180
    "/>

    <!-- Branch Path Definition for Tree 2 (Right) -->
    <path id="branchPath2" fill="none" stroke="url(#trunkGradient)" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" d="
      M 650, 550 
      C 650, 460  620, 390  600, 330
      M 620, 430 
      C 660, 390  680, 330  700, 290
      M 610, 380 
      C 570, 340  540, 300  520, 250
      M 590, 320 
      C 550, 280  510, 240  500, 200
    "/>
    
    <!-- Branch Path Definition for Tree 3 (Center Back) -->
    <path id="branchPath3" fill="none" stroke="url(#trunkGradient)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" opacity="0.7" d="
      M 400, 550 
      C 400, 470  380, 410  380, 350
      M 390, 420 
      C 350, 380  340, 320  330, 280
      M 400, 400 
      C 440, 360  460, 320  470, 280
    "/>

  </defs>

  <!-- ============================================================
       BACKGROUND LAYERS
       ============================================================ -->
  <!-- Sky -->
  <rect width="100%" height="100%" fill="url(#skyGradient)" />

  <!-- Distant Hills/Mountains -->
  <path d="M0,400 C100,350 200,380 300,340 C400,300 500,360 600,320 C700,280 800,340 800,340 L800,600 L0,600 Z" fill="#1a3a4a" opacity="0.6"/>
  <path d="M0,450 C150,400 250,430 400,390 C550,350 650,410 800,380 L800,600 L0,600 Z" fill="#2c5f6b" opacity="0.5"/>

  <!-- Ground -->
  <rect y="450" width="100%" height="150" fill="url(#groundGradient)" />
  <!-- Grass texture lines -->
  <g stroke="#236b3a" stroke-width="1" opacity="0.4">
    <line x1="0" y1="460" x2="800" y2="460" />
    <line x1="0" y1="490" x2="800" y2="490" />
    <line x1="0" y1="520" x2="800" y2="520" />
  </g>

  <!-- ============================================================
       TREES (Trunks & Branches)
       ============================================================ -->
  <!-- Tree 3 (Back) -->
  <use href="#branchPath3" />
  
  <!-- Tree 1 (Left) -->
  <use href="#branchPath1" />
  
  <!-- Tree 2 (Right) -->
  <use href="#branchPath2" />

  <!-- ============================================================
       BLOSSOMS (Placed along branches via transforms)
       ============================================================ -->
  <g id="blossomsLayer">
    <!-- Tree 1 Blossoms (Left Tree) -->
    <use href="#blossomCluster" x="100" y="270" transform="scale(0.6)" style="animation: sway 4s ease-in-out infinite;" />
    <use href="#blossomCluster" x="130" y="310" transform="scale(0.5)" style="animation: sway 4.5s ease-in-out infinite 0.5s;" />
    <use href="#blossomCluster" x="180" y="300" transform="scale(0.7)" style="animation: sway 3.8s ease-in-out infinite 1s;" />
    <use href="#blossomCluster" x="210" y="260" transform="scale(0.55)" style="animation: sway 4.2s ease-in-out infinite 0.2s;" />
    <use href="#blossomCluster" x="270" y="240" transform="scale(0.65)" style="animation: sway 3.5s ease-in-out infinite 1.5s;" />
    <use href="#blossomCluster" x="290" y="180" transform="scale(0.5)" style="animation: sway 4s ease-in-out infinite 0.8s;" />
    <use href="#blossomCluster" x="160" y="360" transform="scale(0.45)" style="animation: sway 5s ease-in-out infinite 2s;" />
    <use href="#blossomCluster" x="230" y="340" transform="scale(0.4)" style="animation: sway 4.8s ease-in-out infinite 2.5s;" />

    <!-- Tree 2 Blossoms (Right Tree) -->
    <use href="#blossomCluster" x="700" y="280" transform="scale(0.6)" style="animation: sway 4.2s ease-in-out infinite 0.3s;" />
    <use href="#blossomCluster" x="680" y="320" transform="scale(0.55)" style="animation: sway 3.9s ease-in-out infinite 1.2s;" />
    <use href="#blossomCluster" x="520" y="240" transform="scale(0.7)" style="animation: sway 4.5s ease-in-out infinite 0.7s;" />
    <use href="#blossomCluster" x="540" y="290" transform="scale(0.5)" style="animation: sway 4s ease-in-out infinite 1.8s;" />
    <use href="#blossomCluster" x="600" y="200" transform="scale(0.65)" style="animation: sway 3.7s ease-in-out infinite 2.2s;" />
    <use href="#blossomCluster" x="510" y="190" transform="scale(0.45)" style="animation: sway 5s ease-in-out infinite 0.1s;" />
    <use href="#blossomCluster" x="670" y="360" transform="scale(0.4)" style="animation: sway 4.8s ease-in-out infinite 1.1s;" />

    <!-- Tree 3 Blossoms (Center Back) -->
    <use href="#blossomCluster" x="330" y="270" transform="scale(0.5)" style="animation: sway 5s ease-in-out infinite 0.4s;" opacity="0.7" />
    <use href="#blossomCluster" x="380" y="340" transform="scale(0.45)" style="animation: sway 4.5s ease-in-out infinite 1.4s;" opacity="0.7" />
    <use href="#blossomCluster" x="460" y="270" transform="scale(0.55)" style="animation: sway 4.8s ease-in-out infinite 2s;" opacity="0.7" />
    <use href="#blossomCluster" x="410" y="330" transform="scale(0.4)" style="animation: sway 5.2s ease-in-out infinite 0.9s;" opacity="0.7" />
  </g>

  <!-- ============================================================
       FALLING PETALS (Animated via SMIL)
       ============================================================ -->
  <g id="fallingPetals">
    <!-- We generate 15 petals with randomized start times/durations via script-like SMIL attributes -->
    <!-- Petal 1 -->
    <use href="#fallingPetal">
      <animateMotion path="M 100,0 Q 120,200 110,600" dur="12s" repeatCount="indefinite" begin="0s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="6s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="12s" repeatCount="indefinite" begin="0s"/>
    </use>
    <!-- Petal 2 -->
    <use href="#fallingPetal">
      <animateMotion path="M 250,0 Q 230,150 260,600" dur="15s" repeatCount="indefinite" begin="2s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="8s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="15s" repeatCount="indefinite" begin="2s"/>
    </use>
    <!-- Petal 3 -->
    <use href="#fallingPetal">
      <animateMotion path="M 400,0 Q 420,300 390,600" dur="10s" repeatCount="indefinite" begin="4s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="10s" repeatCount="indefinite" begin="4s"/>
    </use>
    <!-- Petal 4 -->
    <use href="#fallingPetal">
      <animateMotion path="M 550,0 Q 530,100 560,600" dur="14s" repeatCount="indefinite" begin="1s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="7s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="14s" repeatCount="indefinite" begin="1s"/>
    </use>
    <!-- Petal 5 -->
    <use href="#fallingPetal">
      <animateMotion path="M 700,0 Q 720,250 710,600" dur="11s" repeatCount="indefinite" begin="3s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="5.5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="11s" repeatCount="indefinite" begin="3s"/>
    </use>
    <!-- Petal 6 -->
    <use href="#fallingPetal">
      <animateMotion path="M 180,0 Q 160,400 190,600" dur="13s" repeatCount="indefinite" begin="5s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="6.5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="13s" repeatCount="indefinite" begin="5s"/>
    </use>
    <!-- Petal 7 -->
    <use href="#fallingPetal">
      <animateMotion path="M 620,0 Q 640,50 610,600" dur="16s" repeatCount="indefinite" begin="6s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="8s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="16s" repeatCount="indefinite" begin="6s"/>
    </use>
    <!-- Petal 8 -->
    <use href="#fallingPetal">
      <animateMotion path="M 350,0 Q 330,200 360,600" dur="9s" repeatCount="indefinite" begin="7s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="4.5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="9s" repeatCount="indefinite" begin="7s"/>
    </use>
    <!-- Petal 9 -->
    <use href="#fallingPetal">
      <animateMotion path="M 50,0 Q 70,300 60,600" dur="12s" repeatCount="indefinite" begin="8s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="6s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="12s" repeatCount="indefinite" begin="8s"/>
    </use>
    <!-- Petal 10 -->
    <use href="#fallingPetal">
      <animateMotion path="M 750,0 Q 730,150 760,600" dur="10s" repeatCount="indefinite" begin="1.5s" rotate="auto-reverse"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.9;1" dur="10s" repeatCount="indefinite" begin="1.5s"/>
    </use>
  </g>

  <!-- ============================================================
       CSS STYLES FOR ANIMATION (Swaying Branches)
       ============================================================ -->
  <style type="text/css"><![CDATA[
    @keyframes sway {
      0%   { transform: rotate(-2deg) translateX(-1px); transform-origin: bottom center; }
      50%  { transform: rotate(2deg) translateX(1px); transform-origin: bottom center; }
      100% { transform: rotate(-2deg) translateX(-1px); transform-origin: bottom center; }
    }
    
    /* Apply sway to blossom clusters (handled via inline style animation property above) */
    /* Note: SVG SMIL animateTransform is more reliable for rotation origin than CSS on <use> elements in some browsers.
       The inline 'style="animation: sway..."' on <use> elements works in modern browsers (Chrome/Firefox/Edge). */
  ]]></style>

  <!-- Fallback SMIL Animation for Swaying (More robust for <use> elements) -->
  <!-- We apply a gentle rotation to the whole blossom layer groups if CSS fails, 
       but here we target the specific use elements via ID would be better. 
       For simplicity and broad support, the CSS keyframes above target the <use> elements directly. -->
       
  <!-- Title/Description for Accessibility -->
  <title>Cherry Blossom Grove at Dawn</title>
  <desc>An animated SVG scene featuring three cherry blossom trees in full bloom with falling petals and swaying branches against a gradient dawn sky.</desc>

</svg>
```

</details>

## 13. qwen3.5_397b-cloud

**SVG size:** 4458 characters  
**Complexity:** Low  
**Shape elements:** 25  
**Distinct colors:** 8  
**Raw response:** 4688 characters

![qwen3.5_397b-cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/qwen3.5_397b-cloud.svg)

### Analysis

This SVG contains approximately **25 shape elements** and uses **8 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating petals
- Includes gradient fills for richer visual depth (great for skies and petals)
- Uses transforms (translate, rotate, scale) for positioning petals and branches
- Includes animation elements -- the model attempted motion (e.g., falling petals)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Tree elements are present (trunk, branches) -- the model understood the structure
- Flower / blossom elements are present (petals, blooms) -- the model captured the subject
- Pink tones are used -- the model understood the cherry blossom color cue

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate petal detail. Best for embedding in web pages where small file size matters more than visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Definitions for reusable elements -->
  <defs>
    <!-- A single cherry blossom flower -->
    <g id="flower">
      <circle cx="0" cy="0" r="4" fill="#FFB7C5" />
      <circle cx="3" cy="0" r="3" fill="#FFB7C5" />
      <circle cx="-3" cy="0" r="3" fill="#FFB7C5" />
      <circle cx="1.5" cy="2.5" r="3" fill="#FFB7C5" />
      <circle cx="-1.5" cy="2.5" r="3" fill="#FFB7C5" />
      <circle cx="0" cy="0" r="1.5" fill="#FF69B4" />
    </g>
    
    <!-- A falling petal -->
    <g id="petal">
      <ellipse cx="0" cy="0" rx="3" ry="1.5" fill="#FFC0CB" transform="rotate(45)" />
    </g>

    <!-- Sky Gradient -->
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#E0F7FA;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFFFFF;stop-opacity:1" />
    </linearGradient>
  </defs>

  <!-- Background Sky -->
  <rect width="600" height="400" fill="url(#skyGradient)" />

  <!-- Ground -->
  <ellipse cx="300" cy="380" rx="350" ry="40" fill="#E8F5E9" />

  <!-- Tree 1 (Left) -->
  <g transform="translate(100, 350)">
    <!-- Trunk -->
    <path d="M0,0 C-10,-40 -20,-100 10,-150 C30,-180 50,-150 40,-100 C60,-60 80,-40 90,-10" 
          stroke="#8D6E63" stroke-width="12" fill="none" stroke-linecap="round" />
    <path d="M0,0 C-5,-30 -10,-60 0,-90" 
          stroke="#8D6E63" stroke-width="6" fill="none" stroke-linecap="round" />
    
    <!-- Foliage Clouds (Back) -->
    <circle cx="-20" cy="-160" r="60" fill="#FFC0CB" opacity="0.8" />
    <circle cx="40" cy="-140" r="70" fill="#FFC0CB" opacity="0.8" />
    <circle cx="0" cy="-100" r="50" fill="#FFC0CB" opacity="0.8" />
    
    <!-- Foliage Clouds (Front/Darker) -->
    <circle cx="-30" cy="-150" r="50" fill="#FFB7C5" />
    <circle cx="50" cy="-130" r="60" fill="#FFB7C5" />
    <circle cx="10" cy="-90" r="45" fill="#FFB7C5" />
  </g>

  <!-- Tree 2 (Right) -->
  <g transform="translate(450, 360)">
    <!-- Trunk -->
    <path d="M0,0 C10,-50 20,-120 -10,-180 C-30,-210 -50,-180 -40,-130 C-60,-90 -80,-60 -90,-30" 
          stroke="#795548" stroke-width="14" fill="none" stroke-linecap="round" />
    <path d="M0,0 C5,-30 10,-60 0,-90" 
          stroke="#795548" stroke-width="8" fill="none" stroke-linecap="round" />

    <!-- Foliage Clouds (Back) -->
    <circle cx="-40" cy="-190" r="75" fill="#FFC0CB" opacity="0.8" />
    <circle cx="20" cy="-160" r="65" fill="#FFC0CB" opacity="0.8" />
    <circle cx="-10" cy="-120" r="55" fill="#FFC0CB" opacity="0.8" />

    <!-- Foliage Clouds (Front) -->
    <circle cx="-50" cy="-180" r="65" fill="#FFB7C5" />
    <circle cx="30" cy="-150" r="55" fill="#FFB7C5" />
    <circle cx="-20" cy="-110" r="50" fill="#FFB7C5" />
  </g>

  <!-- Scattered Individual Flowers (Detail) -->
  <g fill="#FF69B4">
    <!-- Left Tree Flowers -->
    <use href="#flower" x="60" y="180" />
    <use href="#flower" x="140" y="200" />
    <use href="#flower" x="90" y="240" />
    <use href="#flower" x="40" y="220" />
    <use href="#flower" x="110" y="160" />
    
    <!-- Right Tree Flowers -->
    <use href="#flower" x="380" y="160" />
    <use href="#flower" x="460" y="190" />
    <use href="#flower" x="420" y="230" />
    <use href="#flower" x="350" y="210" />
    <use href="#flower" x="480" y="150" />
  </g>

  <!-- Falling Petals Animation Group -->
  <g opacity="0.7">
    <use href="#petal" x="150" y="250">
      <animate attributeName="y" from="250" to="400" dur="4s" repeatCount="indefinite" begin="0s"/>
      <animate attributeName="x" from="150" to="130" dur="4s" repeatCount="indefinite" begin="0s"/>
    </use>
    <use href="#petal" x="400" y="220">
      <animate attributeName="y" from="220" to="400" dur="5s" repeatCount="indefinite" begin="1s"/>
      <animate attributeName="x" from="400" to="420" dur="5s" repeatCount="indefinite" begin="1s"/>
    </use>
    <use href="#petal" x="250" y="280">
      <animate attributeName="y" from="280" to="400" dur="6s" repeatCount="indefinite" begin="2s"/>
      <animate attributeName="x" from="250" to="260" dur="6s" repeatCount="indefinite" begin="2s"/>
    </use>
    <use href="#petal" x="500" y="200">
      <animate attributeName="y" from="200" to="400" dur="3.5s" repeatCount="indefinite" begin="0.5s"/>
      <animate attributeName="x" from="500" to="490" dur="3.5s" repeatCount="indefinite" begin="0.5s"/>
    </use>
  </g>
</svg>
```

</details>

## 14. deepseek-v4-flash_0731-cloud

**SVG size:** 15165 characters  
**Complexity:** Very high  
**Shape elements:** 152  
**Distinct colors:** 9  
**Raw response:** 15165 characters

![deepseek-v4-flash_0731-cloud cherry blossom SVG](/assets/img/diagrams/ollama-svg-blossom/deepseek-v4-flash_0731-cloud.svg)

### Analysis

This SVG contains approximately **152 shape elements** and uses **9 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating motifs
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a large, detailed SVG (over 10,000 characters) packed with shapes, gradients, and structure. Best when you want maximum visual richness over file size.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
    <defs>
        <!-- Sky Gradient -->
        <linearGradient id="skyGradient" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#1A1B41"/>
            <stop offset="30%" stop-color="#4B2A5E"/>
            <stop offset="60%" stop-color="#A5557B"/>
            <stop offset="100%" stop-color="#F2A68D"/>
        </linearGradient>
        <!-- Glowing Sun/Moon Gradient -->
        <radialGradient id="moonGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#FFFFFF"/>
            <stop offset="30%" stop-color="rgba(255, 240, 220, 0.9)"/>
            <stop offset="60%" stop-color="rgba(255, 200, 200, 0.4)"/>
            <stop offset="100%" stop-color="rgba(255, 200, 200, 0)"/>
        </radialGradient>
        <!-- Falling Petal Symbol -->
        <g id="petal">
            <path d="M 0,0 C 5,-10 15,-10 20,-2 C 18,0 18,2 20,6 C 15,10 5,10 0,0 Z" fill="#FCC1CD"/>
        </g>
        <!-- Deep Pink Petal for Ground Clusters -->
        <g id="petal-deep">
            <path d="M 0,0 C 5,-10 15,-10 20,-2 C 18,0 18,2 20,6 C 15,10 5,10 0,0 Z" fill="#F4A0B8"/>
        </g>
    </defs>
    <!-- Background Sky -->
    <rect width="800" height="600" fill="url(#skyGradient)"/>
    <!-- Glowing Sun -->
    <circle cx="400" cy="320" r="180" fill="url(#moonGlow)"/>
    <circle cx="400" cy="320" r="60" fill="#FFFFFF" opacity="0.9"/>
    <!-- Subtle Distant Clouds -->
    <path d="M 100,200 Q 150,180 200,200 Q 250,190 300,210 Q 250,220 200,220 Q 150,230 100,200 Z" fill="#FFFFFF" opacity="0.1"/>
    <path d="M 500,150 Q 550,130 600,150 Q 650,140 700,160 Q 650,170 600,170 Q 550,180 500,150 Z" fill="#FFFFFF" opacity="0.15"/>
    <path d="M 600,250 Q 630,240 660,250 Q 690,240 720,260 Q 690,270 660,270 Q 630,280 600,250 Z" fill="#FFFFFF" opacity="0.1"/>
    <!-- Distant Birds -->
    <g fill="#1A1015" opacity="0.6">
        <path d="M 500,150 Q 508,140 516,150 Q 524,140 532,150 Z"/>
        <path d="M 540,135 Q 546,127 552,135 Q 558,127 564,135 Z"/>
        <path d="M 480,165 Q 486,157 492,165 Q 498,157 504,165 Z"/>
        <path d="M 650,120 Q 656,114 662,120 Q 668,114 674,120 Z"/>
    </g>
    <!-- Rolling Hills (Background to Foreground) -->
    <path d="M 0,400 Q 200,350 500,450 T 800,400 L 800,600 L 0,600 Z" fill="#3E5C50"/>
    <path d="M -50,480 Q 250,400 600,500 T 850,480 L 850,600 L -50,600 Z" fill="#2A4238"/>
    <path d="M -50,600 Q 150,500 400,550 T 850,550 L 850,600 L -50,600 Z" fill="#16261F"/>
    <!-- Tree Ground Shadows -->
    <ellipse cx="180" cy="550" rx="130" ry="25" fill="#0F201A" opacity="0.7"/>
    <ellipse cx="580" cy="550" rx="90" ry="20" fill="#0F201A" opacity="0.7"/>
    <!-- ================= LEFT CHERRY BLOSSOM TREE ================= -->
    <!-- Trunk and Branches -->
    <g fill="none" stroke="#2B1B20" stroke-linecap="round" stroke-linejoin="round">
        <!-- Main Trunk -->
        <path d="M 150,550 C 130,450 160,350 180,250" stroke-width="30"/>
        <!-- Major Branches -->
        <path d="M 160,400 C 100,350 60,300 40,200" stroke-width="18"/>
        <path d="M 170,350 C 220,300 260,250 300,200" stroke-width="16"/>
        <path d="M 175,300 C 150,220 120,150 100,100" stroke-width="12"/>
        <path d="M 180,280 C 220,200 240,150 250,100" stroke-width="12"/>
        <!-- Minor Branches -->
        <path d="M 120,240 C 100,200 80,180 60,160" stroke-width="8"/>
        <path d="M 200,280 C 220,240 240,220 270,210" stroke-width="8"/>
        <path d="M 140,180 C 130,140 110,120 100,80" stroke-width="6"/>
        <path d="M 220,180 C 230,140 250,120 270,100" stroke-width="6"/>
        <path d="M 80,220 C 60,200 50,180 30,150" stroke-width="5"/>
        <path d="M 240,230 C 260,210 280,200 310,190" stroke-width="5"/>
        <path d="M 190,250 C 200,220 220,200 240,180" stroke-width="4"/>
    </g>
    <!-- Trunk Textures &amp; Bark Details -->
    <g fill="none" stroke-linecap="round">
        <path d="M 155,530 C 140,450 165,350 183,260" stroke="#4A303A" stroke-width="8"/>
        <path d="M 165,480 C 150,420 170,360 185,290" stroke="#4A303A" stroke-width="6"/>
        <path d="M 175,540 C 160,480 175,420 188,330" stroke="#1A1015" stroke-width="10"/>
        <path d="M 145,520 C 130,460 150,380 170,280" stroke="#1A1015" stroke-width="4"/>
        <path d="M 110,300 C 90,260 70,220 50,180" stroke="#4A303A" stroke-width="3"/>
        <path d="M 250,200 C 260,180 280,160 300,150" stroke="#4A303A" stroke-width="3"/>
    </g>
    <!-- Left Tree Canopy (Base Layer - Deep Pink) -->
    <g fill="#E87A9E" opacity="0.85">
        <circle cx="100" cy="220" r="70"/>
        <circle cx="180" cy="240" r="80"/>
        <circle cx="240" cy="200" r="60"/>
        <circle cx="140" cy="160" r="75"/>
        <circle cx="220" cy="150" r="65"/>
        <circle cx="80" cy="180" r="50"/>
        <circle cx="180" cy="100" r="70"/>
        <circle cx="260" cy="130" r="45"/>
        <circle cx="40" cy="150" r="40"/>
        <circle cx="280" cy="100" r="35"/>
        <circle cx="300" cy="160" r="30"/>
        <circle cx="80" cy="100" r="45"/>
        <circle cx="260" cy="240" r="40"/>
        <circle cx="120" cy="270" r="50"/>
        <circle cx="200" cy="280" r="45"/>
    </g>
    <!-- Left Tree Canopy (Mid Layer - Mid Pink) -->
    <g fill="#F4A0B8" opacity="0.9">
        <circle cx="120" cy="200" r="50"/>
        <circle cx="200" cy="220" r="60"/>
        <circle cx="160" cy="180" r="55"/>
        <circle cx="240" cy="180" r="45"/>
        <circle cx="100" cy="160" r="40"/>
        <circle cx="200" cy="140" r="50"/>
        <circle cx="140" cy="120" r="45"/>
        <circle cx="260" cy="160" r="30"/>
        <circle cx="60" cy="170" r="30"/>
        <circle cx="220" cy="120" r="35"/>
        <circle cx="160" cy="240" r="40"/>
        <circle cx="280" cy="140" r="25"/>
        <circle cx="120" cy="140" r="35"/>
        <circle cx="180" cy="220" r="45"/>
    </g>
    <!-- Left Tree Canopy (Top Layer - Light Pink &amp; White Highlights) -->
    <g fill="#FCC1CD" opacity="0.95">
        <circle cx="140" cy="190" r="40"/>
        <circle cx="180" cy="200" r="45"/>
        <circle cx="200" cy="170" r="35"/>
        <circle cx="160" cy="150" r="30"/>
        <circle cx="220" cy="140" r="25"/>
        <circle cx="120" cy="170" r="30"/>
        <circle cx="180" cy="120" r="35"/>
        <circle cx="240" cy="160" r="20"/>
        <circle cx="140" cy="140" r="25"/>
        <circle cx="160" cy="210" r="30"/>
        <circle cx="100" cy="190" r="25"/>
        <circle cx="200" cy="190" r="30"/>
        <circle cx="80" cy="150" r="20"/>
        <circle cx="260" cy="130" r="15"/>
    </g>
    <!-- Left Tree Canopy (Pure White Core Highlights) -->
    <g fill="#FFFFFF" opacity="0.9">
        <circle cx="180" cy="180" r="20"/>
        <circle cx="160" cy="160" r="15"/>
        <circle cx="200" cy="190" r="18"/>
        <circle cx="140" cy="200" r="12"/>
        <circle cx="220" cy="160" r="10"/>
        <circle cx="120" cy="150" r="12"/>
    </g>
    <!-- ================= RIGHT CHERRY BLOSSOM TREE ================= -->
    <!-- Trunk and Branches -->
    <g fill="none" stroke="#2B1B20" stroke-linecap="round" stroke-linejoin="round">
        <!-- Main Trunk -->
        <path d="M 600,550 C 590,480 580,420 580,380" stroke-width="20"/>
        <!-- Major Branches -->
        <path d="M 585,450 C 550,410 510,390 480,360" stroke-width="12"/>
        <path d="M 580,420 C 620,380 650,350 680,320" stroke-width="12"/>
        <path d="M 580,390 C 570,340 560,300 550,260" stroke-width="10"/>
        <!-- Minor Branches -->
        <path d="M 530,380 C 510,360 490,340 470,320" stroke-width="6"/>
        <path d="M 620,360 C 640,340 650,320 660,290" stroke-width="6"/>
        <path d="M 560,300 C 540,280 520,260 510,240" stroke-width="5"/>
        <path d="M 590,320 C 610,300 620,280 630,260" stroke-width="5"/>
        <path d="M 500,350 C 480,340 460,330 440,320" stroke-width="4"/>
        <path d="M 650,330 C 670,310 690,290 710,280" stroke-width="4"/>
        <path d="M 570,350 C 550,330 540,310 520,290" stroke-width="3"/>
    </g>
    <!-- Trunk Textures &amp; Bark Details -->
    <g fill="none" stroke-linecap="round">
        <path d="M 605,530 C 595,480 585,430 585,390" stroke="#4A303A" stroke-width="6"/>
        <path d="M 595,500 C 585,450 575,410 575,370" stroke="#1A1015" stroke-width="4"/>
        <path d="M 590,470 C 580,420 570,380 565,340" stroke="#4A303A" stroke-width="3"/>
        <path d="M 540,400 C 520,380 500,360 480,350" stroke="#4A303A" stroke-width="3"/>
        <path d="M 630,380 C 650,360 660,340 670,320" stroke="#4A303A" stroke-width="3"/>
    </g>
    <!-- Right Tree Canopy (Base Layer - Deep Pink) -->
    <g fill="#E87A9E" opacity="0.85">
        <circle cx="520" cy="360" r="50"/>
        <circle cx="580" cy="380" r="60"/>
        <circle cx="640" cy="340" r="45"/>
        <circle cx="540" cy="300" r="55"/>
        <circle cx="620" cy="280" r="50"/>
        <circle cx="580" cy="260" r="60"/>
        <circle cx="460" cy="280" r="30"/>
        <circle cx="500" cy="240" r="35"/>
        <circle cx="680" cy="260" r="30"/>
        <circle cx="700" cy="300" r="25"/>
        <circle cx="650" cy="220" r="20"/>
        <circle cx="480" cy="320" r="40"/>
    </g>
    <!-- Right Tree Canopy (Mid Layer - Mid Pink) -->
    <g fill="#F4A0B8" opacity="0.9">
        <circle cx="550" cy="340" r="40"/>
        <circle cx="600" cy="360" r="45"/>
        <circle cx="500" cy="320" r="35"/>
        <circle cx="650" cy="310" r="35"/>
        <circle cx="560" cy="280" r="40"/>
        <circle cx="620" cy="300" r="40"/>
        <circle cx="590" cy="250" r="45"/>
        <circle cx="520" cy="280" r="30"/>
        <circle cx="660" cy="280" r="25"/>
        <circle cx="540" cy="240" r="30"/>
        <circle cx="480" cy="290" r="25"/>
        <circle cx="680" cy="300" r="20"/>
    </g>
    <!-- Right Tree Canopy (Top Layer - Light Pink &amp; White Highlights) -->
    <g fill="#FCC1CD" opacity="0.95">
        <circle cx="560" cy="320" r="30"/>
        <circle cx="590" cy="340" r="35"/>
        <circle cx="540" cy="280" r="25"/>
        <circle cx="610" cy="280" r="30"/>
        <circle cx="580" cy="300" r="35"/>
        <circle cx="620" cy="320" r="25"/>
        <circle cx="570" cy="260" r="20"/>
        <circle cx="600" cy="250" r="25"/>
        <circle cx="530" cy="310" r="20"/>
        <circle cx="640" cy="290" r="20"/>
        <circle cx="510" cy="270" r="15"/>
        <circle cx="660" cy="310" r="15"/>
    </g>
    <!-- Right Tree Canopy (Pure White Core Highlights) -->
    <g fill="#FFFFFF" opacity="0.9">
        <circle cx="580" cy="300" r="15"/>
        <circle cx="560" cy="310" r="12"/>
        <circle cx="600" cy="280" r="15"/>
        <circle cx="590" cy="320" r="10"/>
        <circle cx="550" cy="290" r="10"/>
        <circle cx="620" cy="300" r="8"/>
    </g>
    <!-- ================= FALLING &amp; GROUND PETALS ================= -->
    <g id="falling-petals">
        <!-- Floating in the air -->
        <use href="#petal" transform="translate(320, 400) rotate(45)"/>
        <use href="#petal" transform="translate(350, 450) rotate(-30)"/>
        <use href="#petal" transform="translate(280, 350) rotate(15)"/>
        <use href="#petal" transform="translate(420, 380) rotate(-60)"/>
        <use href="#petal" transform="translate(450, 300) rotate(80)"/>
        <use href="#petal" transform="translate(380, 250) rotate(-15)"/>
        <use href="#petal" transform="translate(500, 450) rotate(45)"/>
        <use href="#petal" transform="translate(480, 520) rotate(-80)"/>
        <use href="#petal" transform="translate(550, 400) rotate(20)"/>
        <use href="#petal" transform="translate(680, 450) rotate(-45)"/>
        <use href="#petal" transform="translate(720, 380) rotate(70)"/>
        <use href="#petal" transform="translate(750, 300) rotate(-10)"/>
        <use href="#petal" transform="translate(650, 350) rotate(35)"/>
        <use href="#petal" transform="translate(600, 280) rotate(-55)"/>
        <use href="#petal" transform="translate(700, 220) rotate(85)"/>
        <use href="#petal" transform="translate(520, 200) rotate(-25)"/>
        <use href="#petal" transform="translate(400, 150) rotate(60)"/>
        <use href="#petal" transform="translate(300, 180) rotate(-40)"/>
        <use href="#petal" transform="translate(200, 250) rotate(15)"/>
        <use href="#petal" transform="translate(150, 300) rotate(-70)"/>
        <use href="#petal" transform="translate(100, 350) rotate(30)"/>
        <use href="#petal" transform="translate(80, 400) rotate(-20)"/>
        <use href="#petal" transform="translate(180, 500) rotate(45)"/>
        <use href="#petal" transform="translate(350, 580) rotate(75)"/>
        <use href="#petal" transform="translate(550, 560) rotate(-50)"/>
        <use href="#petal" transform="translate(650, 580) rotate(25)"/>
        <use href="#petal" transform="translate(750, 550) rotate(-65)"/>
        <use href="#petal" transform="translate(700, 500) rotate(10)"/>
        <use href="#petal" transform="translate(600, 520) rotate(-35)"/>
        <use href="#petal" transform="translate(680, 550) rotate(40)"/>
        <!-- Scattered on the Ground (Left Tree) -->
        <use href="#petal-deep" transform="translate(100, 560) rotate(15)"/>
        <use href="#petal-deep" transform="translate(130, 580) rotate(-45)"/>
        <use href="#petal-deep" transform="translate(150, 565) rotate(70)"/>
        <use href="#petal-deep" transform="translate(220, 570) rotate(-30)"/>
        <use href="#petal-deep" transform="translate(250, 555) rotate(50)"/>
        <use href="#petal-deep" transform="translate(180, 585) rotate(-10)"/>
        <use href="#petal-deep" transform="translate(280, 560) rotate(85)"/>
        <use href="#petal-deep" transform="translate(80, 575) rotate(-60)"/>
        <!-- Scattered on the Ground (Right Tree) -->
        <use href="#petal-deep" transform="translate(520, 560) rotate(-15)"/>
        <use href="#petal-deep" transform="translate(560, 575) rotate(45)"/>
        <use href="#petal-deep" transform="translate(610, 565) rotate(-70)"/>
        <use href="#petal-deep" transform="translate(580, 585) rotate(20)"/>
        <use href="#petal-deep" transform="translate(640, 555) rotate(-40)"/>
        <use href="#petal-deep" transform="translate(660, 570) rotate(10)"/>
        <use href="#petal-deep" transform="translate(500, 570) rotate(75)"/>
        <use href="#petal-deep" transform="translate(600, 550) rotate(-25)"/>
    </g>
    <!-- Magical Atmosphere Sparkles -->
    <g fill="#FFFFFF" opacity="0.7">
        <circle cx="100" cy="100" r="2"/>
        <circle cx="300" cy="180" r="1.5"/>
        <circle cx="200" cy="80" r="2.5"/>
        <circle cx="480" cy="250" r="2"/>
        <circle cx="700" cy="300" r="1.5"/>
        <circle cx="400" cy="200" r="1"/>
        <circle cx="250" cy="280" r="1.5"/>
        <circle cx="550" cy="220" r="2"/>
        <circle cx="150" cy="130" r="1"/>
        <circle cx="650" cy="200" r="1.5"/>
    </g>
</svg>
```

</details>


## Retired Models

The following models were listed in the Ollama registry but returned HTTP 410 Gone when queried. They were retired from Ollama Cloud and are no longer available for inference:

- `bjoernb/claude-opus-4-5:latest` -- glm-4.7 was retired at 2026-07-15
- `deepseek-v3.1:671b-cloud` -- deepseek-v3.1:671b was retired at 2026-07-15
- `glm-5:cloud` -- glm-5 was retired at 2026-07-15
- `qwen3-vl:235b-cloud` -- qwen3-vl:235b was retired at 2026-06-16

## How to Pick the Best Model: A Decision Guide

Now that you have seen all 14 cherry blossom SVGs, here is a structured way to decide which Ollama cloud model is best for your SVG generation needs:

### Step 1: Define your priority

- **Visual quality first**: Scroll back through the SVGs above and pick the one that looks best to your eye. For nature scenes, trust your eye -- the shape and color counts are useful, but a model with 30 shapes and a great palette can beat one with 100 shapes and muddy colors.
- **Code quality first**: Open the raw SVG source for each model (use the disclosure toggles) and look for `<defs>`, `<use>`, gradients, and clean indentation. Models that produce structured code are easier to recolor (e.g., for autumn vs spring) and reuse.
- **Speed first**: If you are building a real-time app, prioritize the models that responded in under 20 seconds (gpt-oss:120b-cloud, nemotron-3-super:cloud, gemma4:31b-cloud).
- **File size first**: For web embedding, smaller is better. Look at the SVG size column in the summary table.

### Step 2: Cross-check across prompts

A model that does well on cherry blossoms might fail on a character prompt. Check our other benchmarks:

- [Duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/) -- character + vehicle
- [Duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/) -- dynamic action scene
- [Duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/) -- vehicle with multiple parts
- This post (cherry blossoms) -- nature / scenery with no central character

A model that consistently produces good results across all four prompts is a safer pick than one that only shines on a single type of scene.

### Step 3: Test with your own prompt

Every model has strengths and weaknesses. The only way to know for sure which model is best for your specific use case is to test it with your own prompt. The Ollama Cloud API is OpenAI-compatible, so you can use any standard client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="glm-5.1:cloud",  # change this to test different models
    messages=[{"role": "user", "content": "Make an svg image of <your prompt>"}],
)
print(resp.choices[0].message.content)
```

## Why This Benchmark Matters: Nature vs Character Prompts

Most LLM SVG benchmarks only test one type of prompt. We deliberately run four different scene types because models specialize:

- **Character prompts** (the duck series) reward models that understand anatomy, proportions, and accessories. Models like `deepseek-v4-pro` and `kimi-k2.6` tend to do well here.
- **Nature prompts** (this cherry blossom post) reward models that understand organic shapes, palettes, and repetition. Models like `glm-5.1` and `nemotron-3-ultra` tend to do well here, because they invest in many small petals and gradients.

If you are picking a model for a specific project, look at the benchmark closest to your use case. If you want a general-purpose model, pick one that does *okay* across all four rather than one that aces one and fails the rest.

## Conclusion: You Decide the Winner

This comparison shows that 14 out of 18 active Ollama cloud models can generate valid SVG artwork from a nature prompt about cherry blossom trees. The results vary dramatically in complexity, style, and technique -- and there is no single "best" model.

Our takeaways after running four SVG benchmarks (bicycle, parachute, jeep, blossom):

- **glm-5.1:cloud** and **nemotron-3-ultra:cloud** consistently produce the longest, most detailed SVGs across all prompt types. Best when you want maximum visual richness.
- **deepseek-v4-pro:cloud** and **deepseek-v4-flash:cloud** consistently produce well-structured, technically advanced SVGs with `<defs>`, `<use>`, and transforms. A strong default choice for code quality and editability.
- **gpt-oss:120b-cloud** and **gemma4:31b-cloud** are consistently among the fastest (under 20 seconds) and produce compact SVGs. Best for speed-sensitive applications.
- **kimi-k2.6:cloud** output size varies wildly by prompt -- it produced a 9,993-char jeep SVG but only a 1,670-char blossom SVG. Worth testing on your specific prompt.
- **glm-5.2:cloud** and **minimax-m3:cloud** offer a reliable balance of detail, speed, and code quality across all four prompts.

But the real verdict is yours. Scroll back through the SVGs, compare them visually, check the raw code, and pick the model that best fits your needs. Every model in this comparison is available right now on Ollama Cloud -- so you can reproduce these results in minutes.

## Links

- [Previous: Duck Driving a Bicycle Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/)
- [Previous: Duck Jumping From a Plane Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/)
- [Previous: Duck Driving a Jeep Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/)
- [Ollama Official Website](https://ollama.com)
- [Ollama Cloud Documentation](https://ollama.com/cloud)
- [SVG Specification (MDN)](https://developer.mozilla.org/en-US/docs/Web/SVG)
- [OpenAI API Reference (used by Ollama)](https://platform.openai.com/docs/api-reference/chat)
