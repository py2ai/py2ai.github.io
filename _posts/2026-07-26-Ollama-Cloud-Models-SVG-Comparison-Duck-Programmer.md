---
layout: post
title: "Which Ollama Cloud Model is Best? Duck Programmer Debugging at 3AM SVG Comparison (13 Models)"
description: "Compare 14 Ollama cloud models on a developer-culture prompt: a duck programmer debugging code at 3am with coffee. Find the best LLM for SVG art. You decide the winner."
date: 2026-07-26
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/
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
  - Programmer
  - Developer Meme
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama cloud model comparison, deepseek vs glm vs qwen, LLM SVG benchmark, AI image generation comparison, programmer meme SVG, duck programmer, debugging at 3am, developer humor, which Ollama model is best, Ollama cloud models 2026, AI coding meme, programmer duck, AI developer art"
---

# Which Ollama Cloud Model is Best? Duck Programmer Debugging at 3AM SVG Comparison (13 Models)

After testing LLMs on ducks, vehicles, and nature scenes, we wanted to know: **can today's top models draw developer culture?** This time we asked 14 Ollama cloud models to draw **a duck programmer debugging code at 3am with coffee** -- a relatable dev meme that tests scene composition, screens with code, lighting, and character expression.

The prompt was: `Make an svg image about a duck programmer debugging code at 3am with coffee`

This is the fifth in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), and [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/).

**Why a programmer duck?** Unlike previous prompts, this scene combines multiple hard problems for SVG generation: (1) a character with expression (tired eyes, sweat drop), (2) a screen with rendered code/text, (3) indoor lighting with glow effects, (4) scene props (laptop, coffee mug, papers, pizza box), and (5) a relatable narrative moment. A model that draws a great duck jeep may struggle here, and vice versa -- which is exactly why we run multiple prompts.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Developer/Scene SVGs

Indoor developer-scene prompts reward different things than outdoor nature prompts. Here are the criteria to use:

- **Scene composition**: Does the SVG have a desk, screen, character, and props arranged in a believable layout? Or is it a jumble of shapes?
- **Code on screen**: Look for actual rendered text (`<text>` elements with monospace font), not just colored rectangles pretending to be code.
- **Lighting and atmosphere**: 3am debugging implies a dark room with screen glow. Does the model use gradients, glows, or filters to convey this?
- **Character expression**: Is the duck tired, stressed, or focused? Expression requires eyes, eyebrows, sweat drops, or posture. Models that only draw a generic duck miss the prompt's emotional cue.
- **Props**: Coffee, laptop, papers, and a clock showing 3:00 are the props the prompt implies. More props = better prompt adherence.
- **SVG code quality**: Does it use `<defs>`, `<use>`, gradients, and clean structure? Better code is easier to tweak (e.g., to recolor or animate).

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with zero post-processing (other than adding `width="100%" height="auto"` for responsive embedding).

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary Table: Compare All 14 Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `deepseek-v4-flash_cloud` | 14873 | 126 | 32 | Very high | Richest scene |
| 2 | `deepseek-v4-pro_cloud` | 9197 | 72 | 14 | Very high | Richest scene |
| 3 | `gemma4_31b-cloud` | 3056 | 31 | 20 | High | Detailed |
| 4 | `gemma4_cloud` | 3162 | 28 | 16 | Medium | Balanced |
| 5 | `glm-5.1_cloud` | 30938 | 194 | 79 | Very high | Richest scene |
| 6 | `glm-5.2_cloud` | 10821 | 89 | 27 | Very high | Richest scene |
| 7 | `gpt-oss_120b-cloud` | 3090 | 21 | 15 | Medium | Balanced |
| 8 | `kimi-k2.6_cloud` | 6269 | 31 | 17 | High | Detailed |
| 9 | `minimax-m2.7_cloud` | 5324 | 43 | 25 | High | Detailed |
| 10 | `minimax-m3_cloud` | 11304 | 82 | 41 | Very high | Richest scene |
| 11 | `nemotron-3-super_cloud` | 3664 | 32 | 11 | High | Detailed |
| 12 | `nemotron-3-ultra_cloud` | 19457 | 93 | 39 | Very high | Richest scene |
| 13 | `qwen3.5_397b-cloud` | 5746 | 54 | 17 | High | Detailed |
| 14 | `deepseek-v4-flash_0731-cloud` | 12388 | 116 | 32 | Very high | Richest scene |
| 15 | `bjoernb_claude-opus-4-5_latest` | - | - | - | - | Retired (410) |
| 16 | `deepseek-v3.1_671b-cloud` | - | - | - | - | Retired (410) |
| 17 | `glm-5_cloud` | - | - | - | - | Retired (410) |
| 18 | `qwen3-vl_235b-cloud` | - | - | - | - | Retired (410) |

**14 out of 18** active models produced a valid SVG. The 4 retired models returned HTTP 410 Gone (removed from Ollama Cloud on 2026-07-15).

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most detailed programmer scene SVG**: pick models labeled "Very high" complexity in the table above
- **You want the fastest response**: look at the per-model sections below for the elapsed time
- **You want the cleanest, most reusable SVG code**: pick models that use `<defs>`, `<use>`, and transforms (see raw source below each SVG)
- **You want a small, efficient SVG for web embedding**: pick models with "Compact" verdict
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"
- **You want to compare within a model family**: pick `deepseek-v4-pro` vs `deepseek-v4-flash`, or `glm-5.1` vs `glm-5.2`, or `minimax-m2.7` vs `minimax-m3`

Now read on for the full per-model breakdown and judge for yourself.

## 1. deepseek-v4-flash_cloud

**SVG size:** 14873 characters  
**Complexity:** Very high  
**Shape elements:** 126  
**Distinct colors:** 32  
**Raw response:** 15429 characters  
**Generation time:** 39.6s

![deepseek-v4-flash_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/deepseek-v4-flash_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With over 14,873 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering scene props, screen code, lighting, and character expression.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <defs>
    <!-- Screen Glow Gradient -->
    <radialGradient id="screenGlow" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
      <stop offset="0%" stop-color="#00e5ff" stop-opacity="0.4" />
      <stop offset="100%" stop-color="#00e5ff" stop-opacity="0" />
    </radialGradient>
    <!-- Night Sky Gradient -->
    <linearGradient id="nightSky" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>
    <!-- Moon Glow -->
    <radialGradient id="moonGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fef08a" stop-opacity="0.8" />
      <stop offset="100%" stop-color="#fef08a" stop-opacity="0" />
    </radialGradient>
  </defs>
  <!-- Background / Night Sky -->
  <rect width="800" height="600" fill="url(#nightSky)" />
  <!-- Window -->
  <rect x="50" y="50" width="220" height="280" rx="10" fill="#0f172a" stroke="#334155" stroke-width="8" />
  <!-- Window Panes -->
  <line x1="160" y1="50" x2="160" y2="330" stroke="#334155" stroke-width="8" />
  <line x1="50" y1="190" x2="270" y2="190" stroke="#334155" stroke-width="8" />
  <!-- Window Crossbars -->
  <rect x="50" y="50" width="220" height="280" rx="10" fill="none" stroke="#1e293b" stroke-width="4" />
  <!-- Moon & Glow -->
  <circle cx="130" cy="120" r="60" fill="url(#moonGlow)" />
  <circle cx="130" cy="120" r="35" fill="#fef08a" />
  <!-- Moon Craters -->
  <circle cx="120" cy="110" r="6" fill="#eab308" opacity="0.4" />
  <circle cx="145" cy="125" r="8" fill="#eab308" opacity="0.4" />
  <circle cx="125" cy="135" r="4" fill="#eab308" opacity="0.4" />
  <!-- Stars -->
  <circle cx="300" cy="80" r="2" fill="#fff" opacity="0.8" />
  <circle cx="350" cy="150" r="1.5" fill="#fff" opacity="0.6" />
  <circle cx="400" cy="100" r="2.5" fill="#fff" opacity="0.9" />
  <circle cx="450" cy="180" r="1" fill="#fff" opacity="0.5" />
  <circle cx="500" cy="70" r="2" fill="#fff" opacity="0.7" />
  <circle cx="700" cy="90" r="2" fill="#fff" opacity="0.8" />
  <circle cx="750" cy="160" r="1.5" fill="#fff" opacity="0.6" />
  <circle cx="680" cy="200" r="1" fill="#fff" opacity="0.5" />
  <circle cx="80" cy="250" r="1.5" fill="#fff" opacity="0.7" />
  <circle cx="220" cy="280" r="2" fill="#fff" opacity="0.6" />
  <!-- Wall Clock (3 AM) -->
  <g transform="translate(650, 150)">
    <circle cx="0" cy="0" r="50" fill="#e2e8f0" stroke="#475569" stroke-width="6" />
    <circle cx="0" cy="0" r="4" fill="#1e293b" />
    <!-- Hour markers -->
    <line x1="0" y1="-40" x2="0" y2="-35" stroke="#475569" stroke-width="4" />
    <line x1="40" y1="0" x2="35" y2="0" stroke="#475569" stroke-width="4" />
    <line x1="0" y1="40" x2="0" y2="35" stroke="#475569" stroke-width="4" />
    <line x1="-40" y1="0" x2="-35" y2="0" stroke="#475569" stroke-width="4" />
    <!-- Hour Hand (pointing to 3) -->
    <line x1="0" y1="0" x2="30" y2="0" stroke="#1e293b" stroke-width="6" stroke-linecap="round" />
    <!-- Minute Hand (pointing to 12) -->
    <line x1="0" y1="0" x2="0" y2="-35" stroke="#1e293b" stroke-width="4" stroke-linecap="round" />
    <!-- Digital 3am text -->
    <text x="0" y="20" font-family="monospace" font-size="10" fill="#ef4444" text-anchor="middle" font-weight="bold">3:00</text>
  </g>
  <!-- Desk -->
  <rect x="0" y="420" width="800" height="180" fill="#3e2723" />
  <rect x="0" y="420" width="800" height="15" fill="#5d4037" />
  <!-- Desk Wood Grain (subtle) -->
  <line x1="100" y1="450" x2="300" y2="450" stroke="#4e342e" stroke-width="2" opacity="0.5" />
  <line x1="400" y1="480" x2="700" y2="480" stroke="#4e342e" stroke-width="2" opacity="0.5" />
  <line x1="50" y1="520" x2="400" y2="520" stroke="#4e342e" stroke-width="2" opacity="0.5" />
  <line x1="500" y1="550" x2="750" y2="550" stroke="#4e342e" stroke-width="2" opacity="0.5" />
  <!-- Screen Glow (Behind Duck and Laptop) -->
  <polygon points="450,250 750,250 800,450 400,450" fill="url(#screenGlow)" />
  <!-- Duck Programmer -->
  <g id="duck">
    <!-- Body -->
    <ellipse cx="280" cy="460" rx="90" ry="100" fill="#facc15" />
    <!-- Body shadow -->
    <ellipse cx="280" cy="480" rx="80" ry="80" fill="#eab308" opacity="0.5" />
    <!-- Tail Feathers -->
    <path d="M 190 420 Q 150 400 160 440 Q 140 430 170 460 Q 150 470 180 480 Z" fill="#facc15" stroke="#eab308" stroke-width="3" />
    <!-- Head -->
    <circle cx="300" cy="310" r="60" fill="#facc15" />
    <!-- Messy Feather (3am struggle) -->
    <path d="M 260 255 Q 250 240 265 245 Q 260 230 275 240 Z" fill="#facc15" stroke="#eab308" stroke-width="2" />
    <path d="M 280 255 Q 285 235 290 250 Z" fill="#facc15" stroke="#eab308" stroke-width="2" />
    <!-- Propeller Beanie -->
    <rect x="260" y="255" width="80" height="15" rx="5" fill="#3b82f6" />
    <rect x="260" y="255" width="80" height="5" rx="2" fill="#60a5fa" />
    <!-- Propeller Base -->
    <circle cx="300" cy="250" r="6" fill="#1e293b" />
    <!-- Propeller Blades -->
    <line x1="270" y1="250" x2="330" y2="250" stroke="#94a3b8" stroke-width="4" />
    <ellipse cx="260" cy="250" rx="15" ry="6" fill="#f59e0b" />
    <ellipse cx="340" cy="250" rx="15" ry="6" fill="#f59e0b" />
    <!-- Beak (3/4 view facing right) -->
    <path d="M 345 300 Q 400 310 350 330 Z" fill="#f97316" />
    <path d="M 345 300 Q 380 305 350 315 Z" fill="#fb923c" />
    <!-- Beak shadow -->
    <path d="M 350 315 Q 380 320 350 330 Z" fill="#ea580c" />
    <!-- Tired Eyes -->
    <!-- Left Eye (Droopy) -->
    <path d="M 285 295 Q 295 305 305 295" fill="none" stroke="#1e293b" stroke-width="5" stroke-linecap="round" />
    <!-- Right Eye (Half closed) -->
    <path d="M 320 300 Q 330 310 340 300" fill="none" stroke="#1e293b" stroke-width="5" stroke-linecap="round" />
    <!-- Under-eye bags / Dark circles -->
    <path d="M 285 305 Q 295 312 305 305" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" opacity="0.6" />
    <path d="M 320 310 Q 330 317 340 310" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" opacity="0.6" />
    <!-- Glasses -->
    <circle cx="295" cy="300" r="22" fill="none" stroke="#67e8f9" stroke-width="4" />
    <circle cx="330" cy="305" r="22" fill="none" stroke="#67e8f9" stroke-width="4" />
    <!-- Glasses Bridge -->
    <path d="M 317 300 Q 320 295 323 300" fill="none" stroke="#67e8f9" stroke-width="4" />
    <!-- Glasses Arms -->
    <line x1="273" y1="295" x2="250" y2="285" stroke="#67e8f9" stroke-width="4" stroke-linecap="round" />
    <line x1="352" y1="300" x2="370" y2="290" stroke="#67e8f9" stroke-width="4" stroke-linecap="round" />
    <!-- Sweat Drop (Stress) -->
    <path d="M 270 270 Q 270 280 265 280 Q 260 280 260 270 Q 260 260 265 255 Q 270 260 270 270 Z" fill="#38bdf8" opacity="0.8" />
    <!-- Wing (Typing on Laptop) -->
    <path d="M 330 400 Q 420 430 430 480 Q 400 500 350 480 Z" fill="#facc15" stroke="#eab308" stroke-width="4" />
    <!-- Wing fingers/feathers -->
    <path d="M 400 460 Q 420 460 430 470" fill="none" stroke="#eab308" stroke-width="3" stroke-linecap="round" />
    <path d="M 390 470 Q 410 470 420 480" fill="none" stroke="#eab308" stroke-width="3" stroke-linecap="round" />
    <path d="M 380 480 Q 400 480 410 490" fill="none" stroke="#eab308" stroke-width="3" stroke-linecap="round" />
  </g>
  <!-- Laptop -->
  <g id="laptop" transform="translate(480, 380)">
    <!-- Laptop Base -->
    <polygon points="0,100 200,100 240,40 40,40" fill="#475569" />
    <polygon points="0,100 200,100 240,40 40,40" fill="none" stroke="#1e293b" stroke-width="3" />
    <!-- Keyboard Area -->
    <rect x="20" y="60" width="160" height="30" rx="4" fill="#1e293b" />
    <!-- Keys -->
    <g fill="#64748b" stroke="#475569" stroke-width="1">
      <rect x="25" y="65" width="10" height="8" rx="1" />
      <rect x="38" y="65" width="10" height="8" rx="1" />
      <rect x="51" y="65" width="10" height="8" rx="1" />
      <rect x="64" y="65" width="10" height="8" rx="1" />
      <rect x="77" y="65" width="10" height="8" rx="1" />
      <rect x="90" y="65" width="10" height="8" rx="1" />
      <rect x="103" y="65" width="10" height="8" rx="1" />
      <rect x="116" y="65" width="10" height="8" rx="1" />
      <rect x="129" y="65" width="10" height="8" rx="1" />
      <rect x="142" y="65" width="10" height="8" rx="1" />
      <rect x="155" y="65" width="20" height="8" rx="1" />
      <rect x="25" y="78" width="20" height="8" rx="1" />
      <rect x="50" y="78" width="10" height="8" rx="1" />
      <rect x="63" y="78" width="10" height="8" rx="1" />
      <rect x="76" y="78" width="10" height="8" rx="1" />
      <rect x="89" y="78" width="10" height="8" rx="1" />
      <rect x="102" y="78" width="10" height="8" rx="1" />
      <rect x="115" y="78" width="10" height="8" rx="1" />
      <rect x="128" y="78" width="10" height="8" rx="1" />
      <rect x="141" y="78" width="10" height="8" rx="1" />
      <rect x="154" y="78" width="20" height="8" rx="1" />
    </g>
    <!-- Trackpad -->
    <rect x="70" y="90" width="60" height="8" rx="2" fill="#334155" />
    <!-- Laptop Screen -->
    <polygon points="40,40 240,40 220,-40 60,-40" fill="#0f172a" stroke="#1e293b" stroke-width="3" />
    <!-- Screen Background -->
    <polygon points="45,35 235,35 216,-35 64,-35" fill="#1e1b4b" />
    <!-- Code on Screen -->
    <g font-family="monospace" font-size="8" fill="#a5f3fc">
      <!-- Line numbers -->
      <text x="50" y="-25" fill="#475569">1</text>
      <text x="50" y="-15" fill="#475569">2</text>
      <text x="50" y="-5" fill="#475569">3</text>
      <text x="50" y="5" fill="#475569">4</text>
      <text x="50" y="15" fill="#475569">5</text>
      <text x="50" y="25" fill="#475569">6</text>
      <!-- Code content -->
      <text x="60" y="-25" fill="#c084fc">import</text>
      <text x="80" y="-25" fill="#fbbf24">coffee</text>
      <text x="60" y="-15" fill="#c084fc">import</text>
      <text x="80" y="-15" fill="#fbbf24">sleep</text>
      <text x="60" y="-5" fill="#f472b6">while</text>
      <text x="80" y="-5" fill="#e2e8f0">True</text>
      <text x="95" y="-5" fill="#e2e8f0">:</text>
      <text x="70" y="5" fill="#f472b6">try</text>
      <text x="85" y="5" fill="#e2e8f0">:</text>
      <text x="80" y="15" fill="#e2e8f0">debug()</text>
      <text x="70" y="25" fill="#f472b6">except</text>
      <text x="100" y="25" fill="#fbbf24">Bug</text>
      <text x="115" y="25" fill="#e2e8f0">:</text>
      <text x="80" y="35" fill="#e2e8f0">cry()</text>
    </g>
    <!-- The Literal Bug (Ladybug) on Screen -->
    <g transform="translate(180, -15)">
      <!-- Bug Body -->
      <ellipse cx="0" cy="0" rx="8" ry="6" fill="#ef4444" />
      <!-- Bug Head -->
      <circle cx="-8" cy="0" r="4" fill="#1e293b" />
      <!-- Bug Antennae -->
      <line x1="-10" y1="-2" x2="-15" y2="-8" stroke="#1e293b" stroke-width="1.5" />
      <line x1="-8" y1="-4" x2="-10" y2="-10" stroke="#1e293b" stroke-width="1.5" />
      <!-- Bug Spots -->
      <circle cx="-2" cy="-2" r="1.5" fill="#1e293b" />
      <circle cx="3" cy="-1" r="1.5" fill="#1e293b" />
      <circle cx="0" cy="3" r="1.5" fill="#1e293b" />
      <!-- Bug Legs -->
      <line x1="-4" y1="4" x2="-6" y2="8" stroke="#1e293b" stroke-width="1" />
      <line x1="0" y1="5" x2="0" y2="9" stroke="#1e293b" stroke-width="1" />
      <line x1="4" y1="4" x2="6" y2="8" stroke="#1e293b" stroke-width="1" />
    </g>
    <!-- Highlighted bug line -->
    <rect x="75" y="20" width="50" height="10" fill="#fef08a" opacity="0.3" />
  </g>
  <!-- Coffee Mug -->
  <g id="coffee" transform="translate(150, 420)">
    <!-- Mug Shadow -->
    <ellipse cx="30" cy="65" rx="35" ry="10" fill="#1e293b" opacity="0.5" />
    <!-- Mug Handle -->
    <path d="M 50 20 Q 75 20 75 40 Q 75 60 50 60" fill="none" stroke="#e2e8f0" stroke-width="8" stroke-linecap="round" />
    <!-- Mug Body -->
    <rect x="0" y="0" width="60" height="65" rx="8" fill="#e2e8f0" />
    <!-- Mug Design (Sass/Tea stain) -->
    <circle cx="30" cy="35" r="15" fill="none" stroke="#cbd5e1" stroke-width="2" />
    <text x="30" y="39" font-family="monospace" font-size="10" fill="#94a3b8" text-anchor="middle">:(){ :|:& };</text>
    <!-- Coffee Inside -->
    <ellipse cx="30" cy="0" rx="25" ry="8" fill="#451a03" />
    <!-- Coffee Rim -->
    <ellipse cx="30" cy="0" rx="25" ry="8" fill="none" stroke="#cbd5e1" stroke-width="2" />
    <!-- Steam -->
    <g fill="none" stroke="#cbd5e1" stroke-width="3" stroke-linecap="round" opacity="0.6">
      <path d="M 15 -5 Q 10 -15 15 -25 T 15 -40" />
      <path d="M 30 -8 Q 25 -20 30 -30 T 30 -45" />
      <path d="M 45 -5 Q 40 -15 45 -25 T 45 -35" />
    </g>
  </g>
  <!-- Additional Ambient Elements -->
  <!-- Empty Pizza Box (Bottom right) -->
  <g transform="translate(620, 480)">
    <rect x="0" y="0" width="80" height="80" rx="5" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2" />
    <rect x="5" y="5" width="70" height="70" rx="3" fill="#f1f5f9" />
    <path d="M 40 5 L 40 75" stroke="#cbd5e1" stroke-width="1" />
    <path d="M 5 40 L 75 40" stroke="#cbd5e1" stroke-width="1" />
    <!-- Grease spots -->
    <circle cx="25" cy="25" r="10" fill="#fef08a" opacity="0.5" />
    <circle cx="55" cy="55" r="12" fill="#fef08a" opacity="0.5" />
    <circle cx="25" cy="55" r="8" fill="#fef08a" opacity="0.5" />
  </g>
  <!-- Scattered Papers (Bottom left) -->
  <g transform="translate(50, 500)">
    <rect x="0" y="0" width="40" height="50" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" transform="rotate(-15)" />
    <line x1="5" y1="10" x2="35" y2="10" stroke="#94a3b8" stroke-width="1" transform="rotate(-15)" />
    <line x1="5" y1="20" x2="35" y2="20" stroke="#94a3b8" stroke-width="1" transform="rotate(-15)" />
    <line x1="5" y1="30" x2="25" y2="30" stroke="#94a3b8" stroke-width="1" transform="rotate(-15)" />
    <rect x="10" y="5" width="40" height="50" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" transform="rotate(10)" />
    <line x1="15" y1="15" x2="45" y2="15" stroke="#94a3b8" stroke-width="1" transform="rotate(10)" />
    <line x1="15" y1="25" x2="45" y2="25" stroke="#94a3b8" stroke-width="1" transform="rotate(10)" />
    <line x1="15" y1="35" x2="35" y2="35" stroke="#94a3b8" stroke-width="1" transform="rotate(10)" />
  </g>
  <!-- Overall Vibe Text (Top Right) -->
  <text x="750" y="50" font-family="monospace" font-size="14" fill="#ef4444" text-anchor="end" font-weight="bold">3 AM DEBUGGING</text>
  <text x="750" y="65" font-family="monospace" font-size="10" fill="#94a3b8" text-anchor="end">Session: 6 hrs</text>
  <text x="750" y="80" font-family="monospace" font-size="10" fill="#94a3b8" text-anchor="end">Bugs: 42</text>
  <text x="750" y="95" font-family="monospace" font-size="10" fill="#94a3b8" text-anchor="end">Sanity: 0</text>
</svg>
```

</details>

## 2. deepseek-v4-pro_cloud

**SVG size:** 9197 characters  
**Complexity:** Very high  
**Shape elements:** 72  
**Distinct colors:** 14  
**Raw response:** 9658 characters  
**Generation time:** 53.0s

![deepseek-v4-pro_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/deepseek-v4-pro_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With over 9,197 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering scene props, screen code, lighting, and character expression.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <defs>
    <!-- Background Gradient -->
    <radialGradient id="bgGlow" cx="50%" cy="50%" r="70%" fx="50%" fy="50%">
      <stop offset="0%" stop-color="#1b2330" />
      <stop offset="100%" stop-color="#0a0c10" />
    </radialGradient>
    <!-- Lamp Light Gradient -->
    <radialGradient id="lampGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#f0a500" stop-opacity="0.8" />
      <stop offset="50%" stop-color="#f0a500" stop-opacity="0.3" />
      <stop offset="100%" stop-color="#f0a500" stop-opacity="0" />
    </radialGradient>
    <!-- Screen Glow Gradient -->
    <linearGradient id="screenGlow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#58a6ff" stop-opacity="0.4" />
      <stop offset="100%" stop-color="#58a6ff" stop-opacity="0" />
    </linearGradient>
    <!-- Coffee Steam Filter -->
    <filter id="blur">
      <feGaussianBlur stdDeviation="2" />
    </filter>
  </defs>
  <!-- Background -->
  <rect width="800" height="600" fill="url(#bgGlow)" />
  <!-- Window (Top Right) -->
  <g transform="translate(620, 40)">
    <rect x="0" y="0" width="140" height="180" rx="10" fill="#161b22" stroke="#30363d" stroke-width="4" />
    <!-- Window Panes -->
    <line x1="70" y1="0" x2="70" y2="180" stroke="#30363d" stroke-width="4" />
    <line x1="0" y1="90" x2="140" y2="90" stroke="#30363d" stroke-width="4" />
    <!-- Moon -->
    <path d="M 100 30 A 25 25 0 1 1 80 60 A 30 30 0 1 0 100 30 Z" fill="#f4d03f" />
    <!-- Stars -->
    <circle cx="30" cy="40" r="2" fill="#fff" />
    <circle cx="50" cy="70" r="1.5" fill="#fff" />
    <circle cx="110" cy="80" r="2" fill="#fff" />
    <circle cx="20" cy="120" r="1.5" fill="#fff" />
    <circle cx="120" cy="140" r="2" fill="#fff" />
  </g>
  <!-- Clock (Top Center) -->
  <g transform="translate(400, 100)">
    <circle cx="0" cy="0" r="45" fill="#161b22" stroke="#30363d" stroke-width="4" />
    <circle cx="0" cy="0" r="40" fill="#0d1117" />
    <!-- Hour markers -->
    <line x1="0" y1="-35" x2="0" y2="-30" stroke="#8b949e" stroke-width="3" />
    <line x1="0" y1="35" x2="0" y2="30" stroke="#8b949e" stroke-width="3" />
    <line x1="-35" y1="0" x2="-30" y2="0" stroke="#8b949e" stroke-width="3" />
    <line x1="35" y1="0" x2="30" y2="0" stroke="#8b949e" stroke-width="3" />
    <!-- Hands (3:00 AM) -->
    <line x1="0" y1="0" x2="0" y2="-25" stroke="#58a6ff" stroke-width="4" stroke-linecap="round" />
    <line x1="0" y1="0" x2="25" y2="0" stroke="#58a6ff" stroke-width="3" stroke-linecap="round" />
    <circle cx="0" cy="0" r="4" fill="#f85149" />
  </g>
  <!-- Desk -->
  <rect x="0" y="450" width="800" height="150" fill="#1f2428" />
  <rect x="0" y="450" width="800" height="8" fill="#30363d" />
  <!-- Shadows on Desk -->
  <ellipse cx="500" cy="460" rx="140" ry="15" fill="#0d1117" opacity="0.6" />
  <ellipse cx="300" cy="520" rx="80" ry="15" fill="#0d1117" opacity="0.6" />
  <ellipse cx="142.5" cy="460" rx="25" ry="8" fill="#0d1117" opacity="0.6" />
  <!-- Desk Lamp (Right Side) -->
  <g>
    <!-- Lamp Base -->
    <ellipse cx="700" cy="450" rx="30" ry="10" fill="#30363d" />
    <rect x="695" y="300" width="10" height="150" fill="#484f58" />
    <!-- Lamp Arms -->
    <line x1="700" y1="300" x2="600" y2="220" stroke="#484f58" stroke-width="8" stroke-linecap="round" />
    <line x1="600" y1="220" x2="500" y2="240" stroke="#484f58" stroke-width="8" stroke-linecap="round" />
    <!-- Lamp Head -->
    <path d="M 480 220 L 520 260 L 500 270 L 460 230 Z" fill="#f0a500" />
    <!-- Light Cone -->
    <polygon points="460,230 520,260 650,500 300,500" fill="url(#lampGlow)" />
  </g>
  <!-- Laptop -->
  <g>
    <!-- Screen Back -->
    <rect x="380" y="220" width="240" height="180" rx="10" fill="#21262d" stroke="#30363d" stroke-width="4" />
    <!-- Screen Bezel -->
    <rect x="385" y="225" width="230" height="170" rx="5" fill="#0d1117" />
    <!-- Screen Display -->
    <rect x="390" y="230" width="220" height="160" rx="2" fill="#0d1117" />
    <!-- Code on Screen -->
    <text x="400" y="250" font-family="monospace" font-size="12" fill="#58a6ff">
      <tspan x="400" dy="0">1  const duck = {</tspan>
      <tspan x="400" dy="18">2    name: "Quack",</tspan>
      <tspan x="400" dy="18">3    debug: () => {</tspan>
      <tspan x="400" dy="18">4      while(true) {</tspan>
      <tspan x="400" dy="18">5        coffee.drink();</tspan>
      <tspan x="400" dy="18">6        console.log("🦆");</tspan>
      <tspan x="400" dy="18">7      }</tspan>
      <tspan x="400" dy="18">8    }</tspan>
      <tspan x="400" dy="18">9  }</tspan>
    </text>
    <!-- Screen Glow Effect -->
    <polygon points="390,230 610,230 610,390 390,390" fill="url(#screenGlow)" />
    <!-- Keyboard Base -->
    <rect x="370" y="410" width="260" height="20" rx="4" fill="#21262d" stroke="#30363d" stroke-width="2" />
    <!-- Keys -->
    <rect x="380" y="415" width="240" height="10" rx="2" fill="#0d1117" />
    <!-- Trackpad -->
    <rect x="480" y="415" width="40" height="10" rx="2" fill="#30363d" />
  </g>
  <!-- Sticky Notes -->
  <g>
    <!-- Note on Desk -->
    <rect x="500" y="460" width="60" height="60" fill="#f0a500" transform="rotate(-15 500 460)" />
    <line x1="510" y1="480" x2="540" y2="480" stroke="#0d1117" stroke-width="2" transform="rotate(-15 500 460)" />
    <line x1="510" y1="490" x2="530" y2="490" stroke="#0d1117" stroke-width="2" transform="rotate(-15 500 460)" />
    <line x1="510" y1="500" x2="545" y2="500" stroke="#0d1117" stroke-width="2" transform="rotate(-15 500 460)" />
    <!-- Note on Laptop -->
    <rect x="420" y="200" width="50" height="50" fill="#58a6ff" transform="rotate(10 420 200)" />
    <line x1="430" y1="220" x2="450" y2="220" stroke="#0d1117" stroke-width="2" transform="rotate(10 420 200)" />
    <line x1="430" y1="230" x2="460" y2="230" stroke="#0d1117" stroke-width="2" transform="rotate(10 420 200)" />
  </g>
  <!-- Duck Programmer -->
  <g>
    <!-- Duck Body (Hoodie) -->
    <path d="M 230 400 C 200 450, 200 520, 300 520 C 400 520, 400 450, 370 400 Z" fill="#30363d" />
    <!-- Hoodie Strings -->
    <line x1="280" y1="400" x2="270" y2="430" stroke="#f0a500" stroke-width="3" stroke-linecap="round" />
    <line x1="320" y1="400" x2="330" y2="430" stroke="#f0a500" stroke-width="3" stroke-linecap="round" />
    <!-- Duck Head -->
    <circle cx="300" cy="280" r="55" fill="#f0a500" />
    <!-- Hood -->
    <path d="M 245 280 C 245 220, 355 220, 355 280 C 355 300, 245 300, 245 280 Z" fill="#30363d" />
    <!-- Beak -->
    <path d="M 250 270 L 180 290 L 250 310 Z" fill="#f85149" />
    <!-- Beak Detail -->
    <line x1="250" y1="290" x2="190" y2="290" stroke="#c0392b" stroke-width="2" />
    <!-- Eyes (Tired) -->
    <!-- Left Eye -->
    <circle cx="280" cy="270" r="10" fill="#fff" />
    <circle cx="275" cy="270" r="5" fill="#0d1117" />
    <!-- Eye Bag -->
    <path d="M 270 280 Q 280 285 290 280" stroke="#8b949e" stroke-width="2" fill="none" />
    <!-- Right Eye (partially visible) -->
    <circle cx="320" cy="270" r="8" fill="#fff" />
    <circle cx="315" cy="270" r="4" fill="#0d1117" />
    <path d="M 312 278 Q 320 283 328 278" stroke="#8b949e" stroke-width="2" fill="none" />
    <!-- Glasses -->
    <circle cx="280" cy="270" r="16" fill="none" stroke="#0d1117" stroke-width="3" />
    <circle cx="320" cy="270" r="14" fill="none" stroke="#0d1117" stroke-width="3" />
    <line x1="296" y1="270" x2="306" y2="270" stroke="#0d1117" stroke-width="3" />
    <!-- Screen Reflection on Glasses -->
    <ellipse cx="280" cy="270" rx="8" ry="4" fill="#58a6ff" opacity="0.6" transform="rotate(-30 280 270)" />
    <ellipse cx="320" cy="270" rx="6" ry="3" fill="#58a6ff" opacity="0.6" transform="rotate(-30 320 270)" />
    <!-- Duck Wings (Typing) -->
    <!-- Left Wing -->
    <path d="M 250 420 Q 280 440 380 420" fill="none" stroke="#f0a500" stroke-width="20" stroke-linecap="round" />
    <!-- Right Wing -->
    <path d="M 350 420 Q 380 440 400 420" fill="none" stroke="#f0a500" stroke-width="20" stroke-linecap="round" />
  </g>
  <!-- Coffee Mug -->
  <g>
    <!-- Mug Shadow -->
    <ellipse cx="142.5" cy="450" rx="25" ry="8" fill="#0d1117" opacity="0.6" />
    <!-- Mug Body -->
    <rect x="120" y="380" width="45" height="55" rx="5" fill="#f85149" />
    <!-- Mug Handle -->
    <path d="M 165 390 Q 185 390 185 410 Q 185 430 165 430" fill="none" stroke="#f85149" stroke-width="6" stroke-linecap="round" />
    <!-- Coffee Inside -->
    <ellipse cx="142.5" cy="380" rx="22.5" ry="6" fill="#3e2723" />
    <!-- Coffee Stain on Desk -->
    <ellipse cx="200" cy="480" rx="15" ry="10" fill="#3e2723" opacity="0.4" />
    <!-- Steam -->
    <g filter="url(#blur)" opacity="0.6">
      <path d="M 130 370 Q 125 350 135 340 T 130 310" fill="none" stroke="#8b949e" stroke-width="3" stroke-linecap="round" />
      <path d="M 150 365 Q 155 345 145 335 T 150 305" fill="none" stroke="#8b949e" stroke-width="3" stroke-linecap="round" />
      <path d="M 140 375 Q 135 355 145 345 T 140 315" fill="none" stroke="#8b949e" stroke-width="2" stroke-linecap="round" />
    </g>
  </g>
</svg>
```

</details>

## 3. gemma4_31b-cloud

**SVG size:** 3056 characters  
**Complexity:** High  
**Shape elements:** 31  
**Distinct colors:** 20  
**Raw response:** 3865 characters  
**Generation time:** 29.7s

![gemma4_31b-cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/gemma4_31b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With 3,056 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <!-- Background: Dark Room -->
  <rect width="800" height="600" fill="#1a1a2e" />
  <!-- Window with Moon (Night Time) -->
  <rect x="600" y="50" width="120" height="150" rx="10" fill="#0f0f1a" stroke="#333" stroke-width="4" />
  <circle cx="660" cy="120" r="30" fill="#f0f0f0" />
  <circle cx="640" cy="100" r="15" fill="#0f0f1a" opacity="0.5" /> <!-- Moon crater -->
  <!-- Desk -->
  <rect x="100" y="450" width="600" height="20" rx="5" fill="#4a3728" />
  <rect x="150" y="470" width="20" height="130" fill="#3a2718" />
  <rect x="630" y="470" width="20" height="130" fill="#3a2718" />
  <!-- Monitor and Glow -->
  <ellipse cx="400" cy="400" rx="250" ry="100" fill="#00f2ff" opacity="0.1" /> <!-- Screen Glow -->
  <rect x="250" y="200" width="300" height="200" rx="10" fill="#333" stroke="#555" stroke-width="8" />
  <rect x="265" y="215" width="270" height="170" rx="5" fill="#0d0d0d" />
  <!-- Code Lines on Screen -->
  <g stroke="#4ecdc4" stroke-width="3" stroke-linecap="round">
    <line x1="280" y1="235" x2="350" y1="235" />
    <line x1="280" y1="255" x2="420" y1="255" />
    <line x1="280" y1="275" x2="380" y1="275" />
    <line x1="310" y1="295" x2="450" y1="295" />
    <!-- Red Error Line -->
    <line x1="280" y1="315" x2="400" y1="315" stroke="#ff5f5f" />
    <circle cx="275" cy="315" r="4" fill="#ff5f5f" />
  </g>
  <!-- Coffee Cup -->
  <path d="M580 430 h40 a10 10 0 0 1 10 10 v20 a10 10 0 0 1 -10 10 h-40 a10 10 0 0 1 -10 -10 v-20 a10 10 0 0 1 10 -10 Z" fill="#eee" />
  <path d="M620 435 h15 a5 5 0 0 1 5 5 v10 a5 5 0 0 1 -5 5 h-15" fill="none" stroke="#eee" stroke-width="5" />
  <!-- Steam -->
  <path d="M590 420 q5 -10 10 0 t10 0" fill="none" stroke="#ccc" stroke-width="2" opacity="0.6" />
  <path d="M600 415 q5 -10 10 0 t10 0" fill="none" stroke="#ccc" stroke-width="2" opacity="0.6" />
  <!-- The Programmer Duck -->
  <g transform="translate(350, 350)">
    <!-- Body -->
    <ellipse cx="40" cy="40" rx="50" ry="40" fill="#ffd700" />
    <!-- Head -->
    <circle cx="70" cy="10" r="30" fill="#ffd700" />
    <!-- Beak -->
    <path d="M95 10 l15 5 l-15 5 Z" fill="#ff8c00" />
    <!-- Eye (Tired/Sleepy) -->
    <path d="M75 5 q5 5 10 0" fill="none" stroke="#333" stroke-width="3" />
    <circle cx="80" cy="12" r="2" fill="#333" />
    <!-- Wing/Hand on Keyboard -->
    <path d="M20 40 q-20 0 -20 10 t20 10" fill="#ffd700" stroke="#e6c200" stroke-width="2" />
    <!-- Headset -->
    <path d="M55 0 a20 20 0 0 1 30 0" fill="none" stroke="#222" stroke-width="5" />
    <rect x="50" y="0" width="15" height="20" rx="5" fill="#222" />
    <rect x="90" y="0" width="15" height="20" rx="5" fill="#222" />
  </g>
  <!-- Keyboard -->
  <rect x="330" y="430" width="140" height="20" rx="5" fill="#222" />
  <!-- Digital Clock -->
  <rect x="650" y="20" width="80" height="30" rx="5" fill="#000" stroke="#444" />
  <text x="665" y="42" font-family="monospace" font-size="20" fill="#0f0" font-weight="bold">03:00 AM</text>
</svg>
```

</details>

## 4. gemma4_cloud

**SVG size:** 3162 characters  
**Complexity:** Medium  
**Shape elements:** 28  
**Distinct colors:** 16  
**Raw response:** 3943 characters  
**Generation time:** 16.9s

![gemma4_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/gemma4_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Includes animation elements -- the model attempted motion (e.g., steam, blinking)
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With 3,162 characters of SVG markup, this is a balanced output -- enough detail to convey the scene without being overwhelming.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="auto">
  <!-- Background: Dark Room -->
  <rect width="500" height="500" fill="#1a1a2e" />
  <!-- Desk -->
  <rect x="50" y="350" width="400" height="20" rx="10" fill="#3d2b1f" />
  <!-- The Monitor (Glowing) -->
  <rect x="150" y="150" width="200" height="130" rx="10" fill="#222" stroke="#444" stroke-width="8"/>
  <rect x="160" y="160" width="180" height="110" rx="5" fill="#001f3f" />
  <!-- "Code" on Screen -->
  <rect x="170" y="170" width="60" height="6" rx="3" fill="#4ade80" />
  <rect x="170" y="180" width="100" height="6" rx="3" fill="#f87171" /> <!-- Bug line -->
  <rect x="170" y="190" width="80" height="6" rx="3" fill="#60a5fa" />
  <rect x="170" y="200" width="120" height="6" rx="3" fill="#fbbf24" />
  <rect x="170" y="210" width="40" height="6" rx="3" fill="#4ade80" />
  <!-- Screen Glow Effect -->
  <ellipse cx="250" cy="220" rx="150" ry="80" fill="rgba(0, 150, 255, 0.15)" />
  <!-- The Duck -->
  <g id="duck">
    <!-- Body -->
    <ellipse cx="250" cy="320" rx="50" ry="35" fill="#ffd700" />
    <!-- Head -->
    <circle cx="280" cy="280" r="25" fill="#ffd700" />
    <!-- Beak -->
    <path d="M 300 280 L 320 285 L 300 290 Z" fill="#ff8c00" />
    <!-- Eye/Glasses -->
    <circle cx="285" cy="275" r="4" fill="black" />
    <!-- Blue Light Glasses -->
    <rect x="270" y="265" width="25" height="15" rx="5" fill="rgba(0, 191, 255, 0.5)" stroke="#fff" stroke-width="1" />
    <rect x="290" y="265" width="25" height="15" rx="5" fill="rgba(0, 191, 255, 0.5)" stroke="#fff" stroke-width="1" />
    <line x1="295" y1="272" x2="295" y2="272" stroke="#fff" stroke-width="2" />
  </g>
  <!-- Coffee Mug -->
  <g id="coffee">
    <rect x="340" y="310" width="30" height="40" rx="5" fill="#ddd" />
    <path d="M 370 320 Q 380 320 380 330 Q 380 340 370 340" fill="none" stroke="#ddd" stroke-width="4" />
    <!-- Steam -->
    <path d="M 345 300 Q 350 290 355 300" stroke="white" fill="none" stroke-width="2" opacity="0.6">
      <animate attributeName="opacity" values="0.2;1;0.2" dur="2s" repeatCount="indefinite" />
    </path>
    <path d="M 355 300 Q 360 290 365 300" stroke="white" fill="none" stroke-width="2" opacity="0.6">
      <animate attributeName="opacity" values="0.2;1;0.2" dur="2.5s" repeatCount="indefinite" />
    </path>
  </g>
  <!-- 3 AM Clock -->
  <circle cx="80" cy="80" r="30" fill="#333" stroke="#666" stroke-width="3" />
  <line x1="80" y1="80" x2="80" y2="60" stroke="white" stroke-width="2" /> <!-- Hour hand at 12 (or 3) -->
  <line x1="80" y1="80" x2="100" y2="80" stroke="white" stroke-width="2" /> <!-- Min hand at 3 -->
  <text x="72" y="125" fill="#888" font-family="Arial" font-size="14" font-weight="bold">3:00 AM</text>
  <!-- Floating "Bugs" -->
  <g fill="#f87171">
    <circle cx="120" cy="200" r="4" />
    <circle cx="380" cy="180" r="4" />
    <circle cx="400" cy="250" r="4" />
    <circle cx="100" cy="280" r="4" />
  </g>
  <!-- Caption -->
  <text x="250" y="450" text-anchor="middle" fill="#fff" font-family="Courier New, monospace" font-size="20">
    &lt;Rubber Duck Debugging... /&gt;
  </text>
</svg>
```

</details>

## 5. glm-5.1_cloud

**SVG size:** 30938 characters  
**Complexity:** Very high  
**Shape elements:** 194  
**Distinct colors:** 79  
**Raw response:** 36377 characters  
**Generation time:** 95.9s

![glm-5.1_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/glm-5.1_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With over 30,938 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering scene props, screen code, lighting, and character expression.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <defs>
    <!-- Background gradient -->
    <radialGradient id="bgGrad" cx="50%" cy="40%" r="70%">
      <stop offset="0%" stop-color="#111930"/>
      <stop offset="100%" stop-color="#060a14"/>
    </radialGradient>
    <!-- Monitor screen glow -->
    <radialGradient id="screenGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#1a5a8a" stop-opacity="0.5"/>
      <stop offset="100%" stop-color="#0a0e1a" stop-opacity="0"/>
    </radialGradient>
    <!-- Coffee gradient -->
    <linearGradient id="coffeeGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5a3218"/>
      <stop offset="100%" stop-color="#6f3e2a"/>
    </linearGradient>
    <!-- Desk gradient -->
    <linearGradient id="deskGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4a3220"/>
      <stop offset="100%" stop-color="#2a1a0e"/>
    </linearGradient>
    <!-- Duck body gradient -->
    <radialGradient id="duckBody" cx="50%" cy="40%" r="55%">
      <stop offset="0%" stop-color="#ffe566"/>
      <stop offset="100%" stop-color="#ffc830"/>
    </radialGradient>
    <!-- Duck head gradient -->
    <radialGradient id="duckHead" cx="50%" cy="45%" r="55%">
      <stop offset="0%" stop-color="#ffe566"/>
      <stop offset="100%" stop-color="#ffd23f"/>
    </radialGradient>
    <!-- Monitor bezel gradient -->
    <linearGradient id="bezelGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#2a2a3a"/>
      <stop offset="100%" stop-color="#1a1a28"/>
    </linearGradient>
    <!-- Screen scanlines -->
    <pattern id="scanlines" patternUnits="userSpaceOnUse" width="4" height="4">
      <line x1="0" y1="0" x2="4" y2="0" stroke="#000" stroke-width="0.5" opacity="0.15"/>
    </pattern>
    <!-- Mug gradient -->
    <linearGradient id="mugGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#e0d5c5"/>
      <stop offset="50%" stop-color="#f5efe5"/>
      <stop offset="100%" stop-color="#c8baa8"/>
    </linearGradient>
    <!-- Moon glow -->
    <radialGradient id="moonGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fffde0" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#fffde0" stop-opacity="0"/>
    </radialGradient>
    <!-- Keyboard gradient -->
    <linearGradient id="kbGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#333345"/>
      <stop offset="100%" stop-color="#222235"/>
    </linearGradient>
    <!-- Coffee inside glow -->
    <radialGradient id="coffeeTop" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#7a4528"/>
      <stop offset="100%" stop-color="#5a3218"/>
    </radialGradient>
    <filter id="glowFilter">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="softGlow">
      <feGaussianBlur stdDeviation="8"/>
    </filter>
    <clipPath id="screenClip">
      <rect x="252" y="88" width="296" height="195" rx="2"/>
    </clipPath>
  </defs>
  <!-- ===== BACKGROUND ===== -->
  <rect width="800" height="600" fill="url(#bgGrad)"/>
  <!-- Window -->
  <rect x="80" y="30" width="180" height="200" rx="5" fill="#0c1225" stroke="#1a2545" stroke-width="3"/>
  <rect x="83" y="33" width="174" height="194" rx="3" fill="#070d1e"/>
  <!-- Moon -->
  <circle cx="150" cy="80" r="40" fill="url(#moonGlow)" filter="url(#softGlow)"/>
  <circle cx="150" cy="80" r="22" fill="#fffde0" opacity="0.9"/>
  <circle cx="143" cy="74" r="4" fill="#efe8c0" opacity="0.5"/>
  <circle cx="157" cy="85" r="3" fill="#efe8c0" opacity="0.4"/>
  <!-- Stars in window -->
  <g class="stars">
    <circle cx="110" cy="60" r="1.5" fill="#fff" opacity="0.7" style="animation: twinkle 3s ease-in-out 0.2s infinite"/>
    <circle cx="200" cy="55" r="1" fill="#fff" opacity="0.5" style="animation: twinkle 4s ease-in-out 1s infinite"/>
    <circle cx="130" cy="100" r="1.2" fill="#fff" opacity="0.6" style="animation: twinkle 2.5s ease-in-out 0.5s infinite"/>
    <circle cx="180" cy="130" r="1" fill="#fff" opacity="0.4" style="animation: twinkle 3.5s ease-in-out 1.5s infinite"/>
    <circle cx="100" cy="150" r="1.3" fill="#fff" opacity="0.5" style="animation: twinkle 2.8s ease-in-out 0.8s infinite"/>
    <circle cx="220" cy="80" r="0.8" fill="#fff" opacity="0.3" style="animation: twinkle 3.2s ease-in-out 2s infinite"/>
    <circle cx="160" cy="170" r="1" fill="#fff" opacity="0.4" style="animation: twinkle 4.2s ease-in-out 0.3s infinite"/>
  </g>
  <!-- More stars outside window -->
  <circle cx="400" cy="25" r="1" fill="#fff" opacity="0.3" style="animation: twinkle 5s ease-in-out 0.7s infinite"/>
  <circle cx="650" cy="40" r="1.2" fill="#fff" opacity="0.25" style="animation: twinkle 3.8s ease-in-out 1.2s infinite"/>
  <circle cx="720" cy="80" r="0.8" fill="#fff" opacity="0.2" style="animation: twinkle 4.5s ease-in-out infinite"/>
  <!-- ===== DESK ===== -->
  <rect x="50" y="420" width="700" height="18" rx="3" fill="url(#deskGrad)"/>
  <rect x="50" y="420" width="700" height="4" rx="2" fill="#5a3e28" opacity="0.6"/>
  <!-- Desk legs -->
  <rect x="100" y="438" width="16" height="120" rx="2" fill="#2a1a0e"/>
  <rect x="684" y="438" width="16" height="120" rx="2" fill="#2a1a0e"/>
  <!-- Cross bar -->
  <rect x="100" y="520" width="600" height="8" rx="2" fill="#2a1a0e" opacity="0.6"/>
  <!-- ===== MONITOR ===== -->
  <!-- Monitor stand -->
  <rect x="370" y="390" width="60" height="30" rx="2" fill="#222235"/>
  <rect x="340" y="410" width="120" height="12" rx="4" fill="#2a2a3a"/>
  <!-- Monitor body -->
  <rect x="245" y="80" width="310" height="215" rx="8" fill="url(#bezelGrad)" stroke="#333348" stroke-width="2"/>
  <!-- Inner bezel highlight -->
  <rect x="248" y="83" width="304" height="209" rx="6" fill="none" stroke="#3a3a50" stroke-width="0.5" opacity="0.5"/>
  <!-- Screen -->
  <g clip-path="url(#screenClip)">
    <rect x="252" y="88" width="296" height="195" fill="#0d1a2a" class="monitor-screen"/>
    <!-- Code lines -->
    <g font-family="'Courier New', monospace" font-size="8.5">
      <!-- Line numbers -->
      <g fill="#3a5a7a" font-size="7">
        <text x="258" y="106">1</text>
        <text x="258" y="118">2</text>
        <text x="258" y="130">3</text>
        <text x="258" y="142">4</text>
        <text x="258" y="154">5</text>
        <text x="258" y="166">6</text>
        <text x="258" y="178">7</text>
        <text x="258" y="190">8</text>
        <text x="258" y="202">9</text>
        <text x="258" y="214">10</text>
        <text x="258" y="226">11</text>
        <text x="258" y="238">12</text>
        <text x="258" y="250">13</text>
        <text x="258" y="262">14</text>
        <text x="258" y="274">15</text>
      </g>
      <!-- Line number gutter line -->
      <line x1="272" y1="92" x2="272" y2="280" stroke="#1a3050" stroke-width="1"/>
      <!-- Code content -->
      <!-- Line 1: import -->
      <text x="278" y="106" fill="#5eafff">import</text>
      <text x="318" y="106" fill="#ffd23f">{ useState }</text>
      <text x="388" y="106" fill="#5eafff">from</text>
      <text x="408" y="106" fill="#4afb7a">'react'</text>
      <text x="448" y="106" fill="#666">;</text>
      <!-- Line 2: blank -->
      <!-- Line 3: function -->
      <text x="278" y="130" fill="#c792ea">function</text>
      <text x="324" y="130" fill="#ffcb6b">DuckApp</text>
      <text x="374" y="130" fill="#ffd23f">()</text>
      <text x="384" y="130" fill="#fff">{</text>
      <!-- Line 4: const -->
      <text x="288" y="142" fill="#c792ea">const</text>
      <text x="318" y="142" fill="#ffcb6b">[data, setData]</text>
      <text x="410" y="142" fill="#89ddff">=</text>
      <text x="420" y="142" fill="#82aaff">useState</text>
      <text x="465" y="142" fill="#ffd23f">(null)</text>
      <text x="498" y="142" fill="#666">;</text>
      <!-- Line 5: const error -->
      <text x="288" y="154" fill="#c792ea">const</text>
      <text x="318" y="154" fill="#ffcb6b">[bugs, setBugs]</text>
      <text x="420" y="154" fill="#89ddff">=</text>
      <text x="430" y="154" fill="#82aaff">useState</text>
      <text x="475" y="154" fill="#ffd23f">(Infinity)</text>
      <text x="528" y="154" fill="#666">;</text>
      <!-- Line 6: blank -->
      <!-- Line 7: useEffect with bug -->
      <text x="288" y="178" fill="#c792ea">useEffect</text>
      <text x="346" y="178" fill="#ffd23f">(() =></text>
      <text x="384" y="178" fill="#fff">{</text>
      <!-- Line 8: fetch -->
      <text x="298" y="190" fill="#82aaff">fetchData</text>
      <text x="348" y="190" fill="#ffd23f">()</text>
      <text x="360" y="190" fill="#89ddff">.</text>
      <text x="366" y="190" fill="#82aaff">then</text>
      <text x="388" y="190" fill="#ffd23f">(res =></text>
      <text x="430" y="190" fill="#ff5e5e">res.json()</text>
      <text x="474" y="190" fill="#666">)</text>
      <!-- Line 9: catch — highlighted error line -->
      <rect x="278" y="197" width="265" height="12" fill="#ff5e5e" opacity="0.1" rx="1"/>
      <text x="298" y="202" fill="#82aaff">.catch</text>
      <text x="332" y="202" fill="#ffd23f">(err =></text>
      <text x="376" y="202" fill="#ff5e5e">console.log(err)</text>
      <text x="456" y="202" fill="#666">)</text>
      <!-- Error underline -->
      <line x1="376" y1="204" x2="454" y2="204" stroke="#ff5e5e" stroke-width="1" stroke-dasharray="2,1"/>
      <!-- Line 10: blank -->
      <!-- Line 11: return -->
      <text x="288" y="226" fill="#c792ea">return</text>
      <text x="324" y="226" fill="#ffd23f">(</text>
      <!-- Line 12: div -->
      <text x="298" y="238" fill="#ff5e5e">&lt;div&gt;</text>
      <!-- Line 13: bug in rendering -->
      <rect x="278" y="245" width="265" height="12" fill="#ff5e5e" opacity="0.1" rx="1"/>
      <text x="308" y="250" fill="#4afb7a">{bugs.map(b =&gt;</text>
      <text x="420" y="250" fill="#ff5e5e">/* CRASH */</text>
      <text x="484" y="250" fill="#666">)}</text>
      <!-- Line 14 -->
      <text x="298" y="262" fill="#ff5e5e">&lt;/div&gt;</text>
      <!-- Line 15 -->
      <text x="288" y="274" fill="#fff">}</text>
    </g>
    <!-- Blinking cursor -->
    <rect x="498" y="154" width="5" height="10" fill="#5eafff" class="cursor-blink"/>
    <!-- Scanlines overlay -->
    <rect x="252" y="88" width="296" height="195" fill="url(#scanlines)" opacity="0.4"/>
  </g>
  <!-- Screen reflection -->
  <rect x="252" y="88" width="296" height="195" rx="2" fill="none" stroke="#2a4a6a" stroke-width="0.5" opacity="0.3"/>
  <line x1="260" y1="92" x2="290" y2="92" stroke="#4a7aaa" stroke-width="0.5" opacity="0.15"/>
  <!-- Monitor power LED -->
  <circle cx="400" cy="285" r="2.5" fill="#4afb7a" opacity="0.8"/>
  <circle cx="400" cy="285" r="4" fill="#4afb7a" opacity="0.2"/>
  <!-- Screen glow on surroundings -->
  <ellipse cx="400" cy="350" rx="200" ry="80" fill="url(#screenGlow)" class="screen-glow"/>
  <!-- ===== KEYBOARD ===== -->
  <g transform="translate(290, 370)">
    <rect x="0" y="0" width="220" height="50" rx="4" fill="url(#kbGrad)" stroke="#3a3a50" stroke-width="1"/>
    <!-- Key rows -->
    <g fill="#4a4a60" font-family="monospace" font-size="5">
      <!-- Row 1 -->
      <g>
        <rect x="8" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="20" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="32" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="44" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="56" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="68" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="80" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="92" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="104" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="116" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="128" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="140" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="152" y="6" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="164" y="6" width="28" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
      </g>
      <!-- Row 2 -->
      <g>
        <rect x="8" y="16" width="14" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="24" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="36" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="48" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="60" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="72" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="84" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="96" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="108" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="120" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="132" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="144" y="16" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="156" y="16" width="36" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
      </g>
      <!-- Row 3 - spacebar row -->
      <g>
        <rect x="8" y="26" width="18" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="28" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="40" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="52" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="64" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="76" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="88" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="100" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="112" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="124" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="136" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="148" y="26" width="10" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
        <rect x="160" y="26" width="32" height="8" rx="1" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
      </g>
      <!-- Spacebar -->
      <rect x="50" y="36" width="110" height="8" rx="2" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
      <rect x="8" y="36" width="36" height="8" rx="2" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
      <rect x="164" y="36" width="28" height="8" rx="2" fill="#3a3a50" stroke="#4a4a60" stroke-width="0.5"/>
      <!-- WASD highlight (duck is gaming the debugger) -->
      <rect x="36" y="16" width="10" height="8" rx="1" fill="#4a6a50" stroke="#6afa8a" stroke-width="0.5" opacity="0.6"/>
    </g>
  </g>
  <!-- ===== COFFEE MUG ===== -->
  <g transform="translate(590, 340)">
    <!-- Mug shadow -->
    <ellipse cx="30" cy="80" rx="28" ry="5" fill="#000" opacity="0.3"/>
    <!-- Mug body -->
    <path d="M0,10 L5,75 Q30,82 55,75 L60,10 Z" fill="url(#mugGrad)" stroke="#b8a88a" stroke-width="0.5"/>
    <!-- Mug inside (coffee visible) -->
    <ellipse cx="30" cy="12" rx="30" ry="8" fill="#4a2810"/>
    <ellipse cx="30" cy="12" rx="28" ry="6" fill="url(#coffeeTop)"/>
    <!-- Coffee highlight -->
    <ellipse cx="24" cy="11" rx="10" ry="3" fill="#7a4a2a" opacity="0.5"/>
    <!-- Mug handle -->
    <path d="M60,25 Q82,25 82,48 Q82,68 60,68" fill="none" stroke="#d0c4a8" stroke-width="5"/>
    <path d="M60,25 Q78,27 78,48 Q78,65 60,68" fill="none" stroke="#e0d5c5" stroke-width="3"/>
    <!-- Mug rim highlight -->
    <ellipse cx="30" cy="10" rx="30" ry="8" fill="none" stroke="#f0e8d8" stroke-width="1" opacity="0.4"/>
    <!-- Caffeine label -->
    <text x="30" y="55" text-anchor="middle" font-family="'Courier New', monospace" font-size="7" fill="#5a3218" font-weight="bold" opacity="0.6">H2O</text>
    <!-- Steam -->
    <g transform="translate(0, -5)">
      <path d="M15,-5 Q12,-15 18,-22" fill="none" stroke="#aaa" stroke-width="1.5" opacity="0.4" class="steam-1"/>
      <path d="M30,-8 Q27,-20 33,-28" fill="none" stroke="#aaa" stroke-width="1.5" opacity="0.3" class="steam-2"/>
      <path d="M44,-4 Q41,-14 47,-20" fill="none" stroke="#aaa" stroke-width="1.5" opacity="0.35" class="steam-3"/>
    </g>
  </g>
  <!-- ===== DUCK ===== -->
  <g transform="translate(195, 290)">
    <!-- Duck shadow on desk -->
    <ellipse cx="40" cy="128" rx="55" ry="8" fill="#000" opacity="0.25"/>
    <!-- Duck body -->
    <ellipse cx="40" cy="100" rx="55" ry="35" fill="url(#duckBody)"/>
    <!-- Body highlight -->
    <ellipse cx="35" cy="90" rx="30" ry="15" fill="#ffe566" opacity="0.4"/>
    <!-- Belly -->
    <ellipse cx="40" cy="108" rx="35" ry="18" fill="#fff3c4" opacity="0.5"/>
    <!-- Duck wing (reaching toward keyboard) -->
    <path d="M70,88 Q95,82 108,100 Q112,108 100,112 Q85,108 75,105" fill="#f0b820" stroke="#e0a810" stroke-width="1"/>
    <!-- Wing feather lines -->
    <path d="M80,95 Q90,92 100,98" fill="none" stroke="#d09810" stroke-width="0.8" opacity="0.5"/>
    <path d="M78,100 Q88,97 98,103" fill="none" stroke="#d09810" stroke-width="0.8" opacity="0.5"/>
    <!-- Tail feathers -->
    <path d="M-10,88 Q-25,75 -20,60" fill="#f0b820" stroke="#d09010" stroke-width="1.5"/>
    <path d="M-5,85 Q-18,70 -15,55" fill="#e8a810" stroke="#d09010" stroke-width="1"/>
    <path d="M0,82 Q-10,68 -8,52" fill="#f0b820" stroke="#d09010" stroke-width="1"/>
    <!-- Neck -->
    <path d="M25,75 Q20,55 22,35 Q25,20 35,15" fill="#ffd23f" stroke="#e0a810" stroke-width="1"/>
    <path d="M50,78 Q55,58 50,38 Q47,22 40,15" fill="#ffd23f" stroke="#e0a810" stroke-width="1"/>
    <!-- Duck head -->
    <circle cx="37" cy="10" r="22" fill="url(#duckHead)"/>
    <!-- Head highlight -->
    <circle cx="33" cy="5" r="10" fill="#ffe566" opacity="0.4"/>
    <!-- Duck eyes (tired!) -->
    <g transform="translate(37, 5)">
      <!-- Left eye -->
      <g class="duck-eye" transform="translate(-8, 0)">
        <ellipse cx="0" cy="0" rx="5" ry="5.5" fill="#fff"/>
        <circle cx="-1" cy="1" r="3" fill="#111"/>
        <circle cx="-2" cy="-1" r="1" fill="#fff" opacity="0.7"/>
        <!-- Bags under eyes -->
        <path d="M-5,5 Q0,9 5,5" fill="none" stroke="#c89020" stroke-width="1.5" opacity="0.6"/>
      </g>
      <!-- Right eye -->
      <g class="duck-eye" transform="translate(8, 0)">
        <ellipse cx="0" cy="0" rx="5" ry="5.5" fill="#fff"/>
        <circle cx="1" cy="1" r="3" fill="#111"/>
        <circle cx="0" cy="-1" r="1" fill="#fff" opacity="0.7"/>
        <!-- Bags under eyes -->
        <path d="M-5,5 Q0,9 5,5" fill="none" stroke="#c89020" stroke-width="1.5" opacity="0.6"/>
      </g>
    </g>
    <!-- Duck beak -->
    <path d="M37,14 L52,18 L37,22 Z" fill="#ff8c21" stroke="#e07010" stroke-width="0.5"/>
    <!-- Beak nostril -->
    <circle cx="43" cy="18" r="1" fill="#e07010" opacity="0.5"/>
    <!-- Head tuft (bedhead from 3am) -->
    <path d="M30,-10 Q28,-22 35,-25" fill="none" stroke="#ffd23f" stroke-width="2.5" stroke-linecap="round"/>
    <path d="M35,-11 Q33,-24 40,-26" fill="none" stroke="#ffc830" stroke-width="2" stroke-linecap="round"/>
    <path d="M40,-10 Q40,-22 47,-23" fill="none" stroke="#ffd23f" stroke-width="2.5" stroke-linecap="round"/>
    <!-- Coffee stain on beak -->
    <circle cx="42" cy="18" r="2" fill="#5a3218" opacity="0.3"/>
  </g>
  <!-- ===== RUBBER DUCK (on desk — rubber duck debugging!) ===== -->
  <g transform="translate(610, 372)">
    <!-- Mini rubber duck -->
    <ellipse cx="0" cy="0" rx="16" ry="12" fill="#ffcb05"/>
    <!-- Head -->
    <circle cx="-10" cy="-10" r="9" fill="#ffcb05"/>
    <!-- Beak -->
    <path d="M-16,-10 L-22,-8 L-16,-6 Z" fill="#ff8c21"/>
    <!-- Eye -->
    <circle cx="-12" cy="-12" r="2" fill="#111"/>
    <circle cx="-12.5" cy="-12.5" r="0.7" fill="#fff"/>
    <!-- Wing -->
    <path d="M0,-5 Q8,-8 10,0 Q8,5 0,5" fill="#f0b820" opacity="0.7"/>
    <!-- Speech bubble from rubber duck -->
    <g transform="translate(30, -40)">
      <rect x="0" y="0" width="80" height="24" rx="6" fill="#fff" opacity="0.9"/>
      <polygon points="10,24 18,24 8,34" fill="#fff" opacity="0.9"/>
      <text x="40" y="16" text-anchor="middle" font-family="'Courier New', monospace" font-size="7" fill="#ff5e5e" font-weight="bold">quack?</text>
    </g>
  </g>
  <!-- ===== BUG on screen (literal bug crawling on code) ===== -->
  <g transform="translate(380, 175)" class="bug">
    <!-- Bug body -->
    <ellipse cx="0" cy="0" rx="4" ry="3" fill="#ff3a3a"/>
    <ellipse cx="0" cy="-3" rx="2.5" ry="2" fill="#cc2020"/>
    <!-- Legs -->
    <line x1="-4" y1="-1" x2="-7" y2="-4" stroke="#cc2020" stroke-width="0.7"/>
    <line x1="-4" y1="1" x2="-7" y2="3" stroke="#cc2020" stroke-width="0.7"/>
    <line x1="-4" y1="0" x2="-7" y2="0" stroke="#cc2020" stroke-width="0.7"/>
    <line x1="4" y1="-1" x2="7" y2="-4" stroke="#cc2020" stroke-width="0.7"/>
    <line x1="4" y1="1" x2="7" y2="3" stroke="#cc2020" stroke-width="0.7"/>
    <line x1="4" y1="0" x2="7" y2="0" stroke="#cc2020" stroke-width="0.7"/>
    <!-- Antennae -->
    <line x1="-1" y1="-5" x2="-3" y2="-8" stroke="#cc2020" stroke-width="0.7"/>
    <line x1="1" y1="-5" x2="3" y2="-8" stroke="#cc2020" stroke-width="0.7"/>
    <circle cx="-3" cy="-8" r="0.5" fill="#cc2020"/>
    <circle cx="3" cy="-8" r="0.5" fill="#cc2020"/>
    <!-- Eyes -->
    <circle cx="-1" cy="-3" r="0.6" fill="#fff"/>
    <circle cx="1" cy="-3" r="0.6" fill="#fff"/>
  </g>
  <!-- ===== CLOCK showing 3:00 AM ===== -->
  <g transform="translate(695, 90)">
    <!-- Clock face -->
    <circle cx="0" cy="0" r="35" fill="#1a1a28" stroke="#3a3a50" stroke-width="3"/>
    <circle cx="0" cy="0" r="32" fill="#0d1117" stroke="#2a2a40" stroke-width="1"/>
    <!-- Hour markers -->
    <g stroke="#4a6a8a" stroke-width="1.5">
      <line x1="0" y1="-28" x2="0" y2="-25"/>
      <line x1="14" y1="-24" x2="12.5" y2="-21.5" transform="rotate(0,0,0)"/>
    </g>
    <!-- All 12 markers -->
    <g stroke="#4a6a8a" stroke-width="1">
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(0)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(30)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(60)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(90)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(120)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(150)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(180)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(210)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(240)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(270)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(300)"/>
      <line x1="0" y1="-28" x2="0" y2="-25" transform="rotate(330)"/>
    </g>
    <!-- Numbers -->
    <g font-family="'Courier New', monospace" font-size="7" fill="#5a8aba" text-anchor="middle">
      <text x="0" y="-18">12</text>
      <text x="20" y="-4">3</text>
      <text x="0" y="16">6</text>
      <text x="-20" y="-4">9</text>
    </g>
    <!-- Hour hand (pointing at 3) -->
    <line x1="0" y1="3" x2="18" y2="-1" stroke="#ff9f43" stroke-width="2.5" stroke-linecap="round"/>
    <!-- Minute hand (pointing at 12) -->
    <line x1="0" y1="3" x2="0" y2="-22" stroke="#ddd" stroke-width="1.5" stroke-linecap="round"/>
    <!-- Second hand (ticking) -->
    <line x1="0" y1="5" x2="0" y2="-20" stroke="#ff5e5e" stroke-width="0.7"/>
    <!-- Center dot -->
    <circle cx="0" cy="0" r="2.5" fill="#ff9f43"/>
    <!-- "AM" label -->
    <text x="0" y="10" text-anchor="middle" font-family="'Courier New', monospace" font-size="6" fill="#ff9f43" font-weight="bold" opacity="0.8">AM</text>
  </g>
  <!-- ===== ERROR NOTIFICATION pop-up ===== -->
  <g transform="translate(530, 105)">
    <rect x="0" y="0" width="110" height="50" rx="6" fill="#2a1520" stroke="#ff5e5e" stroke-width="1.5" opacity="0.9"/>
    <!-- Error icon -->
    <circle cx="16" cy="15" r="8" fill="#ff5e5e" opacity="0.2"/>
    <text x="16" y="19" text-anchor="middle" font-family="monospace" font-size="12" fill="#ff5e5e" font-weight="bold">!</text>
    <!-- Error text -->
    <text x="50" y="16" text-anchor="middle" font-family="monospace" font-size="6" fill="#ff8888" font-weight="bold">ERROR</text>
    <text x="55" y="28" text-anchor="middle" font-family="monospace" font-size="5" fill="#ff8888">TypeError:</text>
    <text x="55" y="37" text-anchor="middle" font-family="monospace" font-size="5" fill="#ff6666">bugs.map is not</text>
    <text x="55" y="45" text-anchor="middle" font-family="monospace" font-size="5" fill="#ff6666">a function</text>
    <!-- Flash effect -->
    <rect x="0" y="0" width="110" height="50" rx="6" fill="#ff5e5e" opacity="0.15" class="error-flash"/>
  </g>
  <!-- ===== THOUGHT BUBBLES (duck thinking about solutions) ===== -->
  <g transform="translate(120, 260)" opacity="0.5">
    <circle cx="50" cy="10" r="3" fill="#aaa"/>
    <circle cx="40" cy="5" r="5" fill="#aaa"/>
    <ellipse cx="20" cy="-5" rx="30" ry="14" fill="#aaa" opacity="0.3"/>
    <text x="20" y="-2" text-anchor="middle" font-family="'Courier New', monospace" font-size="6" fill="#fff" opacity="0.8">why tho</text>
  </g>
  <!-- ===== ZZZ floating (but duck is NOT sleeping, fighting it) ===== -->
  <g transform="translate(100, 250)">
    <text x="0" y="0" font-family="'Courier New', monospace" font-size="14" fill="#5a8aba" opacity="0.4" style="animation: floatZ 3s ease-out infinite">z</text>
    <text x="12" y="-10" font-family="'Courier New', monospace" font-size="11" fill="#5a8aba" opacity="0.3" style="animation: floatZ 3s ease-out 0.5s infinite">z</text>
    <text x="20" y="-18" font-family="'Courier New', monospace" font-size="8" fill="#5a8aba" opacity="0.2" style="animation: floatZ 3s ease-out 1s infinite">z</text>
  </g>
  <!-- ===== EMPTY COFFEE CANS on floor ===== -->
  <g transform="translate(660, 530)">
    <!-- Can 1 -->
    <rect x="0" y="0" width="18" height="24" rx="3" fill="#8a5a3a" stroke="#6a4020" stroke-width="1"/>
    <text x="9" y="14" text-anchor="middle" font-family="monospace" font-size="4" fill="#d0a880" font-weight="bold">CAF</text>
    <!-- Can 2 (knocked over) -->
    <g transform="translate(-35, 10) rotate(-25)">
      <rect x="0" y="0" width="18" height="24" rx="3" fill="#7a4a2a" stroke="#5a3018" stroke-width="1"/>
      <text x="9" y="14" text-anchor="middle" font-family="monospace" font-size="4" fill="#c09870" font-weight="bold">CAF</text>
    </g>
    <!-- Can 3 -->
    <rect x="-60" y="5" width="18" height="24" rx="3" fill="#9a6a4a" stroke="#6a4020" stroke-width="1"/>
    <text x="-51" y="19" text-anchor="middle" font-family="monospace" font-size="4" fill="#e0b890" font-weight="bold">CAF</text>
  </g>
  <!-- ===== MOUSE ===== -->
  <g transform="translate(540, 395)">
    <rect x="0" y="0" width="24" height="38" rx="10" fill="#333345" stroke="#4a4a60" stroke-width="1"/>
    <!-- Scroll wheel -->
    <rect x="8" y="6" width="8" height="4" rx="2" fill="#4a4a60"/>
    <!-- Buttons line -->
    <line x1="12" y1="2" x2="12" y2="14" stroke="#4a4a60" stroke-width="0.5"/>
    <!-- Cord -->
    <path d="M12,0 Q12,-15 20,-20" fill="none" stroke="#4a4a60" stroke-width="1.5"/>
  </g>
  <!-- ===== TITLE TEXT ===== -->
  <g transform="translate(400, 470)">
    <text x="0" y="0" text-anchor="middle" font-family="'Courier New', monospace" font-size="18" fill="#ff9f43" font-weight="bold" letter-spacing="3" opacity="0.8">3AM DEBUG CLUB</text>
    <text x="0" y="22" text-anchor="middle" font-family="'Courier New', monospace" font-size="10" fill="#5a8aba" opacity="0.6">it works on my machine...</text>
  </g>
  <!-- ===== SUBTLE GRID PATTERN ON WALL ===== -->
  <g opacity="0.03" stroke="#5a8aba" stroke-width="0.5">
    <line x1="0" y1="50" x2="800" y2="50"/>
    <line x1="0" y1="100" x2="800" y2="100"/>
    <line x1="0" y1="150" x2="800" y2="150"/>
    <line x1="0" y1="200" x2="800" y2="200"/>
    <line x1="0" y1="250" x2="800" y2="250"/>
    <line x1="0" y1="300" x2="800" y2="300"/>
    <line x1="0" y1="350" x2="800" y2="350"/>
    <line x1="0" y1="400" x2="800" y2="400"/>
  </g>
  <!-- ===== AMBIENT LIGHT from screen ===== -->
  <ellipse cx="400" cy="300" rx="250" ry="150" fill="#1a3a5c" opacity="0.04" filter="url(#softGlow)"/>
  <!-- ===== DESK LAMP (off, because it's 3am and only screen light matters) ===== -->
  <g transform="translate(690, 300)">
    <!-- Lamp base -->
    <ellipse cx="0" cy="120" rx="18" ry="5" fill="#2a2a3a"/>
    <!-- Lamp arm -->
    <rect x="-2" y="20" width="4" height="100" fill="#3a3a4a"/>
    <!-- Lamp shade (off) -->
    <path d="M-20,20 L20,20 L12,0 L-12,0 Z" fill="#3a3a4a" stroke="#4a4a5a" stroke-width="1"/>
    <!-- "OFF" indicator -->
    <circle cx="0" cy="25" r="2" fill="#333"/>
  </g>
  <!-- ===== PIZZA BOX (empty, tragic) ===== -->
  <g transform="translate(88, 438)">
    <rect x="0" y="-12" width="50" height="12" rx="1" fill="#c4a060" stroke="#a08040" stroke-width="0.5"/>
    <rect x="1" y="-11" width="48" height="3" fill="#d4b070" opacity="0.5"/>
    <text x="25" y="-2" text-anchor="middle" font-family="monospace" font-size="4" fill="#8a6030" opacity="0.7">EMPTY ;_;</text>
  </g>
</svg>
```

</details>

## 6. glm-5.2_cloud

**SVG size:** 10821 characters  
**Complexity:** Very high  
**Shape elements:** 89  
**Distinct colors:** 27  
**Raw response:** 10878 characters  
**Generation time:** 77.2s

![glm-5.2_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/glm-5.2_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With over 10,821 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering scene props, screen code, lighting, and character expression.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <defs>
    <linearGradient id="wallGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0B0C10"/>
      <stop offset="100%" stop-color="#1F2833"/>
    </linearGradient>
    <radialGradient id="screenGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#3B82F6" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#3B82F6" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="deskGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#3E2723"/>
      <stop offset="100%" stop-color="#1B1110"/>
    </linearGradient>
    <linearGradient id="mugGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#D32F2F"/>
      <stop offset="100%" stop-color="#B71C1C"/>
    </linearGradient>
    <linearGradient id="hoodieGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4B5563"/>
      <stop offset="100%" stop-color="#374151"/>
    </linearGradient>
    <filter id="blur">
      <feGaussianBlur stdDeviation="5"/>
    </filter>
  </defs>
  <!-- Background Wall -->
  <rect width="800" height="600" fill="url(#wallGrad)"/>
  <!-- Window -->
  <rect x="600" y="40" width="130" height="160" fill="#0F172A" stroke="#334155" stroke-width="6" rx="4"/>
  <line x1="665" y1="40" x2="665" y2="200" stroke="#334155" stroke-width="6"/>
  <line x1="600" y1="120" x2="730" y2="120" stroke="#334155" stroke-width="6"/>
  <!-- Moon -->
  <path d="M 685 70 A 30 30 0 1 0 685 130 A 20 20 0 1 1 685 70 Z" fill="#E2E8F0"/>
  <circle cx="640" cy="65" r="1.5" fill="#FFF" opacity="0.8"/>
  <circle cx="620" cy="85" r="1" fill="#FFF" opacity="0.6"/>
  <circle cx="700" cy="160" r="1" fill="#FFF" opacity="0.7"/>
  <!-- Wall Clock -->
  <g transform="translate(100, 100)">
    <circle cx="0" cy="0" r="35" fill="#1F2833" stroke="#66FCF1" stroke-width="3"/>
    <circle cx="0" cy="0" r="2" fill="#66FCF1"/>
    <line x1="0" y1="0" x2="0" y2="-22" stroke="#C5C6C7" stroke-width="3" stroke-linecap="round"/>
    <line x1="0" y1="0" x2="22" y2="0" stroke="#C5C6C7" stroke-width="3" stroke-linecap="round"/>
    <text x="0" y="55" fill="#66FCF1" font-family="monospace" font-size="12" text-anchor="middle" font-weight="bold">3:00 AM</text>
  </g>
  <!-- Monitor Glow -->
  <ellipse cx="400" cy="280" rx="350" ry="250" fill="url(#screenGlow)"/>
  <!-- Monitor & Screen -->
  <g transform="translate(250, 130)">
    <rect x="0" y="0" width="300" height="200" rx="8" fill="#1E1E1E" stroke="#333" stroke-width="4"/>
    <rect x="8" y="8" width="284" height="184" rx="4" fill="#0F172A"/>
    <!-- Code Lines -->
    <g font-family="monospace" font-size="12">
      <text x="20" y="35" fill="#A78BFA">const</text>
      <text x="55" y="35" fill="#E2E8F0">duck</text>
      <text x="85" y="35" fill="#A78BFA">=</text>
      <text x="95" y="35" fill="#FCD34D">require(</text>
      <text x="155" y="35" fill="#84CC16">'./duck'</text>
      <text x="195" y="35" fill="#FCD34D">);</text>
      <text x="20" y="55" fill="#A78BFA">async function</text>
      <text x="110" y="55" fill="#E2E8F0">debug() {</text>
      <text x="40" y="75" fill="#A78BFA">try</text>
      <text x="70" y="75" fill="#E2E8F0">{</text>
      <text x="60" y="95" fill="#A78BFA">await</text>
      <text x="100" y="95" fill="#E2E8F0">duck.fix();</text>
      <text x="40" y="115" fill="#A78BFA">} catch</text>
      <text x="95" y="115" fill="#E2E8F0">(err) {</text>
      <text x="60" y="135" fill="#F87171">console.error(err);</text>
      <text x="60" y="155" fill="#F87171">throw new Error('Bug!');</text>
      <text x="40" y="175" fill="#E2E8F0">}</text>
      <text x="20" y="195" fill="#E2E8F0">}</text>
      <text x="20" y="220" fill="#F87171" font-weight="bold">&gt; Error: Cannot read properties of undefined</text>
      <text x="20" y="240" fill="#F87171" font-weight="bold">&gt; at Object.&lt;anonymous&gt; (app.js:12:19)</text>
      <text x="20" y="260" fill="#FCD34D">&gt; Process exited with code 1</text>
      <rect x="8" y="270" width="284" height="12" fill="#0F172A"/>
      <circle cx="20" cy="276" r="3" fill="#FFF"/>
      <text x="30" y="280" fill="#94A3B8" font-size="10">debugging duck...</text>
    </g>
    <!-- Monitor Stand -->
    <rect x="130" y="200" width="40" height="20" fill="#1E1E1E"/>
    <rect x="100" y="220" width="100" height="10" rx="5" fill="#1E1E1E"/>
  </g>
  <!-- Desk -->
  <rect x="0" y="400" width="800" height="200" fill="url(#deskGrad)"/>
  <rect x="0" y="400" width="800" height="12" fill="#4E342E"/>
  <rect x="0" y="412" width="800" height="4" fill="#3E2723"/>
  <!-- Rubber Duck Debugging (Meta) -->
  <g transform="translate(120, 390)">
    <ellipse cx="0" cy="0" rx="18" ry="12" fill="#FBBF24"/>
    <circle cx="12" cy="-10" r="10" fill="#FBBF24"/>
    <path d="M 20 -12 Q 28 -10, 28 -5 Q 28 0, 20 2 Z" fill="#F97316"/>
    <circle cx="12" cy="-12" r="1.5" fill="#000"/>
    <path d="M 4 -15 L 7 -22 L 10 -16" fill="#FBBF24"/>
  </g>
  <!-- Keyboard -->
  <g transform="translate(250, 390)">
    <rect x="0" y="0" width="300" height="45" rx="5" fill="#1A1A1A"/>
    <g fill="#333">
      <!-- Row 1 -->
      <rect x="10" y="6" width="14" height="10" rx="2"/>
      <rect x="28" y="6" width="14" height="10" rx="2"/>
      <rect x="46" y="6" width="14" height="10" rx="2"/>
      <rect x="64" y="6" width="14" height="10" rx="2"/>
      <rect x="82" y="6" width="14" height="10" rx="2"/>
      <rect x="100" y="6" width="14" height="10" rx="2"/>
      <rect x="118" y="6" width="14" height="10" rx="2"/>
      <rect x="136" y="6" width="14" height="10" rx="2"/>
      <rect x="154" y="6" width="14" height="10" rx="2"/>
      <rect x="172" y="6" width="14" height="10" rx="2"/>
      <rect x="190" y="6" width="14" height="10" rx="2"/>
      <rect x="208" y="6" width="14" height="10" rx="2"/>
      <rect x="226" y="6" width="14" height="10" rx="2"/>
      <rect x="244" y="6" width="46" height="10" rx="2"/>
      <!-- Row 2 -->
      <rect x="10" y="20" width="14" height="10" rx="2"/>
      <rect x="28" y="20" width="14" height="10" rx="2"/>
      <rect x="46" y="20" width="14" height="10" rx="2"/>
      <rect x="64" y="20" width="14" height="10" rx="2"/>
      <rect x="82" y="20" width="14" height="10" rx="2"/>
      <rect x="100" y="20" width="14" height="10" rx="2"/>
      <rect x="118" y="20" width="14" height="10" rx="2"/>
      <rect x="136" y="20" width="14" height="10" rx="2"/>
      <rect x="154" y="20" width="14" height="10" rx="2"/>
      <rect x="172" y="20" width="14" height="10" rx="2"/>
      <rect x="190" y="20" width="14" height="10" rx="2"/>
      <rect x="208" y="20" width="14" height="10" rx="2"/>
      <rect x="226" y="20" width="64" height="10" rx="2"/>
    </g>
    <rect x="190" y="20" width="14" height="10" rx="2" fill="#D32F2F"/> <!-- Panic key -->
  </g>
  <!-- Coffee Mug -->
  <g transform="translate(560, 370)">
    <!-- Steam -->
    <path d="M 15 -10 Q 25 -20, 15 -30 Q 5 -40, 15 -50" fill="none" stroke="#FFF" stroke-width="3" opacity="0.5" stroke-linecap="round"/>
    <path d="M 35 -10 Q 45 -20, 35 -30 Q 25 -40, 35 -50" fill="none" stroke="#FFF" stroke-width="3" opacity="0.3" stroke-linecap="round"/>
    <path d="M 55 -10 Q 65 -20, 55 -30 Q 45 -40, 55 -50" fill="none" stroke="#FFF" stroke-width="3" opacity="0.5" stroke-linecap="round"/>
    <!-- Mug Body -->
    <rect x="0" y="0" width="60" height="40" rx="4" fill="url(#mugGrad)"/>
    <rect x="0" y="0" width="60" height="8" fill="#B71C1C"/>
    <path d="M 60 8 Q 80 8, 80 24 Q 80 36, 60 36" fill="none" stroke="#D32F2F" stroke-width="6"/>
    <!-- Text on Mug -->
    <text x="30" y="28" fill="#FFF" font-family="monospace" font-size="12" text-anchor="middle" font-weight="bold">&lt;/&gt;</text>
  </g>
  <!-- Duck Programmer (Body & Hoodie) -->
  <g>
    <!-- Hoodie Back -->
    <path d="M 330 330 C 280 330, 250 400, 250 450 L 550 450 C 550 400, 520 330, 470 330 Z" fill="url(#hoodieGrad)"/>
    <!-- Body (White Duck) -->
    <path d="M 350 380 Q 350 480, 400 480 Q 450 480, 450 380 Z" fill="#F8F9FA"/>
    <!-- Hood Front -->
    <path d="M 340 320 Q 310 330, 300 350 Q 320 340, 340 340 Z" fill="#374151"/>
    <path d="M 460 320 Q 490 330, 500 350 Q 480 340, 460 340 Z" fill="#374151"/>
    <!-- Strings -->
    <path d="M 360 350 L 360 380" stroke="#1F2937" stroke-width="4" stroke-linecap="round"/>
    <path d="M 440 350 L 440 380" stroke="#1F2937" stroke-width="4" stroke-linecap="round"/>
  </g>
  <!-- Duck Head & Face -->
  <g>
    <!-- Neck -->
    <rect x="375" y="280" width="50" height="50" fill="#F8F9FA"/>
    <!-- Head -->
    <ellipse cx="400" cy="280" rx="55" ry="45" fill="#F8F9FA"/>
    <!-- Messy Feathers (3am Bedhead) -->
    <path d="M 370 240 L 380 210 L 390 235 L 405 215 L 415 240 L 430 225 L 435 250" stroke="#F8F9FA" stroke-width="8" fill="none" stroke-linecap="round"/>
    <!-- Glasses -->
    <circle cx="375" cy="280" r="20" fill="#1E1E1E" opacity="0.2"/>
    <circle cx="425" cy="280" r="20" fill="#1E1E1E" opacity="0.2"/>
    <circle cx="375" cy="280" r="20" fill="none" stroke="#1F2937" stroke-width="4"/>
    <circle cx="425" cy="280" r="20" fill="none" stroke="#1F2937" stroke-width="4"/>
    <line x1="395" y1="280" x2="405" y2="280" stroke="#1F2937" stroke-width="4"/>
    <!-- Tired Eyes -->
    <line x1="360" y1="270" x2="390" y2="270" stroke="#1F2937" stroke-width="4" stroke-linecap="round"/>
    <line x1="410" y1="270" x2="440" y2="270" stroke="#1F2937" stroke-width="4" stroke-linecap="round"/>
    <!-- Eye Bags -->
    <path d="M 365 295 Q 375 305, 385 295" fill="none" stroke="#CBD5E1" stroke-width="3" stroke-linecap="round"/>
    <path d="M 415 295 Q 425 305, 435 295" fill="none" stroke="#CBD5E1" stroke-width="3" stroke-linecap="round"/>
    <!-- Beak -->
    <path d="M 365 295 Q 400 325, 435 295 Q 400 315, 365 295 Z" fill="#F97316"/>
    <path d="M 365 295 Q 400 310, 435 295 L 430 315 Q 400 330, 370 315 Z" fill="#FB923C"/>
  </g>
  <!-- Duck Wing holding Coffee -->
  <path d="M 450 390 Q 530 380, 560 370 Q 565 390, 560 410 Q 520 420, 450 410 Z" fill="#F8F9FA"/>
  <path d="M 490 380 Q 530 375, 560 370" fill="none" stroke="#E2E8F0" stroke-width="4"/>
  <!-- Thought Bubble -->
  <g transform="translate(480, 180)">
    <circle cx="0" cy="0" r="5" fill="#FFF" opacity="0.8"/>
    <circle cx="10" cy="-10" r="8" fill="#FFF" opacity="0.8"/>
    <rect x="20" y="-40" width="140" height="40" rx="10" fill="#FFF" opacity="0.9"/>
    <text x="90" y="-15" fill="#1F2937" font-family="monospace" font-size="14" text-anchor="middle" font-weight="bold">It worked locally...</text>
  </g>
  <!-- Desk Shadows -->
  <ellipse cx="400" cy="405" rx="200" ry="10" fill="#000" opacity="0.4"/>
  <ellipse cx="590" cy="410" rx="50" ry="6" fill="#000" opacity="0.3"/>
</svg>
```

</details>

## 7. gpt-oss_120b-cloud

**SVG size:** 3090 characters  
**Complexity:** Medium  
**Shape elements:** 21  
**Distinct colors:** 15  
**Raw response:** 3951 characters  
**Generation time:** 19.8s

![gpt-oss_120b-cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/gpt-oss_120b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With 3,090 characters of SVG markup, this is a balanced output -- enough detail to convey the scene without being overwhelming.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300" width="100%" height="auto">
  <!-- Night sky background -->
  <rect width="100%" height="100%" fill="#0d0d25"/>
  <!-- Desk -->
  <rect x="30" y="200" width="440" height="80" rx="8" fill="#3c2e2e"/>
  <!-- Monitor base -->
  <rect x="100" y="130" width="120" height="10" fill="#555"/>
  <!-- Monitor screen -->
  <rect x="105" y="70" width="110" height="70" rx="4" fill="#111"/>
  <!-- “code” on screen -->
  <g fill="#85c540" font-size="8" font-family="monospace">
    <text x="112" y="85">def debug():</text>
    <text x="112" y="95">    if bug_found:</text>
    <text x="112" y="105">        raise QuackError()</text>
    <text x="112" y="115">    # try again</text>
  </g>
  <!-- Coffee mug -->
  <g id="coffee-mug" transform="translate(260,190)">
    <rect x="0" y="-30" width="30" height="40" rx="4" fill="#cfa75e"/>
    <path d="M30 -30 a10 10 0 0 1 0 40" fill="none" stroke="#cfa75e" stroke-width="4"/>
    <!-- steam -->
    <path d="M10 -35 q-2 -8 2 -12" stroke="#eee" stroke-width="1.5" fill="none"/>
    <path d="M18 -35 q-2 -8 2 -12" stroke="#eee" stroke-width="1.5" fill="none"/>
  </g>
  <!-- Clock showing 3:00 AM -->
  <g transform="translate(380,40)" fill="#fff">
    <circle cx="0" cy="0" r="30" fill="#222"/>
    <text x="-14" y="6" font-size="12">3:00</text>
    <text x="-26" y="24" font-size="10" fill="#999">AM</text>
    <!-- tiny hour hand -->
    <line x1="0" y1="0" x2="0" y2="-12" stroke="#fff" stroke-width="2"/>
    <!-- tiny minute hand -->
    <line x1="0" y1="0" x2="8" y2="0" stroke="#fff" stroke-width="2"/>
  </g>
  <!-- Duck programmer -->
  <g id="duck" transform="translate(180,200)">
    <!-- Body -->
    <ellipse cx="0" cy="-30" rx="30" ry="40" fill="#ffdb58"/>
    <!-- Head -->
    <circle cx="0" cy="-80" r="20" fill="#ffdb58"/>
    <!-- Beak -->
    <polygon points="-10,-80 -10,-70 5,-75" fill="#ffa500"/>
    <!-- Eyes -->
    <circle cx="-7" cy="-85" r="3" fill="#000"/>
    <circle cx="7" cy="-85" r="3" fill="#000"/>
    <!-- Arms (typing) -->
    <path d="M-20 -45 q-10 -10 -5 -20" stroke="#ffdb58" stroke-width="6" fill="none"/>
    <path d="M20 -45 q10 -10 5 -20" stroke="#ffdb58" stroke-width="6" fill="none"/>
    <!-- Keyboard (simplified) -->
    <rect x="-30" y="-20" width="60" height="10" rx="2" fill="#333"/>
    <g fill="#999" font-size="6" font-family="monospace">
      <text x="-28" y="-13">Esc F1 F2</text>
      <text x="-28" y="-5">Tab Q W E</text>
    </g>
    <!-- Thought bubble with a bug icon -->
    <g transform="translate(-80,-130)">
      <ellipse cx="0" cy="0" rx="45" ry="20" fill="#fff" stroke="#bbb"/>
      <text x="-35" y="5" font-size="9" fill="#555">🐞 BUG?</text>
    </g>
  </g>
  <!-- Soft glow for late‑night work -->
  <defs>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <rect x="0" y="0" width="100%" height="100%" fill="none" filter="url(#glow)"/>
</svg>
```

</details>

## 8. kimi-k2.6_cloud

**SVG size:** 6269 characters  
**Complexity:** High  
**Shape elements:** 31  
**Distinct colors:** 17  
**Raw response:** 12782 characters  
**Generation time:** 39.4s

![kimi-k2.6_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/kimi-k2.6_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With 6,269 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
            <defs>
                <!-- Gradients -->
                <linearGradient id="screenGlow" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#00ff00;stop-opacity:0.2" />
                    <stop offset="100%" style="stop-color:#00ff00;stop-opacity:0" />
                </linearGradient>
                <radialGradient id="lampLight" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" style="stop-color:#ffaa00;stop-opacity:0.4" />
                    <stop offset="100%" style="stop-color:#ffaa00;stop-opacity:0" />
                </radialGradient>
                <linearGradient id="duckBody" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#ffffff" />
                    <stop offset="100%" style="stop-color:#e0e0e0" />
                </linearGradient>
                <!-- Filters -->
                <filter id="blurFilter">
                    <feGaussianBlur in="SourceGraphic" stdDeviation="2" />
                </filter>
            </defs>
            <!-- 1. Background / Room -->
            <rect width="800" height="600" fill="#1a1b26" />
            <!-- Window -->
            <g transform="translate(100, 50)">
                <rect x="0" y="0" width="200" height="250" fill="#0b0c15" stroke="#333" stroke-width="4"/>
                <!-- Moon -->
                <circle cx="160" cy="50" r="20" fill="#ffffe0" opacity="0.8" />
                <!-- Stars (animated via JS) -->
                <g id="stars"></g>
            </g>
            <!-- 2. Furniture -->
            <!-- Desk -->
            <path d="M0,450 L800,450 L800,600 L0,600 Z" fill="#2e2f3e" />
            <rect x="0" y="450" width="800" height="10" fill="#1f2029" />
            <!-- 3. The Duck (Programmer) -->
            <g id="duck" transform="translate(450, 280)">
                <!-- Chair Back -->
                <path d="M-60,170 L-60,50 Q-60,0 -10,0 L10,0 Q60,0 60,50 L60,170 Z" fill="#111" />
                <!-- Body -->
                <ellipse cx="0" cy="120" rx="70" ry="80" fill="url(#duckBody)" />
                <!-- Head -->
                <circle cx="0" cy="40" r="55" fill="url(#duckBody)" />
                <!-- Headphones -->
                <path d="M-55,40 Q-55,-30 0,-30 Q55,-30 55,40" fill="none" stroke="#333" stroke-width="8" />
                <rect x="-65" y="25" width="15" height="30" rx="5" fill="#222" />
                <rect x="50" y="25" width="15" height="30" rx="5" fill="#222" />
                <!-- Beak -->
                <path d="M-15,45 Q0,35 15,45 L25,55 Q0,65 -25,55 Z" fill="#ff8906" />
                <!-- Eyes -->
                <g id="eyes">
                    <!-- Left Eye -->
                    <circle cx="-20" cy="25" r="8" fill="#000" />
                    <circle cx="-18" cy="23" r="2" fill="#fff" />
                    <!-- Right Eye (Tired/Baggy) -->
                    <circle cx="20" cy="25" r="8" fill="#000" />
                    <circle cx="22" cy="23" r="2" fill="#fff" />
                    <!-- Tired lines -->
                    <path d="M10,35 Q20,40 30,35" stroke="#c0392b" stroke-width="1" fill="none" opacity="0.6"/>
                </g>
                <!-- Sweat Drop -->
                <path id="sweat" d="M50,10 Q55,0 50,-10 Q45,0 50,10" fill="#a7a9be" opacity="0" />
                <!-- Wing (resting on keyboard) -->
                <path d="M-40,90 Q-70,130 -30,140 L20,130" fill="#e0e0e0" stroke="#ccc" stroke-width="1" />
            </g>
            <!-- 4. Computer Setup -->
            <g transform="translate(250, 250)">
                <!-- Monitor Stand -->
                <rect x="80" y="150" width="40" height="40" fill="#111" />
                <path d="M60,190 L140,190 L150,200 L50,200 Z" fill="#222" />
                <!-- Monitor -->
                <rect x="0" y="0" width="200" height="150" rx="5" fill="#000" stroke="#333" stroke-width="2" />
                <!-- Screen Content (Code) -->
                <foreignObject x="5" y="5" width="190" height="140">
                    <div xmlns="http://www.w3.org/1999/xhtml" 
                         style="width:100%; height:100%; background:#1e1e1e; color:#4ec9b0; font-family:monospace; font-size:10px; padding:5px; overflow:hidden; word-wrap:break-word; line-height:12px;">
                        <span style="color:#569cd6;">if</span>(duck.quack()) {<br/>
                        &nbsp;&nbsp;debug();<br/>
                        } <span style="color:#569cd6;">else</span> {<br/>
                        &nbsp;&nbsp;<span style="color:#ce9178;">"need coffee"</span>;<br/>
                        }<br/>
                        <span style="color:#dcdcaa;">console</span>.log(<span id="error-text" style="color:#f44747;">undefined</span>);
                    </div>
                </foreignObject>
            </g>
            <!-- 5. Coffee Mug -->
            <g transform="translate(500, 420)">
                <!-- Steam (Animated) -->
                <path id="steam1" d="M10,-20 Q20,-40 10,-60" stroke="#fff" stroke-width="2" fill="none" opacity="0.5" />
                <path id="steam2" d="M30,-20 Q20,-40 30,-60" stroke="#fff" stroke-width="2" fill="none" opacity="0.5" />
                <!-- Mug Body -->
                <rect x="0" y="0" width="40" height="50" rx="2" fill="#e0e0e0" />
                <!-- Mug Handle -->
                <path d="M40,10 Q55,10 55,25 Q55,40 40,40" fill="none" stroke="#e0e0e0" stroke-width="4" />
                <!-- Coffee liquid -->
                <rect x="2" y="5" width="36" height="5" fill="#3c2f2f" />
            </g>
            <!-- Lighting Overlays -->
            <!-- Monitor Glow on Duck -->
            <path d="M250,400 L450,400 L450,280 L250,280 Z" fill="url(#screenGlow)" style="mix-blend-mode: screen;" />
            <!-- Desk Lamp Light -->
            <circle cx="550" cy="300" r="150" fill="url(#lampLight)" style="pointer-events:none;" />
            <!-- Lamp Stand (Back) -->
            <path d="M550,450 L550,350" stroke="#444" stroke-width="4" />
            <circle cx="550" cy="350" r="10" fill="#444" />
        </svg>
```

</details>

## 9. minimax-m2.7_cloud

**SVG size:** 5324 characters  
**Complexity:** High  
**Shape elements:** 43  
**Distinct colors:** 25  
**Raw response:** 5721 characters  
**Generation time:** 90.2s

![minimax-m2.7_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/minimax-m2.7_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With 5,324 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" width="100%" height="auto">
  <defs>
    <!-- Night Sky Gradient -->
    <linearGradient id="nightSky" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#0f0c29;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#302b63;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#24243e;stop-opacity:1" />
    </linearGradient>
    <!-- Steam Animation -->
    <style>
      .steam {
        animation: rise 2s infinite ease-out;
        opacity: 0;
      }
      @keyframes rise {
        0% { transform: translateY(0); opacity: 0.8; }
        50% { opacity: 0.4; }
        100% { transform: translateY(-20px); opacity: 0; }
      }
      .steam:nth-child(2) { animation-delay: 0.5s; }
      .steam:nth-child(3) { animation-delay: 1s; }
    </style>
  </defs>
  <!-- Background -->
  <rect width="100%" height="100%" fill="url(#nightSky)" />
  <!-- Moon -->
  <circle cx="520" cy="60" r="40" fill="#fdfd96" opacity="0.9" />
  <circle cx="530" cy="50" r="35" fill="url(#nightSky)" opacity="0.2" />
  <!-- Clock showing 3:00 -->
  <g transform="translate(80, 80)">
    <circle cx="0" cy="0" r="30" fill="#fff" opacity="0.2" stroke="#fff" stroke-width="2"/>
    <line x1="0" y1="0" x2="0" y2="-20" stroke="#fff" stroke-width="3" stroke-linecap="round"/> <!-- Hour hand -->
    <line x1="0" y1="0" x2="15" y2="0" stroke="#fff" stroke-width="2" stroke-linecap="round"/> <!-- Minute hand -->
    <text x="0" y="45" font-family="monospace" fill="#ff6b6b" font-size="14" text-anchor="middle" font-weight="bold">3:00 AM</text>
  </g>
  <!-- Desk -->
  <rect x="50" y="300" width="500" height="10" fill="#5c4033" />
  <rect x="70" y="310" width="20" height="90" fill="#4a3728" />
  <rect x="510" y="310" width="20" height="90" fill="#4a3728" />
  <!-- Monitor -->
  <g transform="translate(150, 120)">
    <!-- Stand -->
    <rect x="80" y="150" width="40" height="30" fill="#333" />
    <rect x="50" y="180" width="100" height="10" rx="2" fill="#222" />
    <!-- Screen Frame -->
    <rect x="0" y="0" width="300" height="180" rx="10" fill="#222" />
    <!-- Screen -->
    <rect x="10" y="10" width="280" height="160" fill="#1e1e1e" />
    <!-- Code Lines -->
    <rect x="20" y="30" width="50" height="4" fill="#569cd6" /> <!-- blue func -->
    <rect x="75" y="30" width="40" height="4" fill="#dcdcaa" /> <!-- yellow var -->
    <rect x="120" y="30" width="30" height="4" fill="#c586c0" /> <!-- purple keyword -->
    <rect x="20" y="45" width="80" height="4" fill="#608b4e" /> <!-- green comment -->
    <rect x="20" y="60" width="100" height="4" fill="#9cdcfe" /> <!-- light blue string -->
    <!-- Error Line (The Bug Context) -->
    <rect x="20" y="90" width="60" height="4" fill="#ce9178" /> <!-- orange -->
    <rect x="90" y="90" width="20" height="4" fill="#d4d4d4" />
    <!-- The Bug Icon on Screen -->
    <g transform="translate(230, 80)">
       <circle cx="0" cy="0" r="15" fill="#f44747" opacity="0.8" />
       <path d="M-5,-5 L5,5 M5,-5 L-5,5" stroke="white" stroke-width="3" />
       <text x="0" y="4" fill="white" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">!</text>
    </g>
    <!-- Magnifying Glass -->
    <g transform="translate(220, 100) rotate(-20)">
      <circle cx="0" cy="0" r="20" stroke="#61afef" stroke-width="4" fill="none" opacity="0.5"/>
      <line x1="14" y1="14" x2="30" y2="30" stroke="#61afef" stroke-width="4" stroke-linecap="round"/>
    </g>
  </g>
  <!-- Coffee Mug -->
  <g transform="translate(480, 240)">
    <path d="M10,0 L60,0 L55,40 Q35,45 15,40 Z" fill="#e0e0e0" stroke="#999" stroke-width="1"/>
    <path d="M15,5 L55,5" stroke="#ccc" stroke-width="1"/>
    <!-- Coffee Liquid -->
    <ellipse cx="35" cy="5" rx="25" ry="5" fill="#6f4e37" />
    <!-- Handle -->
    <path d="M55,10 Q75,10 75,25 Q75,40 55,40" stroke="#e0e0e0" stroke-width="5" fill="none" />
    <!-- Steam -->
    <g fill="none" stroke="white" stroke-width="2" stroke-linecap="round" opacity="0.6">
      <path class="steam" d="M25,-5 Q30,-15 25,-25" />
      <path class="steam" d="M35,-5 Q40,-15 35,-25" />
      <path class="steam" d="M45,-5 Q50,-15 45,-25" />
    </g>
  </g>
  <!-- The Duck Programmer -->
  <g transform="translate(80, 180)">
    <!-- Tail feathers -->
    <path d="M-20,110 Q-40,110 -45,90 L-20,95 Z" fill="#e6c200" />
    <!-- Body -->
    <ellipse cx="20" cy="100" rx="50" ry="45" fill="#ffdd00" />
    <!-- Wing (typing/holding) -->
    <ellipse cx="50" cy="110" rx="25" ry="15" fill="#e6c200" transform="rotate(-10 50 110)" />
    <!-- Head -->
    <circle cx="30" cy="50" r="35" fill="#ffdd00" />
    <!-- Beak -->
    <path d="M45,50 L75,55 L45,65 Z" fill="#ff9900" />
    <!-- Glasses -->
    <g stroke="#333" stroke-width="2" fill="rgba(255,255,255,0.3)">
      <circle cx="25" cy="45" r="12" />
      <line x1="37" y1="45" x2="55" y2="45" /> <!-- Bridge -->
      <circle cx="65" cy="45" r="12" />
      <line x1="77" y1="45" x2="90" y2="40" /> <!-- Arm to head -->
    </g>
    <!-- Eyes (looking at screen) -->
    <circle cx="25" cy="45" r="3" fill="black" />
    <circle cx="65" cy="45" r="3" fill="black" />
    <!-- Sweat drop (stress) -->
    <path d="M80,30 Q85,20 80,15 Q75,20 80,30" fill="#4fc3f7" opacity="0.8" />
  </g>
</svg>
```

</details>

## 10. minimax-m3_cloud

**SVG size:** 11304 characters  
**Complexity:** Very high  
**Shape elements:** 82  
**Distinct colors:** 41  
**Raw response:** 12708 characters  
**Generation time:** 90.1s

![minimax-m3_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/minimax-m3_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Includes animation elements -- the model attempted motion (e.g., steam, blinking)
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With over 11,304 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering scene props, screen code, lighting, and character expression.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 500" width="100%" height="auto">
  <defs>
    <linearGradient id="nightSky" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#0f0f23"/>
      <stop offset="100%" style="stop-color:#1a1a3e"/>
    </linearGradient>
    <radialGradient id="monitorGlow" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#4a90e2;stop-opacity:0.3"/>
      <stop offset="100%" style="stop-color:#4a90e2;stop-opacity:0"/>
    </radialGradient>
    <linearGradient id="desk" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#8b4513"/>
      <stop offset="100%" style="stop-color:#5d3a1f"/>
    </linearGradient>
  </defs>
  <!-- Night sky background -->
  <rect width="600" height="500" fill="url(#nightSky)"/>
  <!-- Stars -->
  <circle cx="50" cy="40" r="1" fill="white"/>
  <circle cx="120" cy="80" r="1.5" fill="white"/>
  <circle cx="200" cy="30" r="1" fill="white"/>
  <circle cx="500" cy="50" r="1.5" fill="white"/>
  <circle cx="550" cy="100" r="1" fill="white"/>
  <circle cx="450" cy="20" r="1" fill="white"/>
  <circle cx="80" cy="120" r="1" fill="white"/>
  <circle cx="380" cy="60" r="1" fill="white"/>
  <circle cx="280" cy="100" r="1" fill="white"/>
  <!-- Moon -->
  <circle cx="520" cy="70" r="22" fill="#f0e68c" opacity="0.85"/>
  <circle cx="525" cy="65" r="18" fill="#0f0f23"/>
  <circle cx="515" cy="75" r="3" fill="#d4cf6e" opacity="0.6"/>
  <circle cx="513" cy="62" r="2" fill="#d4cf6e" opacity="0.6"/>
  <!-- Monitor glow -->
  <ellipse cx="300" cy="290" rx="220" ry="120" fill="url(#monitorGlow)"/>
  <!-- Desk -->
  <rect x="40" y="380" width="520" height="15" fill="url(#desk)"/>
  <rect x="40" y="395" width="520" height="105" fill="#3d2817"/>
  <rect x="60" y="395" width="12" height="105" fill="#4a3220"/>
  <rect x="528" y="395" width="12" height="105" fill="#4a3220"/>
  <!-- Floor -->
  <rect x="0" y="480" width="600" height="20" fill="#1a1a2e"/>
  <!-- Laptop -->
  <rect x="220" y="290" width="180" height="95" rx="3" fill="#1a1a1a" stroke="#333" stroke-width="1"/>
  <rect x="225" y="295" width="170" height="80" fill="#0d1117"/>
  <!-- Screen bezel -->
  <rect x="225" y="295" width="170" height="80" fill="none" stroke="#222" stroke-width="1"/>
  <!-- Code lines on screen -->
  <text x="232" y="306" font-family="monospace" font-size="6" fill="#c9d1d9">function</text>
  <text x="260" y="306" font-family="monospace" font-size="6" fill="#79c0ff">debug</text>
  <text x="280" y="306" font-family="monospace" font-size="6" fill="#c9d1d9">() {</text>
  <text x="237" y="316" font-family="monospace" font-size="6" fill="#ff7b72">  if</text>
  <text x="248" y="316" font-family="monospace" font-size="6" fill="#c9d1d9">(error) {</text>
  <text x="242" y="326" font-family="monospace" font-size="6" fill="#79c0ff">    fix</text>
  <text x="255" y="326" font-family="monospace" font-size="6" fill="#c9d1d9">(error);</text>
  <text x="237" y="336" font-family="monospace" font-size="6" fill="#ff7b72">  }</text>
  <text x="232" y="346" font-family="monospace" font-size="6" fill="#8b949e">  // TODO</text>
  <text x="232" y="356" font-family="monospace" font-size="6" fill="#ff7b72">  return</text>
  <text x="258" y="356" font-family="monospace" font-size="6" fill="#ffa657">success</text>
  <text x="232" y="366" font-family="monospace" font-size="6" fill="#c9d1d9">}</text>
  <!-- Error highlight -->
  <rect x="252" y="321" width="35" height="6" fill="#ff5555" opacity="0.25"/>
  <line x1="287" y1="324" x2="297" y2="335" stroke="#ff5555" stroke-width="0.5"/>
  <!-- Cursor -->
  <rect x="293" y="320" width="1.5" height="6" fill="#c9d1d9">
    <animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/>
  </rect>
  <!-- Laptop base -->
  <rect x="210" y="383" width="200" height="6" fill="#0a0a0a"/>
  <rect x="275" y="383" width="70" height="3" fill="#2a2a2a"/>
  <!-- Duck body -->
  <ellipse cx="300" cy="370" rx="55" ry="32" fill="#ffd700"/>
  <ellipse cx="300" cy="370" rx="55" ry="32" fill="url(#bodyShade)" opacity="0.3"/>
  <!-- Duck chest -->
  <ellipse cx="300" cy="370" rx="38" ry="25" fill="#fff8dc"/>
  <!-- Duck wings -->
  <ellipse cx="252" cy="365" rx="14" ry="22" fill="#ffb700" transform="rotate(-25 252 365)"/>
  <ellipse cx="348" cy="365" rx="14" ry="22" fill="#ffb700" transform="rotate(25 348 365)"/>
  <!-- Wings hovering over keyboard -->
  <ellipse cx="270" cy="355" rx="10" ry="15" fill="#ffc700" transform="rotate(-30 270 355)"/>
  <ellipse cx="330" cy="355" rx="10" ry="15" fill="#ffc700" transform="rotate(30 330 355)"/>
  <!-- Duck head -->
  <circle cx="300" cy="320" r="32" fill="#ffd700"/>
  <!-- Duck hair tuft -->
  <path d="M 290 295 Q 295 285 300 290 Q 305 285 310 295" fill="#ffb700" stroke="#e0a100" stroke-width="0.5"/>
  <!-- Duck beak -->
  <ellipse cx="305" cy="342" rx="16" ry="9" fill="#ff8c00"/>
  <line x1="289" y1="342" x2="321" y2="342" stroke="#d97700" stroke-width="1"/>
  <ellipse cx="305" cy="345" rx="14" ry="5" fill="#ff9d2e"/>
  <!-- Glasses -->
  <circle cx="288" cy="318" r="9" fill="rgba(200,220,255,0.2)" stroke="#222" stroke-width="2"/>
  <circle cx="312" cy="318" r="9" fill="rgba(200,220,255,0.2)" stroke="#222" stroke-width="2"/>
  <line x1="297" y1="318" x2="303" y2="318" stroke="#222" stroke-width="2"/>
  <path d="M 277 316 Q 273 313 269 315" fill="none" stroke="#222" stroke-width="1.5"/>
  <path d="M 323 316 Q 327 313 331 315" fill="none" stroke="#222" stroke-width="1.5"/>
  <!-- Tired eyes -->
  <circle cx="288" cy="318" r="3.5" fill="white"/>
  <circle cx="312" cy="318" r="3.5" fill="white"/>
  <circle cx="290" cy="319" r="2.5" fill="#1a1a1a"/>
  <circle cx="314" cy="319" r="2.5" fill="#1a1a1a"/>
  <!-- Bags under eyes -->
  <path d="M 282 324 Q 288 326 294 324" fill="none" stroke="#cc8800" stroke-width="0.8" opacity="0.6"/>
  <path d="M 306 324 Q 312 326 318 324" fill="none" stroke="#cc8800" stroke-width="0.8" opacity="0.6"/>
  <!-- Eyebrows -->
  <path d="M 280 308 Q 288 305 295 308" fill="none" stroke="#cc8800" stroke-width="1.5"/>
  <path d="M 305 308 Q 312 305 320 308" fill="none" stroke="#cc8800" stroke-width="1.5"/>
  <!-- Coffee mug -->
  <rect x="115" y="335" width="50" height="45" fill="#fff" stroke="#222" stroke-width="2"/>
  <rect x="115" y="378" width="50" height="6" fill="#222"/>
  <path d="M 165 350 Q 185 350 185 365 Q 185 380 165 380" fill="none" stroke="#222" stroke-width="2"/>
  <ellipse cx="140" cy="340" rx="20" ry="4" fill="#3d2817"/>
  <ellipse cx="140" cy="340" rx="18" ry="3" fill="#5d3a1f"/>
  <!-- Coffee text -->
  <text x="120" y="360" font-family="Arial" font-size="9" font-weight="bold" fill="#ff5252">CODE</text>
  <!-- Steam -->
  <path d="M 125 332 Q 122 322 128 315 Q 124 305 130 295" fill="none" stroke="#aabbcc" stroke-width="1.5" opacity="0.5">
    <animate attributeName="opacity" values="0.3;0.7;0.3" dur="2s" repeatCount="indefinite"/>
  </path>
  <path d="M 140 332 Q 137 322 143 315 Q 139 305 145 295" fill="none" stroke="#aabbcc" stroke-width="1.5" opacity="0.5">
    <animate attributeName="opacity" values="0.5;0.3;0.5" dur="2.5s" repeatCount="indefinite"/>
  </path>
  <path d="M 155 332 Q 152 322 158 315 Q 154 305 160 295" fill="none" stroke="#aabbcc" stroke-width="1.5" opacity="0.5">
    <animate attributeName="opacity" values="0.4;0.6;0.4" dur="1.8s" repeatCount="indefinite"/>
  </path>
  <!-- Clock on wall -->
  <rect x="440" y="80" width="110" height="90" rx="8" fill="#1a1a1a" stroke="#444" stroke-width="2"/>
  <rect x="448" y="88" width="94" height="55" fill="#000"/>
  <text x="455" y="115" font-family="monospace" font-size="24" font-weight="bold" fill="#ff3333">3:00</text>
  <text x="510" y="115" font-family="monospace" font-size="12" fill="#ff3333">AM</text>
  <rect x="448" y="148" width="94" height="18" fill="#0a0a0a"/>
  <text x="455" y="160" font-family="monospace" font-size="7" fill="#ff3333">STATUS: DEBUGGING</text>
  <!-- Clock LEDs -->
  <circle cx="538" cy="93" r="2" fill="#00ff00">
    <animate attributeName="opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite"/>
  </circle>
  <!-- Floating code symbols -->
  <text x="50" y="240" font-family="monospace" font-size="20" fill="#ff5555" opacity="0.6" font-weight="bold">{ }</text>
  <text x="490" y="220" font-family="monospace" font-size="18" fill="#79c0ff" opacity="0.6" font-weight="bold">&lt;/&gt;</text>
  <text x="80" y="320" font-family="monospace" font-size="16" fill="#ffa657" opacity="0.6">;));</text>
  <text x="460" y="350" font-family="monospace" font-size="14" fill="#c9d1d9" opacity="0.6">// FIXME</text>
  <text x="120" y="180" font-family="monospace" font-size="14" fill="#a371f7" opacity="0.6">[0x1A]</text>
  <text x="430" y="160" font-family="monospace" font-size="12" fill="#ffa657" opacity="0.6">!error</text>
  <!-- Sleepy z's -->
  <text x="350" y="290" font-family="Arial" font-size="16" font-weight="bold" fill="#fff" opacity="0.4">z</text>
  <text x="365" y="270" font-family="Arial" font-size="20" font-weight="bold" fill="#fff" opacity="0.5">Z</text>
  <text x="385" y="245" font-family="Arial" font-size="14" font-weight="bold" fill="#fff" opacity="0.3">z</text>
  <!-- Bug icon (debugging symbol) -->
  <g transform="translate(450, 280)">
    <ellipse cx="0" cy="0" rx="7" ry="9" fill="#ff5555"/>
    <circle cx="-2" cy="-2" r="1" fill="#330000"/>
    <circle cx="2" cy="-2" r="1" fill="#330000"/>
    <line x1="-7" y1="-3" x2="-11" y2="-7" stroke="#ff5555" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="-7" y1="0" x2="-12" y2="0" stroke="#ff5555" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="-7" y1="3" x2="-11" y2="7" stroke="#ff5555" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="7" y1="-3" x2="11" y2="-7" stroke="#ff5555" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="7" y1="0" x2="12" y2="0" stroke="#ff5555" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="7" y1="3" x2="11" y2="7" stroke="#ff5555" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="0" y1="-9" x2="-3" y2="-14" stroke="#ff5555" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="0" y1="-9" x2="3" y2="-14" stroke="#ff5555" stroke-width="1.5" stroke-linecap="round"/>
  </g>
  <!-- Keyboard hints -->
  <rect x="105" y="395" width="80" height="3" fill="#444" opacity="0.4"/>
  <!-- Phone/notebook -->
  <rect x="450" y="345" width="60" height="35" rx="2" fill="#1a1a1a" stroke="#333" stroke-width="1"/>
  <rect x="455" y="350" width="50" height="20" fill="#0a0a0a"/>
  <text x="460" y="358" font-family="monospace" font-size="4" fill="#00ff66">TODO:</text>
  <text x="460" y="364" font-family="monospace" font-size="4" fill="#888">- fix bug</text>
  <text x="460" y="370" font-family="monospace" font-size="4" fill="#888">- sleep</text>
  <!-- Subtle light glow from screen -->
  <ellipse cx="300" cy="350" rx="100" ry="20" fill="#4a90e2" opacity="0.1"/>
  <!-- Title text -->
  <text x="300" y="470" font-family="'Courier New', monospace" font-size="14" font-weight="bold" fill="#5d8aa8" text-anchor="middle" opacity="0.8">// 3:00 AM - still debugging</text>
  <!-- Small decorative dots -->
  <circle cx="200" cy="450" r="2" fill="#ff5555" opacity="0.4"/>
  <circle cx="400" cy="450" r="2" fill="#79c0ff" opacity="0.4"/>
</svg>
```

</details>

## 11. nemotron-3-super_cloud

**SVG size:** 3664 characters  
**Complexity:** High  
**Shape elements:** 32  
**Distinct colors:** 11  
**Raw response:** 4201 characters  
**Generation time:** 15.3s

![nemotron-3-super_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/nemotron-3-super_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With 3,664 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400" width="100%" height="auto">
  <!-- Desk -->
  <rect x="0" y="300" width="500" height="100" fill="#1b263b"/>
  <!-- Laptop base -->
  <rect x="150" y="260" width="200" height="20" rx="4" ry="4" fill="#212f3d"/>
  <!-- Laptop screen (dark with code) -->
  <rect x="150" y="120" width="200" height="140" rx="8" ry="8" fill="#0a1128"/>
  <!-- Code lines -->
  <g fill="#e0e0e0" font-family="monospace" font-size="14">
    <text x="160" y="150">// TODO: Fix the bug</text>
    <text x="160" y="170">if (duck.isSleepy()) {</text>
    <text x="160" y="190">  drinkCoffee();</text>
    <text x="160" y="210">}</text>
    <text x="160" y="230">console.log('Quack!');</text>
  </g>
  <!-- Coffee mug -->
  <g transform="translate(350,260)">
    <!-- Mug body -->
    <path d="M0 0
             h30
             a15 15 0 0 1 15 15
             v40
             a15 15 0 0 1 -15 15
             h-30
             a15 15 0 0 1 -15 -15
             v-40
             a15 15 0 0 1 15 -15
             Z"
          fill="#8b5e3c"/>
    <!-- Mug handle -->
    <path d="M45 20
             a10 10 0 0 1 0 20
             h-10
             a10 10 0 0 0 0 -20
             Z"
          fill="#6d4c41"/>
    <!-- Steam -->
    <g fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round">
      <path d="M15 0 q-5 -10 -10 -20"/>
      <path d="M20 0 q0 -12 -8 -22"/>
      <path d="M25 0 q5 -8 10 -18"/>
    </g>
  </g>
  <!-- Duck programmer -->
  <g transform="translate(120,260)">
    <!-- Body -->
    <ellipse cx="0" cy="0" rx="40" ry="30" fill="#f4a261"/>
    <!-- Wing (left) -->
    <path d="M-40 0
             c-10 -15, -30 -5, -30 15
             s10 20, 30 5"
          fill="#e76f51"/>
    <!-- Wing (right) -->
    <path d="M40 0
             c10 -15, 30 -5, 30 15
             s-10 20, -30 5"
          fill="#e76f51"/>
    <!-- Head -->
    <circle cx="0" cy="-35" r="20" fill="#f4a261"/>
    <!-- Beak -->
    <polygon points="-5,-30 5,-30 0,-20" fill="#e9c46a"/>
    <!-- Eyes -->
    <circle cx="-8" cy="-40" r="3" fill="#fff"/>
    <circle cx="8" cy="-40" r="3" fill="#fff"/>
    <circle cx="-8" cy="-40" r="1" fill="#000"/>
    <circle cx="8" cy="-40" r="1" fill="#000"/>
    <!-- Glasses -->
    <ellipse cx="-12" cy="-40" rx="7" ry="5" fill="none" stroke="#fff" stroke-width="2"/>
    <ellipse cx="12" cy="-40" rx="7" ry="5" fill="none" stroke="#fff" stroke-width="2"/>
    <line x1="-19" y1="-40" x2="19" y2="-40" stroke="#fff" stroke-width="2"/>
  </g>
  <!-- Clock showing 3am -->
  <g transform="translate(80,80)">
    <circle cx="0" cy="0" r="30" fill="#1b263b" stroke="#fff" stroke-width="2"/>
    <!-- Ticks -->
    <g stroke="#fff" stroke-width="2">
      <line x1="0" y1="-25" x2="0" y2="-30"/>
      <line x1="21.65" y1="-12.5" x2="25" y2="-15"/>
      <line x1="25" y1="0" x2="30" y2="0"/>
      <line x1="21.65" y1="12.5" x2="25" y2="15"/>
      <line x1="0" y1="25" x2="0" y2="30"/>
      <line x1="-21.65" y1="12.5" x2="-25" y2="15"/>
      <line x1="-30" y1="0" x2="-25" y2="0"/>
      <line x1="-21.65" y1="-12.5" x2="-25" y2="-15"/>
    </g>
    <!-- Hands (3 o'clock) -->
    <line x1="0" y1="0" x2="0" y2="-20" stroke="#fff" stroke-width="3"/>
    <line x1="0" y1="0" x2="15" y2="0" stroke="#fff" stroke-width="3"/>
    <!-- Center dot -->
    <circle cx="0" cy="0" r="3" fill="#fff"/>
    <!-- Label -->
    <text x="0" y="40" text-anchor="middle" fill="#fff" font-size="14">3 am</text>
  </g>
  <!-- Optional title -->
  <text x="250" y="30" text-anchor="middle" fill="#fff" font-size="20" font-family="sans-serif">
    Duck Programmer Debugging at 3 am
  </text>
</svg>
```

</details>

## 12. nemotron-3-ultra_cloud

**SVG size:** 19457 characters  
**Complexity:** Very high  
**Shape elements:** 93  
**Distinct colors:** 39  
**Raw response:** 21303 characters  
**Generation time:** 77.8s

![nemotron-3-ultra_cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/nemotron-3-ultra_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating UI elements
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Includes animation elements -- the model attempted motion (e.g., steam, blinking)
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With over 19,457 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in rendering scene props, screen code, lighting, and character expression.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <!-- Definitions: Gradients, Filters, Patterns -->
  <defs>
    <!-- Night Sky Gradient -->
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#0d0d1a;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1a1a2e;stop-opacity:1" />
    </linearGradient>
    <!-- Monitor Glow Gradient -->
    <radialGradient id="monitorGlow" cx="50%" cy="30%" r="60%" fx="50%" fy="30%">
      <stop offset="0%" style="stop-color:#1e2d3d;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0f1a24;stop-opacity:1" />
    </radialGradient>
    <!-- Screen Content Gradient -->
    <linearGradient id="screenBg" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#0d1117;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#161b22;stop-opacity:1" />
    </linearGradient>
    <!-- Duck Body Gradient -->
    <linearGradient id="duckBody" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FFD700;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#F5C000;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#E6B800;stop-opacity:1" />
    </linearGradient>
    <!-- Duck Head Gradient (slightly lighter) -->
    <linearGradient id="duckHead" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FFE066;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFD700;stop-opacity:1" />
    </linearGradient>
    <!-- Orange Beak/Feet Gradient -->
    <linearGradient id="orangeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FF9F1C;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#E87C00;stop-opacity:1" />
    </linearGradient>
    <!-- Coffee Brown Gradient -->
    <linearGradient id="coffeeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#4B2E1E;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#3D2314;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2E1A0F;stop-opacity:1" />
    </linearGradient>
    <!-- Cup Gradient -->
    <linearGradient id="cupGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#ffffff;stop-opacity:0.9" />
      <stop offset="50%" style="stop-color:#f0f0f0;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#dcdcdc;stop-opacity:1" />
    </linearGradient>
    <!-- Keyboard Key Gradient -->
    <linearGradient id="keyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#2d2d2d;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1a1a1a;stop-opacity:1" />
    </linearGradient>
    <!-- Glow Filter for Screen Text -->
    <filter id="textGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="1.5" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <!-- Soft Shadow Filter -->
    <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur"/>
      <feOffset dx="2" dy="4" result="offsetBlur"/>
      <feFlood flood-color="#000000" flood-opacity="0.4" result="flood"/>
      <feComposite in="flood" in2="offsetBlur" operator="in" result="shadow"/>
      <feMerge>
        <feMergeNode in="shadow"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <!-- Steam Path Definition -->
    <path id="steamPath" d="M0,0 Q-5,-10 0,-20 Q5,-30 0,-40" fill="none" stroke="rgba(255,255,255,0.4)" stroke-width="2" stroke-linecap="round" opacity="0">
      <animate attributeName="opacity" values="0;0.6;0" dur="2s" repeatCount="indefinite" begin="0s"/>
      <animateTransform attributeName="transform" type="translate" values="0,0; 0,-15; 0,0" dur="2s" repeatCount="indefinite" begin="0s"/>
    </path>
  </defs>
  <!-- Background -->
  <rect width="800" height="600" fill="url(#skyGradient)"/>
  <!-- Stars -->
  <g fill="#ffffff" opacity="0.7">
    <circle cx="50" cy="80" r="1.2"><animate attributeName="opacity" values="0.3;1;0.3" dur="3s" repeatCount="indefinite"/></circle>
    <circle cx="720" cy="40" r="0.8"><animate attributeName="opacity" values="0.3;1;0.3" dur="2.5s" repeatCount="indefinite"/></circle>
    <circle cx="150" cy="150" r="1"><animate attributeName="opacity" values="0.3;1;0.3" dur="4s" repeatCount="indefinite"/></circle>
    <circle cx="650" cy="120" r="0.6"><animate attributeName="opacity" values="0.3;1;0.3" dur="3.5s" repeatCount="indefinite"/></circle>
    <circle cx="400" cy="60" r="1.5"><animate attributeName="opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite"/></circle>
    <circle cx="30" cy="300" r="0.8"><animate attributeName="opacity" values="0.3;1;0.3" dur="5s" repeatCount="indefinite"/></circle>
    <circle cx="770" cy="250" r="1"><animate attributeName="opacity" values="0.3;1;0.3" dur="3s" repeatCount="indefinite"/></circle>
  </g>
  <!-- Moon -->
  <circle cx="700" cy="80" r="35" fill="#f5f5e8" filter="url(#dropShadow)">
     <animate attributeName="r" values="35;36;35" dur="8s" repeatCount="indefinite"/>
  </circle>
  <circle cx="710" cy="72" r="5" fill="#e0e0d0" opacity="0.5"/>
  <circle cx="695" cy="90" r="3" fill="#e0e0d0" opacity="0.4"/>
  <!-- Desk Surface -->
  <path d="M50,480 L750,480 L750,580 L50,580 Z" fill="#1c1c28" filter="url(#dropShadow)"/>
  <path d="M50,480 L750,480" stroke="#333" stroke-width="2"/>
  <!-- Monitor Stand & Base -->
  <g transform="translate(300, 480)">
    <rect x="160" y="-20" width="80" height="20" rx="2" fill="#252525" filter="url(#dropShadow)"/>
    <path d="M180,0 L190,-40 L210,-40 L220,0 Z" fill="#252525" filter="url(#dropShadow)"/>
  </g>
  <!-- Monitor Body -->
  <g transform="translate(150, 100)" filter="url(#dropShadow)">
    <!-- Screen Bezel -->
    <rect x="0" y="0" width="500" height="320" rx="8" fill="#111" stroke="#222" stroke-width="2"/>
    <!-- Screen Content Area -->
    <rect x="10" y="10" width="480" height="300" rx="4" fill="url(#screenBg)"/>
    <!-- Screen Content: Code Editor -->
    <g id="codeEditor" font-family="'JetBrains Mono', 'Fira Code', 'Consolas', monospace" font-size="14" fill="#d4d4d4" line-height="1.6">
      <!-- Line Numbers -->
      <g fill="#6a6a6a" text-anchor="end" font-size="12">
        <text x="30" y="40">1</text>
        <text x="30" y="60">2</text>
        <text x="30" y="80">3</text>
        <text x="30" y="100">4</text>
        <text x="30" y="120">5</text>
        <text x="30" y="140">6</text>
        <text x="30" y="160">7</text>
        <text x="30" y="180">8</text>
        <text x="30" y="200">9</text>
        <text x="30" y="220">10</text>
        <text x="30" y="240">11</text>
        <text x="30" y="260">12</text>
        <text x="30" y="280">13</text>
      </g>
      <!-- Code Lines -->
      <g font-size="13" dominant-baseline="hanging">
        <text x="45" y="38"><tspan fill="#569cd6">const</tspan> <tspan fill="#dcdcaa">duck</tspan> = <tspan fill="#9cdcfe">require</tspan>(<tspan fill="#ce9178">'rubber-duck'</tspan>);</text>
        <text x="45" y="58"><tspan fill="#569cd6">let</tspan> <tspan fill="#dcdcaa">sanity</tspan> = <tspan fill="#b5cea8">100</tspan>;</text>
        <text x="45" y="78"></text>
        <text x="45" y="98"><tspan fill="#569cd6">function</tspan> <tspan fill="#dcdcaa">debugLife</tspan>() {</text>
        <text x="65" y="118"><tspan fill="#569cd6">while</tspan> (<tspan fill="#9cdcfe">sanity</tspan> > <tspan fill="#b5cea8">0</tspan>) {</text>
        <text x="85" y="138"><tspan fill="#dcdcaa">console.log</tspan>(<tspan fill="#ce9178">"Quack? "</tspan> + <tspan fill="#9cdcfe">sanity</tspan>);</text>
        <text x="85" y="158"><tspan fill="#dcdcaa">sanity</tspan> -= <tspan fill="#b5cea8">Math.random</tspan>() * <tspan fill="#b5cea8">10</tspan>;</text>
        <text x="85" y="178"><tspan fill="#d4d4d4"><!-- Cursor Blink --></tspan><tspan id="cursor" fill="#ffff00" opacity="1">|</tspan></text>
        <text x="65" y="198">}</text>
        <text x="45" y="218">}</text>
        <text x="45" y="238"></text>
        <text x="45" y="258"><tspan fill="#f44747" filter="url(#textGlow)">// ERROR: Segmentation Fault (Core Dumped)</tspan></text>
        <text x="45" y="278"><tspan fill="#f44747" filter="url(#textGlow)">// Line 6: Unexpected token '🦆' in production</tspan></text>
      </g>
    </g>
    <!-- Blinking Cursor Animation -->
    <animate xlink:href="#cursor" attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/>
    <!-- Status Bar -->
    <rect x="10" y="280" width="480" height="30" fill="#007acc" opacity="0.1" rx="0 0 4 4"/>
    <text x="20" y="302" font-family="monospace" font-size="11" fill="#888">Ln 13, Col 42  |  UTF-8  |  JavaScript  |  🦆 Mode: PANIC</text>
  </g>
  <!-- Keyboard -->
  <g transform="translate(220, 430)" filter="url(#dropShadow)">
    <rect x="0" y="0" width="360" height="50" rx="4" fill="#1f1f1f" stroke="#333" stroke-width="1"/>
    <g font-family="sans-serif" font-size="10" fill="#aaa" text-anchor="middle" dominant-baseline="middle">
      <!-- Row 1 -->
      <g id="keysRow1">
        <rect x="5" y="5" width="35" height="20" rx="3" fill="url(#keyGrad)" stroke="#333"/>
        <text x="22.5" y="15">ESC</text>
        <rect x="50" y="5" width="30" height="20" rx="3" fill="url(#keyGrad)" stroke="#333"/>
        <text x="65" y="15">F1</text>
        <rect x="85" y="5" width="30" height="20" rx="3" fill="url(#keyGrad)" stroke="#333"/>
        <text x="100" y="15">F2</text>
        <!-- ... abbreviated for brevity, visual representation -->
      </g>
      <!-- Visual Key Blocks -->
      <rect x="5" y="5" width="340" height="20" rx="3" fill="url(#keyGrad)" stroke="#333" opacity="0.8"/>
      <rect x="5" y="30" width="340" height="15" rx="3" fill="url(#keyGrad)" stroke="#333" opacity="0.6"/>
      <!-- Spacebar highlight (being pressed) -->
      <rect x="100" y="30" width="160" height="15" rx="3" fill="#007acc" opacity="0.3" stroke="#007acc"/>
    </g>
  </g>
  <!-- Coffee Cup -->
  <g transform="translate(550, 380)" filter="url(#dropShadow)">
    <!-- Cup Body -->
    <path d="M0,0 L60,0 L65,60 L-5,60 Z" fill="url(#cupGrad)" stroke="#ccc" stroke-width="1"/>
    <!-- Coffee Liquid -->
    <path d="M3,5 L57,5 L60,55 L0,55 Z" fill="url(#coffeeGrad)"/>
    <!-- Highlight on liquid -->
    <ellipse cx="20" cy="15" rx="15" ry="5" fill="#ffffff" opacity="0.05"/>
    <!-- Handle -->
    <path d="M60,15 Q80,15 80,35 Q80,50 60,50" fill="none" stroke="url(#cupGrad)" stroke-width="8" stroke-linecap="round"/>
    <path d="M60,15 Q75,15 75,35 Q75,50 60,50" fill="none" stroke="#eee" stroke-width="2" stroke-linecap="round" opacity="0.5"/>
    <!-- Steam (Animated) -->
    <g opacity="0.8">
      <use xlink:href="#steamPath" x="15" y="0" transform="scale(1)">
         <animate attributeName="opacity" values="0;0.5;0" dur="3s" repeatCount="indefinite" begin="0s"/>
         <animateTransform attributeName="transform" type="translate" values="15,0; 15,-20; 15,0" dur="3s" repeatCount="indefinite" begin="0s"/>
      </use>
      <use xlink:href="#steamPath" x="35" y="0" transform="scale(0.8)">
         <animate attributeName="opacity" values="0;0.5;0" dur="3s" repeatCount="indefinite" begin="0.5s"/>
         <animateTransform attributeName="transform" type="translate" values="35,0; 35,-20; 35,0" dur="3s" repeatCount="indefinite" begin="0.5s"/>
      </use>
      <use xlink:href="#steamPath" x="50" y="0" transform="scale(0.6)">
         <animate attributeName="opacity" values="0;0.4;0" dur="2.5s" repeatCount="indefinite" begin="1s"/>
         <animateTransform attributeName="transform" type="translate" values="50,0; 50,-15; 50,0" dur="2.5s" repeatCount="indefinite" begin="1s"/>
      </use>
    </g>
    <!-- "Java" Text on cup -->
    <text x="30" y="40" text-anchor="middle" font-family="serif" font-size="14" font-weight="bold" fill="#333" opacity="0">☕</text>
    <text x="30" y="42" text-anchor="middle" font-family="monospace" font-size="9" fill="#555" transform="rotate(-10 30 42)">java</text>
  </g>
  <!-- The Duck Programmer -->
  <g transform="translate(100, 280)" filter="url(#dropShadow)">
    <!-- Body -->
    <ellipse cx="0" cy="40" rx="55" ry="65" fill="url(#duckBody)"/>
    <!-- White Chest Patch -->
    <ellipse cx="0" cy="55" rx="25" ry="35" fill="#FFF8DC" opacity="0.8"/>
    <!-- Wings (Arms) Typing -->
    <!-- Left Wing -->
    <path d="M-55,20 Q-90,0 -85,40 Q-70,60 -40,55" fill="url(#duckBody)" stroke="#E6B800" stroke-width="1"/>
    <!-- Right Wing (raised higher) -->
    <path d="M55,10 Q95,-10 100,30 Q90,55 45,50" fill="url(#duckBody)" stroke="#E6B800" stroke-width="1"/>
    <!-- Head -->
    <ellipse cx="0" cy="-45" rx="40" ry="45" fill="url(#duckHead)"/>
    <!-- Eyes (Tired, bags under eyes) -->
    <g>
      <!-- Left Eye -->
      <ellipse cx="-15" cy="-50" rx="12" ry="10" fill="white"/>
      <ellipse cx="-13" cy="-50" rx="5" ry="6" fill="#1a1a2e"/> <!-- Pupil -->
      <ellipse cx="-11" cy="-52" rx="2" ry="2" fill="white"/> <!-- Reflection -->
      <!-- Eye Bag -->
      <path d="M-25,-40 Q-15,-38 -5,-40" fill="none" stroke="#E6B800" stroke-width="3" stroke-linecap="round" opacity="0.5"/>
      <!-- Eyelid (Tired) -->
      <path d="M-27,-55 Q-15,-50 -3,-55" fill="url(#duckHead)" stroke="none"/>
      <!-- Right Eye -->
      <ellipse cx="15" cy="-50" rx="12" ry="10" fill="white"/>
      <ellipse cx="17" cy="-50" rx="5" ry="6" fill="#1a1a2e"/> <!-- Pupil looking right -->
      <ellipse cx="19" cy="-52" rx="2" ry="2" fill="white"/>
      <!-- Eye Bag -->
      <path d="M5,-40 Q15,-38 25,-40" fill="none" stroke="#E6B800" stroke-width="3" stroke-linecap="round" opacity="0.5"/>
      <!-- Eyelid -->
      <path d="M3,-55 Q15,-50 27,-55" fill="url(#duckHead)" stroke="none"/>
    </g>
    <!-- Beak -->
    <path d="M-20,-30 Q0,-10 20,-30 Q0,-25 -20,-30" fill="url(#orangeGrad)"/>
    <!-- Nostril -->
    <ellipse cx="-5" cy="-26" rx="2" ry="1" fill="#E87C00"/>
    <ellipse cx="5" cy="-26" rx="2" ry="1" fill="#E87C00"/>
    <!-- Beak line -->
    <path d="M-18,-30 Q0,-20 18,-30" fill="none" stroke="#E87C00" stroke-width="1.5"/>
    <!-- Headphones -->
    <path d="M-40,-60 Q-40,-90 0,-90 Q40,-90 40,-60" fill="none" stroke="#333" stroke-width="6" stroke-linecap="round"/>
    <circle cx="-40" cy="-60" r="10" fill="#222" stroke="#444"/>
    <circle cx="40" cy="-60" r="10" fill="#222" stroke="#444"/>
    <circle cx="-40" cy="-60" r="5" fill="#444"/>
    <circle cx="40" cy="-60" r="5" fill="#444"/>
    <!-- Headphone wire -->
    <path d="M-40,-50 Q-50,-30 -60,20" fill="none" stroke="#333" stroke-width="1.5" stroke-dasharray="2,2"/>
    <!-- Feet (under desk) -->
    <g transform="translate(0, 105)">
      <path d="M-25,0 L-25,20 L-40,20" fill="none" stroke="url(#orangeGrad)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M25,0 L25,20 L40,20" fill="none" stroke="url(#orangeGrad)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
      <!-- Toes -->
      <g stroke="url(#orangeGrad)" stroke-width="3" stroke-linecap="round">
        <line x1="-25" y1="20" x2="-35" y2="25"/>
        <line x1="-25" y1="20" x2="-25" y2="25"/>
        <line x1="-25" y1="20" x2="-15" y2="25"/>
        <line x1="25" y1="20" x2="15" y2="25"/>
        <line x1="25" y1="20" x2="25" y2="25"/>
        <line x1="25" y1="20" x2="35" y2="25"/>
      </g>
    </g>
  </g>
  <!-- Wall Clock (3:00 AM) -->
  <g transform="translate(650, 150)" filter="url(#dropShadow)">
    <circle cx="0" cy="0" r="50" fill="#fafafa" stroke="#ddd" stroke-width="4"/>
    <circle cx="0" cy="0" r="5" fill="#333"/>
    <!-- Hour Marks -->
    <g stroke="#333" stroke-width="2" stroke-linecap="round">
      <line x1="0" y1="-45" x2="0" y2="-38"/>
      <line x1="0" y1="45" x2="0" y2="38"/>
      <line x1="-45" y1="0" x2="-38" y2="0"/>
      <line x1="45" y1="0" x2="38" y2="0"/>
    </g>
    <!-- Hands -->
    <!-- Hour Hand (3) -->
    <line x1="0" y1="0" x2="30" y2="0" stroke="#e74c3c" stroke-width="4" stroke-linecap="round">
       <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="43200s" repeatCount="indefinite"/>
    </line>
    <!-- Minute Hand (12) -->
    <line x1="0" y1="0" x2="0" y2="-40" stroke="#333" stroke-width="3" stroke-linecap="round">
       <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="3600s" repeatCount="indefinite"/>
    </line>
    <!-- "AM" Label -->
    <text x="0" y="65" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#333">3:00 AM</text>
  </g>
  <!-- Bug on screen (Visual Metaphor) -->
  <g transform="translate(580, 280)" opacity="0.9">
    <animateTransform attributeName="transform" type="translate" values="580,280; 590,275; 575,285; 580,280" dur="0.5s" repeatCount="indefinite"/>
    <!-- Body -->
    <ellipse cx="0" cy="0" rx="8" ry="10" fill="#e74c3c"/>
    <ellipse cx="0" cy="-8" rx="6" ry="6" fill="#c0392b"/>
    <!-- Legs -->
    <g stroke="#962d1a" stroke-width="1.5" stroke-linecap="round">
      <line x1="-8" y1="-2" x2="-15" y2="-8"/>
      <line x1="-4" y1="4" x2="-12" y2="10"/>
      <line x1="0" y1="6" x2="-5" y2="14"/>
      <line x1="8" y1="-2" x2="15" y2="-8"/>
      <line x1="4" y1="4" x2="12" y2="10"/>
      <line x1="0" y1="6" x2="5" y2="14"/>
    </g>
    <!-- Antennae -->
    <g stroke="#962d1a" stroke-width="1.5" fill="none">
      <path d="M-4,-12 Q-8,-18 -6,-22"/>
      <path d="M4,-12 Q8,-18 6,-22"/>
    </g>
    <!-- Eyes -->
    <circle cx="-3" cy="-10" r="1.5" fill="white"/>
    <circle cx="3" cy="-10" r="1.5" fill="white"/>
  </g>
  <!-- Rubber Duck on Desk (Debugging Buddy) -->
  <g transform="translate(100, 450)" filter="url(#dropShadow)">
    <ellipse cx="0" cy="0" rx="20" ry="25" fill="#FFD700"/>
    <ellipse cx="0" cy="-25" rx="15" ry="18" fill="#FFE066"/>
    <path d="M-8,-30 Q0,-22 8,-30 Q0,-26 -8,-30" fill="url(#orangeGrad)"/>
    <circle cx="-5" cy="-30" r="2" fill="white" opacity="0.8"/>
    <circle cx="5" cy="-30" r="2" fill="white" opacity="0.8"/>
    <circle cx="-5" cy="-31" r="3" fill="black"/>
    <circle cx="5" cy="-31" r="3" fill="black"/>
  </g>
  <!-- Messy Papers/Sticky Notes -->
  <g transform="translate(50, 400)" filter="url(#dropShadow)">
    <rect x="0" y="0" width="80" height="100" fill="#fff9c4" transform="rotate(-5)"/>
    <text x="10" y="25" font-family="cursive" font-size="10" fill="#333" transform="rotate(-5 10 25)">TODO:</text>
    <text x="10" y="40" font-family="cursive" font-size="9" fill="#555" transform="rotate(-5 10 40)">1. Fix bug</text>
    <text x="10" y="52" font-family="cursive" font-size="9" fill="#555" transform="rotate(-5 10 52)">2. Sleep?</text>
    <text x="10" y="64" font-family="cursive" font-size="9" fill="#e74c3c" transform="rotate(-5 10 64)">3. Quack</text>
  </g>
  <g transform="translate(670, 400)" filter="url(#dropShadow)">
    <rect x="0" y="0" width="70" height="50" fill="#ffcdd2" transform="rotate(3)"/>
    <text x="10" y="20" font-family="cursive" font-size="10" fill="#c62828" transform="rotate(3 10 20)">BUG</text>
    <text x="10" y="35" font-family="cursive" font-size="9" fill="#555" transform="rotate(3 10 35)">Line 404</text>
  </g>
</svg>
```

</details>

## 13. qwen3.5_397b-cloud

**SVG size:** 5746 characters  
**Complexity:** High  
**Shape elements:** 54  
**Distinct colors:** 17  
**Raw response:** 5848 characters  
**Generation time:** 57.1s

![qwen3.5_397b-cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/qwen3.5_397b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for screen glows and dark rooms)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning props and characters
- Renders actual text on screen (code, labels, clock display)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck character is present -- the model understood the subject
- Laptop/screen elements are present -- the model understood the coding setup
- Coffee/mug elements are present -- the model captured the props
- Time indicator (3:00) is present -- the model captured the time cue

With 5,746 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400" width="100%" height="auto">
  <!-- Definitions for gradients and filters -->
  <defs>
    <linearGradient id="screenGlow" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#38bdf8;stop-opacity:0.2" />
      <stop offset="100%" style="stop-color:#0f172a;stop-opacity:0" />
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <!-- Background (Night Mode) -->
  <rect width="500" height="400" fill="#0f172a" />
  <!-- Window with moon -->
  <rect x="380" y="20" width="100" height="100" fill="#1e293b" stroke="#334155" stroke-width="2"/>
  <circle cx="430" cy="70" r="20" fill="#fde047" opacity="0.8"/>
  <circle cx="420" cy="60" r="3" fill="#1e293b"/>
  <circle cx="440" cy="75" r="4" fill="#1e293b"/>
  <!-- Desk -->
  <rect x="0" y="300" width="500" height="100" fill="#334155" />
  <rect x="0" y="300" width="500" height="10" fill="#475569" />
  <!-- Monitor -->
  <rect x="150" y="80" width="200" height="140" rx="5" fill="#1e293b" stroke="#475569" stroke-width="4"/>
  <rect x="155" y="85" width="190" height="130" rx="2" fill="#020617" />
  <!-- Monitor Stand -->
  <rect x="230" y="220" width="40" height="10" fill="#475569"/>
  <rect x="210" y="230" width="80" height="10" rx="2" fill="#475569"/>
  <!-- Code on Screen -->
  <g font-family="monospace" font-size="10" fill="#94a3b8">
    <text x="165" y="105">1. const duck = new Dev();</text>
    <text x="165" y="120">2. while (true) {</text>
    <text x="175" y="135">3.   if (bug.exists()) {</text>
    <text x="185" y="150" fill="#f87171">4.     throw new Panic(); // WHY??</text>
    <text x="175" y="165">5.   }</text>
    <text x="175" y="180">6.   coffee.drink();</text>
    <text x="165" y="195">7. }</text>
    <text x="165" y="210" fill="#38bdf8">> _ cursor blinking...</text>
  </g>
  <!-- Screen Glow Overlay -->
  <rect x="155" y="85" width="190" height="130" rx="2" fill="url(#screenGlow)" />
  <!-- The Duck Programmer -->
  <g transform="translate(250, 280)">
    <!-- Hoodie Body -->
    <path d="M-60,20 Q-60,-40 0,-40 Q60,-40 60,20 L60,40 L-60,40 Z" fill="#334155" />
    <!-- Hoodie Strings -->
    <path d="M-10,-30 L-15,-10" stroke="#94a3b8" stroke-width="2" fill="none"/>
    <path d="M10,-30 L15,-10" stroke="#94a3b8" stroke-width="2" fill="none"/>
    <!-- Head -->
    <circle cx="0" cy="-50" r="35" fill="#facc15" />
    <!-- Eyes (Tired with bags) -->
    <ellipse cx="-12" cy="-55" rx="8" ry="5" fill="#fef3c7" />
    <ellipse cx="12" cy="-55" rx="8" ry="5" fill="#fef3c7" />
    <circle cx="-12" cy="-55" r="3" fill="#000" />
    <circle cx="12" cy="-55" r="3" fill="#000" />
    <!-- Eye bags -->
    <path d="M-20,-48 Q-12,-45 -4,-48" stroke="#f59e0b" stroke-width="1" fill="none"/>
    <path d="M4,-48 Q12,-45 20,-48" stroke="#f59e0b" stroke-width="1" fill="none"/>
    <!-- Beak -->
    <path d="M-5,-45 L5,-45 L0,-35 Z" fill="#f97316" />
    <!-- Glasses -->
    <g stroke="#cbd5e1" stroke-width="2" fill="none">
      <circle cx="-12" cy="-55" r="9" />
      <circle cx="12" cy="-55" r="9" />
      <line x1="-3" y1="-55" x2="3" y2="-55" />
      <!-- Glare on glasses -->
      <line x1="-16" y1="-59" x2="-8" y2="-51" stroke="white" stroke-width="1" opacity="0.6"/>
      <line x1="8" y1="-59" x2="16" y2="-51" stroke="white" stroke-width="1" opacity="0.6"/>
    </g>
  </g>
  <!-- Coffee Mug -->
  <g transform="translate(130, 310)">
    <path d="M0,0 L0,30 Q0,40 10,40 L30,40 Q40,40 40,30 L40,0 Z" fill="#cbd5e1" />
    <path d="M40,10 Q55,10 55,20 Q55,30 40,30" fill="none" stroke="#cbd5e1" stroke-width="3"/>
    <rect x="2" y="5" width="36" height="25" fill="#78350f" rx="2"/> <!-- Coffee liquid -->
    <!-- Steam -->
    <path d="M10,-5 Q15,-15 10,-25" stroke="#94a3b8" stroke-width="2" fill="none" opacity="0.6"/>
    <path d="M20,-5 Q25,-15 20,-25" stroke="#94a3b8" stroke-width="2" fill="none" opacity="0.6"/>
    <path d="M30,-5 Q35,-15 30,-25" stroke="#94a3b8" stroke-width="2" fill="none" opacity="0.6"/>
  </g>
  <!-- Clock showing 3:00 AM -->
  <g transform="translate(420, 150)">
    <circle cx="0" cy="0" r="25" fill="#0f172a" stroke="#fde047" stroke-width="2"/>
    <line x1="0" y1="0" x2="0" y2="-15" stroke="#fde047" stroke-width="2" /> <!-- Hour hand -->
    <line x1="0" y1="0" x2="20" y2="0" stroke="#fde047" stroke-width="1" />  <!-- Min hand -->
    <circle cx="0" cy="0" r="2" fill="#fde047"/>
    <text x="0" y="35" font-family="monospace" font-size="10" fill="#fde047" text-anchor="middle">3:00 AM</text>
  </g>
  <!-- Keyboard (Abstract) -->
  <rect x="180" y="320" width="140" height="40" rx="5" fill="#1e293b" stroke="#475569"/>
  <g fill="#64748b">
    <rect x="190" y="330" width="10" height="10" rx="2"/>
    <rect x="205" y="330" width="10" height="10" rx="2"/>
    <rect x="220" y="330" width="10" height="10" rx="2"/>
    <rect x="235" y="330" width="10" height="10" rx="2"/>
    <rect x="250" y="330" width="10" height="10" rx="2"/>
    <rect x="265" y="330" width="10" height="10" rx="2"/>
    <rect x="280" y="330" width="10" height="10" rx="2"/>
    <rect x="295" y="330" width="10" height="10" rx="2"/>
    <rect x="190" y="345" width="10" height="10" rx="2"/>
    <rect x="205" y="345" width="10" height="10" rx="2"/>
    <rect x="220" y="345" width="10" height="10" rx="2"/>
    <rect x="235" y="345" width="10" height="10" rx="2"/>
    <rect x="250" y="345" width="10" height="10" rx="2"/>
    <rect x="265" y="345" width="10" height="10" rx="2"/>
    <rect x="280" y="345" width="25" height="10" rx="2"/>
  </g>
</svg>
```

</details>

## 14. deepseek-v4-flash_0731-cloud

**SVG size:** 12388 characters  
**Complexity:** Very high  
**Shape elements:** 116  
**Distinct colors:** 32  
**Raw response:** 12388 characters

![deepseek-v4-flash_0731-cloud duck programmer SVG](/assets/img/diagrams/ollama-svg-programmer/deepseek-v4-flash_0731-cloud.svg)

### Analysis

This SVG contains approximately **116 shape elements** and uses **32 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Includes animation elements -- the model attempted motion
- Contains XML comments, showing the model's internal documentation of its work

This is a large, detailed SVG (over 10,000 characters) packed with shapes, gradients, and structure. Best when you want maximum visual richness over file size.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
    <defs>
        <!-- Background Gradient (Late Night Coding Vibe) -->
        <linearGradient id="bg-grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#0f0c29" />
            <stop offset="50%" stop-color="#302b63" />
            <stop offset="100%" stop-color="#24243e" />
        </linearGradient>
        <!-- Monitor Glow -->
        <radialGradient id="monitor-glow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#4af" stop-opacity="0.8" />
            <stop offset="100%" stop-color="#4af" stop-opacity="0" />
        </radialGradient>
        <!-- Duck Glow (from monitor) -->
        <linearGradient id="duck-glow" x1="1" y1="0" x2="0" y2="0">
            <stop offset="0%" stop-color="#4af" stop-opacity="0.5" />
            <stop offset="100%" stop-color="#4af" stop-opacity="0" />
        </linearGradient>
    </defs>
    <!-- Background -->
    <rect width="800" height="600" fill="url(#bg-grad)" />
    <!-- Window &amp; Night Sky -->
    <g id="window">
        <rect x="50" y="50" width="200" height="250" rx="10" fill="#1a1a2e" stroke="#4a4a6a" stroke-width="8" />
        <!-- Moon -->
        <circle cx="120" cy="100" r="30" fill="#fdfd96" />
        <circle cx="120" cy="100" r="50" fill="#fdfd96" opacity="0.2" />
        <circle cx="120" cy="100" r="70" fill="#fdfd96" opacity="0.1" />
        <!-- Moon Craters -->
        <circle cx="110" cy="90" r="5" fill="#e5e57a" />
        <circle cx="130" cy="110" r="4" fill="#e5e57a" />
        <circle cx="115" cy="115" r="3" fill="#e5e57a" />
        <!-- Stars -->
        <circle cx="80" cy="70" r="1.5" fill="#fff" />
        <circle cx="180" cy="60" r="2" fill="#fff" />
        <circle cx="200" cy="140" r="1.5" fill="#fff" />
        <circle cx="90" cy="200" r="2" fill="#fff" />
        <circle cx="220" cy="180" r="1" fill="#fff" />
        <circle cx="150" cy="150" r="1" fill="#fff" />
        <circle cx="170" cy="220" r="1.5" fill="#fff" />
        <!-- Window Crossbars -->
        <line x1="150" y1="50" x2="150" y2="300" stroke="#4a4a6a" stroke-width="8" />
        <line x1="50" y1="175" x2="250" y2="175" stroke="#4a4a6a" stroke-width="8" />
    </g>
    <!-- Wall Clock (3:00 AM) -->
    <g id="clock">
        <circle cx="680" cy="100" r="45" fill="#222" stroke="#555" stroke-width="6" />
        <!-- Tick Marks -->
        <line x1="680" y1="60" x2="680" y2="70" stroke="#fff" stroke-width="3" />
        <line x1="680" y1="130" x2="680" y2="140" stroke="#fff" stroke-width="3" />
        <line x1="640" y1="100" x2="650" y2="100" stroke="#fff" stroke-width="3" />
        <line x1="710" y1="100" x2="720" y2="100" stroke="#fff" stroke-width="3" />
        <!-- Hands -->
        <line x1="680" y1="100" x2="680" y2="65" stroke="#fff" stroke-width="4" stroke-linecap="round" />
        <line x1="680" y1="100" x2="705" y2="100" stroke="#ff4d4d" stroke-width="5" stroke-linecap="round" />
        <circle cx="680" cy="100" r="5" fill="#fff" />
        <text x="680" y="155" fill="#fff" font-family="monospace" font-size="12" text-anchor="middle" opacity="0.8">3:00 AM</text>
    </g>
    <!-- Desk and Chair -->
    <g id="furniture">
        <!-- Chair Back -->
        <rect x="60" y="250" width="30" height="150" rx="10" fill="#1a1a2e" />
        <rect x="50" y="240" width="50" height="20" rx="5" fill="#16213e" />
        <!-- Chair Seat -->
        <rect x="80" y="420" width="160" height="20" rx="8" fill="#16213e" />
        <!-- Chair Legs -->
        <rect x="150" y="440" width="20" height="40" fill="#1a1a2e" />
        <rect x="130" y="480" width="60" height="10" rx="3" fill="#111" />
        <!-- Desk Top -->
        <rect x="50" y="450" width="700" height="20" rx="5" fill="#2d1b0f" />
        <rect x="60" y="470" width="680" height="10" fill="#1a0f07" />
        <!-- Desk Legs -->
        <rect x="150" y="480" width="15" height="120" fill="#2d1b0f" />
        <rect x="635" y="480" width="15" height="120" fill="#2d1b0f" />
        <!-- Power Strip &amp; Cables Under Desk -->
        <rect x="300" y="490" width="60" height="15" rx="3" fill="#d4d4d4" />
        <circle cx="310" cy="497" r="2" fill="#111" />
        <circle cx="320" cy="497" r="2" fill="#111" />
        <circle cx="330" cy="497" r="2" fill="#111" />
        <circle cx="340" cy="497" r="2" fill="#111" />
        <path d="M 300 497 C 280 497, 260 520, 220 520" fill="none" stroke="#111" stroke-width="4" />
        <path d="M 360 497 C 400 497, 450 520, 500 520" fill="none" stroke="#111" stroke-width="4" />
    </g>
    <!-- Pizza Box -->
    <g id="pizza-box">
        <rect x="640" y="415" width="80" height="30" rx="3" fill="#d4a373" />
        <path d="M 640 415 L 720 415 L 680 395 L 640 415" fill="#b5835a" />
        <line x1="680" y1="395" x2="680" y2="415" stroke="#b5835a" stroke-width="2" />
        <!-- Pizza Slice -->
        <path d="M 660 410 L 675 410 L 667 400 Z" fill="#f4a261" />
        <circle cx="667" cy="405" r="1.5" fill="#e76f51" />
        <circle cx="670" cy="407" r="1.5" fill="#e76f51" />
    </g>
    <!-- Monitor and Code -->
    <g id="monitor">
        <!-- Stand -->
        <rect x="440" y="380" width="60" height="60" fill="#333" />
        <rect x="420" y="440" width="100" height="15" rx="5" fill="#222" />
        <!-- Frame -->
        <rect x="380" y="220" width="200" height="160" rx="10" fill="#111" stroke="#333" stroke-width="8" />
        <!-- Screen -->
        <rect x="390" y="230" width="180" height="140" rx="5" fill="#0a1f2e" />
        <!-- Code Lines -->
        <rect x="400" y="245" width="80" height="8" rx="4" fill="#c792ea" />
        <rect x="490" y="245" width="30" height="8" rx="4" fill="#fff" opacity="0.5" />
        <rect x="530" y="245" width="10" height="8" rx="4" fill="#ffcb8b" />
        <rect x="400" y="260" width="50" height="8" rx="4" fill="#82aaff" />
        <rect x="460" y="260" width="60" height="8" rx="4" fill="#fff" opacity="0.6" />
        <rect x="530" y="260" width="20" height="8" rx="4" fill="#f78c6c" />
        <rect x="410" y="275" width="30" height="8" rx="4" fill="#c792ea" />
        <rect x="450" y="275" width="40" height="8" rx="4" fill="#ffcb8b" />
        <rect x="500" y="275" width="10" height="8" rx="4" fill="#ffcb8b" />
        <rect x="420" y="290" width="40" height="8" rx="4" fill="#82aaff" />
        <rect x="470" y="290" width="30" height="8" rx="4" fill="#f78c6c" />
        <rect x="410" y="305" width="30" height="8" rx="4" fill="#c792ea" />
        <rect x="450" y="305" width="40" height="8" rx="4" fill="#ffcb8b" />
        <rect x="420" y="320" width="40" height="8" rx="4" fill="#82aaff" />
        <rect x="470" y="320" width="30" height="8" rx="4" fill="#f78c6c" />
        <rect x="510" y="320" width="40" height="8" rx="4" fill="#c792ea" />
        <rect x="400" y="335" width="10" height="8" rx="4" fill="#ffcb8b" />
        <!-- Blinking Cursor -->
        <rect x="420" y="335" width="8" height="10" fill="#fff">
            <animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite" />
        </rect>
        <!-- Sticky Notes on Monitor -->
        <rect x="590" y="230" width="25" height="25" fill="#ffeb3b" transform="rotate(15, 590, 230)" />
        <rect x="610" y="220" width="25" height="25" fill="#ff5722" transform="rotate(-10, 610, 220)" />
        <line x1="590" y1="235" x2="610" y2="235" stroke="#333" stroke-width="1.5" transform="rotate(15, 590, 230)" />
        <line x1="590" y1="245" x2="610" y2="245" stroke="#333" stroke-width="1.5" transform="rotate(15, 590, 230)" />
        <!-- Monitor Glow Effect -->
        <circle cx="480" cy="300" r="200" fill="url(#monitor-glow)" opacity="0.3" pointer-events="none" />
    </g>
    <!-- Duck Programmer -->
    <g id="duck-programmer">
        <!-- Hoodie Back/Hood -->
        <circle cx="160" cy="310" r="55" fill="#152c47" />
        <!-- Duck Body (Hoodie) -->
        <rect x="80" y="330" width="160" height="130" rx="40" fill="#1e3a5f" />
        <!-- Hoodie Pocket -->
        <path d="M 110 430 L 210 430 L 200 450 L 120 450 Z" fill="#152c47" />
        <!-- Hoodie Drawstrings -->
        <line x1="130" y1="340" x2="120" y2="370" stroke="#fff" stroke-width="3" stroke-linecap="round" />
        <line x1="190" y1="340" x2="200" y2="370" stroke="#fff" stroke-width="3" stroke-linecap="round" />
        <circle cx="120" cy="370" r="4" fill="#fff" />
        <circle cx="200" cy="370" r="4" fill="#fff" />
        <!-- Duck Head -->
        <circle cx="160" cy="280" r="45" fill="#e6f2ff" />
        <!-- Duck Beak -->
        <path d="M 200 270 C 240 260, 260 270, 250 285 C 240 290, 210 290, 200 285 Z" fill="#f4a261" />
        <path d="M 200 285 C 210 295, 240 295, 250 285 C 240 305, 210 305, 200 290 Z" fill="#e76f51" />
        <!-- Duck Eyes (Tired) -->
        <!-- Right Eye -->
        <circle cx="190" cy="270" r="6" fill="#fff" />
        <circle cx="193" cy="270" r="3" fill="#000" />
        <!-- Left Eye -->
        <circle cx="145" cy="270" r="6" fill="#fff" />
        <circle cx="148" cy="270" r="3" fill="#000" />
        <!-- Eyelids (Half-closed) -->
        <rect x="139" y="262" width="12" height="5" rx="2" fill="#e6f2ff" />
        <rect x="184" y="262" width="12" height="5" rx="2" fill="#e6f2ff" />
        <!-- Dark Circles under Eyes -->
        <path d="M 138 280 Q 145 283 152 280" fill="none" stroke="#7a8a9a" stroke-width="2" />
        <path d="M 183 280 Q 190 283 197 280" fill="none" stroke="#7a8a9a" stroke-width="2" />
        <!-- Floating Zzz (Sleepiness) -->
        <text x="120" y="240" fill="#fff" font-family="monospace" font-size="12" opacity="0.8">Z</text>
        <text x="105" y="220" fill="#fff" font-family="monospace" font-size="14" opacity="0.6">Z</text>
        <text x="90" y="195" fill="#fff" font-family="monospace" font-size="16" opacity="0.4">Z</text>
        <!-- Headset -->
        <path d="M 120 250 A 40 40 0 0 1 200 250" fill="none" stroke="#222" stroke-width="6" />
        <rect x="110" y="245" width="15" height="25" rx="5" fill="#333" />
        <rect x="195" y="245" width="15" height="25" rx="5" fill="#333" />
        <!-- Microphone -->
        <path d="M 210 255 L 225 255 L 225 270 L 210 270 Z" fill="#333" />
        <circle cx="217" cy="270" r="4" fill="#555" />
        <!-- Duck Wing (Typing) -->
        <path d="M 120 390 Q 200 400 230 420 Q 190 435 130 430 Z" fill="#d9e6f2" />
        <!-- Monitor Glow on Duck -->
        <rect x="120" y="220" width="200" height="250" fill="url(#duck-glow)" opacity="0.4" pointer-events="none" />
    </g>
    <!-- Desk Accessories -->
    <g id="desk-accessories">
        <!-- Keyboard -->
        <rect x="200" y="420" width="140" height="30" rx="5" fill="#111" />
        <rect x="205" y="425" width="15" height="10" fill="#333" />
        <rect x="225" y="425" width="15" height="10" fill="#333" />
        <rect x="245" y="425" width="15" height="10" fill="#333" />
        <rect x="265" y="425" width="15" height="10" fill="#333" />
        <rect x="285" y="425" width="15" height="10" fill="#333" />
        <rect x="305" y="425" width="15" height="10" fill="#333" />
        <rect x="325" y="425" width="15" height="10" fill="#333" />
        <!-- Coffee Cup -->
        <rect x="360" y="400" width="25" height="30" rx="3" fill="#c43" />
        <path d="M 385 405 C 395 405, 395 425, 385 425" fill="none" stroke="#c43" stroke-width="4" />
        <ellipse cx="372" cy="400" rx="12" ry="4" fill="#3a1c0d" />
        <!-- Coffee Steam Animation -->
        <path d="M 365 390 Q 370 380 365 370" fill="none" stroke="#fff" stroke-width="2" opacity="0.6">
            <animate attributeName="opacity" values="0.6;0.1;0.6" dur="2s" repeatCount="indefinite" />
        </path>
        <path d="M 375 395 Q 380 385 375 375" fill="none" stroke="#fff" stroke-width="2" opacity="0.4">
            <animate attributeName="opacity" values="0.4;0.1;0.4" dur="1.5s" repeatCount="indefinite" />
        </path>
        <!-- Tiny Rubber Duck (Debugging Buddy) -->
        <circle cx="375" cy="440" r="8" fill="#ffd700" />
        <circle cx="375" cy="432" r="5" fill="#ffd700" />
        <path d="M 379 432 L 385 433 L 384 435 Z" fill="#ff8c00" />
        <circle cx="377" cy="431" r="1" fill="#000" />
    </g>
</svg>
```

</details>


## How to Pick the Best Model: A Decision Guide

Now that you have seen all 14 programmer SVGs, here is a structured way to decide which Ollama cloud model is best for your SVG generation needs:

### Step 1: Define your priority

- **Visual quality first**: Scroll back through the SVGs above and pick the one that looks best to your eye. For developer scenes, trust your eye -- the shape and color counts are useful, but a model with 30 shapes and a great palette can beat one with 100 shapes and muddy colors.
- **Code quality first**: Open the raw SVG source for each model (use the disclosure toggles) and look for `<defs>`, `<use>`, gradients, and clean indentation. Models that produce structured code are easier to recolor and reuse.
- **Speed first**: If you are building a real-time app, prioritize the models that responded in under 30 seconds (see the per-model sections for elapsed times).
- **File size first**: For web embedding, smaller is better. Look at the SVG size column in the summary table.

### Step 2: Cross-check across prompts

A model that does well on the programmer duck might fail on a nature prompt. Check our other benchmarks:

- [Duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/) -- character + vehicle
- [Duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/) -- dynamic action scene
- [Duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/) -- vehicle with multiple parts
- [Cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/) -- nature / scenery with no central character
- This post (programmer duck) -- indoor scene with screen + lighting + props

A model that consistently produces good results across all five prompts is a safer pick than one that only shines on a single type of scene.

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

## Why This Benchmark Matters: Indoor Scene vs Outdoor Scene Prompts

Most LLM SVG benchmarks only test one type of prompt. We deliberately run five different scene types because models specialize:

- **Character + vehicle prompts** (the duck bicycle/jeep series) reward models that understand anatomy, proportions, and mechanical parts. Models like `deepseek-v4-pro` and `kimi-k2.6` tend to do well here.
- **Dynamic action prompts** (the duck parachute) reward models that understand motion, perspective, and dynamic composition.
- **Nature prompts** (cherry blossom) reward models that understand organic shapes, palettes, and repetition. Models like `glm-5.1` and `nemotron-3-ultra` tend to do well here.
- **Indoor scene prompts** (this programmer duck post) reward models that understand lighting, screen text, props, and narrative. Models that can render readable text on a screen and convey a mood (tired, focused) are rare.

If you are picking a model for a specific project, look at the benchmark closest to your use case. If you want a general-purpose model, pick one that does *okay* across all five rather than one that aces one and fails the rest.

## Conclusion: You Decide the Winner

This comparison shows that 14 out of 18 active Ollama cloud models can generate valid SVG artwork from a developer-culture prompt about a duck programmer debugging code at 3am. The results vary dramatically in complexity, style, and technique -- and there is no single "best" model.

Our takeaways after running five SVG benchmarks (bicycle, parachute, jeep, blossom, programmer):

- **glm-5.1:cloud** and **nemotron-3-ultra:cloud** consistently produce the longest, most detailed SVGs across all prompt types. Best when you want maximum visual richness.
- **deepseek-v4-pro:cloud** and **deepseek-v4-flash:cloud** consistently produce well-structured, technically advanced SVGs with `<defs>`, `<use>`, and transforms. A strong default choice for code quality and editability.
- **gpt-oss:120b-cloud** and **gemma4:31b-cloud** are consistently among the fastest and produce compact SVGs. Best for speed-sensitive applications.
- **kimi-k2.6:cloud** output size varies wildly by prompt -- worth testing on your specific prompt.
- **glm-5.2:cloud** and **minimax-m3:cloud** offer a reliable balance of detail, speed, and code quality across all five prompts.

But the real verdict is yours. Scroll back through the SVGs, compare them visually, check the raw code, and pick the model that best fits your needs. Every model in this comparison is available right now on Ollama Cloud -- so you can reproduce these results in minutes.

## Links

- [Previous: Duck Driving a Bicycle Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/)
- [Previous: Duck Jumping From a Plane Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/)
- [Previous: Duck Driving a Jeep Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/)
- [Previous: Cherry Blossom Trees Comparison](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/)
- [Ollama Official Website](https://ollama.com)
- [Ollama Cloud Documentation](https://ollama.com/cloud)
- [SVG Specification (MDN)](https://developer.mozilla.org/en-US/docs/Web/SVG)
- [OpenAI API Reference (used by Ollama)](https://platform.openai.com/docs/api-reference/chat)
