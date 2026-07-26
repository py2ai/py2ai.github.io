---
layout: post
title: "Which Ollama Cloud Model is Best? Duck Driving a Jeep SVG Comparison"
description: "Compare 9 Ollama cloud models side by side to find the best LLM for SVG generation. See how each model draws a duck driving a jeep. Pick the winner yourself."
date: 2026-07-26
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/
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
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama cloud model comparison, deepseek vs glm vs qwen, LLM SVG benchmark, AI image generation comparison, duck jeep SVG, which Ollama model is best, Ollama cloud models 2026"
---

# Which Ollama Cloud Model is Best? Duck Driving a Jeep SVG Comparison

If you are wondering **which Ollama cloud model is best for SVG generation**, this post is for you. We sent the exact same prompt to 9 different state-of-the-art LLMs and let them draw a duck driving a jeep. The results are wildly different, and they reveal which models are actually good at structured drawing tasks.

This is the third in our SVG benchmark series. The prompt was: `Make an svg image about a duck driving a jeep`.
You can compare with our previous benchmarks:
[duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/) and [duck jumping from a plane with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/).

**The goal of this post is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** Different models shine at different things: raw detail, clean structure, fast generation, or visual fidelity. We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for SVG

Before looking at the results, here are the criteria you should use to judge each model:

- **Visual fidelity**: Does the output actually look like a duck driving a jeep? Are the wheels round, the beak duck-like, and the proportions correct?
- **Detail level**: How much scene detail (windows, doors, terrain, sky) does the model add? More detail is not always better, but it shows effort.
- **SVG code quality**: Does it use `<defs>`, `<use>`, gradients, and filters, or just raw shapes? Cleaner code means easier to edit and reuse.
- **Output size**: Very small SVGs (under 2,000 chars) are usually too simple. Very large SVGs (over 15,000 chars) are rich but may be slow to render.
- **Generation speed**: If you are building an app that streams SVGs to users, faster matters. We report generation time for each model.
- **Concept adherence**: Did the model actually draw a jeep (with wheels, doors, windshield) and a duck (with a beak, body, eyes)? Some models lose the plot.

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with zero post-processing.

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Speed | Complexity | Verdict |
|---|-------|----------|--------|--------|-------|------------|---------|
| 1 | `deepseek-v4-pro_cloud` | 6664 | 62 | 21 | - | Medium | Best all-rounder |
| 2 | `gemma4_31b-cloud` | 1680 | 19 | 13 | - | Low | Fastest |
| 3 | `glm-5.1_cloud` | 17901 | 160 | 50 | - | Very high | Most detailed |
| 4 | `glm-5.2_cloud` | 6996 | 66 | 26 | - | Medium | Balanced |
| 5 | `kimi-k2.6_cloud` | 9966 | 51 | 32 | - | Medium | Most technical |
| 6 | `minimax-m2.7_cloud` | 2722 | 26 | 15 | - | Low | Minimalist |
| 7 | `nemotron-3-super_cloud` | 1807 | 15 | 10 | - | Low | Compact |
| 8 | `nemotron-3-ultra_cloud` | 16076 | 92 | 40 | - | Very high | Richest scene |
| 9 | `qwen3.5_397b-cloud` | 3106 | 29 | 16 | - | Low | Efficient |
| 10 | `bjoernb/claude-opus-4-5:latest` | - | - | - | - | - | Retired (410) |
| 11 | `deepseek-v3.1:671b-cloud` | - | - | - | - | - | Retired (410) |
| 12 | `glm-5:cloud` | - | - | - | - | - | Retired (410) |
| 13 | `qwen3-vl:235b-cloud` | - | - | - | - | - | Retired (410) |

**9 out of 13** active models produced a valid SVG. The 4 retired models returned HTTP 410 Gone (removed from Ollama Cloud on 2026-07-15).

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most detailed, visually rich SVG**: pick `glm-5.1:cloud` or `nemotron-3-ultra:cloud`
- **You want the fastest response**: pick `gemma4:31b-cloud` (under 10 seconds)
- **You want the cleanest, most reusable SVG code**: pick `deepseek-v4-pro:cloud` or `kimi-k2.6:cloud`
- **You want a small, efficient SVG for web embedding**: pick `nemotron-3-super:cloud` or `qwen3.5:397b-cloud`
- **You want a balance of detail and speed**: pick `glm-5.2:cloud`

Now read on for the full per-model breakdown and judge for yourself.

## 1. deepseek-v4-pro_cloud

**SVG size:** 6664 characters  
**Complexity:** Medium  
**Shape elements:** 62  
**Distinct colors:** 21  
**Raw response:** 7062 characters

![deepseek-v4-pro_cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/deepseek-v4-pro_cloud.svg)

### Analysis

This SVG contains approximately **62 shape elements** and uses **21 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth and shading
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable. Good for general use.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <defs>
    <!-- Sky Gradient -->
    <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#4A90E2" />
      <stop offset="100%" stop-color="#87CEEB" />
    </linearGradient>
    
    <!-- Jeep Body Gradient -->
    <linearGradient id="jeepGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#66BB6A" />
      <stop offset="100%" stop-color="#2E7D32" />
    </linearGradient>

    <!-- Duck Body Gradient -->
    <linearGradient id="duckGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFE082" />
      <stop offset="100%" stop-color="#FFD700" />
    </linearGradient>
  </defs>

  <!-- Background Sky -->
  <rect width="800" height="400" fill="url(#skyGrad)" />
  
  <!-- Clouds -->
  <path d="M 100 100 Q 120 80 140 100 Q 160 90 170 110 Q 180 130 150 130 L 100 130 Q 80 130 90 110 Q 80 90 100 100" fill="#FFF" opacity="0.8" />
  <path d="M 600 150 Q 620 130 640 150 Q 660 140 670 160 Q 680 180 650 180 L 600 180 Q 580 180 590 160 Q 580 140 600 150" fill="#FFF" opacity="0.8" />
  <path d="M 300 80 Q 320 60 340 80 Q 360 70 370 90 Q 380 110 350 110 L 300 110 Q 280 110 290 90 Q 280 70 300 80" fill="#FFF" opacity="0.8" />

  <!-- Background Ground -->
  <rect y="400" width="800" height="200" fill="#A0522D" />
  
  <!-- Road Lines -->
  <line x1="0" y1="500" x2="800" y2="500" stroke="#FFD700" stroke-width="5" stroke-dasharray="20,20" />

  <!-- Shadow under Jeep -->
  <ellipse cx="400" cy="490" rx="200" ry="15" fill="#000" opacity="0.3" />

  <!-- ================= JEEP (BACK LAYER) ================= -->
  <!-- Jeep Chassis -->
  <rect x="200" y="320" width="350" height="100" rx="10" fill="url(#jeepGrad)" />
  
  <!-- Jeep Hood -->
  <rect x="200" y="300" width="180" height="20" rx="5" fill="url(#jeepGrad)" />
  
  <!-- Roll Cage (Rear) -->
  <path d="M 450 240 L 450 320 M 520 240 L 520 320 M 450 240 L 520 240" fill="none" stroke="#1B5E20" stroke-width="8" stroke-linejoin="round" />

  <!-- ================= DUCK ================= -->
  <!-- Duck Body -->
  <ellipse cx="400" cy="280" rx="50" ry="65" fill="url(#duckGrad)" />
  
  <!-- Duck Tail Feathers -->
  <path d="M 440 320 Q 480 300 490 330 Q 470 340 440 320" fill="#FFD700" />
  
  <!-- Duck Head -->
  <circle cx="400" cy="200" r="40" fill="url(#duckGrad)" />
  
  <!-- Duck Head Tuft -->
  <path d="M 390 165 Q 400 130 415 140 Q 410 155 400 165" fill="#FFD700" />
  
  <!-- Duck Beak -->
  <path d="M 435 190 L 490 200 L 435 220 Z" fill="#FF8C00" />
  <path d="M 435 210 Q 455 215 470 210" fill="none" stroke="#E65C00" stroke-width="3" stroke-linecap="round" />
  
  <!-- Duck Aviator Sunglasses -->
  <circle cx="410" cy="195" r="14" fill="#333" />
  <circle cx="440" cy="195" r="14" fill="#333" />
  <line x1="424" y1="195" x2="426" y2="195" stroke="#333" stroke-width="4" />
  <path d="M 396 195 L 410 195 M 440 195 L 454 195" stroke="#333" stroke-width="4" />

  <!-- ================= JEEP (FRONT LAYER) ================= -->
  <!-- Windshield Frame & Glass -->
  <polygon points="250,320 250,240 300,220 300,320" fill="#B3E5FC" opacity="0.6" stroke="#1B5E20" stroke-width="6" stroke-linejoin="round" />
  
  <!-- Rear-view Mirror -->
  <rect x="240" y="250" width="10" height="20" rx="2" fill="#333" />
  <line x1="245" y1="260" x2="250" y2="260" stroke="#333" stroke-width="4" />

  <!-- Steering Column & Wheel -->
  <line x1="300" y1="365" x2="300" y2="390" stroke="#333" stroke-width="8" />
  <circle cx="300" cy="340" r="25" fill="none" stroke="#333" stroke-width="8" />
  <line x1="300" y1="315" x2="300" y2="365" stroke="#333" stroke-width="8" />

  <!-- Duck Wing (on steering wheel) -->
  <path d="M 380 280 Q 330 320 300 340" fill="none" stroke="#FFD700" stroke-width="25" stroke-linecap="round" />

  <!-- Front Grille -->
  <rect x="190" y="320" width="20" height="80" rx="3" fill="#333" />
  <line x1="195" y1="330" x2="195" y2="390" stroke="#666" stroke-width="2" />
  <line x1="200" y1="330" x2="200" y2="390" stroke="#666" stroke-width="2" />
  <line x1="205" y1="330" x2="205" y2="390" stroke="#666" stroke-width="2" />

  <!-- Headlights -->
  <circle cx="210" cy="340" r="15" fill="#FFEB3B" />
  <circle cx="210" cy="340" r="8" fill="#FFF" />
  
  <!-- Headlight Beams -->
  <polygon points="210,340 50,300 50,380" fill="#FFF" opacity="0.3" />
  <polygon points="210,340 50,320 50,360" fill="#FFF" opacity="0.5" />

  <!-- Front Bumper -->
  <rect x="180" y="400" width="390" height="15" rx="5" fill="#9E9E9E" />

  <!-- Door Line & Handle -->
  <line x1="350" y1="320" x2="350" y2="420" stroke="#1B5E20" stroke-width="4" />
  <rect x="360" y="360" width="15" height="5" rx="2" fill="#333" />

  <!-- Exhaust Pipe & Smoke -->
  <rect x="550" y="380" width="20" height="10" rx="3" fill="#666" />
  <rect x="570" y="375" width="20" height="15" rx="3" fill="#888" />
  <circle cx="600" cy="370" r="10" fill="#D3D3D3" opacity="0.6" />
  <circle cx="620" cy="360" r="15" fill="#D3D3D3" opacity="0.6" />
  <circle cx="650" cy="350" r="20" fill="#D3D3D3" opacity="0.6" />

  <!-- Spare Tire Mount & Spare Tire -->
  <rect x="540" y="340" width="10" height="40" fill="#666" />
  <circle cx="565" cy="360" r="30" fill="#333" />
  <circle cx="565" cy="360" r="15" fill="#BDBDBD" />
  <circle cx="565" cy="360" r="5" fill="#333" />

  <!-- ================= WHEELS ================= -->
  <!-- Front Wheel -->
  <circle cx="280" cy="440" r="45" fill="#333" />
  <circle cx="280" cy="440" r="25" fill="#BDBDBD" />
  <circle cx="280" cy="440" r="10" fill="#333" />
  <path d="M 235 400 A 45 45 0 0 1 325 400" fill="none" stroke="#555" stroke-width="4" />

  <!-- Rear Wheel -->
  <circle cx="500" cy="440" r="45" fill="#333" />
  <circle cx="500" cy="440" r="25" fill="#BDBDBD" />
  <circle cx="500" cy="440" r="10" fill="#333" />
  <path d="M 455 400 A 45 45 0 0 1 545 400" fill="none" stroke="#555" stroke-width="4" />

  <!-- ================= ACTION EFFECTS ================= -->
  <!-- Motion Lines -->
  <line x1="600" y1="400" x2="700" y2="400" stroke="#FFF" stroke-width="4" stroke-linecap="round" />
  <line x1="620" y1="420" x2="720" y2="420" stroke="#FFF" stroke-width="4" stroke-linecap="round" />
  <line x1="580" y1="440" x2="680" y2="440" stroke="#FFF" stroke-width="4" stroke-linecap="round" />

  <!-- Dust Clouds -->
  <circle cx="550" cy="450" r="20" fill="#D3D3D3" opacity="0.8" />
  <circle cx="580" cy="430" r="25" fill="#D3D3D3" opacity="0.8" />
  <circle cx="620" cy="460" r="15" fill="#D3D3D3" opacity="0.8" />
  <circle cx="650" cy="440" r="18" fill="#D3D3D3" opacity="0.8" />
</svg>
```

</details>

## 2. gemma4_31b-cloud

**SVG size:** 1680 characters  
**Complexity:** Low  
**Shape elements:** 19  
**Distinct colors:** 13  
**Raw response:** 2326 characters

![gemma4_31b-cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/gemma4_31b-cloud.svg)

### Analysis

This SVG contains approximately **19 shape elements** and uses **13 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail. Best for embedding in web pages where small file size matters.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Background/Ground -->
  <rect fill="#e0f7fa" />
  <rect x="0" y="220" width="400" height="80" fill="#8bc34a" />

  <!-- Jeep Body -->
  <rect x="100" y="150" width="200" height="60" rx="10" fill="#f44336" />
  <rect x="120" y="120" width="140" height="40" rx="5" fill="#ef5350" />
  
  <!-- Windshield/Window -->
  <rect x="130" y="125" width="120" height="30" rx="3" fill="#b3e5fc" stroke="#333" stroke-width="2" />
  
  <!-- Wheels -->
  <circle cx="130" cy="210" r="25" fill="#333" />
  <circle cx="130" cy="210" r="10" fill="#bdbdbd" />
  <circle cx="270" cy="210" r="25" fill="#333" />
  <circle cx="270" cy="210" r="10" fill="#bdbdbd" />
  
  <!-- Headlights -->
  <circle cx="300" cy="170" r="8" fill="#fff176" stroke="#fbc02d" stroke-width="2" />
  
  <!-- THE DUCK -->
  <!-- Body -->
  <ellipse cx="190" cy="130" rx="30" ry="25" fill="#ffd600" />
  <!-- Head -->
  <circle cx="210" cy="110" r="18" fill="#ffd600" />
  <!-- Beak -->
  <path d="M 225 110 Q 240 110 230 120" fill="#ff9800" stroke="#e65100" stroke-width="1" />
  <!-- Eye -->
  <circle cx="218" cy="105" r="2" fill="#000" />
  
  <!-- Duck's Arms/Wings on steering wheel -->
  <path d="M 190 130 Q 210 130 215 140" fill="none" stroke="#ffd600" stroke-width="6" stroke-linecap="round" />
  
  <!-- Steering Wheel -->
  <circle cx="215" cy="140" r="12" fill="none" stroke="#333" stroke-width="3" />
  <line x1="215" y1="140" x2="215" y2="128" stroke="#333" stroke-width="3" />

  <!-- Exhaust smoke -->
  <circle cx="90" cy="180" r="5" fill="#bdbdbd" opacity="0.6" />
  <circle cx="70" cy="170" r="8" fill="#bdbdbd" opacity="0.4" />
</svg>
```

</details>

## 3. glm-5.1_cloud

**SVG size:** 17901 characters  
**Complexity:** Very high  
**Shape elements:** 160  
**Distinct colors:** 50  
**Raw response:** 21943 characters

![glm-5.1_cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/glm-5.1_cloud.svg)

### Analysis

This SVG contains approximately **160 shape elements** and uses **50 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth and shading
- Adds `<text>` labels, showing the model tried to annotate the scene
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

With over 15,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in adding fine details like the jeep body panels, terrain, sky, and the duck's accessories. Best for users who want maximum visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 550" aria-label="A cartoon duck driving a green jeep through a scenic landscape">
  <defs>
    <!-- Gradients -->
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4AA3DF"/>
      <stop offset="70%" stop-color="#A8D8EA"/>
      <stop offset="100%" stop-color="#D4E8CA"/>
    </linearGradient>
    <linearGradient id="groundGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#9B8365"/>
      <stop offset="100%" stop-color="#7A6548"/>
    </linearGradient>
    <linearGradient id="jeepBodyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5B8C3E"/>
      <stop offset="50%" stop-color="#4A7A2E"/>
      <stop offset="100%" stop-color="#3D6626"/>
    </linearGradient>
    <linearGradient id="jeepSideGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4A7A2E"/>
      <stop offset="100%" stop-color="#3D6626"/>
    </linearGradient>
    <linearGradient id="tireGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#333"/>
      <stop offset="100%" stop-color="#1a1a1a"/>
    </linearGradient>
    <linearGradient id="hubcapGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#E0E0E0"/>
      <stop offset="100%" stop-color="#999"/>
    </linearGradient>
    <radialGradient id="sunGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFE066"/>
      <stop offset="60%" stop-color="#FFD426"/>
      <stop offset="100%" stop-color="#FFAA00"/>
    </radialGradient>
    <linearGradient id="windshieldGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#B8E6FF" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="#80CCFF" stop-opacity="0.6"/>
    </linearGradient>
    <linearGradient id="duckBodyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFE066"/>
      <stop offset="100%" stop-color="#FFCC00"/>
    </linearGradient>
    <linearGradient id="beakGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#FF8C00"/>
      <stop offset="100%" stop-color="#E67600"/>
    </linearGradient>
    <linearGradient id="roadGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#8B7355"/>
      <stop offset="100%" stop-color="#6B5535"/>
    </linearGradient>
    <linearGradient id="mountainGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#6B8E6B"/>
      <stop offset="100%" stop-color="#4A7A4A"/>
    </linearGradient>

    <!-- Tire tread pattern -->
    <pattern id="treadPattern" x="0" y="0" patternUnits="userSpaceOnUse">
      <rect width="6" height="6" fill="#222"/>
      <line x1="0" y1="3" x2="6" y2="3" stroke="#333" stroke-width="1.5"/>
      <line x1="3" y1="0" x2="3" y2="6" stroke="#333" stroke-width="1"/>
    </pattern>

    <!-- Clip for wheel -->
    <clipPath id="wheelClip1">
      <circle cx="270" cy="418" r="36"/>
    </clipPath>
    <clipPath id="wheelClip2">
      <circle cx="545" cy="418" r="36"/>
    </clipPath>
  </defs>

  <!-- SKY -->
  <rect width="800" height="550" fill="url(#skyGrad)"/>

  <!-- SUN -->
  <g class="sun-group">
    <circle cx="680" cy="80" r="50" fill="url(#sunGrad)"/>
    <!-- Sun rays -->
    <g stroke="#FFD426" stroke-width="3" stroke-linecap="round" opacity="0.6">
      <line x1="680" y1="15" x2="680" y2="5"/>
      <line x1="680" y1="145" x2="680" y2="155"/>
      <line x1="615" y1="80" x2="605" y2="80"/>
      <line x1="745" y1="80" x2="755" y2="80"/>
      <line x1="634" y1="34" x2="627" y2="27"/>
      <line x1="726" y1="34" x2="733" y2="27"/>
      <line x1="634" y1="126" x2="627" y2="133"/>
      <line x1="726" y1="126" x2="733" y2="133"/>
    </g>
  </g>

  <!-- CLOUDS -->
  <g class="cloud1" opacity="0.85">
    <ellipse cx="150" cy="80" rx="50" ry="25" fill="white"/>
    <ellipse cx="120" cy="70" rx="35" ry="20" fill="white"/>
    <ellipse cx="180" cy="72" rx="30" ry="18" fill="white"/>
  </g>
  <g class="cloud2" opacity="0.7">
    <ellipse cx="450" cy="60" rx="45" ry="20" fill="white"/>
    <ellipse cx="420" cy="52" rx="30" ry="16" fill="white"/>
    <ellipse cx="475" cy="55" rx="25" ry="14" fill="white"/>
  </g>

  <!-- MOUNTAINS -->
  <polygon points="0,350 100,230 200,310 300,200 400,290 500,220 600,280 700,210 800,300 800,350" fill="url(#mountainGrad)" opacity="0.5"/>
  <polygon points="0,350 80,280 180,320 280,250 380,310 500,260 620,300 720,260 800,310 800,350" fill="#5A8A5A" opacity="0.4"/>

  <!-- GROUND -->
  <rect x="0" y="340" width="800" height="210" fill="url(#groundGrad)"/>

  <!-- Road -->
  <rect x="0" y="400" width="800" height="80" fill="#777" rx="2"/>
  <rect x="0" y="400" width="800" height="5" fill="#888"/>
  <rect x="0" y="475" width="800" height="5" fill="#666"/>
  <!-- Road dashes -->
  <g>
    <rect x="20" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="100" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="180" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="260" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="340" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="420" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="500" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="580" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="660" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
    <rect x="740" y="437" width="40" height="5" rx="2" fill="#FFD700" opacity="0.9"/>
  </g>

  <!-- Road edges -->
  <line x1="0" y1="400" x2="800" y2="400" stroke="#555" stroke-width="2"/>
  <line x1="0" y1="480" x2="800" y2="480" stroke="#555" stroke-width="2"/>

  <!-- Grass tufts on roadside -->
  <g fill="#6B8E4A" opacity="0.7">
    <path d="M50,398 Q55,380 58,398"/>
    <path d="M55,398 Q62,375 65,398"/>
    <path d="M130,398 Q135,382 138,398"/>
    <path d="M620,398 Q625,380 628,398"/>
    <path d="M625,398 Q632,375 635,398"/>
    <path d="M720,398 Q725,382 728,398"/>
  </g>

  <!-- JEEP + DUCK GROUP (bouncing) -->
  <g class="jeep-group">

    <!-- ==================== JEEP ==================== -->

    <!-- Jeep shadow -->
    <ellipse cx="410" cy="478" rx="160" ry="8" fill="rgba(0,0,0,0.25)"/>

    <!-- Jeep undercarriage -->
    <rect x="245" y="395" width="320" height="15" rx="4" fill="#2a2a2a"/>

    <!-- Jeep body - lower panel -->
    <rect x="240" y="310" width="330" height="90" rx="5" fill="url(#jeepBodyGrad)"/>

    <!-- Jeep side panel detail -->
    <rect x="245" y="355" width="320" height="8" rx="2" fill="#3D6626"/>
    <!-- Lower body accent -->
    <rect x="245" y="385" width="320" height="15" rx="3" fill="#333" opacity="0.4"/>

    <!-- Door outline -->
    <rect x="380" y="320" width="100" height="75" rx="3" fill="none" stroke="#3D6626" stroke-width="2"/>
    <!-- Door handle -->
    <rect x="460" y="355" width="16" height="5" rx="2" fill="#C0C0C0"/>

    <!-- Door mirror -->
    <rect x="370" y="325" width="8" height="14" rx="2" fill="#666"/>
    <rect x="368" y="328" width="6" height="8" rx="1" fill="#A8D8EA"/>

    <!-- Jeep rear section -->
    <rect x="540" y="300" width="40" height="100" rx="3" fill="url(#jeepSideGrad)"/>
    <!-- Tail lights -->
    <rect x="572" y="360" width="8" height="15" rx="2" fill="#FF3333"/>
    <rect x="572" y="340" width="8" height="10" rx="2" fill="#FFAA33"/>

    <!-- Jeep hood -->
    <path d="M240,310 L240,280 L380,265 L380,310 Z" fill="url(#jeepBodyGrad)" stroke="#3D6626" stroke-width="1"/>

    <!-- Hood ridges -->
    <line x1="260" y1="290" x2="370" y2="278" stroke="#3D6626" stroke-width="1.5"/>
    <line x1="260" y1="298" x2="370" y2="286" stroke="#3D6626" stroke-width="1.5"/>

    <!-- Jeep front fender -->
    <path d="M240,310 Q220,310 220,340 L220,380 Q220,400 240,400 L240,310 Z" fill="url(#jeepBodyGrad)" stroke="#3D6626" stroke-width="1"/>

    <!-- Front bumper -->
    <rect x="218" y="388" width="30" height="12" rx="3" fill="#555"/>

    <!-- Front grille (iconic Jeep 7-slot) -->
    <rect x="224" y="318" width="18" height="65" rx="3" fill="#333"/>
    <g fill="#666">
      <rect x="228" y="325" width="10" height="6" rx="1"/>
      <rect x="228" y="336" width="10" height="6" rx="1"/>
      <rect x="228" y="347" width="10" height="6" rx="1"/>
      <rect x="228" y="358" width="10" height="6" rx="1"/>
      <rect x="228" y="369" width="10" height="6" rx="1"/>
      <rect x="228" y="380" width="10" height="5" rx="1"/>
    </g>

    <!-- Headlights -->
    <circle cx="232" cy="308" r="12" fill="#FFE066" stroke="#CCC" stroke-width="2"/>
    <circle cx="232" cy="308" r="7" fill="#FFF8DC"/>
    <circle cx="232" cy="306" r="3" fill="white" opacity="0.8"/>

    <!-- Jeep windshield frame -->
    <rect x="365" y="225" width="120" height="90" rx="5" fill="#444" stroke="#333" stroke-width="2"/>
    <!-- Windshield glass -->
    <rect x="370" y="230" width="110" height="80" rx="3" fill="url(#windshieldGrad)"/>
    <!-- Windshield reflection -->
    <path d="M375,235 L410,235 L375,270 Z" fill="white" opacity="0.2"/>
    <!-- Windshield divider -->
    <line x1="425" y1="230" x2="425" y2="310" stroke="#444" stroke-width="3"/>

    <!-- Roll bars -->
    <rect x="370" y="222" width="8" height="95" rx="3" fill="#555"/>
    <rect x="490" y="222" width="8" height="95" rx="3" fill="#555"/>
    <rect x="370" y="222" width="128" height="8" rx="3" fill="#555"/>
    <!-- Roll bar diagonal -->
    <line x1="378" y1="226" x2="490" y2="226" stroke="#666" stroke-width="2"/>

    <!-- Rear roll bar -->
    <rect x="530" y="240" width="8" height="70" rx="3" fill="#555"/>
    <rect x="378" y="230" width="160" height="8" rx="3" fill="#555" opacity="0.5"/>

    <!-- Jeep roof (open top - just a roll bar structure) -->

    <!-- Steering wheel (visible through windshield) -->
    <circle cx="410" cy="295" r="16" fill="none" stroke="#333" stroke-width="4"/>
    <circle cx="410" cy="295" r="12" fill="none" stroke="#555" stroke-width="2"/>
    <!-- Steering column -->
    <line x1="410" y1="305" x2="410" y2="320" stroke="#333" stroke-width="4"/>

    <!-- Dashboard -->
    <rect x="370" y="308" width="110" height="8" rx="2" fill="#333"/>
    <!-- Dashboard gauges -->
    <circle cx="395" cy="311" r="4" fill="#222" stroke="#888" stroke-width="1"/>
    <circle cx="410" cy="311" r="4" fill="#222" stroke="#888" stroke-width="1"/>
    <circle cx="425" cy="311" r="3" fill="#222" stroke="#888" stroke-width="1"/>

    <!-- ==================== DUCK ==================== -->
    <g class="duck-head-group">

      <!-- Duck body (sitting in jeep) -->
      <ellipse cx="400" cy="295" rx="38" ry="30" fill="url(#duckBodyGrad)"/>
      <!-- Duck body highlight -->
      <ellipse cx="395" cy="288" rx="20" ry="15" fill="#FFE87C" opacity="0.5"/>

      <!-- Duck wing (holding steering wheel area) -->
      <path d="M380,290 Q360,280 355,270 Q360,285 375,295" fill="#FFCC00" stroke="#E6B800" stroke-width="1"/>
      <!-- Wing detail line -->
      <path d="M365,278 Q372,285 378,292" fill="none" stroke="#E6B800" stroke-width="1"/>

      <!-- Other wing (right side) -->
      <path d="M430,290 Q445,282 448,275 Q440,288 428,298" fill="#FFCC00" stroke="#E6B800" stroke-width="1"/>

      <!-- Duck neck -->
      <ellipse cx="390" cy="265" rx="15" ry="20" fill="#FFE066"/>

      <!-- Duck head -->
      <ellipse cx="385" cy="245" rx="22" ry="20" fill="#FFE066"/>
      <!-- Head highlight -->
      <ellipse cx="380" cy="240" rx="12" ry="10" fill="#FFE87C" opacity="0.4"/>

      <!-- Duck cap (driving cap) -->
      <path d="M363,240 Q365,222 385,218 Q405,222 407,240 L408,238 Q410,232 405,225 Q395,215 385,214 Q375,215 365,225 Q360,232 362,238 Z" fill="#C0392B"/>
      <!-- Cap brim -->
      <path d="M362,238 Q360,242 368,244 L408,240 Q412,236 410,232 L408,238 Z" fill="#A93226"/>
      <!-- Cap button -->
      <circle cx="385" cy="217" r="3" fill="#E74C3C"/>

      <!-- Duck eyes -->
      <ellipse cx="376" cy="242" rx="6" ry="7" fill="white"/>
      <ellipse cx="394" cy="242" rx="6" ry="7" fill="white"/>
      <!-- Pupils -->
      <ellipse cx="378" cy="243" rx="3.5" ry="4" fill="#222"/>
      <ellipse cx="396" cy="243" rx="3.5" ry="4" fill="#222"/>
      <!-- Eye highlights -->
      <circle cx="380" cy="241" r="1.5" fill="white"/>
      <circle cx="398" cy="241" r="1.5" fill="white"/>
      <!-- Eyebrows (determined look) -->
      <path d="M370,237 Q376,234 382,237" fill="none" stroke="#8B6914" stroke-width="2" stroke-linecap="round"/>
      <path d="M388,237 Q394,234 400,237" fill="none" stroke="#8B6914" stroke-width="2" stroke-linecap="round"/>

      <!-- Duck beak -->
      <path d="M368,250 Q380,248 385,252 Q380,256 368,254 Z" fill="url(#beakGrad)"/>
      <!-- Beak nostril -->
      <circle cx="374" cy="251" r="1.5" fill="#CC7000"/>
      <!-- Mouth line -->
      <line x1="370" y1="252" x2="383" y2="252" stroke="#CC7000" stroke-width="1"/>

      <!-- Duck scarf (flying in wind) -->
      <path d="M400,262 Q420,258 430,265 Q435,268 428,272" fill="#E74C3C" stroke="#C0392B" stroke-width="1"/>
      <!-- Scarf end fluttering -->
      <g class="scarf-end">
        <path d="M428,272 Q445,265 455,275 Q460,280 450,285 Q440,278 428,272" fill="#E74C3C" stroke="#C0392B" stroke-width="1"/>
        <path d="M450,285 Q465,278 472,288 Q470,295 460,290" fill="#C0392B"/>
      </g>

      <!-- Duck tail feathers (sticking up behind) -->
      <path d="M420,280 Q430,265 425,255 Q420,265 418,275" fill="#FFCC00" stroke="#E6B800" stroke-width="1"/>
      <path d="M422,278 Q435,260 430,248 Q424,262 420,275" fill="#FFE066" stroke="#E6B800" stroke-width="1"/>

    </g><!-- end duck-head-group -->

  </g><!-- end jeep-group -->

  <!-- WHEELS (not bouncing with jeep, staying on road) -->
  <!-- Rear wheel -->
  <g>
    <circle cx="270" cy="418" r="38" fill="url(#tireGrad)"/>
    <circle cx="270" cy="418" r="38" fill="url(#treadPattern)" opacity="0.5"/>
    <circle cx="270" cy="418" r="30" fill="#333"/>
    <circle cx="270" cy="418" r="24" fill="url(#hubcapGrad)"/>
    <!-- Spokes -->
    <g class="wheel-spokes" style="transform-origin: 270px 418px;">
      <line x1="270" y1="396" x2="270" y2="440" stroke="#888" stroke-width="2"/>
      <line x1="248" y1="418" x2="292" y2="418" stroke="#888" stroke-width="2"/>
      <line x1="254" y1="402" x2="286" y2="434" stroke="#888" stroke-width="2"/>
      <line x1="286" y1="402" x2="254" y2="434" stroke="#888" stroke-width="2"/>
    </g>
    <circle cx="270" cy="418" r="6" fill="#555"/>
    <circle cx="270" cy="418" r="3" fill="#777"/>
    <!-- Center cap -->
    <circle cx="270" cy="418" r="8" fill="none" stroke="#AAA" stroke-width="1"/>
  </g>

  <!-- Front wheel -->
  <g>
    <circle cx="545" cy="418" r="38" fill="url(#tireGrad)"/>
    <circle cx="545" cy="418" r="38" fill="url(#treadPattern)" opacity="0.5"/>
    <circle cx="545" cy="418" r="30" fill="#333"/>
    <circle cx="545" cy="418" r="24" fill="url(#hubcapGrad)"/>
    <!-- Spokes -->
    <g class="wheel-spokes" style="transform-origin: 545px 418px;">
      <line x1="545" y1="396" x2="545" y2="440" stroke="#888" stroke-width="2"/>
      <line x1="523" y1="418" x2="567" y2="418" stroke="#888" stroke-width="2"/>
      <line x1="529" y1="402" x2="561" y2="434" stroke="#888" stroke-width="2"/>
      <line x1="561" y1="402" x2="529" y2="434" stroke="#888" stroke-width="2"/>
    </g>
    <circle cx="545" cy="418" r="6" fill="#555"/>
    <circle cx="545" cy="418" r="3" fill="#777"/>
    <circle cx="545" cy="418" r="8" fill="none" stroke="#AAA" stroke-width="1"/>
  </g>

  <!-- DUST TRAIL behind jeep -->
  <g>
    <circle class="dust1" cx="180" cy="420" r="8" fill="#B8A88A" opacity="0.5"/>
    <circle class="dust2" cx="160" cy="410" r="12" fill="#B8A88A" opacity="0.4"/>
    <circle class="dust3" cx="140" cy="415" r="10" fill="#B8A88A" opacity="0.3"/>
    <ellipse class="dust1" cx="120" cy="425" rx="20" ry="8" fill="#B8A88A" opacity="0.25"/>
  </g>

  <!-- Small trees/bushes in background -->
  <g opacity="0.6">
    <!-- Tree left -->
    <rect x="68" y="355" width="6" height="20" fill="#5D4037"/>
    <ellipse cx="71" cy="350" rx="14" ry="16" fill="#4A7A4A"/>
    <ellipse cx="66" cy="348" rx="10" ry="12" fill="#5B8C3E"/>

    <!-- Bush right -->
    <ellipse cx="690" cy="370" rx="18" ry="12" fill="#4A7A4A"/>
    <ellipse cx="700" cy="368" rx="12" ry="10" fill="#5B8C3E"/>
    <rect x="706" y="370" width="4" height="15" fill="#5D4037"/>

    <!-- Small bush -->
    <ellipse cx="50" cy="375" rx="10" ry="7" fill="#5B8C3E"/>
    <ellipse cx="760" cy="380" rx="12" ry="8" fill="#4A7A4A"/>
  </g>

  <!-- Small flowers on roadside -->
  <g>
    <circle cx="100" cy="395" r="3" fill="#FF6B6B"/>
    <circle cx="100" cy="395" r="1.5" fill="#FFE066"/>
    <circle cx="650" cy="393" r="3" fill="#FF6B6B"/>
    <circle cx="650" cy="393" r="1.5" fill="#FFE066"/>
    <circle cx="700" cy="396" r="2.5" fill="#E74C3C"/>
    <circle cx="700" cy="396" r="1.2" fill="#FFE066"/>
  </g>

  <!-- "QUACK!" text bubble -->
  <g>
    <rect x="300" y="170" width="90" height="35" rx="12" fill="white" stroke="#333" stroke-width="2"/>
    <polygon points="370,205 385,220 360,205" fill="white" stroke="#333" stroke-width="2"/>
    <polygon points="370,205 385,220 361,205" fill="white"/>
    <text x="345" y="194" font-family="'Comic Sans MS', cursive, sans-serif" font-size="18" font-weight="bold" fill="#C0392B" text-anchor="middle">QUACK!</text>
  </g>

  <!-- License plate -->
  <rect x="265" y="395" width="40" height="12" rx="2" fill="white" stroke="#333" stroke-width="1"/>
  <text x="285" y="404" font-family="monospace" font-size="7" font-weight="bold" fill="#333" text-anchor="middle">QUACK</text>

</svg>
```

</details>

## 4. glm-5.2_cloud

**SVG size:** 6996 characters  
**Complexity:** Medium  
**Shape elements:** 66  
**Distinct colors:** 26  
**Raw response:** 7034 characters

![glm-5.2_cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/glm-5.2_cloud.svg)

### Analysis

This SVG contains approximately **66 shape elements** and uses **26 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth and shading
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable. Good for general use.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">
  <defs>
    <!-- Sky Gradient -->
    <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#87CEEB" />
      <stop offset="100%" stop-color="#E0F6FF" />
    </linearGradient>
    
    <!-- Jeep Body Gradient -->
    <linearGradient id="jeepGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#008080" />
      <stop offset="100%" stop-color="#005959" />
    </linearGradient>
    
    <!-- Tire Gradient -->
    <radialGradient id="tireGrad" cx="50%" cy="50%" r="50%">
      <stop offset="60%" stop-color="#1A1A1A" />
      <stop offset="100%" stop-color="#000000" />
    </radialGradient>
  </defs>

  <!-- Sky Background -->
  <rect x="0" y="0" width="800" height="500" fill="url(#skyGrad)" />

  <!-- Sun -->
  <circle cx="680" cy="100" r="45" fill="#FFD700" opacity="0.8" />
  <circle cx="680" cy="100" r="60" fill="#FFD700" opacity="0.3" />

  <!-- Clouds -->
  <path d="M 150 120 Q 180 90 220 120 Q 260 100 280 130 Q 260 150 150 120 Z" fill="#FFFFFF" opacity="0.8" />
  <path d="M 450 80 Q 480 50 520 80 Q 560 60 580 90 Q 560 110 450 80 Z" fill="#FFFFFF" opacity="0.8" />

  <!-- Hills -->
  <path d="M 0 350 Q 200 280 400 350 Q 600 320 800 350 L 800 500 L 0 500 Z" fill="#556B2F" />
  <path d="M 0 380 Q 150 330 350 380 Q 550 350 800 380 L 800 500 L 0 500 Z" fill="#6B8E23" />

  <!-- Dirt Road -->
  <path d="M 0 420 L 800 420 L 800 500 L 0 500 Z" fill="#8B5A2B" />
  <path d="M 0 420 L 800 420 L 800 430 L 0 430 Z" fill="#A0522D" />

  <!-- Dust Trail behind Jeep -->
  <ellipse cx="120" cy="400" rx="40" ry="15" fill="#8B5A2B" opacity="0.6" />
  <ellipse cx="80" cy="390" rx="50" ry="20" fill="#8B5A2B" opacity="0.4" />
  <ellipse cx="40" cy="380" rx="30" ry="15" fill="#8B5A2B" opacity="0.2" />

  <!-- Jeep Shadow -->
  <ellipse cx="400" cy="410" rx="180" ry="20" fill="#000000" opacity="0.3" />

  <!-- Back Wheel -->
  <g id="backWheel">
    <circle cx="280" cy="380" r="55" fill="url(#tireGrad)" />
    <circle cx="280" cy="380" r="30" fill="#333333" />
    <circle cx="280" cy="380" r="25" fill="#555555" />
    <!-- Spokes -->
    <line x1="280" y1="355" x2="280" y2="405" stroke="#222" stroke-width="6" />
    <line x1="255" y1="380" x2="305" y2="380" stroke="#222" stroke-width="6" />
    <line x1="262" y1="362" x2="298" y2="398" stroke="#222" stroke-width="6" />
    <line x1="298" y1="362" x2="262" y2="398" stroke="#222" stroke-width="6" />
    <circle cx="280" cy="380" r="8" fill="#111" />
  </g>

  <!-- Front Wheel -->
  <g id="frontWheel">
    <circle cx="520" cy="380" r="55" fill="url(#tireGrad)" />
    <circle cx="520" cy="380" r="30" fill="#333333" />
    <circle cx="520" cy="380" r="25" fill="#555555" />
    <!-- Spokes -->
    <line x1="520" y1="355" x2="520" y2="405" stroke="#222" stroke-width="6" />
    <line x1="495" y1="380" x2="545" y2="380" stroke="#222" stroke-width="6" />
    <line x1="502" y1="362" x2="538" y2="398" stroke="#222" stroke-width="6" />
    <line x1="538" y1="362" x2="502" y2="398" stroke="#222" stroke-width="6" />
    <circle cx="520" cy="380" r="8" fill="#111" />
  </g>

  <!-- Jeep Body -->
  <path d="M 230 380 
           A 55 55 0 0 0 330 380 
           L 470 380 
           A 55 55 0 0 0 570 380 
           L 570 320 
           L 540 320 
           L 510 260 
           L 260 260 
           L 230 320 
           Z" 
        fill="url(#jeepGrad)" stroke="#003333" stroke-width="3" />

  <!-- Jeep Lower Body Details -->
  <path d="M 230 380 A 55 55 0 0 0 330 380 L 470 380 A 55 55 0 0 0 570 380" fill="none" stroke="#222" stroke-width="5" />
  <rect x="240" y="350" width="320" height="15" fill="#003333" opacity="0.5" />

  <!-- Jeep Fenders -->
  <path d="M 225 380 A 60 60 0 0 0 335 380 L 330 380 A 55 55 0 0 1 230 380 Z" fill="#222" />
  <path d="M 465 380 A 60 60 0 0 0 575 380 L 570 380 A 55 55 0 0 1 470 380 Z" fill="#222" />

  <!-- Front Grille and Headlights -->
  <rect x="540" y="290" width="30" height="60" rx="4" fill="#111" />
  <line x1="545" y1="295" x2="545" y2="345" stroke="#444" stroke-width="2" />
  <line x1="550" y1="295" x2="550" y2="345" stroke="#444" stroke-width="2" />
  <line x1="555" y1="295" x2="555" y2="345" stroke="#444" stroke-width="2" />
  <line x1="560" y1="295" x2="560" y2="345" stroke="#444" stroke-width="2" />
  <line x1="565" y1="295" x2="565" y2="345" stroke="#444" stroke-width="2" />
  <circle cx="545" cy="310" r="8" fill="#FFD700" stroke="#222" stroke-width="2" />

  <!-- Hood Detail -->
  <path d="M 510 260 L 540 320 L 480 320 L 480 260 Z" fill="#006666" stroke="#003333" stroke-width="2" />

  <!-- Windshield -->
  <polygon points="480,260 510,260 510,190 480,190" fill="rgba(200, 220, 255, 0.7)" stroke="#333" stroke-width="4" />
  <line x1="495" y1="190" x2="495" y2="260" stroke="#333" stroke-width="3" />

  <!-- Roll Cage -->
  <path d="M 350 380 L 350 190 L 460 190 L 460 380" fill="none" stroke="#222" stroke-width="8" stroke-linejoin="round" />
  <path d="M 350 190 L 480 190" fill="none" stroke="#222" stroke-width="8" stroke-linecap="round" />

  <!-- Side Mirror -->
  <path d="M 510 210 L 530 210 L 530 230 L 510 230 Z" fill="#222" />
  <circle cx="525" cy="220" r="8" fill="#CCCCCC" />

  <!-- Door Handle -->
  <rect x="380" y="300" width="20" height="6" rx="3" fill="#222" />

  <!-- Interior & Seat -->
  <rect x="340" y="280" width="80" height="100" fill="#222" />
  <rect x="340" y="270" width="80" height="20" rx="5" fill="#333" />

  <!-- Steering Wheel -->
  <circle cx="420" cy="250" r="25" fill="none" stroke="#111" stroke-width="6" />
  <circle cx="420" cy="250" r="6" fill="#111" />
  <line x1="420" y1="225" x2="420" y2="275" stroke="#111" stroke-width="6" />
  <line x1="395" y1="250" x2="445" y2="250" stroke="#111" stroke-width="6" />

  <!-- Duck Body -->
  <ellipse cx="370" cy="230" rx="35" ry="45" fill="#FFD700" stroke="#E6B800" stroke-width="2" />
  
  <!-- Duck Wing (Reaching for steering wheel) -->
  <path d="M 370 220 Q 390 240 415 250" stroke="#FFD700" stroke-width="18" fill="none" stroke-linecap="round" />
  <path d="M 370 220 Q 390 240 415 250" stroke="#E6B800" stroke-width="18" fill="none" stroke-linecap="round" opacity="0.3" />
  
  <!-- Duck Head -->
  <circle cx="370" cy="170" r="28" fill="#FFD700" stroke="#E6B800" stroke-width="2" />
  
  <!-- Duck Beak -->
  <path d="M 345 165 Q 320 170 345 185 Z" fill="#FFA500" stroke="#CC8400" stroke-width="1" />
  
  <!-- Duck Eye -->
  <circle cx="385" cy="165" r="6" fill="#FFFFFF" />
  <circle cx="386" cy="166" r="3" fill="#000000" />
  
  <!-- Duck Cap -->
  <path d="M 345 185 A 28 28 0 0 1 395 185 Z" fill="#D11B1B" />
  <path d="M 370 175 L 405 165 L 395 185 Z" fill="#D11B1B" stroke="#A01212" stroke-width="1" />
  <rect x="345" y="183" width="50" height="4" fill="#A01212" />

  <!-- Duck Tail Feathers sticking out -->
  <path d="M 335 250 Q 310 260 330 270 Z" fill="#FFD700" stroke="#E6B800" stroke-width="1" />

</svg>
```

</details>

## 5. kimi-k2.6_cloud

**SVG size:** 9966 characters  
**Complexity:** Medium  
**Shape elements:** 51  
**Distinct colors:** 32  
**Raw response:** 11079 characters

![kimi-k2.6_cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/kimi-k2.6_cloud.svg)

### Analysis

This SVG contains approximately **51 shape elements** and uses **32 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth and shading
- Applies SVG filters (e.g., drop shadows, blurs) for visual effects
- Uses transforms (translate, rotate, scale) for precise positioning
- Includes animation elements -- the model attempted motion effects
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable. Good for general use.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 600" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <!-- Sky Gradient -->
            <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#4facfe;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#00f2fe;stop-opacity:1" />
            </linearGradient>

            <!-- Ground Gradient -->
            <linearGradient id="groundGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#e6d16c;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#d4b53c;stop-opacity:1" />
            </linearGradient>

            <!-- Jeep Body Gradient -->
            <linearGradient id="jeepGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#4caf50;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#2e7d32;stop-opacity:1" />
            </linearGradient>

            <!-- Metal Gradient -->
            <linearGradient id="metalGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#9e9e9e;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#616161;stop-opacity:1" />
            </linearGradient>
            
            <!-- Shadow Filter -->
            <filter id="dropShadow" x="-20%" y="-20%">
                <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
                <feOffset dx="2" dy="2" result="offsetblur"/>
                <feComponentTransfer>
                    <feFuncA type="linear" slope="0.3"/>
                </feComponentTransfer>
                <feMerge> 
                    <feMergeNode/>
                    <feMergeNode in="SourceGraphic"/> 
                </feMerge>
            </filter>
        </defs>

        <!-- Background: Sky and Sun -->
        <rect x="0" y="0" width="800" height="600" fill="url(#skyGradient)" />
        <circle cx="700" cy="100" r="60" fill="#FFD700" opacity="0.8" />
        
        <!-- Background: Moving Clouds -->
        <g fill="#ffffff" opacity="0.6">
            <path d="M100,100 Q120,80 140,100 T180,100 T220,100 Q200,130 180,120 T100,100" />
            <path d="M500,150 Q520,130 540,150 T580,150 T620,150 Q600,180 580,170 T500,150" />
            <path d="M300,80 Q320,60 340,80 T380,80 T420,80 Q400,110 380,100 T300,80" />
        </g>

        <!-- Background: Distant Mountains -->
        <path d="M0,450 L150,300 L300,450 L450,320 L600,450 L800,380 L800,450 L0,450" fill="#5D4037" opacity="0.3" />
        <path d="M0,450 L100,350 L250,450 L500,350 L800,450 Z" fill="#795548" opacity="0.5" />

        <!-- Ground -->
        <rect x="0" y="450" width="800" height="150" fill="url(#groundGradient)" />
        <!-- Road/Path -->
        <path d="M0,500 C200,490 600,490 800,500 L800,600 L0,600 Z" fill="#8d6e63" />
        <path d="M0,510 L800,510 M0,530 L800,530 M0,550 L800,550" stroke="#6d4c41" stroke-width="2" stroke-dasharray="20,20" />

        <!-- THE JEEP & DRIVER ASSEMBLY (Centered) -->
        <g transform="translate(150, 180)">
            
            <!-- Back Wheel (Left Side) -->
            <g transform="translate(60, 240)">
                <circle cx="0" cy="0" r="45" fill="#212121" /> <!-- Tire -->
                <circle cx="0" cy="0" r="25" fill="#bdbdbd" stroke="#757575" stroke-width="2"/> <!-- Rim -->
                <circle cx="0" cy="0" r="5" fill="#424242" /> <!-- Hub -->
            </g>

            <!-- Back Wheel (Right Side - Far side, slightly smaller/darker for perspective) -->
            <g transform="translate(380, 240)">
                <circle cx="0" cy="0" r="45" fill="#212121" opacity="0.9"/>
                <circle cx="0" cy="0" r="25" fill="#9e9e9e" stroke="#616161" stroke-width="2"/>
            </g>

            <!-- Jeep Chassis Shadow -->
            <ellipse cx="220" cy="280" rx="200" ry="20" fill="black" opacity="0.3" />

            <!-- Jeep Body -->
            <!-- Main shape -->
            <path d="M20,240 L20,160 L100,160 L120,100 L380,100 L380,240 Z" fill="url(#jeepGradient)" stroke="#1b5e20" stroke-width="3" />
            
            <!-- Interior / Back area -->
            <path d="M120,160 L120,100 L380,100 L380,160 Z" fill="#1b5e20" />
            
            <!-- Roll Bar -->
            <path d="M140,160 L140,40 L340,40 L340,160" fill="none" stroke="#424242" stroke-width="8" stroke-linecap="round" />
            <line x1="140" y1="50" x2="340" y2="50" stroke="#424242" stroke-width="6" />

            <!-- Windshield Frame -->
            <path d="M120,100 L100,160 L120,160 Z" fill="#388e3c" />
            
            <!-- Front Grill/Bumper -->
            <rect x="0" y="220" width="20" height="40" fill="#424242" rx="2" />
            <rect x="10" y="240" width="20" height="20" fill="#616161" rx="2" />
            
            <!-- Headlight -->
            <circle cx="20" cy="180" r="12" fill="#ffeb3b" stroke="#fbc02d" stroke-width="2" />
            <path d="M32,175 L150,150" stroke="#ffeb3b" stroke-width="0" fill="url(#lightBeam)" opacity="0.4"/> 
            
            <!-- Tail Light -->
            <rect x="380" y="180" width="8" height="20" fill="#d32f2f" />

            <!-- Driver's Seat (Inside) -->
            <path d="M180,240 L180,150 L220,150 L220,240" fill="#795548" />
            <path d="M180,150 L180,100 L220,100 L220,150" fill="#5d4037" />

            <!-- THE DUCK DRIVER -->
            <g id="duckDriver" transform="translate(200, 90)">
                <!-- Body -->
                <ellipse cx="0" cy="60" rx="30" ry="40" fill="#ffeb3b" />
                
                <!-- Wing (on steering wheel) -->
                <path d="M-20,60 Q-40,50 -35,30" fill="none" stroke="#fbc02d" stroke-width="12" stroke-linecap="round" />

                <!-- Steering Wheel (Foreground) -->
                <ellipse cx="-25" cy="80" rx="10" ry="25" fill="none" stroke="#424242" stroke-width="6" transform="rotate(-20)" />

                <!-- Head -->
                <circle cx="0" cy="0" r="35" fill="#ffeb3b" />

                <!-- Hat (Adventure Style) -->
                <path d="M-36,-10 Q0,-50 36,-10 Z" fill="#d84315" /> <!-- Top -->
                <path d="M-40,-10 L40,-10 L45,0 L-45,0 Z" fill="#bf360c" /> <!-- Brim -->
                <rect x="-5" y="-45" width="10" height="8" fill="#bf360c" rx="2" /> <!-- Button -->

                <!-- Beak -->
                <path d="M20,-5 Q50,-5 55,5 Q50,15 20,15" fill="#ff9800" stroke="#ef6c00" stroke-width="1" />
                
                <!-- Eye -->
                <circle cx="10" cy="-10" r="5" fill="white" />
                <circle cx="12" cy="-10" r="2" fill="black" />
                
                <!-- Smile -->
                <path d="M10,15 Q20,20 25,15" fill="none" stroke="#f57f17" stroke-width="2" stroke-linecap="round" />
            </g>

            <!-- Windshield Glass (Transparent overlay) -->
            <path d="M120,100 L100,160 L120,160 Z" fill="#add8e6" opacity="0.4" />
            <path d="M380,100 L380,160 L120,160" fill="#add8e6" opacity="0.3" />

            <!-- Front Wheel (Foreground) -->
            <g transform="translate(60, 240)">
                <!-- Wheel bounce animation via SMIL (Internal SVG animation, no CSS) -->
                <animateTransform attributeName="transform" type="translate" values="60,240; 60,235; 60,240" dur="0.4s" repeatCount="indefinite" additive="replace"/>
                
                <!-- Tire Tread Detail -->
                <circle cx="0" cy="0" r="45" fill="#212121" />
                <circle cx="0" cy="0" r="40" fill="none" stroke="#424242" stroke-width="4" stroke-dasharray="10,5" />
                <circle cx="0" cy="0" r="25" fill="#bdbdbd" stroke="#757575" stroke-width="2"/>
                <circle cx="0" cy="0" r="5" fill="#424242" />
                
                <!-- Spokes -->
                <line x1="-25" y1="0" x2="25" y2="0" stroke="#757575" stroke-width="4" />
                <line x1="0" y1="-25" x2="0" y2="25" stroke="#757575" stroke-width="4" />
            </g>
            
            <!-- Motion Lines -->
            <g stroke="#fff" stroke-width="2" stroke-linecap="round" opacity="0.6">
                <path d="M-50,250 L-100,250">
                    <animate attributeName="d" values="M-50,250 L-100,250; M-50,250 L-120,250" dur="0.2s" repeatCount="indefinite" />
                </path>
                <path d="M-30,280 L-80,280">
                    <animate attributeName="d" values="M-30,280 L-80,280; M-30,280 L-100,280" dur="0.25s" repeatCount="indefinite" />
                </path>
            </g>

            <!-- Exhaust Smoke -->
            <g transform="translate(390, 240)">
                <circle cx="0" cy="0" r="5" fill="#eee" opacity="0">
                    <animate attributeName="cx" values="0; 40" dur="1s" repeatCount="indefinite" />
                    <animate attributeName="cy" values="0; -20" dur="1s" repeatCount="indefinite" />
                    <animate attributeName="r" values="5; 15" dur="1s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.8; 0" dur="1s" repeatCount="indefinite" />
                </circle>
                <circle cx="0" cy="0" r="5" fill="#eee" opacity="0">
                    <animate attributeName="cx" values="0; 40" dur="1s" begin="0.5s" repeatCount="indefinite" />
                    <animate attributeName="cy" values="0; -20" dur="1s" begin="0.5s" repeatCount="indefinite" />
                    <animate attributeName="r" values="5; 15" dur="1s" begin="0.5s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.8; 0" dur="1s" begin="0.5s" repeatCount="indefinite" />
                </circle>
            </g>

        </g>
    </svg>
```

</details>

## 6. minimax-m2.7_cloud

**SVG size:** 2722 characters  
**Complexity:** Low  
**Shape elements:** 26  
**Distinct colors:** 15  
**Raw response:** 2904 characters

![minimax-m2.7_cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/minimax-m2.7_cloud.svg)

### Analysis

This SVG contains approximately **26 shape elements** and uses **15 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail. Best for embedding in web pages where small file size matters.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">
  <!-- Background: Sky and Grass -->
  <rect fill="#87CEEB" />
  <rect x="0" y="300" width="600" height="100" fill="#7CFC00" />
  
  <!-- Clouds -->
  <path d="M100,80 Q120,50 150,80 T200,80 T250,80" fill="none" stroke="white" stroke-width="5" stroke-linecap="round" opacity="0.8"/>
  <path d="M400,60 Q420,30 450,60 T500,60" fill="none" stroke="white" stroke-width="5" stroke-linecap="round" opacity="0.8"/>

  <!-- Jeep Group -->
  <g transform="translate(100, 100)">
    
    <!-- Jeep Wheels (Back) -->
    <circle cx="80" cy="200" r="35" fill="#333" />
    <circle cx="80" cy="200" r="15" fill="#AAA" /> <!-- Hub -->
    
    <circle cx="320" cy="200" r="35" fill="#333" />
    <circle cx="320" cy="200" r="15" fill="#AAA" /> <!-- Hub -->

    <!-- Spare Tire on back -->
    <circle cx="20" cy="120" r="25" fill="#333" />
    <circle cx="20" cy="120" r="10" fill="#AAA" />
    
    <!-- Jeep Body -->
    <!-- Main chassis and cabin shape -->
    <path d="M10,180 L10,100 L150,100 L200,20 L360,20 L400,100 L410,100 L410,180 L380,180 L380,210 L360,210 L360,180 L140,180 L140,210 L120,210 L120,180 Z" fill="#4CAF50" stroke="#2E7D32" stroke-width="3"/>
    
    <!-- Wheel Wells -->
    <path d="M60,180 A40,40 0 0,1 100,180" fill="#388E3C" />
    <path d="M300,180 A40,40 0 0,1 340,180" fill="#388E3C" />

    <!-- Windshield and Side Window -->
    <path d="M145,100 L195,25 L350,25 L370,100 Z" fill="#B3E5FC" stroke="#0288D1" stroke-width="2"/>
    <line x1="300" y1="25" x2="300" y2="100" stroke="#0288D1" stroke-width="2" />

    <!-- Headlight -->
    <circle cx="405" cy="110" r="8" fill="#FFEB3B" stroke="#FBC02D" stroke-width="2"/>

    <!-- Grille -->
    <line x1="400" y1="125" x2="400" y2="175" stroke="#333" stroke-width="2"/>
    <line x1="405" y1="125" x2="405" y2="175" stroke="#333" stroke-width="2"/>

    <!-- Steering Wheel -->
    <ellipse cx="260" cy="90" rx="10" ry="30" fill="none" stroke="#333" stroke-width="4" transform="rotate(-20 260 90)"/>

    <!-- The Duck -->
    <g transform="translate(220, 50)">
      <!-- Wing (reaching for wheel) -->
      <path d="M10,40 Q-10,60 20,70" fill="#FBC02D" />
      
      <!-- Body -->
      <ellipse cx="20" cy="60" rx="30" ry="40" fill="#FFD700" />
      
      <!-- Head -->
      <circle cx="30" cy="30" r="25" fill="#FFD700" />
      
      <!-- Beak -->
      <path d="M50,30 L70,35 L50,40 Z" fill="#FF9800" />
      
      <!-- Eye -->
      <circle cx="40" cy="25" r="3" fill="#000" />
      
      <!-- Cap (Optional fun detail) -->
      <path d="M15,10 Q30,-15 45,10 Z" fill="#F44336" />
      <rect x="10" y="8" width="40" height="5" fill="#F44336" />
    </g>
  </g>
</svg>
```

</details>

## 7. nemotron-3-super_cloud

**SVG size:** 1807 characters  
**Complexity:** Low  
**Shape elements:** 15  
**Distinct colors:** 10  
**Raw response:** 2447 characters

![nemotron-3-super_cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/nemotron-3-super_cloud.svg)

### Analysis

This SVG contains approximately **15 shape elements** and uses **10 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail. Best for embedding in web pages where small file size matters.

<details><summary>View raw SVG source</summary>

```xml
<svg
     viewBox="0 0 300 200"
     xmlns="http://www.w3.org/2000/svg"
     version="1.1">

  <!-- Background -->
  <rect width="300" height="200" fill="#cce6ff"/>

  <!-- Ground -->
  <rect x="0" y="170" width="300" height="30" fill="#8b5a2b"/>

  <!-- Jeep Body -->
  <g id="jeep" transform="translate(50,120)">
    <!-- Jeep chassis -->
    <rect x="0" y="0" width="200" height="60"
          fill="#004080" stroke="#000" stroke-width="2"/>
    <!-- Cab roof -->
    <rect x="30" y="-20" width="140" height="20"
          fill="#004080" stroke="#000" stroke-width="2"/>
    <!-- Windshield -->
    <rect x="40" y="-18" width="120" height="16"
          fill="#add8e6" stroke="#000" stroke-width="1"/>
    <!-- Wheels -->
    <circle cx="30" cy="55" r="12" fill="#222" stroke="#000" stroke-width="2"/>
    <circle cx="170" cy="55" r="12" fill="#222" stroke="#000" stroke-width="2"/>
    <!-- Wheel hubs -->
    <circle cx="30" cy="55" r="4" fill="#fff"/>
    <circle cx="170" cy="55" r="4" fill="#fff"/>
  </g>

  <!-- Duck -->
  <g id="duck" transform="translate(130,90) scale(0.8)">
    <!-- Body -->
    <ellipse cx="0" cy="0" rx="30" ry="20" fill="#ffcc00" stroke="#000" stroke-width="2"/>
    <!-- Head -->
    <circle cx="-30" cy="-10" r="15" fill="#ffcc00" stroke="#000" stroke-width="2"/>
    <!-- Beak -->
    <polygon points="-45,-10 -55,-5 -45,0"
             fill="#ff8800" stroke="#000" stroke-width="2"/>
    <!-- Eye -->
    <circle cx="-38" cy="-13" r="3" fill="#000"/>
    <!-- Wing -->
    <path d="M 0,-5
               q 10,-20 20,-5
               q 5,0 5,10"
          fill="#ffbb00" stroke="#000" stroke-width="2"/>
    <!-- Tail -->
    <path d="M 20,0
               q 10,5 10,15
               q 0,5 -5,5"
          fill="#ffbb00" stroke="#000" stroke-width="2"/>
  </g>

</svg>
```

</details>

## 8. nemotron-3-ultra_cloud

**SVG size:** 16076 characters  
**Complexity:** Very high  
**Shape elements:** 92  
**Distinct colors:** 40  
**Raw response:** 17462 characters

![nemotron-3-ultra_cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/nemotron-3-ultra_cloud.svg)

### Analysis

This SVG contains approximately **92 shape elements** and uses **40 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth and shading
- Applies SVG filters (e.g., drop shadows, blurs) for visual effects
- Uses transforms (translate, rotate, scale) for precise positioning
- Includes animation elements -- the model attempted motion effects
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

With over 15,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in adding fine details like the jeep body panels, terrain, sky, and the duck's accessories. Best for users who want maximum visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400">
  <!-- Definitions for Gradients & Filters -->
  <defs>
    <!-- Jeep Body Gradient (Olive Green) -->
    <linearGradient id="jeepBodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#556B2F;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#3A4D20;stop-opacity:1" />
    </linearGradient>
    <!-- Jeep Highlight -->
    <linearGradient id="jeepHighlight" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#6B8E23;stop-opacity:0.5" />
      <stop offset="100%" style="stop-color:#556B2F;stop-opacity:0" />
    </linearGradient>
    <!-- Tire Gradient -->
    <radialGradient id="tireGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#333" />
      <stop offset="100%" style="stop-color:#000" />
    </radialGradient>
    <!-- Rim Gradient -->
    <radialGradient id="rimGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#CCC" />
      <stop offset="100%" style="stop-color:#666" />
    </radialGradient>
    <!-- Duck Body Gradient -->
    <linearGradient id="duckBodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FFD700" />
      <stop offset="100%" style="stop-color:#FFA500" />
    </linearGradient>
    <!-- Duck Head Gradient -->
    <linearGradient id="duckHeadGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FFF8DC" />
      <stop offset="100%" style="stop-color:#FFD700" />
    </linearGradient>
    <!-- Beak Gradient -->
    <linearGradient id="beakGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#FF8C00" />
      <stop offset="100%" style="stop-color:#E67300" />
    </linearGradient>
    <!-- Sky Gradient -->
    <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#87CEEB" />
      <stop offset="100%" style="stop-color:#E0F7FA" />
    </linearGradient>
    <!-- Ground Gradient -->
    <linearGradient id="groundGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#8D6E63" />
      <stop offset="100%" style="stop-color:#5D4037" />
    </linearGradient>
    <!-- Shadow Filter -->
    <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur"/>
      <feOffset dx="3" dy="4" result="offsetBlur"/>
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.3"/>
      </feComponentTransfer>
      <feMerge> 
        <feMergeNode in="offsetBlur"/>
        <feMergeNode in="SourceGraphic"/> 
      </feMerge>
    </filter>
  </defs>

  <!-- Background -->
  <rect width="600" height="400" fill="url(#skyGrad)" />
  
  <!-- Sun -->
  <circle cx="520" cy="60" r="40" fill="#FFD600" opacity="0.9">
    <animate attributeName="opacity" values="0.9;1;0.9" dur="4s" repeatCount="indefinite" />
  </circle>
  <g fill="#FFD600" opacity="0.7">
    <line x1="520" y1="10" x2="520" y2="0" stroke-width="3" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 520 60" to="360 520 60" dur="20s" repeatCount="indefinite"/></line>
    <line x1="520" y1="110" x2="520" y2="120" stroke-width="3" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 520 60" to="360 520 60" dur="20s" repeatCount="indefinite"/></line>
    <line x1="470" y1="60" x2="460" y2="60" stroke-width="3" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 520 60" to="360 520 60" dur="20s" repeatCount="indefinite"/></line>
    <line x1="580" y1="60" x2="590" y2="60" stroke-width="3" stroke-linecap="round"><animateTransform attributeName="transform" type="rotate" from="0 520 60" to="360 520 60" dur="20s" repeatCount="indefinite"/></line>
  </g>

  <!-- Clouds -->
  <g fill="white" opacity="0.8" filter="url(#dropShadow)">
    <circle cx="80" cy="50" r="20" /><circle cx="110" cy="40" r="25" /><circle cx="140" cy="50" r="20" /><circle cx="105" cy="65" r="15" />
    <animateTransform attributeName="transform" type="translate" values="0,0; 30,0; 0,0" dur="30s" repeatCount="indefinite" />
  </g>
  <g fill="white" opacity="0.6" filter="url(#dropShadow)">
    <circle cx="400" cy="80" r="15" /><circle cx="425" cy="70" r="20" /><circle cx="450" cy="80" r="15" />
    <animateTransform attributeName="transform" type="translate" values="0,0; -20,0; 0,0" dur="25s" repeatCount="indefinite" />
  </g>

  <!-- Ground -->
  <path d="M0,320 Q300,300 600,320 L600,400 L0,400 Z" fill="url(#groundGrad)" />
  <!-- Grass tufts -->
  <g fill="#4CAF50" opacity="0.8">
    <path d="M20,325 L25,310 L30,325" /><path d="M80,322 L85,308 L90,322" />
    <path d="M200,328 L205,315 L210,328" /><path d="M450,324 L455,312 L460,324" />
    <path d="M550,326 L555,314 L560,326" />
  </g>

  <!-- ================================================== -->
  <!-- JEEP GROUP (Positioned centered-ish)               -->
  <!-- ================================================== -->
  <g transform="translate(150, 180)" filter="url(#dropShadow)">
    
    <!-- Wheels (Back & Front) -->
    <g id="wheelBack">
      <circle cx="60" cy="140" r="38" fill="url(#tireGrad)" />
      <circle cx="60" cy="140" r="24" fill="url(#rimGrad)" />
      <circle cx="60" cy="140" r="12" fill="#222" />
      <!-- Lug nuts -->
      <g stroke="#444" stroke-width="2" fill="none">
        <line x1="60" y1="116" x2="60" y2="124" />
        <line x1="60" y1="156" x2="60" y2="164" />
        <line x1="36" y1="140" x2="44" y2="140" />
        <line x1="76" y1="140" x2="84" y2="140" />
      </g>
    </g>

    <g id="wheelFront">
      <circle cx="280" cy="140" r="38" fill="url(#tireGrad)" />
      <circle cx="280" cy="140" r="24" fill="url(#rimGrad)" />
      <circle cx="280" cy="140" r="12" fill="#222" />
      <g stroke="#444" stroke-width="2" fill="none">
        <line x1="280" y1="116" x2="280" y2="124" />
        <line x1="280" y1="156" x2="280" y2="164" />
        <line x1="256" y1="140" x2="264" y2="140" />
        <line x1="296" y1="140" x2="304" y2="140" />
      </g>
    </g>

    <!-- Wheel Animation (Rotation) -->
    <animateTransform href="#wheelBack" attributeName="transform" type="rotate" from="0 60 140" to="360 60 140" dur="1.5s" repeatCount="indefinite" />
    <animateTransform href="#wheelFront" attributeName="transform" type="rotate" from="0 280 140" to="360 280 140" dur="1.5s" repeatCount="indefinite" />

    <!-- Chassis / Floor Pan -->
    <path d="M30,100 L320,100 L320,145 L30,145 Z" fill="#2C3E1F" />

    <!-- Main Body Tub -->
    <path d="M40,40 L310,40 L310,105 Q290,115 270,115 L110,115 Q90,115 70,105 L40,105 Z" fill="url(#jeepBodyGrad)" stroke="#2E4018" stroke-width="2" />
    <!-- Highlight on hood -->
    <path d="M42,42 L250,42 L250,60 L42,60 Z" fill="url(#jeepHighlight)" opacity="0.6" />

    <!-- Windshield Frame (Folded down style or up? Let's do UP for driver visibility) -->
    <path d="M75,40 L75,10 L250,10 L250,40 Z" fill="none" stroke="#3A4D20" stroke-width="4" stroke-linejoin="round" />
    <!-- Windshield Glass -->
    <path d="M78,38 L78,13 L247,13 L247,38 Z" fill="#BBDEFB" opacity="0.4" stroke="#90CAF9" stroke-width="1" />

    <!-- Roll Bar / Cage -->
    <path d="M75,10 L75,-15 L250,-15 L250,10" fill="none" stroke="#4A5D2A" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
    <path d="M75,-15 L90,-30" stroke="#4A5D2A" stroke-width="4" stroke-linecap="round" />
    <path d="M250,-15 L235,-30" stroke="#4A5D2A" stroke-width="4" stroke-linecap="round" />

    <!-- Front Grille (Iconic 7-slot) -->
    <g transform="translate(310, 55)">
      <rect x="0" y="0" width="12" height="55" rx="2" fill="#1A1A1A" />
      <g stroke="#333" stroke-width="1.5">
        <line x1="2" y1="5" x2="10" y2="5" />
        <line x1="2" y1="13" x2="10" y2="13" />
        <line x1="2" y1="21" x2="10" y2="21" />
        <line x1="2" y1="29" x2="10" y2="29" />
        <line x1="2" y1="37" x2="10" y2="37" />
        <line x1="2" y1="45" x2="10" y2="45" />
        <line x1="2" y1="53" x2="10" y2="53" />
      </g>
    </g>

    <!-- Headlights -->
    <circle cx="320" cy="75" r="10" fill="#FFFDE7" stroke="#333" stroke-width="2" />
    <circle cx="320" cy="105" r="10" fill="#FFFDE7" stroke="#333" stroke-width="2" />
    <!-- Light Glow -->
    <g opacity="0.3" fill="#FFF9C4">
      <polygon points="320,75 360,60 360,90" />
      <polygon points="320,105 360,90 360,120" />
      <animate attributeName="opacity" values="0.3;0.5;0.3" dur="2s" repeatCount="indefinite" />
    </g>

    <!-- Fenders -->
    <path d="M30,105 Q40,85 60,105 Q80,85 100,105 L100,120 Q80,110 60,120 Q40,110 30,120 Z" fill="url(#jeepBodyGrad)" stroke="#2E4018" stroke-width="1" />
    <path d="M250,105 Q260,85 280,105 Q300,85 320,105 L320,120 Q300,110 280,120 Q260,110 250,120 Z" fill="url(#jeepBodyGrad)" stroke="#2E4018" stroke-width="1" />

    <!-- Rear Tail Light -->
    <rect x="35" y="85" width="8" height="20" rx="2" fill="#C62828" opacity="0.8" />
    
    <!-- Spare Tire on Back -->
    <circle cx="35" cy="70" r="22" fill="url(#tireGrad)" stroke="#222" stroke-width="2" />
    <circle cx="35" cy="70" r="10" fill="url(#rimGrad)" />
    <circle cx="35" cy="70" r="4" fill="#111" />
    
    <!-- Side Mirror -->
    <path d="M75,35 L60,25" stroke="#4A5D2A" stroke-width="4" stroke-linecap="round" />
    <circle cx="60" cy="25" r="6" fill="#333" />
    
    <!-- Door Handle -->
    <rect x="120" y="75" width="25" height="6" rx="3" fill="#4A5D2A" />
    
    <!-- Snorkel (Front Passenger Side) -->
    <path d="M240,10 Q240,-10 255,-15 L260,-15 L260,10 Z" fill="url(#jeepBodyGrad)" stroke="#2E4018" stroke-width="2" />
    <circle cx="257" cy="-18" r="6" fill="#333" />

  </g> <!-- End Jeep Group -->

  <!-- ================================================== -->
  <!-- DUCK GROUP (Driver)                                -->
  <!-- ================================================== -->
  <!-- Positioned roughly at Jeep coords: Jeep starts at 150,180. Driver seat ~ x=150+120=270, y=180+50=230 -->
  <g transform="translate(270, 190)" filter="url(#dropShadow)">
    
    <!-- Body -->
    <ellipse cx="0" cy="20" rx="35" ry="30" fill="url(#duckBodyGrad)" stroke="#E6AC00" stroke-width="1.5" />
    <!-- White chest patch -->
    <ellipse cx="0" cy="30" rx="20" ry="18" fill="#FFF8DC" opacity="0.6" />

    <!-- Wing (Left/Visible) - Animated Waving -->
    <g transform="translate(-35, 10)">
      <path id="wing" d="M0,0 Q-40,-20 -50,10 Q-30,25 0,30 Z" fill="url(#duckBodyGrad)" stroke="#E6AC00" stroke-width="1.5" />
      <animateTransform 
        href="#wing" 
        attributeName="transform" 
        type="rotate" 
        values="0,0,0; -25,0,0; 0,0,0; 15,0,0; 0,0,0" 
        keyTimes="0;0.2;0.4;0.6;1" 
        dur="3s" 
        repeatCount="indefinite" />
    </g>

    <!-- Neck -->
    <path d="M15,-10 Q25,-30 25,-45" fill="none" stroke="url(#duckBodyGrad)" stroke-width="22" stroke-linecap="round" />
    <path d="M15,-10 Q25,-30 25,-45" fill="none" stroke="#FFF8DC" stroke-width="10" stroke-linecap="round" opacity="0.4" />

    <!-- Head -->
    <circle cx="25" cy="-55" r="22" fill="url(#duckHeadGrad)" stroke="#E6AC00" stroke-width="1" />
    
    <!-- Eye -->
    <g transform="translate(35, -60)">
      <ellipse cx="0" cy="0" rx="6" ry="7" fill="white" />
      <ellipse cx="2" cy="-1" rx="3" ry="3.5" fill="#111" />
      <!-- Blink Animation -->
      <animate attributeName="ry" values="7;0;7" dur="4s" repeatCount="indefinite" keyTimes="0;0.1;0.2" calcMode="spline" keySplines="0.4 0 0.2 1; 0.4 0 0.2 1" />
      <animate attributeName="rx" values="6;0;6" dur="4s" repeatCount="indefinite" keyTimes="0;0.1;0.2" calcMode="spline" keySplines="0.4 0 0.2 1; 0.4 0 0.2 1" />
      <!-- Pupil follows blink -->
      <animate attributeName="ry" values="3.5;0;3.5" dur="4s" repeatCount="indefinite" keyTimes="0;0.1;0.2" calcMode="spline" keySplines="0.4 0 0.2 1; 0.4 0 0.2 1" href="#pupil" />
      <animate attributeName="rx" values="3;0;3" dur="4s" repeatCount="indefinite" keyTimes="0;0.1;0.2" calcMode="spline" keySplines="0.4 0 0.2 1; 0.4 0 0.2 1" href="#pupil" />
    </g>
    <ellipse id="pupil" cx="37" cy="-61" rx="3" ry="3.5" fill="#111" />

    <!-- Beak (Upper) -->
    <path d="M47,-55 Q65,-55 70,-50 Q65,-45 47,-45 Z" fill="url(#beakGrad)" stroke="#CC6600" stroke-width="1" />
    <!-- Beak (Lower) -->
    <path d="M47,-45 Q60,-40 50,-38 Q45,-40 47,-45 Z" fill="#FF8C00" stroke="#CC6600" stroke-width="0.5" />
    <!-- Nostril -->
    <ellipse cx="52" cy="-51" rx="1.5" ry="1" fill="#CC6600" />

    <!-- "Cool" Eyebrow / Feathers -->
    <path d="M30,-70 Q35,-78 45,-72" fill="none" stroke="#E6AC00" stroke-width="3" stroke-linecap="round" />
    
    <!-- Captain's Hat -->
    <g transform="translate(10, -80)">
      <!-- Brim -->
      <ellipse cx="0" cy="10" rx="28" ry="6" fill="#1B5E20" stroke="#0D3B15" stroke-width="1.5" />
      <!-- Crown -->
      <path d="M-20,10 Q-20,0 -5,-15 Q10,-20 20,0 Q20,10 20,10 L-20,10 Z" fill="#2E7D32" stroke="#1B5E20" stroke-width="1.5" />
      <!-- Badge/Star -->
      <polygon points="0,-5 2,-1 6,-1 3,1 4,5 0,3 -4,5 -3,1 -6,-1 -2,-1" fill="#FFD700" stroke="#B8860B" stroke-width="0.5" transform="translate(0, -10) scale(0.8)" />
    </g>

    <!-- Steering Wheel Hands (Wings gripping wheel) -->
    <!-- The Jeep steering wheel is roughly at Jeep coords (150+110, 180+60) -> (260, 240) -->
    <!-- Duck center is at (270, 190). Relative to duck: (-10, 50) -->
    <g transform="translate(-10, 50)">
      <!-- Left Wing Hand -->
      <path d="M0,0 Q-15,-10 -25,-5 Q-20,5 -5,10 Z" fill="url(#duckBodyGrad)" stroke="#E6AC00" stroke-width="1.5" />
      <!-- Right Wing Hand -->
      <path d="M0,0 Q15,-10 25,-5 Q20,5 5,10 Z" fill="url(#duckBodyGrad)" stroke="#E6AC00" stroke-width="1.5" transform="scale(-1,1)" />
    </g>

  </g> <!-- End Duck Group -->

  <!-- Steering Wheel (Drawn on top of duck hands/body but behind duck head) -->
  <!-- Jeep Center: 150, 180. Wheel center relative to Jeep: ~110, 65. Absolute: 260, 245 -->
  <g transform="translate(260, 245)" filter="url(#dropShadow)">
    <circle cx="0" cy="0" r="22" fill="none" stroke="#3E2723" stroke-width="5" />
    <circle cx="0" cy="0" r="6" fill="#4E342E" stroke="#2E1A13" stroke-width="2" />
    <!-- Spokes -->
    <line x1="0" y1="-22" x2="0" y2="-6" stroke="#3E2723" stroke-width="4" stroke-linecap="round" />
    <line x1="0" y1="6" x2="0" y2="22" stroke="#3E2723" stroke-width="4" stroke-linecap="round" />
    <line x1="-22" y1="0" x2="-6" y2="0" stroke="#3E2723" stroke-width="4" stroke-linecap="round" />
    <line x1="6" y1="0" x2="22" y2="0" stroke="#3E2723" stroke-width="4" stroke-linecap="round" />
    <!-- Rotate animation synced with wheels? -->
    <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="1.5s" repeatCount="indefinite" />
  </g>

  <!-- Dust Clouds behind tires -->
  <g transform="translate(150, 180)" opacity="0.4" fill="#A1887F">
    <g transform="translate(60, 160)">
      <circle r="8"><animate attributeName="r" values="5;15;5" dur="1s" repeatCount="indefinite" /><animate attributeName="opacity" values="0.4;0;0.4" dur="1s" repeatCount="indefinite" /></circle>
      <circle cx="10" cy="-5" r="6"><animate attributeName="r" values="4;12;4" dur="1.2s" repeatCount="indefinite" /><animate attributeName="opacity" values="0.3;0;0.3" dur="1.2s" repeatCount="indefinite" /></circle>
    </g>
    <g transform="translate(280, 160)">
      <circle r="8"><animate attributeName="r" values="5;15;5" dur="1s" repeatCount="indefinite" /><animate attributeName="opacity" values="0.4;0;0.4" dur="1s" repeatCount="indefinite" /></circle>
      <circle cx="-10" cy="-5" r="6"><animate attributeName="r" values="4;12;4" dur="1.2s" repeatCount="indefinite" /><animate attributeName="opacity" values="0.3;0;0.3" dur="1.2s" repeatCount="indefinite" /></circle>
    </g>
  </g>

</svg>
```

</details>

## 9. qwen3.5_397b-cloud

**SVG size:** 3106 characters  
**Complexity:** Low  
**Shape elements:** 29  
**Distinct colors:** 16  
**Raw response:** 3142 characters

![qwen3.5_397b-cloud duck jeep SVG](/assets/img/diagrams/ollama-svg-jeep/qwen3.5_397b-cloud.svg)

### Analysis

This SVG contains approximately **29 shape elements** and uses **16 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Duck is present (the model understood the main subject)
- Jeep / vehicle elements are present (the model understood the second subject)

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail. Best for embedding in web pages where small file size matters.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Background / Sky -->
  <rect width="500" height="400" fill="#87CEEB" />
  
  <!-- Ground -->
  <rect y="300" width="500" height="100" fill="#8FBC8F" />
  
  <!-- Road -->
  <rect y="320" width="500" height="60" fill="#555555" />
  <line x1="0" y1="350" x2="500" y2="350" stroke="#FFFFFF" stroke-width="4" stroke-dasharray="20, 20" />

  <!-- Jeep Group -->
  <g transform="translate(50, 50)">
    
    <!-- Jeep Body -->
    <path d="M20 200 L20 120 L100 120 L130 80 L320 80 L320 200 Z" fill="#556B2F" stroke="#2F4F4F" stroke-width="3"/>
    <!-- Jeep Fenders -->
    <rect x="10" y="180" width="80" height="40" rx="10" fill="#556B2F" />
    <rect x="280" y="180" width="80" height="40" rx="10" fill="#556B2F" />
    
    <!-- Windshield -->
    <path d="M140 80 L310 80 L300 160 L150 160 Z" fill="#ADD8E6" opacity="0.8" stroke="#2F4F4F" stroke-width="2"/>
    
    <!-- Steering Wheel -->
    <circle cx="240" cy="140" r="25" fill="none" stroke="#333" stroke-width="5" />
    <line x1="240" y1="115" x2="240" y2="165" stroke="#333" stroke-width="5" />

    <!-- Duck -->
    <g transform="translate(160, 100)">
      <!-- Body -->
      <ellipse cx="0" cy="60" rx="35" ry="45" fill="#FFD700" stroke="#DAA520" stroke-width="2"/>
      
      <!-- Wing (holding wheel) -->
      <path d="M10 50 Q 40 50 60 30" fill="none" stroke="#FFD700" stroke-width="12" stroke-linecap="round"/>
      <path d="M10 50 Q 40 50 60 30" fill="none" stroke="#DAA520" stroke-width="2" stroke-linecap="round"/>

      <!-- Head -->
      <circle cx="0" cy="0" r="30" fill="#FFD700" stroke="#DAA520" stroke-width="2"/>
      
      <!-- Beak -->
      <path d="M25 5 Q 45 10 25 15 Z" fill="#FF8C00" stroke="#DAA520" stroke-width="1"/>
      
      <!-- Eye -->
      <circle cx="10" cy="-5" r="4" fill="white"/>
      <circle cx="12" cy="-5" r="2" fill="black"/>
      
      <!-- Sunglasses (Cool Duck) -->
      <path d="M5 -5 L25 -5 L25 5 L5 5 Z" fill="#333" opacity="0.8"/>
      <line x1="5" y1="0" x2="-5" y2="-2" stroke="#333" stroke-width="2"/>
    </g>

    <!-- Wheels -->
    <!-- Back Wheel -->
    <g transform="translate(60, 220)">
      <circle cx="0" cy="0" r="40" fill="#333" stroke="#000" stroke-width="2"/>
      <circle cx="0" cy="0" r="25" fill="#777" stroke="#555" stroke-width="2"/>
      <circle cx="0" cy="0" r="10" fill="#CCC"/>
    </g>
    
    <!-- Front Wheel -->
    <g transform="translate(300, 220)">
      <circle cx="0" cy="0" r="40" fill="#333" stroke="#000" stroke-width="2"/>
      <circle cx="0" cy="0" r="25" fill="#777" stroke="#555" stroke-width="2"/>
      <circle cx="0" cy="0" r="10" fill="#CCC"/>
    </g>
    
    <!-- Headlight -->
    <circle cx="310" cy="130" r="8" fill="#FFFFE0" stroke="#555" stroke-width="1"/>
  </g>
  
  <!-- Motion Lines -->
  <path d="M40 330 L10 330" stroke="white" stroke-width="3" stroke-linecap="round"/>
  <path d="M50 360 L20 360" stroke="white" stroke-width="3" stroke-linecap="round"/>
  <path d="M30 250 L10 250" stroke="white" stroke-width="2" stroke-linecap="round"/>
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

Now that you have seen all 9 SVGs, here is a structured way to decide which Ollama cloud model is best for your SVG generation needs:

### Step 1: Define your priority

- **Visual quality first**: Scroll back through the SVGs above and pick the one that looks best to your eye. There is no substitute for visual judgment. The shape and color counts are useful, but a model with 60 shapes can look worse than one with 20 if the composition is off.
- **Code quality first**: Open the raw SVG source for each model (use the disclosure toggles) and look for `<defs>`, `<use>`, gradients, and clean indentation. Models that produce structured code are easier to edit and reuse programmatically.
- **Speed first**: If you are building a real-time app, prioritize the models that responded in under 15 seconds (gemma4:31b-cloud, nemotron-3-super:cloud).
- **File size first**: For web embedding, smaller is better. Look at the SVG size column in the summary table.

### Step 2: Cross-check across prompts

A model that does well on one prompt might fail on another. Check our other benchmarks:

- [Duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/) -- simpler scene
- [Duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/) -- dynamic action scene
- This post (duck in a jeep) -- vehicle with multiple parts

A model that consistently produces good results across all three prompts is a safer pick than one that only shines occasionally.

### Step 3: Test with your own prompt

Every model has strengths and weaknesses. The only way to know for sure which model is best for your specific use case is to test it with your own prompt. The Ollama Cloud API is OpenAI-compatible, so you can use any standard client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
resp = client.chat.completions.create(
    model="deepseek-v4-pro:cloud",  # change this to test different models
    messages=[{"role": "user", "content": "Make an svg image of <your prompt>"}],
)
print(resp.choices[0].message.content)
```

## Conclusion: You Decide the Winner

This comparison shows that 9 out of 13 active Ollama cloud models can generate valid SVG artwork from a natural language prompt involving a vehicle (jeep) with multiple parts. The results vary dramatically in complexity, style, and technique -- and there is no single "best" model.

Our takeaways after running three SVG benchmarks (bicycle, parachute, jeep):

- **deepseek-v4-pro:cloud** consistently produces well-structured, technically advanced SVGs with `<defs>`, `<use>`, and transforms. A strong default choice for code quality.
- **glm-5.1:cloud** and **nemotron-3-ultra:cloud** consistently produce the longest, most detailed SVGs. Best when you want maximum visual richness.
- **gemma4:31b-cloud** is consistently the fastest (under 25 seconds) and produces simple, compact SVGs. Best for speed-sensitive applications.
- **kimi-k2.6:cloud** often uses animations, which is unique among the models -- but the animations sometimes make content invisible unless you fix them (as we did in the parachute post).
- **glm-5.2:cloud** offers a reliable balance of detail, speed, and code quality.

But the real verdict is yours. Scroll back through the SVGs, compare them visually, check the raw code, and pick the model that best fits your needs. Every model in this comparison is available right now on Ollama Cloud -- so you can reproduce these results in minutes.

## Links

- [Previous: Duck Driving a Bicycle Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/)
- [Previous: Duck Jumping From a Plane Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/)
- [Ollama Official Website](https://ollama.com)
- [Ollama Cloud Documentation](https://ollama.com/cloud)
- [SVG Specification (MDN)](https://developer.mozilla.org/en-US/docs/Web/SVG)
- [OpenAI API Reference (used by Ollama)](https://platform.openai.com/docs/api-reference/chat)
