---
layout: post
title: "Which Ollama Cloud Model is Best? Baby Shark Fish SVG Comparison (13 Models)"
description: "Compare 14 Ollama cloud models on a marine-life prompt: a baby shark fish. Find the best LLM for SVG animal art. You decide the winner."
date: 2026-07-26
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/
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
  - Baby Shark
  - Marine Life
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama cloud model comparison, baby shark SVG, AI shark drawing, LLM SVG benchmark, AI image generation comparison, deepseek vs glm vs qwen, which Ollama model is best, Ollama cloud models 2026, AI animal art, marine life SVG, fish SVG art, baby shark meme, AI art comparison, underwater SVG scene"
---

# Which Ollama Cloud Model is Best? Baby Shark Fish SVG Comparison (13 Models)

After testing LLMs on ducks, vehicles, dev scenes, and nature, we wanted to know: **can today's top models draw marine life?** This time we asked 14 Ollama cloud models to draw **a baby shark fish** -- a prompt that tests organic curves, underwater atmosphere, character anatomy, and cute styling.

The prompt was: `Make an svg image of a baby shark fish`

This is the sixth in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/), and [duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/).

**Why a baby shark?** Unlike previous prompts, this scene combines multiple hard problems for SVG generation: (1) organic curved shapes (shark body, fins, tail) -- harder than rectangles, (2) underwater atmosphere (bubbles, gradients, waves), (3) anatomical correctness (dorsal fin, pectoral fins, gills, caudal tail), (4) character emotion (the "baby" cue implies cute, rounded proportions with big eyes), and (5) scene context (coral, seaweed, small fish). A model that draws a great jeep may struggle with organic curves, and vice versa -- which is exactly why we run multiple prompts.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Animal/Marine SVGs

Marine-life prompts reward different things than indoor or vehicle prompts. Here are the criteria to use:

- **Anatomical accuracy**: Does the SVG have a shark body, dorsal fin, pectoral fins, tail, and gills? Or is it a generic fish shape?
- **Organic curves**: Sharks have smooth, curved bodies. Does the model use `<path>` with bezier curves, or only straight lines and rectangles?
- **Underwater atmosphere**: Look for bubbles, waves, gradients, or blue color palettes that convey water. A shark on a white background misses the scene.
- **Baby/cute styling**: The prompt says "baby." Does the model render a cute, rounded shark with big eyes, or a scary adult shark? Prompt adherence matters.
- **Scene context**: Coral, seaweed, small fish, or a reef add depth and show the model understood the habitat.
- **SVG code quality**: Does it use `<defs>`, `<use>`, gradients, and clean structure? Better code is easier to tweak (e.g., to recolor or animate).

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with zero post-processing (other than adding `width="100%" height="auto"` for responsive embedding and fixing any XML errors so the SVG renders in browsers).

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `deepseek-v4-flash:cloud` | 4917 | 42 | 15 | High | Detailed |
| 2 | `deepseek-v4-pro:cloud` | 3287 | 27 | 8 | Medium | Balanced |
| 3 | `gemma4:31b-cloud` | 1628 | 15 | 9 | Medium | Balanced |
| 4 | `gemma4:cloud` | 1616 | 13 | 8 | Compact | Compact |
| 5 | `glm-5.1:cloud` | 11903 | 69 | 23 | Very high | Richest scene |
| 6 | `glm-5.2:cloud` | 6479 | 43 | 14 | High | Detailed |
| 7 | `gpt-oss:120b-cloud` | 1213 | 8 | 6 | Compact | Compact |
| 8 | `kimi-k2.6:cloud` | 3425 | 13 | 8 | Medium | Balanced |
| 9 | `minimax-m2.7:cloud` | 2341 | 18 | 10 | Medium | Balanced |
| 10 | `minimax-m3:cloud` | 2860 | 24 | 6 | Medium | Balanced |
| 11 | `nemotron-3-super:cloud` | 939 | 7 | 5 | Compact | Compact |
| 12 | `nemotron-3-ultra:cloud` | 5200 | 25 | 8 | Medium | Detailed |
| 13 | `qwen3.5:397b-cloud` | 2023 | 19 | 7 | Medium | Balanced |
| 14 | `deepseek-v4-flash:0731-cloud` | 6060 | 53 | 6 | High | Detailed |
| 15 | `bjoernb/claude-opus-4-5:latest` | - | - | - | - | Retired (410) |
| 16 | `deepseek-v3.1:671b-cloud` | - | - | - | - | Retired (410) |
| 17 | `glm-5:cloud` | - | - | - | - | Retired (410) |
| 18 | `qwen3-vl:235b-cloud` | - | - | - | - | Retired (410) |

**14 out of 18** active models produced a valid SVG. The 4 retired models returned HTTP 410 Gone (removed from Ollama Cloud on 2026-07-15).

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most detailed shark scene SVG**: pick models labeled "Very high" complexity in the table above
- **You want the fastest response**: look at the per-model sections below for the elapsed time
- **You want the cleanest, most reusable SVG code**: pick models that use `<defs>`, `<use>`, and transforms (see raw source below each SVG)
- **You want a small, efficient SVG for web embedding**: pick models with "Compact" verdict
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"
- **You want to compare within a model family**: pick `deepseek-v4-pro` vs `deepseek-v4-flash`, or `glm-5.1` vs `glm-5.2`, or `minimax-m2.7` vs `minimax-m3`

Now read on for the full per-model breakdown and judge for yourself.

## 1. `deepseek-v4-flash:cloud`

**SVG size:** 4917 characters  
**Complexity:** High  
**Shape elements:** 42  
**Distinct colors:** 15  
**Raw response:** 5311 characters  
**Generation time:** 72.6s

![deepseek-v4-flash:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/deepseek-v4-flash_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Bubbles are present -- the model created an underwater atmosphere
- Water/ocean scene is present -- the model understood the habitat
- Coral/seaweed is present -- the model decorated the underwater scene
- Baby/cute styling is present -- the model captured the "baby" cue

With 4,917 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="auto">
  <defs>
    <!-- Drop shadow for the shark -->
    <filter id="shadow" x="-10%" y="-10%" width="130%" height="130%">
      <feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="#000" flood-opacity="0.15" />
    </filter>
    <!-- Clip path for the shark's belly -->
    <clipPath id="bodyClip">
      <path d="M 420 250 
               C 420 180, 380 140, 300 140 
               C 240 140, 200 180, 160 220 
               C 200 320, 240 360, 300 360 
               C 380 360, 420 320, 420 250 Z" />
    </clipPath>
  </defs>
  <!-- Background -->
  <rect width="500" height="500" fill="#E0F7FA" />
  <circle cx="250" cy="250" r="230" fill="#B2EBF2" opacity="0.5" />
  <!-- Background Elements: Seaweed -->
  <g stroke-linecap="round" fill="none">
    <!-- Left Seaweed -->
    <path d="M 30 450 Q 50 400 30 350 Q 10 300 30 250" stroke="#4CAF50" stroke-width="8" />
    <path d="M 60 450 Q 80 380 60 320 Q 40 280 60 220" stroke="#66BB6A" stroke-width="6" />
    <!-- Right Seaweed -->
    <path d="M 470 450 Q 450 400 470 350 Q 490 300 470 250" stroke="#4CAF50" stroke-width="8" />
    <path d="M 440 450 Q 420 380 440 320 Q 460 280 440 220" stroke="#66BB6A" stroke-width="6" />
  </g>
  <!-- Background Bubbles -->
  <g fill="#FFFFFF" opacity="0.6">
    <circle cx="100" cy="150" r="15" />
    <circle cx="80" cy="120" r="8" />
    <circle cx="120" cy="100" r="10" />
    <circle cx="400" cy="100" r="12" />
    <circle cx="420" cy="70" r="6" />
    <circle cx="380" cy="80" r="8" />
    <circle cx="150" cy="380" r="10" />
    <circle cx="180" cy="400" r="6" />
    <circle cx="400" cy="350" r="14" />
    <circle cx="430" cy="380" r="7" />
  </g>
  <!-- Main Shark Group -->
  <g filter="url(#shadow)">
    <!-- Far Side Pectoral Fin -->
    <path d="M 320 280 Q 340 330 360 300 Z" fill="#8BB8D6" opacity="0.6" />
    <!-- Tail Fin -->
    <path d="M 160 220 
             C 80 170, 20 200, 40 250 
             C 20 300, 80 330, 160 280 
             C 130 280, 100 260, 100 250 
             C 100 240, 130 220, 160 220 Z" fill="#8BB8D6" />
    <!-- Anal Fin -->
    <path d="M 180 320 Q 190 360 210 340 Z" fill="#A2D5F2" />
    <!-- Body and Belly (Clipped) -->
    <g clip-path="url(#bodyClip)">
      <!-- Body Base -->
      <path d="M 420 250 
               C 420 180, 380 140, 300 140 
               C 240 140, 200 180, 160 220 
               C 200 320, 240 360, 300 360 
               C 380 360, 420 320, 420 250 Z" fill="#A2D5F2" />
      <!-- Belly -->
      <path d="M 160 220 
               C 200 320, 240 360, 300 360 
               C 380 360, 420 320, 420 250 
               C 380 300, 280 310, 160 220 Z" fill="#FFFFFF" />
    </g>
    <!-- Dorsal Fin -->
    <path d="M 220 145 Q 260 160 300 145 C 280 70, 240 70, 220 145 Z" fill="#A2D5F2" />
    <!-- Dorsal Fin Inner Highlight -->
    <path d="M 235 145 Q 260 155 285 145 C 270 90, 250 90, 235 145 Z" fill="#D5F0F7" />
    <!-- Pectoral Fin -->
    <path d="M 300 300 Q 335 320 370 320 C 360 370, 320 370, 300 300 Z" fill="#A2D5F2" />
    <!-- Pectoral Fin Inner Highlight -->
    <path d="M 315 310 Q 340 320 360 315 C 350 350, 330 350, 315 310 Z" fill="#D5F0F7" />
    <!-- Gills -->
    <g fill="none" stroke="#8BB8D6" stroke-width="4" stroke-linecap="round">
      <path d="M 290 220 Q 285 230 290 240" />
      <path d="M 300 220 Q 295 230 300 240" />
      <path d="M 310 220 Q 305 230 310 240" />
    </g>
    <!-- Cheek Blush -->
    <ellipse cx="345" cy="245" rx="18" ry="12" fill="#FFB6C1" opacity="0.8" />
    <!-- Eye -->
    <circle cx="340" cy="200" r="18" fill="#2C3E50" />
    <circle cx="345" cy="195" r="6" fill="#FFFFFF" />
    <circle cx="335" cy="205" r="2" fill="#FFFFFF" />
    <!-- Eyebrow -->
    <path d="M 325 175 Q 340 170 355 175" fill="none" stroke="#2C3E50" stroke-width="4" stroke-linecap="round" />
    <!-- Mouth -->
    <path d="M 360 250 Q 375 265 360 280" fill="none" stroke="#2C3E50" stroke-width="4" stroke-linecap="round" />
    <!-- Teeth -->
    <path d="M 363 255 L 366 261 L 369 255 L 372 261 L 375 255 L 378 261 L 381 255" fill="#FFFFFF" stroke="#2C3E50" stroke-width="2" stroke-linejoin="round" />
    <!-- Cute Crown -->
    <g>
      <!-- Crown Base -->
      <path d="M 320 150 L 330 120 L 340 135 L 350 110 L 360 135 L 370 120 L 380 150 Z" fill="#FFD700" stroke="#FFA500" stroke-width="3" stroke-linejoin="round" />
      <!-- Crown Jewels -->
      <circle cx="340" cy="135" r="4" fill="#FF0000" />
      <circle cx="350" cy="110" r="4" fill="#00BFFF" />
      <circle cx="360" cy="135" r="4" fill="#FF0000" />
    </g>
  </g>
  <!-- Foreground Sparkles -->
  <g fill="#FFD700">
    <path d="M 120 150 L 125 160 L 135 165 L 125 170 L 120 180 L 115 170 L 105 165 L 115 160 Z" />
    <path d="M 400 180 L 403 185 L 408 188 L 403 191 L 400 196 L 397 191 L 392 188 L 397 185 Z" />
  </g>
</svg>
```

</details>

## 2. `deepseek-v4-pro:cloud`

**SVG size:** 3287 characters  
**Complexity:** Medium  
**Shape elements:** 27  
**Distinct colors:** 8  
**Raw response:** 3656 characters  
**Generation time:** 59.3s

![deepseek-v4-pro:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/deepseek-v4-pro_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for ocean water and lighting)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Bubbles are present -- the model created an underwater atmosphere
- Water/ocean scene is present -- the model understood the habitat
- Baby/cute styling is present -- the model captured the "baby" cue

With 3,287 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="auto">
  <defs>
    <!-- Body Gradient -->
    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#5DADE2" />
      <stop offset="100%" stop-color="#2E86C1" />
    </linearGradient>
    <!-- Belly Gradient -->
    <linearGradient id="bellyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FFFFFF" />
      <stop offset="100%" stop-color="#D6EAF8" />
    </linearGradient>
    <!-- Drop Shadow -->
    <filter id="shadow" x="-10%" y="-10%" width="130%" height="130%">
      <feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="#000000" flood-opacity="0.15" />
    </filter>
  </defs>
  <!-- Background Water Waves -->
  <g id="background-waves">
    <path d="M 0 450 Q 50 430 100 450 T 200 450 T 300 450 T 400 450 T 500 450 L 500 500 L 0 500 Z" fill="#85C1E9" opacity="0.4" />
    <path d="M 0 470 Q 50 450 100 470 T 200 470 T 300 470 T 400 470 T 500 470 L 500 500 L 0 500 Z" fill="#5DADE2" opacity="0.4" />
  </g>
  <!-- Bubbles -->
  <g id="bubbles" fill="none" stroke="#AED6F1" stroke-width="3">
    <circle cx="80" cy="150" r="15" />
    <circle cx="60" cy="110" r="8" />
    <circle cx="100" cy="80" r="20" />
    <circle cx="450" cy="100" r="12" />
    <circle cx="480" cy="70" r="6" />
    <circle cx="420" cy="60" r="18" />
  </g>
  <!-- Baby Shark -->
  <g id="shark" filter="url(#shadow)">
    <!-- Tail Fin -->
    <path d="M 390 250 Q 450 150 490 150 Q 450 250 490 350 Q 450 350 390 250 Z" fill="url(#bodyGrad)" />
    <!-- Dorsal Fin -->
    <path d="M 230 160 Q 270 60 310 160 Z" fill="url(#bodyGrad)" />
    <!-- Pectoral Fin (Left) -->
    <path d="M 150 300 Q 100 380 50 380 Q 100 330 150 300 Z" fill="url(#bodyGrad)" />
    <!-- Pectoral Fin (Right) -->
    <path d="M 250 300 Q 300 380 350 380 Q 300 330 250 300 Z" fill="url(#bodyGrad)" />
    <!-- Main Body -->
    <path d="M 100 250 C 100 120, 250 120, 330 180 C 370 210, 400 230, 400 250 C 400 270, 370 290, 330 320 C 250 380, 100 380, 100 250 Z" fill="url(#bodyGrad)" />
    <!-- Belly -->
    <ellipse cx="200" cy="280" rx="90" ry="45" fill="url(#bellyGrad)" />
    <!-- Gills -->
    <path d="M 280 220 Q 300 250 280 280" fill="none" stroke="#2E86C1" stroke-width="4" stroke-linecap="round" />
    <path d="M 300 220 Q 320 250 300 280" fill="none" stroke="#2E86C1" stroke-width="4" stroke-linecap="round" />
    <!-- Eyes -->
    <!-- Left Eye -->
    <circle cx="150" cy="230" r="20" fill="#FFFFFF" />
    <circle cx="150" cy="230" r="10" fill="#17202A" />
    <circle cx="153" cy="227" r="3" fill="#FFFFFF" />
    <circle cx="147" cy="233" r="1.5" fill="#FFFFFF" />
    <!-- Right Eye -->
    <circle cx="250" cy="230" r="20" fill="#FFFFFF" />
    <circle cx="250" cy="230" r="10" fill="#17202A" />
    <circle cx="253" cy="227" r="3" fill="#FFFFFF" />
    <circle cx="247" cy="233" r="1.5" fill="#FFFFFF" />
    <!-- Cute Smile -->
    <path d="M 170 270 Q 200 290 230 270" fill="none" stroke="#17202A" stroke-width="4" stroke-linecap="round" />
    <!-- Blush Cheeks -->
    <ellipse cx="130" cy="260" rx="15" ry="8" fill="#F1948A" opacity="0.6" />
    <ellipse cx="270" cy="260" rx="15" ry="8" fill="#F1948A" opacity="0.6" />
  </g>
</svg>
```

</details>

## 3. `gemma4:31b-cloud`

**SVG size:** 1628 characters  
**Complexity:** Medium  
**Shape elements:** 15  
**Distinct colors:** 9  
**Raw response:** 2322 characters  
**Generation time:** 8.6s

![gemma4:31b-cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/gemma4_31b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Bubbles are present -- the model created an underwater atmosphere
- Baby/cute styling is present -- the model captured the "baby" cue

With 1,628 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg width="400" height="auto" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Background Circle for a pop of color -->
  <circle cx="200" cy="150" r="120" fill="#e0f7fa" />
  <!-- Tail Fin -->
  <path d="M100 160 L60 130 L60 200 Z" fill="#42a5f5" stroke="#1976d2" stroke-width="4" stroke-linejoin="round" />
  <!-- Main Body -->
  <path d="M100 160 C 100 100, 280 100, 300 160 C 300 220, 100 220, 100 160" fill="#64b5f6" stroke="#1976d2" stroke-width="4" />
  <!-- Dorsal Fin (Top) -->
  <path d="M180 125 Q 200 80, 230 125" fill="#64b5f6" stroke="#1976d2" stroke-width="4" stroke-linejoin="round" />
  <!-- Pectoral Fin (Side) -->
  <path d="M170 170 Q 160 190, 190 190" fill="#42a5f5" stroke="#1976d2" stroke-width="4" stroke-linejoin="round" />
  <!-- Eye (Left) -->
  <circle cx="250" cy="145" r="8" fill="black" />
  <circle cx="253" cy="142" r="3" fill="white" />
  <!-- Eye (Right) -->
  <circle cx="280" cy="145" r="8" fill="black" />
  <circle cx="283" cy="142" r="3" fill="white" />
  <!-- Mouth/Smile -->
  <path d="M240 170 Q 265 185, 290 170" fill="none" stroke="#1976d2" stroke-width="4" stroke-linecap="round" />
  <!-- Little Blush Cheeks -->
  <circle cx="240" cy="155" r="5" fill="#ff8a80" opacity="0.6" />
  <circle cx="290" cy="155" r="5" fill="#ff8a80" opacity="0.6" />
  <!-- Bubbles -->
  <circle cx="320" cy="120" r="10" fill="white" stroke="#b3e5fc" stroke-width="2" opacity="0.8" />
  <circle cx="340" cy="100" r="6" fill="white" stroke="#b3e5fc" stroke-width="2" opacity="0.8" />
  <circle cx="310" cy="80" r="4" fill="white" stroke="#b3e5fc" stroke-width="2" opacity="0.8" />
</svg>
```

</details>

## 4. `gemma4:cloud`

**SVG size:** 1616 characters  
**Complexity:** Compact  
**Shape elements:** 13  
**Distinct colors:** 8  
**Raw response:** 2428 characters  
**Generation time:** 44.5s

![gemma4:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/gemma4_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Bubbles are present -- the model created an underwater atmosphere
- Baby/cute styling is present -- the model captured the "baby" cue

With 1,616 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg width="400" height="auto" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Background Circle for a little pop of color -->
  <circle cx="200" cy="150" r="120" fill="#e0f7fa" />
  <!-- Tail Fin -->
  <path d="M100 160 L60 130 L60 200 Z" fill="#42a5f5" stroke="#1565c0" stroke-width="4" stroke-linejoin="round" />
  <!-- Main Body -->
  <path d="M100 160 Q120 110 220 120 Q320 130 330 170 Q320 220 220 220 Q120 210 100 160" fill="#64b5f6" stroke="#1565c0" stroke-width="4" />
  <!-- Dorsal Fin (Top) -->
  <path d="M200 130 Q220 80 250 125" fill="#64b5f6" stroke="#1565c0" stroke-width="4" stroke-linejoin="round" />
  <!-- Pectoral Fin (Side) -->
  <path d="M210 180 Q230 220 260 190" fill="#42a5f5" stroke="#1565c0" stroke-width="4" stroke-linejoin="round" />
  <!-- Eye -->
  <circle cx="270" cy="155" r="8" fill="white" stroke="black" stroke-width="1" />
  <circle cx="274" cy="153" r="3" fill="black" />
  <!-- Mouth/Smile -->
  <path d="M280 175 Q300 190 315 175" fill="none" stroke="#1565c0" stroke-width="3" stroke-linecap="round" />
  <!-- Gills -->
  <line x1="240" y1="160" x2="235" y2="175" stroke="#1565c0" stroke-width="2" stroke-linecap="round" />
  <line x1="245" y1="160" x2="240" y2="175" stroke="#1565c0" stroke-width="2" stroke-linecap="round" />
  <line x1="250" y1="160" x2="245" y2="175" stroke="#1565c0" stroke-width="2" stroke-linecap="round" />
  <!-- Tiny Bubbles -->
  <circle cx="340" cy="140" r="5" fill="white" stroke="#b3e5fc" stroke-width="1" opacity="0.7" />
  <circle cx="350" cy="120" r="3" fill="white" stroke="#b3e5fc" stroke-width="1" opacity="0.7" />
</svg>
```

</details>

## 5. `glm-5.1:cloud`

**SVG size:** 11903 characters  
**Complexity:** Very high  
**Shape elements:** 69  
**Distinct colors:** 23  
**Raw response:** 16204 characters  
**Generation time:** 12.0s

![glm-5.1:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/glm-5.1_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for ocean water and lighting)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the shark and scene elements
- Includes animation elements -- the model attempted motion (e.g., swimming, bubbles)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Bubbles are present -- the model created an underwater atmosphere
- Water/ocean scene is present -- the model understood the habitat
- Small fish are present -- the model added scene context
- Coral/seaweed is present -- the model decorated the underwater scene
- Baby/cute styling is present -- the model captured the "baby" cue

With 11,903 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 700 600" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
    <defs>
      <!-- Water gradient background -->
      <linearGradient id="waterBg" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#0a1628"/>
        <stop offset="50%" stop-color="#0d3b5e"/>
        <stop offset="100%" stop-color="#1a6b7a"/>
      </linearGradient>
      <!-- Shark body gradient -->
      <linearGradient id="sharkBody" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#5ecfe6"/>
        <stop offset="40%" stop-color="#3ab8d4"/>
        <stop offset="100%" stop-color="#1e8fa8"/>
      </linearGradient>
      <!-- Belly gradient -->
      <linearGradient id="bellyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#e8f8fc"/>
        <stop offset="100%" stop-color="#b8eaf5"/>
      </linearGradient>
      <!-- Fin gradient -->
      <linearGradient id="finGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#4ac4db"/>
        <stop offset="100%" stop-color="#1e95ab"/>
      </linearGradient>
      <!-- Sand gradient -->
      <linearGradient id="sandGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#2a7a6e"/>
        <stop offset="100%" stop-color="#1a5a50"/>
      </linearGradient>
      <!-- Bubble radial -->
      <radialGradient id="bubbleGrad" cx="35%" cy="35%">
        <stop offset="0%" stop-color="#ffffff" stop-opacity="0.8"/>
        <stop offset="100%" stop-color="#7ecfe0" stop-opacity="0.15"/>
      </radialGradient>
      <!-- Glow filter -->
      <filter id="glow">
        <feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
      <!-- Soft shadow -->
      <filter id="shadow">
        <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#063440" flood-opacity="0.4"/>
      </filter>
    </defs>
    <!-- Background -->
    <rect width="700" height="600" fill="url(#waterBg)"/>
    <!-- Light rays from above -->
    <polygon class="light-ray" points="200,0 260,600 140,600" fill="#5ecfe6" opacity="0.1"/>
    <polygon class="light-ray" points="380,0 450,600 310,600" fill="#5ecfe6" opacity="0.08" style="animation-delay: 1s"/>
    <polygon class="light-ray" points="520,0 590,600 460,600" fill="#5ecfe6" opacity="0.06" style="animation-delay: 2s"/>
    <!-- Sandy bottom -->
    <ellipse cx="350" cy="590" rx="400" ry="40" fill="url(#sandGrad)" opacity="0.6"/>
    <!-- Seaweed -->
    <g class="seaweed" style="animation-delay: 0s">
      <path d="M80,580 Q75,540 85,510 Q95,480 80,450 Q65,420 80,390" fill="none" stroke="#1a7a5a" stroke-width="6" stroke-linecap="round" opacity="0.7"/>
      <path d="M90,580 Q95,550 85,525 Q75,500 90,475" fill="none" stroke="#1a9a6a" stroke-width="4" stroke-linecap="round" opacity="0.5"/>
    </g>
    <g class="seaweed" style="animation-delay: 1.2s">
      <path d="M620,580 Q615,545 625,520 Q635,495 620,470 Q605,445 620,420" fill="none" stroke="#1a7a5a" stroke-width="6" stroke-linecap="round" opacity="0.7"/>
      <path d="M630,580 Q635,555 625,530 Q615,505 630,480" fill="none" stroke="#1a9a6a" stroke-width="4" stroke-linecap="round" opacity="0.5"/>
    </g>
    <g class="seaweed" style="animation-delay: 0.6s">
      <path d="M560,585 Q555,560 565,540 Q575,520 560,500" fill="none" stroke="#1a8a5a" stroke-width="5" stroke-linecap="round" opacity="0.6"/>
    </g>
    <!-- Small background bubbles -->
    <circle class="bubble" cx="120" cy="500" r="4" fill="url(#bubbleGrad)" style="animation-duration: 5s; animation-delay: 0s"/>
    <circle class="bubble" cx="180" cy="520" r="3" fill="url(#bubbleGrad)" style="animation-duration: 6s; animation-delay: 1.5s"/>
    <circle class="bubble" cx="500" cy="490" r="5" fill="url(#bubbleGrad)" style="animation-duration: 5.5s; animation-delay: 0.8s"/>
    <circle class="bubble" cx="580" cy="510" r="3" fill="url(#bubbleGrad)" style="animation-duration: 7s; animation-delay: 2s"/>
    <circle class="bubble" cx="400" cy="530" r="4" fill="url(#bubbleGrad)" style="animation-duration: 6.5s; animation-delay: 3s"/>
    <circle class="bubble" cx="300" cy="500" r="2.5" fill="url(#bubbleGrad)" style="animation-duration: 5.8s; animation-delay: 0.5s"/>
    <circle class="bubble" cx="650" cy="480" r="3.5" fill="url(#bubbleGrad)" style="animation-duration: 6.2s; animation-delay: 2.5s"/>
    <!-- ========= BABY SHARK ========= -->
    <g class="shark-group" filter="url(#shadow)">
      <!-- Tail (animated separately) -->
      <g class="tail">
        <path d="M195,300 Q150,270 120,230 Q135,270 130,290 Q125,310 120,340 Q150,320 195,310 Z"
              fill="url(#finGrad)" stroke="#1790a5" stroke-width="1.5"/>
        <!-- Tail detail lines -->
        <path d="M185,298 Q155,280 135,250" fill="none" stroke="#1aafca" stroke-width="1" opacity="0.5"/>
        <path d="M185,306 Q155,320 135,345" fill="none" stroke="#1aafca" stroke-width="1" opacity="0.5"/>
      </g>
      <!-- Body -->
      <path d="M210,260 Q280,225 370,235 Q440,240 470,270 Q490,290 470,320 Q440,355 370,360 Q280,365 210,340 Q195,330 195,300 Q195,270 210,260 Z"
            fill="url(#sharkBody)" stroke="#1790a5" stroke-width="2"/>
      <!-- Belly -->
      <path d="M230,310 Q300,340 380,345 Q430,343 460,320 Q440,345 380,355 Q300,358 230,335 Z"
            fill="url(#bellyGrad)" opacity="0.8"/>
      <!-- Body highlight -->
      <path d="M240,262 Q310,242 380,245 Q430,248 460,265"
            fill="none" stroke="#8ae6f5" stroke-width="2" opacity="0.5" stroke-linecap="round"/>
      <!-- Body shimmer sparkles -->
      <circle class="sparkle" cx="300" cy="260" r="2.5" fill="#ffffff" opacity="0.5" style="animation-delay: 0s"/>
      <circle class="sparkle" cx="360" cy="255" r="2" fill="#ffffff" opacity="0.4" style="animation-delay: 0.5s"/>
      <circle class="sparkle" cx="420" cy="265" r="2.5" fill="#ffffff" opacity="0.5" style="animation-delay: 1s"/>
      <circle class="sparkle" cx="280" cy="275" r="1.8" fill="#ffffff" opacity="0.3" style="animation-delay: 1.5s"/>
      <!-- Dorsal fin (animated separately) -->
      <g class="dorsal-fin">
        <path d="M330,238 Q340,195 360,175 Q355,200 365,238 Z"
              fill="url(#finGrad)" stroke="#1790a5" stroke-width="1.5"/>
        <!-- Fin detail -->
        <path d="M342,230 Q348,205 357,185" fill="none" stroke="#1aafca" stroke-width="1" opacity="0.5"/>
      </g>
      <!-- Pectoral fin (animated separately) -->
      <g class="pectoral-fin">
        <path d="M340,320 Q360,350 380,380 Q365,355 350,340 Q345,335 340,330 Z"
              fill="url(#finGrad)" stroke="#1790a5" stroke-width="1.5"/>
      </g>
      <!-- Smaller pectoral fin (far side) -->
      <path d="M320,320 Q310,345 315,360 Q320,345 325,330 Z"
            fill="#2ea0b5" stroke="#1790a5" stroke-width="1" opacity="0.6"/>
      <!-- Gill lines -->
      <line x1="430" y1="275" x2="435" y2="305" stroke="#1a9ab5" stroke-width="1.5" opacity="0.5" stroke-linecap="round"/>
      <line x1="440" y1="278" x2="445" y2="308" stroke="#1a9ab5" stroke-width="1.5" opacity="0.4" stroke-linecap="round"/>
      <line x1="450" y1="282" x2="454" y2="310" stroke="#1a9ab5" stroke-width="1.5" opacity="0.3" stroke-linecap="round"/>
      <!-- Mouth (open/close animation) -->
      <g class="mouth-open" style="transform-origin: 468px 300px">
        <path d="M465,290 Q480,300 465,315 Q455,305 465,290 Z"
              fill="#0d4a5e" stroke="#0a3a48" stroke-width="1"/>
        <!-- Tongue -->
        <ellipse cx="463" cy="308" rx="5" ry="3" fill="#e0707080" opacity="0.6"/>
      </g>
      <!-- Teeth (cute little teeth) -->
      <g>
        <path d="M462,290 L464,296 L466,290" fill="#ffffff" opacity="0.9"/>
        <path d="M468,292 L469,298 L471,292" fill="#ffffff" opacity="0.9"/>
        <path d="M460,310 L462,305 L464,310" fill="#ffffff" opacity="0.9"/>
        <path d="M466,312 L468,307 L470,312" fill="#ffffff" opacity="0.9"/>
      </g>
      <!-- Eye white -->
      <ellipse cx="450" cy="268" rx="16" ry="18" fill="#ffffff" stroke="#1790a5" stroke-width="1.5"/>
      <!-- Iris -->
      <ellipse cx="453" cy="270" rx="9" ry="11" fill="#1a3050"/>
      <!-- Pupil -->
      <ellipse cx="455" cy="270" rx="5" ry="7" fill="#0a1520"/>
      <!-- Eye highlight -->
      <ellipse cx="457" cy="265" rx="3.5" ry="4" fill="#ffffff" opacity="0.9"/>
      <ellipse cx="450" cy="275" rx="1.5" ry="2" fill="#ffffff" opacity="0.5"/>
      <!-- Eye blink lid -->
      <ellipse class="eye-lid" cx="450" cy="268" rx="17" ry="19" fill="#3ab8d4" style="transform-origin: 450px 268px"/>
      <!-- Eyebrow (cute expression) -->
      <path d="M438,252 Q445,248 462,253" fill="none" stroke="#1790a5" stroke-width="2" stroke-linecap="round"/>
      <!-- Cheek blush -->
      <ellipse class="cheek-glow" cx="440" cy="295" rx="12" ry="7" fill="#f5a0a0" opacity="0.4"/>
      <!-- Nostril -->
      <circle cx="475" cy="272" r="2.5" fill="#1790a5" opacity="0.6"/>
      <!-- Shark mouth smile line -->
      <path d="M458,300 Q462,305 458,310" fill="none" stroke="#0d4a5e" stroke-width="1" opacity="0.4" stroke-linecap="round"/>
    </g>
    <!-- END BABY SHARK -->
    <!-- Foreground bubbles (coming from shark) -->
    <circle class="bubble" cx="490" cy="260" r="6" fill="url(#bubbleGrad)" style="animation-duration: 4s; animation-delay: 0.3s"/>
    <circle class="bubble" cx="500" cy="275" r="4" fill="url(#bubbleGrad)" style="animation-duration: 4.5s; animation-delay: 1s"/>
    <circle class="bubble" cx="485" cy="250" r="3" fill="url(#bubbleGrad)" style="animation-duration: 3.8s; animation-delay: 2s"/>
    <circle class="bubble" cx="495" cy="270" r="5" fill="url(#bubbleGrad)" style="animation-duration: 5s; animation-delay: 0.7s"/>
    <circle class="bubble" cx="480" cy="255" r="2.5" fill="url(#bubbleGrad)" style="animation-duration: 4.2s; animation-delay: 1.8s"/>
    <!-- Small starfish on the sand -->
    <g transform="translate(150,565) scale(0.6)" opacity="0.7">
      <path d="M0,-20 L5,-6 L20,-6 L8,4 L12,18 L0,10 L-12,18 L-8,4 L-20,-6 L-5,-6 Z"
            fill="#e07060" stroke="#c05545" stroke-width="1.5"/>
    </g>
    <!-- Shell on sand -->
    <g transform="translate(530,572)" opacity="0.6">
      <ellipse cx="0" cy="0" rx="12" ry="8" fill="#d4a070" stroke="#b08050" stroke-width="1"/>
      <path d="M-8,0 Q0,-6 8,0" fill="none" stroke="#b08050" stroke-width="0.8"/>
      <path d="M-6,2 Q0,-3 6,2" fill="none" stroke="#b08050" stroke-width="0.6"/>
    </g>
    <!-- Small fish in background -->
    <g transform="translate(130,180) scale(0.5)" opacity="0.35">
      <ellipse cx="0" cy="0" rx="20" ry="10" fill="#7ecfe0"/>
      <polygon points="-20,0 -30,-8 -30,8" fill="#7ecfe0"/>
      <circle cx="10" cy="-2" r="2.5" fill="#ffffff"/>
    </g>
    <g transform="translate(580,150) scale(0.4)" opacity="0.25">
      <ellipse cx="0" cy="0" rx="20" ry="10" fill="#7ecfe0"/>
      <polygon points="-20,0 -30,-8 -30,8" fill="#7ecfe0"/>
      <circle cx="10" cy="-2" r="2.5" fill="#ffffff"/>
    </g>
    <!-- Floating particles -->
    <circle cx="200" cy="300" r="1" fill="#ffffff" opacity="0.15">
      <animate attributeName="opacity" values="0.05;0.2;0.05" dur="3s" repeatCount="indefinite"/>
    </circle>
    <circle cx="500" cy="200" r="1.2" fill="#ffffff" opacity="0.1">
      <animate attributeName="opacity" values="0.05;0.25;0.05" dur="4s" repeatCount="indefinite"/>
    </circle>
    <circle cx="350" cy="450" r="0.8" fill="#ffffff" opacity="0.12">
      <animate attributeName="opacity" values="0.05;0.18;0.05" dur="3.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="600" cy="380" r="1" fill="#ffffff" opacity="0.1">
      <animate attributeName="opacity" values="0.08;0.22;0.08" dur="4.2s" repeatCount="indefinite"/>
    </circle>
  </svg>
```

</details>

## 6. `glm-5.2:cloud`

**SVG size:** 6479 characters  
**Complexity:** High  
**Shape elements:** 43  
**Distinct colors:** 14  
**Raw response:** 6526 characters  
**Generation time:** 63.8s

![glm-5.2:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/glm-5.2_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for ocean water and lighting)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the shark and scene elements
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Bubbles are present -- the model created an underwater atmosphere
- Water/ocean scene is present -- the model understood the habitat
- Small fish are present -- the model added scene context
- Coral/seaweed is present -- the model decorated the underwater scene
- Baby/cute styling is present -- the model captured the "baby" cue

With 6,479 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 600" width="100%" height="auto">
  <defs>
    <!-- Ocean Background Gradient -->
    <linearGradient id="ocean" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#2BB9DB" />
      <stop offset="60%" stop-color="#1D7A9E" />
      <stop offset="100%" stop-color="#154C6E" />
    </linearGradient>
    <!-- Shark Body Gradient -->
    <linearGradient id="sharkBody" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#A4CADE" />
      <stop offset="100%" stop-color="#6A92B9" />
    </linearGradient>
    <!-- Shark Belly Gradient -->
    <linearGradient id="sharkBelly" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#F5FBFD" />
      <stop offset="100%" stop-color="#D8E8F2" />
    </linearGradient>
    <!-- Bubble Gradient -->
    <radialGradient id="bubble" cx="0.3" cy="0.3" r="0.7">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.8" />
      <stop offset="80%" stop-color="#FFFFFF" stop-opacity="0.1" />
      <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0.2" />
    </radialGradient>
    <!-- Drop Shadow for Shark -->
    <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#0B2C40" flood-opacity="0.4" />
    </filter>
    <!-- Sun Ray Gradient -->
    <linearGradient id="sunray" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.15" />
      <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0" />
    </linearGradient>
  </defs>
  <!-- Background -->
  <rect width="600" height="600" fill="url(#ocean)" />
  <!-- Sun Rays -->
  <polygon points="100,-20 250,600 150,600 50,-20" fill="url(#sunray)" />
  <polygon points="350,-20 450,600 400,600 300,-20" fill="url(#sunray)" />
  <polygon points="500,-20 580,600 540,600 480,-20" fill="url(#sunray)" />
  <!-- Background Bubbles -->
  <circle cx="120" cy="80" r="15" fill="url(#bubble)" />
  <circle cx="90" cy="150" r="8" fill="url(#bubble)" />
  <circle cx="480" cy="120" r="25" fill="url(#bubble)" />
  <circle cx="520" cy="200" r="10" fill="url(#bubble)" />
  <circle cx="50" cy="250" r="12" fill="url(#bubble)" />
  <circle cx="450" cy="450" r="18" fill="url(#bubble)" />
  <circle cx="180" cy="500" r="10" fill="url(#bubble)" />
  <!-- Main Shark Group -->
  <g filter="url(#shadow)">
    <!-- Tail Fin (Behind Body) -->
    <path d="M 440 290 
             C 470 240 500 220 515 205 
             C 505 250 495 275 495 290 
             C 495 305 505 330 515 375 
             C 500 360 470 340 440 290 Z" 
          fill="#7AA4CB" />
    <!-- Dorsal Fin (Behind Body) -->
    <path d="M 270 195 
             C 290 145 320 135 340 195 Z" 
          fill="#7AA4CB" />
    <!-- Pectoral Fin (Behind Body) -->
    <path d="M 220 345 
             C 260 400 300 405 325 350 Z" 
          fill="#7AA4CB" />
    <!-- Main Body -->
    <path d="M 125 285 
             C 125 205 230 175 320 190 
             C 400 205 430 235 450 290 
             C 430 345 400 375 320 380 
             C 230 365 125 360 125 285 Z" 
          fill="url(#sharkBody)" />
    <!-- Belly -->
    <path d="M 125 285 
             C 130 335 180 380 280 375 
             C 370 370 410 340 440 300 
             C 400 355 350 385 250 370 
             C 170 355 130 310 125 285 Z" 
          fill="url(#sharkBelly)" />
    <!-- Gill Slits -->
    <path d="M 260 235 C 255 250 255 265 260 280" stroke="#5C8AB2" stroke-width="4" stroke-linecap="round" fill="none" />
    <path d="M 275 240 C 270 255 270 270 275 285" stroke="#5C8AB2" stroke-width="4" stroke-linecap="round" fill="none" />
    <path d="M 290 245 C 285 260 285 275 290 290" stroke="#5C8AB2" stroke-width="4" stroke-linecap="round" fill="none" />
    <!-- Cheek Blush -->
    <circle cx="195" cy="300" r="18" fill="#FF9AA2" opacity="0.6" />
    <!-- Eye -->
    <g id="eye">
      <circle cx="210" cy="245" r="26" fill="#FFFFFF" />
      <circle cx="205" cy="250" r="16" fill="#2A2A2A" />
      <!-- Catchlights -->
      <circle cx="198" cy="243" r="6" fill="#FFFFFF" />
      <circle cx="213" cy="255" r="2.5" fill="#FFFFFF" opacity="0.8" />
    </g>
    <!-- Eyebrow -->
    <path d="M 190 215 C 200 205 220 205 230 210" stroke="#4F7D9F" stroke-width="5" stroke-linecap="round" fill="none" />
    <!-- Mouth &amp; Teeth -->
    <g id="mouth">
      <!-- Mouth Interior -->
      <path d="M 125 275 
               C 125 300 145 320 175 320 
               C 155 305 140 290 125 275 Z" 
            fill="#E8758A" />
      <!-- Tongue/Bottom Lip -->
      <path d="M 125 275 
               C 130 285 150 310 175 320 
               C 160 310 140 285 125 275 Z" 
            fill="#FFA5B8" />
      <!-- Top Teeth -->
      <polygon points="125,275 135,281 130,287" fill="#FFFFFF" />
      <polygon points="135,281 146,289 140,294" fill="#FFFFFF" />
      <polygon points="146,289 158,298 152,302" fill="#FFFFFF" />
      <polygon points="158,298 170,308 165,312" fill="#FFFFFF" />
    </g>
    <!-- Shark Nostril -->
    <circle cx="140" cy="265" r="2.5" fill="#4F7D9F" />
  </g>
  <!-- Foreground Bubbles -->
  <circle cx="380" cy="150" r="12" fill="url(#bubble)" />
  <circle cx="350" cy="100" r="6" fill="url(#bubble)" />
  <circle cx="70" cy="420" r="20" fill="url(#bubble)" />
  <circle cx="100" cy="380" r="10" fill="url(#bubble)" />
  <!-- Seabed -->
  <path d="M 0 480 
           C 100 460 200 500 300 480 
           C 400 460 500 500 600 470 
           L 600 600 L 0 600 Z" 
        fill="#0E3048" />
  <!-- Seaweed -->
  <path d="M 60 550 Q 40 480 60 400 Q 80 320 50 260" stroke="#246B5A" stroke-width="10" fill="none" stroke-linecap="round" />
  <path d="M 80 550 Q 100 490 80 420 Q 60 350 90 300" stroke="#2E8B57" stroke-width="8" fill="none" stroke-linecap="round" />
  <path d="M 530 550 Q 510 490 530 430 Q 550 370 520 320" stroke="#246B5A" stroke-width="10" fill="none" stroke-linecap="round" />
  <path d="M 550 550 Q 570 500 550 440 Q 530 380 560 340" stroke="#2E8B57" stroke-width="8" fill="none" stroke-linecap="round" />
  <!-- Little Starfish on the right -->
  <g transform="translate(480, 540) scale(0.8)">
    <path d="M 0,-25 
             C 10,-10 20,-10 25,0 
             C 10,5 10,15 0,25 
             C -10,15 -10,5 -25,0 
             C -10,-10 -10,-10 0,-25 Z" 
          fill="#FF8C42" stroke="#E55A2B" stroke-width="3" />
    <circle cx="0" cy="0" r="4" fill="#E55A2B" />
  </g>
</svg>
```

</details>

## 7. `gpt-oss:120b-cloud`

**SVG size:** 1213 characters  
**Complexity:** Compact  
**Shape elements:** 8  
**Distinct colors:** 6  
**Raw response:** 1934 characters  
**Generation time:** 6.9s

![gpt-oss:120b-cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/gpt-oss_120b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character

With 1,213 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg
    viewBox="0 0 200 120"
    xmlns="http://www.w3.org/2000/svg"
    width="200"
    height="auto"
    style="background:#e0f7ff"
>
  <!-- Body -->
  <ellipse
      cx="80"
      cy="60"
      rx="70"
      ry="35"
      fill="#a0c4ff"
      stroke="#5a8ac4"
      stroke-width="2"
  />
  <!-- Tail -->
  <polygon
      points="150,45 180,60 150,75"
      fill="#7a7a7a"
      stroke="#5a8ac4"
      stroke-width="2"
  />
  <!-- Dorsal fin -->
  <polygon
      points="70,20 95,20 80,45"
      fill="#7a7a7a"
      stroke="#5a8ac4"
      stroke-width="2"
  />
  <!-- Pectoral fin (left) -->
  <polygon
      points="60,70 45,85 55,73"
      fill="#7a7a7a"
      stroke="#5a8ac4"
      stroke-width="2"
  />
  <!-- Pectoral fin (right) -->
  <polygon
      points="100,70 115,85 105,73"
      fill="#7a7a7a"
      stroke="#5a8ac4"
      stroke-width="2"
  />
  <!-- Eye -->
  <circle
      cx="45"
      cy="55"
      r="5"
      fill="#fff"
      stroke="#5a8ac4"
      stroke-width="1"
  />
  <circle cx="46" cy="54" r="2" fill="#000" />
  <!-- Mouth (smile) -->
  <path
      d="M 40 70 Q 55 80 70 70"
      stroke="#5a8ac4"
      stroke-width="2"
      fill="none"
      stroke-linecap="round"
  />
</svg>
```

</details>

## 8. `kimi-k2.6:cloud`

**SVG size:** 3425 characters  
**Complexity:** Medium  
**Shape elements:** 13  
**Distinct colors:** 8  
**Raw response:** 17103 characters  
**Generation time:** 51.4s

![kimi-k2.6:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/kimi-k2.6_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for ocean water and lighting)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the shark and scene elements
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Baby/cute styling is present -- the model captured the "baby" cue

With 3,425 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 300" width="100%" height="auto">
            <defs>
                <!-- Body Gradient -->
                <linearGradient id="sharkBody" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#00BFFF;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#0080FF;stop-opacity:1" />
                </linearGradient>
                <!-- Belly Gradient -->
                <linearGradient id="sharkBelly" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#FFFFFF;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#E0F7FA;stop-opacity:1" />
                </linearGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                    <feMerge>
                        <feMergeNode in="coloredBlur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            </defs>

            <g id="shark-group">
                <!-- Tail (Back Fin) -->
                <g id="tail-fin" transform="translate(380, 150)">
                    <path d="M0,0 Q30,-40 60,-70 Q50,0 60,70 Q30,40 0,0" fill="url(#sharkBody)" stroke="#0066CC" stroke-width="2" stroke-linejoin="round" />
                </g>

                <!-- Dorsal Fin (Top) -->
                <path d="M220,120 Q240,60 280,50 L290,120 Z" fill="url(#sharkBody)" stroke="#0066CC" stroke-width="2" stroke-linejoin="round" />

                <!-- Pectoral Fin (Side/Bottom) -->
                <path d="M220,160 Q200,200 180,220 L250,180 Z" fill="#0080FF" stroke="#0066CC" stroke-width="2" stroke-linejoin="round" />

                <!-- Main Body -->
                <!-- Using a complex path for a shark silhouette -->
                <path id="body-shape"
                      d="M100,150 Q100,100 180,90 L350,120 Q420,130 380,150 Q420,170 350,180 L180,210 Q100,200 100,150 Z"
                      fill="url(#sharkBody)"
                      stroke="#0066CC"
                      stroke-width="3"
                      stroke-linejoin="round" />

                <!-- Belly Patch -->
                <path d="M110,150 Q250,150 360,160 L340,180 Q220,180 110,150 Z" fill="url(#sharkBelly)" opacity="0.8" />

                <!-- Gills -->
                <g stroke="#005599" stroke-width="3" stroke-linecap="round">
                    <path d="M250,110 Q250,130 250,150" />
                    <path d="M265,112 Q265,132 265,152" />
                    <path d="M280,114 Q280,134 280,154" />
                </g>

                <!-- Face Details -->
                <!-- Eye -->
                <g id="eye-group" transform="translate(140, 130)">
                    <circle cx="0" cy="0" r="10" fill="white" stroke="#0066CC" stroke-width="2"/>
                    <circle id="pupil" cx="2" cy="0" r="4" fill="black" />
                    <circle cx="4" cy="-3" r="2" fill="white" opacity="0.8" />
                </g>

                <!-- Mouth -->
                <path id="mouth" d="M120,170 Q140,185 160,175" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round" />

                <!-- Cheek Blush -->
                <ellipse cx="150" cy="160" rx="10" ry="5" fill="#FFB6C1" opacity="0.4" />
            </g>
        </svg>

```

</details>

## 9. `minimax-m2.7:cloud`

**SVG size:** 2341 characters  
**Complexity:** Medium  
**Shape elements:** 18  
**Distinct colors:** 10  
**Raw response:** 2520 characters  
**Generation time:** 59.3s

![minimax-m2.7:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/minimax-m2.7_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning the shark and scene elements
- Includes animation elements -- the model attempted motion (e.g., swimming, bubbles)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Bubbles are present -- the model created an underwater atmosphere
- Water/ocean scene is present -- the model understood the habitat

With 2,341 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200" width="100%" height="auto">
  <!-- Background: Light Water -->
  <rect width="300" height="200" fill="#E0F7FA" rx="15" />
  <!-- Bubbles -->
  <circle cx="220" cy="60" r="6" fill="#B2EBF2" opacity="0.7">
    <animate attributeName="cy" values="60;40" dur="3s" repeatCount="indefinite" />
    <animate attributeName="opacity" values="0.7;0" dur="3s" repeatCount="indefinite" />
  </circle>
  <circle cx="240" cy="80" r="9" fill="#B2EBF2" opacity="0.7">
    <animate attributeName="cy" values="80;50" dur="4s" repeatCount="indefinite" />
    <animate attributeName="opacity" values="0.7;0" dur="4s" repeatCount="indefinite" />
  </circle>
  <circle cx="210" cy="90" r="4" fill="#B2EBF2" opacity="0.7">
    <animate attributeName="cy" values="90;70" dur="2.5s" repeatCount="indefinite" />
    <animate attributeName="opacity" values="0.7;0" dur="2.5s" repeatCount="indefinite" />
  </circle>
  <!-- Group: The Shark -->
  <g transform="translate(10, 10)">
    <!-- Dorsal Fin (Back) -->
    <path d="M 160 60 L 180 25 L 200 60 Z" fill="#0288D1" />
    <!-- Tail Fin -->
    <path d="M 210 90 L 270 50 L 240 90 L 270 130 Z" fill="#0288D1" />
    <!-- Main Body -->
    <path d="M 40 90 Q 150 10 210 90 Q 150 170 40 90 Z" fill="#29B6F6" />
    <!-- Belly Patch (Lighter Color) -->
    <path d="M 45 90 Q 145 150 205 90 Q 145 110 45 90 Z" fill="#E1F5FE" />
    <!-- Pectoral Fin (Side) -->
    <path d="M 110 115 L 90 150 L 140 125 Z" fill="#0288D1" />
    <!-- Eye -->
    <g>
      <circle cx="80" cy="85" r="12" fill="white" stroke="#01579B" stroke-width="1"/>
      <circle cx="80" cy="85" r="5" fill="black" />
      <circle cx="82" cy="83" r="2" fill="white" /> <!-- Eye Shine -->
    </g>
    <!-- Eye -->
    <g>
      <circle cx="110" cy="85" r="12" fill="white" stroke="#01579B" stroke-width="1"/>
      <circle cx="110" cy="85" r="5" fill="black" />
      <circle cx="112" cy="83" r="2" fill="white" /> <!-- Eye Shine -->
    </g>
    <!-- Mouth (Smile) -->
    <path d="M 70 105 Q 95 125 120 105" fill="none" stroke="#01579B" stroke-width="3" stroke-linecap="round" />
    <!-- Gill Lines -->
    <path d="M 125 80 L 125 100" stroke="#0277BD" stroke-width="2" opacity="0.5" />
    <path d="M 130 82 L 130 98" stroke="#0277BD" stroke-width="2" opacity="0.5" />
  </g>
</svg>
```

</details>

## 10. `minimax-m3:cloud`

**SVG size:** 2860 characters  
**Complexity:** Medium  
**Shape elements:** 24  
**Distinct colors:** 6  
**Raw response:** 3450 characters  
**Generation time:** 13.6s

![minimax-m3:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/minimax-m3_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for ocean water and lighting)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Bubbles are present -- the model created an underwater atmosphere
- Water/ocean scene is present -- the model understood the habitat
- Baby/cute styling is present -- the model captured the "baby" cue

With 2,860 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="400" height="auto">
  <!-- Background (ocean) -->
  <defs>
    <linearGradient id="ocean" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#87CEEB;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#4682B4;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="sharkBody" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#708090;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#5F7A8A;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#C0C0C0;stop-opacity:1" />
    </linearGradient>
  </defs>
  <!-- Ocean background -->
  <rect width="400" height="300" fill="url(#ocean)"/>
  <!-- Bubbles -->
  <circle cx="50" cy="50" r="5" fill="white" opacity="0.6"/>
  <circle cx="70" cy="80" r="3" fill="white" opacity="0.6"/>
  <circle cx="40" cy="120" r="4" fill="white" opacity="0.6"/>
  <circle cx="350" cy="60" r="4" fill="white" opacity="0.6"/>
  <circle cx="370" cy="100" r="3" fill="white" opacity="0.6"/>
  <!-- Baby Shark Body -->
  <ellipse cx="200" cy="170" rx="130" ry="55" fill="url(#sharkBody)" stroke="#2F4F4F" stroke-width="2"/>
  <!-- Belly (lighter color) -->
  <ellipse cx="200" cy="195" rx="100" ry="30" fill="#E8E8E8" opacity="0.7"/>
  <!-- Tail Fin -->
  <path d="M 320 170 Q 370 140 360 170 Q 370 200 320 180 Z" fill="#708090" stroke="#2F4F4F" stroke-width="2"/>
  <!-- Dorsal Fin -->
  <path d="M 200 115 Q 210 75 230 110 Z" fill="#708090" stroke="#2F4F4F" stroke-width="2"/>
  <!-- Side Fin (pectoral) -->
  <path d="M 180 185 Q 160 210 140 200 Q 155 190 175 185 Z" fill="#708090" stroke="#2F4F4F" stroke-width="2"/>
  <!-- Head -->
  <ellipse cx="90" cy="165" rx="55" ry="50" fill="url(#sharkBody)" stroke="#2F4F4F" stroke-width="2"/>
  <!-- Snout -->
  <path d="M 40 165 Q 25 160 30 170 Q 25 180 40 175 Z" fill="#708090" stroke="#2F4F4F" stroke-width="2"/>
  <!-- Mouth (smiling) -->
  <path d="M 45 185 Q 70 200 90 185" fill="none" stroke="#2F4F4F" stroke-width="2.5" stroke-linecap="round"/>
  <!-- Gills -->
  <path d="M 120 150 Q 125 170 120 190" fill="none" stroke="#2F4F4F" stroke-width="1.5"/>
  <path d="M 128 150 Q 133 170 128 190" fill="none" stroke="#2F4F4F" stroke-width="1.5"/>
  <path d="M 136 150 Q 141 170 136 190" fill="none" stroke="#2F4F4F" stroke-width="1.5"/>
  <!-- Eyes (big and cute) -->
  <circle cx="85" cy="150" r="14" fill="white" stroke="#2F4F4F" stroke-width="2"/>
  <circle cx="88" cy="152" r="8" fill="#2F4F4F"/>
  <circle cx="90" cy="149" r="3" fill="white"/>
  <!-- Cheek blush -->
  <circle cx="75" cy="180" r="6" fill="#FFB6C1" opacity="0.6"/>
  <!-- Small sparkles -->
  <g fill="white" opacity="0.8">
    <circle cx="280" cy="140" r="1.5"/>
    <circle cx="290" cy="155" r="1"/>
    <circle cx="270" cy="160" r="1"/>
  </g>
</svg>
```

</details>

## 11. `nemotron-3-super:cloud`

**SVG size:** 939 characters  
**Complexity:** Compact  
**Shape elements:** 7  
**Distinct colors:** 5  
**Raw response:** 1154 characters  
**Generation time:** 12.1s

![nemotron-3-super:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/nemotron-3-super_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Baby/cute styling is present -- the model captured the "baby" cue

With 939 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg width="200" height="auto" viewBox="0 0 200 150"
     xmlns="http://www.w3.org/2000/svg" aria-labelledby="title desc">
  <title id="title">Baby Shark</title>
  <desc id="desc">A cute, simplified baby shark silhouette.</desc>
  <!-- Body -->
  <path d="
        M20,75
        C20,30 80,10 120,40
        C140,30 150,60 140,80
        C150,100 120,110 100,90
        C80,110 50,100 50,80
        C50,60 20,75 20,75 Z"
        fill="#aad3e8"/>
  <!-- Belly (lighter underside) -->
  <ellipse cx="100" cy="95" rx="40" ry="20" fill="#f0f8ff"/>
  <!-- Dorsal fin -->
  <polygon points="100,55 115,30 115,70" fill="#aad3e8"/>
  <!-- Tail fin -->
  <polygon points="30,80 0,60 30,40" fill="#aad3e8"/>
  <!-- Pectoral fin (left side) -->
  <path d="M70,90 q10,-15 20,0"
        stroke="#aad3e8" stroke-width="4" fill="none"/>
  <!-- Eye -->
  <circle cx="115" cy="70" r="4" fill="white"/>
  <circle cx="116" cy="70" r="2" fill="black"/>
</svg>
```

</details>

## 12. `nemotron-3-ultra:cloud`

**SVG size:** 5200 characters  
**Complexity:** Medium  
**Shape elements:** 25  
**Distinct colors:** 8  
**Raw response:** 6135 characters  
**Generation time:** 236.2s

![nemotron-3-ultra:cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/nemotron-3-ultra_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for ocean water and lighting)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the shark and scene elements
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Gills are present -- the model captured anatomical detail
- Bubbles are present -- the model created an underwater atmosphere
- Baby/cute styling is present -- the model captured the "baby" cue

With 5,200 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg width="400" height="auto" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Definitions for Gradients &amp; Filters -->
  <defs>
    <!-- Main Body Gradient (Blue/Grey) -->
    <linearGradient id="bodyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#4A90E2;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#357ABD;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2C6BA0;stop-opacity:1" />
    </linearGradient>
    <!-- Belly Gradient (Lighter) -->
    <linearGradient id="bellyGradient" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#FFFFFF;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#E0F0FF;stop-opacity:1" />
    </linearGradient>
    <!-- Fin Gradient -->
    <linearGradient id="finGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#357ABD;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2C6BA0;stop-opacity:1" />
    </linearGradient>
    <!-- Eye Shine -->
    <radialGradient id="eyeShine" cx="30%" cy="30%" r="70%">
      <stop offset="0%" style="stop-color:#FFFFFF;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFFFFF;stop-opacity:0" />
    </radialGradient>
    <!-- Drop Shadow Filter -->
    <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="3" result="blur"/>
      <feOffset dx="2" dy="3" result="offsetBlur"/>
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.3"/>
      </feComponentTransfer>
      <feMerge> 
        <feMergeNode in="offsetBlur"/>
        <feMergeNode in="SourceGraphic"/> 
      </feMerge>
    </filter>
  </defs>
  <!-- Main Group with Shadow -->
  <g filter="url(#dropShadow)">
    <!-- TAIL (Caudal Fin) - Behind Body -->
    <path d="M 100 200 
             Q 50 150, 40 200 
             Q 55 200, 60 200 
             Q 50 250, 100 200 Z" 
          fill="url(#finGradient)" stroke="#2C6BA0" stroke-width="1.5"/>
    <!-- DORSAL FIN (Top) -->
    <path d="M 220 100 
             Q 200 40, 180 100 
             Q 195 100, 200 100 
             Q 210 70, 220 100 Z" 
          fill="url(#finGradient)" stroke="#2C6BA0" stroke-width="1.5"/>
    <!-- MAIN BODY -->
    <!-- Using an ellipse/path combo for a chubby shark shape -->
    <ellipse cx="220" cy="200" rx="150" ry="80" fill="url(#bodyGradient)" stroke="#2C6BA0" stroke-width="2"/>
    <!-- Nose Tip Override (Rounder) -->
    <ellipse cx="365" cy="200" rx="20" ry="75" fill="url(#bodyGradient)" stroke="#2C6BA0" stroke-width="2"/>
    <!-- BELLY (White Underbelly) -->
    <path d="M 100 200 
             Q 180 260, 300 230 
             Q 340 220, 360 200 
             Q 340 180, 300 170 
             Q 180 140, 100 200 Z" 
          fill="url(#bellyGradient)" opacity="0.9" stroke="none"/>
    <!-- GILLS (3 Lines) -->
    <g stroke="#2C6BA0" stroke-width="1.5" stroke-linecap="round" opacity="0.6">
      <line x1="130" y1="185" x2="160" y2="195" />
      <line x1="130" y1="200" x2="160" y2="205" />
      <line x1="130" y1="215" x2="160" y2="215" />
    </g>
    <!-- PECTORAL FINS (Side Fins) -->
    <!-- Left Fin (Closer) -->
    <path d="M 180 220 
             Q 140 280, 100 250 
             Q 120 240, 150 230 
             Q 160 225, 180 220 Z" 
          fill="url(#finGradient)" stroke="#2C6BA0" stroke-width="1.5"/>
    <!-- Right Fin (Further back - perspective) -->
    <path d="M 190 190 
             Q 160 140, 130 170 
             Q 145 175, 165 185 
             Q 175 190, 190 190 Z" 
          fill="url(#finGradient)" stroke="#2C6BA0" stroke-width="1.5" opacity="0.8"/>
    <!-- EYE (Big and Cute) -->
    <g transform="translate(300, 170)">
      <!-- Eye White / Base -->
      <circle cx="0" cy="0" r="28" fill="#FFFFFF" stroke="#2C6BA0" stroke-width="2"/>
      <!-- Iris (Blue) -->
      <circle cx="0" cy="0" r="18" fill="#2E86DE"/>
      <!-- Pupil (Black) -->
      <circle cx="5" cy="-3" r="10" fill="#1A1A2E"/>
      <!-- Highlights -->
      <circle cx="-6" cy="-8" r="5" fill="url(#eyeShine)"/>
      <circle cx="10" cy="5" r="2.5" fill="#FFFFFF" opacity="0.8"/>
    </g>
    <!-- EYEBROW (Expression) -->
    <path d="M 275 145 Q 300 130, 325 145" 
          fill="none" stroke="#2C6BA0" stroke-width="3" stroke-linecap="round"/>
    <!-- SMILE -->
    <path d="M 340 210 Q 365 230, 380 215" 
          fill="none" stroke="#2C6BA0" stroke-width="3" stroke-linecap="round"/>
    <!-- TONGUE (Optional Peek) -->
    <ellipse cx="365" cy="220" rx="8" ry="4" fill="#FF6B8A" opacity="0.8"/>
    <!-- BLUSH MARKS -->
    <g fill="#FFB6C1" opacity="0.6">
      <ellipse cx="290" cy="205" rx="12" ry="6" transform="rotate(-15 290 205)"/>
      <ellipse cx="330" cy="210" rx="10" ry="5" transform="rotate(10 330 210)"/>
    </g>
  </g> <!-- End Shadow Group -->
  <!-- BUBBLES (Decoration) -->
  <g fill="none" stroke="#4A90E2" stroke-width="1.5" opacity="0.5">
    <circle cx="380" cy="80" r="10"/>
    <circle cx="395" cy="40" r="6"/>
    <circle cx="350" cy="50" r="4"/>
    <circle cx="50" cy="300" r="8"/>
    <circle cx="30" cy="340" r="5"/>
  </g>
</svg>
```

</details>

## 13. `qwen3.5:397b-cloud`

**SVG size:** 2023 characters  
**Complexity:** Medium  
**Shape elements:** 19  
**Distinct colors:** 7  
**Raw response:** 2203 characters  
**Generation time:** 32.0s

![qwen3.5:397b-cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/qwen3.5_397b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Fin/dorsal fin is present -- the model captured the shark's defining feature
- Mouth is present -- the model rendered the shark's mouth
- Tail is present -- the model captured the shark's propulsion
- Eyes are present -- the model gave the shark character
- Baby/cute styling is present -- the model captured the "baby" cue

With 2,023 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background Circle (Optional) -->
  <circle cx="250" cy="250" r="240" fill="#E0F7FA" />
  <!-- Tail Fin (Back Layer) -->
  <path d="M 360 250 Q 450 200 480 250 Q 450 300 360 250" fill="#4FC3F7" stroke="#0277BD" stroke-width="4" stroke-linejoin="round"/>
  <!-- Main Body -->
  <ellipse cx="250" cy="250" rx="140" ry="110" fill="#4FC3F7" stroke="#0277BD" stroke-width="4"/>
  <!-- Belly -->
  <ellipse cx="250" cy="270" rx="100" ry="70" fill="#FFFFFF" stroke="#0277BD" stroke-width="4"/>
  <!-- Dorsal Fin (Top) -->
  <path d="M 200 160 Q 250 60 300 160" fill="#4FC3F7" stroke="#0277BD" stroke-width="4" stroke-linejoin="round"/>
  <!-- Side Fins -->
  <path d="M 170 310 Q 130 350 180 330" fill="#4FC3F7" stroke="#0277BD" stroke-width="4" stroke-linejoin="round"/>
  <path d="M 330 310 Q 370 350 320 330" fill="#4FC3F7" stroke="#0277BD" stroke-width="4" stroke-linejoin="round"/>
  <!-- Eyes (White part) -->
  <circle cx="210" cy="230" r="28" fill="#FFFFFF" stroke="#0277BD" stroke-width="4"/>
  <circle cx="290" cy="230" r="28" fill="#FFFFFF" stroke="#0277BD" stroke-width="4"/>
  <!-- Pupils -->
  <circle cx="215" cy="235" r="10" fill="#000000"/>
  <circle cx="295" cy="235" r="10" fill="#000000"/>
  <!-- Eye Shine -->
  <circle cx="220" cy="225" r="4" fill="#FFFFFF"/>
  <circle cx="300" cy="225" r="4" fill="#FFFFFF"/>
  <!-- Blush -->
  <ellipse cx="180" cy="260" rx="15" ry="8" fill="#FFCDD2" opacity="0.8"/>
  <ellipse cx="320" cy="260" rx="15" ry="8" fill="#FFCDD2" opacity="0.8"/>
  <!-- Mouth -->
  <path d="M 210 280 Q 250 340 290 280" fill="#FF8A80" stroke="#0277BD" stroke-width="4" stroke-linejoin="round"/>
  <!-- Teeth -->
  <path d="M 220 285 L 230 305 L 240 285" fill="#FFFFFF" stroke="#0277BD" stroke-width="2"/>
  <path d="M 245 295 L 255 315 L 265 295" fill="#FFFFFF" stroke="#0277BD" stroke-width="2"/>
  <path d="M 270 285 L 280 305 L 290 285" fill="#FFFFFF" stroke="#0277BD" stroke-width="2"/>
</svg>
```

</details>

## 14. `deepseek-v4-flash:0731-cloud`

**SVG size:** 6060 characters  
**Complexity:** High  
**Shape elements:** 53  
**Distinct colors:** 6  
**Raw response:** 6060 characters  
**Generation time:** ~100s

![deepseek-v4-flash:0731-cloud baby shark SVG](/assets/img/diagrams/ollama-svg-shark/deepseek-v4-flash_0731-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- The model rendered its interpretation of the "baby shark" prompt -- review the SVG above for how well it captured the subject, composition, and key visual cues

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="100%" height="auto">
  <defs>
    <!-- Shark Body Gradient -->
    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#81D4FA" />
      <stop offset="100%" stop-color="#29B6F6" />
    </linearGradient>
    <!-- Drop Shadow for the Shark -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="8" flood-color="#000000" flood-opacity="0.1" />
    </filter>
  </defs>
  <!-- Background Ocean Circle -->
  <circle cx="200" cy="200" r="180" fill="#E0F7FA" />
  <!-- Background Waves -->
  <g stroke="#B2EBF2" stroke-width="4" fill="none" stroke-linecap="round">
    <path d="M 40 120 Q 60 100, 80 120 T 120 120" />
    <path d="M 280 280 Q 300 260, 320 280 T 360 280" />
    <path d="M 330 140 Q 350 120, 370 140 T 410 140" />
    <path d="M 20 60 Q 40 40, 60 60 T 100 60" />
    <path d="M 120 320 Q 140 300, 160 320 T 200 320" />
  </g>
  <!-- Seaweed / Coral -->
  <g fill="none" stroke-width="6" stroke-linecap="round">
    <path d="M 40 380 Q 50 360, 40 340 Q 30 320, 40 300" stroke="#80CBC4" />
    <path d="M 60 380 Q 70 350, 60 320 Q 50 290, 60 260" stroke="#4DB6AC" />
    <path d="M 350 380 Q 340 350, 350 320 Q 360 290, 350 260" stroke="#80CBC4" />
    <path d="M 370 380 Q 380 360, 370 340" stroke="#4DB6AC" />
  </g>
  <!-- Bubbles -->
  <g fill="#FFFFFF" fill-opacity="0.7" stroke="#B2EBF2" stroke-width="2">
    <circle cx="60" cy="100" r="15" />
    <circle cx="45" cy="130" r="8" />
    <circle cx="80" cy="80" r="5" />
    <circle cx="330" cy="80" r="12" />
    <circle cx="310" cy="50" r="6" />
    <circle cx="360" cy="100" r="7" />
    <circle cx="180" cy="60" r="5" />
    <circle cx="260" cy="40" r="9" />
  </g>
  <!-- Bubble Highlights -->
  <g fill="#FFFFFF">
    <circle cx="55" cy="95" r="3" />
    <circle cx="42" cy="127" r="2" />
    <circle cx="78" cy="78" r="1.5" />
    <circle cx="326" cy="76" r="2.5" />
    <circle cx="308" cy="47" r="1.5" />
    <circle cx="357" cy="97" r="2" />
    <circle cx="178" cy="58" r="1.5" />
    <circle cx="257" cy="37" r="2" />
  </g>
  <!-- Sparkles -->
  <g fill="#FFFFFF">
    <path d="M 120 70 Q 120 80, 130 80 Q 120 80, 120 90 Q 120 80, 110 80 Q 120 80, 120 70" />
    <path d="M 280 240 Q 280 250, 290 250 Q 280 250, 280 260 Q 280 250, 270 250 Q 280 250, 280 240" />
    <path d="M 320 180 Q 320 185, 325 185 Q 320 185, 320 190 Q 320 185, 315 185 Q 320 185, 320 180" />
    <path d="M 80 240 Q 80 245, 85 245 Q 80 245, 80 250 Q 80 245, 75 245 Q 80 245, 80 240" />
  </g>
  <!-- Baby Shark Group -->
  <g filter="url(#shadow)">
    <!-- Tail Fin Inner Lines -->
    <g stroke="#0288D1" stroke-width="4" fill="none" stroke-linecap="round">
      <path d="M 358 195 Q 370 180, 380 125" />
      <path d="M 358 205 Q 370 220, 380 275" />
    </g>
    <!-- Dorsal Fin (Top) -->
    <path d="M 180 125 C 190 60, 230 50, 250 75 C 260 95, 255 120, 245 130 C 220 110, 200 120, 180 125 Z" 
          fill="url(#bodyGrad)" stroke="#0288D1" stroke-width="6" stroke-linejoin="round" />
    <!-- Pectoral Fin (Side) -->
    <path d="M 140 240 C 130 280, 160 310, 190 300 C 180 270, 160 250, 140 240 Z" 
          fill="url(#bodyGrad)" stroke="#0288D1" stroke-width="6" stroke-linejoin="round" />
    <!-- Main Body -->
    <path d="M 90 190 
             C 90 140, 150 110, 230 120 
             C 300 130, 340 160, 360 170 
             L 390 110 
             Q 370 180, 360 200 
             Q 370 220, 390 290 
             L 350 230 
             C 300 270, 180 290, 110 270 
             C 85 260, 80 230, 90 190 Z" 
          fill="url(#bodyGrad)" stroke="#0288D1" stroke-width="6" stroke-linejoin="round" />
    <!-- Belly (Lighter color) -->
    <path d="M 90 190 
             C 95 250, 140 275, 200 280 
             C 270 280, 320 260, 350 230 
             C 310 265, 220 285, 140 265 
             C 105 255, 90 230, 90 190 Z" 
          fill="#FFFFFF" stroke="#FFFFFF" stroke-width="12" stroke-linejoin="round" />
    <!-- Belly Outline (to re-add the body outline over the belly) -->
    <path d="M 90 190 
             C 95 250, 140 275, 200 280 
             C 270 280, 320 260, 350 230 
             C 310 265, 220 285, 140 265 
             C 105 255, 90 230, 90 190 Z" 
          fill="none" stroke="#0288D1" stroke-width="6" stroke-linejoin="round" />
    <!-- Nostril -->
    <path d="M 85 195 Q 90 192, 95 195" fill="none" stroke="#0288D1" stroke-width="4" stroke-linecap="round" />
    <!-- Gills -->
    <g fill="none" stroke="#0288D1" stroke-width="4" stroke-linecap="round">
      <path d="M 200 185 Q 205 195, 200 205" />
      <path d="M 210 182 Q 215 195, 210 208" />
    </g>
    <!-- Mouth &amp; Tongue -->
    <path d="M 105 215 Q 130 245, 165 220 Q 130 220, 105 215" 
          fill="#D32F2F" stroke="#0288D1" stroke-width="4" stroke-linejoin="round" />
    <path d="M 115 225 Q 130 240, 145 225 Z" 
          fill="#FF8A80" stroke="#0288D1" stroke-width="2" stroke-linejoin="round" />
    <!-- Small Lip -->
    <path d="M 100 215 Q 110 208, 120 215" 
          fill="url(#bodyGrad)" stroke="#0288D1" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
    <!-- Left Eye (Far) -->
    <circle cx="140" cy="165" r="9" fill="#FFFFFF" stroke="#0288D1" stroke-width="4" />
    <circle cx="143" cy="165" r="4.5" fill="#333333" />
    <circle cx="145" cy="163" r="1.5" fill="#FFFFFF" />
    <!-- Right Eye (Near) -->
    <circle cx="180" cy="165" r="12" fill="#FFFFFF" stroke="#0288D1" stroke-width="4" />
    <circle cx="185" cy="165" r="6" fill="#333333" />
    <circle cx="187" cy="163" r="2" fill="#FFFFFF" />
    <!-- Blush / Cheeks -->
    <circle cx="145" cy="195" r="7" fill="#FFCDD2" opacity="0.8" />
    <circle cx="185" cy="195" r="9" fill="#FFCDD2" opacity="0.8" />
    <!-- Cute Back Details -->
    <g stroke="#0288D1" stroke-width="4" stroke-linecap="round">
      <path d="M 280 155 L 290 155" />
      <path d="M 295 165 L 305 165" />
    </g>
  </g>
</svg>
```

</details>


## Conclusion

That's all 14 baby shark SVGs from the active Ollama cloud models. The marine-life prompt tested a different skill set than our previous prompts: organic curves, underwater atmosphere, anatomical features, and cute character styling.

**No single model is best at everything.** A model that produces a stunning cherry blossom may produce a basic shark, and vice versa. The best model for you depends on your specific use case:

- For **maximum detail**: look at the "Very high" complexity models
- For **speed**: look at the generation times in each section
- For **code quality**: look at the raw SVG source -- clean, well-structured code is easier to customize
- For **concept adherence**: check which models captured the shark's fins, teeth, tail, and baby styling

Try the same prompt yourself with [Ollama Cloud](https://ollama.com/cloud) and see if you agree with our analysis. And check out the other posts in our SVG benchmark series:

- [Duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/)
- [Duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/)
- [Duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/)
- [Cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/)
- [Duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/)
