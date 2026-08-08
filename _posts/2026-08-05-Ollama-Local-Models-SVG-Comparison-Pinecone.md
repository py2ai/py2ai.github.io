---
layout: post
title: "Which Ollama Local Model is Best? Pinecone SVG Comparison (15 Models)"
description: "Compare 15 Ollama local models on a pinecone SVG prompt. Find the best LLM for Fibonacci and nature SVG scenes. You decide the winner."
date: 2026-08-05
header-img: "img/post-bg.jpg"
permalink: /Ollama-Local-Models-SVG-Comparison-Pinecone/
tags:
  - AI
  - Ollama
  - SVG
  - LLM
  - Comparison
  - Benchmark
  - Best LLM
  - SVG generation
  - Pinecone
  - Fibonacci
  - Nature
  - Mathematics
  - Botanical
  - Spiral
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama local model comparison, pinecone SVG, AI pinecone drawing, LLM SVG benchmark, AI Fibonacci SVG, Fibonacci spiral SVG, golden ratio SVG, phyllotaxis SVG, pinecone scales, AI nature art, complex SVG scene, botanical illustration, golden angle SVG"
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/local-deep-research/local-deep-research-architecture.svg
---

# Which Ollama Local Model is Best? Pinecone SVG Comparison (15 Models)

Continuing our Fibonacci series, we asked our models to draw **a pinecone** -- one of nature's most striking examples of Fibonacci spirals. A pinecone's scales spiral outward from the base in two directions, with the number of spirals in each direction typically being consecutive Fibonacci numbers (like 5 and 8, or 8 and 13). This prompt tests whether models can render that spiral scale pattern, the overall elongated egg shape, the woody brown color palette, and the 3D depth that makes each scale appear to overlap the one below it.

The prompt was: `Make an svg image of a pinecone`

This is the thirteenth in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/), [duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/), [baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/), [octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/), [FIFA World Cup 2026](/Ollama-Cloud-Models-SVG-Comparison-Fifa-Worldcup-2026/), [elephant on a skateboard](/Ollama-Cloud-Models-SVG-Comparison-Elephant-Skateboard/), [flying helicopter](/Ollama-Cloud-Models-SVG-Comparison-Flying-Helicopter/), [sunflower with seeds](/Ollama-Local-Models-SVG-Comparison-Sunflower/), [pineapple](/Ollama-Local-Models-SVG-Comparison-Pineapple/), [nautilus shell](/Ollama-Local-Models-SVG-Comparison-Nautilus-Shell/), [dates palm tree](/Ollama-Local-Models-SVG-Comparison-Dates-Palm/).

**Why a pinecone?** The pinecone prompt is a **depth and spiral stress test** because it combines: (1) **Fibonacci spiral scales** -- the scales spiral outward in two directions following Fibonacci numbers; a model with mathematical intuition should attempt overlapping scale patterns rather than flat stripes, (2) **3D depth via overlap** -- each scale overlaps the one below it, creating a sense of depth; this requires careful z-ordering and shape layering, (3) **Egg shape** -- a pinecone is an elongated egg/oval, rounded at the base and pointed at the top, (4) **Woody texture** -- the scales have a matte brown color with subtle shading, (5) **Scale geometry** -- each scale is a curved diamond or teardrop shape, not a simple rectangle, and (6) **Density vs. readability** -- a real pinecone has many scales; the model must decide how many to draw while keeping the SVG file manageable.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Pinecone SVGs

The pinecone prompt rewards different things than previous prompts. Here are the criteria to use:

- **Spiral scale pattern**: Does the model arrange scales in visible spiral patterns? Or just rows/columns? True Fibonacci spirals are the gold standard.
- **3D depth and overlap**: Do the scales overlap each other to create depth? Or are they flat side-by-side?
- **Overall shape**: Is the body an elongated egg shape? Or a circle/cylinder?
- **Scale geometry**: Are the scales curved diamonds or teardrops? Or simple rectangles/triangles?
- **Color palette**: Does it use woody brown tones with shading? Or flat single color?
- **Recognizability**: Would someone immediately recognize this as a pinecone?
- **SVG code quality**: Does it use `<defs>`, `<use>`, and transforms to efficiently generate repeating scales?

## How It Works

The script discovers all locally installed models via the Ollama API (`/api/tags`), then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with minimal post-processing (adding `width="100%" height="auto"` for responsive embedding and fixing XML errors so the SVG renders in browsers).

Unlike our cloud model benchmarks, these models run entirely on the local GPU -- no cloud subscription or network round-trip required. This means generation times reflect local hardware performance, and model sizes range from 1B to 31B parameters. Embedding, vision, and OCR models are automatically skipped.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `cieloforge/qwen2.5-14B-instruct-spec:latest` | 643 | 4 | 3 | Compact | Compact |
| 2 | `deepseek-r1:1.5b` | 2671 | 6 | 3 | Compact | Compact |
| 3 | `deepseek-r1:7b` | 873 | 3 | 1 | Compact | Compact |
| 4 | `gemma3:1b-it-qat` | 265 | 2 | 2 | Compact | Compact |
| 5 | `gemma3:4b` | 620 | 6 | 7 | Compact | Compact |
| 6 | `gemma4:12b` | 1735 | 19 | 5 | Compact | Compact |
| 7 | `gemma4:26b-a4b-it-qat` | 3181 | 4 | 12 | Compact | Compact |
| 8 | `lfm2.5:latest` | 1247 | 6 | 4 | Compact | Compact |
| 9 | `llama3.1:8b` | 511 | 3 | 2 | Compact | Compact |
| 10 | `qwen2.5-coder:7b` | 346 | 2 | 1 | Compact | Compact |
| 11 | `qwen2.5:3b` | 271 | 3 | 2 | Compact | Compact |
| 12 | `qwen2.5:7b` | 611 | 5 | 4 | Compact | Compact |
| 13 | `qwen3.5:4b` | 1002 | 7 | 5 | Compact | Compact |
| 14 | `qwen3.5:9b` | 170 | 0 | 0 | Compact | Compact |
| 15 | `qwen3.5:latest` | 3076 | 20 | 17 | Medium | Balanced |
| 16 | `gemma4:31b-it-qat` | - | - | - | - | Failed |
| 17 | `gemma4:latest` | - | - | - | - | Failed |
| 18 | `jeffgreen311/Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged:latest` | - | - | - | - | Failed |
| 19 | `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2:latest` | - | - | - | - | Failed |
| 20 | `qwen2.5-coder-14b:latest` | - | - | - | - | Failed |
| 21 | `qwen3:14b` | - | - | - | - | Failed |
| 22 | `qwen3:8b` | - | - | - | - | Failed |
| 23 | `SetneufPT/Qwen3.6-27B-MTP_Q3_32K_16GB-GPU:latest` | - | - | - | - | Failed |
| 24 | `SetneufPT/Qwen3.6-27B.MTP_Q3_32K_16GB-GPU:latest` | - | - | - | - | Failed |
| 25 | `VladimirGav/Qwen3.6-27B-16GB-VRAM-Uncensored:latest` | - | - | - | - | Failed |

**15 out of 25** models produced a valid SVG. The 10 that failed either returned an error or did not include a valid `<svg>...</svg>` block in their response.

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most accurate Fibonacci spiral scales**: look for models whose SVG shows visible spiral patterns in the scale arrangement
- **You want 3D depth with overlapping scales**: check the per-model analysis for overlap and z-ordering
- **You want the most detailed pinecone**: pick models labeled "Very high" complexity in the table above
- **You want efficient repeating-scale code**: look for models that used `<use>` and `<defs>` elements
- **You want a small SVG for web embedding**: pick models with "Compact" verdict
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"

Now read on for the full per-model breakdown and judge for yourself.

## 1. `cieloforge/qwen2.5-14B-instruct-spec:latest`

**SVG size:** 643 characters  
**Complexity:** Compact  
**Shape elements:** 4  
**Distinct colors:** 3  
**Raw response:** 1452 characters  
**Generation time:** 67.4s

![cieloforge/qwen2.5-14B-instruct-spec:latest pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/cieloforge_qwen2.5-14B-instruct-spec_latest.svg)

### Analysis

This SVG contains approximately **4 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background for the overall shape -->
  <rect x="0" y="0" width="200" height="300" fill="#d4e5f1" />
  <!-- First pointy portion of the pinecone -->
  <polygon points="96.7, 8.5 16.7, 71.3 112.7, 173.6 24.0, 132.0 62.3, 41.9" fill="#ffc107" />
  <!-- Second pointy portion of the pinecone -->
  <polygon points="50.8, 183.1 4.4, 152.5 124.7, 261.8 102.0, 129.4" fill="#ffc107" />
  <!-- Long stem of the pinecone -->
  <path d="M100.0,361 L145.0,361 M100.0,408 L145.0,408 M100.0,270 L145.0,270" stroke="#aaa" fill="none" stroke-width="5" />
</svg>
```

</details>

## 2. `deepseek-r1:1.5b`

**SVG size:** 2671 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 3  
**Raw response:** 3328 characters  
**Generation time:** 18.7s

![deepseek-r1:1.5b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/deepseek-r1_1.5b.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" name="pinecone" viewBox="0 0 240 180" width="100%" height="auto">
  <g id="petals">
    <!-- Main Ring Group -->
    <ring groups=num-of-rings=7>
      <rect ring-seq rotate=sweepDirection="0.5" opacity="0.6" 
        angle=ranges start-angle start-degree end-angle end-degree reverse
          <a href="# petalAngles" @-1 29 0 'A' stroke-stroke-linecap="round" stroke-width>
        />
      </rect>
    </ring>
    <!-- Petals -->
    <g id="petalGroup">
      <path d="M16.875,21.75263347  0 0 0 429C3.545,20.477307 5.404,18.37
             8.015,16.002 10.94,14.2643 10.94,10.2083 0 0 10.94,-6.0263
            -9.7083,-4.087-9.7083-3.5427 0 0-3.5746 4.9762c(2.135,4.0395)
             (2.6829,10.14) 4.08,16.049-2.964, -3.007z"
          stroke-width="2" stroke="#1d7578"
          fillopacity="0"
        />
      </g>
    <!-- Center Point -->
    <circle cx="120" cy="89"
      d="M135.234,85 68.478 -79.091 24.554 0
      A29.47 29.47 0 0 1 14.934 29.47C3.6404,1.805
       1.0989,1.0989.7351,2.113L243.534,95c-.0234.0234-2.036 13.907 2.083
       12.898-.0668.8051.8055 .0234.0234.34 1.109L65.234,47.228c-
       .0234-.0234-.7766-1.109-.4024-1.678a37.47 37.47
      11.284 31.855 0 0 14.34.1635.126.126
  .96.0834.14C135.234,109 170 286.5
    -270.32 L243.534,89zM0,0c0 .65.35.35 1
     1 1.58 -1.066 .66 1.07L-26 5a2.127 2.127
    0 0 18-4.917c-1.05-.51-.93-.74-
      .76-1.29 3.09 -5.993 4 5z"
        opacity="0" fillStyle="transparent"/>
      </g>
    <!-- Close Paths -->
    <path endcap="round" stroke-width="1" stroke="#2d7e80" 
      d="M16.875,81 0 0 0 429C3.545,20.477307 5.404,18.37
     8.015,16.002 10.94,14.2643 10.94,10.2084 0 0 10.94,-6.0263
    -9.708,3-9.708,3.5427 0 3.5746 4.9763c(2.135,4.0395)
     (2.683,10.14) 4.08,16.048-2.964,-3.007z"/>
    <path endcap="round" stroke-width="2" stroke="#45b34e"
      d="M69,17a27 27 0 1 1 27zM0 0
       a826 826 0 1 1-165.72 165.72A8.347 8.347
        0 1 195.717 195.717 6.315A84.069 84.069
         0 1 4.754-8.323 6.683L69,0a11.153 11.153
          0 1 0 0 3.31-7.805A7.407 7.407 0 1
           7.42 21.966A2.073 2.073 0 1.5 13.514
            13.514A2.073 2.073 0 1 12.227-.486
           .9804 9.603 3.686.522 7.282c(31.066)-50
            -4.62-33.8-12.97-57.915z"/>
    <path endcap="round" stroke-width="1" opacity="0" fillstyle="transparent".206.122C29.031 97.645
      60.728 133.682 90 183.567A33.575 33.575 0 0
     114.352 120.16C161.941 42.37 134.83 28.475
      20.348 6.521 14.857-.63.206-.205zM120 97a31.465 31.465 0 0 1-185.7
      -4.042A44.708 44.708 0 0 1-62.914-.47
     .279C19.343.808 5.896 3.931 2.43.3.77.3
    z"/>
  </g>
</svg>
```

</details>

## 3. `deepseek-r1:7b`

**SVG size:** 873 characters  
**Complexity:** Compact  
**Shape elements:** 3  
**Distinct colors:** 1  
**Raw response:** 1834 characters  
**Generation time:** 57.4s

![deepseek-r1:7b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/deepseek-r1_7b.svg)

### Analysis

This SVG contains approximately **3 shape elements** and uses **1 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox=" -10 -10 20 20 " width="100%" height="auto">
  <rect x="-10" y="-10" width="20" height="20" fill="brown"/>
  <path d="M5,8
           r4,-6L7.395,8H5l-1.396, -2.846-a3.58 3.58 0 0 0 -6.33,1.323C5.47, 7.695 4, 7.426 3.236, 8c-.196, .596-1.396, 1.095-3.236, 1.327a7.24, 7.24 0 0 0 1.432,-6.664C4.354,.886 6.368 .118 9, -2c1.09, -.422 2.9, -.797 4.108,-1.323a4.992,4.992 0 0 0 2.758,7.712C6.475,15.15 7.427,17.05 8.5,17c.7.7.668,1.326-.668,1.952a4.992,4.992 0 0 0 -2.758,7.712C8.204,15.12 .779,11.782 3.31,-6c2.703,-5.182 4.405,-9.846 4.699,-12.743a7,7 0 0 0 -6,0c-.556, -2.307-3.473,-4.366-4.679,-4.485l-.088, -.307 1.395,1.395c3.562,0 4.679,0 4.679,-2a2.99,2.99 0 0 0 -5.508,-1c-.693, .075-1.19,1.184-1.992,1.848c-.8,.7.219,1.445.93,1.772l2.705,-2H5z"/>
          <path d="M0,0 l5,5m5,-5 l-3.61,3.62" />
          </svg>
```

</details>

## 4. `gemma3:1b-it-qat`

**SVG size:** 265 characters  
**Complexity:** Compact  
**Shape elements:** 2  
**Distinct colors:** 2  
**Raw response:** 2558 characters  
**Generation time:** 41.9s

![gemma3:1b-it-qat pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/gemma3_1b-it-qat.svg)

### Analysis

This SVG contains approximately **2 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 100 150" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <path d="M 60,70 C 60,80 L 90,70 S 80,75 C 80,73 L 90,75" fill="none" stroke="#465B42" stroke-width="1">
    <circle cx="60" cy="70" r="30" fill="#FFD700"/>
  </path>
</svg>
```

</details>

## 5. `gemma3:4b`

**SVG size:** 620 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 7  
**Raw response:** 4273 characters  
**Generation time:** 96.4s

![gemma3:4b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/gemma3_4b.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Cone Body -->
  <path d="M50,10 L10,60 L90,60 Z" fill="#778899" stroke="#465F7E"/>
  <!-- Scales (overlapping to create texture) -->
  <ellipse cx="30" cy="55" rx="15" ry="12" fill="#98C1D9" stroke="#465F7E"/>
  <ellipse cx="70" cy="55" rx="15" ry="12" fill="#C1D9E4" stroke="#465F7E"/>
  <ellipse cx="45" cy="65" rx="12" ry="10" fill="#CEBFF3" stroke="#465F7E"/>
  <ellipse cx="55" cy="80" rx="10" ry="8" fill="#D0C9DA" stroke="#465F7E"/>
  <ellipse cx="65" cy="70" rx="12" ry="10" fill="#A3ADBD" stroke="#465F7E"/>
</svg>
```

</details>

## 6. `gemma4:12b`

**SVG size:** 1735 characters  
**Complexity:** Compact  
**Shape elements:** 19  
**Distinct colors:** 5  
**Raw response:** 2517 characters  
**Generation time:** 156.4s

![gemma4:12b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/gemma4_12b.svg)

### Analysis

This SVG contains approximately **19 shape elements** and uses **5 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 250" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <defs>
    <linearGradient id="pineconeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#7d5c4c;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#3e2c22;stop-opacity:1" />
    </linearGradient>
  </defs>
  <!-- Background Shadow -->
  <ellipse cx="100" cy="140" rx="65" ry="100" fill="#2d1f1a" />
  <!-- Main Body Shape -->
  <path d="M100,30 
           C80,30 60,70 60,100 
           C60,140 75,190 90,210 
           C100,220 110,210 120,200
           C130,180 145,130 145,100
           C151,70 130,30 100,30 Z" fill="url(#pineconeGrad)" />
  <!-- Lower Scaled Layers (Darker) -->
  <g fill="#4a352a">
    <path d="M85,190 Q90,215 100,215 Q110,215 115,190" />
    <path d="M76,180 Q80,205 93,205 Q106,205 110,180" />
    <path d="M68,160 Q72,185 81,193" />
    <path d="M124,180 Q130,205 115,205 Q100,205 100,180" />
    <path d="M137,160 Q133,185 123,193" />
  </g>
  <!-- Middle Scaled Layers (Mid Tone) -->
  <g fill="#6b4c3f">
    <path d="M82,170 Q85,190 95,192" />
    <path d="M110,170 Q115,190 105,192" />
    <path d="M74,145 Q78,165 86,168" />
    <path d="M132,145 Q128,165 118,168" />
  </g>
  <!-- Upper Scaled Layers (Lighter/Highlight) -->
  <g fill="#9c7d6b">
    <path d="M95,130 Q98,145 102,145" />
    <path d="M90,110 Q92,125 95,125" />
    <path d="M106,110 Q108,125 107,125" />
    <path d="M103,90 Q105,100 104,100" />
  </g>
  <!-- Decorative Spikes/Scales -->
  <g fill="#4a352a" opacity="0.6">
    <path d="M75,80 Q78,95 80,90" />
    <path d="M125,80 Q122,95 120,90" />
    <path d="M85,60 Q88,75 90,70" />
    <path d="M115,60 Q112,75 110,70" />
  </g>
</svg>
```

</details>

## 7. `gemma4:26b-a4b-it-qat`

**SVG size:** 3181 characters  
**Complexity:** Compact  
**Shape elements:** 4  
**Distinct colors:** 12  
**Raw response:** 4171 characters  
**Generation time:** 303.3s

![gemma4:26b-a4b-it-qat pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/gemma4_26b-a4b-it-qat.svg)

### Analysis

This SVG contains approximately **4 shape elements** and uses **12 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating motifs like seeds, scales, or leaflets
- Includes gradient fills for richer visual depth
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background / Shadow for Depth -->
  <defs>
    <radialGradient id="coreGrad" cx="50%" cy="40%" r="50%">
      <stop offset="0%" stop-color="#5d4037" />
      <stop offset="100%" stop-color="#2e1a16" />
    </radialGradient>
    <!-- The "Scale" Shape -->
    <path id="scaleShape" d="M 0,-10 C 15,-10 20,5 0,20 C -20,5 -15,-10 0,-10 Z" />
    <!-- Shadow filter for the scales -->
    <filter id="dropShadow" x="-20%" y="-20%" width="150%" height="150%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="1.5" />
      <feOffset dx="1" dy="1" result="offset" />
      <feSpecularLighting in="SourceAlpha" specularConstant="0.5" specularExponent="2" lighting-color="#ffffff">
        <fePointLight x="-40" y="-50" z="30" />
      </feSpecularLighting>
    </filter>
  </defs>
  <!-- 1. The Dark Core (Base Shape) -->
  <ellipse cx="100" cy="160" rx="55" ry="130" fill="#2e1a16" />
  <!-- Grouping scales to allow for potential transformations -->
  <g transform="translate(100, 170)">
    <!-- Bottom Layer (Largest Scales) -->
    <g id="layer-bottom">
      <use href="#scaleShape" x="-45" y="80" fill="#3e2723" transform="rotate(-10)" />
      <use href="#scaleShape" x="0" y="90" fill="#4e342e" transform="rotate(0)" />
      <use href="#scaleShape" x="45" y="80" fill="#3e2723" transform="rotate(10)" />
    </g>
    <!-- Second Layer -->
    <g id="layer-mid-low">
      <use href="#scaleShape" x="-40" y="50" fill="#5d4037" transform="rotate(-15)" />
      <use href="#scaleShape" x="-12" y="60" fill="#795548" transform="rotate(-5)" />
      <use href="#scaleShape" x="12" y="60" fill="#795548" transform="rotate(5)" />
      <use href="#scaleShape" x="40" y="50" fill="#5d4037" transform="rotate(15)" />
    </g>
    <!-- Third Layer -->
    <g id="layer-mid-high">
      <use href="#scaleShape" x="-35" y="20" fill="#8d6e63" transform="rotate(-20)" />
      <use href="#scaleShape" x="-12" y="30" fill="#a1887f" transform="rotate(-5)" />
      <use href="#scaleShape" x="12" y="30" fill="#a1887f" transform="rotate(5)" />
      <use href="#scaleShape" x="35" y="20" fill="#8d6e63" transform="rotate(20)" />
    </g>
    <!-- Fourth Layer -->
    <g id="layer-top">
      <use href="#scaleShape" x="-25" y="-20" fill="#a1887f" transform="rotate(-15)" />
      <use href="#scaleShape" x="0" y="-10" fill="#d7ccc8" transform="rotate(0)" />
      <use href="#scaleShape" x="25" y="-20" fill="#a1888f" transform="rotate(15)" />
    </g>
    <!-- The Tip (Smallest Scales) -->
    <g id="layer-tip">
      <use href="#scaleShape" x="-10" y="-60" fill="#efebe9" transform="rotate(-5) scale(0.7)" />
      <use href="#scaleShape" x="0" y="-70" fill="#ffffff" transform="scale(0.5)" />
      <use href="#scaleShape" x="10" y="-60" fill="#efebe9" transform="rotate(5) scale(0.7)" />
    </g>
  </g>
  <!-- Adding some texture/highlights to the whole pinecone -->
  <ellipse cx="85" cy="120" rx="20" ry="40" fill="white" opacity="0.05" transform="rotate(10 85 120)" />
  <ellipse cx="115" cy="200" rx="15" ry="30" fill="white" opacity="0.05" transform="rotate(-10 115 200)" />
</svg>
```

</details>

## 8. `lfm2.5:latest`

**SVG size:** 1247 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 4  
**Raw response:** 2207 characters  
**Generation time:** 446.6s

![lfm2.5:latest pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/lfm2.5_latest.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 200 130"
      aria-label="Stylized pinecone" width="100%" height="auto">
  <!-- Background (optional) -->
  <rect width="100%" height="20" fill="#FDFDFD"/>
  <!-- ==== PINE CONE ==== -->
  <g id="pinecone" transform="translate(50, 70)">
    <!-- Trunk (stem) -->
    <line x1="95" y1="110" x2="95" y2="80" stroke="#8B4513"
          stroke-width="6"/>
    <!-- Body – a large elliptical “cap” -->
    <ellipse cx="100" cy="55" rx="80" ry="40"
            fill="url(#earth-tones)"/>
    <!-- Top curve (the rounded part of the cone) -->
    <path d="M45 85 Q30 55 15 45 L70 45 Q110 65 L95 85 Z"
          fill="#A0522D"/>
    <!-- Overlapping scale wedges – left and right -->
    <!-- Left wedge -->
    <polygon points="100,115 80,95 120,105"
              fill="#8B4513"/>
    <!-- Right wedge -->
    <polygon points="60,115 50,85 70,105"
              fill="#8B4513"/>
  </g>
  <!-- ==== EARTH TONES FILL ==== -->
  <defs>
    <linearGradient id="earth-tones" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   style="stop-color:#8B4513;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#654321;stop-opacity:0.8"/>
    </linearGradient>
  </defs>
</svg>
```

</details>

## 9. `llama3.1:8b`

**SVG size:** 511 characters  
**Complexity:** Compact  
**Shape elements:** 3  
**Distinct colors:** 2  
**Raw response:** 1168 characters  
**Generation time:** 184.3s

![llama3.1:8b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/llama3.1_8b.svg)

### Analysis

This SVG contains approximately **3 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="auto">
  <!-- Outer circle (pinecone surface) -->
  <circle cx="50" cy="50" r="40"
           fill="#789012"/>  <!-- Brown pinecone color -->
  <!-- Spiral ridges ( pinecone scales) -->
  <path d="M 40 30 A 8,8 0 1,1 60,80
            A 8,8 0 1,1 40,70 Z"
           fill="#4B5326"/>  <!-- Medium green-brown scale color -->
  <!-- Center spike (pinecone tip) -->
  <path d="M 50 -20 L 50 40
          M 52 -15 L 48 35"/>
</svg>
```

</details>

## 10. `qwen2.5-coder:7b`

**SVG size:** 346 characters  
**Complexity:** Compact  
**Shape elements:** 2  
**Distinct colors:** 1  
**Raw response:** 768 characters  
**Generation time:** 176.7s

![qwen2.5-coder:7b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/qwen2.5-coder_7b.svg)

### Analysis

This SVG contains approximately **2 shape elements** and uses **1 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 95" width="100%" height="auto">
    <style>
        .pinecone {
            fill: green;
        }
    </style>
    <!-- Base */
    <path class="pinecone" d="M20,85 Q40,65 60,85 T90,85" stroke-width="8"/>
    <!- Cone -->
    <ellipse cx="50" cy="70" rx="15" ry="50" fill="brown"/>
</svg>
```

</details>

## 11. `qwen2.5:3b`

**SVG size:** 271 characters  
**Complexity:** Compact  
**Shape elements:** 3  
**Distinct colors:** 2  
**Raw response:** 1214 characters  
**Generation time:** 100.1s

![qwen2.5:3b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/qwen2.5_3b.svg)

### Analysis

This SVG contains approximately **3 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <circle cx="20" cy="50" r="40" fill="#FFCC00" />
  <path d="
      M 68,37 Q 91,10,93,0 L 96,0 L 100,10 l -2.5,-10
      Z
    " stroke="black" fill="none"/>
  <circle cx="100" cy="84" r="12" />
</svg>
```

</details>

## 12. `qwen2.5:7b`

**SVG size:** 611 characters  
**Complexity:** Compact  
**Shape elements:** 5  
**Distinct colors:** 4  
**Raw response:** 1440 characters  
**Generation time:** 103.4s

![qwen2.5:7b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/qwen2.5_7b.svg)

### Analysis

This SVG contains approximately **5 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background layer -->
  <rect x="0" y="0" width="200" height="200" fill="#DAD7C1" />
  <!-- Cone body -->
  <polygon points="50,100 150,40 190,100 150,160 50,160 0,100"
           fill="#8B632F" />
  <!-- Leaf detail -->
  <polygon points="75,120 125,120 100,145"
           fill="#9E6C3A" />
  <!-- Small leaves -->
  <polyline points="80,150 110,140 100,165"
            fill="#F3B95D" stroke="none" fill-opacity="0.7" />
  <polyline points="120,150 90,140 100,165"
            fill="#F3B95D" stroke="none" fill-opacity="0.7" />
</svg>
```

</details>

## 13. `qwen3.5:4b`

**SVG size:** 1002 characters  
**Complexity:** Compact  
**Shape elements:** 7  
**Distinct colors:** 5  
**Raw response:** 1025 characters  
**Generation time:** 186.3s

![qwen3.5:4b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/qwen3.5_4b.svg)

### Analysis

This SVG contains approximately **7 shape elements** and uses **5 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="-30 67 -92 121" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
    <!-- Background Circle (Shadow/Base) -->
    <circle cx="-63.5" cy="98.5" r="45" fill="#BFAA8C"/>
    <!-- Pinecone Body Scales / Texture Layers -->
    <!-- Layer 1: Back/Center Shadow Scale -->
    <polygon points="-72,90 -60,93 -74,98" fill="#BBA75F" />
    <polygon points="-70,96 -61,98 -71,102" fill="#AEB68C" opacity="0.3"/>
    <!-- Layer 2: Middle Front Scale -->
    <path d="M-45,-20 L-28,-28 L-22,-18 L-49,-16 Z M-27,-20 L-15,-27 -8,-17" 
          fill="#D4C3A8"/>
    <!-- Layer 3: Sides/Ridges -->
    <path d="M-50,98 L-40,103 M-60,110 L-52,114 -58,120" stroke="#BBA75F" 
          fill="none"/>
    <!-- Main Cone Body Shape (Oval/Trapezoid) -->
    <path d="M-90,-32 C-60,-32 40,-12 60,8 L55,105 L-75,105 Z" fill="#AABFAE"/>
    <!-- Detail Ridge Lines (To simulate texture and depth) -->
    <path d="M-1,9 M-8,37 M4,-41 M16,-25 M16,37 M-16,9 -13,-30" 
          fill="#BFAA8C"/>
</svg>
```

</details>

## 14. `qwen3.5:9b`

**SVG size:** 170 characters  
**Complexity:** Compact  
**Shape elements:** 0  
**Distinct colors:** 0  
**Raw response:** 918 characters  
**Generation time:** 216.1s

![qwen3.5:9b pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/qwen3.5_9b.svg)

### Analysis

This SVG contains approximately **0 shape elements** and uses **0 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning and rotation

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="-18 -9 36 18" preserveAspectRatio="xMidYMax slice" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
 <g transform="scale(1,-1)">
  </g>
</svg>
```

</details>

## 15. `qwen3.5:latest`

**SVG size:** 3076 characters  
**Complexity:** Medium  
**Shape elements:** 20  
**Distinct colors:** 17  
**Raw response:** 3118 characters  
**Generation time:** 270.3s

![qwen3.5:latest pinecone SVG](/assets/img/diagrams/ollama-svg-pinecone/qwen3.5_latest.svg)

### Analysis

This SVG contains approximately **20 shape elements** and uses **17 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<pattern>` elements for repeating textures (e.g., pineapple skin or trunk scars)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Includes animation elements -- the model attempted motion
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 250" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <defs>
    <!-- Gradient for pinecone body -->
    <linearGradient id="brownGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#5d4e37;stop-opacity:1"/>
      <stop offset="50%" style="stop-color:#6b5a40;stop-opacity:1"/>
      <stop offset="100%" style="stop-color:#4a3f2c;stop-opacity:1"/>
    </linearGradient>
    <radialGradient id="brownGrad2" cx="30%" cy="30%" r="70%">
      <stop offset="0%" style="stop-color:#8b6f55;stop-opacity:1"/>
      <stop offset="40%" style="stop-color:#6d5a44;stop-opacity:1"/>
      <stop offset="80%" style="stop-color:#4a3f2c;stop-opacity:1"/>
    </radialGradient>
  </defs>
  <!-- Main pinecone base -->
  <ellipse cx="100" cy="65" rx="60" ry="30" fill="#7d684e" opacity="0.9" />
  <circle cx="100" cy="220" r="20" fill="#2a2218" opacity="0.7"/>
  <!-- Scale layer 1 (lowest) -->
  <path d="M35,65 Q100,95 165,65 L175,45 Q100,62 25,45 Z" fill="#8b725a"/>
  <!-- Scale layer 2 -->
  <path d="M30,45 Q100,95 170,45 L180,25 Q100,65 20,25 Z" fill="#9c826a"/>
  <!-- Scale layer 3 -->
  <path d="M32,35 Q100,105 168,35 L178,15 Q100,72 22,15 Z" fill="#ac947a"/>
  <!-- Scale layer 4 -->
  <path d="M34,30 Q100,110 166,30 L176,10 Q100,82 24,10 Z" fill="#9c8b5a"/>
  <!-- Scale layer 5 -->
  <path d="M36,25 Q100,115 164,25 L172,5 Q100,85 26,5 Z" fill="#9c8b4a"/>
  <!-- Scale layer 6 -->
  <path d="M37,22 Q100,120 162,22 L168,-2 Q100,92 28,-2 Z" fill="#ac9d5a"/>
  <!-- Scale layer 7 (middle) -->
  <path d="M45,18 Q100,95 155,18 L163-1 Q100,78 47,-1 Z" fill="#bba27a"/>
  <!-- Scale layer 8 (middle) -->
  <path d="M50,16 Q100,92 150,16 L143-5 Q100,75 52,-5 Z" fill="#bba27a"/>
  <!-- Scale layer 9 (upper) -->
  <path d="M58,14 Q100,82 148,14 L135-12 Q100,65 66,-12 Z" fill="#caba7a"/>
  <!-- Scale layer 10 (upper) -->
  <path d="M65,12 Q98,72 141,12 L135-15 Q95,-8 80,-18 Z" fill="#bba26a"/>
  <!-- Scale layer 11 (upper-mid) -->
  <path d="M72,10 Q85,52 145,10 L230-31 Q95,-32 60,-2 Z" fill="#bba25a"/>
  <!-- Scale layer 12 (upper-mid) -->
  <path d="M75,8 Q82,48 226.5-22 Q95,-35 90,-2Z" fill="#caba4a"/>
  <!-- Top scales detail -->
  <circle cx="100" cy="-15" r="6" fill="#7d684e"/>
  <ellipse cx="94" cy="-25" rx="3" ry="5" transform="rotate(15)" fill="#6b5a40" />
  <!-- Highlight lines for texture -->
  <defs>
    <pattern id="scalePattern" x="0" y="0" width="10" height="12">
      <path d="M-5,0 Q5,-8 -3.5,-9 L8.5,-7 Z M4,6 Q-6,1 3,7 L-5,7 Z"/>
    </pattern>
  </defs>
  <!-- Highlight overlay -->
  <ellipse cx="5" cy="-20" rx="4" ry="6" fill="white" opacity="0.2">
    <animate attributeName="opacity" values="0.1;0.3;0.1" dur="3s" repeatCount="indefinite"/>
  </ellipse>
  <ellipse cx="-5" cy="-8" rx="4" ry="6" fill="#ffffff" opacity="0.15">
    <animate attributeName="opacity" values="0.2;0.1;0.2" dur="3s" repeatCount="indefinite"/>
  </ellipse>
  <!-- Pinecone shadow on ground -->
  <path d="M40,80 Q95,-6 160,79 L180,85 L475,222 Z M40,80 Z" 
        fill="#3d3a30" opacity="0.3"/>
</svg>
```

</details>

## Failed Models

The following 10 models did not produce a valid SVG (either returned an error or the response did not contain a parseable `<svg>...</svg>` block):

- `gemma4:31b-it-qat` -- TimeoutError: timed out
- `gemma4:latest` -- TimeoutError: timed out
- `jeffgreen311/Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged:latest` -- TimeoutError: timed out
- `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2:latest` -- TimeoutError: timed out
- `qwen2.5-coder-14b:latest` -- no SVG block in response
- `qwen3:14b` -- TimeoutError: timed out
- `qwen3:8b` -- TimeoutError: timed out
- `SetneufPT/Qwen3.6-27B-MTP_Q3_32K_16GB-GPU:latest` -- TimeoutError: timed out
- `SetneufPT/Qwen3.6-27B.MTP_Q3_32K_16GB-GPU:latest` -- TimeoutError: timed out
- `VladimirGav/Qwen3.6-27B-16GB-VRAM-Uncensored:latest` -- TimeoutError: timed out

## Conclusion

We asked 25 Ollama local models to draw **a pinecone** -- a subject deeply connected to Fibonacci mathematics. The results reveal each model's natural instinct for mathematical patterns, organic curves, and natural color palettes.

**Key takeaways:**

- **Mathematical intuition varies widely**: some models attempted Fibonacci spiral arrangements naturally, while others defaulted to simple grids or flat shapes. The models that attempted spirals demonstrate a deeper understanding of natural geometry.
- **Detail and file size trade-off**: models that produced the richest scenes (Very high complexity) also generated the largest SVG files. For web embedding, "Balanced" or "Detailed" models may be more practical.
- **Color palettes differ dramatically**: some models used sophisticated gradients and 15+ distinct colors, while others used as few as 4 flat colors. More colors generally means a more lifelike result.
- **Code structure quality varies**: the best models used `<defs>`, `<use>`, and `<pattern>` elements to efficiently generate repeating structures (seeds, scales, leaflets). This is a strong signal of model capability for practical SVG work.
- **No single model is best at everything**: the "right" model depends on whether you prioritize mathematical accuracy, visual beauty, file size, or code quality.

**Try it yourself**: run the benchmark script with your own prompt and see how the models perform on your specific use case. The full code is available in the [ollama-svg-benchmark repository](https://github.com/py2ai/ollama-svg-benchmark).

---

*Which model do you think drew the best pinecone? Let us know in the comments below!*
