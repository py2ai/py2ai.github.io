---
layout: post
title: "Which Ollama Local Model is Best? Nautilus Shell SVG Comparison (15 Models)"
description: "Compare 15 Ollama local models on a nautilus shell SVG prompt. Find the best LLM for Fibonacci and nature SVG scenes. You decide the winner."
date: 2026-08-05
header-img: "img/post-bg.jpg"
permalink: /Ollama-Local-Models-SVG-Comparison-Nautilus-Shell/
tags:
  - AI
  - Ollama
  - SVG
  - LLM
  - Comparison
  - Benchmark
  - Best LLM
  - SVG generation
  - Nautilus
  - Fibonacci
  - Nature
  - Mathematics
  - Golden-Spiral
  - Marine
  - Logarithmic
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama local model comparison, nautilus shell SVG, AI nautilus drawing, LLM SVG benchmark, AI Fibonacci SVG, golden spiral SVG, logarithmic spiral SVG, golden ratio SVG, mathematical art SVG, AI nature art, complex SVG scene, marine biology, shell illustration, phi spiral SVG"
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/local-deep-research/local-deep-research-architecture.svg
---

# Which Ollama Local Model is Best? Nautilus Shell SVG Comparison (15 Models)

The nautilus shell is the iconic embodiment of the **golden spiral** -- a logarithmic spiral whose growth factor is phi (the golden ratio, approximately 1.618). We asked our models to draw **a nautilus shell** to test whether they can render this most mathematical of natural forms. A real nautilus shell is a perfect logarithmic spiral with chambers, a pearlescent surface with brown zebra stripes, and a hollow center where the spiral begins. This prompt tests curve precision (the spiral must expand smoothly), color gradients (pearlescent white to brown stripes), and structural detail (chambers inside the spiral).

The prompt was: `Make an svg image of a nautilus shell`

This is the fourteenth in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/), [duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/), [baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/), [octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/), [FIFA World Cup 2026](/Ollama-Cloud-Models-SVG-Comparison-Fifa-Worldcup-2026/), [elephant on a skateboard](/Ollama-Cloud-Models-SVG-Comparison-Elephant-Skateboard/), [flying helicopter](/Ollama-Cloud-Models-SVG-Comparison-Flying-Helicopter/), [sunflower with seeds](/Ollama-Local-Models-SVG-Comparison-Sunflower/), [pineapple](/Ollama-Local-Models-SVG-Comparison-Pineapple/), [pinecone](/Ollama-Local-Models-SVG-Comparison-Pinecone/), [dates palm tree](/Ollama-Local-Models-SVG-Comparison-Dates-Palm/).

**Why a nautilus shell?** The nautilus shell prompt is a **curve precision stress test** because it combines: (1) **Logarithmic spiral** -- the shell's shape is a perfect golden spiral that grows by phi each quarter turn; a model with mathematical intuition should use `<path>` with Bezier curves to approximate this smooth expansion, (2) **Chamber divisions** -- a real nautilus shell has internal chambers separated by walls; the model should show these as curved lines crossing the spiral, (3) **Pearlescent coloring** -- the shell surface has iridescent white/cream tones with brown zebra-like stripes, (4) **Hollow center** -- the spiral opens from a tiny center to a large opening, (5) **Bilateral symmetry of stripes** -- the brown stripes on the exterior follow the curve of the shell, and (6) **Curve smoothness** -- the spiral must be smooth, not angular or polygonal.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Nautilus Shell SVGs

The nautilus shell prompt rewards different things than previous prompts. Here are the criteria to use:

- **Spiral accuracy**: Does the model draw a smooth logarithmic spiral that expands gradually? Or is it a flat circle / angular polygon?
- **Chamber divisions**: Does it show internal chamber walls as curved lines crossing the spiral?
- **Stripe pattern**: Does it render brown zebra stripes on the exterior following the curve?
- **Color gradient**: Does it use pearlescent white/cream tones with shading? Or flat color?
- **Spiral smoothness**: Is the spiral drawn with smooth Bezier curves? Or straight line segments?
- **Hollow center**: Does the spiral start from a tiny center point and expand outward?
- **SVG code quality**: Does it use `<path>` with `d` attribute curves (`C`, `Q`, `A` commands)? Or just straight lines?

## How It Works

The script discovers all locally installed models via the Ollama API (`/api/tags`), then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with minimal post-processing (adding `width="100%" height="auto"` for responsive embedding and fixing XML errors so the SVG renders in browsers).

Unlike our cloud model benchmarks, these models run entirely on the local GPU -- no cloud subscription or network round-trip required. This means generation times reflect local hardware performance, and model sizes range from 1B to 31B parameters. Embedding, vision, and OCR models are automatically skipped.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `cieloforge/qwen2.5-14B-instruct-spec:latest` | 1355 | 5 | 2 | Compact | Compact |
| 2 | `deepseek-r1:1.5b` | 1849 | 4 | 0 | Compact | Compact |
| 3 | `deepseek-r1:7b` | 207 | 0 | 0 | Compact | Compact |
| 4 | `gemma3:1b-it-qat` | 544 | 3 | 4 | Compact | Compact |
| 5 | `gemma3:4b` | 685 | 3 | 3 | Compact | Compact |
| 6 | `gemma4:12b` | 1131 | 6 | 4 | Compact | Compact |
| 7 | `gemma4:26b-a4b-it-qat` | 2258 | 7 | 9 | Compact | Compact |
| 8 | `lfm2.5:latest` | 1074 | 5 | 10 | Compact | Compact |
| 9 | `llama3.1:8b` | 675 | 2 | 0 | Compact | Compact |
| 10 | `qwen2.5-coder-14b:latest` | 708 | 2 | 3 | Compact | Compact |
| 11 | `qwen2.5-coder:7b` | 568 | 2 | 0 | Compact | Compact |
| 12 | `qwen2.5:7b` | 331 | 1 | 2 | Compact | Compact |
| 13 | `qwen3.5:4b` | 1875 | 5 | 5 | Compact | Compact |
| 14 | `qwen3.5:9b` | 1867 | 6 | 7 | Compact | Compact |
| 15 | `qwen3.5:latest` | 1047 | 2 | 4 | Compact | Compact |
| 16 | `gemma4:31b-it-qat` | - | - | - | - | Failed |
| 17 | `gemma4:latest` | - | - | - | - | Failed |
| 18 | `jeffgreen311/Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged:latest` | - | - | - | - | Failed |
| 19 | `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2:latest` | - | - | - | - | Failed |
| 20 | `qwen2.5:3b` | - | - | - | - | Failed |
| 21 | `qwen3:14b` | - | - | - | - | Failed |
| 22 | `qwen3:8b` | - | - | - | - | Failed |
| 23 | `SetneufPT/Qwen3.6-27B-MTP_Q3_32K_16GB-GPU:latest` | - | - | - | - | Failed |
| 24 | `SetneufPT/Qwen3.6-27B.MTP_Q3_32K_16GB-GPU:latest` | - | - | - | - | Failed |
| 25 | `VladimirGav/Qwen3.6-27B-16GB-VRAM-Uncensored:latest` | - | - | - | - | Failed |

**15 out of 25** models produced a valid SVG. The 10 that failed either returned an error or did not include a valid `<svg>...</svg>` block in their response.

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most mathematically accurate golden spiral**: look for models whose SVG shows a smooth expanding spiral curve
- **You want chamber divisions for biological accuracy**: check the per-model analysis for internal chamber walls
- **You want the most visually rich shell**: pick models labeled "Very high" complexity in the table above
- **You want smooth Bezier curve rendering**: look for models that used `<path>` with curve commands
- **You want a small SVG for web embedding**: pick models with "Compact" verdict
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"

Now read on for the full per-model breakdown and judge for yourself.

## 1. `cieloforge/qwen2.5-14B-instruct-spec:latest`

**SVG size:** 1355 characters  
**Complexity:** Compact  
**Shape elements:** 5  
**Distinct colors:** 2  
**Raw response:** 2121 characters  
**Generation time:** 142.8s

![cieloforge/qwen2.5-14B-instruct-spec:latest nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/cieloforge_qwen2.5-14B-instruct-spec_latest.svg)

### Analysis

This SVG contains approximately **5 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
    <!-- Outline of the nautilus shell -->
    <path fill="#dce2e9"
          d="M
             150,140 
             c-30,-30, -60,-20, -60,40 
             c0,20,20,30, 40,50 
             c20,20, 20,20,40,40
             C180,140, 90,270, 90,270 
              M
             20,10
           z" />
    <!-- Details around the shell's opening -->
    <path fill="#f6eeea"
          d="M
            20,15
            c-5,4, -7,3, -8,-2
            C17,9, 27,17, 37,29
            c10,12, 14,16, 20,26
            C36,50, 100,120, 110,150
           Z"/>
    <!-- Details around the shell's closing -->
    <path fill="#f6eeea"
          d="M
            94,80
            c-7,-1, -8,-3, -9,-2
            C77,76, 87,74, 97,75
            C107,76, 117,83, 127,92
            C137,101, 100,160, 90,150
           Z"/>
    <!-- Details on the right side of the shell -->
    <path fill="#f6eeea"
          d="M
            140,180
            c-3,-2, -4,-2, -5,0
            C169,174, 171,164, 183,143
            C195,123, 165,90, 130,50
           Z"/>
    <!-- Details on the left side of the shell -->
    <path fill="#f6eeea"
          d="M
            70,90
            c-4,-2, -6,-3, -8,-5
            C53,89, 48,91, 40,93
            C28,102, 74,134, 90,90
           Z"/>
</svg>
```

</details>

## 2. `deepseek-r1:1.5b`

**SVG size:** 1849 characters  
**Complexity:** Compact  
**Shape elements:** 4  
**Distinct colors:** 0  
**Raw response:** 2708 characters  
**Generation time:** 146.0s

![deepseek-r1:1.5b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/deepseek-r1_1.5b.svg)

### Analysis

This SVG contains approximately **4 shape elements** and uses **0 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" id="nautilus" width="100%" height="auto">
  <g transform="translate(-50px 100px)" :label="Nautilus Shell">
    <!-- Main outer path (radius 0-150) -->
    <path d="M90,170.5286L40 10C58.5714 130 80 130 107.7273 130C142.2726 130 175 120 170 10L90 -168.755T-15.3505 0a2.7731 2.7731 0 0 1C-40.2723 90 91.8578 94.2107 91.8578 98L61.5 95.3313c-.4142.5408-1.4586 0 -1.3476-.5208C75 90 79.8658 94.668 86.1248 94.668 61.3125 93.3004 93.8088 93.8088 104.477L90 170.5286c-1.5527-.3166-3.9017-.7088-4.4586-1.4572C137.3787 120.9009 141.2469 130 145.2478 130C151.0321 120.9009 153.321 110 156.8791 170a3.4719 3.4719 0 0 0c-3.4719-3.4719 0 0 -.1143-.6332L20.4045 -90C86.1078 86.67 90 90 90 90zM10 72.1763h55.8237c-1.5625 0 3.0312-4.0625 3.3175-7.2422 3.7378-3.2422a.9025 2.2948 0 1 -.1023-.694L-18 -18z"/>
    </g>
    <!-- Inner chamber at radius ~30-100 -->
    <path d="M150,87.8668l-14.0711 14.0711a2.7721 2.7721 0 0 1Mc179.309 -51.8668m30 16.0846l-17.0711-17.0711a2.7721 2.7721 0 0 1M89.5631 4.4957l-20.2064 20.2064a2.6405 2.6405 0 0 1Mc59.5135 76.316m23.1383-14.0711l-17.0711-17.0711a2.7721 2.7721 0 0 1M103.3135 52.9568l-9.6263-9.6263a1.6333 0 0 0 1Mc45.401 -27m20.0868 0l9.3423-9.3423a1.0757 1.0757 0 0 1M27.9108 4.1122l-2.7605 2.7605a1.7759 1.7759 0 0 1Mc33.4906 18.5705m11.6667 42.7108l-14.3045-14.3045a2.9466 2.9466 0 0 1M121.2051 -11.2962L90 -55zM147.8663 90l-14.0711-14.0711a2.7721 2.7721 0 0 1Mc195.5359 44.4109m30-16.0846l-17.0711-17.0711a2.7721 2.7721 0 0 1M89.9108-54.8475L150 -103zM118.6667-90l-20.2064-20.2064a2.6405 2.6405 0 0 1M46.5151-25.4569z"/>
    <!-- Text on a white background -->
    <path d="M300,100l-8 -8L120,78zM200,220l-12 12L94.2738,172.151c-.6335.0445-.8804-.6565-2.7445-.2385a17.2189 17.2189 0 0 1(Cl)(Cl)Cl)"
    <!-- Center line -- />
    <path d="M150 50 M150 130z"/>
    [/g]
</svg>
```

</details>

## 3. `deepseek-r1:7b`

**SVG size:** 207 characters  
**Complexity:** Compact  
**Shape elements:** 0  
**Distinct colors:** 0  
**Raw response:** 2699 characters  
**Generation time:** 116.8s

![deepseek-r1:7b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/deepseek-r1_7b.svg)

### Analysis

This SVG contains approximately **0 shape elements** and uses **0 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="100%" height="auto">
     <g transform="translate(200,200)" id="nautilus">
       <!-- Shell elements will go here -->
     </g>
   </svg>
```

</details>

## 4. `gemma3:1b-it-qat`

**SVG size:** 544 characters  
**Complexity:** Compact  
**Shape elements:** 3  
**Distinct colors:** 4  
**Raw response:** 3147 characters  
**Generation time:** 142.9s

![gemma3:1b-it-qat nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/gemma3_1b-it-qat.svg)

### Analysis

This SVG contains approximately **3 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 150" width="100%" height="auto">
  <!-- Base of the shell -->
  <path d="M 80,120 L 110,120 Z" fill="#E9DABE" stroke="#756AB9" stroke-width="5"/>
  <!-- Spiral/Shell Shape - Two layers-->
  <path d="M 45,120 C 50,80 30,110 70,120 L 50,110 C 60,180 100,130 70,120 Z" fill="#F9A3A4" stroke="#C35E5F" stroke-width="5"/>
  <!-- Upper portion of the spiral - slightly rounded -->
  <path d="M 65,180 C 70,180 80,130 65,130 L 65,130 Z" fill="#F9A3A4" stroke="#C35E5F" stroke-width="5"/>
</svg>
```

</details>

## 5. `gemma3:4b`

**SVG size:** 685 characters  
**Complexity:** Compact  
**Shape elements:** 3  
**Distinct colors:** 3  
**Raw response:** 3892 characters  
**Generation time:** 176.3s

![gemma3:4b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/gemma3_4b.svg)

### Analysis

This SVG contains approximately **3 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Body of the Nautilus Shell -->
  <path d="M100,150 C60,150 10,100 10,80 C10,70 60,45 95,45 C130,45 170,70 170,80 C170,100 130,125 95,125 C60,125 10,100 10,80 C10,70 60,45 100,45 L100,150 Z"
        fill="#E6DDCB" stroke="#A9937C" stroke-width="2"/>
  <!-- Inner Shell (Smallest chamber) -->
  <path d="M85,90 C85,75 65,75 65,90 C65,105 85,105 85,90 Z"
        fill="#D3C2A5" stroke="#A9937C" stroke-width="1"/>
  <!-- Outer Shell (Larger chamber) - Visible only partially -->
  <path d="M100,125 C60,125 100,150 100,125 Z"
        fill="#D3C2A5" stroke="#A9937C" stroke-width="1"/>
</svg>
```

</details>

## 6. `gemma4:12b`

**SVG size:** 1131 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 4  
**Raw response:** 2393 characters  
**Generation time:** 134.6s

![gemma4:12b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/gemma4_12b.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <defs>
    <linearGradient id="shellGradient" x1="0%" y1="100%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#d4a373;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#fefae1;stop-opacity:1" />
    </linearGradient>
  </defs>
  <!-- The main outer shell -->
  <path d="M40,160 C30,150 20,130 20,110 C20,90 35,75 55,70 C80,65 100,70 110,85 C120,100 120,115 110,130 C100,145 80,155 60,155" fill="none" stroke="url(#shellGradient)" stroke-width="12" stroke-linecap="round" />
  <!-- Chamber Dividers -->
  <path d="M55,70 C75,68 90,73 95,85" fill="none" stroke="#c4a17d" stroke-width="2" />
  <path d="M110,85 C115,100 115,115 110,125" fill="none" stroke="#c4a17d" stroke-width="2" />
  <path d="M95,85 C98,100 98,110 95,113" fill="none" stroke="#c4a17d" stroke-width="2" />
  <!-- Interior Detail/Shading -->
  <path d="M40,160 C45,155 55,150 60,155" fill="none" stroke="#bc8a5f" stroke-width="2" />
  <ellipse cx="80" cy="100" rx="30" ry="40" fill="none" stroke="#e9edc9" stroke-width="1" opacity="0.5"/>
</svg>
```

</details>

## 7. `gemma4:26b-a4b-it-qat`

**SVG size:** 2258 characters  
**Complexity:** Compact  
**Shape elements:** 7  
**Distinct colors:** 9  
**Raw response:** 3331 characters  
**Generation time:** 284.2s

![gemma4:26b-a4b-it-qat nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/gemma4_26b-a4b-it-qat.svg)

### Analysis

This SVG contains approximately **7 shape elements** and uses **9 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background for presentation (optional) -->
  <rect width="200" height="200" fill="#f0f0f0" rx="10" />
  <defs>
    <!-- Gradient for the shell body to give it depth and a pearly look -->
    <radialGradient id="shellGradient" cx="40%" cy="40%" r="60%">
      <stop offset="0%" stop-color="#fffaf0" /> <!-- Floral White (center/light) -->
      <stop offset="50%" stop-color="#f5deb3" /> <!-- Wheat (midtone) -->
      <stop offset="100%" stop-color="#d2b48c" /> <!-- Tan (shadow/edges) -->
    </radialGradient>
    <!-- Gradient for the chamber lines -->
    <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#8b4513" opacity="0.6" />
      <stop offset="100%" stop-color="#a0522d" opacity="0.2" />
    </linearGradient>
  </defs>
  <!-- The Main Shell Body (The Silhouette) -->
  <path d="M 100,100 
             C 110,80 140,60 160,90 
             C 180,120 170,160 130,175 
             C 100,185 60,160 50,110 
             C 45,60 90,30 130,20 
             C 160,15 180,50 170,80 
             Q 160,100 130,115
             Q 140,90 120,75
             Q 110,100 80,110
             C 60,120 60,150 90,160
             C 110,170 140,150 145,120
             Q 145,80 115,70
             Q 100,80 100,100 Z" 
        fill="url(#shellGradient)" 
        stroke="#a0522d" 
        stroke-width="1.5" />
  <!-- Chamber Lines (The Spiral Structure) -->
  <g fill="none" stroke="url(#lineGradient)" stroke-width="1.5" stroke-linecap="round">
    <!-- Outer chamber line -->
    <path d="M 140,120 C 160,130 160,160 110,170 C 80,175 70,140 90,120 C 110,100 130,80 135,75" />
    <!-- Middle chamber line -->
    <path d="M 125,115 C 140,125 140,150 110,155 C 90,160 85,140 100,120 C 110,110 120,100 125,95" />
    <!-- Inner chamber line -->
    <path d="M 115,110 C 125,115 125,130 110,135 C 100,140 95,130 105,120" />
    <!-- The core spiral point -->
    <path d="M 110,120 Q 115,115 112,110" stroke="#8b4513" opacity="0.5" />
  </g>
  <!-- Subtle Highlights to give organic feel -->
  <ellipse cx="135" cy="135" rx="10" ry="15" fill="white" opacity="0.2" transform="rotate(20, 135, 135)" />
</svg>
```

</details>

## 8. `lfm2.5:latest`

**SVG size:** 1074 characters  
**Complexity:** Compact  
**Shape elements:** 5  
**Distinct colors:** 10  
**Raw response:** 2165 characters  
**Generation time:** 424.8s

![lfm2.5:latest nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/lfm2.5_latest.svg)

### Analysis

This SVG contains approximately **5 shape elements** and uses **10 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Shell layers -->
  <!-- Outer layer with white fill -->
  <polygon points="70,20 85,8 90,30 Q80,50 95,25 L100,5 Q100,45 K60,65 L55,40 M40,55 Q70,35 K75,25 L82,10 Q82,45 K25,80 M65,50 Q70,38 K55,54 L42,35 Z" fill="#ffffff" stroke="#000" stroke-width="0.5"/>
  <!-- Inner layer for depth -->
  <ellipse cx="30" cy="60" rx="8" ry="4" fill="#e6f7ff" stroke="#4a90e2" stroke-width="0.3"/>
  <ellipse cx="60" cy="20" rx="10" ry="6" fill="#f4d1a5" stroke="#9aaed4" stroke-width="0.2"/>
  <ellipse cx="110" sy="70" rx="12" ry="8" fill="#dfbff3" stroke="#3498db" stroke-width="0.2"/>
  <!-- Gradient for shell appearance -->
  <defs>
    <radialGradient id="shellGrad">
      <stop offset="0%" gradientUnits="ObjectPercentages" stop-color="#e6f7ff"/>
      <stop offset="100%" gradientUnits="ObjectPercentages" stop-color="#add8e6"/>
    </radialGradient>
  </defs>
  <!-- Add shell outline -->
  <path d="M50,20 Q30,5 10,5 M70,20 Q110,5 90,15" stroke="#333" stroke-width="0.4"/>
</svg>
```

</details>

## 9. `llama3.1:8b`

**SVG size:** 675 characters  
**Complexity:** Compact  
**Shape elements:** 2  
**Distinct colors:** 0  
**Raw response:** 2336 characters  
**Generation time:** 202.2s

![llama3.1:8b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/llama3.1_8b.svg)

### Analysis

This SVG contains approximately **2 shape elements** and uses **0 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50" width="100%" height="auto">
    <!-- Path for the outer curve (calculated) -->
    <path id="outer-shell" d="
        M25,5 
        A20,20 0 1,0 45,25 
        A20,20 0 1,0 25,45 
        Z"
        fill-rule="evenodd"/>
    <!-- Path for the inner curve -->
    <path id="inner-shell" d="
        M10.75,12.5 
        A10,10 0 0,1 40,17.5 
        A10,10 0 0,1 10.75,27.5 
        Z"
        fill-rule="evenodd"/>
    <!-- Add color to outer-shell -->
    <style>
      #outer-shell {
          fill: #87CEEB; /* a blue-ish colour */
      }
      #inner-shell {
          fill: white;
      }
    </style>
</svg>
```

</details>

## 10. `qwen2.5-coder-14b:latest`

**SVG size:** 708 characters  
**Complexity:** Compact  
**Shape elements:** 2  
**Distinct colors:** 3  
**Raw response:** 1474 characters  
**Generation time:** 241.3s

![qwen2.5-coder-14b:latest nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/qwen2.5-coder-14b_latest.svg)

### Analysis

This SVG contains approximately **2 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background -->
  <rect x="0" y="0" width="400" height="400" fill="#f7f7f7"/>
  <!-- Nautilus Shell Path -->
  <path d="M 200, 200 
             c 30, 0 58.561, 19.969 70.394, 49.543 
             c 13.177, 29.842 -24.651, 43.586 -29.139, 66.35 
             c -5.049, 26.882 -22.199, 44.337 -48.206, 49.187 
             c -26.107, 4.849 -49.473, -15.385 -53.08, -41.316 
             c -3.94, -30.426 24.995, -56.069 53.081, -41.316 
             c 14.402, 4.675 29.407, -11.526 29.139, -29.953 
             c -0.355, -28.408 -38.219, -49.304 -70.394, -49.543 
             Z" fill="#ffcc00" stroke="#000" stroke-width="1"/>
</svg>
```

</details>

## 11. `qwen2.5-coder:7b`

**SVG size:** 568 characters  
**Complexity:** Compact  
**Shape elements:** 2  
**Distinct colors:** 0  
**Raw response:** 1701 characters  
**Generation time:** 230.9s

![qwen2.5-coder:7b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/qwen2.5-coder_7b.svg)

### Analysis

This SVG contains approximately **2 shape elements** and uses **0 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
    <!-- Define color and stroke -->
    <style>
        .nautilus-shell {
            fill: #ffffff;
            stroke: #000000;
            stroke-width: 2px;
        }
    </style>
    <!-- Path data for the nautilus shell -->
    <path class="nautilus-shell" d="M100,35 Q89,70 60,85 T30,30 T60, -25 89,-70 Q100, -130 121, -85 T150,35 Z"/>
    <!-- Additional path for the top part -->
    <path class="nautilus-shell" d="M50,25 q-20,75 50,75 h60 q70,-70 50,-150 z"/>
</svg>
```

</details>

## 12. `qwen2.5:7b`

**SVG size:** 331 characters  
**Complexity:** Compact  
**Shape elements:** 1  
**Distinct colors:** 2  
**Raw response:** 1552 characters  
**Generation time:** 169.6s

![qwen2.5:7b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/qwen2.5_7b.svg)

### Analysis

This SVG contains approximately **1 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="100%" height="auto">
    <title>Nautilus Shell</title>
    <path d="M 56,80 Q 147.723,192 238,283 C 242,286 245,289 247,292 L 373,322 L 324,295 Q 302,274 164,192 Q 54.5,113.5 -28,80 H 56 V 80 Z"
          fill="#FFD700" stroke="#000000" stroke-width="3"/>
</svg>
```

</details>

## 13. `qwen3.5:4b`

**SVG size:** 1875 characters  
**Complexity:** Compact  
**Shape elements:** 5  
**Distinct colors:** 5  
**Raw response:** 2169 characters  
**Generation time:** 143.0s

![qwen3.5:4b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/qwen3.5_4b.svg)

### Analysis

This SVG contains approximately **5 shape elements** and uses **5 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<pattern>` elements for repeating textures (e.g., pineapple skin or trunk scars)
- Includes gradient fills for richer visual depth
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="-20 -160 84 390" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
    <!-- Background -->
    <rect x="-50" y="-170" width="120" height="380" fill="#ffffff"/>
    <!-- Gradient Definitions for Shell -->
    <defs>
        <linearGradient id="nautilusGrad" x1="49%" y1="0%" x2="66%" y2="50%">
            <stop offset="0%" style="stop-color:#f3dfd8;stop-opacity:1" /> 
            <stop offset="70%" style="stop-color:#e6cbb8;stop-opacity:1" /> 
            <stop offset="90%" style="stop-color:#d4b5a2;stop-opacity:1" />
        </linearGradient>
        <!-- Pattern for Chamber Lines -->
        <pattern id="chamberLines" patternUnits="userSpaceOnUse" width="30" height="30">
            <circle cx="-8" cy="5.67e-32px" r=".1" fill="#d4b5a2"/>
        </pattern>
        <!-- Shadow Gradient for Depth -->
       <radialGradient id="shadowGradCenter" x1="9%" y1="40%" x2="8%" y2="-5%">> 
            <stop offset="0%" style="stop-color:rgba(0,0,0,.3)" /> 
            <stop offset="60%" stop-opacity=".5"/>
        </radialGradient>
    </defs>
    <!-- Main Shell Group -->
    <g fill-rule="evenodd" opacity=".9">
        <!-- Outer Spiral Curve (Logarithmic Approximation) -->
        <path d="M 18,20 C 45,60 30,75 12.5,55 
                 Q -10,65 -35,40 L -30,-30 Q -65,-35 -95,-35
                 S -80,-50 -65,-20 C -60,-30 -75,-50 -90,-60 
                 T 10,-80 q 45,-40 45,14" fill="url(#nautilusGrad)" stroke="#cbbbae"/>
        <!-- Inner Structure / Septa -->
        <g id="chambers">
            <!-- Vertical/Spiral ribs approximation using rotated ellipses/lines -->
            <path d="M 20,35 L 
                     -16,-8 C 
                        4-70 -9,-140 64,15" fill="#fff"/>
     </g>
        <!-- Internal Chamber Divider Details (Ridges) -->
        <circle cx="10.2%" cy="-35px" r=".9%" />
    </g>
</svg>
```

</details>

## 14. `qwen3.5:9b`

**SVG size:** 1867 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 7  
**Raw response:** 1863 characters  
**Generation time:** 156.3s

![qwen3.5:9b nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/qwen3.5_9b.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-15 -20 70 68" preserveAspectRatio="xMidYMax meet" width="100%" height="auto">
  <defs>
    <!-- Gradient for shell material -->
    <linearGradient id="shellGrad" x1="0%" y1="100%" x2="0%" y2="0%">
      <stop offset="0%" style="stop-color:#8B4513;stop-opacity:1" /> <!-- SaddleBrown -->
      <stop offset="66.67%" style="stop-color:#D2A9F5;stop-opacity:1"/> <!-- Lavender -> Purple-ish for contrast hint, or use #E0CDAE (wheat) -->
    </linearGradient>
  </defs>
  <!-- The nautilus uses a logarithmic-style spiral. We draw arcs from center outward. 
       Note: For simplicity we use a series of paths that grow in size by fixed ratio each turn. -->
  <g id="nautilus-shell">
    <!-- Whorl segments drawn sequentially from inside to outside to create overlap effect -->
    <!-- Inner whorl (center) - smallest, tightest curve -->
    <path d="M-5,-10 A34.69 28.73 0 0 1 5.09,-8 l0,2 L-10,-19" 
          fill="#8B5D3B" stroke="#6F4E37" opacity="0.9"/>
    <!-- Second whorl - slightly larger and starting rotation -->
    <path d="M-10,-19 A5,2 0 0 1 10,-11 l12,.5 L2,-28" 
          fill="#A67C52" opacity=".85"/>
    <!-- Third whorl - medium size -->
    <path d="M-34.69,28.73 A20 30 0 1 1 -55,-1 l6,-1 L-49,.5 A34.69 28.73 0 1 1 -55.59,-1" 
          fill="#BFAE8D" opacity=".8"/>
    <!-- Outer Whorl (fully developed final coil) -->
    <path d="M-2,0 A45 60 0 0 1 37,32 l-3.5,5 L-9,-15 a37 20 1 0 -1 -8" 
          fill="#D4C5B0" opacity=".7"/>
    <!-- Central aperture/embryo area -->
     <ellipse cx="-6" cy="10" rx="4" ry="2.5" transform="rotate(-35)" fill="#E0CCAF" />
  </g>
      <!-- Subtle shadows and highlights to give volume (pseudo-3D effect) --> 
    <path d="M -8, -9 Q -17, -4,-6,5 A24.69 28.73 0 0 1 -4.09 ,-6 l-.6 , L -10,-8" fill="#6D4C41"/> <!-- shadow hint on one curve -->
</svg>
```

</details>

## 15. `qwen3.5:latest`

**SVG size:** 1047 characters  
**Complexity:** Compact  
**Shape elements:** 2  
**Distinct colors:** 4  
**Raw response:** 3793 characters  
**Generation time:** 440.7s

![qwen3.5:latest nautilus shell SVG](/assets/img/diagrams/ollama-svg-nautilus/qwen3.5_latest.svg)

### Analysis

This SVG contains approximately **2 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-48 9 -156 37" width="100%" height="auto">
  <defs>
    <!-- Pearl/Iridescent Gradient for body -->
    <linearGradient id="shellNacre" x1="%%" y1="%" x2="%%" y2="%">
      <stop offset="0%" style="stop-color:#fffefc"/> 
      <stop offset="45%" style="stop-color:#d3eaf6"/> 
      <stop offset="80%" style="stop-color:#aeeff7"/> <!-- Blueish pearl sheen -->
    </linearGradient>
    <!-- Shadow for depth on the edge --> 
    <radialGradient id="shellShadow" cx="50%" cy="-20%">
       <stop offset="60%" stop-color="#bfbfc9"/>
       <stop offset="100%" stop-color="#e3f5ff"/>
    </radialGradient>
  </defs>
  <!-- The main outer silhouette and inner structure -->
  <path fill-rule="evenodd" clip-rule="evenodd" />
    -- Main shell body outline (Side Profile Approximation) ---
    M 209,13 
     Q -48,65 -15,-67   Outer Curve Base to Apex Area -> Lip Curve
      <!-- Inner Chamber Detail Lines -->
       ...
    <path fill="url(#shellNacre)" stroke="#dcebf1" stroke-width="2.0"/>
  </svg>
```

</details>

## Failed Models

The following 10 models did not produce a valid SVG (either returned an error or the response did not contain a parseable `<svg>...</svg>` block):

- `gemma4:31b-it-qat` -- TimeoutError: timed out
- `gemma4:latest` -- TimeoutError: timed out
- `jeffgreen311/Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged:latest` -- TimeoutError: timed out
- `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2:latest` -- no SVG block in response
- `qwen2.5:3b` -- no SVG block in response
- `qwen3:14b` -- TimeoutError: timed out
- `qwen3:8b` -- TimeoutError: timed out
- `SetneufPT/Qwen3.6-27B-MTP_Q3_32K_16GB-GPU:latest` -- TimeoutError: timed out
- `SetneufPT/Qwen3.6-27B.MTP_Q3_32K_16GB-GPU:latest` -- TimeoutError: timed out
- `VladimirGav/Qwen3.6-27B-16GB-VRAM-Uncensored:latest` -- TimeoutError: timed out

## Conclusion

We asked 25 Ollama local models to draw **a nautilus shell** -- a subject deeply connected to Fibonacci mathematics. The results reveal each model's natural instinct for mathematical patterns, organic curves, and natural color palettes.

**Key takeaways:**

- **Mathematical intuition varies widely**: some models attempted Fibonacci spiral arrangements naturally, while others defaulted to simple grids or flat shapes. The models that attempted spirals demonstrate a deeper understanding of natural geometry.
- **Detail and file size trade-off**: models that produced the richest scenes (Very high complexity) also generated the largest SVG files. For web embedding, "Balanced" or "Detailed" models may be more practical.
- **Color palettes differ dramatically**: some models used sophisticated gradients and 15+ distinct colors, while others used as few as 4 flat colors. More colors generally means a more lifelike result.
- **Code structure quality varies**: the best models used `<defs>`, `<use>`, and `<pattern>` elements to efficiently generate repeating structures (seeds, scales, leaflets). This is a strong signal of model capability for practical SVG work.
- **No single model is best at everything**: the "right" model depends on whether you prioritize mathematical accuracy, visual beauty, file size, or code quality.

**Try it yourself**: run the benchmark script with your own prompt and see how the models perform on your specific use case. The full code is available in the [ollama-svg-benchmark repository](https://github.com/py2ai/ollama-svg-benchmark).

---

*Which model do you think drew the best nautilus shell? Let us know in the comments below!*
