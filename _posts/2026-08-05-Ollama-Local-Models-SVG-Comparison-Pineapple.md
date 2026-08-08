---
layout: post
title: "Which Ollama Local Model is Best? Pineapple SVG Comparison (14 Models)"
description: "Compare 14 Ollama local models on a pineapple SVG prompt. Find the best LLM for Fibonacci and nature SVG scenes. You decide the winner."
date: 2026-08-05
header-img: "img/post-bg.jpg"
permalink: /Ollama-Local-Models-SVG-Comparison-Pineapple/
tags:
  - AI
  - Ollama
  - SVG
  - LLM
  - Comparison
  - Benchmark
  - Best LLM
  - SVG generation
  - Pineapple
  - Fibonacci
  - Nature
  - Mathematics
  - Fruit
  - Tropical
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama local model comparison, pineapple SVG, AI pineapple drawing, LLM SVG benchmark, AI Fibonacci SVG, Fibonacci spiral SVG, mathematical art SVG, golden ratio SVG, phyllotaxis SVG, pineapple skin pattern, AI fruit drawing, AI nature art, complex SVG scene, botanical illustration"
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/local-deep-research/local-deep-research-architecture.svg
---

# Which Ollama Local Model is Best? Pineapple SVG Comparison (14 Models)

After testing models on sunflowers and mechanical scenes, we turned to another Fibonacci marvel of nature: **the pineapple**. A pineapple's skin is covered in hexagonal scales that form three sets of spirals -- each following Fibonacci numbers (typically 8, 13, and 21). This prompt tests whether models can render that textured skin pattern, the crown of spiky leaves on top, the golden-yellow color gradient of ripe flesh, and the overall oval shape that makes a pineapple instantly recognizable.

The prompt was: `Make an svg image of a pineapple`

This is the twelfth in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/), [duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/), [baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/), [octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/), [FIFA World Cup 2026](/Ollama-Cloud-Models-SVG-Comparison-Fifa-Worldcup-2026/), [elephant on a skateboard](/Ollama-Cloud-Models-SVG-Comparison-Elephant-Skateboard/), [flying helicopter](/Ollama-Cloud-Models-SVG-Comparison-Flying-Helicopter/), [sunflower with seeds](/Ollama-Local-Models-SVG-Comparison-Sunflower/), [pinecone](/Ollama-Local-Models-SVG-Comparison-Pinecone/), [nautilus shell](/Ollama-Local-Models-SVG-Comparison-Nautilus-Shell/), [dates palm tree](/Ollama-Local-Models-SVG-Comparison-Dates-Palm/).

**Why a pineapple?** The pineapple prompt is a **texture and pattern stress test** because it combines: (1) **Fibonacci diagonal spirals** -- the hexagonal scales on a pineapple skin form three intersecting spiral sets (8, 13, 21), and a model with mathematical intuition should attempt a diamond or hexagonal grid pattern rather than a flat oval, (2) **Crown leaves** -- a pineapple has a distinctive crown of stiff green spiky leaves on top; omitting it makes the drawing unrecognizable, (3) **Color gradient** -- a ripe pineapple transitions from green near the top to golden-yellow at the bottom, (4) **Oval body shape** -- the body is an elongated oval, not a circle or rectangle, (5) **Texture density** -- a real pineapple has dozens of scale segments; the model must balance detail with file size, and (6) **Symmetry** -- the body is bilaterally symmetric, while the crown leaves radiate from the top.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Pineapple SVGs

The pineapple prompt rewards different things than previous prompts. Here are the criteria to use:

- **Skin texture pattern**: Does the model render the diamond/hexagonal scale pattern on the skin? Or is it just a flat oval? The Fibonacci spiral pattern is the ultimate test.
- **Crown of leaves**: Does it include the green spiky crown on top? How many leaves? Are they individually drawn or just suggested?
- **Color gradient**: Does it use a green-to-yellow gradient on the body? Or flat color?
- **Body shape**: Is the body an oval (taller than wide)? Or a circle/rectangle?
- **Scale detail**: Does each scale have a subtle 3D shading or highlight? Or are they flat diamonds?
- **Overall recognizability**: If you showed this SVG to someone, would they immediately say "pineapple"?
- **SVG code quality**: Does it use `<pattern>`, `<defs>`, or `<use>` to efficiently generate the repeating scale texture?

## How It Works

The script discovers all locally installed models via the Ollama API (`/api/tags`), then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with minimal post-processing (adding `width="100%" height="auto"` for responsive embedding and fixing XML errors so the SVG renders in browsers).

Unlike our cloud model benchmarks, these models run entirely on the local GPU -- no cloud subscription or network round-trip required. This means generation times reflect local hardware performance, and model sizes range from 1B to 31B parameters. Embedding, vision, and OCR models are automatically skipped.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `cieloforge/qwen2.5-14B-instruct-spec:latest` | 394 | 3 | 2 | Compact | Compact |
| 2 | `deepseek-r1:1.5b` | 359 | 2 | 4 | Compact | Compact |
| 3 | `deepseek-r1:7b` | 1047 | 8 | 3 | Compact | Compact |
| 4 | `gemma3:1b-it-qat` | 761 | 5 | 2 | Compact | Compact |
| 5 | `gemma3:4b` | 729 | 7 | 5 | Compact | Compact |
| 6 | `gemma4:12b` | 1803 | 12 | 6 | Compact | Compact |
| 7 | `gemma4:26b-a4b-it-qat` | 1486 | 21 | 6 | Compact | Compact |
| 8 | `lfm2.5:latest` | 618 | 6 | 4 | Compact | Compact |
| 9 | `llama3.1:8b` | 577 | 1 | 1 | Compact | Compact |
| 10 | `qwen2.5-coder-14b:latest` | 517 | 2 | 3 | Compact | Compact |
| 11 | `qwen2.5-coder:7b` | 519 | 5 | 5 | Compact | Compact |
| 12 | `qwen2.5:7b` | 574 | 6 | 5 | Compact | Compact |
| 13 | `qwen3.5:9b` | 2266 | 17 | 6 | Medium | Balanced |
| 14 | `qwen3.5:latest` | 3975 | 4 | 4 | Compact | Compact |
| 15 | `gemma4:31b-it-qat` | - | - | - | - | Failed |
| 16 | `gemma4:latest` | - | - | - | - | Failed |
| 17 | `jeffgreen311/Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged:latest` | - | - | - | - | Failed |
| 18 | `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2:latest` | - | - | - | - | Failed |
| 19 | `qwen2.5:3b` | - | - | - | - | Failed |
| 20 | `qwen3.5:4b` | - | - | - | - | Failed |
| 21 | `qwen3:14b` | - | - | - | - | Failed |
| 22 | `qwen3:8b` | - | - | - | - | Failed |
| 23 | `SetneufPT/Qwen3.6-27B-MTP_Q3_32K_16GB-GPU:latest` | - | - | - | - | Failed |
| 24 | `SetneufPT/Qwen3.6-27B.MTP_Q3_32K_16GB-GPU:latest` | - | - | - | - | Failed |
| 25 | `VladimirGav/Qwen3.6-27B-16GB-VRAM-Uncensored:latest` | - | - | - | - | Failed |

**14 out of 25** models produced a valid SVG. The 11 that failed either returned an error or did not include a valid `<svg>...</svg>` block in their response.

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most accurate Fibonacci skin pattern**: look for models whose SVG shows a visible diamond/hexagonal grid on the body
- **You want the most visually rich pineapple**: pick models labeled "Very high" complexity in the table above
- **You want a recognizable pineapple with a crown**: check the per-model analysis for crown leaf presence
- **You want gradient coloring (green to gold)**: look for models that used `<linearGradient>` on the body
- **You want efficient code for the repeating pattern**: look for models that used `<pattern>` or `<use>` elements
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"

Now read on for the full per-model breakdown and judge for yourself.

## 1. `cieloforge/qwen2.5-14B-instruct-spec:latest`

**SVG size:** 394 characters  
**Complexity:** Compact  
**Shape elements:** 3  
**Distinct colors:** 2  
**Raw response:** 639 characters  
**Generation time:** 39.8s

![cieloforge/qwen2.5-14B-instruct-spec:latest pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/cieloforge_qwen2.5-14B-instruct-spec_latest.svg)

### Analysis

This SVG contains approximately **3 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating motifs like seeds, scales, or leaflets

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80" width="100%" height="auto">
  <defs>
    <rect id="base" x="10" y="30" width="60" height="30" fill="#f2f1d6"/>
    <polygon id="spikes" points="50,70 60,60 80,100 30,100 10,60"/>
  </defs>
  <circle cx="50" cy="40" r="20" fill="#ff4e5f"/>
  <use href="#base" style="fill:#cfaabb"/>
  <use href="#spikes" style="fill:#a9573b"/>
</svg>
```

</details>

## 2. `deepseek-r1:1.5b`

**SVG size:** 359 characters  
**Complexity:** Compact  
**Shape elements:** 2  
**Distinct colors:** 4  
**Raw response:** 631 characters  
**Generation time:** 44.2s

![deepseek-r1:1.5b pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/deepseek-r1_1.5b.svg)

### Analysis

This SVG contains approximately **2 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Uses standard SVG primitives (rect, circle, path, polygon) to compose the scene

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <circle cx="400" cy="300" r="250" fill="hot" />
  <loop allow="repeat" times="16">
    <rect cx="400" cy="300" rx="250" fill="none-attrnone"
          fill="h*((7.98*cos($angle)) - 451.6)"
          "$angle":
            "none-attrname" />
  </loop>
  <close-path stroke="#ff0000"/>
</svg>
```

</details>

## 3. `deepseek-r1:7b`

**SVG size:** 1047 characters  
**Complexity:** Compact  
**Shape elements:** 8  
**Distinct colors:** 3  
**Raw response:** 1723 characters  
**Generation time:** 24.3s

![deepseek-r1:7b pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/deepseek-r1_7b.svg)

### Analysis

This SVG contains approximately **8 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Pineapple illustration created using text and basic shapes -->
  <!-- Core circle -->
  <circle cx="290" cy="30" r="75" fill="black" stroke="white" stroke-width="1" opacity="0.7"/>
  <!-- Fruitlet circles (using text paths) -->
  <!-- First layer of fruitlets -->
  <circle cx="-60" cy="-40" r="20" fill="none" stroke="white" stroke-width="1"/>
  <circle cx="-85" cy="-95" r="20" fill="none" stroke="white" stroke-width="1" rotation="135"/>
  <!-- Second layer of fruitlets (larger and spaced out) -->
  <circle cx="-46" cy="-71" r="26" fill="none" stroke="white" stroke-width="1"/>
  <circle cx="-99" cy="-80" r="26" fill="none" stroke="white" stroke-width="1"/>
  <circle cx="-31" cy="-95" r="26" fill="none" stroke="white" stroke-width="1"/>
  <circle cx="-74" cy="-56" r="26" fill="none" stroke="white" stroke-width="1"/>
  <!-- Tree trunk (simplified) -->
  <line x1="300" y1="100" x2="300" y2="400" penstrokeweight="2" stroke="brown4" stroke-linecap="round"/>
</svg>
```

</details>

## 4. `gemma3:1b-it-qat`

**SVG size:** 761 characters  
**Complexity:** Compact  
**Shape elements:** 5  
**Distinct colors:** 2  
**Raw response:** 3348 characters  
**Generation time:** 43.9s

![gemma3:1b-it-qat pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/gemma3_1b-it-qat.svg)

### Analysis

This SVG contains approximately **5 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
    <!-- This is the main pineapple shape -->
    <polynomial id="pineapple" cx="15" cy="18" r="2" />
      <!--  Create some leaves.  This makes it a bit more realistic. -->
     <line x1="20" y1="15" x2="30" y2="18" stroke="black" stroke-width="2"*pixelSizeScale">
        <line x1="20" y1="18" x2="30" y2="20">
     </line>
    <line x1="30" y1="15" x2="40" y2="18" stroke="black" stroke-width="2"*pixelSizeScale">
        <line x1="30" y1="18" x2="40" y2="20">
     </line>
    <!--  Add a crown or "spiky" top. This is really simple in SVG -->
    <polygon id="pineappleCrown" points="5.79,6.66 38.2,7.1 38.2,6.66 43.94,7.1 57.92,6.73 45.4,7.5 48.3,4.3 57.93,7.1" fill="white" />
  </svg>
```

</details>

## 5. `gemma3:4b`

**SVG size:** 729 characters  
**Complexity:** Compact  
**Shape elements:** 7  
**Distinct colors:** 5  
**Raw response:** 3861 characters  
**Generation time:** 75.2s

![gemma3:4b pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/gemma3_4b.svg)

### Analysis

This SVG contains approximately **7 shape elements** and uses **5 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 250" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Base of the Pineapple (Spike) -->
  <path d="M100,20 L100,220 C100,240 150,240 150,220 L150,20" fill="#FDD7E4"/>
  <!-- Top of the Pineapple (Cone) -->
  <polygon points="100,80 130,60 170,60" fill="#FEE9D7"/>
  <!-- Outer Leaves -->
  <path d="M150,20 L150,40 L170,30 L190,40 L190,20 Z" fill="#CDBEF8"/> 
  <path d="M150,50 L150,70 L170,60 L190,70 L190,50 Z" fill="#CDBEF8"/>
  <!-- Smaller Leaves -->
  <path d="M130,40 L130,55 L140,45 L140,55 Z" fill="#BEA786"/>
  <path d="M130,60 L130,75 L140,65 L140,75 Z" fill="#BEA786"/>
  <!-- Stem -->
   <line x1="100" y1="220" x2="100" y2="240" stroke="#E6C39B" stroke-width="5"/>
</svg>
```

</details>

## 6. `gemma4:12b`

**SVG size:** 1803 characters  
**Complexity:** Compact  
**Shape elements:** 12  
**Distinct colors:** 6  
**Raw response:** 2184 characters  
**Generation time:** 133.9s

![gemma4:12b pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/gemma4_12b.svg)

### Analysis

This SVG contains approximately **12 shape elements** and uses **6 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Shadow/Depth -->
  <ellipse cx="100" cy="180" rx="65" ry="90" fill="#D4AC0D" />
  <!-- Main Body -->
  <ellipse cx="100" cy="180" rx="65" ry="90" fill="#FFD700" stroke="#B8860B" stroke-width="2"/>
  <!-- Texture/Patterns (The "scales") -->
  <g fill="#B8860B" opacity="0.4">
    <path d="M80 130 Q 90 140 100 130 M100 150 Q 110 160 120 150 M80 170 Q 90 180 100 170" fill="none" stroke="#B8860B" stroke-width="4" stroke-linecap="round"/>
    <path d="M100 130 Q 90 120 80 130 M120 150 Q 110 140 100 150 M100 170 Q 110 160 120 170" fill="none" stroke="#B8860B" stroke-width="4" stroke-linecap="round"/>
    <path d="M90 110 Q 100 100 110 110 M70 130 Q 80 120 90 130 M110 130 Q 120 120 130 130" fill="none" stroke="#B8860B" stroke-width="4" stroke-linecap="round"/>
    <path d="M70 150 Q 80 140 90 150 M110 150 Q 120 140 130 150" fill="none" stroke="#B8860B" stroke-width="4" stroke-linecap="round"/>
    <path d="M100 210 Q 90 220 80 210 M100 230 Q 110 240 120 230" fill="none" stroke="#B8860B" stroke-width="4" stroke-linecap="round"/>
  </g>
  <!-- Leaves (Crown) -->
  <g fill="#2E8B57">
    <!-- Left Leaf -->
    <path d="M100 100 C 60 70, 40 130, 100 110" fill="#2E8B57" stroke="#1E5D3D" stroke-width="2"/>
    <!-- Center Leaf -->
    <path d="M100 90 C 90 40, 110 40, 100 90" fill="#3CB371" stroke="#1E5D3D" stroke-width="2"/>
    <!-- Right Leaf -->
    <path d="M100 100 C 140 70, 160 130, 100 110" fill="#2E8B57" stroke="#1E5D3D" stroke-width="2"/>
    <!-- Backer Leaves -->
    <path d="M90 110 Q 70 60, 40 90" fill="#2E8B57" stroke="#1E5D3D" stroke-width="2" opacity="0.8"/>
    <path d="M110 110 Q 130 60, 160 90" fill="#2E8B57" stroke="#1E5D3D" stroke-width="2" opacity="0.8"/>
  </g>
</svg>
```

</details>

## 7. `gemma4:26b-a4b-it-qat`

**SVG size:** 1486 characters  
**Complexity:** Compact  
**Shape elements:** 21  
**Distinct colors:** 6  
**Raw response:** 2561 characters  
**Generation time:** 258.6s

![gemma4:26b-a4b-it-qat pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/gemma4_26b-a4b-it-qat.svg)

### Analysis

This SVG contains approximately **21 shape elements** and uses **6 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 280" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background (Optional) -->
  <rect width="200" height="280" fill="#fdfefe" />
  <!-- The Leaves (Crown) -->
  <g id="leaves" fill="#2D5A27">
    <!-- Back leaves -->
    <path d="M100 90 Q70 30 60 80 Z" />
    <path d="M100 90 Q130 30 140 80 Z" />
    <!-- Side leaves -->
    <path d="M100 100 Q50 60 70 110 Z" />
    <path d="M100 100 Q150 60 130 110 Z" />
    <!-- Front leaves -->
    <path d="M100 110 Q80 70 90 120 Z" />
    <path d="M100 110 Q120 70 110 120 Z" />
  </g>
  <!-- The Body of the Pineapple -->
  <ellipse cx="100" cy="180" rx="55" ry="75" fill="#FFD700" stroke="#DAA520" stroke-width="3" />
  <!-- Texture (The "scales") -->
  <g id="texture" fill="#E4A900">
    <!-- Row 1 -->
    <circle cx="80" cy="140" r="6" />
    <circle cx="100" cy="135" r="7" />
    <circle cx="120" cy="140" r="6" />
    <!-- Row 2 -->
    <circle cx="70" cy="170" r="8" />
    <circle cx="90" cy="165" r="7" />
    <circle cx="110" cy="165" r="8" />
    <circle cx="130" cy="170" r="7" />
    <!-- Row 3 -->
    <circle cx="80" cy="200" r="7" />
    <circle cx="100" cy="210" r="8" />
    <circle cx="120" cy="200" r="7" />
    <!-- Row 4 -->
    <circle cx="90" cy="235" r="6" />
    <circle cx="110" cy="235" r="6" />
  </g>
  <!-- Highlights to add dimension -->
  <path d="M75 160 Q80 180 85 210" stroke="white" stroke-width="4" fill="none" opacity="0.3" stroke-linecap="round" />
</svg>
```

</details>

## 8. `lfm2.5:latest`

**SVG size:** 618 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 4  
**Raw response:** 1189 characters  
**Generation time:** 419.9s

![lfm2.5:latest pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/lfm2.5_latest.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 150 100" width="100%" height="auto">
  <!-- Pineapple Base -->
  <rect x="20" y="60" width="130" height="65" fill="#8B4513"/>
  <!-- Scales Texture (approximate texture representation) -->
  <circle cx="60" cy="80" r="5" fill="#A0522D"/>
  <circle cx="100" cy="30" r="6" fill="#8B4513"/>
  <rect x="90" y="50" width="10" height="10" fill="#FFD700" stroke="#8B4513" stroke-width="1"/>
  <circle cx="120" cy="0" r="7" fill="#8B4513" />
  <!-- Thorny Crown -->
  <path d="M60,80 Q70,75 55,95 Q100,90 80,45 M90,50 Q85,30 65,10" stroke="#555" stroke-width="1.5"/>
</svg>
```

</details>

## 9. `llama3.1:8b`

**SVG size:** 577 characters  
**Complexity:** Compact  
**Shape elements:** 1  
**Distinct colors:** 1  
**Raw response:** 1405 characters  
**Generation time:** 153.5s

![llama3.1:8b pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/llama3.1_8b.svg)

### Analysis

This SVG contains approximately **1 shape elements** and uses **1 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-20 -70 240 240" width="100%" height="auto">
    <g fill="#F7DC6F">
        <path d="M -10,100 c 20,-70,-90,-50,-40,-80 C 5,65 105,90 85,150 55,70 45,-15 5,0 Z M 75,-30 a 95,55 0,1 0,-200 135 95,-55 0,1 0,230 -170 35,40 7,-28 50,20 40,10 C 60,-35 25,45 65,100 70,5 55,90 35,-75 Z M -15,-110 a 40,80 0,0 1,130,30 a 170,-90 0,0 1,0 20 a 50,50 0,0 1,0 120 l -10,60 c -85,30,-160,-100 -140,-150 Z M 55,-55 s 10,15 -40,75 70,-65 C -10,-90 45,-145 25,-135 s 0,20 -20,20 M 65,-175 q 2,3 50,-35 c 6,10 60,-30 12,40 Z" />
    </g>
</svg>
```

</details>

## 10. `qwen2.5-coder-14b:latest`

**SVG size:** 517 characters  
**Complexity:** Compact  
**Shape elements:** 2  
**Distinct colors:** 3  
**Raw response:** 688 characters  
**Generation time:** 186.5s

![qwen2.5-coder-14b:latest pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/qwen2.5-coder-14b_latest.svg)

### Analysis

This SVG contains approximately **2 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-10 -10 250 450" width="100%" height="auto">
  <defs>
    <radialGradient id="pineappleGrad" fx="0.3" fy="0.7">
        <stop offset="0%" stop-color="#ffcc66"></stop>
        <stop offset="100%" stop-color="#ff9900"></stop>
    </radialGradient>
  </defs>
  <path d="M120,50 Q100,30 80,50 T40,50 Q60,70 80,50 Q120,30 140,50" fill="url(#pineappleGrad)" />
  <path d="M120,150 Q100,130 80,150 T40,150 Q60,170 80,150 Q120,130 140,150" fill="url(#pineappleGrad)" />
</svg>
```

</details>

## 11. `qwen2.5-coder:7b`

**SVG size:** 519 characters  
**Complexity:** Compact  
**Shape elements:** 5  
**Distinct colors:** 5  
**Raw response:** 2512 characters  
**Generation time:** 183.4s

![qwen2.5-coder:7b pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/qwen2.5-coder_7b.svg)

### Analysis

This SVG contains approximately **5 shape elements** and uses **5 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
<!-- Pineapple stem -->
<path d="M 20,35 H 80 Q 120 45 120 85 L 60 90 Z" fill="#8b4513"/>
<!-- Outline of the pineapple -->
<ellipse cx="85" cy="115" rx="35" ry="55" stroke="black" fill="#ff6347"/>
<!-- Pineapple slices -->
<path d="M 100,85 L 90,110 Q 100,140 110,110 Z" fill="#f7dc6f"/>
<path d="M 55,140 H 35 Q 25 110 45,140 Z" fill="#f7dc6f"/>
<!-- Pineapple core -->
<ellipse cx="80" cy="130" rx="10" ry="5" stroke="black" fill="#be9e26"/>
</svg>
```

</details>

## 12. `qwen2.5:7b`

**SVG size:** 574 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 5  
**Raw response:** 1317 characters  
**Generation time:** 148.6s

![qwen2.5:7b pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/qwen2.5_7b.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **5 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 150 150" width="100%" height="auto">
    <!-- Background -->
    <rect width="150" height="150" fill="#FBBA4C" />
    <!-- Crown (Top part) -->
    <circle cx="75" cy="55" r="29.8" fill="#FDBD6A" />
    <path d="M 52 35 Q 70 10, 103 10 T 103 35" fill="#FDBD6A" />
    <!-- Pineapple Body -->
    <ellipse cx="75" cy="85" rx="55" ry="45" fill="#EE8D2E" />
    <!-- Leaf Details -->
    <path d="M 40 130 Q 90 125, 90 60 T 140 60" fill="#5B7551" />
    <path d="M 100 85 Q 150 95, 120 130 T 95 130" fill="#7A8D3C" />
</svg>
```

</details>

## 13. `qwen3.5:9b`

**SVG size:** 2266 characters  
**Complexity:** Medium  
**Shape elements:** 17  
**Distinct colors:** 6  
**Raw response:** 2306 characters  
**Generation time:** 131.3s

![qwen3.5:9b pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/qwen3.5_9b.svg)

### Analysis

This SVG contains approximately **17 shape elements** and uses **6 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Pineapple Body (bottom) -->
  <ellipse cx="100" cy="260" rx="85" ry="70" fill="#FFD700"/>
  <!-- Top part of body -->
  <path d="M 35,290 L 185,290 
           Q 200,240 185,200 T 160,100 T 100,-50 T 40,200"
        fill="#FFD700"/>
  <!-- Diagonal texture pattern - left side -->
  <path d="M 35,290 L 85,310 
           Q 90,260 85,240 T 40,290" fill="#FFEEDD"/>
  <!-- Diagonal texture pattern - right side -->
  <path d="M 75,290 L 185,290 
           Q 185,360 135,340 T 75,315" fill="#FFEEDD"/>
  <!-- Vertical texture lines - left -->
  <path d="M 53,290 L 58,255 M 53,285 L 63,220 M 48,290 L 57,215" fill="#FFEEDD"/>
  <!-- Vertical texture lines - right -->
  <path d="M 112,290 L 110,-20 Q 163,-10 166,40 T 108,170 M 115,290 L 110,-5" stroke="#FFEEDD" fill="none"/>
  <!-- Pineapple Outline - Brown Border -->
  <path d="M 35,290 C 70,260 
         Q 40,290 85,290 L 185,290 Q 145,260 180,290
         A 85,85,0,0,-135,180" 
        fill="#FFD700"/>
    <!-- Green Leaf Crown -->
  <g>
    <!-- Main stem connection -->
    <path d="M 100,.2 L 65,-40 Q 69,-45 73,-35 T 85,1" 
          fill="#FFD700"/>
    <!-- Leaves - left side (curved) -->
    <ellipse cx="65" cy="-30" rx="25" ry="15" transform="rotate(-45)" fill="#4CAF50"/>
    <ellipse cx="55" cy="-40" rx="28" ry="17" transform="rotate(20)" fill="#6BFF9E"/>
    <!-- Leaves - right side (curved) -->
    <ellipse cx="135" cy="-30" rx="25" ry="15" transform="rotate(-45 180 75)" fill="#4CAF50"/>
    <ellipse cx="90" cy="-60" rx="23" ry="16" transform="rotate(15, -10, 20)" fill="#6BFF9E"/>
      <!-- Outer leaves -->
    <path d="M 85,-4 Q 70,-40 60,-45 Q 45,-35 30,-30 
              L 30,-15 T 40,25" stroke="#2D5A27" fill="none"/>
    <path d="M 115,-4 Q 130,-40 140,-45 Q 155,-35 170,-30 
              L 170,-15 T 160,28" stroke="#2D5A27" fill="none"/>
    <!-- Top inner leaves -->
    <ellipse cx="90" cy="-45" rx="24" ry="16" transform="rotate(30)" fill="#6BFF9E"/>
    <ellipse cx="110" cy="-48" rx="24" ry="17" transform="rotate(-20, 10 -15)" fill="#4CAF50"/>
  </g>
   <!-- Final Outline -->  
<rect width="200" height="300" fill="none" stroke="#8B6914" 
      stroke-width="3.5"/>
</svg>
```

</details>

## 14. `qwen3.5:latest`

**SVG size:** 3975 characters  
**Complexity:** Compact  
**Shape elements:** 4  
**Distinct colors:** 4  
**Raw response:** 5173 characters  
**Generation time:** 363.2s

![qwen3.5:latest pineapple SVG](/assets/img/diagrams/ollama-svg-pineapple/qwen3.5_latest.svg)

### Analysis

This SVG contains approximately **4 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<pattern>` elements for repeating textures (e.g., pineapple skin or trunk scars)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewBox="-5 -6 486 296" width="100%" height="auto">
  <defs>
    <!-- Pineapple Body Gradient (Yellow to Orange-Brown) -->
    <linearGradient id="pineSkin" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#fcecb8;stop-opacity:1" />
      <stop offset="45%" style="stop-color:fefae0;stop-opacity:1" />
      <stop offset="95%" style="stop-color:#d6c723;stop-opacity:1" />
    </linearGradient>
    <!-- Leaf Gradient (Dark to Light Green) -->
    <radialGradient id="leafGrad">
      <stop offset="0%" stop-color="#4cbf8a"/>
      <stop offset="5%" stop-color="#6cc392"/>
      <stop offset="20%" style="stop-color:#1e7b4c; stop-opacity:0.9" />
    </radialGradient>
   <!-- Shadow Gradient for skin texture -->
     <linearGradient id="highlightGrad">
        <offset/>
        <stop class="off"/>  -- using a standard filter approach would be heavy, simple circles used instead
      </defs>
       <!-- Simple Brown spot pattern definition (Reused often) -->
      <pattern id="pineTexture" x="0" y="0" width="38.156741892" height="52.528443583" 
                 patternUnits="userSpaceOnUse">
        <!-- This creates the rough scale look by adding small darker spots -->
      </pattern>
  </defs>
  <g id="pineapple-fruit-body" transform="translate(0, -1)">
    <!-- Main Body Shape: An elongated egg/ellipse shape with slight rotation for organic feel -->
    <path d="M239.5,-7 
             C348,.6 421,107.5 409,154.5
             L-9,160 
            Curve to left side -C200,-20" /> 
    <!-- Let's do this with an actual clean path for a better shape -->
     <path d="M387 -7 C 395, -16 , 450, 17.5 , 315 .8
               Curve to bottom: L 260, 155 
               Up curve other side back up" />
        <!-- RE-WRITING THE SHAPES FOR HIGH QUALITY VISUAL -->        
    </g> 
      <!-- I will use standard bezier paths for high quality visual result in code below.
     <svg version="1.1"> 
  <rect x="0" y="389" width="275" height="46"/>   <!- Placeholder to keep file short -->
</html> 
```
**Correction**: The attempt above failed to render a proper SVG due to syntax errors in the text block below it. Here is the **correct, functional code**. You can copy and paste this into an `.svg` file or view directly here:
</think>
Here is a standalone SVG image of a pineapple created with clean paths and gradients for lighting effects.
You can save the following code as `pineapple.svg`, open it in your web browser, or just display the raw text below to see it immediately.
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-5 -6 481 293">
  <!-- DEFINITIONS -->
  <defs>
    <!-- Main body color (Yellowish-Gold) -->
    <linearGradient id="pinegrad" x1="0%" y1="0%" x2="74.95%" y2="-68%">      
      <stop offset="0%" stop-color="#FEFCB8"/>
      <stop offset="33%"/> 
       <stop offset="33%"/>   
    </linearGradient>
    <!-- Leaf gradient (Dark green to light green) -->
    <radialGradient id="topGrad">       
     <stop offset="0%" stop-color="#4CBF8A" />      
      <stop offset="5%" stop-color="#6CC392"/>    
      <offset/>     
       <offset class="" style ="offset:; "  /> 
        <!-- To ensure we can use a simple single-path shape for the leaves -->
          <!-- A radial gradient is good enough here, simpler than path defs -->
    </radialGradient>
     <!-- Pattern spots (The skin texture of pineapple) 
                We simulate this with small circles added manually later to avoid code bloat in Defs 
      <pattern id="pineSpots" x="-50%" y="-7.3462918" width="15." height=".">        
           </pattern>    
    -->
  </defs>
 <!-- PINEAPPLE CROWN (Leaves) -->
<g transform = "translate(24 -3)"> 
   <g stroke="#0F5D1C" fill="" id ="pineapple-topleaves">      
      <!-- Leaf Group: This group rotates to spread leaves horizontally around a center point. 6 Leaves.-->       
      <path class="leaf-shape"/>     
    </svg>
```

</details>

## Failed Models

The following 11 models did not produce a valid SVG (either returned an error or the response did not contain a parseable `<svg>...</svg>` block):

- `gemma4:31b-it-qat` -- TimeoutError: timed out
- `gemma4:latest` -- TimeoutError: timed out
- `jeffgreen311/Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged:latest` -- TimeoutError: timed out
- `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2:latest` -- no SVG block in response
- `qwen2.5:3b` -- no SVG block in response
- `qwen3.5:4b` -- no SVG block in response
- `qwen3:14b` -- TimeoutError: timed out
- `qwen3:8b` -- TimeoutError: timed out
- `SetneufPT/Qwen3.6-27B-MTP_Q3_32K_16GB-GPU:latest` -- TimeoutError: timed out
- `SetneufPT/Qwen3.6-27B.MTP_Q3_32K_16GB-GPU:latest` -- TimeoutError: timed out
- `VladimirGav/Qwen3.6-27B-16GB-VRAM-Uncensored:latest` -- TimeoutError: timed out

## Conclusion

We asked 25 Ollama local models to draw **a pineapple** -- a subject deeply connected to Fibonacci mathematics. The results reveal each model's natural instinct for mathematical patterns, organic curves, and natural color palettes.

**Key takeaways:**

- **Mathematical intuition varies widely**: some models attempted Fibonacci spiral arrangements naturally, while others defaulted to simple grids or flat shapes. The models that attempted spirals demonstrate a deeper understanding of natural geometry.
- **Detail and file size trade-off**: models that produced the richest scenes (Very high complexity) also generated the largest SVG files. For web embedding, "Balanced" or "Detailed" models may be more practical.
- **Color palettes differ dramatically**: some models used sophisticated gradients and 15+ distinct colors, while others used as few as 4 flat colors. More colors generally means a more lifelike result.
- **Code structure quality varies**: the best models used `<defs>`, `<use>`, and `<pattern>` elements to efficiently generate repeating structures (seeds, scales, leaflets). This is a strong signal of model capability for practical SVG work.
- **No single model is best at everything**: the "right" model depends on whether you prioritize mathematical accuracy, visual beauty, file size, or code quality.

**Try it yourself**: run the benchmark script with your own prompt and see how the models perform on your specific use case. The full code is available in the [ollama-svg-benchmark repository](https://github.com/py2ai/ollama-svg-benchmark).

---

*Which model do you think drew the best pineapple? Let us know in the comments below!*
