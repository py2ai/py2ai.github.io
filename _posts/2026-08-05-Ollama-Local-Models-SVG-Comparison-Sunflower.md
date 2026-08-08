---
layout: post
title: "Which Ollama Local Model is Best? Sunflower SVG Comparison (16 Models)"
description: "Compare 16 Ollama local models on a sunflower SVG prompt. Find the best LLM for Fibonacci and nature SVG scenes. You decide the winner."
date: 2026-08-05
header-img: "img/post-bg.jpg"
permalink: /Ollama-Local-Models-SVG-Comparison-Sunflower/
tags:
  - AI
  - Ollama
  - SVG
  - LLM
  - Comparison
  - Benchmark
  - Best LLM
  - SVG generation
  - Sunflower
  - Fibonacci
  - Nature
  - Mathematics
  - Botanical
  - Phyllotaxis
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama local model comparison, sunflower SVG, AI sunflower drawing, LLM SVG benchmark, AI Fibonacci SVG, Fibonacci spiral SVG, mathematical art SVG, AI nature drawing, golden ratio SVG, phyllotaxis SVG, sunflower seeds SVG, AI art comparison, complex SVG scene, nature illustration, botanical SVG"
featured-img: ai-coding-frameworks/ai-coding-frameworks
image: /assets/img/diagrams/local-deep-research/local-deep-research-architecture.svg
---

# Which Ollama Local Model is Best? Sunflower SVG Comparison (16 Models)

After testing LLMs on ducks, vehicles, marine life, chess, sports, and machines, we wanted to know: **can today's top models capture the mathematical beauty of nature?** This time we asked our Ollama local models to draw **a sunflower with seeds** -- a prompt that tests whether models understand Fibonacci spiral patterns (real sunflower seeds are arranged in intersecting clockwise and counter-clockwise spirals that follow Fibonacci numbers like 21, 34, 55, and 89), radial symmetry, color gradients (dark center seeds, bright yellow petals, green stem and leaves), and organic curves that blend mathematics with aesthetics.

The prompt was: `Make an svg image of a sunflower with seeds`

This is the eleventh in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/), [duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/), [baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/), [octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/), [FIFA World Cup 2026](/Ollama-Cloud-Models-SVG-Comparison-Fifa-Worldcup-2026/), [elephant on a skateboard](/Ollama-Cloud-Models-SVG-Comparison-Elephant-Skateboard/), [flying helicopter](/Ollama-Cloud-Models-SVG-Comparison-Flying-Helicopter/), [pineapple](/Ollama-Local-Models-SVG-Comparison-Pineapple/), [pinecone](/Ollama-Local-Models-SVG-Comparison-Pinecone/), [nautilus shell](/Ollama-Local-Models-SVG-Comparison-Nautilus-Shell/), [dates palm tree](/Ollama-Local-Models-SVG-Comparison-Dates-Palm/).

**Why a sunflower?** This prompt is a **mathematical stress test** for SVG generation because it combines: (1) **Fibonacci spirals** -- real sunflower seeds follow Fibonacci sequences, and a model with strong mathematical intuition should attempt spiral arrangements rather than a simple grid of dots, (2) **Two-scale structure** -- a sunflower has a dark seed-filled center and bright outer petals, requiring the model to distinguish two visual zones, (3) **Radial symmetry** -- petals and seeds radiate from the center, testing the model's understanding of rotational geometry and `<transform>` usage, (4) **Color theory** -- yellow petals, brown or black seeds, green stem and leaves -- the model must choose a coherent palette, (5) **Organic vs. geometric** -- the model must balance mathematical precision (spirals) with organic softness (petal shapes), and (6) **Scale and density** -- a real sunflower has hundreds of seeds; the model must decide how many to draw and how to arrange them without making the SVG file enormous.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Sunflower SVGs

The sunflower prompt rewards different things than previous prompts. Here are the criteria to use:

- **Fibonacci spiral pattern**: Does the model arrange seeds in spiral patterns (clockwise and counter-clockwise)? Or does it just place them in a grid, concentric circles, or random dots? True Fibonacci spirals are the gold standard.
- **Petal detail**: Does it draw distinct petals around the center? How many petals? Real sunflowers have 34, 55, or 89 petals (Fibonacci numbers).
- **Two-zone structure**: Is there a clear dark seed center and bright petal ring? Or is it just a flat yellow circle?
- **Color depth**: Does it use gradients for the seeds (dark brown to black) and petals (yellow to orange)? Or flat fills only?
- **Radial symmetry**: Is the flower radially symmetric? Are the petals evenly spaced around the center?
- **Stem and leaves**: Does it include a green stem and leaves for context? Or just the flower head floating in a void?
- **SVG code quality**: Does it use `<defs>`, `<use>`, and transforms to efficiently generate repeating seeds and petals? Good code structure is a sign of model competence.

## How It Works

The script discovers all locally installed models via the Ollama API (`/api/tags`), then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with minimal post-processing (adding `width="100%" height="auto"` for responsive embedding and fixing XML errors so the SVG renders in browsers).

Unlike our cloud model benchmarks, these models run entirely on the local GPU -- no cloud subscription or network round-trip required. This means generation times reflect local hardware performance, and model sizes range from 1B to 31B parameters. Embedding, vision, and OCR models are automatically skipped.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `cieloforge/qwen2.5-14B-instruct-spec:latest` | 3050 | 25 | 2 | Medium | Balanced |
| 2 | `deepseek-r1:1.5b` | 2974 | 20 | 1 | Medium | Balanced |
| 3 | `deepseek-r1:7b` | 872 | 6 | 2 | Compact | Compact |
| 4 | `gemma3:1b-it-qat` | 544 | 4 | 2 | Compact | Compact |
| 5 | `gemma3:4b` | 1015 | 13 | 3 | Compact | Compact |
| 6 | `gemma4:12b` | 2840 | 39 | 7 | Medium | Balanced |
| 7 | `gemma4:26b-a4b-it-qat` | 2884 | 29 | 7 | Medium | Balanced |
| 8 | `lfm2.5:latest` | 78 | 0 | 0 | Compact | Compact |
| 9 | `llama3.1:8b` | 1153 | 14 | 8 | Compact | Compact |
| 10 | `qwen2.5-coder-14b:latest` | 443 | 6 | 2 | Compact | Compact |
| 11 | `qwen2.5-coder:7b` | 594 | 4 | 3 | Compact | Compact |
| 12 | `qwen2.5:3b` | 1810 | 21 | 7 | Compact | Compact |
| 13 | `qwen2.5:7b` | 1938 | 8 | 5 | Compact | Compact |
| 14 | `qwen3.5:4b` | 2661 | 19 | 4 | Medium | Balanced |
| 15 | `qwen3.5:9b` | 1533 | 1 | 7 | Compact | Compact |
| 16 | `qwen3.5:latest` | 991 | 7 | 5 | Compact | Compact |
| 17 | `gemma4:31b-it-qat` | - | - | - | - | Failed |
| 18 | `gemma4:latest` | - | - | - | - | Failed |
| 19 | `jeffgreen311/Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged:latest` | - | - | - | - | Failed |
| 20 | `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2:latest` | - | - | - | - | Failed |
| 21 | `qwen3:14b` | - | - | - | - | Failed |
| 22 | `qwen3:8b` | - | - | - | - | Failed |
| 23 | `SetneufPT/Qwen3.6-27B-MTP_Q3_32K_16GB-GPU:latest` | - | - | - | - | Failed |
| 24 | `SetneufPT/Qwen3.6-27B.MTP_Q3_32K_16GB-GPU:latest` | - | - | - | - | Failed |
| 25 | `VladimirGav/Qwen3.6-27B-16GB-VRAM-Uncensored:latest` | - | - | - | - | Failed |

**16 out of 25** models produced a valid SVG. The 9 that failed either returned an error or did not include a valid `<svg>...</svg>` block in their response.

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most mathematically accurate seed spirals**: look for models whose SVG shows visible spiral patterns in the seed arrangement
- **You want the most visually beautiful sunflower**: pick models labeled "Very high" complexity in the table above
- **You want a small, efficient SVG for web embedding**: pick models with "Compact" verdict
- **You want accurate botanical detail (petals, stem, leaves)**: check the per-model analysis for stem and leaf presence
- **You want gradient-rich coloring**: look for models that used `<linearGradient>` or `<radialGradient>`
- **You want to compare within a model family**: pick `deepseek-v4-pro` vs `deepseek-v4-flash`, or `glm-5.1` vs `glm-5.2`
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"

Now read on for the full per-model breakdown and judge for yourself.

## 1. `cieloforge/qwen2.5-14B-instruct-spec:latest`

**SVG size:** 3050 characters  
**Complexity:** Medium  
**Shape elements:** 25  
**Distinct colors:** 2  
**Raw response:** 3340 characters  
**Generation time:** 148.9s

![cieloforge/qwen2.5-14B-instruct-spec:latest sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/cieloforge_qwen2.5-14B-instruct-spec_latest.svg)

### Analysis

This SVG contains approximately **25 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes animation elements -- the model attempted motion
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 248 350" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <defs>
    <!-- Define styles -->
    <style>
      .seeds {
        fill: #FF7F50;
        stroke: none;
      }
    </style>
  </defs>
  <!-- Sunflower petal paths -->
  <path d="M124,63 L153.928,102.072 A153.93,153.93 0 0,0 183.856,109.472 C205.144,109.472 224,127.808 224,149.32 A128,128 0 0,1 124,167 H124 Z" />
  <path d="M124,63 L94.072,102.072 A153.93,153.93 0 0,0 64,109.472 C45.544,109.472 26.056,127.808 26.056,149.32 A128,128 0 0,1 124,167 H124 Z" />
  <path d="M124,63 L85.28,102.072 A153.93,153.93 0 0,0 55.2,109.472 C36.224,109.472 16.952,127.808 16.952,149.32 A128,128 0 0,1 124,167 H124 Z" />
  <path d="M124,63 L55.2,102.072 A153.93,153.93 0 0,0 29.2,109.472 C10.984,109.472 -5.568,127.808 -5.568,149.32 A128,128 0 0,1 124,167 H124 Z" />
  <path d="M124,63 L26.952,102.072 A153.93,153.93 0 0,0 22.952,109.472 C10.192,109.472 -4.664,127.808 -4.664,149.32 A128,128 0 0,1 124,167 H124 Z" />
  <!-- Petal and petal head -->
  <path d="M152,104 C163.02,105.19 161.76,108.23 161.76,108.23 L175.2,99.35 H150 Z M157,164 C168.38,165.08 176.94,171.82 176.94,171.82 L177.77,172.58 H150 Z" fill="#FFE7BA" />
  <!-- Seed paths -->
  <path class="seeds" d="M123.5,64 C123,91.54 123.5,112 123,112 L123.5 86 C126.76,40 217.16,40 217.16,40 L182.5,59 H174.5" />
  <path class="seeds" d="M125.5,66 C123.5,71 123.5,112 123,112 L123.5 83 C128.76,24 230.69,24 230.69,24 L216.5,33 H179.5" />
  <path class="seeds" d="M127.5,65 C124,89 220.5,89 220.5,89 L225,93 H185" />
  <path class="seeds" d="M123,71 C123.5,94 221.6,94 221.6,94 L226,98 H187" />
  <g fill="#F3E1BC">
    <path class="seeds" d="M100.5,63 C123.5,94 221.6,94 221.6,94 L216.5,98 H179.5 V142 H50 V142 C32.075,25.15 31.384,-1.843 50,-1.843 V146 L37,154 H65" />
    <path class="seeds" d="M109.5,63 C70,71 210,71 210,71 L205,75 H88" />
    <path class="seeds" d="M118.5,63 C84,91 125,91 125,91 L128,95 H98">
    <path class="seeds" d="M127.5,66 C124.5,88 180.9,88 180.9,88 L185,92 H155">
    <path class="seeds" d="M92,70 C83,80 97,80 97,80 L100,84 H46">
      <animate attributeName="x1" from="92" to="10" dur="5s" repeatCount="indefinite" />
    </path>
    <!-- Additional seeds -->
    <g class="seeds">
      <circle cx="37.5" cy="285" r="4"/>
      <circle cx="49.5" cy="301" r="4"/>
      <circle cx="67.5" cy="289" r="4"/>
      <circle cx="85.5" cy="305" r="4"/>
      <circle cx="52.5" cy="297" r="4"/>
    </g>
    <!-- Petals -->
    <path d="M69,167 L48.718,159 A19.743,19.743 0 0,1 44.457,175 C23.112,175 12.6,184.4 11.594,196.06 A17.828,17.828 0 0,1 8,198 H5 Z" />
    <path d="M69,167 L48.718,175 C23.112,175 12.6,184.4 11.594,196.06 A17.828,17.828 0 0,1 8,198 H5 Z" />
    <!-- Petal tail -->
    <path d="M93,167 L52.778,158 A14.784,14.784 0 0,0 49,164 C38.7,164 29.818,170.414 28.8,181.28 A13.35,13.35 0 0,1 21,183 H16 Z" />
    <!-- Pistil -->
    <circle cx="97" cy="192" r="4.8"/>
    <!-- Ancestors line -->
    <line x1="44.457+50" y1="175-150" x2="50" y2="160"/>
  </g>
</svg>
```

</details>

## 2. `deepseek-r1:1.5b`

**SVG size:** 2974 characters  
**Complexity:** Medium  
**Shape elements:** 20  
**Distinct colors:** 1  
**Raw response:** 3860 characters  
**Generation time:** 150.3s

![deepseek-r1:1.5b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/deepseek-r1_1.5b.svg)

### Analysis

This SVG contains approximately **20 shape elements** and uses **1 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 240 180" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <path fill="currentColor" d" />
    <!-- petals -->
    <circle cx="0" cy="75" r="160" stroke="#f00" stroke-width="2" fill="none" />
    <polygon points="4 35, 91, -5 91, -6 148 91, 81 180 50.47 180"/>
    <polygon points="69, 81 121 105 ,18 130" fill="none" />
    <!-- petals -->
    <circle cx="-100" cy="35" r="76" stroke="#f00" stroke-width="2" fill="none" />
    <path d="M-49.44,58.865c-4.23-.37-9.55-3.138-12.47 0-12.47 3.95 12.04 12.47 22.34c3.105 0 4.76.667 7.185,3.042a3.715,3.715 0 01-7.942-2.37c-4.37-4.23-9.1-9.1-12.47-12.47a16,16 0 01-49.683 58.865zm-22.23i-2.915,0zm-20,0c-1.32-.23-3-3.3 -6-7.47a5.907 5.907 0 019.65.798zM-75.48-55.815c7.972.21 10.58 2.365 12.9 6.568a3.407 3.407 0 01-76.04-55.815zM75,75c3.242.23 6.478.947 12.471,3.464a16 16 0 01-49-.648zM17.07-66.125c1.246.005 2.486.218 5.023.48a16.147 16.147 0 01-69.102-9.682zM24.172 26.41c-4.033-1.27-8.5-3.818-13.2-10.95a24.247 24.247 0 01-75-22.02z">
    </path>
    <!-- central circle -->
    <circle cx="0" cy="-75" r="62" stroke="#f00" stroke-width="2" fill="none" />
    <!-- seeds -->
    <circle cx='-48' cy='39' r='12' stroke="#f00" stroke-width="2" fill='none'>
    <path d="M15.119 75.586c-1.457.578-3.926,1.457-7.356,3.926-7.356-3.926 -1.457-3.926-1.457-7.356 0a16.276 16.276 0 01 38.387-75.585zM10.876,25.306c1.906-.473 6.85 -5.898 9.552-12.627a45.381 45.381 0 01 57.263 22.875z" />
    </circle>
    <circle cx='-52' cy='39' r='12'>
    <path d="M-93.835 24c0 .0149-.0757.5958-1.4958.6r6.07 .757a24 24 0 1-50-11.37z" />
    </circle>
    <circle cx='-56' cy='39' />
    <path d="M-56 3'14 7.5'C6 8.25 9.69 10.5 11.25 14a11 11 0 01-27 14c0-.07-5.1-.358-9.9-1'c0 .0149.075 2.786g4.804 4.744a16 16 0 11-24 2.888l-2.475.31zM73.646 22c-.077.039-5.76.087-10.96-.058a28 28 0 1-56 2.192l1.647-1.63Ca5.29-.278 12 2.346 17 9.27C6.07 9.673 9.67 11.57 11.37 14a16 16 0 12-30 14z" />
    </circle>
    <circle cx='-56' cy='39' r='12'>
    <path d="M-58 22c0 .03-.074.101-.146.265a24 24 0 1-58 22zM61.79 23c.038.002.104.125.266.28ax16 16 0 0118-26.937zM31.073 61.705c-.019-.003-2.79-1.9-9.455-3a16 16 0 00-52 3z" />
    </circle>
    <circle cx='-31' cy='53' r='8' />
    <path d="M47.855 73c0 .03-.073.098-1.672.427a32.319 32.319 0 01-79.682 109.909zM52.313 68.95c-.01.002-.059.018-1.499.039a33.389 33.389 0 01 19-3.179zM10.75 47.6c-1.498.882-3.889 1.497-5.686-.577a26.654 26.654 0 00 15.23 -5.67zM25 1.8c1.016.058 2.745.125 5.656.196a25 25 0 11-73.5 25.806z" />
    </circle>
    <circle cx='-23' cy='53' r='12'>
    <path d="M204.403 64c.005.003-.005.014-1.985.639a56 56 0 00-347 217.687zM6.447 132.6c3.366.104 9.07 1.789 22.216.991a36 36 0 00 156-40.053zM69 102.945c-.365.008-2.9.058-11.7 2.837a40 40 0 1-60 24.28z" />
    </circle>
    <circle cx='-1' cy='17.5' r='160'>M-29-9.92c-.035.007-.07 .121-1.57.608a41 41 0 01 75-109zM160 160c0 0 160 60 160-60 160 0 a160 160 0 00-160-60z"/>
    </circle>
</svg>
```

</details>

## 3. `deepseek-r1:7b`

**SVG size:** 872 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 2  
**Raw response:** 1349 characters  
**Generation time:** 102.3s

![deepseek-r1:7b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/deepseek-r1_7b.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!--Background -->
  <rect x="0" y="0" width="600" height="400" fill="#FFF8F5"></rect>
  <!-- Sunflower base (small circle for structure) -->
  <circle cx="300" cy="200" r="15" fill="#FFD7AD"></circle>
  <!-- Petals formed by arcs -->
  <path d="M460,190 L380,180 M300,150 L220,170 M220,170 L320,130 M460,190 L380,200"></path>
  <!-- Seed clusters -->
  <!-- Many seed elements added with positions calculated to create spiral effect -->
  <g transform="translate(300,200)">
    <!-- Randomly placed small circles as seeds -->
    <!-- Each with size .75 radius, position varying around center -->
    <circle cx="-41" cy="-98" r="0.75"></circle>
    <circle cx="-156" cy="-38" r="0.75"></circle>
    <circle cx="-284" cy="-22" r="0.75"></circle>
    <!-- More seed circles added here -->
  </g>
</svg>
```

</details>

## 4. `gemma3:1b-it-qat`

**SVG size:** 544 characters  
**Complexity:** Compact  
**Shape elements:** 4  
**Distinct colors:** 2  
**Raw response:** 2981 characters  
**Generation time:** 127.3s

![gemma3:1b-it-qat sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/gemma3_1b-it-qat.svg)

### Analysis

This SVG contains approximately **4 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <path d="M 30 40 L 60 30 H 80 L 90 C 75 40, 85 30, 80 70 L 50 70 Z" fill="#F5F5DC" />
    <path d="M 30 10 l -0.26 -0.15 l -0.70 -0.25 z" stroke="#795548" stroke-width="2" fill="#795548">  <!-- Center lines -->
    <path d="M 30 30 C 35,45,30,50,25,65 L 40 10 Z" stroke="#795548" stroke-width="2" fill="#795548"> <!-- Lower lines -->
    <path d="M 30 80 C 35,75,30,70,25,60 L 40 80 Z" stroke="#795548" stroke-width="2" fill="#795548"> <!--  upper lines -->
  </path>
</svg>
```

</details>

## 5. `gemma3:4b`

**SVG size:** 1015 characters  
**Complexity:** Compact  
**Shape elements:** 13  
**Distinct colors:** 3  
**Raw response:** 3519 characters  
**Generation time:** 179.7s

![gemma3:4b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/gemma3_4b.svg)

### Analysis

This SVG contains approximately **13 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Flower Center -->
  <circle cx="100" cy="120" r="40" fill="#FFDA63"/>
  <!-- Lower Petals -->
  <polygon points="50,180 150,180 100,-50" fill="#FFD700"/>
  <polygon points="100,180 150,180 100,-50" fill="#FFDA63"/>
  <!-- Upper Petals -->
  <polygon points="20,180 100,180 50, -50" fill="#FFDA63"/>
  <polygon points="100,180 50, 180 100, -50" fill="#FFD700"/>
  <!-- Seed Pod -->
   <ellipse cx="100" cy="250" rx="45" ry="23" fill="#DAA520"/>
  <!-- Seeds -->
    <circle cx="90" cy="240" r="4.5" fill="#FFD700" /> 
    <circle cx="110" cy="240" r="4.5" fill="#FFD700" />
    <circle cx="85" cy="235" r="3.5" fill="#FFD700"/> 
    <circle cx="115" cy="235" r="3.5" fill="#FFD700"/>
    <circle cx="98" cy="245" r="3" fill="#FFD700"/>
    <circle cx="105" cy="245" r="3" fill="#FFD700"/>
    <circle cx="102" cy="260" r="2.8" fill="#FFD700" />
   <!-- More seeds... can add many more to make it look fuller! -->
</svg>
```

</details>

## 6. `gemma4:12b`

**SVG size:** 2840 characters  
**Complexity:** Medium  
**Shape elements:** 39  
**Distinct colors:** 7  
**Raw response:** 3451 characters  
**Generation time:** 170.2s

![gemma4:12b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/gemma4_12b.svg)

### Analysis

This SVG contains approximately **39 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 400 500" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background (Optional) -->
  <rect width="400" height="500" fill="#fdfbf7" />
  <!-- Stem and Leaves -->
  <path d="M200 210 Q 205 350, 200 480" stroke="#4a7c2c" stroke-width="12" fill="none" />
  <path d="M200 320 C 140 300, 130 360, 180 370" fill="#5fa14a" /> <!-- Left leaf -->
  <path d="M200 360 C 260 340, 270 390, 220 400" fill="#5fa14a" /> <!-- Right leaf -->
  <!-- Petals (Back Layer) -->
  <g fill="#ffcc00">
    <ellipse cx="200" cy="150" rx="30" ry="80" transform="rotate(0, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="30" ry="80" transform="rotate(45, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="30" ry="80" transform="rotate(90, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="30" ry="80" transform="rotate(135, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="30" ry="80" transform="rotate(180, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="30" ry="80" transform="rotate(225, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="30" ry="80" transform="rotate(270, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="30" ry="80" transform="rotate(315, 200, 150)"/>
  </g>
  <!-- Petals (Front Layer - staggered) -->
  <g fill="#ffdb30">
    <ellipse cx="200" cy="150" rx="25" ry="75" transform="rotate(22.5, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="25" ry="75" transform="rotate(67.5, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="25" ry="75" transform="rotate(112.5, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="25" ry="75" transform="rotate(157.5, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="25" ry="75" transform="rotate(202.5, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="25" ry="75" transform="rotate(247.5, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="25" ry="75" transform="rotate(292.5, 200, 150)"/>
    <ellipse cx="200" cy="150" rx="25" ry="75" transform="rotate(337.5, 200, 150)"/>
  </g>
  <!-- Sunflower Center (The Seed Head) -->
  <circle cx="200" cy="150" r="60" fill="#5c4033" />
  <!-- Seed Texture Layer -->
  <g fill="#7a5a4a">
    <circle cx="170" cy="130" r="3"/>
    <circle cx="230" cy="125" r="4"/>
    <circle cx="200" cy="150" r="5"/>
    <circle cx="160" cy="155" r="3"/>
    <circle cx="240" cy="155" r="4"/>
    <circle cx="200" cy="185" r="4"/>
    <circle cx="185" cy="175" r="3"/>
    <circle cx="215" cy="170" r="3"/>
    <circle cx="190" cy="135" r="4"/>
    <circle cx="210" cy="140" r="3"/>
    <circle cx="180" cy="160" r="3"/>
    <circle cx="220" cy="165" r="3"/>
    <!-- Adding a few more random seeds to create texture -->
    <circle cx="175" cy="145" r="2"/>
    <circle cx="225" cy="135" r="2"/>
    <circle cx="195" cy="115" r="2"/>
    <circle cx="205" cy="185" r="2"/>
    <circle cx="165" cy="140" r="3"/>
    <circle cx="235" cy="140" r="3"/>
  </g>
</svg>
```

</details>

## 7. `gemma4:26b-a4b-it-qat`

**SVG size:** 2884 characters  
**Complexity:** Medium  
**Shape elements:** 29  
**Distinct colors:** 7  
**Raw response:** 3778 characters  
**Generation time:** 330.7s

![gemma4:26b-a4b-it-qat sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/gemma4_26b-a4b-it-qat.svg)

### Analysis

This SVG contains approximately **29 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background (Optional: can be removed for transparency) -->
  <rect width="200" height="300" fill="#fafafa" rx="15" />
  <!-- Stem -->
  <path d="M100 280 Q95 200 100 120" stroke="#4d7c2c" stroke-width="8" fill="none" stroke-linecap="round" />
  <!-- Leaves -->
  <ellipse cx="80" cy="220" rx="30" ry="12" fill="#6a9e3f" transform="rotate(-30 80 220)" />
  <ellipse cx="125" cy="170" rx="25" ry="10" fill="#6a9e3f" transform="rotate(45 125 170)" />
  <!-- Petals (Grouped for centering) -->
  <g id="petals-group" transform="translate(100, 80)">
    <!-- We create multiple petals by rotating long thin ellipses/paths -->
    <!-- I am manually placing several to ensure a full circle look -->
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(0)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(30)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(60)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(90)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(120)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(150)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(180)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(210)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(240)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(270)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(300)" />
    <path d="M0,-55 Q10,-20 0,0 Q-10,-20 0,-55" fill="#FFD700" transform="rotate(330)" />
  </g>
  <!-- Flower Center (The Disk) -->
  <circle cx="100" cy="80" r="40" fill="#3e2723" stroke="#2d1b1a" stroke-width="2" />
  <!-- Seeds (Pattern of small dots inside the center) -->
  <g id="seeds">
    <!-- This creates a grid of little seed dots -->
    <circle cx="90" cy="70" r="2.5" fill="#5d4037" />
    <circle cx="110" cy="70" r="2.5" fill="#5d4037" />
    <circle cx="100" cy="60" r="2.5" fill="#5d4037" />
    <circle cx="85" cy="80" r="2.5" fill="#5d4037" />
    <circle cx="115" cy="80" r="2.5" fill="#5d4037" />
    <circle cx="100" cy="80" r="2.5" fill="#5d4037" />
    <circle cx="100" cy="95" r="2.5" fill="#5d4037" />
    <circle cx="85" cy="95" r="2.5" fill="#5d4037" />
    <circle cx="115" cy="95" r="2.5" fill="#5d4037" />
    <circle cx="95" cy="100" r="2.5" fill="#5d4037" />
    <circle cx="105" cy="100" r="2.5" fill="#5d4037" />
  </g>
  <!-- Inner glow on the center for texture -->
  <circle cx="100" cy="80" r="35" fill="none" stroke="#5d4037" stroke-width="1" opacity="0.5" />
</svg>
```

</details>

## 8. `lfm2.5:latest`

**SVG size:** 78 characters  
**Complexity:** Compact  
**Shape elements:** 0  
**Distinct colors:** 0  
**Raw response:** 2500 characters  
**Generation time:** 363.3s

![lfm2.5:latest sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/lfm2.5_latest.svg)

### Analysis

This SVG contains approximately **0 shape elements** and uses **0 distinct colors**.

**Notable SVG techniques used:**

- Uses standard SVG primitives (rect, circle, path, polygon) to compose the scene

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" … width="100%" height="auto"> … </svg>
```

</details>

## 9. `llama3.1:8b`

**SVG size:** 1153 characters  
**Complexity:** Compact  
**Shape elements:** 14  
**Distinct colors:** 8  
**Raw response:** 2724 characters  
**Generation time:** 176.7s

![llama3.1:8b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/llama3.1_8b.svg)

### Analysis

This SVG contains approximately **14 shape elements** and uses **8 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 420" width="100%" height="auto">
  <!-- Background Circle -->
  <circle cx="210" cy="210" r="170" fill="#F2C464"/>
  <g transform="translate(70, 190)">
    <!-- Sun Ray Pattern -->
    <path d="M -25, 10 L 35, -15" fill="#FFFF00"/>
    <path d="M -15, -5 L -55, 20" fill="#FFFF00" />
    <path d="M 30, -10 L 60, 10" fill="#FFFF00"/>
    <!-- Center of the Sun -->
    <circle cx="0" cy="0" r="7.9" fill="#F2C464"/>
    <circle cx="5.4" cy="-3" r="7.4" fill="#D6B857"/>
  </g>
  <!-- Sunflower Petals -->
  <path d="M 170, 100 C 230, -30, 190, 300, 150, 20" stroke="#008000" fill="#008000" />
  <path d="M 250, 220 C 210, 420, 320, 40, 370, 260" stroke="#008000" fill="#008000" />
  <path d="M 190, 280 C 230, 10, 140, 130, 360, -80" stroke="#008000" fill="#008000" />
  <path d="M 240, 150 C 160, -120, 300, -170, 370, -120" stroke="#008000" fill="#008000" />
  <!-- Seeds -->
  <circle cx="100" cy="200" r="3.5" fill="#964B00"/>
  <circle cx="110" cy="220" r="3.2" fill="#784E00"/>
  <circle cx="120" cy="240" r="3.0" fill="#7C6A4F"/>
  <circle cx="100" cy="250" r="2.8" fill="#77433F"/>
</svg>
```

</details>

## 10. `qwen2.5-coder-14b:latest`

**SVG size:** 443 characters  
**Complexity:** Compact  
**Shape elements:** 6  
**Distinct colors:** 2  
**Raw response:** 816 characters  
**Generation time:** 201.4s

![qwen2.5-coder-14b:latest sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/qwen2.5-coder-14b_latest.svg)

### Analysis

This SVG contains approximately **6 shape elements** and uses **2 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Draw the sunflower head -->
  <ellipse cx="100" cy="100" rx="80" ry="100" fill="gold" />
  <!-- Draw the seeds -->
  <circle cx="100" cy="100" r="5" fill="green" />
  <circle cx="120" cy="90" r="5" fill="green" />
  <circle cx="110" cy="110" r="5" fill="green" />
  <circle cx="90" cy="120" r="5" fill="green" />
  <circle cx="70" cy="110" r="5" fill="green" />
</svg>
```

</details>

## 11. `qwen2.5-coder:7b`

**SVG size:** 594 characters  
**Complexity:** Compact  
**Shape elements:** 4  
**Distinct colors:** 3  
**Raw response:** 1772 characters  
**Generation time:** 195.6s

![qwen2.5-coder:7b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/qwen2.5-coder_7b.svg)

### Analysis

This SVG contains approximately **4 shape elements** and uses **3 distinct colors**.

**Notable SVG techniques used:**

- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="100%" height="auto">
    <!-- Sunflower Base -->
    <circle cx="32" cy="32" r="30" fill="#FFD95F" />
    <!-- Flower Petals -->
    <path fill="#FFC125" d="
        M32,14 q8,-6 12,-18 t-12,-18 q-4,6 -8,18 z 
        M32,50 q8,6 12,18 t-12,18 q-4,-6 -8,-18 z
    "/>
    <!-- Seeds -->
    <path fill="#E8A700" d="
        M29,29 c-3,3 -3,9 0,12 l -10,10 C19,45 19,41 16,38 z
    "/>
    <path fill="#E8A700" d="
        M35,35 c3,-3 9,-3 12,0 l 10,-10 C51,24 51,28 48,31 z
    "/>
    <!-- Add more seeds as needed -->
</svg>
```

</details>

## 12. `qwen2.5:3b`

**SVG size:** 1810 characters  
**Complexity:** Compact  
**Shape elements:** 21  
**Distinct colors:** 7  
**Raw response:** 2592 characters  
**Generation time:** 186.8s

![qwen2.5:3b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/qwen2.5_3b.svg)

### Analysis

This SVG contains approximately **21 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Basic Structure -->
  <rect style="fill:#FFF;" x="15" y="15" width="180" height="145"></rect>
  <!-- Background Circle for Sunflower Center -->
  <ellipse cx="55" cy="65" rx="25" ry="35" fill="#ccc"/>
  <!-- Stalk Segment 1 -->
  <line x1="70" y1="65" x2="95" y2="53" style="stroke:#000;stroke-width:4"/>
  <!-- Seed Segments (Circles and Ellipses) -->
  <circle cx="83" cy="61" r="5" fill="#FFF"/>
  <ellipse cx="79" cy="61" rx="2.5" ry="7" fill="#FFD4A0"/>
  <!-- Seed Segments (Circles and Ellipses) -->
  <circle cx="83" cy="61" r="5" fill="#FFD4A0"/>
  <ellipse cx="79" cy="62" rx="2.5" ry="3.5" fill="#FF684D"/>
  <!-- Seed Segments (Circles and Ellipses) -->
  <circle cx="83" cy="61" r="5" fill="#FF684D"/>
  <ellipse cx="79" cy="62.5" rx="2.5" ry="2.3" fill="#07BD5B"/>
  <!-- Seed Segments (Circles and Ellipses) -->
  <circle cx="81" cy="60.5" r="4.5" fill="#FFD4A0"/>
  <ellipse cx="81" cy="60.2" rx="2.3" ry="2.3" fill="#9ACD32"/>
  <!-- Seed Segments (Circles and Ellipses) -->
  <circle cx="77" cy="57" r="4.5" fill="#FF684D"/>
  <ellipse cx="78" cy="57.1" rx="2.3" ry="1.9" fill="#A0C429"/>
  <!-- Seed Segments (Lines and Ellipses) -->
  <polygon points="70,58 85,62 83,58"/>
  <ellipse cx="72.5" cy="59" rx="1.5" ry="1" fill="#FFD4A0"/>
  <!-- Seed Segments (Lines and Ellipses) -->
  <polygon points="72.5,60 77,62 78,61.5"/>
  <ellipse cx="73.9" cy="60.5" rx="1.1" ry="0.9" fill="#FF684D"/>
  <!-- Seed Segments (Lines and Ellipses) -->
  <polygon points="82.5,58.5 75,58.9 73,60.5"/>
  <ellipse cx="80" cy="59.5" rx="0.85" ry="1" fill="#FFD4A0"/>
  <!-- Seed Segments (Lines and Ellipses) -->
  <polygon points="71.5,57.3 69,57.2 64.5,57"/>
  <ellipse cx="72.8" cy="57.8" rx="0.95" ry="0.5" fill="#FF684D"/>
</svg>
```

</details>

## 13. `qwen2.5:7b`

**SVG size:** 1938 characters  
**Complexity:** Compact  
**Shape elements:** 8  
**Distinct colors:** 5  
**Raw response:** 2706 characters  
**Generation time:** 185.9s

![qwen2.5:7b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/qwen2.5_7b.svg)

### Analysis

This SVG contains approximately **8 shape elements** and uses **5 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="100%" height="auto">
    <!-- Circle for the center of the sunflower -->
    <circle cx="200" cy="200" r="50" fill="#FFD700"/>
    <!-- Petals (use a series of paths) -->
    <path d="M189.4 164c-36-27-56-62-70-104C93 63 48 20 35 10s29-9 65 16 37-4 74-12l-7.4 18c-29-15-54-31-75-48C78 47 130 26 173 28S216 68 258 83 301 104 338 130s92 99 136 135c7 7-77-7-191-71-98-41-194-93-273-141zm3.2-1C221 150 293 127 296 22s213 240 300 268c-9 10-51-5-68+3z" fill="#FFD700"/>
    <path d="M229.3 188.2s-31-8-47-20C195 146 179 162 173 220l62.7-9-24+10c-11 0-18-9-18-19s9-18 18-18l22-5zm74.6-67.1c-11-13-31-18-46-18-32 0-35+15-40.5 55C39.4 160 71.5 150 113 126 138 101 170-26 213-25s151 100 211 129c9 10 19 45 41+51l12.7-3z" fill="#FFD700"/>
    <path d="M286.1 233.1s-15.8-34.3-16-37.4c5-7 9-2-8+8zm-6-32c-4 2-8 11-8 17 0 29 10 53 28 65C331 273 297 269 288 253c-6-10-18-14-28+5c-11 8-18 23.7-26 37S216 201 200 209v5z" fill="#FFD700"/>
    <!-- Small details and seeds (dots) -->
    <defs>
        <linearGradient id="seedColors">
            <stop offset="0%" stop-color="#FFFFFF"/>
            <stop offset="40%" stop-color="#EEDC82"/>
            <stop offset="100%" stop-color="#9B7F4A"/>
        </linearGradient>
    </defs>
    <circle cx="200" cy="350" r="4" fill="url(#seedColors)" stroke-width="0.5" />
    <!-- More seeds with random locations -->
    <g transform="translate(180 370)">
        <ellipse transform="scale(.6)" rx="2" ry="1.8" fill="url(#seedColors)" />
        <circle cx="4" cy="0" r="2" fill="url(#seedColors)" stroke-width="0"/>
    </g>
    <!-- More seeds in the background -->
    <g transform="translate(35 170)">
        <ellipse rx="19" ry="25" fill="#EEDC82" stroke-width="0.6" />
        <!-- Individual seed dots for a sunflower appearance -->
        <!-- Add more as desired to fill the gaps between petals and add detail -->
    </g>
</svg>
```

</details>

## 14. `qwen3.5:4b`

**SVG size:** 2661 characters  
**Complexity:** Medium  
**Shape elements:** 19  
**Distinct colors:** 4  
**Raw response:** 2765 characters  
**Generation time:** 144.1s

![qwen3.5:4b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/qwen3.5_4b.svg)

### Analysis

This SVG contains approximately **19 shape elements** and uses **4 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="100%" height="auto">
    <!-- Definition for stem texture pattern (optional) -->
    <defs>
        /* Sunflower Gradient */
        <radialGradient id="headGrad" cx="35%" cy="45%">
            <stop offset="0%" style="stop-color:#FFE68C; stop-opacity:1"/> 
            <stop offset="70%" style="stop-color:#F29A1B; stop-opacity:1"/> 
        </radialGradient>
        /* Stem Texture */
    </defs>
    <!-- Petals -->
    <g id="petals">
        <path d="M 86.5,347 C 104,290 118,228 112.5 170 c-17,8 -55,22 -85,52" fill="#FFD166"/>
        <path d="M 344.5,118 C 397,50 402,-5 334.5,29 c-31,10 -76,10 -95 10" fill="#FFD166"/>
        <path d="M 200,380 C 140,420 80,440 40.5,440 c-5,-19 -7,-47 -14,-50" fill="#FFD166"/>
        <path d="M 288,34 C 268,12 248,-2 248,-50 c-15,-4 -40,0 -19,47" fill="#FFD166"/>
    </g>
    <!-- Flower Head with Seeds -->
    <circle cx="312.5" cy="256" r="112" id="flowerHead" />
        <fill url="#headGrad" /> 
        <!-- Seed Lines Texture (Brown/Black Radiating Spikes) representing seeds -->
        <g fill="none" stroke="#3A1908" stroke-width="3.5" style="mix-blend-mode: multiply;">
            <!-- Inner Seeds Cluster (Left Side of center mass) -->
             <line x1="265" y1="247.5" x2="240" y2="292"/> 
             <line x1="308" y1="234" x2="265" y2="278"/>
            <!-- Middle Ring Area -->
             <line x1="297.5" y1="202.5" x2="335" y2="250"/> 
             <line x1="240" y1="236.5" x2="280" y2="265"/>
             <!-- Outer Ring Area -->
             <line x1="327.5" y1="237.5" x2="359" y2="247"/> 
             <line x1="224" y1="208.5" x2="260" y2="226"/>
        </g>
         <!-- Seeds Texture (Top/Sides) -->
      <path d="M 329,270 C 340,250 330,218 355,205 C 375,196 360,193 341,246" stroke="#3A1908"/>
       <path d="M 239,296 C 210,266 220,215 250,220 C 270,225 255,240 236,283" stroke="#3A1908"/>
       <path d="M 340,185 C 360,162 334,148 370,160 C 398,172 380,185 355,210" stroke="#3A1908"/>
       <!-- Additional darker seed texture in center --> 
 <path d="M 334.5,229 L 362,232 M 267.5,257 Q 284,240 277,251" stroke="#A73E0D"/>
       <!-- Bottom of seeds --> 
      <path d="M 220, 296 L 242, 308 M 328.5, 259 Q 318, 265 324, 274" stroke="#A73E0D"/>
    </circle>
      <!-- Stem -->
   <path d="M 256,256 L 256,440 C 290,418 310.5, 450 256, 476" fill="#3A5C32"/>
    <!-- Left Leaf -->
   <path d="M 260,210 Q 240, 290 208,270 C 180,285 178,250 108,230" stroke="#3A5C32"/>
    <!-- Right Leaf -->   
   <path d="M 264.5,290 Q 280, 310 308,320 C 340,307 338,290 259,280" stroke="#3A5C32"/>
</svg>
```

</details>

## 15. `qwen3.5:9b`

**SVG size:** 1533 characters  
**Complexity:** Compact  
**Shape elements:** 1  
**Distinct colors:** 7  
**Raw response:** 7821 characters  
**Generation time:** 400.7s

![qwen3.5:9b sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/qwen3.5_9b.svg)

### Analysis

This SVG contains approximately **1 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="-127 -86 254 132" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
 <!-- Define colors and gradients -->
 <defs>
    <!-- Stem gradient (Dark green to light shadow) -->
    <linearGradient id="stemGrad" x1="-50%" y1="0%" x2="50%" y2="126%">
        <stop offset="0%" stop-color="#7cb342" />
        <stop offset="85%" stop-color="#2e7d32" />
    </linearGradient>
     <!-- Petal Gradient: Green base to bright yellow tip -->
    <radialGradient id="petalGrad1" cx=".5" cy=".5">
      <stop offset="0%" stop-opacity=".8469" stop-color="#ffeb3b" /> 
      <stop offset="2.7031%" stop-opacity="var(--opacity-0)" stop-stop-color="#fdd835" />
       <linearGradient id="flowerPetalGrad">
          <stop offset="0%" stop-color="#64e9ad"> <!-- Leaf color center --> 
          <stop offset="29.17% " stop-colr="#ffea5c"><!-- Bright yellow outer tip-->     
         </li>  linearGradient id=PetalGreen"><linearGradient id="petalGrad">
            <offset="0%" #a4d35f"></stop></stop><path x="29.17%" />fill=#8bb86f</color>/</stop> --> 
        <!-- Gradient Stop logic: dark green near center (inside petal) to yellow tip-->
    </defs>
     <linearGradient id="petalGrad">
         <stop offset="0%" stop-color="#4caf50"> <!-- Green where stem meets head -->
         <offset=29.17%">stop=#388e3c</color></stop><stop offset="--opacity-</stop />/--> 
      <linearGradient id="petalGrad">
          <stop color="#bdf9a6" stop-color="green" />
       </p
    --offset=50%" fill=color=#7cb342"></gradient>
</svg>
```

</details>

## 16. `qwen3.5:latest`

**SVG size:** 991 characters  
**Complexity:** Compact  
**Shape elements:** 7  
**Distinct colors:** 5  
**Raw response:** 1595 characters  
**Generation time:** 467.9s

![qwen3.5:latest sunflower SVG](/assets/img/diagrams/ollama-svg-sunflower/qwen3.5_latest.svg)

### Analysis

This SVG contains approximately **7 shape elements** and uses **5 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Uses `<use>` references to reuse defined elements -- ideal for repeating motifs like seeds, scales, or leaflets
- Uses transforms (translate, rotate, scale) for precise positioning and rotation
- Uses Bezier curves (`C`, `Q`, `A` path commands) for smooth organic shapes
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500" width="100%" height="auto">
  <!-- Background -->
  <rect width="100%" height="100%" fill="#f9f4e6"/>
  <!-- Stem -->
  <path d="M-37,285 Q-76,400 -5,480" stroke="#4a7c2a" stroke-width="14" fill="none"/>
  <!-- Leaves on stem -->
  <path d=" M-49.4,-256 L-32.6,-256 Q-15.9,-208 -1.8,-270 Z" 
        transform="translate(0, 0)" fill="#3d6b1e"/>
  <!-- Large leaf above -->
  <path d="-35 470 q5,-20,-2,-50 t-20,-15 l-80 ,10" 
        stroke="#2f5a19" stroke-width="16" fill="none"/>
    <!-- Large leaf below -->
  <path d="-35,470 q5,-20,-2,-50 t-20,-15 l80 ,10" 
        transform="rotate(180 -50.6928 1)" fill="#2f5a19"/>
<!-- Seeds -->
<ellipse cx="240" cy="-273" rx="38,14" ry="0,-33"></ellipse> 
<!-- Sunflower Head Center (brown disk)-->
<g transform="translate(240 -3.6)"> 
    <circle r="95" fill="#a78d7a"/>   
    <!-- Seed Patterns Inside -->
        <use x="-1, y= 1" /></g> 
<defs><text dy=".2em">Seeds</text></svg>
```

</details>

## Failed Models

The following 9 models did not produce a valid SVG (either returned an error or the response did not contain a parseable `<svg>...</svg>` block):

- `gemma4:31b-it-qat` -- TimeoutError: timed out
- `gemma4:latest` -- TimeoutError: timed out
- `jeffgreen311/Eve-V2-Unleashed-Qwen3.5-8B-Liberated-4K-4B-Merged:latest` -- TimeoutError: timed out
- `kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2:latest` -- no SVG block in response
- `qwen3:14b` -- TimeoutError: timed out
- `qwen3:8b` -- TimeoutError: timed out
- `SetneufPT/Qwen3.6-27B-MTP_Q3_32K_16GB-GPU:latest` -- TimeoutError: timed out
- `SetneufPT/Qwen3.6-27B.MTP_Q3_32K_16GB-GPU:latest` -- TimeoutError: timed out
- `VladimirGav/Qwen3.6-27B-16GB-VRAM-Uncensored:latest` -- TimeoutError: timed out

## Conclusion

We asked 25 Ollama local models to draw **a sunflower** -- a subject deeply connected to Fibonacci mathematics. The results reveal each model's natural instinct for mathematical patterns, organic curves, and natural color palettes.

**Key takeaways:**

- **Mathematical intuition varies widely**: some models attempted Fibonacci spiral arrangements naturally, while others defaulted to simple grids or flat shapes. The models that attempted spirals demonstrate a deeper understanding of natural geometry.
- **Detail and file size trade-off**: models that produced the richest scenes (Very high complexity) also generated the largest SVG files. For web embedding, "Balanced" or "Detailed" models may be more practical.
- **Color palettes differ dramatically**: some models used sophisticated gradients and 15+ distinct colors, while others used as few as 4 flat colors. More colors generally means a more lifelike result.
- **Code structure quality varies**: the best models used `<defs>`, `<use>`, and `<pattern>` elements to efficiently generate repeating structures (seeds, scales, leaflets). This is a strong signal of model capability for practical SVG work.
- **No single model is best at everything**: the "right" model depends on whether you prioritize mathematical accuracy, visual beauty, file size, or code quality.

**Try it yourself**: run the benchmark script with your own prompt and see how the models perform on your specific use case. The full code is available in the [ollama-svg-benchmark repository](https://github.com/py2ai/ollama-svg-benchmark).

---

*Which model do you think drew the best sunflower? Let us know in the comments below!*
