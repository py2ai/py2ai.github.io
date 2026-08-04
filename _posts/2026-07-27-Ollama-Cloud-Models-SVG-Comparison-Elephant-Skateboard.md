---
layout: post
title: "Which Ollama Cloud Model is Best? Elephant on a Skateboard SVG Comparison (13 Models)"
description: "Compare 14 Ollama cloud models on a whimsical prompt: an elephant on a skateboard. Find the best LLM for funny animal SVG scenes. You decide the winner."
date: 2026-07-27
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Elephant-Skateboard/
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
  - Elephant
  - Skateboard
  - Animal
  - Funny
  - Creative AI
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama cloud model comparison, elephant skateboard SVG, AI elephant drawing, LLM SVG benchmark, AI image generation comparison, deepseek vs glm vs qwen, which Ollama model is best, Ollama cloud models 2026, AI creative art, funny animal SVG, elephant SVG, AI art comparison, complex SVG scene, animal on skateboard, whimsical AI art"
---

# Which Ollama Cloud Model is Best? Elephant on a Skateboard SVG Comparison (13 Models)

After testing LLMs on ducks, vehicles, dev scenes, marine life, chess, and the FIFA World Cup, we wanted to know: **can today's top models draw a whimsical, physically impossible scene?** This time we asked 14 Ollama cloud models to draw **an elephant on a skateboard** -- a prompt that tests organic anatomy (a massive elephant), mechanical precision (a skateboard with wheels), scale mismatch (a huge animal on a tiny board), and creative humor.

The prompt was: `Make an svg of elephant on a skateboard`

This is the ninth in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/), [duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/), [baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/), [octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/), and [FIFA World Cup 2026](/Ollama-Cloud-Models-SVG-Comparison-Fifa-Worldcup-2026/).

**Why an elephant on a skateboard?** This prompt is a delightfully absurd stress test for SVG generation because it combines multiple hard problems: (1) **Organic anatomy** -- an elephant has a massive body, four thick legs, a trunk, large ears (African) or small ears (Asian), tusks, a tail, and wrinkles, all requiring complex `<path>` curves, (2) **Mechanical precision** -- a skateboard has a specific deck shape, 4 wheels, and 2 trucks (axle assemblies) that must be positioned symmetrically, (3) **Scale mismatch** -- an elephant weighs 6 tons and a skateboard is meant for a 80kg human; the model must decide how to reconcile this (tiny skateboard under a huge elephant? oversized skateboard?), (4) **Balance and pose** -- a skateboarding elephant must somehow balance on the board, which requires the model to think about physics and pose, (5) **Concept combination** -- the model must understand both "elephant" and "skateboard" and combine them in a single coherent scene, (6) **Humor and whimsy** -- the prompt is inherently funny, and a good model should capture that playfulness (e.g., the elephant looking excited, wearing a helmet, or doing a trick). A model that draws a great trophy may fail here because elephants require organic curves, not geometric shapes.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Whimsical Animal SVGs

The elephant-skateboard prompt rewards different things than previous prompts. Here are the criteria to use:

- **Elephant anatomy**: Does the SVG have a trunk, four legs, large ears, a tail, and tusks? Or is it a generic blob? The trunk is the hardest part -- it requires complex bezier curves and must be positioned correctly on the face.
- **Skateboard accuracy**: Does the SVG have a deck (the board), 4 wheels, and 2 trucks? Or is it just a rectangle? A good skateboard has the iconic shape with upturned ends.
- **Scale and proportion**: How does the model handle the size mismatch? A tiny skateboard under a huge elephant is funny but unrealistic; an oversized skateboard looks more plausible. The model's choice reveals its creative judgment.
- **Pose and balance**: Is the elephant standing on the skateboard? Doing a trick? Looking excited? A good scene captures the dynamism of skateboarding, not just a static elephant on a static board.
- **Scene context**: Did the model add motion lines, a ground/surface, a sky, or other context? Or is the elephant-skateboard floating in a void?
- **Humor and whimsy**: Does the SVG capture the playfulness of the prompt? Look for details like a helmet, sunglasses, a grin, or a trick pose that show the model "got the joke".
- **SVG code quality**: Does it use `<defs>`, `<use>`, gradients, and clean structure? Better code is easier to tweak (e.g., to recolor the elephant or change the skateboard design).

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with minimal post-processing (adding `width="100%" height="auto"` for responsive embedding and fixing XML errors so the SVG renders in browsers).

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `deepseek-v4-flash:cloud` | 5554 | 49 | 17 | High | Detailed |
| 2 | `deepseek-v4-pro:cloud` | 8153 | 67 | 10 | Very high | Richest scene |
| 3 | `gemma4:31b-cloud` | 1240 | 13 | 6 | Compact | Compact |
| 4 | `gemma4:cloud` | 1155 | 12 | 6 | Compact | Compact |
| 5 | `glm-5.1:cloud` | 11862 | 78 | 26 | Very high | Richest scene |
| 6 | `glm-5.2:cloud` | 5364 | 60 | 18 | Very high | Richest scene |
| 7 | `gpt-oss:120b-cloud` | 1659 | 20 | 7 | Medium | Balanced |
| 8 | `kimi-k2.6:cloud` | 7511 | 30 | 15 | High | Detailed |
| 9 | `minimax-m2.7:cloud` | 3010 | 24 | 8 | Medium | Balanced |
| 10 | `minimax-m3:cloud` | 5627 | 63 | 23 | Very high | Richest scene |
| 11 | `nemotron-3-super:cloud` | 1995 | 22 | 7 | Medium | Balanced |
| 12 | `nemotron-3-ultra:cloud` | 7239 | 39 | 14 | High | Detailed |
| 13 | `qwen3.5:397b-cloud` | 2604 | 22 | 15 | Medium | Balanced |
| 14 | `deepseek-v4-flash:0731-cloud` | 4669 | 48 | 14 | Medium | Balanced |
| 15 | `bjoernb/claude-opus-4-5:latest` | - | - | - | - | Retired (410) |
| 16 | `deepseek-v3.1:671b-cloud` | - | - | - | - | Retired (410) |
| 17 | `glm-5:cloud` | - | - | - | - | Retired (410) |
| 18 | `qwen3-vl:235b-cloud` | - | - | - | - | Retired (410) |

**14 out of 18** active models produced a valid SVG. The 4 retired models returned HTTP 410 Gone (removed from Ollama Cloud on 2026-07-15).

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most detailed elephant skateboard SVG**: pick models labeled "Very high" complexity in the table above
- **You want the fastest response**: look at the per-model sections below for the elapsed time
- **You want the cleanest, most reusable SVG code**: pick models that use `<defs>`, `<use>`, and transforms (see raw source below each SVG)
- **You want a small, efficient SVG for web embedding**: pick models with "Compact" verdict
- **You want accurate elephant anatomy (trunk, ears, tusks)**: check the per-model analysis -- the trunk is the hardest part
- **You want a funny/whimsical scene**: look for models that added details like motion lines, a helmet, or a trick pose
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"
- **You want to compare within a model family**: pick `deepseek-v4-pro` vs `deepseek-v4-flash`, or `glm-5.1` vs `glm-5.2`, or `minimax-m2.7` vs `minimax-m3`

Now read on for the full per-model breakdown and judge for yourself.

## 1. `deepseek-v4-flash:cloud`

**SVG size:** 5554 characters  
**Complexity:** High  
**Shape elements:** 49  
**Distinct colors:** 17  
**Raw response:** 6112 characters  
**Generation time:** 48.6s

![deepseek-v4-flash:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/deepseek-v4-flash_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the elephant and tilting the skateboard
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders tusks -- the model captured fine anatomical detail
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Renders trucks (the axle assemblies) -- the model captured fine skateboard mechanical detail
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface

With 5,554 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
    <defs>
        <!-- Drop shadow for depth -->
        <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
            <feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="#000" flood-opacity="0.2"/>
        </filter>
        <filter id="light-shadow" x="-10%" y="-10%" width="120%" height="120%">
            <feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#000" flood-opacity="0.15"/>
        </filter>
    </defs>
    <!-- Background -->
    <rect width="800" height="600" fill="#F0F8FF" />
    <circle cx="400" cy="300" r="220" fill="#FFEAA7" opacity="0.4" />
    <!-- Background Stars/Sparkles -->
    <path d="M 120 150 Q 130 150 130 140 Q 130 150 140 150 Q 130 150 130 160 Q 130 150 120 150 Z" fill="#FFD166" />
    <path d="M 680 200 Q 690 200 690 190 Q 690 200 700 200 Q 690 200 690 210 Q 690 200 680 200 Z" fill="#FFD166" />
    <path d="M 200 100 Q 210 100 210 90 Q 210 100 220 100 Q 210 100 210 110 Q 210 100 200 100 Z" fill="#FFD166" />
    <!-- Ground Shadow -->
    <ellipse cx="400" cy="560" rx="240" ry="20" fill="rgba(0,0,0,0.15)" />
    <!-- Motion Lines (Wind) -->
    <g id="motion-lines" stroke="#A0B2BE" stroke-width="6" stroke-linecap="round" fill="none" opacity="0.6">
        <path d="M 660 450 L 720 450" />
        <path d="M 680 480 L 740 480" />
        <path d="M 650 510 L 700 510" />
        <path d="M 120 480 L 160 480" />
        <path d="M 100 510 L 150 510" />
    </g>
    <!-- Skateboard -->
    <g id="skateboard" filter="url(#shadow)">
        <!-- Deck -->
        <path d="M 150 480 Q 180 450 250 450 L 550 450 Q 620 450 650 480 L 650 490 Q 620 510 550 510 L 250 510 Q 180 510 150 490 Z" fill="#E63946" stroke="#C1121F" stroke-width="4" stroke-linejoin="round" />
        <!-- Deck Design: Stripe &amp; Lightning Bolt -->
        <path d="M 250 460 L 550 460 L 550 470 L 250 470 Z" fill="#F8F9FA" opacity="0.9" />
        <path d="M 350 460 L 380 460 L 370 480 L 400 480 L 360 510 L 380 490 L 350 490 Z" fill="#FFD166" />
        <!-- Trucks -->
        <path d="M 210 510 L 230 510 L 220 530 Z" fill="#495057" />
        <path d="M 570 510 L 590 510 L 580 530 Z" fill="#495057" />
        <!-- Wheels -->
        <circle cx="220" cy="540" r="30" fill="#1D3557" />
        <circle cx="220" cy="540" r="12" fill="#F1FAEE" />
        <circle cx="580" cy="540" r="30" fill="#1D3557" />
        <circle cx="580" cy="540" r="12" fill="#F1FAEE" />
    </g>
    <!-- Elephant -->
    <g id="elephant" filter="url(#shadow)">
        <!-- Back Legs -->
        <rect x="460" y="420" width="30" height="70" rx="15" fill="#8A9BA8" />
        <rect x="330" y="420" width="30" height="70" rx="15" fill="#8A9BA8" />
        <!-- Tail -->
        <path d="M 510 320 Q 550 300 560 360" fill="none" stroke="#8A9BA8" stroke-width="10" stroke-linecap="round" />
        <path d="M 560 360 Q 570 380 550 390 Q 560 370 560 360 Z" fill="#8A9BA8" />
        <!-- Body -->
        <ellipse cx="400" cy="350" rx="120" ry="140" fill="#A0B2BE" />
        <!-- Body Highlight -->
        <path d="M 320 240 Q 400 210 480 240" fill="none" stroke="#FFFFFF" stroke-width="8" stroke-linecap="round" opacity="0.4" />
        <!-- Back Ear -->
        <ellipse cx="350" cy="270" rx="45" ry="75" transform="rotate(15 350 270)" fill="#8A9BA8" />
        <!-- Front Legs -->
        <rect x="440" y="410" width="35" height="80" rx="15" fill="#A0B2BE" />
        <rect x="310" y="410" width="35" height="80" rx="15" fill="#A0B2BE" />
        <!-- Front Leg Toes -->
        <circle cx="320" cy="485" r="4" fill="#5A6B7A" />
        <circle cx="330" cy="485" r="4" fill="#5A6B7A" />
        <circle cx="340" cy="485" r="4" fill="#5A6B7A" />
        <circle cx="450" cy="485" r="4" fill="#5A6B7A" />
        <circle cx="460" cy="485" r="4" fill="#5A6B7A" />
        <circle cx="470" cy="485" r="4" fill="#5A6B7A" />
        <!-- Head -->
        <circle cx="290" cy="280" r="85" fill="#A0B2BE" />
        <!-- Front Ear -->
        <ellipse cx="250" cy="260" rx="55" ry="85" transform="rotate(-15 250 260)" fill="#A0B2BE" />
        <ellipse cx="250" cy="260" rx="35" ry="65" transform="rotate(-15 250 260)" fill="#FFB5A7" opacity="0.6" />
        <!-- Trunk -->
        <path d="M 230 330 C 160 370, 130 280, 170 210 C 190 170, 240 150, 260 150 C 250 160, 240 170, 230 180 C 200 210, 190 250, 210 300 C 215 320, 230 320, 240 310 Z" fill="#A0B2BE" />
        <!-- Trunk Highlight -->
        <path d="M 220 320 C 170 350, 150 280, 180 220" fill="none" stroke="#FFFFFF" stroke-width="6" stroke-linecap="round" opacity="0.4" />
        <!-- Tusks -->
        <path d="M 230 350 Q 190 380 180 400 Q 200 390 240 360 Z" fill="#FFFFFF" />
        <!-- Eye -->
        <circle cx="260" cy="270" r="10" fill="#1D3557" />
        <circle cx="263" cy="267" r="4" fill="#FFFFFF" />
        <!-- Cheek Blush -->
        <ellipse cx="270" cy="300" rx="18" ry="12" fill="#FFB5A7" opacity="0.8" />
        <!-- Helmet -->
        <path d="M 210 240 C 210 160, 320 160, 340 240 Z" fill="#2A9D8F" />
        <path d="M 200 240 Q 270 250 350 240" fill="none" stroke="#21867A" stroke-width="12" stroke-linecap="round" />
        <path d="M 220 240 L 230 330" fill="none" stroke="#21867A" stroke-width="6" />
        <circle cx="230" cy="330" r="4" fill="#21867A" />
        <!-- Helmet Highlight -->
        <path d="M 230 180 Q 270 170 310 190" fill="none" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.5" />
    </g>
</svg>
```

</details>

## 2. `deepseek-v4-pro:cloud`

**SVG size:** 8153 characters  
**Complexity:** Very high  
**Shape elements:** 67  
**Distinct colors:** 10  
**Raw response:** 8564 characters  
**Generation time:** 95.2s

![deepseek-v4-pro:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/deepseek-v4-pro_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the elephant body and skateboard texture)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the elephant and tilting the skateboard
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders tusks -- the model captured fine anatomical detail
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Renders trucks (the axle assemblies) -- the model captured fine skateboard mechanical detail
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface
- Sky/atmosphere is present -- the model created an outdoor scene

With 8,153 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <defs>
    <!-- Gradients -->
    <linearGradient id="elephantGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#A0C4E2" />
      <stop offset="100%" stop-color="#8BAEC2" />
    </linearGradient>
    <linearGradient id="deckGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#F4A261" />
      <stop offset="100%" stop-color="#E76F51" />
    </linearGradient>
    <radialGradient id="wheelGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#2A9D8F" />
      <stop offset="100%" stop-color="#264653" />
    </radialGradient>
    <!-- Drop Shadow for Elephant -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="#000" flood-opacity="0.15" />
    </filter>
  </defs>
  <!-- Background Elements (Motion Lines &amp; Wind) -->
  <g id="background-elements">
    <!-- Ground Shadow -->
    <ellipse cx="400" cy="580" rx="200" ry="15" fill="#000" opacity="0.1" />
    <!-- Motion Lines -->
    <line x1="150" y1="500" x2="220" y2="500" stroke="#333" stroke-width="4" stroke-linecap="round" opacity="0.4" />
    <line x1="170" y1="520" x2="230" y2="520" stroke="#333" stroke-width="4" stroke-linecap="round" opacity="0.4" />
    <line x1="130" y1="540" x2="210" y2="540" stroke="#333" stroke-width="4" stroke-linecap="round" opacity="0.4" />
    <line x1="180" y1="560" x2="230" y2="560" stroke="#333" stroke-width="4" stroke-linecap="round" opacity="0.4" />
    <!-- Wind Lines -->
    <path d="M 600 200 Q 650 200 700 220" stroke="#333" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.3" />
    <path d="M 620 240 Q 670 240 720 260" stroke="#333" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.3" />
    <path d="M 580 280 Q 630 280 680 300" stroke="#333" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.3" />
    <!-- Sparkles / Stars -->
    <path d="M 150 200 L 155 215 L 170 220 L 155 225 L 150 240 L 145 225 L 130 220 L 145 215 Z" fill="#E9C46A" opacity="0.8" />
    <path d="M 650 150 L 653 160 L 663 163 L 653 166 L 650 176 L 647 166 L 637 163 L 647 160 Z" fill="#E9C46A" opacity="0.8" />
    <path d="M 680 300 L 682 306 L 688 308 L 682 310 L 680 316 L 678 310 L 672 308 L 678 306 Z" fill="#E9C46A" opacity="0.8" />
  </g>
  <!-- Skateboard -->
  <g id="skateboard">
    <!-- Deck -->
    <rect x="250" y="480" width="300" height="20" rx="10" fill="url(#deckGrad)" />
    <!-- Grip Tape -->
    <rect x="250" y="480" width="300" height="10" rx="5" fill="#333" />
    <!-- Grip Tape Texture -->
    <line x1="260" y1="485" x2="540" y2="485" stroke="#555" stroke-width="2" stroke-dasharray="4 4" />
    <line x1="260" y1="490" x2="540" y2="490" stroke="#555" stroke-width="2" stroke-dasharray="4 4" />
    <!-- Deck Stripe / Lightning Bolt -->
    <path d="M 360 485 L 380 485 L 370 495 L 390 495 L 360 505 L 370 495 L 350 495 Z" fill="#FFF" opacity="0.8" />
    <!-- Trucks -->
    <!-- Front Truck -->
    <rect x="290" y="500" width="20" height="30" fill="#D3D3D3" />
    <path d="M 280 530 L 320 530 L 310 540 L 290 540 Z" fill="#D3D3D3" />
    <!-- Back Truck -->
    <rect x="490" y="500" width="20" height="30" fill="#D3D3D3" />
    <path d="M 480 530 L 520 530 L 510 540 L 490 540 Z" fill="#D3D3D3" />
    <!-- Wheels -->
    <!-- Front Left -->
    <circle cx="280" cy="550" r="15" fill="url(#wheelGrad)" />
    <circle cx="280" cy="550" r="6" fill="#264653" />
    <!-- Front Right -->
    <circle cx="320" cy="550" r="15" fill="url(#wheelGrad)" />
    <circle cx="320" cy="550" r="6" fill="#264653" />
    <!-- Back Left -->
    <circle cx="480" cy="550" r="15" fill="url(#wheelGrad)" />
    <circle cx="480" cy="550" r="6" fill="#264653" />
    <!-- Back Right -->
    <circle cx="520" cy="550" r="15" fill="url(#wheelGrad)" />
    <circle cx="520" cy="550" r="6" fill="#264653" />
  </g>
  <!-- Elephant -->
  <g id="elephant" filter="url(#shadow)">
    <!-- Tail -->
    <path d="M 530 280 Q 570 250 580 300" stroke="url(#elephantGrad)" stroke-width="12" fill="none" stroke-linecap="round" />
    <path d="M 580 300 Q 590 310 580 320 Q 570 310 580 300" fill="url(#elephantGrad)" />
    <!-- Back Right Leg (Pushing) -->
    <rect x="520" y="420" width="45" height="120" rx="20" fill="url(#elephantGrad)" transform="rotate(-45 520 420)" />
    <!-- Back Right Foot -->
    <ellipse cx="480" cy="560" rx="35" ry="15" fill="url(#elephantGrad)" transform="rotate(-15 480 560)" />
    <!-- Back Right Toenails -->
    <ellipse cx="470" cy="560" rx="6" ry="10" fill="#FFF" transform="rotate(-15 480 560)" />
    <ellipse cx="480" cy="560" rx="6" ry="10" fill="#FFF" transform="rotate(-15 480 560)" />
    <ellipse cx="490" cy="560" rx="6" ry="10" fill="#FFF" transform="rotate(-15 480 560)" />
    <!-- Back Right Knee Pad -->
    <circle cx="500" cy="460" r="22" fill="#E9C46A" />
    <!-- Back Left Leg (On Board) -->
    <rect x="440" y="380" width="45" height="100" rx="20" fill="url(#elephantGrad)" transform="rotate(-10 440 380)" />
    <!-- Back Left Toenails -->
    <ellipse cx="450" cy="475" rx="6" ry="10" fill="#FFF" />
    <ellipse cx="460" cy="475" rx="6" ry="10" fill="#FFF" />
    <ellipse cx="470" cy="475" rx="6" ry="10" fill="#FFF" />
    <!-- Back Left Knee Pad -->
    <circle cx="460" cy="460" r="22" fill="#E9C46A" />
    <!-- Front Left Leg (On Board) -->
    <rect x="300" y="380" width="45" height="100" rx="20" fill="url(#elephantGrad)" transform="rotate(10 300 380)" />
    <!-- Front Left Toenails -->
    <ellipse cx="310" cy="475" rx="6" ry="10" fill="#FFF" />
    <ellipse cx="320" cy="475" rx="6" ry="10" fill="#FFF" />
    <ellipse cx="330" cy="475" rx="6" ry="10" fill="#FFF" />
    <!-- Front Left Knee Pad -->
    <circle cx="320" cy="460" r="22" fill="#E9C46A" />
    <!-- Front Right Leg (On Board) -->
    <rect x="360" y="390" width="45" height="90" rx="20" fill="url(#elephantGrad)" transform="rotate(5 360 390)" />
    <!-- Front Right Toenails -->
    <ellipse cx="370" cy="475" rx="6" ry="10" fill="#FFF" />
    <ellipse cx="380" cy="475" rx="6" ry="10" fill="#FFF" />
    <ellipse cx="390" cy="475" rx="6" ry="10" fill="#FFF" />
    <!-- Front Right Knee Pad -->
    <circle cx="380" cy="460" r="22" fill="#E9C46A" />
    <!-- Main Body -->
    <ellipse cx="400" cy="300" rx="140" ry="160" fill="url(#elephantGrad)" transform="rotate(-10 400 300)" />
    <!-- Left Ear (Behind Head) -->
    <ellipse cx="300" cy="200" rx="60" ry="80" fill="#8BAEC2" transform="rotate(20 300 200)" />
    <!-- Head -->
    <circle cx="260" cy="240" r="90" fill="url(#elephantGrad)" />
    <!-- Right Ear (In Front of Head) -->
    <ellipse cx="280" cy="180" rx="50" ry="70" fill="url(#elephantGrad)" transform="rotate(10 280 180)" />
    <!-- Trunk -->
    <path d="M 200 280 C 140 320, 140 400, 170 440 C 190 460, 220 450, 210 420 C 200 390, 210 340, 240 310 Z" fill="url(#elephantGrad)" />
    <!-- Tusk -->
    <path d="M 190 280 C 160 300, 140 320, 150 330 C 170 330, 180 310, 200 290 Z" fill="#FFF" />
    <!-- Eye &amp; Sunglasses -->
    <path d="M 190 210 Q 230 200 270 210 L 260 230 Q 230 220 200 230 Z" fill="#333" />
    <!-- Sunglasses Reflection -->
    <path d="M 200 215 L 240 215" stroke="#FFF" stroke-width="2" stroke-linecap="round" />
    <!-- Scarf / Bandana -->
    <path d="M 280 270 Q 300 290 320 270 L 310 250 L 290 250 Z" fill="#E76F51" />
    <path d="M 320 270 Q 340 280 350 300 L 340 310 Q 330 290 310 280 Z" fill="#E76F51" />
    <!-- Helmet -->
    <path d="M 190 160 C 190 90, 330 90, 330 160 Z" fill="#E9C46A" />
    <!-- Helmet Visor -->
    <path d="M 190 160 Q 260 140 330 160 Q 260 150 190 160" fill="#333" />
    <!-- Helmet Straps -->
    <line x1="220" y1="160" x2="220" y2="200" stroke="#333" stroke-width="4" />
    <line x1="300" y1="160" x2="300" y2="200" stroke="#333" stroke-width="4" />
  </g>
  <!-- Shadows for Feet -->
  <ellipse cx="480" cy="580" rx="30" ry="8" fill="#000" opacity="0.2" />
  <ellipse cx="400" cy="480" rx="100" ry="10" fill="#000" opacity="0.2" />
</svg>
```

</details>

## 3. `gemma4:31b-cloud`

**SVG size:** 1240 characters  
**Complexity:** Compact  
**Shape elements:** 13  
**Distinct colors:** 6  
**Raw response:** 1898 characters  
**Generation time:** 5.8s

![gemma4:31b-cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/gemma4_31b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface

With 1,240 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background Circle -->
  <circle cx="100" cy="100" r="90" fill="#f0f8ff" />
  <!-- Skateboard Deck -->
  <rect x="50" y="150" width="100" height="10" rx="5" fill="#d2691e" />
  <!-- Skateboard Wheels -->
  <circle cx="70" cy="162" r="6" fill="#333" />
  <circle cx="130" cy="162" r="6" fill="#333" />
  <!-- Elephant Body -->
  <path d="M70 150 C 60 150, 60 100, 80 90 C 100 80, 140 80, 150 100 C 160 120, 150 150, 130 150" fill="#a9a9a9" />
  <!-- Elephant Head -->
  <circle cx="155" cy="95" r="25" fill="#a9a9a9" />
  <!-- Ear -->
  <ellipse cx="140" cy="95" rx="12" ry="18" fill="#808080" />
  <!-- Trunk -->
  <path d="M175 95 Q 185 95, 180 115 Q 175 125, 165 120" stroke="#a9a9a9" stroke-width="8" stroke-linecap="round" fill="none" />
  <!-- Eye -->
  <circle cx="165" cy="90" r="2" fill="#333" />
  <!-- Legs -->
  <rect x="80" y="140" width="12" height="15" rx="2" fill="#a9a9a9" />
  <rect x="110" y="140" width="12" height="15" rx="2" fill="#a9a9a9" />
  <rect x="130" y="140" width="12" height="15" rx="2" fill="#a9a9a9" />
  <!-- Tail -->
  <path d="M70 120 Q 60 120, 65 130" stroke="#a9a9a9" stroke-width="3" fill="none" />
</svg>
```

</details>

## 4. `gemma4:cloud`

**SVG size:** 1155 characters  
**Complexity:** Compact  
**Shape elements:** 12  
**Distinct colors:** 6  
**Raw response:** 1934 characters  
**Generation time:** 5.6s

![gemma4:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/gemma4_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface

With 1,155 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background Circle -->
  <circle cx="100" cy="100" r="90" fill="#f0f0f0" />
  <!-- Skateboard -->
  <!-- Deck -->
  <rect x="50" y="150" width="100" height="10" rx="5" fill="#d32f2f" />
  <!-- Wheels -->
  <circle cx="70" cy="162" r="6" fill="#333" />
  <circle cx="130" cy="162" r="6" fill="#333" />
  <!-- Elephant Body -->
  <ellipse cx="100" cy="110" rx="45" ry="35" fill="#90a4ae" />
  <!-- Head -->
  <circle cx="135" cy="100" r="25" fill="#90a4ae" />
  <!-- Ear -->
  <ellipse cx="125" cy="95" rx="12" ry="18" fill="#78909c" />
  <!-- Trunk -->
  <path d="M 155 100 Q 170 100 170 120" stroke="#90a4ae" stroke-width="10" fill="none" stroke-linecap="round" />
  <!-- Eye -->
  <circle cx="142" cy="95" r="2" fill="#333" />
  <!-- Legs -->
  <!-- Back leg -->
  <rect x="70" y="120" width="12" height="30" rx="4" fill="#90a4ae" />
  <!-- Front leg -->
  <rect x="115" y="120" width="12" height="30" rx="4" fill="#90a4ae" />
  <!-- Tail -->
  <path d="M 55 110 Q 45 110 45 120" stroke="#90a4ae" stroke-width="3" fill="none" stroke-linecap="round" />
</svg>
```

</details>

## 5. `glm-5.1:cloud`

**SVG size:** 11862 characters  
**Complexity:** Very high  
**Shape elements:** 78  
**Distinct colors:** 26  
**Raw response:** 16000 characters  
**Generation time:** 68.2s

![glm-5.1:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/glm-5.1_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the elephant body and skateboard texture)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Includes animation elements -- the model attempted motion (e.g., rolling wheels, balancing)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders tusks -- the model captured fine anatomical detail
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Renders trucks (the axle assemblies) -- the model captured fine skateboard mechanical detail
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface

With 11,862 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 600 450" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
    <defs>
      <!-- Gradients -->
      <linearGradient id="elephantGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#a8a8b8"/>
        <stop offset="100%" stop-color="#787888"/>
      </linearGradient>
      <linearGradient id="bellyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#c0c0cc"/>
        <stop offset="100%" stop-color="#a0a0b0"/>
      </linearGradient>
      <linearGradient id="deckGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#e94560"/>
        <stop offset="50%" stop-color="#ff6b6b"/>
        <stop offset="100%" stop-color="#e94560"/>
      </linearGradient>
      <linearGradient id="wheelGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#555570"/>
        <stop offset="100%" stop-color="#333348"/>
      </linearGradient>
      <radialGradient id="eyeShine" cx="35%" cy="35%" r="50%">
        <stop offset="0%" stop-color="#ffffff"/>
        <stop offset="100%" stop-color="#ddeeff"/>
      </radialGradient>
      <linearGradient id="groundGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="transparent"/>
        <stop offset="20%" stop-color="#e94560"/>
        <stop offset="80%" stop-color="#e94560"/>
        <stop offset="100%" stop-color="transparent"/>
      </linearGradient>
      <filter id="shadowFilter">
        <feDropShadow dx="0" dy="4" stdDeviation="3" flood-color="#000" flood-opacity="0.3"/>
      </filter>
      <filter id="glow">
        <feGaussianBlur stdDeviation="4" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
    </defs>
    <!-- Background stars/dots -->
    <circle cx="50" cy="40" r="1.5" fill="#ffffff" opacity="0.3"/>
    <circle cx="150" cy="80" r="1" fill="#ffffff" opacity="0.2"/>
    <circle cx="520" cy="50" r="1.5" fill="#ffffff" opacity="0.25"/>
    <circle cx="480" cy="100" r="1" fill="#ffffff" opacity="0.2"/>
    <circle cx="100" cy="120" r="1" fill="#ffffff" opacity="0.15"/>
    <circle cx="550" cy="160" r="1.5" fill="#ffffff" opacity="0.2"/>
    <circle cx="30" cy="180" r="1" fill="#ffffff" opacity="0.15"/>
    <circle cx="570" cy="250" r="1" fill="#ffffff" opacity="0.2"/>
    <circle cx="250" cy="30" r="1.2" fill="#ffffff" opacity="0.18"/>
    <circle cx="400" cy="60" r="1" fill="#ffffff" opacity="0.15"/>
    <!-- Speed lines behind skateboard -->
    <g class="speed-line">
      <line x1="140" y1="350" x2="80" y2="350" stroke="#e94560" stroke-width="2.5" stroke-linecap="round" opacity="0.7"/>
    </g>
    <g class="speed-line">
      <line x1="130" y1="365" x2="60" y2="365" stroke="#e94560" stroke-width="2" stroke-linecap="round" opacity="0.5"/>
    </g>
    <g class="speed-line">
      <line x1="145" y1="380" x2="75" y2="380" stroke="#ff6b6b" stroke-width="1.5" stroke-linecap="round" opacity="0.4"/>
    </g>
    <g class="speed-line">
      <line x1="135" y1="395" x2="90" y2="395" stroke="#ff6b6b" stroke-width="1.5" stroke-linecap="round" opacity="0.3"/>
    </g>
    <!-- Ground glow line -->
    <line x1="50" y1="395" x2="550" y2="395" stroke="url(#groundGrad)" stroke-width="3" class="ground-glow" stroke-linecap="round"/>
    <!-- Skateboard shadow -->
    <ellipse cx="300" cy="392" rx="120" ry="6" fill="#000" opacity="0.2"/>
    <!-- ===== SKATEBOARD ===== -->
    <!-- Deck -->
    <path d="M150,355 Q140,355 135,348 Q130,340 138,335 L160,335 Q165,340 170,345 L430,345 Q435,340 440,335 L462,335 Q470,340 465,348 Q460,355 450,355 Z"
          fill="url(#deckGrad)" filter="url(#shadowFilter)"/>
    <!-- Deck stripe detail -->
    <rect x="200" y="338" width="200" height="3" rx="1.5" fill="#ffffff" opacity="0.3"/>
    <!-- Deck top edge highlight -->
    <path d="M155,345 L445,345" stroke="#ffffff" stroke-width="1" opacity="0.15" stroke-linecap="round"/>
    <!-- Trucks (metal parts connecting wheels to deck) -->
    <rect x="195" y="355" width="30" height="10" rx="3" fill="#888899"/>
    <rect x="375" y="355" width="30" height="10" rx="3" fill="#888899"/>
    <!-- Wheels -->
    <g>
      <!-- Back left wheel -->
      <circle cx="195" cy="372" r="12" fill="url(#wheelGrad)" stroke="#222238" stroke-width="1.5"/>
      <circle cx="195" cy="372" r="5" fill="#444460" stroke="#555570" stroke-width="0.5"/>
      <line x1="195" y1="363" x2="195" y2="381" stroke="#555570" stroke-width="0.8" opacity="0.5"/>
      <line x1="186" y1="372" x2="204" y2="372" stroke="#555570" stroke-width="0.8" opacity="0.5"/>
      <!-- Back right wheel (behind) -->
      <circle cx="210" cy="374" r="11" fill="#2a2a3e" stroke="#222238" stroke-width="1.5" opacity="0.6"/>
      <!-- Front left wheel -->
      <circle cx="405" cy="372" r="12" fill="url(#wheelGrad)" stroke="#222238" stroke-width="1.5"/>
      <circle cx="405" cy="372" r="5" fill="#444460" stroke="#555570" stroke-width="0.5"/>
      <line x1="405" y1="363" x2="405" y2="381" stroke="#555570" stroke-width="0.8" opacity="0.5"/>
      <line x1="396" y1="372" x2="414" y2="372" stroke="#555570" stroke-width="0.8" opacity="0.5"/>
      <!-- Front right wheel (behind) -->
      <circle cx="390" cy="374" r="11" fill="#2a2a3e" stroke="#222238" stroke-width="1.5" opacity="0.6"/>
    </g>
    <!-- ===== ELEPHANT ===== -->
    <g filter="url(#shadowFilter)">
      <!-- Back left leg (behind body) -->
      <path d="M410,295 L405,340 Q403,348 410,348 L425,348 Q432,348 430,340 L425,295 Z"
            fill="#6e6e80" stroke="#5a5a6a" stroke-width="1"/>
      <!-- Front left leg (behind body) -->
      <path d="M240,290 L235,340 Q233,348 240,348 L255,348 Q262,348 260,340 L255,290 Z"
            fill="#6e6e80" stroke="#5a5a6a" stroke-width="1"/>
      <!-- Tail -->
      <g class="tail-group">
        <path d="M440,255 Q465,240 470,260 Q475,280 465,290 Q460,295 458,288 Q462,275 455,265 Q450,260 440,262"
              fill="none" stroke="#787888" stroke-width="4" stroke-linecap="round"/>
        <!-- Tail tuft -->
        <ellipse cx="467" cy="293" rx="5" ry="4" fill="#5a5a6a"/>
      </g>
      <!-- Body -->
      <ellipse cx="340" cy="270" rx="115" ry="65" fill="url(#elephantGrad)"/>
      <!-- Belly highlight -->
      <ellipse cx="340" cy="285" rx="90" ry="35" fill="url(#bellyGrad)" opacity="0.5"/>
      <!-- Back right leg -->
      <path d="M395,295 L390,340 Q388,348 395,348 L410,348 Q417,348 415,340 L410,295 Z"
            fill="url(#elephantGrad)" stroke="#5a5a6a" stroke-width="1"/>
      <!-- Foot detail -->
      <ellipse cx="403" cy="348" rx="12" ry="4" fill="#6a6a78"/>
      <!-- Toenails -->
      <circle cx="393" cy="347" r="2" fill="#c0c0cc" opacity="0.6"/>
      <circle cx="399" cy="346" r="2" fill="#c0c0cc" opacity="0.6"/>
      <circle cx="405" cy="346" r="2" fill="#c0c0cc" opacity="0.6"/>
      <!-- Front right leg -->
      <path d="M230,285 L225,340 Q223,348 230,348 L245,348 Q252,348 250,340 L245,290 Z"
            fill="url(#elephantGrad)" stroke="#5a5a6a" stroke-width="1"/>
      <!-- Foot detail -->
      <ellipse cx="238" cy="348" rx="12" ry="4" fill="#6a6a78"/>
      <!-- Toenails -->
      <circle cx="228" cy="347" r="2" fill="#c0c0cc" opacity="0.6"/>
      <circle cx="234" cy="346" r="2" fill="#c0c0cc" opacity="0.6"/>
      <circle cx="240" cy="346" r="2" fill="#c0c0cc" opacity="0.6"/>
      <!-- Head -->
      <circle cx="230" cy="210" r="60" fill="url(#elephantGrad)"/>
      <!-- Head highlight -->
      <circle cx="220" cy="195" r="40" fill="#b0b0c0" opacity="0.3"/>
      <!-- Ear (behind head) -->
      <g class="ear-group">
        <path d="M270,175 Q310,140 320,180 Q330,220 300,245 Q280,260 270,240 Z"
              fill="#9898a8" stroke="#7a7a8a" stroke-width="1.5"/>
        <!-- Inner ear -->
        <path d="M280,185 Q300,165 305,190 Q310,215 295,230 Z"
              fill="#d4a0a0" opacity="0.6"/>
      </g>
      <!-- Ear (front, visible) -->
      <path d="M200,175 Q160,130 145,175 Q130,220 170,245 Q190,255 200,235 Z"
            fill="#a0a0b0" stroke="#7a7a8a" stroke-width="1.5"/>
      <!-- Inner ear detail -->
      <path d="M192,188 Q168,158 162,185 Q156,215 180,232 Z"
            fill="#d4a0a0" opacity="0.6"/>
      <!-- Ear veins -->
      <path d="M180,195 Q170,210 175,225" fill="none" stroke="#c09090" stroke-width="0.8" opacity="0.4"/>
      <path d="M188,192 Q182,205 185,220" fill="none" stroke="#c09090" stroke-width="0.6" opacity="0.3"/>
      <!-- Trunk -->
      <g class="trunk-group">
        <path d="M210,240 Q195,260 185,285 Q175,310 180,325 Q185,340 195,340 Q205,340 210,330 Q215,320 210,310 Q205,305 200,310"
              fill="none" stroke="url(#elephantGrad)" stroke-width="18" stroke-linecap="round"/>
        <path d="M210,240 Q195,260 185,285 Q175,310 180,325 Q185,340 195,340 Q205,340 210,330 Q215,320 210,310 Q205,305 200,310"
              fill="none" stroke="#9090a0" stroke-width="12" stroke-linecap="round" opacity="0.4"/>
        <!-- Trunk tip curl -->
        <circle cx="200" cy="315" r="9" fill="#a0a0b0"/>
        <!-- Trunk tip nostril -->
        <ellipse cx="195" cy="318" rx="3" ry="2" fill="#7a7a8a" opacity="0.5"/>
      </g>
      <!-- Eye -->
      <ellipse cx="205" cy="195" rx="12" ry="13" fill="#ffffff"/>
      <circle cx="207" cy="194" r="8" fill="#2a2a3e"/>
      <circle cx="210" cy="191" r="3.5" fill="url(#eyeShine)"/>
      <!-- Eyelash/eyebrow -->
      <path d="M193,185 Q200,180 215,183" fill="none" stroke="#5a5a6a" stroke-width="2" stroke-linecap="round"/>
      <!-- Mouth/smile line -->
      <path d="M218,240 Q225,248 232,245" fill="none" stroke="#6a6a7a" stroke-width="1.5" stroke-linecap="round"/>
      <!-- Tusks -->
      <path d="M218,235 Q215,255 220,265" fill="none" stroke="#fffff0" stroke-width="4" stroke-linecap="round"/>
      <path d="M218,235 Q215,255 220,265" fill="none" stroke="#eeeedd" stroke-width="2" stroke-linecap="round" opacity="0.5"/>
      <!-- Wrinkles on body -->
      <path d="M290,250 Q310,245 330,250" fill="none" stroke="#7a7a8a" stroke-width="1" opacity="0.4"/>
      <path d="M280,265 Q310,260 340,265" fill="none" stroke="#7a7a8a" stroke-width="1" opacity="0.3"/>
      <path d="M370,255 Q390,250 410,255" fill="none" stroke="#7a7a8a" stroke-width="1" opacity="0.3"/>
      <path d="M250,230 Q260,225 275,228" fill="none" stroke="#7a7a8a" stroke-width="0.8" opacity="0.3"/>
      <!-- Head wrinkle -->
      <path d="M190,170 Q210,165 230,170" fill="none" stroke="#7a7a8a" stroke-width="1" opacity="0.3"/>
    </g>
    <!-- Dust/spark particles from wheels -->
    <g>
      <circle class="particle" cx="180" cy="370" r="3" fill="#e94560" opacity="0.6"/>
      <circle class="particle" cx="170" cy="375" r="2" fill="#ff6b6b" opacity="0.5"/>
      <circle class="particle" cx="190" cy="365" r="2.5" fill="#ffa07a" opacity="0.4"/>
      <circle class="particle" cx="175" cy="380" r="1.5" fill="#e94560" opacity="0.5"/>
    </g>
    <!-- Stars/sparkles around elephant -->
    <g filter="url(#glow)">
      <polygon points="150,150 153,158 161,158 155,163 157,171 150,167 143,171 145,163 139,158 147,158" fill="#e94560" opacity="0.6">
        <animate attributeName="opacity" values="0.6;0.2;0.6" dur="2s" repeatCount="indefinite"/>
      </polygon>
      <polygon points="480,120 482,126 488,126 483,130 485,136 480,132 475,136 477,130 472,126 478,126" fill="#ff6b6b" opacity="0.5">
        <animate attributeName="opacity" values="0.5;0.1;0.5" dur="1.8s" repeatCount="indefinite"/>
      </polygon>
      <polygon points="500,280 502,286 508,286 503,290 505,296 500,292 495,296 497,290 492,286 498,286" fill="#ffa07a" opacity="0.4">
        <animate attributeName="opacity" values="0.4;0.1;0.4" dur="1.5s" repeatCount="indefinite"/>
      </polygon>
    </g>
  </svg>
```

</details>

## 6. `glm-5.2:cloud`

**SVG size:** 5364 characters  
**Complexity:** Very high  
**Shape elements:** 60  
**Distinct colors:** 18  
**Raw response:** 5394 characters  
**Generation time:** 53.3s

![glm-5.2:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/glm-5.2_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the elephant body and skateboard texture)
- Uses transforms (translate, rotate, scale) for positioning the elephant and tilting the skateboard
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders tusks -- the model captured fine anatomical detail
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Renders trucks (the axle assemblies) -- the model captured fine skateboard mechanical detail
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface
- Sky/atmosphere is present -- the model created an outdoor scene

With 5,364 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <defs>
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0B002A"/>
      <stop offset="50%" stop-color="#2E0E4A"/>
      <stop offset="100%" stop-color="#5D1A7A"/>
    </linearGradient>
    <linearGradient id="sunGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#FFD700"/>
      <stop offset="30%" stop-color="#FF8000"/>
      <stop offset="60%" stop-color="#FF00FF"/>
      <stop offset="100%" stop-color="#9D00FF"/>
    </linearGradient>
  </defs>
  <!-- Background Sky -->
  <rect width="800" height="600" fill="url(#skyGrad)"/>
  <!-- Stars -->
  <g fill="#FFFFFF">
    <circle cx="100" cy="100" r="2"/>
    <circle cx="200" cy="50" r="1.5"/>
    <circle cx="700" cy="150" r="2.5"/>
    <circle cx="650" cy="80" r="1.5"/>
    <circle cx="150" cy="200" r="2"/>
    <circle cx="550" cy="250" r="1"/>
    <circle cx="300" cy="150" r="1.5"/>
    <circle cx="750" cy="250" r="2"/>
  </g>
  <!-- Setting Sun -->
  <circle cx="400" cy="400" r="150" fill="url(#sunGrad)"/>
  <!-- Sun Cutout Bands -->
  <rect x="250" y="340" width="300" height="6" fill="#100815"/>
  <rect x="250" y="360" width="300" height="8" fill="#100815"/>
  <rect x="250" y="380" width="300" height="12" fill="#100815"/>
  <rect x="250" y="395" width="300" height="20" fill="#100815"/>
  <!-- Ground -->
  <rect x="0" y="400" width="800" height="200" fill="#100815"/>
  <!-- Vaporwave Grid -->
  <g stroke="#FF00FF" stroke-width="2" opacity="0.6">
    <line x1="0" y1="600" x2="400" y2="400"/>
    <line x1="100" y1="600" x2="400" y2="400"/>
    <line x1="200" y1="600" x2="400" y2="400"/>
    <line x1="300" y1="600" x2="400" y2="400"/>
    <line x1="500" y1="600" x2="400" y2="400"/>
    <line x1="600" y1="600" x2="400" y2="400"/>
    <line x1="700" y1="600" x2="400" y2="400"/>
    <line x1="800" y1="600" x2="400" y2="400"/>
  </g>
  <g stroke="#00CED1" stroke-width="2" opacity="0.8">
    <line x1="0" y1="440" x2="800" y2="440"/>
    <line x1="0" y1="488" x2="800" y2="488"/>
    <line x1="0" y1="544" x2="800" y2="544"/>
    <line x1="0" y1="600" x2="800" y2="600"/>
  </g>
  <!-- Motion Lines -->
  <g stroke-linecap="round" opacity="0.7">
    <line x1="250" y1="300" x2="300" y2="300" stroke="#FFFFFF" stroke-width="4"/>
    <line x1="220" y1="350" x2="290" y2="350" stroke="#FFFFFF" stroke-width="4"/>
    <line x1="200" y1="250" x2="280" y2="250" stroke="#00CED1" stroke-width="4"/>
  </g>
  <!-- Board Shadow -->
  <ellipse cx="400" cy="415" rx="130" ry="14" fill="#000000" opacity="0.5"/>
  <!-- Skateboard -->
  <g transform="translate(400, 410) rotate(-12)">
    <!-- Trucks -->
    <rect x="-80" y="10" width="40" height="10" rx="2" fill="#7F8C8D"/>
    <rect x="40" y="10" width="40" height="10" rx="2" fill="#7F8C8D"/>
    <!-- Wheels -->
    <circle cx="-60" cy="25" r="14" fill="#111111" stroke="#F4D03F" stroke-width="4"/>
    <circle cx="60" cy="25" r="14" fill="#111111" stroke="#F4D03F" stroke-width="4"/>
    <!-- Deck -->
    <path d="M -100 10 Q -110 5 -100 0 L 100 0 Q 110 5 100 10 Z" fill="#111111" stroke="#00CED1" stroke-width="4"/>
  </g>
  <!-- Elephant Group -->
  <g transform="translate(400, 300)">
    <!-- Tail -->
    <path d="M -65 -10 Q -90 20 -80 50 L -70 50 Q -85 20 -65 -5 Z" fill="#6B7287"/>
    <path d="M -80 50 L -90 65 L -70 60 Z" fill="#9BA4B5"/>
    <!-- Back Legs -->
    <rect x="-50" y="35" width="30" height="50" rx="12" fill="#6B7287"/>
    <rect x="15" y="35" width="30" height="50" rx="12" fill="#7E8B98"/>
    <line x1="-40" y1="82" x2="-30" y2="82" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
    <line x1="25" y1="82" x2="35" y2="82" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
    <!-- Body -->
    <ellipse cx="0" cy="0" rx="70" ry="55" fill="#9BA4B5"/>
    <ellipse cx="0" cy="15" rx="65" ry="30" fill="#B0BAC7"/>
    <!-- Front Legs -->
    <path d="M -45 20 Q -60 50 -30 75 L -10 75 Q -20 50 -15 20 Z" fill="#9BA4B5"/>
    <path d="M -5 20 Q -15 50 5 70 L 25 70 Q 15 50 25 20 Z" fill="#818B9C"/>
    <line x1="-35" y1="72" x2="-25" y2="72" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
    <line x1="5" y1="67" x2="15" y2="67" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round"/>
    <!-- Left Ear -->
    <path d="M 20 -40 C -10 -90 -50 -60 -35 -20 C -20 10 10 0 20 -40 Z" fill="#818B9C"/>
    <path d="M 10 -35 C -10 -65 -35 -45 -25 -20 C -10 0 5 0 10 -35 Z" fill="#A0AEB8"/>
    <!-- Head -->
    <ellipse cx="40" cy="-40" rx="55" ry="50" fill="#9BA4B5"/>
    <!-- Trunk -->
    <path d="M 90 -10 C 140 10 110 50 90 60 C 70 70 60 40 80 30 C 90 20 80 10 90 -10 Z" fill="#9BA4B5"/>
    <path d="M 85 10 C 100 20 85 40 80 45" stroke="#6B7287" stroke-width="3" fill="none"/>
    <ellipse cx="88" cy="60" rx="8" ry="4" fill="#6B7287"/>
    <!-- Tusk -->
    <path d="M 55 5 Q 75 25 60 30 Q 70 15 55 5 Z" fill="#FFDFBA"/>
    <!-- Eye -->
    <circle cx="70" cy="-50" r="7" fill="#1A1126"/>
    <circle cx="73" cy="-53" r="2" fill="#FFFFFF"/>
    <!-- Backward Cap -->
    <path d="M 0 -70 Q 30 -110 80 -70 Q 80 -55 0 -55 Z" fill="#FF00FF"/>
    <path d="M 80 -70 C 130 -70 130 -45 75 -45 L 80 -70 Z" fill="#D600D6"/>
    <polygon points="40,-95 45,-80 55,-80 45,-70 55,-60 35,-75 45,-75" fill="#FFFFFF"/>
  </g>
</svg>
```

</details>

## 7. `gpt-oss:120b-cloud`

**SVG size:** 1659 characters  
**Complexity:** Medium  
**Shape elements:** 20  
**Distinct colors:** 7  
**Raw response:** 2359 characters  
**Generation time:** 9.1s

![gpt-oss:120b-cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/gpt-oss_120b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface

With 1,659 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 200 150"
    style="background:#f0f8ff" width="100%" height="auto">
  <!-- ==== Skateboard ==== -->
  <!-- board -->
  <rect x="70" y="110" width="80" height="8" rx="2" fill="#555" />
  <!-- wheels -->
  <circle cx="85"  cy="124" r="6" fill="#111" />
  <circle cx="115" cy="124" r="6" fill="#111" />
  <circle cx="145" cy="124" r="6" fill="#111" />
  <circle cx="155" cy="124" r="6" fill="#111" />
  <!-- ==== Elephant ==== -->
  <!-- body -->
  <ellipse cx="100" cy="70" rx="45" ry="30" fill="#777" />
  <!-- head -->
  <circle cx="100" cy="35" r="20" fill="#777" />
  <!-- ears -->
  <ellipse cx="75"  cy="35" rx="12" ry="20" fill="#777" />
  <ellipse cx="125" cy="35" rx="12" ry="20" fill="#777" />
  <!-- eyes -->
  <circle cx="92" cy="30" r="2.5" fill="#fff" />
  <circle cx="108" cy="30" r="2.5" fill="#fff" />
  <!-- pupils -->
  <circle cx="92" cy="30" r="1" fill="#000" />
  <circle cx="108" cy="30" r="1" fill="#000" />
  <!-- trunk -->
  <path d="M100 45 
           q-5 10 -5 20 
           q0 10 5 20 
           q5 -10 5 -20 
           q0 -10 -5 -20"
        fill="#777" />
  <!-- front legs (holding board) -->
  <rect x="80"  y="80" width="10" height="20" fill="#777" />
  <rect x="110" y="80" width="10" height="20" fill="#777" />
  <!-- back legs -->
  <rect x="70"  y="95" width="10" height="20" fill="#777" />
  <rect x="120" y="95" width="10" height="20" fill="#777" />
  <!-- tail -->
  <path d="M138 70 q5 5 2 12" stroke="#777" stroke-width="3" fill="none" />
  <!-- Optional ground line -->
  <line x1="0" y1="130" x2="200" y2="130" stroke="#bbb" stroke-width="1" />
</svg>
```

</details>

## 8. `kimi-k2.6:cloud`

**SVG size:** 7511 characters  
**Complexity:** High  
**Shape elements:** 30  
**Distinct colors:** 15  
**Raw response:** 22175 characters  
**Generation time:** 49.9s

![kimi-k2.6:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/kimi-k2.6_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the elephant body and skateboard texture)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the elephant and tilting the skateboard
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface
- Sky/atmosphere is present -- the model created an outdoor scene

With 7,511 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" id="main-svg" viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid slice" width="100%" height="auto">
            <defs>
                <!-- Gradients -->
                <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#4facfe;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#00f2fe;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="sunsetGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#ff9966;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#ff5e62;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="elephantBody" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#95a5a6;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#7f8c8d;stop-opacity:1" />
                </linearGradient>
                <filter id="shadowBlur">
                    <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
                    <feOffset dx="2" dy="4" result="offsetblur" />
                    <feComponentTransfer>
                        <feFuncA type="linear" slope="0.3" />
                    </feComponentTransfer>
                    <feMerge>
                        <feMergeNode />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
            </defs>
            <!-- Background Elements -->
            <g id="bg-layer">
                <rect id="sky-rect" x="-500" y="-500" width="3000" height="2000" fill="url(#skyGradient)" />
                <!-- Sun/Moon -->
                <circle id="sun" cx="850" cy="100" r="60" fill="#FFD700" opacity="0.9" />
                <!-- Clouds (Parallax Layer 1) -->
                <g id="clouds-back" opacity="0.6" fill="#fff"></g>
                <!-- Mountains/City (Parallax Layer 2) -->
                <g id="mountains" transform="translate(0, 350)"></g>
                <!-- Road -->
                <rect x="-100" y="500" width="3000" height="200" fill="#34495e" />
                <g id="road-lines"></g>
            </g>
            <!-- The Character Group -->
            <g id="character-group" transform="translate(400, 380)">
                <!-- Skateboard Shadow -->
                <ellipse cx="0" cy="115" rx="140" ry="10" fill="#000" opacity="0.2" />
                <!-- Skateboard -->
                <g id="skateboard" transform="translate(0, 90)">
                    <!-- Wheels -->
                    <g id="wheel-back" transform="translate(-90, 15)">
                        <circle cx="0" cy="0" r="18" fill="#2c3e50" stroke="#ecf0f1" stroke-width="2"/>
                        <g class="wheel-spokes">
                            <line x1="-18" y1="0" x2="18" y2="0" stroke="#7f8c8d" stroke-width="2"/>
                            <line x1="0" y1="-18" x2="0" y2="18" stroke="#7f8c8d" stroke-width="2"/>
                        </g>
                    </g>
                    <g id="wheel-front" transform="translate(90, 15)">
                        <circle cx="0" cy="0" r="18" fill="#2c3e50" stroke="#ecf0f1" stroke-width="2"/>
                        <g class="wheel-spokes">
                            <line x1="-18" y1="0" x2="18" y2="0" stroke="#7f8c8d" stroke-width="2"/>
                            <line x1="0" y1="-18" x2="0" y2="18" stroke="#7f8c8d" stroke-width="2"/>
                        </g>
                    </g>
                    <!-- Deck -->
                    <path d="M-120,0 Q-120,-15 -90,-15 L90,-15 Q120,-15 120,0 Q120,10 90,10 L-90,10 Q-120,10 -120,0" fill="#e74c3c" stroke="#c0392b" stroke-width="2"/>
                    <path d="M-120,0 Q-120,-15 -90,-15 L90,-15 Q120,-15 120,0 Q120,10 90,10 L-90,10 Q-120,10 -120,0" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="10,5" opacity="0.5" transform="translate(0, -5)"/>
                </g>
                <!-- Elephant -->
                <g id="elephant" transform="translate(0, -20)">
                    <!-- Back Leg (Left) -->
                    <g id="leg-back" transform="translate(-60, 40)">
                        <path d="M0,0 L0,50 L10,50 L15,0 Z" fill="#7f8c8d" stroke="#555" stroke-width="2" stroke-linejoin="round"/>
                        <rect x="0" y="45" width="12" height="10" rx="3" fill="#333"/> <!-- Shoe -->
                    </g>
                    <!-- Back Arm (Left) -->
                    <g id="arm-back" transform="translate(-50, -30)">
                         <path d="M0,0 Q-20,20 -10,40" fill="none" stroke="#7f8c8d" stroke-width="18" stroke-linecap="round"/>
                         <circle cx="-10" cy="40" r="10" fill="#95a5a6"/> <!-- Hand -->
                    </g>
                    <!-- Body -->
                    <ellipse cx="0" cy="0" rx="90" ry="80" fill="url(#elephantBody)" stroke="#555" stroke-width="2"/>
                    <path d="M-40,-20 Q0,40 40,-20" fill="none" stroke="#7f8c8d" stroke-width="3" opacity="0.5"/> <!-- Belly fold -->
                    <!-- Head -->
                    <g id="head-group" transform="translate(60, -60)">
                        <!-- Ears -->
                        <path d="M-10,10 Q-60,-20 -50,60 Q-20,80 -10,40" fill="#95a5a6" stroke="#555" stroke-width="2"/>
                        <!-- Head Shape -->
                        <circle cx="0" cy="0" r="50" fill="url(#elephantBody)" stroke="#555" stroke-width="2"/>
                        <!-- Eye -->
                        <g id="eye-group" transform="translate(15, -10)">
                            <circle cx="0" cy="0" r="8" fill="white"/>
                            <circle id="pupil" cx="2" cy="0" r="3" fill="black"/>
                            <path d="M-8,-8 Q0,-12 8,-8" fill="none" stroke="#555" stroke-width="2"/> <!-- Eyebrow -->
                        </g>
                        <!-- Trunk -->
                        <g id="trunk" transform="translate(30, 20)">
                            <path id="trunk-path" d="M0,0 Q20,30 10,60 T30,90" fill="none" stroke="#95a5a6" stroke-width="18" stroke-linecap="round"/>
                        </g>
                    </g>
                    <!-- Front Leg (Right) -->
                    <g id="leg-front" transform="translate(40, 50)">
                         <path d="M0,0 L0,40 L10,40 L15,0 Z" fill="#95a5a6" stroke="#555" stroke-width="2" stroke-linejoin="round"/>
                         <rect x="0" y="35" width="12" height="10" rx="3" fill="#e74c3c"/> <!-- Shoe -->
                         <!-- Sock -->
                         <rect x="2" y="25" width="11" height="12" fill="white"/>
                    </g>
                    <!-- Front Arm (Right) -->
                    <g id="arm-front" transform="translate(50, -20)">
                         <path d="M0,0 Q20,20 10,50" fill="none" stroke="#95a5a6" stroke-width="18" stroke-linecap="round"/>
                         <circle cx="10" cy="50" r="10" fill="#95a5a6"/> <!-- Hand -->
                         <!-- Wristband -->
                         <rect x="5" y="35" width="14" height="6" fill="#e74c3c" transform="rotate(20 12 38)"/>
                    </g>
                </g>
            </g>
            <!-- Foreground Particles (Dust/Stars) -->
            <g id="particles"></g>
        </svg>
```

</details>

## 9. `minimax-m2.7:cloud`

**SVG size:** 3010 characters  
**Complexity:** Medium  
**Shape elements:** 24  
**Distinct colors:** 8  
**Raw response:** 3282 characters  
**Generation time:** 37.1s

![minimax-m2.7:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/minimax-m2.7_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the elephant body and skateboard texture)
- Uses transforms (translate, rotate, scale) for positioning the elephant and tilting the skateboard
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders tusks -- the model captured fine anatomical detail
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Renders trucks (the axle assemblies) -- the model captured fine skateboard mechanical detail
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding

With 3,010 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 400" width="100%" height="auto">
  <defs>
    <linearGradient id="deckGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#d35400;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#e67e22;stop-opacity:1" />
    </linearGradient>
  </defs>
  <!-- Motion Lines -->
  <g stroke="#bdc3c7" stroke-width="4" stroke-linecap="round">
    <line x1="50" y1="330" x2="120" y2="330" />
    <line x1="20" y1="310" x2="80" y2="310" />
    <line x1="60" y1="350" x2="110" y2="350" />
  </g>
  <!-- Skateboard -->
  <g transform="translate(0, 20)">
    <!-- Wheels (Back) -->
    <circle cx="180" cy="330" r="15" fill="#2c3e50" />
    <circle cx="320" cy="330" r="15" fill="#2c3e50" />
    <!-- Trucks -->
    <rect x="170" y="310" width="20" height="10" fill="#7f8c8d" />
    <rect x="310" y="310" width="20" height="10" fill="#7f8c8d" />
    <!-- Deck -->
    <!-- Using a path for a slightly curved board shape -->
    <path d="M130,310 Q250,300 370,310 L365,300 Q250,290 135,300 Z" fill="#34495e" /> <!-- Bottom of deck -->
    <path d="M135,300 Q250,290 365,300 L365,280 Q250,270 135,280 Z" fill="url(#deckGradient)" /> <!-- Grip tape top -->
  </g>
  <!-- Elephant -->
  <g transform="translate(0, 20)">
    <!-- Tail -->
    <path d="M130,230 Q110,230 110,250 T105,270" fill="none" stroke="#95a5a6" stroke-width="4" stroke-linecap="round"/>
    <circle cx="105" cy="270" r="3" fill="#7f8c8d"/>
    <!-- Back Legs (Darker for depth) -->
    <path d="M150,280 L150,310 Q150,320 140,320 L130,320" fill="none" stroke="#7f8c8d" stroke-width="25" stroke-linecap="round"/>
    <path d="M270,280 L270,310 Q270,320 260,320 L250,320" fill="none" stroke="#7f8c8d" stroke-width="25" stroke-linecap="round"/>
    <!-- Front Legs (Main color) -->
    <path d="M180,280 L180,310 Q180,320 170,320 L160,320" fill="none" stroke="#95a5a6" stroke-width="25" stroke-linecap="round"/>
    <path d="M300,280 L300,310 Q300,320 290,320 L280,320" fill="none" stroke="#95a5a6" stroke-width="25" stroke-linecap="round"/>
    <!-- Body -->
    <ellipse cx="230" cy="240" rx="90" ry="70" fill="#95a5a6" />
    <!-- Head -->
    <circle cx="300" cy="190" r="55" fill="#95a5a6" />
    <!-- Trunk -->
    <path d="M330,200 Q360,200 360,230 Q360,260 340,250" fill="none" stroke="#95a5a6" stroke-width="18" stroke-linecap="round" />
    <!-- Tusk -->
    <path d="M315,220 Q310,250 290,245" fill="none" stroke="#ecf0f1" stroke-width="6" stroke-linecap="round" />
    <!-- Ear -->
    <path d="M270,170 Q220,150 220,200 Q220,250 280,220 Z" fill="#bdc3c7" stroke="#95a5a6" stroke-width="2"/>
    <path d="M270,170 Q230,160 230,200 Q230,230 270,215" fill="#95a5a6" opacity="0.3"/>
    <!-- Eye -->
    <circle cx="310" cy="180" r="4" fill="#2c3e50" />
    <circle cx="312" cy="178" r="1.5" fill="#ffffff" />
    <!-- Mouth -->
    <path d="M320,220 Q330,225 325,235" fill="none" stroke="#2c3e50" stroke-width="2" stroke-linecap="round" />
  </g>
</svg>
```

</details>

## 10. `minimax-m3:cloud`

**SVG size:** 5627 characters  
**Complexity:** Very high  
**Shape elements:** 63  
**Distinct colors:** 23  
**Raw response:** 6239 characters  
**Generation time:** 99.6s

![minimax-m3:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/minimax-m3_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Renders actual text (labels, title, or decorative text)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders tusks -- the model captured fine anatomical detail
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Renders trucks (the axle assemblies) -- the model captured fine skateboard mechanical detail
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface
- Sky/atmosphere is present -- the model created an outdoor scene

With 5,627 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="auto">
  <!-- Background sky -->
  <rect width="500" height="500" fill="#87CEEB"/>
  <!-- Clouds -->
  <g opacity="0.9">
    <ellipse cx="80" cy="80" rx="30" ry="15" fill="white"/>
    <ellipse cx="105" cy="75" rx="25" ry="13" fill="white"/>
    <ellipse cx="60" cy="85" rx="20" ry="10" fill="white"/>
  </g>
  <g opacity="0.9">
    <ellipse cx="420" cy="120" rx="35" ry="18" fill="white"/>
    <ellipse cx="450" cy="115" rx="28" ry="14" fill="white"/>
    <ellipse cx="395" cy="125" rx="22" ry="11" fill="white"/>
  </g>
  <!-- Sun -->
  <circle cx="60" cy="60" r="30" fill="#FFD700" opacity="0.3"/>
  <circle cx="60" cy="60" r="25" fill="#FFD700"/>
  <!-- Ground -->
  <rect y="400" width="500" height="100" fill="#7CCD7C"/>
  <rect y="400" width="500" height="10" fill="#5BAF5B"/>
  <!-- Motion lines -->
  <g stroke="#fff" stroke-width="3" stroke-linecap="round" opacity="0.7">
    <line x1="30" y1="280" x2="80" y2="280"/>
    <line x1="20" y1="305" x2="70" y2="305"/>
    <line x1="40" y1="330" x2="85" y2="330"/>
  </g>
  <!-- Skateboard shadow -->
  <ellipse cx="250" cy="408" rx="180" ry="6" fill="#000" opacity="0.2"/>
  <!-- Skateboard deck -->
  <path d="M 95 348 Q 95 338 105 338 L 115 338 Q 115 332 122 332 L 378 332 Q 385 332 385 338 L 395 338 Q 405 338 405 348 L 405 360 Q 405 368 395 368 L 105 368 Q 95 368 95 360 Z" 
        fill="#D2691E" stroke="#8B4513" stroke-width="2.5"/>
  <!-- Deck design -->
  <path d="M 200 340 L 300 340 L 305 360 L 195 360 Z" fill="#FF6347" opacity="0.8"/>
  <text x="250" y="355" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="11" font-weight="bold">SK8</text>
  <!-- Trucks -->
  <rect x="138" y="368" width="52" height="10" fill="#444" rx="2"/>
  <rect x="310" y="368" width="52" height="10" fill="#444" rx="2"/>
  <!-- Axles -->
  <line x1="138" y1="375" x2="190" y2="375" stroke="#222" stroke-width="3"/>
  <line x1="310" y1="375" x2="362" y2="375" stroke="#222" stroke-width="3"/>
  <!-- Wheels -->
  <g>
    <circle cx="150" cy="385" r="11" fill="#222"/>
    <circle cx="150" cy="385" r="4" fill="#888"/>
    <circle cx="178" cy="385" r="11" fill="#222"/>
    <circle cx="178" cy="385" r="4" fill="#888"/>
  </g>
  <g>
    <circle cx="322" cy="385" r="11" fill="#222"/>
    <circle cx="322" cy="385" r="4" fill="#888"/>
    <circle cx="350" cy="385" r="11" fill="#222"/>
    <circle cx="350" cy="385" r="4" fill="#888"/>
  </g>
  <!-- ELEPHANT -->
  <!-- Back legs (behind body) -->
  <rect x="170" y="280" width="24" height="60" fill="#9A9A9A" stroke="#555" stroke-width="2" rx="4"/>
  <rect x="215" y="280" width="24" height="60" fill="#B0B0B0" stroke="#666" stroke-width="2" rx="4"/>
  <!-- Toenails for back legs -->
  <g fill="#444">
    <ellipse cx="177" cy="338" rx="3" ry="2"/>
    <ellipse cx="182" cy="338" rx="3" ry="2"/>
    <ellipse cx="187" cy="338" rx="3" ry="2"/>
    <ellipse cx="222" cy="338" rx="3" ry="2"/>
    <ellipse cx="227" cy="338" rx="3" ry="2"/>
    <ellipse cx="232" cy="338" rx="3" ry="2"/>
  </g>
  <!-- Body -->
  <ellipse cx="250" cy="240" rx="125" ry="62" fill="#B8B8B8" stroke="#666" stroke-width="2.5"/>
  <!-- Front legs (in front of body) -->
  <rect x="285" y="280" width="24" height="60" fill="#B0B0B0" stroke="#666" stroke-width="2" rx="4"/>
  <rect x="330" y="280" width="24" height="60" fill="#9A9A9A" stroke="#555" stroke-width="2" rx="4"/>
  <!-- Toenails for front legs -->
  <g fill="#444">
    <ellipse cx="292" cy="338" rx="3" ry="2"/>
    <ellipse cx="297" cy="338" rx="3" ry="2"/>
    <ellipse cx="302" cy="338" rx="3" ry="2"/>
    <ellipse cx="337" cy="338" rx="3" ry="2"/>
    <ellipse cx="342" cy="338" rx="3" ry="2"/>
    <ellipse cx="347" cy="338" rx="3" ry="2"/>
  </g>
  <!-- Tail -->
  <path d="M 130 230 Q 110 235 105 255 Q 103 270 115 275" stroke="#666" stroke-width="3" fill="none" stroke-linecap="round"/>
  <circle cx="115" cy="275" r="5" fill="#444"/>
  <!-- Head -->
  <circle cx="360" cy="220" r="50" fill="#B8B8B8" stroke="#666" stroke-width="2.5"/>
  <!-- Ear -->
  <path d="M 330 178 Q 295 168 288 205 Q 288 240 320 245 Q 350 240 350 210 Q 350 182 330 178 Z" 
        fill="#999" stroke="#666" stroke-width="2.5"/>
  <!-- Inner ear -->
  <path d="M 320 200 Q 305 205 305 220 Q 318 232 328 228 Q 333 215 328 200 Z" fill="#D4A0A0"/>
  <!-- Trunk -->
  <path d="M 390 240 Q 425 252 430 285 Q 430 318 408 328 Q 392 332 388 320" 
        stroke="#999" stroke-width="18" fill="none" stroke-linecap="round"/>
  <path d="M 390 240 Q 425 252 430 285 Q 430 318 408 328 Q 392 332 388 320" 
        stroke="#666" stroke-width="2" fill="none" stroke-linecap="round"/>
  <!-- Trunk wrinkles -->
  <g stroke="#666" stroke-width="1.5" fill="none" stroke-linecap="round">
    <path d="M 405 250 Q 410 255 415 260"/>
    <path d="M 420 270 Q 425 278 426 286"/>
    <path d="M 422 302 Q 418 315 408 325"/>
  </g>
  <!-- Eye -->
  <circle cx="365" cy="205" r="6" fill="#000"/>
  <circle cx="367" cy="203" r="2" fill="#fff"/>
  <!-- Eyebrow -->
  <path d="M 354 192 Q 365 187 376 193" stroke="#444" stroke-width="2.5" fill="none" stroke-linecap="round"/>
  <!-- Tusk -->
  <path d="M 380 250 Q 386 265 392 278" stroke="#fff" stroke-width="6" fill="none" stroke-linecap="round"/>
  <path d="M 381 252 Q 386 263 391 274" stroke="#ddd" stroke-width="1.5" fill="none" stroke-linecap="round"/>
  <!-- Smile -->
  <path d="M 358 252 Q 366 256 374 252" stroke="#444" stroke-width="2" fill="none" stroke-linecap="round"/>
  <!-- Blush -->
  <ellipse cx="338" cy="228" rx="8" ry="5" fill="#FFB6C1" opacity="0.6"/>
</svg>
```

</details>

## 11. `nemotron-3-super:cloud`

**SVG size:** 1995 characters  
**Complexity:** Medium  
**Shape elements:** 22  
**Distinct colors:** 7  
**Raw response:** 3705 characters  
**Generation time:** 24.9s

![nemotron-3-super:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/nemotron-3-super_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding

With 1,995 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Skateboard -->
  <rect x="60" y="140" width="80" height="10" rx="2" fill="#8B4513"/>
  <circle cx="80" cy="155" r="6" fill="#5D4037"/>
  <circle cx="120" cy="155" r="6" fill="#5D4037"/>
  <circle cx="80" cy="155" r="3" fill="#FFFFFF"/>
  <circle cx="120" cy="155" r="3" fill="#FFFFFF"/>
  <!-- Elephant Body -->
  <ellipse cx="100" cy="110" rx="40" ry="30" fill="#D7CCC8"/>
  <!-- Elephant Head -->
  <circle cx="100" cy="80" r="25" fill="#D7CCC8"/>
  <!-- Elephant Ears -->
  <path d="M75,65 C60,50 60,100 75,85" fill="#D7CCC8"/>
  <path d="M125,65 C140,50 140,100 125,85" fill="#D7CCC8"/>
  <!-- Elephant Trunk (holding a peace sign!) -->
  <path d="M100,85 
           q10,-15 20,-5 
           q5,5 10,0 
           q5,-5 10,0 
           q10,5 15,-5 
           q5,-5 10,0" 
        stroke="#D7CCC8" 
        stroke-width="4" 
        fill="none"
        stroke-linecap="round"/>
  <circle cx="130" cy="65" r="3" fill="#D7CCC8"/> <!-- Peace sign tip -->
  <circle cx="140" cy="55" r="3" fill="#D7CCC8"/> <!-- Peace sign tip -->
  <!-- Elephant Eyes -->
  <circle cx="90" cy="75" r="3" fill="#000000"/>
  <circle cx="110" cy="75" r="3" fill="#000000"/>
  <circle cx="92" cy="73" r="1" fill="#FFFFFF"/>
  <circle cx="112" cy="73" r="1" fill="#FFFFFF"/>
  <!-- Elephant Legs (suggesting balance) -->
  <rect x="75" y="130" width="10" height="20" fill="#D7CCC8"/>
  <rect x="115" y="130" width="10" height="20" fill="#D7CCC8"/>
  <!-- Tail -->
  <path d="M60,110 
           q-10,5 -15,10 
           q-5,5 -5,15" 
        stroke="#D7CCC8" 
        stroke-width="2" 
        fill="none"
        stroke-linecap="round"/>
  <!-- Motion lines (fun touch!) -->
  <line x1="50" y1="150" x2="30" y2="155" stroke="#CCCCCC" stroke-width="1"/>
  <line x1="50" y1="152" x2="30" y2="157" stroke="#CCCCCC" stroke-width="1"/>
  <line x1="50" y1="154" x2="30" y2="159" stroke="#CCCCCC" stroke-width="1"/>
</svg>
```

</details>

## 12. `nemotron-3-ultra:cloud`

**SVG size:** 7239 characters  
**Complexity:** High  
**Shape elements:** 39  
**Distinct colors:** 14  
**Raw response:** 8408 characters  
**Generation time:** 34.6s

![nemotron-3-ultra:cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/nemotron-3-ultra_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the elephant body and skateboard texture)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the elephant and tilting the skateboard
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders tusks -- the model captured fine anatomical detail
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Renders trucks (the axle assemblies) -- the model captured fine skateboard mechanical detail
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface

With 7,239 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Definitions for Gradients -->
  <defs>
    <!-- Elephant Body Gradient -->
    <linearGradient id="elephantGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#6D8A9E;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#4A6A7E;stop-opacity:1" />
    </linearGradient>
    <!-- Ear Gradient -->
    <linearGradient id="earGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#8AA5B8;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#5D7D91;stop-opacity:1" />
    </linearGradient>
    <!-- Skateboard Deck Gradient -->
    <linearGradient id="deckGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#E8C58A;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#D4A86A;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#C09050;stop-opacity:1" />
    </linearGradient>
    <!-- Wheel Gradient -->
    <radialGradient id="wheelGrad" cx="50%" cy="50%" r="50%" fx="35%" fy="35%">
      <stop offset="0%" style="stop-color:#FFFFFF;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#333333;stop-opacity:1" />
    </radialGradient>
    <!-- Truck Gradient -->
    <linearGradient id="truckGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#CCCCCC;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#999999;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#CCCCCC;stop-opacity:1" />
    </linearGradient>
    <!-- Shadow Filter -->
    <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
      <feOffset dx="2" dy="4" result="offsetblur"/>
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.2"/>
      </feComponentTransfer>
      <feMerge> 
        <feMergeNode in="offsetblur"/>
        <feMergeNode in="SourceGraphic"/> 
      </feMerge>
    </filter>
  </defs>
  <!-- Main Group with Shadow -->
  <g filter="url(#dropShadow)">
    <!-- SKATEBOARD (Drawn first so elephant overlaps) -->
    <g id="skateboard">
      <!-- Deck -->
      <path d="M60,210 
               C60,200 80,195 120,195 
               C280,195 300,200 300,210 
               C300,220 280,225 120,225 
               C80,225 60,220 60,210 Z" 
            fill="url(#deckGrad)" stroke="#8B6B42" stroke-width="1.5"/>
      <!-- Grip Tape Texture (Subtle lines) -->
      <g clip-path="url(#deckClip)">
        <defs>
          <clipPath id="deckClip">
            <path d="M60,210 
                     C60,200 80,195 120,195 
                     C280,195 300,200 300,210 
                     C300,220 280,225 120,225 
                     C80,225 60,220 60,210 Z"/>
          </clipPath>
        </defs>
        <rect x="55" y="190" width="250" height="40" fill="#2C2C2C" opacity="0.15"/>
      </g>
      <!-- Trucks (Metal axles) -->
      <!-- Front Truck -->
      <rect x="105" y="212" width="30" height="6" rx="2" fill="url(#truckGrad)" stroke="#666" stroke-width="0.5"/>
      <!-- Back Truck -->
      <rect x="265" y="212" width="30" height="6" rx="2" fill="url(#truckGrad)" stroke="#666" stroke-width="0.5"/>
      <!-- Wheels -->
      <!-- Front Left -->
      <circle cx="100" cy="228" r="14" fill="url(#wheelGrad)" stroke="#111" stroke-width="1.5"/>
      <circle cx="100" cy="228" r="6" fill="#222"/>
      <!-- Front Right -->
      <circle cx="135" cy="228" r="14" fill="url(#wheelGrad)" stroke="#111" stroke-width="1.5"/>
      <circle cx="135" cy="228" r="6" fill="#222"/>
      <!-- Back Left -->
      <circle cx="265" cy="228" r="14" fill="url(#wheelGrad)" stroke="#111" stroke-width="1.5"/>
      <circle cx="265" cy="228" r="6" fill="#222"/>
      <!-- Back Right -->
      <circle cx="300" cy="228" r="14" fill="url(#wheelGrad)" stroke="#111" stroke-width="1.5"/>
      <circle cx="300" cy="228" r="6" fill="#222"/>
    </g>
    <!-- ELEPHANT -->
    <g id="elephant" transform="translate(0, -10)">
      <!-- Back Legs (Behind Body) -->
      <path d="M180,200 L175,240 L195,240 L190,200 Z" fill="url(#elephantGrad)" stroke="#3D5A6E" stroke-width="1"/>
      <path d="M230,200 L225,240 L245,240 L240,200 Z" fill="url(#elephantGrad)" stroke="#3D5A6E" stroke-width="1"/>
      <!-- Back Feet Details -->
      <ellipse cx="185" cy="242" rx="12" ry="6" fill="#4A6A7E"/>
      <ellipse cx="235" cy="242" rx="12" ry="6" fill="#4A6A7E"/>
      <!-- Tail -->
      <path d="M280,170 Q300,180 310,175" stroke="url(#elephantGrad)" stroke-width="6" fill="none" stroke-linecap="round"/>
      <!-- Tail Tuft -->
      <path d="M310,175 Q320,165 315,160 Q325,170 318,175" fill="#4A6A7E"/>
      <!-- Body -->
      <ellipse cx="210" cy="160" rx="80" ry="55" fill="url(#elephantGrad)" stroke="#3D5A6E" stroke-width="1.5"/>
      <!-- Front Legs -->
      <path d="M150,195 L145,240 L165,240 L160,195 Z" fill="url(#elephantGrad)" stroke="#3D5A6E" stroke-width="1"/>
      <path d="M200,195 L195,240 L215,240 L210,195 Z" fill="url(#elephantGrad)" stroke="#3D5A6E" stroke-width="1"/>
      <!-- Front Feet Details -->
      <ellipse cx="155" cy="242" rx="12" ry="6" fill="#4A6A7E"/>
      <ellipse cx="205" cy="242" rx="12" ry="6" fill="#4A6A7E"/>
      <!-- Head -->
      <ellipse cx="120" cy="145" rx="45" ry="40" fill="url(#elephantGrad)" stroke="#3D5A6E" stroke-width="1.5"/>
      <!-- Ears -->
      <!-- Left Ear (Back) -->
      <path d="M90,120 Q60,80 70,160 Q85,150 90,120" fill="url(#earGrad)" stroke="#3D5A6E" stroke-width="1"/>
      <!-- Right Ear (Front) -->
      <path d="M140,115 Q175,70 170,160 Q145,150 140,115" fill="url(#earGrad)" stroke="#3D5A6E" stroke-width="1"/>
      <!-- Ear Inner Detail -->
      <path d="M145,125 Q165,95 158,150" fill="none" stroke="#5D7D91" stroke-width="2" opacity="0.5"/>
      <!-- Trunk -->
      <path d="M85,145 
               Q70,155 60,170 
               Q50,185 55,200 
               Q65,210 75,205" 
            stroke="url(#elephantGrad)" stroke-width="18" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      <!-- Trunk Tip Highlight -->
      <path d="M55,200 Q65,210 75,205" stroke="#8AA5B8" stroke-width="4" fill="none" stroke-linecap="round" opacity="0.5"/>
      <!-- Eyes -->
      <!-- Left Eye -->
      <circle cx="105" cy="135" r="6" fill="#1A1A2E"/>
      <circle cx="107" cy="133" r="2" fill="#FFFFFF"/>
      <!-- Right Eye -->
      <circle cx="130" cy="135" r="6" fill="#1A1A2E"/>
      <circle cx="132" cy="133" r="2" fill="#FFFFFF"/>
      <!-- Eyebrows / Expression -->
      <path d="M95,125 Q105,120 115,125" stroke="#3D5A6E" stroke-width="2" fill="none" stroke-linecap="round"/>
      <path d="M120,125 Q130,120 140,125" stroke="#3D5A6E" stroke-width="2" fill="none" stroke-linecap="round"/>
      <!-- Tusks -->
      <path d="M80,155 Q70,165 65,180" stroke="#F5F5F0" stroke-width="6" fill="none" stroke-linecap="round"/>
      <path d="M95,155 Q100,165 98,180" stroke="#F5F5F0" stroke-width="6" fill="none" stroke-linecap="round"/>
    </g>
  </g>
  <!-- Ground Shadow -->
  <ellipse cx="200" cy="245" rx="100" ry="8" fill="#000000" opacity="0.15"/>
</svg>
```

</details>

## 13. `qwen3.5:397b-cloud`

**SVG size:** 2604 characters  
**Complexity:** Medium  
**Shape elements:** 22  
**Distinct colors:** 15  
**Raw response:** 2651 characters  
**Generation time:** 25.0s

![qwen3.5:397b-cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/qwen3.5_397b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning the elephant and tilting the skateboard
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "elephant" in text/comments -- the model captured the subject
- Renders a trunk -- the model captured the elephant's most distinctive feature
- Renders tusks -- the model captured fine anatomical detail
- Renders ears -- the model captured the elephant's large ears (key for African vs Asian distinction)
- Renders legs -- the model captured the elephant's anatomy
- Renders a tail -- the model captured full anatomical detail
- Renders eyes -- the model gave the elephant character
- Renders "skateboard" in text/comments -- the model captured the object
- Renders a deck (the board itself) -- the model captured the skateboard's main platform
- Renders wheels -- the model captured the skateboard's rolling mechanism (typically 4 wheels)
- Renders trucks (the axle assemblies) -- the model captured fine skateboard mechanical detail
- Elephant-skateboard interaction is present -- the model combined both concepts in one scene
- Motion/speed lines are present -- the model captured the dynamic feel of skateboarding
- Ground/surface is present -- the model placed the skateboard on a surface

With 2,604 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background Circle (Optional decoration) -->
  <circle cx="250" cy="250" r="200" fill="#f0f4f8" />
  <!-- SKATEBOARD -->
  <g id="skateboard">
    <!-- Wheels -->
    <circle cx="150" cy="380" r="25" fill="#333" stroke="#555" stroke-width="2"/>
    <circle cx="350" cy="380" r="25" fill="#333" stroke="#555" stroke-width="2"/>
    <!-- Trucks (Axles) -->
    <rect x="130" y="340" width="40" height="10" fill="#888" rx="2" />
    <rect x="330" y="340" width="40" height="10" fill="#888" rx="2" />
    <!-- Deck -->
    <rect x="100" y="330" width="300" height="30" rx="15" fill="#FF6B6B" stroke="#D64545" stroke-width="2"/>
    <!-- Grip Tape Texture (Lines) -->
    <path d="M120 335 L120 355 M160 335 L160 355 M200 335 L200 355 M240 335 L240 355 M280 335 L280 355 M320 335 L320 355 M360 335 L360 355" stroke="#444" stroke-width="2" opacity="0.5"/>
  </g>
  <!-- ELEPHANT -->
  <g id="elephant" transform="translate(0, -20)">
    <!-- Back Leg (Right side) -->
    <rect x="280" y="260" width="35" height="80" rx="10" fill="#95a5a6" />
    <!-- Tail -->
    <path d="M220 280 Q200 300 190 320" stroke="#7f8c8d" stroke-width="6" fill="none" stroke-linecap="round"/>
    <circle cx="190" cy="320" r="6" fill="#333" />
    <!-- Body -->
    <ellipse cx="250" cy="250" rx="110" ry="80" fill="#95a5a6" />
    <!-- Belly shading -->
    <ellipse cx="250" cy="290" rx="80" ry="40" fill="#bdc3c7" opacity="0.4"/>
    <!-- Front Leg (Left side) -->
    <rect x="200" y="260" width="35" height="80" rx="10" fill="#95a5a6" />
    <!-- Foot -->
    <ellipse cx="217" cy="340" rx="25" ry="10" fill="#7f8c8d" />
    <!-- Head Group -->
    <g transform="translate(300, 200)">
      <!-- Ear (Back) -->
      <path d="M20 10 Q60 -20 70 20 Q60 80 20 60" fill="#7f8c8d" stroke="#666" stroke-width="2"/>
      <!-- Head Shape -->
      <circle cx="30" cy="30" r="55" fill="#95a5a6" />
      <!-- Ear (Front - slightly lighter) -->
      <path d="M25 15 Q55 -10 65 25 Q55 75 25 55" fill="#bdc3c7" stroke="#999" stroke-width="1"/>
      <!-- Eye -->
      <circle cx="45" cy="20" r="6" fill="white" />
      <circle cx="47" cy="20" r="2" fill="black" />
      <!-- Tusk -->
      <path d="M35 55 Q50 65 45 75" stroke="white" stroke-width="8" fill="none" stroke-linecap="round"/>
      <!-- Trunk -->
      <path d="M10 50 Q-10 80 0 110 Q10 130 30 125" stroke="#95a5a6" stroke-width="22" fill="none" stroke-linecap="round"/>
      <!-- Trunk Tip -->
      <circle cx="30" cy="125" r="12" fill="#95a5a6" />
    </g>
  </g>
</svg>
```

</details>

## 14. `deepseek-v4-flash:0731-cloud`

**SVG size:** 4669 characters  
**Complexity:** Medium  
**Shape elements:** 48  
**Distinct colors:** 14  
**Raw response:** 4669 characters  
**Generation time:** ~100s

![deepseek-v4-flash:0731-cloud elephant skateboard SVG](/assets/img/diagrams/ollama-svg-elephant/deepseek-v4-flash_0731-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- The model rendered its interpretation of the "elephant skateboard" prompt -- review the SVG above for how well it captured the subject, composition, and key visual cues

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <defs>
    <!-- Gradient for the wheels to give a slight 3D effect -->
    <radialGradient id="wheelGrad" cx="30%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#636e72" />
      <stop offset="100%" stop-color="#2d3436" />
    </radialGradient>
    <radialGradient id="hubGrad" cx="30%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#dfe6e9" />
      <stop offset="100%" stop-color="#b2bec3" />
    </radialGradient>
  </defs>
  <!-- Background Layer -->
  <rect width="800" height="600" fill="#fdfbf7" />
  <circle cx="400" cy="300" r="240" fill="#ffebd6" />
  <!-- Sun Decoration -->
  <circle cx="140" cy="140" r="45" fill="#f9f871" />
  <circle cx="140" cy="140" r="60" fill="none" stroke="#f9f871" stroke-width="4" stroke-dasharray="10 10" />
  <!-- Ground Shadow -->
  <ellipse cx="400" cy="530" rx="200" ry="15" fill="#e8d8c3" />
  <!-- Motion Lines (Speed effect) -->
  <line x1="180" y1="470" x2="240" y2="470" stroke="#ff9f1c" stroke-width="6" stroke-linecap="round" />
  <line x1="150" y1="500" x2="230" y2="500" stroke="#ff9f1c" stroke-width="6" stroke-linecap="round" />
  <line x1="170" y1="530" x2="220" y2="530" stroke="#ff9f1c" stroke-width="6" stroke-linecap="round" />
  <!-- Elephant Tail -->
  <path d="M 280 330 Q 230 360 240 410" stroke="#8ba3c7" stroke-width="12" fill="none" stroke-linecap="round" />
  <path d="M 240 410 L 230 420 M 240 410 L 245 425 M 240 410 L 235 430" stroke="#8ba3c7" stroke-width="4" fill="none" stroke-linecap="round" />
  <!-- Far Legs -->
  <rect x="300" y="380" width="25" height="80" rx="12" fill="#7a96b8" />
  <rect x="440" y="380" width="25" height="80" rx="12" fill="#7a96b8" />
  <!-- Far Legs Toenails -->
  <circle cx="310" cy="455" r="4" fill="#f7c8d0" />
  <circle cx="320" cy="455" r="4" fill="#f7c8d0" />
  <circle cx="450" cy="455" r="4" fill="#f7c8d0" />
  <circle cx="460" cy="455" r="4" fill="#f7c8d0" />
  <!-- Elephant Body -->
  <ellipse cx="370" cy="350" rx="100" ry="80" fill="#8ba3c7" />
  <!-- Saddle / Blanket -->
  <path d="M 310 320 C 310 280, 430 280, 430 320 C 430 360, 310 360, 310 320 Z" fill="#2ec4b6" />
  <path d="M 310 320 C 310 280, 430 280, 430 320" fill="none" stroke="#f9f871" stroke-width="6" stroke-dasharray="10 5" />
  <!-- Elephant Head -->
  <ellipse cx="500" cy="310" rx="65" ry="75" fill="#8ba3c7" />
  <!-- Elephant Ear -->
  <path d="M 480 260 C 440 250, 420 350, 460 390 C 480 410, 510 380, 500 340 Z" fill="#8ba3c7" />
  <path d="M 480 280 C 455 275, 440 340, 465 370 C 475 380, 490 365, 485 340 Z" fill="#f7c8d0" />
  <!-- Elephant Trunk -->
  <path d="M 540 280 Q 600 300 590 420 Q 585 460 560 460 Q 550 460 550 450 Q 570 440 570 400 Q 570 330 520 340 Z" fill="#8ba3c7" />
  <!-- Tusk -->
  <path d="M 530 380 Q 560 390 560 410 Q 550 400 525 390 Z" fill="#ffffff" />
  <!-- Eye -->
  <circle cx="530" cy="290" r="6" fill="#2d3436" />
  <circle cx="532" cy="288" r="2" fill="#ffffff" />
  <!-- Cap / Hat -->
  <path d="M 460 260 C 460 220, 520 220, 530 260 Z" fill="#ff6b6b" />
  <path d="M 525 260 Q 560 255 570 265 L 525 265 Z" fill="#ee5253" />
  <!-- Near Legs -->
  <rect x="335" y="380" width="25" height="80" rx="12" fill="#8ba3c7" />
  <rect x="475" y="380" width="25" height="80" rx="12" fill="#8ba3c7" />
  <!-- Near Legs Toenails -->
  <circle cx="345" cy="455" r="4" fill="#f7c8d0" />
  <circle cx="355" cy="455" r="4" fill="#f7c8d0" />
  <circle cx="365" cy="455" r="4" fill="#f7c8d0" />
  <circle cx="485" cy="455" r="4" fill="#f7c8d0" />
  <circle cx="495" cy="455" r="4" fill="#f7c8d0" />
  <circle cx="505" cy="455" r="4" fill="#f7c8d0" />
  <!-- Skateboard Deck -->
  <rect x="250" y="460" width="300" height="20" rx="10" fill="#ff9f1c" />
  <!-- Skateboard Trucks -->
  <path d="M 310 480 L 330 480 L 325 495 L 315 495 Z" fill="#636e72" />
  <path d="M 470 480 L 490 480 L 485 495 L 475 495 Z" fill="#636e72" />
  <!-- Skateboard Wheels -->
  <circle cx="320" cy="505" r="25" fill="url(#wheelGrad)" />
  <circle cx="320" cy="505" r="10" fill="url(#hubGrad)" />
  <circle cx="480" cy="505" r="25" fill="url(#wheelGrad)" />
  <circle cx="480" cy="505" r="10" fill="url(#hubGrad)" />
  <!-- Grip Tape -->
  <rect x="250" y="460" width="300" height="5" rx="2" fill="#2d3436" />
  <!-- Decorative Sparkles -->
  <path d="M 600 200 L 605 210 L 615 212 L 605 214 L 600 224 L 595 214 L 585 212 L 595 210 Z" fill="#f9f871" />
  <path d="M 220 220 L 223 226 L 230 228 L 223 230 L 220 236 L 217 230 L 210 228 L 217 226 Z" fill="#ff9f1c" />
  <circle cx="620" cy="380" r="5" fill="#ff6b6b" />
  <circle cx="200" cy="350" r="6" fill="#2ec4b6" />
</svg>
```

</details>


## Conclusion

That's all 14 elephant-on-a-skateboard SVGs from the active Ollama cloud models. This whimsical-animal prompt tested a different skill set than our previous prompts: organic anatomy (elephant trunk, ears, legs), mechanical precision (skateboard deck, wheels, trucks), scale mismatch handling, and creative humor.

**No single model is best at everything.** A model that produces a stunning World Cup poster may produce a basic elephant, and vice versa. The best model for you depends on your specific use case:

- For **maximum detail**: look at the "Very high" complexity models
- For **speed**: look at the generation times in each section
- For **anatomical accuracy**: check which models captured the elephant's trunk, ears, and tusks
- For **skateboard accuracy**: check which models rendered 4 wheels, a deck, and trucks
- For **humor and whimsy**: look for models that added motion lines, poses, or playful details
- For **code quality**: look at the raw SVG source -- clean, well-structured code is easier to customize

Try the same prompt yourself with [Ollama Cloud](https://ollama.com/cloud) and see if you agree with our analysis. And check out the other posts in our SVG benchmark series:

- [Duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/)
- [Duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/)
- [Duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/)
- [Cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/)
- [Duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/)
- [Baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/)
- [Octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/)
- [FIFA World Cup 2026](/Ollama-Cloud-Models-SVG-Comparison-Fifa-Worldcup-2026/)
