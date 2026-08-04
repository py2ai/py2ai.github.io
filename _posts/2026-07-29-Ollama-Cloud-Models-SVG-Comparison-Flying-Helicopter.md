---
layout: post
title: "Which Ollama Cloud Model is Best? Flying Helicopter SVG Comparison (14 Models)"
description: "Compare 14 Ollama cloud models on a flying helicopter SVG prompt. Find the best LLM for aviation and mechanical SVG scenes. You decide the winner."
date: 2026-07-30
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Flying-Helicopter/
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
  - Helicopter
  - Aviation
  - Animation
  - Flying
  - Mechanical
author: "PyShine"
seo:
  keywords: "best Ollama model for SVG, best LLM for SVG generation, Ollama cloud model comparison, helicopter SVG, AI helicopter drawing, LLM SVG benchmark, AI image generation comparison, deepseek vs glm vs qwen, which Ollama model is best, Ollama cloud models 2026, AI aviation art, flying helicopter SVG, animated SVG, rotor blades SVG, mechanical SVG, AI art comparison, complex SVG scene, aviation illustration, helicopter diagram"
---

# Which Ollama Cloud Model is Best? Flying Helicopter SVG Comparison (14 Models)

After testing LLMs on ducks, vehicles, dev scenes, marine life, chess, the FIFA World Cup, and an elephant on a skateboard, we wanted to know: **can today's top models draw a machine with moving parts?** This time we asked 14 Ollama cloud models to draw **a flying helicopter** -- a prompt that tests mechanical precision (rotor blades, fuselage, tail boom, landing skids), aerodynamic understanding (how a helicopter flies), scene context (sky, clouds, motion), and animation potential (spinning rotors, hovering).

The prompt was: `Make an svg image of a flying helicopter`

This is the tenth in our SVG benchmark series. See also: [duck on a bicycle](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), [duck with a parachute](/Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/), [duck driving a jeep](/Ollama-Cloud-Models-SVG-Comparison-Duck-Jeep/), [cherry blossom trees](/Ollama-Cloud-Models-SVG-Comparison-Cherry-Blossom/), [duck programmer debugging at 3am](/Ollama-Cloud-Models-SVG-Comparison-Duck-Programmer/), [baby shark fish](/Ollama-Cloud-Models-SVG-Comparison-Baby-Shark/), [octopus playing chess](/Ollama-Cloud-Models-SVG-Comparison-Octopus-Chess/), [FIFA World Cup 2026](/Ollama-Cloud-Models-SVG-Comparison-Fifa-Worldcup-2026/), and [elephant on a skateboard](/Ollama-Cloud-Models-SVG-Comparison-Elephant-Skateboard/).

**Why a flying helicopter?** This prompt is a mechanical stress test for SVG generation because it combines multiple hard problems: (1) **Mechanical anatomy** -- a helicopter has a fuselage (main body), a cockpit (where the pilot sits), a main rotor (the large spinning blades on top that generate lift), a tail rotor (the smaller vertical rotor that counteracts torque), a tail boom (the long extension to the tail), landing skids or wheels, and often navigation lights, (2) **Rotational motion** -- the rotors must spin, which means a good model should use `<animateTransform>` or CSS `@keyframes` to animate them, (3) **Flight dynamics** -- a flying helicopter should be in the air, not on the ground, so the model should add a sky, clouds, or ground below, (4) **Proportions and symmetry** -- the main rotor must be centered above the fuselage, the tail rotor must be at the end of the tail boom, and the landing skids must be symmetric, (5) **Detail vs. simplicity** -- a helicopter has many small parts (windows, door handles, antenna, exhaust), and the model must decide how much detail to include, (6) **Scene context** -- a helicopter flying in a void is boring; a good model adds clouds, stars, a sun/moon, or a landscape below. A model that draws a great elephant may fail here because helicopters require precise mechanical shapes, not organic curves.

**The goal is not to declare a winner -- it is to give you the data so you can pick the best model for your own use case.** We show you the SVG, the stats, and a short analysis for each. You decide.

## How to Choose the Best Ollama Model for Mechanical SVGs

The helicopter prompt rewards different things than previous prompts. Here are the criteria to use:

- **Helicopter anatomy**: Does the SVG have a fuselage, cockpit, main rotor, tail rotor, tail boom, and landing skids? Or is it just a generic blob with a stick on top? The main rotor is the most critical -- without it, it's not a helicopter.
- **Rotor detail**: Are the rotor blades rendered as distinct elements? Does the model attempt to show them spinning (via animation)? A great SVG shows the main rotor and tail rotor separately.
- **Proportions and symmetry**: Is the main rotor centered above the fuselage? Are the landing skids symmetric? Is the tail boom proportional to the body? Bad proportions make the helicopter look wrong even if all parts are present.
- **Flight context**: Is the helicopter in the air? Does the SVG include a sky, clouds, stars, or a ground below? Or is the helicopter floating in a void?
- **Animation**: Did the model use `<animate>` or `@keyframes` to make the rotors spin, the helicopter hover, or the clouds drift? Animation is what makes an SVG feel "alive" -- and it's a strong signal of model capability.
- **Mechanical detail**: Does the SVG include navigation lights, a door, windows, an antenna, exhaust, or a searchlight? These small details show the model's attention to realism.
- **SVG code quality**: Does it use `<defs>`, `<use>`, gradients, and clean structure? Better code is easier to tweak (e.g., to recolor the helicopter or change the rotor speed).

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering with minimal post-processing (adding `width="100%" height="auto"` for responsive embedding and fixing XML errors so the SVG renders in browsers).

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary Table: Compare All Models at a Glance

Use this table to quickly compare models on the metrics that matter. The **verdict** column is a one-line summary to help you shortlist -- but read the per-model sections below for the full picture before you decide.

| # | Model | SVG Size | Shapes | Colors | Complexity | Verdict |
|---|-------|----------|--------|--------|------------|---------|
| 1 | `deepseek-v4-flash:cloud` | 8796 | 55 | 12 | High | Detailed |
| 2 | `deepseek-v4-pro:cloud` | 6814 | 56 | 9 | High | Detailed |
| 3 | `gemma4:31b-cloud` | 1865 | 16 | 8 | Medium | Balanced |
| 4 | `gemma4:cloud` | 1737 | 17 | 6 | Medium | Balanced |
| 5 | `glm-5.1:cloud` | 10242 | 73 | 19 | Very high | Richest scene |
| 6 | `glm-5.2:cloud` | 8044 | 43 | 13 | High | Detailed |
| 7 | `gpt-oss:120b-cloud` | 1990 | 19 | 9 | Medium | Balanced |
| 8 | `kimi-k2.6:cloud` | 4052 | 17 | 9 | Medium | Balanced |
| 9 | `minimax-m2.7:cloud` | 2608 | 16 | 10 | Medium | Balanced |
| 10 | `minimax-m3:cloud` | 4053 | 35 | 13 | High | Detailed |
| 11 | `nemotron-3-super:cloud` | 796 | 7 | 4 | Compact | Compact |
| 12 | `nemotron-3-ultra:cloud` | 7241 | 39 | 15 | High | Detailed |
| 13 | `qwen3.5:397b-cloud` | 3707 | 22 | 7 | Medium | Balanced |
| 14 | `deepseek-v4-flash:0731-cloud` | 7972 | 66 | 9 | High | Detailed |
| 15 | `bjoernb/claude-opus-4-5:latest` | - | - | - | - | Retired (410) |
| 16 | `deepseek-v3.1:671b-cloud` | - | - | - | - | Retired (410) |
| 17 | `glm-5:cloud` | - | - | - | - | Retired (410) |
| 18 | `qwen3-vl:235b-cloud` | - | - | - | - | Retired (410) |

**14 out of 18** active models produced a valid SVG. The 4 retired models returned HTTP 410 Gone (removed from Ollama Cloud on 2026-07-15).

## Quick Recommendation by Use Case

If you just want a shortcut, here is which model to pick based on what you care about:

- **You want the most detailed helicopter SVG**: pick models labeled "Very high" complexity in the table above
- **You want the fastest response**: look at the per-model sections below for the elapsed time
- **You want the cleanest, most reusable SVG code**: pick models that use `<defs>`, `<use>`, and transforms (see raw source below each SVG)
- **You want a small, efficient SVG for web embedding**: pick models with "Compact" verdict
- **You want accurate helicopter anatomy (rotor, tail, skids)**: check the per-model analysis -- the main rotor is the most critical part
- **You want animation (spinning rotors, hovering)**: look for models that used `<animate>` or `@keyframes`
- **You want a balance of detail and speed**: pick models labeled "Balanced" or "Detailed"
- **You want to compare within a model family**: pick `deepseek-v4-pro` vs `deepseek-v4-flash`, or `glm-5.1` vs `glm-5.2`, or `minimax-m2.7` vs `minimax-m3`

Now read on for the full per-model breakdown and judge for yourself.

## 1. `deepseek-v4-flash:cloud`

**SVG size:** 8796 characters  
**Complexity:** High  
**Shape elements:** 55  
**Distinct colors:** 12  
**Raw response:** 9410 characters  
**Generation time:** 30.0s

![deepseek-v4-flash:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/deepseek-v4-flash_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders clouds -- the model placed the helicopter in a sky scene
- Renders sky -- the model captured the flying context
- Renders a sun/moon -- the model added celestial detail
- Renders ground/terrain -- the model placed the helicopter above a surface
- Renders exhaust/smoke -- the model captured engine output detail
- Renders navigation lights -- the model captured aviation lighting detail

With 8,796 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
    <defs>
        <!-- Background Sky Gradient -->
        <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#4A90E2" />
            <stop offset="60%" stop-color="#87CEEB" />
            <stop offset="100%" stop-color="#E0F6FF" />
        </linearGradient>
        <!-- Helicopter Body Gradient -->
        <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#5DADE2" />
            <stop offset="100%" stop-color="#1B4F72" />
        </linearGradient>
        <!-- Cockpit Window Gradient -->
        <linearGradient id="windowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#D4E6F1" stop-opacity="0.9" />
            <stop offset="100%" stop-color="#2980B9" stop-opacity="0.9" />
        </linearGradient>
        <!-- Motion Blur Filters -->
        <filter id="motionBlurH" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="15 0" />
        </filter>
        <filter id="motionBlurV" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceGraphic" stdDeviation="0 15" />
        </filter>
        <!-- Drop Shadow for Helicopter -->
        <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="8" dy="15" stdDeviation="6" flood-color="#000" flood-opacity="0.3" />
        </filter>
    </defs>
    <!-- Background -->
    <rect width="800" height="600" fill="url(#skyGrad)" />
    <!-- Sun / Glowing Orb -->
    <g transform="translate(650, 150)">
        <circle cx="0" cy="0" r="80" fill="#FFD700" opacity="0.1" />
        <circle cx="0" cy="0" r="60" fill="#FFD700" opacity="0.2" />
        <circle cx="0" cy="0" r="40" fill="#FFD700" opacity="0.8" />
        <circle cx="0" cy="0" r="30" fill="#FFF" />
    </g>
    <!-- Clouds -->
    <g fill="#FFFFFF" opacity="0.8">
        <path d="M 100 200 Q 120 180, 140 200 Q 160 190, 180 200 Q 200 220, 180 230 L 100 230 Q 80 220, 100 200 Z" />
        <path d="M 600 400 Q 620 380, 640 400 Q 660 390, 680 400 Q 700 420, 680 430 L 600 430 Q 580 420, 600 400 Z" opacity="0.6" />
        <path d="M 50 500 Q 70 480, 90 500 Q 110 490, 130 500 Q 150 520, 130 530 L 50 530 Q 30 520, 50 500 Z" opacity="0.5" />
        <path d="M 700 200 Q 720 180, 740 200 Q 760 190, 780 200 Q 800 220, 780 230 L 700 230 Q 680 220, 700 200 Z" opacity="0.4" />
    </g>
    <!-- Ground Shadow -->
    <ellipse cx="400" cy="550" rx="200" ry="25" fill="#000" opacity="0.15" />
    <!-- Motion Lines (Speed) -->
    <g stroke="#A0A0A0" stroke-width="3" stroke-linecap="round" fill="none" opacity="0.6">
        <line x1="50" y1="270" x2="180" y2="270" />
        <line x1="80" y1="300" x2="200" y2="300" />
        <line x1="60" y1="330" x2="160" y2="330" />
        <line x1="550" y1="250" x2="650" y2="250" />
        <line x1="520" y1="280" x2="620" y2="280" />
        <line x1="580" y1="320" x2="700" y2="320" />
    </g>
    <!-- Flying Helicopter Group -->
    <g transform="translate(100, 50) scale(1.3)" filter="url(#dropShadow)">
        <!-- === TAIL GROUP === -->
        <g id="tail">
            <!-- Tail Boom -->
            <path d="M 280 295 L 150 295 L 150 305 L 280 305 Z" fill="url(#bodyGrad)" stroke="#0D3B66" stroke-width="1.5" />
            <!-- Vertical Tail Fin -->
            <path d="M 160 295 Q 140 260, 120 240 L 100 240 Q 100 280, 100 305 Z" fill="#E74C3C" stroke="#0D3B66" stroke-width="1.5" />
            <!-- Horizontal Tail Fin -->
            <rect x="110" y="285" width="50" height="8" rx="4" fill="#E74C3C" stroke="#0D3B66" stroke-width="1.5" />
            <!-- Tail Rotor Disc (Motion Blur) -->
            <ellipse cx="100" cy="275" rx="4" ry="25" fill="#A0A0A0" opacity="0.6" filter="url(#motionBlurV)" />
            <!-- Tail Rotor Hub -->
            <circle cx="100" cy="275" r="3" fill="#333" />
            <!-- Tail Rotor Solid Core -->
            <line x1="100" y1="252" x2="100" y2="298" stroke="#333" stroke-width="1.5" />
        </g>
        <!-- === MAIN BODY GROUP === -->
        <g id="fuselage">
            <!-- Main Body Shape -->
            <path d="M 280 290 C 280 260, 380 250, 430 270 C 460 285, 470 310, 430 320 C 380 330, 300 320, 280 310 Z" fill="url(#bodyGrad)" stroke="#0D3B66" stroke-width="1.5" />
            <!-- Cockpit Windows -->
            <path d="M 380 270 C 410 270, 440 285, 440 300 C 440 310, 420 315, 390 310 C 370 305, 360 280, 380 270 Z" fill="url(#windowGrad)" stroke="#0D3B66" stroke-width="1.5" />
            <!-- Window Frame Divider -->
            <path d="M 380 270 C 410 270, 440 285, 440 300" fill="none" stroke="#0D3B66" stroke-width="1.5" />
            <!-- Pilot Silhouette -->
            <ellipse cx="410" cy="285" rx="6" ry="10" fill="#222" />
            <circle cx="410" cy="278" r="5" fill="#222" />
            <!-- Cockpit Dashboard Reflection -->
            <path d="M 385 285 L 400 285 L 395 295 L 385 295 Z" fill="#FFF" opacity="0.3" />
            <!-- Engine Exhaust / Side Details -->
            <path d="M 300 280 L 320 280 L 320 290 L 300 290 Z" fill="#555" stroke="#0D3B66" stroke-width="1" />
            <circle cx="340" cy="285" r="2.5" fill="#E74C3C" />
            <circle cx="355" cy="285" r="2.5" fill="#E74C3C" />
            <!-- Door Line -->
            <line x1="330" y1="270" x2="330" y2="315" stroke="#0D3B66" stroke-width="1" opacity="0.5" />
            <!-- Door Handle -->
            <rect x="335" y="295" width="8" height="2" rx="1" fill="#FFF" opacity="0.8" />
            <!-- Searchlight / Utility Pod -->
            <rect x="430" y="315" width="10" height="15" rx="2" fill="#FFD700" stroke="#333" stroke-width="1" />
            <polygon points="435,330 425,350 445,350" fill="#FFD700" opacity="0.6" />
        </g>
        <!-- === LANDING GEAR GROUP === -->
        <g id="landingGear">
            <!-- Left Skid -->
            <path d="M 320 320 L 320 340 Q 320 345, 325 345 L 375 345 Q 380 345, 380 340 L 380 320" fill="none" stroke="#333" stroke-width="4" stroke-linejoin="round" />
            <!-- Right Skid -->
            <path d="M 400 320 L 400 340 Q 400 345, 405 345 L 455 345 Q 460 345, 460 340 L 460 320" fill="none" stroke="#333" stroke-width="4" stroke-linejoin="round" />
            <!-- Struts -->
            <line x1="330" y1="320" x2="330" y2="340" stroke="#555" stroke-width="3" />
            <line x1="370" y1="320" x2="370" y2="340" stroke="#555" stroke-width="3" />
            <line x1="410" y1="320" x2="410" y2="340" stroke="#555" stroke-width="3" />
            <line x1="450" y1="320" x2="450" y2="340" stroke="#555" stroke-width="3" />
            <!-- Skid Tips (Red/Orange) -->
            <path d="M 320 340 Q 320 345, 325 345" fill="none" stroke="#E74C3C" stroke-width="4" stroke-linecap="round" />
            <path d="M 375 345 Q 380 345, 380 340" fill="none" stroke="#E74C3C" stroke-width="4" stroke-linecap="round" />
            <path d="M 400 340 Q 400 345, 405 345" fill="none" stroke="#E74C3C" stroke-width="4" stroke-linecap="round" />
            <path d="M 455 345 Q 460 345, 460 340" fill="none" stroke="#E74C3C" stroke-width="4" stroke-linecap="round" />
        </g>
        <!-- === MAIN ROTOR GROUP === -->
        <g id="mainRotor">
            <!-- Rotor Mast -->
            <rect x="330" y="230" width="10" height="20" fill="#555" stroke="#0D3B66" stroke-width="1.5" />
            <!-- Rotor Disc (Motion Blur Effect) -->
            <ellipse cx="335" cy="230" rx="140" ry="20" fill="#B0C4DE" opacity="0.2" />
            <ellipse cx="335" cy="230" rx="140" ry="10" fill="#A0A0A0" opacity="0.4" filter="url(#motionBlurH)" />
            <!-- Rotor Hub -->
            <circle cx="335" cy="230" r="8" fill="#333" stroke="#0D3B66" stroke-width="1.5" />
            <!-- Solid Rotor Blades Core -->
            <line x1="195" y1="230" x2="475" y2="230" stroke="#333" stroke-width="4" stroke-linecap="round" />
            <line x1="195" y1="230" x2="475" y2="230" stroke="#E74C3C" stroke-width="1.5" stroke-linecap="round" />
            <!-- Rotor Blade Tips (High Speed Indicators) -->
            <ellipse cx="195" cy="230" rx="8" ry="3" fill="#E74C3C" />
            <ellipse cx="475" cy="230" rx="8" ry="3" fill="#E74C3C" />
            <!-- Spinning Disc Dashed Lines -->
            <ellipse cx="335" cy="230" rx="130" ry="15" fill="none" stroke="#FFF" stroke-width="1" stroke-dasharray="10, 15" opacity="0.6" />
            <ellipse cx="335" cy="230" rx="100" ry="10" fill="none" stroke="#FFF" stroke-width="1" stroke-dasharray="5, 10" opacity="0.4" />
        </g>
    </g>
</svg>
```

</details>

## 2. `deepseek-v4-pro:cloud`

**SVG size:** 6814 characters  
**Complexity:** High  
**Shape elements:** 56  
**Distinct colors:** 9  
**Raw response:** 7267 characters  
**Generation time:** 30.0s

![deepseek-v4-pro:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/deepseek-v4-pro_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses `<use>` references to reuse defined elements -- ideal for repeating rotor blades or cloud patterns
- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders clouds -- the model placed the helicopter in a sky scene
- Renders sky -- the model captured the flying context
- Renders a sun/moon -- the model added celestial detail
- Renders ground/terrain -- the model placed the helicopter above a surface
- Renders exhaust/smoke -- the model captured engine output detail
- Renders navigation lights -- the model captured aviation lighting detail

With 6,814 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="100%" height="auto">
  <defs>
    <!-- Sky Gradient -->
    <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#4facfe" />
      <stop offset="100%" stop-color="#00f2fe" />
    </linearGradient>
    <!-- Drop Shadow for Helicopter -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="8" dy="12" stdDeviation="6" flood-color="#000" flood-opacity="0.25" />
    </filter>
    <!-- Reusable Cloud Component -->
    <g id="cloud">
      <circle cx="0" cy="0" r="30" />
      <circle cx="40" cy="-10" r="40" />
      <circle cx="80" cy="0" r="30" />
      <rect x="0" y="0" width="80" height="30" rx="15" />
    </g>
  </defs>
  <!-- Background -->
  <rect width="500" height="500" fill="url(#skyGrad)" />
  <!-- Sun -->
  <circle cx="400" cy="100" r="45" fill="#FFD700" opacity="0.9" />
  <circle cx="400" cy="100" r="60" fill="#FFD700" opacity="0.3" />
  <!-- Background Clouds -->
  <use href="#cloud" x="50" y="80" fill="#FFFFFF" opacity="0.6" transform="scale(0.8)" />
  <use href="#cloud" x="300" y="150" fill="#FFFFFF" opacity="0.5" transform="scale(0.6)" />
  <use href="#cloud" x="-20" y="200" fill="#FFFFFF" opacity="0.4" transform="scale(0.9)" />
  <!-- Motion Lines (Speed effect) -->
  <g stroke="#FFFFFF" stroke-linecap="round" opacity="0.7">
    <line x1="40" y1="250" x2="140" y2="250" stroke-width="4" />
    <line x1="70" y1="270" x2="150" y2="270" stroke-width="2" />
    <line x1="20" y1="230" x2="100" y2="230" stroke-width="3" />
    <line x1="50" y1="290" x2="110" y2="290" stroke-width="2" />
    <line x1="10" y1="210" x2="60" y2="210" stroke-width="1.5" />
  </g>
  <!-- Helicopter Group (Tilted for dynamic flight) -->
  <g transform="translate(250, 250) rotate(-8) translate(-250, -250)" filter="url(#shadow)">
    <!-- Tail Rotor -->
    <g transform="translate(80, 275)">
      <!-- Rotor Blades (Motion Blur) -->
      <ellipse cx="0" cy="0" rx="4" ry="22" fill="#4A5568" opacity="0.4" />
      <ellipse cx="0" cy="0" rx="3" ry="20" fill="#4A5568" opacity="0.6" />
      <ellipse cx="0" cy="0" rx="2" ry="18" fill="#4A5568" />
      <!-- Rotor Hub -->
      <circle cx="0" cy="0" r="4" fill="#2B2D42" />
      <circle cx="0" cy="0" r="2" fill="#718096" />
    </g>
    <!-- Tail Boom -->
    <path d="M 150 260 L 80 270 L 80 280 L 150 280 Z" fill="#2B2D42" />
    <!-- Tail Fin (Vertical Stabilizer) -->
    <path d="M 80 270 L 60 240 L 70 240 L 80 270 Z" fill="#F6AD55" />
    <path d="M 80 280 L 65 300 L 75 300 L 80 280 Z" fill="#F6AD55" />
    <!-- Horizontal Stabilizer -->
    <path d="M 120 270 L 140 280 L 140 285 L 120 275 Z" fill="#2B2D42" />
    <!-- Main Fuselage -->
    <path d="M 180 240 
             L 300 240 
             C 320 240, 330 250, 330 260 
             L 330 270 
             C 330 280, 320 290, 300 290 
             L 180 290 
             C 160 290, 150 280, 150 270 
             L 150 260 
             C 150 250, 160 240, 180 240 Z" fill="#2B2D42" />
    <!-- Fuselage Accent Stripe -->
    <path d="M 180 260 L 320 260" stroke="#F6AD55" stroke-width="4" />
    <path d="M 180 264 L 320 264" stroke="#F6AD55" stroke-width="1.5" opacity="0.5" />
    <!-- Panel Lines (Details) -->
    <line x1="220" y1="240" x2="220" y2="290" stroke="#1A202C" stroke-width="1.5" opacity="0.4" />
    <line x1="260" y1="240" x2="260" y2="290" stroke="#1A202C" stroke-width="1.5" opacity="0.4" />
    <line x1="180" y1="275" x2="300" y2="275" stroke="#1A202C" stroke-width="1.5" opacity="0.4" />
    <!-- Engine Intake -->
    <ellipse cx="200" cy="240" rx="10" ry="15" fill="#1A202C" />
    <ellipse cx="200" cy="240" rx="6" ry="10" fill="#4A5568" />
    <!-- Engine Exhaust -->
    <rect x="220" y="235" width="15" height="10" rx="3" fill="#F6AD55" />
    <path d="M 235 240 L 245 240" stroke="#F6AD55" stroke-width="2" opacity="0.6" />
    <!-- Cockpit Glass -->
    <path d="M 300 240 C 320 240, 330 250, 330 260 C 330 270, 320 280, 300 280 C 310 280, 315 270, 315 260 C 315 250, 310 240, 300 240 Z" fill="#A0C4FF" opacity="0.8" />
    <!-- Cockpit Highlight -->
    <path d="M 300 240 C 310 240, 315 250, 315 260 C 315 270, 310 280, 300 280" fill="none" stroke="#FFFFFF" stroke-width="2" opacity="0.6" />
    <!-- Cabin Windows -->
    <rect x="260" y="245" width="20" height="15" rx="4" fill="#A0C4FF" opacity="0.7" />
    <rect x="230" y="245" width="15" height="15" rx="3" fill="#A0C4FF" opacity="0.7" />
    <!-- Searchlight -->
    <circle cx="330" cy="265" r="6" fill="#F6AD55" />
    <circle cx="330" cy="265" r="3" fill="#FFFFFF" opacity="0.8" />
    <path d="M 330 265 L 360 285" stroke="#F6AD55" stroke-width="3" opacity="0.5" />
    <path d="M 330 265 L 365 280" stroke="#F6AD55" stroke-width="1.5" opacity="0.8" />
    <!-- Antenna -->
    <line x1="180" y1="240" x2="160" y2="220" stroke="#4A5568" stroke-width="2" />
    <circle cx="160" cy="220" r="2" fill="#F6AD55" />
    <!-- Landing Skids -->
    <!-- Struts -->
    <line x1="200" y1="290" x2="200" y2="310" stroke="#4A5568" stroke-width="6" stroke-linecap="round" />
    <line x1="280" y1="290" x2="280" y2="310" stroke="#4A5568" stroke-width="6" stroke-linecap="round" />
    <!-- Skids -->
    <rect x="170" y="310" width="140" height="8" rx="4" fill="#4A5568" />
    <!-- Skid Highlights -->
    <rect x="170" y="312" width="140" height="2" fill="#718096" />
    <rect x="170" y="310" width="140" height="1" fill="#1A202C" opacity="0.5" />
    <!-- Main Rotor Mast -->
    <rect x="245" y="210" width="10" height="30" fill="#4A5568" />
    <rect x="243" y="210" width="14" height="5" fill="#718096" />
    <!-- Main Rotor Blades (Spinning Effect) -->
    <g transform="translate(250, 210)">
      <!-- Motion Blur / Spinning Disc -->
      <ellipse cx="0" cy="0" rx="160" ry="18" fill="#4A5568" opacity="0.15" />
      <ellipse cx="0" cy="0" rx="160" ry="14" fill="#4A5568" opacity="0.25" />
      <ellipse cx="0" cy="0" rx="160" ry="10" fill="#4A5568" opacity="0.4" />
      <ellipse cx="0" cy="0" rx="160" ry="6" fill="#4A5568" opacity="0.6" />
      <!-- Visible Rotor Blades -->
      <rect x="-160" y="-8" width="320" height="16" rx="8" fill="#4A5568" />
      <rect x="-160" y="-8" width="320" height="16" rx="8" fill="#4A5568" transform="rotate(90)" />
      <!-- Rotor Hub -->
      <circle cx="0" cy="0" r="8" fill="#2B2D42" />
      <circle cx="0" cy="0" r="4" fill="#718096" />
    </g>
  </g>
  <!-- Foreground Clouds (Depth effect) -->
  <use href="#cloud" x="350" y="400" fill="#FFFFFF" opacity="0.8" transform="scale(1.2)" />
  <use href="#cloud" x="-30" y="350" fill="#FFFFFF" opacity="0.7" transform="scale(1)" />
  <use href="#cloud" x="200" y="450" fill="#FFFFFF" opacity="0.6" transform="scale(0.9)" />
</svg>
```

</details>

## 3. `gemma4:31b-cloud`

**SVG size:** 1865 characters  
**Complexity:** Medium  
**Shape elements:** 16  
**Distinct colors:** 8  
**Raw response:** 2699 characters  
**Generation time:** 15.0s

![gemma4:31b-cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/gemma4_31b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Includes animation elements -- the model attempted motion (e.g., spinning rotors, hovering, cloud drift)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders clouds -- the model placed the helicopter in a sky scene
- Renders sky -- the model captured the flying context
- Renders ground/terrain -- the model placed the helicopter above a surface
- Includes animation -- the model captured the dynamic motion of flying (spinning rotors, hovering)
- Renders navigation lights -- the model captured aviation lighting detail

With 1,865 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background (Optional: Light Blue Sky) -->
  <rect width="400" height="300" fill="#e0f7fa" />
  <!-- Clouds for depth/motion -->
  <g fill="#ffffff" opacity="0.8">
    <ellipse cx="80" cy="100" rx="30" ry="20" />
    <ellipse cx="110" cy="110" rx="30" ry="20" />
    <ellipse cx="300" cy="150" rx="40" ry="25" />
    <ellipse cx="330" cy="160" rx="30" ry="20" />
  </g>
  <!-- Helicopter Group -->
  <g transform="translate(120, 100)">
    <!-- Tail Boom -->
    <path d="M120 40 L200 30 L200 20 L120 30 Z" fill="#37474f" />
    <!-- Tail Rotor -->
    <rect x="200" y="20" width="5" height="20" rx="2" fill="#78909c" />
    <!-- Main Body (Fuselage) -->
    <path d="M20 60 C 20 30, 120 30, 120 60 L 120 90 C 120 110, 20 110, 20 90 Z" fill="#455a64" />
    <!-- Cockpit Window -->
    <path d="M30 45 C 30 35, 80 35, 80 45 L 80 70 C 80 80, 30 80, 30 70 Z" fill="#bbdefb" />
    <!-- Landing Skids -->
    <line x1="40" y1="110" x2="40" y2="120" stroke="#263238" stroke-width="4" />
    <line x1="100" y1="110" x2="100" y2="120" stroke="#263238" stroke-width="4" />
    <rect x="30" y="120" width="80" height="5" rx="2" fill="#263238" />
    <!-- Rotor Mast -->
    <rect x="67" y="30" width="6" height="10" fill="#263238" />
    <!-- Main Rotor (Animated) -->
    <g>
      <rect x="20" y="27" width="100" height="6" rx="3" fill="#263238">
        <animateTransform 
          attributeName="transform" 
          type="rotate" 
          from="0 70 30" 
          to="360 70 30" 
          dur="0.1s" 
          repeatCount="indefinite" />
      </rect>
    </g>
  </g>
  <!-- Motion Lines -->
  <g stroke="#b0bec5" stroke-width="3" stroke-linecap="round">
    <line x1="100" y1="150" x2="130" y2="150" />
    <line x1="80" y1="170" x2="110" y2="170" />
  </g>
</svg>
```

</details>

## 4. `gemma4:cloud`

**SVG size:** 1737 characters  
**Complexity:** Medium  
**Shape elements:** 17  
**Distinct colors:** 6  
**Raw response:** 2627 characters  
**Generation time:** 15.0s

![gemma4:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/gemma4_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Includes animation elements -- the model attempted motion (e.g., spinning rotors, hovering, cloud drift)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders clouds -- the model placed the helicopter in a sky scene
- Renders sky -- the model captured the flying context
- Renders ground/terrain -- the model placed the helicopter above a surface
- Includes animation -- the model captured the dynamic motion of flying (spinning rotors, hovering)

With 1,737 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Background Sky -->
  <rect width="400" height="300" fill="#e0f7fa" />
  <!-- Clouds -->
  <circle cx="50" cy="80" r="30" fill="white" />
  <circle cx="80" cy="80" r="30" fill="white" />
  <circle cx="110" cy="80" r="30" fill="white" />
  <circle cx="300" cy="150" r="40" fill="white" />
  <circle cx="340" cy="150" r="40" fill="white" />
  <circle cx="380" cy="150" r="40" fill="white" />
  <!-- Helicopter Group -->
  <g transform="translate(120, 100)">
    <!-- Tail Boom -->
    <path d="M120 40 L220 30 L220 45 L120 55 Z" fill="#34495e" />
    <!-- Tail Rotor -->
    <rect x="220" y="25" width="5" height="25" rx="2" fill="#7f8c8d" />
    <ellipse cx="225" cy="37" rx="15" ry="5" fill="rgba(127, 140, 141, 0.5)" />
    <!-- Main Body (Fuselage) -->
    <path d="M20 40 Q 0 40 0 70 Q 0 110 40 110 L110 110 Q 130 110 130 80 Q 130 40 110 40 Z" fill="#e74c3c" />
    <!-- Cockpit Window -->
    <path d="M30 50 Q 20 50 20 70 Q 20 90 40 90 L70 90 Q 80 90 80 50 Z" fill="#aed6f4" stroke="#34495e" stroke-width="2" />
    <!-- Landing Skids -->
    <line x1="40" y1="110" x2="40" y2="130" stroke="#34495e" stroke-width="4" />
    <line x1="90" y1="110" x2="90" y2="130" stroke="#34495e" stroke-width="4" />
    <path d="M30 130 L110 130" stroke="#34495e" stroke-width="6" stroke-linecap="round" />
    <!-- Rotor Mast -->
    <rect x="60" y="20" width="10" height="20" fill="#34495e" />
    <!-- Main Rotor Blades (with animation) -->
    <g>
      <ellipse cx="65" cy="20" rx="100" ry="10" fill="rgba(52, 73, 94, 0.6)" />
      <animate attributeName="opacity" values="0.4;1;0.4" dur="0.1s" repeatCount="indefinite" />
    </g>
  </g>
</svg>
```

</details>

## 5. `glm-5.1:cloud`

**SVG size:** 10242 characters  
**Complexity:** Very high  
**Shape elements:** 73  
**Distinct colors:** 19  
**Raw response:** 14827 characters  
**Generation time:** 60.0s

![glm-5.1:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/glm-5.1_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders clouds -- the model placed the helicopter in a sky scene
- Renders sky -- the model captured the flying context
- Renders stars -- the model created a night scene
- Renders ground/terrain -- the model placed the helicopter above a surface
- Renders exhaust/smoke -- the model captured engine output detail
- Renders navigation lights -- the model captured aviation lighting detail

With 10,242 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 900 700" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flying helicopter SVG illustration" width="100%" height="auto">
  <defs>
    <!-- Gradients -->
    <linearGradient id="bodyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f0f4f8"/>
      <stop offset="40%" stop-color="#c8d0da"/>
      <stop offset="100%" stop-color="#8898a8"/>
    </linearGradient>
    <linearGradient id="bodySideGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#b0bcc8"/>
      <stop offset="100%" stop-color="#6a7a8a"/>
    </linearGradient>
    <linearGradient id="redStripe" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ff5a4a"/>
      <stop offset="100%" stop-color="#b82820"/>
    </linearGradient>
    <linearGradient id="glassGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#a0e0ff"/>
      <stop offset="50%" stop-color="#60b8e0"/>
      <stop offset="100%" stop-color="#3090c0"/>
    </linearGradient>
    <linearGradient id="tailGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#c8d0da"/>
      <stop offset="100%" stop-color="#8898a8"/>
    </linearGradient>
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0b1a2e"/>
      <stop offset="50%" stop-color="#1a3a5c"/>
      <stop offset="100%" stop-color="#2a6a9a"/>
    </linearGradient>
    <linearGradient id="groundGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1a4a2a"/>
      <stop offset="100%" stop-color="#0a2a16"/>
    </linearGradient>
    <radialGradient id="searchlightGrad" cx="0.5" cy="0" r="0.8">
      <stop offset="0%" stop-color="#ffffcc" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#ffffcc" stop-opacity="0"/>
    </radialGradient>
    <!-- Filters -->
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="3" dy="6" stdDeviation="8" flood-color="#000000" flood-opacity="0.3"/>
    </filter>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <!-- Sky background -->
  <rect width="900" height="700" fill="url(#skyGrad)"/>
  <!-- Stars -->
  <g opacity="0.6">
    <circle cx="80" cy="60" r="1.5" fill="#ffffff"/>
    <circle cx="200" cy="40" r="1" fill="#ffffff"/>
    <circle cx="350" cy="80" r="1.2" fill="#ffffff"/>
    <circle cx="500" cy="30" r="1" fill="#ffffff"/>
    <circle cx="650" cy="70" r="1.5" fill="#ffffff"/>
    <circle cx="780" cy="50" r="1" fill="#ffffff"/>
    <circle cx="120" cy="130" r="1" fill="#ffffff"/>
    <circle cx="420" cy="110" r="1.3" fill="#ffffff"/>
    <circle cx="600" cy="120" r="1" fill="#ffffff"/>
    <circle cx="830" cy="100" r="1.5" fill="#ffffff"/>
    <circle cx="50" cy="180" r="1" fill="#ffffff"/>
    <circle cx="750" cy="160" r="1.2" fill="#ffffff"/>
  </g>
  <!-- Clouds (background) -->
  <g class="cloud1" opacity="0.15">
    <ellipse cx="950" cy="180" rx="100" ry="30" fill="#ffffff"/>
    <ellipse cx="920" cy="170" rx="60" ry="22" fill="#ffffff"/>
    <ellipse cx="980" cy="170" rx="50" ry="20" fill="#ffffff"/>
  </g>
  <g class="cloud2" opacity="0.1">
    <ellipse cx="1050" cy="300" rx="120" ry="28" fill="#ffffff"/>
    <ellipse cx="1020" cy="290" rx="70" ry="20" fill="#ffffff"/>
    <ellipse cx="1090" cy="292" rx="55" ry="18" fill="#ffffff"/>
  </g>
  <g class="cloud3" opacity="0.12">
    <ellipse cx="1000" cy="450" rx="90" ry="25" fill="#ffffff"/>
    <ellipse cx="970" cy="442" rx="55" ry="18" fill="#ffffff"/>
    <ellipse cx="1030" cy="440" rx="45" ry="16" fill="#ffffff"/>
  </g>
  <!-- Ground -->
  <ellipse cx="450" cy="680" rx="500" ry="40" fill="url(#groundGrad)" opacity="0.4"/>
  <!-- ===== HELICOPTER GROUP ===== -->
  <g class="helicopter-group" filter="url(#shadow)">
    <!-- Searchlight beam -->
    <polygon points="430,420 390,580 510,580 470,420" fill="url(#searchlightGrad)" opacity="0.25"/>
    <!-- Downwash effect -->
    <g class="downwash" opacity="0.3">
      <line x1="400" y1="430" x2="380" y2="470" stroke="#ffffff" stroke-width="1" opacity="0"/>
      <line x1="440" y1="430" x2="440" y2="475" stroke="#ffffff" stroke-width="1" opacity="0"/>
      <line x1="470" y1="430" x2="490" y2="470" stroke="#ffffff" stroke-width="1" opacity="0"/>
      <line x1="420" y1="430" x2="410" y2="472" stroke="#ffffff" stroke-width="1" opacity="0"/>
    </g>
    <!-- Landing skids -->
    <g stroke="#5a6a7a" stroke-width="3" fill="none" stroke-linecap="round">
      <!-- Left skid -->
      <line x1="370" y1="405" x2="370" y2="420"/>
      <line x1="370" y1="420" x2="350" y2="425"/>
      <line x1="350" y1="425" x2="510" y2="425"/>
      <!-- Right skid -->
      <line x1="500" y1="405" x2="500" y2="420"/>
      <line x1="500" y1="420" x2="510" y2="425"/>
      <!-- Cross bars -->
      <line x1="370" y1="420" x2="500" y2="420" stroke-width="2.5"/>
      <line x1="370" y1="405" x2="500" y2="405" stroke-width="2.5"/>
    </g>
    <!-- Tail boom -->
    <path d="M530,290 C580,280 680,240 730,225 L740,230 C690,250 580,290 530,300 Z"
          fill="url(#tailGrad)" stroke="#7888a0" stroke-width="1"/>
    <!-- Tail boom red stripe -->
    <path d="M540,295 C590,285 670,248 720,232 L722,237 C670,255 590,292 540,302 Z"
          fill="url(#redStripe)" opacity="0.7"/>
    <!-- Tail fin / vertical stabilizer -->
    <path d="M720,225 L730,180 L745,185 L740,230 Z"
          fill="#c8d0da" stroke="#7888a0" stroke-width="1"/>
    <path d="M725,225 L730,195 L738,198 L735,228 Z"
          fill="url(#redStripe)" opacity="0.6"/>
    <!-- Horizontal stabilizer -->
    <path d="M710,245 L740,238 L748,242 L718,250 Z"
          fill="#b0bcc8" stroke="#7888a0" stroke-width="1"/>
    <!-- Tail rotor -->
    <g class="tail-rotor">
      <rect x="727" y="183" width="6" height="45" rx="2" fill="#4a5a6a" opacity="0.8"/>
      <rect x="724" y="183" width="12" height="5" rx="1.5" fill="#6a7a8a"/>
      <rect x="724" y="223" width="12" height="5" rx="1.5" fill="#6a7a8a"/>
    </g>
    <!-- Main fuselage / body -->
    <path d="M340,310
             C340,270 370,250 420,245
             L530,265
             C560,270 560,310 540,320
             C540,340 540,370 535,395
             C530,410 460,420 430,420
             C400,420 360,410 350,395
             C340,370 340,340 340,310 Z"
          fill="url(#bodyGrad)" stroke="#7888a0" stroke-width="1.5"/>
    <!-- Body side shadow -->
    <path d="M340,310
             C340,270 370,250 420,245
             L530,265
             C560,270 560,310 540,320
             L540,320
             C520,325 440,330 380,330
             C350,330 340,325 340,310 Z"
          fill="url(#bodySideGrad)" opacity="0.3"/>
    <!-- Red stripe on body -->
    <path d="M345,335
             C345,330 370,325 420,322
             L540,320
             L540,330
             C440,335 370,340 345,340 Z"
          fill="url(#redStripe)" opacity="0.85"/>
    <!-- Cockpit glass -->
    <path d="M350,305
             C350,280 375,260 405,255
             L430,252
             L430,305
             Z"
          fill="url(#glassGrad)" stroke="#4090b0" stroke-width="1.5" opacity="0.9"/>
    <!-- Glass highlight -->
    <path d="M360,295
             C360,280 380,268 400,262
             L410,260
             L410,295
             Z"
          fill="#ffffff" opacity="0.25"/>
    <!-- Side windows -->
    <rect x="445" y="272" width="40" height="28" rx="5" fill="url(#glassGrad)" stroke="#4090b0" stroke-width="1.2" opacity="0.85"/>
    <rect x="495" y="275" width="28" height="25" rx="4" fill="url(#glassGrad)" stroke="#4090b0" stroke-width="1.2" opacity="0.85"/>
    <!-- Window highlights -->
    <rect x="448" y="275" width="15" height="10" rx="3" fill="#ffffff" opacity="0.2"/>
    <rect x="498" y="278" width="10" height="8" rx="2" fill="#ffffff" opacity="0.2"/>
    <!-- Door line -->
    <line x1="440" y1="260" x2="440" y2="400" stroke="#7888a0" stroke-width="1" opacity="0.5"/>
    <!-- Door handle -->
    <rect x="443" y="340" width="10" height="3" rx="1.5" fill="#5a6a7a"/>
    <!-- Engine housing on top -->
    <path d="M400,248 L460,248 L470,245 L470,240 L400,240 Z"
          fill="#b0bcc8" stroke="#7888a0" stroke-width="1"/>
    <path d="M410,240 L460,240 L465,237 L415,237 Z"
          fill="#a0acb8" stroke="#7888a0" stroke-width="0.8"/>
    <!-- Rotor mast -->
    <rect x="447" y="170" width="6" height="72" rx="2" fill="#5a6a7a"/>
    <rect x="444" y="235" width="12" height="8" rx="2" fill="#6a7a8a"/>
    <!-- Rotor hub -->
    <circle cx="450" cy="155" r="8" fill="#4a5a6a" stroke="#3a4a5a" stroke-width="1.5"/>
    <circle cx="450" cy="155" r="4" fill="#3a4a5a"/>
    <!-- Main rotor blades (spinning) -->
    <g class="main-rotor">
      <rect x="210" y="150" width="480" height="8" rx="3" fill="#5a6a7a" opacity="0.85"/>
      <rect x="210" y="151" width="480" height="3" rx="1" fill="#8a9aaa" opacity="0.4"/>
    </g>
    <!-- Rotor blur disc (visual spinning effect) -->
    <ellipse class="rotor-blur" cx="450" cy="155" rx="240" ry="6" fill="#8898a8" opacity="0.2"/>
    <!-- Exhaust -->
    <g class="exhaust">
      <ellipse cx="548" cy="385" rx="8" ry="5" fill="#ff8844" opacity="0.5"/>
      <ellipse cx="558" cy="388" rx="12" ry="6" fill="#ff6633" opacity="0.3"/>
      <ellipse cx="572" cy="391" rx="16" ry="7" fill="#ff4422" opacity="0.15"/>
    </g>
    <!-- Navigation lights -->
    <circle cx="340" cy="310" r="4" fill="#ff2222" opacity="0.9" filter="url(#glow)"/>
    <circle cx="540" cy="320" r="4" fill="#22ff44" opacity="0.9" filter="url(#glow)"/>
    <circle cx="740" cy="225" r="3" fill="#ffffff" opacity="0.8" filter="url(#glow)"/>
    <!-- Antenna -->
    <line x1="480" y1="245" x2="485" y2="220" stroke="#5a6a7a" stroke-width="1.5" stroke-linecap="round"/>
    <circle cx="485" cy="218" r="2" fill="#ff4444" opacity="0.8"/>
  </g>
  <!-- Foreground clouds (in front) -->
  <g opacity="0.08">
    <ellipse cx="300" cy="550" rx="200" ry="40" fill="#ffffff"/>
    <ellipse cx="650" cy="600" rx="150" ry="30" fill="#ffffff"/>
  </g>
</svg>
```

</details>

## 6. `glm-5.2:cloud`

**SVG size:** 8044 characters  
**Complexity:** High  
**Shape elements:** 43  
**Distinct colors:** 13  
**Raw response:** 8041 characters  
**Generation time:** 30.0s

![glm-5.2:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/glm-5.2_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Uses `<use>` references to reuse defined elements -- ideal for repeating rotor blades or cloud patterns
- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Defines reusable `<symbol>` elements -- shows mature SVG authoring
- Includes animation elements -- the model attempted motion (e.g., spinning rotors, hovering, cloud drift)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders clouds -- the model placed the helicopter in a sky scene
- Renders sky -- the model captured the flying context
- Renders ground/terrain -- the model placed the helicopter above a surface
- Includes animation -- the model captured the dynamic motion of flying (spinning rotors, hovering)
- Renders exhaust/smoke -- the model captured engine output detail
- Renders navigation lights -- the model captured aviation lighting detail

With 8,044 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" text-rendering="geometricPrecision" font-family="ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif" width="100%" height="auto">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#a9d6e8"/>
      <stop offset="0.55" stop-color="#cfe7f0"/>
      <stop offset="1" stop-color="#eaf4f1"/>
    </linearGradient>
    <linearGradient id="body" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#e8412c"/>
      <stop offset="1" stop-color="#b12a1c"/>
    </linearGradient>
    <linearGradient id="bodyDark" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#9c2418"/>
      <stop offset="1" stop-color="#6e180e"/>
    </linearGradient>
    <linearGradient id="window" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#bfe9f5"/>
      <stop offset="1" stop-color="#5b8fa3"/>
    </linearGradient>
    <radialGradient id="rotorBlur" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#222" stop-opacity="0.55"/>
      <stop offset="0.7" stop-color="#222" stop-opacity="0.18"/>
      <stop offset="1" stop-color="#222" stop-opacity="0"/>
    </radialGradient>
    <symbol id="cloud" viewBox="0 0 120 40">
      <g fill="#ffffff">
        <ellipse cx="30" cy="25" rx="22" ry="14" />
        <ellipse cx="55" cy="20" rx="28" ry="18" />
        <ellipse cx="85" cy="26" rx="24" ry="14" />
        <ellipse cx="100" cy="28" rx="16" ry="10" />
      </g>
    </symbol>
    <style>
      @media (prefers-reduced-motion: reduce){
        animate, animateTransform { display:none }
      }
    </style>
  </defs>
  <!-- Sky -->
  <rect width="800" height="500" fill="url(#sky)"/>
  <!-- Distant clouds (parallax) -->
  <g opacity="0.85">
    <use href="#cloud" x="60" y="90" width="160" height="54"/>
    <use href="#cloud" x="520" y="60" width="200" height="68"/>
    <use href="#cloud" x="380" y="160" width="120" height="42"/>
    <animateTransform attributeName="transform" type="translate"
      values="0 0; -40 0; 0 0" dur="22s" repeatCount="indefinite"/>
  </g>
  <g opacity="0.6">
    <use href="#cloud" x="200" y="220" width="100" height="34"/>
    <use href="#cloud" x="600" y="200" width="130" height="44"/>
    <animateTransform attributeName="transform" type="translate"
      values="0 0; -80 0; 0 0" dur="16s" repeatCount="indefinite"/>
  </g>
  <!-- Helicopter group with gentle bob -->
  <g>
    <animateTransform attributeName="transform" type="translate"
      values="0 0; 0 -10; 0 4; 0 -6; 0 0"
      keyTimes="0; 0.3; 0.55; 0.8; 1"
      dur="6s" repeatCount="indefinite"/>
    <!-- Motion streaks behind tail -->
    <g stroke="#ffffff" stroke-linecap="round" opacity="0.5">
      <line x1="120" y1="250" x2="200" y2="250" stroke-width="2"/>
      <line x1="100" y1="268" x2="195" y2="268" stroke-width="2"/>
      <line x1="140" y1="284" x2="210" y2="284" stroke-width="2"/>
      <animate attributeName="opacity" values="0.2;0.6;0.2" dur="1.2s" repeatCount="indefinite"/>
    </g>
    <!-- Shadow on ground-ish haze -->
    <ellipse cx="400" cy="395" rx="120" ry="10" fill="#000" opacity="0.08" />
    <!-- Landing skids -->
    <g stroke="#2b2b2b" stroke-width="5" stroke-linecap="round" fill="none">
      <line x1="280" y1="345" x2="520" y2="345"/>
      <line x1="300" y1="320" x2="300" y2="345"/>
      <line x1="500" y1="320" x2="500" y2="345"/>
    </g>
    <!-- Tail boom -->
    <path d="M 480 250 Q 560 240 650 250 L 660 270 Q 560 280 480 270 Z" fill="url(#body)"/>
    <path d="M 480 268 Q 560 278 650 270 L 650 274 Q 560 282 480 274 Z" fill="url(#bodyDark)" opacity="0.6"/>
    <!-- Tail fin / vertical stabilizer -->
    <path d="M 620 250 L 660 215 L 672 218 L 660 255 Z" fill="url(#body)"/>
    <path d="M 620 250 L 660 215 L 664 217 L 632 252 Z" fill="#ffffff" opacity="0.18"/>
    <!-- Tail rotor (spinning disc, side view) -->
    <g transform="translate(666 235)">
      <circle r="14" fill="url(#rotorBlur)">
        <animateTransform attributeName="transform" type="rotate"
          from="0" to="360" dur="0.12s" repeatCount="indefinite"/>
      </circle>
      <line x1="-12" y1="0" x2="12" y2="0" stroke="#1f1f1f" stroke-width="1.2" opacity="0.5"/>
    </g>
    <!-- Main fuselage -->
    <path d="M 280 245
             Q 270 215 310 205
             Q 360 195 410 200
             Q 470 205 495 230
             Q 505 255 495 285
             Q 480 320 410 325
             Q 340 330 305 315
             Q 275 300 280 245 Z"
          fill="url(#body)"/>
    <!-- Belly shadow -->
    <path d="M 295 305 Q 360 332 470 318 Q 490 305 485 295 Q 380 320 300 295 Z"
          fill="url(#bodyDark)" opacity="0.55"/>
    <!-- Cockpit windshield -->
    <path d="M 290 235 Q 285 215 315 210 Q 345 205 360 220 L 350 250 Q 320 250 290 250 Z"
          fill="url(#window)"/>
    <path d="M 295 235 Q 290 220 315 215 L 318 218 Q 298 224 300 240 Z"
          fill="#ffffff" opacity="0.35"/>
    <!-- Side window -->
    <path d="M 372 215 Q 395 210 420 215 L 418 245 Q 395 248 372 245 Z"
          fill="url(#window)" opacity="0.9"/>
    <path d="M 376 218 Q 392 215 410 218 L 405 222 Q 390 220 378 222 Z"
          fill="#ffffff" opacity="0.3"/>
    <!-- Door seam -->
    <path d="M 368 210 L 368 305" stroke="#6e180e" stroke-width="1.5" fill="none" opacity="0.6"/>
    <!-- Door handle -->
    <rect x="380" y="260" width="10" height="3" rx="1" fill="#3a0d06"/>
    <!-- Body highlight -->
    <path d="M 300 215 Q 360 200 460 212 Q 480 218 470 222 Q 380 210 305 222 Z"
          fill="#ffffff" opacity="0.22"/>
    <!-- Engine intake / detail -->
    <circle cx="455" cy="240" r="6" fill="#2b2b2b" />
    <circle cx="455" cy="240" r="3" fill="#444" />
    <!-- Exhaust -->
    <path d="M 470 270 L 495 268 L 498 280 L 472 282 Z" fill="#333"/>
    <path d="M 495 270 L 510 269 L 512 278 L 498 280 Z" fill="#1f1f1f" opacity="0.8"/>
    <!-- Heat shimmer -->
    <g opacity="0.4">
      <ellipse cx="520" cy="272" rx="14" ry="3" fill="#ffffff">
        <animate attributeName="opacity" values="0;0.5;0" dur="2s" repeatCount="indefinite"/>
      </ellipse>
    </g>
    <!-- Rotor mast -->
    <rect x="392" y="178" width="16" height="22" rx="3" fill="#2b2b2b"/>
    <rect x="394" y="180" width="12" height="18" rx="2" fill="#4a4a4a"/>
    <!-- Main rotor: spinning disc + faint blades -->
    <g>
      <!-- blur disc -->
      <ellipse cx="400" cy="175" rx="170" ry="10" fill="url(#rotorBlur)" />
      <ellipse cx="400" cy="175" rx="170" ry="3" fill="#222" opacity="0.25" />
      <!-- spinning blades (counter-rotating for motion feel) -->
      <g>
        <rect x="240" y="173" width="320" height="5" rx="2" fill="#1f1f1f" opacity="0.35"/>
        <rect x="240" y="173" width="320" height="5" rx="2" fill="#1f1f1f" opacity="0.35" transform="rotate(60 400 175)"/>
        <rect x="240" y="173" width="320" height="5" rx="2" fill="#1f1f1f" opacity="0.35" transform="rotate(120 400 175)"/>
        <animateTransform attributeName="transform" type="rotate"
          from="0 400 175" to="360 400 175" dur="0.18s" repeatCount="indefinite"/>
      </g>
    </g>
    <!-- Antenna -->
    <line x1="430" y1="200" x2="432" y2="188" stroke="#1f1f1f" stroke-width="1.5"/>
    <circle cx="432" cy="187" r="1.5" fill="#1f1f1f"/>
    <!-- Nav lights blink -->
    <circle cx="280" cy="248" r="2.5" fill="#ff3232">
      <animate attributeName="opacity" values="1;0.15;1" dur="1.4s" repeatCount="indefinite"/>
    </circle>
    <circle cx="668" cy="235" r="2.5" fill="#32ff6a">
      <animate attributeName="opacity" values="0.15;1;0.15" dur="1.4s" repeatCount="indefinite"/>
    </circle>
  </g>
  <!-- Foreground drifting wisps for depth -->
  <g opacity="0.45">
    <use href="#cloud" x="-80" y="320" width="220" height="60"/>
    <use href="#cloud" x="600" y="360" width="260" height="70"/>
    <animateTransform attributeName="transform" type="translate"
      values="0 0; -160 0; 0 0" dur="12s" repeatCount="indefinite"/>
  </g>
</svg>
```

</details>

## 7. `gpt-oss:120b-cloud`

**SVG size:** 1990 characters  
**Complexity:** Medium  
**Shape elements:** 19  
**Distinct colors:** 9  
**Raw response:** 2056 characters  
**Generation time:** 15.0s

![gpt-oss:120b-cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/gpt-oss_120b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders ground/terrain -- the model placed the helicopter above a surface

With 1,990 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 400 250"
    preserveAspectRatio="xMidYMid meet" width="100%" height="auto">
  <!-- Background (transparent) -->
  <rect width="100%" height="100%" fill="none"/>
  <!-- Main fuselage -->
  <path d="M80,130
           C80,115 120,115 150,130
           L260,130
           C280,130 300,150 300,150
           C290,120 260,110 240,110
           L150,110
           C120,110 80,115 80,130 Z"
        fill="#3a6ea5"
        stroke="#2c4a78"
        stroke-width="2"/>
  <!-- Tail boom -->
  <rect x="300" y="130" width="70" height="8" fill="#3a6ea5" stroke="#2c4a78" stroke-width="2"/>
  <!-- Tail rotor hub -->
  <circle cx="380" cy="134" r="6" fill="#444"/>
  <!-- Tail rotor blades -->
  <g transform="rotate(45 380 134)">
    <rect x="378" y="124" width="4" height="20" fill="#888"/>
    <rect x="378" y="144" width="4" height="20" fill="#888"/>
  </g>
  <g transform="rotate(-45 380 134)">
    <rect x="378" y="124" width="4" height="20" fill="#888"/>
    <rect x="378" y="144" width="4" height="20" fill="#888"/>
  </g>
  <!-- Main rotor hub -->
  <circle cx="200" cy="95" r="8" fill="#444"/>
  <!-- Main rotor blades -->
  <g stroke="#666" stroke-width="4" stroke-linecap="round">
    <line x1="200" y1="95" x2="130" y2="95"/>
    <line x1="200" y1="95" x2="270" y2="95"/>
    <line x1="200" y1="95" x2="200" y2="25"/>
    <line x1="200" y1="95" x2="200" y2="165"/>
  </g>
  <!-- Skids -->
  <line x1="110" y1="140" x2="110" y2="165" stroke="#555" stroke-width="6"/>
  <line x1="260" y1="140" x2="260" y2="165" stroke="#555" stroke-width="6"/>
  <line x1="110" y1="165" x2="260" y2="165" stroke="#555" stroke-width="6"/>
  <path d="M110,165 Q185,185 260,165" fill="none" stroke="#555" stroke-width="6"/>
  <!-- Windows -->
  <rect x="150" y="120" width="30" height="12" fill="#cce5ff" stroke="#99badd" stroke-width="1"/>
  <rect x="190" y="120" width="30" height="12" fill="#cce5ff" stroke="#99badd" stroke-width="1"/>
</svg>
```

</details>

## 8. `kimi-k2.6:cloud`

**SVG size:** 4052 characters  
**Complexity:** Medium  
**Shape elements:** 17  
**Distinct colors:** 9  
**Raw response:** 20744 characters  
**Generation time:** 60.0s

![kimi-k2.6:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/kimi-k2.6_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders sky -- the model captured the flying context
- Renders ground/terrain -- the model placed the helicopter above a surface
- Renders navigation lights -- the model captured aviation lighting detail

With 4,052 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" id="game-canvas" preserveAspectRatio="xMidYMid slice" width="100%" height="auto">
        <defs>
            <!-- Gradients -->
            <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#4facfe;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#00f2fe;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="heliBodyGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#3b82f6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#1d4ed8;stop-opacity:1" />
            </linearGradient>
            <linearGradient id="glassGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#e0f2fe;stop-opacity:0.8" />
                <stop offset="100%" style="stop-color:#bae6fd;stop-opacity:0.6" />
            </linearGradient>
            <filter id="shadow">
                <feDropShadow dx="0" dy="10" stdDeviation="5" flood-color="#000" flood-opacity="0.3"/>
            </filter>
        </defs>
        <!-- Sky Background -->
        <rect width="100%" height="100%" fill="url(#skyGradient)" />
        <!-- Scenery Group -->
        <g id="scenery-layer"></g>
        <!-- Searchlight Beam (Hidden by default) -->
        <path id="searchlight" d="M0,0 L-100,500 L100,500 Z" fill="rgba(255,255,200,0.3)" style="visibility: hidden; pointer-events: none;" />
        <!-- The Helicopter Group -->
        <g id="helicopter-group" filter="url(#shadow)">
            <!-- Landing Skids -->
            <path d="M 100 240 L 100 250 M 220 240 L 220 250 M 80 250 L 240 250" stroke="#333" stroke-width="6" fill="none" stroke-linecap="round" />
            <!-- Struts -->
            <path d="M 120 220 L 100 240 M 200 220 L 220 240" stroke="#555" stroke-width="4" fill="none" />
            <!-- Tail Boom -->
            <path d="M 220 180 L 380 160 L 380 130 L 220 160 Z" fill="#60a5fa" stroke="#1e3a8a" stroke-width="2"/>
            <!-- Vertical Stabilizer -->
            <path d="M 370 160 L 380 130 L 400 110 L 410 120 L 380 160 Z" fill="#3b82f6" stroke="#1e3a8a" stroke-width="2" />
            <!-- Tail Rotor -->
            <g id="tail-rotor-group">
                <circle cx="410" cy="115" r="3" fill="#333" />
                <rect id="tail-blade-1" x="410" y="95" width="4" height="40" fill="#1e40af" rx="2" class="tail-rotor-blade" />
                <rect id="tail-blade-2" x="400" y="113" width="40" height="4" fill="#1e40af" rx="2" class="tail-rotor-blade" />
            </g>
            <!-- Main Fuselage -->
            <path d="M 80 180 Q 80 120 150 120 L 220 130 Q 260 140 260 180 Q 260 220 220 220 L 120 220 Q 80 220 80 180 Z" fill="url(#heliBodyGradient)" stroke="#1e3a8a" stroke-width="2" />
            <!-- Cockpit Window -->
            <path d="M 150 125 L 200 135 Q 230 140 230 180 L 150 180 Z" fill="url(#glassGradient)" stroke="#7dd3fc" stroke-width="1" />
            <!-- Door/Details -->
            <rect x="110" y="150" width="40" height="50" rx="5" fill="none" stroke="#2563eb" stroke-width="2" />
            <!-- Main Rotor Mast -->
            <rect x="145" y="100" width="10" height="20" fill="#555" />
            <!-- Main Rotor Blades -->
            <g id="main-rotor-group">
                <circle cx="150" cy="120" r="6" fill="#333" />
                <!-- Blade 1 -->
                <path d="M 150 120 L 150 -40 L 155 -40 L 155 120 Z" fill="#1e3a8a" opacity="0.9" class="rotor-blade" />
                <!-- Blade 2 -->
                <path d="M 150 120 L 150 280 L 145 280 L 145 120 Z" fill="#1e3a8a" opacity="0.9" class="rotor-blade" />
                <!-- Spinning Disc Effect (Fast rotation simulation) -->
                <ellipse id="rotor-disc" cx="150" cy="120" rx="160" ry="20" fill="rgba(200,200,255,0.1)" stroke="rgba(255,255,255,0.3)" stroke-width="2" style="visibility:hidden"/>
            </g>
        </g>
    </svg>
```

</details>

## 9. `minimax-m2.7:cloud`

**SVG size:** 2608 characters  
**Complexity:** Medium  
**Shape elements:** 16  
**Distinct colors:** 10  
**Raw response:** 2935 characters  
**Generation time:** 15.0s

![minimax-m2.7:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/minimax-m2.7_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Includes animation elements -- the model attempted motion (e.g., spinning rotors, hovering, cloud drift)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders clouds -- the model placed the helicopter in a sky scene
- Renders ground/terrain -- the model placed the helicopter above a surface
- Includes animation -- the model captured the dynamic motion of flying (spinning rotors, hovering)

With 2,608 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="100%" height="auto">
  <defs>
    <!-- CSS Animation for the rotor spinning -->
    <style>
      .rotor {
        transform-box: fill-box;
        transform-origin: center;
        animation: spin 0.2s linear infinite;
      }
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    </style>
    <!-- Gradient for the cockpit window -->
    <linearGradient id="windowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#a2d9ff;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2e86c1;stop-opacity:1" />
    </linearGradient>
  </defs>
  <!-- Background clouds / speed lines to imply motion -->
  <g fill="#e0e0e0" opacity="0.6">
    <circle cx="50" cy="300" r="10" />
    <circle cx="80" cy="320" r="15" />
    <circle cx="320" cy="100" r="8" />
    <circle cx="350" cy="80" r="12" />
    <path d="M 0 250 L 40 250 M 10 270 L 70 270" stroke="#e0e0e0" stroke-width="4" stroke-linecap="round"/>
  </g>
  <!-- Tail Section -->
  <!-- Tail Fin -->
  <path d="M 300 160 L 320 140 L 320 180 Z" fill="#c0392b" />
  <!-- Tail Boom -->
  <rect x="180" y="160" width="120" height="20" rx="5" fill="#e74c3c" />
  <!-- Tail Rotor (Spinning) -->
  <g class="rotor">
    <ellipse cx="320" cy="160" rx="4" ry="25" fill="#7f8c8d" />
  </g>
  <!-- Main Body -->
  <!-- Fuselage -->
  <path d="M 60 180 
           C 60 140, 100 120, 160 120 
           L 200 120 
           C 220 120, 220 160, 200 160 
           L 160 160 
           C 140 160, 140 180, 160 180 
           L 60 180 Z" 
        fill="#e74c3c" />
  <!-- Cockpit Window -->
  <path d="M 100 125 
           C 120 125, 130 140, 145 140 
           C 145 160, 120 165, 100 165 
           Z" 
        fill="url(#windowGrad)" stroke="#ecf0f1" stroke-width="2" />
  <!-- Landing Skids -->
  <path d="M 80 210 L 80 230 Q 80 240 100 240 L 170 240" 
        fill="none" stroke="#34495e" stroke-width="6" stroke-linecap="round" />
  <!-- Struts connecting skids to body -->
  <path d="M 110 185 L 110 230 M 160 185 L 160 230" 
        stroke="#34495e" stroke-width="4" />
  <!-- Main Rotor Mast -->
  <rect x="135" y="90" width="10" height="30" fill="#555" />
  <!-- Main Rotor Blades (Spinning) -->
  <g class="rotor">
    <!-- Rotor Hub -->
    <circle cx="140" cy="90" r="8" fill="#333" />
    <!-- Blades -->
    <ellipse cx="140" cy="90" rx="120" ry="15" fill="#95a5a6" opacity="0.8" />
    <ellipse cx="140" cy="90" rx="15" ry="120" fill="#95a5a6" opacity="0.8" />
  </g>
</svg>
```

</details>

## 10. `minimax-m3:cloud`

**SVG size:** 4053 characters  
**Complexity:** High  
**Shape elements:** 35  
**Distinct colors:** 13  
**Raw response:** 4812 characters  
**Generation time:** 15.0s

![minimax-m3:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/minimax-m3_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Renders actual text (labels, title, or decorative text)
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders clouds -- the model placed the helicopter in a sky scene
- Renders sky -- the model captured the flying context
- Renders a sun/moon -- the model added celestial detail
- Renders ground/terrain -- the model placed the helicopter above a surface

With 4,053 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" width="100%" height="auto">
  <!-- Sky background -->
  <defs>
    <linearGradient id="sky" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#87CEEB;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#E0F6FF;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#E63946;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#A4161A;stop-opacity:1" />
    </linearGradient>
    <radialGradient id="sun" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#FFF9C4;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFD54F;stop-opacity:0" />
    </radialGradient>
  </defs>
  <!-- Sky -->
  <rect width="600" height="400" fill="url(#sky)"/>
  <!-- Sun glow -->
  <circle cx="500" cy="80" r="60" fill="url(#sun)"/>
  <circle cx="500" cy="80" r="25" fill="#FFEB3B"/>
  <!-- Clouds -->
  <g opacity="0.8">
    <ellipse cx="100" cy="80" rx="40" ry="15" fill="white"/>
    <ellipse cx="120" cy="75" rx="30" ry="12" fill="white"/>
    <ellipse cx="85" cy="78" rx="25" ry="10" fill="white"/>
    <ellipse cx="380" cy="120" rx="35" ry="13" fill="white"/>
    <ellipse cx="400" cy="115" rx="28" ry="11" fill="white"/>
  </g>
  <!-- Main rotor blur (motion effect) -->
  <ellipse cx="300" cy="155" rx="110" ry="4" fill="#333" opacity="0.3"/>
  <ellipse cx="300" cy="155" rx="100" ry="3" fill="#555" opacity="0.5"/>
  <ellipse cx="300" cy="155" rx="90" ry="2" fill="#777" opacity="0.6"/>
  <!-- Rotor mast -->
  <rect x="295" y="155" width="10" height="30" fill="#333"/>
  <!-- Helicopter body -->
  <!-- Tail boom -->
  <rect x="420" y="195" width="120" height="22" fill="url(#bodyGrad)" rx="3"/>
  <!-- Main cabin -->
  <ellipse cx="320" cy="210" rx="130" ry="55" fill="url(#bodyGrad)"/>
  <!-- Cabin top -->
  <ellipse cx="280" cy="185" rx="60" ry="25" fill="url(#bodyGrad)"/>
  <!-- Cockpit window -->
  <path d="M 210 200 Q 210 180 235 180 Q 260 180 260 200 L 260 220 Q 235 225 210 220 Z" fill="#4FC3F7" opacity="0.8" stroke="#0277BD" stroke-width="2"/>
  <!-- Side window -->
  <ellipse cx="320" cy="195" rx="40" ry="15" fill="#4FC3F7" opacity="0.8" stroke="#0277BD" stroke-width="2"/>
  <!-- Door -->
  <rect x="340" y="205" width="50" height="45" fill="#B71C1C" stroke="#7F0000" stroke-width="2" rx="3"/>
  <circle cx="380" cy="227" r="3" fill="#333"/>
  <!-- Stripe decoration -->
  <rect x="200" y="225" width="290" height="8" fill="#FFD700" opacity="0.9"/>
  <!-- Landing skids -->
  <rect x="220" y="265" width="180" height="6" fill="#424242" rx="2"/>
  <rect x="240" y="245" width="6" height="30" fill="#424242"/>
  <rect x="374" y="245" width="6" height="30" fill="#424242"/>
  <!-- Front skid support -->
  <rect x="225" y="245" width="6" height="25" fill="#424242"/>
  <rect x="389" y="245" width="6" height="25" fill="#424242"/>
  <!-- Tail rotor -->
  <circle cx="540" cy="205" r="5" fill="#333"/>
  <ellipse cx="540" cy="205" rx="3" ry="20" fill="#333" opacity="0.4"/>
  <ellipse cx="540" cy="205" rx="20" ry="3" fill="#333" opacity="0.4"/>
  <ellipse cx="540" cy="205" rx="2" ry="15" fill="#555" opacity="0.6"/>
  <!-- Tail fin -->
  <path d="M 530 195 L 540 175 L 545 195 Z" fill="url(#bodyGrad)"/>
  <!-- Motion lines showing flying -->
  <g stroke="#666" stroke-width="2" opacity="0.5">
    <line x1="540" y1="195" x2="560" y2="190"/>
    <line x1="540" y1="205" x2="565" y2="205"/>
    <line x1="540" y1="215" x2="560" y2="220"/>
  </g>
  <!-- Main rotor blades (showing rotation) -->
  <g transform="rotate(15 300 155)">
    <rect x="200" y="153" width="200" height="4" fill="#222"/>
  </g>
  <g transform="rotate(-20 300 155)">
    <rect x="200" y="153" width="200" height="4" fill="#222" opacity="0.7"/>
  </g>
  <!-- Title text -->
  <text x="300" y="370" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#333" text-anchor="middle" opacity="0.7">Flying Helicopter</text>
</svg>
```

</details>

## 11. `nemotron-3-super:cloud`

**SVG size:** 796 characters  
**Complexity:** Compact  
**Shape elements:** 7  
**Distinct colors:** 4  
**Raw response:** 794 characters  
**Generation time:** 15.0s

![nemotron-3-super:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/nemotron-3-super_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders a tail -- the model captured the helicopter's tail structure

With 796 characters of SVG markup, this is a compact output -- minimal but potentially efficient for web embedding.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 150"
     xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <g transform="translate(100,75)">
    <!-- Helicopter body -->
    <ellipse cx="0" cy="0" rx="40" ry="20" fill="#8B4513"/>
    <!-- Cabin/windows -->
    <rect x="-20" y="-30" width="40" height="20" rx="5" fill="#ADD8E6"/>
    <!-- Tail boom -->
    <line x1="40" y1="0" x2="80" y2="0"
          stroke="#8B4513" stroke-width="4"/>
    <!-- Tail rotor -->
    <circle cx="80" cy="0" r="8" fill="#555555"/>
    <!-- Main rotor hub -->
    <circle cx="0" cy="-30" r="6" fill="#333"/>
    <!-- Main rotor blades -->
    <line x1="-30" y1="-30" x2="30" y2="-30"
          stroke="#333" stroke-width="4"/>
    <line x1="0" y1="-50" x2="0" y2="-10"
          stroke="#333" stroke-width="4"/>
  </g>
</svg>
```

</details>

## 12. `nemotron-3-ultra:cloud`

**SVG size:** 7241 characters  
**Complexity:** High  
**Shape elements:** 39  
**Distinct colors:** 15  
**Raw response:** 8807 characters  
**Generation time:** 30.0s

![nemotron-3-ultra:cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/nemotron-3-ultra_cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders stars -- the model created a night scene
- Renders navigation lights -- the model captured aviation lighting detail

With 7,241 characters of SVG markup, this is a detailed output that balances richness with readability.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300" width="100%" height="auto">
  <!-- Definitions for gradients and blur effects -->
  <defs>
    <!-- Metallic body gradient -->
    <linearGradient id="bodyGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#e0e0e0;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#ffffff;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#a0a0a0;stop-opacity:1" />
    </linearGradient>
    <!-- Dark window gradient -->
    <linearGradient id="windowGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a3a5c;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0d1b2a;stop-opacity:1" />
    </linearGradient>
    <!-- Rotor blur gradient (for motion effect) -->
    <radialGradient id="rotorBlur" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
      <stop offset="0%" style="stop-color:#333333;stop-opacity:0.6" />
      <stop offset="70%" style="stop-color:#333333;stop-opacity:0.1" />
      <stop offset="100%" style="stop-color:#333333;stop-opacity:0" />
    </radialGradient>
    <!-- Tail rotor blur -->
    <linearGradient id="tailRotorBlur" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#333333;stop-opacity:0" />
      <stop offset="50%" style="stop-color:#333333;stop-opacity:0.5" />
      <stop offset="100%" style="stop-color:#333333;stop-opacity:0" />
    </linearGradient>
    <!-- Drop shadow filter -->
    <filter id="dropShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
      <feOffset dx="2" dy="4" result="offsetblur"/>
      <feComponentTransfer>
        <feFuncA type="linear" slope="0.3"/>
      </feComponentTransfer>
      <feMerge> 
        <feMergeNode in="offsetblur"/>
        <feMergeNode in="SourceGraphic"/> 
      </feMerge>
    </filter>
  </defs>
  <!-- Main Group with Shadow -->
  <g filter="url(#dropShadow)">
    <!-- TAIL BOOM -->
    <g id="tailBoom">
      <!-- Horizontal stabilizer fins -->
      <path d="M460 155 L520 130 L520 135 L460 160 Z" fill="#c0c0c0" stroke="#999" stroke-width="1"/>
      <path d="M460 175 L520 200 L520 195 L460 170 Z" fill="#c0c0c0" stroke="#999" stroke-width="1"/>
      <!-- Tail boom tube -->
      <rect x="340" y="155" width="200" height="20" rx="10" fill="url(#bodyGradient)" stroke="#999" stroke-width="1"/>
      <!-- Tail rotor hub -->
      <circle cx="540" cy="165" r="12" fill="#333" stroke="#111" stroke-width="1"/>
      <!-- TAIL ROTOR (Blurred for motion) -->
      <rect x="520" y="160" width="40" height="10" rx="5" fill="url(#tailRotorBlur)" transform-origin="540 165"/>
      <rect x="520" y="160" width="40" height="10" rx="5" fill="url(#tailRotorBlur)" transform="rotate(60, 540, 165)"/>
      <rect x="520" y="160" width="40" height="10" rx="5" fill="url(#tailRotorBlur)" transform="rotate(120, 540, 165)"/>
    </g>
    <!-- MAIN FUSELAGE (BODY) -->
    <g id="fuselage">
      <!-- Belly / Skid attachment points -->
      <path d="M120 220 Q160 240 240 230 L240 220 Z" fill="#555" opacity="0.5"/>
      <!-- Main Body Shape -->
      <path d="
        M80 180 
        C 60 180, 50 150, 80 130 
        L 300 130 
        C 330 130, 340 145, 340 165 
        L 340 185 
        L 80 185 
        Z" 
        fill="url(#bodyGradient)" stroke="#999" stroke-width="1.5"/>
      <!-- Engine Housing (Top rear) -->
      <path d="M260 110 Q280 90 310 110 L310 130 L260 130 Z" fill="#d0d0d0" stroke="#999" stroke-width="1"/>
      <rect x="270" y="100" width="30" height="15" rx="3" fill="#444" opacity="0.8"/> <!-- Intake -->
      <!-- Cockpit Windows -->
      <path d="M95 135 Q110 115 180 115 L180 170 Q160 180 110 175 Z" fill="url(#windowGradient)" stroke="#000" stroke-width="0.5"/>
      <path d="M185 135 Q220 115 280 120 L280 170 Q250 180 200 175 Z" fill="url(#windowGradient)" stroke="#000" stroke-width="0.5"/>
      <!-- Sliding Door Window (Side) -->
      <rect x="200" y="155" width="70" height="25" rx="3" fill="url(#windowGradient)" stroke="#000" stroke-width="0.5"/>
      <!-- Accent Stripe -->
      <path d="M90 175 L320 175" stroke="#e63946" stroke-width="4" stroke-linecap="round" opacity="0.9"/>
      <path d="M90 165 L320 165" stroke="#1d3557" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
    </g>
    <!-- LANDING SKIDS -->
    <g id="skids" stroke="#444" stroke-width="6" fill="none" stroke-linecap="round">
      <!-- Front Skid -->
      <path d="M110 200 L110 230 Q130 240 160 235"/>
      <path d="M110 230 L160 235" stroke-width="4"/> 
      <!-- Cross brace -->
      <line x1="115" y1="220" x2="140" y2="220" stroke-width="3"/>
      <!-- Rear Skid -->
      <path d="M280 200 L280 230 Q260 240 230 235"/>
      <path d="M280 230 L230 235" stroke-width="4"/>
      <!-- Cross brace -->
      <line x1="275" y1="220" x2="250" y2="220" stroke-width="3"/>
      <!-- Connecting tubes -->
      <line x1="160" y1="235" x2="230" y2="235" stroke="#666" stroke-width="3"/>
    </g>
    <!-- MAIN ROTOR MAST -->
    <g id="mast">
      <rect x="185" y="90" width="10" height="50" fill="#555" stroke="#333" stroke-width="1"/>
      <circle cx="190" cy="95" r="15" fill="#333" stroke="#111" stroke-width="1"/>
      <circle cx="190" cy="95" r="8" fill="#555"/> <!-- Hub center -->
    </g>
    <!-- MAIN ROTOR BLADES (Blurred Disk for Motion) -->
    <g id="mainRotor" transform-origin="190 95">
      <!-- Blur Disk -->
      <ellipse cx="190" cy="95" rx="180" ry="15" fill="url(#rotorBlur)" transform="rotate(0, 190, 95)"/>
      <ellipse cx="190" cy="95" rx="180" ry="15" fill="url(#rotorBlur)" transform="rotate(45, 190, 95)"/>
      <ellipse cx="190" cy="95" rx="180" ry="15" fill="url(#rotorBlur)" transform="rotate(90, 190, 95)"/>
      <ellipse cx="190" cy="95" rx="180" ry="15" fill="url(#rotorBlur)" transform="rotate(135, 190, 95)"/>
      <!-- Static Blade Tips (Hint of shape) - Optional, adds realism to blur -->
      <g stroke="#222" stroke-width="3" stroke-linecap="round" opacity="0.4">
        <line x1="190" y1="95" x2="360" y2="95" />
        <line x1="190" y1="95" x2="190" y2="-80" />
        <line x1="190" y1="95" x2="20" y2="95" />
        <line x1="190" y1="95" x2="190" y2="270" />
      </g>
    </g>
    <!-- DETAILS: Lights &amp; Antennas -->
    <g id="details">
      <!-- Navigation Lights -->
      <circle cx="75" cy="165" r="4" fill="#e63946" filter="url(#glow)"/> <!-- Port Red -->
      <circle cx="335" cy="165" r="4" fill="#2a9d8f" filter="url(#glow)"/> <!-- Starboard Green -->
      <circle cx="535" cy="165" r="3" fill="#ffffff" filter="url(#glow)"/> <!-- Tail White -->
      <!-- Antenna -->
      <path d="M310 110 Q320 70 315 50" stroke="#555" stroke-width="2" fill="none" stroke-linecap="round"/>
      <circle cx="315" cy="50" r="3" fill="#e63946"/>
    </g>
  </g> <!-- End Shadow Group -->
  <!-- Glow Filter for Lights -->
  <defs>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
</svg>
```

</details>

## 13. `qwen3.5:397b-cloud`

**SVG size:** 3707 characters  
**Complexity:** Medium  
**Shape elements:** 22  
**Distinct colors:** 7  
**Raw response:** 4003 characters  
**Generation time:** 15.0s

![qwen3.5:397b-cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/qwen3.5_397b-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth (great for the helicopter body and sky)
- Applies SVG filters (e.g., drop shadows, blurs) for atmospheric effects
- Uses transforms (translate, rotate, scale) for positioning the helicopter and orienting rotors
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- Renders "helicopter" in text/comments -- the model captured the subject
- Renders a fuselage/body -- the model captured the helicopter's main structure
- Renders a cockpit/cabin -- the model captured where the pilot sits
- Renders windows/glass -- the model captured transparency detail
- Renders rotors -- the model captured the helicopter's most distinctive feature
- Renders rotor blades -- the model captured the lifting surface detail
- Renders a main rotor -- the model captured the primary lift mechanism
- Renders a tail rotor/boom -- the model captured the anti-torque mechanism
- Renders landing skids/gear -- the model captured the helicopter's ground contact
- Renders a tail -- the model captured the helicopter's tail structure
- Renders a tail fin/stabilizer -- the model captured fine aerodynamic detail
- Renders ground/terrain -- the model placed the helicopter above a surface
- Renders navigation lights -- the model captured aviation lighting detail

With 3,707 characters of SVG markup, this is a balanced output -- enough detail to be recognizable without being bloated.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg" width="100%" height="auto">
  <!-- Definitions for Gradients and Filters -->
  <defs>
    <!-- Body Gradient -->
    <linearGradient id="bodyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#3498db;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2980b9;stop-opacity:1" />
    </linearGradient>
    <!-- Cockpit Glass Gradient -->
    <linearGradient id="glassGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#2c3e50;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#1a252f;stop-opacity:1" />
    </linearGradient>
    <!-- Rotor Blur Filter -->
    <filter id="motionBlur" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="4, 0" />
    </filter>
  </defs>
  <!-- Wind / Speed Lines (Background) -->
  <g stroke="#bdc3c7" stroke-width="3" stroke-linecap="round" opacity="0.6">
    <line x1="50" y1="150" x2="120" y2="150" />
    <line x1="20" y1="200" x2="80" y2="200" />
    <line x1="80" y1="250" x2="140" y2="250" />
    <line x1="400" y1="180" x2="460" y2="180" />
    <line x1="420" y1="220" x2="480" y2="220" />
  </g>
  <!-- Main Group: Helicopter -->
  <g transform="translate(50, 50)">
    <!-- Tail Rotor (Back) -->
    <g transform="translate(360, 110)">
       <!-- Rotor blur circle -->
       <ellipse cx="0" cy="0" rx="15" ry="35" fill="#95a5a6" opacity="0.6" filter="url(#motionBlur)" />
       <!-- Rotor center hub -->
       <circle cx="0" cy="0" r="5" fill="#7f8c8d" />
    </g>
    <!-- Tail Boom -->
    <path d="M280 140 L380 110 L380 130 L280 160 Z" fill="url(#bodyGrad)" />
    <path d="M280 140 L380 110" stroke="#2980b9" stroke-width="2" fill="none"/>
    <!-- Landing Skids -->
    <g stroke="#7f8c8d" stroke-width="8" fill="none" stroke-linecap="round">
      <!-- Left Skid -->
      <path d="M120 230 Q 200 250 280 230" />
      <!-- Right Skid (slightly offset for perspective) -->
      <path d="M140 245 Q 220 265 300 245" stroke-width="6" opacity="0.8"/>
    </g>
    <!-- Skid Supports -->
    <line x1="150" y1="200" x2="130" y2="235" stroke="#7f8c8d" stroke-width="4" />
    <line x1="250" y1="200" x2="230" y2="240" stroke="#7f8c8d" stroke-width="4" />
    <!-- Main Body (Fuselage) -->
    <path d="M50 120 
             Q 50 60 150 60 
             Q 280 60 300 140 
             Q 310 180 280 200 
             L 80 200 
             Q 40 190 50 120 Z" 
          fill="url(#bodyGrad)" stroke="#2980b9" stroke-width="2"/>
    <!-- Cockpit Window -->
    <path d="M260 80 
             Q 290 90 290 130 
             L 280 160 
             L 200 160 
             L 180 100 
             Q 190 80 260 80 Z" 
          fill="url(#glassGrad)" />
    <!-- Window Reflection -->
    <path d="M270 90 L 280 120 L 260 120 Z" fill="white" opacity="0.2" />
    <!-- Door Line Detail -->
    <path d="M160 90 Q 160 140 180 160" stroke="#2980b9" stroke-width="2" fill="none" opacity="0.5"/>
    <!-- Main Rotor Assembly -->
    <g transform="translate(180, 60)">
      <!-- Rotor Mast -->
      <rect x="-5" y="-10" width="10" height="20" fill="#7f8c8d" />
      <!-- Spinning Blades (Blur) -->
      <ellipse cx="0" cy="0" rx="160" ry="12" fill="#ecf0f1" opacity="0.7" filter="url(#motionBlur)" />
      <!-- Solid Blade Lines (for structure) -->
      <line x1="-140" y1="0" x2="140" y2="0" stroke="#bdc3c7" stroke-width="4" />
      <line x1="0" y1="-5" x2="0" y2="5" stroke="#bdc3c7" stroke-width="4" transform="rotate(45)" />
      <!-- Rotor Hub -->
      <circle cx="0" cy="0" r="8" fill="#95a5a6" stroke="#7f8c8d" stroke-width="2"/>
    </g>
  </g>
</svg>
```

</details>

## 14. `deepseek-v4-flash:0731-cloud`

**SVG size:** 7972 characters  
**Complexity:** High  
**Shape elements:** 66  
**Distinct colors:** 9  
**Raw response:** 7972 characters  
**Generation time:** ~100s

![deepseek-v4-flash:0731-cloud flying helicopter SVG](/assets/img/diagrams/ollama-svg-helicopter/deepseek-v4-flash_0731-cloud.svg)

### Analysis

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

**Concept adherence:**

- The model rendered its interpretation of the "flying helicopter" prompt -- review the SVG above for how well it captured the subject, composition, and key visual cues

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
  <defs>
    <!-- Sky Gradient -->
    <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#87CEEB"/>
      <stop offset="100%" stop-color="#E0F6FF"/>
    </linearGradient>
    <!-- Rotor Motion Blur Gradient -->
    <radialGradient id="rotorBlur" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#F1FAEE" stop-opacity="0.5"/>
      <stop offset="80%" stop-color="#F1FAEE" stop-opacity="0.2"/>
      <stop offset="100%" stop-color="#F1FAEE" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <!-- Background -->
  <rect width="800" height="600" fill="url(#skyGrad)"/>
  <!-- Sun -->
  <circle cx="100" cy="100" r="70" fill="#F4A261" opacity="0.3"/>
  <circle cx="100" cy="100" r="50" fill="#F4A261" opacity="0.9"/>
  <!-- Background Clouds / Speed Lines -->
  <g fill="#FFFFFF" opacity="0.6">
    <path d="M 100 200 Q 150 190 200 200 Q 250 190 300 200 Q 250 210 200 210 Q 150 210 100 200 Z"/>
    <path d="M 500 400 Q 550 390 600 400 Q 650 390 700 400 Q 650 410 600 410 Q 550 410 500 400 Z"/>
    <path d="M 600 150 Q 630 140 660 150 Q 690 140 720 150 Q 690 160 660 160 Q 630 160 600 150 Z"/>
  </g>
  <g stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.5">
    <line x1="200" y1="150" x2="280" y2="150"/>
    <line x1="600" y1="500" x2="680" y2="500"/>
    <line x1="150" y1="350" x2="200" y2="350"/>
    <line x1="650" y1="250" x2="720" y2="250"/>
  </g>
  <!-- Helicopter Ground Shadow -->
  <ellipse cx="400" cy="550" rx="120" ry="20" fill="#1D3557" opacity="0.15"/>
  <!-- Main Helicopter Group -->
  <g transform="translate(400, 300)">
    <!-- Tail Rotor (Background elements) -->
    <g>
      <!-- Tail Rotor Blades -->
      <ellipse cx="-220" cy="-20" rx="5" ry="25" fill="#1D3557" opacity="0.7"/>
      <ellipse cx="-220" cy="-20" rx="5" ry="25" fill="#1D3557" opacity="0.7" transform="rotate(90, -220, -20)"/>
      <!-- Tail Rotor Motion Blur -->
      <ellipse cx="-220" cy="-20" rx="15" ry="30" fill="none" stroke="#F1FAEE" stroke-width="4" opacity="0.6"/>
      <ellipse cx="-220" cy="-20" rx="8" ry="35" fill="none" stroke="#F1FAEE" stroke-width="2" opacity="0.4"/>
    </g>
    <!-- Background Landing Gear Skid -->
    <g>
      <path d="M -30 25 L -40 45" stroke="#1D3557" stroke-width="4" stroke-linecap="round"/>
      <path d="M 10 25 L 0 45" stroke="#1D3557" stroke-width="4" stroke-linecap="round"/>
      <path d="M 50 25 L 40 45" stroke="#1D3557" stroke-width="4" stroke-linecap="round"/>
      <path d="M -60 45 L 40 45" stroke="#1D3557" stroke-width="6" stroke-linecap="round" opacity="0.6"/>
      <circle cx="-60" cy="45" r="3" fill="#1D3557" opacity="0.6"/>
      <circle cx="40" cy="45" r="3" fill="#1D3557" opacity="0.6"/>
    </g>
    <!-- Tail Boom -->
    <g>
      <path d="M -80 -10 L -220 -30 L -220 -10 L -80 10 Z" fill="#E63946"/>
      <path d="M -80 -5 L -220 -20 L -220 -15 L -80 0 Z" fill="#F4F4F9"/>
      <path d="M -80 0 L -220 -15 L -220 -10 L -80 10 Z" fill="#B32D3A"/>
      <!-- Tail Rotor Hub -->
      <circle cx="-220" cy="-20" r="6" fill="#1D3557"/>
    </g>
    <!-- Forward Motion Speed Lines -->
    <g fill="none" stroke="#FFFFFF" stroke-width="3" stroke-linecap="round" opacity="0.7">
      <path d="M 120 0 Q 150 0 180 -10"/>
      <path d="M 130 15 Q 160 15 190 5"/>
      <path d="M 110 -20 Q 140 -25 160 -30"/>
    </g>
    <!-- Main Fuselage / Body -->
    <g>
      <!-- Base Body -->
      <path d="M -80 30 L -80 -10 Q -80 -60 -20 -60 Q 40 -60 70 -30 Q 120 -10 110 15 Q 100 30 70 30 L -80 30 Z" fill="#E63946"/>
      <!-- Bottom White Stripe -->
      <path d="M -78 20 L 68 20 Q 90 20 100 25 L -78 25 Z" fill="#F4F4F9"/>
      <!-- Bottom Shadow (Dark Red) -->
      <path d="M -80 25 L 70 25 Q 85 25 95 28 L -80 28 Z" fill="#B32D3A"/>
      <!-- Top White Stripe -->
      <path d="M -78 -10 Q -78 -50 -20 -50 Q 35 -50 60 -30 L 45 -20 Q 25 -35 -20 -35 Q -55 -35 -60 -10 Z" fill="#F4F4F9"/>
      <!-- Engine Intake -->
      <ellipse cx="-60" cy="-20" rx="10" ry="15" fill="#1D3557"/>
      <ellipse cx="-58" cy="-22" rx="6" ry="10" fill="#333333"/>
      <!-- Exhaust Pipe -->
      <path d="M -80 -5 L -95 0 L -95 5 L -80 5 Z" fill="#457B9D"/>
    </g>
    <!-- Cockpit and Windows -->
    <g>
      <!-- Windshield -->
      <path d="M 55 -28 Q 90 -15 105 5 L 85 10 Q 65 -10 45 -15 Z" fill="#A8DADC" stroke="#1D3557" stroke-width="2"/>
      <!-- Pilot Silhouette -->
      <circle cx="60" cy="-20" r="8" fill="#1D3557" opacity="0.4"/>
      <path d="M 50 -12 L 70 -12 L 70 -5 L 50 -5 Z" fill="#1D3557" opacity="0.4"/>
      <!-- Control Panel -->
      <path d="M 45 -15 L 70 -15 L 80 5 L 45 5 Z" fill="#1D3557" opacity="0.3"/>
      <!-- Side Window 1 -->
      <rect x="10" y="-45" width="30" height="20" rx="5" fill="#A8DADC" stroke="#1D3557" stroke-width="2"/>
      <!-- Side Window 2 -->
      <rect x="-30" y="-45" width="30" height="20" rx="5" fill="#A8DADC" stroke="#1D3557" stroke-width="2"/>
      <!-- Rear Window -->
      <path d="M -60 -40 L -40 -40 L -40 -15 L -60 -15 Z" fill="#A8DADC" stroke="#1D3557" stroke-width="2"/>
      <!-- Door Lines -->
      <path d="M -20 -55 L -20 25" stroke="#1D3557" stroke-width="2" opacity="0.5"/>
      <path d="M 20 -55 L 20 25" stroke="#1D3557" stroke-width="2" opacity="0.5"/>
    </g>
    <!-- Wing Stub &amp; Armaments (Optional Add-on) -->
    <g>
      <!-- Wing Stub -->
      <path d="M 10 15 L 50 25 L 50 35 L 10 25 Z" fill="#E63946" stroke="#1D3557" stroke-width="2"/>
      <!-- Rocket Pod -->
      <rect x="45" y="20" width="30" height="20" rx="3" fill="#457B9D" stroke="#1D3557" stroke-width="2"/>
      <!-- Rocket Tubes -->
      <circle cx="75" cy="25" r="3" fill="#F4F4F9"/>
      <circle cx="75" cy="30" r="3" fill="#F4F4F9"/>
      <circle cx="75" cy="35" r="3" fill="#F4F4F9"/>
      <!-- Foreground Wing Stub Shadow -->
      <path d="M 10 25 L 50 35 L 50 30 L 10 20 Z" fill="#B32D3A"/>
    </g>
    <!-- Foreground Landing Gear Skids -->
    <g>
      <!-- Struts -->
      <path d="M -40 25 L -50 50" stroke="#1D3557" stroke-width="5" stroke-linecap="round"/>
      <path d="M 0 25 L -10 50" stroke="#1D3557" stroke-width="5" stroke-linecap="round"/>
      <path d="M 40 25 L 30 50" stroke="#1D3557" stroke-width="5" stroke-linecap="round"/>
      <!-- Skid Tube -->
      <path d="M -70 50 L 50 50" stroke="#1D3557" stroke-width="8" stroke-linecap="round"/>
      <!-- Skid End Caps -->
      <circle cx="-70" cy="50" r="4" fill="#1D3557"/>
      <circle cx="50" cy="50" r="4" fill="#1D3557"/>
      <!-- Skid Highlights -->
      <path d="M -65 48 L 45 48" stroke="#F4F4F9" stroke-width="2" stroke-linecap="round" opacity="0.4"/>
    </g>
    <!-- Main Rotor Assembly -->
    <g>
      <!-- Mast -->
      <rect x="-12" y="-90" width="12" height="30" fill="#457B9D"/>
      <!-- Swash Plate / Hub -->
      <ellipse cx="-6" cy="-60" rx="15" ry="6" fill="#1D3557"/>
      <!-- Rotor Blades -->
      <g opacity="0.85">
        <!-- Blade 1 (Front) -->
        <path d="M -6 -63 L 180 -70 L 180 -64 L -6 -57 Z" fill="#1D3557"/>
        <!-- Blade 2 (Right) -->
        <path d="M -6 -63 L 180 -70 L 180 -64 L -6 -57 Z" fill="#1D3557" transform="rotate(90, -6, -63)"/>
        <!-- Blade 3 (Back) -->
        <path d="M -6 -63 L 180 -70 L 180 -64 L -6 -57 Z" fill="#1D3557" opacity="0.6" transform="rotate(180, -6, -63)"/>
        <!-- Blade 4 (Left) -->
        <path d="M -6 -63 L 180 -70 L 180 -64 L -6 -57 Z" fill="#1D3557" opacity="0.6" transform="rotate(270, -6, -63)"/>
      </g>
      <!-- Rotor Motion Blur Discs -->
      <ellipse cx="-6" cy="-63" rx="190" ry="25" fill="url(#rotorBlur)"/>
      <ellipse cx="-6" cy="-63" rx="190" ry="8" fill="none" stroke="#F1FAEE" stroke-width="6" opacity="0.4"/>
      <ellipse cx="-6" cy="-63" rx="100" ry="15" fill="none" stroke="#F1FAEE" stroke-width="4" opacity="0.3"/>
    </g>
  </g>
</svg>
```

</details>


## Conclusion

That's all 14 flying helicopter SVGs from the active Ollama cloud models. This mechanical-aviation prompt tested a different skill set than our previous prompts: mechanical anatomy (fuselage, cockpit, rotors, tail boom, landing skids), rotational motion (spinning rotors), flight dynamics (sky, clouds, ground), and animation potential.

**No single model is best at everything.** A model that produces a stunning elephant may produce a basic helicopter, and vice versa. The best model for you depends on your specific use case:

- For **maximum detail**: look at the "Very high" complexity models
- For **speed**: look at the generation times in each section
- For **anatomical accuracy**: check which models captured the main rotor, tail rotor, and landing skids
- For **animation**: check which models used `<animate>` or `@keyframes` for spinning rotors
- For **scene context**: look for models that added clouds, sky, stars, or ground
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
- [Elephant on a skateboard](/Ollama-Cloud-Models-SVG-Comparison-Elephant-Skateboard/)
