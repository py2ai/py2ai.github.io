---
layout: post
title: "Ollama Cloud Models SVG Comparison: A Duck Driving a Bicycle"
description: "Asking 9 Ollama cloud models the same prompt and comparing the SVG artwork each one generates. A visual benchmark of SOTA LLM SVG generation."
date: 2026-07-25
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - Ollama
  - SVG
  - LLM
  - Comparison
author: "PyShine"
image: /assets/img/diagrams/ollama/ollama-api-ecosystem.svg
---

# Ollama Cloud Models SVG Comparison: A Duck Driving a Bicycle

What happens when you ask 9 different state-of-the-art LLMs to draw the exact same thing? In this post, we sent the prompt `Make an svg image of a duck driving a bicycle` to every cloud model available in the local Ollama registry and compared the results side by side.

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering.

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary of Results

| # | Model | SVG Size (chars) | Status |
|---|-------|-----------------|--------|
| 1 | `deepseek-v4-pro_cloud` | 8693 | Generated |
| 2 | `gemma4_31b-cloud` | 1582 | Generated |
| 3 | `glm-5.1_cloud` | 15284 | Generated |
| 4 | `glm-5.2_cloud` | 6217 | Generated |
| 5 | `kimi-k2.6_cloud` | 8262 | Generated |
| 6 | `minimax-m2.7_cloud` | 3490 | Generated |
| 7 | `nemotron-3-super_cloud` | 1954 | Generated |
| 8 | `nemotron-3-ultra_cloud` | 8548 | Generated |
| 9 | `qwen3.5_397b-cloud` | 3247 | Generated |
| 10 | `deepseek-v4-flash_0731-cloud` | 11814 | Generated |
| 11 | `bjoernb/claude-opus-4-5:latest` | - | Retired (410) |
| 12 | `deepseek-v3.1:671b-cloud` | - | Retired (410) |
| 13 | `glm-5:cloud` | - | Retired (410) |
| 14 | `qwen3-vl:235b-cloud` | - | Retired (410) |

**10 out of 14** models successfully produced a valid SVG. The 4 retired models returned HTTP 410 Gone (they were removed from Ollama Cloud on 2026-07-15).

## 1. deepseek-v4-pro_cloud

**SVG size:** 8693 characters  
**Raw response:** 9128 characters

![deepseek-v4-pro_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/deepseek-v4-pro_cloud.svg)

### Analysis

This SVG contains approximately **82 shape elements** and uses **16 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <!-- Background -->
  <rect x="0" y="0" width="800" height="600" fill="#87CEEB" />
  
  <!-- Sun -->
  <circle cx="100" cy="100" r="50" fill="#FFD700" />
  
  <!-- Clouds -->
  <path d="M 200 100 Q 220 80 240 100 Q 260 80 280 100 Q 300 120 280 130 L 200 130 Q 180 120 200 100" fill="#FFFFFF" opacity="0.8" />
  <path d="M 600 150 Q 620 130 640 150 Q 660 130 680 150 Q 700 170 680 180 L 600 180 Q 580 170 600 150" fill="#FFFFFF" opacity="0.8" />
  <path d="M 400 80 Q 420 60 440 80 Q 460 60 480 80 Q 500 100 480 110 L 400 110 Q 380 100 400 80" fill="#FFFFFF" opacity="0.8" />

  <!-- Birds -->
  <path d="M 150 120 Q 155 115 160 120 Q 165 115 170 120" fill="none" stroke="#333" stroke-width="2" />
  <path d="M 700 60 Q 705 55 710 60 Q 715 55 720 60" fill="none" stroke="#333" stroke-width="2" />
  <path d="M 300 80 Q 310 70 320 80 Q 330 70 340 80" fill="none" stroke="#333" stroke-width="2" />

  <!-- Road -->
  <rect x="0" y="500" width="800" height="100" fill="#555555" />
  <line x1="0" y1="550" x2="800" y2="550" stroke="#FFEB3B" stroke-width="6" stroke-dasharray="40 20" />

  <!-- Grass -->
  <path d="M 50 500 L 60 480 L 70 500 Z" fill="#4CAF50" />
  <path d="M 70 500 L 80 470 L 90 500 Z" fill="#4CAF50" />
  <path d="M 750 500 L 760 480 L 770 500 Z" fill="#4CAF50" />
  <path d="M 770 500 L 780 470 L 790 500 Z" fill="#4CAF50" />

  <!-- Shadow under the bike -->
  <ellipse cx="400" cy="550" rx="250" ry="15" fill="rgba(0,0,0,0.3)" />

  <!-- Motion Lines -->
  <line x1="50" y1="300" x2="120" y2="300" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.6" />
  <line x1="80" y1="320" x2="140" y2="320" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.6" />
  <line x1="60" y1="340" x2="110" y2="340" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.6" />

  <!-- ================= BICYCLE ================= -->
  
  <!-- Back Wheel -->
  <circle cx="200" cy="450" r="100" fill="none" stroke="#222" stroke-width="12" />
  <circle cx="200" cy="450" r="94" fill="none" stroke="#CCC" stroke-width="2" />
  <!-- Back Spokes -->
  <line x1="200" y1="360" x2="200" y2="540" stroke="#CCC" stroke-width="2" />
  <line x1="110" y1="450" x2="290" y2="450" stroke="#CCC" stroke-width="2" />
  <line x1="136" y1="386" x2="264" y2="514" stroke="#CCC" stroke-width="2" />
  <line x1="136" y1="514" x2="264" y2="386" stroke="#CCC" stroke-width="2" />
  <line x1="200" y1="360" x2="200" y2="540" stroke="#CCC" stroke-width="2" transform="rotate(30 200 450)" />
  <line x1="200" y1="360" x2="200" y2="540" stroke="#CCC" stroke-width="2" transform="rotate(60 200 450)" />
  <line x1="200" y1="360" x2="200" y2="540" stroke="#CCC" stroke-width="2" transform="rotate(120 200 450)" />
  <line x1="200" y1="360" x2="200" y2="540" stroke="#CCC" stroke-width="2" transform="rotate(150 200 450)" />
  <!-- Rear Hub -->
  <circle cx="200" cy="450" r="10" fill="#555" stroke="#333" stroke-width="3" />

  <!-- Front Wheel -->
  <circle cx="600" cy="450" r="100" fill="none" stroke="#222" stroke-width="12" />
  <circle cx="600" cy="450" r="94" fill="none" stroke="#CCC" stroke-width="2" />
  <!-- Front Spokes -->
  <line x1="600" y1="360" x2="600" y2="540" stroke="#CCC" stroke-width="2" />
  <line x1="510" y1="450" x2="690" y2="450" stroke="#CCC" stroke-width="2" />
  <line x1="536" y1="386" x2="664" y2="514" stroke="#CCC" stroke-width="2" />
  <line x1="536" y1="514" x2="664" y2="386" stroke="#CCC" stroke-width="2" />
  <line x1="600" y1="360" x2="600" y2="540" stroke="#CCC" stroke-width="2" transform="rotate(30 600 450)" />
  <line x1="600" y1="360" x2="600" y2="540" stroke="#CCC" stroke-width="2" transform="rotate(60 600 450)" />
  <line x1="600" y1="360" x2="600" y2="540" stroke="#CCC" stroke-width="2" transform="rotate(120 600 450)" />
  <line x1="600" y1="360" x2="600" y2="540" stroke="#CCC" stroke-width="2" transform="rotate(150 600 450)" />
  <!-- Front Hub -->
  <circle cx="600" cy="450" r="10" fill="#555" stroke="#333" stroke-width="3" />

  <!-- Back Pedal and Crank -->
  <line x1="400" y1="450" x2="380" y2="470" stroke="#333" stroke-width="8" stroke-linecap="round" />
  <line x1="360" y1="470" x2="400" y2="470" stroke="#333" stroke-width="10" stroke-linecap="round" />

  <!-- Back Leg and Foot (Duck) -->
  <path d="M 340 360 L 360 460" stroke="#FF8C00" stroke-width="12" fill="none" stroke-linecap="round" />
  <path d="M 340 460 L 380 460" stroke="#FF8C00" stroke-width="12" fill="none" stroke-linecap="round" />

  <!-- Back Wing (Duck) -->
  <path d="M 340 280 Q 380 250 420 280" stroke="#FFD700" stroke-width="20" fill="none" stroke-linecap="round" />

  <!-- Bike Frame -->
  <!-- Chainstay -->
  <line x1="200" y1="450" x2="400" y2="450" stroke="#FF4500" stroke-width="12" stroke-linecap="round" />
  <!-- Seatstay -->
  <line x1="200" y1="450" x2="350" y2="300" stroke="#FF4500" stroke-width="12" stroke-linecap="round" />
  <!-- Seat tube -->
  <line x1="350" y1="300" x2="400" y2="450" stroke="#FF4500" stroke-width="12" stroke-linecap="round" />
  <!-- Down tube -->
  <line x1="400" y1="450" x2="550" y2="300" stroke="#FF4500" stroke-width="12" stroke-linecap="round" />
  <!-- Top tube -->
  <line x1="350" y1="300" x2="550" y2="300" stroke="#FF4500" stroke-width="12" stroke-linecap="round" />
  <!-- Head tube -->
  <line x1="550" y1="300" x2="560" y2="350" stroke="#FF4500" stroke-width="16" stroke-linecap="round" />
  <!-- Fork -->
  <path d="M 550 300 L 560 350 L 600 450" stroke="#FF4500" stroke-width="12" fill="none" stroke-linejoin="round" />

  <!-- Chain and Chainring -->
  <line x1="200" y1="450" x2="400" y2="450" stroke="#333" stroke-width="4" stroke-dasharray="6 3" />
  <circle cx="400" cy="450" r="18" fill="#555" stroke="#333" stroke-width="4" />

  <!-- Handlebars -->
  <line x1="550" y1="300" x2="540" y2="250" stroke="#333" stroke-width="10" stroke-linecap="round" />
  <path d="M 500 250 Q 540 220 560 250" stroke="#333" stroke-width="10" fill="none" stroke-linecap="round" />
  
  <!-- Handlebar Bell -->
  <circle cx="560" cy="240" r="8" fill="#FFD700" stroke="#333" stroke-width="2" />
  <path d="M 560 240 L 560 250" stroke="#333" stroke-width="2" />

  <!-- Seat -->
  <line x1="330" y1="300" x2="370" y2="300" stroke="#333" stroke-width="15" stroke-linecap="round" />
  <line x1="350" y1="300" x2="350" y2="320" stroke="#333" stroke-width="8" stroke-linecap="round" />

  <!-- Taillight -->
  <rect x="345" y="280" width="10" height="15" fill="#FF0000" rx="3" />
  <circle cx="350" cy="287" r="3" fill="#FFFFFF" />

  <!-- Headlight -->
  <circle cx="560" cy="350" r="10" fill="#FFEB3B" stroke="#333" stroke-width="2" />

  <!-- Front Pedal and Crank -->
  <line x1="400" y1="450" x2="420" y2="470" stroke="#333" stroke-width="8" stroke-linecap="round" />
  <line x1="400" y1="470" x2="440" y2="470" stroke="#333" stroke-width="10" stroke-linecap="round" />

  <!-- ================= DUCK ================= -->
  
  <!-- Duck Body -->
  <ellipse cx="360" cy="300" rx="60" ry="70" fill="#FFD700" />
  
  <!-- Duck Tail -->
  <path d="M 300 300 Q 260 280 270 320 Q 280 340 310 340" fill="#FFD700" />

  <!-- Duck Neck -->
  <path d="M 400 260 L 430 190 L 450 190 L 430 280 Z" fill="#FFD700" />

  <!-- Duck Head -->
  <circle cx="450" cy="170" r="35" fill="#FFD700" />

  <!-- Duck Tuft (Hair) -->
  <path d="M 450 135 Q 455 120 465 130 Q 460 140 450 135" fill="#FFD700" />

  <!-- Duck Blush -->
  <circle cx="455" cy="175" r="6" fill="#FF9999" opacity="0.6" />

  <!-- Duck Beak -->
  <path d="M 470 155 Q 500 150 520 170 Q 500 190 470 185 Z" fill="#FF8C00" />
  <!-- Beak Smile -->
  <path d="M 485 175 Q 495 185 505 175" stroke="#333" stroke-width="2" fill="none" />

  <!-- Duck Eye -->
  <circle cx="465" cy="160" r="5" fill="#000000" />
  <circle cx="463" cy="158" r="2" fill="#FFFFFF" />

  <!-- Helmet -->
  <path d="M 440 145 C 440 100, 480 100, 480 145 Z" fill="#1E90FF" />
  <!-- Helmet Visor -->
  <path d="M 480 140 L 500 145 L 480 150 Z" fill="#333" />
  <!-- Helmet Strap -->
  <path d="M 440 145 L 450 160 L 460 145" stroke="#333" stroke-width="2" fill="none" />

  <!-- Front Leg and Foot (Duck) -->
  <path d="M 370 360 L 400 470" stroke="#FF8C00" stroke-width="12" fill="none" stroke-linecap="round" />
  <path d="M 380 470 L 420 470" stroke="#FF8C00" stroke-width="12" fill="none" stroke-linecap="round" />

  <!-- Front Wing (Duck) - Holding the handlebar -->
  <path d="M 350 300 Q 400 260 500 250" stroke="#FFD700" stroke-width="25" fill="none" stroke-linecap="round" />

  <!-- Shadow of Duck on the Seat -->
  <ellipse cx="360" cy="300" rx="50" ry="15" fill="rgba(0,0,0,0.2)" />

</svg>
```

</details>

## 2. gemma4_31b-cloud

**SVG size:** 1582 characters  
**Raw response:** 2565 characters

![gemma4_31b-cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/gemma4_31b-cloud.svg)

### Analysis

This SVG contains approximately **17 shape elements** and uses **6 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Background: Simple Ground -->
  <line x1="20" y1="160" x2="180" y2="160" stroke="#888" stroke-width="4" stroke-linecap="round" />

  <!-- Bicycle Frame -->
  <path d="M70 160 L100 130 L130 130 L150 160" fill="none" stroke="#e74c3c" stroke-width="4" stroke-linejoin="round" />
  <line x1="100" y1="130" x2="100" y2="150" stroke="#e74c3c" stroke-width="4" />
  
  <!-- Wheels -->
  <circle cx="70" cy="160" r="15" fill="none" stroke="#333" stroke-width="3" />
  <circle cx="150" cy="160" r="15" fill="none" stroke="#333" stroke-width="3" />
  <circle cx="70" cy="160" r="2" fill="#333" />
  <circle cx="150" cy="160" r="2" fill="#333" />

  <!-- Handlebars -->
  <path d="M130 130 L135 115 L145 115" fill="none" stroke="#333" stroke-width="3" stroke-linecap="round" />

  <!-- Duck Body -->
  <ellipse cx="100" cy="110" rx="25" ry="20" fill="#FFD700" />
  
  <!-- Duck Head -->
  <circle cx="125" cy="90" r="12" fill="#FFD700" />
  
  <!-- Duck Beak -->
  <path d="M135 90 L145 92 L135 94 Z" fill="#FF8C00" />
  
  <!-- Duck Eye -->
  <circle cx="128" cy="87" r="1.5" fill="black" />

  <!-- Duck Wing -->
  <ellipse cx="100" cy="110" rx="10" ry="6" fill="#FFE135" />

  <!-- Duck Leg/Foot on Pedal -->
  <line x1="100" y1="130" x2="105" y2="145" stroke="#FF8C00" stroke-width="3" />
  <line x1="105" y1="145" x2="112" y2="145" stroke="#FF8C00" stroke-width="3" />

  <!-- Little Hat (Optional Cute Detail) -->
  <rect x="120" y="78" fill="#333" />
  <path d="M126 78 L126 72 L132 72 L132 78" fill="#333" />
</svg>
```

</details>

## 3. glm-5.1_cloud

**SVG size:** 15284 characters  
**Raw response:** 18846 characters

![glm-5.1_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/glm-5.1_cloud.svg)

### Analysis

This SVG contains approximately **118 shape elements** and uses **43 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Contains XML comments, showing the model's internal documentation of its work

With over 10,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in adding fine details, accessories, and scene elements.

<details><summary>View raw SVG source</summary>

```xml
<svg class="scene-container" viewBox="0 0 800 700" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <!-- Sky gradient -->
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#264653"/>
      <stop offset="60%" stop-color="#2a9d8f" stop-opacity="0.3"/>
      <stop offset="100%" stop-color="#e9c46a" stop-opacity="0.2"/>
    </linearGradient>

    <!-- Ground gradient -->
    <linearGradient id="groundGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#588157"/>
      <stop offset="100%" stop-color="#3a5a40"/>
    </linearGradient>

    <!-- Duck body gradient -->
    <radialGradient id="duckBodyGrad" cx="0.4" cy="0.3">
      <stop offset="0%" stop-color="#f4e285"/>
      <stop offset="100%" stop-color="#e9c46a"/>
    </radialGradient>

    <!-- Duck head gradient -->
    <radialGradient id="duckHeadGrad" cx="0.4" cy="0.3">
      <stop offset="0%" stop-color="#f4e285"/>
      <stop offset="100%" stop-color="#daa520"/>
    </radialGradient>

    <!-- Wheel rim gradient -->
    <linearGradient id="rimGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#888"/>
      <stop offset="50%" stop-color="#ccc"/>
      <stop offset="100%" stop-color="#666"/>
    </linearGradient>

    <!-- Tire gradient -->
    <radialGradient id="tireGrad" cx="0.5" cy="0.5" r="0.5">
      <stop offset="85%" stop-color="#333"/>
      <stop offset="100%" stop-color="#1a1a1a"/>
    </radialGradient>

    <!-- Road gradient -->
    <linearGradient id="roadGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#6b705c"/>
      <stop offset="100%" stop-color="#4a4e3f"/>
    </linearGradient>

    <!-- Beak gradient -->
    <linearGradient id="beakGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f77f00"/>
      <stop offset="100%" stop-color="#e36414"/>
    </linearGradient>

    <!-- Scarf gradient -->
    <linearGradient id="scarfGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#e63946"/>
      <stop offset="100%" stop-color="#d62828"/>
    </linearGradient>

    <!-- Frame gradient -->
    <linearGradient id="frameGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#e76f51"/>
      <stop offset="100%" stop-color="#c44536"/>
    </linearGradient>

    <!-- Sun glow -->
    <radialGradient id="sunGlow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0%" stop-color="#f4e285" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#f4e285" stop-opacity="0"/>
    </radialGradient>
  </defs>

  <!-- BACKGROUND SKY -->
  <rect x="0" y="0" fill="url(#skyGrad)"/>

  <!-- Sun -->
  <circle cx="650" cy="100" r="80" fill="url(#sunGlow)"/>
  <circle cx="650" cy="100" r="35" fill="#f4e285" opacity="0.9"/>

  <!-- Clouds -->
  <g class="cloud1" opacity="0.3">
    <ellipse cx="180" cy="110" rx="50" ry="18" fill="#a8dadc"/>
    <ellipse cx="210" cy="100" rx="40" ry="22" fill="#a8dadc"/>
    <ellipse cx="150" cy="105" rx="35" ry="15" fill="#a8dadc"/>
  </g>
  <g class="cloud2" opacity="0.25">
    <ellipse cx="500" cy="80" rx="55" ry="16" fill="#a8dadc"/>
    <ellipse cx="530" cy="72" rx="38" ry="20" fill="#a8dadc"/>
    <ellipse cx="475" cy="76" rx="30" ry="14" fill="#a8dadc"/>
  </g>

  <!-- Distant hills -->
  <ellipse cx="200" cy="510" rx="250" ry="40" fill="#344e41" opacity="0.4"/>
  <ellipse cx="600" cy="520" rx="300" ry="35" fill="#344e41" opacity="0.3"/>

  <!-- Ground -->
  <rect x="0" y="530" width="800" height="170" fill="url(#groundGrad)"/>

  <!-- Road -->
  <rect x="0" y="530" width="800" height="60" fill="url(#roadGrad)" rx="2"/>
  <line x1="0" y1="560" x2="800" y2="560" stroke="#a3b18a" stroke-width="2" stroke-dasharray="20,15" opacity="0.5"/>

  <!-- Road edge lines -->
  <line x1="0" y1="532" x2="800" y2="532" stroke="#a3b18a" stroke-width="1.5" opacity="0.3"/>
  <line x1="0" y1="588" x2="800" y2="588" stroke="#a3b18a" stroke-width="1.5" opacity="0.3"/>

  <!-- Ground texture marks (moving) -->
  <g class="ground-marks" opacity="0.2">
    <line x1="50" y1="600" x2="80" y2="600" stroke="#2d3a1e" stroke-width="1.5"/>
    <line x1="150" y1="610" x2="175" y2="610" stroke="#2d3a1e" stroke-width="1"/>
    <line x1="280" y1="595" x2="310" y2="595" stroke="#2d3a1e" stroke-width="1.5"/>
    <line x1="420" y1="605" x2="445" y2="605" stroke="#2d3a1e" stroke-width="1"/>
    <line x1="550" y1="598" x2="575" y2="598" stroke="#2d3a1e" stroke-width="1.5"/>
    <line x1="680" y1="612" x2="710" y2="612" stroke="#2d3a1e" stroke-width="1"/>
  </g>

  <!-- === BICYCLE === -->
  <g id="bicycle">

    <!-- REAR WHEEL -->
    <g id="rearWheel">
      <!-- Tire -->
      <circle cx="310" cy="530" r="58" fill="none" stroke="#2b2b2b" stroke-width="12"/>
      <circle cx="310" cy="530" r="58" fill="none" stroke="#3d3d3d" stroke-width="8"/>
      <!-- Rim -->
      <circle cx="310" cy="530" r="50" fill="none" stroke="url(#rimGrad)" stroke-width="3"/>
      <!-- Hub -->
      <circle cx="310" cy="530" r="6" fill="#999" stroke="#777" stroke-width="2"/>
      <!-- Spokes (animated) -->
      <g class="wheel-spokes-rear" style="transform-origin: 310px 530px;">
        <line x1="310" y1="480" x2="310" y2="580" stroke="#aaa" stroke-width="1.2"/>
        <line x1="260" y1="530" x2="360" y2="530" stroke="#aaa" stroke-width="1.2"/>
        <line x1="274" y1="494" x2="346" y2="566" stroke="#aaa" stroke-width="1.2"/>
        <line x1="346" y1="494" x2="274" y2="566" stroke="#aaa" stroke-width="1.2"/>
        <line x1="268" y1="510" x2="352" y2="550" stroke="#aaa" stroke-width="0.8"/>
        <line x1="352" y1="510" x2="268" y2="550" stroke="#aaa" stroke-width="0.8"/>
        <line x1="290" y1="478" x2="330" y2="582" stroke="#aaa" stroke-width="0.8"/>
        <line x1="330" y1="478" x2="290" y2="582" stroke="#aaa" stroke-width="0.8"/>
      </g>
    </g>

    <!-- FRONT WHEEL -->
    <g id="frontWheel">
      <!-- Tire -->
      <circle cx="530" cy="530" r="58" fill="none" stroke="#2b2b2b" stroke-width="12"/>
      <circle cx="530" cy="530" r="58" fill="none" stroke="#3d3d3d" stroke-width="8"/>
      <!-- Rim -->
      <circle cx="530" cy="530" r="50" fill="none" stroke="url(#rimGrad)" stroke-width="3"/>
      <!-- Hub -->
      <circle cx="530" cy="530" r="6" fill="#999" stroke="#777" stroke-width="2"/>
      <!-- Spokes (animated) -->
      <g class="wheel-spokes" style="transform-origin: 530px 530px;">
        <line x1="530" y1="480" x2="530" y2="580" stroke="#aaa" stroke-width="1.2"/>
        <line x1="480" y1="530" x2="580" y2="530" stroke="#aaa" stroke-width="1.2"/>
        <line x1="494" y1="494" x2="566" y2="566" stroke="#aaa" stroke-width="1.2"/>
        <line x1="566" y1="494" x2="494" y2="566" stroke="#aaa" stroke-width="1.2"/>
        <line x1="488" y1="510" x2="572" y2="550" stroke="#aaa" stroke-width="0.8"/>
        <line x1="572" y1="510" x2="488" y2="550" stroke="#aaa" stroke-width="0.8"/>
        <line x1="510" y1="478" x2="550" y2="582" stroke="#aaa" stroke-width="0.8"/>
        <line x1="550" y1="478" x2="510" y2="582" stroke="#aaa" stroke-width="0.8"/>
      </g>
    </g>

    <!-- FRAME -->
    <!-- Seat tube -->
    <line x1="370" y1="385" x2="340" y2="510" stroke="url(#frameGrad)" stroke-width="7" stroke-linecap="round"/>
    <!-- Top tube -->
    <line x1="370" y1="385" x2="490" y2="395" stroke="url(#frameGrad)" stroke-width="6" stroke-linecap="round"/>
    <!-- Down tube -->
    <line x1="370" y1="385" x2="530" y2="520" stroke="url(#frameGrad)" stroke-width="6" stroke-linecap="round"/>
    <!-- Chain stay -->
    <line x1="340" y1="510" x2="310" y2="530" stroke="url(#frameGrad)" stroke-width="5" stroke-linecap="round"/>
    <!-- Seat stay -->
    <line x1="340" y1="510" x2="310" y2="530" stroke="url(#frameGrad)" stroke-width="4" stroke-linecap="round"/>
    <!-- Fork -->
    <line x1="490" y1="395" x2="530" y2="530" stroke="url(#frameGrad)" stroke-width="5" stroke-linecap="round"/>

    <!-- Chain ring area -->
    <circle cx="340" cy="510" r="14" fill="none" stroke="#888" stroke-width="3"/>
    <circle cx="340" cy="510" r="5" fill="#aaa"/>

    <!-- Pedals -->
    <line x1="325" y1="520" x2="355" y2="500" stroke="#777" stroke-width="3" stroke-linecap="round"/>
    <rect x="320" y="517" width="12" height="5" rx="2" fill="#555" stroke="#444" stroke-width="1"/>
    <rect x="350" y="497" width="12" height="5" rx="2" fill="#555" stroke="#444" stroke-width="1"/>

    <!-- Chain (simplified) -->
    <path d="M340,524 Q325,540 310,530" fill="none" stroke="#555" stroke-width="2" stroke-dasharray="3,2"/>

    <!-- Seat post -->
    <line x1="370" y1="385" x2="370" y2="365" stroke="#777" stroke-width="4" stroke-linecap="round"/>

    <!-- Seat -->
    <ellipse cx="370" cy="360" rx="22" ry="8" fill="#5c4033" stroke="#4a3226" stroke-width="2"/>
    <ellipse cx="370" cy="358" rx="20" ry="6" fill="#6d4c3d" opacity="0.6"/>

    <!-- Handlebar stem -->
    <line x1="490" y1="395" x2="495" y2="370" stroke="#888" stroke-width="4" stroke-linecap="round"/>

    <!-- Handlebars -->
    <path d="M475,370 Q495,362 515,370" fill="none" stroke="#888" stroke-width="5" stroke-linecap="round"/>
    <!-- Handlebar grips -->
    <circle cx="475" cy="370" r="4" fill="#333"/>
    <circle cx="515" cy="370" r="4" fill="#333"/>

  </g>

  <!-- === DUCK === -->
  <g class="duck-body-group">

    <!-- Duck tail feathers -->
    <path d="M305,370 Q280,345 295,325 Q305,340 315,355" fill="#daa520" stroke="#b8860b" stroke-width="1.5"/>
    <path d="M310,365 Q290,340 300,320 Q310,335 320,350" fill="#e9c46a" stroke="#daa520" stroke-width="1"/>
    <path d="M315,360 Q300,335 308,318 Q316,332 323,348" fill="#f4e285" stroke="#daa520" stroke-width="1"/>

    <!-- Duck body -->
    <ellipse cx="390" cy="355" rx="80" ry="50" fill="url(#duckBodyGrad)" stroke="#daa520" stroke-width="2"/>

    <!-- Belly highlight -->
    <ellipse cx="400" cy="365" rx="50" ry="25" fill="#f4e285" opacity="0.4"/>

    <!-- Duck wing -->
    <path d="M350,340 Q330,310 360,300 Q380,305 395,330 Q385,345 370,350 Z" fill="#daa520" stroke="#b8860b" stroke-width="1.5"/>
    <path d="M355,335 Q342,315 365,308" fill="none" stroke="#b8860b" stroke-width="1" opacity="0.5"/>

    <!-- Scarf -->
    <g class="scarf-group">
      <path d="M420,310 Q430,315 440,308 Q445,310 442,318 Q438,325 430,320 Q425,325 420,318 Z" fill="url(#scarfGrad)" stroke="#b71c1c" stroke-width="1"/>
      <!-- Scarf tail fluttering -->
      <path d="M420,318 Q400,340 380,335 Q375,330 385,325 Q395,328 415,312" fill="url(#scarfGrad)" stroke="#b71c1c" stroke-width="1" opacity="0.9"/>
      <path d="M380,335 Q370,340 365,332 Q372,330 380,335" fill="#d62828" stroke="#b71c1c" stroke-width="0.5"/>
    </g>

    <!-- Duck neck -->
    <path d="M420,340 Q440,310 445,285 Q448,275 445,270 Q435,280 430,300 Q420,325 410,345" fill="#e9c46a" stroke="#daa520" stroke-width="1.5"/>

    <!-- Duck head -->
    <ellipse cx="450" cy="268" rx="32" ry="28" fill="url(#duckHeadGrad)" stroke="#daa520" stroke-width="1.5"/>

    <!-- Head highlight -->
    <ellipse cx="445" cy="260" rx="15" ry="10" fill="#f4e285" opacity="0.3"/>

    <!-- Duck eye -->
    <ellipse cx="460" cy="260" rx="7" ry="8" fill="#1a1a2e"/>
    <ellipse cx="458" cy="258" rx="3" ry="3.5" fill="#fff"/>
    <ellipse cx="462" cy="262" rx="1.5" ry="1.5" fill="#fff" opacity="0.5"/>

    <!-- Eyebrow (cheeky expression) -->
    <path d="M452,250 Q458,247 466,250" fill="none" stroke="#8b6914" stroke-width="2" stroke-linecap="round"/>

    <!-- Duck beak -->
    <path d="M475,270 Q498,268 500,275 Q498,282 475,280" fill="url(#beakGrad)" stroke="#d45d00" stroke-width="1.5"/>
    <!-- Beak line -->
    <line x1="476" y1="275" x2="498" y2="275" stroke="#c43e00" stroke-width="1"/>
    <!-- Nostril -->
    <ellipse cx="488" cy="272" rx="2" ry="1.5" fill="#c43e00" opacity="0.6"/>

    <!-- Duck cap (helmet for safety!) -->
    <path d="M422,255 Q430,238 455,235 Q475,233 480,242 Q482,250 478,255" fill="#2a9d8f" stroke="#1a7a6e" stroke-width="2"/>
    <!-- Cap brim -->
    <path d="M420,255 Q435,252 470,252 Q480,253 480,255 Q478,260 470,260 Q440,262 420,258" fill="#1a7a6e" stroke="#155e54" stroke-width="1"/>
    <!-- Cap button -->
    <circle cx="452" cy="236" r="3" fill="#f4e285"/>

    <!-- Duck feet on pedals -->
    <g>
      <!-- Left foot (on left pedal) -->
      <path d="M328,520 L320,528 L345,528 L340,520" fill="#f77f00" stroke="#e36414" stroke-width="1.5"/>
      <!-- Right foot (on right pedal) -->
      <path d="M350,498 L342,506 L367,506 L362,498" fill="#f77f00" stroke="#e36414" stroke-width="1.5"/>
    </g>

    <!-- Duck arms/wings reaching handlebars -->
    <path d="M400,340 Q430,350 465,365 Q475,368 475,370" fill="none" stroke="#daa520" stroke-width="5" stroke-linecap="round"/>
    <path d="M380,350 Q420,360 465,370" fill="none" stroke="#e9c46a" stroke-width="4" stroke-linecap="round"/>

    <!-- Wing tips on handlebars -->
    <ellipse cx="475" cy="370" rx="6" ry="5" fill="#daa520" stroke="#b8860b" stroke-width="1"/>
    <ellipse cx="515" cy="370" rx="6" ry="5" fill="#daa520" stroke="#b8860b" stroke-width="1"/>

  </g>

  <!-- Small dust puffs behind rear wheel -->
  <g opacity="0.15">
    <circle cx="260" cy="555" r="8" fill="#a8dadc">
      <animate attributeName="cx" values="260;230;200" dur="2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.15;0.05;0" dur="2s" repeatCount="indefinite"/>
    </circle>
    <circle cx="250" cy="548" r="5" fill="#a8dadc">
      <animate attributeName="cx" values="250;215;185" dur="2.3s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.12;0.04;0" dur="2.3s" repeatCount="indefinite"/>
    </circle>
    <circle cx="270" cy="560" r="6" fill="#a8dadc">
      <animate attributeName="cx" values="270;240;210" dur="1.8s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.1;0.03;0" dur="1.8s" repeatCount="indefinite"/>
    </circle>
  </g>

  <!-- Small stars in the sky -->
  <circle cx="100" cy="60" r="1.5" fill="#f4e285" opacity="0.5"/>
  <circle cx="300" cy="40" r="1" fill="#f4e285" opacity="0.4"/>
  <circle cx="450" cy="55" r="1.5" fill="#f4e285" opacity="0.3"/>
  <circle cx="700" cy="180" r="1" fill="#f4e285" opacity="0.3"/>
  <circle cx="550" cy="130" r="1.2" fill="#f4e285" opacity="0.4"/>

  <!-- Small flowers on the ground -->
  <g opacity="0.6">
    <circle cx="120" cy="610" r="4" fill="#e63946"/>
    <circle cx="122" cy="608" r="2" fill="#f4e285"/>
    <circle cx="640" cy="605" r="4" fill="#e63946"/>
    <circle cx="642" cy="603" r="2" fill="#f4e285"/>
    <circle cx="720" cy="615" r="3.5" fill="#f77f00"/>
    <circle cx="722" cy="613" r="1.8" fill="#f4e285"/>
    <circle cx="80" cy="618" r="3.5" fill="#f77f00"/>
    <circle cx="82" cy="616" r="1.8" fill="#f4e285"/>
  </g>

  <!-- Grass tufts -->
  <g opacity="0.4" stroke="#3a5a40" stroke-width="1.5" stroke-linecap="round" fill="none">
    <path d="M50,590 Q55,575 52,565"/>
    <path d="M55,590 Q58,578 60,568"/>
    <path d="M200,588 Q203,573 200,563"/>
    <path d="M205,588 Q208,576 210,566"/>
    <path d="M600,590 Q603,575 600,565"/>
    <path d="M605,590 Q608,578 610,568"/>
    <path d="M750,592 Q753,577 750,567"/>
    <path d="M755,592 Q758,580 760,570"/>
  </g>

</svg>
```

</details>

## 4. glm-5.2_cloud

**SVG size:** 6217 characters  
**Raw response:** 6255 characters

![glm-5.2_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/glm-5.2_cloud.svg)

### Analysis

This SVG contains approximately **57 shape elements** and uses **18 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Uses `<use>` references to reuse defined elements, reducing duplication
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <defs>
    <g id="spokes">
      <line x1="0" y1="-50" x2="0" y2="50" stroke="#BDC3C7" stroke-width="2"/>
      <line x1="-50" y1="0" x2="50" y2="0" stroke="#BDC3C7" stroke-width="2"/>
      <line x1="-35" y1="-35" x2="35" y2="35" stroke="#BDC3C7" stroke-width="2"/>
      <line x1="35" y1="-35" x2="-35" y2="35" stroke="#BDC3C7" stroke-width="2"/>
    </g>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="#87CEEB"/>
  
  <!-- Sun -->
  <circle cx="700" cy="100" r="50" fill="#FFD700"/>
  <circle cx="700" cy="100" r="60" fill="#FFD700" opacity="0.3"/>

  <!-- Clouds -->
  <path d="M100 150 Q 150 130 200 150 Q 200 180 100 180 Z" fill="#FFFFFF" opacity="0.9"/>
  <path d="M120 140 Q 160 120 200 140 Q 200 160 120 160 Z" fill="#FFFFFF" opacity="0.7"/>
  
  <path d="M400 100 Q 450 80 500 100 Q 500 130 400 130 Z" fill="#FFFFFF" opacity="0.8"/>
  <path d="M420 90 Q 460 70 500 90 Q 500 110 420 110 Z" fill="#FFFFFF" opacity="0.6"/>

  <!-- Ground and Road -->
  <path d="M0 450 Q 400 420 800 450 L800 600 L0 600 Z" fill="#A2D9A2"/>
  <path d="M0 460 Q 400 440 800 460 L800 600 L0 600 Z" fill="#6B6B6B"/>
  <path d="M0 460 Q 400 440 800 460" stroke="#FFFFFF" stroke-width="4" stroke-dasharray="20 15" fill="none"/>

  <!-- Motion Lines -->
  <line x1="50" y1="200" x2="120" y2="200" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.6"/>
  <line x1="80" y1="250" x2="180" y2="250" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.6"/>
  <line x1="30" y1="300" x2="110" y2="300" stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.6"/>

  <!-- Bicycle Frame -->
  <g id="bicycle-frame">
    <!-- Rear Wheel -->
    <g transform="translate(220, 400)">
      <circle cx="0" cy="0" r="60" fill="none" stroke="#2C3E50" stroke-width="10"/>
      <circle cx="0" cy="0" r="50" fill="none" stroke="#95A5A6" stroke-width="4"/>
      <g>
        <use href="#spokes"/>
        <animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="1.5s" repeatCount="indefinite"/>
      </g>
    </g>

    <!-- Front Wheel -->
    <g transform="translate(580, 400)">
      <circle cx="0" cy="0" r="60" fill="none" stroke="#2C3E50" stroke-width="10"/>
      <circle cx="0" cy="0" r="50" fill="none" stroke="#95A5A6" stroke-width="4"/>
      <g>
        <use href="#spokes"/>
        <animateTransform attributeName="transform" type="rotate" from="0 0 0" to="360 0 0" dur="1.5s" repeatCount="indefinite"/>
      </g>
    </g>

    <!-- Chain -->
    <line x1="220" y1="400" x2="400" y2="400" stroke="#555" stroke-width="3" stroke-dasharray="3 2"/>

    <!-- Chain Stay -->
    <line x1="220" y1="400" x2="400" y2="400" stroke="#E74C3C" stroke-width="8"/>
    <!-- Seat Stay -->
    <line x1="220" y1="400" x2="300" y2="280" stroke="#E74C3C" stroke-width="8"/>
    <!-- Seat Tube -->
    <line x1="400" y1="400" x2="300" y2="280" stroke="#E74C3C" stroke-width="8"/>
    <!-- Top Tube -->
    <line x1="300" y1="280" x2="580" y2="280" stroke="#E74C3C" stroke-width="8"/>
    <!-- Down Tube -->
    <line x1="400" y1="400" x2="580" y2="280" stroke="#E74C3C" stroke-width="8"/>
    <!-- Front Fork -->
    <line x1="580" y1="280" x2="580" y2="400" stroke="#E74C3C" stroke-width="8"/>

    <!-- Seat -->
    <rect x="280" y="265" width="40" height="10" rx="5" fill="#2C3E50"/>
    
    <!-- Handlebars -->
    <path d="M580 280 C 620 260, 630 280, 620 300 Q 610 310 600 300" fill="none" stroke="#2C3E50" stroke-width="6" stroke-linecap="round"/>

    <!-- Crank and Pedals -->
    <line x1="400" y1="400" x2="430" y2="430" stroke="#7F8C8D" stroke-width="6"/>
    <rect x="425" y="425" width="15" height="6" fill="#2C3E50" transform="rotate(-45 432.5 428)"/>
    <!-- Back Pedal -->
    <line x1="400" y1="400" x2="370" y2="370" stroke="#7F8C8D" stroke-width="6"/>
    <rect x="365" y="365" width="15" height="6" fill="#2C3E50" transform="rotate(135 372.5 368)"/>
  </g>

  <!-- The Duck -->
  <g id="duck">
    <!-- Tail -->
    <polygon points="170,245 130,215 170,245" fill="#FFFFFF"/>
    <polygon points="170,245 130,215 175,235" fill="#F0F8FF"/>
    
    <!-- Body -->
    <ellipse cx="240" cy="250" rx="60" ry="40" fill="#FFFFFF"/>
    
    <!-- Neck -->
    <path d="M270 220 Q 290 180 340 160 Q 360 170 330 210 Q 290 240 270 220 Z" fill="#FFFFFF"/>
    
    <!-- Head -->
    <circle cx="340" cy="160" r="25" fill="#FFFFFF"/>
    
    <!-- Beak -->
    <path d="M360 150 Q 410 150 410 160 Z" fill="#FFA500"/>
    <path d="M360 160 Q 410 160 410 170 Q 410 170 360 170 Z" fill="#FFCC00"/>
    <line x1="360" y1="160" x2="410" y2="160" stroke="#E67E22" stroke-width="1"/>

    <!-- Eye -->
    <circle cx="345" cy="155" r="4" fill="#000000"/>
    <circle cx="346" cy="154" r="1.5" fill="#FFFFFF"/>

    <!-- Helmet -->
    <path d="M315 135 Q 340 120 365 135 Q 365 150 315 150 Z" fill="#E74C3C"/>
    <rect x="315" y="145" width="50" height="5" fill="#C0392B"/>
    <rect x="325" y="150" width="30" height="8" fill="#E74C3C"/>
    
    <!-- Scarf blowing in wind -->
    <path d="M270 200 Q 200 160 120 180 Q 130 200 120 220 Q 180 200 240 210 Z" fill="#E74C3C"/>
    <path d="M270 200 Q 220 170 160 190 Q 180 210 160 220 Q 200 200 240 210 Z" fill="#C0392B"/>

    <!-- Back Leg (Background) -->
    <path d="M220 280 Q 280 330 370 370" stroke="#FFA500" stroke-width="5" fill="none" stroke-linecap="round"/>
    <!-- Back Webbed Foot -->
    <path d="M370 370 L 390 380 L 380 360 L 365 365 Z" fill="#E67E22"/>

    <!-- Front Leg (Foreground) -->
    <path d="M250 280 Q 350 380 430 430" stroke="#FFA500" stroke-width="6" fill="none" stroke-linecap="round"/>
    <!-- Front Webbed Foot on Pedal -->
    <path d="M420 425 L 450 440 L 445 420 L 425 420 Z" fill="#FFA500"/>

    <!-- Wing -->
    <path d="M220 220 Q 250 260 300 240 Q 280 210 220 220 Z" fill="#F0F8FF"/>
    <path d="M240 225 Q 260 250 290 235" stroke="#D6EAF8" stroke-width="2" fill="none"/>

    <!-- Arm holding handlebar -->
    <path d="M250 210 Q 400 180 590 285" stroke="#FFFFFF" stroke-width="12" fill="none" stroke-linecap="round"/>
    <circle cx="590" cy="285" r="8" fill="#FFA500"/>
  </g>
</svg>
```

</details>

## 5. kimi-k2.6_cloud

**SVG size:** 8262 characters  
**Raw response:** 30431 characters

![kimi-k2.6_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/kimi-k2.6_cloud.svg)

### Analysis

This SVG contains approximately **41 shape elements** and uses **24 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" id="scene" viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid slice" class="w-full h-full max-w-[1400px]">
            <defs>
                <!-- Sky Gradient -->
                <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#38bdf8;stop-opacity:1" id="skyStop1" />
                    <stop offset="100%" style="stop-color:#bae6fd;stop-opacity:1" id="skyStop2" />
                </linearGradient>

                <!-- Ground Gradient -->
                <linearGradient id="groundGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#4ade80;stop-opacity:1" id="groundStop1" />
                    <stop offset="100%" style="stop-color:#16a34a;stop-opacity:1" id="groundStop2" />
                </linearGradient>

                <!-- Sun Gradient -->
                <radialGradient id="sunGradient">
                    <stop offset="0%" style="stop-color:#fef3c7;stop-opacity:1" />
                    <stop offset="80%" style="stop-color:#f59e0b;stop-opacity:1" />
                </radialGradient>

                <!-- Duck Body Gradient -->
                <radialGradient id="duckBodyGrad" cx="30%" cy="30%" r="70%">
                    <stop offset="0%" stop-color="#ffffff" />
                    <stop offset="100%" stop-color="#e2e8f0" />
                </radialGradient>
                
                <!-- Wheel Clip Path -->
                <clipPath id="wheelClip">
                    <circle cx="0" cy="0" r="50" />
                </clipPath>
            </defs>

            <!-- Background Layer -->
            <rect id="skyRect" x="0" y="0" fill="url(#skyGradient)" />
            
            <!-- Sun/Moon -->
            <g id="celestialBody" transform="translate(850, 100)">
                <circle r="60" fill="url(#sunGradient)" />
                <circle r="65" fill="#fbbf24" opacity="0.3" />
            </g>

            <!-- Clouds (Generated via JS) -->
            <g id="cloudsLayer"></g>

            <!-- Mountains -->
            <path d="M-100 450 L200 250 L500 450 L800 300 L1100 450 Z" fill="#15803d" opacity="0.6" id="mountainBack" />
            <path d="M-50 450 L300 350 L600 450 L900 380 L1150 450 Z" fill="#166534" opacity="0.8" id="mountainFront" />

            <!-- Ground -->
            <rect id="groundRect" x="0" y="450" width="1000" height="150" fill="url(#groundGradient)" />

            <!-- Road Lines -->
            <g id="roadLines"></g>

            <!-- Shadow -->
            <ellipse cx="500" cy="520" rx="100" ry="15" fill="#000" opacity="0.2" id="duckShadow" />

            <!-- THE BIKE & DUCK GROUP -->
            <g id="bikeGroup" transform="translate(500, 450)">
                
                <!-- Back Wheel -->
                <g id="wheelBack" transform="translate(-100, 0)">
                    <circle r="50" fill="transparent" stroke="#334155" stroke-width="8" />
                    <circle r="45" fill="transparent" stroke="#94a3b8" stroke-width="2" />
                    <g id="spokesBack" stroke="#cbd5e1" stroke-width="1">
                        <!-- Spokes generated by JS -->
                    </g>
                </g>

                <!-- Front Wheel -->
                <g id="wheelFront" transform="translate(100, 0)">
                    <circle r="50" fill="transparent" stroke="#334155" stroke-width="8" />
                    <circle r="45" fill="transparent" stroke="#94a3b8" stroke-width="2" />
                    <g id="spokesFront" stroke="#cbd5e1" stroke-width="1">
                        <!-- Spokes generated by JS -->
                    </g>
                </g>

                <!-- Frame -->
                <path d="M-100 0 L0 -60 L100 0 L0 0 Z" fill="none" stroke="#ef4444" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" />
                <path d="M-100 0 L0 -90" fill="none" stroke="#ef4444" stroke-width="8" stroke-linecap="round" />
                <path d="M0 -60 L0 -90" fill="none" stroke="#ef4444" stroke-width="8" stroke-linecap="round" />
                <path d="M0 -90 L60 -110" fill="none" stroke="#ef4444" stroke-width="6" stroke-linecap="round" /> <!-- Handlebar stem -->
                <line x1="60" y1="-110" x2="80" y2="-115" stroke="#334155" stroke-width="6" stroke-linecap="round" /> <!-- Handlebar -->
                <path d="M-100 0 L-110 -15" stroke="#334155" stroke-width="6" stroke-linecap="round" /> <!-- Rear Peg -->

                <!-- Pedals & Crank -->
                <g id="pedalAssembly" transform="translate(0,0)">
                    <line x1="0" y1="0" x2="0" y2="0" stroke="#475569" stroke-width="5" id="crankArm" />
                    <circle r="8" fill="#64748b" />
                </g>

                <!-- THE DUCK -->
                <g id="duck" transform="translate(0, -90)"> <!-- Positioned on seat -->
                    
                    <!-- Far Leg (Darker) -->
                    <g id="legFar" transform="translate(5, 35)">
                         <path d="M0 0 L0 20 L10 20" fill="none" stroke="#fbbf24" stroke-width="6" stroke-linecap="round" id="farLegPath"/>
                         <path d="M0 0 L0 0" fill="none" stroke="#fbbf24" stroke-width="6" stroke-linecap="round" opacity="0.5" /> <!-- placeholder for thigh -->
                    </g>

                    <!-- Body -->
                    <ellipse cx="0" cy="0" rx="35" ry="40" fill="#fcd34d" id="duckTorso" />
                    <ellipse cx="0" cy="5" rx="25" ry="30" fill="#ffffff" opacity="0.4" /> <!-- Belly highlight -->

                    <!-- Wing (Flapping) -->
                    <path id="wing" d="M10 -10 Q30 0 10 20" fill="#fbbf24" stroke="#d97706" stroke-width="2" transform-origin="10 0" />

                    <!-- Head -->
                    <circle cx="0" cy="-45" r="30" fill="#fcd34d" id="duckHead" />
                    
                    <!-- Beak -->
                    <path d="M20 -45 L40 -40 L20 -35 Z" fill="#f97316" id="duckBeak" />
                    <path d="M20 -45 L35 -42 L20 -40 Z" fill="#fb923c" />

                    <!-- Eye -->
                    <g id="eyeGroup" transform="translate(8, -55)">
                        <circle r="6" fill="white" />
                        <circle id="pupil" r="2.5" fill="black" cx="1" cy="0" />
                    </g>

                    <!-- Near Leg -->
                    <g id="legNear" transform="translate(-5, 35)">
                         <path d="M0 0 L0 20 L10 20" fill="none" stroke="#fbbf24" stroke-width="6" stroke-linecap="round" id="nearLegPath"/>
                    </g>

                    <!-- Accessories -->
                    <g id="accessories">
                        <!-- Glasses -->
                        <g id="accGlasses" display="none" transform="translate(5, -55)">
                            <circle r="8" fill="#000" opacity="0.7" />
                            <circle cx="20" r="8" fill="#000" opacity="0.7" />
                            <line x1="0" y1="0" x2="20" y2="0" stroke="#000" stroke-width="2" />
                            <line x1="0" y1="0" x2="-10" y2="-5" stroke="#000" stroke-width="2" />
                        </g>
                        <!-- Top Hat -->
                        <g id="accHat" display="none" transform="translate(0, -75)">
                            <rect x="-20" y="0" width="40" height="5" fill="#1e293b" />
                            <rect x="-12" y="-30" width="24" height="30" fill="#1e293b" />
                            <rect x="-12" y="-10" width="24" height="5" fill="#ef4444" />
                        </g>
                        <!-- Bow Tie -->
                        <g id="accBowtie" display="none" transform="translate(0, -15)">
                            <path d="M-10 -5 L0 0 L-10 5 Z" fill="#ef4444" />
                            <path d="M10 -5 L0 0 L10 5 Z" fill="#ef4444" />
                            <circle r="3" fill="#b91c1c" />
                        </g>
                    </g>
                </g>
            </g>
            
            <!-- Wind lines -->
            <g id="windLines"></g>

        </svg>
```

</details>

## 6. minimax-m2.7_cloud

**SVG size:** 3490 characters  
**Raw response:** 3684 characters

![minimax-m2.7_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/minimax-m2.7_cloud.svg)

### Analysis

This SVG contains approximately **35 shape elements** and uses **14 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="600" height="400" fill="#E0F7FA" />
  <rect y="300" width="600" height="100" fill="#81C784" />
  
  <!-- Motion Lines -->
  <g stroke="white" stroke-width="4" stroke-linecap="round" opacity="0.6">
    <line x1="50" y1="150" x2="100" y2="150" />
    <line x1="20" y1="180" x2="80" y2="180" />
    <line x1="40" y1="210" x2="90" y2="210" />
  </g>

  <!-- Bicycle Group -->
  <g id="bicycle">
    <!-- Wheels -->
    <g stroke="#333" stroke-width="8" fill="none">
        <circle cx="120" cy="250" r="60" />
        <circle cx="450" cy="250" r="60" />
    </g>
    
    <!-- Spokes -->
    <g stroke="#CCC" stroke-width="2">
        <g transform="translate(120, 250)">
            <line x1="0" y1="-50" x2="0" y2="50" />
            <line x1="-50" y1="0" x2="50" y2="0" />
            <line x1="-35" y1="-35" x2="35" y2="35" />
            <line x1="35" y1="-35" x2="-35" y2="35" />
        </g>
        <g transform="translate(450, 250)">
            <line x1="0" y1="-50" x2="0" y2="50" />
            <line x1="-50" y1="0" x2="50" y2="0" />
            <line x1="-35" y1="-35" x2="35" y2="35" />
            <line x1="35" y1="-35" x2="-35" y2="35" />
        </g>
    </g>

    <!-- Frame -->
    <path d="M120 250 L280 250 L320 160 L120 250 M280 250 L320 160 L410 160 L280 250 M410 160 L450 250 M320 160 L340 120" 
          stroke="#D32F2F" stroke-width="10" fill="none" stroke-linejoin="round" stroke-linecap="round"/>
    
    <!-- Seat Post -->
    <line x1="320" y1="160" x2="310" y2="110" stroke="#333" stroke-width="8"/>
    <!-- Seat -->
    <ellipse cx="310" cy="110" rx="30" ry="10" fill="#8B0000" />
    
    <!-- Handlebars -->
    <path d="M340 120 Q 340 90 310 90" stroke="#333" stroke-width="6" fill="none"/>
    <circle cx="310" cy="90" r="8" fill="#333" />

    <!-- Pedal Gear -->
    <circle cx="280" cy="250" r="20" fill="#555" />
    <line x1="280" y1="250" x2="300" y2="270" stroke="#222" stroke-width="6" /> <!-- Crank -->
    <rect x="295" y="265" width="15" height="8" fill="#222" rx="2"/> <!-- Pedal -->
  </g>

  <!-- Duck Group -->
  <g id="duck">
    <!-- Back Leg (Left) -->
    <path d="M280 190 L290 240 L260 260" stroke="#FFA000" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    
    <!-- Body -->
    <ellipse cx="290" cy="180" rx="55" ry="65" fill="#FFC107" transform="rotate(-10 290 180)" />
    
    <!-- Head -->
    <circle cx="330" cy="120" r="40" fill="#FFC107" />
    
    <!-- Beak -->
    <path d="M355 125 L420 130 L355 145 Z" fill="#FF9800" stroke="#E65100" stroke-width="1"/>
    
    <!-- Eye -->
    <circle cx="350" cy="110" r="6" fill="#212121" />
    <circle cx="352" cy="108" r="2" fill="white" />
    
    <!-- Helmet -->
    <path d="M300 100 Q 330 60 360 100" fill="#1976D2" />
    <circle cx="330" cy="90" r="42" fill="none" stroke="#1976D2" stroke-width="5"/>

    <!-- Wing (Right Arm) holding handlebar -->
    <path d="M320 170 Q 350 200 330 120" stroke="#FFA000" stroke-width="12" fill="none" stroke-linecap="round" />
    <!-- Hand/Thumb gripping -->
    <ellipse cx="325" cy="95" rx="6" ry="8" fill="#FFA000" />

    <!-- Front Leg (Right) -->
    <path d="M300 200 L310 250 L330 260" stroke="#FFA000" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>

    <!-- Foot on Pedal -->
    <path d="M310 260 L340 260 L340 275 L310 275 Z" fill="#FF9800" />
  </g>
</svg>
```

</details>

## 7. nemotron-3-super_cloud

**SVG size:** 1954 characters  
**Raw response:** 1979 characters

![nemotron-3-super_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/nemotron-3-super_cloud.svg)

### Analysis

This SVG contains approximately **23 shape elements** and uses **9 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 300 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="100%" height="100%" fill="#e0f7ff"/>

  <!-- Bicycle -->
  <!-- Wheels -->
  <circle cx="70" cy="150" r="20" fill="#fff" stroke="#333" stroke-width="2"/>
  <circle cx="230" cy="150" r="20" fill="#fff" stroke="#333" stroke-width="2"/>
  <!-- Spokes -->
  <line x1="70" y1="130" x2="70" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="70" y1="150" x2="90" y2="150" stroke="#333" stroke-width="1"/>
  <line x1="230" y1="130" x2="230" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="230" y1="150" x2="250" y2="150" stroke="#333" stroke-width="1"/>
  <!-- Frame -->
  <path d="M90,130 L120,90 L150,110 L190,110 L210,130" fill="none" stroke="#333" stroke-width="4"/>
  <!-- Seat -->
  <ellipse cx="120" cy="85" rx="15" ry="8" fill="#8B4513"/>
  <!-- Handlebars -->
  <line x1="190" y1="110" x2="210" y2="90" stroke="#333" stroke-width="4"/>
  <line x1="210" y1="90" x2="220" y2="100" stroke="#333" stroke-width="4"/>
  <!-- Pedals -->
  <circle cx="150" cy="110" r="4" fill="#555"/>
  <circle cx="190" cy="110" r="4" fill="#555"/>
  <line x1="150" y1="106" x2="150" y2="114" stroke="#555" stroke-width="2"/>
  <line x1="190" y1="106" x2="190" y2="114" stroke="#555" stroke-width="2"/>

  <!-- Duck -->
  <!-- Body -->
  <ellipse cx="140" cy="100" rx="30" ry="20" fill="#FFD700"/>
  <!-- Head -->
  <circle cx="100" cy="90" r="18" fill="#FFD700"/>
  <!-- Beak -->
  <polygon points="82,90 66,85 82,80" fill="#FFA500"/>
  <!-- Eye -->
  <circle cx="106" cy="85" r="3" fill="#000"/>
  <!-- Wing -->
  <path d="M130,100 q10,-20 30,0" fill="none" stroke="#8B8000" stroke-width="4"/>
  <!-- Tail -->
  <polygon points="170,105 190,95 190,115" fill="#FFA500"/>
  <!-- Legs (on pedals) -->
  <line x1="140" y1="110" x2="150" y2="110" stroke="#8B4513" stroke-width="3"/>
  <line x1="140" y1="110" x2="130" y2="110" stroke="#8B4513" stroke-width="3"/>
</svg>
```

</details>

## 8. nemotron-3-ultra_cloud

**SVG size:** 8548 characters  
**Raw response:** 9537 characters

![nemotron-3-ultra_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/nemotron-3-ultra_cloud.svg)

### Analysis

This SVG contains approximately **67 shape elements** and uses **19 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Applies SVG filters (e.g., drop shadows) for visual effects
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300">
  <!-- Definitions for Gradients & Reusable Parts -->
  <defs>
    <!-- Duck Body Gradient -->
    <linearGradient id="duckBodyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#FFD700;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFA500;stop-opacity:1" />
    </linearGradient>
    <!-- Bike Frame Gradient -->
    <linearGradient id="bikeFrameGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#4A90E2;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#2C3E50;stop-opacity:1" />
    </linearGradient>
    <!-- Tire Gradient -->
    <radialGradient id="tireGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#444;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#111;stop-opacity:1" />
    </radialGradient>
    <!-- Rim Gradient -->
    <linearGradient id="rimGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#DDD;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#888;stop-opacity:1" />
    </linearGradient>
    <!-- Ground Shadow -->
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

  <!-- Ground Line -->
  <line x1="0" y1="240" x2="400" y2="240" stroke="#888" stroke-width="2" stroke-dasharray="5,5" opacity="0.5"/>

  <!-- ==================== BICYCLE ==================== -->
  <g id="bicycle" filter="url(#dropShadow)">
    <!-- Rear Wheel -->
    <g id="rearWheel" transform="translate(110, 200)">
      <circle r="45" fill="url(#tireGrad)" stroke="#222" stroke-width="3"/>
      <circle r="36" fill="none" stroke="url(#rimGrad)" stroke-width="2"/>
      <!-- Spokes -->
      <g stroke="url(#rimGrad)" stroke-width="1" opacity="0.8">
        <line x1="0" y1="0" x2="0" y2="-36"/>
        <line x1="0" y1="0" x2="36" y2="0"/>
        <line x1="0" y1="0" x2="-36" y2="0"/>
        <line x1="0" y1="0" x2="0" y2="36"/>
        <line x1="0" y1="0" x2="25.5" y2="-25.5"/>
        <line x1="0" y1="0" x2="-25.5" y2="-25.5"/>
        <line x1="0" y1="0" x2="25.5" y2="25.5"/>
        <line x1="0" y1="0" x2="-25.5" y2="25.5"/>
      </g>
      <circle r="6" fill="url(#rimGrad)" stroke="#555" stroke-width="1"/> <!-- Hub -->
    </g>

    <!-- Front Wheel -->
    <g id="frontWheel" transform="translate(300, 200)">
      <circle r="45" fill="url(#tireGrad)" stroke="#222" stroke-width="3"/>
      <circle r="36" fill="none" stroke="url(#rimGrad)" stroke-width="2"/>
      <g stroke="url(#rimGrad)" stroke-width="1" opacity="0.8">
        <line x1="0" y1="0" x2="0" y2="-36"/>
        <line x1="0" y1="0" x2="36" y2="0"/>
        <line x1="0" y1="0" x2="-36" y2="0"/>
        <line x1="0" y1="0" x2="0" y2="36"/>
        <line x1="0" y1="0" x2="25.5" y2="-25.5"/>
        <line x1="0" y1="0" x2="-25.5" y2="-25.5"/>
        <line x1="0" y1="0" x2="25.5" y2="25.5"/>
        <line x1="0" y1="0" x2="-25.5" y2="25.5"/>
      </g>
      <circle r="6" fill="url(#rimGrad)" stroke="#555" stroke-width="1"/>
    </g>

    <!-- Frame -->
    <g stroke="url(#bikeFrameGrad)" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" fill="none">
      <!-- Top tube & Down tube -->
      <path d="M 110 155 L 260 155 L 260 200 L 110 200 Z" />
      <!-- Seat tube -->
      <line x1="160" y1="155" x2="160" y2="200" />
      <!-- Chain stays (rear triangle bottom) -->
      <path d="M 110 200 L 160 155" />
      <!-- Seat stays (rear triangle top) -->
      <path d="M 160 105 L 110 200" />
      <!-- Fork -->
      <path d="M 260 155 L 300 155 L 300 200" />
      <!-- Handlebars stem -->
      <line x1="260" y1="155" x2="260" y2="110" />
      <!-- Handlebars -->
      <path d="M 240 110 L 260 100 L 280 110" stroke-width="4" />
      <!-- Grips -->
      <circle cx="238" cy="110" r="5" fill="#333" />
      <circle cx="282" cy="110" r="5" fill="#333" />
    </g>

    <!-- Seat -->
    <path d="M 145 105 Q 160 95 180 105" stroke="url(#bikeFrameGrad)" stroke-width="8" stroke-linecap="round" fill="none" />
    <ellipse cx="160" cy="100" rx="20" ry="6" fill="#2C3E50" />
    <ellipse cx="160" cy="98" rx="18" ry="4" fill="#34495E" />

    <!-- Pedals & Crank -->
    <g transform="translate(160, 200)" stroke="#555" stroke-width="3" stroke-linecap="round">
      <!-- Crank Arms -->
      <line x1="0" y1="0" x2="-30" y2="-15" id="crankLeft" /> 
      <line x1="0" y1="0" x2="30" y2="15" id="crankRight" />
      <!-- Pedals -->
      <line x1="-30" y1="-15" x2="-50" y2="-20" stroke="#777" stroke-width="4" />
      <line x1="30" y1="15" x2="50" y2="20" stroke="#777" stroke-width="4" />
    </g>

    <!-- Chain (Simple representation) -->
    <path d="M 150 200 Q 180 200 200 200 Q 230 200 260 200" stroke="#444" stroke-width="1.5" fill="none" opacity="0.6" />
    <path d="M 150 202 Q 180 202 200 202 Q 230 202 260 202" stroke="#444" stroke-width="1.5" fill="none" opacity="0.6" />
  </g>

  <!-- ==================== DUCK ==================== -->
  <g id="duck" transform="translate(0, 0)" filter="url(#dropShadow)">
    <!-- Tail Feathers (Back) -->
    <path d="M 115 125 Q 100 110 95 130 Q 105 125 115 125" fill="url(#duckBodyGrad)" stroke="#E69500" stroke-width="1"/>
    <path d="M 115 125 Q 100 130 95 115 Q 105 125 115 125" fill="url(#duckBodyGrad)" stroke="#E69500" stroke-width="1"/>

    <!-- Body -->
    <ellipse cx="170" cy="125" rx="55" ry="40" fill="url(#duckBodyGrad)" stroke="#E69500" stroke-width="1.5"/>
    <!-- White Chest/Belly highlight -->
    <ellipse cx="180" cy="135" rx="35" ry="25" fill="#FFF8DC" opacity="0.4" />

    <!-- Wing (Resting on handlebar) -->
    <path d="M 210 120 Q 240 95 265 110 Q 250 125 210 120" fill="url(#duckBodyGrad)" stroke="#E69500" stroke-width="1.5" />
    <path d="M 215 122 Q 235 105 250 115" stroke="#E69500" stroke-width="1" fill="none" opacity="0.5"/>

    <!-- Neck -->
    <path d="M 215 100 Q 235 75 240 60 Q 245 75 225 95" fill="url(#duckBodyGrad)" stroke="#E69500" stroke-width="1.5" />

    <!-- Head -->
    <ellipse cx="240" cy="50" rx="28" ry="22" fill="url(#duckBodyGrad)" stroke="#E69500" stroke-width="1.5"/>
    
    <!-- Eye -->
    <g transform="translate(250, 45)">
      <ellipse cx="0" cy="0" rx="6" ry="7" fill="white" />
      <ellipse cx="2" cy="1" rx="3" ry="3.5" fill="#111" />
      <circle cx="4" cy="0" r="1.2" fill="white" /> <!-- Highlight -->
      <!-- Eyebrow -->
      <path d="M -8 -10 Q 0 -14 8 -10" stroke="#333" stroke-width="1.5" fill="none" stroke-linecap="round"/>
    </g>

    <!-- Beak -->
    <path d="M 265 50 Q 295 50 290 60 Q 295 50 265 58 Z" fill="#FF8C00" stroke="#E67300" stroke-width="1.5" />
    <!-- Nostril -->
    <ellipse cx="280" cy="53" rx="1.5" ry="1" fill="#E67300" />
    <!-- Lower Beak / Smile -->
    <path d="M 265 58 Q 275 60 285 58" stroke="#E67300" stroke-width="2" fill="none" stroke-linecap="round"/>

    <!-- Feet on Pedals (Connected to crank arms at 160,200) -->
    <!-- Left Foot (Forward/Down-ish) -->
    <g transform="translate(130, 185) rotate(-20)">
      <path d="M 0 0 L 0 20" stroke="#FF8C00" stroke-width="4" stroke-linecap="round" />
      <path d="M 0 20 L -10 28 L 0 28 L 10 28" stroke="#FF8C00" stroke-width="3" stroke-linecap="round" fill="none" />
    </g>
    <!-- Right Foot (Back/Up-ish) -->
    <g transform="translate(190, 215) rotate(20)">
      <path d="M 0 0 L 0 20" stroke="#FF8C00" stroke-width="4" stroke-linecap="round" />
      <path d="M 0 20 L -10 28 L 0 28 L 10 28" stroke="#FF8C00" stroke-width="3" stroke-linecap="round" fill="none" />
    </g>
  </g>

  <!-- Motion Lines (Speed Effect) -->
  <g opacity="0.4" stroke="#4A90E2" stroke-width="2" stroke-linecap="round">
    <line x1="50" y1="100" x2="20" y2="100" />
    <line x1="50" y1="130" x2="20" y2="130" />
    <line x1="50" y1="160" x2="20" y2="160" />
    <line x1="50" y1="190" x2="20" y2="190" />
  </g>
  
  <!-- Bell on Handlebars -->
  <g transform="translate(235, 100)">
    <circle cx="0" cy="0" r="8" fill="#DDD" stroke="#999" stroke-width="1"/>
    <circle cx="0" cy="0" r="4" fill="#AAA" />
    <rect x="-1" y="8" width="2" height="6" fill="#888" rx="1"/>
  </g>

</svg>
```

</details>

## 9. qwen3.5_397b-cloud

**SVG size:** 3247 characters  
**Raw response:** 3258 characters

![qwen3.5_397b-cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/qwen3.5_397b-cloud.svg)

### Analysis

This SVG contains approximately **31 shape elements** and uses **11 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Definitions for reusable parts -->
  <defs>
    <style>
      .tire { fill: none; stroke: #333; stroke-width: 8; }
      .rim { fill: none; stroke: #AAA; stroke-width: 4; }
      .spoke { stroke: #CCC; stroke-width: 2; }
      .frame { fill: none; stroke: #3498DB; stroke-width: 6; stroke-linecap: round; stroke-linejoin: round; }
      .duck-body { fill: #FFD700; stroke: #DAA520; stroke-width: 2; }
      .duck-beak { fill: #FF8C00; stroke: #DAA520; stroke-width: 1; }
      .duck-eye { fill: #000; }
      .duck-wing { fill: #F0E68C; stroke: #DAA520; stroke-width: 1; }
      .ground-shadow { fill: #000; opacity: 0.2; }
    </style>
  </defs>

  <!-- Ground Shadow -->
  <ellipse cx="250" cy="350" rx="180" ry="10" class="ground-shadow" />

  <!-- Rear Wheel -->
  <g transform="translate(150, 300)">
    <circle r="60" class="tire" />
    <circle r="50" class="rim" />
    <!-- Spokes -->
    <line x1="0" y1="-50" x2="0" y2="50" class="spoke" />
    <line x1="-50" y1="0" x2="50" y2="0" class="spoke" />
    <line x1="-35" y1="-35" x2="35" y2="35" class="spoke" />
    <line x1="35" y1="-35" x2="-35" y2="35" class="spoke" />
  </g>

  <!-- Front Wheel -->
  <g transform="translate(350, 300)">
    <circle r="60" class="tire" />
    <circle r="50" class="rim" />
    <!-- Spokes -->
    <line x1="0" y1="-50" x2="0" y2="50" class="spoke" />
    <line x1="-50" y1="0" x2="50" y2="0" class="spoke" />
    <line x1="-35" y1="-35" x2="35" y2="35" class="spoke" />
    <line x1="35" y1="-35" x2="-35" y2="35" class="spoke" />
  </g>

  <!-- Bike Frame (Back Part) -->
  <path d="M150 300 L220 300 L200 220 L150 300" class="frame" />
  
  <!-- Duck Legs -->
  <path d="M210 260 L210 300 M230 260 L230 300" stroke="#FF8C00" stroke-width="4" stroke-linecap="round" />
  <ellipse cx="210" cy="300" rx="10" ry="5" fill="#FF8C00" />
  <ellipse cx="230" cy="300" rx="10" ry="5" fill="#FF8C00" />

  <!-- Duck Body -->
  <ellipse cx="220" cy="240" rx="50" ry="35" class="duck-body" />

  <!-- Duck Wing -->
  <ellipse cx="210" cy="245" rx="20" ry="12" class="duck-wing" />

  <!-- Duck Neck & Head -->
  <path d="M240 220 Q250 200 260 210" fill="none" stroke="#FFD700" stroke-width="25" stroke-linecap="round" />
  <circle cx="270" cy="200" r="22" class="duck-body" />

  <!-- Duck Beak -->
  <path d="M285 195 L310 200 L285 205 Z" class="duck-beak" />

  <!-- Duck Eye -->
  <circle cx="275" cy="195" r="3" class="duck-eye" />
  <circle cx="276" cy="194" r="1" fill="#FFF" />

  <!-- Bike Frame (Front Part & Handlebars) -->
  <path d="M220 300 L200 220 L260 220 L290 180" class="frame" />
  <path d="M290 180 L280 180 M290 180 L300 180" stroke="#333" stroke-width="6" stroke-linecap="round" />
  
  <!-- Seat -->
  <ellipse cx="190" cy="210" rx="20" ry="8" fill="#333" transform="rotate(-10 190 210)" />

  <!-- Duck Arm/Wing holding handle -->
  <path d="M230 240 Q260 220 290 180" fill="none" stroke="#FFD700" stroke-width="8" stroke-linecap="round" />
  
  <!-- Bandana (Optional detail for fun) -->
  <path d="M255 185 Q270 175 285 185 L270 200 Z" fill="#E74C3C" />
  <circle cx="255" cy="185" r="3" fill="#E74C3C" />
  <circle cx="285" cy="185" r="3" fill="#E74C3C" />

</svg>
```

</details>

## 10. deepseek-v4-flash_0731-cloud

**SVG size:** 11814 characters  
**Raw response:** 11814 characters

![deepseek-v4-flash_0731-cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/deepseek-v4-flash_0731-cloud.svg)

### Analysis

This SVG contains approximately **99 shape elements** and uses **11 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a large, detailed SVG (over 10,000 characters) packed with shapes, gradients, and structure. Best when you want maximum visual richness over file size.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" width="100%" height="auto">
    <defs>
        <!-- Background Sky Gradient -->
        <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#a0e9ff" />
            <stop offset="100%" stop-color="#c1f4c5" />
        </linearGradient>
        <!-- Duck Body Gradient -->
        <radialGradient id="duckGrad" cx="40%" cy="30%" r="60%">
            <stop offset="0%" stop-color="#ffe066" />
            <stop offset="100%" stop-color="#ffb703" />
        </radialGradient>
        <!-- Duck Head Gradient -->
        <radialGradient id="headGrad" cx="35%" cy="30%" r="70%">
            <stop offset="0%" stop-color="#ffe066" />
            <stop offset="100%" stop-color="#f4a261" />
        </radialGradient>
    </defs>
    <!-- Background Layer -->
    <rect width="800" height="600" fill="url(#skyGrad)" />
    <!-- Sun -->
    <circle cx="700" cy="150" r="60" fill="#ffd166" opacity="0.8" />
    <circle cx="700" cy="150" r="80" fill="#ffd166" opacity="0.2" />
    <!-- Clouds -->
    <g fill="#ffffff" opacity="0.8">
        <circle cx="150" cy="120" r="30" />
        <circle cx="180" cy="100" r="40" />
        <circle cx="220" cy="120" r="30" />
        <rect x="150" y="120" width="70" height="30" rx="15" />
    </g>
    <g fill="#ffffff" opacity="0.6">
        <circle cx="550" cy="80" r="40" />
        <circle cx="600" cy="60" r="50" />
        <circle cx="650" cy="80" r="30" />
        <rect x="550" y="80" width="100" height="40" rx="20" />
    </g>
    <!-- Speed Lines (Background) -->
    <g stroke="#ffffff" stroke-linecap="round" opacity="0.5">
        <line x1="50" y1="200" x2="150" y2="200" stroke-width="4" />
        <line x1="100" y1="250" x2="180" y2="250" stroke-width="3" opacity="0.4" />
        <line x1="80" y1="150" x2="140" y2="150" stroke-width="5" opacity="0.6" />
        <line x1="650" y1="280" x2="750" y2="280" stroke-width="4" />
        <line x1="680" y1="320" x2="760" y2="320" stroke-width="3" opacity="0.4" />
    </g>
    <!-- Road -->
    <path d="M 0 530 L 800 530 L 800 600 L 0 600 Z" fill="#6c757d" />
    <path d="M 0 550 L 800 550" stroke="#ffd166" stroke-width="4" stroke-dasharray="20, 15" />
    <!-- ================= BICYCLE ================= -->
    <!-- Back Wheel -->
    <g>
        <circle cx="200" cy="450" r="80" fill="none" stroke="#2b2d42" stroke-width="12" />
        <circle cx="200" cy="450" r="70" fill="none" stroke="#a8dadc" stroke-width="2" stroke-dasharray="10, 5" />
        <!-- Spokes -->
        <g stroke="#a8dadc" stroke-width="2">
            <line x1="200" y1="370" x2="200" y2="530" />
            <line x1="120" y1="450" x2="280" y2="450" />
            <line x1="143" y1="393" x2="257" y2="507" />
            <line x1="257" y1="393" x2="143" y2="507" />
        </g>
        <circle cx="200" cy="450" r="10" fill="#e63946" />
        <circle cx="200" cy="450" r="74" fill="none" stroke="#ffffff" stroke-width="4" stroke-dasharray="10, 30" opacity="0.4" />
    </g>
    <!-- Front Wheel -->
    <g>
        <circle cx="600" cy="450" r="80" fill="none" stroke="#2b2d42" stroke-width="12" />
        <circle cx="600" cy="450" r="70" fill="none" stroke="#a8dadc" stroke-width="2" stroke-dasharray="10, 5" />
        <!-- Spokes -->
        <g stroke="#a8dadc" stroke-width="2">
            <line x1="600" y1="370" x2="600" y2="530" />
            <line x1="520" y1="450" x2="680" y2="450" />
            <line x1="543" y1="393" x2="657" y2="507" />
            <line x1="657" y1="393" x2="543" y2="507" />
        </g>
        <circle cx="600" cy="450" r="10" fill="#e63946" />
        <circle cx="600" cy="450" r="74" fill="none" stroke="#ffffff" stroke-width="4" stroke-dasharray="10, 30" opacity="0.4" />
    </g>
    <!-- Chain &amp; Gears -->
    <g>
        <circle cx="400" cy="450" r="30" fill="none" stroke="#a8dadc" stroke-width="6" />
        <circle cx="200" cy="450" r="15" fill="none" stroke="#a8dadc" stroke-width="6" />
        <!-- Chain links -->
        <line x1="200" y1="435" x2="400" y2="420" stroke="#a8dadc" stroke-width="4" stroke-dasharray="4, 2" />
        <line x1="200" y1="465" x2="400" y2="480" stroke="#a8dadc" stroke-width="4" stroke-dasharray="4, 2" />
    </g>
    <!-- Bike Frame -->
    <g stroke="#1d3557" stroke-linecap="round" stroke-linejoin="round" fill="none">
        <!-- Chainstay -->
        <line x1="200" y1="450" x2="400" y2="450" stroke-width="12" />
        <!-- Seatstay -->
        <line x1="200" y1="450" x2="300" y2="320" stroke-width="12" />
        <!-- Seat tube -->
        <line x1="300" y1="320" x2="400" y2="450" stroke-width="16" />
        <!-- Down tube -->
        <line x1="400" y1="450" x2="550" y2="320" stroke-width="16" />
        <!-- Top tube -->
        <line x1="300" y1="320" x2="550" y2="320" stroke-width="16" />
        <!-- Fork -->
        <line x1="550" y1="320" x2="600" y2="450" stroke-width="14" />
        <!-- Stem -->
        <line x1="550" y1="320" x2="530" y2="290" stroke-width="12" />
    </g>
    <!-- Top Tube Highlight -->
    <line x1="305" y1="315" x2="545" y2="315" stroke="#ffffff" stroke-width="3" stroke-linecap="round" opacity="0.3" />
    <!-- Seat -->
    <g>
        <ellipse cx="300" cy="330" rx="30" ry="10" fill="#1d3557" opacity="0.5" />
        <path d="M 270 320 Q 300 300 330 320 Q 330 330 300 330 Q 270 330 270 320" fill="#1d3557" />
    </g>
    <!-- Handlebars -->
    <g>
        <path d="M 530 290 Q 560 270 570 300" fill="none" stroke="#1d3557" stroke-width="12" stroke-linecap="round" />
        <path d="M 565 295 L 575 310" fill="none" stroke="#e63946" stroke-width="14" stroke-linecap="round" />
    </g>
    <!-- Crank &amp; Pedals -->
    <g>
        <circle cx="400" cy="450" r="20" fill="#a8dadc" stroke="#1d3557" stroke-width="6" />
        <!-- Crank Arms -->
        <line x1="400" y1="450" x2="370" y2="410" stroke="#1d3557" stroke-width="10" stroke-linecap="round" />
        <line x1="400" y1="450" x2="430" y2="490" stroke="#1d3557" stroke-width="10" stroke-linecap="round" />
        <!-- Pedal Bases -->
        <rect x="355" y="405" width="30" height="10" rx="3" fill="#e63946" transform="rotate(-15 370 410)" />
        <rect x="415" y="485" width="30" height="10" rx="3" fill="#e63946" transform="rotate(15 430 490)" />
    </g>
    <!-- ================= DUCK ================= -->
    <!-- Duck Tail -->
    <g>
        <path d="M 260 250 Q 210 230 220 270 Q 230 290 270 300" fill="url(#duckGrad)" stroke="#ffb703" stroke-width="3" stroke-linecap="round" />
        <path d="M 230 250 L 220 240 M 225 265 L 210 260" stroke="#e85d04" stroke-width="3" stroke-linecap="round" />
    </g>
    <!-- Duck Legs &amp; Feet -->
    <g>
        <!-- Back Leg -->
        <path d="M 300 330 Q 320 370 370 410" fill="none" stroke="#f4a261" stroke-width="12" stroke-linecap="round" />
        <!-- Front Leg -->
        <path d="M 350 330 Q 360 400 430 490" fill="none" stroke="#f4a261" stroke-width="12" stroke-linecap="round" />
        <!-- Back Foot (Webbed) -->
        <g transform="translate(370, 410) rotate(-45)">
            <ellipse cx="0" cy="15" rx="8" ry="15" fill="#f4a261" />
            <ellipse cx="-10" cy="10" rx="6" ry="12" fill="#f4a261" />
            <ellipse cx="10" cy="10" rx="6" ry="12" fill="#f4a261" />
        </g>
        <!-- Front Foot (Webbed) -->
        <g transform="translate(430, 490) rotate(-15)">
            <ellipse cx="0" cy="15" rx="8" ry="15" fill="#f4a261" />
            <ellipse cx="-10" cy="10" rx="6" ry="12" fill="#f4a261" />
            <ellipse cx="10" cy="10" rx="6" ry="12" fill="#f4a261" />
        </g>
    </g>
    <!-- Duck Body -->
    <g>
        <ellipse cx="320" cy="270" rx="80" ry="70" fill="url(#duckGrad)" />
        <!-- Body Highlight -->
        <ellipse cx="300" cy="245" rx="30" ry="15" fill="#ffffff" opacity="0.3" transform="rotate(-30 300 245)" />
    </g>
    <!-- Duck Neck -->
    <path d="M 350 230 Q 370 180 380 150 L 420 160 Q 390 220 380 280 Z" fill="url(#duckGrad)" />
    <!-- Duck Head -->
    <g>
        <circle cx="420" cy="140" r="50" fill="url(#headGrad)" />
        <!-- Head Highlight -->
        <ellipse cx="405" cy="120" rx="15" ry="8" fill="#ffffff" opacity="0.3" transform="rotate(-20 405 120)" />
    </g>
    <!-- Feather Tuft (sticking out from helmet) -->
    <g>
        <path d="M 375 105 Q 360 90 365 80" fill="none" stroke="#ffd166" stroke-width="4" stroke-linecap="round" />
        <path d="M 370 105 Q 350 100 345 90" fill="none" stroke="#ffd166" stroke-width="4" stroke-linecap="round" />
    </g>
    <!-- Duck Face -->
    <g>
        <!-- Eye -->
        <circle cx="440" cy="125" r="6" fill="#000000" />
        <circle cx="442" cy="123" r="2" fill="#ffffff" />
        <!-- Blush -->
        <ellipse cx="425" cy="155" rx="8" ry="4" fill="#ff4d6d" opacity="0.6" />
        <!-- Upper Beak -->
        <path d="M 465 130 C 490 125, 510 140, 510 150 C 490 150, 470 145, 460 140 Z" fill="#f4a261" />
        <!-- Lower Beak -->
        <path d="M 465 145 C 480 155, 490 160, 495 165 C 480 170, 465 160, 460 155 Z" fill="#e76f51" />
        <!-- Beak Separation Line -->
        <path d="M 460 140 C 470 145, 490 145, 510 150" fill="none" stroke="#d00000" stroke-width="2" />
        <!-- Happy Mouth Line -->
        <path d="M 465 155 C 475 160, 485 160, 490 162" fill="none" stroke="#d00000" stroke-width="2" />
    </g>
    <!-- Cycling Helmet -->
    <g>
        <!-- Helmet Dome -->
        <path d="M 368 110 A 52 52 0 0 1 472 110 Z" fill="#e63946" />
        <path d="M 365 110 L 475 110" stroke="#1d3557" stroke-width="4" stroke-linecap="round" />
        <!-- Helmet Vents -->
        <g stroke="#1d3557" stroke-width="3" stroke-linecap="round">
            <line x1="390" y1="95" x2="390" y2="110" />
            <line x1="420" y1="90" x2="420" y2="110" />
            <line x1="450" y1="95" x2="450" y2="110" />
        </g>
        <!-- Helmet Strap -->
        <path d="M 375 110 Q 380 150 410 160" fill="none" stroke="#1d3557" stroke-width="3" />
    </g>
    <!-- Scarf (Wind effect) -->
    <g>
        <!-- Scarf Wrap -->
        <path d="M 370 180 C 390 190, 400 210, 380 220 C 360 215, 340 200, 350 180 Z" fill="#e63946" />
        <!-- Scarf Tail 1 -->
        <path d="M 375 215 C 300 240, 240 180, 180 220" fill="none" stroke="#e63946" stroke-width="15" stroke-linecap="round" />
        <!-- Scarf Tail 2 -->
        <path d="M 370 210 C 300 260, 220 210, 150 250" fill="none" stroke="#e63946" stroke-width="10" stroke-linecap="round" />
        <!-- Scarf Stripes -->
        <g stroke="#ffb703" stroke-width="4" stroke-linecap="round">
            <line x1="300" y1="230" x2="290" y2="215" />
            <line x1="240" y1="210" x2="230" y2="195" />
            <line x1="190" y1="220" x2="180" y2="210" />
            <line x1="260" y1="240" x2="250" y2="225" />
            <line x1="200" y1="235" x2="190" y2="220" />
        </g>
    </g>
    <!-- Duck Wing (Holding Handlebar) -->
    <g>
        <!-- Main Wing -->
        <path d="M 360 250 Q 460 270 565 300 Q 460 320 370 300" fill="#ffb703" stroke="#e85d04" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" />
        <!-- Wing Feathers Details -->
        <g stroke="#e85d04" stroke-width="3" stroke-linecap="round">
            <path d="M 450 290 L 460 300" />
            <path d="M 480 295 L 490 305" />
            <path d="M 510 300 L 520 310" />
        </g>
        <!-- Wing "Hand" wrapping around the handlebar -->
        <path d="M 560 295 Q 570 310 565 315" fill="none" stroke="#ffb703" stroke-width="6" stroke-linecap="round" />
        <path d="M 565 300 Q 575 315 570 320" fill="none" stroke="#ffb703" stroke-width="6" stroke-linecap="round" />
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

## Conclusion

This comparison shows that 10 out of 14 active Ollama cloud models can generate valid SVG artwork from a natural language prompt. The results vary dramatically in complexity, style, and technique:

- **Simplest output**: gemma4:31b-cloud produced a compact SVG with basic shapes
- **Most detailed output**: glm-5.1:cloud generated a 15,000+ character SVG with rich scene elements
- **Most technically advanced**: deepseek-v4-pro:cloud used `<defs>`, `<use>`, transforms, and motion lines
- **Fastest response**: gemma4:31b-cloud completed in just 8 seconds

All SVGs were generated with zero post-processing -- what you see is exactly what each model produced. The variety of approaches (from minimalist geometric to richly detailed scenes) highlights how different training data and model architectures lead to distinctly different creative outputs.

## Links

- [Ollama Official Website](https://ollama.com)
- [Ollama Cloud Documentation](https://ollama.com/cloud)
- [SVG Specification (MDN)](https://developer.mozilla.org/en-US/docs/Web/SVG)
