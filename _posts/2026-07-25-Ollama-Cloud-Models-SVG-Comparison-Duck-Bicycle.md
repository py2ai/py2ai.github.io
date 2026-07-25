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
---

# Ollama Cloud Models SVG Comparison: A Duck Driving a Bicycle

What happens when you ask 9 different state-of-the-art LLMs to draw the exact same thing? In this post, we sent the prompt `Make an svg image of a duck driving a bicycle` to every cloud model available in the local Ollama registry and compared the results side by side.

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering.

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary of Results

| # | Model | SVG Size (chars) | Status |
|---|-------|-----------------|--------|
| 1 | `deepseek-v4-pro_cloud` | 8044 | Generated |
| 2 | `gemma4_31b-cloud` | 1445 | Generated |
| 3 | `glm-5.1_cloud` | 14174 | Generated |
| 4 | `glm-5.2_cloud` | 5482 | Generated |
| 5 | `kimi-k2.6_cloud` | 6538 | Generated |
| 6 | `minimax-m2.7_cloud` | 3808 | Generated |
| 7 | `nemotron-3-super_cloud` | 1950 | Generated |
| 8 | `nemotron-3-ultra_cloud` | 11633 | Generated |
| 9 | `qwen3.5_397b-cloud` | 3492 | Generated |
| 10 | `bjoernb/claude-opus-4-5:latest` | - | Retired (410) |
| 11 | `deepseek-v3.1:671b-cloud` | - | Retired (410) |
| 12 | `glm-5:cloud` | - | Retired (410) |
| 13 | `qwen3-vl:235b-cloud` | - | Retired (410) |

**9 out of 13** models successfully produced a valid SVG. The 4 retired models returned HTTP 410 Gone (they were removed from Ollama Cloud on 2026-07-15).

## 1. deepseek-v4-pro_cloud

**SVG size:** 8044 characters  
**Raw response:** 8447 characters

![deepseek-v4-pro_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/deepseek-v4-pro_cloud.svg)

### Analysis

This SVG contains approximately **73 shape elements** and uses **10 distinct colors**.

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
    <!-- Reusable Wheel Component -->
    <g id="wheel">
      <!-- Tire -->
      <circle cx="0" cy="0" r="80" stroke="#264653" stroke-width="16" fill="none"/>
      <!-- Rim -->
      <circle cx="0" cy="0" r="72" stroke="#A8DADC" stroke-width="4" fill="none"/>
      <!-- Spokes Group -->
      <g id="spokes">
        <line x1="0" y1="-72" x2="0" y2="72" stroke="#A8DADC" stroke-width="2"/>
        <line x1="-72" y1="0" x2="72" y2="0" stroke="#A8DADC" stroke-width="2"/>
        <line x1="-51" y1="-51" x2="51" y2="51" stroke="#A8DADC" stroke-width="2"/>
        <line x1="-51" y1="51" x2="51" y2="-51" stroke="#A8DADC" stroke-width="2"/>
      </g>
      <!-- Rotated Spokes -->
      <use href="#spokes" transform="rotate(30)" />
      <use href="#spokes" transform="rotate(60)" />
      <!-- Hub -->
      <circle cx="0" cy="0" r="10" fill="#264653"/>
    </g>
  </defs>

  <!-- Background -->
  <rect x="0" y="0" fill="#E0FBFC"/>
  
  <!-- Sun -->
  <circle cx="100" cy="100" r="70" fill="#FFD166" opacity="0.3"/>
  <circle cx="100" cy="100" r="50" fill="#FFD166"/>
  
  <!-- Clouds -->
  <path d="M 200 100 Q 220 80 240 100 Q 260 80 280 100 Q 300 120 280 130 L 200 130 Q 180 120 200 100" fill="#FFF"/>
  <path d="M 600 150 Q 620 130 640 150 Q 660 130 680 150 Q 700 170 680 180 L 600 180 Q 580 170 600 150" fill="#FFF"/>
  
  <!-- Road -->
  <rect x="0" y="520" width="800" height="80" fill="#A8DADC"/>
  <line x1="0" y1="560" x2="800" y2="560" stroke="#FFF" stroke-width="6" stroke-dasharray="40 20"/>
  
  <!-- Shadow under the bike -->
  <ellipse cx="400" cy="530" rx="250" ry="15" fill="#8D99AE" opacity="0.5"/>

  <!-- Motion Lines -->
  <line x1="50" y1="400" x2="100" y2="400" stroke="#264653" stroke-width="4" stroke-linecap="round"/>
  <line x1="30" y1="430" x2="120" y2="430" stroke="#264653" stroke-width="4" stroke-linecap="round"/>
  <line x1="60" y1="460" x2="110" y2="460" stroke="#264653" stroke-width="4" stroke-linecap="round"/>

  <!-- ================= BICYCLE ================= -->
  
  <!-- Bike Frame -->
  <g stroke="#E76F51" stroke-width="14" stroke-linecap="round" stroke-linejoin="round">
    <!-- Chainstay -->
    <line x1="200" y1="450" x2="400" y2="450"/>
    <!-- Seatstay -->
    <line x1="200" y1="450" x2="350" y2="300"/>
    <!-- Seat tube -->
    <line x1="400" y1="450" x2="350" y2="300"/>
    <!-- Downtube -->
    <line x1="400" y1="450" x2="550" y2="250"/>
    <!-- Top tube -->
    <line x1="350" y1="300" x2="550" y2="250"/>
  </g>
  
  <!-- Head Tube -->
  <line x1="540" y1="240" x2="560" y2="260" stroke="#E76F51" stroke-width="16" stroke-linecap="round"/>
  
  <!-- Fork -->
  <path d="M 550 250 Q 580 350 600 450" stroke="#E76F51" stroke-width="10" fill="none" stroke-linecap="round"/>

  <!-- Frame Decals -->
  <path d="M 350 300 L 400 450" stroke="#FFF" stroke-width="4" stroke-dasharray="10 10" opacity="0.4"/>
  <path d="M 400 450 L 550 250" stroke="#FFF" stroke-width="4" stroke-dasharray="10 10" opacity="0.4"/>

  <!-- Wheels -->
  <use href="#wheel" x="200" y="450" />
  <use href="#wheel" x="600" y="450" />

  <!-- Chain & Pedals -->
  <!-- Rear Cog -->
  <circle cx="200" cy="450" r="15" fill="#264653"/>
  <!-- Chainring -->
  <circle cx="400" cy="450" r="30" fill="#F4A261" stroke="#264653" stroke-width="6"/>
  <circle cx="400" cy="450" r="15" fill="#264653"/>
  <!-- Chain -->
  <line x1="200" y1="440" x2="400" y2="440" stroke="#264653" stroke-width="4" stroke-dasharray="6 4"/>
  <line x1="200" y1="460" x2="400" y2="460" stroke="#264653" stroke-width="4" stroke-dasharray="6 4"/>
  
  <!-- Crank Arms & Pedals -->
  <line x1="400" y1="450" x2="400" y2="490" stroke="#264653" stroke-width="10" stroke-linecap="round"/>
  <rect x="380" y="490" width="40" height="10" rx="3" fill="#264653"/>
  <line x1="400" y1="450" x2="380" y2="480" stroke="#264653" stroke-width="10" stroke-linecap="round"/>
  <rect x="360" y="480" width="40" height="10" rx="3" fill="#264653"/>

  <!-- Seat Post -->
  <line x1="350" y1="300" x2="350" y2="280" stroke="#264653" stroke-width="10" stroke-linecap="round"/>

  <!-- ================= DUCK ================= -->
  
  <!-- Duck Body -->
  <g id="duck-body">
    <!-- Main Body -->
    <ellipse cx="340" cy="230" rx="60" ry="50" fill="#FFD166" transform="rotate(-15, 340, 230)"/>
    <!-- White Belly -->
    <ellipse cx="345" cy="240" rx="40" ry="35" fill="#FFF" transform="rotate(-15, 345, 240)"/>
    <!-- Neck -->
    <path d="M 380 200 Q 400 140 420 130 L 400 120 Q 370 150 350 200" fill="#FFD166"/>
  </g>

  <!-- Duck Legs -->
  <g id="duck-legs">
    <!-- Back Leg -->
    <path d="M 340 270 L 360 350 L 380 480" stroke="#F4A261" stroke-width="12" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="360" cy="350" r="8" fill="#F4A261"/>
    <!-- Back Foot -->
    <path d="M 380 480 L 395 490 L 380 485 L 365 490 Z" fill="#F4A261"/>
    
    <!-- Front Leg -->
    <path d="M 360 270 L 380 350 L 400 490" stroke="#F4A261" stroke-width="12" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="380" cy="350" r="8" fill="#F4A261"/>
    <!-- Front Foot -->
    <path d="M 400 490 L 415 500 L 400 495 L 385 500 Z" fill="#F4A261"/>
  </g>

  <!-- Seat -->
  <ellipse cx="350" cy="275" rx="35" ry="12" fill="#264653"/>
  <ellipse cx="350" cy="273" rx="25" ry="6" fill="#3A5A6B"/>

  <!-- Handlebars -->
  <g id="handlebars">
    <!-- Stem -->
    <line x1="550" y1="250" x2="550" y2="220" stroke="#264653" stroke-width="10" stroke-linecap="round"/>
    <!-- Bar -->
    <path d="M 520 220 Q 550 200 580 220" stroke="#264653" stroke-width="10" fill="none" stroke-linecap="round"/>
    <!-- Grips -->
    <line x1="515" y1="220" x2="525" y2="220" stroke="#E76F51" stroke-width="14" stroke-linecap="round"/>
    <line x1="575" y1="220" x2="585" y2="220" stroke="#E76F51" stroke-width="14" stroke-linecap="round"/>
    <!-- Bell -->
    <circle cx="530" cy="210" r="10" fill="#F4A261"/>
    <circle cx="530" cy="210" r="6" fill="#E76F51"/>
    <line x1="530" y1="210" x2="530" y2="220" stroke="#264653" stroke-width="4"/>
  </g>

  <!-- Duck Wing (Holding Handlebar) -->
  <g id="duck-wing">
    <path d="M 380 210 C 420 160, 480 180, 550 220 C 500 240, 420 240, 380 230 Z" fill="#FFD166"/>
    <!-- Wingtip gripping handlebar -->
    <circle cx="550" cy="220" r="12" fill="#FFD166"/>
    <!-- Feather details -->
    <path d="M 400 200 Q 450 190 500 220" stroke="#E0A800" stroke-width="3" fill="none"/>
    <path d="M 410 215 Q 460 205 510 230" stroke="#E0A800" stroke-width="3" fill="none"/>
  </g>

  <!-- Duck Head & Face -->
  <g id="duck-head">
    <!-- Head -->
    <circle cx="420" cy="120" r="35" fill="#FFD166"/>
    <!-- Head Highlight -->
    <ellipse cx="410" cy="105" rx="15" ry="10" fill="#FFF" opacity="0.3" transform="rotate(-30, 410, 105)"/>
    
    <!-- Beak -->
    <path d="M 450 110 Q 480 110 480 125 Q 460 130 450 130" fill="#F4A261"/>
    <!-- Nostril -->
    <circle cx="465" cy="115" r="2" fill="#264653"/>
    <!-- Smile -->
    <path d="M 450 125 Q 460 130 470 125" stroke="#264653" stroke-width="2" fill="none"/>
    
    <!-- Eye -->
    <circle cx="435" cy="110" r="6" fill="#264653"/>
    <circle cx="437" cy="108" r="2" fill="#FFF"/>
    <!-- Eyebrow (Determined look) -->
    <path d="M 425 100 Q 435 95 445 100" stroke="#264653" stroke-width="3" fill="none" stroke-linecap="round"/>
    
    <!-- Baseball Cap -->
    <path d="M 395 90 Q 420 75 445 90 L 445 95 L 395 95 Z" fill="#E76F51"/>
    <path d="M 445 90 Q 465 90 465 95 Q 455 100 445 95" fill="#264653"/>
  </g>

  <!-- Bike Accessories (Front & Back) -->
  <!-- Headlight -->
  <rect x="590" y="240" width="15" height="20" rx="3" fill="#F4A261" transform="rotate(15, 590, 240)"/>
  <circle cx="597" cy="250" r="8" fill="#FFF" opacity="0.8"/>
  <!-- Taillight -->
  <rect x="190" y="300" width="10" height="15" rx="2" fill="#E76F51" transform="rotate(-15, 190, 300)"/>
  <circle cx="195" cy="307" r="5" fill="#FFD166"/>

</svg>
```

</details>

## 2. gemma4_31b-cloud

**SVG size:** 1445 characters  
**Raw response:** 2015 characters

![gemma4_31b-cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/gemma4_31b-cloud.svg)

### Analysis

This SVG contains approximately **16 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Background Circle -->
  <circle cx="100" cy="100" r="90" fill="#e0f7fa" />

  <!-- Bicycle Wheels -->
  <circle cx="60" cy="150" r="20" fill="none" stroke="#333" stroke-width="4" />
  <circle cx="60" cy="150" r="2" fill="#333" />
  <circle cx="140" cy="150" r="20" fill="none" stroke="#333" stroke-width="4" />
  <circle cx="140" cy="150" r="2" fill="#333" />

  <!-- Bicycle Frame -->
  <path d="M60 150 L100 150 L130 110 L90 110 Z" fill="none" stroke="#d32f2f" stroke-width="4" stroke-linejoin="round" />
  <line x1="100" y1="150" x2="90" y2="110" stroke="#d32f2f" stroke-width="4" />
  
  <!-- Handlebars -->
  <line x1="130" y1="110" x2="140" y2="100" stroke="#333" stroke-width="4" />
  <line x1="135" y1="100" x2="145" y2="100" stroke="#333" stroke-width="4" stroke-linecap="round" />

  <!-- Duck Body -->
  <ellipse cx="90" cy="110" rx="25" ry="20" fill="#FFEB3B" />
  
  <!-- Duck Head -->
  <circle cx="115" cy="85" r="15" fill="#FFEB3B" />
  
  <!-- Duck Beak -->
  <path d="M125 82 L135 85 L125 88 Z" fill="#FF9800" />
  
  <!-- Duck Eye -->
  <circle cx="120" cy="82" r="2" fill="#000" />

  <!-- Duck Wing -->
  <ellipse cx="85" cy="110" rx="10" ry="6" fill="#FDD835" />

  <!-- Pedals/Legs -->
  <line x1="90" y1="130" x2="90" y2="150" stroke="#FF9800" stroke-width="3" />
  <line x1="100" y1="130" x2="100" y2="150" stroke="#FF9800" stroke-width="3" />
</svg>
```

</details>

## 3. glm-5.1_cloud

**SVG size:** 14174 characters  
**Raw response:** 16458 characters

![glm-5.1_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/glm-5.1_cloud.svg)

### Analysis

This SVG contains approximately **122 shape elements** and uses **45 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

With over 10,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in adding fine details, accessories, and scene elements.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <!-- Sky gradient -->
      <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#87CEEB"/>
        <stop offset="60%" stop-color="#B0E0FF"/>
        <stop offset="100%" stop-color="#E8F5E9"/>
      </linearGradient>
      <!-- Ground gradient -->
      <linearGradient id="groundGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#66BB6A"/>
        <stop offset="100%" stop-color="#388E3C"/>
      </linearGradient>
      <!-- Duck body gradient -->
      <radialGradient id="duckBody" cx="0.4" cy="0.3" r="0.7">
        <stop offset="0%" stop-color="#FFE082"/>
        <stop offset="100%" stop-color="#FBC02D"/>
      </radialGradient>
      <!-- Duck head gradient -->
      <radialGradient id="duckHead" cx="0.4" cy="0.3" r="0.6">
        <stop offset="0%" stop-color="#FFF176"/>
        <stop offset="100%" stop-color="#FDD835"/>
      </radialGradient>
      <!-- Sun glow -->
      <radialGradient id="sunGlow" cx="0.5" cy="0.5" r="0.5">
        <stop offset="0%" stop-color="#FFF9C4"/>
        <stop offset="40%" stop-color="#FFEE58" stop-opacity="0.8"/>
        <stop offset="100%" stop-color="#FFEE58" stop-opacity="0"/>
      </radialGradient>
      <!-- Spoke pattern for wheels -->
      <clipPath id="wheelClip1">
        <circle cx="265" cy="430" r="72"/>
      </clipPath>
      <clipPath id="wheelClip2">
        <circle cx="535" cy="430" r="72"/>
      </clipPath>
      <!-- Road texture -->
      <pattern id="roadDash" x="0" y="0" patternUnits="userSpaceOnUse">
        <rect x="0" y="2" width="24" height="4" rx="2" fill="#FFEB3B" opacity="0.9"/>
      </pattern>
    </defs>

    <!-- Sky -->
    <rect x="0" y="0" width="800" height="600" fill="url(#skyGrad)"/>

    <!-- Sun -->
    <circle cx="680" cy="90" r="80" fill="url(#sunGlow)"/>
    <circle cx="680" cy="90" r="38" fill="#FFF176"/>
    <circle cx="680" cy="90" r="34" fill="#FFEE58"/>

    <!-- Clouds -->
    <g opacity="0.85">
      <ellipse cx="150" cy="100" rx="60" ry="24" fill="white"/>
      <ellipse cx="120" cy="90" rx="40" ry="20" fill="white"/>
      <ellipse cx="185" cy="92" rx="35" ry="18" fill="white"/>

      <ellipse cx="480" cy="70" rx="50" ry="20" fill="white"/>
      <ellipse cx="455" cy="62" rx="35" ry="16" fill="white"/>
      <ellipse cx="510" cy="64" rx="30" ry="15" fill="white"/>
    </g>

    <!-- Distant hills -->
    <ellipse cx="200" cy="440" rx="260" ry="60" fill="#81C784" opacity="0.5"/>
    <ellipse cx="600" cy="445" rx="300" ry="55" fill="#66BB6A" opacity="0.5"/>

    <!-- Ground -->
    <rect x="0" y="460" width="800" height="140" fill="url(#groundGrad)"/>

    <!-- Road -->
    <rect x="0" y="470" width="800" height="70" rx="2" fill="#616161"/>
    <rect x="0" y="500" width="800" height="10" fill="url(#roadDash)"/>
    <!-- Road edges -->
    <rect x="0" y="470" width="800" height="3" fill="#9E9E9E" opacity="0.5"/>
    <rect x="0" y="537" width="800" height="3" fill="#424242" opacity="0.5"/>

    <!-- Grass tufts on ground -->
    <g fill="#43A047" opacity="0.7">
      <ellipse cx="50" cy="460" rx="15" ry="5"/>
      <ellipse cx="130" cy="462" rx="12" ry="4"/>
      <ellipse cx="700" cy="458" rx="18" ry="5"/>
      <ellipse cx="750" cy="463" rx="10" ry="4"/>
      <ellipse cx="30" cy="555" rx="20" ry="6"/>
      <ellipse cx="770" cy="560" rx="15" ry="5"/>
    </g>

    <!-- Small flowers on ground -->
    <g>
      <circle cx="80" cy="560" r="4" fill="#FF8A80"/>
      <circle cx="82" cy="557" r="3" fill="#FF5252"/>
      <circle cx="730" cy="555" r="4" fill="#FF80AB"/>
      <circle cx="732" cy="552" r="3" fill="#F50057"/>
      <circle cx="160" cy="565" r="3.5" fill="#FFFF8D"/>
      <circle cx="660" cy="560" r="3.5" fill="#B388FF"/>
    </g>

    <!-- ============ BICYCLE ============ -->
    <g id="bicycle">

      <!-- Rear wheel -->
      <g>
        <circle cx="265" cy="430" r="72" fill="none" stroke="#37474F" stroke-width="7"/>
        <circle cx="265" cy="430" r="68" fill="none" stroke="#546E7A" stroke-width="2"/>
        <circle cx="265" cy="430" r="14" fill="#455A64" stroke="#37474F" stroke-width="3"/>
        <!-- Spokes -->
        <g stroke="#78909C" stroke-width="1.5" opacity="0.7">
          <line x1="265" y1="362" x2="265" y2="498"/>
          <line x1="197" y1="430" x2="333" y2="430"/>
          <line x1="217" y1="382" x2="313" y2="478"/>
          <line x1="313" y1="382" x2="217" y2="478"/>
          <line x1="265" y1="362" x2="237" y2="493" transform="rotate(22,265,430)"/>
          <line x1="265" y1="362" x2="237" y2="493" transform="rotate(68,265,430)"/>
          <line x1="265" y1="362" x2="237" y2="493" transform="rotate(112,265,430)"/>
          <line x1="265" y1="362" x2="237" y2="493" transform="rotate(158,265,430)"/>
        </g>
      </g>

      <!-- Front wheel -->
      <g>
        <circle cx="535" cy="430" r="72" fill="none" stroke="#37474F" stroke-width="7"/>
        <circle cx="535" cy="430" r="68" fill="none" stroke="#546E7A" stroke-width="2"/>
        <circle cx="535" cy="430" r="14" fill="#455A64" stroke="#37474F" stroke-width="3"/>
        <!-- Spokes -->
        <g stroke="#78909C" stroke-width="1.5" opacity="0.7">
          <line x1="535" y1="362" x2="535" y2="498"/>
          <line x1="467" y1="430" x2="603" y2="430"/>
          <line x1="487" y1="382" x2="583" y2="478"/>
          <line x1="583" y1="382" x2="487" y2="478"/>
          <line x1="535" y1="362" x2="507" y2="493" transform="rotate(22,535,430)"/>
          <line x1="535" y1="362" x2="507" y2="493" transform="rotate(68,535,430)"/>
          <line x1="535" y1="362" x2="507" y2="493" transform="rotate(112,535,430)"/>
          <line x1="535" y1="362" x2="507" y2="493" transform="rotate(158,535,430)"/>
        </g>
      </g>

      <!-- Tire details (treads) -->
      <circle cx="265" cy="430" r="72" fill="none" stroke="#263238" stroke-width="1" stroke-dasharray="6,8" opacity="0.3"/>
      <circle cx="535" cy="430" r="72" fill="none" stroke="#263238" stroke-width="1" stroke-dasharray="6,8" opacity="0.3"/>

      <!-- Frame -->
      <g stroke="#E53935" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" fill="none">
        <!-- Seat tube (seat to bottom bracket) -->
        <line x1="350" y1="310" x2="355" y2="410"/>
        <!-- Down tube (head tube to bottom bracket) -->
        <line x1="490" y1="310" x2="355" y2="410"/>
        <!-- Top tube (seat to head tube) -->
        <line x1="350" y1="310" x2="490" y2="310"/>
        <!-- Chain stay (bottom bracket to rear axle) -->
        <line x1="355" y1="410" x2="265" y2="430"/>
        <!-- Seat stay (seat to rear axle) -->
        <line x1="350" y1="310" x2="265" y2="430"/>
        <!-- Fork (head tube to front axle) -->
        <line x1="490" y1="310" x2="535" y2="430"/>
      </g>

      <!-- Chain ring -->
      <circle cx="355" cy="410" r="20" fill="none" stroke="#455A64" stroke-width="3"/>
      <circle cx="355" cy="410" r="6" fill="#455A64"/>

      <!-- Chain -->
      <polyline points="265,430 275,432 335,418 355,410 335,402 275,428 265,430" 
                fill="none" stroke="#78909C" stroke-width="2" opacity="0.6"/>

      <!-- Seat -->
      <ellipse cx="345" cy="295" rx="28" ry="10" fill="#4E342E" stroke="#3E2723" stroke-width="2"/>
      <ellipse cx="345" cy="294" rx="22" ry="7" fill="#5D4037" opacity="0.5"/>

      <!-- Seat post -->
      <line x1="350" y1="305" x2="348" y2="310" stroke="#546E7A" stroke-width="4" stroke-linecap="round"/>

      <!-- Handlebars -->
      <g stroke="#546E7A" stroke-width="5" stroke-linecap="round" fill="none">
        <!-- Stem -->
        <line x1="490" y1="310" x2="500" y2="280"/>
        <!-- Handlebar cross piece -->
        <path d="M475,280 Q500,270 525,280" stroke="#546E7A" stroke-width="5"/>
        <!-- Grips -->
        <line x1="470" y1="282" x2="478" y2="284" stroke="#795548" stroke-width="8" stroke-linecap="round"/>
        <line x1="522" y1="284" x2="530" y2="282" stroke="#795548" stroke-width="8" stroke-linecap="round"/>
      </g>

      <!-- Pedals -->
      <g>
        <!-- Pedal crank - right (up position) -->
        <line x1="355" y1="410" x2="375" y2="390" stroke="#455A64" stroke-width="4" stroke-linecap="round"/>
        <rect x="370" y="385" width="18" height="6" rx="2" fill="#607D8B" stroke="#455A64" stroke-width="1.5" transform="rotate(-45,379,388)"/>

        <!-- Pedal crank - left (down position) -->
        <line x1="355" y1="410" x2="335" y2="430" stroke="#455A64" stroke-width="4" stroke-linecap="round"/>
        <rect x="326" y="426" width="18" height="6" rx="2" fill="#607D8B" stroke="#455A64" stroke-width="1.5" transform="rotate(-45,335,429)"/>
      </g>

      <!-- ============ DUCK ============ -->
      <g id="duck">

        <!-- Duck tail -->
        <path d="M280,275 Q260,250 270,235 Q280,240 290,260 Q295,270 290,280 Z" fill="#FBC02D" stroke="#F9A825" stroke-width="1.5"/>
        <path d="M275,270 Q265,255 272,245 Q278,250 282,262 Z" fill="#FFE082" opacity="0.6"/>

        <!-- Duck body -->
        <ellipse cx="360" cy="270" rx="75" ry="45" fill="url(#duckBody)" stroke="#F9A825" stroke-width="2"/>
        <!-- Belly highlight -->
        <ellipse cx="365" cy="282" rx="50" ry="25" fill="#FFF176" opacity="0.3"/>

        <!-- Duck wing -->
        <path d="M330,255 Q310,270 325,290 Q340,280 355,260 Q345,250 330,255 Z" fill="#F9A825" stroke="#F57F17" stroke-width="1.5"/>
        <!-- Wing feather detail -->
        <path d="M332,260 Q322,272 330,283" fill="none" stroke="#E65100" stroke-width="1" opacity="0.4"/>
        <path d="M340,256 Q330,268 337,281" fill="none" stroke="#E65100" stroke-width="1" opacity="0.3"/>

        <!-- Duck neck -->
        <path d="M400,260 Q415,240 420,210 Q425,200 420,195" fill="#FDD835" stroke="#F9A825" stroke-width="2"/>
        <path d="M390,265 Q405,245 410,215 Q412,205 408,200" fill="#FFF176" stroke="none" opacity="0.4"/>

        <!-- Duck head -->
        <ellipse cx="425" cy="188" rx="30" ry="26" fill="url(#duckHead)" stroke="#F9A825" stroke-width="2"/>
        <!-- Head highlight -->
        <ellipse cx="418" cy="180" rx="14" ry="10" fill="#FFF9C4" opacity="0.4"/>

        <!-- Duck beak -->
        <path d="M450,188 L490,183 L450,198 Z" fill="#FF8F00" stroke="#E65100" stroke-width="1.5" stroke-linejoin="round"/>
        <!-- Beak detail -->
        <line x1="452" y1="192" x2="482" y2="188" stroke="#E65100" stroke-width="1" opacity="0.5"/>
        <!-- Nostril -->
        <ellipse cx="462" cy="189" rx="2.5" ry="1.5" fill="#E65100" opacity="0.5"/>

        <!-- Duck eye -->
        <ellipse cx="435" cy="182" rx="7" ry="8" fill="white" stroke="#37474F" stroke-width="1"/>
        <ellipse cx="437" cy="182" rx="4" ry="5" fill="#1a1a2e"/>
        <circle cx="439" cy="180" r="2" fill="white" opacity="0.9"/>
        <!-- Eyebrow (determined expression!) -->
        <path d="M430,172 Q436,168 443,173" fill="none" stroke="#5D4037" stroke-width="2.5" stroke-linecap="round"/>

        <!-- Duck crest / tuft on head -->
        <path d="M420,165 Q415,148 425,145 Q430,155 425,165" fill="#F9A825" stroke="#F57F17" stroke-width="1"/>
        <path d="M428,163 Q432,145 440,148 Q435,158 430,165" fill="#FFCA28" stroke="#F57F17" stroke-width="1"/>
        <path d="M415,168 Q408,155 415,150 Q420,160 418,167" fill="#FFE082" stroke="#F57F17" stroke-width="1"/>

        <!-- Duck legs going to pedals -->
        <!-- Right leg (to upper pedal) -->
        <path d="M375,300 Q385,340 375,388" fill="none" stroke="#FF8F00" stroke-width="5" stroke-linecap="round"/>
        <!-- Right foot on pedal -->
        <path d="M375,388 L368,385 L385,383 Z" fill="#FF8F00" stroke="#E65100" stroke-width="1"/>

        <!-- Left leg (to lower pedal) -->
        <path d="M355,300 Q345,360 335,428" fill="none" stroke="#FF8F00" stroke-width="5" stroke-linecap="round"/>
        <!-- Left foot on pedal -->
        <path d="M335,428 L328,425 L345,424 Z" fill="#FF8F00" stroke="#E65100" stroke-width="1"/>

        <!-- Scarf blowing in the wind -->
        <path d="M408,218 Q395,225 385,215 Q375,225 360,220" fill="#E53935" stroke="#C62828" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M360,220 Q345,228 330,222" fill="#E53935" stroke="#C62828" stroke-width="1.5" stroke-linecap="round"/>
        <path d="M330,222 Q315,230 305,225" fill="#EF5350" stroke="#C62828" stroke-width="1" stroke-linecap="round"/>

        <!-- Wing gripping handlebar -->
        <path d="M395,260 Q430,265 470,282" fill="none" stroke="#F9A825" stroke-width="6" stroke-linecap="round"/>
        <path d="M470,282 Q475,284 478,284" fill="none" stroke="#F9A825" stroke-width="5" stroke-linecap="round"/>
      </g>

    </g>

    <!-- Speed lines / motion effect -->
    <g stroke="#B0BEC5" stroke-width="2" stroke-linecap="round" opacity="0.5">
      <line x1="120" y1="300" x2="170" y2="300"/>
      <line x1="100" y1="330" x2="155" y2="330"/>
      <line x1="130" y1="360" x2="175" y2="360"/>
      <line x1="90" y1="390" x2="145" y2="390"/>
      <line x1="110" y1="420" x2="165" y2="420"/>
    </g>

    <!-- Dust puffs behind bicycle -->
    <g opacity="0.4">
      <circle cx="200" cy="490" r="12" fill="#BCAAA4"/>
      <circle cx="175" cy="495" r="8" fill="#BCAAA4"/>
      <circle cx="155" cy="488" r="6" fill="#BCAAA4"/>
      <circle cx="220" cy="498" r="10" fill="#D7CCC8"/>
      <circle cx="185" cy="502" r="7" fill="#D7CCC8"/>
    </g>

    <!-- Small birds in sky -->
    <g stroke="#546E7A" stroke-width="1.5" fill="none" opacity="0.4">
      <path d="M100,180 Q107,172 114,178 Q121,172 128,180"/>
      <path d="M580,140 Q585,134 590,139 Q595,134 600,140"/>
      <path d="M300,160 Q305,154 310,159 Q315,154 320,160"/>
    </g>

    <!-- Butterfly -->
    <g transform="translate(150, 320)" opacity="0.7">
      <ellipse cx="-6" cy="0" rx="6" ry="4" fill="#FF80AB" transform="rotate(-20)"/>
      <ellipse cx="6" cy="0" rx="6" ry="4" fill="#FF80AB" transform="rotate(20)"/>
      <line x1="0" y1="-2" x2="0" y2="4" stroke="#4E342E" stroke-width="1"/>
    </g>

  </svg>
```

</details>

## 4. glm-5.2_cloud

**SVG size:** 5482 characters  
**Raw response:** 5520 characters

![glm-5.2_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/glm-5.2_cloud.svg)

### Analysis

This SVG contains approximately **58 shape elements** and uses **21 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <!-- Background -->
  <rect width="800" height="600" fill="#87CEEB" />
  
  <!-- Sun -->
  <circle cx="680" cy="100" r="45" fill="#FFD54F" />
  <circle cx="680" cy="100" r="60" fill="#FFD54F" opacity="0.3" />

  <!-- Ground -->
  <path d="M0,450 Q400,420 800,450 L800,600 L0,600 Z" fill="#C8E6C9" />
  <path d="M0,450 Q400,420 800,450" stroke="#81C784" stroke-width="6" fill="none" stroke-linecap="round" />
  
  <!-- Motion Lines -->
  <g stroke="#FFFFFF" stroke-width="4" stroke-linecap="round" opacity="0.7">
    <line x1="40" y1="250" x2="140" y2="250" />
    <line x1="40" y1="300" x2="110" y2="300" />
    <line x1="40" y1="200" x2="90" y2="200" />
  </g>

  <!-- Back Wheel -->
  <g id="back-wheel">
    <circle cx="250" cy="450" r="70" fill="none" stroke="#212121" stroke-width="12" />
    <circle cx="250" cy="450" r="58" fill="none" stroke="#9E9E9E" stroke-width="4" />
    <g stroke="#BDBDBD" stroke-width="2">
      <line x1="250" y1="392" x2="250" y2="508" />
      <line x1="200" y1="410" x2="300" y2="490" />
      <line x1="200" y1="490" x2="300" y2="410" />
      <line x1="192" y1="450" x2="308" y2="450" />
    </g>
  </g>

  <!-- Front Wheel -->
  <g id="front-wheel">
    <circle cx="550" cy="450" r="70" fill="none" stroke="#212121" stroke-width="12" />
    <circle cx="550" cy="450" r="58" fill="none" stroke="#9E9E9E" stroke-width="4" />
    <g stroke="#BDBDBD" stroke-width="2">
      <line x1="550" y1="392" x2="550" y2="508" />
      <line x1="500" y1="410" x2="600" y2="490" />
      <line x1="500" y1="490" x2="600" y2="410" />
      <line x1="492" y1="450" x2="608" y2="450" />
    </g>
  </g>

  <!-- Bicycle Frame -->
  <g id="frame" stroke="#1565C0" stroke-width="8" stroke-linecap="round" fill="none">
    <!-- Chain stay -->
    <line x1="400" y1="450" x2="250" y2="450" />
    <!-- Seat stay -->
    <line x1="310" y1="290" x2="250" y2="450" />
    <!-- Seat tube -->
    <line x1="310" y1="290" x2="400" y2="450" />
    <!-- Down tube -->
    <line x1="540" y1="320" x2="400" y2="450" />
    <!-- Top tube -->
    <line x1="310" y1="290" x2="540" y2="320" />
    <!-- Fork -->
    <line x1="540" y1="320" x2="550" y2="450" />
  </g>

  <!-- Handlebar Stem and Grip -->
  <line x1="540" y1="320" x2="580" y2="280" stroke="#1565C0" stroke-width="8" stroke-linecap="round" />
  <path d="M 580 280 Q 600 270 610 290 Q 600 295 585 290 Z" fill="#333333" />
  
  <!-- Seat -->
  <path d="M 270 290 Q 310 280 340 295 Q 340 305 320 305 L 280 305 Q 260 305 270 290 Z" fill="#2E7D32" />

  <!-- Crankset and Pedals -->
  <circle cx="400" cy="450" r="15" fill="#666666" stroke="#333333" stroke-width="3" />
  <!-- Back Pedal Arm -->
  <line x1="400" y1="450" x2="370" y2="420" stroke="#333333" stroke-width="5" stroke-linecap="round" />
  <rect x="360" y="415" width="20" height="10" rx="3" fill="#424242" transform="rotate(-20 370 420)" />
  
  <!-- Duck Back Leg (Behind Frame) -->
  <path d="M 290 310 Q 330 370 370 425" stroke="#E68900" stroke-width="14" fill="none" stroke-linecap="round" />
  <path d="M 365 420 L 395 435 L 390 445 L 360 430 Z" fill="#FF8C00" />
  <line x1="365" y1="425" x2="390" y2="435" stroke="#E68900" stroke-width="2" />

  <!-- Duck Body -->
  <g id="duck-body">
    <ellipse cx="270" cy="240" rx="75" ry="85" fill="#FFD700" transform="rotate(10 270 240)" />
    <ellipse cx="230" cy="190" rx="30" ry="50" fill="#FFD700" transform="rotate(20 230 190)" />
    
    <!-- Tail Feathers -->
    <path d="M 210 280 Q 180 290 190 310 Q 210 300 220 280 Z" fill="#FFC107" />
  </g>

  <!-- Duck Head -->
  <circle cx="210" cy="140" r="45" fill="#FFD700" />
  
  <!-- Duck Beak -->
  <path d="M 165 130 C 130 130, 120 140, 115 155 C 130 160, 160 155, 165 145 Z" fill="#FF8C00" />
  <path d="M 115 155 C 125 170, 155 165, 165 150 L 115 155 Z" fill="#E68900" />
  
  <!-- Duck Eye -->
  <circle cx="195" cy="125" r="12" fill="white" />
  <circle cx="195" cy="125" r="6" fill="#212121" />
  <circle cx="192" cy="122" r="2" fill="white" />

  <!-- Duck Helmet -->
  <path d="M 165 140 A 45 45 0 0 1 255 140 C 255 95, 165 95, 165 140 Z" fill="#D32F2F" />
  <rect x="180" y="105" width="10" height="15" rx="2" fill="#B71C1C" />
  <rect x="205" y="100" width="10" height="15" rx="2" fill="#B71C1C" />
  <rect x="230" y="105" width="10" height="15" rx="2" fill="#B71C1C" />
  <path d="M 165 140 Q 175 160 195 155" stroke="#B71C1C" stroke-width="3" fill="none" stroke-linecap="round" />

  <!-- Duck Wing (Reaching to handlebars) -->
  <path d="M 270 210 Q 400 160 580 270 Q 590 280 570 285 Q 400 230 270 240 Z" fill="#FFCA28" />
  <!-- Wing Feather Details -->
  <path d="M 350 195 Q 450 190 560 270" stroke="#F57F17" stroke-width="2" fill="none" />
  <path d="M 320 205 Q 400 200 510 260" stroke="#F57F17" stroke-width="2" fill="none" />
  <path d="M 290 215 Q 350 215 470 250" stroke="#F57F17" stroke-width="2" fill="none" />

  <!-- Front Pedal Arm -->
  <line x1="400" y1="450" x2="390" y2="435" stroke="#333333" stroke-width="5" stroke-linecap="round" />
  <rect x="380" y="430" width="20" height="10" rx="3" fill="#424242" transform="rotate(-20 390 435)" />
  
  <!-- Duck Front Leg (Over Frame) -->
  <path d="M 270 320 Q 350 380 390 440" stroke="#FF8C00" stroke-width="14" fill="none" stroke-linecap="round" />
  <path d="M 385 435 L 420 450 L 415 455 L 380 445 Z" fill="#FF8C00" />
  <line x1="385" y1="440" x2="415" y2="450" stroke="#E68900" stroke-width="2" />

</svg>
```

</details>

## 5. kimi-k2.6_cloud

**SVG size:** 6538 characters  
**Raw response:** 23317 characters

![kimi-k2.6_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/kimi-k2.6_cloud.svg)

### Analysis

This SVG contains approximately **31 shape elements** and uses **14 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Applies SVG filters (e.g., drop shadows) for visual effects
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" id="main-svg" class="w-full h-full absolute inset-0" viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid slice">
            <defs>
                <!-- Gradients -->
                <linearGradient id="sun-gradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#fef08a;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#fbbf24;stop-opacity:1" />
                </linearGradient>
                <linearGradient id="duck-body-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#fde047;stop-opacity:1" /> <!-- yellow-300 -->
                    <stop offset="100%" style="stop-color:#facc15;stop-opacity:1" /> <!-- yellow-400 -->
                </linearGradient>
                <filter id="shadow-blur">
                    <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
                    <feOffset dx="2" dy="4" result="offsetblur"/>
                    <feComponentTransfer>
                        <feFuncA type="linear" slope="0.3"/>
                    </feComponentTransfer>
                    <feMerge> 
                        <feMergeNode/>
                        <feMergeNode in="SourceGraphic"/> 
                    </feMerge>
                </filter>
            </defs>

            <!-- Sun -->
            <circle cx="850" cy="100" r="60" fill="url(#sun-gradient)" opacity="0.9" />

            <!-- Clouds Group -->
            <g id="clouds-layer"></g>

            <!-- Road -->
            <rect x="-100" y="450" fill="#475569" /> <!-- Road Surface -->
            <line x1="-100" y1="460" x2="1200" y2="460" stroke="#334155" stroke-width="5" /> <!-- Curb -->
            
            <!-- Road Striping (Animated) -->
            <g id="road-stripes"></g>

            <!-- Bicycle & Duck Group (Centered) -->
            <g id="rider-group" transform="translate(400, 320)">
                
                <!-- Shadow -->
                <ellipse cx="100" cy="140" rx="120" ry="10" fill="black" opacity="0.2" id="bike-shadow" />

                <!-- Back Wheel -->
                <g id="wheel-back" transform="translate(0, 100)">
                    <circle r="45" fill="none" stroke="#1e293b" stroke-width="6" /> <!-- Tire -->
                    <circle r="38" fill="none" stroke="#cbd5e1" stroke-width="2" /> <!-- Rim -->
                    <circle r="3" fill="#334155" /> <!-- Axle -->
                    <!-- Spokes (Added via JS for rotation simplicity) -->
                    <g class="spokes"></g>
                </g>

                <!-- Front Wheel -->
                <g id="wheel-front" transform="translate(200, 100)">
                    <circle r="45" fill="none" stroke="#1e293b" stroke-width="6" />
                    <circle r="38" fill="none" stroke="#cbd5e1" stroke-width="2" />
                    <circle r="3" fill="#334155" />
                    <g class="spokes"></g>
                </g>

                <!-- Frame -->
                <path d="M0,100 L80,100 L140,40 L200,100" fill="none" stroke="#ef4444" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" /> <!-- Chainstay/Seatstay/Top tube -->
                <path d="M80,100 L60,20 L140,40" fill="none" stroke="#ef4444" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" /> <!-- Down tube -->
                <path d="M200,100 L190,20" fill="none" stroke="#ef4444" stroke-width="6" stroke-linecap="round" /> <!-- Fork -->
                <path d="M60,20 L30,20" fill="none" stroke="#1e293b" stroke-width="4" stroke-linecap="round" /> <!-- Handlebar stem -->
                <path d="M15,20 L45,20 L45,10" fill="none" stroke="#1e293b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" /> <!-- Handlebars -->

                <!-- Chain & Pedals -->
                <circle cx="80" cy="100" r="15" fill="none" stroke="#64748b" stroke-width="2" /> <!-- Chainring -->
                <path d="M0,100 L80,100" stroke="#64748b" stroke-width="2" /> <!-- Chain -->
                
                <!-- Pedal Group (Rotates) -->
                <g id="pedals" transform="translate(80, 100)">
                    <rect x="-25" y="-4" width="50" height="8" rx="2" fill="#334155" />
                </g>

                <!-- Seat -->
                <path d="M50,20 L40,10 L20,10" fill="none" stroke="#1e293b" stroke-width="6" stroke-linejoin="round" />

                <!-- The Duck -->
                <g id="duck-group" transform="translate(35, -20)">
                    <!-- Tail -->
                    <path d="M-20,10 L-30,0 L-25,15 Z" fill="#f59e0b" />
                    
                    <!-- Body -->
                    <ellipse cx="0" cy="0" rx="35" ry="25" fill="url(#duck-body-grad)" filter="url(#shadow-blur)" />
                    
                    <!-- Wing -->
                    <path d="M-10,5 Q5,15 20,5 Q10,20 -10,5" fill="#eab308" />

                    <!-- Neck & Head -->
                    <g id="duck-head" transform="translate(25, -15)">
                        <path d="M0,0 Q0,-20 15,-25 Q30,-20 30,0 Q20,10 0,0" fill="url(#duck-body-grad)" />
                        <!-- Beak -->
                        <path d="M28,-15 L45,-10 L28,-5" fill="#f97316" stroke="#ea580c" stroke-width="1" />
                        <!-- Eye -->
                        <circle cx="20" cy="-18" r="3" fill="white" />
                        <circle cx="21" cy="-18" r="1" fill="black" />
                        <!-- Sunglasses (if wanted, maybe toggleable later, default cool duck) -->
                        <rect x="10" y="-22" width="20" height="6" rx="2" fill="#1e293b" />
                        <line x1="30" y1="-19" x2="35" y2="-21" stroke="#1e293b" stroke-width="2" />
                    </g>

                    <!-- Legs/Pedaling -->
                    <!-- Right Leg (Behind) -->
                    <path id="leg-right" d="M0,15 L10,30 L25,30" fill="none" stroke="#f97316" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
                </g>
                
                <!-- Bell -->
                <g transform="translate(30, 15)">
                    <path d="M0,0 Q5,-10 10,0" fill="none" stroke="#fbbf24" stroke-width="3" />
                    <circle cx="5" cy="-10" r="3" fill="#fbbf24" />
                </g>
            </g>
            
            <!-- Foreground Grass (Parallax) -->
            <g id="grass-layer"></g>

        </svg>
```

</details>

## 6. minimax-m2.7_cloud

**SVG size:** 3808 characters  
**Raw response:** 3999 characters

![minimax-m2.7_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/minimax-m2.7_cloud.svg)

### Analysis

This SVG contains approximately **30 shape elements** and uses **11 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Applies SVG filters (e.g., drop shadows) for visual effects
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Definitions for gradients/shadows -->
  <defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
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

  <!-- Ground Shadow -->
  <ellipse cx="250" cy="310" rx="180" ry="20" fill="#e0e0e0" />

  <!-- BICYCLE -->
  <g transform="translate(0, 10)">
    <!-- Wheels -->
    <g stroke="#333" stroke-width="4" fill="none">
      <!-- Back Wheel -->
      <circle cx="150" cy="280" r="50" stroke-width="8" />
      <g stroke="#555" stroke-width="2">
        <line x1="150" y1="230" x2="150" y2="330" />
        <line x1="100" y1="280" x2="200" y2="280" />
        <line x1="115" y1="245" x2="185" y2="315" />
        <line x1="185" y1="245" x2="115" y2="315" />
      </g>
      
      <!-- Front Wheel -->
      <circle cx="350" cy="280" r="50" stroke-width="8" />
      <g stroke="#555" stroke-width="2">
        <line x1="350" y1="230" x2="350" y2="330" />
        <line x1="300" y1="280" x2="400" y2="280" />
        <line x1="315" y1="245" x2="385" y2="315" />
        <line x1="385" y1="245" x2="315" y2="315" />
      </g>
    </g>

    <!-- Frame -->
    <g stroke="#e74c3c" stroke-width="6" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <!-- Chain stays & Seat tube -->
      <path d="M150 280 L220 280 L190 190 L150 280" /> 
      <!-- Down tube & Top tube -->
      <path d="M220 280 L340 190 L190 190 L220 280" />
      <!-- Fork -->
      <path d="M340 190 L350 280" />
    </g>
    
    <!-- Handlebars -->
    <path d="M340 190 L340 160 L320 150" fill="none" stroke="#333" stroke-width="5" stroke-linecap="round"/>
    <!-- Handlebar Grip -->
    <line x1="320" y1="150" x2="305" y2="155" stroke="#222" stroke-width="8" stroke-linecap="round"/>

    <!-- Seat -->
    <path d="M175 185 L205 185 Q215 185 215 175 L215 165 Q215 155 205 155 L180 155 Q170 155 170 165 Z" fill="#333" />
    <line x1="190" y1="190" x2="190" y2="240" stroke="#555" stroke-width="4" /> <!-- Seat post -->
  </g>

  <!-- DUCK -->
  <g transform="translate(0, 10)" filter="url(#shadow)">
    <!-- Back Leg (Orange) -->
    <path d="M200 190 L230 250 L250 270" stroke="#FF8C00" stroke-width="8" fill="none" stroke-linecap="round" />
    
    <!-- Body Group -->
    <g>
      <!-- Body Shape -->
      <path d="M160 200 Q140 150 190 130 Q240 130 240 190 Q240 220 200 220 Z" fill="#FCE205" />
      
      <!-- Wing -->
      <path d="M180 160 Q200 160 210 180 Q200 190 180 180 Z" fill="#E6B800" />
      
      <!-- Head -->
      <circle cx="210" cy="120" r="35" fill="#FCE205" />
      
      <!-- Beak -->
      <path d="M235 115 L275 110 Q280 125 270 130 L235 125 Z" fill="#FF8C00" />
      <circle cx="220" cy="115" r="4" fill="#000" /> <!-- Eye -->
      <circle cx="222" cy="113" r="1.5" fill="#fff" /> <!-- Eye highlight -->

      <!-- Front Leg (Orange) - Pedal connection -->
      <path d="M210 200 Q220 220 230 250 L250 270" stroke="#FF8C00" stroke-width="8" fill="none" stroke-linecap="round" />
      <!-- Foot on Pedal -->
      <ellipse cx="255" cy="275" rx="10" ry="6" fill="#FF8C00" />
    </g>

    <!-- Arm Reaching for Handlebars -->
    <path d="M210 160 Q240 180 300 160" stroke="#FCE205" stroke-width="8" fill="none" stroke-linecap="round" />
  </g>
  
  <!-- Motion Lines -->
  <g stroke="#aaa" stroke-width="2" stroke-linecap="round">
    <line x1="50" y1="300" x2="100" y2="300" />
    <line x1="20" y1="320" x2="80" y2="320" />
  </g>

</svg>
```

</details>

## 7. nemotron-3-super_cloud

**SVG size:** 1950 characters  
**Raw response:** 1975 characters

![nemotron-3-super_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/nemotron-3-super_cloud.svg)

### Analysis

This SVG contains approximately **21 shape elements** and uses **8 distinct colors**.

**Notable SVG techniques used:**

- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="100%" height="100%" fill="#87CEEB"/>

  <!-- Bicycle -->
  <!-- Wheels -->
  <circle cx="100" cy="150" r="30" fill="#FFFFFF" stroke="#333" stroke-width="2"/>
  <circle cx="300" cy="150" r="30" fill="#FFFFFF" stroke="#333" stroke-width="2"/>
  <!-- Frame -->
  <!-- Rear triangle -->
  <line x1="fork") -->
  <line x1="100" y1="120" x2="130" y2="80" stroke="#333" stroke-width="3"/>
  <line x1="130" y1="80" x2="180" y2="70" stroke="#333" stroke-width="3"/>
  <!-- Down tube -->
  <line x1="180" y1="70" x2="260" y2="110" stroke="#333" stroke-width="3"/>
  <!-- Seat tube -->
  <line x1="260" y1="110" x2="260" y2="80" stroke="#333" stroke-width="3"/>
  <!-- Top tube -->
  <line x1="180" y1="70" x2="260" y2="80" stroke="#333" stroke-width="3"/>
  <!-- Seat -->
  <rect x="250" y="75" width="20" height="5" rx="2" ry="2" fill="#555"/>
  <!-- Handlebars -->
  <line x1="130" y1="80" x2="110" y2="60" stroke="#333" stroke-width="3"/>
  <line x1="130" y1="80" x2="150" y2="60" stroke="#333" stroke-width="3"/>
  <!-- Pedals -->
  <line x1="230" y1="110" x2="250" y2="110" stroke="#333" stroke-width="2"/>
  <line x1="270" y1="110" x2="290" y2="110" stroke="#333" stroke-width="2"/>

  <!-- Duck -->
  <!-- Body -->
  <ellipse cx="300" cy="110" rx="40" ry="25" fill="#F5DEB3" stroke="#8B4513" stroke-width="2"/>
  <!-- Head -->
  <circle cx="340" cy="90" r="20" fill="#F5DEB3" stroke="#8B4513" stroke-width="2"/>
  <!-- Beak -->
  <polygon points="360,90 380,95 360,100" fill="#FFA500"/>
  <!-- Eye -->
  <circle cx="335" cy="85" r="3" fill="#000"/>
  <!-- Wing (simple) -->
  <path d="M280,100 Q260,80 250,110" stroke="#8B4513" stroke-width="2" fill="none"/>
  <!-- Legs (on pedals) -->
  <line x1="300" y1="135" x2="280" y2="150" stroke="#8B4513" stroke-width="2"/>
  <line x1="300" y1="135" x2="320" y2="150" stroke="#8B4513" stroke-width="2"/>
</svg>
```

</details>

## 8. nemotron-3-ultra_cloud

**SVG size:** 11633 characters  
**Raw response:** 12821 characters

![nemotron-3-ultra_cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/nemotron-3-ultra_cloud.svg)

### Analysis

This SVG contains approximately **92 shape elements** and uses **26 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

With over 10,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in adding fine details, accessories, and scene elements.

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
      <stop offset="100%" style="stop-color:#1a5bb8;stop-opacity:1" />
    </linearGradient>
    <!-- Metal Gradient -->
    <linearGradient id="metalGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#E0E0E0;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#A0A0A0;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#808080;stop-opacity:1" />
    </linearGradient>
    <!-- Tire Gradient -->
    <radialGradient id="tireGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:#444;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#111;stop-opacity:1" />
    </radialGradient>
    <!-- Ground Shadow -->
    <radialGradient id="shadowGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" style="stop-color:rgba(0,0,0,0.2);stop-opacity:1" />
      <stop offset="100%" style="stop-color:rgba(0,0,0,0);stop-opacity:0" />
    </radialGradient>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="#E8F4FD" />

  <!-- Ground Line -->
  <line x1="0" y1="220" x2="400" y2="220" stroke="#A8D0E6" stroke-width="2" />

  <!-- Shadows -->
  <ellipse cx="130" cy="222" rx="45" ry="5" fill="url(#shadowGrad)" />
  <ellipse cx="280" cy="222" rx="45" ry="5" fill="url(#shadowGrad)" />

  <!-- ==================== BICYCLE (Back to Front Layering) ==================== -->
  <g id="bicycle">
    <!-- Rear Wheel -->
    <g id="rearWheel" transform="translate(130, 185)">
      <circle r="35" fill="url(#tireGrad)" />
      <circle r="28" fill="none" stroke="#555" stroke-width="4" />
      <!-- Spokes -->
      <g stroke="#999" stroke-width="1.5">
        <line x1="0" y1="0" x2="0" y2="-28" />
        <line x1="0" y1="0" x2="0" y2="28" />
        <line x1="0" y1="0" x2="-28" y2="0" />
        <line x1="0" y1="0" x2="28" y2="0" />
        <line x1="0" y1="0" x2="-19.8" y2="-19.8" />
        <line x1="0" y1="0" x2="19.8" y2="-19.8" />
        <line x1="0" y1="0" x2="-19.8" y2="19.8" />
        <line x1="0" y1="0" x2="19.8" y2="19.8" />
      </g>
      <!-- Hub -->
      <circle r="5" fill="url(#metalGrad)" />
      <!-- Cassette -->
      <circle r="8" fill="none" stroke="#666" stroke-width="2" />
    </g>

    <!-- Front Wheel -->
    <g id="frontWheel" transform="translate(280, 185)">
      <circle r="35" fill="url(#tireGrad)" />
      <circle r="28" fill="none" stroke="#555" stroke-width="4" />
      <g stroke="#999" stroke-width="1.5">
        <line x1="0" y1="0" x2="0" y2="-28" />
        <line x1="0" y1="0" x2="0" y2="28" />
        <line x1="0" y1="0" x2="-28" y2="0" />
        <line x1="0" y1="0" x2="28" y2="0" />
        <line x1="0" y1="0" x2="-19.8" y2="-19.8" />
        <line x1="0" y1="0" x2="19.8" y2="-19.8" />
        <line x1="0" y1="0" x2="-19.8" y2="19.8" />
        <line x1="0" y1="0" x2="19.8" y2="19.8" />
      </g>
      <circle r="5" fill="url(#metalGrad)" />
    </g>

    <!-- Frame Geometry Points (Mental Map) -->
    <!-- Rear Axle: (130, 185) -->
    <!-- Bottom Bracket: (190, 185) -->
    <!-- Seat Tube Top: (155, 115) -->
    <!-- Head Tube Top: (255, 115) -->
    <!-- Front Axle: (280, 185) -->

    <!-- Chainstays (Rear Triangle Lower) -->
    <path d="M 130 185 L 190 185" stroke="url(#bikeFrameGrad)" stroke-width="6" stroke-linecap="round" />
    <!-- Seatstays (Rear Triangle Upper) -->
    <path d="M 130 185 Q 145 150 155 115" stroke="url(#bikeFrameGrad)" stroke-width="5" stroke-linecap="round" fill="none" />
    <!-- Seat Tube -->
    <path d="M 190 185 L 155 115" stroke="url(#bikeFrameGrad)" stroke-width="6" stroke-linecap="round" />
    <!-- Down Tube -->
    <path d="M 190 185 Q 210 170 255 115" stroke="url(#bikeFrameGrad)" stroke-width="7" stroke-linecap="round" fill="none" />
    <!-- Top Tube -->
    <path d="M 155 115 Q 180 100 255 115" stroke="url(#bikeFrameGrad)" stroke-width="6" stroke-linecap="round" fill="none" />
    <!-- Head Tube -->
    <path d="M 255 115 L 265 135" stroke="url(#bikeFrameGrad)" stroke-width="8" stroke-linecap="round" />
    <!-- Fork Legs -->
    <path d="M 265 135 Q 255 160 245 185" stroke="url(#bikeFrameGrad)" stroke-width="5" stroke-linecap="round" fill="none" />
    <path d="M 275 135 Q 285 160 295 185" stroke="url(#bikeFrameGrad)" stroke-width="5" stroke-linecap="round" fill="none" />
    <!-- Fork Crown -->
    <path d="M 245 185 L 295 185" stroke="url(#bikeFrameGrad)" stroke-width="5" stroke-linecap="round" />

    <!-- Handlebars -->
    <g transform="translate(270, 115)">
      <path d="M 0 0 L -45 -10" stroke="url(#metalGrad)" stroke-width="5" stroke-linecap="round" fill="none" />
      <path d="M 0 0 L 45 -10" stroke="url(#metalGrad)" stroke-width="5" stroke-linecap="round" fill="none" />
      <!-- Grips -->
      <circle cx="-45" cy="-10" r="6" fill="#333" />
      <circle cx="45" cy="-10" r="6" fill="#333" />
      <!-- Brake Levers -->
      <path d="M -30 -15 L -40 -25" stroke="#444" stroke-width="3" stroke-linecap="round" fill="none" />
      <path d="M 30 -15 L 40 -25" stroke="#444" stroke-width="3" stroke-linecap="round" fill="none" />
    </g>

    <!-- Seat -->
    <path d="M 155 115 Q 140 100 125 105" stroke="#333" stroke-width="10" stroke-linecap="round" fill="none" />
    <ellipse cx="135" cy="105" rx="18" ry="6" fill="#222" transform="rotate(-15 135 105)" />

    <!-- Pedals / Crankset -->
    <g transform="translate(190, 185)">
      <!-- Crank Arms -->
      <line x1="0" y1="0" x2="-70" y2="-10" stroke="url(#metalGrad)" stroke-width="5" stroke-linecap="round" />
      <line x1="0" y1="0" x2="70" y2="10" stroke="url(#metalGrad)" stroke-width="5" stroke-linecap="round" />
      <!-- Chainring -->
      <circle r="30" fill="none" stroke="#555" stroke-width="4" />
      <circle r="12" fill="url(#metalGrad)" />
      <!-- Pedals -->
      <rect x="-85" y="-20" width="30" height="10" rx="2" fill="#444" transform="rotate(-10 -70 -15)" />
      <rect x="55" y="10" width="30" height="10" rx="2" fill="#444" transform="rotate(170 70 15)" />
    </g>

    <!-- Chain (Simplified) -->
    <path d="M 190 185 Q 180 185 160 185 Q 130 185 130 185" stroke="#444" stroke-width="2" fill="none" stroke-dasharray="2,2" opacity="0.6" />
    <path d="M 130 155 L 190 155" stroke="#444" stroke-width="2" fill="none" stroke-dasharray="2,2" opacity="0.6" />

    <!-- Accessories: Basket on Front -->
    <g transform="translate(285, 85)">
      <path d="M 0 0 L -5 30 L 25 30 L 20 0 Z" fill="#8B5A2B" stroke="#5D3A1A" stroke-width="2" />
      <path d="M 0 0 L 0 30" stroke="#5D3A1A" stroke-width="1.5" stroke-dasharray="3,2" />
      <path d="M 20 0 L 20 30" stroke="#5D3A1A" stroke-width="1.5" stroke-dasharray="3,2" />
      <rect x="-5" y="30" width="30" height="3" fill="#5D3A1A" rx="1.5" />
      <!-- Flower in basket -->
      <circle cx="10" cy="20" r="4" fill="#FF69B4" />
      <circle cx="10" cy="20" r="2" fill="#FFFF00" />
    </g>
  </g>

  <!-- ==================== DUCK ==================== -->
  <g id="duck" transform="translate(0, -10)">
    <!-- Tail Feathers (Behind Body) -->
    <path d="M 110 125 Q 90 110 80 130 Q 95 125 110 125" fill="url(#duckBodyGrad)" stroke="#E6A000" stroke-width="1" />
    <path d="M 110 130 Q 90 140 80 135 Q 95 130 110 130" fill="url(#duckBodyGrad)" stroke="#E6A000" stroke-width="1" />

    <!-- Body -->
    <ellipse cx="155" cy="130" rx="50" ry="35" fill="url(#duckBodyGrad)" stroke="#E6A000" stroke-width="1.5" />
    
    <!-- Wing (Animated position suggestion - resting on handlebars) -->
    <path d="M 170 125 Q 195 100 220 110 Q 210 125 170 125" fill="#FFD700" stroke="#E6A000" stroke-width="1.5" opacity="0.9" />
    <!-- Wing Feathers -->
    <path d="M 180 118 Q 200 105 210 115" fill="none" stroke="#E6A000" stroke-width="1" stroke-linecap="round" />
    <path d="M 185 122 Q 200 110 210 118" fill="none" stroke="#E6A000" stroke-width="1" stroke-linecap="round" />

    <!-- Neck -->
    <path d="M 185 110 Q 200 90 205 75 Q 210 65 205 60" stroke="url(#duckBodyGrad)" stroke-width="22" stroke-linecap="round" fill="none" />

    <!-- Head -->
    <ellipse cx="205" cy="50" rx="22" ry="20" fill="url(#duckBodyGrad)" stroke="#E6A000" stroke-width="1.5" />
    
    <!-- Head Feathers (Top) -->
    <path d="M 195 35 Q 200 20 210 35" fill="#FFD700" stroke="#E6A000" stroke-width="1" />
    <path d="M 205 32 Q 210 18 215 32" fill="#FFD700" stroke="#E6A000" stroke-width="1" />

    <!-- Beak -->
    <path d="M 227 48 Q 260 50 227 55" fill="#FF8C00" stroke="#E67300" stroke-width="1.5" />
    <!-- Beak Nostril -->
    <ellipse cx="235" cy="50" rx="2" ry="1" fill="#E67300" />
    <!-- Lower Beak -->
    <path d="M 225 53 Q 250 55 225 58" fill="#FF8C00" stroke="#E67300" stroke-width="1" />
    
    <!-- Eye -->
    <g transform="translate(200, 45)">
      <ellipse cx="0" cy="0" rx="6" ry="7" fill="white" />
      <ellipse cx="2" cy="1" rx="3" ry="3.5" fill="#111" />
      <circle cx="4" cy="-1" r="1.5" fill="white" /> <!-- Highlight -->
      <!-- Eyebrow / Expression -->
      <path d="M -8 -10 Q 0 -14 8 -10" stroke="#333" stroke-width="2" fill="none" stroke-linecap="round" />
    </g>

    <!-- Helmet -->
    <path d="M 185 30 Q 195 10 215 10 Q 235 10 245 30 Q 245 40 235 45 Q 195 45 185 30" 
          fill="#FF4444" stroke="#CC0000" stroke-width="2" opacity="0.95" />
    <!-- Helmet Vents -->
    <ellipse cx="195" cy="18" rx="5" ry="3" fill="#CC0000" opacity="0.5" />
    <ellipse cx="215" cy="16" rx="5" ry="3" fill="#CC0000" opacity="0.5" />
    <ellipse cx="230" cy="20" rx="4" ry="2.5" fill="#CC0000" opacity="0.5" />
    <!-- Helmet Strap -->
    <path d="M 185 32 Q 180 45 175 55" stroke="#333" stroke-width="3" fill="none" stroke-linecap="round" />
    <path d="M 245 32 Q 250 45 255 55" stroke="#333" stroke-width="3" fill="none" stroke-linecap="round" />
    <path d="M 175 55 L 255 55" stroke="#333" stroke-width="3" fill="none" stroke-linecap="round" />

    <!-- Legs / Feet on Pedals -->
    <!-- Left Leg (Back Pedal - Lower) -->
    <g transform="translate(120, 175)">
        <path d="M 0 0 L -10 25" stroke="#FF8C00" stroke-width="6" stroke-linecap="round" />
        <!-- Foot/Webbing -->
        <path d="M -10 25 L -20 30 L -5 30 L -10 25" fill="#FF8C00" stroke="#E67300" stroke-width="1" />
        <path d="M -10 25 L 0 30 L -5 30 Z" fill="#FF8C00" stroke="#E67300" stroke-width="1" />
    </g>
    
    <!-- Right Leg (Front Pedal - Higher) -->
    <g transform="translate(260, 175)">
        <path d="M 0 0 L 10 15" stroke="#FF8C00" stroke-width="6" stroke-linecap="round" />
        <path d="M 10 15 L 20 18 L 10 22 L 10 15" fill="#FF8C00" stroke="#E67300" stroke-width="1" />
        <path d="M 10 15 L 0 22 L 5 22 Z" fill="#FF8C00" stroke="#E67300" stroke-width="1" />
    </g>

  </g>

  <!-- Speed Lines / Motion Effect -->
  <g opacity="0.4" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round">
    <line x1="50" y1="100" x2="90" y2="100" />
    <line x1="40" y1="120" x2="85" y2="120" />
    <line x1="50" y1="140" x2="90" y2="140" />
    <line x1="60" y1="80" x2="100" y2="80" />
  </g>

</svg>
```

</details>

## 9. qwen3.5_397b-cloud

**SVG size:** 3492 characters  
**Raw response:** 3710 characters

![qwen3.5_397b-cloud duck bicycle SVG](/assets/img/diagrams/ollama-svg-compare/qwen3.5_397b-cloud.svg)

### Analysis

This SVG contains approximately **29 shape elements** and uses **14 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Background Circle -->
  <circle cx="200" cy="150" r="140" fill="#e0f7fa" />
  
  <!-- Ground -->
  <line x1="20" y1="250" x2="380" y2="250" stroke="#8d6e63" stroke-width="4" stroke-linecap="round" />

  <!-- Bicycle Group -->
  <g transform="translate(0, 10)">
    
    <!-- Back Wheel -->
    <circle cx="100" cy="220" r="40" fill="none" stroke="#333" stroke-width="6" />
    <circle cx="100" cy="220" r="36" fill="none" stroke="#ddd" stroke-width="2" />
    <!-- Spokes -->
    <path d="M100 180 L100 260 M60 220 L140 220 M72 192 L128 248 M128 192 L72 248" stroke="#ddd" stroke-width="2" />

    <!-- Front Wheel -->
    <circle cx="300" cy="220" r="40" fill="none" stroke="#333" stroke-width="6" />
    <circle cx="300" cy="220" r="36" fill="none" stroke="#ddd" stroke-width="2" />
    <!-- Spokes -->
    <path d="M300 180 L300 260 M260 220 L340 220 M272 192 L328 248 M328 192 L272 248" stroke="#ddd" stroke-width="2" />

    <!-- Bike Frame -->
    <path d="M100 220 L180 220 L280 160 L200 160 L180 220" fill="none" stroke="#d32f2f" stroke-width="8" stroke-linejoin="round" stroke-linecap="round" />
    <path d="M180 220 L140 160" fill="none" stroke="#d32f2f" stroke-width="8" stroke-linecap="round" /> <!-- Seat post -->
    
    <!-- Handlebars -->
    <path d="M280 160 L290 140 L270 140" fill="none" stroke="#555" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" />
    
    <!-- Seat -->
    <ellipse cx="140" cy="155" rx="20" ry="8" fill="#333" />

    <!-- Pedals/Crank -->
    <circle cx="180" cy="220" r="10" fill="#555" />
    <rect x="175" y="230" width="10" height="20" fill="#555" /> <!-- Pedal arm -->
    <rect x="170" y="245" width="20" height="8" fill="#333" /> <!-- Pedal foot rest -->
  </g>

  <!-- Duck Group -->
  <g transform="translate(0, 10)">
    <!-- Back Leg (hidden slightly) -->
    <path d="M150 180 L175 235" stroke="#ff9800" stroke-width="8" stroke-linecap="round" />
    <path d="M165 235 L190 235" stroke="#ff9800" stroke-width="8" stroke-linecap="round" /> <!-- Foot on pedal -->

    <!-- Body -->
    <ellipse cx="140" cy="175" rx="35" ry="25" fill="#ffeb3b" stroke="#fbc02d" stroke-width="2" />
    
    <!-- Wing -->
    <path d="M125 175 Q145 195 165 175" fill="#fdd835" stroke="#fbc02d" stroke-width="2" />

    <!-- Neck -->
    <path d="M150 160 Q160 140 165 120" stroke="#ffeb3b" stroke-width="22" fill="none" stroke-linecap="round" />

    <!-- Head -->
    <circle cx="170" cy="115" r="22" fill="#ffeb3b" stroke="#fbc02d" stroke-width="2" />

    <!-- Eye -->
    <circle cx="178" cy="110" r="3" fill="#000" />
    <circle cx="179" cy="109" r="1" fill="#fff" />

    <!-- Beak -->
    <path d="M188 115 L210 120 L188 125 Z" fill="#ff9800" stroke="#e65100" stroke-width="1" />

    <!-- Front Wing (Holding Handlebars) -->
    <path d="M160 170 Q190 160 260 150" fill="none" stroke="#ffeb3b" stroke-width="12" stroke-linecap="round" />
    <!-- Wing Tip/Hand -->
    <circle cx="260" cy="150" r="8" fill="#ffeb3b" stroke="#fbc02d" stroke-width="2" />

    <!-- Front Leg -->
    <path d="M155 185 L180 235" stroke="#ff9800" stroke-width="8" stroke-linecap="round" />
    <path d="M170 235 L195 235" stroke="#ff9800" stroke-width="8" stroke-linecap="round" /> <!-- Foot on pedal -->
  </g>
  
  <!-- Speed Lines -->
  <path d="M40 150 L10 150 M50 180 L15 180 M60 210 L25 210" stroke="#90a4ae" stroke-width="3" stroke-linecap="round" opacity="0.6" />
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

This comparison shows that 9 out of 13 active Ollama cloud models can generate valid SVG artwork from a natural language prompt. The results vary dramatically in complexity, style, and technique:

- **Simplest output**: gemma4:31b-cloud produced a compact 1,445-character SVG with basic shapes
- **Most detailed output**: glm-5.1:cloud generated a 14,000+ character SVG with rich scene elements
- **Most technically advanced**: deepseek-v4-pro:cloud used `<defs>`, `<use>`, transforms, and motion lines
- **Fastest response**: gemma4:31b-cloud completed in just 7 seconds

All SVGs were generated with zero post-processing -- what you see is exactly what each model produced. The variety of approaches (from minimalist geometric to richly detailed scenes) highlights how different training data and model architectures lead to distinctly different creative outputs.

## Links

- [Ollama Official Website](https://ollama.com)
- [Ollama Cloud Documentation](https://ollama.com/cloud)
- [SVG Specification (MDN)](https://developer.mozilla.org/en-US/docs/Web/SVG)
