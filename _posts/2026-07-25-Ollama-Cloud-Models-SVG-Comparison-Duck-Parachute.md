---
layout: post
title: "Ollama Cloud Models SVG Comparison: A Duck Jumping From a Plane"
description: "Asking 9 Ollama cloud models to draw a duck jumping from a plane with a parachute. A visual benchmark of SOTA LLM SVG generation."
date: 2026-07-25
header-img: "img/post-bg.jpg"
permalink: /Ollama-Cloud-Models-SVG-Comparison-Duck-Parachute/
featured-img: ai-coding-frameworks/ai-coding-frameworks
tags:
  - AI
  - Ollama
  - SVG
  - LLM
  - Comparison
author: "PyShine"
---

# Ollama Cloud Models SVG Comparison: A Duck Jumping From a Plane

Following up on our previous [duck-on-a-bicycle comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), we sent a new, more challenging prompt to every cloud model in the local Ollama registry. This time the prompt was: `Make an svg image where a duck is jumping from a plane and landing with a parachute`.

This prompt tests the models on a more complex scene: a dynamic action (jumping), two distinct objects (duck and plane), spatial relationships (falling from sky), and a specific mechanism (parachute). It is a harder test of scene composition than the bicycle prompt.

## How It Works

The script discovers all cloud-hosted models via the Ollama API (`/api/tags`), pulls each model, then sends the identical prompt through the OpenAI-compatible endpoint (`http://localhost:11434/v1/chat/completions`). Each model's response is parsed for an `<svg>...</svg>` block, and the extracted SVG is saved for rendering.

Cloud models are identified by the `remote_host` field in the API response -- these models are hosted on Ollama Cloud rather than running locally. This means even very large models (671B parameters) can be queried instantly without local GPU resources.

## Summary of Results

| # | Model | SVG Size (chars) | Status |
|---|-------|-----------------|--------|
| 1 | `deepseek-v4-pro_cloud` | 8308 | Generated |
| 2 | `gemma4_31b-cloud` | 2795 | Generated |
| 3 | `glm-5.1_cloud` | 15528 | Generated |
| 4 | `glm-5.2_cloud` | 11128 | Generated |
| 5 | `kimi-k2.6_cloud` | 7141 | Generated |
| 6 | `minimax-m2.7_cloud` | 3089 | Generated |
| 7 | `nemotron-3-super_cloud` | 1606 | Generated |
| 8 | `nemotron-3-ultra_cloud` | 15952 | Generated |
| 9 | `qwen3.5_397b-cloud` | 3277 | Generated |
| 10 | `deepseek-v4-flash_0731-cloud` | 6400 | Generated |
| 11 | `bjoernb/claude-opus-4-5:latest` | - | Retired (410) |
| 12 | `deepseek-v3.1:671b-cloud` | - | Retired (410) |
| 13 | `glm-5:cloud` | - | Retired (410) |
| 14 | `qwen3-vl:235b-cloud` | - | Retired (410) |

**10 out of 14** models successfully produced a valid SVG. The 4 retired models returned HTTP 410 Gone (they were removed from Ollama Cloud on 2026-07-15).

## 1. deepseek-v4-pro_cloud

**SVG size:** 8308 characters  
**Raw response:** 8682 characters

![deepseek-v4-pro_cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/deepseek-v4-pro_cloud.svg)

### Analysis

This SVG contains approximately **77 shape elements** and uses **21 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Uses `<use>` references to reuse defined elements, reducing duplication
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
    <defs>
        <!-- Sky Gradient -->
        <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#4a90e2" />
            <stop offset="100%" stop-color="#a0c8f0" />
        </linearGradient>
        
        <!-- Ground Gradient -->
        <linearGradient id="groundGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#7ec850" />
            <stop offset="100%" stop-color="#4a8c2a" />
        </linearGradient>

        <!-- Cloud Shape -->
        <g id="cloud" fill="#ffffff" opacity="0.8">
            <circle cx="0" cy="0" r="30" />
            <circle cx="40" cy="-10" r="40" />
            <circle cx="80" cy="5" r="25" />
            <circle cx="110" cy="15" r="20" />
            <circle cx="-20" cy="10" r="20" />
            <rect x="-20" y="0" width="130" height="35" rx="17.5" />
        </g>

        <!-- Parachute Canopy Clip Path -->
        <clipPath id="canopyClip">
            <path d="M -120,0 
                     Q -100,-20 -80,0 
                     Q -60,-20 -40,0 
                     Q -20,-20 0,0 
                     Q 20,-20 40,0 
                     Q 60,-20 80,0 
                     Q 100,-20 120,0 
                     Q 0,-150 -120,0 Z" />
        </clipPath>
    </defs>

    <!-- Sky Background -->
    <rect width="800" height="600" fill="url(#skyGrad)" />

    <!-- Clouds -->
    <use href="#cloud" x="50" y="50" transform="scale(0.5)" opacity="0.5" />
    <use href="#cloud" x="250" y="180" transform="scale(0.7)" opacity="0.6" />
    <use href="#cloud" x="550" y="100" transform="scale(0.8)" opacity="0.5" />
    <use href="#cloud" x="700" y="250" transform="scale(0.6)" opacity="0.7" />
    <use href="#cloud" x="100" y="300" transform="scale(0.9)" opacity="0.8" />
    <use href="#cloud" x="600" y="350" transform="scale(0.7)" opacity="0.6" />

    <!-- Ground -->
    <rect y="500" width="800" height="100" fill="url(#groundGrad)" />
    
    <!-- Ground Hills -->
    <ellipse cx="200" cy="500" rx="300" ry="40" fill="#6ab04c" />
    <ellipse cx="600" cy="500" rx="400" ry="50" fill="#5a9e3a" />

    <!-- Target on Ground -->
    <g id="target" transform="translate(500, 520)">
        <circle cx="0" cy="0" r="45" fill="none" stroke="#ffffff" stroke-width="4" />
        <circle cx="0" cy="0" r="40" fill="#e74c3c" />
        <circle cx="0" cy="0" r="30" fill="#ffffff" />
        <circle cx="0" cy="0" r="20" fill="#e74c3c" />
        <circle cx="0" cy="0" r="10" fill="#ffffff" />
        <circle cx="0" cy="0" r="5" fill="#e74c3c" />
    </g>

    <!-- Trees / Bushes -->
    <g id="trees" fill="#27ae60">
        <!-- Left Trees -->
        <circle cx="300" cy="510" r="20" />
        <circle cx="320" cy="520" r="15" />
        <circle cx="280" cy="520" r="15" />
        <!-- Right Trees -->
        <circle cx="600" cy="510" r="20" />
        <circle cx="620" cy="520" r="15" />
        <circle cx="580" cy="520" r="15" />
    </g>

    <!-- Jump Trajectory -->
    <path d="M 200,120 Q 300,200 400,350" fill="none" stroke="#ffffff" stroke-width="3" stroke-dasharray="8,8" opacity="0.6" />

    <!-- Jump Smoke -->
    <g id="jump-smoke" transform="translate(180, 120)" fill="#ffffff" opacity="0.7">
        <circle cx="0" cy="0" r="15" />
        <circle cx="15" cy="-5" r="20" />
        <circle cx="30" cy="5" r="15" />
        <circle cx="10" cy="15" r="18" />
    </g>

    <!-- Plane -->
    <g id="plane" transform="translate(150, 100) scale(0.8)">
        <!-- Body -->
        <path d="M 0,0 L 120,0 L 130,15 L 120,30 L 0,30 Z" fill="#e74c3c" />
        <!-- Body Highlight -->
        <path d="M 0,0 L 120,0 L 110,5 L 0,5 Z" fill="#f1948a" opacity="0.5" />
        <!-- Tail -->
        <path d="M 0,0 L -30,-20 L -20,0 Z" fill="#c0392b" />
        <path d="M 0,30 L -30,50 L -20,30 Z" fill="#c0392b" />
        <!-- Wings -->
        <path d="M 40,0 L 20,-40 L 80,-40 L 100,0 Z" fill="#f1c40f" />
        <path d="M 40,30 L 20,70 L 80,70 L 100,30 Z" fill="#f1c40f" />
        <!-- Wing Struts -->
        <line x1="30" y1="0" x2="30" y2="-40" stroke="#c0392b" stroke-width="2" />
        <line x1="70" y1="0" x2="70" y2="-40" stroke="#c0392b" stroke-width="2" />
        <line x1="30" y1="30" x2="30" y2="70" stroke="#c0392b" stroke-width="2" />
        <line x1="70" y1="30" x2="70" y2="70" stroke="#c0392b" stroke-width="2" />
        <!-- Propeller -->
        <ellipse cx="130" cy="15" rx="5" ry="25" fill="#34495e" />
        <!-- Propeller Blur (motion) -->
        <ellipse cx="130" cy="15" rx="15" ry="5" fill="#95a5a6" opacity="0.5" />
        <!-- Cockpit -->
        <ellipse cx="100" cy="15" rx="15" ry="10" fill="#ecf0f1" />
        <!-- Pilot Silhouette -->
        <circle cx="100" cy="15" r="5" fill="#2c3e50" />
        <!-- Plane Stripes -->
        <path d="M 20,0 L 30,0 L 30,30 L 20,30 Z" fill="#c0392b" />
        <path d="M 60,0 L 70,0 L 70,30 L 60,30 Z" fill="#c0392b" />
    </g>

    <!-- Wind / Motion Lines -->
    <g stroke="#ffffff" stroke-width="2" opacity="0.4" stroke-linecap="round">
        <line x1="330" y1="300" x2="330" y2="330" />
        <line x1="470" y1="320" x2="470" y2="360" />
        <line x1="350" y1="380" x2="350" y2="420" />
        <line x1="450" y1="400" x2="450" y2="440" />
        <line x1="380" y1="280" x2="380" y2="300" />
        <line x1="420" y1="290" x2="420" y2="310" />
    </g>

    <!-- Parachute -->
    <g id="parachute" transform="translate(400, 200)">
        <!-- Canopy -->
        <g clip-path="url(#canopyClip)">
            <!-- Red Base -->
            <rect x="-120" y="-150" width="240" height="150" fill="#e74c3c" />
            <!-- White Stripes -->
            <rect x="-72" y="-150" width="48" height="150" fill="#ffffff" />
            <rect x="24" y="-150" width="48" height="150" fill="#ffffff" />
        </g>
        
        <!-- Canopy Seams (Radiating lines) -->
        <g stroke="#c0392b" stroke-width="1.5" opacity="0.4">
            <line x1="-120" y1="0" x2="0" y2="-150" />
            <line x1="-80" y1="0" x2="0" y2="-150" />
            <line x1="-40" y1="0" x2="0" y2="-150" />
            <line x1="0" y1="0" x2="0" y2="-150" />
            <line x1="40" y1="0" x2="0" y2="-150" />
            <line x1="80" y1="0" x2="0" y2="-150" />
            <line x1="120" y1="0" x2="0" y2="-150" />
        </g>

        <!-- Parachute Strings -->
        <g stroke="#555" stroke-width="2">
            <line x1="-100" y1="0" x2="-25" y2="150" />
            <line x1="-50" y1="0" x2="-10" y2="150" />
            <line x1="0" y1="0" x2="0" y2="150" />
            <line x1="50" y1="0" x2="10" y2="150" />
            <line x1="100" y1="0" x2="25" y2="150" />
        </g>
    </g>

    <!-- Duck -->
    <g id="duck-group" transform="translate(400, 350)">
        <!-- Harness -->
        <path d="M -25,-15 L 25,-15 L 25,25 L -25,25 Z" fill="none" stroke="#555" stroke-width="3" stroke-linejoin="round" />
        
        <!-- Duck Body (Rotated slightly to look down) -->
        <g transform="rotate(15)">
            <!-- Body -->
            <path d="M -30,0 C -40,-20 -20,-35 0,-35 C 20,-35 40,-20 30,0 C 40,10 30,30 0,30 C -30,30 -40,10 -30,0 Z" fill="#f1c40f" />
            <!-- Head -->
            <circle cx="30" cy="-30" r="18" fill="#f1c40f" />
            <!-- Head Highlight -->
            <circle cx="25" cy="-35" r="5" fill="#ffffff" opacity="0.5" />
            <!-- Beak -->
            <path d="M 45,-35 L 65,-25 L 45,-15 Z" fill="#e67e22" />
            <!-- Mouth -->
            <path d="M 50,-25 L 60,-25" stroke="#000" stroke-width="2" />
            <!-- Eye -->
            <circle cx="38" cy="-32" r="3" fill="#000" />
            <!-- Wing -->
            <path d="M -10,-10 Q -40,-40 -20,-50 Q 0,-30 -10,-10" fill="#f39c12" />
            <!-- Tail -->
            <path d="M -30,0 L -50,-10 L -45,10 Z" fill="#f39c12" />
            <!-- Feet -->
            <path d="M -15,25 L -20,45 L -10,45 L -5,25" fill="#e67e22" />
            <path d="M 15,25 L 10,45 L 20,45 L 25,25" fill="#e67e22" />
            <!-- Sweat Drop (Funny detail) -->
            <path d="M 50,-40 Q 55,-50 60,-40 Q 55,-35 50,-40 Z" fill="#3498db" />
        </g>
    </g>
</svg>
```

</details>

## 2. gemma4_31b-cloud

**SVG size:** 2795 characters  
**Raw response:** 3807 characters

![gemma4_31b-cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/gemma4_31b-cloud.svg)

### Analysis

This SVG contains approximately **21 shape elements** and uses **6 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Adds `<text>` labels, showing the model tried to annotate the scene
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Background: Sky and Ground -->
  <rect fill="#87CEEB" />
  <rect x="0" y="350" width="800" height="50" fill="#7CFC00" />
  
  <!-- PANEL 1: THE JUMP -->
  <g transform="translate(50, 50)">
    <!-- Plane -->
    <path d="M0 20 L150 20 L170 0 L180 0 L170 20 L200 20 L200 40 L170 40 L160 60 L160 40 L0 40 Z" fill="#B0B0B0" stroke="#333" stroke-width="2"/>
    <rect x="120" y="15" width="30" height="15" rx="5" fill="#A0A0A0" /> <!-- Window -->
    
    <!-- Jumping Duck -->
    <g transform="translate(160, 50) rotate(45)">
      <ellipse cx="10" cy="10" rx="12" ry="10" fill="yellow" stroke="black" stroke-width="1"/> <!-- Body -->
      <circle cx="20" cy="5" r="6" fill="yellow" stroke="black" stroke-width="1"/> <!-- Head -->
      <path d="M24 5 L28 6 L24 7 Z" fill="orange" stroke="black" stroke-width="0.5"/> <!-- Beak -->
      <path d="M5 15 Q10 20 15 15" stroke="orange" stroke-width="3" fill="none"/> <!-- Feet -->
    </g>
    <text x="0" y="-20" font-family="Arial" font-weight="bold" fill="#333">1. GERONIMO!</text>
  </g>

  <!-- PANEL 2: THE DESCENT -->
  <g transform="translate(300, 50)">
    <!-- Parachute -->
    <path d="M50 50 Q100 0 150 50 Z" fill="#FF4500" stroke="#333" stroke-width="2"/>
    <line x1="50" y1="50" x2="100" y2="100" stroke="black" stroke-width="1"/>
    <line x1="150" y1="50" x2="100" y2="100" stroke="black" stroke-width="1"/>
    <line x1="100" y1="50" x2="100" y2="100" stroke="black" stroke-width="1"/>
    
    <!-- Floating Duck -->
    <g transform="translate(90, 100)">
      <ellipse cx="10" cy="10" rx="12" ry="10" fill="yellow" stroke="black" stroke-width="1"/>
      <circle cx="20" cy="5" r="6" fill="yellow" stroke="black" stroke-width="1"/>
      <path d="M24 5 L28 6 L24 7 Z" fill="orange" stroke="black" stroke-width="0.5"/>
      <path d="M5 15 Q10 20 15 15" stroke="orange" stroke-width="3" fill="none"/>
    </g>
    <text x="50" y="-20" font-family="Arial" font-weight="bold" fill="#333">2. FLOATING...</text>
  </g>

  <!-- PANEL 3: THE LANDING -->
  <g transform="translate(550, 50)">
    <!-- Parachute on ground -->
    <path d="M30 280 Q80 240 130 280 Z" fill="#FF4500" stroke="#333" stroke-width="2" opacity="0.7"/>
    
    <!-- Landed Duck -->
    <g transform="translate(80, 280)">
      <ellipse cx="10" cy="10" rx="12" ry="10" fill="yellow" stroke="black" stroke-width="1"/>
      <circle cx="20" cy="5" r="6" fill="yellow" stroke="black" stroke-width="1"/>
      <path d="M24 5 L28 6 L24 7 Z" fill="orange" stroke="black" stroke-width="0.5"/>
      <path d="M5 15 L10 20 L15 15" stroke="orange" stroke-width="3" fill="none"/>
    </g>
    <text x="40" y="-20" font-family="Arial" font-weight="bold" fill="#333">3. SAFE LANDING!</text>
  </g>
</svg>
```

</details>

## 3. glm-5.1_cloud

**SVG size:** 15528 characters  
**Raw response:** 23137 characters

![glm-5.1_cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/glm-5.1_cloud.svg)

### Analysis

This SVG contains approximately **119 shape elements** and uses **60 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Applies SVG filters (e.g., drop shadows) for visual effects
- Uses transforms (translate, rotate, scale) for precise positioning
- Adds `<text>` labels, showing the model tried to annotate the scene
- Contains XML comments, showing the model's internal documentation of its work

With over 10,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in adding fine details, the plane fuselage, parachute cords, and sky scene elements.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 1200" id="duckScene">
  <defs>
    <!-- Sky gradient -->
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0d1b2a"/>
      <stop offset="25%" stop-color="#1b3a5c"/>
      <stop offset="55%" stop-color="#4a90d9"/>
      <stop offset="75%" stop-color="#7ec8e3"/>
      <stop offset="90%" stop-color="#b8e4f0"/>
      <stop offset="100%" stop-color="#d4f0e8"/>
    </linearGradient>

    <!-- Ground gradient -->
    <linearGradient id="groundGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#5a9e4b"/>
      <stop offset="30%" stop-color="#4a8c3f"/>
      <stop offset="100%" stop-color="#3a6e30"/>
    </linearGradient>

    <!-- Parachute gradient -->
    <radialGradient id="chuteGrad1" cx="0.5" cy="0.8" r="0.7">
      <stop offset="0%" stop-color="#ff6b35"/>
      <stop offset="60%" stop-color="#e84e0f"/>
      <stop offset="100%" stop-color="#c43d00"/>
    </radialGradient>
    <radialGradient id="chuteGrad2" cx="0.5" cy="0.8" r="0.7">
      <stop offset="0%" stop-color="#ffd166"/>
      <stop offset="60%" stop-color="#f0b429"/>
      <stop offset="100%" stop-color="#d69e00"/>
    </radialGradient>
    <radialGradient id="chuteGrad3" cx="0.5" cy="0.8" r="0.7">
      <stop offset="0%" stop-color="#06d6a0"/>
      <stop offset="60%" stop-color="#05b88a"/>
      <stop offset="100%" stop-color="#049a72"/>
    </radialGradient>

    <!-- Duck body gradient -->
    <radialGradient id="duckBody" cx="0.4" cy="0.3" r="0.8">
      <stop offset="0%" stop-color="#ffe066"/>
      <stop offset="70%" stop-color="#f0c000"/>
      <stop offset="100%" stop-color="#d4a000"/>
    </radialGradient>

    <!-- Duck head gradient -->
    <radialGradient id="duckHead" cx="0.4" cy="0.3" r="0.7">
      <stop offset="0%" stop-color="#fff176"/>
      <stop offset="70%" stop-color="#f0c800"/>
      <stop offset="100%" stop-color="#d4a800"/>
    </radialGradient>

    <!-- Sun glow -->
    <radialGradient id="sunGlow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0%" stop-color="#fff9c4" stop-opacity="1"/>
      <stop offset="40%" stop-color="#ffee58" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#ffee58" stop-opacity="0"/>
    </radialGradient>

    <!-- Water gradient -->
    <linearGradient id="waterGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4a90d9" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#2979b8" stop-opacity="0.8"/>
    </linearGradient>

    <!-- Cloud filter for softness -->
    <filter id="cloudSoft">
      <feGaussianBlur in="SourceGraphic" stdDeviation="2"/>
    </filter>

    <filter id="glowFilter">
      <feGaussianBlur in="SourceGraphic" stdDeviation="6"/>
    </filter>

    <!-- Tree gradient -->
    <linearGradient id="treeGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#2d7a1e"/>
      <stop offset="100%" stop-color="#1a5c10"/>
    </linearGradient>
  </defs>

  <!-- SKY -->
  <rect fill="url(#skyGrad)"/>

  <!-- Stars in upper sky -->
  <g class="stars">
    <circle class="star" cx="80" cy="40" r="1.5" fill="#ffffff"/>
    <circle class="star" cx="200" cy="80" r="1" fill="#ffffff"/>
    <circle class="star" cx="750" cy="50" r="1.5" fill="#ffffff"/>
    <circle class="star" cx="600" cy="30" r="1" fill="#ffffff"/>
    <circle class="star" cx="400" cy="60" r="1.2" fill="#ffffff"/>
  </g>

  <!-- Sun -->
  <circle cx="720" cy="280" r="120" fill="url(#sunGlow)" opacity="0.5"/>
  <circle cx="720" cy="280" r="45" fill="#ffee58" opacity="0.9"/>
  <circle cx="720" cy="280" r="35" fill="#fff9c4"/>

  <!-- Clouds -->
  <g class="cloud1" opacity="0.85">
    <ellipse cx="150" cy="200" rx="70" ry="30" fill="white" opacity="0.9"/>
    <ellipse cx="190" cy="190" rx="50" ry="25" fill="white" opacity="0.95"/>
    <ellipse cx="120" cy="195" rx="40" ry="20" fill="white" opacity="0.85"/>
  </g>

  <g class="cloud2" opacity="0.7">
    <ellipse cx="650" cy="150" rx="80" ry="28" fill="white" opacity="0.85"/>
    <ellipse cx="700" cy="140" rx="55" ry="22" fill="white" opacity="0.9"/>
    <ellipse cx="610" cy="145" rx="45" ry="20" fill="white" opacity="0.8"/>
  </g>

  <g class="cloud3" opacity="0.6">
    <ellipse cx="400" cy="350" rx="65" ry="22" fill="white" opacity="0.8"/>
    <ellipse cx="440" cy="342" rx="45" ry="18" fill="white" opacity="0.85"/>
    <ellipse cx="365" cy="346" rx="35" ry="15" fill="white" opacity="0.75"/>
  </g>

  <!-- Small clouds scattered -->
  <g class="cloud2" opacity="0.4">
    <ellipse cx="100" cy="500" rx="50" ry="15" fill="white"/>
    <ellipse cx="130" cy="495" rx="35" ry="12" fill="white"/>
  </g>
  <g class="cloud1" opacity="0.35">
    <ellipse cx="780" cy="450" rx="55" ry="16" fill="white"/>
    <ellipse cx="810" cy="445" rx="38" ry="13" fill="white"/>
  </g>

  <!-- PLANE -->
  <g class="plane-group">
    <g transform="translate(350, 80) scale(0.8)">
      <!-- Fuselage -->
      <ellipse cx="0" cy="0" rx="60" ry="14" fill="#c0c0c0"/>
      <ellipse cx="0" cy="-2" rx="58" ry="10" fill="#d8d8d8"/>
      <!-- Cockpit window -->
      <ellipse cx="40" cy="-4" rx="12" ry="7" fill="#4a90d9" opacity="0.8"/>
      <ellipse cx="40" cy="-5" rx="10" ry="5" fill="#7ec8e3" opacity="0.6"/>
      <!-- Wings -->
      <path d="M-10,0 L-25,-45 L5,-45 L10,0Z" fill="#b0b0b0"/>
      <path d="M-10,0 L-25,45 L5,45 L10,0Z" fill="#a0a0a0"/>
      <!-- Tail -->
      <path d="M-50,-2 L-65,-30 L-45,-28 L-45,0Z" fill="#b0b0b0"/>
      <path d="M-50,2 L-60,15 L-45,14Z" fill="#a8a8a8"/>
      <!-- Engine pods -->
      <ellipse cx="-8" cy="-35" rx="8" ry="5" fill="#999"/>
      <ellipse cx="-8" cy="35" rx="8" ry="5" fill="#999"/>
      <!-- Red stripe -->
      <path d="M-55,0 Q0,-4 55,0 Q0,4 -55,0Z" fill="#ff6b35" opacity="0.8"/>
      <!-- Door open (duck jumped!) -->
      <rect x="-5" y="5" width="10" height="12" rx="2" fill="#e0e0e0" stroke="#999" stroke-width="0.5"/>
      <path d="M-5,5 L-8,-5 L2,-5 L5,5Z" fill="#888" opacity="0.6"/>
    </g>
  </g>

  <!-- WIND / MOTION LINES near duck -->
  <g opacity="0.4">
    <line class="wind-line" x1="340" y1="420" x2="310" y2="418" stroke="white" stroke-width="1.5" stroke-dasharray="4 8"/>
    <line class="wind-line" x1="350" y1="440" x2="315" y2="438" stroke="white" stroke-width="1.5" stroke-dasharray="4 8"/>
    <line class="wind-line" x1="330" y1="460" x2="295" y2="458" stroke="white" stroke-width="1.5" stroke-dasharray="4 8"/>
    <line class="wind-line" x1="345" y1="480" x2="300" y2="477" stroke="white" stroke-width="1.5" stroke-dasharray="4 8"/>
  </g>

  <!-- PARACHUTE + DUCK GROUP -->
  <g class="parachute-group">

    <!-- Parachute canopy -->
    <g class="parachute-canopy">
      <!-- Canopy panels -->
      <path d="M310,130 Q340,50 450,50 Q560,50 590,130" fill="url(#chuteGrad1)" stroke="#c43d00" stroke-width="1.5"/>
      <path d="M350,130 Q370,60 450,55 Q530,60 550,130" fill="url(#chuteGrad2)" stroke="#d69e00" stroke-width="1"/>
      <path d="M380,130 Q400,70 450,65 Q500,70 520,130" fill="url(#chuteGrad3)" stroke="#049a72" stroke-width="1"/>

      <!-- Canopy outline -->
      <path d="M310,130 Q340,45 450,45 Q560,45 590,130" fill="none" stroke="#aa3500" stroke-width="2.5"/>

      <!-- Panel seam lines -->
      <path d="M350,130 Q370,55 450,50" fill="none" stroke="#b84400" stroke-width="1" opacity="0.6"/>
      <path d="M410,130 Q430,52 450,50" fill="none" stroke="#b84400" stroke-width="1" opacity="0.6"/>
      <path d="M490,130 Q470,52 450,50" fill="none" stroke="#b84400" stroke-width="1" opacity="0.6"/>
      <path d="M550,130 Q530,55 450,50" fill="none" stroke="#b84400" stroke-width="1" opacity="0.6"/>

      <!-- Canopy top detail -->
      <ellipse cx="450" cy="52" rx="15" ry="6" fill="#ff8c42" opacity="0.5"/>

      <!-- Vent hole -->
      <ellipse cx="450" cy="68" rx="8" ry="4" fill="#1a3a5c" opacity="0.3"/>
    </g>

    <!-- Suspension lines -->
    <g stroke="#8B7355" stroke-width="1.2" fill="none" opacity="0.9">
      <path d="M320,130 Q350,250 420,340"/>
      <path d="M360,130 Q380,250 425,340"/>
      <path d="M410,130 Q420,250 430,340"/>
      <path d="M450,130 Q450,250 440,340"/>
      <path d="M490,130 Q470,250 445,340"/>
      <path d="M540,130 Q510,250 450,340"/>
      <path d="M580,130 Q530,250 455,340"/>
    </g>

    <!-- DUCK -->
    <g class="duck-body-group" transform="translate(435, 340)">

      <!-- Goggles strap -->
      <path d="M-30,-52 Q0,-68 30,-52" fill="none" stroke="#444" stroke-width="3"/>
      <path d="M-32,-52 Q0,-70 32,-52" fill="none" stroke="#666" stroke-width="1.5"/>

      <!-- Goggles -->
      <ellipse cx="-14" cy="-52" rx="13" ry="10" fill="#333" stroke="#222" stroke-width="2"/>
      <ellipse cx="14" cy="-52" rx="13" ry="10" fill="#333" stroke="#222" stroke-width="2"/>
      <!-- Goggle lenses -->
      <ellipse cx="-14" cy="-52" rx="10" ry="7" fill="#5bc0eb" opacity="0.7"/>
      <ellipse cx="14" cy="-52" rx="10" ry="7" fill="#5bc0eb" opacity="0.7"/>
      <!-- Lens reflections -->
      <ellipse cx="-17" cy="-55" rx="4" ry="3" fill="white" opacity="0.5"/>
      <ellipse cx="11" cy="-55" rx="4" ry="3" fill="white" opacity="0.5"/>
      <!-- Bridge -->
      <path d="M-1,-52 Q0,-55 1,-52" fill="#222" stroke="#222" stroke-width="2"/>

      <!-- Head -->
      <ellipse cx="0" cy="-38" rx="26" ry="22" fill="url(#duckHead)"/>

      <!-- Bill -->
      <path d="M20,-38 L50,-35 L48,-30 L20,-30Z" fill="#ff8c00" stroke="#e67600" stroke-width="1"/>
      <path d="M20,-34 L45,-33 L44,-31 L20,-31Z" fill="#e67a00" opacity="0.5"/>
      <!-- Nostril -->
      <ellipse cx="35" cy="-35" rx="2" ry="1" fill="#cc6a00" opacity="0.6"/>

      <!-- Eyes (behind goggles but visible) -->
      <circle cx="-8" cy="-42" r="4" fill="#1a1a1a"/>
      <circle cx="8" cy="-42" r="4" fill="#1a1a1a"/>
      <circle cx="-7" cy="-43" r="1.5" fill="white"/>
      <circle cx="9" cy="-43" r="1.5" fill="white"/>

      <!-- Head tuft -->
      <path d="M-5,-58 Q-2,-68 5,-60 Q8,-70 12,-60 Q15,-65 10,-57" fill="#f0c800" stroke="#d4a000" stroke-width="1"/>

      <!-- Neck -->
      <path d="M-15,-20 Q-18,0 -20,10 L20,10 Q18,0 15,-20Z" fill="url(#duckHead)"/>

      <!-- Body -->
      <ellipse cx="0" cy="30" rx="35" ry="40" fill="url(#duckBody)"/>
      <!-- Belly highlight -->
      <ellipse cx="-5" cy="25" rx="22" ry="25" fill="#ffe066" opacity="0.4"/>

      <!-- Wings (spread out - skydiving pose!) -->
      <!-- Left wing -->
      <path d="M-35,10 Q-80,-10 -95,15 Q-85,25 -60,20 Q-45,18 -35,25Z" fill="#f0c000" stroke="#d4a000" stroke-width="1.5"/>
      <path d="M-50,14 Q-70,8 -85,18" fill="none" stroke="#c49000" stroke-width="1" opacity="0.5"/>
      <!-- Right wing -->
      <path d="M35,10 Q80,-10 95,15 Q85,25 60,20 Q45,18 35,25Z" fill="#f0c000" stroke="#d4a000" stroke-width="1.5"/>
      <path d="M50,14 Q70,8 85,18" fill="none" stroke="#c49000" stroke-width="1" opacity="0.5"/>

      <!-- Tail feathers -->
      <path d="M-10,65 Q-15,85 -5,80 Q0,78 5,80 Q15,85 10,65" fill="#e0a800" stroke="#c49000" stroke-width="1"/>
      <path d="M-5,65 Q-8,78 0,75 Q8,78 5,65" fill="#d4a000" stroke="#b88800" stroke-width="0.5"/>

      <!-- Feet (dangling) -->
      <g transform="translate(-12, 75)">
        <path d="M0,0 L-10,25 L-15,22 M0,0 L-5,28 L-10,25 M0,0 L5,25 L0,22" fill="#ff8c00" stroke="#e67a00" stroke-width="1.5" stroke-linecap="round"/>
      </g>
      <g transform="translate(12, 75)">
        <path d="M0,0 L-10,25 L-15,22 M0,0 L-5,28 L-10,25 M0,0 L5,25 L0,22" fill="#ff8c00" stroke="#e67a00" stroke-width="1.5" stroke-linecap="round"/>
      </g>

      <!-- Scarf (fluttering in the wind!) -->
      <path d="M-18,0 Q-25,-5 -30,2 Q-35,8 -42,5" fill="none" stroke="#ff6b35" stroke-width="4" stroke-linecap="round"/>
      <path d="M-18,2 Q-28,0 -35,8 Q-40,15 -50,10" fill="none" stroke="#ff6b35" stroke-width="3" stroke-linecap="round" opacity="0.7"/>

      <!-- Harness straps -->
      <path d="M-15,10 L-20,-5" stroke="#8B7355" stroke-width="2.5" fill="none"/>
      <path d="M15,10 L20,-5" stroke="#8B7355" stroke-width="2.5" fill="none"/>
      <path d="M-8,65 L0,70" stroke="#8B7355" stroke-width="2" fill="none"/>
      <path d="M8,65 L0,70" stroke="#8B7355" stroke-width="2" fill="none"/>

    </g>
  </g>

  <!-- GROUND / LANDSCAPE -->
  <g>
    <!-- Far hills -->
    <path d="M0,950 Q100,880 200,920 Q350,860 450,900 Q600,850 700,910 Q800,870 900,920 L900,1050 L0,1050Z" fill="#3a7d32" opacity="0.5"/>

    <!-- Water / lake -->
    <ellipse cx="200" cy="1020" rx="120" ry="30" fill="url(#waterGrad)"/>
    <ellipse cx="200" cy="1015" rx="80" ry="15" fill="#7ec8e3" opacity="0.3"/>

    <!-- Near ground -->
    <path d="M0,980 Q150,950 300,970 Q500,940 650,965 Q800,945 900,970 L900,1200 L0,1200Z" fill="url(#groundGrad)"/>

    <!-- Ground texture lines -->
    <path d="M0,1000 Q200,990 400,1005 Q600,995 900,1000" fill="none" stroke="#3a6e30" stroke-width="1" opacity="0.5"/>
    <path d="M0,1040 Q250,1030 500,1045 Q750,1035 900,1040" fill="none" stroke="#3a6e30" stroke-width="1" opacity="0.4"/>

    <!-- Trees -->
    <g transform="translate(100, 940)">
      <rect x="-4" y="0" width="8" height="25" rx="3" fill="#5a3a1a"/>
      <ellipse cx="0" cy="-10" rx="18" ry="22" fill="url(#treeGrad)"/>
      <ellipse cx="-8" cy="-5" rx="12" ry="15" fill="#2d8a1e" opacity="0.7"/>
    </g>

    <g transform="translate(300, 950)">
      <rect x="-3" y="0" width="6" height="20" rx="2" fill="#5a3a1a"/>
      <ellipse cx="0" cy="-8" rx="14" ry="18" fill="url(#treeGrad)"/>
    </g>

    <g transform="translate(500, 935)">
      <rect x="-5" y="0" width="10" height="30" rx="4" fill="#5a3a1a"/>
      <ellipse cx="0" cy="-14" rx="22" ry="28" fill="url(#treeGrad)"/>
      <ellipse cx="10" cy="-8" rx="15" ry="18" fill="#2d8a1e" opacity="0.7"/>
    </g>

    <g transform="translate(680, 945)">
      <rect x="-3" y="0" width="6" height="18" rx="2" fill="#5a3a1a"/>
      <ellipse cx="0" cy="-6" rx="12" ry="15" fill="url(#treeGrad)"/>
    </g>

    <g transform="translate(800, 940)">
      <rect x="-4" y="0" width="8" height="22" rx="3" fill="#5a3a1a"/>
      <ellipse cx="0" cy="-9" rx="16" ry="20" fill="url(#treeGrad)"/>
    </g>

    <!-- Small bushes -->
    <ellipse cx="50" cy="970" rx="15" ry="10" fill="#3a8a2a" opacity="0.7"/>
    <ellipse cx="420" cy="965" rx="12" ry="8" fill="#3a8a2a" opacity="0.6"/>
    <ellipse cx="600" cy="960" rx="10" ry="7" fill="#3a8a2a" opacity="0.5"/>
    <ellipse cx="850" cy="968" rx="14" ry="9" fill="#3a8a2a" opacity="0.6"/>

    <!-- Landing target circle -->
    <ellipse cx="450" cy="990" rx="50" ry="12" fill="none" stroke="#ff6b35" stroke-width="2" stroke-dasharray="6 4" opacity="0.7"/>
    <ellipse cx="450" cy="990" rx="30" ry="7" fill="none" stroke="#ff6b35" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.5"/>
    <ellipse cx="450" cy="990" rx="10" ry="3" fill="#ff6b35" opacity="0.4"/>
  </g>

  <!-- Birds in distance -->
  <g opacity="0.4">
    <path d="M120,300 Q125,295 130,300 Q135,295 140,300" fill="none" stroke="#333" stroke-width="1.5"/>
    <path d="M160,280 Q164,276 168,280 Q172,276 176,280" fill="none" stroke="#333" stroke-width="1.2"/>
    <path d="M780,220 Q784,216 788,220 Q792,216 796,220" fill="none" stroke="#333" stroke-width="1.2"/>
  </g>

  <!-- "JUMP!" text trailing from plane -->
  <text x="300" y="120" font-family="'Arial Black', sans-serif" font-size="14" fill="white" opacity="0.5" font-weight="bold" letter-spacing="3">JUMP!</text>

</svg>
```

</details>

## 4. glm-5.2_cloud

**SVG size:** 11128 characters  
**Raw response:** 13180 characters

![glm-5.2_cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/glm-5.2_cloud.svg)

### Analysis

This SVG contains approximately **83 shape elements** and uses **26 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Uses `<use>` references to reuse defined elements, reducing duplication
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Includes animation elements for motion effects
- Contains XML comments, showing the model's internal documentation of its work

With over 10,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in adding fine details, the plane fuselage, parachute cords, and sky scene elements.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" text-rendering="geometricPrecision" font-family="ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif">
  <title>Duck parachuting from a plane</title>
  <desc>A cartoon duck jumps from a plane and descends under a red-and-white parachute toward a landing target.</desc>

  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#7DC8F0"/>
      <stop offset="0.6" stop-color="#BFE3F5"/>
      <stop offset="1" stop-color="#EAF4FB"/>
    </linearGradient>
    <radialGradient id="sun" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0" stop-color="#FFF4D6" stop-opacity="0.9"/>
      <stop offset="1" stop-color="#FFF4D6" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="grass" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#8BC34A"/>
      <stop offset="1" stop-color="#5D9B3A"/>
    </linearGradient>
    <linearGradient id="planeBody" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#E63946"/>
      <stop offset="1" stop-color="#B71C2C"/>
    </linearGradient>
    <radialGradient id="cloud" cx="0.5" cy="0.4" r="0.6">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.95"/>
      <stop offset="1" stop-color="#DCE9F2" stop-opacity="0.7"/>
    </radialGradient>
    <radialGradient id="domeShade" cx="0.5" cy="1" r="0.8">
      <stop offset="0" stop-color="#000" stop-opacity="0.18"/>
      <stop offset="1" stop-color="#000" stop-opacity="0"/>
    </radialGradient>
    <symbol id="cloud-shape" viewBox="0 0 120 50">
      <ellipse cx="30" cy="32" rx="22" ry="16" fill="url(#cloud)"/>
      <ellipse cx="55" cy="25" rx="28" ry="20" fill="url(#cloud)"/>
      <ellipse cx="85" cy="32" rx="24" ry="17" fill="url(#cloud)"/>
      <ellipse cx="100" cy="35" rx="16" ry="12" fill="url(#cloud)"/>
    </symbol>
  </defs>

  <style>
    @media (prefers-reduced-motion: reduce){
      animate, animateTransform, animateMotion { display: none; }
      * { animation: none !important; }
    }
  </style>

  <!-- Sky -->
  <rect fill="url(#sky)"/>

  <!-- Sun -->
  <g>
    <circle cx="700" cy="100" r="160" fill="url(#sun)"/>
    <circle cx="700" cy="100" r="38" fill="#FFE8A3">
      <animate attributeName="r" values="38;42;38" dur="6s" repeatCount="indefinite"/>
    </circle>
  </g>

  <!-- Drifting clouds -->
  <g opacity="0.85">
    <use href="#cloud-shape" x="20" y="180" width="160" height="60">
      <animateTransform attributeName="transform" type="translate"
        values="0 0; 60 0; 0 0" keyTimes="0;0.5;1" dur="20s" repeatCount="indefinite"/>
    </use>
    <use href="#cloud-shape" x="520" y="80" width="200" height="70">
      <animateTransform attributeName="transform" type="translate"
        values="0 0; -50 0; 0 0" keyTimes="0;0.5;1" dur="25s" repeatCount="indefinite"/>
    </use>
    <use href="#cloud-shape" x="280" y="240" width="140" height="55">
      <animateTransform attributeName="transform" type="translate"
        values="0 0; 30 0; 0 0" keyTimes="0;0.5;1" dur="18s" repeatCount="indefinite"/>
    </use>
  </g>

  <!-- Distant birds -->
  <g stroke="#3a3a3a" stroke-width="1.5" fill="none" stroke-linecap="round">
    <g>
      <path d="M 340 150 q 5 -5 10 0 q 5 -5 10 0"/>
      <path d="M 365 168 q 4 -4 8 0 q 4 -4 8 0"/>
      <animateTransform attributeName="transform" type="translate"
        values="0 0; -40 -8; 0 0" dur="15s" repeatCount="indefinite"/>
    </g>
  </g>

  <!-- Ground -->
  <rect x="0" y="520" width="800" height="80" fill="url(#grass)"/>

  <!-- Pond -->
  <ellipse cx="180" cy="560" rx="70" ry="14" fill="#8FCAF0" opacity="0.85"/>
  <ellipse cx="180" cy="558" rx="62" ry="9" fill="#B6DCF2" opacity="0.7"/>

  <!-- Landing target -->
  <g transform="translate(540, 555)">
    <ellipse cx="0" cy="0" rx="72" ry="14" fill="#F5F5F5"/>
    <ellipse cx="0" cy="0" rx="55" ry="11" fill="#E63946"/>
    <ellipse cx="0" cy="0" rx="38" ry="7.5" fill="#F5F5F5"/>
    <ellipse cx="0" cy="0" rx="18" ry="4" fill="#E63946"/>
  </g>

  <!-- Grass tufts -->
  <g stroke="#3F7A1E" stroke-width="2" stroke-linecap="round" fill="none">
    <path d="M 60 530 q 2 -8 4 -10"/><path d="M 70 532 q 2 -8 4 -10"/>
    <path d="M 280 540 q 2 -8 4 -10"/><path d="M 290 540 q -2 -8 -4 -10"/>
    <path d="M 380 540 q 2 -8 4 -10"/><path d="M 388 540 q -2 -8 -4 -10"/>
    <path d="M 680 530 q 2 -8 4 -10"/><path d="M 690 532 q -2 -8 -4 -10"/>
  </g>

  <!-- Plane (top-left) -->
  <g id="plane">
    <ellipse cx="110" cy="110" rx="55" ry="14" fill="url(#planeBody)"/>
    <ellipse cx="160" cy="110" rx="10" ry="9" fill="#F5F5F5"/>
    <path d="M 55 110 L 45 88 L 67 100 Z" fill="#E63946"/>
    <path d="M 55 110 L 50 95 L 62 102 Z" fill="#B71C2C" opacity="0.7"/>
    <path d="M 95 100 L 75 70 L 110 78 L 120 100 Z" fill="#E63946"/>
    <path d="M 95 100 L 75 70 L 90 78 L 100 100 Z" fill="#B71C2C" opacity="0.55"/>
    <path d="M 100 118 L 85 132 L 122 122 Z" fill="#B71C2C"/>
    <rect x="105" y="103" width="14" height="8" rx="2" fill="#BFE3F5"/>
    <rect x="122" y="103" width="14" height="8" rx="2" fill="#BFE3F5"/>
    <rect x="125" y="105" width="3" height="4" fill="#fff" opacity="0.6"/>
    <!-- open jump door -->
    <rect x="88" y="106" width="10" height="12" rx="1" fill="#2B2B2B"/>
    <path d="M 88 106 q 5 -3 10 0" fill="#1a1a1a"/>

    <!-- propeller -->
    <g transform="translate(172, 110)">
      <circle r="3.5" fill="#2B2B2B"/>
      <g>
        <line x1="-20" y1="0" x2="20" y2="0" stroke="#2B2B2B" stroke-width="3.5" stroke-linecap="round"/>
        <line x1="0" y1="-20" x2="0" y2="20" stroke="#2B2B2B" stroke-width="2" stroke-linecap="round" opacity="0.35"/>
        <animateTransform attributeName="transform" type="rotate"
          from="0" to="360" dur="0.4s" repeatCount="indefinite"/>
      </g>
    </g>

    <!-- plane hovers gently -->
    <animateTransform attributeName="transform" type="translate"
      values="0 0; 8 -3; 0 0; -4 2; 0 0" keyTimes="0; 0.3; 0.5; 0.8; 1"
      dur="10s" repeatCount="indefinite"/>
  </g>

  <!-- Just-jumped mini duck with motion trail -->
  <g>
    <!-- motion dashes trailing above the duck -->
    <g stroke="#ffffff" stroke-width="2.5" stroke-linecap="round" fill="none">
      <path d="M 92 124 L 92 132">
        <animate attributeName="opacity" values="0.15;0.85;0.15" dur="1.1s" repeatCount="indefinite"/>
      </path>
      <path d="M 96 128 L 96 138">
        <animate attributeName="opacity" values="0.15;0.85;0.15" dur="1.1s" begin="0.2s" repeatCount="indefinite"/>
      </path>
      <path d="M 100 132 L 100 142">
        <animate attributeName="opacity" values="0.15;0.85;0.15" dur="1.1s" begin="0.4s" repeatCount="indefinite"/>
      </path>
    </g>
    <!-- tiny duck, free-falling -->
    <g transform="translate(98, 158)">
      <g>
        <ellipse cx="0" cy="0" rx="9" ry="7" fill="#ffffff"/>
        <circle cx="-7" cy="-5" r="5" fill="#ffffff"/>
        <path d="M -12 -5 L -17 -3 L -12 -1 Z" fill="#F4A024"/>
        <circle cx="-8" cy="-6" r="1.1" fill="#222"/>
        <!-- flailing legs -->
        <g stroke="#F4A024" stroke-width="1.6" stroke-linecap="round" fill="none">
          <path d="M -2 6 L -3 11"/>
          <path d="M 3 6 L 5 11"/>
        </g>
        <animateTransform attributeName="transform" type="translate"
          values="0 0; 1 8; 0 0" dur="1.3s" repeatCount="indefinite"/>
      </g>
    </g>
  </g>

  <!-- Main parachute + duck assembly -->
  <g id="chute-duck">
    <!-- Parachute suspension lines -->
    <g stroke="#3a3a3a" stroke-width="1.1" stroke-linecap="round" opacity="0.75" fill="none">
      <line x1="380" y1="280" x2="520" y2="388"/>
      <line x1="436" y1="280" x2="520" y2="388"/>
      <line x1="492" y1="280" x2="520" y2="388"/>
      <line x1="548" y1="280" x2="520" y2="388"/>
      <line x1="604" y1="280" x2="520" y2="388"/>
      <line x1="660" y1="280" x2="520" y2="388"/>
    </g>

    <!-- Parachute dome (8 gore segments) -->
    <g>
      <path d="M 520,280 L 380,280 A 140 140 0 0 1 390.6,226.4 Z" fill="#E63946"/>
      <path d="M 520,280 L 390.6,226.4 A 140 140 0 0 1 421,181 Z" fill="#F5F5F5"/>
      <path d="M 520,280 L 421,181 A 140 140 0 0 1 466.4,150.6 Z" fill="#E63946"/>
      <path d="M 520,280 L 466.4,150.6 A 140 140 0 0 1 520,140 Z" fill="#F5F5F5"/>
      <path d="M 520,280 L 520,140 A 140 140 0 0 1 573.6,150.6 Z" fill="#E63946"/>
      <path d="M 520,280 L 573.6,150.6 A 140 140 0 0 1 619,181 Z" fill="#F5F5F5"/>
      <path d="M 520,280 L 619,181 A 140 140 0 0 1 649.4,226.4 Z" fill="#E63946"/>
      <path d="M 520,280 L 649.4,226.4 A 140 140 0 0 1 660,280 Z" fill="#F5F5F5"/>

      <!-- shading along the rim -->
      <path d="M 380,280 A 140 140 0 0 1 660,280 L 660,288 A 148 148 0 0 0 380,288 Z" fill="url(#domeShade)"/>
      <!-- top highlight -->
      <path d="M 466.4,150.6 A 140 140 0 0 1 573.6,150.6" stroke="#ffffff" stroke-width="3" fill="none" opacity="0.55" stroke-linecap="round"/>
    </g>

    <!-- Duck under canopy -->
    <g>
      <!-- harness -->
      <rect x="510" y="384" width="20" height="10" rx="2" fill="#7A4A28"/>
      <line x1="515" y1="394" x2="513" y2="404" stroke="#7A4A28" stroke-width="2.5" stroke-linecap="round"/>
      <line x1="525" y1="394" x2="527" y2="404" stroke="#7A4A28" stroke-width="2.5" stroke-linecap="round"/>

      <!-- body -->
      <ellipse cx="520" cy="412" rx="42" ry="32" fill="#ffffff"/>
      <ellipse cx="520" cy="422" rx="38" ry="20" fill="#EAEDEF" opacity="0.55"/>
      <!-- wing -->
      <path d="M 512 402 Q 488 410 496 426 Q 512 432 522 420 Z" fill="#EAEDEF"/>
      <path d="M 510 405 Q 498 410 502 422" stroke="#C5CAD0" stroke-width="1" fill="none"/>

      <!-- head -->
      <circle cx="498" cy="372" r="22" fill="#ffffff"/>
      <ellipse cx="488" cy="364" rx="9" ry="6" fill="#ffffff" opacity="0.7"/>
      <!-- eye -->
      <circle cx="492" cy="369" r="4.2" fill="#222"/>
      <circle cx="493.6" cy="367.4" r="1.4" fill="#fff"/>
      <!-- beak -->
      <path d="M 476 372 Q 462 374 461 380 Q 468 385 478 380 Z" fill="#F4A024"/>
      <path d="M 462 380 L 478 380" stroke="#C97D12" stroke-width="0.8"/>
      <path d="M 476 372 Q 468 373 466 377" stroke="#C97D12" stroke-width="0.7" fill="none"/>
      <!-- small cheek blush -->
      <circle cx="486" cy="378" r="2.5" fill="#F4A024" opacity="0.25"/>

      <!-- dangling feet -->
      <g stroke="#F4A024" stroke-width="2.6" stroke-linecap="round" fill="none">
        <path d="M 508 440 L 506 452"/>
        <path d="M 500 452 L 512 452"/>
        <path d="M 532 440 L 534 452"/>
        <path d="M 526 452 L 538 452"/>
      </g>

      <!-- gentle bob -->
      <animateTransform attributeName="transform" type="translate"
        values="0 0; 0 -4; 0 0" dur="2.6s" repeatCount="indefinite"/>
    </g>

    <!-- whole rig sways side to side -->
    <animateTransform attributeName="transform" type="rotate"
      values="-2.5 520 388; 2.5 520 388; -2.5 520 388"
      keyTimes="0;0.5;1" dur="6s" repeatCount="indefinite"
      calcMode="spline" keySplines="0.4 0 0.6 1; 0.4 0 0.6 1"/>
  </g>

</svg>
```

</details>

## 5. kimi-k2.6_cloud

**SVG size:** 7141 characters  
**Raw response:** 13972 characters

![kimi-k2.6_cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/kimi-k2.6_cloud.svg)

### Analysis

This SVG contains approximately **23 shape elements** and uses **19 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Applies SVG filters (e.g., drop shadows) for visual effects
- Uses transforms (translate, rotate, scale) for precise positioning
- Includes animation elements for motion effects
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately detailed SVG (5,000-10,000 characters), balancing visual richness with code clarity. The model included enough detail to make the scene recognizable while keeping the markup manageable.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" id="mainSvg" preserveAspectRatio="xMidYMid slice">
            <!-- Definitions for Gradients and reusable shapes -->
            <defs>
                <linearGradient id="groundGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:#76c949;stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#4a8c28;stop-opacity:1" />
                </linearGradient>
                
                <filter id="shadow" x="-50%" y="-50%">
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
            <!-- Sun -->
            <circle cx="700" cy="80" r="40" fill="#FFD700" opacity="0.8">
                <animate attributeName="r" values="40;42;40" dur="4s" repeatCount="indefinite" />
            </circle>

            <!-- Clouds -->
            <g opacity="0.8">
                <path d="M50,100 Q70,80 90,100 T130,100 T170,100" fill="none" stroke="white" stroke-width="20" stroke-linecap="round" />
                <animateTransform attributeName="transform" type="translate" from="-200 0" to="900 0" dur="60s" repeatCount="indefinite" />
            </g>
            <g opacity="0.6">
                <path d="M600,150 Q620,130 640,150 T680,150" fill="none" stroke="white" stroke-width="15" stroke-linecap="round" />
                <animateTransform attributeName="transform" type="translate" from="900 0" to="-200 0" dur="45s" repeatCount="indefinite" />
            </g>

            <!-- The Ground -->
            <g transform="translate(0, 500)">
                <rect x="0" y="0" width="800" height="100" fill="url(#groundGrad)" />
                <!-- Trees -->
                <g transform="translate(100, -40)">
                    <rect x="-5" y="0" width="10" height="40" fill="#8B4513" />
                    <circle cx="0" cy="-10" r="25" fill="#228B22" />
                </g>
                <g transform="translate(600, -30)">
                    <rect x="-5" y="0" width="10" height="30" fill="#8B4513" />
                    <path d="M-20,0 L0,-40 L20,0 Z" fill="#228B22" />
                </g>
            </g>

            <!-- THE PLANE -->
            <g id="planeGroup">
                <!-- Plane Body -->
                <path d="M-60,0 Q-60,-20 0,-20 L60,-10 Q80,0 60,10 L0,20 Q-60,20 -60,0 Z" fill="#eee" stroke="#ccc" stroke-width="1" />
                <!-- Cockpit -->
                <path d="M10,-18 L30,-18 L25,-5 L10,-5 Z" fill="#87CEEB" opacity="0.8" />
                <!-- Wing -->
                <path d="M0,5 L20,5 L10,25 L-10,25 Z" fill="#ccc" />
                <!-- Tail -->
                <path d="M-50,-5 L-70,-15 L-60,-5 Z" fill="#ff6b6b" />
                
                <!-- Initial Fly In Animation -->
                <animateMotion id="planeFlyIn" path="M -100,100 L 100,100" dur="2s" fill="freeze" />
                
                <!-- Hover Animation (Loops after fly in) -->
                <animateTransform id="planeHover" attributeName="transform" type="translate" values="0,0; 0,5; 0,0" dur="2s" repeatCount="indefinite" begin="planeFlyIn.end" />
            </g>

            <!-- THE JUMP SEQUENCE (Duck + Parachute) -->
            <!-- Group Wrapper for the whole falling logic -->
            <g id="skydiverGroup" opacity="0">
                
                <!-- Parachute -->
                <g id="parachute" transform="scale(0)">
                    <!-- Canopy -->
                    <path d="M-40,0 Q-20,-50 0,-50 Q20,-50 40,0 L0,0 Z" fill="#ff4757" stroke="#2f3542" stroke-width="1" />
                    <path d="M-40,0 Q-20,-50 0,-50" fill="#ff6b81" />
                    <!-- Strings -->
                    <line x1="-35" y1="0" x2="0" y2="40" stroke="#333" stroke-width="1" />
                    <line x1="35" y1="0" x2="0" y2="40" stroke="#333" stroke-width="1" />
                    <line x1="0" y1="-50" x2="0" y2="40" stroke="#333" stroke-width="1" />
                    
                    <!-- Parachute Opens (Scale Up) -->
                    <animateTransform id="chuteOpen" attributeName="transform" type="scale" from="0" to="1" begin="jumpAnim.begin + 2s" dur="0.5s" fill="freeze" />
                    <!-- Parachute Drift -->
                    <animateTransform attributeName="transform" type="translate" values="0,0; 10,0; 0,0" dur="3s" repeatCount="indefinite" begin="chuteOpen.end" additive="sum" />
                </g>

                <!-- The Duck Character -->
                <g id="duck" transform="translate(0, 40)">
                    <!-- Body -->
                    <ellipse cx="0" cy="0" rx="15" ry="10" fill="#ffdd59" stroke="#ffa502" stroke-width="1" />
                    <!-- Wing -->
                    <path d="M-5,-5 Q5,-5 5,5" fill="none" stroke="#ffa502" stroke-width="2" />
                    <!-- Head -->
                    <circle cx="-8" cy="-8" r="8" fill="#ffdd59" />
                    <!-- Beak -->
                    <path d="M-16,-8 L-22,-5 L-16,-2 Z" fill="#ff6348" />
                    <!-- Eye -->
                    <circle cx="-10" cy="-10" r="1.5" fill="#000" />
                    <!-- Goggles (Skydiving gear!) -->
                    <rect x="-14" y="-12" width="8" height="4" rx="2" fill="#a4b0be" stroke="#57606f" stroke-width="0.5" />
                </g>

                <!-- Overall Movement of Skydiver -->
                <!-- Path: Start at plane -> Freefall -> Parachute Float -> Land -->
                <animateMotion id="jumpAnim" 
                    path="M 100,110 C 100,200 150,300 350,490" 
                    dur="6s" 
                    fill="freeze"
                    begin="indefinite" 
                />

                <!-- Rotation: Horizontal fall, then vertical float, then upright land -->
                <animateTransform 
                    id="fallRotation"
                    attributeName="transform" 
                    type="rotate" 
                    values="0 0 0; 90 0 0; 0 0 0" 
                    keyTimes="0; 0.3; 1" 
                    dur="6s" 
                    fill="freeze"
                    begin="jumpAnim.begin" 
                    additive="sum"
                />
                
                <!-- Fade In for the skydiver group -->
                <animate attributeName="opacity" from="0" to="1" dur="0.1s" fill="freeze" begin="jumpAnim.begin" />
            </g>

            <!-- Plane Fly Away Animation (Triggered after landing) -->
            <animateMotion id="planeFlyOut" xlink:href="#planeGroup" path="M 100,100 L 900,50" dur="3s" fill="freeze" begin="indefinite" />

        </svg>
```

</details>

## 6. minimax-m2.7_cloud

**SVG size:** 3089 characters  
**Raw response:** 3295 characters

![minimax-m2.7_cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/minimax-m2.7_cloud.svg)

### Analysis

This SVG contains approximately **34 shape elements** and uses **12 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Includes animation elements for motion effects
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
  <!-- Sky Background -->
  <rect width="500" height="500" fill="#87CEEB" />
  
  <!-- Clouds -->
  <g fill="#FFFFFF" opacity="0.8">
    <circle cx="100" cy="80" r="30" />
    <circle cx="140" cy="80" r="40" />
    <circle cx="180" cy="90" r="30" />
    
    <circle cx="350" cy="150" r="25" />
    <circle cx="390" cy="140" r="35" />
    <circle cx="420" cy="160" r="25" />

    <circle cx="50" cy="300" r="20" />
    <circle cx="80" cy="310" r="25" />
  </g>

  <!-- The Plane (Top Right) -->
  <g transform="translate(380, 60) rotate(-15)">
    <!-- Tail -->
    <path d="M-10,-15 L10,-15 L0,-40 Z" fill="#555" />
    <!-- Wing -->
    <path d="M10,0 L60,30 L60,40 L10,10 Z" fill="#777" />
    <!-- Fuselage -->
    <path d="M-20,0 Q80,-20 120,0 Q80,20 -20,0 Z" fill="#CCC" />
    <circle cx="80" cy="-5" r="5" fill="#333" /> <!-- Window -->
    <circle cx="60" cy="-5" r="5" fill="#333" /> <!-- Window -->
    <!-- Propeller hint -->
    <ellipse cx="122" cy="0" rx="2" ry="15" fill="#444" opacity="0.5" />
  </g>

  <!-- Parachute Ropes (Behind duck, in front of sky) -->
  <line x1="200" y1="50" x2="190" y2="150" stroke="#555" stroke-width="1" />
  <line x1="250" y1="50" x2="260" y2="150" stroke="#555" stroke-width="1" />
  <line x1="300" y1="50" x2="310" y2="150" stroke="#555" stroke-width="1" />

  <!-- The Duck -->
  <g transform="translate(250, 160)">
    <!-- Scarf blowing in wind -->
    <path d="M-10,-15 Q-30,-10 -60,-30 L-65,-25 Q-30,0 -5,-5 Z" fill="#FF4444" />
    
    <!-- Legs -->
    <path d="M-5,30 L-5,50" stroke="#FFA500" stroke-width="4" />
    <path d="M5,30 L5,50" stroke="#FFA500" stroke-width="4" />
    
    <!-- Body -->
    <ellipse cx="0" cy="10" rx="25" ry="30" fill="#FFD700" />
    
    <!-- Wings holding strings -->
    <path d="M-20,10 Q-40,0 -15,-20" stroke="#FFD700" stroke-width="8" fill="none" />
    <path d="M20,10 Q40,0 15,-20" stroke="#FFD700" stroke-width="8" fill="none" />

    <!-- Head -->
    <circle cx="0" cy="-25" r="20" fill="#FFD700" />
    
    <!-- Beak -->
    <path d="M0,-22 L12,-18 L0,-14 Z" fill="#FFA500" />
    
    <!-- Eyes (Wide open for skydiving) -->
    <circle cx="-7" cy="-28" r="5" fill="white" />
    <circle cx="7" cy="-28" r="5" fill="white" />
    <circle cx="-7" cy="-28" r="2" fill="black" />
    <circle cx="7" cy="-28" r="2" fill="black" />

    <!-- Helmet -->
    <path d="M-22,-25 A 22,22 0 0,1 22,-25 L 22,-20 L -22,-20 Z" fill="#FFFFFF" />
  </g>

  <!-- Parachute Canopy (Dome) -->
  <path d="M150,50 A 100,60 0 0,1 350,50 L 330,55 L 170,55 Z" fill="#FF4500" stroke="#D32F2F" stroke-width="2" />
  
  <!-- Parachute Stripes/Details -->
  <path d="M200,50 A 50,60 0 0,1 300,50" fill="none" stroke="white" stroke-width="2" opacity="0.5" />
  <path d="M250,50 L 250,50" fill="none" stroke="white" stroke-width="2" opacity="0.5" />

  <!-- Animation: Gentle float -->
  <animateTransform attributeName="transform" type="translate" 
    values="250 160; 250 155; 250 160" 
    dur="3s" repeatCount="indefinite" additive="sum" />
</svg>
```

</details>

## 7. nemotron-3-super_cloud

**SVG size:** 1606 characters  
**Raw response:** 1879 characters

![nemotron-3-super_cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/nemotron-3-super_cloud.svg)

### Analysis

This SVG contains approximately **16 shape elements** and uses **8 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 500 400"
     xmlns="http://www.w3.org/2000/svg">
  <!-- Sky -->
  <rect width="500" height="400" fill="#87CEEB"/>
  <!-- Ground -->
  <rect y="350" width="500" height="50" fill="#228B22"/>

  <!-- Plane -->
  <rect x="50" y="80" width="150" height="20" fill="#708090"/> <!-- fuselage -->
  <polygon points="50,100 30,130 50,160" fill="#708090"/>      <!-- left wing -->
  <polygon points="200,100 220,130 200,160" fill="#708090"/>   <!-- right wing -->
  <polygon points="200,90 210,80 220,90 210,100" fill="#708090"/> <!-- tail -->

  <!-- Duck -->
  <ellipse cx="250" cy="250" rx="40" ry="25" fill="#FFFFCC"/> <!-- body -->
  <circle cx="290" cy="230" r="15" fill="#FFFFCC"/>           <!-- head -->
  <polygon points="305,225 320,230 305,235" fill="#FFA500"/>   <!-- beak -->
  <circle cx="295" cy="225" r="3" fill="#000"/>                <!-- eye -->
  <ellipse cx="230" cy="260" rx="20" ry="10" fill="#FFE4B4"
           transform="rotate(-30 230 260)"/>                    <!-- wing -->

  <!-- Parachute -->
  <ellipse cx="250" cy="150" rx="50" ry="30"
           fill="white" stroke="#000" stroke-width="2"/>       <!-- canopy -->
  <!-- Suspension lines -->
  <line x1="200" y1="180" x2="230" y2="250"
        stroke="#000" stroke-width="2"/>
  <line x1="300" y1="180" x2="270" y2="250"
        stroke="#000" stroke-width="2"/>
  <line x1="250" y1="150" x2="285" y2="230"
        stroke="#000" stroke-width="2"/>

  <!-- Dashed trajectory from plane to duck -->
  <line x1="200" y1="100" x2="250" y2="250"
        stroke="#555" stroke-width="1.5" stroke-dasharray="4,2"/>
</svg>
```

</details>

## 8. nemotron-3-ultra_cloud

**SVG size:** 15952 characters  
**Raw response:** 17298 characters

![nemotron-3-ultra_cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/nemotron-3-ultra_cloud.svg)

### Analysis

This SVG contains approximately **82 shape elements** and uses **34 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured SVG)
- Includes gradient fills for richer visual depth
- Applies SVG filters (e.g., drop shadows) for visual effects
- Uses transforms (translate, rotate, scale) for precise positioning
- Includes animation elements for motion effects
- Adds `<text>` labels, showing the model tried to annotate the scene
- Contains XML comments, showing the model's internal documentation of its work

With over 10,000 characters of SVG markup, this is one of the most detailed outputs in the comparison. The model invested significant effort in adding fine details, the plane fuselage, parachute cords, and sky scene elements.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <!-- Definitions: Gradients & Filters -->
  <defs>
    <!-- Sky Gradient -->
    <linearGradient id="skyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#0d47a1"/>
      <stop offset="40%" style="stop-color:#2196f3"/>
      <stop offset="100%" style="stop-color:#bbdefb"/>
    </linearGradient>
    
    <!-- Ground Gradient -->
    <linearGradient id="groundGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#4caf50"/>
      <stop offset="100%" style="stop-color:#2e7d32"/>
    </linearGradient>

    <!-- Plane Body Gradient -->
    <linearGradient id="planeBodyGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#e0e0e0"/>
      <stop offset="50%" style="stop-color:#ffffff"/>
      <stop offset="100%" style="stop-color:#bdbdbd"/>
    </linearGradient>

    <!-- Parachute Gradient -->
    <linearGradient id="chuteGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#ff5252"/>
      <stop offset="50%" style="stop-color:#ff1744"/>
      <stop offset="100%" style="stop-color:#b71c1c"/>
    </linearGradient>
    
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

    <!-- Cloud Filter (Softness) -->
    <filter id="cloudBlur">
      <feGaussianBlur stdDeviation="2"/>
    </filter>
  </defs>

  <!-- Background Sky -->
  <rect width="800" height="600" fill="url(#skyGradient)"/>

  <!-- Sun -->
  <circle cx="720" cy="80" r="50" fill="#ffeb3b" opacity="0.9" filter="url(#dropShadow)">
     <animate attributeName="opacity" values="0.9;1;0.9" dur="4s" repeatCount="indefinite"/>
  </circle>

  <!-- Clouds Layer (Background) -->
  <g filter="url(#cloudBlur)" fill="white" opacity="0.8">
    <g transform="translate(100, 80)">
      <ellipse cx="0" cy="0" rx="40" ry="20"/>
      <ellipse cx="35" cy="-10" rx="30" ry="18"/>
      <ellipse cx="-30" cy="-5" rx="25" ry="15"/>
    </g>
    <g transform="translate(550, 120)">
      <ellipse cx="0" cy="0" rx="50" ry="25"/>
      <ellipse cx="45" cy="-15" rx="35" ry="20"/>
      <ellipse cx="-40" cy="-8" rx="30" ry="18"/>
    </g>
    <g transform="translate(300, 50)">
      <ellipse cx="0" cy="0" rx="35" ry="18"/>
      <ellipse cx="30" cy="-8" rx="25" ry="15"/>
    </g>
    <!-- Animated Cloud -->
    <g transform="translate(50, 200)">
      <animateTransform attributeName="transform" type="translate" values="50,200; 200,200; 50,200" dur="30s" repeatCount="indefinite"/>
      <ellipse cx="0" cy="0" rx="45" ry="22" fill="white" opacity="0.7"/>
      <ellipse cx="40" cy="-12" rx="30" ry="18" fill="white" opacity="0.7"/>
    </g>
  </g>

  <!-- Ground -->
  <path d="M0,480 Q200,460 400,480 Q600,500 800,480 L800,600 L0,600 Z" fill="url(#groundGradient)"/>
  <!-- Ground Details -->
  <g fill="#388e3c" opacity="0.5">
     <ellipse cx="100" cy="500" rx="30" ry="10"/>
     <ellipse cx="350" cy="510" rx="40" ry="12"/>
     <ellipse cx="650" cy="495" rx="25" ry="8"/>
     <ellipse cx="750" cy="520" rx="20" ry="6"/>
  </g>

  <!-- ================================================== -->
  -- 1. THE PLANE (Top Left - Exit Point)              --
  <!-- ================================================== -->
  <g transform="translate(100, 100)" filter="url(#dropShadow)">
    <!-- Propeller Blur Animation -->
    <g transform="translate(110, 25)">
      <animateTransform attributeName="transform" type="rotate" values="0 0 0; 360 0 0" dur="0.1s" repeatCount="indefinite"/>
      <ellipse cx="0" cy="0" rx="55" ry="4" fill="#333" opacity="0.4"/>
      <ellipse cx="0" cy="0" rx="4" ry="55" fill="#333" opacity="0.4"/>
    </g>
    <!-- Propeller Hub -->
    <circle cx="110" cy="25" r="6" fill="#424242"/>

    <!-- Fuselage -->
    <path d="M0,25 L100,15 L100,35 Z" fill="url(#planeBodyGrad)" stroke="#9e9e9e" stroke-width="1"/>
    <rect x="100" y="15" width="80" height="20" fill="url(#planeBodyGrad)" stroke="#9e9e9e" stroke-width="1" rx="2"/>
    <!-- Cockpit -->
    <path d="M160,15 L190,20 L190,30 L160,35 Z" fill="#4fc3f7" stroke="#0288d1" stroke-width="1"/>
    <rect x="165" y="19" width="20" height="12" fill="#01579b" opacity="0.6"/> <!-- Windshield -->
    
    <!-- Tail -->
    <path d="M180,15 L210,5 L215,5 L185,15 Z" fill="#e0e0e0" stroke="#9e9e9e" stroke-width="1"/>
    <path d="M180,35 L210,45 L215,45 L185,35 Z" fill="#e0e0e0" stroke="#9e9e9e" stroke-width="1"/>
    <path d="M190,5 L195,5 L185,15 L180,15 Z" fill="#ff5252" stroke="#c62828" stroke-width="1"/> <!-- Vertical Stabilizer -->
    
    <!-- Wings -->
    <path d="M115,15 L150,-20 L160,-18 L130,15 Z" fill="url(#planeBodyGrad)" stroke="#9e9e9e" stroke-width="1"/>
    <path d="M115,35 L150,70 L160,68 L130,35 Z" fill="url(#planeBodyGrad)" stroke="#9e9e9e" stroke-width="1"/>
    
    <!-- Exhaust Puffs -->
    <g fill="white" opacity="0.6">
      <circle cx="10" cy="25" r="5">
        <animate attributeName="cx" values="10;0;-10" dur="1s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.6;0" dur="1s" repeatCount="indefinite"/>
        <animate attributeName="r" values="5;10" dur="1s" repeatCount="indefinite"/>
      </circle>
      <circle cx="10" cy="25" r="5">
        <animate attributeName="cx" values="10;0;-10" dur="1s" repeatCount="indefinite" begin="0.5s"/>
        <animate attributeName="opacity" values="0.6;0" dur="1s" repeatCount="indefinite" begin="0.5s"/>
        <animate attributeName="r" values="5;10" dur="1s" repeatCount="indefinite" begin="0.5s"/>
      </circle>
    </g>
    
    <!-- "Jump!" Speech Bubble -->
    <g transform="translate(190, -10)" opacity="0">
      <animate attributeName="opacity" values="0;1;1;0" dur="4s" begin="0.5s" repeatCount="indefinite"/>
      <path d="M0,0 L40,-20 L45,-15 L20,0 L45,15 L40,20 Z" fill="white" stroke="#333" stroke-width="1"/>
      <text x="22" y="5" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#333" text-anchor="middle">GO!</text>
    </g>
  </g>

  <!-- ================================================== -->
  -- 2. DUCK FREE FALL (Middle Right)                   --
  <!-- ================================================== -->
  <g transform="translate(550, 180)" filter="url(#dropShadow)">
     <!-- Motion Lines -->
     <g stroke="#fff" stroke-width="1.5" opacity="0.5">
        <line x1="0" y1="50" x2="0" y2="90">
          <animate attributeName="y1" values="50;120" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="90;160" dur="0.5s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.5;0" dur="0.5s" repeatCount="indefinite"/>
        </line>
        <line x1="-15" y1="55" x2="-15" y2="95" transform="rotate(-10)">
           <animate attributeName="y1" values="55;125" dur="0.5s" repeatCount="indefinite" begin="0.1s"/>
           <animate attributeName="y2" values="95;165" dur="0.5s" repeatCount="indefinite" begin="0.1s"/>
           <animate attributeName="opacity" values="0.5;0" dur="0.5s" repeatCount="indefinite" begin="0.1s"/>
        </line>
        <line x1="15" y1="55" x2="15" y2="95" transform="rotate(10)">
           <animate attributeName="y1" values="55;125" dur="0.5s" repeatCount="indefinite" begin="0.2s"/>
           <animate attributeName="y2" values="95;165" dur="0.5s" repeatCount="indefinite" begin="0.2s"/>
           <animate attributeName="opacity" values="0.5;0" dur="0.5s" repeatCount="indefinite" begin="0.2s"/>
        </line>
     </g>

    <!-- Duck Body (Streamlined) -->
    <ellipse cx="0" cy="0" rx="22" ry="16" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5"/>
    <!-- Head -->
    <circle cx="25" cy="-10" r="14" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5"/>
    <!-- Bill (Open screaming) -->
    <path d="M35,-15 L55,-5 L35,5 Z" fill="#ff9800" stroke="#e65100" stroke-width="1"/>
    <path d="M35,-2 L50,-2 L35,2 Z" fill="#ffcc80"/> <!-- Tongue -->
    <!-- Eye (Wide panic) -->
    <circle cx="30" cy="-13" r="5" fill="white" stroke="#333" stroke-width="0.5"/>
    <circle cx="32" cy="-13" r="2.5" fill="black"/>
    <ellipse cx="34" cy="-15" rx="1" ry="1.5" fill="white"/> <!-- Highlight -->
    <!-- Eyebrow (Angled down) -->
    <path d="M24,-18 L36,-10" stroke="#333" stroke-width="2" fill="none" stroke-linecap="round"/>
    <!-- Wing (Flapping back) -->
    <path d="M-5,5 Q-30,0 -20,-15 Q-10,-5 -5,5" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5">
      <animateTransform attributeName="transform" type="rotate" values="0 0 0; -20 0 0; 0 0 0" dur="0.4s" repeatCount="indefinite"/>
    </path>
    <!-- Legs (Tucked) -->
    <g stroke="#ff9800" stroke-width="2" stroke-linecap="round">
      <line x1="-8" y1="12" x2="-12" y2="20"/>
      <line x1="2" y1="12" x2="-2" y2="20"/>
    </g>
    <!-- "Aaaah" Text -->
    <text x="0" y="-40" font-family="Comic Sans MS, cursive" font-size="14" font-weight="bold" fill="#fff" stroke="#d32f2f" stroke-width="0.5" text-anchor="middle" opacity="0.8">
       <animate attributeName="opacity" values="0.8;0.3;0.8" dur="0.5s" repeatCount="indefinite"/>AAAAH!
    </text>
  </g>

  <!-- ================================================== -->
  -- 3. DUCK UNDER PARACHUTE (Lower Center)             --
  <!-- ================================================== -->
  <g transform="translate(350, 300)" filter="url(#dropShadow)">
    <!-- Parachute Canopy -->
    <path d="M-70,0 Q0,-80 70,0 Q50,20 -50,20 Z" fill="url(#chuteGrad)" stroke="#b71c1c" stroke-width="2"/>
    <!-- Panels -->
    <g stroke="#c62828" stroke-width="1" opacity="0.5">
      <path d="M-70,0 Q0,-80 0,0"/>
      <path d="M0,0 Q0,-80 70,0"/>
      <path d="M-35,0 Q0,-40 0,0"/>
      <path d="M0,0 Q0,-40 35,0"/>
    </g>
    <!-- Vent Hole -->
    <circle cx="0" cy="-70" r="6" fill="#1a1a1a"/>

    <!-- Suspension Lines -->
    <g stroke="#424242" stroke-width="1.5" stroke-linecap="round">
      <line x1="-60" y1="10" x2="-25" y2="60"/>
      <line x1="-20" y1="5" x2="-5" y2="60"/>
      <line x1="20" y1="5" x2="5" y2="60"/>
      <line x1="60" y1="10" x2="25" y2="60"/>
    </g>

    <!-- Harness / Backpack -->
    <rect x="-12" y="55" width="24" height="15" rx="3" fill="#3e2723" stroke="#212121" stroke-width="1"/>
    
    <!-- Duck Body (Hanging) -->
    <g transform="translate(0, 70)">
      <!-- Body -->
      <ellipse cx="0" cy="10" rx="18" ry="14" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5"/>
      <!-- Head (Looking down, happy/relieved) -->
      <circle cx="22" cy="-2" r="12" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5"/>
      <!-- Bill (Smile) -->
      <path d="M30,-6 L44,0 L30,6 Z" fill="#ff9800" stroke="#e65100" stroke-width="1"/>
      <path d="M32,0 L40,0 L32,3 Z" fill="#ffccbc"/> 
      <!-- Eye (Happy/Squint) -->
      <path d="M26,-6 Q30,-10 34,-6" stroke="#333" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      <!-- Eyebrow (Relaxed) -->
      <path d="M22,-12 Q28,-15 34,-12" stroke="#333" stroke-width="1.5" fill="none"/>
      <!-- Wings (Holding risers) -->
      <path d="M-8,5 Q-25,10 -15,30" stroke="#fff9c4" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      <path d="M8,5 Q25,10 15,30" stroke="#fff9c4" stroke-width="8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
      <!-- Legs (Dangling) -->
      <g stroke="#ff9800" stroke-width="3" stroke-linecap="round">
        <line x1="-6" y1="22" x2="-8" y2="35">
           <animateTransform attributeName="transform" type="rotate" values="-10 -6 22; 10 -6 22; -10 -6 22" dur="2s" repeatCount="indefinite"/>
        </line>
        <line x1="6" y1="22" x2="8" y2="35">
           <animateTransform attributeName="transform" type="rotate" values="10 6 22; -10 6 22; 10 6 22" dur="2s" repeatCount="indefinite"/>
        </line>
      </g>
      
      <!-- "Whee!" Text -->
      <text x="0" y="-30" font-family="Comic Sans MS, cursive" font-size="12" fill="#fff" stroke="#1b5e20" stroke-width="0.5" text-anchor="middle" font-weight="bold">WHEEE!</text>
    </g>
    
    <!-- Slight Sway Animation for whole parachute group -->
    <animateTransform attributeName="transform" type="translate" values="350,300; 355,305; 350,300" dur="4s" repeatCount="indefinite" additive="sum"/>
    <animateTransform attributeName="transform" type="rotate" values="-2; 2; -2" dur="4s" repeatCount="indefinite" additive="sum" origin="0 -70"/>
  </g>

  <!-- ================================================== -->
  -- 4. LANDED DUCK (Bottom Right)                      --
  <!-- ================================================== -->
  <g transform="translate(650, 470)" filter="url(#dropShadow)">
    <!-- Collapsed Parachute on Ground -->
    <path d="M-40,0 Q-20,-10 0,0 Q20,-10 40,0 Q20,15 -20,15 Z" fill="#ff5252" opacity="0.7" stroke="#b71c1c" stroke-width="1"/>
    <path d="M-10,0 Q-5,-5 0,0 Q5,-5 10,0" stroke="#b71c1c" stroke-width="0.5" fill="none" opacity="0.5"/>
    
    <!-- Duck Standing, Wings Up (Victory) -->
    <g transform="translate(0, -10)">
      <!-- Shadow -->
      <ellipse cx="0" cy="35" rx="25" ry="5" fill="#000" opacity="0.1"/>
      
      <!-- Legs -->
      <g stroke="#ff9800" stroke-width="4" stroke-linecap="round">
        <line x1="-10" y1="20" x2="-12" y2="35"/>
        <line x1="10" y1="20" x2="12" y2="35"/>
        <!-- Feet -->
        <polyline points="-12,35 -18,35 -12,38" fill="none"/>
        <polyline points="12,35 18,35 12,38" fill="none"/>
      </g>
      
      <!-- Body -->
      <ellipse cx="0" cy="10" rx="20" ry="16" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5"/>
      <!-- Tail feathers -->
      <path d="M-20,10 Q-30,5 -25,15 Q-20,10 -20,10" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5"/>
      
      <!-- Head -->
      <circle cx="24" cy="-6" r="13" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5"/>
      <!-- Bill (Grin) -->
      <path d="M32,-10 L48,-4 L32,2 Z" fill="#ff9800" stroke="#e65100" stroke-width="1"/>
      <!-- Eye (Closed Happy) -->
      <path d="M20,-10 Q27,-14 34,-10" stroke="#333" stroke-width="3" fill="none" stroke-linecap="round"/>
      <!-- Eyebrow -->
      <path d="M18,-14 Q27,-18 36,-14" stroke="#333" stroke-width="1.5" fill="none"/>
      
      <!-- Wings (Raised High) -->
      <path d="M-5,0 Q-35,-30 -10,-40 Q-5,-20 -5,0" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5">
        <animateTransform attributeName="transform" type="rotate" values="0 0 0; -5 0 0; 0 0 0" dur="0.8s" repeatCount="indefinite"/>
      </path>
      <path d="M5,0 Q35,-30 10,-40 Q5,-20 5,0" fill="#fff9c4" stroke="#fbc02d" stroke-width="1.5">
        <animateTransform attributeName="transform" type="rotate" values="0 0 0; 5 0 0; 0 0 0" dur="0.8s" repeatCount="indefinite"/>
      </path>
      
      <!-- "Naailed it!" Text -->
      <text x="0" y="-45" font-family="Arial Black, sans-serif" font-size="11" fill="#1b5e20" font-weight="bold" text-anchor="middle" opacity="0">NAILED IT!
        <animate attributeName="opacity" values="0;1;1;0" dur="4s" begin="1s" repeatCount="indefinite"/>
      </text>
    </g>
  </g>

  <!-- Title / Caption -->
  <text x="400" y="30" font-family="Georgia, serif" font-size="24" fill="white" text-anchor="middle" font-weight="bold" opacity="0.9" filter="url(#dropShadow)">
    Operation: Quack Drop
  </text>
  <text x="400" y="55" font-family="Georgia, serif" font-size="14" fill="#e3f2fd" text-anchor="middle" opacity="0.8">
    1. Exit  →  2. Freefall  →  3. Canopy  →  4. Stick the Landing
  </text>

</svg>
```

</details>

## 9. qwen3.5_397b-cloud

**SVG size:** 3277 characters  
**Raw response:** 3378 characters

![qwen3.5_397b-cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/qwen3.5_397b-cloud.svg)

### Analysis

This SVG contains approximately **33 shape elements** and uses **15 distinct colors**.

**Notable SVG techniques used:**

- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a compact SVG (under 5,000 characters) that uses minimal markup to convey the scene. The model prioritized efficiency and simplicity over elaborate detail.

<details><summary>View raw SVG source</summary>

```xml
<svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Background: Sky -->
  <rect width="600" height="400" fill="#87CEEB" />
  
  <!-- Clouds -->
  <g fill="#FFFFFF" opacity="0.8">
    <circle cx="100" cy="80" r="30" />
    <circle cx="140" cy="90" r="40" />
    <circle cx="80" cy="100" r="30" />
    
    <circle cx="450" cy="50" r="25" />
    <circle cx="490" cy="60" r="35" />
    <circle cx="430" cy="70" r="25" />
  </g>

  <!-- Ground -->
  <path d="M0 350 Q 300 320 600 350 L 600 400 L 0 400 Z" fill="#90EE90" />

  <!-- The Plane -->
  <g transform="translate(50, 80) rotate(-5)">
    <!-- Speed lines -->
    <line x1="-20" y1="20" x2="-60" y2="20" stroke="#555" stroke-width="2" stroke-linecap="round" />
    <line x1="-10" y1="30" x2="-50" y2="30" stroke="#555" stroke-width="2" stroke-linecap="round" />
    
    <!-- Fuselage -->
    <ellipse cx="100" cy="50" rx="90" ry="25" fill="#D3D3D3" stroke="#555" stroke-width="2"/>
    <!-- Cockpit -->
    <path d="M 140 35 Q 170 35 180 50 L 140 50 Z" fill="#ADD8E6" stroke="#555" stroke-width="2"/>
    <!-- Wing -->
    <path d="M 80 50 L 120 50 L 110 10 L 90 10 Z" fill="#A9A9A9" stroke="#555" stroke-width="2"/>
    <!-- Tail -->
    <path d="M 20 50 L 40 50 L 30 10 Z" fill="#A9A9A9" stroke="#555" stroke-width="2"/>
    <!-- Propeller (blurred) -->
    <ellipse cx="190" cy="50" rx="5" ry="25" fill="#333" opacity="0.6"/>
  </g>

  <!-- The Duck and Parachute -->
  <g transform="translate(300, 180)">
    
    <!-- Parachute Canopy -->
    <path d="M -60 0 Q 0 -70 60 0" fill="#FF4500" stroke="#8B0000" stroke-width="2"/>
    <!-- Parachute Stripes -->
    <path d="M -40 0 Q -10 -50 20 0" fill="none" stroke="#FFF" stroke-width="3" opacity="0.5"/>
    <path d="M -10 0 Q 10 -60 40 0" fill="none" stroke="#FFF" stroke-width="3" opacity="0.5"/>
    
    <!-- Parachute Strings -->
    <line x1="-60" y1="0" x2="-15" y2="60" stroke="#333" stroke-width="1" />
    <line x1="60" y1="0" x2="15" y2="60" stroke="#333" stroke-width="1" />
    <line x1="0" y1="-10" x2="0" y2="60" stroke="#333" stroke-width="1" />

    <!-- Duck Body -->
    <ellipse cx="0" cy="75" rx="18" ry="24" fill="#FFD700" stroke="#DAA520" stroke-width="2"/>
    
    <!-- Duck Head -->
    <circle cx="0" cy="50" r="14" fill="#FFD700" stroke="#DAA520" stroke-width="2"/>
    
    <!-- Beak -->
    <path d="M 10 52 L 22 56 L 10 60 Z" fill="#FFA500" stroke="#DAA520" stroke-width="1"/>
    
    <!-- Eye -->
    <circle cx="4" cy="46" r="2" fill="#000" />
    <circle cx="5" cy="45" r="0.8" fill="#FFF" />
    
    <!-- Goggles -->
    <path d="M -5 44 Q 0 40 5 44 L 5 50 Q 0 54 -5 50 Z" fill="none" stroke="#333" stroke-width="2"/>
    <rect x="-6" y="44" width="12" height="6" fill="#87CEEB" opacity="0.6"/>

    <!-- Wings (flapping) -->
    <path d="M -10 70 Q -30 60 -20 80" fill="#FFD700" stroke="#DAA520" stroke-width="2"/>
    <path d="M 10 70 Q 30 60 20 80" fill="#FFD700" stroke="#DAA520" stroke-width="2"/>

    <!-- Feet -->
    <path d="M -5 95 L -10 105 L -2 105 Z" fill="#FFA500" stroke="#DAA520" stroke-width="1"/>
    <path d="M 5 95 L 10 105 L 2 105 Z" fill="#FFA500" stroke="#DAA520" stroke-width="1"/>
    
    <!-- Harness -->
    <rect x="-10" y="60" width="20" height="15" fill="#333" rx="2"/>
  </g>
</svg>
```

</details>

## 10. deepseek-v4-flash_0731-cloud

**SVG size:** 6400 characters  
**Raw response:** 6400 characters

![deepseek-v4-flash_0731-cloud duck parachute SVG](/assets/img/diagrams/ollama-svg-parachute/deepseek-v4-flash_0731-cloud.svg)

### Analysis

This SVG contains approximately **56 shape elements** and uses **7 distinct colors**.

**Notable SVG techniques used:**

- Uses `<defs>` blocks for reusable components (a sign of well-structured, editable SVG)
- Includes gradient fills for richer visual depth
- Uses transforms (translate, rotate, scale) for precise positioning
- Contains XML comments, showing the model's internal documentation of its work

This is a moderately sized SVG (5,000-10,000 characters) that balances detail with manageable file size. A good middle ground for web embedding and visual richness.

<details><summary>View raw SVG source</summary>

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800" width="100%" height="auto">
  <defs>
    <!-- Sky Gradient -->
    <linearGradient id="skyGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#4A90E2" />
      <stop offset="100%" stop-color="#A9E0FF" />
    </linearGradient>
    <!-- Parachute Canopy Gradient -->
    <linearGradient id="canopyGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FF4500" />
      <stop offset="50%" stop-color="#FF8C00" />
      <stop offset="100%" stop-color="#FF4500" />
    </linearGradient>
    <!-- Duck Body Gradient -->
    <radialGradient id="duckGrad" cx="400" cy="550" r="150" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#FFF8DC" />
      <stop offset="40%" stop-color="#FFD700" />
      <stop offset="100%" stop-color="#FFA500" />
    </radialGradient>
    <!-- Duck Wing Gradient -->
    <linearGradient id="wingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FFE066" />
      <stop offset="100%" stop-color="#FFCC00" />
    </linearGradient>
  </defs>
  <!-- Background -->
  <rect width="800" height="800" fill="url(#skyGrad)" />
  <!-- Sun -->
  <circle cx="100" cy="100" r="90" fill="#FFD700" opacity="0.2" />
  <circle cx="100" cy="100" r="70" fill="#FFD700" opacity="0.4" />
  <circle cx="100" cy="100" r="50" fill="#FFD700" opacity="0.9" />
  <!-- Clouds -->
  <g fill="#FFF" opacity="0.8">
    <!-- Cloud 1 (Top Right) -->
    <circle cx="650" cy="100" r="50" />
    <circle cx="700" cy="120" r="60" />
    <circle cx="740" cy="150" r="45" />
    <circle cx="680" cy="160" r="40" />
    <!-- Cloud 2 (Mid Left) -->
    <circle cx="120" cy="350" r="40" />
    <circle cx="160" cy="330" r="50" />
    <circle cx="200" cy="360" r="35" />
    <!-- Cloud 3 (Bottom Right) -->
    <circle cx="680" cy="600" r="45" opacity="0.6" />
    <circle cx="730" cy="620" r="55" opacity="0.6" />
    <circle cx="760" cy="650" r="40" opacity="0.6" />
  </g>
  <!-- Wind / Speed Lines -->
  <g fill="none" stroke="#FFF" stroke-width="4" stroke-linecap="round" opacity="0.6">
    <path d="M 120 200 Q 150 200, 170 220" />
    <path d="M 630 180 Q 660 180, 680 200" />
    <path d="M 640 250 Q 670 250, 690 270" />
    <path d="M 220 400 L 280 400" stroke-dasharray="10 10" />
    <path d="M 520 450 L 600 450" stroke-dasharray="15 15" />
    <path d="M 180 600 L 240 600" stroke-dasharray="8 8" />
    <path d="M 550 650 L 620 650" stroke-dasharray="12 12" />
  </g>
  <!-- Ropes -->
  <g stroke="#555" stroke-width="3" stroke-linecap="round">
    <line x1="253.3" y1="280" x2="370" y2="550" />
    <line x1="326.6" y1="280" x2="370" y2="550" />
    <line x1="400" y1="280" x2="370" y2="550" />
    <line x1="473.3" y1="280" x2="370" y2="550" />
    <line x1="546.6" y1="280" x2="370" y2="550" />
  </g>
  <!-- Parachute Canopy -->
  <g>
    <!-- Main Canopy Shape -->
    <path d="M 180 280 
             C 180 80, 620 80, 620 280 
             Q 586.6 240, 546.6 280 
             Q 513.3 240, 473.3 280 
             Q 440 240, 400 280 
             Q 366.6 240, 326.6 280 
             Q 293.3 240, 253.3 280 
             Q 226.6 240, 180 280 Z" 
          fill="url(#canopyGrad)" />
    <!-- Canopy Seams (White Stripes) -->
    <path d="M 400 80 Q 300 180, 253.3 280" fill="none" stroke="#FFF" stroke-width="5" opacity="0.9" />
    <path d="M 400 80 Q 360 180, 326.6 280" fill="none" stroke="#FFF" stroke-width="5" opacity="0.9" />
    <path d="M 400 80 Q 440 180, 473.3 280" fill="none" stroke="#FFF" stroke-width="5" opacity="0.9" />
    <path d="M 400 80 Q 500 180, 546.6 280" fill="none" stroke="#FFF" stroke-width="5" opacity="0.9" />
  </g>
  <!-- Duck -->
  <g>
    <!-- Tail -->
    <path d="M 290 570 L 240 530 L 280 610 Z" fill="url(#duckGrad)" />
    <!-- Body -->
    <ellipse cx="370" cy="580" rx="90" ry="70" fill="url(#duckGrad)" />
    <!-- Neck -->
    <path d="M 410 510 L 480 500 L 470 580 L 400 580 Z" fill="url(#duckGrad)" />
    <!-- Head -->
    <circle cx="450" cy="490" r="50" fill="url(#duckGrad)" />
    <!-- Head Tuft -->
    <path d="M 440 445 Q 430 425, 445 420 Q 445 435, 450 440 Z" fill="url(#duckGrad)" />
    <path d="M 450 440 Q 455 410, 465 415 Q 460 430, 455 440 Z" fill="url(#duckGrad)" />
    <!-- Beak -->
    <path d="M 490 480 Q 550 480, 550 500 Q 550 515, 490 510 Z" fill="#FF8C00" />
    <circle cx="515" cy="485" r="2.5" fill="#CC7000" />
    <path d="M 495 505 Q 515 515, 530 505" fill="none" stroke="#CC7000" stroke-width="2" stroke-linecap="round" />
    <!-- Eye -->
    <circle cx="465" cy="475" r="7" fill="#000" />
    <circle cx="467" cy="473" r="2.5" fill="#FFF" />
    <!-- Wing -->
    <path d="M 330 570 C 330 540, 410 550, 410 590 C 410 620, 350 620, 330 570 Z" fill="url(#wingGrad)" />
    <!-- Wing Feather Details -->
    <path d="M 350 570 Q 370 580, 390 580" fill="none" stroke="#E6A800" stroke-width="3" stroke-linecap="round" />
    <path d="M 345 580 Q 365 590, 385 590" fill="none" stroke="#E6A800" stroke-width="3" stroke-linecap="round" />
    <path d="M 350 590 Q 370 600, 380 600" fill="none" stroke="#E6A800" stroke-width="3" stroke-linecap="round" />
    <!-- Highlights -->
    <ellipse cx="435" cy="470" rx="15" ry="10" fill="#FFF" opacity="0.4" transform="rotate(-30 435 470)" />
    <ellipse cx="330" cy="550" rx="20" ry="15" fill="#FFF" opacity="0.4" transform="rotate(-45 330 550)" />
    <!-- Feet -->
    <!-- Left Foot -->
    <path d="M 330 640 Q 320 670, 300 680 L 330 690 L 350 680 Q 340 660, 340 640 Z" fill="#FF8C00" />
    <!-- Right Foot -->
    <path d="M 390 640 Q 380 670, 360 680 L 390 690 L 410 680 Q 400 660, 400 640 Z" fill="#FF8C00" />
  </g>
  <!-- Harness -->
  <g>
    <!-- Harness Straps -->
    <path d="M 310 550 Q 370 580, 430 550" fill="none" stroke="#8B4513" stroke-width="8" stroke-linecap="round" />
    <path d="M 310 610 Q 370 640, 430 610" fill="none" stroke="#8B4513" stroke-width="8" stroke-linecap="round" />
    <line x1="330" y1="550" x2="330" y2="610" stroke="#8B4513" stroke-width="8" stroke-linecap="round" />
    <line x1="410" y1="550" x2="410" y2="610" stroke="#8B4513" stroke-width="8" stroke-linecap="round" />
    <!-- Harness Ring (Connection point for ropes) -->
    <circle cx="370" cy="550" r="14" fill="#A9A9A9" stroke="#555" stroke-width="4" />
    <circle cx="370" cy="550" r="6" fill="#555" />
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

## Comparison With the Bicycle Prompt

Compared to the [previous duck-on-a-bicycle comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/), this parachute prompt is more challenging because it requires:

- **Two distinct objects**: a duck and a plane (vs. just a duck + bicycle)
- **Vertical composition**: sky, falling motion, ground (vs. horizontal ground scene)
- **A specific mechanism**: parachute with cords and canopy (vs. just two wheels)
- **Spatial relationships**: duck below plane, parachute above duck

Observations across models:

- Most models successfully rendered the parachute as a canopy shape (often a semicircle or arc)
- Several models added motion lines to indicate falling, which was not present in the bicycle scene
- The plane was the most varied element: some drew a simple rectangle, others a detailed fuselage with windows
- Sky backgrounds (clouds, sun) were more common in this prompt than ground backgrounds in the bicycle prompt

## Conclusion

This comparison shows that 10 out of 14 active Ollama cloud models can generate valid SVG artwork from a complex natural language prompt involving multiple objects, spatial relationships, and a specific mechanism. The results vary dramatically in complexity, style, and technique:

- **Simplest output**: nemotron-3-super:cloud produced a compact SVG with basic shapes
- **Most detailed output**: nemotron-3-ultra:cloud generated a 15,000+ character SVG with rich scene elements
- **Most technically advanced**: deepseek-v4-pro:cloud used `<defs>`, `<use>`, and transforms

All SVGs were generated with zero post-processing -- what you see is exactly what each model produced. The variety of approaches (from minimalist geometric to richly detailed scenes) highlights how different training data and model architectures lead to distinctly different creative outputs, especially when the prompt demands complex scene composition.

## Links

- [Previous: Duck Driving a Bicycle Comparison](/Ollama-Cloud-Models-SVG-Comparison-Duck-Bicycle/)
- [Ollama Official Website](https://ollama.com)
- [Ollama Cloud Documentation](https://ollama.com/cloud)
- [SVG Specification (MDN)](https://developer.mozilla.org/en-US/docs/Web/SVG)
