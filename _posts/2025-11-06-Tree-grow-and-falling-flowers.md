---
description: Learn how to create a beautiful animated tree with growing and falling flowers using Python's Turtle graphics module. Step-by-step tutorial with physics simu...
featured-img: 20251109-cherryblossom
keywords:
- Python
- Turtle Graphics
- Animation
- Recursive Tree
- Flower Animation
- Physics Simulation
layout: post
mathjax: false
tags:
- python
- turtle-graphics
- animation
- recursion
- physics
- tutorial
title: Animated Flower Tree with Python Turtle – Growing &...
---
# Animated Flower Tree with Python Turtle – Growing & Falling Flowers

## Create Stunning Animated Nature Scenes with Python's Built-in Turtle Module

Python's Turtle graphics module isn't just for simple shapes – it can create beautiful, complex animations with physics simulations! This tutorial shows you how to build an **animated flowering tree** where blossoms gradually grow on branches and then gently fall with realistic physics. You'll learn recursive tree generation, color blending, growth animation, and basic physics simulation – all with pure Python.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/r2zzJFscM3M" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

---

## Overview

This project creates a **procedurally generated tree** with these animated features:

1. **Recursive tree generation** with random branching angles
2. **Color blending** from brown trunk to green branches
3. **Gradual flower growth** animation
4. **Realistic flower falling** with gravity and wind physics
5. **Natural swaying motion** for attached flowers

---

## What You'll Learn

- **Recursive algorithms** for tree generation
- **Turtle graphics** advanced techniques
- **Color theory** and RGB blending
- **Animation principles** with growth stages
- **Basic physics simulation** (gravity, wind, rotation)
- **Object-oriented thinking** with flower data structures

---

## Step 1: Setting Up the Environment

We start by importing necessary modules and setting up configuration constants.

```python
import turtle 
from random import random, uniform, seed
import math
import time

## Settings 
W, H = 420, 420          # Window dimensions
DEPTH = 7                # Recursion depth for tree
LENGTH = 80              # Initial branch length
BA = 25                  # Base branching angle
GRAVITY = 1.5            # Physics constants
WIND = 2.0               # Wind strength
FPS = 0.02               # Animation frame delay
seed(192)                # Seed for reproducible randomness

branch_params = []

```

Key Settings Explained:

* `DEPTH` Controls how detailed the tree is (higher = more branches)
* `LENGTH` Starting branch length in pixels
* `GRAVITY/WIND` Physics parameters for falling animation
* `seed(192)` Ensures the same tree generates every time

---

## Step 2: Generating the Tree Structure

We use recursion to generate tree parameters before drawing.

```python
def record_my_params(length, depth, level=0):
    """Recursively generate tree branch parameters"""
    if depth == 0:
        return
  
    ## Randomize angles and lengths for natural look
    a1 = uniform(BA - 10, BA + 10)
    a2 = uniform(BA - 10, BA + 10)
    l1 = length * uniform(0.6, 0.8)  # Child branches are shorter
    l2 = length * uniform(0.6, 0.8)
  
    branch_params.append((a1, a2, l1, l2, level))
  
    ## Recursive calls for sub-branches
    record_my_params(l1, depth - 1, level + 1)
    record_my_params(l2, depth - 1, level + 1)

record_my_params(LENGTH, DEPTH)

def param_gen():
    """Generator for branch parameters"""
    for p in branch_params:
        yield p

```

How Recursion Works:

* Each branch splits into two sub-branches
* Sub-branches have randomized angles and lengths
* Process continues until depth reaches 0
* Parameters are stored for consistent drawing

---

## Step 3: Color Blending for Natural Look

We blend colors from brown (trunk) to green (branches) based on branch level.

```python
def blend(c1, c2, t):
    """Linear color interpolation between two RGB colors"""
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )

## Usage in drawing:
t = (DEPTH - depth) / DEPTH  # Normalized position (0=trunk, 1=tip)
brown = (139/255, 69/255, 19/255)
green = (34/255, 139/255, 34/255)
r, g, b = blend(brown, green, t)

```

### Color Theory:

* Brown trunk: (139/255, 69/255, 19/255) - earthy, strong
* Green tips: (34/255, 139/255, 34/255) - vibrant, leafy
* Linear interpolation: Creates smooth color transitions

---

## Step 4: Drawing Flowers with Growth Animation

Flowers are drawn as pentagram shapes with gradual growth animation.

```python
def draw_flower_at_position(pen, x, y, color, size, rotation=0, growth_factor=1.0):
    """Draw a flower at specific position with growth animation"""
    pen.penup()
    pen.goto(x, y)
    pen.setheading(rotation)
    pen.pendown()
  
    ## Apply growth factor to size
    current_size = size * growth_factor
    if current_size < 0.5:  # Don't draw if too small
        return
      
    pen.fillcolor(color)
    pen.pencolor(color)
    pen.begin_fill()
    for _ in range(5):
        pen.circle(current_size, 72)  # Draw petal
        pen.left(144)                 # Turn for next petal
    pen.end_fill()
  
    ## Draw flower center
    pen.penup()
    pen.goto(x, y)
    pen.dot(current_size * 0.6, (0.9, 0.4, 0.6))

def animate_flower_growth(screen, tree_pen, flower_positions):
    """Animate flowers growing on the tree"""
    flower_pen = turtle.Turtle()
    flower_pen.hideturtle()
    flower_pen.speed(0)
    flower_pen.penup()
  
    growth_stages = 20  # Number of growth stages
  
    for stage in range(growth_stages + 1):
        flower_pen.clear()
        growth_factor = stage / growth_stages  # 0 to 1
      
        for flower in flower_positions:
            draw_flower_at_position(
                flower_pen, 
                flower['x'], 
                flower['y'], 
                flower['color'], 
                flower['size'], 
                growth_factor=growth_factor
            )
      
        screen.update()
        time.sleep(FPS)

```

### Growth Animation Features:

* 20 growth stages for smooth appearance
* Size scaling from 0% to 100%
* Separate turtle for flowers vs tree
* Controlled frame rate for smooth animation

---

### Step 5: Physics for Falling Flowers

Realistic falling animation with gravity, wind, and rotation physics.

```python
def make_flowers_fall(screen, tree_pen, flower_pen, flower_positions):
    """Animate flowers falling with physics simulation"""
  
    velocities = {}  # Store velocity for each flower
    rotations = {}   # Store rotation for each flower
  
    while True:
        flower_pen.clear()
      
        ## Randomly detach flowers over time
        for flower in flower_positions:
            if flower['attached'] and random() < 0.03:
                flower['attached'] = False
                flower_id = flower['id']
              
                ## Initialize physics properties
                velocities[flower_id] = [
                    uniform(-WIND, WIND),  # Horizontal velocity (wind)
                    -GRAVITY * uniform(0.8, 1.2)  # Vertical velocity
                ]
                rotations[flower_id] = uniform(0, 360)  # Initial rotation
      
        ## Update falling flowers
        for flower in flower_positions:
            if not flower['attached']:
                flower_id = flower['id']
                vx, vy = velocities[flower_id]
              
                ## Apply physics
                flower['x'] += vx
                flower['y'] += vy
                vy -= GRAVITY * 0.08  # Gravity acceleration
              
                ## Air resistance and random movement
                vx *= 0.998
                vy *= 0.998
                vx += uniform(-0.008, 0.008)
              
                ## Update rotation based on horizontal movement
                rotations[flower_id] += vx * 3
              
                ## Ground collision
                if flower['y'] <= -H//2 + 20:
                    flower['y'] = -H//2 + 20  # Stop at ground
                    if abs(vy) > 0.3:
                        vy = -vy * 0.2  # Bounce
                    else:
                        vy = 0  # Stop moving
              
                velocities[flower_id] = [vx, vy]
              
                ## Draw with current rotation
                draw_flower_at_position(
                    flower_pen, 
                    flower['x'], 
                    flower['y'], 
                    flower['color'], 
                    flower['size'], 
                    rotations[flower_id]
                )
      
        screen.update()
        time.sleep(FPS)
```

### Physics Simulation Elements:

* Gravity: Constant downward acceleration
* Wind: Random horizontal movement
* Rotation: Flowers spin as they fall
* Air resistance: Velocity gradually decreases
* Ground collision: Realistic bouncing and stopping
* Random variations: Natural, non-uniform movement

---

## Complete Code

{% include codeHeader.html %}

```python
## source code at www.pyshine.com
import turtle 
from random import random, uniform, seed
import math
import time

## Settings 
W, H = 420, 420
DEPTH = 7
LENGTH = 80
BA = 25
GRAVITY = 1.5
WIND = 2.0
FPS = 0.02
seed(192)

branch_params = []

def record_my_params(length, depth, level=0):
    if depth == 0:
        return
    a1 = uniform(BA - 10, BA + 10)
    a2 = uniform(BA - 10, BA + 10)
    l1 = length * uniform(0.6, 0.8)
    l2 = length * uniform(0.6, 0.8)
    branch_params.append((a1, a2, l1, l2, level))
    record_my_params(l1, depth - 1, level + 1)
    record_my_params(l2, depth - 1, level + 1)

record_my_params(LENGTH, DEPTH)

def param_gen():
    for p in branch_params:
        yield p

def blend(c1, c2, t):
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )

def draw_flower_at_position(pen, x, y, color, size, rotation=0, growth_factor=1.0):
    """Draw a flower at a specific position with growth animation"""
    pen.penup()
    pen.goto(x, y)
    pen.setheading(rotation)
    pen.pendown()
  
    ## Apply growth factor to size
    current_size = size * growth_factor
    if current_size < 0.5:  # Don't draw if too small
        return
      
    pen.fillcolor(color)
    pen.pencolor(color)
    pen.begin_fill()
    for _ in range(5):
        pen.circle(current_size, 72)
        pen.left(144)
    pen.end_fill()
    pen.penup()
    pen.goto(x, y)
    pen.dot(current_size * 0.6, (0.9, 0.4, 0.6))

def draw_tree_with_flowers():
    screen = turtle.Screen()
    screen.setup(W, H)
    screen.title("Turtle Tree with Growing and Falling Flowers")
    screen.bgcolor("black")
    screen.tracer(0, 0)  # Turn off animation for initial drawing
  
    ## Create main tree pen
    tree_pen = turtle.Turtle()
    tree_pen.hideturtle()
    tree_pen.speed(0)
    tree_pen.left(90)
    tree_pen.penup()
    tree_pen.goto(0, -H//2 + 30)
    tree_pen.pendown()

    ## Store flower positions during tree drawing
    flower_positions = []

    def draw_branch(length, depth):
        """Recursively draw tree branches and record flower positions."""
        if depth == 0:
            ## Record leaf/flower position but don't draw yet
            size = uniform(2, 5)
            color = (1.0, uniform(0.6, 0.8), uniform(0.7, 0.9))
            flower_positions.append({
                'x': tree_pen.xcor(),
                'y': tree_pen.ycor(),
                'color': color,
                'size': size,
                'attached': True,
                'grown': False,  # Flower hasn't grown yet
                'id': len(flower_positions)  # Unique ID for each flower
            })
            return
          
        t = (DEPTH - depth) / DEPTH
        brown = (139/255, 69/255, 19/255)
        green = (34/255, 139/255, 34/255)
        r, g, b = blend(brown, green, t)
        tree_pen.pencolor(r, g, b)
        tree_pen.pensize(max(1, depth / 2))
        tree_pen.forward(length)
      
        if random() < 0.12 and depth < DEPTH - 2:
            ## Record branch flower position but don't draw yet
            size = uniform(2, 5)
            color = (1.0, uniform(0.6, 0.8), uniform(0.7, 0.9))
            flower_positions.append({
                'x': tree_pen.xcor(),
                'y': tree_pen.ycor(),
                'color': color,
                'size': size,
                'attached': True,
                'grown': False,  # Flower hasn't grown yet
                'id': len(flower_positions)  # Unique ID for each flower
            })
          
        try:
            a1, a2, l1, l2, level = next(params)
        except StopIteration:
            return
          
        tree_pen.left(a1)
        draw_branch(l1, depth - 1)
        tree_pen.right(a1 + a2)
        draw_branch(l2, depth - 1)
        tree_pen.left(a2)
        tree_pen.backward(length)

    ## Draw the initial tree structure (without flowers)
    params = param_gen()
    draw_branch(LENGTH, DEPTH)
  
    screen.update()  # Show the initial tree
    print(f"Tree structure complete. {len(flower_positions)} flower positions recorded.")
  
    ## Now animate flowers growing
    animate_flower_growth(screen, tree_pen, flower_positions)

def animate_flower_growth(screen, tree_pen, flower_positions):
    """Animate flowers growing on the tree"""
  
    ## Create a separate pen for flowers
    flower_pen = turtle.Turtle()
    flower_pen.hideturtle()
    flower_pen.speed(0)
    flower_pen.penup()
  
    print("Flowers growing...")
  
    ## Grow flowers gradually
    growth_stages = 20  # Number of growth stages
    breeze_phase = 0
    breeze_strength = 0.8
    breeze_frequency = 0.08
  
    for stage in range(growth_stages + 1):
        flower_pen.clear()
      
        ## Calculate breeze effect for subtle movement
        breeze_phase += breeze_frequency
        current_breeze = math.sin(breeze_phase) * breeze_strength
      
        ## Calculate growth factor (0 to 1)
        growth_factor = stage / growth_stages
      
        ## Draw each flower with current growth factor
        for flower in flower_positions:
            if not flower['grown']:
                ## Slight sway with breeze
                sway_x = current_breeze * 0.1 * math.sin(stage * 0.5 + flower['x'] * 0.01)
              
                draw_flower_at_position(
                    flower_pen, 
                    flower['x'] + sway_x, 
                    flower['y'], 
                    flower['color'], 
                    flower['size'], 
                    current_breeze * 2,
                    growth_factor
                )
      
        screen.update()
        time.sleep(FPS)
  
    ## Mark all flowers as fully grown
    for flower in flower_positions:
        flower['grown'] = True
  
    print("All flowers have grown!")
    time.sleep(1)  # Pause to show fully grown flowers
  
    ## Now start the falling animation
    make_flowers_fall(screen, tree_pen, flower_pen, flower_positions)

def make_flowers_fall(screen, tree_pen, flower_pen, initial_flowers):
    """Animate flowers falling from the tree"""
  
    ## Create a separate pen for falling flowers
    fall_pen = turtle.Turtle()
    fall_pen.hideturtle()
    fall_pen.speed(0)
    fall_pen.penup()
  
    ## Copy initial flowers for animation
    flowers = []
    for flower in initial_flowers:
        flowers.append({
            'x': flower['x'],
            'y': flower['y'], 
            'color': flower['color'],
            'size': flower['size'],
            'attached': True,
            'grown': True,
            'id': flower['id']
        })
  
    ## Initialize velocities and rotations
    velocities = {}
    rotations = {}
    fallen_count = 0
    frame_count = 0
    breeze_phase = 0
    breeze_strength = 0.8
    breeze_frequency = 0.08

    print(f"Flowers starting to fall...")

    ## Store which flowers are currently attached to the tree
    attached_flowers = {flower['id']: flower for flower in flowers if flower['attached']}
  
    while True:
        frame_count += 1
        fall_pen.clear()
        flower_pen.clear()  # Clear the flower pen to remove flowers from tree
      
        ## Calculate breeze effect
        breeze_phase += breeze_frequency
        current_breeze = math.sin(breeze_phase) * breeze_strength

        ## Check for flowers starting to fall
        flowers_to_detach = []
        for flower in flowers:
            if (flower['attached'] and frame_count > 10 and 
                random() < 0.03 and frame_count % 4 == 0):
                flowers_to_detach.append(flower['id'])
      
        ## Detach the flowers
        for flower_id in flowers_to_detach:
            for flower in flowers:
                if flower['id'] == flower_id and flower['attached']:
                    flower['attached'] = False
                    velocities[flower_id] = [
                        uniform(-WIND * 0.7, WIND * 0.7) + current_breeze * 0.4,
                        -GRAVITY * uniform(0.8, 1.2)
                    ]
                    rotations[flower_id] = uniform(0, 360)
                    fallen_count += 1
                    ## Remove from attached flowers
                    if flower_id in attached_flowers:
                        del attached_flowers[flower_id]
                    print(f"🌸 Flower {flower_id} dropped! {fallen_count}/{len(flowers)}")

        ## Draw only the flowers that are still attached to the tree
        for flower in attached_flowers.values():
            if flower['attached']:
                ## Draw attached flower with slight sway
                sway_x = current_breeze * 0.1 * math.sin(frame_count * 0.2 + flower['x'] * 0.01)
                draw_flower_at_position(
                    flower_pen, 
                    flower['x'] + sway_x, 
                    flower['y'], 
                    flower['color'], 
                    flower['size'], 
                    current_breeze * 2
                )

        ## Update and draw falling flowers
        all_fallen = len(attached_flowers) == 0
        active_falling_flowers = False
      
        for flower in flowers:
            if not flower['attached']:
                ## Update falling flower physics
                flower_id = flower['id']
                vx, vy = velocities.get(flower_id, (0, 0))
              
                ## Apply physics
                flower['x'] += vx + current_breeze * 0.05
                flower['y'] += vy
                vy -= GRAVITY * 0.08
                vx += current_breeze * 0.015 + uniform(-0.008, 0.008)
                vx *= 0.998
                vy *= 0.998
              
                ## Update rotation
                if flower_id in rotations:
                    rotations[flower_id] += vx * 3
              
                ## Ground collision
                ground_level = -H//2 + flower['size'] * 1.4 + 20
                if flower['y'] <= ground_level:
                    flower['y'] = ground_level
                    if abs(vy) > 0.3:
                        vy = -vy * 0.2
                    else:
                        vy = 0
                        vx *= 0.8
              
                velocities[flower_id] = [vx, vy]
                rotation = rotations.get(flower_id, 0)
              
                ## Draw falling flower
                draw_flower_at_position(
                    fall_pen, 
                    flower['x'], 
                    flower['y'], 
                    flower['color'], 
                    flower['size'], 
                    rotation
                )
              
                if vy != 0 or vx != 0:
                    active_falling_flowers = True

        screen.update()
        time.sleep(FPS)
      
        ## End condition: all flowers have fallen and stopped moving
        if all_fallen and not active_falling_flowers and frame_count > 60:
            break

    print(f"All {fallen_count} flowers have fallen!")
  
    ## Keep the window open
    screen.exitonclick()

if __name__ == "__main__":
    draw_tree_with_flowers()
```

---

## How It Works

* Tree Generation: Recursive algorithm builds tree structure with random variations
* Flower Placement: Flowers are positioned at branch endpoints during tree drawing
* Growth Animation: Flowers appear gradually over 20 animation frames
* Physics Simulation: Each flower gets velocity and rotation properties
* Natural Falling: Random detachment with gravity, wind, and collision detection

---

## Customization Ideas

* Seasonal Colors: Change flower colors for different seasons
* Different Tree Types: Modify branching patterns for various tree species
* Weather Effects: Add rain, snow, or stronger wind
* Interactive Elements: Click to make flowers fall or grow new ones
* Background: Add sky, ground, or other landscape elements

---

## Key Learnings

* Recursive Algorithms: Perfect for tree-like structures and branching patterns
* Turtle Graphics: Advanced techniques beyond basic shapes
* Animation Principles: Gradual growth, smooth transitions, frame control
* Physics Simulation: Gravity, wind, rotation, and collision detection
* Color Theory: Natural color blending and RGB manipulation
* Data Structures: Using dictionaries to track multiple animated objects

---

This project demonstrates that Python's Turtle graphics can create much more than simple drawings. By combining recursive algorithms, color theory, and physics simulation, you can build beautiful, complex animations that bring digital nature to life. The techniques learned here can be applied to game development, data visualization, or interactive art projects.

Try modifying the parameters to create your own unique animated scenes!

---

**Website:** https://www.pyshine.com
**Author:** PyShine
