---
layout: post
title: "Visualizing Pascal's Law with Python and Pygame"
description: "A detailed beginner-to-intermediate tutorial explaining Pascal’s Law using an interactive Pygame hydraulic press simulation."
featured-img: 20260110Hydra/20260110Hydra
keywords:
- Python
- Pygame
- Physics Simulation
- Pascal's Law
- Hydraulic Press
- STEM
---

# Visualizing Pascal's Law with Python and Pygame

This project demonstrates **Pascal's Law**, a fundamental principle of fluid mechanics, using an interactive **Pygame simulation**. By dragging pistons and adjusting areas, users can visually and numerically understand how pressure and force behave in a confined fluid.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/1LPkqUvLulU" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

---

## What Is Pascal’s Law?

**Pascal’s Law** states:

> *Pressure applied to a confined fluid is transmitted equally in all directions throughout the fluid.*

Mathematically:

```
P = F / A
```

Where:
- **P** = Pressure (Pascals, Pa)
- **F** = Force (Newtons, N)
- **A** = Area (square meters, m²)

This principle is the reason **hydraulic presses, car brakes, lifts, and excavators** can multiply force.

---

## What This Simulation Shows

This program visually simulates a **hydraulic system** with:

- A **small input piston** (left)
- A **large output piston** (right)
- A **U-shaped fluid pipe** connecting them

### Key Features

- Real-time fluid visualization
- Mouse-controlled piston movement
- Adjustable output piston area
- Automatic force multiplication
- Live display of SI units (N, m², Pa)

---

## Technologies Used

- **Python 3**
- **Pygame** (graphics + interaction)
- **Physics equations** (real SI units)

---

## Physics Model Used

### Pressure Calculation

Pressure is calculated using fluid depth:

```
P = ρ g h
```

Where:
- **ρ** = fluid density (1000 kg/m³ for water)
- **g** = gravity (9.81 m/s²)
- **h** = depth of piston (meters)

---

### Force Transmission

Because pressure is equal everywhere:

```
F_in / A_small = F_out / A_big
```

So:

```
F_out = F_in × (A_big / A_small)
```

This is why **small force on a small piston creates large force on a large piston**.

---

## Understanding the Code Structure

### 1. Window Setup (Portrait Mode)

```python
W, H = 500, 650
S = pygame.display.set_mode((W, H))
```

A tall window helps visualize vertical piston motion.

---

### 2. SI Units Configuration

```python
A_small = 0.01  # m²
A_big = 0.08    # m²
fluid_density = 1000
g = 9.81
```

All calculations are done using **real physics units**.

---

### 3. Area-to-Width Scaling

```python
big_w = int(base_width * (A_big / A_small) ** 0.5)
```

Why square root?

Because:
```
Area ∝ width²
```

So width must scale with √area to remain visually correct.

---

### 4. Constant Fluid Volume Constraint

To simulate incompressible fluid:

```python
initial_fluid_area = calculate_fluid_area(...)
```

When one piston moves down, the other **must rise** to conserve volume.

---

### 5. Mouse Interaction

- Drag **left piston** → increases pressure
- Drag **right handle** → changes piston area

This lets users **experiment freely**.

---

### Complete code
{% include codeHeader.html %}

```python
"""This demonstrates Pascal's Law:
that pressure applied to a confined fluid is transmitted 
equally throughout the fluid.

Key Features:
- Real-time piston movement and corresponding 
    pressure/force change.
- Adjustable output piston area via mouse drag to
    visualize force multiplication.
- Realistic SI units (Force in N, Area in meter-square., 
    Pressure in Pa).
- Dynamic info panel displaying Pascal's law and live 
    values.
"""

import pygame, sys
from pygame.draw import rect, lines
from pygame import Rect
pygame.init()

# Portrait window setup
W, H = 500, 650
S = pygame.display.set_mode((W, H))
pygame.display.set_caption("Hydraulic Pressure")

font = pygame.font.SysFont("Arial", 16, bold=True)
BLUE = (0, 120, 255)
GRAY = (180, 180, 180)
DARK = (15, 15, 25)
WHITE = (255, 255, 255)
OUTLINE = (100, 100, 100)
HANDLE_COLOR = (255, 200, 50)
ORANGE = (255, 165, 0)  # New color for dragging

# SI Units Setup
# Areas in m^2, 
# Pressure in Pa (Pascals), 
# Force in N (Newtons)
A_small = 0.01 # 0.01 m^2 = 100 cm^2(small piston)
A_big = 0.08          # 0.08 m^2 = 800 cm^2 (big piston)
fluid_density = 1000  # kg/m³ (water density)
g = 9.81              # m/s² (gravity)

# Geometry
left_x, right_x = 150, 350
pipe_top, pipe_bottom = 440, 450
shaft_len = 60
piston_height = 25

# Calculate widths based on areas to 
# maintain visual equality
base_width = 40  # base width for A_small 
piston_width = base_width
# cylinder is 10px wider than piston
cylinder_w = base_width + 10  

# Calculate big piston width to be 
# visually proportional to area
big_w = int(base_width * (A_big / A_small) ** 0.5)  
big_h = 50

piston_y = 300
dragging = False
area_drag = False
offset_y = 0
y_min, y_max = 300, 410

# Calculate initial total blue fluid area 
# (all three regions)
def calculate_fluid_area(left_y, right_y, big_width):
    """Calculate total blue fluid area including
    all three regions"""
    # 1. Left cylinder blue area 
    # (from piston to pipe bottom)
    left_area = 40 * (pipe_bottom - left_y)
    
    # 2. Pipe region blue area 
    # (horizontal U-shaped part)
    pipe_area = 40 * (pipe_bottom - pipe_top)
    
    # 3. Right cylinder blue area 
    # (from piston to pipe bottom)
    right_area = big_width * \
        (pipe_bottom - (right_y + big_h))
    
    return left_area + pipe_area + right_area
def input_pressure(y):
    """Pressure applied by left piston in Pascals"""
    # Convert pixels to meters (100px = 1m)
    depth = max(0, (y - y_min) / 100)  
    return depth * fluid_density * g  # Pressure
def calculate_right_piston_position(left_y, big_width):
    """Calculate right piston position to maintain 
    constant total fluid area"""
    # Current left and pipe areas 
    # (these change with left piston movement)
    current_left_area = 40 * (pipe_bottom - left_y)
    current_pipe_area = 40 * (pipe_bottom - pipe_top)  
    
    # Calculate required right area to maintain 
    # constant total
    required_right_area = initial_fluid_area - \
                          current_left_area - \
                          current_pipe_area
    
    # Calculate right piston position from required 
    # right area
    if big_width > 0:
        right_fluid_height = required_right_area / \
                            big_width
        right_y = pipe_bottom - right_fluid_height -\
                            big_h
    else:
        right_y = initial_right_y
    
    # Ensure the piston stays within bounds
    return max(220, min(400, right_y))

# Initial fluid area calculation
initial_right_y = 400
initial_fluid_area = calculate_fluid_area(
                    piston_y,initial_right_y, big_w)
# Main Loop

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: 
            pygame.quit(); sys.exit()
        if e.type == pygame.MOUSEBUTTONDOWN:
            # Left piston drag
            if (left_x - piston_width//2 < \
                e.pos[0] < left_x + piston_width//2
                and piston_y - piston_height//2 < \
                e.pos[1] < piston_y + piston_height):
                
                dragging, offset_y = True, \
                e.pos[1] - piston_y
            # Right piston area handle drag
            handle_rect = Rect(right_x + 
                          big_w//2 - 5, 350, 10, 50)
            if handle_rect.collidepoint(e.pos):
                area_drag = True
        if e.type == pygame.MOUSEBUTTONUP:
            dragging = area_drag = False
        if e.type == pygame.MOUSEMOTION:
            if dragging:
                piston_y = max(y_min, 
                        min(y_max, e.pos[1]-offset_y))
            if area_drag:
                # Change big piston area and 
                # update width accordingly
                new_big_w = max(40, 
                    min(200, e.pos[0]-right_x+big_w//2))
                A_big = A_small * \
                    (new_big_w / base_width) ** 2  
                big_w = new_big_w

    # --- Physics ---
    P_in = input_pressure(piston_y)  # Pressure Pascals
    P_out = P_in  # Same pressure (Pascal's principle)
    F_in = P_in * A_small  # Force in Newtons
    F_out = P_out * A_big  # Force in Newtons
    
    # Calculate mechanical advantage safely
    mechanical_advantage = F_out / F_in if \
                            F_in != 0 else 0
    
    # Calculate right piston position to maintain 
    # constant total fluid area
    right_y = calculate_right_piston_position(
                            piston_y, big_w)
    
    # Calculate current total fluid area for display
    current_fluid_area = calculate_fluid_area(
                            piston_y, right_y, big_w)

    # --- Drawing ---
    S.fill(DARK)

    # U-shaped pipe - ALL THREE BLUE REGIONS:
    # 1. Horizontal pipe region (constant)
    rect(S, BLUE, (left_x - 20, 
                   pipe_top, 
                   right_x - left_x + 40, 
                   pipe_bottom - pipe_top))
    # 2. Left cylinder blue region (changes with piston)
    rect(S, BLUE, (left_x - 20, 
                   piston_y, 
                   40, 
                   pipe_bottom - piston_y))
    # 3. Right cylinder blue region (changes with piston)
    rect(S, BLUE, (right_x - big_w//2, 
                   right_y + big_h, 
                   big_w, 
                   pipe_bottom - (right_y + big_h)))

    # Left cylinder outline
    cyl_rect = Rect(left_x - cylinder_w//2, 
                           y_min - 10, 
                           cylinder_w, 
                           y_max - y_min + 50)
    rect(S, OUTLINE, cyl_rect, 3)

    # Left piston shaft + T-head - CHANGE COLOR 
    if dragging:
        # Use ORANGE when dragging
        rect(S, ORANGE, 
             (left_x - 5, 
              piston_y - shaft_len, 
              10, 
              shaft_len))
        head = Rect(left_x - piston_width//2, 
                    piston_y, 
                    piston_width, 
                    piston_height)
        top_bar = Rect(left_x - piston_width, 
                       piston_y - 10, 
                       piston_width*2, 
                       10)
        rect(S, ORANGE, head)
        rect(S, ORANGE, top_bar)
    else:
        # Use GRAY when not dragging
        rect(S, GRAY, 
                (left_x - 5,
                piston_y - shaft_len, 
                10, 
                shaft_len))
        head = Rect(left_x - piston_width//2, 
                           piston_y, 
                           piston_width, 
                           piston_height)
        top_bar = Rect(left_x - piston_width, 
                              piston_y - 10, 
                              piston_width*2, 
                              10)
        rect(S, GRAY, head)
        rect(S, GRAY, top_bar)

    # Right piston
    right_rect = Rect(right_x - big_w//2, 
                      right_y, 
                      big_w, 
                      big_h)
    rect(S, GRAY, right_rect)
    rect(S, OUTLINE, right_rect, 2)

    # Handle to drag width/area - 
    # CHANGE COLOR WHEN DRAGGING
    handle_rect = Rect(right_x + big_w//2 - 5, 
                       350, 10, 50)
    if area_drag:
        rect(S, ORANGE, handle_rect)  # Orange dragging
    else:
        rect(S, HANDLE_COLOR, handle_rect)  # Normal color

    # Pipe outline
    lines(S, WHITE, False, [
        (left_x - 20, pipe_top),
        (left_x - 20, pipe_bottom),
        (right_x + big_w//2, pipe_bottom),
        (right_x + big_w//2, pipe_top)
    ], 2)

    # Pascal's Law formula and explanation added to the info panel
    info_lines = [
        "Pascal's Law:",
        "Pressure applied to confined fluid is transmitted equally",
        "Formula: P = F / A (Pressure = Force divided by Area)",
        "",
        f"Input Force: {F_in:.1f} N | Output Force: {F_out:.1f} N",
        f"Input Area: {A_small:.4f} m² | Output Area: {A_big:.4f} m²",
        "From Pascal's law: F_in / A_small = F_out/A_big",
        f"Input Pressure: {F_in:.1f} N / {A_small:.4f} m² = {P_in:.1f} Pa",
        f"Output Pressure: {F_out:.1f} N / {A_big:.4f} m² = {P_out:.1f} Pa"
            ]
    
    for i, line in enumerate(info_lines):
        S.blit(font.render(line, True, WHITE), 
               (30, 20 + i*22))

    pygame.display.flip()
    
```

## Live Information Panel

The top panel displays:

- Pascal’s Law explanation
- Input & output forces
- Input & output areas
- Pressure equality proof

Example:

```
Input Pressure: 981 Pa
Output Pressure: 981 Pa
```

---

## Educational Use Cases

This simulation is excellent for:

- Physics classrooms
- STEM demonstrations
- Engineering intuition
- Interactive learning projects
- Educational games

---

## Real-World Applications of Pascal’s Law

- Hydraulic car jacks
- Excavators
- Aircraft brakes
- Industrial presses
- Power steering systems

---

## Common Questions (FAQ)

### Why doesn’t force stay the same?
Because **area changes**. Pressure stays constant, not force.

---

### Why is the output piston larger?
To demonstrate **force multiplication**.

---

### Why is gravity included?
It converts piston depth into **real pressure values**.

---

### Is this physically accurate?
Yes for **ideal fluids** (incompressible, no losses).

---

## Possible Extensions

- Add pressure gauges
- Animate fluid particles
- Add oil vs water density
- Add real hydraulic ratios
- Export data graphs

---

## Conclusion

This Pygame project transforms Pascal’s Law from a formula into a **hands-on visual experience**. By interacting with pistons and areas, users gain an intuitive understanding of pressure, force, and mechanical advantage.

If you can *see* physics, you can *understand* it.

---

Happy learning!