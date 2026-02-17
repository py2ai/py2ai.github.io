---
layout: post
title: "Spring-Mass System Simulation with Pygame - Hooke's Law Physics"
description: "Learn to create an interactive spring-mass system simulation using Pygame. Understand Hooke's Law, physics parameters, and real-time visualization with mouse interaction."
featured-img: 2026-spring-mass/2026-spring-mass
keywords:
- pygame spring simulation
- hooke's law
- physics simulation
- python pygame
- spring-mass system
- interactive physics
- damping simulation
- python tutorial
- physics programming
- game development
categories:
- Python tutorial series
tags:
- Python
- Pygame
- Physics
- Simulation
- Hooke's Law
- Interactive
- Tutorial
mathjax: true
---

# Spring-Mass System Simulation with Pygame - Hooke's Law Physics

Create an interactive physics simulation demonstrating Hooke's Law using Pygame! This tutorial will guide you through building a spring-mass system that responds to mouse interaction, showing real-time physics with damping effects.

## 📐 Understanding Hooke's Law

Hooke's Law describes the relationship between the force exerted by a spring and its displacement from equilibrium:

$$F = -kx$$

Where:
- **F** = Force exerted by the spring (Newtons)
- **k** = Spring constant (N/m) - stiffness of the spring
- **x** = Displacement from equilibrium position (meters)

The negative sign indicates that the force is always opposite to the displacement, creating a restoring force that pulls the mass back to equilibrium.

## 🎯 Physics Parameters Explained

### Spring Constant (k)
- **Value**: 0.04 in our simulation
- **Meaning**: Higher values = stiffer spring
- **Effect**: Faster oscillation frequency
- **Formula**: $T = 2\pi\sqrt{m/k}$ (period of oscillation)

### Mass (m)
- **Value**: 1.0 kg in our simulation
- **Meaning**: Mass of the attached object
- **Effect**: Heavier mass = slower oscillation
- **Formula**: $a = F/m$ (acceleration)

### Damping Factor
- **Value**: 0.01 in our simulation
- **Meaning**: Energy loss over time
- **Effect**: Causes oscillations to decay
- **Formula**: $F_{damping} = -cv$ where c is damping coefficient

## 🚀 Complete Code Implementation

### Prerequisites

Install required packages:

```bash
pip install pygame
```

### Full Source Code

{% include codeHeader.html %}
```python
import pygame
import sys

# Initialize Pygame
pygame.init()

# Window setup
WIDTH, HEIGHT = 450, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Spring-Mass System")

clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

# Physics parameters
k = 0.04         # Spring constant
mass = 1          # Mass
damping = 0.01    # Damping factor

anchor_x = WIDTH // 2
anchor_y = 50
rest_length = 200

# Initial state
position = rest_length
velocity = 0
dragging = False

def draw_spring(surface, x1, y1, x2, y2, coils=15):
    """Draw a coiled spring between two points."""
    points = []
    for i in range(coils + 1):
        t = i / coils
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if i not in (0, coils):
            x += (-1)**i * 10
        points.append((x, y))
    pygame.draw.lines(surface,
                  (200, 100, 0),
                  False, points, 2)

while True:
    dt = clock.tick(60) / 1000  # Convert to seconds

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            dragging = True

        if event.type == pygame.MOUSEBUTTONUP:
            dragging = False

        if event.type == pygame.MOUSEMOTION and dragging:
            mouse_y = pygame.mouse.get_pos()[1]
            position = max(50, mouse_y - anchor_y)
            velocity = 0

    # Physics update (if not dragging)
    if not dragging:
        displacement = position - rest_length
        force = -k * displacement
        acceleration = force / mass
        velocity += acceleration
        velocity *= (1 - damping)
        position += velocity

    screen.fill((20, 20, 30))

    mass_y = anchor_y + position

    # Draw anchor
    pygame.draw.circle(screen, (0, 255, 0),
                   (anchor_x, anchor_y), 5)

    # Draw spring
    draw_spring(screen, anchor_x, anchor_y,
                anchor_x, mass_y)

    # Draw mass
    pygame.draw.rect(screen, (0, 150, 255),
                 (anchor_x - 25, mass_y, 50, 40))

    # Info text
    info = [
        "Hooke's Law: F = -k x",
        f"k = {k}",
        f"Displacement = {position - rest_length:.2f}",
        f"Velocity = {velocity:.2f}"
    ]

    for i, line in enumerate(info):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (20, 20 + i * 25))

    pygame.display.flip()
```

## 📊 Code Breakdown

### 1. Initialization

```python
pygame.init()
WIDTH, HEIGHT = 450, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)
```

- **pygame.init()**: Initialize all Pygame modules
- **set_mode()**: Create the display window
- **Clock()**: Control frame rate and timing
- **SysFont()**: Load system font for text rendering

### 2. Physics Parameters

```python
k = 0.04         # Spring constant (stiffness)
mass = 1          # Mass of the object
damping = 0.01    # Energy loss factor
rest_length = 200   # Equilibrium position
```

These parameters control the physics behavior:
- **k**: Higher values make the spring stiffer
- **mass**: Affects acceleration ($a = F/m$)
- **damping**: Causes oscillations to decay over time
- **rest_length**: Position where spring force is zero

### 3. Spring Drawing Function

```python
def draw_spring(surface, x1, y1, x2, y2, coils=15):
    points = []
    for i in range(coils + 1):
        t = i / coils
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if i not in (0, coils):
            x += (-1)**i * 10
        points.append((x, y))
    pygame.draw.lines(surface, (200, 100, 0), False, points, 2)
```

This creates a realistic coiled spring:
- **Linear interpolation**: Creates straight line between endpoints
- **Oscillating offset**: `(-1)**i * 10` creates zigzag pattern
- **15 coils**: Number of spring coils to draw
- **Color**: (200, 100, 0) = Orange/brown spring

### 4. Event Handling

```python
for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()

    if event.type == pygame.MOUSEBUTTONDOWN:
        dragging = True

    if event.type == pygame.MOUSEBUTTONUP:
        dragging = False

    if event.type == pygame.MOUSEMOTION and dragging:
        mouse_y = pygame.mouse.get_pos()[1]
        position = max(50, mouse_y - anchor_y)
        velocity = 0
```

Interactive features:
- **MOUSEBUTTONDOWN**: Start dragging the mass
- **MOUSEBUTTONUP**: Release the mass
- **MOUSEMOTION**: Update position while dragging
- **max(50, ...)**: Prevent mass from going above anchor

### 5. Physics Simulation

```python
if not dragging:
    displacement = position - rest_length
    force = -k * displacement
    acceleration = force / mass
    velocity += acceleration
    velocity *= (1 - damping)
    position += velocity
```

This implements the physics equations:

1. **Calculate displacement**: Distance from equilibrium
2. **Apply Hooke's Law**: $F = -kx$
3. **Calculate acceleration**: $a = F/m$ (Newton's Second Law)
4. **Update velocity**: $v = v + a \cdot dt$
5. **Apply damping**: $v = v \cdot (1 - c)$ (energy loss)
6. **Update position**: $x = x + v \cdot dt$

### 6. Rendering

```python
screen.fill((20, 20, 30))  # Dark blue background

pygame.draw.circle(screen, (0, 255, 0), (anchor_x, anchor_y), 5)  # Green anchor
draw_spring(screen, anchor_x, anchor_y, anchor_x, mass_y)  # Orange spring
pygame.draw.rect(screen, (0, 150, 255), (anchor_x - 25, mass_y, 50, 40))  # Blue mass
```

Visual elements:
- **Anchor point**: Green circle at top
- **Spring**: Coiled line connecting anchor to mass
- **Mass**: Blue rectangle representing the weight

## 🔬 Physics Theory

### Simple Harmonic Motion

When damping is zero, the system exhibits simple harmonic motion:

$$x(t) = A\cos(\omega t + \phi)$$

Where:
- **A**: Amplitude (maximum displacement)
- **$\omega$**: Angular frequency = $\sqrt{k/m}$
- **t**: Time
- **$\phi$**: Phase constant

### Damped Harmonic Motion

With damping, the motion decays over time:

$$x(t) = Ae^{-\gamma t}\cos(\omega_d t + \phi)$$

Where:
- **$\gamma$**: Damping coefficient
- **$\omega_d$**: Damped frequency = $\sqrt{\omega^2 - \gamma^2}$

### Energy Considerations

**Potential Energy** (stored in spring):
$$PE = \frac{1}{2}kx^2$$

**Kinetic Energy** (motion of mass):
$$KE = \frac{1}{2}mv^2$$

**Total Energy** (without damping):
$$E = PE + KE = \text{constant}$$

Damping removes energy from the system, causing oscillations to decay.

## 🎮 Experimenting with Parameters

### Try Different Spring Constants

```python
# Stiffer spring (faster oscillation)
k = 0.08

# Softer spring (slower oscillation)
k = 0.02
```

### Try Different Masses

```python
# Heavier mass (slower oscillation)
mass = 2

# Lighter mass (faster oscillation)
mass = 0.5
```

### Try Different Damping

```python
# More damping (faster decay)
damping = 0.05

# Less damping (longer oscillation)
damping = 0.001

# No damping (never stops)
damping = 0
```

## 📈 Extending the Simulation

### Add Multiple Springs

```python
# Create a double spring system
spring1_length = 150
spring2_length = 150
mass1_pos = spring1_length
mass2_pos = spring1_length + spring2_length

# Draw both springs
draw_spring(screen, anchor_x, anchor_y, anchor_x, anchor_y + mass1_pos)
draw_spring(screen, anchor_x, anchor_y + mass1_pos, anchor_x, anchor_y + mass2_pos)
```

### Add Gravity

```python
gravity = 9.8  # m/s²

# Update physics with gravity
if not dragging:
    displacement = position - rest_length
    spring_force = -k * displacement
    gravity_force = mass * gravity * 0.1  # Scale for pixels
    total_force = spring_force + gravity_force
    acceleration = total_force / mass
```

### Add Energy Display

```python
# Calculate energies
potential_energy = 0.5 * k * displacement**2
kinetic_energy = 0.5 * mass * velocity**2
total_energy = potential_energy + kinetic_energy

# Display energies
energy_info = [
    f"PE: {potential_energy:.2f}",
    f"KE: {kinetic_energy:.2f}",
    f"Total: {total_energy:.2f}"
]
```

## 🎯 Real-World Applications

Spring-mass systems appear everywhere:

1. **Automotive Suspension**: Car shock absorbers use damped springs
2. **Buildings**: Tuned mass dampers reduce earthquake damage
3. **Clocks**: Pendulum clocks use harmonic motion
4. **Musical Instruments**: Guitar strings vibrate as springs
5. **Seismographs**: Detect earthquakes using spring-mass sensors

## 🐛 Common Issues and Solutions

### Mass Goes Through Anchor

**Problem**: Mass moves above the anchor point

**Solution**: Add constraint in mouse motion handler:
```python
position = max(50, mouse_y - anchor_y)
```

### Oscillations Don't Stop

**Problem**: System oscillates forever

**Solution**: Increase damping:
```python
damping = 0.05  # Higher value
```

### Simulation Too Fast/Slow

**Problem**: Frame rate issues

**Solution**: Adjust clock tick:
```python
dt = clock.tick(60) / 1000  # 60 FPS
```

## 📚 Further Learning

Explore more physics simulations:

- [Pendulum Simulation with Pygame]({{ site.baseurl }}{% post_url 2025-11-12-Tick-Tick-Wall-Clock %})
- [Gravitational Time Dilation Simulation in Python]({{ site.baseurl }}{% post_url 2025-10-30-Time-Runs-Differently-Under-Gravity %})
- [AC to DC Full Wave Rectifier in Pygame]({{ site.baseurl }}{% post_url 2025-10-29-AC-to-DC-Full-Wave-Rectifier-in-Pygame %})
- [How to Make a Tree with Falling Flowers]({{ site.baseurl }}{% post_url 2025-11-04-Make-a-tree-with-Falling-Flowers %})

## Conclusion

This spring-mass simulation demonstrates fundamental physics principles through interactive visualization. By understanding Hooke's Law and implementing it in code, you've created a system that responds naturally to forces and energy loss.

Experiment with different parameters to see how they affect the motion. This simulation provides a foundation for more complex physics projects and real-world applications.

Keep exploring, keep coding, and enjoy the beautiful physics of harmonic motion!

## Related Posts

- [Python Cheatsheet Every Learner Must Know - Save Hours of Time]({{ site.baseurl }}{% post_url 2026-02-17-Python-Cheatsheet %})
- [Top 10 AI Models You Need to Know in 2026 - Complete Guide]({{ site.baseurl }}{% post_url 2026-02-16-Top-10-AI-Models %})
- [How to Make a Zombie Shooter Game in Pygame (Beginner Tutorial)]({{ site.baseurl }}{% post_url 2025-02-07-How-to-make-Zombie-Shooter-game %})
- [Let's Build a Simple "Battleship" Game]({{ site.baseurl }}{% post_url 2024-05-30-Make-a-battleship-game %})
