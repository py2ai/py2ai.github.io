---
description: "<div style=\"text-align:center;\">\n  <iframe width=\"320\" height=\"568\" \n      src=\"https://www.youtube.com/embed/-LxLf4hZzHY?rel=0&autoplay=0\" \n      title=\"You..."
featured-img: 26072022-python-logo
keywords:
- Vpython
- GPS Satellite Simulation
layout: post
mathjax: true
tags:
- gps satellites
- vpython tutorial
- gps satellites basic tutorial
title: How to make a GPS Satellite Simulation in Python
---
For more details:

# Check out our YouTube Short!

<div style="text-align:center;">
  <iframe width="320" height="568" 
      src="https://www.youtube.com/embed/-LxLf4hZzHY?rel=0&autoplay=0" 
      title="YouTube Short" frameborder="0" 
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
      allowfullscreen>
  </iframe>
</div>

---

## 3D Visualization of a GPS Satellite Constellation

**A detailed, step-by-step tutorial and annotated markdown blog** that explains a VPython script which renders a 3D visualization of the GPS satellite constellation (24 operational satellites shown across 6 orbital planes). This tutorial covers setup, math, per-line code explanation, optimization tips, debugging, and suggested extensions.

## Overview

The script creates a **3D animation of GPS satellites** orbiting Earth using VPython. It models the **real GPS constellation** with six orbital planes, each containing four satellites, at an altitude of about 20,180 km. The satellites orbit with a 55° inclination and leave colorful trails to show their paths.

## Requirements

To run this visualization, you need:

- Python 3.8+
- The `vpython` library

Install it via pip:

```bash
pip install vpython
```

> VPython runs in a browser or a standalone window depending on your setup.

## Full Source Code

{% include codeHeader.html %}

```python
from vpython import sphere, vector, rate, color, \
    textures, label, canvas, cylinder, mag
import math

## Scene setup
scene = canvas(
    title = "GPS Satellites:24 | Orbital Planes: 6",
    width = 400,
    height = 600,
    background = color.black,
    center = vector(0,0,0)
)

## Earth and GPS parameters
EARTH_RADIUS = 6371e3 # 6,371 km
GPS_ALTITUDE = 20180e3 # 20,180 km
GPS_RADIUS = EARTH_RADIUS + GPS_ALTITUDE
SCALE = 1e-7
DISP_R = EARTH_RADIUS * SCALE
DISPLAY_GPS_RADIUS = GPS_RADIUS * SCALE

## Orbital parameters
GPS_ORBITAL_PERIOD = 43080 # seconds (11h 58m)
SPEED_FACTOR = 3000 # Time scale multiplier for visuals
TIME_STEP = 1/60
omega_gps = 2 * math.pi / GPS_ORBITAL_PERIOD * SPEED_FACTOR

## Create Earth
earth = sphere(radius = DISP_R,
               texture=textures.earth,
               shininess=0.1)

## Create GPS satellites
num_planes = 6
sats_per_plane = 4
total_satellites = num_planes * sats_per_plane
satellites = []
sat_labels = []
connection_lines = []

## Colors for each plane
plane_colors = [color.red, color.orange, color.yellow,
                color.green, color.cyan, color.magenta]

inclination = math.radians(55)
satellite_count = 0

for plane in range(num_planes):
    raan = 2 * math.pi * plane / num_planes  # RAAN evenly spaced
    for sat_in_plane in range(sats_per_plane):
        true_anomaly = 2 * math.pi * sat_in_plane / sats_per_plane
      
        sat = sphere(
            radius = DISP_R * 0.1,
            color = color.white,
            make_trail = True,
            trail_type = "curve",
            trail_radius = DISP_R * 0.025,
            trail_color = plane_colors[plane],
            retain=2000
        )

        sat_label = label(
            pos = vector(0,0,0),
            text = f"GPS {satellite_count+1}",
            color = plane_colors[plane],
            box=False,
            line=False,
            height=10
        )

        connection_line = cylinder(
            radius = DISP_R * 0.005,
            color = plane_colors[plane],
            opacity = 0.6
        )

        sat.raan = raan
        sat.true_anomaly = true_anomaly
        sat.plane = plane
        satellites.append(sat)
        sat_labels.append(sat_label)
        connection_lines.append(connection_line)
        satellite_count += 1

print(f"Created {len(satellites)} GPS satellites in {num_planes} orbital planes")

## Simulation loop
while True:
    rate(60)
    earth.rotate(angle=0.001, axis=vector(0,1,0))

    for i, sat in enumerate(satellites):
        sat.true_anomaly += omega_gps * TIME_STEP
        r = DISPLAY_GPS_RADIUS

        ## Position in orbital plane
        x_orb = r * math.cos(sat.true_anomaly)
        y_orb = 0
        z_orb = r * math.sin(sat.true_anomaly)

        ## Apply inclination rotation
        x_incl = x_orb
        y_incl = y_orb * math.cos(inclination) - z_orb * math.sin(inclination)
        z_incl = y_orb * math.sin(inclination) + z_orb * math.cos(inclination)

        ## Apply RAAN rotation
        x_final = x_incl * math.cos(sat.raan) - y_incl * math.sin(sat.raan)
        y_final = x_incl * math.sin(sat.raan) + y_incl * math.cos(sat.raan)
        z_final = z_incl

        sat.pos = vector(x_final, y_final, z_final)
        sat_labels[i].pos = sat.pos + vector(0, DISP_R * 0.5 , 0)

        if mag(sat.pos) > 0:
            sat_dir = sat.pos / mag(sat.pos)
            ground_point = sat_dir * DISP_R
            connection_lines[i].pos = ground_point
            connection_lines[i].axis = sat.pos - ground_point

        if sat.true_anomaly >= 2 * math.pi:
            sat.true_anomaly -= 2 * math.pi
            sat.clear_trail()
```

## Detailed Explanation

### Scene and Scale

We use a `canvas` as our 3D scene and apply scaling to shrink Earth's real radius (6,371 km) and orbit altitude (20,180 km) to manageable VPython units using `SCALE = 1e-7`. This keeps visual proportions correct while fitting in the viewport.

### Orbital Parameters

The GPS system uses:

- **6 orbital planes** separated by 60° in RAAN.
- **4 satellites per plane**, spaced 90° apart in true anomaly.
- **Inclination:** 55° relative to Earth's equator.
- **Orbital Period:** ~12 hours.

To make orbits visible, we scale the angular speed by `SPEED_FACTOR = 3000` so satellites complete revolutions faster.

### Satellite Initialization

Each satellite is created as a small white `sphere` with a colored trail. The color corresponds to its orbital plane. We also attach a `label` above each satellite and a `cylinder` line connecting it to Earth's surface.

The nested loop structure:

- Outer loop (`plane`) sets the plane’s RAAN.
- Inner loop (`sat_in_plane`) places satellites evenly spaced in that plane.

We store each satellite’s RAAN, true anomaly, and plane index for later updates.

### Orbital Motion

Each frame (at 60 FPS):

1. Increment the true anomaly based on angular velocity.
2. Compute the position in the orbital plane.
3. Apply two rotations:
   - **Inclination rotation** around the x-axis.
   - **RAAN rotation** around the z-axis.
4. Update the satellite’s position.
5. Update its label and connection line.
6. Reset the trail after completing a full orbit.

### Earth Rotation

The Earth sphere is rotated slowly with:

```python
earth.rotate(angle=0.001, axis=vector(0,1,0))
```

This simulates Earth’s spin and adds realism.

## Mathematical Summary

| Step | Rotation Axis | Formula                       | Purpose              |
| ---- | ------------- | ----------------------------- | -------------------- |
| 1    | X-axis        | y' = y*cos(i) - z*sin(i)    | Inclination rotation |
| 2    | Z-axis        | x'' = x'*cos(Ω) - y'*sin(Ω) | RAAN rotation        |

`Ω` (RAAN) shifts the orbital plane around Earth’s z-axis. `i` (inclination) tilts the plane.

## Performance Tips

- Reduce `retain` in trails to <1000 to save memory.
- Use `rate(30)` if animation is slow.
- Decrease number of satellites for testing.
- Avoid frequent creation/deletion of VPython objects inside the loop.

## Enhancements You Can Try

1. **Add real-time controls:** buttons for pause/resume, orbit speed, and trail toggle.
2. **Add real GPS data:** parse TLE files for actual satellite positions.
3. **Add ground stations:** show visible satellites from a chosen location.
4. **Add camera animation:** orbit camera around Earth.

## Summary

This VPython project demonstrates how to visualize satellite constellations in 3D. You learned:

- How to model orbital planes and inclination.
- How to use VPython primitives (`sphere`, `cylinder`, `label`).
- How to simulate orbital motion using angular updates.

It’s a powerful educational example for both orbital mechanics and real-time 3D visualization in Python.

---

**Website:** https://www.pyshine.com
**Author:** PyShine