---
description: Beginner-friendly tutorial to create random avatars in Python using python_avatars. Step-by-step guide to generate personalized SVG avatars.
featured-img: 20251121-python-avatars/20251121-python-avatars
keywords:
- Python
- python_avatars
- avatar
- SVG
- beginner
- tutorial
- random avatars
layout: post
mathjax: false
tags:
- python
- avatar
- SVG
- python_avatars
- beginner
- tutorial
title: Python Avatars – Generate Random SVG Avatars for Beginners
---

## Introduction

Creating unique avatars is fun and useful for websites, apps, or games.  
In this tutorial, you'll learn how to generate **random avatars** using Python with the `python_avatars` library.

No prior experience is required — this guide is for **absolute beginners**.

---

## Understanding the Requirements

You will learn how to:

- Install and use `python_avatars`
- Generate a **random avatar**
- Customize features like hair, eyes, mouth, clothing, and colors
- Save your avatar as an **SVG file**

### Prerequisites:

- Python 3 installed
- Basic knowledge of Python (variables, imports, functions)

---

## Setting Up Your Environment

Install the `python_avatars` package:

```bash
pip install python-avatars
```

You also need Python's built-in `random` module, which requires no installation.

---

## Writing the Avatar Script

Create a new Python file, e.g., **`avatar_gen.py`**, and add:

```python
import python_avatars as pa
import random as r

# Create an avatar with random features
avatar = pa.Avatar(
    style=pa.AvatarStyle.CIRCLE,
    **{p: r.choice(list(enum)) for p, enum in {
        "background_color": pa.BackgroundColor,
        "top": pa.HairType,
        "eyes": pa.EyeType,
        "mouth": pa.MouthType,
        "clothing": pa.ClothingType,
        "hair_color": pa.HairColor,
        "clothing_color": pa.ClothingColor
    }.items()},
    skin_color=f"#{r.randint(0,0xFFFFFF):06x}"  # Random skin color
)

# Render the avatar as SVG
avatar.render("avatar.svg")
```

---

## Code Explanation

### 1. Importing Libraries

```python
import python_avatars as pa
import random as r
```

- `python_avatars` → main library to create avatars
- `random` → choose random features

### 2. Creating an Avatar

```python
avatar = pa.Avatar(
    style=pa.AvatarStyle.CIRCLE,
    **{p: r.choice(list(enum)) for p, enum in { ... }.items()},
    skin_color=f"#{r.randint(0,0xFFFFFF):06x}"
)
```

- `style=pa.AvatarStyle.CIRCLE` → gives your avatar a circular shape
- `**{p: r.choice(list(enum)) ...}` → randomly selects a value for each attribute
- `skin_color=f"#{r.randint(0,0xFFFFFF):06x}"` → random hexadecimal color for skin

### 3. Random Features

- `background_color` → random background
- `top` → hair or hat type
- `eyes`, `mouth` → facial features
- `clothing` → shirt or outfit
- `hair_color`, `clothing_color` → random colors

### 4. Rendering the Avatar

```python
avatar.render("avatar.svg")
```

- Saves your generated avatar as `avatar.svg` in the current folder
- You can open it in any browser or SVG viewer

---

## Running the Script

1. Save the file as **`avatar_gen.py`**
2. Open your terminal and navigate to the folder
3. Run:

```bash
python avatar_gen.py
```

4. Check the folder for **`avatar.svg`** and open it in a browser

---

## Customizing Your Avatars

You can change features manually:

```python
avatar = pa.Avatar(
    style=pa.AvatarStyle.CIRCLE,
    top=pa.HairType.LONG,
    eyes=pa.EyeType.HAPPY,
    mouth=pa.MouthType.SMILE,
    background_color=pa.BackgroundColor.BLUE,
    skin_color="#FFCCAA",
    hair_color=pa.HairColor.BLACK,
    clothing=pa.ClothingType.SHIRT,
    clothing_color=pa.ClothingColor.RED
)
```

This allows you to create **unique avatars** instead of fully random ones.

---

## Conclusion

You now know how to:

- Generate random avatars in Python
- Customize avatar features
- Save avatars as SVG files

This is perfect for profile pictures, games, or web projects.  
Experiment with different styles, colors, and shapes to make avatars **truly unique**!

---

**Website:** https://www.pyshine.com  
**Author:** PyShine
