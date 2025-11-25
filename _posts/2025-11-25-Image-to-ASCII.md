---
description: Learn how to convert any image into animated ASCII art using Python and OpenCV. Perfect for beginners exploring image processing and terminal graphics.
featured-img: 20251126-asciiart/20251126-asciiart
keywords:
- Python
- ASCII
- OpenCV
- terminal animation
- image processing
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- ascii
- opencv
- terminal
- beginner
- tutorial
title: Convert Images to Animated ASCII Art in Python – Beginner's Guide
---
## Introduction

Ever wondered how to turn an image into awesome ASCII art directly inside your terminal?
In this beginner-friendly tutorial, you'll learn how to load an image, convert it to grayscale, map pixels to ASCII characters, and display it line-by-line with a smooth animated effect.

This project is perfect for beginners learning:

- Basic **image processing**
- **OpenCV** fundamentals
- ASCII art generation
- Terminal animation

---

## Requirements

Install the required library:

```bash
pip install opencv-python
```

You also need an image file named:

```
ia.png
```

Place it in the same folder as your script. Of course you can try any other image name but remember to change that name in the code as well.

---

## Understanding the Code

### 1. Importing Libraries

```python
import cv2
import time
import os
```

- **cv2** → Image loading and resizing
- **time** → Adding animation timing
- **os** → Optional file management

---

## 2. ASCII Character Set

```python
ASCII = "@%#*+=-:. "
```

Characters are arranged from **darkest** (`@`) to **lightest** ().
Each grayscale pixel finds its closest match in this set.

---

## 3. Load the Image

```python
img = cv2.imread("ia.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image issue!")
```

Loads the image as grayscale.
If missing → raises an error.

---

## 4. Resize While Keeping Aspect Ratio

```python
h, w = img.shape
new_w = 90
ratio = h / w
new_h = int(new_w * ratio * 0.55)
resized = cv2.resize(img, (new_w, new_h))
```

Why multiply by **0.55**?
Because terminal text cells are **taller** than they are **wide**, so we shrink vertically to maintain the correct proportions.

---

## 5. Convert Pixels to ASCII (With Animation!)

```python
for i, row in enumerate(resized):
    print("".join([ASCII[int(p) * len(ASCII) // 256] for p in row]))
    time.sleep(0.05)
```

- Converts each pixel intensity to a matching ASCII character
- Builds each line as a string
- Prints it with a short delay to create the animation effect

---

## Full Source Code

```python
import cv2
import time
import os

ASCII = "@%#*+=-:. "
img = cv2.imread("ia.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image issue!")

h, w = img.shape
new_w = 90
ratio = h / w
new_h = int(new_w * ratio * 0.55)
resized = cv2.resize(img, (new_w, new_h))

for i, row in enumerate(resized):
    print("".join([ASCII[int(p) * len(ASCII) // 256]
                   for p in row]))
    time.sleep(0.05)
```

---

## Running the Script

Save the script as:

```
ascii_art.py
```

Run it:

```bash
python ascii_art.py
```

Your ASCII animation will begin drawing line-by-line!

---

## Conclusion

You now know how to:

- Load and process images with OpenCV
- Convert image pixels to ASCII brightness levels
- Create cool terminal animations

Try modifying:

- The ASCII character set
- Image size
- Animation speed

Enjoy experimenting, and happy coding!

---

**Website:** https://www.pyshine.com
**Author:** PyShine
