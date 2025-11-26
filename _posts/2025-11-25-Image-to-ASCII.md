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
pip install opencv-contrib-python
```

You also need an image file named:

```
ia.png
```

Place it in the same folder as your script. Of course you can try any other image name but remember to change that name in the code as well.

---

## Understanding the Code

## 1. Importing Libraries

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
{% include codeHeader.html %}
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

## Full Source Code (Colored version)
Try the following for colorful version:

{% include codeHeader.html %}
```python
import cv2, time

ASCII = " .:-=+*#%@"
img = cv2.imread("ia.png")
if img is None: raise FileNotFoundError()
h, w = img.shape[:2]
new_w = 80
new_h = int(new_w * (h / w) * 0.5)
img = cv2.resize(img, (new_w, new_h))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for gy, row in zip(gray, img):
    line = "".join(
        f"\033[38;2;{r};{g};{b}m{ASCII[p*9//255]}\033[0m"
        for p, (b, g, r) in zip(gy, row)
    )
    print(line)
    time.sleep(0.01)

```

## Breaking Down the Code for COLORFUL ASCII ART

1. Importing Libraries

```python
import cv2, time
```
- cv2 → OpenCV library for image processing
- time → To add a small delay for animated printing

2. ASCII Characters

```python
ASCII = " .:-=+*#%@"
```

Each character represents a different brightness level

" " is the lightest, "@" is the darkest

3. Loading the Image

```python
img = cv2.imread("ia.png")
if img is None: 
    raise FileNotFoundError()
```

- Reads the image file
- Throws an error if the image is missing

4. Resizing the Image

```python
h, w = img.shape[:2]
new_w = 80
new_h = int(new_w * (h / w) * 0.5)
img = cv2.resize(img, (new_w, new_h))
```

- Maintains aspect ratio
- Scales the width to 80 characters
- 0.5 factor compensates for character height vs width

5. Converting to Grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

- Converts image to grayscale
- Each pixel now represents brightness (0–255)

6. Mapping Pixels to ASCII Characters with Color

```python
for gy, row in zip(gray, img):
    line = "".join(
        f"\033[38;2;{r};{g};{b}m{ASCII[p*9//255]}\033[0m"
        for p, (b, g, r) in zip(gy, row)
    )
    print(line)
    time.sleep(0.01)
```

- Loops over each row of pixels
- Maps brightness to ASCII character
- Colors characters using ANSI escape codes:

```
\033[38;2;R;G;Bm
```

- Sets RGB foreground color
- `\033[0m` resets formatting
- `time.sleep(0.01)` adds a small delay to create a smooth printing effect

## Conclusion

You now know how to:

- Load and process images with OpenCV
- Convert image pixels to ASCII brightness levels
- Create cool terminal animations

Try modifying:

- The ASCII character set
- Image size
- Animation speed

Tips:

- Increase new_w for higher-resolution ASCII art.
- Use a terminal with truecolor support for best results.
- You can experiment with different ASCII character sets for different artistic styles.

Enjoy experimenting, and happy coding!

---

**Website:** https://www.pyshine.com
**Author:** PyShine
