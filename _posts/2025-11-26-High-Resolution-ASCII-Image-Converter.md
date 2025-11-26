---
description: Learn how to convert images to high-resolution ASCII art in Python. Step-by-step beginner-friendly guide to build a pixel-aligned full-color ASCII converter.
featured-img: 20251126-ascii-hd/20251126-ascii-hd
keywords:
- Python
- ASCII art
- image conversion
- OpenCV
- PIL
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- ascii
- image-processing
- beginner
- tutorial
title: High-Resolution ASCII Image Converter in Python – Beginner's Guide
---
## Introduction

Have you ever wanted to turn your favorite images into **high-resolution ASCII art**?
In this tutorial, we’ll build a Python program that converts images into **full-color, pixel-aligned ASCII art**.

This guide is written for **beginners**, and no prior experience with image processing is required.

Example input image:

![ascii_hd_cat]({{"assets/img/posts/cat.png" | absolute_url}} )

## Understanding the Requirements

Before we start coding, let's understand the tools and concepts:

### Requirements:

- **Python 3 installed**
- Basic familiarity with Python functions, lists, and loops
- Libraries:
  - `numpy`
  - `opencv-python`
  - `Pillow` (PIL)

The program will:

- Load an image
- Resize it to match ASCII character grid
- Match each block of pixels to a character
- Recreate the image using ASCII characters with original colors
- Provide character usage statistics

---

## Installing Required Libraries

Run the following command:

```bash
pip install numpy opencv-contrib-python pillow
```

---

## Creating the ASCII Converter

Create a new Python file called **`ascii_matcher_hd.py`** and start coding.

### 1. Import Modules

```python
import os, platform
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageOps
```

- `os` and `platform` → to detect system fonts
- `numpy` → efficient numerical operations
- `cv2` → image processing
- `PIL` → drawing ASCII characters on images

---

### 2. Define ASCII Characters

```python
ASCII = [chr(i) for i in range(32, 127)]
```

- ASCII characters from **space (32)** to **tilde (126)**.
- These will be used to recreate the image.

---

### 3. Load a Monospace Font

```python
def load_monospace_font(size=20):
    system = platform.system()
    mac = ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"]
    win = [r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf"]
    linux = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"]
    search = mac if system == "Darwin" else win if system == "Windows" else linux
    for p in search:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                pass
    print("⚠ Using fallback font")
    return ImageFont.load_default()

font = load_monospace_font(20)
```

- Cross-platform font loading ensures consistent ASCII alignment.
- Uses a fallback if no system font is found.

---

### 4. Get Font Cell Size

```python
def get_font_cell(font):
    bbox = font.getbbox("M")
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])

CELL = get_font_cell(font)
print("Font cell =", CELL)
```

- Each ASCII character is drawn in a **fixed cell size**.
- Important for accurate image reconstruction.

---

### 5. Precompute Glyph Bitmaps

```python
def glyph_bitmap(ch):
    w, h = CELL
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.text((0, 0), ch, font=font, fill=255)
    arr = np.array(im, dtype=np.float32) / 255.0
    return arr

GLYPHS = [(ch, glyph_bitmap(ch)) for ch in ASCII]
```

- Precomputes grayscale representation for each ASCII character.
- Speeds up the matching process.

---

### 6. Enhance Image Contrast

```python
def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply((gray * 255).astype(np.uint8)) / 255.0
    gamma = 1.2
    g = np.power(g, gamma)
    return np.clip(g, 0, 1)
```

- Uses **CLAHE** for low-contrast images.
- Applies a gamma correction for brightness enhancement.

---

### 7. Convert Image to ASCII

```python
def image_to_ascii(img_path, out_path, cols=160, brightness=1.0):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    cw, ch = CELL
    rows = int(cols * (h/w) * (cw/ch))
    target_px = (cols*cw, rows*ch)
    img_small = cv2.resize(img, target_px, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY) / 255.0
    gray = enhance_contrast(gray)
    gray = np.clip(gray * brightness, 0, 1)

    out = Image.new("RGB", target_px, (0,0,0))
    draw = ImageDraw.Draw(out)

    char_usage = {ch: 0 for ch in ASCII}
    total_used = 0

    for r in range(rows):
        for c in range(cols):
            x0 = c*cw
            y0 = r*ch
            patch = gray[y0:y0+ch, x0:x0+cw]

            best = min(GLYPHS, key=lambda g: np.sum(np.abs(patch - g[1])))
            best_char = best[0]

            char_usage[best_char] += 1
            total_used += 1

            color = tuple(int(v) for v in img_small[y0:y0+ch, x0:x0+cw].mean(axis=(0,1)))
            draw.text((x0, y0), best_char, font=font, fill=color)

    out.save(out_path)
    print("Saved:", out_path)

    print("\n=== CHARACTER USAGE REPORT ===")
    print("Total characters used:", total_used)
    for ch, count in sorted(char_usage.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"'{ch}' : {count}")
```

- Resizes the image to match **ASCII grid**.
- Matches each pixel block with the **best-fitting ASCII character**.
- Draws ASCII characters using **average color** from the original image.
- Prints **character usage statistics**.

---

### 8. Example Run

```python
if __name__ == "__main__":
    image_to_ascii("cat.png", "ascii_hd.png", cols=200, brightness=1.6)
```

- Converts `cat.png` into `ascii_hd.png`.
- Adjust `cols` and `brightness` for different resolution and contrast.

---

## Running the Script

1. Save your file as `ascii_matcher_hd.py`.
2. Place an image in the same folder, e.g., `cat.png`.
3. Run the script:

```bash
python ascii_matcher_hd.py
```

## Complete code

{% include codeHeader.html %}

```python

import os, platform
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageOps


# ASCII CHARSET (printables)
ASCII = [chr(i) for i in range(32, 127)]


# LOAD CROSS-PLATFORM MONOSPACE FONT
def load_monospace_font(size=20):
    system = platform.system()

    mac = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf"
    ]
    win = [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf"
    ]
    linux = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
    ]

    search = mac if system == "Darwin" else win if system == "Windows" else linux

    for p in search:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                pass

    print("Using fallback font")
    return ImageFont.load_default()

font = load_monospace_font(20)


# GET FONT CELL SIZE (precise height)
def get_font_cell(font):
    bbox = font.getbbox("M")
    return (bbox[2] - bbox[0], bbox[3] - bbox[1])

CELL = get_font_cell(font)
print("Font cell =", CELL)


# PRECOMPUTE GLYPH BITMAPS
def glyph_bitmap(ch):
    w, h = CELL
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.text((0, 0), ch, font=font, fill=255)
    arr = np.array(im, dtype=np.float32) / 255.0
    return arr

GLYPHS = [(ch, glyph_bitmap(ch)) for ch in ASCII]


# NORMALIZATION + CONTRAST BOOST FUNCTION
def enhance_contrast(gray):
    # CLAHE = best for low contrast images
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    g = clahe.apply((gray * 255).astype(np.uint8)) / 255.0

    # light gamma for more brightness pop
    gamma = 1.2
    g = np.power(g, gamma)

    return np.clip(g, 0, 1)


# MAIN: HI-RES PIXEL-ALIGNED FULL-COLOR ASCII CONVERTER
def image_to_ascii(img_path, out_path, cols=160, brightness=1.0):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    cw, ch = CELL
    rows = int(cols * (h/w) * (cw/ch))

    target_px = (cols*cw, rows*ch)
    img_small = cv2.resize(img, target_px, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY) / 255.0
    gray = enhance_contrast(gray)
    gray = np.clip(gray * brightness, 0, 1)

    out = Image.new("RGB", target_px, (0,0,0))
    draw = ImageDraw.Draw(out)

  
    # CHARACTER USAGE COUNTER
    char_usage = {ch: 0 for ch in ASCII}
    total_used = 0

    for r in range(rows):
        for c in range(cols):

            x0 = c*cw
            y0 = r*ch
            patch = gray[y0:y0+ch, x0:x0+cw]

            # best ASCII match by L1 distance
            best = min(GLYPHS, key=lambda g: np.sum(np.abs(patch - g[1])))
            best_char = best[0]

            char_usage[best_char] += 1
            total_used += 1

            # original mean color
            color = tuple(int(v) for v in img_small[y0:y0+ch, x0:x0+cw].mean(axis=(0,1)))
            draw.text((x0, y0), best_char, font=font, fill=color)

    out.save(out_path)
    print("Saved:", out_path)


    # PRINT CHARACTER STATISTICS
    print("\n=== CHARACTER USAGE REPORT ===")
    print("Total characters used:", total_used)

    for ch, count in sorted(char_usage.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"'{ch}' : {count}")


if __name__ == "__main__":
    image_to_ascii("cat.png", "ascii_hd.png", cols=200, brightness=1.6)

```


Output image via these parameters :

`image_to_ascii("cat.png","ascii_hd.png",cols=400,brightness=1.6)`


![ascii_hd_cat]({{"assets/img/posts/ascii_hd_cat.png" | absolute_url}} )

```bash
python main.py
Font cell = (12, 15)
Saved: ascii_hd.png

=== CHARACTER USAGE REPORT ===
Total characters used: 128000
'N' : 56261
' ' : 28717
'M' : 21530
'U' : 10621
'%' : 6287
'|' : 1990
'W' : 520
'`' : 380
'0' : 284
'[' : 158
'*' : 154
'C' : 107
'R' : 89
'J' : 83
'~' : 81
'@' : 80
'K' : 80
'B' : 75
'w' : 65
'<' : 40
'Z' : 38
'#' : 36
'^' : 34
'D' : 33
'L' : 33
'r' : 32
'>' : 24
'/' : 14
'-' : 13
'm' : 11
'q' : 11
'9' : 10
'!' : 8
'3' : 8
'2' : 7
'H' : 7
'+' : 6
'A' : 6
'a' : 6
'1' : 5
'\' : 5
'd' : 5
'k' : 5
'$' : 4
'4' : 4
'X' : 4
'u' : 4
'6' : 3
'8' : 3
':' : 3
'f' : 3
'e' : 2
'{' : 2
'"' : 1
'(' : 1
')' : 1
'7' : 1
'P' : 1
'Q' : 1
'S' : 1
'o' : 1
'y' : 1
```

---

4. Check the generated **ASCII art image**.

---

## Conclusion

You now have a **high-resolution, full-color ASCII converter** in Python!This tutorial introduces:

- Image resizing and processing with OpenCV
- Using PIL for drawing ASCII characters
- Precomputing glyphs for efficiency
- Pixel-aligned ASCII reconstruction
- Character usage analytics

Experiment with different images, font sizes, columns, and brightness to create stunning ASCII art.

---

**Website:** https://www.pyshine.com
**Author:** PyShine
