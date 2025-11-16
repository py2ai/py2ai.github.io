---
description: Step-by-step beginner-friendly guide to create a live-updating Excel chart using Python's pandas, matplotlib, and watchdog libraries.
featured-img: 20251110-python-matplotlib
keywords:
- Python
- Excel
- pandas
- matplotlib
- watchdog
- automation
- live-update
- data-visualization
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- excel
- pandas
- matplotlib
- watchdog
- automation
- beginner
- data-visualization
- tutorial
title: Real-Time Excel Chart Updater with Python and Watchdog
---

# Real-Time Excel Chart Updater with Python and Watchdog

## Automatically Refresh Charts When Excel Files Change

This tutorial shows you how to build a **real-time Excel chart visualizer** using **Python**, **pandas**, **matplotlib**, and **watchdog**.
The script detects changes in an Excel file and automatically **updates a live chart** without manual refresh — perfect for dashboards, live monitoring, and educational demos.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/ROtFo05RJZk" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

---

## Overview

This project continuously monitors an Excel file named **`garden_data.xlsx`**.
Whenever the file changes, the Python program automatically **reloads it** and **redraws the chart**.

Ideal for:

- Real-time data logging dashboards
- Automatically visualizing IoT or sensor data
- Tracking and plotting financial trends in Excel

---

## How It Works

1. **Watchdog** detects when the Excel file changes.
2. **pandas** loads and parses the Excel data.
3. **matplotlib** renders an interactive live chart.
4. The chart updates instantly — no need to rerun the program.

---

## Setup Instructions

First, install the dependencies:

```bash
pip install pandas matplotlib watchdog numpy openpyxl
```

Next, create a file named **`garden_data.xlsx`** in the same folder.
Example sheet content:

| Plant | Height |
| ----- | ------ |
| Rose  | 30     |
| Tulip | 25     |
| Lily  | 35     |

Now, you’re ready to run the script!

---

## Core Components

### 1. Drawing the Chart

The `draw_chart(ax)` function automatically detects numeric columns from your Excel sheet and plots them.
It also handles errors gracefully if the file is missing or contains no numeric data.

```python
def draw_chart(ax):
    ax.clear()
    try:
        xls = pd.ExcelFile(filename)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error reading file:\n{e}",
                ha='center', va='center', fontsize=10)
        plt.draw()
        return

    sheet_names = xls.sheet_names
    numeric_sheets = []

    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            numeric_sheets.append((sheet, df, numeric_df))

    if not numeric_sheets:
        ax.text(0.5, 0.5, "No numeric data found", 
                ha='center', va='center')
        plt.draw()
        return

    # Use the first numeric sheet found
    sheet, df, numeric_df = numeric_sheets[0]
    x_candidates = df.select_dtypes(
        exclude=[np.number]).columns.tolist()
    y_candidates = numeric_df.columns.tolist()

    ycol = y_candidates[0]
    xcol = x_candidates[0] if x_candidates else ycol

    try:
        ax.bar(df[xcol], df[ycol], color="steelblue")
    except Exception:
        ax.bar(range(len(df[ycol])), df[ycol],
               color="steelblue")
    ax.set_title(f"{sheet} — {ycol} vs {xcol}")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    plt.tight_layout()
    plt.draw()
```

---

### 2. Watching for File Changes

The `Watcher` class listens for file modification events using **watchdog**.
Whenever the Excel file changes, a flag is triggered to refresh the chart.

```python
class Watcher(FileSystemEventHandler):
    def on_modified(self, event):
        global needs_update
        if event.src_path.endswith(filename):
            needs_update = True
```

---

### 3. Main Loop

The main loop runs continuously with `plt.ion()` (interactive mode).
It checks for updates and redraws the chart automatically.

```python
if __name__ == "__main__":
    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 3))
    event_handler = Watcher()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print(f"Watching '{filename}'... (Ctrl+C to stop)")

    try:
        while True:
            if needs_update:
                print("Excel refreshing chart...")
                draw_chart(ax)
                needs_update = False
            plt.pause(0.01)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

---

## Complete Code

Here’s the entire script in one block:

{% include codeHeader.html %}

```python
import pandas as pd
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np

filename = "garden_data.xlsx"
needs_update = True

def draw_chart(ax):
    ax.clear()
    try:
        xls = pd.ExcelFile(filename)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error reading file:\n{e}",
                ha='center', va='center', fontsize=10)
        plt.draw()
        return

    sheet_names = xls.sheet_names
    numeric_sheets = []

    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            numeric_sheets.append((sheet, df, numeric_df))

    if not numeric_sheets:
        ax.text(0.5, 0.5, "No numeric data found", 
                ha='center', va='center')
        plt.draw()
        return

    # Use the first numeric sheet found
    sheet, df, numeric_df = numeric_sheets[0]
    x_candidates = df.select_dtypes(
        exclude=[np.number]).columns.tolist()
    y_candidates = numeric_df.columns.tolist()

    ycol = y_candidates[0]
    xcol = x_candidates[0] if x_candidates else ycol

    try:
        ax.bar(df[xcol], df[ycol], color="steelblue")
    except Exception:
        ax.bar(range(len(df[ycol])), df[ycol],
               color="steelblue")
    ax.set_title(f"{sheet} — {ycol} vs {xcol}")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    plt.tight_layout()
    plt.draw()


class Watcher(FileSystemEventHandler):
    def on_modified(self, event):
        global needs_update
        if event.src_path.endswith(filename):
            needs_update = True


if __name__ == "__main__":
    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 3))
    event_handler = Watcher()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    print(f"Watching '{filename}'... (Ctrl+C to stop)")

    try:
        while True:
            if needs_update:
                print("Excel refreshing chart...")
                draw_chart(ax)
                needs_update = False
            plt.pause(0.01)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

---

## How to Run

1. Save the above script as **`excel_chart_watcher.py`**
2. Place **`garden_data.xlsx`** in the same directory
3. Run:

```bash
python excel_chart_watcher.py
```

Make any edits to the Excel file — and watch your chart refresh instantly! ⚡

---

## Key Learnings

- Real-time file watching with **watchdog**
- Automated Excel parsing with **pandas**
- Smart detection of numeric vs non-numeric columns
- Live visualization using **matplotlib**

---

## Further Ideas

- Display multiple charts (for each sheet)
- Add filtering or smoothing for better clarity
- Save charts as images automatically
- Integrate with **Tkinter** or **Streamlit** for dashboards

---

**You’ve built a live Excel chart updater in Python!**
Perfect for automation, teaching, and data monitoring projects.

---

**Website:** https://www.pyshine.com
**Author:** PyShine
