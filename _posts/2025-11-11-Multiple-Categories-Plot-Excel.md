---
description: Learn how to create a live-updating Excel chart in Python using Pandas, Matplotlib, and Watchdog. Step-by-step beginner-friendly guide for real-time data vis...
featured-img: 20251111-python-multimatplotlib
keywords:
- Python
- Excel
- Pandas
- Matplotlib
- Watchdog
- data visualization
- live chart
- real-time update
- automation
- beginner tutorial
layout: post
mathjax: false
tags:
- python
- excel
- pandas
- matplotlib
- watchdog
- data-visualization
- automation
- beginner
- tutorial
title: Garden Data Live Chart Tutorial with Python
---


# Garden Data Live Chart in Python

## Beginner-Friendly Tutorial – Live-Refreshing Excel Charts

This tutorial walks you through creating a **live-updating chart viewer** for Excel sheets using **Python**, **Pandas**, **Matplotlib**, and **Watchdog**. By the end, you will have a tool that:
- Reads multiple sheets from an Excel file.
- Displays **grouped bar charts** with proper labels.
- Automatically refreshes when the Excel file is updated.
- Allows switching between sheets using clickable buttons.

<div class="video-container">
  <iframe 
    width="560" 
    height="315" 
    src="https://www.youtube.com/embed/gxV5FiEzLFE" 
    title="YouTube video player" 
    frameborder="0" 
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
    allowfullscreen>
  </iframe>
</div>

--- 

## Table of Contents
- [Overview](#overview)
- [Setting Up Your Environment](#setting-up-your-environment)
- [Reading Excel Sheets](#reading-excel-sheets)
- [Drawing Charts with Matplotlib](#drawing-charts-with-matplotlib)
- [Adding Sheet Selection Buttons](#adding-sheet-selection-buttons)
- [Watching for File Changes](#watching-for-file-changes)
- [Putting It All Together](#putting-it-all-together)
- [Complete Code](#complete-code)
- [Running the Script](#running-the-script)
- [Key Learnings](#key-learnings)
- [Further Improvements](#further-improvements)

## Overview

We are building a **live chart viewer** for an Excel file named `garden_data.xlsx`. Each sheet in the Excel file can be visualized as a **grouped bar chart**. The program automatically detects changes in the Excel file and updates the chart, making it useful for monitoring live data.

Key features:
- Live-refreshing grouped bar charts.
- Automatic detection of Excel sheet changes.
- Multiple numeric columns per chart.
- Interactive sheet selection buttons.
- Beginner-friendly structure and code clarity.

## Setting Up Your Environment

Install the required packages if you haven't already:

```bash
pip install pandas matplotlib watchdog numpy
```

Import the necessary libraries in your Python script:

```python
import pandas as pd
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
from matplotlib.widgets import Button
```

## Reading Excel Sheets

Define the Excel file to monitor and helper variables:

```python
filename = "garden_data.xlsx"
needs_update = True
current_sheet = None
sheet_buttons = []
```

Function to get all sheet names:

```python
def get_sheet_names():
    try:
        xls = pd.ExcelFile(filename)
        return xls.sheet_names
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
```

This function uses **Pandas** to safely read the sheet names, handling errors if the file is missing or corrupted.

## Drawing Charts with Matplotlib

This function handles drawing **grouped bar charts** for a selected sheet:

```python
def draw_chart(ax, sheet_name=None):
    ax.clear()
    try:
        xls = pd.ExcelFile(filename)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error reading file:\n{e}", ha='center', va='center', fontsize=10)
        plt.draw()
        return

    sheet_names = xls.sheet_names
    if sheet_name is None or sheet_name not in sheet_names:
        if sheet_names:
            sheet_name = sheet_names[0]
        else:
            ax.text(0.5, 0.5, "No sheets found", ha='center', va='center')
            plt.draw()
            return

    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.empty:
            ax.text(0.5, 0.5, f"No data in '{sheet_name}'", ha='center', va='center')
            plt.draw()
            return

        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not categorical_cols or not numeric_cols:
            ax.text(0.5, 0.5, f"Need both categorical and numeric columns in '{sheet_name}'", ha='center', va='center')
            plt.draw()
            return

        xcol = categorical_cols[0]
        categories = df[xcol].astype(str).tolist()
        n_numeric = len(numeric_cols)
        n_categories = len(categories)
        colors = plt.cm.Set3(np.linspace(0, 1, n_numeric))
        bar_width = 0.8 / n_numeric
        x_pos = np.arange(n_categories)

        bars = []
        for i, (numeric_col, color) in enumerate(zip(numeric_cols, colors)):
            bar_positions = x_pos + i * bar_width - (0.8 - bar_width) / 2
            bar = ax.bar(bar_positions, df[numeric_col], width=bar_width, color=color, alpha=0.8, label=numeric_col)
            bars.append(bar)
            for j, value in enumerate(df[numeric_col]):
                if not pd.isna(value):
                    ax.text(bar_positions[j], value + (max(df[numeric_col]) * 0.01), f'{value:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(f"{sheet_name}", fontsize=12, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=30, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if n_numeric > 1:
            ax.legend(loc='best', ncol=min(1, n_numeric), fontsize=9, framealpha=0.9)

        ax.set_xlabel(xcol, fontsize=10, labelpad=10)
        ax.xaxis.set_label_coords(1.0, -0.15)
        y_label = numeric_cols[0] if n_numeric == 1 else "Values"
        ax.set_ylabel(y_label, fontsize=10, labelpad=10)
        ax.yaxis.set_label_coords(-0.1, 1.0)

        y_max = max([df[col].max() for col in numeric_cols if not df[col].isna().all()])
        ax.set_ylim(0, y_max * 1.15)

    except Exception as e:
        ax.text(0.5, 0.5, f"Error processing sheet '{sheet_name}':\n{e}", ha='center', va='center', fontsize=10)
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")

    plt.draw()
```

This function:
- Handles empty sheets and missing numeric/categorical columns.
- Creates **grouped bars** if multiple numeric columns exist.
- Adds **value labels** on top of each bar.
- Styles the chart with **grid, spines, and rotated labels**.
- Adjusts y-axis to fit the tallest bar.

## Adding Sheet Selection Buttons

```python
def on_sheet_button_clicked(sheet_name):
    global current_sheet, needs_update
    current_sheet = sheet_name
    needs_update = True
    for btn, s_name in sheet_buttons:
        if s_name == sheet_name:
            btn.color = '#4CAF50'
            btn.hovercolor = '#45a049'
            btn.label.set_color('black')
            btn.label.set_fontweight('bold')
        else:
            btn.color = '#f8f9fa'
            btn.hovercolor = '#e9ecef'
            btn.label.set_color('black')
            btn.label.set_fontweight('normal')
        btn.ax.figure.canvas.draw_idle()


def create_sheet_buttons(sheet_names, fig):
    global sheet_buttons, current_sheet
    for btn, _ in sheet_buttons:
        btn.ax.remove()
    sheet_buttons.clear()
    if not sheet_names: return
    n_sheets = len(sheet_names)
    max_buttons_per_row = 4
    button_height = 0.06
    button_spacing = 0.01
    rows = (n_sheets + max_buttons_per_row - 1) // max_buttons_per_row
    buttons_per_row = min(n_sheets, max_buttons_per_row)
    button_width = (0.8 - (buttons_per_row - 1) * button_spacing) / buttons_per_row
    start_y = 0.15 - (rows - 1) * (button_height + 0.02)

    for i, sheet_name in enumerate(sheet_names):
        row = i // buttons_per_row
        col = i % buttons_per_row
        x_pos = 0.1 + col * (button_width + button_spacing)
        y_pos = start_y - row * (button_height + 0.02)
        btn_ax = plt.axes([x_pos, y_pos, button_width, button_height])
        btn = Button(btn_ax, sheet_name, color='#f8f9fa', hovercolor='#e9ecef')
        btn.label.set_fontsize(8)
        btn.label.set_fontweight('normal')
        btn.label.set_color('black')
        for spine in btn_ax.spines.values():
            spine.set_color('#dee2e6')
            spine.set_linewidth(1)
        btn.on_clicked(lambda event, sn=sheet_name: on_sheet_button_clicked(sn))
        sheet_buttons.append((btn, sheet_name))
    if sheet_names:
        current_sheet = sheet_names[0]
        on_sheet_button_clicked(current_sheet)
```

## Watching for File Changes

```python
class Watcher(FileSystemEventHandler):
    def on_modified(self, event):
        global needs_update
        if event.src_path.endswith(filename):
            needs_update = True
```

This uses **Watchdog** to detect when the Excel file is modified and triggers a refresh of the chart.

## Putting It All Together

Initialize the figure, axes, buttons, and observer:

```python
plt.ion()
fig = plt.figure(figsize=(5, 7.5))
ax = plt.axes([0.15, 0.4, 0.7, 0.5])
ax_bg = plt.axes([0.05, 0.05, 0.9, 0.25])
ax_bg.set_facecolor('#f1f3f4')
## ... add labels and style
sheet_names = get_sheet_names()
if sheet_names:
    create_sheet_buttons(sheet_names, fig)
draw_chart(ax, current_sheet)

observer = Observer()
event_handler = Watcher()
observer.schedule(event_handler, ".", recursive=False)
observer.start()

try:
    while True:
        if needs_update:
            draw_chart(ax, current_sheet)
            new_sheet_names = get_sheet_names()
            if set(new_sheet_names) != set(sheet_names):
                sheet_names = new_sheet_names
                create_sheet_buttons(sheet_names, fig)
            needs_update = False
        plt.pause(0.01)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

## Complete Code
{% include codeHeader.html %}
```python
## Source code at www.pyshine.com
import pandas as pd
import matplotlib.pyplot as plt
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
from matplotlib.widgets import Button

filename = "garden_data.xlsx"
needs_update = True
current_sheet = None
sheet_buttons = []

def get_sheet_names():
    """Get all sheet names from the Excel file"""
    try:
        xls = pd.ExcelFile(filename)
        return xls.sheet_names
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def draw_chart(ax, sheet_name=None):
    ax.clear()
    try:
        xls = pd.ExcelFile(filename)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error reading file:\n{e}", 
                ha='center', va='center', fontsize=10)
        plt.draw()
        return

    sheet_names = xls.sheet_names
    
    if sheet_name is None or sheet_name not in sheet_names:
        if sheet_names:
            sheet_name = sheet_names[0]
        else:
            ax.text(0.5, 0.5, "No sheets found", 
                    ha='center', va='center')
            plt.draw()
            return
    
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Check if dataframe is empty
        if df.empty:
            ax.text(0.5, 0.5, f"No data in '{sheet_name}'", 
                    ha='center', va='center')
            plt.draw()
            return
            
        # Identify categorical and numeric columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not categorical_cols or not numeric_cols:
            ax.text(0.5, 0.5, f"Need both categorical and numeric columns in '{sheet_name}'", 
                    ha='center', va='center')
            plt.draw()
            return
        
        # Use first categorical column as x-axis
        xcol = categorical_cols[0]
        categories = df[xcol].astype(str).tolist()
        
        # Create grouped bar chart for multiple numeric columns
        n_numeric = len(numeric_cols)
        n_categories = len(categories)
        
        # Set up colors for different numeric columns
        colors = plt.cm.Set3(np.linspace(0, 1, n_numeric))
        
        # Calculate bar positions
        bar_width = 0.8 / n_numeric
        x_pos = np.arange(n_categories)
        
        # Create bars for each numeric column
        bars = []
        for i, (numeric_col, color) in enumerate(zip(numeric_cols, colors)):
            bar_positions = x_pos + i * bar_width - (0.8 - bar_width) / 2
            bar = ax.bar(bar_positions, df[numeric_col], width=bar_width, 
                        color=color, alpha=0.8, label=numeric_col)
            bars.append(bar)
            
            # Add value labels on top of bars
            for j, value in enumerate(df[numeric_col]):
                if not pd.isna(value):
                    ax.text(bar_positions[j], value + (max(df[numeric_col]) * 0.01), 
                           f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Customize the chart
        ax.set_title(f"{sheet_name}", fontsize=12, fontweight='bold', pad=20)  # Increased pad for legend space
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=30, ha='right')
        
        # Style the chart
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend for numeric columns at top middle
        if n_numeric > 1:
            # Place legend at top center, above the plot
            ax.legend(loc='best', 
                     ncol=min(1, n_numeric), fontsize=9, framealpha=0.9)
        
        # Position x-axis label at the end of the x-axis (right side)
        ax.set_xlabel(xcol, fontsize=10, labelpad=10)
        ax.xaxis.set_label_coords(1.0, -0.15)  # Right end, slightly below
        
        # Add y-axis label for numeric values
        if n_numeric == 1:
            y_label = numeric_cols[0]
        else:
            y_label = "Values"
        ax.set_ylabel(y_label, fontsize=10, labelpad=10)
        ax.yaxis.set_label_coords(-0.1, 1.0)  # Left side, at top
        
        # Adjust y-axis limits to accommodate value labels
        y_max = max([df[col].max() for col in numeric_cols if not df[col].isna().all()])
        ax.set_ylim(0, y_max * 1.15)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error processing sheet '{sheet_name}':\n{e}",
                ha='center', va='center', fontsize=10)
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")
    
    plt.draw()

def on_sheet_button_clicked(sheet_name):
    """Callback when a sheet button is clicked"""
    global current_sheet, needs_update
    current_sheet = sheet_name
    needs_update = True
    
    # Update button colors to show active state with black text
    for btn, s_name in sheet_buttons:
        if s_name == sheet_name:
            # Active button - colored background with black text
            btn.color = '#4CAF50'  # Green background for active
            btn.hovercolor = '#45a049'
            btn.label.set_color('black')  # Black text
            btn.label.set_fontweight('bold')
        else:
            # Inactive button - light background with black text
            btn.color = '#f8f9fa'  # Light gray background
            btn.hovercolor = '#e9ecef'
            btn.label.set_color('black')  # Black text
            btn.label.set_fontweight('normal')
        
        # Force immediate redraw of the button
        btn.ax.figure.canvas.draw_idle()

def create_sheet_buttons(sheet_names, fig):
    """Create buttons for sheet selection at the bottom"""
    global sheet_buttons, current_sheet
    
    # Clear existing buttons
    for btn, _ in sheet_buttons:
        btn.ax.remove()
    sheet_buttons.clear()
    
    if not sheet_names:
        return
    
    # Calculate button dimensions based on number of sheets
    n_sheets = len(sheet_names)
    max_buttons_per_row = 4
    button_height = 0.06
    button_spacing = 0.01
    
    if n_sheets <= max_buttons_per_row:
        rows = 1
        buttons_per_row = n_sheets
        button_width = (0.8 - (buttons_per_row - 1) * button_spacing) / buttons_per_row
        start_y = 0.15
    else:
        rows = (n_sheets + max_buttons_per_row - 1) // max_buttons_per_row
        buttons_per_row = min(n_sheets, max_buttons_per_row)
        button_width = (0.8 - (buttons_per_row - 1) * button_spacing) / buttons_per_row
        start_y = 0.15 - (rows - 1) * (button_height + 0.02)
    
    # Create buttons in a grid
    for i, sheet_name in enumerate(sheet_names):
        row = i // buttons_per_row
        col = i % buttons_per_row
        
        x_pos = 0.1 + col * (button_width + button_spacing)
        y_pos = start_y - row * (button_height + 0.02)
        
        btn_ax = plt.axes([x_pos, y_pos, button_width, button_height])
        
        # Create button with initial inactive style
        btn = Button(btn_ax, sheet_name, color='#f8f9fa', hovercolor='#e9ecef')
        
        # Style the button - set text color to black
        btn.label.set_fontsize(8)
        btn.label.set_fontweight('normal')
        btn.label.set_color('black')  # Always black text
        
        # Add border to all buttons
        for spine in btn_ax.spines.values():
            spine.set_color('#dee2e6')
            spine.set_linewidth(1)
        
        # Use lambda to capture the current sheet_name
        btn.on_clicked(lambda event, sn=sheet_name: on_sheet_button_clicked(sn))
        
        sheet_buttons.append((btn, sheet_name))
    
    # Set initial active button
    if sheet_names:
        current_sheet = sheet_names[0]
        on_sheet_button_clicked(current_sheet)

class Watcher(FileSystemEventHandler):
    def on_modified(self, event):
        global needs_update
        if event.src_path.endswith(filename):
            needs_update = True

if __name__ == "__main__":
    plt.ion()
    # You can use any figure size now - legend will stay inside
    fig = plt.figure(figsize=(5, 7.5))  # Smaller figure size as requested
    
    # Create main axes for chart - adjusted for top legend
    ax = plt.axes([0.15, 0.4, 0.7, 0.5])  # Centered with margins
    
    # Get initial sheet names
    sheet_names = get_sheet_names()
    
    # Create button container background at bottom
    ax_bg = plt.axes([0.05, 0.05, 0.9, 0.25])  # Adjusted height
    ax_bg.set_facecolor('#f1f3f4')
    ax_bg.spines['top'].set_visible(True)
    ax_bg.spines['top'].set_color('#dadce0')
    ax_bg.spines['right'].set_visible(False)
    ax_bg.spines['bottom'].set_visible(False)
    ax_bg.spines['left'].set_visible(False)
    ax_bg.tick_params(which='both', bottom=False, left=False, 
                     labelbottom=False, labelleft=False)
    
    # Add title for button section
    ax_bg.text(0.5, 0.9, 'SHEET SELECTOR', transform=ax_bg.transAxes,
               ha='center', va='center', fontsize=11, fontweight='bold',
               color='black')
    
    # Add instruction text
    ax_bg.text(0.5, 0.82, 'Click any sheet to view its data', transform=ax_bg.transAxes,
               ha='center', va='center', fontsize=9, color='#666666')
    
    if sheet_names:
        create_sheet_buttons(sheet_names, fig)
    
    # Style the main figure
    fig.patch.set_facecolor('white')
    
    # Draw initial chart
    if sheet_names:
        draw_chart(ax, current_sheet)
    
    event_handler = Watcher()
    observer = Observer()
    observer.schedule(event_handler, ".", recursive=False)
    observer.start()
    
    print(f"Watching '{filename}'... (Ctrl+C to stop)")
    print("Click sheet buttons at the bottom to view different data")

    try:
        while True:
            if needs_update:
                print(f"Refreshing chart for sheet: {current_sheet}")
                draw_chart(ax, current_sheet)
                
                # Update buttons if sheet list changed
                new_sheet_names = get_sheet_names()
                if set(new_sheet_names) != set(sheet_names):
                    sheet_names = new_sheet_names
                    create_sheet_buttons(sheet_names, fig)
                    if sheet_names and current_sheet not in sheet_names:
                        current_sheet = sheet_names[0]
                        on_sheet_button_clicked(current_sheet)
                
                needs_update = False
            plt.pause(0.01)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

## Running the Script

1. Ensure your Excel file `garden_data.xlsx` exists.
2. Save the script as `garden_live_chart.py`.
3. Run:

```bash
python garden_live_chart.py
```
4. Click sheet buttons to switch sheets.
5. Modify the Excel file while the script is running to see live updates.

## Key Learnings

- Reading Excel sheets dynamically with Pandas.
- Handling multiple numeric columns in grouped bar charts.
- Creating interactive Matplotlib buttons.
- Detecting file changes using Watchdog.
- Combining all components in a live-updating visualization.

## Further Improvements

- Add support for multiple categorical columns.
- Enhance chart styling (colors, fonts, themes).
- Export charts to images automatically.
- Support larger datasets with scrolling buttons.
- Add filters for numeric ranges or categories.

---

This tutorial provides a **complete beginner-friendly guide** to creating **interactive, live-refreshing Excel visualizations** in Python.
---

**Website:** https://www.pyshine.com
**Author:** PyShine

