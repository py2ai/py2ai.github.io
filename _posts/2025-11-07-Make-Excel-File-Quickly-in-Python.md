---
description: Learn how to create Excel workbooks with multiple sheets using Python's openpyxl library. Step-by-step tutorial for absolute beginners.
featured-img: 20251109-excelpython-openpyxl
keywords:
- Python
- Excel
- openpyxl
- spreadsheet
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- excel
- openpyxl
- beginner
- data-processing
- tutorial
title: Create Excel Files with Python – Complete Beginner's Guide
---


# Create Excel Files with Python – Complete Beginner's Guide

## Learn How to Automate Excel File Creation with Python's openpyxl Library

Excel spreadsheets are everywhere in the business world, but did you know you can create them automatically using Python? This beginner-friendly tutorial will teach you how to generate Excel files with multiple sheets using the **openpyxl** library. You'll learn everything from installation to creating organized data sheets – perfect for automating reports or organizing data!

---

## Table of Contents
- [What We're Building](#what-were-building)
- [Why Use Python for Excel?](#why-use-python-for-excel)
- [Step 0: Installing openpyxl](#step-0-installing-openpyxl)
- [Step 1: Importing the Library](#step-1-importing-the-library)
- [Step 2: Creating Your First Workbook](#step-2-creating-your-first-workbook)
- [Step 3: Understanding Sheets and Active Sheet](#step-3-understanding-sheets-and-active-sheet)
- [Step 4: Naming Your First Sheet](#step-4-naming-your-first-sheet)
- [Step 5: Adding Data to Cells](#step-5-adding-data-to-cells)
- [Step 6: Adding Multiple Rows with append()](#step-6-adding-multiple-rows-with-append)
- [Step 7: Creating a Second Sheet](#step-7-creating-a-second-sheet)
- [Step 8: Organizing Related Data](#step-8-organizing-related-data)
- [Step 9: Saving Your Excel File](#step-9-saving-your-excel-file)
- [Complete Code](#complete-code)
- [Understanding the Output](#understanding-the-output)
- [Common Mistakes to Avoid](#common-mistakes-to-avoid)
- [Next Steps](#next-steps)

---

## What We're Building

In this tutorial, we'll create an Excel file called **"garden_data.xlsx"** that contains two sheets:

1. **"Trees" sheet**: Lists different types of trees and their heights
2. **"Fruits" sheet**: Shows fruits and their colors

This demonstrates how to organize related but separate data in different tabs within the same Excel file.

---

## Why Use Python for Excel?

- **Automate repetitive tasks**: Generate weekly/monthly reports automatically
- **Handle large datasets**: Process thousands of rows efficiently
- **Integrate with other systems**: Pull data from databases or APIs into Excel
- **Reduce errors**: Eliminate manual copy-paste mistakes
- **Schedule tasks**: Run Python scripts to update Excel files automatically

---

## Step 0: Installing openpyxl

Before we start coding, we need to install the openpyxl library. Open your command prompt or terminal and type:

```bash
pip install openpyxl
```
### What is pip?

pip is Python's package manager that helps install external libraries

It comes automatically with Python (if you installed Python from python.org)

### Verifying Installation:
After installation, you can verify it worked by opening Python and typing:

```
import openpyxl
```

If no error appears, you're ready to go!

---

## Step 1: Importing the Library

```
from openpyxl import Workbook
```

Let's break this down:

* `from openpyxl`: We're taking something from the openpyxl library
* `import Workbook`: We're specifically importing the Workbook class

### What's a Workbook?

* In Excel terms, a Workbook is the entire Excel file (the .xlsx file)
* It can contain multiple Sheets (the tabs you see at the bottom)
* The Workbook class is like a blueprint for creating Excel files

---

## Step 2: Creating Your First Workbook

```python
## Create a new Excel workbook
workbook = Workbook()
filename = "garden_data.xlsx"
```

### Understanding this code:

* `workbook = Workbook()`: Creates a new, empty Excel workbook in memory
* Think of this as opening Excel and creating a "New Blank Workbook"
* `filename = "garden_data.xlsx"`: Sets the name for our output file

### Important Notes:

* The workbook exists only in computer memory until we save it
* We can name our file anything, but .xlsx is the standard Excel extension

---

## Step 3: Understanding Sheets and Active Sheet

```
## The default workbook starts with one sheet
sheet1 = workbook.active
```

### What is a Sheet?

* Sheets are the individual tabs in an Excel file
* By default, every new Excel workbook starts with one sheet (usually named "Sheet1")

### What does `.active` mean?

* `workbook.active` gets the currently active (selected) sheet
* Since we just created the workbook, there's only one sheet, so it's automatically active

---

## Step 4: Naming Your First Sheet

```python
sheet1.title = "Trees"
```

### Why rename sheets?

* Default names like "Sheet1" aren't descriptive
* Good sheet names help people understand what data they contain
* In our case, "Trees" clearly indicates this sheet contains tree data

### Sheet Name Rules:

* Maximum 31 characters
* Cannot contain: `: \ / ? * [ ]`
* Cannot be blank

---

## Step 5: Adding Data to Cells

```python
## Add data to the Trees sheet
sheet1["A1"] = "Tree Name"   # First Column
sheet1["B1"] = "Height (m)"  # Second Column
```

### Understanding Excel Cell References:

* `A1` means Column A, Row 1
* `B1` means Column B, Row 1
* This is the same coordinate system you use in Excel

### What we're creating:

```
| A        | B          |
|----------|------------|
| Tree Name| Height (m) |
```

### Why add headers?

* Headers describe what each column contains
* Makes your data understandable to others
* Follows good data organization practices

---

## Step 6: Adding Multiple Rows with append()

```python
sheet1.append(["Mango", 10])   # First Row
sheet1.append(["Apple", 6])    # Second Row
sheet1.append(["Banana", 5])   # Third Row
```

### How append() works:

* `append()` adds a new row at the bottom of your data
* Each item in the list becomes a cell in that row
* `["Mango", 10]` becomes: Cell A2 = "Mango", Cell B2 = 10

### Visualizing the data:

```
| A         | B          |
|-----------|------------|
| Tree Name | Height (m) |
| Mango     | 10         |
| Apple     | 6          |
| Banana    | 5          |
```

### Why use append() instead of cell references?

* Faster for adding multiple rows
* Less error-prone than manually tracking row numbers
* More readable code

---

## Step 7: Creating a Second Sheet

```python
## Create another sheet for Fruits
sheet2 = workbook.create_sheet(title="Fruits")
```

### Multiple Sheets Organization:

* Different types of data belong in different sheets

* "Trees" sheet: Information about trees

* "Fruits" sheet: Information about fruits

* This keeps related data together and separate from unrelated data

### What happens behind the scenes:

* Python creates a new sheet tab in the workbook
* The new sheet becomes the active sheet temporarily
* We now have two sheets: "Trees" and "Fruits"

---

## Step 8: Organizing Related Data

```python
sheet2["A1"] = "Fruit Name"
sheet2["B1"] = "Color"
sheet2.append(["Mango", "Yellow"])
sheet2.append(["Apple", "Red"])
sheet2.append(["Banana", "Yellow"])
```

### Creating Consistent Structure:

* Both sheets have similar layouts (headers in row 1, data below)
* This makes the Excel file easy to navigate and understand
* Notice how "Mango" appears in both sheets but with different information

## The Fruits sheet looks like:

```
| A         | B      |
|-----------|--------|
| Fruit Name| Color  |
| Mango     | Yellow |
| Apple     | Red    |
| Banana    | Yellow |
```

---

## Step 9: Saving Your Excel File

```python
## Save the workbook
workbook.save(filename)

print(f"{filename} created, Trees and Fruits!")
```

## Why do we need to save?

* Until now, everything existed only in computer memory
* `workbook.save()` writes the data to an actual file on your computer
* Without saving, your work would be lost when the program ends

## File Location:

* The file saves in the same folder where your Python script is running
* You can specify a path: `workbook.save("C:/Users/Name/Documents/garden_data.xlsx")`

### The print() statement:

* Provides confirmation that the script ran successfully
* `f"{filename}"` inserts the filename into the message
* Output: `garden_data.xlsx created, Trees and Fruits!`

---

## Complete Code

```python
## simple_excel_two_sheets.py
from openpyxl import Workbook

## Create a new Excel workbook
workbook = Workbook()
filename = "garden_data.xlsx"
## The default workbook starts with one sheet
sheet1 = workbook.active
sheet1.title = "Trees"

## Add data to the Trees sheet
sheet1["A1"] = "Tree Name"   # First Column
sheet1["B1"] = "Height (m)"  # Second Column
sheet1.append(["Mango", 10]) # First Row
sheet1.append(["Apple", 6])  # Second Row
sheet1.append(["Banana", 5]) # Third Row

## Create another sheet for Fruits
sheet2 = workbook.create_sheet(title="Fruits")
sheet2["A1"] = "Fruit Name"
sheet2["B1"] = "Color"
sheet2.append(["Mango", "Yellow"])
sheet2.append(["Apple", "Red"])
sheet2.append(["Banana", "Yellow"])

## Save the workbook
workbook.save(filename)

print(f"{filename} created, Trees and Fruits!")
```

---

## Understanding the Output

After running the code, you'll find "garden_data.xlsx" in your project folder. When you open it:

### Sheet 1 - "Trees":

* Shows three types of trees with their heights
* Organized with clear headers
* Ready for further analysis or reporting

### Sheet 2 - "Fruits":

* Shows the same plants but focused on fruit characteristics
* Demonstrates how different aspects of related data can be separated

### File Properties:

* Format: Standard Excel (.xlsx) format
* Compatibility: Can be opened in Excel, Google Sheets, LibreOffice
* Structure: Professional-looking with proper headers and organization

---

## Common Mistakes to Avoid

1. Forgetting to save: Your work won't be preserved without workbook.save()
2. Incorrect cell references:
    * Wrong: `sheet1["1A"]` (number first)
    * Right: `sheet1["A1"]` (letter first)

3. Missing installation: Forgetting to run `pip install openpyxl` first
4. File permission issues: Trying to save to protected folders
5. Overwriting files: Saving with same filename overwrites existing files

---

## Next Steps

Now that you can create basic Excel files, try these enhancements:

### Add More Data:

```python
## Add more tree data
sheet1.append(["Oak", 15])
sheet1.append(["Pine", 20])

```

### Format Cells:

```python
from openpyxl.styles import Font

## Make headers bold
sheet1["A1"].font = Font(bold=True)
sheet1["B1"].font = Font(bold=True)

```

### Add Formulas:

```python
## Calculate average height
sheet1["A5"] = "Average Height:"
sheet1["B5"] = "=AVERAGE(B2:B4)"
```

### Read Existing Files:

```python
from openpyxl import load_workbook

## Open the file we just created
workbook = load_workbook("garden_data.xlsx")
sheet = workbook.active
print(sheet["A2"].value)  # Prints "Mango"
```

---

Congratulations! You've just automated Excel file creation using Python. This fundamental skill opens doors to automated reporting, data processing, and much more. The simple garden data example demonstrates principles you can apply to business reports, scientific data, or any structured information that needs organizing.

*Try modifying the code to create your own custom Excel files – the possibilities are endless!*


