---
description: Step-by-step beginner-friendly tutorial to create Excel workbooks with multiple sheets using Python's pandas library.
featured-img: 20251109-excelpython
keywords:
- Python
- Excel
- pandas
- spreadsheet
- beginner
- tutorial
layout: post
mathjax: false
tags:
- python
- excel
- pandas
- beginner
- data-processing
- tutorial
title: Create Excel Files with Python and pandas
---
# Table of Contents

1. [Introduction](#introduction)
2. [Understanding the Requirements](#understanding-the-requirements)
3. [Setting Up Your Environment](#setting-up-your-environment)
4. [Breaking Down the Code](#breaking-down-the-code)
   1. [Importing pandas](#importing-pandas)
   2. [Preparing Data](#preparing-data)
   3. [Writing to Excel](#writing-to-excel)
5. [Running the Script](#running-the-script)
6. [Conclusion](#conclusion)

---

## Introduction

In this tutorial, we will learn how to create an Excel workbook with multiple sheets using Python's **pandas** library. This guide is written for absolute beginners and explains every line of the code.

---

## Understanding the Requirements

Before we start, you need:

- Python 3 installed
- pandas library installed (`pip install pandas openpyxl`)
- A terminal or IDE to run the Python script

We will create an Excel file named `garden_data.xlsx` with two sheets:

1. **Trees** – shows the quantity of Mango and Apple trees.
2. **Fruits** – shows the color of Mango and Apple fruits.

---

## Setting Up Your Environment

1. Install pandas and openpyxl:

```bash
pip install pandas openpyxl
```

2. Open your preferred editor and create a new file named `create_excel.py`.

---

## Breaking Down the Code

### Importing pandas

{% include codeHeader.html %}

```python
import pandas as pd
```

- `pandas` is a powerful library for data analysis and manipulation in Python.
- We use it here to create DataFrames and export them to Excel.

### Preparing Data

{% include codeHeader.html %}

```python
filename = "garden_data.xlsx"

sheets_data = {
    "Trees": {"Mango": 10, "Apple": 6},
    "Fruits": {"Mango": "Yellow", "Apple": "Red"}
}
```

- `filename` is the name of the Excel file that will be generated.
- `sheets_data` is a dictionary where:
  - Each key is the sheet name.
  - Each value is another dictionary containing the data for that sheet.

### Writing to Excel

{% include codeHeader.html %}

```python
with pd.ExcelWriter(filename) as writer:
    for sheet, data in sheets_data.items():
        pd.DataFrame(list(data.items()),
                     columns=["Name", "Value"])
          .to_excel(writer, sheet_name=sheet, index=False)
```

- `pd.ExcelWriter(filename)` opens a new Excel file to write multiple sheets.
- `for sheet, data in sheets_data.items()` loops through each sheet and its data.
- `pd.DataFrame(list(data.items()), columns=["Name", "Value"])` converts the dictionary into a DataFrame suitable for Excel.
- `.to_excel(writer, sheet_name=sheet, index=False)` writes the DataFrame to the Excel sheet without adding the index column.
- Using `with` ensures that the file is properly saved and closed after writing.

### Confirming File Generation

```python
print(f"{filename} Generated!")
```

- Prints a message to confirm that the Excel file has been created successfully.

---

## Running the Script

1. Save the file as `create_excel.py`.
2. Run it in your terminal or IDE:

```bash
python create_excel.py
```

3. Check your project folder for `garden_data.xlsx`.
4. Open the Excel file to see the two sheets with the correct data.

---

## Conclusion

You have successfully created an Excel workbook with multiple sheets using Python's pandas library. This beginner-friendly tutorial demonstrated:

- How to structure data for multiple sheets
- How to convert dictionaries to pandas DataFrames
- How to write multiple sheets to a single Excel file

Feel free to expand this example by adding more sheets or more complex data. Pandas makes working with Excel files both simple and powerful!

---

**Website:** https://www.pyshine.com
**Author:** PyShine
