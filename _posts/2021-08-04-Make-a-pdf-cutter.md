---
categories:
- Python tutorial series
- Tools
description: "Split and cut PDF files into individual pages in Python using PyPDF2. Extract specific pages, divide large PDFs, and automate PDF manipulation with this step-by-step tutorial."
featured-img: 2021-12-24-pdf-paper
keywords:
- python split pdf
- pypdf2 tutorial
- cut pdf pages python
- pdf manipulation python
- extract pdf pages
- python pdf cutter
- divide pdf file
- pypdf2 split
layout: post
mathjax: true
tags:
- Python
- PDF
- PyPDF2
- Tutorial
- Script
- PDF Tools
title: How to Split a PDF into Pages in Python with PyPDF2
---


Hi friends, let's say in a folder you have a pdf file which has 6 pages, and now you want to cut that pdf file because you are interested only in the 2nd page.
So, for this all we need is to install pypdf using `pip3 install pypdf2`, and then in the same folder run the python script below:

# cut.py
{% include codeHeader.html %}
```python
from PyPDF2 import PdfFileWriter, PdfFileReader
import argparse
parser = argparse.ArgumentParser()


parser.add_argument("-pdf",
                
                    required=True,
                    help="input pdf file location",
                    )

args = parser.parse_args()

inputpdf = PdfFileReader(open(args.pdf, "rb"))
for i in range(inputpdf.numPages):
    output = PdfFileWriter()
    output.addPage(inputpdf.getPage(i))
    with open("document-page%s.pdf" % i, "wb") as outputStream:
        output.write(outputStream)

print(f"PDF named {args.pdf} splitted into {i+1} pages")
```

In the terminal, we need to pass -pdf argument to get the location of the pdf file, in this case our `a.pdf` file is in the same folder as `cut.py`.

## Usage:

```
python3.6 cut.py -pdf a.pdf
```

Output:

```
PDF named a.pdf splitted into 6 pages
```
