---
description: In this tutorial, we'll delve into the usage of various output widgets in PySide6 to display information to users in graphical user interface (GUI) applicati...
featured-img: 26072022-python-logo
keywords:
- PySide6 output widgets
- Progress bar in PySide6
- Digital segment display PySide6
- QLabel PySide6
- PySide6 GUI tutorial
- Qt widgets
- GUI application design
- Python output widgets
- PySide6 widgets example
layout: post
mathjax: true
tags:
- Python
- PySide6
- GUI Development
- Output Widgets
- Progress Bar
- Digital Segment Display
- Text Label
- Qt
- Programming Tutorial
title: Exploring Output Widgets in PySide6 (Part 6)
---

In this tutorial, we'll delve into the usage of various output widgets in PySide6 to display information to users in graphical user interface (GUI) applications. Output widgets allow us to present data in different formats, such as progress bars, digital segment displays, and text labels. We'll build a simple example application to demonstrate how to incorporate different output widgets in PySide6.

# Prerequisites
Before we begin, ensure you have Python and PySide6 installed on your system. You can install PySide6 using pip:

`pip install PySide6`

## Designing the GUI
Our GUI will consist of a main window with different output widgets, including a progress bar, a digital segment display, and a text label.


## output_widgets_app.py
{% include codeHeader.html %}
```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QProgressBar, QLCDNumber, QLabel

class OutputWidgetsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Output Widgets Demo')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        ## Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(50)
        layout.addWidget(self.progress_bar)

        ## Digital Segment Display
        self.segment_display = QLCDNumber()
        self.segment_display.display(123)
        layout.addWidget(self.segment_display)

        ## Text Label
        self.text_label = QLabel("Hello, PySide6!")
        layout.addWidget(self.text_label)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    outputWidgetsApp = OutputWidgetsApp()
    outputWidgetsApp.show()
    sys.exit(app.exec())
```

## Understanding the Code

* We create a class OutputWidgetsApp inheriting from QWidget.
* In the initUI method, we set up the main window and create a vertical layout.
* We create a progress bar using QProgressBar and set its initial value.
* We create a digital segment display using QLCDNumber and display a numeric value.
* We create a text label using QLabel and set its text.
* We add these output widgets to the layout.

## Running the Application
Save the code to a file (e.g., output_widgets_app.py) and execute it using Python:

`python output_widgets_app.py`

You should see the GUI window with different output widgets, including a progress bar, a digital segment display, and a text label.

## Conclusion

In this tutorial, we explored how to incorporate various output widgets in PySide6 to display information to users in GUI applications. Output widgets allow us to present data in different formats, such as progress bars for visualizing progress, digital segment displays for showing numeric values, and text labels for displaying text. You can further customize and enhance this application by adding more output widgets, incorporating animations, or styling the GUI components to match your design preferences. Experiment with the code and explore the possibilities of PySide6 to create even more informative and visually appealing GUI applications!
