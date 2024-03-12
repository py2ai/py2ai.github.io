---
layout: post
title: Interacting with Output Widgets Using Input Widgets in PySide6 (Part 7)
mathjax: true
featured-img: 26072022-python-logo
summary:  Control the Progress bar with Slider widget and more
---


In this tutorial, we'll explore how to interact with output widgets by providing input through input widgets in PySide6. We'll build a simple example application where input widgets such as sliders and spin boxes are used to provide input, which will be displayed by output widgets like progress bars and digital segment displays.

# Prerequisites

Before we begin, ensure you have Python and PySide6 installed on your system. You can install PySide6 using pip:

`pip install PySide6`

# Designing the GUI

Our GUI will consist of a main window with input widgets like sliders and spin boxes, and output widgets like progress bars and digital segment displays to display the input values.

## input_output_widgets_app.py
{% include codeHeader.html %}
```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QSlider, QSpinBox, QProgressBar, QLCDNumber
from PySide6.QtCore import Qt

class InputOutputWidgetsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Input-Output Widgets Demo')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setTickInterval(10)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.updateOutputWidgets)
        layout.addWidget(self.slider)

        # Spin Box
        self.spin_box = QSpinBox()
        self.spin_box.setMinimum(0)
        self.spin_box.setMaximum(100)
        self.spin_box.setValue(50)
        self.spin_box.valueChanged.connect(self.updateOutputWidgets)
        layout.addWidget(self.spin_box)

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Digital Segment Display
        self.segment_display = QLCDNumber()
        layout.addWidget(self.segment_display)

        self.setLayout(layout)

        # Update output widgets initially
        self.updateOutputWidgets()

    def updateOutputWidgets(self):
        # Update progress bar
        self.progress_bar.setValue(self.slider.value())

        # Update digital segment display
        self.segment_display.display(self.spin_box.value())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    inputOutputWidgetsApp = InputOutputWidgetsApp()
    inputOutputWidgetsApp.show()
    sys.exit(app.exec())
```

# Understanding the Code

* We create a class InputOutputWidgetsApp inheriting from QWidget.
* We set up input widgets such as a slider and a spin box, and output widgets such as a progress bar and a digital segment display.
* We connect the valueChanged signals of the slider and the spin box to a common updateOutputWidgets method, which updates the output widgets whenever the input values change.
* The updateOutputWidgets method updates the value of the progress bar with the slider value and updates the value displayed on the digital segment display with the spin box value.

# Running the Application
Save the code to a file (e.g., input_output_widgets_app.py) and execute it using Python:

`python input_output_widgets_app.py`

You should see the GUI window with input widgets (slider and spin box) and output widgets (progress bar and digital segment display). Adjusting the values of the input widgets should dynamically update the output widgets accordingly.

# Conclusion

In this tutorial, we learned how to interact with output widgets by providing input through input widgets in PySide6. By connecting the signals of input widgets to appropriate methods, we can dynamically update the output widgets based on user input. This interaction allows us to create more dynamic and responsive GUI applications. Experiment with the code and explore the possibilities of PySide6 to create even more interactive and user-friendly applications!
