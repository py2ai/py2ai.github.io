---
layout: post
title: Building a Calculator Application with PySide6 (Part 2)
mathjax: true
featured-img: 26072022-python-logo
summary:  Beyond the Basics
---


Welcome back to our PySide6 tutorial series! In the previous tutorial, we explored the fundamentals of PySide6 by creating a simple Todo List application. Now, let's take our PySide6 skills to the next level by building a calculator application. A calculator is a classic example of a GUI application that involves handling user input and performing calculations.

# Prerequisites

Before we dive into building the calculator application, make sure you have Python and PySide6 installed on your system. If you haven't done this yet, please refer to the instructions in Part 1 of this tutorial series.

# Designing the Calculator Application

Our calculator application will consist of a main window with a display area to show the input and output of calculations, and buttons for numeric input, arithmetic operations, and special functions like clearing the display. Let's start by designing the layout:

## calculartor.py

{% include codeHeader.html %}
```python
import sys
from functools import partial
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton, QLineEdit

class CalculatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Calculator')
        self.setGeometry(100, 100, 300, 400)

        layout = QVBoxLayout()

        self.display = QLineEdit()
        self.display.setReadOnly(True)
        layout.addWidget(self.display)

        buttonsLayout = QGridLayout()

        buttons = [
            '7', '8', '9', '/',
            '4', '5', '6', '*',
            '1', '2', '3', '-',
            'C', '0', '=', '+'
        ]

        row = 0
        col = 0

        for button in buttons:
            btn = QPushButton(button)
            if button == '=':
                btn.clicked.connect(self.calculate)
            elif button == 'C':
                btn.clicked.connect(self.clearDisplay)
            else:
                btn.clicked.connect(partial(self.appendToDisplay, button))

            buttonsLayout.addWidget(btn, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1

        layout.addLayout(buttonsLayout)
        self.setLayout(layout)

    def appendToDisplay(self, char):
        self.display.setText(self.display.text() + char)

    def calculate(self):
        try:
            result = eval(self.display.text())
            self.display.setText(str(result))
        except Exception as e:
            self.display.setText("Error")

    def clearDisplay(self):
        self.display.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    calculatorApp = CalculatorApp()
    calculatorApp.show()
    sys.exit(app.exec())

```

# Understanding the Code

* We create a class CalculatorApp inheriting from QWidget.
* In the initUI method, we set up the calculator's layout, including the display area and buttons for numeric input, arithmetic operations, clearing, and calculation.
* Numeric buttons and arithmetic operation buttons are connected to their respective functions using the clicked.connect() method.
* The appendToDisplay method appends characters to the display.
* The calculate method evaluates the expression in the display and displays the result.
* The clearDisplay method clears the display.
* Running the Calculator Application
* Save the code to a file (e.g., calculator.py) and execute it using Python:

`python calculator.py`

You should see the calculator application window with the numeric input buttons, arithmetic operation buttons, and the display area. You can now start performing calculations using your PySide6 calculator!

# Conclusion

In this tutorial, we built a simple calculator application using PySide6. We designed the UI layout and added functionality to handle user input and perform calculations. You can further customize and enhance this application by adding more features such as support for parentheses, memory functions, or scientific operations. Experiment with the code and explore the possibilities of PySide6 to create even more advanced GUI applications!








