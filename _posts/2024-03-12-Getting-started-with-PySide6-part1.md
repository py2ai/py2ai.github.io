---
layout: post
title: Building Your First Simple GUI (Part 1)
mathjax: true
featured-img: 26072022-python-logo
description: Learning PySide6 for GUI development
---

Welcome to the world of PySide6! PySide6 is a powerful Python module that allows you to create cross-platform graphical user interfaces (GUIs) with ease. Whether you're a beginner or an experienced developer, PySide6 provides a straightforward way to build interactive applications for desktop and mobile platforms.

In this tutorial series, we'll walk through the basics of PySide6, starting with building a simple GUI application. By the end of this tutorial, you'll have a solid understanding of how PySide6 works and be ready to dive deeper into creating more complex applications.

# Prerequisites

Before we begin, make sure you have Python installed on your system. You can download and install Python from the official Python website: python.org. Additionally, you'll need to install PySide6. You can install it using pip:

`pip install PySide6`

# Setting Up Your Development Environment

Once Python and PySide6 are installed, you're ready to set up your development environment. You can use any text editor or integrated development environment (IDE) of your choice. Some popular options include Visual Studio Code, PyCharm, and Sublime Text.

# Creating Your First PySide6 Application

Let's dive right in and create a simple GUI application using PySide6. In this example, we'll create a window with a button that displays a message when clicked.

## lab1.py
{% include codeHeader.html %}
```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('My First PySide6 App with PyShine')
        self.setGeometry(100, 100, 300, 200)

        self.button = QPushButton('Click Me!', self)
        self.button.setGeometry(100, 100, 100, 50)
        self.button.clicked.connect(self.displayMessage)

    def displayMessage(self):
        QMessageBox.information(self, 'Message', 'Hello, PySide6!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
```

# Understanding the Code

Let's break down the code step by step:

1. We import necessary modules from PySide6 and sys.
2. We define a class called MainWindow that inherits from QWidget, the base class for all UI objects in PySide6.
3. In the `__init__` method, we call the superclass constructor and initialize the UI using the initUI method.
4. The initUI method sets the window title, size, and creates a button with its click event connected to the displayMessage method.
5. The `displayMessage` method shows a message box with the text "Hello, PySide6!" when the button is clicked.
6. Finally, we create an instance of QApplication, create an instance of MainWindow, show it, and start the application event loop.


# Running Your Application

To run the application, save the code to a file (e.g., simple_gui.py) and execute it using Python:

`python lab1.py`

You should see a window with a button labeled "Click Me!". Clicking the button will display a message box with the text "Hello, PySide6!".

Congratulations! You've successfully created your first PySide6 GUI application. In the next part of this tutorial series, we'll explore more advanced features of PySide6 and build upon what we've learned here. Stay tuned for more!







