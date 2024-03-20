---
layout: post
title: Creating a Guess Country from Flag Game in Python (Part 7)
mathjax: true
featured-img: 26072022-python-logo
summary:  Guess a country name
---

In this tutorial, we'll create a graphical user interface (GUI) application using PySide6, a Python binding for the Qt toolkit. Our application will display flags of random countries, and the user will have to guess the name of the country. We'll utilize the flagpy library to fetch flag images and country names. Let's dive into the step-by-step process of building this game.


# Prerequisites

Before starting, ensure you have the following installed:

Python (3.x recommended)
PySide6 library (pip install PySide6)
flagpy library (pip install flagpy)

# Understanding the Code

Let's break down the provided code before we proceed with the tutorial.

* We import necessary modules and libraries.
* We define a class RandomCountryFlagApp that inherits from QMainWindow.
* In the constructor (__init__), we set up the main window, including labels to display flag and country name, and buttons for user interaction.
* We define methods to display a random flag (display_random_flag) and to display the country name (display_country_info).
* Finally, we instantiate the application and the main window and execute the application loop.

## Step 1: Setting Up the GUI Structure

Let's start by creating the basic structure of our GUI application.

{% include codeHeader.html %}
```python
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PySide6.QtGui import QPixmap, QImage, Qt
from flagpy import get_country_list, get_flag_img
import numpy as np
np.random.seed(1)
class RandomCountryFlagApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Guess Flag")
        self.setGeometry(100, 100, 400, 400)

        layout = QVBoxLayout()

        # Label to display country flag
        self.flag_label = QLabel()
        layout.addWidget(self.flag_label)

        # Label to display country name
        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self.name_label)

        # Button to display country name and flag
        self.show_button = QPushButton("Show Name")
        self.show_button.clicked.connect(self.display_country_info)
        layout.addWidget(self.show_button)

        # Button to show next random flag and guess name
        self.next_button = QPushButton("Next Flag")
        self.next_button.clicked.connect(self.display_random_flag)
        layout.addWidget(self.next_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        self.checked_in = []
        self.display_random_flag()

```

In this step, we've set up the main window, added labels for displaying the flag and country name, and created buttons for displaying the name and showing the next random flag. We've connected button clicks to their respective methods.

## Step 2: Displaying Random Flags

Next, let's implement the functionality to display random flags when the application starts and when the user clicks the "Next Flag" button.

{% include codeHeader.html %}
```python
    def display_random_flag(self):
            countries = get_country_list()
            while True:
                random_index = np.random.randint(0, len(countries))
                if (random_index not in self.checked_in):
                    self.checked_in.append(random_index)
                    break
                if len(self.checked_in) == len(countries):
                    
                    break
                    
            
            
            self.current_country = countries[random_index]
            flag_image = get_flag_img(self.current_country)
            qt_image = QImage(flag_image.tobytes(), flag_image.width, flag_image.height, flag_image.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.flag_label.setPixmap(pixmap)
            self.name_label.clear()


```

In this method:

* We fetch the list of countries using get_country_list() from the flagpy library.
* We generate a random index to select a country.
* We ensure that the selected index is not already checked (flag not displayed before).
* If all flags have been displayed, we reset the checked_in list to start again.
* We fetch the flag image using get_flag_img() and display it in the GUI.

## Step 3: Displaying Country Name

Now, let's implement the functionality to display the name of the country when the user clicks the "Show Name" button.

{% include codeHeader.html %}
```python
    def display_country_info(self):
        self.name_label.setText(self.current_country)

```

This method simply sets the text of the name_label to the current country's name.

## Step 4: Running the Application

Finally, let's run the application loop to start the GUI.

{% include codeHeader.html %}
```python
if __name__ == "__main__":
    app = QApplication([])
    window = RandomCountryFlagApp()
    window.show()
    app.exec()

```

## Conclusion

In this tutorial, we've created a simple "Guess the Country Name" game GUI using PySide6. The application displays flags of random countries, and the user can guess the country name by clicking a button. You can further enhance this application by adding scoring, time limits, or hints to make it more engaging. Have fun exploring and customizing the game!


