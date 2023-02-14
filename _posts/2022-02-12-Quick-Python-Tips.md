---
layout: post
title: Good python coding examples for the software development
mathjax: true
featured-img: 26072022-python-logo
summary:  Some good tips to use while writing python codes
---

Hello friends! Below are some examples that you can use to write better codes.

1. Use meaningful variable names:
```python
# Good
name = "John Doe"

# Bad
n = "John Doe"
```

2. Write descriptive docstrings for functions and modules:
```python
def greet(name):
    """This function greets the person passed in as a parameter"""
    print("Hello, " + name + ". How are you today?")

```

3. Use try-except blocks to handle exceptions:
```python
try:
    # some code here
except Exception as e:
    # handle exception here
```

4. Use the ‘with’ statement when working with files:
```python
with open("file.txt", "r") as file:
    data = file.read()
    # do something with the file data
```

5. Avoid using a single leading underscore (_) to indicate a weak “internal use” variable. Use double leading underscores to avoid naming collisions in subclasses.

```python
# Good
class MyClass:
    def __init__(self):
        self.__variable = 42

# Bad
class MyClass:
    def __init__(self):
        self._variable = 42

```

6. Use list comprehensions instead of map and filter when possible:
```python
# Good
squared_numbers = [x**2 for x in numbers]

# Bad
squared_numbers = map(lambda x: x**2, numbers)

```

7. Use the built-in Python functions and libraries before writing your own implementation:
```python
# Good
import os
os.remove("file.txt")

# Bad
import shutil
shutil.rmtree("file.txt")

```

8. Use PEP 8 style guide for writing code:
```python
# Good
def greet_person(person_name):
    print("Hello, " + person_name + "!")

# Bad
def greetPerson(personName):
    print("Hello, " + personName + "!")

```










