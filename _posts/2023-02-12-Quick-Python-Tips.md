---
description: Hello friends! Below are some examples that you can use to write better codes.
featured-img: 26072022-python-logo
keywords:
- Python
- Coding
- Software Development
layout: post
mathjax: true
tags:
- Python
- Coding
- Best Practices
title: Good python coding examples for the software development
---



Hello friends! Below are some examples that you can use to write better codes.

# 1. Use meaningful variable names:

```python

## Good
name = "John Doe"

## Bad
n = "John Doe"

```

### 2. Write descriptive docstrings for functions and modules:

```python

def greet(name):
    """This function greets the person passed in as a parameter"""
    print("Hello, " + name + ". How are you today?")

```

### 3. Use try-except blocks to handle exceptions:

```python
try:
    ## some code here
except Exception as e:
    ## handle exception here
```

### 4. Use the ‘with’ statement when working with files:

```python
with open("file.txt", "r") as file:
    data = file.read()
    ## do something with the file data
```

### 5. Avoid using a single leading underscore `(_)` to indicate a weak “internal use” variable. Use double leading underscores to avoid naming collisions in subclasses.

```python
## Good
class MyClass:
    def __init__(self):
        self.__variable = 42

## Bad
class MyClass:
    def __init__(self):
        self._variable = 42

```

### 6. Use list comprehensions instead of map and filter when possible:

```python
## Good
squared_numbers = [x**2 for x in numbers]

## Bad
squared_numbers = map(lambda x: x**2, numbers)

```

### 7. Use the built-in Python functions and libraries before writing your own implementation:

```python
## Good
import os
os.remove("file.txt")

## Bad
import shutil
shutil.rmtree("file.txt")

```

### 8. Use PEP 8 style guide for writing code:

```python
## Good
def greet_person(person_name):
    print("Hello, " + person_name + "!")

## Bad
def greetPerson(personName):
    print("Hello, " + personName + "!")

```

### 9. Use is and is not instead of equality (==) and inequality (!=) when comparing objects to None:

```python
## Good
if value is None:
    ## do something

## Bad
if value == None:
    ## do something

```

### 10. Use context managers (with statement) when working with resources like files, sockets, and databases to ensure that they are properly cleaned up when done:

```python
## Good
with open("file.txt", "w") as file:
    file.write("Hello, World!")

## Bad
file = open("file.txt", "w")
file.write("Hello, World!")
file.close()

```
### 11. Use Python’s built-in logging module instead of print statements for logging and debugging:

```python
## Good
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("Debug message")

## Bad
print("Debug message")

```

### 12. Use functions to extract complex logic and make code reusable:

```python
## Good
def calculate_sum(a, b):
    return a + b

result = calculate_sum(2, 3)

## Bad
result = 2 + 3

```

### 13. Use the with statement when working with databases to ensure that database connections are properly closed:

```python
## Good
import sqlite3

with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()

## Bad
import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
conn.close()

```

### 14. Use list comprehensions instead of for loops for simple data manipulations:

```python
## Good
squared_numbers = [x**2 for x in range(1, 10)]

## Bad
squared_numbers = []
for x in range(1, 10):
    squared_numbers.append(x**2)

```

### 15. Use the if __name__ == "__main__" statement when writing script files:

```python
def main():
    ## Main code here

if __name__ == "__main__":
    main()

```
This ensures that the code in the script file is only executed when the file is run as the main program, and not when it is imported as a module.

### 16. Use the with statement when working with locks to ensure that the lock is properly released:

```python
import threading

lock = threading.Lock()

## Good
with lock:
    ## Critical section

## Bad
lock.acquire()
try:
    ## Critical section
finally:
    lock.release()

```

### 17. Use Python’s built-in unittest module for writing and running tests:
```python

import unittest

class TestSum(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()

```

### 18. Use the enum module to define named constants:

```python
import enum

class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

## Good
if color == Color.RED:
    ## do something

## Bad
if color == 1:
    ## do something

```

### 19. Use the collections module for working with data structures such as defaultdict, Counter, and namedtuple:

```python
from collections import defaultdict

## Good
word_count = defaultdict(int)
for word in words:
    word_count[word] += 1

## Bad
word_count = {}
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

```

### 20. Use the itertools module for working with iterators and generators:

```python
import itertools

## Good
squared_numbers = list(map(lambda x: x**2, range(1, 10)))

## Bad
squared_numbers = []
for x in range(1, 10):
    squared_numbers.append(x**2)


```

### 21. Use try-except blocks to handle exceptions:

```python
## Good
try:
    ## code that may raise an exception
except Exception as e:
    ## handle the exception

## Bad
try:
    ## code that may raise an exception
except:
    ## handle the exception

```

### 22. Use the with statement to automatically close files:

```python
## Good
with open("file.txt", "r") as f:
    ## read from file

## Bad
f = open("file.txt", "r")
try:
    ## read from file
finally:
    f.close()

```

### 23. Use the os module to interact with the operating system:

```python 
import os

## Good
if os.path.exists("file.txt"):
    ## do something

## Bad
import os.path
if os.path.exists("file.txt"):
    ## do something

```
### 24. Use the shutil module to manipulate files and directories:

```python
import shutil

## Good
shutil.copyfile("src.txt", "dst.txt")

## Bad
import os
os.system("cp src.txt dst.txt")

```

### 25. Use the re module to work with regular expressions:

```python
import re

## Good
match = re.search(r"\d+", "Number: 42")

## Bad
import string
match = string.find("Number: 42", "42")
```

### 26. Use the datetime module to work with dates and times:

```python
import datetime

## Good
now = datetime.datetime.now()

## Bad
import time
now = time.localtime()

```

### 27. Use the math module to perform mathematical operations:

```python
import math

## Good
result = math.sqrt(16)

## Bad
result = 16**0.5

```

### 28. Use the random module to generate random numbers:

```python
import random

## Good
random_number = random.randint(1, 100)

## Bad
import time
random_number = int(time.time() % 100)

```

### 29. Use str.format() method for string formatting:

```python
## Good
name = "John"
print("Hello, {}!".format(name))

## Bad
name = "John"
print("Hello, " + name + "!")

```

### 30. Use list comprehensions instead of loops for simple operations:

```python
## Good
numbers = [1, 2, 3, 4, 5]
squared_numbers = [n**2 for n in numbers]

## Bad
numbers = [1, 2, 3, 4, 5]
squared_numbers = []
for n in numbers:
    squared_numbers.append(n**2)

```

### 31. Use generators instead of lists when working with large data sets:

```python
## Good
def squares(n):
    for i in range(n):
        yield i**2

## Bad
def squares(n):
    squares = []
    for i in range(n):
        squares.append(i**2)
    return squares

```

### 32. Use the enum module to define named constants:

```python
import enum

## Good
class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

## Bad
RED = 1
GREEN = 2
BLUE = 3

```

### 33. Use the collections module for working with collections:

```python
import collections

## Good
counter = collections.Counter("hello world")

## Bad
counter = {}
for c in "hello world":
    if c in counter:
        counter[c] += 1
    else:
        counter[c] = 1

```

### 34. Use the functools module for working with functions:

```python
Use the functools module for working with functions:

```

### 35. Use the itertools module for working with iterators:

```python
import itertools

## Good
even_numbers = itertools.islice(itertools.count(0, 2), 5)

## Bad
even_numbers = []
for i in range(0, 10, 2):
    even_numbers.append(i)

```

### 36. Use the logging module for logging messages:

```python
import logging

## Good
logging.basicConfig(level=logging.INFO)
logging.debug("Debug message")
logging.info("Info message")

## Bad
if debug:
    print("Debug message")
print("Info message")

```

### 37. Use with statement when working with files or other resources:

```python
## Good
with open("file.txt", "r") as f:
    content = f.read()

## Bad
f = open("file.txt", "r")
content = f.read()
f.close()

```

### 38. Use the unittest module for writing unit tests:

```python
import unittest

## Good
class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")

## Bad
def test_upper():
    assert "foo".upper() == "FOO"

```

### 39. Use assert statements to check for preconditions and postconditions:

```python
## Good
def divide(a, b):
    assert b != 0, "division by zero"
    return a / b

## Bad
def divide(a, b):
    if b == 0:
        raise Exception("division by zero")
    return a / b

```

### 40. Use the os and shutil modules for working with files and directories:

```python
import os
import shutil

## Good
if not os.path.exists("directory"):
    os.makedirs("directory")
shutil.copy("file1.txt", "directory")

## Bad
if not os.path.exists("directory"):
    os.system("mkdir directory")
os.system("cp file1.txt directory")

```

### 41. Use the contextlib module for creating context managers:

```python
import contextlib

## Good
@contextlib.contextmanager
def make_temp_directory():
    temp_directory = tempfile.mkdtemp()
    try:
        yield temp_directory
    finally:
        shutil.rmtree(temp_directory)

## Bad
temp_directory = tempfile.mkdtemp()
try:
    ## do something with temp_directory
finally:
    shutil.rmtree(temp_directory)

```

### 42. Use the argparse module for parsing command line arguments:


```python
import argparse

## Good
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="name of the person")
parser.add_argument("-a", "--age", help="age of the person", type=int)
args = parser.parse_args()

## Bad
import sys

if len(sys.argv) < 3:
    print("usage: script.py -n name -a age")
    sys.exit(1)
if sys.argv[1] == "-n":
    name = sys.argv[2]

```

### 43. Use the collections module for working with data structures:

```python
import collections

## Good
counter = collections.Counter([1, 2, 2, 3, 3, 3])
print(counter)

## Bad
counter = {}
for item in [1, 2, 2, 3, 3, 3]:
    if item in counter:
        counter[item] += 1
    else:
        counter[item] = 1
print(counter)

```

### 44. Use the re module for working with regular expressions:

```python
import re

## Good
result = re.search("\d{3}-\d{2}-\d{4}", "Social Security Number: 123-45-6789")
print(result.group(0))

## Bad
string = "Social Security Number: 123-45-6789"
start = string.find("123")
end = start + len("123-45-6789")
print(string[start:end])

```

### 45. Use the datetime module for working with dates and times:

```python
import datetime

## Good
today = datetime.datetime.now().date()
print(today)

## Bad
from time import gmtime, strftime
print(strftime("%Y-%m-%d", gmtime()))

```

### 46. Use the bisect module for working with sorted lists:

```python
import bisect

## Good
arr = [1, 2, 3, 4, 4, 4, 5]
index = bisect.bisect_left(arr, 4)
print(index)

## Bad
arr = [1, 2, 3, 4, 4, 4, 5]
index = 0
for i, item in enumerate(arr):
    if item >= 4:
        index = i
        break
print(index)

```

### 47. Use the itertools module for working with iterators:

```python
import itertools

## Good
numbers = [1, 2, 3, 4, 5]
result = list(itertools.combinations(numbers, 2))
print(result)

## Bad
numbers = [1, 2, 3, 4, 5]
result = []
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        result.append((numbers[i], numbers[j]))
print(result)

```

### 48. Use the functools module for working with functions:

```python
import functools

## Good
def add(a, b):
    return a + b
add = functools.partial(add, b=10)
print(add(5))

## Bad
def add(a, b=10):
    return a + b
print(add(5))

```

### 49. Use the threading module for working with threads:

```python
import threading

## Good
def worker():
    print("worker")

thread = threading.Thread(target=worker)
thread.start()
thread.join()
```

### 50. Use the os module for working with the operating system:

```python
import os

## Good
print(os.path.join("data", "input.txt"))

## Bad
print("data" + os.sep + "input.txt")

```

### 51. Use the shutil module for working with file objects:

```python
import shutil

## Good
shutil.copy2("input.txt", "input_backup.txt")

## Bad
with open("input.txt", "r") as src:
    with open("input_backup.txt", "w") as dst:
        dst.write(src.read())

```
### 52. Use the glob module for working with file patterns:

```python
import glob

## Good
files = glob.glob("*.txt")
print(files)

## Bad
import os
files = [file for file in os.listdir() if file.endswith(".txt")]
print(files)

```

### 53. Use the pickle module for working with binary data:

```python
import pickle

## Good
data = {"name": "John Doe", "age": 30}
with open("data.pkl", "wb") as file:
    pickle.dump(data, file)

## Bad
data = {"name": "John Doe", "age": 30}
with open("data.txt", "w") as file:
    file.write(str(data))

```

### 54. Use the csv module for working with CSV files:

```python
import csv

## Good
data = [["name", "age"], ["John Doe", 30]]
with open("data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

## Bad
data = [["name", "age"], ["John Doe", 30]]
with open("data.csv", "w") as file:
    for row in data:
        file.write(",".join(row) + "\n")

```

### 55. Use the json module for working with JSON data:

```python
import json

## Good
data = {"name": "John Doe", "age": 30}
with open("data.json", "w") as file:
    json.dump(data, file)

## Bad
data = {"name": "John Doe", "age": 30}
with open("data.json", "w") as file:
    file.write(str(data))

```

### 56. Use try...except blocks to handle exceptions:

```python
## Good
try:
    result = 1 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

## Bad
result = 1 / 0
print("This line will never be reached")

```

### 57. Use with statements to work with context managers:

```python
## Good
with open("data.txt", "r") as file:
    data = file.read()

## Bad
file = open("data.txt", "r")
data = file.read()
file.close()

```

### 58. Use assert statements for debugging and testing:

```python
## Good
assert len(data) > 0, "Data is empty"

## Bad
if len(data) == 0:
    raise ValueError("Data is empty")

```

### 59. Use is and is not instead of == and != when comparing to None:

```python
## Good
if data is None:
    print("Data is None")

## Bad
if data == None:
    print("Data is None")

```

### 60. Use else blocks in for and while loops to specify an else block that is executed if the loop terminates normally (i.e., without encountering a break statement):

```python
## Good
for number in range(10):
    if number == 5:
        break
else:
    print("5 not found")

## Bad
found = False
for number in range(10):
    if number == 5:
        found = True
        break
if not found:
    print("5 not found")

```

### 61. Use enumerate() function to loop over a list and get both the index and the value:

```python
## Good
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(index, fruit)

## Bad
fruits = ["apple", "banana", "cherry"]
for i in range(len(fruits)):
    print(i, fruits[i])

```

### 62. Use list comprehensions instead of loops for simple operations on lists:


```python
## Good
squared_numbers = [x ** 2 for x in range(10)]

## Bad
squared_numbers = []
for x in range(10):
    squared_numbers.append(x ** 2)

```

### 63. Use zip() function to loop over multiple lists in parallel:

```python
## Good
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(name, age)

## Bad
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for i in range(len(names)):
    print(names[i], ages[i])

```

### 64. Use dict.get() method instead of dict[key] to avoid KeyError:

```python
## Good
person = {"name": "Alice", "age": 25}
age = person.get("age", 0)

## Bad
person = {"name": "Alice", "age": 25}
age = person["age"]

```

### 65. Use defaultdict from the collections module instead of dict when you need a dictionary with default values:

```python
## Good
from collections import defaultdict

person = defaultdict(str)
person["name"] = "Alice"
print(person["age"])  # Output: ''

## Bad
person = {}
person["name"] = "Alice"
try:
    print(person["age"])
except KeyError:
    print("Age not found")

```

### 66. Use map() and filter() functions to apply operations to lists:

```python
## Good
numbers = [1, 2, 3, 4, 5]
squared_numbers = list(map(lambda x: x ** 2, numbers))
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

## Bad
numbers = [1, 2, 3, 4, 5]
squared_numbers = []
for number in numbers:
    squared_numbers.append(number ** 2)
even_numbers = []
for number in numbers:
    if number % 2 == 0:
        even_numbers.append(number)

```

### 67. Use functools.partial() to partially apply functions:

```python
## Good
from functools import partial

def multiply(a, b):
    return a * b

double = partial(multiply, 2)
print(double(3))  # Output: 6

## Bad
def double(b):
    return 2 * b

```

### 68. Use itertools functions to work with iterators:

```python
## Good
import itertools

numbers = [1, 2, 3, 4, 5]
combinations = list(itertools.combinations(numbers, 2))

## Bad
numbers = [1, 2, 3, 4, 5]
combinations = []
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        combinations.append((numbers[i], numbers[j]))

```

### 69. Use namedtuple from the collections module instead of tuples:

```python
## Good
from collections import namedtuple

Person = namedtuple("Person", ["name", "age"])
person = Person("Alice", 25)

## Bad
person = ("Alice", 25)

```

### 70. Use dataclasses module for defining classes with default implementations for special methods:

```python
## Good
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

person = Person("Alice", 25)

## Bad
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 25)

```

### 71. Use the format method for string formatting:

```python
## Good
name = "Alice"
age = 25
print("My name is {0} and I am {1} years old.".format(name, age))

## Bad
name = "Alice"
age = 25
print("My name is " + name + " and I am " + str(age) + " years old.")

```

### 72. Use enumerate to iterate over lists and keep track of the index:

```python
## Good
names = ["Alice", "Bob", "Charlie"]
for i, name in enumerate(names):
    print(i, name)

## Bad
names = ["Alice", "Bob", "Charlie"]
for i in range(len(names)):
    print(i, names[i])

```

### 73. Use list comprehensions for simple list transformations:

```python
## Good
numbers = [1, 2, 3, 4, 5]
squared_numbers = [number ** 2 for number in numbers]

## Bad
numbers = [1, 2, 3, 4, 5]
squared_numbers = []
for number in numbers:
    squared_numbers.append(number ** 2)

```

### 74. Use the with statement when working with resources that need to be cleaned up, such as files:

```python
## Good
with open("file.txt", "r") as f:
    contents = f.read()

## Bad
f = open("file.txt", "r")
contents = f.read()
f.close()

```

### 75. Use zip to iterate over multiple lists in parallel:

```python
## Good
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(name, age)

## Bad
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for i in range(len(names)):
    print(names[i], ages[i])

```

### 76. Use exceptions to handle errors instead of returning error codes:

```python
## Good
try:
    result = divide(a, b)
except ZeroDivisionError:
    print("Cannot divide by zero.")

## Bad
result = divide(a, b)
if result == ERROR_CODE:
    print("Cannot divide by zero.")

```

### 77. Use the else clause in a for loop to perform an action when the loop finishes without being interrupted:

```python
## Good
numbers = [1, 2, 3, 4, 5]
for number in numbers:
    if number == 3:
        break
else:
    print("Loop finished without being interrupted.")

## Bad
numbers = [1, 2, 3, 4, 5]
interrupted = False
for number in numbers:
    if number == 3:
        interrupted = True
        break
if not interrupted:
    print("Loop finished without being interrupted.")

```

### 78. Use the repr method to provide a string representation of an object that can be used to recreate it:

```python
## Good
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Point({0}, {1})".format(self.x, self.y)

## Bad
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({0}, {1})".format(self.x, self.y)

```

### 79. Use the with statement when working with resources that need to be cleaned up, such as databases:

```python
## Good
with conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()

## Bad
conn = connect()
cursor = conn.cursor()
cursor.execute("SELECT * FROM users")
users = cursor.fetchall()
conn.close()

```

### 80. Use the else clause in a while loop to perform an action when the loop finishes without being interrupted:

```python
## Good
i = 0
while i < 5:
    if i == 3:
        break
    i += 1
else:
    print("Loop finished without being interrupted.")

## Bad
i = 0
interrupted = False
while i < 5:
    if i == 3:
        interrupted = True
        break
    i += 1
if not interrupted:
    print("Loop finished without being interrupted.")

```

### 81. Use the enumerate function when iterating over a sequence to keep track of the index:

```python
## Good
words = ["apple", "banana", "cherry"]
for i, word in enumerate(words):
    print(i, word)

## Bad
words = ["apple", "banana", "cherry"]
i = 0
for word in words:
    print(i, word)
    i += 1

```

### 82. Use the zip function when iterating over multiple sequences in parallel:


```python
## Good
words = ["apple", "banana", "cherry"]
lengths = [5, 6, 6]
for word, length in zip(words, lengths):
    print(word, length)

## Bad
words = ["apple", "banana", "cherry"]
lengths = [5, 6, 6]
for i in range(len(words)):
    print(words[i], lengths[i])

```

### 83. Use list comprehensions instead of filter and map when possible:

```python
## Good
squared = [x ** 2 for x in range(10) if x % 2 == 0]

## Bad
squared = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, range(10))))

```

### 84. Use the None object instead of False or other values to represent a lack of value:

```python
## Good
def divide(a, b):
    if b == 0:
        return None
    return a / b

## Bad
def divide(a, b):
    if b == 0:
        return False
    return a / b

```

### 85. Use the assert statement to check for conditions that should always be true:

```python
## Good
assert divide(4, 2) == 2

## Bad
result = divide(4, 2)
if result != 2:
    raise Exception("Unexpected result.")

```

### 86. Use the collections module to work with data structures such as Counter, defaultdict, and OrderedDict:

```python
## Good
from collections import Counter

word = "mississippi"
counter = Counter(word)

## Bad
word = "mississippi"
counter = {}
for letter in word:
    if letter in counter:
        counter[letter] += 1
    else:
        counter[letter] = 1

```

### 87. Use the str.format() method instead of string concatenation for string formatting:

```python
## Good
name = "Alice"
print("Hello, {}!".format(name))

## Bad
name = "Alice"
print("Hello, " + name + "!")

```

### 88. Use the with statement when working with files or other resources that need to be cleaned up after use:

```python
## Good
with open("file.txt", "r") as file:
    contents = file.read()

## Bad
file = open("file.txt", "r")
contents = file.read()
file.close()

```

### 89. Use the try...except statement to handle exceptions:

```python
## Good
try:
    value = int("abc")
except ValueError:
    value = 0

## Bad
value = int("abc")
except ValueError:
    value = 0

```

### 90. Use the pass statement as a placeholder when a statement is required syntactically, but no action is needed:

```python
## Good
def noop():
    pass

## Bad
def noop():
    ## TODO: implement me!
    return

```

### 91. Use the functools module to work with functions, such as partial and reduce:

```python
## Good
from functools import reduce

numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)

## Bad
numbers = [1, 2, 3, 4, 5]
product = 1
for number in numbers:
    product *= number

```

### 92. Use the itertools module to work with iterators, such as chain and zip_longest:

```python
## Good
from itertools import chain

words = ["apple", "banana", "cherry"]
letters = chain.from_iterable(words)

## Bad
words = ["apple", "banana", "cherry"]
letters = []
for word in words:
    for letter in word:
        letters.append(letter)

```

### 93. Use the enum module to define enumerations:

```python
## Good
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

## Bad
RED = 1
GREEN = 2
BLUE = 3

```

### 94. Use the typing module to specify the types of variables, arguments, and return values:

```python
## Good
from typing import List

def reverse(lst: List[int]) -> List[int]:
    return lst[::-1]

## Bad
def reverse(lst):
    return lst[::-1]

```

### 95. Use the collections module to work with collections, such as Counter and defaultdict:

```python
## Good
from collections import Counter

words = ["apple", "banana", "cherry", "apple", "banana"]
counter = Counter(words)

## Bad
words = ["apple", "banana", "cherry", "apple", "banana"]
counter = {}
for word in words:
    if word in counter:
        counter[word] += 1
    else:
        counter[word] = 1

```

### 96. Use the queue module to work with queues, such as Queue and PriorityQueue:

```python
## Good
from queue import Queue

q = Queue()
q.put("apple")
q.put("banana")
q.put("cherry")

## Bad
q = []
q.append("apple")
q.append("banana")
q.append("cherry")

```

### 97. Use the weakref module to work with weak references:

```python
## Good
import weakref

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

a = Node(1)
b = Node(2)
a.next = weakref.ref(b)

## Bad
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

a = Node(1)
b = Node(2)
a.next = b

```

### 98. Use the copy module to make shallow or deep copies of objects:

```python
## Good
import copy

a = [1, 2, 3]
b = copy.copy(a)
c = copy.deepcopy(a)

## Bad
a = [1, 2, 3]
b = a
c = a[:]

```

### 99. Use the dis module to disassemble and inspect Python bytecode:


```python
## Good
import dis

def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

dis.dis(fib)

## Bad
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib.__code__.co_code)

```

### 100. Use the logging module to log messages, such as debug, info, warning, error, and critical:

```python
## Good
import logging

logging.basicConfig(filename="app.log", level=logging.DEBUG)
logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.critical("This is a critical message")

## Bad
print("[DEBUG] This is a debug message")
print("[INFO] This is an info message")
print("[WARNING] This is a warning message")
print("[ERROR] This is an error message")
print("[CRITICAL] This is a critical message")

```

