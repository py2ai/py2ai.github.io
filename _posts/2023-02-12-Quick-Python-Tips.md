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

9. Use is and is not instead of equality (==) and inequality (!=) when comparing objects to None:

```python
# Good
if value is None:
    # do something

# Bad
if value == None:
    # do something

```

10. Use context managers (with statement) when working with resources like files, sockets, and databases to ensure that they are properly cleaned up when done:

```python
# Good
with open("file.txt", "w") as file:
    file.write("Hello, World!")

# Bad
file = open("file.txt", "w")
file.write("Hello, World!")
file.close()

```
11. Use Python’s built-in logging module instead of print statements for logging and debugging:

```python
# Good
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("Debug message")

# Bad
print("Debug message")

```

12. Use functions to extract complex logic and make code reusable:

```python
# Good
def calculate_sum(a, b):
    return a + b

result = calculate_sum(2, 3)

# Bad
result = 2 + 3

```

13. Use the with statement when working with databases to ensure that database connections are properly closed:

```python
# Good
import sqlite3

with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()

# Bad
import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
conn.close()

```

14. Use list comprehensions instead of for loops for simple data manipulations:

```python
# Good
squared_numbers = [x**2 for x in range(1, 10)]

# Bad
squared_numbers = []
for x in range(1, 10):
    squared_numbers.append(x**2)

```

15. Use the if __name__ == "__main__" statement when writing script files:

```python
def main():
    # Main code here

if __name__ == "__main__":
    main()

```
This ensures that the code in the script file is only executed when the file is run as the main program, and not when it is imported as a module.

16. Use the with statement when working with locks to ensure that the lock is properly released:

```python
import threading

lock = threading.Lock()

# Good
with lock:
    # Critical section

# Bad
lock.acquire()
try:
    # Critical section
finally:
    lock.release()

```

17. Use Python’s built-in unittest module for writing and running tests:
```python

import unittest

class TestSum(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()

```

18. Use the enum module to define named constants:

```python
import enum

class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# Good
if color == Color.RED:
    # do something

# Bad
if color == 1:
    # do something

```

19. Use the collections module for working with data structures such as defaultdict, Counter, and namedtuple:

```python
from collections import defaultdict

# Good
word_count = defaultdict(int)
for word in words:
    word_count[word] += 1

# Bad
word_count = {}
for word in words:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

```

20. Use the itertools module for working with iterators and generators:

```python
import itertools

# Good
squared_numbers = list(map(lambda x: x**2, range(1, 10)))

# Bad
squared_numbers = []
for x in range(1, 10):
    squared_numbers.append(x**2)


```

21. Use try-except blocks to handle exceptions:

```python
# Good
try:
    # code that may raise an exception
except Exception as e:
    # handle the exception

# Bad
try:
    # code that may raise an exception
except:
    # handle the exception

```

22. Use the with statement to automatically close files:

```python
# Good
with open("file.txt", "r") as f:
    # read from file

# Bad
f = open("file.txt", "r")
try:
    # read from file
finally:
    f.close()

```

23. Use the os module to interact with the operating system:

```python 
import os

# Good
if os.path.exists("file.txt"):
    # do something

# Bad
import os.path
if os.path.exists("file.txt"):
    # do something

```
24. Use the shutil module to manipulate files and directories:

```python
import shutil

# Good
shutil.copyfile("src.txt", "dst.txt")

# Bad
import os
os.system("cp src.txt dst.txt")

```

25. Use the re module to work with regular expressions:

```python
import re

# Good
match = re.search(r"\d+", "Number: 42")

# Bad
import string
match = string.find("Number: 42", "42")
```

26. Use the datetime module to work with dates and times:

```python
import datetime

# Good
now = datetime.datetime.now()

# Bad
import time
now = time.localtime()

```

27. Use the math module to perform mathematical operations:

```python
import math

# Good
result = math.sqrt(16)

# Bad
result = 16**0.5

```

28. Use the random module to generate random numbers:

```python
import random

# Good
random_number = random.randint(1, 100)

# Bad
import time
random_number = int(time.time() % 100)

```

29. Use str.format() method for string formatting:

```python
# Good
name = "John"
print("Hello, {}!".format(name))

# Bad
name = "John"
print("Hello, " + name + "!")

```

30. Use list comprehensions instead of loops for simple operations:

```python
# Good
numbers = [1, 2, 3, 4, 5]
squared_numbers = [n**2 for n in numbers]

# Bad
numbers = [1, 2, 3, 4, 5]
squared_numbers = []
for n in numbers:
    squared_numbers.append(n**2)

```

31. Use generators instead of lists when working with large data sets:

```python
# Good
def squares(n):
    for i in range(n):
        yield i**2

# Bad
def squares(n):
    squares = []
    for i in range(n):
        squares.append(i**2)
    return squares

```

32. Use the enum module to define named constants:

```python
import enum

# Good
class Color(enum.Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

# Bad
RED = 1
GREEN = 2
BLUE = 3

```

33. Use the collections module for working with collections:

```python
import collections

# Good
counter = collections.Counter("hello world")

# Bad
counter = {}
for c in "hello world":
    if c in counter:
        counter[c] += 1
    else:
        counter[c] = 1

```

34. Use the functools module for working with functions:

```python
Use the functools module for working with functions:

```

35. Use the itertools module for working with iterators:

```python
import itertools

# Good
even_numbers = itertools.islice(itertools.count(0, 2), 5)

# Bad
even_numbers = []
for i in range(0, 10, 2):
    even_numbers.append(i)

```

36. Use the logging module for logging messages:

```python
import logging

# Good
logging.basicConfig(level=logging.INFO)
logging.debug("Debug message")
logging.info("Info message")

# Bad
if debug:
    print("Debug message")
print("Info message")

```

37. Use with statement when working with files or other resources:

```python
# Good
with open("file.txt", "r") as f:
    content = f.read()

# Bad
f = open("file.txt", "r")
content = f.read()
f.close()

```

38. Use the unittest module for writing unit tests:

```python
import unittest

# Good
class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        self.assertEqual("foo".upper(), "FOO")

# Bad
def test_upper():
    assert "foo".upper() == "FOO"

```

39. Use assert statements to check for preconditions and postconditions:

```python
# Good
def divide(a, b):
    assert b != 0, "division by zero"
    return a / b

# Bad
def divide(a, b):
    if b == 0:
        raise Exception("division by zero")
    return a / b

```

40. Use the os and shutil modules for working with files and directories:

```python
import os
import shutil

# Good
if not os.path.exists("directory"):
    os.makedirs("directory")
shutil.copy("file1.txt", "directory")

# Bad
if not os.path.exists("directory"):
    os.system("mkdir directory")
os.system("cp file1.txt directory")

```

41. Use the contextlib module for creating context managers:

```python
import contextlib

# Good
@contextlib.contextmanager
def make_temp_directory():
    temp_directory = tempfile.mkdtemp()
    try:
        yield temp_directory
    finally:
        shutil.rmtree(temp_directory)

# Bad
temp_directory = tempfile.mkdtemp()
try:
    # do something with temp_directory
finally:
    shutil.rmtree(temp_directory)

```

42. Use the argparse module for parsing command line arguments:


```python
import argparse

# Good
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="name of the person")
parser.add_argument("-a", "--age", help="age of the person", type=int)
args = parser.parse_args()

# Bad
import sys

if len(sys.argv) < 3:
    print("usage: script.py -n name -a age")
    sys.exit(1)
if sys.argv[1] == "-n":
    name = sys.argv[2]

```

43. Use the collections module for working with data structures:

```python
import collections

# Good
counter = collections.Counter([1, 2, 2, 3, 3, 3])
print(counter)

# Bad
counter = {}
for item in [1, 2, 2, 3, 3, 3]:
    if item in counter:
        counter[item] += 1
    else:
        counter[item] = 1
print(counter)

```

44. Use the re module for working with regular expressions:

```python
import re

# Good
result = re.search("\d{3}-\d{2}-\d{4}", "Social Security Number: 123-45-6789")
print(result.group(0))

# Bad
string = "Social Security Number: 123-45-6789"
start = string.find("123")
end = start + len("123-45-6789")
print(string[start:end])

```

45. Use the datetime module for working with dates and times:

```python
import datetime

# Good
today = datetime.datetime.now().date()
print(today)

# Bad
from time import gmtime, strftime
print(strftime("%Y-%m-%d", gmtime()))

```

46. Use the bisect module for working with sorted lists:

```python
import bisect

# Good
arr = [1, 2, 3, 4, 4, 4, 5]
index = bisect.bisect_left(arr, 4)
print(index)

# Bad
arr = [1, 2, 3, 4, 4, 4, 5]
index = 0
for i, item in enumerate(arr):
    if item >= 4:
        index = i
        break
print(index)

```

47. Use the itertools module for working with iterators:

```python
import itertools

# Good
numbers = [1, 2, 3, 4, 5]
result = list(itertools.combinations(numbers, 2))
print(result)

# Bad
numbers = [1, 2, 3, 4, 5]
result = []
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        result.append((numbers[i], numbers[j]))
print(result)

```

48. Use the functools module for working with functions:

```python
import functools

# Good
def add(a, b):
    return a + b
add = functools.partial(add, b=10)
print(add(5))

# Bad
def add(a, b=10):
    return a + b
print(add(5))

```

49. Use the threading module for working with threads:

```python
import threading

# Good
def worker():
    print("worker")

thread = threading.Thread(target=worker)
thread.start()
thread.join()
```

50. Use the os module for working with the operating system:

```python
import os

# Good
print(os.path.join("data", "input.txt"))

# Bad
print("data" + os.sep + "input.txt")

```

51. Use the shutil module for working with file objects:

```python
import shutil

# Good
shutil.copy2("input.txt", "input_backup.txt")

# Bad
with open("input.txt", "r") as src:
    with open("input_backup.txt", "w") as dst:
        dst.write(src.read())

```

