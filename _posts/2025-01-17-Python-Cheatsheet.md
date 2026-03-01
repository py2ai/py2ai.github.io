---
date: 2025-01-17
layout: post
title: "Python Cheatsheet Every Learner Must Know - Save Hours of Time"
description: "Master Python programming with this comprehensive cheatsheet. From basic syntax to advanced concepts, save hours of time with quick reference tables and code examples."
featured-img: 2026-cheatsheet/2026-cheatsheet
keywords:
- python cheatsheet
- python reference
- python quick guide
- python programming
- python basics
- python tutorial
- python code examples
- python learning
- python syntax
- python essentials
categories:
- Python tutorial series
tags:
- Python
- Cheatsheet
- Programming
- Tutorial
- Python Basics
- Python Reference
- Code Examples
- Learning Python
permalink: /2026-02-17-Python-Cheatsheet/
mathjax: true
---

# Python Cheatsheet Every Learner Must Know

Python is one of the most versatile and beginner-friendly programming languages. Whether you're just starting out or looking to refresh your memory, this comprehensive cheatsheet will save you hours of time searching for syntax and common patterns. Keep this guide handy as your quick reference!

## Quick Reference Table

<table>
<thead>
<tr>
<th>Category</th>
<th>Concept</th>
<th>Syntax</th>
<th>Example</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Variables</strong></td>
<td>Assignment</td>
<td><code>x = 5</code></td>
<td><pre><code class="language-python">x = 5</code></pre></td>
</tr>
<tr>
<td></td>
<td>Multiple assignment</td>
<td><code>a, b = 1, 2</code></td>
<td><pre><code class="language-python">a, b = 1, 2</code></pre></td>
</tr>
<tr>
<td><strong>Data Types</strong></td>
<td>String</td>
<td><code>text = "Hello"</code></td>
<td><pre><code class="language-python">text = "Hello"</code></pre></td>
</tr>
<tr>
<td></td>
<td>Integer</td>
<td><code>num = 42</code></td>
<td><pre><code class="language-python">num = 42</code></pre></td>
</tr>
<tr>
<td></td>
<td>Float</td>
<td><code>pi = 3.14</code></td>
<td><pre><code class="language-python">pi = 3.14</code></pre></td>
</tr>
<tr>
<td></td>
<td>Boolean</td>
<td><code>is_true = True</code></td>
<td><pre><code class="language-python">is_true = True</code></pre></td>
</tr>
<tr>
<td></td>
<td>List</td>
<td><code>items = [1, 2, 3]</code></td>
<td><pre><code class="language-python">items = [1, 2, 3]</code></pre></td>
</tr>
<tr>
<td></td>
<td>Tuple</td>
<td><code>coords = (x, y)</code></td>
<td><pre><code class="language-python">coords = (10, 20)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Dictionary</td>
<td><code>data = {"key": "value"}</code></td>
<td><pre><code class="language-python">data = {"name": "Alice"}</code></pre></td>
</tr>
<tr>
<td></td>
<td>Set</td>
<td><code>unique = {1, 2, 3}</code></td>
<td><pre><code class="language-python">unique = {1, 2, 3}</code></pre></td>
</tr>
<tr>
<td><strong>String Operations</strong></td>
<td>Concatenation</td>
<td><code>"Hello" + " World"</code></td>
<td><pre><code class="language-python">"Hello" + " World"</code></pre></td>
</tr>
<tr>
<td></td>
<td>Repetition</td>
<td><code>"Ha" * 3</code></td>
<td><pre><code class="language-python">"Ha" * 3</code></pre></td>
</tr>
<tr>
<td></td>
<td>Slicing</td>
<td><code>text[0:5]</code></td>
<td><pre><code class="language-python">text = "Hello"
text[0:5]</code></pre></td>
</tr>
<tr>
<td></td>
<td>Length</td>
<td><code>len(text)</code></td>
<td><pre><code class="language-python">len("Hello")</code></pre></td>
</tr>
<tr>
<td></td>
<td>Upper/Lower</td>
<td><code>text.upper()</code> / <code>text.lower()</code></td>
<td><pre><code class="language-python">"hello".upper()</code></pre></td>
</tr>
<tr>
<td></td>
<td>Strip whitespace</td>
<td><code>text.strip()</code></td>
<td><pre><code class="language-python">"  text  ".strip()</code></pre></td>
</tr>
<tr>
<td></td>
<td>Replace</td>
<td><code>text.replace("old", "new")</code></td>
<td><pre><code class="language-python">"Hello World".replace("World", "Python")</code></pre></td>
</tr>
<tr>
<td></td>
<td>Split</td>
<td><code>text.split(",")</code></td>
<td><pre><code class="language-python">"a,b,c".split(",")</code></pre></td>
</tr>
<tr>
<td></td>
<td>Join</td>
<td><code>", ".join(list)</code></td>
<td><pre><code class="language-python">", ".join(["a", "b", "c"])</code></pre></td>
</tr>
<tr>
<td><strong>List Operations</strong></td>
<td>Append</td>
<td><code>list.append(item)</code></td>
<td><pre><code class="language-python">items = [1, 2]
items.append(3)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Extend</td>
<td><code>list.extend([1, 2])</code></td>
<td><pre><code class="language-python">items = [1]
items.extend([2, 3])</code></pre></td>
</tr>
<tr>
<td></td>
<td>Insert</td>
<td><code>list.insert(0, item)</code></td>
<td><pre><code class="language-python">items = [1, 2]
items.insert(0, 0)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Remove</td>
<td><code>list.remove(item)</code></td>
<td><pre><code class="language-python">items = [1, 2, 3]
items.remove(2)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Pop</td>
<td><code>list.pop()</code> / <code>list.pop(index)</code></td>
<td><pre><code class="language-python">items = [1, 2, 3]
items.pop()</code></pre></td>
</tr>
<tr>
<td></td>
<td>Index</td>
<td><code>list.index(item)</code></td>
<td><pre><code class="language-python">items = [1, 2, 3]
items.index(2)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Count</td>
<td><code>list.count(item)</code></td>
<td><pre><code class="language-python">items = [1, 2, 2, 3]
items.count(2)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Sort</td>
<td><code>list.sort()</code> / <code>sorted(list)</code></td>
<td><pre><code class="language-python">items = [3, 1, 2]
items.sort()</code></pre></td>
</tr>
<tr>
<td></td>
<td>Reverse</td>
<td><code>list.reverse()</code></td>
<td><pre><code class="language-python">items = [1, 2, 3]
items.reverse()</code></pre></td>
</tr>
<tr>
<td></td>
<td>List comprehension</td>
<td><code>[x*2 for x in list]</code></td>
<td><pre><code class="language-python">[x*2 for x in [1, 2, 3]]</code></pre></td>
</tr>
<tr>
<td><strong>Dictionary Operations</strong></td>
<td>Access value</td>
<td><code>dict["key"]</code></td>
<td><pre><code class="language-python">person = {"name": "Alice"}
person["name"]</code></pre></td>
</tr>
<tr>
<td></td>
<td>Get with default</td>
<td><code>dict.get("key", default)</code></td>
<td><pre><code class="language-python">person.get("age", 25)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Add key-value</td>
<td><code>dict["new"] = "value"</code></td>
<td><pre><code class="language-python">person["city"] = "NY"</code></pre></td>
</tr>
<tr>
<td></td>
<td>Remove key</td>
<td><code>del dict["key"]</code></td>
<td><pre><code class="language-python">del person["age"]</code></pre></td>
</tr>
<tr>
<td></td>
<td>Get keys</td>
<td><code>dict.keys()</code></td>
<td><pre><code class="language-python">list(person.keys())</code></pre></td>
</tr>
<tr>
<td></td>
<td>Get values</td>
<td><code>dict.values()</code></td>
<td><pre><code class="language-python">list(person.values())</code></pre></td>
</tr>
<tr>
<td></td>
<td>Get items</td>
<td><code>dict.items()</code></td>
<td><pre><code class="language-python">list(person.items())</code></pre></td>
</tr>
<tr>
<td></td>
<td>Check key exists</td>
<td><code>"key" in dict</code></td>
<td><pre><code class="language-python">"name" in person</code></pre></td>
</tr>
<tr>
<td><strong>Control Flow</strong></td>
<td>If statement</td>
<td><code>if condition:</code></td>
<td><pre><code class="language-python">if x > 5:</code></pre></td>
</tr>
<tr>
<td></td>
<td>If-else</td>
<td><code>if condition: else:</code></td>
<td><pre><code class="language-python">if x > 5:
    print("Big")</code></pre></td>
</tr>
<tr>
<td></td>
<td>If-elif-else</td>
<td><code>if: elif: else:</code></td>
<td><pre><code class="language-python">if x > 5:
    print("Big")
elif x < 0:
    print("Small")</code></pre></td>
</tr>
<tr>
<td></td>
<td>For loop</td>
<td><code>for item in iterable:</code></td>
<td><pre><code class="language-python">for i in range(5):</code></pre></td>
</tr>
<tr>
<td></td>
<td>While loop</td>
<td><code>while condition:</code></td>
<td><pre><code class="language-python">while x < 10:
    x += 1</code></pre></td>
</tr>
<tr>
<td></td>
<td>Break</td>
<td><code>break</code></td>
<td><pre><code class="language-python">for i in range(10):
    if i == 5:
        break</code></pre></td>
</tr>
<tr>
<td></td>
<td>Continue</td>
<td><code>continue</code></td>
<td><pre><code class="language-python">for i in range(10):
    if i % 2 == 0:
        continue</code></pre></td>
</tr>
<tr>
<td><strong>Loops & Iteration</strong></td>
<td>Range</td>
<td><code>for i in range(5):</code></td>
<td><pre><code class="language-python">for i in range(5):</code></pre></td>
</tr>
<tr>
<td></td>
<td>Enumerate</td>
<td><code>for i, val in enumerate(list):</code></td>
<td><pre><code class="language-python">for i, val in enumerate(["a", "b", "c"]):</code></pre></td>
</tr>
<tr>
<td></td>
<td>Zip</td>
<td><code>for a, b in zip(list1, list2):</code></td>
<td><pre><code class="language-python">for a, b in zip([1, 2], [3, 4]):</code></pre></td>
</tr>
<tr>
<td></td>
<td>List comprehension</td>
<td><code>[x for x in list if condition]</code></td>
<td><pre><code class="language-python">[x for x in range(10) if x % 2 == 0]</code></pre></td>
</tr>
<tr>
<td><strong>Functions</strong></td>
<td>Define function</td>
<td><code>def func_name(params):</code></td>
<td><pre><code class="language-python">def greet(name):</code></pre></td>
</tr>
<tr>
<td></td>
<td>Return value</td>
<td><code>return value</code></td>
<td><pre><code class="language-python">return "Hello"</code></pre></td>
</tr>
<tr>
<td></td>
<td>Default parameter</td>
<td><code>def func(param="default"):</code></td>
<td><pre><code class="language-python">def greet(name="World"):</code></pre></td>
</tr>
<tr>
<td></td>
<td>*args</td>
<td><code>def func(*args):</code></td>
<td><pre><code class="language-python">def sum_all(*args):</code></pre></td>
</tr>
<tr>
<td></td>
<td>**kwargs</td>
<td><code>def func(**kwargs):</code></td>
<td><pre><code class="language-python">def create(**kwargs):</code></pre></td>
</tr>
<tr>
<td></td>
<td>Lambda</td>
<td><code>lambda x: x*2</code></td>
<td><pre><code class="language-python">square = lambda x: x**2</code></pre></td>
</tr>
<tr>
<td></td>
<td>Map</td>
<td><code>list(map(func, iterable))</code></td>
<td><pre><code class="language-python">list(map(str, [1, 2, 3]))</code></pre></td>
</tr>
<tr>
<td></td>
<td>Filter</td>
<td><code>list(filter(func, iterable))</code></td>
<td><pre><code class="language-python">list(filter(lambda x: x > 5, [1, 2, 3, 6]))</code></pre></td>
</tr>
<tr>
<td></td>
<td>Reduce</td>
<td><code>from functools import reduce</code></td>
<td><pre><code class="language-python">reduce(lambda x, y: x + y, [1, 2, 3])</code></pre></td>
</tr>
<tr>
<td><strong>File Operations</strong></td>
<td>Open file</td>
<td><code>open("file.txt", "r")</code></td>
<td><pre><code class="language-python">open("data.txt", "r")</code></pre></td>
</tr>
<tr>
<td></td>
<td>Read file</td>
<td><code>file.read()</code> / <code>file.readlines()</code></td>
<td><pre><code class="language-python">file.read()</code></pre></td>
</tr>
<tr>
<td></td>
<td>Write file</td>
<td><code>file.write("text")</code></td>
<td><pre><code class="language-python">file.write("Hello")</code></pre></td>
</tr>
<tr>
<td></td>
<td>Close file</td>
<td><code>file.close()</code></td>
<td><pre><code class="language-python">file.close()</code></pre></td>
</tr>
<tr>
<td></td>
<td>Context manager</td>
<td><code>with open("file") as f:</code></td>
<td><pre><code class="language-python">with open("data.txt", "r") as f:</code></pre></td>
</tr>
<tr>
<td><strong>Exception Handling</strong></td>
<td>Try-except</td>
<td><code>try: except:</code></td>
<td><pre><code class="language-python">try:
    x = 1/0
except:
    print("Error")</code></pre></td>
</tr>
<tr>
<td></td>
<td>Specific exception</td>
<td><code>except ValueError:</code></td>
<td><pre><code class="language-python">except ValueError as e:</code></pre></td>
</tr>
<tr>
<td></td>
<td>Finally</td>
<td><code>finally:</code></td>
<td><pre><code class="language-python">finally:
    print("Done")</code></pre></td>
</tr>
<tr>
<td></td>
<td>Raise exception</td>
<td><code>raise Exception("msg")</code></td>
<td><pre><code class="language-python">raise ValueError("Invalid input")</code></pre></td>
</tr>
<tr>
<td><strong>Classes & OOP</strong></td>
<td>Define class</td>
<td><code>class MyClass:</code></td>
<td><pre><code class="language-python">class Dog:</code></pre></td>
</tr>
<tr>
<td></td>
<td>Constructor</td>
<td><code>def __init__(self):</code></td>
<td><pre><code class="language-python">def __init__(self, name):</code></pre></td>
</tr>
<tr>
<td></td>
<td>Method</td>
<td><code>def method(self):</code></td>
<td><pre><code class="language-python">def bark(self):</code></pre></td>
</tr>
<tr>
<td></td>
<td>Inheritance</td>
<td><code>class Child(Parent):</code></td>
<td><pre><code class="language-python">class Puppy(Dog):</code></pre></td>
</tr>
<tr>
<td></td>
<td>Super class</td>
<td><code>super().__init__()</code></td>
<td><pre><code class="language-python">super().__init__()</code></pre></td>
</tr>
<tr>
<td></td>
<td>Class variable</td>
<td><code>class_var = value</code></td>
<td><pre><code class="language-python">species = "Canis"</code></pre></td>
</tr>
<tr>
<td></td>
<td>Instance variable</td>
<td><code>self.var = value</code></td>
<td><pre><code class="language-python">self.name = "Buddy"</code></pre></td>
</tr>
<tr>
<td><strong>Modules</strong></td>
<td>Import module</td>
<td><code>import module</code></td>
<td><pre><code class="language-python">import math</code></pre></td>
</tr>
<tr>
<td></td>
<td>Import specific</td>
<td><code>from module import func</code></td>
<td><pre><code class="language-python">from math import sqrt</code></pre></td>
</tr>
<tr>
<td></td>
<td>Import with alias</td>
<td><code>import module as alias</code></td>
<td><pre><code class="language-python">import numpy as np</code></pre></td>
</tr>
<tr>
<td></td>
<td>Import all</td>
<td><code>from module import *</code></td>
<td><pre><code class="language-python">from math import *</code></pre></td>
</tr>
<tr>
<td><strong>String Formatting</strong></td>
<td>f-string</td>
<td><code>f"Value: {var}"</code></td>
<td><pre><code class="language-python">f"Name: {name}"</code></pre></td>
</tr>
<tr>
<td></td>
<td>Format method</td>
<td><code>"{}".format(var)</code></td>
<td><pre><code class="language-python">"Hello {}".format(name)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Percent formatting</td>
<td><code>"%s" % var</code></td>
<td><pre><code class="language-python">"Hello %s" % name</code></pre></td>
</tr>
<tr>
<td><strong>Math Operations</strong></td>
<td>Power</td>
<td><code>2 ** 3</code></td>
<td><pre><code class="language-python">2 ** 3</code></pre></td>
</tr>
<tr>
<td></td>
<td>Floor division</td>
<td><code>7 // 2</code></td>
<td><pre><code class="language-python">7 // 2</code></pre></td>
</tr>
<tr>
<td></td>
<td>Modulo</td>
<td><code>7 % 2</code></td>
<td><pre><code class="language-python">7 % 2</code></pre></td>
</tr>
<tr>
<td></td>
<td>Absolute value</td>
<td><code>abs(-5)</code></td>
<td><pre><code class="language-python">abs(-5)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Round</td>
<td><code>round(3.14159, 2)</code></td>
<td><pre><code class="language-python">round(3.14159, 2)</code></pre></td>
</tr>
<tr>
<td></td>
<td>Max/Min</td>
<td><code>max([1, 2, 3])</code> / <code>min([1, 2, 3])</code></td>
<td><pre><code class="language-python">max([1, 2, 3])</code></pre></td>
</tr>
<tr>
<td></td>
<td>Sum</td>
<td><code>sum([1, 2, 3])</code></td>
<td><pre><code class="language-python">sum([1, 2, 3])</code></pre></td>
</tr>
<tr>
<td><strong>Boolean Operations</strong></td>
<td>And</td>
<td><code>condition1 and condition2</code></td>
<td><pre><code class="language-python">x > 5 and x < 10</code></pre></td>
</tr>
<tr>
<td></td>
<td>Or</td>
<td><code>condition1 or condition2</code></td>
<td><pre><code class="language-python">x < 5 or x > 10</code></pre></td>
</tr>
<tr>
<td></td>
<td>Not</td>
<td><code>not condition</code></td>
<td><pre><code class="language-python">not x > 5</code></pre></td>
</tr>
<tr>
<td></td>
<td>Comparison</td>
<td><code>==</code>, <code>!=</code>, <code>></code>, <code><</code>, <code>>=</code>, <code><=</code></td>
<td><pre><code class="language-python">x == 5
x != 5
x > 5</code></pre></td>
</tr>
<tr>
<td><strong>Type Conversion</strong></td>
<td>To string</td>
<td><code>str(123)</code></td>
<td><pre><code class="language-python">str(123)</code></pre></td>
</tr>
<tr>
<td></td>
<td>To integer</td>
<td><code>int("123")</code></td>
<td><pre><code class="language-python">int("123")</code></pre></td>
</tr>
<tr>
<td></td>
<td>To float</td>
<td><code>float("3.14")</code></td>
<td><pre><code class="language-python">float("3.14")</code></pre></td>
</tr>
<tr>
<td></td>
<td>To list</td>
<td><code>list("abc")</code></td>
<td><pre><code class="language-python">list("abc")</code></pre></td>
</tr>
<tr>
<td></td>
<td>To tuple</td>
<td><code>tuple([1, 2, 3])</code></td>
<td><pre><code class="language-python">tuple([1, 2, 3])</code></pre></td>
</tr>
<tr>
<td></td>
<td>To set</td>
<td><code>set([1, 2, 2])</code></td>
<td><pre><code class="language-python">set([1, 2, 2])</code></pre></td>
</tr>
</tbody>
</table>

## Essential Code Snippets

### 1. String Manipulation

{% include codeHeader.html %}
```python
text = "  Hello, World!  "

# Remove whitespace
clean = text.strip()

# Convert to uppercase
upper = text.upper()

# Replace text
replaced = text.replace("World", "Python")

# Split into words
words = text.split(", ")

# Join list into string
joined = "-".join(["Python", "is", "awesome"])

print(clean, upper, replaced, words, joined)
```

### 2. List Comprehensions

{% include codeHeader.html %}
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]

# With condition
evens = [x for x in range(20) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(3)] for i in range(3)]

print(squares)    # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(evens)      # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
print(matrix)     # [[0, 0, 0], [0, 1, 2], [0, 2, 4]]
```

### 3. Dictionary Operations

{% include codeHeader.html %}
```python
# Create dictionary
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}

# Access values
print(person["name"])           # Alice
print(person.get("country", "USA"))  # USA (default)

# Update dictionary
person["age"] = 31
person["email"] = "alice@example.com"

# Iterate through dictionary
for key, value in person.items():
    print(f"{key}: {value}")

# Get keys and values
keys = list(person.keys())
values = list(person.values())
```

### 4. File Handling

{% include codeHeader.html %}
```python
# Reading a file
with open("input.txt", "r") as file:
    content = file.read()
    lines = file.readlines()

# Writing to a file
with open("output.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is a new line.")

# Appending to a file
with open("log.txt", "a") as file:
    file.write(f"Log entry: {datetime.now()}\n")

# Reading CSV file
import csv
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)
```

### 5. Exception Handling

{% include codeHeader.html %}
```python
try:
    # Code that might raise an exception
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError as e:
    print(f"Invalid value: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
else:
    print("No exceptions occurred")
finally:
    print("This always executes")

# Raising custom exceptions
def validate_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age
```

### 6. Working with Sets

{% include codeHeader.html %}
```python
# Create sets
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

# Set operations
union = set1 | set2           # {1, 2, 3, 4, 5, 6}
intersection = set1 & set2     # {3, 4}
difference = set1 - set2        # {1, 2}
symmetric_diff = set1 ^ set2    # {1, 2, 5, 6}

# Set methods
set1.add(5)
set1.remove(1)
set1.discard(10)  # No error if not exists
is_member = 3 in set1  # True
```

### 7. Lambda Functions

{% include codeHeader.html %}
```python
# Simple lambda
square = lambda x: x ** 2
print(square(5))  # 25

# Lambda with multiple arguments
add = lambda x, y: x + y
print(add(3, 4))  # 7

# Lambda with conditional
is_even = lambda x: x % 2 == 0
print(is_even(4))  # True

# Using lambda with map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Using lambda with filter
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]
```

### 8. Class Basics

{% include codeHeader.html %}
```python
class Dog:
    # Class variable
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance variables
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def birthday(self):
        self.age += 1
        return f"{self.name} is now {self.age} years old"

# Create instance
my_dog = Dog("Buddy", 3)
print(my_dog.bark())        # Buddy says Woof!
print(my_dog.birthday())    # Buddy is now 4 years old
print(my_dog.species)        # Canis familiaris
```

### 9. Time and Date

{% include codeHeader.html %}
```python
from datetime import datetime, timedelta

# Current date and time
now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Create specific date
date = datetime(2026, 2, 17)
print(date)

# Date arithmetic
future = now + timedelta(days=7)
past = now - timedelta(hours=24)

# Parse string to date
date_str = "2026-02-17"
parsed = datetime.strptime(date_str, "%Y-%m-%d")

# Calculate difference
diff = future - now
print(f"Days until: {diff.days}")
```

### 10. Working with JSON

{% include codeHeader.html %}
```python
import json

# Dictionary to JSON
data = {
    "name": "Python",
    "version": 3.12,
    "features": ["easy", "powerful", "versatile"]
}

# Convert to JSON string
json_string = json.dumps(data, indent=2)
print(json_string)

# Parse JSON string
parsed = json.loads(json_string)
print(parsed["features"])

# Read JSON file
with open("data.json", "r") as file:
    data = json.load(file)

# Write JSON file
with open("output.json", "w") as file:
    json.dump(data, file, indent=2)
```

## Common Import Statements

| Purpose | Import Statement |
|---------|----------------|
| Math operations | `import math` |
| Random numbers | `import random` |
| Date and time | `from datetime import datetime` |
| Regular expressions | `import re` |
| JSON handling | `import json` |
| CSV files | `import csv` |
| File paths | `from pathlib import Path` |
| HTTP requests | `import requests` |
| Data analysis | `import pandas as pd` |
| Plotting | `import matplotlib.pyplot as plt` |
| NumPy arrays | `import numpy as np` |
| System operations | `import os` |
| Command line args | `import sys` |
| Time operations | `import time` |
| Copy objects | `import copy` |
| Collections | `from collections import defaultdict, Counter` |
| Itertools | `from itertools import count, cycle` |

## ⚡ Time-Saving Tips

### 1. Use f-strings for formatting
```python
# Old way
name = "Alice"
age = 25
print("Name: %s, Age: %d" % (name, age))

# Better way
print(f"Name: {name}, Age: {age}")
```

### 2. Use enumerate for index and value
```python
# Old way
items = ['a', 'b', 'c']
for i in range(len(items)):
    print(i, items[i])

# Better way
for i, item in enumerate(items):
    print(i, item)
```

### 3. Use zip for parallel iteration
```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age} years old")
```

### 4. Use context managers for files
```python
# Old way
file = open("data.txt", "r")
content = file.read()
file.close()

# Better way
with open("data.txt", "r") as file:
    content = file.read()
```

### 5. Use list comprehensions instead of loops
```python
# Old way
squares = []
for i in range(10):
    squares.append(i**2)

# Better way
squares = [i**2 for i in range(10)]
```

### 6. Use defaultdict for counting
```python
from collections import defaultdict

# Old way
counts = {}
for item in items:
    if item not in counts:
        counts[item] = 0
    counts[item] += 1

# Better way
counts = defaultdict(int)
for item in items:
    counts[item] += 1
```

### 7. Use Counter for frequency analysis
```python
from collections import Counter

words = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
word_counts = Counter(words)

print(word_counts.most_common(2))  # [('apple', 3), ('banana', 2)]
```

### 8. Use pathlib for file paths
```python
from pathlib import Path

# Old way
import os
path = os.path.join('folder', 'file.txt')

# Better way
path = Path('folder') / 'file.txt'

# Check if exists
if path.exists():
    content = path.read_text()
```

## Best Practices

1. **Use meaningful variable names**
   ```python
   # Bad
   x = 5
   y = 10
   
   # Good
   width = 5
   height = 10
   ```

2. **Write docstrings for functions**
   ```python
   def calculate_area(length, width):
       """Calculate the area of a rectangle.
       
       Args:
           length: The length of the rectangle
           width: The width of the rectangle
           
       Returns:
           The area of the rectangle
       """
       return length * width
   ```

3. **Use type hints (Python 3.5+)**
   ```python
   def greet(name: str) -> str:
       return f"Hello, {name}!"
   ```

4. **Handle exceptions gracefully**
   ```python
   try:
       result = risky_operation()
   except SpecificException as e:
       logger.error(f"Error: {e}")
       raise
   ```

5. **Use constants for magic numbers**
   ```python
   # Bad
   if temperature > 37:
       print("High temperature")
   
   # Good
   NORMAL_TEMP = 37
   if temperature > NORMAL_TEMP:
       print("High temperature")
   ```

## 📖 Common Mistakes to Avoid

1. **Modifying list while iterating**
   ```python
   # Bad - will cause issues
   items = [1, 2, 3, 4, 5]
   for item in items:
       if item % 2 == 0:
           items.remove(item)
   
   # Good - create new list
   items = [1, 2, 3, 4, 5]
   items = [item for item in items if item % 2 != 0]
   ```

2. **Using mutable default arguments**
   ```python
   # Bad - shared across calls
   def append_to(item, lst=[]):
       lst.append(item)
       return lst
   
   # Good - use None as default
   def append_to(item, lst=None):
       if lst is None:
           lst = []
       lst.append(item)
       return lst
   ```

3. **Comparing with None**
   ```python
   # Bad
   if x == None:
       pass
   
   # Good
   if x is None:
       pass
   ```

4. **Forgetting to use list() on map/filter**
   ```python
   # In Python 3, map returns iterator
   result = map(lambda x: x*2, [1, 2, 3])
   print(list(result))  # Need to convert to list
   ```

## 🔧 Debugging Tips

1. **Use print statements strategically**
   ```python
   print(f"Debug: x = {x}, y = {y}")
   ```

2. **Use pdb for interactive debugging**
   ```python
   import pdb; pdb.set_trace()
   ```

3. **Use type() to check variable types**
   ```python
   print(type(variable))
   ```

4. **Use dir() to explore objects**
   ```python
   print(dir(my_object))
   ```

5. **Use help() for documentation**
   ```python
   help(str.upper)
   ```

## Conclusion

This Python cheatsheet covers the most essential concepts and operations you'll use daily as a Python developer. Keep this guide bookmarked and refer to it whenever you need a quick reminder. The more you practice these patterns, the more natural they'll become.

Remember, the best way to learn Python is by writing code. Start with small projects, experiment with these concepts, and gradually build up your skills. Happy coding!

## Related Posts

- [Python Tips and Tricks You Must Know - 10 Essential Techniques]({{ site.baseurl }}{% post_url 2024-05-21-Python-tips-you-must-know %})
- [75+ Good Python Coding Examples for Software Development - Best Practices]({{ site.baseurl }}{% post_url 2023-02-12-Quick-Python-Tips %})
- [Learn Python Part 01 - Complete Beginner's Guide with Examples]({{ site.baseurl }}{% post_url 2022-05-28-Learn-Python-Part-01 %})
- [How to Earn Money Online Using Python Programming Skills - 10 Proven Ways]({{ site.baseurl }}{% post_url 2024-01-30-How-to-earn-money-online-using-python-programming-skills %})
