---
layout: post
title: Python tips and tricks you must know!
mathjax: true
featured-img: 26072022-python-logo
description:  Python tips and tricks
---

Python programming skills have seen a surge in demand lately. To assist you in honing your Python coding prowess, we've compiled some nifty Python techniques you can employ to enhance your code. Consider learning one new trick each day for the next month, and peruse our guide on Python best practices to ensure your code stands out.

If you find your Python skills lacking, you can also enhance them through our Python Skill Path.

# 1. Indexing Tricks

{% include codeHeader.html %}
```python
phrase = "Goodbye, world!"
print(phrase[::-1])

"""
!dlrow ,eybdooG
"""
```

Python indexing allows users to access a subset of a sequence. The index represents the position of an element in the sequence. While mutable sequences allow slicing for data extraction and modification, attempting to modify slices of immutable sequences raises a TypeError.

The slicing format in Python is sequence`[start:stop:step]`. If no values are specified for start, stop, and step, the sequence defaults to:

Start: 0
Stop: length of the sequence
Step: 1 (if not specified)
Negative indices can also be passed, facilitating sequence reversal. For instance, in a list with four elements, the 0th index is equivalent to the -4 index, and the last index corresponds to -1. Leveraging this knowledge, the given example code prints the string in reverse order.


# 2. Efficient Variable Swapping

{% include codeHeader.html %}
```python
x = 10
y = 5
print(f"Initial: {x, y}")

"""
Initial: (10, 5)
"""

x, y = y, x + 2
print(f"Swapped: {x, y}")

"""
Swapped: (5, 12)
"""
```

In Python, unpacking iterables into variables via automatic unpacking enables simultaneous assignment in a single line. Similarly, using the * operator allows collecting multiple values into a single variable, known as packing. The combination of automatic packing and unpacking gives rise to simultaneous assignment, streamlining the process of assigning values to multiple variables.

# 3. Set vs. List
{% include codeHeader.html %}
```python
import sys

set_ = {1, 2, 3, 4, 5}
list_ = [1, 2, 3, 4, 5]

print(f"Set size: {sys.getsizeof(set_)} bytes")
print(f"List size: {sys.getsizeof(list_)} bytes")

"""
Set size: 232 bytes
List size: 144 bytes
"""
```
While both sets and lists in Python are iterable and permit indexing, tuples offer distinct advantages over lists. Lists are mutable, allowing modifications, while tuples are immutable, preventing alterations and making them more memory-efficient. Tuples also offer faster performance compared to lists, making them preferable in scenarios where data immutability is desired.


# 4. Generator Functions
{% include codeHeader.html %}
```python
a = [x * 2 for x in range(10)]
b = (x * 2 for x in range(10))

print(a)
print(b)

"""
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
<generator object <genexpr> at 0x7f61f8808b50>
"""
```
List comprehensions serve as a pythonic way of generating lists from iterables, offering faster performance than traditional for loops. However, substituting square brackets [ ] with parentheses ( ) inadvertently creates a generator object, leveraging lazy evaluation to generate elements upon request, thereby conserving memory.


# 5. Object Aliasing
{% include codeHeader.html %}
```python
original = [1, 2, 3, 4, 5]
alias = original

# Modify alias
alias[4] = 7

print(id(original))
print(id(alias))
print(original)  # Changes reflect in the original list

"""
140279273485184
140279273485184
[1, 2, 3, 4, 7]
"""
```
In Python, every entity is an object, and assigning an object to an identifier creates a reference to that object. Consequently, assigning one identifier to another results in two identifiers referencing the same object, known as aliasing. Changes made to one alias affect the other, underscoring the importance of caution when dealing with mutable objects to avoid unintended side effects.

To create a distinct copy of the original object rather than a reference, one can utilize slicing `(b = a[:])` or other methods like `list(a)` or `copy()`.

# 6. Utilizing the ‘not’ Operator
{% include codeHeader.html %}
```python
empty_list = []
print(not empty_list)

"""
True
"""
```
Python's not operator serves as a concise means of checking whether a data structure is empty, returning True if the structure evaluates to False, and vice versa. It effectively inverts the truth value of Boolean expressions and objects, facilitating streamlined conditional checks and enhanced code readability.

Additionally, the not operator finds utility in conditional statements, simplifying logic flow by negating conditional expressions when evaluating truth values.

# 7. Enhanced String Formatting with F-strings
{% include codeHeader.html %}
```python
name = "Alice"
age = 25

print(f"Hello, I'm {name} and I'm {age} years old!")

"""
Hello, I'm Alice and I'm 25 years old!
"""
```
Python 3.6 introduced f-strings, offering a more concise, readable, and efficient method of string formatting compared to traditional methods like format() or string concatenation. F-strings streamline the process of embedding variables within strings, enhancing code clarity and reducing the likelihood of formatting errors.

Moreover, f-strings support advanced features like printing variable names alongside their values ({variable = value}), further enhancing their versatility and utility in Python programming.

# 8. Customizing Print Output with the ‘end’ Parameter

{% include codeHeader.html %}
```python
languages = ["English", "French", "Spanish", "German", "Twi"]
for language in languages:
    print(language, end=" ")

"""
English French Spanish German Twi 
"""
```
The print() function's end parameter offers flexibility in customizing output by specifying the character or string to append at the end of each print call. By default, end is set to "\n", causing the print function to terminate with a newline character. Customizing end enables printing multiple values on the same line or appending custom separators, enhancing output formatting and presentation.

# 9. Augmenting Tuples with Append Operations
{% include codeHeader.html %}
```python
tup = (1, 2, [1, 2, 3])
tup[2].append(4)
print(tup)

"""
(1, 2, [1, 2, 3, 4])
"""
```
Although tuples in Python are immutable, allowing no direct modifications, mutable objects within tuples can be altered. By leveraging the mutability of contained objects, one can append elements to lists embedded within tuples, effectively modifying tuple content indirectly while preserving tuple immutability.

While not a recommended practice due to potential confusion, this technique demonstrates the versatility of Python's data structures and the interplay between mutability and immutability.

# 10. Streamlining Dictionary Merging
{% include codeHeader.html %}
```python
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged_dict = {**dict1, **dict2}
print(merged_dict)

"""
{'a': 1, 'b': 2, 'c': 3, 'd': 4}
"""
```
Merging dictionaries in Python can be achieved using dictionary unpacking within dictionary literals. This concise syntax allows combining multiple dictionaries into a single dictionary, preserving key-value pairs while eliminating duplicates. This technique offers a streamlined approach to dictionary merging, enhancing code readability and maintainability.








