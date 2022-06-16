---
layout: post
title: Learn Python Tips and Tricks Part 01
mathjax: true
summary:  Learn Python with Examples
---

Hi friends! We are starting free Python learning course. If you are new to Python and want to quickly grasp the knowledge of this language, then this course is for you. 

* [Python Installation](#python-installation)
* [Run basic python](#run-basic-python)
* [Run basic python through file](#run-basic-python-through-file)
    * [main.py](#main.py)
* [Variables and Data types in Python](#variables-and-data-types-in-python)
    * [Strings](#strings)
    * [Let's play with string](#let's-play-with-string)
        * [How to add two strings in Python?](#how-to-add-two-strings-in-python)
        * [How to access first element of string variable in Python?](#how-to-access-first-element-of-string-variable-in-python)
        * [How to access last element of string variable in Python?](#how-to-access-last-element-of-string-variable-in-python)
        * [How to reverse a string variable in Python?](#how-to-reverse-a-string-variable-in-python)
        * [How to print a string multiple times by using integer?](#how-to-print-a-string-multiple-times-by-using-integer)
        * [How to find a string in a string in Python?](#how-to-find-a-string-in-a-string-in-python)
        * [How to replace a letter in string with another letter in Python?](#how-to-replace-a-letter-in-string-with-another-letter-in-python)
        * [How to set upper case of a string in Python?](#how-to-set-upper-case-of-a-string-in-python)
        * [How to set lower case of a string in Python?](#how-to-set-lower-case-of-a-string-in-python)
        * [How to check if a string starts with some letter in Python?](#how-to-check-if-a-string-starts-with-some-letter-in-python)
        * [How to check if a string ends with some letter in Python?](#how-to-check-if-a-string-ends-with-some-letter-in-python)
        * [How to find the highest index of occurence of a letter in a string in Python?](#how-to-find-the-highest-index-of-occurence-of-a-letter-in-a-string-in-python)
        * [How to count occurences of a letter in string in Python?](#how-to-count-occurences-of-a-letter-in-string-in-python)
        * [How to convert other data types to string in Python?](#how-to-convert-other-data-types-to-string-in-python)
    * [integers](#integers)
    * [floats - 2.34](#floats---2.34)
    * [booleans - `True` or `False`](#booleans---`true`-or-`false`)
* [Summary](#summary)

# Python Installation

Go ahead and download Python from here https://www.python.org/downloads/ according to your Operating System, i.e., Windows, Mac OS or Linux, etc. We will use Python 3.8.5 version. 

# Run basic python

Once installed, open up a new Terminal and write `python` and hit `Enter` key. You will see the following Python interactive interpreter 

```python
Python 3.8.5 (v3.8.5:580fbb018f, Jul 20 2020, 12:11:27) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 

```

Let's start by importing `this` in Python to print the Golden Rules of Software Engineering.

```python
>>> import this
```

Output:
```text
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

# Run basic python through file

Let's do the same thing by writing the following code in a new file named `main.py` in any location or path in your computer. Open up a new Terminal and cd to the path.



## main.py
{% include codeHeader.html %}
```python
import this
print (this)
```
Then type `python main.py` in the Terminal and hit Enter to get following output.

Output:
```text
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

# Variables and Data types in Python

Everything in Python is object including the variables and data types. The basic data types in Python are:
* strings - 'hello'
* integers -  234
* floats - 2.34
* booleans - `True` or `False`

## Strings
Let's check a variable that contains string data type

```python
>>> a = 'hello'
>>> type(a)
<class 'str'>
```
Variable `a` contains a string of five elements `hello` as `h` `e` `l` `l` `o`. So its length or `len` is 5

```python
>>> len(a)
5
```

## Let's play with string

Lets play with string variable.

### How to add two strings in Python?

```python
>>> a
'hello'
>>> a + a
'hellohello'
```

### How to access first element of string variable in Python?

```python
>>> a
'hello'
>>> a[0]
'h'
```
### How to access last element of string variable in Python?

```python
>>> a
'hello'
>>> a[-1]
'o'
```

### How to reverse a string variable in Python?

Method 1 - Using Slice notation (Fastest way)

```python
>>> a
'hello'
>>> a[::-1]
'olleh'
```

Method 2 - Using join

```python
>>> a
'hello'
>>> ''.join(reversed(a))
'olleh'
```

Method 3 - Using a function

```python
>>> a
'hello'
>>> def reverse(a): return a[0] if len(a)==1 else a[len(a)-1] + reverse(a[0:len(a)-1])
... 
>>> reverse(a)
'olleh'
```

### How to print a string multiple times by using integer?

```python
>>> a
'hello'
>>> a*2
'hellohello'
>>> a*4
'hellohellohellohello'
>>> a*6
'hellohellohellohellohellohello'
>>> a*8
'hellohellohellohellohellohellohellohello'
```

### How to find a string in a string in Python?

```python
>>> a
'hello'
>>> a.find('e')
1
>>> a.find('o')
4
>>> a.find('u')
-1
```
As `u` is not inside string `hello`, the `find` function will return `-1`

### How to replace a letter in string with another letter in Python?

```python
>>> a
'hello'
>>> a.replace('o', 'O')
'hellO'
```

### How to set upper case of a string in Python?

```python
>>> a
'hello'
>>> a.upper()
'HELLO'
>>> a
'hello'
```
Note that calling `.upper()` returns the caps but did not change the value of `a`.

### How to set lower case of a string in Python?

```python
>>> a
'hello'
>>> b=a.upper()
>>> b
'HELLO'
>>> b=b.lower()
>>> b
'hello'
```

### How to check if a string starts with some letter in Python?

```python
>>> a
'hello'
>>> a.startswith('h')
True
>>> a.startswith('w')
False
>>> a.startswith('hel')
True
```

### How to check if a string ends with some letter in Python?

```python
>>> a
'hello'
>>> a.endswith('o')
True
>>> a.endswith('llo')
True
>>> a.endswith('hello')
True
>>> a.endswith('z')
False
```

### How to find the highest index of occurence of a letter in a string in Python?

To find the last occurence or highest index of a sub-string in a string.

```python
>>> a ='a quick brown fox jumps over the lazy dog'
>>> a.find('a')
0
>>> a.rfind('a')
34
>>> a.find('o')
10
>>> a.rfind('o')
39
```

### How to count occurences of a letter in string in Python?

```python
>>> a
'a quick brown fox jumps over the lazy dog'
>>> a.count('a')
2
>>> a.count('o')
4
>>> a.count('z')
1
```

### How to convert other data types to string in Python?

```python
>>> num =2
>>> type(num)
<class 'int'>
>>> str(num)
'2'
>>> num =2.4
>>> type(num)
<class 'float'>
>>> str(num)
'2.4'
>>> num = True
>>> type(num)
<class 'bool'>
>>> str(num)
'True'
```

## integers

```python
>>> num = 2
>>> type(num)
<class 'int'>
```

## floats - 2.34

```python
>>> num = 2
>>> b= float(num)
>>> b
2.0
>>> type(b)
<class 'float'>

```
## booleans - `True` or `False`

```python
>>> a = True
>>> type(a)
<class 'bool'>
>>> a = False
>>> type(a)
<class 'bool'>
```


# Summary

We saw that we can either run Python script via interpretor or using python file. Throughout this course we will continue to use anyone of them...
