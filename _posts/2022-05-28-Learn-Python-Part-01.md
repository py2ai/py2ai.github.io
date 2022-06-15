---
layout: post
title: Learn Python Tips and Tricks Part 01
mathjax: true
summary:  Learn Python with Examples
---

Hi friends! We are starting free Python learning course. If you are new to Python and want to quickly grasp the knowledge of this language, then this course is for you. Go ahead and download Python from here https://www.python.org/downloads/ according to your Operating System, i.e., Windows, Mac OS or Linux, etc. We will use Python 3.8.5 version. 

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
Let's do the same thing by writing the following code in a new file named `main.py` in any location or path in your computer. Open up a new Terminal and cd to the path.

### main.py
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

### Variables and Data types in Python

Everything in Python is object including the variables and data types. The basic data types in Python are:
* strings - 'hello'
* integers -  234
* floats - 2.34
* booleans - `True` or `False`

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

Lets play with string variable.

How to add two strings in Python?

```python
>>> a + a
'hellohello'
```

How to access first element of string variable in Python?

```python
>>> a[0]
'h'
```
How to access last element of string variable in Python?

```python
>>> a[-1]
'o'
```

How to reverse a string variable in Python?

```python
>>> a[::-1]
'olleh'
```






### Summary

We saw that we can either run Python script via interpretor or using python file. Throughout this course we will continue to use anyone of them...
