---
layout: post
title: Importance of One Hot Encoding
categories: [tutorial]
mathjax: true
summary: Why One Hot Encoding is important for Neural Netwoks
---

## What is ONE HOT ENCODING?

An efficient method of encoding the classes to train a network.

### One Hot Encoding
```python
[1,0,0]: Class A
[0,1,0]: Class B
[0,0,1]: Class C
```

### Efficient Encoding
```python
0: Class A
1: Class B
2: Class C
```
In neural networks when we need to pick a class from classes, we have output nodes equal to the number of classes. 
Each node shows the probability that it may matches Class A, Class B or Class C.
It may look like this:
Class A has probability equal to 0.1
Class B has probability equal to 0.2
Class C has probability equal to 0.7

Although it looks inefficient from the storage perspective, but it is very efficient for the training. 
Moreover, it complements argmax function, that saves us writing a lot of code.
