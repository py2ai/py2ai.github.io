---
layout: post
title: Basic Coding in TensorFlow
author: Hussain A.
categories: [tutorial]

mathjax: true
summary: Lets code in TensorFlow
---

Before we proceed the basic coding, please note that according to the tensorflow 1.0.0 release notes,
tf.mul, tf.sub and tf.neg are deprecated in favor of tf.multiply, tf.subtract and tf.negative. Other changes
can be found [here](https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md#breaking-changes-to-the-api).

Lets launch python and import tensorflow as tf.

```python 
import tensorflow as tf
```
Lets initialize a constant string name 'Hello TF World'

```python
text=tf.constant('Hello TF World')
```
Now just type `text` in the console and hit enter, which will result in:

## String Data Type

`>>> text
<tf.Tensor 'Const:0' shape=() dtype=string>`

This means that we have initialized a constant of string datatype.

## Int32 Data Type

Lets go ahead and make a simple multiplier that will multiply two constants.

```python
a = tf.constant(4)
b = tf.constant(2)
multiplication = tf.multiply(a,b)
```

Here a and b are TF constants of dtype=int32, and the result by defult is also of the same data type. Its time to start a
TF session and get the result of the multiplication. 

```python
mySess = tf.Session()
```
Lets run the `mySess` for the `multiplication` graph.

```python
mySess.run(multiplication)
```
and we get a result: 

`>>> mySess.run(multiplication)`

`8`

## Floating point Data Type

```python
a = tf.constant(2.2)
b = tf.constant(2)
```
Now we have `a` as  dtype=float32 and `b` as dtype=int32, therefore if we now write multiplication and run, we will get this error:

`>>>multiplication = tf.multiply(a,b)`

`TypeError: Input 'y' of 'Mul' Op has type int32 that does not match type float32 of argument 'x'.`

Therefore, in order to make a multiply graph we must initialize `b` as `2.0'

```python
b = tf.constant(2.0)
```
And now 

`>>>multiplication = tf.multiply(a,b)`

It will not generate an error, lets run this new floating point graph with `mySession`.

`>>> mySess.run(multiplication)`

`4.4`









