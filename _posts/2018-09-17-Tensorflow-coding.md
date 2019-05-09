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

Therefore, in order to make a multiply graph we must initialize `b` as `2.0`

```python
b = tf.constant(2.0)
```
And now 

`>>>multiplication = tf.multiply(a,b)`

It will not generate an error, lets run this new floating point graph with `mySession`.

`>>> mySess.run(multiplication)`

`4.4`

## Creating TF Variables

The tensorflow parameters are represented by the `tf.Variable`. These
variables are in-memory buffers and contain the tensors or matrices.
The difference between these tensors with other tensors, is their existence
throughout the session run. Other tensors are only instantiated at the 
session run and then cleaned away when the graph is run. Therefore, the 
TF variables has three important properties.

1) TF variables must be initialized for the first time.
2) TF variables can be modified during the parameter optimization.
3) The values in TF variables can be stored and restored in the disk.

Lets create a TF variable that represent weights connecting neurons between two layers of a feed forward neural network:

```python
Weights = tf.Variable(tf.random_normal([100,200],stddev=0.5),name="Weights")
```
Lets check what is its description:

`>>> Weights`

`<tf.Variable 'Weights:0' shape=(100, 200) dtype=float32_ref>`

It means we have created a tensor of rows 100 and colums 200 and the values inside
this tensor is of normal distribution with the standard deviation of 0.5. This tensor
can be connected between a layer with output of 100 and a layer with input of size
200. The name paramter is a unique identifier, by default the tensor is trainable. If we 
dont want it to be trained then, we must pass a flag `trainable=False`.

## View the values inside a tf.Variable 

Lets have a look at the values of above tensor which is represented by `Weights`.

```python

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

Weights = tf.Variable(tf.random_normal([100, 200], stddev=0.5,name="Weights"))
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	v = sess.run(Weights)
	print(v)  # will print the Weights tensor
	plt.imshow(v)
	plt.colorbar()
	plt.show()
	
	
	v.sort()
	hmean = np.mean(v) # take the mean
	
	hstd = np.std(v) # get the standard deviation
	pdf = stats.norm.pdf(v, hmean, hstd)
	plt.plot(v,pdf) # including h here is crucial)
	plt.show()
```
The Weights tensor looks like this with values with standard deviation of 

![]({{ "assets/img/tensor flow/tensor_values.png" | absolute_url }})
Here find loc: [here](http://htmlpreview.github.io/?https://github.com/py2ai/py2ai.github.io/blob/master/assets/img/map2.html)



The distribution of values in the tensor looks like this, obviously, the standard
deviation is 0.5.

![]({{ "assets/img/tensor flow/tensor_distribution.png" | absolute_url }})













