---
layout: post
title: TensorFlow Basics
author: Hussain A.
categories: [tutorial]
featured-img: TensorFlow
mathjax: true
summary: A quick tutorial about tensorflow
---




Hi everybody, today i will write about a little introduction to Tensorflow API. Tensorflow is a standard open source library released in 2015 by Google for building deep learning algorithms. 
Tensor can be considered as an N dimensional array. Tensors are defined by the unit of dimensionality which is known as rank. A scalar is a rank-zero tensor. A vector is rank-one tensor. A 2D matrix is a rank-two tensor and so on.

## Data Types in TensorFlow

Mulitple datatypes are supported by tensorflow such as; 

`tf.float32, tf.float64, tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8, tf.string, tf.bool`

In common TensorFlow automatically, manages the data types and we don’t need to specify the data type, unless there is a special requirement of memory utilization. For example if you only need 32 bit data type then there is no need to use the 64 bit, which will save the memory.

 
## How to build a deep learning Model?

Well building a deep learning model requires following steps:

1) First the dataset should be prepared.
2) The features and labels must be defined.
3) Encoding of the dependent variables should be done.
4) The dataset is then divided into two parts such as; train and test.
5) Then the Tensorflow structure hold these features and labels.
6) The model is built using the number of layers, neurons and other parameters.
7) The model is then trained in such a way that the mean square error, which is actually the difference between the actual output and the desired output, is minimized. Or minimize the loss.
8) Then after a specific value of the loss, the model is ready to predict the test inputs.




## Tensorflow Code

### 1) Buid a computational graph

```python

import tensorflow as tf
Neuron1 = tf.constant(8.0, tf.float32)
Neuron2 = tf.constant(3.0)
print(Neuron1,Neuron2)

```


### 2) Run the computational graph

Session places the graph operations on the devices such as CPUs and GPUs . Session provides methods to execute those graph operations.

```python
sess = tf.Session() # launch the graph and create a session object
print(sess.run([Neuron1,Neuron2]))
sess.close() # free up the resources
```

Another way to launch the session , and automatically close the session when it is done.


```python
with tf.Session() as less:
  output = sess.run([Neuron1,Neuron2])
  print(output)
```


## Example

```python
import Tensorflow as tf
a = tf.constant(8.0)
b = tf.constant(3.0)
c = a*b
sess = tf.Session()
print(sess.run(c))
sess.close()
```
## TensorBoard

Tensor board is used to visualize the computation graph. For this we need to create a FileWriter. 

`FileWriter = tf.summary.FileWriter(‘log_simple_graph’,sess.graph)` 
`sess.graph` is used to create the graph object.

Here the first argument `log_simple_graph` is the output directory name. 
After that the graph can be visualized by the command `tensorboard —logdir = “path/to/the/log_simple_graph”`

Launching this command will start a local web app on port 6006. 
The reason for choosing this port number is that it is inverted goog of the google:P.

In next tutorials i will write about building a deep learning model from scratch using tensorflow API. Have a nice day!











