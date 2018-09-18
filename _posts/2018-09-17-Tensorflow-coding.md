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

