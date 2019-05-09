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
Here find loc: [here](../master/assets/img/map2.html)







<html>
<head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.3/leaflet.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.3/leaflet.css" />
  <style>
    #map650cc3aba7164be8a91f5506dfaf3294 {
      height:100%;
    }
  </style> 
</head>
<body>
<button onclick="getLocation()">Get Position</button>
  <div id="map650cc3aba7164be8a91f5506dfaf3294"></div>

<p id="demo"></p>
<script text="text/javascript">
var x = document.getElementById("demo");
//function to getLocation using HTML5 geolocation
function getLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(showPosition);
    } else { 
        x.innerHTML = "Geolocation is not supported by this browser.";
    }
}
//positions passed here, check lat/long and show it to user 
function showPosition(position) {
    x.innerHTML = "Latitude: " + position.coords.latitude + 
    "<br>Longitude: " + position.coords.longitude;
	
	var blob = new Blob([ position.coords.latitude+","+position.coords.longitude], {type: "text/plain;charset=utf-8"});
	<!-- saveAs(blob, "MYLOCFILE.csv"); -->
	var map = L.map('map650cc3aba7164be8a91f5506dfaf3294');
L.tileLayer(
  "http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  {maxZoom:19, attribution: '<a href="https://github.com/jwass/mplleaflet">mplleaflet</a> | Map data (c) <a href="http://openstreetmap.org">OpenStreetMap</a> contributors'}).addTo(map);
var gjData = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [position.coords.longitude, position.coords.latitude]}, "properties": {"html": "<svg width=\"8px\" height=\"8px\" viewBox=\"-4.0 -4.0 8.0 8.0\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">  <path d=\"M 0.0 -3.0 C 0.7956093000000001 -3.0 1.5587396123545605 -2.683901074764725 2.121320343559643 -2.121320343559643 C 2.683901074764725 -1.5587396123545605 3.0 -0.7956093000000001 3.0 -0.0 C 3.0 0.7956093000000001 2.683901074764725 1.5587396123545605 2.121320343559643 2.121320343559643 C 1.5587396123545605 2.683901074764725 0.7956093000000001 3.0 0.0 3.0 C -0.7956093000000001 3.0 -1.5587396123545605 2.683901074764725 -2.121320343559643 2.121320343559643 C -2.683901074764725 1.5587396123545605 -3.0 0.7956093000000001 -3.0 -0.0 C -3.0 -0.7956093000000001 -2.683901074764725 -1.5587396123545605 -2.121320343559643 -2.121320343559643 C -1.5587396123545605 -2.683901074764725 -0.7956093000000001 -3.0 0.0 -3.0 Z\" stroke=\"#0000FF\" stroke-width=\"1.0\" stroke-opacity=\"1\" fill=\"#0000FF\" fill-opacity=\"1\" /></svg>", "anchor_x": 4.0, "anchor_y": 4.0}}]};
if (gjData.features.length != 0) {
  var gj = L.geoJson(gjData, {
    style: function (feature) {
      return feature.properties;
    },
    pointToLayer: function (feature, latlng) {
      var icon = L.divIcon({'html': feature.properties.html, 
        iconAnchor: [feature.properties.anchor_x, 
                     feature.properties.anchor_y], 
          className: 'empty'});  // What can I do about empty?
      return L.marker(latlng, {icon: icon});
    }
  });
  gj.addTo(map);
  
  map.fitBounds(gj.getBounds());
} else {
  map.setView([0, 0], 1);
}
}
</script>
</body>
</html>
















The distribution of values in the tensor looks like this, obviously, the standard
deviation is 0.5.

![]({{ "assets/img/tensor flow/tensor_distribution.png" | absolute_url }})













