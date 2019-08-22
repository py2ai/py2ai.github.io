---
layout: post
title: Lab5 Training Classification model and Epochs
author: Hussain A.
categories: [tutorial]
mathjax: true
summary: A quick tutorial on Keras Classification
---






## A quick and easy Classification model

Hi there! today we will build a very simple classification model that should looks like this figure:
![]({{ "assets/img/posts/lab5_keras_model.png" | absolute_url }}). Lets import the necessary components. 
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras.utils import plot_model
import keras
import pandas as pd
pyplot.style.use('ggplot')

```






In this lab we have a total of three classes. So the output size of our model is 3 units i.e. a vector of three elements,
and each element represents the degree of classification per class. lets prepare the input data samples.
Our simple three input samples are [1,1,1,1], [2,2,2,2] and [3,3,3,3]. Lets stack them vertically and to make them as a 4x3 matrix in X.

```python
X = np.array([1, 1, 1, 1])
X2 = np.array([2, 2, 2, 2])
X3 = np.array([3, 3, 3, 3])
X =np.vstack((X,X2))
X =np.vstack((X,X3))
```

In this lab we have a total of three classes. Data sample [1,1,1,1] belongs to class 1 and its target represented by [0,0,1]. Similarly, [2,2,2,2] belongs to class 2 
represented by [0,1,0]. And the third input sample [3,3,3,3] belongs to class 3 as shown by [1,0,0]. I put them in this order just for
easy understanding since binary 001=1 and binary 010 =2 and binary 100 =3, but the order can be changed. Care should be taken to keep 
one hot encoded vector and each sample class corresponds to its column index class. The target y per sample X is shown below.


```python
num_classes=3

y = np.array([0,0,1])
y2 = np.array([0,1,0])
y3 = np.array([1,0,0])

y = np.vstack((y,y2))
y = np.vstack((y,y3))

```

Ok now we have our input samples and their target class vectors. Before training lets see how the shapes of X and y look like.
```python
print(X.shape,y.shape)
```

Lets create the model. Our loss function is categorical_crossentropy, the optimizer is Adam and the metrics to monitor accuracy is categorical_accuracy.
The input_dim of model is 4 because each data sample has four elements. The output of first fully connected Dense layer is 50 units. The output of next 
Dense layer is 3 because our target is vector of three elements for example [0,0,1]. 

```python
model = Sequential()

model.add(Dense(50,input_dim=4))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
plot_model(model, to_file='mymodel.png', show_shapes=True)
```


We initialize the Epochs to 0.
```python
Epochs = 0
```
Lets train the model by calling .fit function. We will also save each loss.csv file to analyse the training process. History is stored in history variable
and we kept the shuffle to False. You can set it to True to see what happens but thats an excercise for you. Epochs will be incremented by 100 as for loop
iterates.

```python
for i in range(10):
	Epochs+=100
	# # training
	history = model.fit(X, y, epochs=Epochs, batch_size=1, verbose=2,shuffle=False)
	df = pd.DataFrame.from_dict(history.history)
	df.to_csv(str(Epochs)+'-loss.csv', encoding='utf-8', index=False)
	
	pyplot.plot(history.history['loss'],label='Epochs: '+str(Epochs)+' categorical_crossentropy')
	
	# pyplot.plot(history.history['sparse_categorical_accuracy'],label='sparse_categorical_accuracy')
	pyplot.legend()
	
	pyplot.xlabel('Training Epoch')
	pyplot.ylabel('Loss Value')
	pyplot.legend()
	# pyplot.show()
	pyplot.savefig('Loss-epochs-{}.jpg'.format(Epochs)) 
	pyplot.close()
```


Well! its time to predict. How about checking for a data sample let say X = [1,1,1,1]. Its shape is (4,). So X.shape[0]= 4. And our model expects an input of 
4 elements, so lets reshape it as 1x4.

```python
X=np.array([1,1,1,1])
if X.shape[0]==4:
	X=np.reshape(X,(1,4))
```	
Lets predict its class. The output of model should be nearly or equal to [0,0,1]. I am using numpy round function to keep it vivid upto 2 decimal points instead of 
showing a very long fractional number.

```python
out = np.round_(model.predict(X),2)
print(out)
```



Here are the results of training process.

![]({{ "assets/img/posts/Loss-epochs-100.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-200.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-300.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-400.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-500.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-600.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-700.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-800.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-900.jpg" | absolute_url }})
![]({{ "assets/img/posts/Loss-epochs-1000.jpg" | absolute_url }})



Thats all for this lab 5. In the next lab 6 we will learn about training the CNN model. I hope this lab will be helpful for the beginners. Code is  [available](https://github.com/py2ai/Keras-Labs).
