---
categories:
- Python tutorial series
description: A quick tutorial on Microsoft Excel .xlsx file operations in python
featured-img: xls
keywords:
- file
- read
- write
- development
- code
- programming
- xlsx
- using
layout: post
mathjax: true
title: How to read and write in xlsx file using python
---

Welcome to Pyshine  
I wish and hope that you guyz are fine and enjoying good health

Lets import the required libraries

{% include codeHeader.html %}
```python
import xlwings as xw
import pandas as pd
import math
import numpy as np
```
xlwings can be installed using pip3 install xlwings

Lets have a look at the .xlsx [file](https://github.com/py2ai/Keras-Labs/blob/master/demodata.xlsx) we are going to read
we can read and assign it to the workbook

{% include codeHeader.html %}
```python
workbook = xw.Book(r'demodata.xlsx',"utf8")
```
Lets make sheet sht

{% include codeHeader.html %}
```python
sht = workbook.sheets('demodata')
```
Get the table in the Pandas data frame

{% include codeHeader.html %}
```python
df = sht.range('A1').options(pd.Series,expand='table').value
```
Its time to make a function getDataForKeyWord
Lets say we have a keyword in any column of the .xlsx table 
Imagine its Fri and we want to find the occurences of it in the specified
Column and then we want to find the respective values in any other column
So this function will input the key word to search and also the index of the 
column to search the key word , to output the specific values for that 
keyword and put those values to a dict whose keys are the keyword,index of keyword
and the value is value found in the target column 
Things will become much clear as we proceed
So lets make it

{% include codeHeader.html %}
```python
def getDataForKeyWord(keyword, keyword_column,target_column):
	INFO = df.iloc[:,keyword_column:1+keyword_column].values

	sumDictOf = {}
	nameList = []
	for i in range (len(INFO)):
		text = INFO[i][0]
		print(text)
		try:
			if text.find(keyword)!=-1: # It will be not equal to -1 if keyword is found
				index = text.find(keyword) # here we find the keyword in the column
				L = len(keyword)  # Lets say if keyword is Fri so L = 1+1+1=3
				print(i)
				if (math.isnan(df.iloc[i,target_column:1+target_column].values[0])):
					sumDictOf[(text[0:index+L],i)] = 0
					nameList.append(text[0:index+L])

				else:
					sumDictOf[(text[0:index+L],i)] = df.iloc[i,target_column:1+target_column].values[0]
					nameList.append(text[0:index+L])
		except:
			pass

	return sumDictOf,nameList

sumDictOf,nameList = getDataForKeyWord('Fri',1,3)
```

So above I just enter Fri for the input column 1:2 [weekdays] and the target column 3:4 [prices]
so lets see how the values are obtained 

{% include codeHeader.html %}
```python
print(sumDictOf)
print(nameList)
```
Let me close the .xlsx file and again run , it will auto open it
Lets change the keyword to search	
Good enough next we make a new .xlsx file to put two columns

{% include codeHeader.html %}
```python
wb = xw.Book()
xw.Range('A1').value = np.array(nameList).reshape(len(nameList),1) # first col
xw.Range('B1').value = np.array(list(sumDictOf.values())).reshape(len(list(sumDictOf.values())),1) # Second col
wb.save(r'output.xlsx')
``` 
Lets see , run it, close the .xlsx 
We see that the output file now has all selected keywords and their values.
Thats all for today.
Take care of yourself



	

