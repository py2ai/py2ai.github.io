---
layout: post
title: How to read and write in xlsx file using python
author: Hussain A.
categories: [Python tutorial series]
mathjax: true
summary: A quick tutorial on Microsoft Excel .xlsx file operations in python
---


Welcome to Pyshine  
I wish and hope that you guyz are fine and enjoying good health

Lets import the required libraries
```python
import xlwings as xw
import pandas as pd
import math
import numpy as np
```
xlwings can be installed using pip3 install xlwings

Lets have a look at the .xlsx file we are going to read
we can read and assign it to the workbook
```python
workbook = xw.Book(r'demodata.xlsx',"utf8")
```
Lets make sheet sht
```python
sht = workbook.sheets('demodata')
```
Get the table in the Pandas data frame
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
```python
def getDataForKeyWord(keyword, keyword_column,target_column):
	INFO = df.iloc[:,keyword_column:1+keyword_column].values
	# Here the : means all rows and , keyword_column is the index number of column
	
	# Lets make the dictionary and list to output
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
					# In real life .xlsx data we may come across NaN values
					# so this if was to handle that instant
					# On the other hand if value is a number then
				else:
					sumDictOf[(text[0:index+L],i)] = df.iloc[i,target_column:1+target_column].values[0]
					nameList.append(text[0:index+L])
		except:
			pass

  # Alright so lets output the values
	return sumDictOf,nameList

sumDictOf,nameList = getDataForKeyWord('Fri',1,3)
```

So above I just enter Fri for the input column 1:2 [weekdays] and the target column 3:4 [prices]
so lets see how the values are obtained 
```python
print(sumDictOf)
print(nameList)
```
Let me close the .xlsx file and again run , it will auto open it
Lets change the keyword to search	
Good enough next we make a new .xlsx file to put two columns
```python
wb = xw.Book()
xw.Range('A1').value = np.array(nameList).reshape(len(nameList),1) # first col
xw.Range('B1').value = np.array(list(sumDictOf.values())).reshape(len(list(sumDictOf.values())),1) # Second col
# And lets save it
wb.save(r'output.xlsx')
``` 
Lets see , run it, close the .xlsx 
We see that the output file now has all selected keywords and their values
Thats all for today
Take care of yourself

	

