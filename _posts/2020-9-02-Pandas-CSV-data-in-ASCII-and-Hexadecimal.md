---
layout: post
title: Pandas dataframe with hexadecimal and ascii values
categories: [tutorial series]
mathjax: true
featured-img: pandas
summary: This code will demonstrate how to convert pandas data to numeric form 
---
Source code and sample csv data file are available:[download]
```python
import pandas as pd
import struct
data = pd.read_csv('data.csv',encoding='utf-8').fillna(0).astype(str)
print ("BEFORE:",data.head())
abcd = 1

def get_value(col):
	out=col.copy()
	i=0
	for d in col:
		d=eval(d)
		packed = struct.pack('L',d)
		unpacked = struct.unpack('L',packed)[0]
		out[i] = unpacked
		i+=1
	return out
	
data['col1'] = get_value(data['col1'].values)
data['col2'] = get_value(data['col2'].values)
data['col3'] = get_value(data['col3'].values)
print('AFTER:',data)	

sample=['1234','abcd', '0x12cd']
print(sample,'CONVERTED TO:',get_value(sample))
```
# RESULTS

BEFORE:    col1  col2     col3

0  1234  abcd   0x12cd

1  1234  abcd   0x12cd

2  1234  abcd   0x12cd

3  1234  abcd   0x12cd

AFTER:    col1 col2  col3

0  1234    1  4813

1  1234    1  4813

2  1234    1  4813

3  1234    1  4813

['1234', 'abcd', '0x12cd'] CONVERTED TO: [1234, 1, 4813]

[download](https://drive.google.com/file/d/1nELxf5DpHsOR6OSFQT6NapZBpOhJ_hn8/view?usp=sharing)
