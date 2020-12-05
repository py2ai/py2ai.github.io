---
layout: post
title: What are yield and return statements in Python
categories: [GUI tutorial series]
mathjax: true
featured-img: yield
summary: This code will demonstrate the key differences between yield and return statements in python
---
yield keyword means: 
Provide output and continue

return keyword means: 
Provide output and stop

<br>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/TODMGIezMpE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
<br>


```python
import time
Lets try to make a list using a function
def make_list (num):
	for i in range(num):
		print('local variable i :',i)
		yield i
		
		
num = 3
value = make_list(num)
print('Here is value',value)

Lets make list of the value

print(list(value))




next_value = next(value)
print('Here is next(value)',next_value)
next_value = next(value)
print('Here is next(value)',next_value)
next_value = next(value)
print('Here is next(value)',next_value)

# If we call the next value once more, we will get a traceback. 
# Because the stack of executions is already empty

# next_value = next(value)
# print('Here is next(value)',next_value)

# So to have a list back lets call the make_list again

value = make_list(num)
print(list(value))



```

