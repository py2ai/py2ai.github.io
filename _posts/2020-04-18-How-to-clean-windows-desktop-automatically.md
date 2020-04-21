---
layout: post
title: How to automatically arrange the desktop icons
author: Hussain A.
categories: [GUI tutorial series]
mathjax: true
summary: This code will arrange files in folders according to their extension to keep windows desktop nice and tidy
---


Hello there! Welcome to the PyShine Artificial Intelligence Learning series. I hope you guys will be staying at home and healthy. I was thinking that the majority of PC users do some much work every day, yet just like me they often come across the stuffy desktop. Yes, a Windows PC desktop that gets full of icons here and there, in fact everywhere. For those who regularly manage and put their files and folders carefully, it is a habit to them and salutes to them. But for others who still find it hard to put the files appropriately, it really becomes a huge pile of icons on the desktop. So I was thinking about how Python can be useful in arranging the desktop icons. For the time being, I have this idea that maybe we can find files and folders on the desktop and then make groups of files according to their extensions and then put them to their types in their respective new Type folders.

So let's try to make this happen with simple Python code:

We only need os library for paths and shutil library for moving files:

```python 
import os
import shutil
```
Let's get the path of your desktop:

```python 
path = os.path.expanduser("~/Desktop")
```
Now we need to get the folder list and files list

```python 

subdirs = filter(os.path.isdir, [os.path.join(path,x) for x in os.listdir(path)])
folder_list = list()
file_list = list()

for folderpath in subdirs:
	folder_list.append(folderpath)

print("Folders List: ", folder_list)


file_paths = filter(os.path.isfile,[os.path.join(path,x) for x in os.listdir(path)])
for f in file_paths:
	file_list.append( f )
	
print("Files List: ", file_list)
list = file_list


```
Now that we have both folder list and files list, we dont change the folders (for those who want to put these folders in antother folder can do more enhancements). On the other hand for the files we will find the extension of each file using the function:

```python

def file_extension(path): 
	return os.path.splitext(path)[1] 
	

```
This function will take input a list of files and generate a Dictionary as an output. This output key_dict will have keys as the extension name and values as the list that will contain paths of all files that belong to this extension key of the key_dict.

```python
def getGroupsDict(Files):
	key_dict = {}
	for file in Files:
		key_dict[file_extension(file)] = []
	for file in Files:
		key_dict[file_extension(file)].append(file)
	return key_dict
	
dict = getGroupsDict(list)
print("dict",dict)
```
Once we get the dict from getGroupsDict() function, we will iterate over the dictionary items to make new folders with name as the Upper case of keys of the dict. If a folder already exists we can skip it.

```python
for key,value in dict.items():
	Key = key.upper()
	try:
		os.mkdir(path+'/'+Key)
	except:
		pass
```
Now that all folders are created, its time to clean up the Desktop. The following loop will move all files to their respective new folders.

```python
for key,value in dict.items():
	for file in dict[key]:
		try:
			shutil.move(file,path+'/'+key.upper()) 
		except:
			pass
		


```
Thats it, hopefully you will like this tiny but fruitful code. 








	
