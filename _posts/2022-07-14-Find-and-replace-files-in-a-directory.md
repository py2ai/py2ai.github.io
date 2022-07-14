---
layout: post
title: Learn Python Tips and Tricks Part 02
mathjax: true
summary:  Find all files with any extension and rename them
---

Hi friends! In this part we will learn how to get a list of files in a directory. How to rename the extensions of specific type of files. Imagine you have a directory of thousands files and you have to rename only a particular type of files. Manually doing such stuff would require lots of time and energy. But thanks to Python we can do this job quickly.

* [Folder containing target files](#folder-containing-target-files)
    * [Using glob library to get paths of all files](#using-glob-library-to-get-paths-of-all-files)
        * [main.py](#main.py)
    * [Iterate the list of files](#iterate-the-list-of-files)
        * [main.py](#main.py)
    * [Perform the rename operation on files](#perform-the-rename-operation-on-files)
        * [main.py](#main.py)
* [Summary ](#summary-)

# Folder containing target files

We have a direction named `all_files` with some `.json` files in it.
```
banana.json
mango.json
orange.json
pear.json
apple.json
```
Our goal is to first find these `.json` files and then rename them as `txt`. Note that it is just an example and you can try with any kind of extensions.

## Using glob library to get paths of all files

Our small Python code will provide path of directory, and we use `*.json` in it to refer all json files in this path. So make try this simple code and use your own path accordingly.

### main.py

```python
import os
import glob

path = 'all_files/*.json'
files = glob.glob(path)
print('Files found:',files)
```

Output:

```
python .\main.py
Files found: ['all_files\\apple.json', 'all_files\\banana.json', 'all_files\\mango.json', 'all_files\\orange.json', 'all_files\\pear.json']
```
We can see that `glob.glob()` has returned a list of paths of all json files.

## Iterate the list of files

We can use a simple for loop to iterate over `files` list. Get each `file` path and call it source `src` and destination `dst`.  

### main.py

```python
import os
import glob

path = 'all_files/*.json'
files = glob.glob(path)
print('Files found:',files)

for file in files:
    
    src = file
    dst = file.replace('.json','.txt')
    print('src file:',src)
    print('dst file:',dst)
```

Output:

```
python .\main.py
Files found: ['all_files\\apple.json', 'all_files\\banana.json', 'all_files\\mango.json', 'all_files\\orange.json', 'all_files\\pear.json']
src file: all_files\apple.json
dst file: all_files\apple.txt
src file: all_files\banana.json
dst file: all_files\banana.txt
src file: all_files\mango.json
dst file: all_files\mango.txt
src file: all_files\orange.json
dst file: all_files\orange.txt
src file: all_files\pear.json
dst file: all_files\pear.txt
```

## Perform the rename operation on files

We have the source and destination names, so lets us `os` library to rename them.

### main.py

```python
import os
import glob

path = 'all_files/*.json'
files = glob.glob(path)
print('Files found:',files)

for file in files:
    
    src = file
    dst = file.replace('.json','.txt')
    print('src file:',src)
    print('dst file:',dst)
    os.rename(src,dst)
    
path = 'all_files/*.txt'
files = glob.glob(path)
print('Files renamed:',files)
```

Output:

```
python .\main.py
Files found: ['all_files\\apple.json', 'all_files\\banana.json', 'all_files\\mango.json', 'all_files\\orange.json', 'all_files\\pear.json']
src file: all_files\apple.json
dst file: all_files\apple.txt
src file: all_files\banana.json
dst file: all_files\banana.txt
src file: all_files\mango.json
dst file: all_files\mango.txt
src file: all_files\orange.json
dst file: all_files\orange.txt
src file: all_files\pear.json
dst file: all_files\pear.txt
Files renamed: ['all_files\\apple.txt', 'all_files\\banana.txt', 'all_files\\mango.txt', 'all_files\\orange.txt', 'all_files\\pear.txt']
```

We can see that all files are renamed from .json to .txt. Of course we can do the reverse operation to find all .txt files and renamed them back to .json. But that is left as an excercise for you. 


# Summary 

In this tutorial we saw how to find all files with a particular extension in a directory or folder. How to rename all the files in it.
