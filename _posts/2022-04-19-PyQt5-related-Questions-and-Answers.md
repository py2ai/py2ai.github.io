---
layout: post
title: Issue and solutions related to PyQt5
categories: [GUI tutorial series]
mathjax: true
summary: You can find important issues and their solutions related to PyQt5 here
---

Hi friends, we are going to start a Q and A about PyQt5 here. This page will be dedicated to only Questions that are frequently asked by you and their answers.
We will continue to update this page accordingly.

### Q: I can't find the Qt designer app in windows? I just typed designer in cmd but have this error:
```
designer : The term 'designer' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of the name, or if a path was included, verify 
that the path is correct and try again.
At line:1 char:1
+ designer
+ ~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (designer:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
```
### Solution:
Unfortunatley, in newer versions, the PyQt5 wheels do not provide tools such as Qt Designer that were included in the old binary installers. But the good news is that you can install it via 

`pip install pyqt5-tools~=5.15`

Once installed you can simple type the following to run the Qt designer

```
pyqt5-tools designer
```
