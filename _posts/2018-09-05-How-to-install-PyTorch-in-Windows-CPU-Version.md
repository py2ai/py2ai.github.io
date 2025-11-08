---
categories:
- tutorial
description: A fastest way to install PyTorch in Windows without Conda
keywords:
- installing
- pytorch
- development
- code
- programming
- cpu
- windows
- tutorial
layout: post
mathjax: true
title: Installing Pytorch in Windows (CPU version)
---


Hello there, today i am going to show you an easy way to install PyTorch in Windows 10 or Windows 7. Typical methods available for its installation are based on Conda. However, there also exists an easy way to install PyTorch (CPU support only). It is assumed that you have installed Python 3.6 in windows 7 or 10. If not then please google for the python 3.6 installation and then 
follow these setps:

*1.* First, we need to install Shapely. For this download [Shapely](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely) as Shapely-1.6.3-cp36-cp36m-win_amd64.whl
For Windows 32bit use this Shapely‑1.7.1‑cp36‑cp36m‑win32.whl

Then go to the directory where you have downloaded the whl file and then press SHIFT and right click and select open command prompt here and then execute this 

`pip install Shapely-1.6.3-cp36-cp36m-win_amd64.whl`
For 32 bit
`pip install Shapely‑1.7.1‑cp36‑cp36m‑win32.whl`

*2.*  Secondly, execute the following for intel openmp

`pip install intel-openmp`

*3.* Third and final step is to download [PyTorch](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pytorch), currently the version available is torch‑1.0.1‑cp36‑cp36m‑win_amd64.whl, so download it. Again just as before execute this in command prompt:

`pip install torch‑1.0.1‑cp36‑cp36m‑win_amd64.whl`
For 32 bit version:
`pip install torch==1.6.0`
Congratulations! you have PyTorch (CPU version) ready!! If you like to install PyTorch GPU version, please follow my next tutorial.
