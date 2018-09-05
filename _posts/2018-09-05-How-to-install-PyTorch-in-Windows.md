---
layout: post
title: Installing Pytorch in Windows
author: Hussain A.
categories: [tutorial]
mathjax: true
summary: A fastest way to install PyTorch in Windows without Conda
---

Hello there, today i am going to show you an easy way to install PyTorch in Windows 10 or Windows 7. Typical methods available for 
its installation are based on Conda. However, there also exist an easy way to install PyTorch. It is assumed that you already have installed Python 3.6 in windows 7 or 10. If not then please google for the python 3.6 installation and then 
follow these setps:

*1.* First, we need to install Shapely. For this download [Shapely](http://www.xavierdupre.fr/enseignement/setup/Shapely-1.6.3-cp36-cp36m-win_amd64.whl) as Shapely-1.6.3-cp36-cp36m-win_amd64.whl

Then go to the directory where, you have downloaded the whl file and then press SHIFT and right click and select open command prompt here and then execute this 

`pip install Shapely-1.6.3-cp36-cp36m-win_amd64.whl`

*2.*  Secondly, we need to install Fiona. For this go to [Gohlke](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pytorch) and download 
Fiona‑1.7.13‑cp36‑cp36m‑win_amd64.whl

Then execute this command:


`pip install Fiona‑1.7.13‑cp36‑cp36m‑win_amd64.whl`

*3.* Third and final step is to download [PyTorch](http://www.xavierdupre.fr/enseignement/setup/torch-0.3.0b0+591e73e-py3-none-any.whl). Again just as before execute this in command prompt:

`pip install torch-0.3.0b0+591e73e-py3-none-any.whl`

Congratulations you have PyTorch ready!!.
