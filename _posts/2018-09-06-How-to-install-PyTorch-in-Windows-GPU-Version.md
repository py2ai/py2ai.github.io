---
layout: post
title: Installing Pytorch in Windows (GPU version)
categories: [tutorial]
mathjax: true
description: A fastest way to install PyTorch in Windows without Conda
---

Hi there, today we are installing PyTorch in Windows. It is assumed that you already have installed NVidia GPU card. 
The installation also requires the correct version of CUDA toolkit and the type of graphics card. For example if your 
GPU is GTX 1060 6G, then its a Pascal based graphics card. Also check your version accordingly from the Nvidia official website. 

Now come to the CUDA tool kit version. If you want to know which version of CUDA tool kit is installed in windows. Open up the command prompt and enter this

`nvcc --version`

In my case it shows, the release 9.0, V9.0.176 CUDA compilation Tools. Your Python version should also be known. For my PC it is 
python 3.6.5. Now jump to [peterjc123](https://github.com/peterjc123/pytorch-scripts) and click Windows GPU (0.4.0). It will 
take you to another page where you have a variety of options available.
`JOB NAME TESTS DURATION
Environment: CUDA_VERSION=90, PYTHON_VERSION=3.6, TORCH_CUDA_ARCH_LIST=Maxwell
1 hr 15 min
Environment: CUDA_VERSION=90, PYTHON_VERSION=3.5, TORCH_CUDA_ARCH_LIST=Maxwell
1 hr 12 min
Environment: CUDA_VERSION=91, PYTHON_VERSION=3.6.2, TORCH_CUDA_ARCH_LIST=Maxwell
1 hr 14 min
Environment: CUDA_VERSION=91, PYTHON_VERSION=3.5, TORCH_CUDA_ARCH_LIST=Maxwell
1 hr 13 min
Environment: CUDA_VERSION=90, PYTHON_VERSION=3.6.2, TORCH_CUDA_ARCH_LIST=Pascal
1 hr 14 min
Environment: CUDA_VERSION=90, PYTHON_VERSION=3.5, TORCH_CUDA_ARCH_LIST=Pascal
1 hr 23 min
Environment: CUDA_VERSION=91, PYTHON_VERSION=3.6.2, TORCH_CUDA_ARCH_LIST=Pascal
1 hr 12 min
Environment: CUDA_VERSION=91, PYTHON_VERSION=3.5, TORCH_CUDA_ARCH_LIST=Pascal
1 hr 16 min
Environment: CUDA_VERSION=90, PYTHON_VERSION=3.6.2, TORCH_CUDA_ARCH_LIST=Kepler
1 hr 8 min
Environment: CUDA_VERSION=90, PYTHON_VERSION=3.5, TORCH_CUDA_ARCH_LIST=Kepler
1 hr 14 min
Environment: CUDA_VERSION=91, PYTHON_VERSION=3.6.2, TORCH_CUDA_ARCH_LIST=Kepler
1 hr 14 min
Environment: CUDA_VERSION=91, PYTHON_VERSION=3.5, TORCH_CUDA_ARCH_LIST=Kepler
1 hr 10 min`
 
 Go ahead and click on the relevant option. In my case i choose this option:
 `Environment: CUDA_VERSION=90, PYTHON_VERSION=3.6.2, TORCH_CUDA_ARCH_LIST=Pascal`
 
Eventhough i have Python 3.6.5 but it will still work for any python 3.6.x version. My card is Pascal based and my CUDA toolkit
version is 9.0 which is interpreted as 90. After clicking this option you will land to anther page, scroll down and you will see
these options: 

**JOBS CONSOLE MESSAGES  TESTS  ARTIFACTS**

Click on the **ARTIFACTS** option. After this scroll down and you will find the whl file. For my case the [PyTorch](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pytorch)
is here. Download it and then pip install the whl file. For example:

`pip install torch‑1.0.1‑cp36‑cp36m‑win_amd64.whl`

After succesfull installation we need to check if all things working fine?

For this open up python by typing python in command prompt.

```python
import torch
```
If no error occurs, it means PyTorch is installed, now lets check the cuda support. For this type

```python
torch.cuda.is_available()
```
If you get `True` it means you have succesfully installed the PyTorch. If any problems you can ask me in the comments section. Have a nice day!



