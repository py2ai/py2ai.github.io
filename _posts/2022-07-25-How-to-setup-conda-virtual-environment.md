---
layout: post
title: Learn Python Tips and Tricks Part 03
mathjax: true
summary:  Setting up Conda environment for Python
---

Hello friends! Installing any version of Python on any platform is easy. But this easy task becomes cumbersome when you need to work on cross-platforms, for example, MacOs, Linux/Ubuntu/RedHat, Windows, etc. The situation gets even more worse when you have to work on multiple versions of Python. Now imagine a quick and easy way to work on virtual environments for any platform OS and any version of Python. Today we will walk through the installation steps and quote working tips about Miniconda.

* [Install Miniconda ](#install-miniconda-)
    * [Miniconda installation on Windows ](#miniconda-installation-on-windows-)
        * [Solution to Miniconda activate problem on Windows](#solution-to-miniconda-activate-problem-on-windows)
* [Create virtual environment using conda with a specific Python version](#create-virtual-environment-using-conda-with-a-specific-python-version)
* [Info about existing virtual environments](#info-about-existing-virtual-environments)
* [Activate a conda virtual environment ](#activate-a-conda-virtual-environment-)
* [Run python under a virtual environment](#run-python-under-a-virtual-environment)
* [Install a Python Package using Conda](#install-a-python-package-using-conda)
    * [Package Plan ##](#package-plan-##)
* [Deactivate a virtual environment](#deactivate-a-virtual-environment)
* [Remove a conda virtual environment](#remove-a-conda-virtual-environment)

# Install Miniconda 

Go to this website https://docs.conda.io/en/latest/miniconda.html, and according to your OS, download the install the Miniconda.

## Miniconda installation on Windows 
While installing follow the Default settings in the installation setup. Close and re-open the terminal. Here you might experience an issue saying profile.ps1 cannot be loaded, 
```
. : File C:\Users\ps\Documents\WindowsPowerShell\profile.ps1 cannot be loaded because running scripts is    
disabled on this system. For more information, see about_Execution_Policies at 
https:/go.microsoft.com/fwlink/?LinkID=135170.
At line:1 char:3
+ . 'C:\Users\ps\Documents\WindowsPowerShell\profile.ps1'
+   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) [], PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
```

### Solution to Miniconda activate problem on Windows
The above problem can be resolved by opening up Power Shell as Administrator and typing this command `set-ExecutionPolicy RemoteSigned`
It will ask to press `yes/no`. Simply press `y` and there you go. Close and re-open powershell as standalone or Terminal in Visual Studio Code, there will be no issue. 

# Create virtual environment using conda with a specific Python version

Open up the powershell or vscode based Terminal, you will see (base) default environment at the start.

```(base) PS C:\PyShine>```
To create a virtual environment (e.g., named py36) of your own choice with any specific Python version, use the following command:
```conda create -n py36 python=3.6.5```

The above command will start creating a new virtual environent with `Python 3.6.5 version`, and the environment name will be `py36`. Of course you can give any name

# Info about existing virtual environments

Following command will list the existing environments
```conda env list```
output:
```py36   C:\Users\ps\miniconda3\envs\py36```

# Activate a conda virtual environment 

Activating a conda environment is very simple. We can activate the `py36` env:

```conda activate py36```

# Run python under a virtual environment

Under the `py36` environment, you can type `python` and hit enter.

```python
(base) PS C:\PyShine> conda activate py36
(py36) PS C:\PyShine> python
Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
# Install a Python Package using Conda

You can either use `pip install package` under the activate virtual environment or simply use `conda install package`. Replace the `package` with the required one. For example to install `numpy` package, use `conda install numpy`.

```
(py36) PS C:\PyShine> python
Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()
(py36) PS C:\PyShine> conda install numpy
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: C:\Users\ps\miniconda3\envs\py36

  added / updated specs:
    - numpy


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    intel-openmp-2022.0.0      |    haa95532_3663         2.4 MB
    mkl-2020.2                 |              256       109.3 MB
    mkl-service-2.3.0          |   py36h196d8e1_0          45 KB
    mkl_fft-1.3.0              |   py36h46781fe_0         131 KB
    mkl_random-1.1.1           |   py36h47e9c7a_0         235 KB
    numpy-1.19.2               |   py36hadc3359_0          22 KB
    numpy-base-1.19.2          |   py36ha3acd2a_0         3.8 MB
    ------------------------------------------------------------
                                           Total:       115.9 MB

The following NEW packages will be INSTALLED:

  blas               pkgs/main/win-64::blas-1.0-mkl
  intel-openmp       pkgs/main/win-64::intel-openmp-2022.0.0-haa95532_3663
  mkl                pkgs/main/win-64::mkl-2020.2-256
  mkl-service        pkgs/main/win-64::mkl-service-2.3.0-py36h196d8e1_0
  mkl_fft            pkgs/main/win-64::mkl_fft-1.3.0-py36h46781fe_0
  mkl_random         pkgs/main/win-64::mkl_random-1.1.1-py36h47e9c7a_0
  numpy              pkgs/main/win-64::numpy-1.19.2-py36hadc3359_0
  numpy-base         pkgs/main/win-64::numpy-base-1.19.2-py36ha3acd2a_0
  six                pkgs/main/noarch::six-1.16.0-pyhd3eb1b0_1


Proceed ([y]/n)?

```

Press `y` to continue...

```
Downloading and Extracting Packages
mkl-service-2.3.0    | 45 KB     | ############################################################################ | 100%
numpy-base-1.19.2    | 3.8 MB    | ############################################################################ | 100%
mkl_fft-1.3.0        | 131 KB    | ############################################################################ | 100%
mkl-2020.2           | 109.3 MB  | ############################################################################ | 100%
mkl_random-1.1.1     | 235 KB    | ############################################################################ | 100%
intel-openmp-2022.0. | 2.4 MB    | ############################################################################ | 100%
numpy-1.19.2         | 22 KB     | ############################################################################ | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
(py36) PS C:\PyShine>

```

The `numpy` package is installed successfully. There are multiple ways to confirm it. But the simple one is to use `pip install numpy`
```
(py36) PS C:\PyShine> pip install numpy
Requirement already satisfied: numpy in c:\users\ps\miniconda3\envs\py36\lib\site-packages (1.19.2)
```
You can uninstall this package with `conda uninstall numpy`.

# Deactivate a virtual environment

To activate pass this command

```
(py36) PS C:\PyShine> conda deactivate py36
(base) PS C:\PyShine>
```
This will lead to the default virtual environment `base`. Go on and deactivate it as well 
```
(base) PS C:\PyShine> conda deactivate
PS C:\PyShine>
```
Please note that the Python version and its relevant installed packages only belong to the virtual environment that is activated.

# Remove a conda virtual environment

You can always remove any existing conda environment, e.g., `py36` using the following command:

```conda env remove --name=py36```
