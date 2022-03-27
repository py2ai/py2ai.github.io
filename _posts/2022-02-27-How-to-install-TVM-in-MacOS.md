---
layout: post
title: How to install TVM on MAC OS
categories: [tutorial]
mathjax: true
summary: A quick tutorial for beginners to build from source the TVM in Mac OS
---

Hello there, today's tutorial is about installing Apache TVM on your Mac OS. The installation procedure is definetly available on 
https://tvm.apache.org/docs/install/from_source.html, however, there are some steps that are required for the beginners to successfully install it.
Friends, Apached TVM is an open source ML compiler framework for GPUS, CPUs and deep learning accelerators. The goal of TVM is to enable deep learning
community to optimize and then run computations efficiently on a hardware. TVM has a diverse community of hardware vendors, compiler engineers and ML researchers
. This community has build a unified, programmable software stack to enrich the entire ML technology ecosystem and made it available to everyone. As a result
of this collective effort, the performance goes up. We require the following steps to successfully install TVM on MacOS. Please note that Homebrew is
required for either Intel or Apple's M1 chip processors. We need cmake, llvm and Python3.8, please note that currently Python3.9 have some issues so I will
higly recommend Python3.8

### Step 1
```
brew install gcc git cmake
brew install llvm
```
If Python3.8 is not installed then either use virtualenv or conda env to install it, otherwise use 
```
brew install python@3.8
```
### Step 2
By default brew will not link your llvm installation to correct path, for this you need to force link it
```
brew link llvm --force
```

### Step 3

```
git clone --recursive https://github.com/apache/tvm tvm
```
### Step 4

```
cd tvm
mkdir build
cp cmake/config.cmake build
```
### Step 5
We need to edit edit build/config.cmake file to customize the compilation options, so we can directly add `set(USE_LLVM ON)` and let cmake search for a usable version of LLVM. After that we will build and make tvm.

```
cd build
cmake ..
make -j4
```
### Step 6

Once all make is finished successfully, we need to install Python package for this tvm, so that we can use it in Python codes.

```
export MACOSX_DEPLOYMENT_TARGET=10.9 
cd python; python setup.py install --user; cd ..

```



Below are some useful links about TVM:

[1] Tutorials: https://tvm.apache.org/docs/tutorials/index.html#autoscheduler-template-free-auto-scheduling

[2] Benchmark repo: https://github.com/tlc-pack/TLCBench

[3] OSDI Paper: Ansor : Generating High-Performance Tensor Programs for Deep Learning

[4] Results on Apple M1 chip: https://medium.com/octoml/on-the-apple-m1-beating-apples-core-ml-4-with-30-model-performance-improvements-9d94af7d1b2d.
