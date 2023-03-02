---
layout: post
title: What is SVD 
mathjax: true
featured-img: 26072022-python-logo
summary:  In this tutorial we will learn what is Singular Value Decomposition (SVD) and run it in python
---

 
Hello friends! The Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a given matrix A into three matrices as follows:

```
A = UΣV^T
```
where,

U is an m x m unitary matrix (i.e., UU^T = I, where I is the identity matrix), with its columns called the left-singular vectors of A.
Σ is an m x n diagonal matrix containing the singular values of A in decreasing order on its diagonal. Singular values are non-negative real numbers that measure the "strength" or "importance" of the corresponding singular vectors of A.
V is an n x n unitary matrix (i.e., VV^T = I), with its columns called the right-singular vectors of A.
The SVD provides a useful way to understand the structure of a matrix and can be used for a variety of applications in linear algebra, data analysis, and signal processing.
