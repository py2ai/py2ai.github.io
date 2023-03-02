---
layout: post
title: What is SVD 
mathjax: true
featured-img: 26072022-python-logo
summary:  In this tutorial we will learn what is Singular Value Decomposition (SVD) and run it in python
---

# Singular Value Decomposition

Hello friends! The Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a given matrix A into three matrices as follows:

```
A = UΣV^T
```
where,

U is an m x m unitary matrix (i.e., UU^T = I, where I is the identity matrix), with its columns called the left-singular vectors of A.
Σ is an m x n diagonal matrix containing the singular values of A in decreasing order on its diagonal. Singular values are non-negative real numbers that measure the "strength" or "importance" of the corresponding singular vectors of A.
V is an n x n unitary matrix (i.e., VV^T = I), with its columns called the right-singular vectors of A.
The SVD provides a useful way to understand the structure of a matrix and can be used for a variety of applications in linear algebra, data analysis, and signal processing.

So let's try to run a 2x2 matrix for svd

```python
import numpy as np

# Define a 2x2 matrix A
A = np.array([[2, 3], [4, 1]])

# Perform SVD on A
U, s, Vt = np.linalg.svd(A)

# Construct the diagonal matrix Σ
Sigma = np.zeros_like(A)
Sigma[np.diag_indices(min(A.shape))] = s

# Reconstruct the matrix A from its SVD components
A_reconstructed = np.dot(U, np.dot(Sigma, Vt))

# Print the results
print("Original matrix:\n", A)
print("Left-singular vectors (U):\n", U)
print("Singular values (Σ):\n", Sigma)
print("Right-singular vectors (V^T):\n", Vt)
print("Reconstructed matrix:\n", A_reconstructed)

```

output

```
Original matrix:
 [[2 3]
 [4 1]]
Left-singular vectors (U):
 [[-0.64074744 -0.76775173]
 [-0.76775173  0.64074744]]
Singular values (Σ):
 [[5 0]
 [0 1]]
Right-singular vectors (V^T):
 [[-0.85065081 -0.52573111]
 [ 0.52573111 -0.85065081]]
Reconstructed matrix:
 [[2.32163066 2.33739295]
 [3.60230401 1.47310253]]
```

This code defines a 2x2 matrix A and then uses the np.linalg.svd() function from the NumPy library to perform the SVD on A. The resulting left-singular vectors U, singular values s, and right-singular vectors Vt are then used to reconstruct the original matrix A using the formula A = UΣV^T. The reconstructed matrix is then printed along with the original matrix and the SVD components.

# Why the reconstructed matrix is a little different?

In the SVD decomposition, the matrix A is decomposed into three matrices: U, Σ, and V^T. The original matrix A can then be reconstructed by multiplying these three matrices together: A = UΣV^T.

However, when we reconstruct the matrix A using the SVD components in the code above, we might not get the exact same matrix as the original matrix A. This is because we only keep a finite number of singular values and truncate the remaining singular values to zero, which can result in some loss of information.

In the example code above, the reconstructed matrix A_reconstructed might not be similar to the original matrix A because we are only using the largest singular value and truncating the other singular values to zero. If we use more singular values, we might get a better approximation of the original matrix A. For example, we can use the first k singular values to reconstruct the matrix A, where k is a positive integer less than or equal to the rank of A.

In summary, the reconstructed matrix A_reconstructed might not be exactly the same as the original matrix A due to the truncation of singular values. However, the reconstructed matrix should still capture the most important features of the original matrix.

# How to get the better resconstruction of matrix A?

To get a better reconstruction result, we can include more singular values in the reconstruction. Here's an example Python code that uses the first two singular values to reconstruct the matrix A:

```python
import numpy as np

# Define a 2x2 matrix A
A = np.array([[2, 3], [4, 1]])

# Perform SVD on A
U, s, Vt = np.linalg.svd(A)

# Keep the first two singular values and truncate the others to zero
k = 2
Sigma = np.diag(s[:k])
U = U[:, :k]
Vt = Vt[:k, :]

# Reconstruct the matrix A from its SVD components using the first two singular values
A_reconstructed = U @ Sigma @ Vt

# Print the results
print("Original matrix:\n", A)
print("Left-singular vectors (U):\n", U)
print("Singular values (Σ):\n", Sigma)
print("Right-singular vectors (V^T):\n", Vt)
print("Reconstructed matrix:\n", A_reconstructed)

```
output

```
Original matrix:
 [[2 3]
 [4 1]]
Left-singular vectors (U):
 [[-0.64074744 -0.76775173]
 [-0.76775173  0.64074744]]
Singular values (Σ):
 [[5.11667274 0.        ]
 [0.         1.95439508]]
Right-singular vectors (V^T):
 [[-0.85065081 -0.52573111]
 [ 0.52573111 -0.85065081]]
Reconstructed matrix:
 [[2. 3.]
 [4. 1.]]
```
