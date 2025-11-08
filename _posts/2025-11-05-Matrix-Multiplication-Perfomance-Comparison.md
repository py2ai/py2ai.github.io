---
description: Learn how to compare matrix multiplication speeds using naive Python, compiled C++, and optimized NumPy.
featured-img: 26072022-python-logo
keywords:
- matrix multiplication
- Python
- C++
- NumPy
- performance
layout: post
tags:
- matrix-multiplication
- Python
- C++
- NumPy
- performance
title: Matrix Multiplication Performance Comparison Python vs...
---




# Matrix Multiplication Performance Comparison: Python vs C++ vs NumPy

## # Beginner-Friendly Tutorial to Understand Matrix Computation and Optimization

This tutorial will walk you through performing **matrix multiplication** using three approaches:
1. Naive Python nested loops
2. Compiled C++ code executed from Python
3. Optimized NumPy operations

We'll compare the execution times and understand why optimized libraries like NumPy are much faster.

---

## # Table of Contents
- [Introduction](#introduction)
- [Setup and Imports](#setup-and-imports)
- [Generate Random Matrices](#generate-random-matrices)
- [Naive Python Multiplication](#naive-python-multiplication)
- [C++ Multiplication from Python](#c-multiplication-from-python)
- [NumPy Multiplication](#numpy-multiplication)
- [Performance Comparison](#performance-comparison)
- [Key Takeaways](#key-takeaways)
- [Running the Code](#running-the-code)
- [Why Some Are Faster](#why-some-are-faster)

---

## # Introduction

Matrix multiplication is a fundamental operation in mathematics, computer science, and data science. However, **naive implementations in Python** are slow for large matrices because Python loops are interpreted at runtime.

Optimized libraries like **NumPy** and compiled languages like **C++** can significantly reduce computation time.

---

## # Setup and Imports

We'll use the following Python libraries:

```python
import time          # To measure execution time
import subprocess    # To compile and run C++ code from Python
tempfile             # To create temporary files for C++ code
import os            # To handle file operations
import numpy as np   # Highly optimized matrix operations
```

Make sure you have **NumPy** installed. You can install it using:

```bash
pip install numpy
```
---

## # Generate Random Matrices

We'll generate two square matrices `A` and `B` of size `N x N` with random integers between 0 and 10.

```python
## # Matrix Size
N = 600  # Increase N to see larger performance differences

## # Generate Matrices
A, B = np.random.randint(0, 11, (2, N, N))
```

> Tip: For beginners, start with smaller values like `N=100` to test the code faster.

---

## # Naive Python Multiplication

Using nested loops, we multiply matrices manually.

```python
start_py = time.time()
C = [[0]*N for _ in range(N)]
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]
end_py = time.time()
python_time = end_py - start_py
```

## # How It Works
- We create an empty matrix `C` with zeros.
- Three nested loops compute the dot product.
- Time is recorded using `time.time()`.

**Note:** This is extremely slow for large matrices due to Python's interpreted loops.

---

## # C++ Multiplication from Python

We can speed things up using **C++**, a compiled language. We'll:
1. Write the C++ code to a temporary file.
2. Compile it using `g++`.
3. Run the executable and capture the runtime.

{% raw %} 
```python
cpp_code = f"""
## # include <iostream>
## # include <vector>
## # include <cstdlib>
## # include <ctime>
using namespace std;
int main(){{
    int N = {N};
    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<long long>> C(N, vector<long long>(N, 0));
    srand(time(0));
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++){{
            A[i][j] = rand() % 11;
            B[i][j] = rand() % 11;
        }}
    clock_t start = clock();
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<N;k++)
                C[i][j] += (long long)A[i][k]*B[k][j];
    clock_t end = clock();
    cout << double(end-start)/CLOCKS_PER_SEC << endl;
    return 0;
}}
"""

## # Write to temp file
with tempfile.NamedTemporaryFile(delete=False, suffix=".cpp") as f:
    cpp_file = f.name
    f.write(cpp_code.encode())

exe_file = cpp_file.replace(".cpp", "")

## # Compile
compile_proc = subprocess.run(["g++", cpp_file, "-O2", "-o", exe_file], capture_output=True, text=True)
if compile_proc.returncode != 0:
    print("C++ Compilation failed!", compile_proc.stderr)
else:
    result = subprocess.run([exe_file], capture_output=True, text=True)
    cpp_time = float(result.stdout.strip())

## # Cleanup
ios.remove(cpp_file)
if os.path.exists(exe_file): os.remove(exe_file)
```
{% endraw %}

**Explanation:**
- We generate the matrices inside C++ with `rand()`.
- Use triple nested loops to calculate multiplication.
- Capture execution time using `clock()`.
- Compile with `-O2` optimization flag.

---

## # NumPy Multiplication

NumPy provides optimized matrix multiplication using highly efficient C/Fortran libraries under the hood.

```python
A_np = np.array(A)
B_np = np.array(B)

start_np = time.time()
C_np = np.dot(A_np, B_np)
end_np = time.time()
numpy_time = end_np - start_np
```
## # Why NumPy is Fast
- NumPy leverages **vectorized operations**.
- Internal loops are compiled in **C/Fortran**, avoiding Python overhead.
- Utilizes CPU cache and SIMD instructions for speed.

---

## # Performance Comparison

Finally, let's compare all three methods:

```python
print("\n=== Performance Comparison ===")
print(f"Python: {python_time:.4f} s")
print(f"C++: {cpp_time:.4f} s")
print(f"NumPy: {numpy_time:.4f} s")
```

You will usually find:
```
Python >> C++ >> NumPy
```
for execution speed, with NumPy being the fastest.

---

## # Key Takeaways
- Naive Python loops are slow for large matrices.
- C++ is faster due to compilation, but requires more setup.
- NumPy is both **fast** and **easy to use** in Python.
- Optimized libraries are crucial for performance-critical applications.

---

## # Running the Code

1. Copy the code into a `.py` file.
2. Ensure you have **NumPy** and **g++ compiler** installed.
3. Run the Python script:

---

## # Why Some Are Faster

The speed differences are due to how the code is executed:

- **Python**: Slower because it interprets loops at runtime in a high-level language, which adds overhead for each iteration.
- **C++**: Faster because it is compiled into machine code, reducing runtime overhead. The `-O2` flag also optimizes loops.
- **NumPy**: Fastest because it is written in C and Fortran under the hood and uses highly optimized, vectorized operations that leverage CPU features.


```bash
python matrix_comparison.py
```
## # Complete Code
{% raw %} 
```python
import time, subprocess, tempfile, os
import numpy as np # Highly optimized C/Fortran based

## # Matrix Size
N = 100 # Adjust bigger N shows bigger speed difference
## # Generate Matrices
A,B = np.random.randint(0,11,(2,N,N))

## # Python Naive matrix Multiplication
start_py = time.time()
C = [[0]*N for _ in range(N)]
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j] += A[i][k] * B[k][j]
end_py = time.time()
python_time = end_py - start_py

## # C++ version from Python
cpp_code = f"""
## # include <iostream>
## # include <vector>
## # include <cstdlib>
## # include <ctime>
using namespace std;
int main(){{
    int N = {N};
    vector<vector<int> > A(N, vector<int>(N));
    vector<vector<int> > B(N, vector<int>(N));
    vector<vector<long long> >
    C(N, vector<long long>(N, 0));
    
    srand(time(0));
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++) {{
            A[i][j] = rand() % 11;
            B[i][j] = rand() % 11;
        }}
    clock_t start = clock();
    for (int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            for(int k=0; k<N; k++)
                C[i][j] += (long long)A[i][k]*B[k][j];
    clock_t end  = clock();
    double duration = double(end-start)/CLOCKS_PER_SEC;
    cout << duration << endl;
    return 0;
}}
""" 

## # Write C++ code to temp file
with tempfile.NamedTemporaryFile(delete=False,
                                 suffix=".cpp") as f:
    cpp_file = f.name
    f.write(cpp_code.encode())
exe_file = cpp_file.replace(".cpp", "")
compile_proc = subprocess.run(["g++", cpp_file,
                               "-O2", "-o", exe_file],
                              capture_output=True,
                              text=True)
if compile_proc.returncode !=0:
    print("C++ Compilation failed!")
    print(compile_proc.stderr)
else:
    ## Run C++ executable and capture time
    result = subprocess.run([exe_file],
                            capture_output=True,
                            text=True)
    cpp_time = float(result.stdout.strip())

## # Cleanup temporary files
os.remove(cpp_file)
if os.path.exists(exe_file):
    os.remove(exe_file)

## # Numpy Matrix Multiplication
A_np = np.array(A)
B_np = np.array(B)

start_np = time.time()
C_np = np.dot(A_np, B_np)
end_np = time.time()
numpy_time = end_np - start_np

## # Compare All
print("\n=== Performance Comparison ===")
print(f"Python: {python_time:.4f} s")
print(f"C++: {cpp_time:.4f} s")
print(f"Numpy: {numpy_time:.4f} s")
```
{% endraw %}

You’ll see execution times for Python, C++, and NumPy.

---

This tutorial demonstrates how **choosing the right tools** can drastically impact performance in numerical computing.

