---
description: Learn how to compile and execute C++ code directly from Python using subprocess and tempfile modules.
featured-img: 20251109-c++vpython
keywords:
- Python
- C++
- subprocess
- integration
- tutorial
layout: post
mathjax: false
tags:
- python
- cpp
- automation
- interoperability
- tutorial
title: Run C++ Code from Python – Step-by-Step Guide
---
# Run C++ Code from Python – Step-by-Step Guide

## # Beginner-Friendly Tutorial – Learn How Python Can Compile and Execute C++ Code

Python is an incredibly versatile language, but sometimes you might need the speed and performance of C++. This tutorial explains how to **run C++ code directly from Python**, making it easy to integrate both languages. You’ll learn how to write, compile, and execute a temporary C++ program from a Python script using the **`subprocess`** and **`tempfile`** modules. This is a great way to understand cross-language execution and automation using Python!

## # Table of Contents

- [Overview](#overview)
- [Why Run C++ from Python?](#why-run-c-from-python)
- [Step 1: Writing C++ Code in Python](#step-1-writing-c-code-in-python)
- [Step 2: Compiling C++ Code](#step-2-compiling-c-code)
- [Step 3: Executing the Compiled Program](#step-3-executing-the-compiled-program)
- [Step 4: Cleaning Up Temporary Files](#step-4-cleaning-up-temporary-files)
- [Complete Code](#complete-code)
- [How It Works](#how-it-works)
- [Key Learnings](#key-learnings)
- [Further Experiments](#further-experiments)

## # Overview

This project shows how Python can act as a **controller for compiling and executing C++ code**. The program:

1. Writes a simple **C++ Hello World** program to a temporary file.
2. Uses **`subprocess.run()`** to compile it using the **g++ compiler**.
3. Executes the compiled program and displays the C++ output inside Python.
4. Cleans up all temporary files automatically.

## # Why Run C++ from Python?

- Combine **Python's flexibility** with **C++'s performance**.
- Automate **testing of C++ code** or **generate code dynamically**.
- Build hybrid systems where Python handles logic and C++ handles computation-heavy tasks.

## # Step 1: Writing C++ Code in Python

The script starts with a simple Python print, followed by defining C++ code as a multi-line string.

```python
import subprocess  # To run external programs
import tempfile    # To create temporary files for C++ code
import os          # To handle file operations

print("Hello World from Python!")  # Simple Python print

cpp_code = """
## # include <iostream>
using namespace std;

int main() {
    cout << "Hello World from C++!" << endl;
    return 0;
}
"""
```

Here, the **C++ code is stored as a string** that will later be written to a temporary file.

---

## # Setting Up `g++` on Your System

You need a working C++ compiler (`g++`) to run this tutorial. Follow the steps below depending on your OS.

## # Windows

1. Install **MinGW-w64** (Minimalist GNU for Windows):
   - Visit: [https://winlibs.com/](https://winlibs.com/) or [https://sourceforge.net/projects/mingw-w64/](https://sourceforge.net/projects/mingw-w64/)
   - Download and install it.
2. Add the **`bin` directory** (e.g., `C:\mingw64\bin`) to your **PATH environment variable**.
3. Open Command Prompt and type:
   ```bash
   g++ --version
   ```

   If it prints version info, you’re ready to go!

## # macOS

1. Install **Xcode Command Line Tools** by running:
   ```bash
   xcode-select --install
   ```
2. Verify installation with:
   ```bash
   g++ --version
   ```

   You should see Apple’s `clang` compiler, which works for this tutorial.

## # Linux (Ubuntu/Debian)

1. Open Terminal and install `g++`:
   ```bash
   sudo apt update
   sudo apt install g++
   ```
2. Verify installation:
   ```bash
   g++ --version
   ```

---

## # Step 2: Compiling C++ Code

We create a temporary `.cpp` file and compile it using the **g++ compiler**.

```python
with tempfile.NamedTemporaryFile(delete=False, suffix=".cpp") as f:
    cpp_file = f.name
    f.write(cpp_code.encode())

exe_file = cpp_file.replace(".cpp", "")

compile_proc = subprocess.run(
    ["g++", cpp_file, "-o", exe_file],
    capture_output=True,
    text=True
)
```

- **`tempfile.NamedTemporaryFile()`** creates a unique `.cpp` file in the system's temporary directory.
- **`subprocess.run()`** executes the g++ compiler to compile the file into an executable.
- **`capture_output=True`** captures any compilation messages.

## # Step 3: Executing the Compiled Program

After successful compilation, Python runs the compiled C++ binary.

```python
if compile_proc.returncode != 0:
    print("C++ compilation failed!")
    print(compile_proc.stderr)
else:
    result = subprocess.run([exe_file], capture_output=True, text=True)
    print("Output from C++:")
    print(result.stdout.strip())
```

If compilation fails, Python displays the compiler error messages. If it succeeds, it prints the **C++ program output**.

## # Example Output:

```
Hello World from Python!
Output from C++:
Hello World from C++!
```

## # Step 4: Cleaning Up Temporary Files

Finally, we remove temporary files created during compilation.

```python
os.remove(cpp_file)
if os.path.exists(exe_file):
    os.remove(exe_file)
```

This ensures no leftover files clutter your system.

## # Complete Code

{% include codeHeader.html %}

```python
import subprocess
import tempfile
import os

print("Hello World from Python!")

cpp_code = """
## # include <iostream>
using namespace std;

int main() {
    cout << "Hello World from C++!" << endl;
    return 0;
}
"""

with tempfile.NamedTemporaryFile(delete=False, suffix=".cpp") as f:
    cpp_file = f.name
    f.write(cpp_code.encode())

exe_file = cpp_file.replace(".cpp", "")

compile_proc = subprocess.run(["g++", cpp_file, "-o", exe_file], capture_output=True, text=True)

if compile_proc.returncode != 0:
    print("C++ compilation failed!")
    print(compile_proc.stderr)
else:
    result = subprocess.run([exe_file], capture_output=True, text=True)
    print("Output from C++:")
    print(result.stdout.strip())

os.remove(cpp_file)
if os.path.exists(exe_file):
    os.remove(exe_file)
```

## # How It Works

1. **Python writes C++ code** to a temporary file.
2. **g++ compiles** the code into an executable.
3. **Python runs the executable** and prints its output.
4. **Cleanup** removes temporary files for a tidy finish.

## # Key Learnings

- Use **`subprocess.run()`** to execute system commands from Python.
- Create **temporary files** using `tempfile` safely.
- Capture **compiler output and program results**.
- Integrate Python and C++ seamlessly for automation or hybrid projects.

## # Further Experiments

- Modify the C++ code dynamically (e.g., generate code from user input).
- Run multiple C++ snippets from one Python script.
- Measure **execution time differences** between Python and C++.
- Use **Python to benchmark or visualize C++ results**.

---

With this simple integration, you’ve built a **bridge between Python and C++**, unlocking a world of hybrid programming possibilities. You can extend this example to dynamically generate more complex C++ programs, compile them on the fly, and benchmark their performance using Python scripts.

---

**Website:** https://www.pyshine.com
**Author:** PyShine
