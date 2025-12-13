---
layout: post
title: "Automatically Free a Busy Port in Python Using psutil"
description: "How to detect and safely terminate processes occupying a TCP port in Python using psutil. A beginner-friendly, real-world tutorial."
date: 2025-12-13
author: PyShine
categories:
- Python
- Networking
tags:
- python
- psutil
- sockets
- tcp
- dev-tools
featured-img: 20251213-psutil-port/20251213-psutil-port
---

# Automatically Free a Busy Port in Python Using psutil

If you’ve ever tried to start a Python server and received an error like **"Address already in use"**, this tutorial is for you. We’ll learn how to **detect which process is using a port and safely terminate it** using Python and the `psutil` library.

This guide is written for **Python beginners**, with clear explanations, examples, and real-world use cases.

---

## What Problem Does This Solve?

When running a server (TCP, HTTP, Flask, FastAPI, etc.), the program needs exclusive access to a port (like `12345` or `8000`).

Sometimes:
- A previous server didn’t shut down correctly
- Another application is already using the port
- A crashed process is still running in the background

This causes errors such as:

```
OSError: [Errno 48] Address already in use
```
or in Windows operating system

```
OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
```

This script **automatically finds and kills the process using the port**, then starts the server cleanly.

---

## Installing Required Library

We use the `psutil` library to inspect running processes.

```bash
pip install psutil
```

---

## Full Working Code

```python
# pip install psutil
import socket
import time
import psutil


def scan_and_terminate(proc, port):
    """
    Check if a process is using the given port.
    If yes, terminate that process.
    Returns True if a process was killed.
    """
    for conn in proc.connections(kind="inet"):
        if conn.laddr.port == port:
            name, pid = proc.info["name"], proc.pid
            print(f"[INFO] Port {port} busy!")
            print(f"[INFO] Terminating {name} (PID {pid})")
            proc.kill()
            time.sleep(0.5)
            return True
    return False


def terminate_process_on_port(port):
    """
    Scan all running processes and terminate
    the one holding the given port.
    """
    for proc in psutil.process_iter(attrs=["pid", "name"]):
        try:
            if scan_and_terminate(proc, port):
                return
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass


def start_server():
    """
    Start a TCP server after freeing the port
    """
    HOST, PORT = "127.0.0.1", 12345

    # Ensure port is free
    terminate_process_on_port(PORT)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen()

    print(f"[INFO] Server running on {HOST}:{PORT}")

    while True:
        time.sleep(1)


if __name__ == "__main__":
    start_server()
```

---

## Code Explanation (Beginner Friendly)

### 1. Scanning Processes for Port Usage

```python
proc.connections(kind="inet")
```

This checks all **network connections** of a process (TCP & UDP).

We compare the local port:

```python
if conn.laddr.port == port:
```

If it matches, the process is using the port.

---

### 2. Terminating the Process

```python
proc.kill()
```

This forcefully stops the process holding the port.

We then pause briefly:

```python
time.sleep(0.5)
```

This gives the OS time to release the port.

---

### 3. Iterating Over All Running Processes

```python
psutil.process_iter(attrs=["pid", "name"])
```

This safely loops over **all running processes** without crashing if access is denied.

---

### 4. Starting the TCP Server

```python
s.bind((HOST, PORT))
```

Once the port is free, the server starts successfully.

---

## Why This Code Is Important

- Prevents **"Address already in use"** errors
- Automates cleanup during development
- Makes scripts **self-healing**
- Saves time restarting servers manually

---

## Real-World Use Cases

### ✅ Development Servers
- Flask / FastAPI
- Socket servers
- Local testing environments

### ✅ Game Servers
- Multiplayer socket-based games

### ✅ Embedded & IoT Systems
- Restarting services automatically

### ✅ CI/CD & Automation
- Ensure ports are free before deployment

---

## Safety Notes

- This script **kills processes forcefully**
- Use only in **development environments**
- Avoid running as root unless necessary
- Never blindly kill ports on production machines

---

## Common Questions (FAQ)

**Q: Will this work on macOS, Linux, and Windows?**  
Yes. `psutil` is cross-platform.

**Q: Can I terminate multiple processes on the same port?**  
Ports are exclusive—only one process can bind to a port at a time.

**Q: Can I make it safer?**  
Yes! You can prompt the user before killing the process.

---

## Beginner Exercise

Try modifying the script to:
- Ask the user before terminating a process
- Log killed processes to a file
- Support multiple ports

---

## Conclusion

This script is a powerful example of how Python can interact with the operating system to solve real-world problems. Learning `psutil` opens the door to system monitoring, automation, and professional-grade tools.

Happy hacking!

