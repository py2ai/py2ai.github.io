---
description: A complete beginner-friendly FastAPI tutorial. Learn how to build a simple API that returns random jokes using FastAPI and PyJokes.
featured-img: 20251120-fastapi-jokes/20251120-fastapi-jokes
keywords:
- Python
- FastAPI
- API
- backend
- web development
- tutorial
- beginner
layout: post
mathjax: false
tags:
- python
- fastapi
- api
- webdev
- backend
- beginner
- tutorial
title: Lab1 Build Your First Joke API
---
## Introduction

FastAPI is one of the fastest-growing Python frameworks for building APIs.
It is **super fast**, easy to learn, and perfect for beginners.

In this tutorial, you’ll build your **first FastAPI application** — a simple API that returns a random joke using the `pyjokes` library.

This guide is written for **absolute beginners**, so no prior backend experience is required.

---

## Understanding the Requirements

Before writing any code, let’s understand what tools we’ll use:

### What you need:

- **Python 3 installed**
- Basic understanding of functions and imports
- A terminal or command prompt
- Two simple Python libraries:
  - `fastapi`
  - `uvicorn`
  - `pyjokes`

The application we build will:

- Start a FastAPI server
- Provide an endpoint `/joke`
- Return a random joke in JSON format when visited

---

## Setting Up Your Environment

Open your terminal and install the required libraries:

```bash
pip install fastapi uvicorn pyjokes
```

To confirm installation:

```bash
python -m pip show fastapi
python -m pip show uvicorn
python -m pip show pyjokes
```

Everything should show a valid version number.

---

## Creating Your FastAPI App

Create a new file called **`main.py`** and add the following code:

```python
from fastapi import FastAPI
import pyjokes

app = FastAPI()

@app.get("/joke")
def tell_joke():
    print(joke := pyjokes.get_joke())
    return {"joke": joke}
```

### Code Explanation

#### 1. Importing Modules

```python
from fastapi import FastAPI
import pyjokes
```

- `FastAPI` lets you create API endpoints easily.
- `pyjokes` gives us ready-made programming jokes.

#### 2. Creating an App Instance

```python
app = FastAPI()
```

This creates your FastAPI application. Think of it as the “brain” of your API.

#### 3. Creating an Endpoint

```python
@app.get("/joke")
def tell_joke():
```

- `@app.get("/joke")` tells FastAPI:
  > “When someone visits `/joke`, run the function below.”
  >

#### 4. Returning a Joke

```python
return {"joke": joke}
```

FastAPI automatically converts this Python dict into JSON.

---

## Running the FastAPI App

Use the following command in the folder where `main.py` is saved:

```bash
python -m uvicorn main:app --reload
```

### What this means:

- `uvicorn` → server that runs FastAPI
- `main` → the filename `main.py`
- `app` → the FastAPI instance inside that file
- `--reload` → restart automatically whenever you edit the file

---

## Testing Your API

Once the server is running, open your browser and visit:

```
http://127.0.0.1:8000/joke
```

You should see something like:

```json
{
  "joke": "Why do programmers prefer dark mode? Because light attracts bugs!"
}
```

---

## Bonus: Automatic Documentation!

FastAPI automatically generates documentation pages.

Open these:

### Swagger UI

```
http://127.0.0.1:8000/docs
```

### ReDoc

```
http://127.0.0.1:8000/redoc
```

![1763597234731]({{"assets/img/posts/2025-11-17-Lab1-of-FastAPI-for-Beginners/1763597234731.png" | absolute_url}} )

You get beautiful, interactive API docs **for free**.

---

## Complete Source Code

```python
# pip install fastapi uvicorn pyjokes

from fastapi import FastAPI
import pyjokes

app = FastAPI()

@app.get("/joke")
def tell_joke():
    print(joke := pyjokes.get_joke())
    return {"joke": joke}

# Run using:
# python -m uvicorn main:app --reload
```

---

## Conclusion

You have now built your **first FastAPI project**!

You learned:

- How to install FastAPI
- How to create an API endpoint
- How to return JSON responses
- How to run a development server
- How to access auto-generated API docs

This is only the beginning — FastAPI can handle authentication, databases, async tasks, and much more.

Happy coding!

---

**Website:** https://www.pyshine.com
**Author:** PyShine
