---
description: FastAPI Lab 2 – Build a Personalized Jokes API with categories using FastAPI and PyJokes. Beginner-friendly step-by-step tutorial.
featured-img: 20251120-fastapi-jokes-categories/20251120-fastapi-jokes-categories
keywords:
- Python
- FastAPI
- API
- web development
- backend
- pyjokes
- tutorial
- beginner
layout: post
mathjax: false
tags:
- python
- fastapi
- api
- backend
- tutorial
- beginner
title: FastAPI Lab 2 – Personalized Jokes API with Categories
---

## Introduction

Welcome to **Lab 2** of your FastAPI learning journey!  
In the previous lab, you built a basic Jokes API.  
Now, it's time to **upgrade your API** by adding **categories** and **query parameters**.

This tutorial is beginner‑friendly and explains everything step-by-step.

By the end of this lab, you will know how to:

- Add query parameters to FastAPI routes  
- Pass values dynamically through URLs  
- Create customized API responses  
- Use PyJokes categories

---

## Understanding the Requirements

You will build an API that:

- Accepts a query parameter named `category`
- Supports multiple categories:
  - `neutral`
  - `chuck`
  - `all`
- Returns a random joke from the selected category
- Prints the selected category and joke in the console for debugging

---

## Setting Up

Install the required packages:

```bash
pip install fastapi uvicorn pyjokes
```

Create a file named **`main.py`**.

---

## Writing the API Code

Here is your full Lab 2 code:

```python
# Lab2: Personalized Jokes API with Categories
# pip install fastapi uvicorn pyjokes
# Command to run app: 
# python -m uvicorn main:app --reload

from fastapi import FastAPI, Query
import pyjokes 

app = FastAPI()

@app.get("/joke")
def tell_a_joke(category: str = Query(
    default="neutral",
    description="Category of Joke: neutral, chuck, all"
)):
    """
    Returns a random joke.
    Optional query parameter: category (default: neutral)
    """
    joke = pyjokes.get_joke(category=category)
    print(f"Category: {category} | Joke: {joke}")
    return {"category": category, "joke": joke}
```

---

## Code Explanation

### 1. Importing Modules

```python
from fastapi import FastAPI, Query
import pyjokes
```

- `Query` allows you to define query parameters with defaults and descriptions.

### 2. Creating the App

```python
app = FastAPI()
```

### 3. Defining the Endpoint

```python
@app.get("/joke")
```

This means:  
> When someone visits `/joke`, run the following function.

### 4. Adding a Query Parameter

```python
category: str = Query(default="neutral",
                      description="Category of Joke: neutral, chuck, all")
```

This allows the user to request:

```
/joke?category=neutral
/joke?category=chuck
/joke?category=all
```

If no category is given, it defaults to `"neutral"`.

### 5. Getting the Joke

```python
joke = pyjokes.get_joke(category=category)
```

PyJokes automatically selects a random joke from that category.

### 6. Returning JSON Response

```python
return {"category": category, "joke": joke}
```

---

## Running the API

Run the server using:

```bash
python -m uvicorn main:app --reload
```

---

## Testing Your API

Open your browser or use curl/Postman:

### Neutral jokes (default)

```
http://127.0.0.1:8000/joke
```

### Chuck Norris jokes:

```
http://127.0.0.1:8000/joke?category=chuck
```

### All jokes:

```
http://127.0.0.1:8000/joke?category=all
```

---

## Automatic API Documentation

FastAPI gives you free docs again!

### Swagger UI

```
http://127.0.0.1:8000/docs
```

### ReDoc

```
http://127.0.0.1:8000/redoc
```

Here you can test the `category` query parameter interactively.

---

## Conclusion

In this lab, you learned how to:

- Add query parameters with defaults  
- Make your API dynamic  
- Personalize responses  
- Use PyJokes categories  
- Read logs printed in your server console  

You now have a flexible Jokes API that users can interact with!

---

**Website:** https://www.pyshine.com  
**Author:** PyShine
