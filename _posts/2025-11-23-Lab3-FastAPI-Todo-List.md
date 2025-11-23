---
description: Lab 3 beginner-friendly tutorial to create a Simple To-Do List API using FastAPI and Pydantic. Step-by-step guide for beginners.
featured-img: 20251123-fastapi-todo/20251123-fastapi-todo
keywords:
- Python
- FastAPI
- API
- backend
- Pydantic
- to-do list
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
title: FastAPI Lab 3 – Build a Simple To-Do List API
---

## Introduction

In this lab, you will learn how to build a **Simple To-Do List API** using **FastAPI** and **Pydantic**.  
This API allows users to add tasks and view all tasks.  

This guide is **beginner-friendly**, and no prior experience with APIs is required.

---

## Understanding the Requirements

You will build an API that:

- Stores tasks in memory (no database required)
- Allows adding tasks via POST requests
- Allows viewing all tasks via GET requests
- Uses Pydantic to validate incoming task data

---

## Setting Up

Install FastAPI and Uvicorn:

```bash
pip install fastapi uvicorn
```

---

## Writing the To-Do API

Create a file named **`main.py`** and add:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# In-memory "database"
tasks = []

# Data model for creating tasks
class Task(BaseModel):
    task: str

@app.post("/add")
def add_task(item: Task):
    tasks.append(item.task)
    return {"message": "Task added!", "task": item.task}

@app.get("/tasks")
def get_tasks():
    return {"tasks": tasks}
```

---

## Code Explanation

### 1. Importing Libraries

- `FastAPI` → to create API endpoints  
- `BaseModel` → to define and validate request data

### 2. Creating the App

```python
app = FastAPI()
```

### 3. Creating an In-Memory Database

```python
tasks = []
```

- `tasks` is a simple Python list that stores all task strings

### 4. Defining the Task Model

```python
class Task(BaseModel):
    task: str
```

- Ensures that the incoming request contains a string field named `task`

### 5. Adding Tasks

```python
@app.post("/add")
def add_task(item: Task):
    tasks.append(item.task)
    return {"message": "Task added!", "task": item.task}
```

- Accepts POST requests to `/add`
- Validates the request body with Pydantic `Task` model
- Adds the task to the in-memory list
- Returns a JSON response confirming the addition

### 6. Retrieving All Tasks

```python
@app.get("/tasks")
def get_tasks():
    return {"tasks": tasks}
```

- Accepts GET requests to `/tasks`
- Returns all tasks stored in memory

---

## Running the Server

```bash
python -m uvicorn main:app --reload
```

- `--reload` automatically restarts the server when you edit the code

---

## Testing Your API

### Add a Task

```bash
curl -X POST "http://127.0.0.1:8000/add" -H "Content-Type: application/json" -d '{"task":"Buy groceries"}'
```

Response:

```json
{
  "message": "Task added!",
  "task": "Buy groceries"
}
```

### View All Tasks

```
http://127.0.0.1:8000/tasks
```

Response:

```json
{
  "tasks": ["Buy groceries"]
}
```

---

## Conclusion

You have now created a **Simple To-Do List API** in Python using FastAPI:

- Learned to handle POST and GET requests
- Used Pydantic for request validation
- Stored data in memory
- Built a working backend API for tasks

You can expand this lab in the future by:

- Adding task deletion
- Adding completion status
- Connecting to a database

---

**Website:** https://www.pyshine.com  
**Author:** PyShine
