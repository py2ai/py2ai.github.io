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

Have you ever wondered how apps like  **Google Keep** ,  **Todoist** , or even your phone’s reminder app manage and organize your tasks behind the scenes? In this lab, you’ll step into the world of web APIs and build your very own **Simple To-Do List API** using **FastAPI** and  **Pydantic** .

With this API, you’ll be able to  **add tasks** ,  **view all tasks** , and understand how data flows between clients and servers — the same concepts used in real-world apps.

Don’t worry if you’ve never worked with APIs before — this guide is  **beginner-friendly** , walks you through every step, and requires no prior experience. By the end, you’ll have a working backend that could be the foundation for your own productivity app!

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

## GET Requests

- Purpose: Retrieve or fetch data from a server.
- Effect on Server: Does not change server data (read-only).
- Data Transmission: Data is sent via URL query parameters (visible in the URL).
- Idempotence: Yes — making the same GET request multiple times produces the same result.
- Caching: GET requests can be cached by browsers and proxies.
- Use Case Examples:
- Viewing all tasks in a to-do list API
- Retrieving a user profile
- Loading a web page

```python
@app.get("/tasks")
def get_tasks():
    return {"tasks": tasks}
```

Visiting `/tasks` in your browser or via curl retrieves all tasks.

## POST Requests

- Purpose: Send data to the server to create or update resources.
- Effect on Server: Changes server state (e.g., adds a new record).
- Data Transmission: Data is sent in the request body (not visible in the URL).
- Idempotence: Usually no — repeating a POST request can create multiple resources.
- Caching: POST requests are usually not cached.
- Use Case Examples:
- Adding a new task to a to-do list
- Creating a new user account
- Submitting a form

```python
@app.post("/add")
def add_task(item: Task):
    tasks.append(item.task)
    return {"message": "Task added!", "task": item.task}

```

## Key Differences Between GET and POST

| Feature            | GET                  | POST                |
| ------------------ | -------------------- | ------------------- |
| Purpose            | Retrieve data        | Send or create data |
| Alters Server Data | No                   | Yes                 |
| Data Location      | URL query parameters | Request body        |
| Visibility         | URL visible          | Body hidden         |
| Idempotent         | Yes                  | Usually No          |
| Caching            | Can be cached        | Usually not cached  |

### Summary:

- Use GET to read data without changing anything.
- Use POST to send data or make changes to the server.

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
