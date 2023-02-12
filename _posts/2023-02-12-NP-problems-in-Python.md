---
layout: post
title: What are NP problems
mathjax: true
featured-img: 26072022-python-logo
summary:  Can we solve NP problems?
---

Hello friends! Lets first talk about NP problems and then get to the coding part.

# NP problems
NP (Nondeterministic Polynomial time) problems are a class of computational problems for which no efficient solution algorithm has been found, but for which it is believed that a solution can be verified quickly. In other words, the problem can be solved in an exponential amount of time, but the solution can be checked in polynomial time.

Examples of NP problems include:
* Traveling Salesman Problem, 
* Knapsack Problem, 
* Satisfiability (SAT) Problem.

These problems are of significant interest in the field of theoretical computer science and have been widely studied for many years. Despite much effort, a polynomial-time algorithm for any NP-complete problem has yet to be found.

It is widely believed that NP problems are fundamentally hard and that there is no efficient solution algorithm for them, although this has not been proven. The distinction between NP problems and those that can be solved efficiently (in polynomial time) is one of the most important open questions in computer science and mathematics, and is known as the P versus NP problem.

## Traveling Salesman Problem (TSP) 
The Traveling Salesman Problem (TSP) is a classic optimization problem in computer science and mathematics. It asks the following question: given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the starting city?

In other words, the problem involves finding the minimum-distance round-trip path that visits a given set of cities and returns to the starting point. The TSP has applications in various fields, including operations research, computer science, and transportation planning.

The TSP is an NP-hard problem, meaning that it is computationally difficult to find an exact solution for large sets of cities. As a result, various approximate algorithms have been developed to find near-optimal solutions to the TSP, such as heuristics, local search algorithms, and branch-and-bound algorithms. Let's check them one by one.

```python
import random

def solve_tsp(points):
    tour = []
    used = [False for i in range(len(points))]
    used[0] = True
    tour.append(0)
    for i in range(len(points) - 1):
        best_distance = float('inf')
        best_j = 0
        for j in range(len(points)):
            if not used[j] and dist(points[tour[-1]], points[j]) < best_distance:
                best_distance = dist(points[tour[-1]], points[j])
                best_j = j
        tour.append(best_j)
        used[best_j] = True
    return tour

def dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(50)]
print('Our points are:', points)
print('Solution:', solve_tsp(points))
```
output:
```
Our points are: [(2, 3), (5, 9), (9, 3), (7, 3), (6, 4), (10, 5), (9, 10), (6, 5), (3, 2), (1, 8), (8, 1), (8, 7), (6, 7), (1, 4), (4, 0), (0, 5), (7, 8), (10, 5), (5, 4), (0, 2), (5, 0), (3, 6), (8, 2), (3, 4), (1, 3), (8, 3), (1, 1), (7, 10), (10, 6), (10, 6), (10, 10), (5, 3), (8, 4), (5, 7), (9, 8), (1, 4), (1, 10), (4, 1), (4, 3), (2, 3), (1, 4), (10, 5), (9, 10), (6, 1), (7, 8), (7, 3), (9, 6), (5, 9), (1, 4), (6, 2)]
Solution: [0, 39, 24, 13, 35, 40, 48, 15, 19, 26, 8, 37, 14, 20, 43, 49, 3, 45, 25, 2, 22, 10, 32, 4, 7, 18, 31, 38, 23, 21, 33, 12, 16, 44, 11, 34, 6, 42, 30, 27, 1, 47, 9, 36, 46, 28, 29, 5, 17, 41]
```

