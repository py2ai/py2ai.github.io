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

### A simple heuristic algorithm to solve TSP

Let's try to solve the Traveling Salesman Problem (TSP) using Python:
```python
import random
random.seed(1)
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

def get_total_distance(cities, solution):
    distance = 0
    for i in range(len(solution) - 1):
        distance += dist(cities[solution[i]], cities[solution[i + 1]])
    return distance

points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(50)]

print('=== Heuristic algorithm to solve TSP ===')
print('Our points are:', points)
print('Total points are:', len(points))
solution = solve_tsp(points)
print('Solution:', solution)
print('Total solution points:', len(solution))
print("Total distance:", get_total_distance(points, solution))
```
output:
```
=== Heuristic algorithm to solve TSP ===
Our points are: [(2, 9), (1, 4), (1, 7), (7, 7), (10, 6), (3, 1), (7, 0), (6, 6), (9, 0), (7, 4), (3, 9), (1, 5), (0, 0), (0, 10), (8, 0), (6, 10), (3, 6), (0, 8), (3, 7), (7, 8), (3, 5), (3, 10), (3, 7), (4, 0), (6, 8), (10, 1), (2, 10), (4, 1), (5, 8), (6, 8), (10, 3), (4, 4), (9, 7), (8, 6), (9, 0), (7, 3), (6, 6), (10, 2), (5, 8), (10, 5), (1, 7), (10, 8), (1, 2), (8, 6), (5, 7), (0, 7), (0, 4), (9, 9), (9, 6), (10, 2)]
Total points are: 50
Solution: [0, 10, 21, 26, 13, 17, 45, 2, 40, 11, 1, 46, 42, 5, 27, 23, 6, 14, 8, 34, 25, 37, 49, 30, 39, 4, 48, 32, 33, 43, 3, 19, 24, 29, 28, 38, 44, 7, 36, 9, 35, 31, 20, 16, 18, 22, 15, 47, 41, 12]
Total solution points: 50
Total distance: 72.56692978905966
```
The solve_tsp function implements a greedy algorithm to solve the TSP. It starts at a random point and chooses the nearest unvisited point as the next point to visit. This process is repeated until all points have been visited. The dist function calculates the Euclidean distance between two points.

This is just one example of a heuristic solution to the TSP. There are many other algorithms that can be used, such as simulated annealing, genetic algorithms, and ant colony optimization. The choice of algorithm will depend on the specific problem and the desired trade-off between solution quality and computation time.

### A simple local search algorithm to solve TSP

Here we will do a local search algorithm for solving the Traveling Salesman Problem (TSP) in Python:
```python
import random
random.seed(1)
def local_search_tsp(cities, initial_solution):
    best_solution = initial_solution
    best_distance = get_total_distance(cities, best_solution)
    improved = True
    while improved:
        improved = False
        for i in range(len(best_solution) - 1):
            for j in range(i + 1, len(best_solution)):
                new_solution = best_solution[:]
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                new_distance = get_total_distance(cities, new_solution)
                if new_distance < best_distance:
                    best_solution = new_solution
                    best_distance = new_distance
                    improved = True
    return best_solution

def get_total_distance(cities, solution):
    distance = 0
    for i in range(len(solution) - 1):
        distance += get_distance(cities[solution[i]], cities[solution[i + 1]])
    return distance

def get_distance(city1, city2):
    return ((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2) ** 0.5

def generate_random_solution(n):
    return random.sample(range(n), n)


points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(50)]
print('=== Local search algorithm to solve TSP ===')
print('Our points are:', points)
initial_solution = generate_random_solution(len(points))
best_solution = local_search_tsp(points, initial_solution)
print("Best solution:", best_solution)
print('Total solution points:', len(best_solution))
print("Total distance:", get_total_distance(points, best_solution))
```
```output
=== Local search algorithm to solve TSP ===
Our points are: [(2, 9), (1, 4), (1, 7), (7, 7), (10, 6), (3, 1), (7, 0), (6, 6), (9, 0), (7, 4), (3, 9), (1, 5), (0, 0), (0, 10), (8, 0), (6, 10), (3, 6), (0, 8), (3, 7), (7, 8), (3, 5), (3, 10), (3, 7), (4, 0), (6, 8), (10, 1), (2, 10), (4, 1), (5, 8), (6, 8), (10, 3), (4, 4), (9, 7), (8, 6), (9, 0), (7, 3), (6, 6), (10, 2), (5, 8), (10, 5), (1, 7), (10, 8), (1, 2), (8, 6), (5, 7), (0, 7), (0, 4), (9, 9), (9, 6), (10, 2)]
Best solution: [21, 10, 28, 38, 14, 34, 8, 25, 39, 4, 32, 3, 15, 45, 17, 13, 26, 0, 2, 40, 11, 1, 46, 42, 12, 23, 6, 27, 5, 20, 31, 35, 9, 16, 22, 18, 44, 33, 43, 48, 30, 37, 49, 36, 7, 24, 29, 19, 47, 41]
Total solution points: 50
Total distance: 98.02978032014208
```
