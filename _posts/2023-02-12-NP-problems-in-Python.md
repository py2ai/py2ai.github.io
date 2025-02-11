---
layout: post
title: What are NP problems
mathjax: true
featured-img: 26072022-python-logo
description:  Can we solve NP problems?
tags: [NP problems, computational theory, algorithms]
keywords: NP problems, Traveling Salesman Problem, Knapsack Problem, SAT Problem
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

points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(10)]

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
Our points are: [(2, 9), (1, 4), (1, 7), (7, 7), (10, 6), (3, 1), (7, 0), (6, 6), (9, 0), (7, 4)]
Total points are: 10
Solution: [0, 2, 1, 5, 6, 8, 9, 7, 3, 4]
Total solution points: 10
Total distance: 26.249420033622282
```
Time: 0.03s user 0.01s system 80% cpu 0.052 total

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


points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(10)]
print('=== Local search algorithm to solve TSP ===')
print('Our points are:', points)
print('Total points are:', len(points))
initial_solution = generate_random_solution(len(points))
best_solution = local_search_tsp(points, initial_solution)
print("Best solution:", best_solution)
print('Total solution points:', len(best_solution))
print("Total distance:", get_total_distance(points, best_solution))
```
```output
=== Local search algorithm to solve TSP ===
Our points are: [(2, 9), (1, 4), (1, 7), (7, 7), (10, 6), (3, 1), (7, 0), (6, 6), (9, 0), (7, 4)]
Total points are: 10
Best solution: [0, 2, 1, 5, 7, 3, 4, 9, 6, 8]
Total solution points: 10
Total distance: 28.854613645814542
```
Time: 0.03s user 0.01s system 83% cpu 0.051 total

### A simple branch and bound algorithm to solve TSP

The following code returns the minimum distance to visit all cities and the path that achieves this minimum distance. The function TSP implements the main logic of the algorithm and returns both the minimum distance and the path that achieves this distance. The function lowerBound calculates a lower bound on the minimum distance by considering the closest city to start and multiplying it by the number of unvisited cities. The algorithm continues by branching out to all unvisited cities and updating the minimum distance and the best path if a better solution is found.

```python
import numpy as np
import random, math
random.seed(1)
def TSP(cities, start, currDist, bound, path, visited):
    if len(path) == len(cities):
        return currDist + cities[start][path[0]], path + [start]
    minDist = math.inf
    bestPath = None
    for city in range(len(cities)):
        if city not in visited:
            if currDist + cities[start][city] + bound(cities, city, visited) < minDist:
                newDist, newPath = TSP(cities, city, currDist + cities[start][city], bound, path + [city], visited + [city])
                if newDist < minDist:
                    minDist = newDist
                    bestPath = newPath
    return minDist, bestPath

def lowerBound(cities, start, visited):
    remaining = [city for city in range(len(cities)) if city not in visited]
    if not remaining:
        return 0
    minDist = math.inf
    for city in remaining:
        minDist = min(minDist, cities[start][city])
    return minDist * len(remaining)

def dist(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def get_distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = dist(cities[i], cities[j])

    return dist_matrix

def get_total_distance(cities, solution):
    distance = 0
    for i in range(len(solution) - 1):
        distance += dist(cities[solution[i]], cities[solution[i + 1]])
    return distance


path = []
visited = []
start = 0
points = [(random.randint(0, 10), random.randint(0, 10)) for i in range(10)]
print('=== Branch and Bound algorithm to solve TSP ===')
print('Our points are:', points)
print('Total points are:', len(points))
cities = get_distance_matrix(points)
min_best_distance, best_solution = TSP(cities, start, 0, lowerBound, path, visited)
best_solution = best_solution[0:-1]
print("Best solution:", best_solution)
print('Total solution points:', len(best_solution))
print("Total distance:", get_total_distance(points, best_solution))
```
output:
```
=== Branch and Bound algorithm to solve TSP ===
Our points are: [(2, 9), (1, 4), (1, 7), (7, 7), (10, 6), (3, 1), (7, 0), (6, 6), (9, 0), (7, 4)]
Total points are: 10
Best solution: [0, 2, 1, 5, 6, 8, 9, 4, 3, 7]
Total solution points: 10
Total distance: 27.61890333158648
```
Time:  39.61s user 0.11s system 100% cpu 39.394 total

### Conclusion

The best TSP algorithm depends on several factors such as the size of the problem, the complexity of the cost function, and the desired accuracy of the solution. In general, it can be difficult to determine which algorithm is "best" without considering the specific context of a given problem.

Local search algorithms, such as simulated annealing or iterated local search, can be effective for finding good solutions to small and medium-sized TSP problems. These algorithms typically involve randomly generating a starting solution and then making small, local modifications to try and improve the solution.

Heuristic algorithms, such as the nearest neighbor or the Christofides algorithm, can also be used to find good solutions to TSP problems. These algorithms work by making intelligent choices about which cities to visit next based on some heuristic rules, rather than performing a comprehensive search of all possible solutions.

Branch and bound is a complete search algorithm that can be used to find optimal solutions to TSP problems, but it can be very time-consuming for large problems. The algorithm works by systematically exploring the solution space and eliminating large portions of it that cannot contain the optimal solution. This helps to reduce the size of the search space, but it can still be slow for large problems.

In conclusion, the best TSP algorithm will depend on the specific problem at hand and the requirements for the solution, but in general, local search and heuristic algorithms can be a good starting point for finding good solutions to TSP problems.

## knapsack problem 
The knapsack problem is a combinatorial optimization problem that involves finding the combination of items to include in a knapsack so that the total weight is less than or equal to a given weight limit and the total value is maximized. The problem can be solved using dynamic programming, which involves creating a table to store the solutions to subproblems and using those solutions to build up the solution to the original problem.

Here is a python code to solve the knapsack problem:

```python
def knapsack(items, max_weight):
    n = len(items)
    dp = [[0 for j in range(max_weight + 1)] for i in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, max_weight + 1):
            if items[i - 1][1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - items[i - 1][1]] + items[i - 1][0])
    
    res = dp[n][max_weight]
    w = max_weight
    selected = []
    for i in range(n, 0, -1):
        if res <= 0:
            break
        if res == dp[i - 1][w]:
            continue
        else:
            selected.append(i - 1)
            res = res - items[i - 1][0]
            w = w - items[i - 1][1]
    
    return res, selected

# example usage
items = [(60, 10), (100, 20), (120, 30)]
max_weight = 50
print(knapsack(items, max_weight))

```
output
```
(0, [2, 1])
```
This code returns a tuple of two values, the first value is the maximum value that can be obtained by choosing items and the second value is a list of indices of the items that should be included in the knapsack to obtain the maximum value.
The return res, selected statement at the end of the knapsack function returns a tuple of two values:

res: The maximum value that can be obtained by choosing items from the items list with a weight limit of max_weight. This value is stored in the last cell of the dp table and is calculated through the dynamic programming algorithm.

selected: A list of indices of the items that should be included in the knapsack to obtain the maximum value. This list is constructed by tracing back the dp table from the last cell to the first cell and keeping track of the items that are selected along the way. The indices are relative to the original items list.

The tuple of these two values is returned from the knapsack function so that it can be used in the calling code.

## SAT Problem

The Boolean Satisfiability Problem (SAT) is a classical NP-complete problem, which means that it's a problem for which no known efficient solution exists, but its solutions can be verified quickly.

Here's a simple python code to solve the SAT problem using the brute-force method of trying out all possible combinations of truth values for the variables:

```python
from itertools import product

def solve_sat(clauses, variables):
    for assignment in product([True, False], repeat=variables):
        satisfied = True
        for clause in clauses:
            clause_satisfied = False
            for literal in clause:
                if literal > 0 and assignment[abs(literal) - 1]:
                    clause_satisfied = True
                    break
                elif literal < 0 and not assignment[abs(literal) - 1]:
                    clause_satisfied = True
                    break
            if not clause_satisfied:
                satisfied = False
                break
        if satisfied:
            return assignment
    return None
    
clauses = [[1, -2, 3], [1, 2, -3]]
variables = 3
assignment = solve_sat(clauses, variables)
if assignment is None:
    print("The SAT problem is not satisfiable")
else:
    print("The SAT problem is satisfiable:", assignment)

```
output:
```
The SAT problem is satisfiable: (True, True, True)
```
Note: This implementation is not efficient for large instances of the SAT problem, but it serves as a simple demonstration of how the problem can be solved.

