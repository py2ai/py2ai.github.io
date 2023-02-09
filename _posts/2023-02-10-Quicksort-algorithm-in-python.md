---
layout: post
title: How to sort a list using Quicksort
mathjax: true
featured-img: 26072022-python-logo
summary:  Making functions to sort 
---

Hello friends! There are several variations of the quicksort algorithm, each with its own approach to choosing the pivot element, partitioning the data, and optimizing performance. Some of the most common quicksort schemes include:

* Classic Quicksort: In this version, the pivot is usually chosen as the first, last, or middle element of the array.

* Hoare Partition Scheme: This version of quicksort uses a two-pointer approach to partition the data and was developed by Tony Hoare.

* Lomuto Partition Scheme: Named after Nico Lomuto, this version of quicksort uses a single-pointer approach to partition the data and is simpler to implement than the Hoare scheme.

* Tail Recursive Quicksort: This version of quicksort uses tail recursion optimization to reduce the call stack and improve performance.

* Randomized Quicksort: In this version, the pivot is chosen randomly to improve the performance of the algorithm and avoid worst-case scenarios.

So, there are at least 5 different quicksort schemes. However, there could be more variations or modifications of the quicksort algorithm that have been developed and used in specific contexts.









This implementation uses the Lomuto partition scheme, which is a common partition scheme used in the QuickSort algorithm. 
The quick_sort function is the main function that implements the QuickSort algorithm, while the partition function is used to partition the array around the pivot element. 
The pivot element is the last element in the array, in this implementation.

{% include codeHeader.html %}
```python
def quick_sort(array, low, high):
    if low < high:
        pivot = partition(array, low, high)
        quick_sort(array, low, pivot-1)
        quick_sort(array, pivot+1, high)

# Lomuto partition scheme
def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1

numbers = [114, 70, 2, -8, 1, 3, 5, 6]
n = len(numbers)
quick_sort(numbers, 0, n - 1)
print("Sorted numbers: ", numbers)
```

output:

```
Sorted numbers:  [-8, 1, 2, 3, 5, 6, 70, 114]
```
