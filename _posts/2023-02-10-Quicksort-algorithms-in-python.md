---
layout: post
title: How to sort a list using Quicksort algorithms
mathjax: true
featured-img: 26072022-python-logo
summary:  Different functions to sort numbers
---

Hello friends! There are several variations of the quicksort algorithm, each with its own approach to choosing the pivot element, partitioning the data, and optimizing performance. Some of the most common quicksort schemes include:

* Classic Quicksort: In this version, the pivot is usually chosen as the first, last, or middle element of the array.

* Hoare Partition Scheme: This version of quicksort uses a two-pointer approach to partition the data and was developed by Tony Hoare.

* Lomuto Partition Scheme: Named after Nico Lomuto, this version of quicksort uses a single-pointer approach to partition the data and is simpler to implement than the Hoare scheme.

* Tail Recursive Quicksort: This version of quicksort uses tail recursion optimization to reduce the call stack and improve performance.

* Randomized Quicksort: In this version, the pivot is chosen randomly to improve the performance of the algorithm and avoid worst-case scenarios.

We can look at 5 different quicksort schemes. However, there could be more variations or modifications of the quicksort algorithm that have been developed and used in specific contexts.

# Classic Quicksort Scheme
Here is a Python implementation of the classic QuickSort algorithm:
```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    return quick_sort(less) + [pivot] + quick_sort(greater)

numbers = [114, 70, 2, -8, 1, 3, 5, 6]
n = len(numbers)
print("Sorted numbers: ", quick_sort(numbers))
```
output:
```
Sorted numbers:  [-8, 1, 2, 3, 5, 6, 70, 114]
```

The quick_sort function takes a list arr as input and returns the sorted list. The function uses the first element of the list as the pivot and splits the rest of the list into two sublists: less and greater. The less list contains all elements that are less than or equal to the pivot, and the greater list contains all elements that are greater than the pivot. The function then recursively calls itself on each of these sublists until the base case is reached, when the list has only one element. The base case returns the input list as it is already sorted. The sorted sublists are then combined and returned as the result of the function.


#  Hoare Partition Scheme
The Hoare Partition Scheme is a popular algorithm used for quick sort. It works by selecting a pivot element, and then partitioning the elements into two groups such that all elements less than the pivot are in one group and all elements greater than the pivot are in another. Here's the code to implement Hoare Partition Scheme in Python:
```python
def hoare_partition(arr, low, high):
    pivot = arr[low]
    i = low - 1
    j = high + 1
    while True:
        i = i + 1
        while arr[i] < pivot:
            i = i + 1
        j = j - 1
        while arr[j] > pivot:
            j = j - 1
        if i >= j:
            return j
        arr[i], arr[j] = arr[j], arr[i]

def quick_sort(arr, low, high):
    if low < high:
        p = hoare_partition(arr, low, high)
        quick_sort(arr, low, p)
        quick_sort(arr, p+1, high)

numbers = [114, 70, 2, -8, 1, 3, 5, 6]
n = len(numbers)
quick_sort(numbers, 0, n-1)
print("Sorted numbers: ", numbers)
```
output:
```
Sorted numbers:  [-8, 1, 2, 3, 5, 6, 70, 114]
```

# Lomuto Partition Scheme
This implementation uses the Lomuto partition scheme, which is a common partition scheme used in the QuickSort algorithm. 
The quick_sort function is the main function that implements the QuickSort algorithm, while the partition function is used to partition the array around the pivot element. 
The pivot element is the last element in the array, in this implementation.

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

# Tail Recursive Quicksort
Tail Recursion is a technique where the last statement in a function is a recursive call, allowing the function to return immediately, without having to keep the current function's state in memory. This can be particularly useful for sorting algorithms, such as QuickSort, which can have a large number of recursive calls. Here's a python implementation of Tail Recursive QuickSort:

```python
def tail_recursive_quicksort(arr, low, high):
    while low < high:
        pivot = partition(arr, low, high)
        tail_recursive_quicksort(arr, low, pivot-1)
        low = pivot + 1

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i + 1

numbers = [114, 70, 2, -8, 1, 3, 5, 6]
n = len(numbers)
tail_recursive_quicksort(numbers, 0, n - 1)
print("Sorted numbers: ", numbers)
```
output:
```
Sorted numbers:  [-8, 1, 2, 3, 5, 6, 70, 114]
```
In this implementation, the tail_recursive_quicksort function uses a while loop to handle the recursion, and the partition function is used to find the pivot element. The partition function is the same as in a traditional QuickSort implementation, and it rearranges the elements in the array so that all elements less than or equal to the pivot are to the left of the pivot, and all elements greater than the pivot are to the right. The pivot is then returned, so that the tail_recursive_quicksort function knows where to split the array for the next iteration.

The tail_recursive_quicksort function sorts the array arr between indices low and high. To sort the entire array, you would call tail_recursive_quicksort(arr, 0, len(arr) - 1).

# Randomized Quicksort
Here's an implementation of randomized quicksort in Python:

```python
import random

def partition(arr, low, high):
    pivot_index = random.randint(low, high)
    pivot = arr[pivot_index]
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def randomized_quicksort(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)
        randomized_quicksort(arr, low, pivot_index - 1)
        randomized_quicksort(arr, pivot_index + 1, high)

numbers = [114, 70, 2, -8, 1, 3, 5, 6]
randomized_quicksort(numbers, 0, len(numbers) - 1)
print("Sorted numbers: ", numbers)
```
output:
```
Sorted numbers:  [-8, 1, 2, 3, 5, 6, 70, 114]
```

# Conclusion
There is no one "best" quicksort algorithm in terms of runtime, as the performance of quicksort can depend on various factors such as the size and distribution of the input, the choice of pivot element, and the implementation details. However, some common strategies can be used to improve the average-case runtime of quicksort.

One approach to optimize quicksort is to use a pivot selection strategy that tends to result in more balanced partitions, such as choosing the median of the input as the pivot. This can help to reduce the likelihood of ending up with unbalanced partitions, which can lead to worse-case performance.

Another strategy is to use a hybrid approach that combines quicksort with another sorting algorithm, such as insertion sort. This can help to speed up the sorting of small sub-arrays, which can be a common bottleneck in quicksort.

It's also worth noting that in practice, other sorting algorithms, such as merge sort, can sometimes be faster than quicksort for large inputs. The choice of sorting algorithm often depends on the specific requirements and constraints of the application.

In conclusion, the best quicksort algorithm in terms of runtime will depend on the specific input, requirements, and constraints of the problem.
