```python
from typing import List
from math import ceil
from collections import defaultdict
```

# Binary Search: Easy Problems

## Binary Search

https://leetcode.com/problems/binary-search/

You are given an array of distinct integers nums, sorted in ascending order, and an integer target.

Implement a function to search for target within nums. If it exists, then return its index, otherwise, return -1.

Your solution must run in O(logn) time.

Example 1:

```
Input: nums = [-1,0,2,4,6,8], target = 4
Output: 3
```

Example 2:

```
Input: nums = [-1,0,2,4,6,8], target = 3
Output: -1
```

Constraints:

```
1 <= nums.length <= 10000.
-10000 < nums[i], target < 10000
```


```python
def search(nums: List[int], target: int) -> int:
    """Optimal solution O(logn), O(1)"""
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def binary_search(nums: List[int], target: int) -> int:
    """Recursive solution O(logn), O(logn)"""
    def recursive_search(left: int, right: int) -> int:
        if left > right:
            return -1

        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            return recursive_search(mid + 1, right)
        else:
            return recursive_search(left, mid - 1)

    return recursive_search(0, len(nums) - 1)
```


```python
nums = [-1,0,2,4,6,8]
target = 4
search(nums, target)
```




    3




```python
binary_search(nums, target)
```




    3



# Binary Search: Medium Problems

Problem 1: Search a 2D Matrix

https://leetcode.com/problems/search-a-2d-matrix/

You are given an `m x n` 2-D integer array `matrix` and an integer `target`.

Each row in `matrix` is sorted in non-decreasing order.
The first integer of every row is greater than the last integer of the previous row.
Return true if target exists within matrix or false otherwise.

Can you write a solution that runs in `O(log(m * n))` time?

Example 1:

```
Input: matrix = [[1,2,4,8],[10,11,12,13],[14,20,30,40]]
target = 10
Output: true
```

Example 2:

```
Input: matrix = [[1,2,4,8],[10,11,12,13],[14,20,30,40]]
target = 15
Output: false
```

Constraints:

```
m == matrix.length
n == matrix[i].length
1 <= m, n <= 100
-10000 <= matrix[i][j], target <= 10000
```



```python
def searchMatrix(matrix: List[List[int]], target: int) -> bool:
    """This solution runs in O(logm + logn), or O(log(m*n)). O(1) space"""
    rows, cols = len(matrix), len(matrix[0])

    l, r = 0, cols - 1
    top, bottom = 0, rows - 1

    # O(logm)
    while top <= bottom:

        # Find the middle row
        row = (top + bottom) // 2

        if target > matrix[row][-1]:
            top = row + 1
        elif target < matrix[row][0]:
            bottom = row - 1
        else:
            break

    if not (top <= bottom):
        return False

    row = (top + bottom) // 2

    # O(logn)
    while l <= r:
        mid = (l + r) // 2
        if matrix[row][mid] == target:
            return True
        elif matrix[row][mid] < target:
            l = mid + 1
        else:
            r = mid - 1

    return False

def searchMatrix_one_pass(matrix: List[List[int]], target: int) -> bool:
    """Same as above, but in one pass. O(logm + logn), O(1) space. Essentially this flattens the matrix into a 1D array."""

    rows, cols = len(matrix), len(matrix[0])
    l, r = 0, rows * cols - 1

    while l <= r:
        mid = (l + r) // 2
        row, col = mid // cols, mid % cols
        if target > matrix[row][col]:
            l = mid + 1
        elif target < matrix[row][col]:
            r = mid - 1
        else:
            return True
    return False
```


```python
matrix = [[1,2,4,8],[10,11,12,13],[14,20,30,40]]
target = 10
searchMatrix(matrix, target)
searchMatrix_one_pass(matrix, target)
```




    True




```python
matrix = [[1,2,4,8],[10,11,12,13],[14,20,30,40]]
target = 15
searchMatrix(matrix, target)
searchMatrix_one_pass(matrix, target)
```




    False



## Problem 2: Koko Eating Bananas

https://leetcode.com/problems/koko-eating-bananas/

You are given an integer array `piles` where `piles[i]` is the number of bananas in the ith pile. You are also given an integer `h`, which represents the number of hours you have to eat all the bananas.

You may decide your bananas-per-hour eating rate of `k`. Each hour, you may choose a pile of bananas and eats `k` bananas from that pile. If the pile has less than `k` bananas, you may finish eating the pile but you can not eat from another pile in the same hour.

Return the minimum integer k such that you can eat all the bananas within h hours.

Example 1:

```
Input: piles = [1,4,3,2], h = 9
Output: 2
```

Explanation: With an eating rate of 2, you can eat the bananas in 6 hours. With an eating rate of 1, you would need 10 hours to eat all the bananas (which exceeds h=9), thus the minimum eating rate is 2.

Example 2:

```
Input: piles = [25,10,23,4], h = 4
Output: 25
```

Constraints:

```
1 <= piles.length <= 1,000
piles.length <= h <= 1,000,000
1 <= piles[i] <= 1,000,000,000
```


```python
def minEatingSpeed(piles: List[int], h: int) -> int:
    l, r = 1, max(piles)
    res = r

    # Binary search for the minimum eating speed
    while l <= r:
        mid = (l + r) // 2
        # If Koko can eat all bananas with speed mid, then we need to check if there is a slower speed
        if sum(ceil(pile / mid) for pile in piles) <= h:
            res = mid
            r = mid - 1
        # If Koko cannot eat all bananas with speed mid, then we need to check a faster speed
        else:
            l = mid + 1
    return res

```


```python
piles = [1,4,3,2]
h = 9
minEatingSpeed(piles, h)
```




    2




```python
piles = [25,10,23,4]
h = 4
minEatingSpeed(piles, h)
```




    25




```python
piles = [3,6,7,11]
h = 8
minEatingSpeed(piles, h)
piles = [30,11,23,4,20]
h = 5
minEatingSpeed(piles, h)
piles = [30,11,23,4,20]
h = 6
minEatingSpeed(piles, h)
```




    23



## Problem 3: Find Minimum in Rotated Sorted Array

https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/

You are given an array of length `n` which was originally sorted in ascending order. It has now been rotated between 1 and `n` times. For example, the array `nums = [1,2,3,4,5,6]` might become:

`[3,4,5,6,1,2]` if it was rotated `4` times.
`[1,2,3,4,5,6]` if it was rotated `6` times.

Notice that rotating the array `4` times moves the last four elements of the array to the beginning. Rotating the array `6` times produces the original array.

Assuming all elements in the rotated sorted array nums are unique, return the minimum element of this array.

A solution that runs in `O(n)` time is trivial, can you write an algorithm that runs in `O(log n)` time?

Example 1:

```
Input: nums = [3,4,5,6,1,2]
Output: 1
```

Example 2:

```
Input: nums = [4,5,0,1,2,3]
Output: 0
```

Example 3:

```
Input: nums = [4,5,6,7]
Output: 4
```

Constraints:

```
1 <= nums.length <= 1000
-1000 <= nums[i] <= 1000
```


```python
def findMin(nums: List[int]) -> int:
    """O(logn) solution using binary search"""
    res = nums[0]
    l, r, = 0, len(nums) - 1

    while l <= r:
        # If the section is sorted, then the minimum is the first element
        if nums[l] < nums[r]:
            res = min(res, nums[l])
            break

        # Find the middle element, and update the result if it is smaller
        m = (l+r) // 2
        res = min(res, nums[m])

        # If the left half is smaller (and section is not sorted), then the minimum is in the right half
        if nums[l] < nums[m]:
            l = m + 1
        # If the right half is smaller (and section is not sorted), then the minimum is in the left half
        else:
            r = m - 1
    return res

```


```python
nums = [3,4,5,6,1,2]
findMin(nums)
```




    1




```python
nums = [4,5,0,1,2,3]
findMin(nums)
```




    0




```python
nums = [4,5,6,7]
findMin(nums)
```




    4



## Problem 4: Search in Rotated Sorted Array

https://leetcode.com/problems/search-in-rotated-sorted-array/

You are given an array of length `n` which was originally sorted in ascending order. It has now been rotated between `1` and `n` times. For example, the array `nums = [1,2,3,4,5,6]` might become:

`[3,4,5,6,1,2]` if it was rotated `4` times.
`[1,2,3,4,5,6]` if it was rotated `6` times.

Given the rotated sorted array `nums` and an integer `target`, return the index of `target` within `nums`, or `-1` if it is not present.

You may assume all elements in the sorted rotated array `nums` are unique,

A solution that runs in `O(n)` time is trivial, can you write an algorithm that runs in `O(log n)` time?

Example 1:

```
Input: nums = [3,4,5,6,1,2], target = 1
Output: 4
```

Example 2:

```
Input: nums = [3,5,6,0,1,2], target = 4
Output: -1
```

Constraints:

```
1 <= nums.length <= 1000
-1000 <= nums[i] <= 1000
-1000 <= target <= 1000
```


```python
def search(nums: List[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    pivot = nums[0]

    # Find the pivot
    while l <= r:
        m = (l + r) // 2
        # Check if the target was found
        if target == nums[m]:
            return m

        # If the left half is sorted
        if nums[l] <= nums[m]:
            # If the target is in the left half, then update the right pointer
            if nums[l] <= target <= nums[m]:
                r = m - 1
            else:
                l = m + 1

        # If the right half is sorted
        else:
            # If the target is in the right half, then update the left pointer
            if nums[m] <= target <= nums[r]:
                l = m + 1
            else:
                r = m - 1

    return -1
```


```python
nums = [3,4,5,6,1,2]
target = 1
search(nums, target)
```




    4



## Problem 5: Time Based Key-Value Store

https://leetcode.com/problems/time-based-key-value-store/

Implement a time-based key-value data structure that supports:

- Storing multiple values for the same key at specified time stamps
- Retrieving the key's value at a specified timestamp

Implement the TimeMap class:

- `TimeMap()` Initializes the object.
- `void set(String key, String value, int timestamp)` Stores the key key with the value value at the given time timestamp.
- `String get(String key, int timestamp)` Returns the most recent value of key if set was previously called on it and the most recent timestamp for that key prev_timestamp is less than or equal to the given timestamp (`prev_timestamp <= timestamp`). If there are no values, it returns `""`.

Note: For all calls to set, the timestamps are in strictly increasing order.

Example 1:

```
Input:
["TimeMap", "set", ["alice", "happy", 1], "get", ["alice", 1], "get", ["alice", 2], "set", ["alice", "sad", 3], "get", ["alice", 3]]

Output:
[null, null, "happy", "happy", null, "sad"]
```

Explanation:
```
TimeMap timeMap = new TimeMap();
timeMap.set("alice", "happy", 1);  // store the key "alice" and value "happy" along with timestamp = 1.
timeMap.get("alice", 1);           // return "happy"
timeMap.get("alice", 2);           // return "happy", there is no value stored for timestamp 2, thus we return the value at timestamp 1.
timeMap.set("alice", "sad", 3);    // store the key "alice" and value "sad" along with timestamp = 3.
timeMap.get("alice", 3);           // return "sad"
```

Constraints:

```
1 <= key.length, value.length <= 100
key and value only include lowercase English letters and digits.
1 <= timestamp <= 1000
```


```python
class TimeMap:

    def __init__(self):
        self.vals = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.vals:
            self.vals[key] = []
        self.vals[key].append((timestamp, value))


    def get(self, key: str, timestamp: int) -> str:
        res = ""
        vals = self.vals[key]

        # We can assume the values are sorted by timestamp, given question constraints.
        # Otherwise, we would need to sort, or linear search.
        # vals.sort(key=lambda x: x[0])

        l, r = 0, len(vals) - 1
        while l <= r:
            m = (l + r) // 2
            if vals[m][0] <= timestamp:
                res = vals[m][1]
                l = m + 1
            else:
                r = m - 1
        return res


```


```python
time_map = TimeMap()
time_map.set("alice", "happy", 1)
time_map.get("alice", 1)
time_map.get("alice", 2)
time_map.set("alice", "sad", 3)
time_map.get("alice", 3)
```




    'sad'



# Binary Search: Hard Problems

## Problem 1: Median of Two Sorted Arrays

https://leetcode.com/problems/median-of-two-sorted-arrays/

You are given two integer arrays `nums1` and `nums2` of size `m` and `n` respectively, where each is sorted in ascending order. Return the median value among all elements of the two arrays.

Your solution must run in `O(log(m+n))` time.

Example 1:

```
Input: nums1 = [1,2], nums2 = [3]
Output: 2.0
```

Explanation: Among `[1, 2, 3]` the median is `2`.

Example 2:

```
Input: nums1 = [1,3], nums2 = [2,4]
Output: 2.5
```

Explanation: Among [1, 2, 3, 4] the median is (2 + 3) / 2 = 2.5.

Constraints:

```
nums1.length == m
nums2.length == n
0 <= m <= 1000
0 <= n <= 1000
-10^6 <= nums1[i], nums2[i] <= 10^6
```


```python
def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    # Set a to be the shorter array, and b to be the longer array
    if len(nums1) > len(nums2):
        a, b = nums2, nums1
    else:
        a, b = nums1, nums2

    # Get the total length and half length, set pointers for array a
    total_len = len(a) + len(b)
    half_len = total_len // 2
    l, r = 0, len(a) - 1

    while True:
        i = (l + r) // 2    # Pointer for a
        j = half_len - i - 2    # Pointer for b

        # Get the values of the partitions, if out of bounds, then use -inf or inf
        a_left = a[i] if i >= 0 else float('-inf')
        a_right = a[i+1] if (i+1) < len(a) else float('inf')
        b_left = b[j] if j >= 0 else float('-inf')
        b_right = b[j+1] if (j+1) < len(b) else float('inf')

        # Check if we have found the correct partition
        if a_left <= b_right and b_left <= a_right:
            # We have found the correct partition, now we need to find the median
            if total_len % 2 == 1:
                return min(a_right, b_right)
            else:
                return (max(a_left, b_left) + min(a_right, b_right)) / 2
        # If the partition is incorrect, then we need to move the pointers
        elif a_left > b_right:
            # Move the right pointer for a
            r = i - 1
        else:
            # Move the left pointer for a
            l = i + 1
```


```python
nums1 = [1,2]
nums2 = [3]
findMedianSortedArrays(nums1, nums2)
```




    2




```python
nums1 = [1,3]
nums2 = [2,4]
findMedianSortedArrays(nums1, nums2)
```




    2.5




```python

```
