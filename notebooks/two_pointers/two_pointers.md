```python
from typing import List
```

# Two Pointers: Easy Problems

## Problem 1: Valid Palindrome

https://leetcode.com/problems/valid-palindrome/

Given a string `s`, return true if it is a palindrome, otherwise return false.

A palindrome is a string that reads the same forward and backward. It is also case-insensitive and ignores all non-alphanumeric characters.

Example 1:

```
Input: s = "Was it a car or a cat I saw?"
Output: true
```

Explanation: After considering only alphanumerical characters we have "wasitacaroracatisaw", which is a palindrome.

Example 2:

```
Input: s = "tab a cat"
Output: false
```

Explanation: "tabacat" is not a palindrome.

Constraints:

```
1 <= s.length <= 1000
s is made up of only printable ASCII characters.
```


```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        start, end = 0, len(s) - 1

        while start < end:
            # Skip spaces or other special chars
            while start < end and not s[start].isalnum():
                start += 1
            while start < end and not s[end].isalnum():
                end -= 1

            if s[start].lower() != s[end].lower():
                return False
            start += 1
            end -= 1
        return True

    # def alpha_numeric(self, c):
    #     """Return bool for if char is alphanumeric character"""
    #     return (ord('A') <= ord(c) <= ord('Z') or
    #             ord('a') <= ord(c) <= ord('z') or
    #             ord('0') <= ord(c) <= ord('9'))
```

# Two Pointers: Medium Problems

## Problem 1: Two Integer Sum II

https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/

Given an array of integers numbers that is sorted in non-decreasing order.

Return the indices (1-indexed) of two numbers, `[index1, index2]`, such that they add up to a given target number `target` and `index1 < index2`. Note that `index1` and `index2` cannot be equal, therefore you may not use the same element twice.

There will always be exactly one valid solution.

Your solution must use `O(1)` `O(1)` additional space.

Example 1:
```
Input: numbers = [1,2,3,4], target = 3
Output: [1,2]
```

Explanation:
The sum of 1 and 2 is 3. Since we are assuming a 1-indexed array, `index1 = 1`, `index2 = 2`. We return `[1, 2]`.

Constraints:

```
2 <= numbers.length <= 1000
-1000 <= numbers[i] <= 1000
-1000 <= target <= 1000
```


```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i, j = 0, len(numbers) - 1

        while i < j:
            nums_sum = numbers[i] + numbers[j]
            if nums_sum == target:
                return [i + 1, j + 1]
            elif nums_sum < target:
                i += 1
            else:
                j -= 1
        return []

```


```python
numbers = [1,2,3,4]
target = 3

Solution().twoSum(numbers, target)
```




    [1, 2]



## Problem 2: 3Sum

https://leetcode.com/problems/3sum/

Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` where `nums[i] + nums[j] + nums[k] == 0`, and the indices `i`, `j` and `k` are all distinct.

The output should not contain any duplicate triplets. You may return the output and the triplets in any order.

Example 1:

```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

Explanation:

```
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
```

The distinct triplets are `[-1,0,1]` and `[-1,-1,2]`.

Example 2:

```
Input: nums = [0,1,1]
Output: []
```

Explanation: The only possible triplet does not sum up to 0.

Example 3:

```
Input: nums = [0,0,0]
Output: [[0,0,0]]
```

Explanation: The only possible triplet sums up to 0.

Constraints:

```
3 <= nums.length <= 1000
-10^5 <= nums[i] <= 10^5
```



```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        triplets = []

        for i, a in enumerate(nums):
            # If the current number is greater than 0, we can't find a valid triplet from sorted list
            if a > 0:
                break

            # Skip duplicates
            if i > 0 and a == nums[i - 1]:
                continue

            # Two pointers
            l, r = i + 1, len(nums) - 1
            while l < r:
                nums_sum = a + nums[l] + nums[r]
                if nums_sum > 0:
                    r -= 1
                elif nums_sum < 0:
                    l += 1
                else:
                    triplets.append([a, nums[l], nums[r]])
                    l += 1
                    # Skip duplicates, don't need to check for r because conditions above ensure we've found a valid triplet
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1
        return triplets

```


```python
nums = [-1,0,1,2,-1,-4]

Solution().threeSum(nums)

```




    [[-1, -1, 2], [-1, 0, 1]]



## Problem 3: Container With Most Water

https://leetcode.com/problems/container-with-most-water/

You are given an integer array heights where `heights[i]` represents the height of the `i` bar.

You may choose any two bars to form a container. Return the maximum amount of water a container can store.

Example 1:

```
Input: height = [1,7,2,5,4,7,3,6]
Output: 36
```

Example 2:

```
Input: height = [2,2,2]
Output: 4
```

Constraints:

```
2 <= height.length <= 1000
0 <= height[i] <= 1000
```


```python
class Solution:
    def maxArea(self, heights: List[int]) -> int:
        max_area = 0
        l, r = 0, len(heights) - 1

        while l < r:
            area = (r - l) * min(heights[l], heights[r])
            max_area = max(max_area, area)
            if heights[l] > heights[r]:
                r -= 1
            else:
                l += 1
        return max_area
```


```python
height = [1,7,2,5,4,7,3,6]

Solution().maxArea(height)
```




    36



# Two Pointers: Hard Problems

Problem 1: Trapping Rain Water

https://leetcode.com/problems/trapping-rain-water/


You are given an array non-negative integers height which represent an elevation map. Each value `height[i]` represents the height of a bar, which has a width of `1`.

Return the maximum area of water that can be trapped between the bars.

Example 1:

```
Input: height = [0,2,0,3,1,0,1,3,2,1]
Output: 9
```

Constraints:

```
1 <= height.length <= 1000
0 <= height[i] <= 1000
```


```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0

        # Two pointers
        l, r = 0, len(height) - 1
        # Max heights
        left_max, right_max = height[l], height[r]
        res = 0

        # Move pointers
        while l < r:
            # If left max is less than right max, move left pointer
            if left_max < right_max:
                l += 1
                left_max = max(left_max, height[l])
                # Do not need to check negative because we know left_max >= height[l]
                res += left_max - height[l]
            # If right max is less than left max, move right pointer
            else:
                r -= 1
                right_max = max(right_max, height[r])
                # Do not need to check negative because we know right_max >= height[r]
                res += right_max - height[r]
        return res
```


```python
height = [0,2,0,3,1,0,1,3,2,1]

Solution().trap(height)
```




    9
