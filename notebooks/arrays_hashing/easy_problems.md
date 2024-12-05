# Arrays and Hashing: Easy Problems


```python
from typing import List
```

## Problem 1: Contains Duplicate

https://leetcode.com/problems/contains-duplicate/


Contains Duplicate
Given an integer array nums, return true if any value appears more than once in the array, otherwise return false.

Example 1:

```
Input: nums = [1, 2, 3, 3]
Output: true
```

Example 2:

```
Input: nums = [1, 2, 3, 4]
Output: false
```


```python
class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:
        # return len(set(nums)) < len(nums)
        hash_set = set()
        for num in nums:
            if num in hash_set:
                return True
            else:
                hash_set.add(num)
        return False

```

## Problem 2: Valid Anagram

https://leetcode.com/problems/valid-anagram/

Given two strings s and t, return true if t is an anagram of s, and false otherwise.

Example 1:

```
Input: s = "anagram", t = "nagaram"
Output: true
```

Example 2:

```
Input: s = "rat", t = "car"
Output: false
```


Constraints:

```
1 <= s.length, t.length <= 5 * 104
s and t consist of lowercase English letters.
```

Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?


```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        # Brute force: O(n log n)
        # return sorted(s) == sorted(t)

        # Hash map: O(n+m), O(1)
        # hash_map_s, hash_map_t = {}, {}
        # for i in range(len(s)):
        #     hash_map_s[s[i]] = 1 + hash_map_s.get(s[i], 0)
        #     hash_map_t[t[i]] = 1 + hash_map_t.get(t[i], 0)
        # return hash_map_s == hash_map_t

        # Hash Map Optimised: O(n+m), O(1)
        count = [0] * 26    # 26 letters in the alphabet
        for i in range(len(s)):
            # ord() gets the ASCII value of the character
            # Subtracting the ASCII value of 'a' gives a number between 0 and 25
            # This is the index of the letter in the count array
            count[ord(s[i]) - ord('a')] += 1
            count[ord(t[i]) - ord('a')] -= 1

        # If the strings are anagrams, the count array will be all zeros
        for val in count:
            if val != 0:
                return False
        return True
```

## Problem 3: Two Sum

https://leetcode.com/problems/two-sum/

Given an array of integers nums and an integer target, return the indices i and j such that nums[i] + nums[j] == target and i != j.

You may assume that every input has exactly one pair of indices i and j that satisfy the condition.

Return the answer with the smaller index first.

Example 1:

```
Input:
nums = [3,4,5,6], target = 7
Output: [0,1]

```
Explanation: nums[0] + nums[1] == 7, so we return [0, 1].

Example 2:

```
Input: nums = [4,5,6], target = 10
Output: [0,2]
```

Example 3:

```
Input: nums = [5,5], target = 10
Output: [0,1]
```

Constraints:
```
2 <= nums.length <= 1000
-10,000,000 <= nums[i] <= 10,000,000
-10,000,000 <= target <= 10,000,000
```


```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Brute force: O(n^2)
        # for i, num1 in enumerate(nums):
        #     for j, num2 in enumerate(nums[i+1:]):
        #         if num1 + num2 == target:
        #             return [i, j]
        # return []

        # Hash map: O(n), O(n)
        # Key: number, Value: index, we can check if the difference we neeed is in the hash map
        hash_map = {}
        for i, num in enumerate(nums):
            difference = target - num
            if difference in hash_map:
                return [hash_map[difference], i]
            hash_map[num] = i
        return []
```
