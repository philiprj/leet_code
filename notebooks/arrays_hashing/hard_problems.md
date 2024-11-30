# Arrays and Hashing: Hard Problems


```python
from typing import List
from collections import defaultdict, Counter
import heapq

```

## Problem 1: Group Anagrams

https://leetcode.com/problems/group-anagrams/


Given an array of strings strs, group the
anagrams
 together. You can return the answer in any order.



Example 1:

```
Input: strs = ["eat","tea","tan","ate","nat","bat"]

Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

Explanation:

There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
Example 2:

```
Input: strs = [""]

Output: [[""]]
```


Example 3:

```
Input: strs = ["a"]

Output: [["a"]]
```


Constraints:

```
1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters.
```


```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # Complexity: O(n * m) and O(n * m), where n is the number of words and m is the length of the longest word
        # Use defaultdict to avoid key error
        # The key in this case is a tuple of counts of each letter, and value is a list of words that match that key
        anagrams = defaultdict(list)

        for word in strs:

            # Count the number of each letter in the word
            count = [0] * 26

            # ord() gets the ASCII value of the character
            for c in word:
                count[ord(c) - ord('a')] += 1

            # We can do this because defaultdict will create a new list if the key is not present
            anagrams[tuple(count)].append(word)

            # Alternative solution using sorted(), complexity, O(m log n), O(mn)
            # key = "".join(sorted(word))
            # anagrams[key].append(word)

        return list(anagrams.values())
```

## Problem 2: Top K Frequent Elements

https://leetcode.com/problems/top-k-frequent-elements/

Given an integer array nums and an integer `k`, return the `k` most frequent elements within the array.

The test cases are generated such that the answer is always unique.

You may return the output in any order.

Example 1:
```
Input: nums = [1,2,2,3,3,3], k = 2

Output: [2,3]
```

Example 2:

```
Input: nums = [7,7], k = 1

Output: [7]
```

Constraints:

```
1 <= nums.length <= 10^4.
-1000 <= nums[i] <= 1000
1 <= k <= number of distinct elements in nums.
```


```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # counts = Counter(nums)
        # return [num for num, _ in counts.most_common(k)]

        # Alternative solution using a heap, complexity O(n log k), O(n+k)
        # counts = {}
        # for num in nums:
        #     counts[num] = counts.get(num, 0) + 1

        # heap = []
        # for num, count in counts.items():
        #     heapq.heappush(heap, (count, num))
        #     if len(heap) > k:
        #         heapq.heappop(heap)

        # res = []
        # for _ in range(k):
        #     res.append(heapq.heappop(heap)[1])
        # return res

        # Alternative solution using bucket sort, complexity O(n), O(n)
        count = {}
        freq = [[] for i in range(len(nums) + 1)]   # Create a list of empty lists, one for each frequency from 0 to len(nums)

        for num in nums:
            count[num] = count.get(num, 0) + 1
        for num, cnt in count.items():
            freq[cnt].append(num)

        res = []

        # Iterate over the frequency list in reverse order
        for i in range(len(freq) - 1, 0, -1):
            for num in freq[i]:
                res.append(num)
                if len(res) == k:
                    return res

```

## Problem 3: Encode and Decode Strings

https://leetcode.com/problems/encode-and-decode-strings/

Design an algorithm to encode a list of strings to a single string. The encoded string is then decoded back to the original list of strings.

Please implement encode and decode

Example 1:
```
Input: ["neet","code","love","you"]

Output:["neet","code","love","you"]
```

Example 2:

```
Input: ["we","say",":","yes"]

Output: ["we","say",":","yes"]
```

Constraints:

```
0 <= strs.length < 100
0 <= strs[i].length < 200
strs[i] contains only UTF-8 characters.
```


```python
class Solution:

    def encode(self, strs: List[str]) -> str:
        # Start with a number representing the length of the string, followed by a #, then the string
        # The len is required to know where the end of the string is, incase the string contains a #
        return ''.join(f"{len(s)}#{s}" for s in strs)

    def decode(self, s: str) -> List[str]:
        res = []
        i = 0

        # Iterate over the string, and for each string, get the length, then the string
        while i < len(s):
            j = i
            # Find the length of the string (could be multiple digits)
            while s[j] != '#':
                j += 1
            n_chars = int(s[i:j])

            # Get the string (j will be at the #, so start at j+1), and add n_chars to get the end of the string
            i = j + 1
            j = i + n_chars

            # Add the string to the result, and move i to the end of the string
            res.append(s[i:j])
            i = j

        return res
```

## Problem 4: Products of Array Except Self

https://leetcode.com/problems/product-of-array-except-self/

Given an integer array nums, return an array output where `output[i]` is the product of all the elements of nums except `nums[i]`.

Each product is guaranteed to fit in a 32-bit integer.

Follow-up: Could you solve it in `O(n)` time without using the division operation?

Example 1:

```
Input: nums = [1,2,4,6]
Output: [48,24,12,8]
```

Example 2:

```
Input: nums = [-1,0,1,2,3]
Output: [0,-6,0,0,0]
```

Constraints:
```
2 <= nums.length <= 1000
-20 <= nums[i] <= 20
```


```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # # Naive solution, O(n), O(n)
        # total_product, zero_count = 1, 0
        # for num in nums:
        #     if num:
        #         total_product *= num
        #     # If there is a zero, increment the zero count
        #     else:
        #         zero_count += 1

        # # If there is more than one zero, all products will be zero
        # if zero_count > 1:
        #     return [0] * len(nums)

        # res = [0] * len(nums)

        # for i, num in enumerate(nums):
        #     # Handle the case where there is one zero, not at current index
        #     if zero_count and num:
        #         res[i] = 0
        #     # Handle the case where there is one zero, at current index
        #     elif zero_count and not num:
        #         res[i] = total_product
        #     # Handle the case where there is no zero
        #     else:
        #         res[i] = total_product // num

        # return res

        # Prefix and Suffix Optimal solution, O(n), O(1)
        res = [1] * len(nums)

        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]

        suffix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= suffix
            suffix *= nums[i]

        return res

```

Explaination

```
input: [1,2,3,4]
prefix: [1,2,6,24]      # Left to right
suffix: [24,24,12,4]    # Right to left
output: [24,12,8,6]     # Multiply prefix[i-1] and suffix[i+1] (default to 1 if i=0 or i=n-1)
```

Optimally we can directly compute the prefix and suffix as we iterate over the array, and then use the result array to store the output.

We pass through the array twice, so O(n) time and O(1) space.

First pass at each element we store the current prefix, then increment the prefix value by the current element.
Second pass we multiply the current result element by the current suffix, then increment the suffix value by the current element.


```python
# nums = [1,2,4,6]
nums = [-1,0,1,2,3]
```


```python
sol = Solution()
print(sol.productExceptSelf(nums))
```

    [0, -6, 0, 0, 0]


## Problem 5: Valid Sudoku

https://leetcode.com/problems/valid-sudoku/

You are given a a 9 x 9 Sudoku board board. A Sudoku board is valid if the following rules are followed:

- Each row must contain the digits 1-9 without duplicates.
- Each column must contain the digits 1-9 without duplicates.
- Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without duplicates.

Return true if the Sudoku board is valid, otherwise return false

Note: A board does not need to be full or be solvable to be valid.

Example 1

```
Input: board =
[["1","2",".",".","3",".",".",".","."],
 ["4",".",".","5",".",".",".",".","."],
 [".","9","8",".",".",".",".",".","3"],
 ["5",".",".",".","6",".",".",".","4"],
 [".",".",".","8",".","3",".",".","5"],
 ["7",".",".",".","2",".",".",".","6"],
 [".",".",".",".",".",".","2",".","."],
 [".",".",".","4","1","9",".",".","8"],
 [".",".",".",".","8",".",".","7","9"]]

Output: true
```

Example 2:

```
Input: board =
[["1","2",".",".","3",".",".",".","."],
 ["4",".",".","5",".",".",".",".","."],
 [".","9","1",".",".",".",".",".","3"],
 ["5",".",".",".","6",".",".",".","4"],
 [".",".",".","8",".","3",".",".","5"],
 ["7",".",".",".","2",".",".",".","6"],
 [".",".",".",".",".",".","2",".","."],
 [".",".",".","4","1","9",".",".","8"],
 [".",".",".",".","8",".",".","7","9"]]

Output: false
```
Explanation: There are two 1's in the top-left 3x3 sub-box.

Constraints:

board.length == 9
board[i].length == 9
board[i][j] is a digit 1-9 or '.'



```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        for row in range(9):
            # Create a set to store the numbers in the row
            seen = set()
            for col in range(9):
                if board[row][col] != '.' and board[row][col] in seen:
                    return False
                seen.add(board[row][col])

        for col in range(9):
            # Create a set to store the numbers in the row
            seen = set()
            for row in range(9):
                if board[row][col] != '.' and board[row][col] in seen:
                    return False
                seen.add(board[row][col])

        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = set()
                for k in range(3):
                    for l in range(3):
                        if board[i+k][j+l] != '.':
                            if board[i+k][j+l] in box:
                                return False
                            box.add(board[i+k][j+l])

        for square in range(9):
            seen = set()
            for i in range(3):
                for j in range(3):
                    row = (square//3) * 3 + i
                    col = (square % 3) * 3 + j
                    if board[row][col] == ".":
                        continue
                    if board[row][col] in seen:
                        return False
                    seen.add(board[row][col])

        return True

    def isValidSudoku2(self, board: List[List[str]]) -> bool:
        """Find valid sudoku using a hash set to store the numbers in each row, column, and 3x3 square.
        O(n^2), O(n^2). Optimised over the previous solution by using a hash set to store the numbers, but still n^2

        Args:
            board (List[List[str]])

        Returns:
            bool
        """
        cols = defaultdict(set)     # key is col number  (e.g. cols[0] = {1,2,3})
        rows = defaultdict(set)     # key is row number (rows[0] = {1,2,3})
        squares = defaultdict(set)  # key is (r//3, c//3) (squares[(0,0)] = {1,2,3})

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (board[r][c] in rows[r]) or (board[r][c] in cols[c]) or (board[r][c] in squares[(r//3, c//3)]):
                    return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r//3, c//3)].add(board[r][c])
        return True

    def isValidSudoku3(self, board: List[List[str]]) -> bool:
        """Complex bitwise solution. To return to once better understanding of bitwise"""
        rows = [0] * 9
        cols = [0] * 9
        squares = [0] * 9

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue

                val = int(board[r][c]) - 1
                if (1 << val) & rows[r]:
                    return False
                if (1 << val) & cols[c]:
                    return False
                if (1 << val) & squares[(r // 3) * 3 + (c // 3)]:
                    return False

                rows[r] |= (1 << val)
                cols[c] |= (1 << val)
                squares[(r // 3) * 3 + (c // 3)] |= (1 << val)

        return True

```


```python
board = [
    ["1","2",".",".","3",".",".",".","."],
    ["4",".",".","5",".",".",".",".","."],
    [".","9","8",".",".",".",".",".","3"],
    ["5",".",".",".","6",".",".",".","4"],
    [".",".",".","8",".","3",".",".","5"],
    ["7",".",".",".","2",".",".",".","6"],
    [".",".",".",".",".",".","2",".","."],
    [".",".",".","4","1","9",".",".","8"],
    [".",".",".",".","8",".",".","7","9"]
 ]

sol = Solution()
```


```python
print(sol.isValidSudoku2(board))
```

    True


## Problem 6: Longest Consecutive Sequence

https://leetcode.com/problems/longest-consecutive-sequence/

Given an array of integers nums, return the length of the longest consecutive sequence of elements that can be formed.

A consecutive sequence is a sequence of elements in which each element is exactly 1 greater than the previous element. The elements do not have to be consecutive in the original array.

You must write an algorithm that runs in O(n) time.

Example 1:

```
Input: nums = [2,20,4,10,3,4,5]

Output: 4
```

Explanation: The longest consecutive sequence is [2, 3, 4, 5].

Example 2:

```
Input: nums = [0,3,2,5,4,6,1,1]

Output: 7
```

Constraints:
```
0 <= nums.length <= 1000
-10^9 <= nums[i] <= 10^9
```


```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # # Naive solution, using sorting, O(n log n) due to sorting, O(1)
        # if not nums:
        #     return 0

        # # Could impliment this manually...
        # nums.sort()

        # longest = 1
        # current = 1
        # i = 1

        # for i in range(1, len(nums)):
        #     if nums[i] == nums[i-1]:
        #         continue
        #     if nums[i] == nums[i-1] + 1:
        #         current += 1
        #     else:
        #         longest = max(longest, current)
        #         current = 1

        # return max(longest, current)

        # Optimal solution, using a hash set, O(n), O(n)
        # num_set = set(nums)
        # longest = 0
        # current = 0

        # for num in num_set:
        #     # Check if the number is the start of a sequence
        #     if (num - 1) not in num_set:
        #         current = 1
        #         while num + current in num_set:
        #             current += 1
        #         longest = max(longest, current)
        # return longest


        # Hash Map solution, O(n), O(n).
        # The solution is particularly efficient because:
        # 1. Each number is only processed once
        # 2. We don't need to store or update every number in between sequence endpoints
        # 3. We can quickly merge sequences by looking at adjacent numbers
        num_map = defaultdict(int)
        longest = 0

        for num in nums:
            if not num_map[num]:
                left_length = num_map[num - 1]   # Length of sequence to the left
                right_length = num_map[num + 1]  # Length of sequence to the right
                current_length = left_length + right_length + 1  # Current sequence length, including current number

                # Update the length of the sequence for the current number, and the numbers to either side
                num_map[num] = current_length
                num_map[num - left_length] = current_length
                num_map[num + right_length] = current_length
                longest = max(longest, num_map[num])
        return longest


```


```python
nums = [2,20,4,10,3,4,5]
# nums = [0,3,2,5,4,6,1,1]
```


```python
print(Solution().longestConsecutive(nums))
```

    4
