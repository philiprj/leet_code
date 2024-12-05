```python
from typing import List
```

# Stack: Easy Problems

## Problem  1: Valid Parentheses

https://leetcode.com/problems/valid-parentheses/

You are given a string s consisting of the following characters: '(', ')', '{', '}', '[' and ']'.

The input string s is valid if and only if:

Every open bracket is closed by the same type of close bracket.
Open brackets are closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
Return true if s is a valid string, and false otherwise.

Example 1:

```
Input: s = "[]"
Output: true
```

Example 2:

```
Input: s = "([{}])"
Output: true
```

Example 3:

```
Input: s = "[(])"
Output: false
```

Explanation: The brackets are not closed in the correct order.

Constraints:

```
1 <= s.length <= 1000
```



```python
def isValid(s: str) -> bool:
    close_open = { ")" : "(", "]" : "[", "}" : "{" }
    stack = []

    for c in s:
        # Check if c is closing bracket
        if c in close_open:
            if stack and stack[-1] == close_open[c]:
                stack.pop()
            else:
                return False
        else:
            stack.append(c)
    return not stack

```


```python
s = "[]"
isValid(s) == True


s = "([{}])"
isValid(s) == True

s = "[(])"
isValid(s) == False

```




    True



# Stack: Medium Problems

## Problem 2: Min Stack

https://leetcode.com/problems/min-stack/


Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

- MinStack() initializes the stack object.
- void push(int val) pushes the element val onto the stack.
- void pop() removes the element on the top of the stack.
- int top() gets the top element of the stack.
- int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.

Example 1:

```
Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]
```

Explanation
- MinStack minStack = new MinStack();
- minStack.push(-2);
- minStack.push(0);
- minStack.push(-3);
- minStack.getMin(); // return -3
- minStack.pop();
- minStack.top();    // return 0
- minStack.getMin(); // return -2


Constraints:
```
-231 <= val <= 231 - 1
```
Methods pop, top and getMin operations will always be called on non-empty stacks.
At most 3 * 104 calls will be made to push, pop, top, and getMin.



```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(val)

    def pop(self) -> None:
        self.min_stack.pop()
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```


```python
# Your MinStack object will be instantiated and called as such:
obj = MinStack()
obj.push(-2)
obj.push(0)
obj.push(-3)
print(obj.getMin())
obj.pop()
print(obj.top())
print(obj.getMin())
```

    -3
    0
    -2


## Problem 3: Evaluate Reverse Polish Notation

https://leetcode.com/problems/evaluate-reverse-polish-notation/

You are given an array of strings tokens that represents a valid arithmetic expression in Reverse Polish Notation.

Return the integer that represents the evaluation of the expression.

The operands may be integers or the results of other operations.
The operators include `'+', '-', '*', and '/'`.
Assume that division between integers always truncates toward zero.
Example 1:

```
Input: tokens = ["1","2","+","3","*","4","-"]

Output: 5
```

Explanation: `((1 + 2) * 3) - 4 = 5`

Constraints:
`1 <= tokens.length <= 1000.`
`tokens[i]` is `"+", "-", "*", or "/"`, or a string representing an integer in the range `[-100, 100]`.




```python
def evalRPN(tokens: List[str]) -> int:
    """We know this is valid, so we don't need to check for validity."""
    res = 0
    stack = []
    for token in tokens:
        if token == "+":
                stack.append(stack.pop() + stack.pop())
        elif token == "-":
            x, y = stack.pop(), stack.pop()
            stack.append(y - x)
        elif token == "*":
            stack.append(stack.pop() * stack.pop())
        elif token == "/":
            x, y = stack.pop(), stack.pop()
            stack.append(int(y / x))
        else:
            stack.append(int(token))
    return stack[0]
```


```python
# tokens = ["1","2","+","3","*","4","-"]
# tokens = ["2","1","+","3","*"]
# tokens = ["4","13","5","/","+"]
tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]

evalRPN(tokens) == 22
```




    True



## Problem 4: Generate Parentheses

https://leetcode.com/problems/generate-parentheses/

Note this is a backtracking problem. Not related to the stack.

You are given an integer n. Return all well-formed parentheses strings that you can generate with n pairs of parentheses.

Example 1:

```
Input: n = 1
Output: ["()"]
```

Example 2:

```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

You may return the answer in any order.

Constraints:
```
1 <= n <= 7
```


```python
def generateParenthesis(n: int) -> List[str]:
    # Only add open parenthesis if open < n
    # Only add a closing parenthesis if closed < open
    # Valid if open == closed == n

    res = []
    stack = []

    def backtrack(open_n, closed_n):
        if open_n == closed_n == n:
                res.append("".join(stack))
                return

        if open_n < n:
            stack.append("(")
            backtrack(open_n + 1, closed_n)
            stack.pop()

        if closed_n < open_n:
            stack.append(")")
            backtrack(open_n, closed_n + 1)
            stack.pop()

    backtrack(0, 0)

    return res

def generateParenthesis2(n):
    """Dynamic programming solution. TODO: Understand."""
    res = [[] for _ in range(n+1)]
    res[0] = [""]

    for k in range(n + 1):
        for i in range(k):
            for left in res[i]:
                for right in res[k-i-1]:
                    res[k].append("(" + left + ")" + right)

    return res[-1]

```


```python
generateParenthesis(3)
```




    ['((()))', '(()())', '(())()', '()(())', '()()()']



## Problem 5: Daily Temperatures

https://leetcode.com/problems/daily-temperatures/

You are given an array of integers temperatures where temperatures `[i]` represents the daily temperatures on the ith day.

Return an array result where result `[i]` is the number of days after the ith day before a warmer temperature appears on a future day. If there is no day in the future where a warmer temperature will appear for the ith day, set result `[i]` to 0 instead.

Example 1:

```
Input: temperatures = [30,38,30,36,35,40,28]

Output: [1,4,1,2,1,0,0]
```

Example 2:

```
Input: temperatures = [22,21,20]

Output: [0,0,0]
```

Constraints:

```
1 <= temperatures.length <= 1000.
1 <= temperatures[i] <= 100
```


```python
def dailyTemperatures(temperatures: List[int]) -> List[int]:
    # # brute force: O(n^2), space: O(1)
    # res = [0] * len(temperatures)
    # for i, temp in enumerate(temperatures):
    #     for j in range(i+1, len(temperatures)):
    #         if temperatures[j] > temp:
    #             res[i] = j - i
    #             break
    # return res

    # stack: O(n), space: O(n)
    res = [0] * len(temperatures)
    stack = []
    for i, temp in enumerate(temperatures):
        while stack and temp > temperatures[stack[-1]]:
            stack_index = stack.pop()
            res[stack_index] = i - stack_index

        stack.append(i)

    return res

def dailyTemperatures_dp(temperatures: List[int]) -> List[int]:
    """Dynamic programming solution. O(n) time, O(1) space."""
    n = len(temperatures)
    res = [0] * n

    # We iterate from the second last day to the first day (n-2 to 0).
    for i in range(n - 2, -1, -1):
        # For i, j represents the next day.
        j = i + 1
        while j < n and temperatures[j] <= temperatures[i]:
            # If res[j] is 0, it means there is no future day where the temperature is higher than temperatures[j].
            if res[j] == 0:
                # If there is no future day where the temperature is higher than temperatures[j],
                j = n
                break
            # Otherwise, we move to the next day where the temperature is higher than temperatures[j].
            j += res[j]

        # If j < n, it means we found a future day where the temperature is higher than temperatures[i]. So update the result for i.
        if j < n:
            res[i] = j - i
    return res

```


```python
temperatures = [30,38,30,36,35,40,28]
dailyTemperatures(temperatures) == [1,4,1,2,1,0,0]

temperatures = [30,60,90]
dailyTemperatures(temperatures) == [1,1,0]

temperatures = [73,74,75,71,69,72,76,73]
dailyTemperatures(temperatures) == [1,1,4,2,1,1,0,0]
```




    True




```python
temperatures = [30,38,30,36,35,40,28]
dailyTemperatures_dp(temperatures) == [1,4,1,2,1,0,0]

temperatures = [30,60,90]
dailyTemperatures_dp(temperatures) == [1,1,0]

temperatures = [73,74,75,71,69,72,76,73]
dailyTemperatures_dp(temperatures) == [1,1,4,2,1,1,0,0]
```




    True



## Problem 6: Car Fleet

https://leetcode.com/problems/car-fleet/

There are n cars traveling to the same destination on a one-lane highway.

You are given two arrays of integers position and speed, both of length n.

`position[i]` is the position of the ith car (in miles)
`speed[i]` is the speed of the ith car (in miles per hour)
The destination is at position `target` miles.

A car can not pass another car ahead of it. It can only catch up to another car and then drive at the same speed as the car ahead of it.

A car fleet is a non-empty set of cars driving at the same position and same speed. A single car is also considered a car fleet.

If a car catches up to a car fleet the moment the fleet reaches the destination, then the car is considered to be part of the fleet.

Return the number of different car fleets that will arrive at the destination.

Example 1:

```
Input: target = 10, position = [1,4], speed = [3,2]
Output: 1
```

Explanation: The cars starting at 1 (speed 3) and 4 (speed 2) become a fleet, meeting each other at 10, the destination.

Example 2:

```
Input: target = 10, position = [4,1,0,7], speed = [2,2,1,1]
Output: 3
```

Explanation: The cars starting at 4 and 7 become a fleet at position 10. The cars starting at 1 and 0 never catch up to the car ahead of them. Thus, there are 3 car fleets that will arrive at the destination.

Constraints:

```
n == position.length == speed.length.
1 <= n <= 1000
0 < target <= 1000
0 < speed[i] <= 100
0 <= position[i] < target
```
All the values of position are unique.



```python
def carFleet(target: int, position: List[int], speed: List[int]) -> int:
    """O(n log n) time, O(n) space."""
    # Sort the cars by position, reverse order, O(log(n))
    pair = [(p, s) for p, s in zip(position, speed)]
    pair.sort(reverse=True)

    # Iterate through the cars, O(n)
    stack = []
    for p, s in pair:

        # Get the time taken to reach the target, add to stack
        t = (target - p) / s
        # We add and pop current element to ensure we keep the slower moving car on the stack
        stack.append((target - p) / s)

        # If the time taken is less than the top
        # Use if rather than while because we only need to compare to car ahead which is limiting factor for speed and has already been checked
        if len(stack) >= 2 and stack[-1] <= stack[-2]:
            stack.pop()

    return len(stack)

```


```python
target = 10
position = [1,4]
speed = [3,2]
carFleet(target, position, speed)
```




    1




```python
target = 10
position = [4,1,0,7]
speed = [2,2,1,1]

carFleet(target, position, speed)

```




    3




```python
target = 12
position = [10,8,0,5,3, 3]
speed = [2,4,1,1,3, 4]
carFleet(target, position, speed)

```




    3



# Stack: Hard Problems

## Problem 1: Largest Rectangle In Histogram


https://leetcode.com/problems/largest-rectangle-in-histogram/

You are given an array of integers heights where `heights[i]` represents the height of a bar. The width of each bar is 1.

Return the area of the largest rectangle that can be formed among the bars.

Note: This chart is known as a histogram.

Example 1:

```
Input: heights = [7,1,7,2,2,4]
Output: 8
```

Example 2:

```
Input: heights = [1,3,7]
Output: 7
```

Constraints:

```
1 <= heights.length <= 1000.
0 <= heights[i] <= 1000
```



```python
def largestRectangleArea(heights: List[int]) -> int:
    """O(n) time, O(n) space."""
    max_area = 0
    stack = []  # (index, height)

    # Dummy height to ensure we calculate the area for bars left on stack at end.
    heights = heights + [0]

    for i, h in enumerate(heights):
        start = i

        # While the current height is less than the height at the top of the stack
        while stack and h < stack[-1][1]:
            # Get the index and height of the top of the stack
            index, height = stack.pop()
            # Calculate the area of the rectangle with the current top of the stack
            max_area = max(max_area, height * (i - index))
            # Update the start of the current rectangle to the index of the popped bar
            start = index
        # Add the current bar to the stack, and start is the index of the current bar or the index of the previous bar on stack
        stack.append((start, h))

    return max_area
```


```python
heights = [7,1,7,2,2,4]
largestRectangleArea(heights) == 8
```




    True




```python
heights = [1,3,7]
largestRectangleArea(heights) == 7
```




    True
