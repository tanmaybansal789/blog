+++
date = '2026-03-15T01:41:09+05:30'
draft = true
title = 'DSA: Part 2 - British Informatics Olympiad 2025'
series = ['dsa']
series_order = 2
+++

Below, I'll outline some solutions to some nice problems from the previous year's [British Informatics Olympiad](https://www.olympiad.org.uk/papers/2025/bio/bio25-exam.pdf).

## Question 1: Palindromic Sums


> **Every positive integer** can be represented by a palindromic sum of *at most three* **palindromic numbers** (numbers that remain the same when their digits are reversed).
> To be a valid palindromic sum for an integer, the sum must contain the *smallest possible number* of palindromic numbers. It **can contain duplicates**.
> For example:
> - 12321 is a palindromic number already, so can be formed from just 12321;
> - 9610 is equal to 161 + 9449, which is the only pair of palindromic numbers that sum to 9610.
> - There are triplets that sum to 9610, such as 282 + 1771 + 7557, but those contain too many palindromic numbers;
> - 1031 requires three palindromic numbers such as 2 + 494 + 535 and 4 + 88 + 939.
>
> Write a program that reads in an integer (between 1 and 1,000,000 inclusive) and outputs 1, 2 or 3 palindromic numbers which together form a minimal length palindromic sum for the input.

In the British Informatics Olympiad, Question 1/3 tends to be the simplest - so rather than approaching this with any specially crafted algorithm, it might be easiest to begin with a brute force.
Then, a simple 3-step process is sufficient:
1. Check if `n` is already a palindrome - if so, just return `n`
2. Iterate through all palindromes less than `n`, such that if `a` and `n - a` are both palindromic, it is returned.
3. Iterate through all triplets, performing the same check.

In Python:
```python
def is_palindrome(n):
    return str(n) == str(n)[::-1]

def find_palindromic_sum(n):
    if is_palindrome(n):
        return [n]

    for a in range(1, n):
        if is_palindrome(a) and is_palindrome(n - a):
            return [a, n - a]
        
    for a in range(1, n):
        for b in range(a, (n - a) // 2 + 1):
            if is_palindrome(a) and is_palindrome(b) and is_palindrome(n - a - b):
                return [a, b, n - a - b]

n = int(input())
print(*find_palindromic_sum(n))
```
This algorithm is sufficient for part a), given the constraint that `n < 10**6`, and large `n` tend to have 2-palindrome solutions.
When iterating through all triples, we can deduce a maximum for `b` so that the final number is larger than b:
```
b <= n - a - b
2b <= n - a
b <= 1/2(n - a)
```

### Part B
Part B asks for the palindromic sums that represent 54.
By modifying the code to look like:
```python
def is_palindrome(n):
    return str(n) == str(n)[::-1]

def find_palindromic_sum(n):
    if is_palindrome(n):
        print([n])

    for a in range(1, n):
        if is_palindrome(a) and is_palindrome(n - a):
            print([a, n - a])

    for a in range(1, n):
        for b in range(a, (n - a) // 2 + 1):
            if is_palindrome(a) and is_palindrome(b) and is_palindrome(n - a - b):
                print([a, b, n - a - b])

print(find_palindromic_sum(54))
```
We get the following sums:
```
[1, 9, 44]
[2, 8, 44]
[3, 7, 44]
[4, 6, 44]
[5, 5, 44]
```

### Part C
To find how many integers between 1 and 1,000,000 which require a 3-number palindromic sum, it is equivalent to asking: 
> Find how many integers only require a 2 or 1-length palindromic sum, then subtract that from 1,000,000.
The efficient strategy to solve this is:
- **Precompute all palindromes** from 1 to 1,000,000
- Initialise an empty set
- Add all palindromes, and numbers less/equal to the limit reachable by the sum of 2 palindromes
- Output 1,000,000 - the length of this set.

In Python:
```python
def is_palindrome(n):
    return str(n) == str(n)[::-1]

upper_limit = 10 ** 6

numbers_not_requiring_3 = set()
palindromes = list(filter(is_palindrome, range(1, upper_limit + 1)))

for p in palindromes:
    numbers_not_requiring_3.add(p)

for i in range(len(palindromes)):
    for j in range(i, len(palindromes)):
        a, b = palindromes[i], palindromes[j]
        if a + b <= upper_limit:
            numbers_not_requiring_3.add(a + b)

print(upper_limit - len(numbers_not_requiring_3))
```

## Question 2: Safe Haven
> In this game, there is a grid of size n by n, where cells positions increase row by row (i.e. 1-5 for first row, 6-10 for second row, etc.)
> Red controls square 1. Players alternate, starting with Green. On each turn, a player visits successive positions (wrapping from n^2 to 1) starting after the last controlled square. They count empty squares visited; when this reaches their modifier (r for Red, g for Green), they take control of that square and their turn ends.
>
> A move transfers a player's control from one of their squares to an adjacent opponent's square. Players alternate turns starting with Red.
>
> 1. Select the non-safe haven with the smallest number of opponent squares. Ties broken by: largest number of own squares; then highest position value.
> 2. Select the lowest position in this haven that the player controls and neighbours an opponent square.
> 3. Transfer to the lowest position neighbouring opponent square.
>
> Play continues until no valid moves remain. Each player counts their safe havens.
>
> A safe haven is a connected group of single-colour squares with no adjacent opponent squares. A haven is non-safe if it contains both colours or neighbours the opponent.

This is a characteristic British Informatics Question 2, not particularly intricate in terms of problem solving and optimisation, but a significant implementation challenge, considering data representation, edge case handling, and correct logic.
For this problem, it is first necessary to decide on how to represent the grid.
One simple technique would just to use a n^2-length, flat array. The benefits are that we can directly index by position (though, starting at 0), but when it comes to indexing by 2D coordinates, `row * 5 + column` is a simple formula that can be used to find the flat index of any given safe haven.
Each element of the array can therefore be a string, one of `'.'`, `'R'`, and `'G'`, to signify a cell being empty, red or green respectively.
In Python:
