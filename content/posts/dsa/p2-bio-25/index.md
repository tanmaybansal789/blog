+++
date = '2026-03-15T01:41:09+05:30'
draft = false
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

Therefore, we have 4 subproblems:
1. Game setup
2. Finding havens
3. Finding the optimal move
4. Scoring

Game setup is a simple iterative algorithm - keep track of the the turn, how many empty cells have been visited, and the target number, updated the grid as necessary.

For finding havens, a depth-first search from every cell that is unvisited allows us to iterate throguh every haven (a `set` containing coordinates), checking if they contain both `'R'` and `'G'` cells - *unsafe*.

Then, to find the optimal move, we apply the rules in order of priority. We can use Python's rules for comparing `tuple`s to simplify this - it compares each pair of elements until one is found that is different from the other. To ensure that we *minimise* the number of cells they own, we can negate it in the stats tuple.

To find the best cell to make a move from, we look at all cells in the optimal haven, so that it has the minimum position value and it has a neighbour owned by the opponent.

To simulate play, we move our square in the grid to the neighbouring square, switch turns, and repeat this, until no moves are possible (in the code, this is when `play_moves()` returns `False`).

We then count up the number of havens belonging to each player - we know that all havens are now *safe*, otherwise play would continue.

```python
n, r, g = map(int, input().split())
grid = ['.'] * (n * n)

def setup_grid():
    grid[0] = 'R'

    is_red_turn = False
    position = 0
    for _ in range(n * n - 1):
        empty_count = 0 
        target = r if is_red_turn else g
        while empty_count < target:
            position = (position + 1) % (n * n)
            if grid[position] == '.':
                empty_count += 1
        
        grid[position] = 'R' if is_red_turn else 'G'
        is_red_turn = not is_red_turn

def get_unsafe_havens():
    square_types = set()
    visited = set()
    haven = set()

    def dfs(i, j):
        square_types.add(grid[i * n + j])
        visited.add((i, j))
        haven.add((i, j))

        di, dj = 1, 0
        for _ in range(4):

            ni, nj = i + di, j + dj
            if (0 <= ni < n and 0 <= nj < n) \
                and (ni, nj) not in visited \
                and grid[ni * n + nj] != '.': 
                dfs(ni, nj)

            di, dj = -dj, di

    for i in range(n):  
        for j in range(n):
            if (i, j) not in visited and grid[i * n + j] != '.':
                haven.clear()
                square_types.clear()

                dfs(i, j)

                if 'R' in square_types and 'G' in square_types:
                    yield haven


def find_best_haven(us, them):
    # find the optimal haven to attack
    best_haven, best_stats = None, None
    for haven in get_unsafe_havens():
        them_count = sum(1 for i, j in haven if grid[i * n + j] == them)
        us_count = len(haven) - them_count
        max_position = max(i * n + j for i, j in haven)

        # minimise their count, then maximmise ours, then maximise the maximum square position
        stats = (-them_count, us_count, max_position)
        if best_haven is None or stats > best_stats:
            best_haven, best_stats = haven, stats

    return best_haven

def find_best_move(haven, us, them):
    best_square, best_neighbour = None, None
    for i, j in haven:
        if grid[i * n + j] != us:
            continue

        neighbour = min(((ni, nj)
                        for ni, nj in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
                        if 0 <= ni < n and 0 <= nj < n and grid[ni * n + nj] == them),
                        default=None)

        if neighbour and (not best_square or (i, j) < best_square):
            best_square, best_neighbour = (i, j), neighbour

    return best_square, best_neighbour

def play_move(is_red_turn):
    us, them = ('R', 'G') if is_red_turn else ('G', 'R')
    best_haven = find_best_haven(us, them)
    if not best_haven:
        return False
    
    best_square, best_neighbour = find_best_move(best_haven, us, them)
    if not best_square:
        return False

    si, sj = best_square
    ni, nj = best_neighbour
    grid[si * n + sj] = '.'
    grid[ni * n + nj] = us

    return True

def score_game():
    red_score, green_score = 0, 0
    
    visited = set()
    def dfs(i, j):
        visited.add((i, j)) 
        di, dj = 1, 0
        for _ in range(4):
            ni, nj = i + di, j + dj
            if (0 <= ni < n and 0 <= nj < n) \
                and (ni, nj) not in visited \
                and grid[ni * n + nj] != '.': 
                dfs(ni, nj)
            di, dj = -dj, di 

    for i in range(n):
        for j in range(n):
            if grid[i * n + j] != '.' and (i, j) not in visited:
                dfs(i, j)
                if grid[i * n + j] == 'R': 
                    red_score += 1
                else:       
                    green_score += 1

    return red_score, green_score


# main
setup_grid()

is_red_turn = True
while play_move(is_red_turn):
    is_red_turn = not is_red_turn

print(*score_game())
```

### Part B
> Determine the set-up grid layout, assuming an input of `3 123456789 987654321`,
We know:
As n = 3, we can see that our grid has 3 * 3 = 9 cells. If we have `k` remaining cells, we get back to our starting point after `k` jumps. We can modify the setup function as shown, then execute the program with the given inputs, and add an extra debug print for the found grid.
```python
def setup_grid():
    grid[0] = 'R'

    is_red_turn = False
    position = 0
    for k in reversed(range(1, n * n)):
        empty_count = 0 
        target = r if is_red_turn else g
        target = (target - 1) % k + 1
        while empty_count < target:
            position = (position + 1) % (n * n)
            if grid[position] == '.':
                empty_count += 1
        
        grid[position] = 'R' if is_red_turn else 'G'
        is_red_turn = not is_red_turn
```

We get: `['R', 'G', 'R', 'R', 'G', 'G', 'G', 'R', 'R']`.
Formatting this into the required grid:
```
RGR
RGG
GRR
```

### Part C
> Find modifiers for Red and Green, both less than 50, which will set up a 4⨉4 grid such that there are no neighbouring squares controlled by the same player.
This means that we will have to get a checkerboard pattern, like:
```
RGRG
GRGR
RGRG
GRGR
```

We can brute-force this with the following code:
```python
tgt = list("RGRGGRGRRGRGGRGR")`
for r1 in range(1, 50):
    for g1 in range(1, 50):
        r, g = r1, g1
        grid = ['.'] * (n * n)
        setup_grid()
        if grid == tgt:
            print(r, g)
            break
```
We get the required answer of r = 25, g = 41.

### Part D
> The strategy is updated to include the following step at the start of the turn:
> If there are moves which make at least one safe haven for the current player, the player will play the one with the lowest value position for their controlled square, which takes control of the lowest value neighbouring square controlled by their opponent. If there is no such move, the existing strategy is applied.
> How many safe havens does each player control, using this strategy, with the input 10 810 2025?
A move that creates a safe haven means that there is an unsafe haven which only has **1 opponent square** - if we find such a haven, then we'll look at the cells *adjacent to the opponent square*, and choose the smallest one owned by us, and move it to the opponent square. This check runs before the existing strategy is applied, so we add it the beginning of `find_best_move()`.
