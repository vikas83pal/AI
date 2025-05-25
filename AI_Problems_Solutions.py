# 1. Breadth First Search (BFS)
from collections import deque

def bfs(graph, start):
    visited, queue = set(), deque([start])
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            print(vertex, end=" ")
            visited.add(vertex)
            neighbors = graph.get(vertex, set())
            queue.extend(neighbors - visited)

# Example graph (undirected)
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F'},
    'D': {'B'},
    'E': {'B', 'F'},
    'F': {'C', 'E'}
}

bfs(graph, 'A')


# 2. Depth First Search (DFS)
def dfs(graph, start, visited=None):
    # Initialize the visited set if it's the first call
    if visited is None:
        visited = set()
    
    # Mark the current node as visited
    visited.add(start)

    # Print the current node
    print(start, end=" ")

    # Recursively visit all unvisited neighbors
    for next in graph[start] - visited:
        dfs(graph, next, visited)

# Example graph (undirected)
graph = {
    'A': {'B', 'C'},
    'B': {'A', 'D', 'E'},
    'C': {'A', 'F'},
    'D': {'B'},
    'E': {'B', 'F'},
    'F': {'C', 'E'}
}

# Call DFS starting from node 'A'
print("DFS traversal starting from 'A':")
dfs(graph, 'A')


# 3. Tic-Tac-Toe Game
def tic_tac_toe():
    board = [" " for _ in range(9)]
    def print_board():
        for i in range(3):
            print("|".join(board[i*3:(i+1)*3]))
    def check_win(player):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        return any(all(board[i]==player for i in combo) for combo in wins)
    current = "X"
    for _ in range(9):
        print_board()
        move = int(input(f"Enter position for {current} (0-8): "))
        if board[move] == " ":
            board[move] = current
            if check_win(current):
                print_board()
                print(f"{current} wins!")
                return
            current = "O" if current == "X" else "X"
    print("Draw!")

# 4. 8-Puzzle Problem (Using BFS)
from collections import deque

def puzzle_bfs(start, goal):
    visited = set()
    queue = deque([(start, [])])

    while queue:
        state, path = queue.popleft()

        if state == goal:
            return path + [state]

        visited.add(state)
        i = state.index("0")  # Find the blank tile (represented by '0')

        # Determine legal moves based on the position of '0'
        moves = []
        if i % 3 > 0: moves.append(-1)  # Move left
        if i % 3 < 2: moves.append(1)   # Move right
        if i > 2: moves.append(-3)      # Move up
        if i < 6: moves.append(3)       # Move down

        # Try all legal moves
        for move in moves:
            j = i + move
            lst = list(state)
            lst[i], lst[j] = lst[j], lst[i]  # Swap blank with neighbor
            new_state = "".join(lst)
            if new_state not in visited:
                queue.append((new_state, path + [state]))

    return []  # Return empty if no solution

# Example usage
start = "724506831"  # Example scrambled state
goal = "123456780"   # Goal state

solution_path = puzzle_bfs(start, goal)

# Print the solution path
print("Solution path (steps from start to goal):")
for step in solution_path:
    for i in range(0, 9, 3):
        print(step[i:i+3])
    print("---")


# 5. Water-Jug Problem (4-liter and 3-liter jug to measure 2 liters)
def water_jug_bfs():
    from collections import deque
    visited = set()
    queue = deque([((0,0), [])])
    while queue:
        (a,b), path = queue.popleft()
        if (a,b) in visited: continue
        visited.add((a,b))
        path = path + [(a,b)]
        if a == 2 or b == 2:
            return path
        actions = [(4,b), (a,3), (0,b), (a,0),
                   (a - min(a, 3-b), b + min(a, 3-b)),
                   (a + min(b, 4-a), b - min(b, 4-a))]
        for new_state in actions:
            if new_state not in visited:
                queue.append((new_state, path))
    return []

# 6. Travelling Salesman Problem (Brute Force)
from itertools import permutations

def tsp(graph, start):
    nodes = list(graph.keys())
    nodes.remove(start)
    min_path, min_cost = None, float('inf')

    for perm in permutations(nodes):
        cost = 0
        k = start

        # Calculate path cost from start through all nodes in perm
        for j in perm:
            cost += graph[k][j]
            k = j

        # Add cost to return to start
        cost += graph[k][start]

        # Check if this is the minimum cost path
        if cost < min_cost:
            min_cost = cost
            min_path = (start,) + perm + (start,)

    return min_path, min_cost

# Example graph (fully connected)
graph = {
    'A': {'A': 0, 'B': 10, 'C': 15, 'D': 20},
    'B': {'A': 10, 'B': 0, 'C': 35, 'D': 25},
    'C': {'A': 15, 'B': 35, 'C': 0, 'D': 30},
    'D': {'A': 20, 'B': 25, 'C': 30, 'D': 0}
}

# Starting node
start = 'A'

# Solve TSP
path, cost = tsp(graph, start)

# Print result
print("Minimum cost path:", " -> ".join(path))
print("Minimum total cost:", cost)


# 7. Tower of Hanoi
def tower_of_hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
    else:
        # Move n-1 disks from source to auxiliary
        tower_of_hanoi(n-1, source, auxiliary, target)
        
        # Move the nth disk from source to target
        print(f"Move disk {n} from {source} to {target}")
        
        # Move the n-1 disks from auxiliary to target
        tower_of_hanoi(n-1, auxiliary, target, source)

# Example usage
n = 3  # Number of disks
tower_of_hanoi(n, 'A', 'C', 'B')


# 8. Monkey Banana Problem (Dynamic Programming on 2D grid)
def monkey_banana(grid):
    rows = len(grid)
    
    for i in range(1, rows):
        for j in range(len(grid[i])):
            # Initialize max_banana from above if in bounds
            from_above = grid[i-1][j] if j < len(grid[i-1]) else 0
            from_left_diag = grid[i-1][j-1] if j-1 >= 0 else 0
            
            # Update current cell with best possible path
            grid[i][j] += max(from_above, from_left_diag)
    
    return max(grid[-1])


grid = [
    [7],
    [3, 8],
    [8, 1, 0],
    [2, 7, 4, 4],
    [4, 5, 2, 6, 5]
]

print("Maximum bananas the monkey can collect:", monkey_banana(grid))



# 9. Alpha-Beta Pruning (Minimax)
def alphabeta(node, depth, alpha, beta, maximizingPlayer, values):
    # Base case: if we reach depth 0 (leaf), return its value
    if depth == 0 or node >= len(values):
        return values[node]

    if maximizingPlayer:
        maxEval = float('-inf')
        for child in range(node*2, node*2 + 2):
            if child < len(values):
                eval = alphabeta(child, depth - 1, alpha, beta, False, values)
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
        return maxEval
    else:
        minEval = float('inf')
        for child in range(node*2, node*2 + 2):
            if child < len(values):
                eval = alphabeta(child, depth - 1, alpha, beta, True, values)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
        return minEval

# Example usage
# Binary tree represented as a list of leaf values at depth 3
values = [3, 5, 6, 9, 1, 2, 0, -1]  # These are assumed to be at the leaf level
depth = 3  # Height of the tree
alpha = float('-inf')
beta = float('inf')
start_node = 0

# Call the function and print result
best_value = alphabeta(start_node, depth, alpha, beta, True, values)
print("The optimal value is:", best_value)


# 10. 8-Queens Problem (Backtracking)
def solve_n_queens(n):
    def is_safe(board, row, col):
        # Check column and both diagonals
        for i in range(row):
            if board[i] == col or board[i] - i == col - row or board[i] + i == col + row:
                return False
        return True

    def solve(row, board):
        if row == n:
            result.append(board[:])  # Store a valid solution
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(row + 1, board)
                board[row] = -1  # Backtrack

    result = []
    solve(0, [-1] * n)
    return result

# 🖨️ Function to print board solutions
def print_boards(solutions, n):
    for sol_num, solution in enumerate(solutions, 1):
        print(f"\nSolution {sol_num}:")
        for i in range(n):
            row = ['.'] * n
            row[solution[i]] = 'Q'
            print("".join(row))

# Example usage:
n = 4
solutions = solve_n_queens(n)
print(f"Total solutions for {n}-Queens: {len(solutions)}")
print_boards(solutions, n)
