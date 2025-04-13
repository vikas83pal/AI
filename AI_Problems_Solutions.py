# 1. Breadth First Search (BFS)
from collections import deque

def bfs(graph, start):
    visited, queue = set(), deque([start])
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            print(vertex, end=" ")
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)

# 2. Depth First Search (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")
    for next in graph[start] - visited:
        dfs(graph, next, visited)

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
def puzzle_bfs(start, goal):
    from collections import deque
    visited = set()
    queue = deque([(start, [])])
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path + [state]
        visited.add(state)
        i = state.index("0")
        moves = []
        if i % 3 > 0: moves.append(-1)
        if i % 3 < 2: moves.append(1)
        if i > 2: moves.append(-3)
        if i < 6: moves.append(3)
        for move in moves:
            j = i + move
            lst = list(state)
            lst[i], lst[j] = lst[j], lst[i]
            new_state = "".join(lst)
            if new_state not in visited:
                queue.append((new_state, path + [state]))
    return []

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
def tsp(graph, start):
    from itertools import permutations
    nodes = list(graph.keys())
    nodes.remove(start)
    min_path, min_cost = None, float('inf')
    for perm in permutations(nodes):
        cost = 0
        k = start
        for j in perm:
            cost += graph[k][j]
            k = j
        cost += graph[k][start]
        if cost < min_cost:
            min_cost = cost
            min_path = (start,) + perm + (start,)
    return min_path, min_cost

# 7. Tower of Hanoi
def tower_of_hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
    else:
        tower_of_hanoi(n-1, source, auxiliary, target)
        print(f"Move disk {n} from {source} to {target}")
        tower_of_hanoi(n-1, auxiliary, target, source)

# 8. Monkey Banana Problem (Dynamic Programming on 2D grid)
def monkey_banana(grid):
    rows = len(grid)
    for i in range(1, rows):
        for j in range(len(grid[i])):
            grid[i][j] += max(grid[i-1][j], grid[i-1][j-1] if j > 0 else 0)
    return max(grid[-1])

# 9. Alpha-Beta Pruning (Minimax)
def alphabeta(node, depth, alpha, beta, maximizingPlayer, values):
    if depth == 0:
        return values[node]
    if maximizingPlayer:
        maxEval = float('-inf')
        for child in range(node*2, node*2+2):
            eval = alphabeta(child, depth-1, alpha, beta, False, values)
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float('inf')
        for child in range(node*2, node*2+2):
            eval = alphabeta(child, depth-1, alpha, beta, True, values)
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval

# 10. 8-Queens Problem (Backtracking)
def solve_n_queens(n):
    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col or                board[i] - i == col - row or                board[i] + i == col + row:
                return False
        return True

    def solve(row, board):
        if row == n:
            result.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                solve(row+1, board)
                board[row] = -1

    result = []
    solve(0, [-1]*n)
    return result