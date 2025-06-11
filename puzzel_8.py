import random

def create_board():
    nums = list(range(1 , 9)) + [" "]
    random.shuffle(nums)
    return [nums[i:i+3] for i in range(0, 9, 3)]

def print_board(board):
    print("\n")
    for row in board:
        print(" | ". join(str(x) for x in row))
        print("---+---+---")

def is_solved(board):
    excepted = [1, 2, 3, 4, 5, 6, 7, 8, " "]
    flat_board = sum(board, [])
    return flat_board == excepted

def find_blank(board):
    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                return i, j

def move_title(board, direction):
    i, j = find_blank(board)

    if direction == "W" and i > 0:
        board[i][j], board[i - 1][j] = board[i - 1][j], board[i][j]
    elif direction == "s" and i < 2:
        board[i][j], board[i + 1][j] = board[i + 1][j], board[i][j]
    elif direction == "a" and j > 0:
        board[i][j], board[i ][j - 1] = board[i][j - 1], board[i][j]
    elif direction == "d" and j < 2:
        board[i][j], board[i][j + 1] = board[i][j + 1], board[i][j]
    else:
        print("Invalid move")

def game():
    board = create_board()

    while not is_solved(board):
        print_board(board)
        print("Moves are W(up), S(down), a(left), d(right)")
        move = input("Enter the move: ").lower()
        move_title(board, move)
    print_board(board)
    print("solved")

game()