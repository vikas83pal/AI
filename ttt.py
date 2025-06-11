board = [" " for _ in range(9)]

def print_board():
    print()
    for i in range(3):
        print(board[i * 3], " |", board[i * 3 + 1], "|", board[i * 3 + 2])
        if(i < 2):
            print("---+---+---")

def check_win(player):
    win_pos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    
    return any(all(board[i] == player for i in line) for line in win_pos)

def game():
    current = "X"
    for turn in range(9):
        print_board()
        move = int(input(f"player {current} : choose the position (0 - 9): "))
        if(board[move] != " "):
            print("Spot taken.. try again")
            continue
        board[move] = current
        if check_win(current):
            print_board()
            print(f"player {current}... wins")
            return
        
        current = "O" if current == "X" else "X"
    print_board()
    print("Draw")

game()