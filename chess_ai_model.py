import pygame
import os
import torch
import torch.nn as nn

# Define the model architecture with matching layer names
class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)  # Match checkpoint dimensions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # The output shape will now be [1, 64]


# Instantiate the model
model = ChessModel()

# Load the state dictionary into the model
MODEL_PATH = "chess_model_trained.pth"
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()  # Set the model to evaluation mode

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8

# Colors
WHITE = (222, 184, 135)
BLACK = (139, 69, 19)
SELECTED_COLOR = (255, 255, 0)
VALID_MOVE_COLOR = (0, 0, 255)

# Piece Images (Ensure these images are in the 'assets' folder)
PIECE_IMAGES = {
    'r': pygame.image.load(os.path.join('assets', 'bR.png')),
    'n': pygame.image.load(os.path.join('assets', 'bN.png')),
    'b': pygame.image.load(os.path.join('assets', 'bB.png')),
    'q': pygame.image.load(os.path.join('assets', 'bQ.png')),
    'k': pygame.image.load(os.path.join('assets', 'bK.png')),
    'p': pygame.image.load(os.path.join('assets', 'bP.png')),
    'R': pygame.image.load(os.path.join('assets', 'wR.png')),
    'N': pygame.image.load(os.path.join('assets', 'wN.png')),
    'B': pygame.image.load(os.path.join('assets', 'wB.png')),
    'Q': pygame.image.load(os.path.join('assets', 'wQ.png')),
    'K': pygame.image.load(os.path.join('assets', 'wK.png')),
    'P': pygame.image.load(os.path.join('assets', 'wP.png')),
}

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Two-Player Chess Game")

# Function to draw the chessboard
def draw_board(selected_square=None, valid_moves=None):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            # Highlight the selected square
            if selected_square and selected_square == (row, col):
                pygame.draw.rect(screen, SELECTED_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

            # Highlight valid moves
            if valid_moves and (row, col) in valid_moves:
                pygame.draw.rect(screen, VALID_MOVE_COLOR, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)

# Function to draw pieces
def draw_pieces(board):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece != ".":  # If there's a piece
                piece_image = PIECE_IMAGES.get(piece)
                if piece_image:
                    # Scale the image to fit the square size
                    piece_image = pygame.transform.scale(piece_image, (SQUARE_SIZE, SQUARE_SIZE))
                    screen.blit(piece_image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

# Initialize the chess board
def initialize_board():
    return [
        ["r", "n", "b", "q", "k", "b", "n", "r"],
        ["p", "p", "p", "p", "p", "p", "p", "p"],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", "."],
        ["P", "P", "P", "P", "P", "P", "P", "P"],
        ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ]

# Function to convert board to a tensor
def board_to_tensor(board):
    piece_map = {
        '.': 0, 'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,
        'p': -1, 'r': -2, 'n': -3, 'b': -4, 'q': -5, 'k': -6
    }
    board_tensor = torch.tensor([piece_map[piece] for row in board for piece in row], dtype=torch.float32)
    return board_tensor.view(1, 64)  # Adjust to (1, 64) if the model expects this shape

# Function to get valid moves for a piece
def get_valid_moves(board, row, col):
    piece = board[row][col]
    valid_moves = []

    # Pawn movement logic
    if piece.lower() == 'p':
        direction = -1 if piece.isupper() else 1
        if 0 <= row + direction < 8 and board[row + direction][col] == ".":
            valid_moves.append((row + direction, col))
        for d in [-1, 1]:
            if 0 <= col + d < 8 and 0 <= row + direction < 8 and board[row + direction][col + d] != ".":
                if board[row + direction][col + d].isupper() != piece.isupper():
                    valid_moves.append((row + direction, col + d))

    # Rook and Queen movement logic
    elif piece.lower() == 'r' or piece.lower() == 'q':
        for d in [-1, 1]:
            for i in range(1, 8):
                new_row = row + d * i
                if 0 <= new_row < 8:
                    if board[new_row][col] == ".":
                        valid_moves.append((new_row, col))
                    else:
                        if board[new_row][col].isupper() != piece.isupper():
                            valid_moves.append((new_row, col))
                        break
                else:
                    break
            for i in range(1, 8):
                new_col = col + d * i
                if 0 <= new_col < 8:
                    if board[row][new_col] == ".":
                        valid_moves.append((row, new_col))
                    else:
                        if board[row][new_col].isupper() != piece.isupper():
                            valid_moves.append((row, new_col))
                        break
                else:
                    break

    # Bishop movement logic
    if piece.lower() == 'b' or piece.lower() == 'q':
        for d_row, d_col in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            for i in range(1, 8):
                new_row = row + d_row * i
                new_col = col + d_col * i
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    if board[new_row][new_col] == ".":
                        valid_moves.append((new_row, new_col))
                    else:
                        if board[new_row][new_col].isupper() != piece.isupper():
                            valid_moves.append((new_row, new_col))
                        break
                else:
                    break

    # Knight movement logic
    elif piece.lower() == 'n':
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for move in knight_moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if board[new_row][new_col] == "." or board[new_row][new_col].isupper() != piece.isupper():
                    valid_moves.append((new_row,new_col))

    # King movement logic
    elif piece.lower() == 'k':
        king_moves = [
            (1, 0), (1, 1), (1,-1), (-1 ,0),
            (-1 ,1), (-1 ,-1), (0 ,1), (0 ,-1)
        ]
        for move in king_moves:
            new_row = row + move[0]
            new_col = col + move[1]
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if board[new_row][new_col] == "." or board[new_row][new_col].isupper() != piece.isupper():
                    valid_moves.append((new_row,new_col))

    return valid_moves

# Function to calculate the AI's move
def get_black_move(board):
    # Convert board to tensor and get the model's move prediction
    board_tensor = board_to_tensor(board)
    with torch.no_grad():
        move = model(board_tensor)  # Model's output tensor

    # Assuming the output corresponds to (start_row, start_col) -> (end_row, end_col)
    move = move.view(-1).tolist()  # Flatten the tensor into a list
    
    # Instead of simply selecting the highest probability, evaluate all possible moves
    # and select the most strategic one, for example by choosing a move that captures a piece.
    
    # Example: Selecting the best move based on capturing the opponent's pieces
    best_move = None
    best_score = float('-inf')

    for i in range(64):
        start_row, start_col = divmod(i, 8)
        valid_moves = get_valid_moves(board, start_row, start_col)
        
        for end_row, end_col in valid_moves:
            score = evaluate_move(board, start_row, start_col, end_row, end_col)
            if score > best_score:
                best_score = score
                best_move = (start_row, start_col, end_row, end_col)
    
    return best_move

def evaluate_move(board, start_row, start_col, end_row, end_col):
    # Implement evaluation criteria, for example:
    # - Reward moves that capture opponent's pieces
    # - Penalize moves that place the king in danger
    piece = board[start_row][start_col]
    target_piece = board[end_row][end_col]
    
    score = 0
    if target_piece != '.':
        # Reward capturing an opponent's piece
        if (piece.islower() and target_piece.isupper()) or (piece.isupper() and target_piece.islower()):
            score += 10  # Arbitrary value for capturing an opponent's piece
    
    # Add other evaluation heuristics as needed
    
    return score

# Main game loop
def main():
    board = initialize_board()
    clock = pygame.time.Clock()
    selected_square = None
    valid_moves = []
    player_turn = True  # Keep track of whose turn it is
    running = True
    
    while running:
        draw_board(selected_square, valid_moves)
        draw_pieces(board)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE
                if player_turn:  # Player's turn
                    if selected_square:
                        if (row, col) in valid_moves:
                            # Player makes a move
                            board[row][col] = board[selected_square[0]][selected_square[1]]
                            board[selected_square[0]][selected_square[1]] = "."
                            selected_square = None
                            valid_moves = []

                            # Debugging: print the board after the player's move
                            print("Board after player move:")
                            for r in board:
                                print(r)

                            # Switch turn to AI
                            player_turn = False
                        else:
                            selected_square = None
                            valid_moves = []
                    else:
                        selected_square = (row, col)
                        valid_moves = get_valid_moves(board, row, col)
                else:  # AI's turn
                    # AI makes a move
                    black_move = get_black_move(board)
                    start_row, start_col, end_row, end_col = black_move
                    board[end_row][end_col] = board[start_row][start_col]
                    board[start_row][start_col] = "."

                    # Debugging: print the board after the AI move
                    print("Board after AI move:")
                    for r in board:
                        print(r)

                    # Switch turn to player
                    player_turn = True

        clock.tick(60)

    pygame.quit()


# Run the game
if __name__ == "__main__":
    main()