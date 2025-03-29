import pygame
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import chess
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Define the Chess Environment
class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()
        self.game_over = False
    
    def reset(self):
        self.board = chess.Board()
        self.game_over = False
        return self.board.fen()
    
    def step(self, move):
        if move not in self.board.legal_moves:
            raise ValueError("Illegal Move Attempted!")
        
        self.board.push(move)
        reward = 0  # Default reward

        if self.board.is_game_over():
            self.game_over = True
            result = self.board.result()
            reward = 1 if result == "1-0" else -1 if result == "0-1" else 0

        return self.board.fen(), reward, self.game_over

    def get_legal_moves(self):
        return list(self.board.legal_moves)

# Helper function to convert FEN to tensor
def fen_to_tensor(fen):
    piece_map = {'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
                 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
    board_tensor = np.zeros((8, 8), dtype=np.float32)
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                board_tensor[i, col] = piece_map.get(char, 0)
                col += 1
    return torch.tensor(board_tensor.flatten(), dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Define the Chess Model
class ChessModel(nn.Module):
    def __init__(self, output_size=64):
        super(ChessModel, self).__init__()
        self.fc1 = nn.Linear(8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the Chess GUI
class ChessGUI:
    def __init__(self, size=600):
        self.size = size
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Chess Game")
        self.piece_images = self.load_piece_images()

    def load_piece_images(self):
        piece_images = {}
        pieces = ['K', 'Q', 'R', 'B', 'N', 'P']
        colors = ['w', 'b']
        for color in colors:
            for piece in pieces:
                image_path = os.path.join('assets', f'{color}{piece}.png')
                if os.path.exists(image_path):
                    piece_images[f'{color}{piece}'] = pygame.image.load(image_path)
        return piece_images

    def draw_board(self, board):
        square_size = self.size // 8
        colors = [(222, 184, 135), (139, 69, 19)]
        self.screen.fill((0, 0, 0))  # Clear screen
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * square_size, row * square_size, square_size, square_size))
        for square, piece in board.piece_map().items():
            row, col = divmod(square, 8)
            piece_image = self.piece_images.get(f'{"w" if piece.color else "b"}{piece.symbol().upper()}', None)
            if piece_image:
                self.screen.blit(pygame.transform.scale(piece_image, (square_size, square_size)), (col * square_size, row * square_size))
        pygame.display.flip()

# DQN Agent
class DQNAgent:
    def __init__(self, model, optimizer, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def get_move(self, state, legal_moves):
        state_tensor = fen_to_tensor(state)
        q_values = self.model(state_tensor).squeeze(0)  # Remove batch dimension

        # Only select from legal moves
        legal_q_values = [(move, q_values[move.from_square].item()) for move in legal_moves if move.from_square < 64]
        if not legal_q_values:
            return random.choice(legal_moves)  # Fallback for safety

        legal_q_values.sort(key=lambda x: x[1], reverse=True)

        # Epsilon-greedy strategy
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        return legal_q_values[0][0]  # Best move

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training function with model saving
def train_agent(episodes=250):
    pygame.init()
    gui = ChessGUI()
    model = ChessModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    agent = DQNAgent(model, optimizer)
    game_env = ChessEnvironment()
    max_moves = 200

    for episode in range(episodes):
        state = game_env.reset()
        done = False
        move_count = 0

        while not done and move_count < max_moves:
            legal_moves = game_env.get_legal_moves()
            if legal_moves:
                action = agent.get_move(state, legal_moves)
                print(f"Episode {episode + 1}, Move {move_count + 1}: {action}")

                next_state, reward, done = game_env.step(action)
                print(f"Reward: {reward}, Game Over: {done}")

                state = next_state
                move_count += 1

                if move_count % 5 == 0:  # Redraw GUI every 5 moves to reduce overhead
                    gui.draw_board(game_env.board)

            agent.update_epsilon()  # Update exploration factor once per episode

        print(f"Episode {episode + 1} completed with {move_count} moves. Epsilon: {agent.epsilon:.4f}")

    # Save trained model
    torch.save(model.state_dict(), "chess_model_trained.pth")
    print("Model saved successfully as 'chess_model_trained.pth'")

    pygame.quit()

# Load trained model
def load_trained_model(model_path="chess_model_trained.pth"):
    model = ChessModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Trained model loaded successfully!")
    return model

# Run training
train_agent()
