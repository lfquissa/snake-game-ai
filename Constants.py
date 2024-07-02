import pygame

from Point import Point

pygame.init()
# Model constants
MAX_MEMORY = 100_000
BATCH_SIZE = 500
LR = 0.0001
GAMMA = 0.95
TRAINING_EPISODES = 650
EPSILON = 250
EPS_UPPER = 280
# Game constants
BLOCK_SIZE = 20 * 2
BACKGROUNDCOLOR = (102, 153, 153)
SNAKECOLOR = (153, 255, 102)
SNAKEHEADCOLOR = (255, 255, 0)  # New color for the snake head
SNAKEOUTLINECOLOR = (0, 0, 0)  # Outline color for the snake body
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
FONT = pygame.font.Font(None, 25)
CAPTION = "Snake Game"
WIDTH = 640 * 2
HEIGHT = 480 * 2
TICK_RATE = 10
SENSOR_DIRECTIONS = [
    Point(-1, 0),  # Left
    Point(1, 0),  # Right
    Point(0, -1),  # Up
    Point(0, 1),  # Down
    Point(-1, -1),  # Top-left
    Point(1, -1),  # Top-right
    Point(-1, 1),  # Bottom-left
    Point(1, 1),  # Bottom-right
]
