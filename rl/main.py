from dataclasses import dataclass
import pygame
import random

width = height = 400
cell_size = 10

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)

alpha = 0.1
gamma = 0.95
epsilon = 0.001
episodes = 1_000_000

snake_learning_len = 10
render_every = 5_000

class Snake:
    def __init__(self, screen):
        self.screen = screen
        self.reset()

    def reset(self):
        self.snake = [(width // 2, height // 2)]
        self.direction = random.choice([(0, -1)])
