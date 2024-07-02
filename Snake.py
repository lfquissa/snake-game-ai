import Constants
from Directions import Directions
from Point import Point


class Snake:
    def __init__(self):
        # MARK:- Snake Confing
        self.snake_direction = Directions.RIGHT
        self.snake_head = Point(Constants.WIDTH // 2, Constants.WIDTH // 2)
        self.snake = [
            Point(self.snake_head.x - 2 * Constants.BLOCK_SIZE, self.snake_head.y),
            Point(self.snake_head.x - Constants.BLOCK_SIZE, self.snake_head.y),
            Point(self.snake_head.x, self.snake_head.y),
        ]

    def is_food_in_snake(self, food):
        if food in self.snake:
            return True
        return False

    def change_snake_direction(self, action):
        self.snake_direction = action.next_direction(self.snake_direction)

    def update_snake_position(self):
        if self.snake_direction == Directions.LEFT:
            self.snake_head.x -= Constants.BLOCK_SIZE

        elif self.snake_direction == Directions.RIGHT:
            self.snake_head.x += Constants.BLOCK_SIZE

        elif self.snake_direction == Directions.UP:
            self.snake_head.y -= Constants.BLOCK_SIZE

        elif self.snake_direction == Directions.DOWN:
            self.snake_head.y += Constants.BLOCK_SIZE

        self.snake.append(Point(self.snake_head.x, self.snake_head.y))

    def get_head_surrounding(self):
        head = self.snake[-1]
        left = Point(head.x - Constants.BLOCK_SIZE, head.y)
        right = Point(head.x + Constants.BLOCK_SIZE, head.y)
        up = Point(head.x, head.y - Constants.BLOCK_SIZE)
        down = Point(head.x, head.y + Constants.BLOCK_SIZE)
        return [left, up, right, down]
