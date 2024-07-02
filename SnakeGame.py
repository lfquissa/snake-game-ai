import random

import pygame

import Constants
from Point import Point
from Snake import Snake


class SnakeGame:

    def __init__(self, width=Constants.WIDTH, height=Constants.HEIGHT, seed=0):
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(Constants.CAPTION)
        self.clock = pygame.time.Clock()
        self.default_tick_rate = Constants.TICK_RATE
        self.tick_rate = self.default_tick_rate
        self.prev_tick_rate = self.default_tick_rate
        self.show_ui = True
        self.paused = False
        self.start_new_game()
        random.seed(seed)

    def start_new_game(self):
        self.iteration = 0
        self.iteration_since_last_food = 0
        self.score = 0
        self.food = None
        self.snake = Snake()
        self.place_food()

    def place_food(self):
        while True:
            x = (
                random.randint(0, (self.width // Constants.BLOCK_SIZE) - 1)
            ) * Constants.BLOCK_SIZE
            y = (
                random.randint(0, (self.height // Constants.BLOCK_SIZE) - 1)
            ) * Constants.BLOCK_SIZE
            self.food = Point(x, y)
            if not self.snake.is_food_in_snake(self.food):
                break

    def update_ui(self):
        self.display.fill(Constants.BACKGROUNDCOLOR)

        # Draw food
        pygame.draw.rect(
            self.display,
            Constants.RED,
            pygame.Rect(
                self.food.x, self.food.y, Constants.BLOCK_SIZE, Constants.BLOCK_SIZE
            ),
        )

        # Draw snake body
        for pt in self.snake.snake[:-1]:
            pygame.draw.rect(
                self.display,
                Constants.SNAKECOLOR,
                pygame.Rect(pt.x, pt.y, Constants.BLOCK_SIZE, Constants.BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display,
                Constants.SNAKEOUTLINECOLOR,
                pygame.Rect(pt.x, pt.y, Constants.BLOCK_SIZE, Constants.BLOCK_SIZE),
                1,  # Thickness of the outline
            )

        # Draw snake head
        pygame.draw.rect(
            self.display,
            Constants.SNAKEHEADCOLOR,
            pygame.Rect(
                self.snake.snake[-1].x,
                self.snake.snake[-1].y,
                Constants.BLOCK_SIZE,
                Constants.BLOCK_SIZE,
            ),
        )
        pygame.draw.rect(
            self.display,
            Constants.SNAKEOUTLINECOLOR,
            pygame.Rect(
                self.snake.snake[-1].x,
                self.snake.snake[-1].y,
                Constants.BLOCK_SIZE,
                Constants.BLOCK_SIZE,
            ),
            1,  # Thickness of the outline
        )

        # Draw score
        text = Constants.FONT.render("Score: " + str(self.score), True, Constants.RED)
        self.display.blit(text, [0, 0])

        # Draw tick rate
        tick_rate_text = Constants.FONT.render(
            "Tick Rate: " + str(self.tick_rate), True, Constants.RED
        )
        self.display.blit(tick_rate_text, [self.width - 150, 0])

        pygame.display.flip()

    def handle_user_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    self.tick_rate += 20  # Increase tick rate
                elif event.key == pygame.K_s:
                    self.tick_rate = max(
                        self.default_tick_rate, self.tick_rate - 20
                    )  # Decrease tick rate but not below default
                elif event.key == pygame.K_r:
                    if self.tick_rate == self.default_tick_rate:
                        self.tick_rate = (
                            self.prev_tick_rate
                        )  # Example of flipping to a different value
                    else:
                        self.prev_tick_rate = self.tick_rate
                        self.tick_rate = self.default_tick_rate
                elif event.key == pygame.K_SPACE:  # Handle pause/resume
                    self.paused = not self.paused
                elif event.key == pygame.K_p:  # Handle screenshot
                    pygame.image.save(self.display, "screenshot.png")

    def play_step(self, action):
        if self.show_ui:
            self.handle_user_input()
            if self.paused:  # Check if the game is paused
                return False, 0, self.score

        self.iteration += 1
        self.iteration_since_last_food += 1

        # Move snake
        self.snake.change_snake_direction(action)
        self.snake.update_snake_position()

        # Check if game over
        if game_over := self.check_colision(self.snake.snake_head):
            return game_over, -10, self.score

        # Check if snake ate the food
        if self.snake.snake_head == self.food:
            self.place_food()
            reward = 10
            self.score += 1
            self.iteration_since_last_food = 0
        else:
            reward = 0
            self.snake.snake.pop(0)

        # # Penalty for taking too long
        # if self.iteration_since_last_food > 200:
        #     reward = -1

        if self.iteration_since_last_food >= 600:
            game_over = True
            reward = -10

        # Render the updated game state
        if self.show_ui:
            self.update_ui()
            self.clock.tick(self.tick_rate)

        return game_over, reward, self.score

    def check_colision(self, point):
        if (
            point.x > self.width - Constants.BLOCK_SIZE
            or point.x < 0
            or point.y > self.height - Constants.BLOCK_SIZE
            or point.y < 0
        ):
            return True

        if point in self.snake.snake[:-1]:
            return True

        return False

    def get_game_state(self):
        snake_surrondings = self.snake.get_head_surrounding()
        snake_direction = self.snake.snake_direction

        straigth_block = snake_direction.get_block_of_straight_direction(
            snake_surrondings
        )
        left_block = snake_direction.get_block_of_left_directions(snake_surrondings)
        right_block = snake_direction.get_block_of_right_directions(snake_surrondings)

        danger_straight = 0
        danger_left = 0
        danger_right = 0

        if self.check_colision(straigth_block):
            danger_straight = 1
        if self.check_colision(left_block):
            danger_left = 1
        if self.check_colision(right_block):
            danger_right = 1

        danger_state = [danger_straight, danger_left, danger_right]
        move_state = snake_direction.get_directions_vector()
        food_state = self.get_food_vector()

        return danger_state + move_state + food_state + self.get_sensor_data()

    def get_food_vector(self):
        vector = [0, 0, 0, 0]
        if self.food.x < self.snake.snake[-1].x:
            vector[0] = 1
        if self.food.x > self.snake.snake[-1].x:
            vector[1] = 1
        if self.food.y < self.snake.snake[-1].y:
            vector[2] = 1
        if self.food.y > self.snake.snake[-1].y:
            vector[3] = 1
        return vector

    def get_sensor_data(self):
        head = self.snake.snake_head
        sensor_data = []

        for direction in Constants.SENSOR_DIRECTIONS:
            distance = 0
            sensor_type = [0, 0, 0]  # [wall, food, body]
            current_point = Point(head.x, head.y)

            while True:
                distance += 1
                current_point = Point(
                    current_point.x + direction.x * Constants.BLOCK_SIZE,
                    current_point.y + direction.y * Constants.BLOCK_SIZE,
                )

                if not (
                    0 <= current_point.x < self.width
                    and 0 <= current_point.y < self.height
                ):
                    sensor_type[0] = 1  # Wall
                    break
                if current_point == self.food:
                    sensor_type[1] = 1  # Food
                    break
                if current_point in self.snake.snake[:-1]:
                    sensor_type[2] = 1  # Body
                    break

            sensor_data.extend(sensor_type + [distance])

        return sensor_data
