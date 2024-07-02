from enum import Enum

import numpy as np

from Directions import Directions


class Action(Enum):
    STRAIGHT = [1, 0, 0]
    RIGHT = [0, 1, 0]
    LEFT = [0, 0, 1]

    def next_direction(self, current_direction):
        clock_wise = [Directions.UP, Directions.RIGHT, Directions.DOWN, Directions.LEFT]
        index = clock_wise.index(current_direction)

        if np.array_equal(self.value, [1, 0, 0]):
            return clock_wise[index % 4]
        elif np.array_equal(self.value, [0, 1, 0]):
            return clock_wise[(index + 1) % 4]
        elif np.array_equal(self.value, [0, 0, 1]):
            return clock_wise[(index - 1) % 4]
