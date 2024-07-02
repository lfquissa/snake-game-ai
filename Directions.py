from enum import Enum
from itertools import compress


class Directions(Enum):
    RIGHT = 1
    LEFT = 2
    DOWN = 3
    UP = 4

    def directions(self):
        return [Directions.LEFT, Directions.UP, Directions.RIGHT, Directions.DOWN]

    def get_directions_vector(self):
        directions = self.directions()
        return list(map(lambda x: 1 if x.value == self.value else 0, directions))

    def get_block_of_straight_direction(self, blocks):
        directions = self.get_directions_vector()
        booleans = list(map(lambda x: False if x == 0 else True, directions))
        lista = list(compress(blocks, booleans))
        return lista[0]

    def get_block_of_left_directions(self, blocks):
        directions = self.get_directions_vector()
        index = directions.index(1)
        new_index = (index - 1) % 4
        directions[index] = 0
        directions[new_index] = 1

        booleans = list(map(lambda x: False if x == 0 else True, directions))
        lista = list(compress(blocks, booleans))
        return lista[0]

    def get_block_of_right_directions(self, blocks):
        directions = self.get_directions_vector()
        index = directions.index(1)
        new_index = (index + 1) % 4
        directions[index] = 0
        directions[new_index] = 1

        booleans = list(map(lambda x: False if x == 0 else True, directions))
        lista = list(compress(blocks, booleans))
        return lista[0]
