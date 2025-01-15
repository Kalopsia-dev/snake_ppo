from enum import IntEnum
from typing import Tuple


# Game Axes
Y, X = 0, 1


class State(IntEnum):
    '''Integer values representing the game state.'''
    FOOD   = -1
    GROUND = 0
    HEAD   = 2


class Action(IntEnum):
    '''Actions the snake can take.'''
    LEFT    = 0
    FORWARD = 1
    RIGHT   = 2


class Direction(IntEnum):
    '''Directions the snake can face.
    Integer values represent clockwise rotation.'''
    LEFT  = 0
    UP    = 1
    RIGHT = 2
    DOWN  = 3


    def __add__(self, action: Action):
        '''Return the new direction after performing the given action.'''
        if isinstance(action, Action):
            return Direction((self.value + action.value - 1) % 4)
        raise ValueError('Cannot add non-Action object to Direction.')


    def __iadd__(self, action: Action):
        '''Update the direction after performing the given action.'''
        return self.__add__(action)


    def inverse(self):
        '''Return the opposite direction.'''
        return Direction((self.value + 2) % 4)


    def shift(self,
        position: Tuple[int, int],
    ) -> Tuple[int, int]:
        '''Shift a position in the direction and return the result.'''
        # Determine the new head position.
        match self:
            case Direction.LEFT:
                return (position[Y], position[X] - 1)
            case Direction.UP:
                return (position[Y] - 1, position[X])
            case Direction.RIGHT:
                return (position[Y], position[X] + 1)
            case Direction.DOWN:
                return (position[Y] + 1, position[X])
