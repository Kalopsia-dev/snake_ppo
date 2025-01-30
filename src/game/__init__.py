from game.utils import (
    Action,
    Direction,
    State,
    X, Y,
)

from typing import Tuple, Optional
import numpy as np


class Snake:
    '''Main game class. Handles the game loop and game state.'''

    def __init__(self,
        width: int,
        height: int,
        gui: bool = True,
        **kwargs,
    ) -> None:
        '''Initialise a new game instance.'''
        # Remember the game configuration.
        self.height = max(5, height)
        self.width  = max(5, width)
        self.gui    = False

        # Initialise the game state array.
        self.state = np.zeros((self.height, self.width), dtype = np.int16)

        if gui:
            # Import the GUI module only if needed.
            from game.gui import SnakeGUI

            # Launch a game window and attach it to the game state.
            self.gui = SnakeGUI(game_state = self.state, **kwargs)

        # Prepare the game.
        self.reset()


    def reset(self) -> None:
        '''Initialise the game state.'''
        # Clear prior game variables.
        self.direction = Direction.RIGHT
        self.done      = False
        self.score     = 0

        # Reset the game state.
        self.state.fill(State.GROUND)

        # Add the snake, represented by ascending numbers.
        self.head_pos = (self.height // 2, self.width // 2)
        for i in range(4):
            self.state[self.head_pos[Y], self.head_pos[X] - i] = State.HEAD + i

        # Place food at a random position.
        self.place_food()

        # Update the game window, if available.
        if self.gui:
            self.gui.update(tick = False)


    def place_food(self) -> None:
        '''Place food at a random, empty position.'''
        # Recursively find an empty position for the food. Place and remember it.
        self.food_pos = self.get_blank_position()
        if self.food_pos:
            self.state[*self.food_pos] = State.FOOD
        else:
            # We've run out of space for food. The game is over.
            self.done = True


    def get_blank_position(self) -> Optional[Tuple[int, int]]:
        '''Return a random, empty position within the game area, if available.'''
        # Find all ground positions and randomly select one.
        ground_positions = np.argwhere(self.state == State.GROUND)
        if ground_positions.shape[0] > 0:
            return tuple(ground_positions[np.random.randint(ground_positions.shape[0])])
        # If no empty positions remain, return None.
        return None


    def get_snake_tail(self) -> Tuple[int, int]:
        '''Return the snake's current tail position.'''
        return np.unravel_index(np.argmax(self.state), (self.height, self.width))


    def get_snake_tail_idx(self) -> int:
        '''Return the index of the snake's tail.'''
        return np.max(self.state)


    def get_snake_len(self) -> int:
        '''Return the snake's current length.'''
        # The snake is an ascending sequence of integers, starting at State.HEAD.
        return self.get_snake_tail_idx() - State.HEAD + 1


    def move_snake(self, action: Action) -> None:
        '''Move the snake based on the provided action and its current direction.'''
        # Determine the new head position and direction.
        self.direction += action
        next_head_pos = self.direction.shift(position = self.head_pos)

        # Find the snake's tail.
        tail_pos = self.get_snake_tail()
        tail_idx = self.state[*tail_pos]

        # Wall collision check.
        if (next_head_pos[Y] < 0 or next_head_pos[Y] >= self.height
         or next_head_pos[X] < 0 or next_head_pos[X] >= self.width):
            self.done = True
        # Snake collision check.
        elif tail_idx > self.state[*next_head_pos] >= State.HEAD:
            self.done = True
        # No collision or food.
        else:
            food_collision = self.state[*next_head_pos] == State.FOOD
            if not food_collision:
                # We only need to move the tail if the snake didn't eat food.
                self.state[*tail_pos] = State.GROUND

            # Move the snake forward by incrementing its body indices.
            self.state[self.state >= State.HEAD] += 1
            self.state[*next_head_pos] = State.HEAD
            self.head_pos = next_head_pos

            if food_collision:
                # Place new food and increase the score.
                self.place_food()
                self.score += 1


    def play_step(self, action: Action = Action.FORWARD) -> None:
        '''Play a single step of the game. Return the agent's reward.'''
        if self.gui:
            # Handle user input and the close window event.
            new_direction = self.gui.handle_events()
            if new_direction is not None:
                # We don't want the snake to turn back on itself.
                if self.direction != new_direction.inverse():
                    self.direction = new_direction
            # Human players move the snake based on directions, not actions.
            if self.gui.user_input:
                action = Action.FORWARD

        # Move the snake based on the given action.
        self.move_snake(action)

        # Update the GUI, if available.
        if self.gui:
            self.gui.update(tick = True)


    @staticmethod
    def play(
        width: int,
        height: int,
        **kwargs,
    ) -> None:
        '''Let a human player control the snake.'''
        # Create a new game instance with a GUI accepting user input.
        game = Snake(
            width      = width,
            height     = height,
            gui        = True,
            user_input = True,
            **kwargs,
        )
        # Run the game until it's over.
        while not game.done:
            game.play_step()
        # Print the final score.
        print(f'Game over! Score: {game.score}')
        game.gui.quit()
