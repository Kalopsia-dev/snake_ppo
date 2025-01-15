from game.utils import (
    Direction,
    State,
    X, Y,
)

from typing import List, Tuple, Optional
from enum import Enum
import numpy as np
import pygame


class Colour(Tuple[int, int, int], Enum):
    '''Predefined colours for the Snake game.'''
    BG    = ( 10,  14, 25)
    GRID  = (  2,   6, 17)
    FOOD  = (235,  52, 52)
    SNAKE = (110, 219, 72)


class Quadrant(Tuple[int, int], Enum):
    '''Quadrant constants for the neighbourhood of a snake segment.'''
    TOP_LEFT     = (0, 0)
    TOP_RIGHT    = (0, 1)
    BOTTOM_LEFT  = (1, 0)
    BOTTOM_RIGHT = (1, 1)


class SnakeGUI:
    '''PyGame GUI for the Snake game.'''

    def __init__(self,
        game_state: np.ndarray,
        block_size: int = 24,
        user_input: bool = False,
        fps: int = 3,
    ) -> None:
        '''Initialise a new game instance.'''
        # Remember the game grid size.
        self.height, self.width = game_state.shape
        self.game_state = game_state

        # Launch the game window.
        self.fps        = fps
        self.block_size = block_size
        self.user_input = user_input
        self.display    = pygame.display.set_mode(size = (self.width * self.block_size, self.height * self.block_size))
        self.clock      = pygame.time.Clock()
        pygame.display.set_caption('Snake')

        # Precompute the pixel sizes of displayed objects.
        self.size_food     = int(self.block_size * 0.75)
        self.size_head     = int(self.block_size * 0.75)
        self.size_head_alt = int(self.size_head * 0.8)
        self.size_snake    = self.block_size // 2
        self.size_tail     = self.block_size // 2
        self.size_grid     = max(1, self.block_size // 12)


    def handle_events(self) -> Optional[Direction]:
        '''Handle user input and return the snake's new direction.'''
        # Handle user input.
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    # If the user closes the window, exit the game.
                    self.quit()
                case pygame.KEYDOWN:
                    if not self.user_input:
                        # If user input is disabled, ignore it.
                        continue
                    match event.key:
                        case pygame.K_LEFT:
                            return Direction.LEFT
                        case pygame.K_UP:
                            return Direction.UP
                        case pygame.K_RIGHT:
                            return Direction.RIGHT
                        case pygame.K_DOWN:
                            return Direction.DOWN
        return None


    def quit(self) -> None:
        '''Close the game window.'''
        pygame.quit()


    def update(self,
        tick: bool = True,
    ) -> None:
        '''Update the game window.'''
        # Generate the background.
        self.display.fill(Colour.BG)

        # Draw all game objects.
        game_objects = np.argwhere(self.game_state != State.GROUND)
        all(self.draw(obj) for obj in game_objects)

        # Draw a grid above the game objects.
        self.draw_grid()

        # Update the game window.
        pygame.display.flip()
        if tick:
            self.clock.tick(self.fps)


    def fill(self,
        color: Tuple[int, int, int],
        pos: Tuple[int, int],
        size: Tuple[int, int],
        offset: Tuple[float, float] = (0, 0),
    ) -> None:
        '''Fill a rectangular area with a given colour.'''
        screen_pos_x = (pos[X] + offset[X]) * self.block_size
        screen_pos_y = (pos[Y] + offset[Y]) * self.block_size
        self.display.fill(color, (int(screen_pos_x), int(screen_pos_y), *size))


    def draw(self, pos: Tuple[int, int]) -> True:
        '''Draw a game object at the given position.'''
        # Determine which object to draw.
        index = self.game_state[*pos]
        if index == State.FOOD:
            # Food is simple. A red square, centred and slightly smaller than the grid.
            self.fill(Colour.FOOD, pos, (self.size_food, self.size_food), offset = (0.2, 0.2))

        elif index >= State.HEAD:
            # For snake parts, we must determine the type and orientation. Let's use the descending order of snake parts to our advantage.
            head_segment  = 2 * index + 1 #         n + (n+1) = 2n + 1
            inner_segment = 3 * index     # (n-1) + n + (n+1) = 3n
            tail_segment  = 2 * index - 1 # (n-1) + n         = 2n - 1

            # Read the current segment's neighbourhood.
            neighbours = self.game_state[max(0, pos[Y]-1):pos[Y]+2, max(0, pos[X]-1):pos[X]+2].copy()
            if neighbours.shape != (3, 3):
                # For (literal) edge cases, we need to pad the neighbourhood with ground values.
                padding = ((1 if pos[Y] == 0 else 0, 1 if pos[Y] == self.height - 1 else 0),
                           (1 if pos[X] == 0 else 0, 1 if pos[X] == self.width  - 1 else 0))
                neighbours = np.pad(neighbours, padding, mode='constant', constant_values = 0)

            # Mask irrelevant neighbours so our above formulae work correctly.
            neighbours[(neighbours < index - 1) | (neighbours > index + 1)] = 0

            # Horizontal and vertical sums identify the simpler segments.
            horizontal_sum = np.sum(neighbours, axis = X)[1]
            vertical_sum   = np.sum(neighbours, axis = Y)[1]
            # For all others, we will also need the sums of the four 2x2 subregions of the neighbourhood.
            quadrant_sums  = np.lib.stride_tricks.sliding_window_view(neighbours, window_shape = (2, 2)).sum(axis = (-2, -1))

            # SNAKE BODY
            if horizontal_sum == inner_segment:
                # Horizontal segment
                self.fill(Colour.SNAKE, pos, (self.block_size, self.size_snake), offset = (0.3, 0))
            elif vertical_sum == inner_segment:
                # Vertical segment
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.block_size), offset = (0, 0.3))

            # SNAKE CORNERS
            elif quadrant_sums[*Quadrant.TOP_LEFT] == inner_segment:
                # Top-left corner
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.size_snake), offset = (0, 0.3))
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.size_snake), offset = (0.3, 0))
            elif quadrant_sums[*Quadrant.TOP_RIGHT] == inner_segment:
                # Top-right corner
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.size_snake), offset = (0, 0.3))
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.size_snake), offset = (0.3, 0.5))
            elif quadrant_sums[*Quadrant.BOTTOM_LEFT] == inner_segment:
                # Bottom-left corner
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.size_snake), offset = (0.5, 0.3))
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.size_snake), offset = (0.3, 0))
            elif quadrant_sums[*Quadrant.BOTTOM_RIGHT] == inner_segment:
                # Bottom-right corner
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.size_snake), offset = (0.5, 0.3))
                self.fill(Colour.SNAKE, pos, (self.size_snake, self.size_snake), offset = (0.3, 0.5))

            # SNAKE TAIL
            elif horizontal_sum == tail_segment:
                if quadrant_sums[*Quadrant.TOP_LEFT] == tail_segment:
                    # Tail facing right (connected to the left)
                    self.fill(Colour.SNAKE, pos, (self.size_tail, self.size_tail), offset = (0.3, 0))
                else:
                    # Tail facing left (connected to the right)
                    self.fill(Colour.SNAKE, pos, (self.size_tail, self.size_tail), offset = (0.3, 0.5))
            elif vertical_sum == tail_segment:
                if quadrant_sums[*Quadrant.TOP_LEFT] == tail_segment:
                    # Tail facing down (connected to the top)
                    self.fill(Colour.SNAKE, pos, (self.size_tail, self.size_tail), offset = (0, 0.3))
                else:
                    # Tail facing up (connected to the bottom)
                    self.fill(Colour.SNAKE, pos, (self.size_tail, self.size_tail), offset = (0.5, 0.3))

            # SNAKE HEAD
            elif horizontal_sum == head_segment:
                if quadrant_sums[*Quadrant.TOP_LEFT] == head_segment:
                    # Head facing right (connected to the left)
                    self.fill(Colour.SNAKE, pos, (self.size_head, self.size_head_alt), offset = (0.25, 0))
                else:
                    # Head facing left (connected to the right)
                    self.fill(Colour.SNAKE, pos, (self.size_head, self.size_head_alt), offset = (0.25, 0.3))
            elif vertical_sum == head_segment:
                if quadrant_sums[*Quadrant.TOP_LEFT] == head_segment:
                    # Head facing down (connected to the top)
                    self.fill(Colour.SNAKE, pos, (self.size_head_alt, self.size_head), offset = (0, 0.25))
                else:
                    # Head facing up (connected to the bottom)
                    self.fill(Colour.SNAKE, pos, (self.size_head_alt, self.size_head), offset = (0.3, 0.25))
        return True


    def draw_grid(self) -> None:
        '''Draw a grid over the game window.'''
        for x in range(0, self.width * self.block_size, self.block_size):
            pygame.draw.line(self.display, Colour.GRID, (x, 0), (x, self.height * self.block_size), self.size_grid)
        for y in range(0, self.height * self.block_size, self.block_size):
            pygame.draw.line(self.display, Colour.GRID, (0, y), (self.width * self.block_size, y), self.size_grid)

    def __render_state__(self,
        game_state: np.ndarray,
    ) -> np.ndarray:
        '''Render a single game state as an image.'''
        self.game_state = game_state
        self.update(tick = False)
        # Return the game state as an image. We need to swap the axes for the correct orientation.
        return pygame.surfarray.array3d(self.display).swapaxes(0, 1)


def render_replay(
    game_states: List[np.ndarray],
    output_file: str = 'snake.gif',
    fps: int = 3,
) -> None:
    '''Output a gif of the game states.'''
    # If there are no game states, we can't render anything.
    if not game_states:
        return

    # We don't need to import this library unless we're rendering a GIF.
    import imageio

    # Create a GUI instance to render the game states.
    helper = SnakeGUI(
        game_state = game_states[0],
        user_input = False,
        block_size = 24,
        fps        = fps,
    )
    frames = [helper.__render_state__(state)
              for state in game_states]

    # Save the frames as a GIF.
    imageio.mimsave(output_file, frames, fps = fps, loop = 0)
