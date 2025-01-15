from agent import ACN
from agent.utils import rollout
from game import Snake
from game.utils import (
    Action,
    Direction,
    State,
    X, Y,
)

from typing import List, Tuple, Optional
from enum import Enum
import numpy as np
import torch


class Reward(float, Enum):
    '''Reward values for the game.'''
    FOOD  =  1
    DEATH = -2
    STEP  = -1e-5


class SnakeEnv:
    '''Reinforcement Learning environment for the Snake game.'''

    def __init__(self,
        gui: bool = False,
        **kwargs,
    ) -> None:
        '''Initialise a new game environment.'''
        # Initialise a base Snake game with the given parameters.
        self.game = Snake(user_input = False, gui = gui, **kwargs)

        # The snake can move at most max_steps times its length.
        self.max_steps = self.game.width * self.game.height
        self.step = 0

        # Define the action and observation spaces.
        self.action_space = len(Action)
        self.observation_space = self.get_observation().shape[0]


    def reset(self) -> None:
        '''Reset the game environment and return the initial observation.'''
        # Reset the game state and store the initial observation.
        self.game.reset()
        self.step = 0


    def play_step(self, action: Action) -> int:
        '''Perform an action and return the reward.'''
        # Store the last score before the step.
        last_score = self.get_score()

        # Play the step in the game.
        self.game.play_step(action = action)
        self.step += 1

        # Return the reward for the step.
        return self.get_reward(last_score = last_score)


    def get_reward(self,
        last_score: int,
    ) -> torch.tensor:
        '''Calculate the agent's reward.'''
        # If the snake has covered the entire game area, reward it.
        if np.all(self.game.state >= State.HEAD):
            reward = Reward.FOOD
        # If the snake died or ran out of time, punish it and end the game.
        elif self.game.done or self.get_game_progress() >= 1:
            self.game.done = True
            reward = Reward.DEATH
        # If the snake ate food, reward it.
        elif self.get_score() > last_score:
            reward = Reward.FOOD
        # Otherwise, apply the default step reward.
        else:
            reward = Reward.STEP
        # Return the reward as a tensor.
        return torch.tensor(reward.value, dtype = torch.float32)


    def get_score(self) -> int:
        '''Return the current game score.'''
        return self.game.score


    def get_state(self, copy: bool = False) -> np.ndarray:
        '''Return the current game state.'''
        return self.game.state if not copy else self.game.state.copy()


    def get_game_progress(self) -> float:
        '''Return the game progress as a float between 0 and 1.'''
        return self.step / (self.game.get_snake_len() * self.max_steps)


    def get_observation(self) -> torch.Tensor:
        '''Return an observation tensor for the current game state. Components:
        - The sum of danger values in different directions. (Left, forward, right, forward left, forward right)
        - The snake's current direction as one-hot vector.
        - The relative position of the food as one-hot vector.
        - The snake's step count, normalised by the maximum allowed step count.'''
        # The initial observation contains 5 elements representing nearby dangers.
        observation = self.__observe_dangers__(danger_map = self.__map_dangers__())

        # The next 4 elements represent the snake's current direction as one-hot vectors.
        observation.extend(
            int(self.game.direction == direction)
            for direction in Direction
        )
        # The following 4 elements represent the relative position of the food, again as one-hot vectors.
        observation.extend((
            self.game.head_pos[Y] < self.game.food_pos[Y], # Food above
            self.game.head_pos[Y] > self.game.food_pos[Y], # Food below
            self.game.head_pos[X] < self.game.food_pos[X], # Food left
            self.game.head_pos[X] > self.game.food_pos[X], # Food right
        ))
        # The final element represents the danger posed by the time limit.
        observation.append(self.get_game_progress())
        # Return the observation tensor.
        return torch.tensor(observation, dtype = torch.float32)


    def __map_dangers__(self,
        boundary: int = 1,
    ) -> np.ndarray:
        '''Generate a danger map of the game state. It contains floats between 0 and 1, where 0 is safe and 1 is dangerous.'''
        # We'll apply offsets to account for the game boundaries. Adjust the head position accordingly.
        height, width  = self.game.height + 2 * boundary, self.game.width + 2 * boundary
        head_pos       = (self.game.head_pos[Y] + boundary, self.game.head_pos[X] + boundary)

        # First of all, calculate the Manhattan distance from the snake's head to each cell of the (upcoming) danger map.
        l1_distance  = np.zeros((height, width), dtype = np.int16)             # Initialise with zeros.
        l1_distance += np.abs(np.arange(height) - head_pos[Y])[:, np.newaxis]  # Absolute vertical distance.
        l1_distance += np.abs(np.arange(width)  - head_pos[X])                 # Absolute horizontal distance.

        # Now we'll populate the danger map with the game state, and the snake's head and food.
        danger_map = np.ones_like(l1_distance, dtype=np.float32)               # Initialise with ones for the border.
        danger_map[boundary:-boundary, boundary:-boundary] = self.get_state()  # Copy the game state.

        # Hide snake parts that will vanish by the time the snake reaches them.
        danger_map[(danger_map >= State.HEAD) & (danger_map + l1_distance > self.game.get_snake_tail_idx())] = State.GROUND

        # Normalise the danger map by the distance from the snake's head.
        # We square the L1 distance to reduce the relevance of distant cells.
        return danger_map.clip(0, 1) / (l1_distance ** 2).clip(1, None)


    def __observe_dangers__(self,
        danger_map: np.ndarray,
        boundary: int = 1,
    ) -> List[float]:
        '''Generate an observation based on the danger map. This is done by summing up danger values along different directions.'''
        observation = []
        head_pos = (self.game.head_pos[Y] + boundary, self.game.head_pos[X] + boundary)
        # Observe "left", "forward", and "right", from the snake's current point of view.
        possible_directions = [self.game.direction + action for action in Action]
        for direction in possible_directions:
            match direction:
                case Direction.LEFT:
                    # Calculate the sum of the danger values to the left of the snake's head.
                    danger_vector = danger_map[head_pos[Y], 0:head_pos[X]]
                    observation.append(danger_vector.sum())
                case Direction.UP:
                    # Calculate the sum of the danger values above the snake's head.
                    danger_vector = danger_map[0:head_pos[Y], head_pos[X]]
                    observation.append(danger_vector.sum())
                case Direction.RIGHT:
                    # Calculate the sum of the danger values to the right of the snake's head.
                    danger_vector = danger_map[head_pos[Y], head_pos[X]+1:]
                    observation.append(danger_vector.sum())
                case Direction.DOWN:
                    # Calculate the sum of the danger values below the snake's head.
                    danger_vector = danger_map[head_pos[Y]+1:, head_pos[X]]
                    observation.append(danger_vector.sum())

        # Consider diagonal dangers. First, determine the danger map diagonals intersecting the snake's head.
        diagonal_downwards = np.diagonal(danger_map,            offset = head_pos[X] - head_pos[Y])
        diagonal_upwards   = np.diagonal(np.flipud(danger_map), offset = head_pos[X] + head_pos[Y] - (danger_map.shape[Y] - 1))

        # Clip the diagonals at the snake's head.
        snake_head_value     = danger_map[*head_pos]
        downward_split_index = np.argwhere(diagonal_downwards == snake_head_value)[0].item()
        upward_split_index   = np.argwhere(diagonal_upwards   == snake_head_value)[0].item()

        # Observe "forward left" and "forward right" based on the snake's current movement direction.
        match self.game.direction:
            case Direction.UP:
                # If we're moving upwards, "left up" is left of the snake's head, and "right up" is right of the snake's head.
                danger_left_up    = diagonal_downwards[:downward_split_index]
                danger_right_up   = diagonal_upwards[:upward_split_index]
                observation.extend((danger_left_up.sum(), danger_right_up.sum()))
            case Direction.RIGHT:
                # If we're moving right, "right up" is left of the snake's head, and "right down" is right of the snake's head.
                danger_right_up   = diagonal_upwards[:upward_split_index]
                danger_right_down = diagonal_downwards[downward_split_index + 1:]
                observation.extend((danger_right_up.sum(), danger_right_down.sum()))
            case Direction.DOWN:
                # If we're moving downwards, "right down" is left of the snake's head, and "left down" is right of the snake's head.
                danger_right_down = diagonal_downwards[downward_split_index + 1:]
                danger_left_down  = diagonal_upwards[upward_split_index + 1:]
                observation.extend((danger_right_down.sum(), danger_left_down.sum()))
            case Direction.LEFT:
                # If we're moving left, "left down" is left of the snake's head, and "left up" is right of the snake's head.
                danger_left_down  = diagonal_upwards[upward_split_index + 1:]
                danger_left_up    = diagonal_downwards[:downward_split_index]
                observation.extend((danger_left_down.sum(), danger_left_up.sum()))
        # Return the observed danger values.
        return observation


    def plot_danger_map(self,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> None:
        '''Plot the danger map as a heatmap.'''
        # We don't need to import these libraries unless we're plotting.
        import seaborn as sns ; sns.set_theme('notebook', style='dark')
        import matplotlib.pyplot as plt ; plt.style.use("dark_background")

        # Generate the danger map and replace the snake's head with a safe value.
        danger_map = self.__map_dangers__()

        plt.figure(figsize=self.game.state.shape if not figsize else figsize)
        sns.heatmap(danger_map, cmap='hot', square=True, cbar=False, annot=True, fmt=".2f")
        plt.title('Danger Map', fontsize = 16)

        # Label the axes starting at -1 to account for the game boundaries.
        plt.xticks(np.arange(danger_map.shape[X]) + 0.5, labels = range(-1, danger_map.shape[X] - 1))
        plt.yticks(np.arange(danger_map.shape[Y]) + 0.5, labels = range(-1, danger_map.shape[Y] - 1))
        plt.show()


    def evaluate(self,
        model: ACN,
        num_games: int,
        silent: bool = False,
    ) -> List[torch.Tensor]:
        '''Evaluate the given model by simulating multiple games and averaging the scores.
        Return the best game replay.'''
        model.eval()
        best_game  = None
        best_score = 0
        scores     = 0
        for _ in range(num_games):
            # Simulate an entire game and record the final score.
            _, score, replay = rollout(model = model, env = self)
            scores += score
            if score > best_score:
                # New highscore! Save the game replay.
                best_score = score
                best_game = replay
        if not silent:
            print(f'Mean: {scores / num_games:.2f} | Highscore: {best_score}')
        return best_game
