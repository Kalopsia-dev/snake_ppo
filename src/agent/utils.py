from game.utils import Action

from typing import Dict, List, Tuple
import torch.nn as nn
import torch


def rollout(env,
    model: nn.Module,
    gamma: float = 0.99,
    gae_smoothing: float = 0.97,
) -> Tuple[Dict[str, torch.Tensor], int, List[torch.Tensor]]:
    '''Simulate a single game of Snake using the given agent.
    Return the rollout data and final score.'''
    # Initialise the game state and training metrics.
    train_data = {
        'states':    [],
        'actions':   [],
        'rewards':   [],
        'values':    [],
        'log_probs': [],
    }
    env.reset()
    game_states = []
    while not env.game.done:
        # Get the current state.
        state = env.get_observation()

        # Store the actual game states for replay purposes.
        game_states.append(env.get_state(copy = True))

        # Get the policy logits and state value.
        policy_logits, state_value = model(state)

        # Sample an action from the policy.
        action_distribution = torch.distributions.Categorical(logits = policy_logits)
        action = action_distribution.sample()

        # Choose an action for the next game step.
        reward = env.play_step(Action(action.item()))

        # Record the state, action, reward, and value.
        train_data['states'].append(state)
        train_data['actions'].append(action)
        train_data['rewards'].append(reward)
        train_data['values'].append(state_value)
        train_data['log_probs'].append(action_distribution.log_prob(action))

    # Convert the rollout data to PyTorch tensors.
    train_data = {key: torch.stack(value).detach()
                  for key, value in train_data.items()}

    # Calculate advantages using Generalised Advantage Estimation.
    train_data['values'] = generalised_advantage_estimate(
                               rewards       = train_data['rewards'],
                               values        = train_data['values'],
                               gae_smoothing = gae_smoothing,
                               gamma         = gamma,
                           )
    # Return the rollout data.
    return train_data, env.get_score(), game_states


def generalised_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_smoothing: float = 0.97,
) -> torch.Tensor:
    '''Calculate the Generalised Advantage Estimation (GAE) advantages.'''
    with torch.no_grad():
        # First, shift the state values by one step.
        next_values = values.roll(shifts = -1)
        next_values[-1] = 0 # Terminal state.

        # Compute temporal differences, then the GAE.
        reward_deltas = rewards + gamma * next_values - values
        return discounted_cumsums(reward_deltas, discount_factor = gamma * gae_smoothing)


def discounted_cumsums(reward_deltas: torch.Tensor, discount_factor: float) -> torch.Tensor:
    '''Compute the discounted cumulative sums of rewards.'''
    with torch.no_grad():
        # Generate a vector of ascending powers of the discount factor.
        epsilon = torch.finfo(torch.float64).tiny # Add a tiny value for numerical stability.
        discount_factors = discount_factor ** torch.arange(len(reward_deltas), dtype = torch.float64) + epsilon

        # Compute the cumulative sums and normalise them by the discount factors.
        cumulative_sums = torch.cumsum((reward_deltas.flip(dims = [0]) * discount_factors.flip(dims = [0])), dim = 0).flip(dims = [0])
        return cumulative_sums / discount_factors
