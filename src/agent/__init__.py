from typing import Tuple
import torch.nn as nn
import torch


class ACN(nn.Module):
    '''Actor Critic model, predicts both policy and value of a given observation.'''

    def __init__(self,
        observation_space: int,
        action_space: int,
        hidden_size: int = 128,
        act_fn = nn.ReLU,
    ) -> None:
        '''Initialise the actor critic network.'''
        super().__init__()
        # Store the input and output shapes.
        self.observation_space = observation_space
        self.action_space      = action_space

        # Shared feature layer. Learns to understand the game state.
        self.feature = nn.Sequential(
            nn.Linear(observation_space, hidden_size),
            act_fn(),
            nn.Linear(hidden_size, hidden_size),
            act_fn()
        )
        # Value stream. Learns state values.
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            act_fn(),
            nn.Linear(hidden_size // 2, 1)
        )
        # Policy stream. Learns action probabilities for a given state.
        self.policy_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            act_fn(),
            nn.Linear(hidden_size // 2, action_space)
        )


    def value(self, state) -> torch.Tensor:
        '''Return the value of the given state.'''
        return self.value_stream(self.feature(state)).squeeze()


    def policy(self, state) -> torch.Tensor:
        '''Return the policy logits for a given state.'''
        return self.policy_stream(self.feature(state))


    def forward(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Forward pass through the network. Returns the policy logits and state value.'''
        # Compute the shared features.
        features = self.feature(state)
        # Return the policy logits and state value.
        return self.policy_stream(features), self.value_stream(features).squeeze()


    def save(self, filename: str) -> None:
        '''Export the model parameters to a file.'''
        torch.save(self.state_dict(), filename)


    def load(self, filename: str) -> 'ACN':
        '''Load the model parameters from a file.'''
        self.load_state_dict(torch.load(filename, weights_only = False))
        return self
