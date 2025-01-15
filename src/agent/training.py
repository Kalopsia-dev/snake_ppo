from game.env import SnakeEnv
from agent.utils import rollout
from agent import ACN

from collections import deque
from typing import List
import torch.nn as nn
import numpy as np
import torch
import os


class PPOTrainer:
    '''Trainer class for the PPO model.'''
    def __init__(self,
        model: ACN,
        ppo_clip_value: float = 0.2,
        target_kl_div: float = 0.02,
        max_policy_updates: int = 30,
        policy_lr: float = 2e-5,
        value_lr: float = 2e-6,
        gamma: float = 0.99,
        gae_smoothing: float = 0.97,
    ) -> None:
        '''Initialise the trainer with the given model and optimiser.'''
        # Store the model and hyperparameters.
        self.model              = model
        self.policy             = model.policy_stream
        self.value              = model.value_stream
        self.ppo_clip_value     = ppo_clip_value
        self.target_kl_div      = target_kl_div
        self.max_policy_updates = max_policy_updates
        self.gae_smoothing      = gae_smoothing
        self.gamma              = gamma
        self.n_games            = 0

        # Create separate optimisers for the policy and value networks.
        policy_parameters = list(model.feature.parameters()) + list(model.policy_stream.parameters())
        value_parameters  = list(model.feature.parameters()) + list(model.value_stream.parameters())
        self.policy_optimizer = torch.optim.AdamW(policy_parameters, lr = policy_lr)
        self.value_optimizer  = torch.optim.AdamW(value_parameters,  lr = value_lr)


    def train_policy(self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        '''Train the policy network using Proximal Policy Optimisation.'''
        for _ in range(self.max_policy_updates):
            # Zero the policy gradients.
            self.policy_optimizer.zero_grad()

            # Determine the new policy logits and log probabilities.
            new_policy_logits = torch.distributions.Categorical(logits = self.model.policy(states))
            new_log_probs = new_policy_logits.log_prob(actions)

            # Compute the policy ratio and apply the PPO-Clip objective.
            policy_ratio  = torch.exp(new_log_probs - old_log_probs)
            clamped_ratio = policy_ratio.clamp(1 - self.ppo_clip_value, 1 + self.ppo_clip_value)

            # Compute the PPO objective.
            clamped_loss = clamped_ratio * advantages
            full_loss    = policy_ratio * advantages

            # The policy loss is the minimum of the clamped and full losses.
            policy_loss = -torch.min(clamped_loss, full_loss).mean()

            # Perform a policy update.
            policy_loss.backward()
            self.policy_optimizer.step()

            # Check the KL divergence between the old and new policy. If it exceeds the target, stop training.
            kl_divergence = (old_log_probs - new_log_probs).mean()
            if kl_divergence >= self.target_kl_div:
                break


    def train_value(self,
        states: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        '''Train the value network using the Mean Squared Error loss.'''
        # Zero the value gradients.
        self.value_optimizer.zero_grad()

        # Compute the value predictions and the value loss.
        values = self.model.value(states)
        value_loss = nn.functional.mse_loss(values, returns)

        # Perform a value update.
        value_loss.backward()
        self.value_optimizer.step()


def train(
    env: SnakeEnv,
    trainer: PPOTrainer,
    episodes: int,
    feedback_interval: int = 100,
    checkpoint_dir: str = 'assets/models/',
) -> List[torch.Tensor]:
    '''Training loop for the PPO agent. Returns a replay of the best game.'''
    # Initialise the training variables.
    best_score = 0
    best_game = None
    scores = deque(maxlen = feedback_interval)
    # Start training.
    trainer.model.train()
    for _ in range(episodes):
        # Simulate a single episode of the game.
        train_data, score, replay = rollout(
            env           = env,
            model         = trainer.model,
            gamma         = trainer.gamma,
            gae_smoothing = trainer.gae_smoothing,
        )
        trainer.n_games += 1

        # Randomly shuffle the training data.
        permute_idx = torch.randperm(len(train_data['rewards']))
        train_data  = {key: data[permute_idx]
                       for key, data in train_data.items()}

        # Train the model using the collected data.
        trainer.train_policy(
            states        = train_data['states'],
            actions       = train_data['actions'],
            old_log_probs = train_data['log_probs'],
            advantages    = train_data['values'],
        )
        trainer.train_value(
            states  = train_data['states'],
            returns = train_data['rewards'],
        )

        # Print some statistics.
        scores.append(score)
        if trainer.n_games % feedback_interval == 0:
            print(f'EPISODE {trainer.n_games:{len(str(episodes))}d} | Mean: {sum(scores) / len(scores):4.1f}, Highscore: {best_score:2d} |{box_plot_string(scores)}')

        # Check if we've achieved a new high score.
        if score > best_score:
            best_score = score
            best_game = replay
            if score >= (env.game.width * env.game.height) // 2:
                # Save the model, if it covers a significant portion of the game area.
                trainer.model.save(os.path.join(checkpoint_dir, f'snake_agent_ppo_{best_score:02d}.pth'))

    # Save the final model.
    trainer.model.save(os.path.join(checkpoint_dir, f'snake_agent_ppo_last.pth'))
    print('Training complete!')
    return best_game


def box_plot_string(scores: deque) -> str:
    '''Returns a textual box plot of the scores to show their distribution while training.'''
    # Calculate the box plot statistics.
    min_value       = min(scores)
    lower_quartile  = int(np.round(np.percentile(scores, 25), 0))
    median_value    = int(np.round(np.percentile(scores, 50), 0))
    upper_quartile  = int(np.round(np.percentile(scores, 75), 0))
    max_value       = max(scores)

    # Return the box plot as a string.
    return ' ' * (min_value-1) + '▕' \
         + ' ' * (lower_quartile - min_value) \
         + '░' * (median_value - lower_quartile) \
         + '█' \
         + '░' * (upper_quartile - median_value) \
         + ' ' * (max_value-1 - upper_quartile) + '▏'
