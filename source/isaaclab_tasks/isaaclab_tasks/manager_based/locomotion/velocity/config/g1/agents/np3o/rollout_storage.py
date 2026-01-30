# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications Copyright (c) 2026, Özhan Özen

from __future__ import annotations

from collections.abc import Generator

import torch
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


class N3PORolloutStorage(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            # NP3O additions
            self.costs: torch.Tensor | None = None
            self.cost_values: torch.Tensor | None = None

        def clear(self) -> None:
            super().clear()
            self.costs = None
            self.cost_values = None

    def __init__(self, *args, **kwargs) -> None:
        print("[NP3O DEBUG] Using N3PORolloutStorage")
        super().__init__(*args, **kwargs)

        # Only meaningful for RL
        if self.training_type == "rl":
            self.costs = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
            self.cost_values = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
            self.cost_returns = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
            self.cost_advantages = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
            self.cost_dones = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device).byte()
            self.cost_violation = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
            self.cost_adv_mean = torch.zeros(1, device=self.device)
            self.cost_adv_std = torch.ones(1, device=self.device)

    def add_transitions(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

            # NP3O additions
            if transition.costs is not None:
                self.costs[self.step].copy_(transition.costs.view(-1, 1))
            if transition.cost_values is not None:
                self.cost_values[self.step].copy_(transition.cost_values.view(-1, 1))

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    # GAE for cost function, exactly like rsl_rl compute_returns
    def compute_cost_returns(
        self, last_cost_values: torch.Tensor, gamma: float, lam: float, normalize_advantage: bool = True
    ) -> None:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_cost_values if step == self.num_transitions_per_env - 1 else self.cost_values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.cost_dones[step].float()
            # TD error: r_t + gamma * Vc(s_{t+1}) - Vc(s_t)
            delta = self.costs[step] + next_is_not_terminal * gamma * next_values - self.cost_values[step]
            # Advantage: Ac(s_t, a_t) = delta_t + gamma * lambda * Ac(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: Rc_t = Ac(s_t, a_t) + Vc(s_t)
            self.cost_returns[step] = advantage + self.cost_values[step]

        # Compute the advantages
        self.cost_advantages = self.cost_returns - self.cost_values

        # Save stats of UNNORMALIZED advantages
        self.cost_adv_mean = self.cost_advantages.mean().detach()
        self.cost_adv_std = (self.cost_advantages.std() + 1e-8).detach()

        epsilon = 0.0  # epsilon is assumed 0 for NP3O in the paper

        # Normalize if requested (N-P3O uses normalized cost advantages)
        if normalize_advantage:
            self.cost_advantages = (self.cost_advantages - self.cost_adv_mean) / self.cost_adv_std
            # Per-sample violation term
            # (1-gamma)(R^c_t - epsilon), then mapped into the normalized scale like the paper does
            self.cost_violation = (
                ((1.0 - gamma) * (self.cost_returns - epsilon) + self.cost_adv_mean)
                / self.cost_adv_std
            )
        else:
            self.cost_violation = (1-gamma)*(self.cost_returns-epsilon)

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # NP3O additions
        cost_returns = self.cost_returns.flatten(0, 1)
        cost_advantages = self.cost_advantages.flatten(0, 1)
        cost_violation = self.cost_violation.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Create the mini-batch
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # NP3O additions
                cost_returns_batch = cost_returns[batch_idx]
                cost_advantages_batch = cost_advantages[batch_idx]
                cost_violation_batch = cost_violation[batch_idx]

                hidden_state_a_batch = None
                hidden_state_c_batch = None
                masks_batch = None

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                    cost_returns_batch,
                    cost_advantages_batch,
                    cost_violation_batch,
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # NP3O additions
                cost_returns_batch = self.cost_returns[:, start:stop]
                cost_advantages_batch = self.cost_advantages[:, start:stop]
                cost_violation_batch = self.cost_violation[:, start:stop]

                # Reshape to [num_envs, time, num layers, hidden dim]
                # Original shape: [time, num_layers, num_envs, hidden_dim])
                last_was_done = last_was_done.permute(1, 0)
                # Take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                hidden_state_a_batch = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_a
                ]
                hidden_state_c_batch = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_c
                ]
                # Remove the tuple for GRU
                hidden_state_a_batch = (
                    hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                )
                hidden_state_c_batch = (
                    hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch
                )

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                    cost_returns_batch,
                    cost_advantages_batch,
                    cost_violation_batch,
                )

                first_traj = last_traj
