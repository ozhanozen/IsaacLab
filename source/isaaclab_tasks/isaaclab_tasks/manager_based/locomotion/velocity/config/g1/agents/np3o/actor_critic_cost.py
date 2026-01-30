# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications Copyright (c) 2026, Özhan Özen

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP
from tensordict import TensorDict


class ActorCriticCost(ActorCritic):
    def __init__(self, *args, **kwargs: dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)

        # Reconstruct critic obs dim exactly like rsl_rl does
        obs: TensorDict = args[0]
        obs_groups: dict[str, list[str]] = args[1]
        critic_hidden_dims = kwargs.get("critic_hidden_dims", [256, 256, 256])
        activation = kwargs.get("activation", "elu")

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "ActorCriticCost only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        # Cost Critic with Softplus to ensure non-negative outputs
        self.cost_critic = nn.Sequential(
            MLP(num_critic_obs, 1, critic_hidden_dims, activation),
            nn.Softplus(),
        )
        print(f"Cost Critic MLP: {self.cost_critic}")

    def evaluate_cost(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Evaluate the cost given the observations.

        Uses the cost critic network and critic observations to compute the cost.
        """
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.cost_critic(obs)
