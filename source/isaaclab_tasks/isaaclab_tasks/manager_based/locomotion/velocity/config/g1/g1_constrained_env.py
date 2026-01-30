# Copyright (c) 2026, Özhan Özen.
# All rights reserved.

import torch

from isaaclab.envs import ManagerBasedRLEnv

from ...mdp.costs import cumulative_cost


class G1ConstrainedRLEnv(ManagerBasedRLEnv):
    def step(self, action: torch.Tensor):
        obs, rew, terminated, truncated, extras = super().step(action)

        extras["cost"] = cumulative_cost(self)

        return obs, rew, terminated, truncated, extras
