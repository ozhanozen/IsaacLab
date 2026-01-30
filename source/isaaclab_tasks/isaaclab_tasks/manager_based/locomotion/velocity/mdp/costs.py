# Copyright (c) 2026, Özhan Özen.
# All rights reserved.

from __future__ import annotations

import torch

from isaaclab.envs.mdp.rewards import flat_orientation_l2, joint_pos_limits
from isaaclab.envs.mdp.terminations import illegal_contact
from isaaclab.managers import SceneEntityCfg


def cumulative_cost(env):
    c = torch.zeros(env.num_envs, device=env.device)

    # 1) torso/base contact as binary cost
    c += illegal_contact(
        env,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names="torso_link"),
        threshold=1.0,
    ).float()

    # 2) joint limits (continuous)
    c += joint_pos_limits(env, asset_cfg=SceneEntityCfg("robot", joint_names=[".*_ankle_.*"]))

    # 3) tilt (continuous)
    c += flat_orientation_l2(env)

    assert c.ndim == 1 and c.shape[0] == env.num_envs

    return c
