# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


def _counterpart(name: str) -> str | None:
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    if "_left_" in name:
        return name.replace("_left_", "_right_")
    if "_right_" in name:
        return name.replace("_right_", "_left_")
    return None


def _build_lr_perm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Build (and cache) left-right joint permutation using joint names."""
    perm = getattr(env, "_g1_lr_perm_min", None)
    if perm is not None:
        # Ensure it's on the right device (usually already is)
        if perm.device != env.device:
            perm = perm.to(env.device)
            env._g1_lr_perm_min = perm
        return perm

    robot = env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    idx = {n: i for i, n in enumerate(joint_names)}

    n = len(joint_names)
    perm = torch.arange(n, device=env.device)

    for i, name in enumerate(joint_names):
        other = _counterpart(name)
        if other is None:
            continue
        j = idx.get(other, None)
        if j is None:
            continue
        perm[i] = j

    env._g1_lr_perm_min = perm
    return perm


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment batch with left-right mirror (x2): original + mirrored."""
    env_u = env.unwrapped
    perm = _build_lr_perm(env_u)

    # # DEBUG sanity check
    # if not hasattr(env_u, "_g1_sym_debug_printed"):
    #     env_u._g1_sym_debug_printed = True
    #     robot = env_u.scene["robot"]
    #     joint_names = list(robot.data.joint_names)

    #     # find a few left joints to check mapping
    #     examples = [i for i, n in enumerate(joint_names) if n.startswith("left_")][:10]
    #     print("\n[G1 symmetry debug] called compute_symmetric_states")
    #     print(f"[G1 symmetry debug] num_joints={len(joint_names)} perm_device={perm.device}")
    #     for i in examples:
    #         j = int(perm[i].item())
    #         print(f"  {i:02d} {joint_names[i]}  -->  {j:02d} {joint_names[j]}")

    # observations
    if obs is not None:
        b = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:b] = obs["policy"]
        obs_aug["policy"][b:] = _mirror_policy_obs_minimal(obs["policy"], perm)
    else:
        obs_aug = None

    # actions
    if actions is not None:
        b = actions.shape[0]
        act_aug = torch.zeros(b * 2, actions.shape[1], device=actions.device)
        act_aug[:b] = actions
        act_aug[b:] = actions[:, perm]
    else:
        act_aug = None

    # # DEBUG sanity check
    # if obs is not None and actions is not None and not hasattr(env_u, "_g1_sym_debug_checked"):
    #     env_u._g1_sym_debug_checked = True

    #     # shapes should double
    #     assert obs_aug["policy"].shape[0] == 2 * obs["policy"].shape[0]
    #     assert act_aug.shape[0] == 2 * actions.shape[0]

    #     # check one sample: swapped joints should match for the mirrored half
    #     # compare original q to mirrored q after swapping
    #     pol0 = obs["policy"][:1]
    #     polm = obs_aug["policy"][obs.batch_size[0] : obs.batch_size[0] + 1]

    #     total = pol0.shape[1]
    #     n = (total - 12) // 3
    #     q0 = pol0[:, 12 : 12 + n]
    #     qm = polm[:, 12 : 12 + n]

    #     # should be equal to swapped version
    #     max_err = (qm - q0[:, perm]).abs().max().item()
    #     print(f"[G1 symmetry debug] max joint-pos swap err: {max_err:.3e}")

    return obs_aug, act_aug


def _mirror_policy_obs_minimal(pol: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """
    Assumes flat policy obs layout (no height_scan):
      base_lin_vel(3), base_ang_vel(3), projected_gravity(3), commands(3),
      joint_pos(n), joint_vel(n), last_actions(n)
    """
    out = pol.clone()
    device = out.device

    # Mirror across y: flip the "y-like" components
    out[:, 0:3] *= torch.tensor([1.0, -1.0, 1.0], device=device)      # lin vel
    out[:, 3:6] *= torch.tensor([-1.0, 1.0, -1.0], device=device)     # ang vel
    out[:, 6:9] *= torch.tensor([1.0, -1.0, 1.0], device=device)      # gravity
    out[:, 9:12] *= torch.tensor([1.0, -1.0, -1.0], device=device)    # commands [vx, vy, wz]

    total = out.shape[1]
    assert (total - 12) % 3 == 0, f"Unexpected policy obs dim: {total} (expected 12 + 3*n)"
    n = (total - 12) // 3

    # swap joints in q, dq, last_action blocks
    out[:, 12 : 12 + n] = out[:, 12 : 12 + n][:, perm]
    out[:, 12 + n : 12 + 2 * n] = out[:, 12 + n : 12 + 2 * n][:, perm]
    out[:, 12 + 2 * n : 12 + 3 * n] = out[:, 12 + 2 * n : 12 + 3 * n][:, perm]

    return out