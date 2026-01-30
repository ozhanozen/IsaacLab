# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications Copyright (c) 2026, Özhan Özen

from typing import Mapping, Union, Optional
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.symmetry import g1


@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]



@configclass
class CustomG1CPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "g1_ppo"

        self.obs_groups = {
            "policy": ["policy"],
            "critic": ["policy"],
        }

        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.008,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg = RslRlSymmetryCfg(
                use_data_augmentation=True,
                data_augmentation_func=g1.compute_symmetric_states,
            )
        )


@configclass
class CustomG1NP3ORunnerCfg(CustomG1CPPORunnerCfg):
    kappa_schedule: Optional[Mapping[str, Union[str, float]]] = None
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "g1_np3o"

        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="NP3O",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.008,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg = RslRlSymmetryCfg(
                use_data_augmentation=True,
                data_augmentation_func=g1.compute_symmetric_states,
            ),
        )
        setattr(self.algorithm, "kappa", 0.1)
        setattr(self.algorithm, "cost_value_loss_coef", 1.0)
        setattr(self.algorithm, "normalize_cost_advantage", True)
        setattr(self.algorithm, "normalize_cost_advantage_per_mini_batch", False)

        self.policy = RslRlPpoActorCriticCfg(
            class_name="ActorCriticCost",
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )

        # Kappa scheduling (progressive constraint enforcement)
        self.kappa_schedule = {
            "start": 0.1,
            "growth": 1.0004,  # > 1
            "max": 0.2,
        }

        # Entropy coefficient scheduling
        self.entropy_schedule = {
            "start": 0.008,
            "decay": 0.999,  # < 1
            "min": 0.005,
        }
