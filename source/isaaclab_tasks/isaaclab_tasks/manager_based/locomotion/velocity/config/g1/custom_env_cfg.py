# Copyright (c) 2026, Özhan Özen.
# All rights reserved.

from isaaclab.utils import configclass

from .flat_env_cfg import G1FlatEnvCfg


@configclass
class CustomG1EnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # Make it *yaw-rate*, not heading-based
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        # optional: make intent explicit
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


class CustomG1EnvCfg_PLAY(CustomG1EnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
