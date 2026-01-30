# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications Copyright (c) 2026, Özhan Özen

from __future__ import annotations

import os
import statistics
import time
import warnings
from collections import deque

import rsl_rl
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state
from tensordict import TensorDict

from .actor_critic_cost import ActorCriticCost
from .n3po import NP3O


class NP3ORunner(OnPolicyRunner):
    """On-policy runner adapted for NP3O."""
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        print("[NP3O DEBUG] Using NP3ORunner")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        self.kappa_schedule = self.cfg.get("kappa_schedule", None) if isinstance(self.cfg, dict) else None
        if self.kappa_schedule is not None:
            print("[NP3O DEBUG] Loaded kappa_schedule:", self.kappa_schedule)
        else:
            print("[NP3O DEBUG] No kappa_schedule found")

        self.entropy_schedule = self.cfg.get("entropy_schedule", None) if isinstance(self.cfg, dict) else None
        if self.entropy_schedule is not None:
            print("[NP3O DEBUG] Loaded entropy_schedule:", self.entropy_schedule)
        else:
            print("[NP3O DEBUG] No entropy_schedule found")

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
        """Construct the actor-criticc-cost and N3PO algorithm."""
        # rsl_rl base + new “class resolution”

        # Resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Make eval(...) find our classes (keep upstream style)
        globals()["ActorCriticCost"] = ActorCriticCost
        globals()["NP3O"] = NP3O

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)

        # NP3O overrides init_storage to create N3PORolloutStorage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )

        return alg

    def _maybe_update_entropy_coef(self, it: int) -> None:
        """Entropy coefficient schedule."""
        if self.entropy_schedule is None or not hasattr(self.alg, "entropy_coef"):
            return
        sched = self.entropy_schedule
        start, decay, min_entropy = float(sched["start"]), float(sched["decay"]), float(sched["min"])
        self.alg.entropy_coef = max(min_entropy, start * (decay ** it))
        if self.log_dir is not None and not self.disable_logs:
            self.writer.add_scalar("Train/entropy_coef", self.alg.entropy_coef, it)

    def _maybe_update_kappa(self, it: int) -> None:
        """Kappa schedule for progressive constraint enforcement."""
        if self.kappa_schedule is None or not hasattr(self.alg, "kappa"):
            return
        sched = self.kappa_schedule
        start, growth, kmax = float(sched["start"]), float(sched["growth"]), float(sched["max"])
        self.alg.kappa = min(kmax, start * (growth ** it))
        if self.log_dir is not None and not self.disable_logs:
            self.writer.add_scalar("Train/kappa", self.alg.kappa, it)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Mostly copy rsl_rl learn(), with cost logging + kappa schedule.

        # Initialize writer
        self._prepare_logging_writer()

        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # NP3O additions: cost logs
        costbuffer = deque(maxlen=100)
        costperstepbuffer = deque(maxlen=100)
        cur_cost_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            # Update entropy coefficient and kappa
            self._maybe_update_entropy_coef(it)
            self._maybe_update_kappa(it)

            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards

                        # NP3O: accumulate costs for episode logging
                        if isinstance(extras, dict) and "cost" in extras and torch.is_tensor(extras["cost"]):
                            cur_cost_sum += extras["cost"].to(self.device, dtype=torch.float32).view(-1)

                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

                        # NP3O additions: episode cost logging
                        ep_costs = cur_cost_sum[new_ids][:, 0]
                        ep_lens = cur_episode_length[new_ids][:, 0]
                        ep_cost_per_step = ep_costs / (ep_lens + 1e-8)
                        costbuffer.extend(ep_costs.cpu().numpy().tolist())
                        costperstepbuffer.extend(ep_cost_per_step.cpu().numpy().tolist())

                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_cost_sum[new_ids] = 0

                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())

                # NP3O: log cost stats
                if len(costbuffer) > 0:
                    mean_ep_cost = statistics.mean(costbuffer)
                    mean_cost_per_step = statistics.mean(costperstepbuffer)
                    self.writer.add_scalar("Train/mean_cost", mean_ep_cost, it)
                    self.writer.add_scalar("Train/mean_cost_per_step", mean_cost_per_step, it)

                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()
            # Save code state
            if it == start_iter and not self.disable_logs:
                # Obtain all the diff files
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                # If possible store them to wandb or neptune
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
