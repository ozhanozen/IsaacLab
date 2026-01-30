# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Modifications Copyright (c) 2026, Özhan Özen

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from .rollout_storage import N3PORolloutStorage


class NP3O(PPO):
    def __init__(
        self,
        *args,
        kappa: float = 1.0,
        cost_value_loss_coef: float = 1.0,
        normalize_cost_advantage: bool = False,
        normalize_cost_advantage_per_mini_batch: bool = False,
        **kwargs,
    ) -> None:
        print("[NP3O DEBUG] Using NP3O algorithm")
        super().__init__(*args, **kwargs)

        self.kappa = float(kappa)
        self.cost_value_loss_coef = float(cost_value_loss_coef)
        self.normalize_cost_advantage = bool(normalize_cost_advantage)
        self.normalize_cost_advantage_per_mini_batch = bool(normalize_cost_advantage_per_mini_batch)
        if self.normalize_cost_advantage_per_mini_batch and self.normalize_cost_advantage:
            raise ValueError(
                "Don't combine per-mini-batch cost advantage normalization with normalized violation; "
                "it breaks the constraint offset. Use one or the other."
            )

        # Use extended Transition
        self.transition = N3PORolloutStorage.Transition()

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create NP3O rollout storage
        self.storage = N3PORolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs

        # NP3O addition: cost value estimate
        if not hasattr(self.policy, "evaluate_cost"):
            raise RuntimeError("Policy must implement evaluate_cost(obs, ...) for NP3O.")
        self.transition.cost_values = self.policy.evaluate_cost(obs).detach()

        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # NP3O addition: record costs
        costs = extras.get("cost", None) if isinstance(extras, dict) else None
        if torch.is_tensor(costs):
            self.transition.costs = costs.clone().to(self.device)

            # Bootstrapping cost on time outs (mirror reward bootstrapping)
            if "time_outs" in extras:
                self.transition.costs += self.gamma * torch.squeeze(
                    self.transition.cost_values * extras["time_outs"].unsqueeze(1).to(self.device), 1
                )

            # cost_dones mirrors dones (shape handling done in storage)
            if isinstance(self.storage, N3PORolloutStorage):
                self.storage.cost_dones[self.storage.step].copy_(dones.view(-1, 1))

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

        # NP3O cost returns
        last_cost_values = self.policy.evaluate_cost(obs).detach()
        assert isinstance(self.storage, N3PORolloutStorage)
        self.storage.compute_cost_returns(
            last_cost_values,
            self.gamma,
            self.lam,
            normalize_advantage=(self.normalize_cost_advantage and (not self.normalize_cost_advantage_per_mini_batch)),
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # NP3O cost terms
        mean_cost_value_loss = 0
        mean_cost_penalty = 0
        mean_cost_clip_surr = 0
        mean_cost_adv_mean = 0
        mean_cost_adv_std = 0
        assert isinstance(self.storage, N3PORolloutStorage)

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            cost_returns_batch,
            cost_advantages_batch,
            cost_violation_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # NP3O: Check if we should normalize cost advantages per mini batch
            if self.normalize_cost_advantage_per_mini_batch:
                with torch.no_grad():
                    cost_advantages_batch = (cost_advantages_batch - cost_advantages_batch.mean()) / (
                        cost_advantages_batch.std() + 1e-8
                    )

            # Perform symmetric augmentation (copy rsl_rl, with cost repeats)
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

                # NP3O additions
                cost_returns_batch = cost_returns_batch.repeat(num_aug, 1)
                cost_advantages_batch = cost_advantages_batch.repeat(num_aug, 1)
                cost_violation_batch = cost_violation_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]


            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # NP3O: cost value loss
            value_cost_batch = self.policy.evaluate_cost(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            cost_value_loss = (value_cost_batch - cost_returns_batch).pow(2).mean() # Clipping can be added if desired
            # NP3O: clipped cost surrogate penalty (uses same ratio as actor)
            r_clip = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            c_adv = torch.squeeze(cost_advantages_batch)
            cost_surr_a = ratio * c_adv
            cost_surr_b = r_clip * c_adv

            cost_clip_per_sample = torch.max(cost_surr_a, cost_surr_b)
            viol_per_sample = cost_clip_per_sample + cost_violation_batch.squeeze(-1)
            cost_penalty = self.kappa * torch.relu(viol_per_sample).mean()

            # Simplified version if we ignore the offset
            cost_clip_surrogate = cost_clip_per_sample.mean()
            # cost_penalty = self.kappa * torch.relu(cost_clip_surrogate)

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.cost_value_loss_coef * cost_value_loss
                + cost_penalty
                - self.entropy_coef * entropy_batch.mean()
            )

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

            # NP3O cost terms
            mean_cost_value_loss += cost_value_loss.item()
            mean_cost_penalty += cost_penalty.item()
            mean_cost_clip_surr += cost_clip_surrogate.item()
            mean_cost_adv_mean += cost_advantages_batch.mean().item()
            mean_cost_adv_std += cost_advantages_batch.std().clamp_min(1e-8).item()

            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        mean_cost_value_loss /= num_updates
        mean_cost_penalty /= num_updates
        mean_cost_clip_surr /= num_updates
        mean_cost_adv_mean /= num_updates
        mean_cost_adv_std /= num_updates

        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "cost_value_loss": mean_cost_value_loss,
            "cost_penalty": mean_cost_penalty,
            "cost_clip_surrogate": mean_cost_clip_surr,
            "cost_adv_mean": mean_cost_adv_mean,
            "cost_adv_std": mean_cost_adv_std,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
