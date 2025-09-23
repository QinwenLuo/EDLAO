# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
import math

import torch
from torch import nn
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_policy_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils.device import is_cuda_available, is_npu_available
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor import DataParallelPPOActor

if is_cuda_available:
    pass
elif is_npu_available:
    pass


__all__ = ["EDLAODataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EDLAODataParallelPPOActor(DataParallelPPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)

        if self.config.entropy_coeff != 0 or self.config.use_entropy_advantage:
            self.calculate_entropy = True
            assert not (
                    self.config.entropy_coeff != 0 and self.config.use_entropy_advantage
                ), (
                    "Cannot set entropy_coeff>0 and use_entropy_advantage=True at the same time. They are mutually exclusive."
                )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid silent error
        global_steps = data.meta_info["global_steps"]
        total_training_steps = data.meta_info["total_training_steps"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "difficulties",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = (
            ["multi_modal_inputs"] if has_multi_modal_inputs else []
        )

        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low
                        if self.config.clip_ratio_low is not None
                        else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high
                        if self.config.clip_ratio_high is not None
                        else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)

                    scale = 0.0
                    if self.calculate_entropy:
                        if self.config.entropy_coeff_annealing == "linear":
                            scale = 1 - global_steps / total_training_steps
                        elif self.config.entropy_coeff_annealing == "cosine":
                            scale = 0.5 * (
                                1 + math.cos(
                                    math.pi * global_steps / total_training_steps
                                )
                            )
                        else:
                            scale = 1.0

                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=self.calculate_entropy,
                    )

                    if self.config.use_entropy_advantage:
                        entropy_adv = torch.min(
                                self.config.entropy_advantage_alpha * entropy.detach(),
                                advantages.abs() / self.config.entropy_advantage_kappa,
                            )
                        print("advantages info:", advantages.shape, advantages.dtype, advantages.device)
                        print("entropy info:", entropy.shape, entropy.dtype, entropy.device)
                        print("entropy_adv info:", entropy_adv.shape, entropy_adv.dtype, entropy_adv.device)
                        print("difficulties info:", model_inputs["difficulties"].shape, model_inputs["difficulties"].dtype)
                        print("scale info:", type(scale), getattr(scale, "shape", None))

                        advantages += (
                            entropy_adv
                            * scale
                            * model_inputs["difficulties"].detach()
                        )
                        print("finallyadvantages info:", advantages.shape, advantages.dtype, advantages.device)

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                            compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                        )

                    else:
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                            policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                            )
                        )

                    if entropy_coeff != 0:
                        entropy_coeff = scale * entropy_coeff * data["difficulties"]
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (
                            response_mask.shape[0] / self.config.ppo_mini_batch_size
                        )
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
