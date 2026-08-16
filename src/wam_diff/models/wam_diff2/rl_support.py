# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class WAMDiff2RLPreparedBatch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    block_metadata: dict[str, torch.Tensor]
    logits_to_keep: torch.Tensor


@dataclass
class WAMDiff2RLComputationOutput:
    loss: torch.Tensor
    entropy: torch.Tensor
    kl_loss: torch.Tensor


class WAMDiff2RLAdapter:
    def __init__(self, config) -> None:
        self.config = config
        self.block_size = int(getattr(config, "block_size", 4))
        self.mask_token_id = int(getattr(config, "mask_token_id", 151671))
        text_config = getattr(config, "text_config", None)
        pad_token_id = getattr(text_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(text_config, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(config, "eos_token_id", 0)
        self.pad_token_id = int(pad_token_id)

    def prepare_batch(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        masked_indices: torch.Tensor,
        step_ids: Optional[torch.Tensor] = None,
    ) -> WAMDiff2RLPreparedBatch:
        batch_size, _ = input_ids.shape
        device = input_ids.device
        sample_results = []
        max_len = 0

        for row in range(batch_size):
            packed = self._pack_one_sample(
                input_ids=input_ids[row],
                position_ids=None if position_ids is None else position_ids[..., row, :] if position_ids.ndim == 3 else position_ids[row],
                labels=None if labels is None else labels[row],
                masked_indices=masked_indices[row],
                step_ids=None if step_ids is None else step_ids[row],
                device=device,
            )
            sample_results.append(packed)
            max_len = max(max_len, packed["input_ids"].shape[-1])

        if max_len == 0:
            max_len = 1

        packed_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=input_ids.dtype, device=device)
        packed_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        packed_logits_to_keep = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        sample_ids = torch.full((batch_size, max_len), -1, dtype=torch.int32, device=device)
        token_types = torch.full((batch_size, max_len), -1, dtype=torch.int8, device=device)
        block_ids = torch.full((batch_size, max_len), -1, dtype=torch.int32, device=device)

        packed_position_ids = None
        if position_ids is not None:
            if position_ids.ndim == 3:
                packed_position_ids = torch.zeros((3, batch_size, max_len), dtype=position_ids.dtype, device=device)
            else:
                packed_position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=device)

        for row, packed in enumerate(sample_results):
            packed_len = packed["input_ids"].shape[-1]
            packed_input_ids[row, :packed_len] = packed["input_ids"]
            packed_attention_mask[row, :packed_len] = 1
            packed_logits_to_keep[row, :packed_len] = packed["logits_to_keep"]
            sample_ids[row, :packed_len] = 0
            token_types[row, :packed_len] = packed["token_types"]
            block_ids[row, :packed_len] = packed["block_ids"]
            if packed_position_ids is not None:
                if packed_position_ids.ndim == 3:
                    packed_position_ids[:, row, :packed_len] = packed["position_ids"]
                else:
                    packed_position_ids[row, :packed_len] = packed["position_ids"]

        if packed_position_ids is None:
            packed_position_ids = torch.arange(max_len, device=device).long().unsqueeze(0).expand(batch_size, -1)

        return WAMDiff2RLPreparedBatch(
            input_ids=packed_input_ids,
            position_ids=packed_position_ids,
            attention_mask=packed_attention_mask,
            block_metadata={
                "sample_ids": sample_ids,
                "token_types": token_types,
                "block_ids": block_ids,
            },
            logits_to_keep=packed_logits_to_keep,
        )

    def compute_rl_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        masked_indices: torch.Tensor,
        p_mask: torch.Tensor,
        adv: torch.Tensor,
        is_real: torch.Tensor,
        logp_old_tok: Optional[torch.Tensor] = None,
        logp_ref_tok: Optional[torch.Tensor] = None,
        ppo_eps: float = 0.2,
        kl_beta: float = 0.0,
        use_kl_estimator_k3: bool = True,
        return_entropy: bool = False,
        loss_mean: bool = True,
    ) -> WAMDiff2RLComputationOutput:
        device = logits.device
        zero = logits.new_zeros(())

        labels_masked = labels[masked_indices]
        if labels_masked.numel() == 0:
            return WAMDiff2RLComputationOutput(loss=zero, entropy=zero, kl_loss=zero)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        logp_masked = log_probs.gather(dim=-1, index=labels_masked.unsqueeze(-1)).squeeze(-1)

        is_real_tensor = is_real.to(device=device, dtype=torch.bool)
        p_mask_real = p_mask.to(device=device, dtype=torch.bool) & is_real_tensor.unsqueeze(1)
        p_to_keep_real = p_mask_real[masked_indices]

        if p_to_keep_real.numel() == 0 or not torch.any(p_to_keep_real):
            return WAMDiff2RLComputationOutput(loss=zero, entropy=zero, kl_loss=zero)

        logp_p = logp_masked[p_to_keep_real]

        entropy = zero
        if return_entropy:
            entropy_p = -(log_probs[p_to_keep_real].exp() * log_probs[p_to_keep_real]).sum(dim=-1)
            entropy = entropy_p.mean() if entropy_p.numel() > 0 else zero

        adv_tensor = adv.to(device=device, dtype=torch.float32)
        adv_p = adv_tensor.unsqueeze(1).expand_as(p_mask)[masked_indices][p_to_keep_real]

        if logp_old_tok is not None and logp_old_tok.numel() > 0:
            logp_old_p = logp_old_tok.to(device=device)[masked_indices][p_to_keep_real]
        else:
            logp_old_p = logp_p.detach()

        ratio_p = (logp_p - logp_old_p).clamp(-10.0, 10.0).exp()
        clipped_ratio = ratio_p.clamp(1 - ppo_eps, 1 + ppo_eps + 0.08)
        surrogate_p = torch.minimum(ratio_p * adv_p, clipped_ratio * adv_p)
        policy_loss = -surrogate_p.mean() if loss_mean else -surrogate_p.sum()

        kl_loss = zero
        if kl_beta > 0 and logp_ref_tok is not None and logp_ref_tok.numel() > 0:
            logp_ref_p = logp_ref_tok.to(device=device)[masked_indices][p_to_keep_real]
            kl_seq_p = logp_p - logp_ref_p
            if use_kl_estimator_k3:
                kl_seq_p = (-kl_seq_p).clamp(-10.0, 10.0).exp() - 1.0 + kl_seq_p
            kl_loss = kl_beta * (kl_seq_p.mean() if loss_mean else kl_seq_p.sum())

        return WAMDiff2RLComputationOutput(
            loss=policy_loss + kl_loss,
            entropy=entropy,
            kl_loss=kl_loss.detach(),
        )

    def _pack_one_sample(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        masked_indices: torch.Tensor,
        step_ids: Optional[torch.Tensor],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        active_len = self._infer_active_len(input_ids=input_ids, labels=labels, step_ids=step_ids)
        if active_len <= 0:
            active_len = 1

        response_mask = self._infer_response_mask(
            labels=None if labels is None else labels[:active_len],
            step_ids=None if step_ids is None else step_ids[:active_len],
            length=active_len,
            device=device,
        )
        response_positions = torch.nonzero(response_mask, as_tuple=False).flatten()
        prompt_len = int(response_positions[0].item()) if response_positions.numel() > 0 else active_len

        active_input_ids = input_ids[:active_len]
        active_masked_indices = masked_indices[:active_len]
        active_position_ids = self._slice_position_ids(position_ids, active_len, device)

        prefix_ids = active_input_ids[:prompt_len]
        response_ids = active_input_ids[prompt_len:active_len]
        masked_response = active_masked_indices[prompt_len:active_len]

        if response_ids.numel() > 0:
            noisy_response_ids = response_ids.clone()
            noisy_response_ids[masked_response] = self.mask_token_id
            packed_input_ids = torch.cat([prefix_ids, response_ids, noisy_response_ids], dim=0)

            prefix_types = torch.zeros(prefix_ids.shape[0], dtype=torch.int8, device=device)
            clean_types = torch.ones(response_ids.shape[0], dtype=torch.int8, device=device)
            noisy_types = torch.full((response_ids.shape[0],), 2, dtype=torch.int8, device=device)
            token_types = torch.cat([prefix_types, clean_types, noisy_types], dim=0)

            num_prefix_blocks = (prompt_len + self.block_size - 1) // self.block_size
            prefix_block_ids = torch.arange(prefix_ids.shape[0], device=device, dtype=torch.int32) // self.block_size if prefix_ids.numel() > 0 else torch.empty((0,), dtype=torch.int32, device=device)
            response_blocks = num_prefix_blocks + (torch.arange(response_ids.shape[0], device=device, dtype=torch.int32) // self.block_size)
            block_ids = torch.cat([prefix_block_ids, response_blocks, response_blocks], dim=0)

            if active_position_ids.ndim == 2:
                packed_position_ids = torch.cat(
                    [
                        active_position_ids[:, :prompt_len],
                        active_position_ids[:, prompt_len:active_len],
                        active_position_ids[:, prompt_len:active_len],
                    ],
                    dim=-1,
                )
            else:
                packed_position_ids = torch.cat(
                    [
                        active_position_ids[:prompt_len],
                        active_position_ids[prompt_len:active_len],
                        active_position_ids[prompt_len:active_len],
                    ],
                    dim=-1,
                )

            logits_to_keep = torch.zeros(packed_input_ids.shape[0], dtype=torch.bool, device=device)
            noisy_start = prompt_len + response_ids.shape[0]
            logits_to_keep[noisy_start:noisy_start + response_ids.shape[0]] = masked_response
        else:
            packed_input_ids = active_input_ids
            token_types = torch.zeros(packed_input_ids.shape[0], dtype=torch.int8, device=device)
            block_ids = torch.arange(packed_input_ids.shape[0], device=device, dtype=torch.int32) // self.block_size
            packed_position_ids = active_position_ids
            logits_to_keep = torch.zeros(packed_input_ids.shape[0], dtype=torch.bool, device=device)

        return {
            "input_ids": packed_input_ids,
            "position_ids": packed_position_ids,
            "token_types": token_types,
            "block_ids": block_ids,
            "logits_to_keep": logits_to_keep,
        }

    def _infer_active_len(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor],
        step_ids: Optional[torch.Tensor],
    ) -> int:
        non_pad = torch.nonzero(input_ids != self.pad_token_id, as_tuple=False).flatten()
        active_len = int(non_pad[-1].item()) + 1 if non_pad.numel() > 0 else 0

        if labels is not None:
            response_positions = torch.nonzero(labels != -100, as_tuple=False).flatten()
            if response_positions.numel() > 0:
                active_len = max(active_len, int(response_positions[-1].item()) + 1)

        if step_ids is not None:
            step_positions = torch.nonzero(step_ids > 0, as_tuple=False).flatten()
            if step_positions.numel() > 0:
                active_len = max(active_len, int(step_positions[-1].item()) + 1)

        return active_len

    def _infer_response_mask(
        self,
        labels: Optional[torch.Tensor],
        step_ids: Optional[torch.Tensor],
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if step_ids is not None and torch.any(step_ids > 0):
            return step_ids > 0
        if labels is not None:
            return labels != -100
        return torch.zeros(length, dtype=torch.bool, device=device)

    def _slice_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        active_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if position_ids is None:
            return torch.arange(active_len, device=device).long()
        if position_ids.ndim == 2:
            return position_ids[:, :active_len]
        return position_ids[:active_len]
