# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from wam_diff.losses.weighted_ce import WeightedCrossEntropy


class KDLoss(nn.Module):
    def __init__(self, ignore_index: int = -100, temperature: float = 1.0, fp32_upcast: bool = True):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.fp32_upcast = fp32_upcast

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        num_batch_labels: int | None = None,
    ) -> torch.Tensor:
        """
        Calculates KL(P_teacher‖P_student) averaged over valid tokens.

        Logits are (optionally) cast to fp32 for numerical stability, probabilities
        are obtained with softmax / log_softmax after temperature scaling, and
        padding tokens (== ignore_index) are ignored in the average.

        Args:
            student_logits (torch.Tensor): The logits of the student model.
            teacher_logits (torch.Tensor): The logits of the teacher model.
            labels (torch.Tensor): The labels of the batch.
            num_batch_labels (int | None): The number of valid labels in the batch.

        Important note on num_batch_labels:
            - if `num_batch_labels` is None, it will return the mean over kl_per_token.
            - if `num_batch_labels` is not None, it will return the sum(kl_per_token) / num_batch_labels.
            Please do note that usually, num_batch_labels > #valid labels in labels tensor, for example,
            when doing gradient accumulation.

            We prefer the num_batch_labels variable over counting the number of valid labels in the batch,
            to allow for easier handling when doing gradient accumulation and per-token loss computation.

        Returns:
            The KL loss.
        """
        # Exclude padding / ignored tokens from the loss.
        valid_mask = (labels != self.ignore_index).view(-1)
        if valid_mask.sum() == 0:
            # Entire batch contains only padding - return zero to keep gradients finite.
            return student_logits.new_tensor(0.0)

        if student_logits.ndim > 2:
            student_logits = student_logits.view(-1, student_logits.shape[-1])
        if teacher_logits.ndim > 2:
            teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        if labels.ndim > 1:
            labels = labels.view(-1)
        t_logits = teacher_logits[valid_mask]
        s_logits = student_logits[valid_mask]
        labels = labels[valid_mask]

        # Up-cast logits to fp32 for numerical stability
        if self.fp32_upcast:
            t_logits = t_logits.float()
            s_logits = s_logits.float()
        #  and apply temperature scaling.
        if self.temperature != 1.0:
            t_logits.mul_(1 / self.temperature)
            s_logits.mul_(1 / self.temperature)

        # Probabilities / log-probabilities
        teacher_prob = F.softmax(t_logits, dim=-1, dtype=torch.float32)
        student_logprob = F.log_softmax(s_logits, dim=-1, dtype=torch.float32)

        # mask out infinities originating *only* from student logits
        # (teacher logits infs are extremely rare and do not
        # affect gradients w.r.t. student parameters).
        inf_mask = torch.isinf(s_logits)

        # Compute per-token forward KL contribution and flatten.
        kl_per_token = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0).sum(-1).view(-1)

        # Average over valid tokens.
        if num_batch_labels is not None:
            return -torch.sum(kl_per_token) / num_batch_labels
        else:
            return -torch.mean(kl_per_token)


class BlockKDLoss(nn.Module):
    """KD loss for block diffusion training.

    Extends KDLoss to support:
    - loss_mask: only compute KL on masked positions (same as WeightedCrossEntropy)
    - t weighting: loss_weight = 1/t (noise level weighting, lower t → higher weight)
    - Macro-averaging across samples (same as WeightedCrossEntropy)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = temperature
        self.fp32_upcast = fp32_upcast
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        response_mask: torch.Tensor | None = None,
        num_label_tokens: int | None = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Calculate KL(P_teacher || P_student) on masked positions.

        Args:
            student_logits: [B, L, V] student model logits.
            teacher_logits: [B, L, V] teacher model logits.
            labels: [B, L] ground truth labels.
            loss_mask: [B, L] binary mask, 1 on positions to compute loss.
            t: [B, 1] or [1] noise level for 1/t weighting.
            response_mask: [B, L] binary mask for response positions.
            num_label_tokens: total label tokens across all ranks (for gradient accumulation).
            num_samples: number of samples in the batch (for macro averaging).
        """
        b, l = labels.shape

        # Up-cast to fp32
        if self.fp32_upcast:
            student_logits = student_logits.float()
            teacher_logits = teacher_logits.float()

        # Temperature scaling
        if self.temperature != 1.0:
            teacher_logits = teacher_logits / self.temperature
            student_logits = student_logits / self.temperature

        # Compute per-token KL
        teacher_prob = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)

        # Mask out infinities from student logits
        inf_mask = torch.isinf(student_logits)
        kl_per_token = torch.masked_fill(teacher_prob * student_logprob, inf_mask, 0).sum(-1)  # [B, L]

        # Negate (KL divergence)
        kl_per_token = -kl_per_token

        # Apply 1/t weighting (same as WeightedCrossEntropy)
        if t is not None:
            loss_weight = 1.0 / t  # [B, 1] or [1]
            kl_per_token = kl_per_token * loss_weight

        # Apply loss_mask: only compute on masked positions
        if loss_mask is not None:
            kl_per_token = kl_per_token * loss_mask.float()
        else:
            # Fallback: use labels != ignore_index
            kl_per_token = kl_per_token * (labels != self.ignore_index).float()

        # Macro-averaging across samples (same logic as WeightedCrossEntropy.compute_macro_average_loss)
        if response_mask is not None and num_samples > 0:
            # Use macro averaging to match CE loss
            kl_per_token_for_avg = kl_per_token * response_mask.float()
            # Build sample IDs from response_mask
            mask_bool = response_mask.bool()
            inner_transitions = mask_bool[:, 1:] & (~mask_bool[:, :-1])
            first_col_starts = mask_bool[:, 0:1]
            is_start = torch.cat([first_col_starts, inner_transitions], dim=1)
            flat_ids = is_start.view(-1).cumsum(dim=0)

            flat_kl = kl_per_token_for_avg.reshape(-1)
            flat_mask = mask_bool.reshape(-1).float()

            max_id = flat_ids.max().item()
            if max_id < 0:
                return student_logits.new_tensor(0.0, requires_grad=True)

            num_bins = int(max_id + 1)
            kl_sum_per_sample = torch.zeros(num_bins, device=kl_per_token.device)
            valid_len_per_sample = torch.zeros(num_bins, device=kl_per_token.device)

            kl_sum_per_sample.scatter_add_(0, flat_ids, flat_kl)
            valid_len_per_sample.scatter_add_(0, flat_ids, flat_mask)

            valid_samples_mask = valid_len_per_sample > 0
            if valid_samples_mask.sum() == 0:
                return student_logits.new_tensor(0.0, requires_grad=True)

            kl_mean_per_sample = kl_sum_per_sample / (valid_len_per_sample + 1e-9)
            return kl_mean_per_sample[valid_samples_mask].sum() / num_samples

        elif num_label_tokens is not None:
            return kl_per_token.sum() / num_label_tokens
        else:
            valid = (labels != self.ignore_index).float()
            return kl_per_token.sum() / max(valid.sum(), 1)


class DistilKDLoss(nn.Module):
    """Reverse KL(P_student || P_teacher) for distillation.

    Key differences from BlockKDLoss:
    - Uses reverse KL (student || teacher) instead of forward KL (teacher || student)
    - Only computes on loss_mask positions (supervised tokens)
    - Teacher and student share the same x_t (guaranteed externally)
    - Supports 1/t weighting and macro-averaging (same as WeightedCrossEntropy)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.fp32_upcast = fp32_upcast

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        loss_mask: torch.Tensor,
        t: torch.Tensor | None = None,
        response_mask: torch.Tensor | None = None,
        num_label_tokens: int | None = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Calculate KL(P_student || P_teacher) on supervised (loss_mask) positions.

        Args:
            student_logits: [B, L, V] student model logits.
            teacher_logits: [B, L, V] teacher model logits (same x_t as student).
            loss_mask: [B, L] binary mask, 1 on supervised positions to compute loss.
            t: [B, 1] or [1] noise level for 1/t weighting.
            response_mask: [B, L] binary mask for response positions (macro averaging).
            num_label_tokens: total label tokens across all ranks (for gradient accumulation).
            num_samples: number of samples in the batch (for macro averaging).
        """
        # Up-cast to fp32
        if self.fp32_upcast:
            student_logits = student_logits.float()
            teacher_logits = teacher_logits.float()

        # Temperature scaling
        if self.temperature != 1.0:
            student_logits = student_logits / self.temperature
            teacher_logits = teacher_logits / self.temperature

        # Reverse KL: KL(P_student || P_teacher)
        # = sum P_student * (log P_student - log P_teacher)
        student_logprob = F.log_softmax(student_logits, dim=-1, dtype=torch.float32)
        student_prob = F.softmax(student_logits, dim=-1, dtype=torch.float32)
        teacher_logprob = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)

        # Mask out infinities from student logits
        inf_mask = torch.isinf(student_logits) | torch.isinf(teacher_logits)
        kl_per_token = torch.masked_fill(student_prob * (student_logprob - teacher_logprob), inf_mask, 0).sum(-1)  # [B, L]

        # Apply 1/t weighting (same as WeightedCrossEntropy)
        if t is not None:
            loss_weight = 1.0 / t  # [B, 1] or [1]
            kl_per_token = kl_per_token * loss_weight

        # Only compute on supervised (loss_mask) positions
        kl_per_token = kl_per_token * loss_mask.float()

        # Macro-averaging across samples (same logic as WeightedCrossEntropy)
        if response_mask is not None and num_samples > 0:
            kl_per_token_for_avg = kl_per_token * response_mask.float()
            mask_bool = response_mask.bool()
            inner_transitions = mask_bool[:, 1:] & (~mask_bool[:, :-1])
            first_col_starts = mask_bool[:, 0:1]
            is_start = torch.cat([first_col_starts, inner_transitions], dim=1)
            flat_ids = is_start.view(-1).cumsum(dim=0)

            flat_kl = kl_per_token_for_avg.reshape(-1)
            flat_mask = mask_bool.reshape(-1).float()

            max_id = flat_ids.max().item()
            if max_id < 0:
                return student_logits.new_tensor(0.0, requires_grad=True)

            num_bins = int(max_id + 1)
            kl_sum_per_sample = torch.zeros(num_bins, device=kl_per_token.device)
            valid_len_per_sample = torch.zeros(num_bins, device=kl_per_token.device)

            kl_sum_per_sample.scatter_add_(0, flat_ids, flat_kl)
            valid_len_per_sample.scatter_add_(0, flat_ids, flat_mask)

            valid_samples_mask = valid_len_per_sample > 0
            if valid_samples_mask.sum() == 0:
                return student_logits.new_tensor(0.0, requires_grad=True)

            kl_mean_per_sample = kl_sum_per_sample / (valid_len_per_sample + 1e-9)
            return kl_mean_per_sample[valid_samples_mask].sum() / num_samples

        elif num_label_tokens is not None:
            return kl_per_token.sum() / num_label_tokens
        else:
            return kl_per_token.sum() / max(loss_mask.sum(), 1)


class Block_JSDLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        beta: float = 0.5,
        fp32_upcast: bool = True,
        eps: float = 1e-8,
        detach_teacher: bool = True,
    ):
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        self.temperature = temperature
        self.beta = beta
        self.fp32_upcast = fp32_upcast
        self.eps = eps
        self.detach_teacher = detach_teacher

    @staticmethod
    def compute_macro_average_loss(per_token_loss, loss_mask, response_mask):
        """计算 Sequence Packing 下的 Macro-Average Loss
        利用response_mask的连续性, 通过检测 0->1 的跳变来区分不同的sample片段。

        Args:
            per_token_loss: [B, L] 或 [N], reduction='none'的loss
            loss_mask: [B, L] 或 [N], 0/1 或 Bool, 1 表示有效的计算loss token
            response_mask:    [B, L] 或 [N], 0/1 或 Bool, 1 表示有效的response token
        Returns:
            final_loss: Scalar
        """
        if per_token_loss.dim() == 1:
            per_token_loss = per_token_loss.unsqueeze(0)
            response_mask = response_mask.unsqueeze(0)

        B, L = per_token_loss.shape
        device = per_token_loss.device
        mask_bool = response_mask.bool()

        # =========================================================
        # 1. 自动生成 Sample IDs
        # =========================================================
        # 我们需要识别每个连续 '1' 片段的起点。
        # 起点定义为：当前是 1，且前一个是 0 (上升沿)；或者是一行的开头且为 1。

        # 1.1 计算同一行内的跳变 (当前位 > 前一位) => 0->1 变 True, 1->0 变 False
        # shape: [B, L-1]
        inner_transitions = mask_bool[:, 1:] & (~mask_bool[:, :-1])

        # 1.2 处理每行的第一列
        # 如果第一列是1，它也是一个片段的开始, shape: [B, 1]
        first_col_starts = mask_bool[:, 0:1]

        # 1.3 拼接得到完整的“片段起点”掩码, shape: [B, L]
        # is_start[b, l]为True 表示这里是一个新sample的开始
        is_start = torch.cat([first_col_starts, inner_transitions], dim=1)

        # 1.4 生成全局唯一ID
        # 展平后计算累加和, 每遇到一个True(新起点)，ID就+1。
        # 这样同一个连续片段内的ID会保持不变 (因为中间全是False, cumsum不增加)。
        # padding(0)部分的ID会跟随上一个片段，但会在后续被mask过滤掉，不影响。
        flat_ids = is_start.view(-1).cumsum(dim=0)

        # 2. 准备数据
        flat_loss = per_token_loss.reshape(-1)
        flat_mask = mask_bool.reshape(-1).float()
        flat_loss_mask = loss_mask.reshape(-1).bool()

        # 3. 确保loss只计算mask=True的部分
        # (虽然padding部分的ID可能重复，但乘上mask后变为 0，不会产生贡献)
        flat_loss = flat_loss * flat_mask
        flat_loss[~flat_loss_mask] = 0

        # 4. 准备scatter容器
        max_id = flat_ids.max().item()
        # 防止全 0 mask 导致 max_id 为负数
        if max_id < 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        num_bins = int(max_id + 1)
        loss_sum_per_sample = torch.zeros(num_bins, device=device)
        valid_len_per_sample = torch.zeros(num_bins, device=device)

        # 5. 分组累加
        loss_sum_per_sample.scatter_add_(0, flat_ids, flat_loss)
        valid_len_per_sample.scatter_add_(0, flat_ids, flat_mask)

        # 6. 计算每个子序列的平均loss, 只有长度>0的才是有效样本
        valid_samples_mask = valid_len_per_sample > 0

        # 计算每个样本内部的mean loss
        loss_mean_per_sample = loss_sum_per_sample / (valid_len_per_sample + 1e-9)

        # 7.取所有有效样本的均值(Macro Average)
        # 如果整个batch都是padding，valid_samples_mask全False，需要处理
        if valid_samples_mask.sum() == 0:
             return torch.tensor(0.0, device=device, requires_grad=True)

        # final_loss = loss_mean_per_sample[valid_samples_mask].mean()
        final_loss = loss_mean_per_sample[valid_samples_mask].sum()

        return final_loss

    def forward(
        self,
        student_logits: torch.Tensor,   # [B, L, V]
        teacher_logits: torch.Tensor,   # [B, L, V]
        loss_mask: torch.Tensor,        # [B, L]
        response_mask: torch.Tensor,    # [B, L]
        num_samples: int = 1,
    ) -> torch.Tensor:
        if self.fp32_upcast:
            student_logits = student_logits.float()
            teacher_logits = teacher_logits.float()

        loss_mask = loss_mask.bool()
        response_mask = response_mask.bool()

        tau = self.temperature
        beta = self.beta
        detach_teacher = self.detach_teacher

        # Chunk along L to reduce peak memory: ~6 * [B,chunk,V] instead of ~6 * [B,L,V]
        CHUNK = 512
        B, L, V = student_logits.shape
        jsd_per_token = student_logits.new_zeros(B, L)

        for start in range(0, L, CHUNK):
            end = min(start + CHUNK, L)
            s_logits = student_logits[:, start:end]
            t_logits = teacher_logits[:, start:end]

            s_log_prob = F.log_softmax(s_logits / tau, dim=-1)
            t_log_prob = F.log_softmax(t_logits / tau, dim=-1)

            s_prob = s_log_prob.exp()
            t_prob = t_log_prob.exp()

            if detach_teacher:
                t_prob = t_prob.detach()
                t_log_prob = t_log_prob.detach()

            mix_prob = beta * t_prob + (1.0 - beta) * s_prob
            mix_log_prob = torch.log(mix_prob.clamp_min(self.eps))

            t_to_mix = (t_prob * (t_log_prob - mix_log_prob)).sum(dim=-1)   # [B, chunk]
            s_to_mix = (s_prob * (s_log_prob - mix_log_prob)).sum(dim=-1)   # [B, chunk]

            jsd_per_token[:, start:end] = beta * t_to_mix + (1.0 - beta) * s_to_mix

        jsd_per_token = jsd_per_token * loss_mask.float()

        # 和 WeightedCrossEntropy(reduction="mean") 对齐：
        # sample 内平均，再 sample 间平均
        loss = self.compute_macro_average_loss(
            per_token_loss=jsd_per_token,
            loss_mask=loss_mask,
            response_mask=response_mask,
        ) / max(int(num_samples), 1)

        return loss * (tau ** 2)


class ReverseKDLoss(nn.Module):
    """Reverse-KL KD loss, reduction aligned with WeightedCrossEntropy.

    Computes:

        KL(P_student || P_teacher)

    on supervised positions only.

    Reduction:
        sample 内:
            sum(loss_mask 位置的 reverse-KL) / response_mask 长度

        batch 内:
            sum(sample loss) / num_samples

    This matches WeightedCrossEntropy(reduction="mean").
    """

    def __init__(
        self,
        temperature: float = 1.0,
        fp32_upcast: bool = True,
        detach_teacher: bool = True,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.temperature = temperature
        self.fp32_upcast = fp32_upcast
        self.detach_teacher = detach_teacher

    def forward(
        self,
        student_logits: torch.Tensor,   # [B, L, V]
        teacher_logits: torch.Tensor,   # [B, L, V]
        loss_mask: torch.Tensor,        # [B, L]
        response_mask: torch.Tensor,    # [B, L]
        num_samples: int = 1,
    ) -> torch.Tensor:
        if self.fp32_upcast:
            student_logits = student_logits.float()
            teacher_logits = teacher_logits.float()

        loss_mask = loss_mask.bool()
        response_mask = response_mask.bool()

        tau = self.temperature

        student_log_prob = F.log_softmax(student_logits / tau, dim=-1)
        teacher_log_prob = F.log_softmax(teacher_logits / tau, dim=-1)

        if self.detach_teacher:
            teacher_log_prob = teacher_log_prob.detach()

        student_prob = student_log_prob.exp()

        # [B, L, V] -> [B, L]
        per_token_rkl = (
            student_prob * (student_log_prob - teacher_log_prob)
        ).sum(dim=-1)

        # 只在 supervised positions 上计算 KD
        per_token_rkl = per_token_rkl * loss_mask.float()

        # 对齐 WeightedCrossEntropy(reduction="mean")
        loss = WeightedCrossEntropy.compute_macro_average_loss(
            per_token_loss=per_token_rkl,
            loss_mask=loss_mask,
            response_mask=response_mask,
        ) / max(int(num_samples), 1)

        return loss * (tau ** 2)
