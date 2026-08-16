# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from .configuration_wam_diff2 import WAMDiff2Config, WAMDiff2TextConfig, WAMDiff2VisionConfig
from .rl_support import WAMDiff2RLAdapter

from torch.nn.attention.flex_attention import flex_attention, create_block_mask


# help class for uniform sampling.
@dataclass
class SchedulerOutput:
    r"""Represents a sample of a conditional-flow generated probability path.

    """
    alpha_t: torch.Tensor = field(metadata={"help": "alpha_t"})
    d_alpha_t: torch.Tensor = field(metadata={"help": "Derivative of alpha_t."})
    sigma_t: Optional[torch.Tensor] = field(default=None, metadata={"help": "sigma_t"})
    d_sigma_t: Optional[torch.Tensor] = field(default=None, metadata={"help": "Derivative of sigma_t."})

class CondOTScheduler:
    """CondOT Scheduler."""
    def __call__(self, t: torch.Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

@dataclass
class DiscretePathSample:
    """
    Represents a sample of a conditional-flow generated discrete probability path.

    Attributes:
        x_1 (Tensor): the target sample :math:`X_1`.
        x_0 (Tensor): the source sample :math:`X_0`.
        t (Tensor): the time sample  :math:`t`.
        x_t (Tensor): the sample along the path  :math:`X_t ~ p_t`.
    """
    x_1: torch.Tensor = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: torch.Tensor = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: torch.Tensor = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: torch.Tensor = field(metadata={"help": "samples X_t ~ p_t(X_t), shape (batch_size, ...)."})

def expand_tensor_like(input_tensor: torch.Tensor, expand_to: torch.Tensor):
    assert input_tensor.ndim == 1, "Input tensor must be a 1d vector."
    assert (
        input_tensor.shape[0] == expand_to.shape[0]
    ), f"The first (batch_size) dimension must match. Got shape {input_tensor.shape} and {expand_to.shape}."

    dim_diff = expand_to.ndim - input_tensor.ndim

    t_expanded = input_tensor.clone()
    t_expanded = t_expanded.reshape(-1, *([1] * dim_diff))

    return t_expanded.expand_as(expand_to)

def unsqueeze_to_match(source: torch.Tensor, target: torch.Tensor, how: str = "suffix") -> torch.Tensor:
    return source

class MixtureDiscreteProbPath:
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, determined by the scheduler, when it flips to the target data point :math:`X_1`.
    The scheduler determines the flip probability using the parameter :math:`\sigma_t`, which is a function of time `t`. Specifically, :math:`\sigma_t` represents the probability of remaining at :math:`X_0`, while :math:`1 - \sigma_t` is the probability of flipping to :math:`X_1`:

    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\sigma_t`.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def sample(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> DiscretePathSample:
        r"""Sample from the affine probability path:
            | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
            | return :math:`X_0, X_1, t`, and :math:`X_t \sim p_t`.
        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            DiscretePathSample: a conditional sample at :math:`X_t ~ p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t
        if sigma_t.ndim == 1:
            sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1) # [B, L]

        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t # [B, L]
        x_t = torch.where(source_indices, x_0, x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def posterior_to_velocity(self, posterior_logits: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Convert the factorized posterior to velocity.

        | given :math:`p(X_1|X_t)`. In the factorized case: :math:`\prod_i p(X_1^i | X_t)`.
        | return :math:`u_t`.

        Args:
            posterior_logits (Tensor): logits of the x_1 posterior conditional on x_t, shape (..., vocab size).
            x_t (Tensor): path sample at time t, shape (...).
            t (Tensor): time in [0, 1].

        Returns:
            Tensor: velocity.
        """
        posterior = torch.softmax(posterior_logits, dim=-1) # [b, l, vocab_size]
        vocab_size = posterior.shape[-1]
        x_t = F.one_hot(x_t, num_classes=vocab_size) # [b, l]
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)
        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)

def categorical(probs: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs.flatten(0, -2), num_samples=1, replacement=True).view(*probs.shape[:-1])

# 下面4个函数都是为了sequence packing设计的，但目前梯度反传不work
def flex_block_diffusion_forward_full(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    metadata = kwargs.get("block_metadata")

    # 回退逻辑: 如果metadata为空, 调用原生flash_attention
    if metadata is None:
        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        if attention_interface is None:
            attention_interface = ALL_ATTENTION_FUNCTIONS["eager"]

        attn_output = attention_interface(module, query, key, value, attention_mask, is_causal=False, dropout=dropout, scaling=scaling, **kwargs)
        return attn_output, None

    # block diffusion logic
    sample_ids = metadata["sample_ids"]
    token_types = metadata["token_types"]
    block_ids = metadata["block_ids"]

    def block_diffusion_score_mod(b, h, q_idx, k_idx):
        sq, sk = sample_ids[b, q_idx], sample_ids[b, k_idx]
        tq, tk = token_types[b, q_idx], token_types[b, k_idx]
        bq, bk = block_ids[b, q_idx], block_ids[b, k_idx]

        # 1. 样本隔离
        is_visible = (sq == sk) & (sq != -1)

        # 2. 区域mask隔离
        # 2.1 query是prefix
        # block causal, 只能看到 prefix中block_id <= 自己的block.
        mask_prefix = (tq == 0) & (tk == 0) & (bk <= bq)

        # 2.2 query 是 clean response
        # 可以看到全部的 prefix
        # Block Causal, 可以看到 clean response的前i个blocks.
        mask_clean = (tq == 1) & (
            (tk == 0) |
            ((tk == 1) & (bk <= bq))
        )

        # 2.3 query 是 noisy response
        # 可以看到全部的 prefix.
        # 可以看到 clean response 的前 i-1 个 block.
        # 可以看到自身 noisy block.
        mask_noisy = (tq == 2) & (
            (tk == 0) |
            ((tk == 1) & (bk < bq)) |
            ((tk == 2) & (bk == bq))
        )
        # 只有满足上述任一场景且满足样本隔离，才允许 Attention

        return is_visible & (mask_prefix | mask_clean | mask_noisy)

    query_len = query.shape[-2]
    key_len = key.shape[-2]
    block_mask = create_block_mask(
        block_diffusion_score_mod,
        B=query.shape[0],
        H=query.shape[1],
        Q_LEN=query_len,
        KV_LEN=key_len,
        device=query.device,
        _compile=True
    )

    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    attn_output = flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=True,
        # dropout_p=dropout
    )
    return attn_output, None


def flex_block_diffusion_forward_wise(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    metadata = kwargs.get("block_metadata")

    # 回退逻辑: 如果metadata为空, 调用原生flash_attention
    if metadata is None:
        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        if attention_interface is None:
            attention_interface = ALL_ATTENTION_FUNCTIONS["eager"]

        attn_output = attention_interface(module, query, key, value, attention_mask, is_causal=False, dropout=dropout, scaling=scaling, **kwargs)
        return attn_output, None

    # block diffusion logic
    attn_output = torch.zeros_like(query)
    B, H_q, L, D = query.shape
    device = query.device
    # H_kv = key.shape[1]
    # num_groups = H_q // H_kv  # 算出每个 KV 头对应多少个 Q 头

    sample_ids = metadata["sample_ids"]
    token_types = metadata["token_types"]
    block_ids = metadata["block_ids"]

    for b in range(B):
        b_sample_ids = sample_ids[b]

        # 找到sample_ids变化的边界
        diff = b_sample_ids[1:] != b_sample_ids[:-1]
        change_points = diff.nonzero(as_tuple=True)[0] + 1
        # 组合起始和终点: [0, p1, p2, ..., len]
        boundaries = torch.cat([torch.tensor([0], device=device), change_points, torch.tensor([L], device=device)])

        # 针对该 batch 内的每个 sample 进行切片计算
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            s_id = b_sample_ids[start].item()

            if s_id == -1: # 过滤 padding 部分
                continue

            # 执行切片
            q_s = query[b:b+1, :, start:end, :]  # [1, H, L_s, D]
            k_s = key[b:b+1, :, start:end, :]
            v_s = value[b:b+1, :, start:end, :]

            # 提取该 sample 内部的 token 元数据
            s_token_types = token_types[b, start:end]
            s_block_ids = block_ids[b, start:end]

            # 4. 定义简化的局部 Score Mod
            # 此时不再需要判断 sq == sk，因为切片本身已经保证了样本隔离
            def local_score_mod(b_idx, h_idx, q_idx, k_idx):
                tq, tk = s_token_types[q_idx], s_token_types[k_idx]
                bq, bk = s_block_ids[q_idx], s_block_ids[k_idx]

                # 逻辑 1: prefix
                mask_prefix = (tq == 0) & (tk == 0) & (bk <= bq)
                # 逻辑 2: clean
                mask_clean = (tq == 1) & ((tk == 0) | ((tk == 1) & (bk <= bq)))
                # 逻辑 3: noisy
                mask_noisy = (tq == 2) & (
                    (tk == 0) | ((tk == 1) & (bk < bq)) | ((tk == 2) & (bk == bq))
                )

                is_visible = mask_prefix | mask_clean | mask_noisy
                return is_visible

            # 5. 生成局部 Block Mask
            L_s = end - start
            block_mask = create_block_mask(
                local_score_mod,
                B=1,
                H=None,
                Q_LEN=L_s,
                KV_LEN=L_s,
                device=device,
                _compile=False
            )

            # 6. 计算并将结果写回输出张量
            # 显式使用 [b:b+1, :, start:end, :] 确保梯度路径正确
            attn_output[b:b+1, :, start:end, :] = flex_attention(
                q_s, k_s, v_s,
                block_mask=block_mask,
                scale=scaling,
                enable_gqa=True,
            )

    return attn_output, None


def sdpa_block_diffusion_forward_full(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
):
    metadata = kwargs.get("block_metadata")

    B, H_q, L, D = query.shape
    H_kv = key.shape[1]

    # 1. 准备元数据 (假设形状均为 [B, L])
    s_ids = metadata["sample_ids"].unsqueeze(2)    # [B, L, 1]
    t_types = metadata["token_types"].unsqueeze(2) # [B, L, 1]
    b_ids = metadata["block_ids"].unsqueeze(2)     # [B, L, 1]

    # 2. 构造广播矩阵
    # 利用广播机制计算 [B, L, L] 的关系矩阵
    sample_mask = (s_ids == s_ids.transpose(1, 2)) & (s_ids != -1)
    t_q, t_k = t_types, t_types.transpose(1, 2)
    b_q, b_k = b_ids, b_ids.transpose(1, 2)

    # 逻辑 1: prefix (type 0) -> 看到同 sample 同为 prefix 且 block_id 满足因果
    mask_prefix = (t_q == 0) & (t_k == 0) & (b_k <= b_q)

    # 逻辑 2: clean (type 1) -> 看到同 sample 的所有 prefix，或同为 clean 且因果
    mask_clean = (t_q == 1) & ((t_k == 0) | ((t_k == 1) & (b_k <= b_q)))

    # 逻辑 3: noisy (type 2) -> 看到同 sample 所有 prefix，或之前 block 的 clean，或同 block 的 noisy
    mask_noisy = (t_q == 2) & (
        (t_k == 0) | ((t_k == 1) & (b_k < b_q)) | ((t_k == 2) & (b_k == b_q))
    )

    # 合并所有逻辑并添加 Sample 隔离
    # 最终 mask 形状: [B, 1, L, L]
    full_mask = (mask_prefix | mask_clean | mask_noisy) & sample_mask
    full_mask = full_mask.unsqueeze(1)

    # 3. 处理 GQA
    if H_q != H_kv:
        num_groups = H_q // H_kv
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)

    # 4. 一次性调用 SDPA
    attn_output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=full_mask,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling
    )

    return attn_output, None


def sdpa_block_diffusion_forward_wise(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    metadata = kwargs.get("block_metadata")

    # 1. 回退逻辑
    if metadata is None:
        # 注意：SDPA 接受 attn_mask 或 is_causal 参数
        attn_output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=dropout if module.training else 0.0,
            is_causal=False,
            scale=scaling
        )
        return attn_output, None

    attn_output = torch.zeros_like(query)
    B, H_q, L, D = query.shape
    device = query.device
    dtype = query.dtype
    H_kv = key.shape[1]
    num_groups = H_q // H_kv  # 算出每个 KV 头对应多少个 Q 头

    sample_ids = metadata["sample_ids"]
    token_types = metadata["token_types"]
    block_ids = metadata["block_ids"]

    for b in range(B):
        b_sample_ids = sample_ids[b]
        diff = b_sample_ids[1:] != b_sample_ids[:-1]
        change_points = diff.nonzero(as_tuple=True)[0] + 1
        boundaries = torch.cat([torch.tensor([0], device=device), change_points, torch.tensor([L], device=device)])

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            s_id = b_sample_ids[start].item()

            if s_id == -1:
                continue

            # 2. 提取当前 sample 的数据
            q_s = query[b:b+1, :, start:end, :]  # [1, H, L_s, D]
            k_s = key[b:b+1, :, start:end, :]
            v_s = value[b:b+1, :, start:end, :]

            # 使用 repeat_interleave 将 [1, 8, L_s, D] 变为 [1, 16, L_s, D]
            k_s = k_s.repeat_interleave(num_groups, dim=1)
            v_s = v_s.repeat_interleave(num_groups, dim=1)

            # 3. 向量化构建 4D Mask (针对当前 Sample)
            # 提取元数据并增加维度用于广播: [L_s] -> [L_s, 1] 和 [1, L_s]
            s_t = token_types[b, start:end]
            s_b = block_ids[b, start:end]

            tq, tk = s_t.unsqueeze(1), s_t.unsqueeze(0)
            bq, bk = s_b.unsqueeze(1), s_b.unsqueeze(0)

            # 逻辑 1: prefix (type 0)
            # 只能看到相同为 prefix 的 token，且 block_id 满足因果性
            mask_prefix = (tq == 0) & (tk == 0) & (bk <= bq)

            # 逻辑 2: clean (type 1)
            # 能看到所有 prefix，或者看到同为 clean 且 block_id 满足因果性
            mask_clean = (tq == 1) & ((tk == 0) | ((tk == 1) & (bk <= bq)))

            # 逻辑 3: noisy (type 2)
            # 能看到所有 prefix，或者看到之前的 clean，或者看到同 block 的 noisy
            mask_noisy = (tq == 2) & (
                (tk == 0) | ((tk == 1) & (bk < bq)) | ((tk == 2) & (bk == bq))
            )

            # 合并掩码 [L_s, L_s]
            full_mask = mask_prefix | mask_clean | mask_noisy

            # SDPA 的 attn_mask 期望:
            # Bool 类型: True 表示保留, False 表示遮蔽
            # 形状: (1, 1, L_s, L_s) 自动广播到 Head 维度
            full_mask = full_mask.unsqueeze(0).unsqueeze(0)

            # 4. 调用 SDPA
            attn_output[b:b+1, :, start:end, :] = F.scaled_dot_product_attention(
                q_s, k_s, v_s,
                attn_mask=full_mask,
                dropout_p=dropout if module.training else 0.0,
                is_causal=False,
                scale=scaling
            )

    return attn_output, None


# 为了非sequence packing设计的flex attention
def flex_block_diffusion_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    metadata = kwargs.get("block_metadata")

    # 回退逻辑: 如果metadata为空, 调用原生flash_attention
    if metadata is None:
        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        if attention_interface is None:
            attention_interface = ALL_ATTENTION_FUNCTIONS["eager"]

        attn_output = attention_interface(module, query, key, value, attention_mask, is_causal=False, dropout=dropout, scaling=scaling, **kwargs)
        return attn_output, None

    # block diffusion logic
    token_types = metadata["token_types"]
    block_ids = metadata["block_ids"]

    def block_diffusion_score_mod(b, h, q_idx, k_idx):
        tq, tk = token_types[b, q_idx], token_types[b, k_idx]
        bq, bk = block_ids[b, q_idx], block_ids[b, k_idx]

        # 1. 基础有效性检查 (Padding Mask)
        # 只要 query 或 key 任意一个是 padding (-1)，就不允许 attention
        is_valid = (tq != -1) & (tk != -1)

        # 2. 区域逻辑隔离
        # 2.1 query是prefix (tq=0)
        # block causal: 只能看到 prefix 中 block_id <= 自己的 block
        mask_prefix = (tq == 0) & (tk == 0) & (bk <= bq)

        # 2.2 query 是 clean response (tq=1)
        # 可以看到全部 prefix，以及 clean response 的前 i 个 blocks (block causal)
        mask_clean = (tq == 1) & (
            (tk == 0) |
            ((tk == 1) & (bk <= bq))
        )

        # 2.3 query 是 noisy response (tq=2)
        # 可以看到全部 prefix，clean response 的前 i-1 个 block，以及自身的 noisy block
        mask_noisy = (tq == 2) & (
            (tk == 0) |
            ((tk == 1) & (bk < bq)) |
            ((tk == 2) & (bk == bq))
        )

        # 最终掩码
        return is_valid & (mask_prefix | mask_clean | mask_noisy)

    L = query.shape[-2]
    block_mask = create_block_mask(
        block_diffusion_score_mod,
        B=query.shape[0],
        H=query.shape[1],
        Q_LEN=L,
        KV_LEN=L,
        device=query.device,
        _compile=False
    )

    attn_output = flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=True,
    )
    return attn_output, None

ALL_ATTENTION_FUNCTIONS["flex_block_diffusion"] = flex_block_diffusion_forward_full

class WAMDiff2VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class WAMDiff2VisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class WAMDiff2VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class WAMDiff2VisionPatchMerger(nn.Module):
    def __init__(self, config: WAMDiff2VisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class WAMDiff2VisionAttention(nn.Module):
    def __init__(self, config: WAMDiff2VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            # 对不支持cu_seqlens变长序列加速的注意力实现中，通过物理切分来模拟sequence packing的样本隔离
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class WAMDiff2VisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = WAMDiff2VisionAttention(config=config)
        self.mlp = WAMDiff2VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class WAMDiff2TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: WAMDiff2TextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, WAMDiff2 has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@use_kernel_forward_from_hub("RMSNorm")
class WAMDiff2TextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        WAMDiff2TextRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class WAMDiff2TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: WAMDiff2TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False # modified

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = WAMDiff2TextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = WAMDiff2TextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)   # [b, h=32, seq, d=128]
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)     # [b, h=8, seq, d=128]
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)                # [b, h=8, seq, d=128]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            if use_cache: # update cache.
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else: # only use cache, don't update.
                past_key = past_key_values[self.layer_idx][0] if self.layer_idx < len(past_key_values) else None
                past_value = past_key_values[self.layer_idx][1] if self.layer_idx < len(past_key_values) else None
                if past_key is not None and past_value is not None:
                    key_states = torch.cat([past_key, key_states], dim=-2)
                    value_states = torch.cat([past_value, value_states], dim=-2)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 训练阶段且存在block_metadata时，强制使用flex_block_diffusion
        if self.training and "block_metadata" in kwargs:
            attention_interface = ALL_ATTENTION_FUNCTIONS['flex_block_diffusion']
        elif self.training:
            attention_interface = ALL_ATTENTION_FUNCTIONS['sdpa']
        else:
            # 推理阶段或普通模式，使用配置指定的实现 (如 flash_attention_2 或 eager)
            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # print(attention_interface)

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class WAMDiff2TextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class WAMDiff2TextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: WAMDiff2TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = WAMDiff2TextAttention(config=config, layer_idx=layer_idx)

        self.mlp = WAMDiff2TextMLP(config)
        self.input_layernorm = WAMDiff2TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = WAMDiff2TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class WAMDiff2ModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


@auto_docstring
class WAMDiff2PreTrainedModel(PreTrainedModel):
    config: WAMDiff2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WAMDiff2TextDecoderLayer", "WAMDiff2VisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": WAMDiff2TextDecoderLayer,
        "attentions": WAMDiff2TextAttention,
    }


class WAMDiff2VisionModel(WAMDiff2PreTrainedModel):
    config: WAMDiff2VisionConfig
    _no_split_modules = ["WAMDiff2VisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = WAMDiff2VisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = WAMDiff2VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([WAMDiff2VisionBlock(config) for _ in range(config.depth)])
        self.merger = WAMDiff2VisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                WAMDiff2VisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


@auto_docstring(
    custom_intro=(
        "Text part of WAMDiff2, "
        "not a pure text-only model, as DeepStack integrates visual features into the early hidden states."
    )
)
class WAMDiff2TextModel(WAMDiff2PreTrainedModel):
    config: WAMDiff2TextConfig
    _no_split_modules = ["WAMDiff2TextDecoderLayer"]

    def __init__(self, config: WAMDiff2TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [WAMDiff2TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = WAMDiff2TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = WAMDiff2TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        """
        hidden_states: [b, seq_len, dim].
        visual_pos_masks: [b, seq_len].
        visual_embeds: [total_visual_tokens, dim].
        """
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


@auto_docstring
class WAMDiff2Model(WAMDiff2PreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: WAMDiff2Config
    _no_split_modules = ["WAMDiff2TextDecoderLayer", "WAMDiff2VisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = WAMDiff2VisionModel._from_config(config.vision_config)
        self.language_model = WAMDiff2TextModel._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, WAMDiff2 use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, WAMDiff2ModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        _vision_timing = getattr(self, "_enable_generate_timing", False)
        _vision_device = inputs_embeds.device

        if _vision_timing:
            try:
                if _vision_device.type == "npu":
                    torch.npu.synchronize()
                elif _vision_device.type == "cuda":
                    torch.cuda.synchronize()
            except Exception:
                pass
            _t_vision_start = time.perf_counter()

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if _vision_timing:
            try:
                if _vision_device.type == "npu":
                    torch.npu.synchronize()
                elif _vision_device.type == "cuda":
                    torch.cuda.synchronize()
            except Exception:
                pass
            self._vision_time = time.perf_counter() - _t_vision_start

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return WAMDiff2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

@dataclass
@auto_docstring(
    custom_intro="""
    Base class for WAMDiff2 causal language model (or autoregressive) outputs.
    """
)
class WAMDiff2CausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None


class WAMDiff2ForConditionalGeneration(WAMDiff2PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: WAMDiff2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = WAMDiff2Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.rl_adapter = WAMDiff2RLAdapter(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def prepare_for_rl_training(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        masked_indices: torch.Tensor,
        step_ids: Optional[torch.Tensor] = None,
    ):
        return self.rl_adapter.prepare_batch(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            masked_indices=masked_indices,
            step_ids=step_ids,
        )

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        masked_indices: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        compute_rl_loss: bool = False,
        p_mask: Optional[torch.Tensor] = None,
        adv: Optional[torch.Tensor] = None,
        logp_old_tok: Optional[torch.Tensor] = None,
        logp_ref_tok: Optional[torch.Tensor] = None,
        is_real: Optional[torch.Tensor] = None,
        ppo_eps: float = 0.2,
        kl_beta: float = 0.0,
        use_kl_estimator_k3: bool = True,
        return_entropy: bool = False,
        loss_mean: bool = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, WAMDiff2CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        step_ids = kwargs.pop("step_ids", None)

        if self.training and masked_indices is not None and (compute_rl_loss or return_logits):
            prepared = self.prepare_for_rl_training(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                masked_indices=masked_indices,
                step_ids=step_ids,
            )

            outputs = self.model(
                input_ids=prepared.input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=prepared.position_ids,
                attention_mask=prepared.attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                block_metadata=prepared.block_metadata,
                **kwargs,
            )

            hidden_states = outputs[0]
            masked_hidden_states = hidden_states[prepared.logits_to_keep].contiguous()
            logits = self.lm_head(masked_hidden_states)

            output = WAMDiff2CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                rope_deltas=outputs.rope_deltas,
            )

            if compute_rl_loss:
                if p_mask is None:
                    raise ValueError("`p_mask` must be provided when `compute_rl_loss=True`.")
                if adv is None:
                    raise ValueError("`adv` must be provided when `compute_rl_loss=True`.")
                if is_real is None:
                    raise ValueError("`is_real` must be provided when `compute_rl_loss=True`.")
                if labels is None:
                    raise ValueError("`labels` must be provided when `compute_rl_loss=True`.")

                rl_output = self.rl_adapter.compute_rl_loss(
                    logits=logits,
                    labels=labels,
                    masked_indices=masked_indices,
                    p_mask=p_mask,
                    adv=adv,
                    is_real=is_real,
                    logp_old_tok=logp_old_tok,
                    logp_ref_tok=logp_ref_tok,
                    ppo_eps=ppo_eps,
                    kl_beta=kl_beta,
                    use_kl_estimator_k3=use_kl_estimator_k3,
                    return_entropy=return_entropy,
                    loss_mean=loss_mean,
                )
                output.loss = rl_output.loss
                output.entropy = rl_output.entropy
                output.kl_loss = rl_output.kl_loss

            return output

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return WAMDiff2CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    @staticmethod
    def sample_tokens(logits, temperature=0.0, top_k=0, top_p=1.0):
        """Sample tokens with temperature, top-k, and top-p.

        Uses inverse CDF sampling (cumsum + rand + sum) instead of
        torch.multinomial, which hangs on Ascend NPU.  All operations
        stay on the original device — no D2H/H2D copy needed.
        """
        batch_size = logits.shape[0]
        seq_len = logits.shape[1]
        vocab_size = logits.shape[-1]

        logits_2d = logits.reshape(-1, vocab_size)
        device = logits_2d.device

        if temperature == 0:
            # Greedy sampling
            tokens = torch.argmax(logits_2d, dim=-1, keepdim=True)
            probs = F.softmax(logits_2d, dim=-1)
            token_probs = torch.gather(probs, -1, tokens)
        else:
            # Apply temperature
            logits_scaled = logits_2d / temperature

            # --- Reduce effective effective vocabulary via topk ---
            # This implements top-k filtering when top_k > 0, and enables
            # efficient top-p without a full sort when top_k == 0.
            # 1000 is more than enough to capture the 95% nucleus for
            # typical LLM distributions.
            if top_k > 0:
                effective_k = min(top_k, vocab_size)
            elif top_p < 1.0:
                effective_k = min(1000, vocab_size)
            else:
                effective_k = vocab_size

            if effective_k < vocab_size:
                topk_vals, topk_idx = torch.topk(logits_scaled, effective_k, dim=-1)
            else:
                topk_vals = logits_scaled
                topk_idx = None

            # Softmax on the (possibly reduced) logits
            probs = F.softmax(topk_vals, dim=-1)

            # --- Top-p (nucleus) filtering ---
            # topk already gives us sorted logits, so cumsum on probs is
            # equivalent to sorting by probability.
            if top_p < 1.0:
                cum_probs_tp = torch.cumsum(probs, dim=-1)
                sorted_mask = cum_probs_tp > top_p
                # Shift right: always keep at least the first (highest-prob) token
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                probs = probs.masked_fill(sorted_mask, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # --- Inverse CDF sampling (replaces torch.multinomial) ---
            # token = first index where cumsum(probs) >= u
            cum_probs = torch.cumsum(probs, dim=-1)
            u = torch.rand(probs.shape[0], 1, device=device, dtype=probs.dtype)
            local_idx = (cum_probs < u).sum(dim=-1, keepdim=True).long()
            local_idx = local_idx.clamp(max=probs.shape[-1] - 1)

            token_probs = torch.gather(probs, -1, local_idx)

            if topk_idx is not None:
                tokens = torch.gather(topk_idx, -1, local_idx)
            else:
                tokens = local_idx

        return tokens.view(batch_size, seq_len), token_probs.view(batch_size, seq_len)

    @staticmethod
    def get_num_transfer_tokens(block_length, steps):
        """Calculate how many tokens to unmask at each step."""
        if steps == 0:
            return torch.zeros(1, dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer = torch.full((steps,), base, dtype=torch.int64)
        num_transfer[:remainder] += 1
        return num_transfer

    @torch.no_grad()
    def generate(
        self,
        inputs,
        max_new_tokens: int = 256,
        block_size: int = 32,
        horizon_size: int = 0,
        denoising_steps: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        remasking_strategy: str = "low_confidence_dynamic",
        confidence_threshold: float = 0.9,
        pad_target_penalty: float = 1.0,
        mask_token_id: int = 151671,
        eos_token_id: int = 151645,  # <|im_end|>
        output_history: bool = False,
    ):
        """Batch-capable block diffusion decoding for WAM Diff2.

        Returns:
            Tensor of shape [batch_size, generated_length].
            The returned ids contain only generated response tokens, not prompt tokens.

        Notes:
            - This function assumes rollout is called under torch.no_grad().
            - For this model, rollout is usually safer while the model remains in train()
            mode, because WAMDiff2TextAttention uses SDPA in training mode when no
            block_metadata is provided.
        """
        if block_size is None:
            block_size = max_new_tokens

        device = inputs.input_ids.device
        batch_size = inputs.input_ids.shape[0]
        prompt_len = inputs.input_ids.shape[-1]
        input_dtype = inputs.input_ids.dtype

        # 1. Align generation length to block_size.
        num_res_blocks = (max_new_tokens + block_size - 1) // block_size
        max_new_tokens = num_res_blocks * block_size

        # 2. Build block causal attention mask.
        total_len = prompt_len + max_new_tokens
        block_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
        num_prompt_blocks = (prompt_len + block_size - 1) // block_size

        for b in range(num_prompt_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, prompt_len)
            block_ids[start:end] = b

        for b in range(num_res_blocks):
            start = prompt_len + b * block_size
            end = prompt_len + (b + 1) * block_size
            block_ids[start:end] = num_prompt_blocks + b

        q_blocks = block_ids.unsqueeze(1)
        k_blocks = block_ids.unsqueeze(0)

        # Bool mask: True means visible for PyTorch SDPA.
        block_attention_mask = (q_blocks >= k_blocks).unsqueeze(0).unsqueeze(0)
        block_attention_mask = block_attention_mask.expand(batch_size, 1, total_len, total_len)

        # 3. Use real prompt attention_mask if available.
        prefill_attention_mask = inputs.get("attention_mask", None)
        if prefill_attention_mask is None:
            prefill_attention_mask = torch.ones_like(inputs.input_ids)

        # 4. Compute MRoPE position ids for prompt.
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=inputs.input_ids,
            image_grid_thw=inputs.get("image_grid_thw", None),
            video_grid_thw=inputs.get("video_grid_thw", None),
            attention_mask=prefill_attention_mask,
        )

        # 5. Prefill prompt and build KV cache.
        #
        # Combine block causal mask with padding mask.
        # Shape:
        #   block mask:       [B, 1, S, S]
        #   key padding mask: [B, 1, 1, S]
        #   qry padding mask: [B, 1, S, 1]
        prompt_block_mask = block_attention_mask[:, :, :prompt_len, :prompt_len]

        if prefill_attention_mask is not None:
            prompt_valid = prefill_attention_mask.to(device=device).bool()
            key_padding_mask = prompt_valid[:, None, None, :]
            query_padding_mask = prompt_valid[:, None, :, None]
            prompt_block_mask = prompt_block_mask & key_padding_mask & query_padding_mask

        _timing = getattr(self, "_enable_generate_timing", False)

        # Propagate timing flag to the inner model so vision encoding can be timed.
        if _timing:
            self.model._enable_generate_timing = True
        else:
            self.model._enable_generate_timing = False

        # ---- timing helpers (NPU-compatible) ----
        _t_breakdown = {
            "prefill": 0.0,
            "vision": 0.0,             # vision encoder (image/video feature extraction)
            "decode_forward": 0.0,     # denoising loop forward passes
            "sample_tokens": 0.0,      # sampling (multinomial / argmax)
            "remasking": 0.0,          # transfer mask construction
            "cache_update": 0.0,       # KV cache update forward
            "mask_misc": 0.0,          # mask/where/tensor ops
        }
        _t_call_count = {
            "decode_forward": 0,
            "sample_tokens": 0,
            "remasking": 0,
            "cache_update": 0,
        }

        def _sync():
            if _timing:
                try:
                    if device.type == "npu":
                        torch.npu.synchronize()
                    elif device.type == "cuda":
                        torch.cuda.synchronize()
                except Exception:
                    pass

        if _timing:
            _sync()
            _t_prefill_start = time.perf_counter()

        output = self(
            input_ids=inputs.input_ids,
            attention_mask=prompt_block_mask,
            position_ids=position_ids,
            use_cache=True,
            pixel_values=inputs.get("pixel_values", None),
            pixel_values_videos=inputs.get("pixel_values_videos", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            video_grid_thw=inputs.get("video_grid_thw", None),
        )

        if _timing:
            _sync()
            _t_breakdown["prefill"] = time.perf_counter() - _t_prefill_start
            _t_breakdown["vision"] = getattr(self.model, "_vision_time", 0.0)
            # Disable vision timing on inner model to avoid extra syncs during decode.
            self.model._enable_generate_timing = False

        current_cache = output.past_key_values
        rope_delta_base = rope_deltas

        # Denoising iterations per response block.
        steps_per_block = [max(1, denoising_steps)] * num_res_blocks

        # 6. Init output buffer.
        whole_ids = torch.full(
            (batch_size, total_len),
            fill_value=mask_token_id,
            dtype=input_dtype,
            device=device,
        )
        whole_ids[:, :prompt_len] = inputs.input_ids

        histories = [] if output_history else None

        # Per-sample finished state.
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 7. Init horizon tokens. This must be batch-aware.
        next_block_init = torch.full(
            size=(batch_size, horizon_size),
            fill_value=mask_token_id,
            dtype=input_dtype,
            device=device,
        )

        # 8. Inter-block decoding loop.
        if _timing:
            _sync()
            _t_decode_start = time.perf_counter()
        block_actual_steps = []  # actual denoising steps per block (before early break)
        for b in range(num_res_blocks):
            curr_block_start = prompt_len + b * block_size
            curr_block_end = min(curr_block_start + block_size, total_len)
            curr_block_len = curr_block_end - curr_block_start

            if horizon_size is not None:
                window_end = min(curr_block_end + horizon_size, total_len)
            else:
                window_end = curr_block_end

            current_horizon = window_end - curr_block_end

            # Current block + optional horizon window.
            block_input_ids = torch.cat(
                [
                    torch.full(
                        (batch_size, block_size),
                        fill_value=mask_token_id,
                        dtype=input_dtype,
                        device=device,
                    ),
                    next_block_init[:, :current_horizon],
                ],
                dim=1,
            )

            # Do not generate meaningful tokens for samples that already ended.
            if finished.any():
                block_input_ids[finished, :block_size] = eos_token_id

            # 9. Position ids for current block/window.
            #
            # get_rope_index returns rope_deltas with shape [batch_size, 1].
            # For later decoding, position should be:
            #   base absolute position + per-sample rope delta
            # Final shape should be [3, batch_size, seq_len].
            indices = torch.arange(curr_block_start, window_end, device=device)
            block_position_ids = indices.view(1, -1).expand(batch_size, -1).clone()

            if rope_delta_base is not None:
                block_position_ids = block_position_ids + rope_delta_base.to(
                    device=device,
                    dtype=block_position_ids.dtype,
                )

            block_position_ids = block_position_ids.unsqueeze(0).expand(3, -1, -1)

            # 10. Intra-block denoising loop.
            num_transfer_tokens = self.get_num_transfer_tokens(
                block_size,
                steps_per_block[b],
            ).to(device=device)

            for step in range(steps_per_block[b]):
                is_mask = block_input_ids == mask_token_id

                # Finished samples should not be updated.
                if finished.any():
                    is_mask[finished] = False

                if not is_mask.any():
                    block_actual_steps.append(step)  # broke: actual steps = step (forward skipped this iter)
                    break

                if _timing:
                    _sync()
                    _t_fwd_start = time.perf_counter()

                output = self(
                    input_ids=block_input_ids,
                    position_ids=block_position_ids,
                    attention_mask=block_attention_mask[:, :, curr_block_start:window_end, :window_end],
                    past_key_values=current_cache,
                    use_cache=False,
                )

                logits = output.logits

                if _timing:
                    _sync()
                    _t_breakdown["decode_forward"] += time.perf_counter() - _t_fwd_start
                    _t_call_count["decode_forward"] += 1
                    _t_sample_start = time.perf_counter()

                # Sample / greedy decode candidate x0.
                x0, x0_p = self.sample_tokens(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                if _timing:
                    _sync()
                    _t_breakdown["sample_tokens"] += time.perf_counter() - _t_sample_start
                    _t_call_count["sample_tokens"] += 1
                    _t_rem_start = time.perf_counter()

                if pad_target_penalty != 1.0:
                    is_pad_token = x0 == mask_token_id
                    x0_p = torch.where(is_pad_token, x0_p / pad_target_penalty, x0_p)

                num_to_transfer = num_transfer_tokens[step].item()

                effective_mask = is_mask.clone()
                effective_mask[:, block_size:] = False

                if finished.any():
                    effective_mask[finished] = False

                transfer_mask = torch.zeros_like(x0, dtype=torch.bool)

                if remasking_strategy == "sequential":
                    for j in range(batch_size):
                        if finished[j]:
                            continue

                        mask_positions = effective_mask[j].nonzero(as_tuple=True)[0]
                        if len(mask_positions) == 0:
                            continue

                        num_to_select = min(num_to_transfer, len(mask_positions))
                        selected_positions = mask_positions[:num_to_select]
                        transfer_mask[j, selected_positions] = True

                elif remasking_strategy == "low_confidence_static":
                    confidence = torch.where(
                        effective_mask,
                        x0_p,
                        torch.tensor(-torch.inf, device=device, dtype=x0_p.dtype),
                    )

                    for j in range(batch_size):
                        if finished[j]:
                            continue

                        num_masks = effective_mask[j].sum().item()
                        k = min(num_to_transfer, num_masks)

                        if k > 0 and not torch.all(torch.isinf(confidence[j])):
                            _, idx = torch.topk(confidence[j], k)
                            transfer_mask[j, idx] = True

                elif remasking_strategy == "low_confidence_dynamic":
                    confidence = torch.where(
                        effective_mask,
                        x0_p,
                        torch.tensor(-torch.inf, device=device, dtype=x0_p.dtype),
                    )

                    for j in range(batch_size):
                        if finished[j]:
                            continue

                        high_conf_mask = confidence[j] >= confidence_threshold
                        if high_conf_mask.any():
                            transfer_mask[j] = high_conf_mask
                        else:
                            num_masks = effective_mask[j].sum().item()
                            if num_masks > 0:
                                _, idx = torch.topk(confidence[j], 1)
                                transfer_mask[j, idx] = True

                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

                block_input_ids = torch.where(transfer_mask, x0, block_input_ids)

                if _timing:
                    _sync()
                    _t_breakdown["remasking"] += time.perf_counter() - _t_rem_start
                    _t_call_count["remasking"] += 1
            else:
                # Completed all steps without early break.
                block_actual_steps.append(steps_per_block[b])

            # 10.5 Fill remaining mask tokens with last prediction (avoids outputting mask_token_id).
            remaining_mask = (block_input_ids == mask_token_id)[:, :block_size]
            if finished.any():
                remaining_mask[finished] = False
            if remaining_mask.any():
                block_input_ids = torch.where(remaining_mask, x0, block_input_ids)

            # 11. Commit current block.
            final_block_ids = block_input_ids[:, :block_size]

            # For samples already finished before this block, keep EOS.
            if finished.any():
                final_block_ids = torch.where(
                    finished[:, None],
                    torch.full_like(final_block_ids, eos_token_id),
                    final_block_ids,
                )

            whole_ids[:, curr_block_start:curr_block_end] = final_block_ids[:, :curr_block_len]

            if histories is not None:
                histories.append(whole_ids[:, prompt_len:curr_block_end].clone().cpu())

            # Update per-sample finished state.
            finished |= (final_block_ids == eos_token_id).any(dim=1)

            if finished.all():
                break

            # 12. Clean forward to update KV cache.
            #
            # Only commit the current generated block into cache.
            if _timing:
                _sync()
                _t_cache_start = time.perf_counter()

            outputs = self(
                input_ids=final_block_ids[:, :curr_block_len],
                attention_mask=block_attention_mask[:, :, curr_block_start:curr_block_end, :curr_block_end],
                position_ids=block_position_ids[:, :, :curr_block_len],
                past_key_values=current_cache,
                use_cache=True,
            )

            current_cache = outputs.past_key_values

            if _timing:
                _sync()
                _t_breakdown["cache_update"] += time.perf_counter() - _t_cache_start
                _t_call_count["cache_update"] += 1

            # Optional horizon warm start. Your original code did not update this,
            # so we keep the same behavior. If later you want horizon warm start,
            # update next_block_init here from block_input_ids[:, block_size:].

        if _timing:
            _sync()
            _t_decode = time.perf_counter() - _t_decode_start
            _t_breakdown["decode_total"] = _t_decode
            _t_breakdown["mask_misc"] = (
                _t_decode
                - _t_breakdown["decode_forward"]
                - _t_breakdown["sample_tokens"]
                - _t_breakdown["remasking"]
                - _t_breakdown["cache_update"]
            )
            self._generate_timing = {
                "prefill_time": _t_breakdown["prefill"],
                "vision_time": _t_breakdown["vision"],
                "decode_time": _t_decode,
                "breakdown": dict(_t_breakdown),
                "call_count": dict(_t_call_count),
            }

        # Always expose actual denoising steps per block (independent of _timing).
        self._block_actual_steps = block_actual_steps

        if output_history:
            return whole_ids[:, prompt_len:], histories

        return whole_ids[:, prompt_len:]

    @torch.no_grad()
    def uniform_generate(
        self,
        inputs,
        max_new_tokens: int = 256,
        block_size: int = 32,
        horizon_size: int = 0,
        denoising_steps: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        remasking_strategy: str = 'low_confidence_dynamic',
        confidence_threshold: float = 0.9,
        pad_target_penalty: float = 1.0,
        mask_token_id: int = 151671,
        eos_token_id: int = 151645, # <|im_end|>
        vocab_size: int = 151646,
        output_history: bool = False,
    ):
        """Uniform block diffusion decoding for WAM Diff2.

        `denoising_steps` is the maximum number of denoising iterations for
        each response block, aligned with the server-side rollout semantics.
        """
        device = inputs.input_ids.device
        batch_size = inputs.input_ids.shape[0]
        prompt_len = inputs.input_ids.shape[-1]

        # 1. 调整长度对齐block_size
        num_res_blocks = (max_new_tokens + block_size - 1) // block_size
        max_new_tokens = num_res_blocks * block_size # update max_new_tokens

        # 2. 构建block_attention_mask
        total_len = prompt_len + max_new_tokens
        block_ids = torch.zeros(total_len, dtype=torch.int32, device=device)
        num_prompt_blocks = (prompt_len + block_size - 1) // block_size

        for b in range(num_prompt_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, prompt_len)
            block_ids[start: end] = b

        for b in range(num_res_blocks):
            start = prompt_len + b * block_size
            end = prompt_len + (b + 1) * block_size
            block_ids[start: end] = num_prompt_blocks + b

        q_blocks = block_ids.unsqueeze(1)
        k_blocks = block_ids.unsqueeze(0)
        block_attention_mask = (q_blocks >= k_blocks).unsqueeze(0).unsqueeze(0)

        # 3. prefilling, 仅对prompt部分进行一次推理，获取初始缓存
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=inputs.input_ids,
            image_grid_thw=inputs.get("image_grid_thw", None),
            video_grid_thw=inputs.get("video_grid_thw", None),
            attention_mask= torch.ones_like(inputs.input_ids),
        )

        output = self(
            input_ids=inputs.input_ids,
            attention_mask=block_attention_mask[:, :, :prompt_len, :prompt_len],
            position_ids=position_ids,
            use_cache=True,
            pixel_values=inputs.get("pixel_values", None),
            pixel_values_videos=inputs.get("pixel_values_videos", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            video_grid_thw=inputs.get("video_grid_thw", None)
        )
        # recon_prompt_ids = torch.argmax(output.logits, dim=-1)
        # recon_prompt_text = processor.decode(recon_prompt_ids[0])
        current_cache = output.past_key_values
        rope_delta_base = rope_deltas

        # `denoising_steps` is interpreted as the maximum denoising iterations
        # for each response block to match the server-side rollout semantics.
        steps_per_block = [max(1, denoising_steps)] * num_res_blocks

        # 初始化总结果
        whole_ids = torch.full((batch_size, total_len), fill_value=mask_token_id, dtype=torch.int32)
        whole_ids[:, :prompt_len] = inputs.input_ids
        histories = [] if output_history else None

        # 下一个block的预热初始化
        next_block_init = torch.randint(low=0, high=vocab_size, size=(1, horizon_size), device=device)

        # inter-block loop
        for b in range(num_res_blocks):
            curr_block_start = prompt_len + b * block_size
            curr_block_end = min(curr_block_start + block_size, total_len)

            window_end = min(curr_block_end + horizon_size, total_len) if horizon_size is not None else curr_block_end

            # 1. 初始化当前 block 状态
            current_horizon = window_end - curr_block_end
            block_input_ids = torch.cat([
                torch.randint(low=0, high=vocab_size, size=(batch_size, block_size), device=device),
                next_block_init[:, :current_horizon]
            ], dim=1)

            # 2. 计算当前block的position_ids
            indices = torch.arange(curr_block_start, window_end, device=device)
            position_ids = indices.view(1, -1).expand(batch_size, -1).clone() # [b, block_size]
            if rope_delta_base is not None:
                position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1) + rope_delta_base.view(-1, 1, 1) # apply RoPE offset

            # 3. intra-block loop
            num_steps = steps_per_block[b]

            scheduler = CondOTScheduler()
            path = MixtureDiscreteProbPath(scheduler)
            step_size = 1 / num_steps
            t_discrete = torch.tensor([0.0 + step_size * i for i in range(num_steps)] + [1.0], device=device)

            for step in range(num_steps):
                output = self(
                    input_ids=block_input_ids,
                    position_ids=position_ids,
                    attention_mask=block_attention_mask[:, :, curr_block_start:window_end, :window_end],
                    past_key_values=current_cache,
                    use_cache=False
                )
                logits = output.logits

                # Sample tokens
                x1, x1_p = self.sample_tokens(logits, temperature, top_k, top_p)

                if step == num_steps - 1:
                    block_input_ids = x1
                    break

                t = t_discrete[step: step + 1]
                h = t_discrete[step + 1: step + 2] - t

                scheduler_output = path.scheduler(t=t)
                k_t = scheduler_output.alpha_t      # [b,]
                d_k_t = scheduler_output.d_alpha_t  # [b,]

                delta_1 = F.one_hot(x1, num_classes=vocab_size)  # [b, l, vocab_size]
                u = d_k_t / (1 - k_t) * delta_1                  # [b, l, vocab_size]

                # set u_t(x_t|x_t, x_1) = 0
                delta_t = F.one_hot(block_input_ids, num_classes=vocab_size) # [b, l, vocab_size]
                u = torch.where(delta_t.to(torch.bool), torch.zeros_like(u), u)

                # sample x_t \sim u_t(\cdot |x_t, x_1)
                intensity = u.sum(dim=-1) # [b, l], assuming u_t(x_t|x_t, x1) := 0
                mask_jump = torch.rand(size=x1.shape, device=x1.device) < 1 - torch.exp(-h * intensity)

                if mask_jump.sum() > 0:
                    block_input_ids[mask_jump] = categorical(u[mask_jump].to(torch.float32))

            # 4. block commit, 只提取block_size个tokens作为确定的结果
            final_block_ids = block_input_ids[:, :block_size]
            whole_ids[:, curr_block_start:curr_block_end] = final_block_ids

            if histories is not None:
                histories.append(whole_ids[:, prompt_len: curr_block_end].clone().cpu())

            # early stop if eos_token_id exists in curr block.
            if eos_token_id in final_block_ids:
                break

            # 执行一次clean forward，更新kv cache
            outputs = self(
                input_ids=final_block_ids,
                attention_mask=block_attention_mask[:, :, curr_block_start: curr_block_end, :curr_block_end],
                position_ids=position_ids[:, :, :block_size] if position_ids is not None else None,
                past_key_values=current_cache,
                use_cache=True
            )

            # 5. update state
            current_cache = outputs.past_key_values

        if output_history:
            return whole_ids[:, prompt_len:], histories
        else:
            return whole_ids[:, prompt_len:]

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # WAMDiff2 position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs


__all__ = [
    "WAMDiff2VisionModel",
    "WAMDiff2ForConditionalGeneration",
    "WAMDiff2Model",
    "WAMDiff2PreTrainedModel",
    "WAMDiff2TextModel",
]
