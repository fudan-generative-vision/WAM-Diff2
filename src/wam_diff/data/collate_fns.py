# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from unittest.mock import MagicMock

import os
import re
import torch
import json
import numpy as np
from wam_diff.shared.import_utils import MISSING_QWEN_VL_UTILS_MSG
from typing import Optional, Tuple

try:
    from qwen_vl_utils import process_vision_info
    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False
    process_vision_info = MagicMock()

try:
    from qwen_omni_utils import process_mm_info

    HAVE_QWEN_OMNI_UTILS = True
except ImportError:
    HAVE_QWEN_OMNI_UTILS = False
    process_mm_info = MagicMock()

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from wam_diff.data.utils import default_stop_tokens


def _find_pattern_indices(template, pattern, search_start_index=0, allow_first_token_mismatch=False):
    template_len = len(template)
    pattern_len = len(pattern)
    for i in range(search_start_index, template_len - pattern_len + 1):
        match = template[i : i + pattern_len] == pattern
        if torch.all(match) or (allow_first_token_mismatch and torch.all(match[1:])):
            return i, i + pattern_len
    return -1, -1


def _extract_assistant_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return ""


def build_labels(
    input_ids_batch: torch.Tensor,
    conversations: Sequence[Sequence[Dict[str, Any]]],
    processor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct label and optional loss-mask tensors aligned to assistant responses."""
    tokenizer = getattr(processor, "tokenizer", processor)

    labels_list: List[torch.Tensor] = []

    for encoded, conversation in zip(input_ids_batch, conversations):
        labels = torch.full_like(encoded, -100)
        search_start_index = 0

        for message in conversation:
            if message.get("role") != "assistant":
                continue

            assistant_text = _extract_assistant_text(message)
            if not assistant_text:
                continue

            assistant_tokens = tokenizer(
                assistant_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(encoded.device)

            answer_start, answer_end = _find_pattern_indices(encoded, assistant_tokens, search_start_index)

            if answer_end < len(encoded):
                next_token_str = tokenizer.decode(encoded[answer_end])
                if next_token_str.strip() in default_stop_tokens(processor):
                    answer_end += 1

            if answer_start >= 0:
                labels[answer_start:answer_end] = encoded[answer_start:answer_end]
                search_start_index = answer_end
            else:
                logger.warning(
                    (
                        "Unable to find answer segment in the tokenized conversation. "
                        "Skipping labeling for this and subsequent answers. Details:"
                        "\n- Processed Text: %s"
                        "\n- Tokens: %s"
                        "\n- Target Answer Tokens: %s"
                        "\n- Search Start Index: %d"
                    ),
                    conversation,
                    encoded,
                    assistant_tokens,
                    search_start_index,
                )
                break

        labels_list.append(labels)

    labels_tensor = torch.stack(labels_list)
    return labels_tensor


def phi4_mm_collate_fn(examples, processor):
    """Collate function for Phi-4 MM model audio input"""

    # Extract conversations and audio data
    conversations = [example["conversation"] for example in examples]
    audios = [example["audio"] for example in examples]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    audio_inputs = [(audio["array"], audio["sampling_rate"]) if isinstance(audio, dict) else audio for audio in audios]
    batch = processor(
        text=texts, audios=audio_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )

    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )

    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    batch["labels"] = labels

    # Remove specified batch features if present
    for key in ["input_image_embeds", "image_sizes", "image_attention_mask"]:
        if key in batch:
            del batch[key]
    return batch


def qwen2_5_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = [example["conversation"] for example in examples]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    image_inputs = [process_vision_info(conversation)[0] for conversation in conversations]

    batch = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]

    return batch


def qwen3_omni_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    use_audio_in_video: bool = False,
) -> Dict[str, torch.Tensor]:
    """Collate function for Qwen3 Omni processors."""
    if not HAVE_QWEN_OMNI_UTILS:
        raise ImportError(
            "qwen_omni_utils is required for qwen3_omni_collate_fn. Install it with: pip install qwen-omni-utils"
        )

    conversations = [example["conversation"] for example in examples]
    texts = [
        processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
        for conversation in conversations
    ]

    all_audios = []
    all_images = []
    all_videos = []
    for conversation in conversations:
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        all_audios.append(audios)
        all_images.append(images)
        all_videos.append(videos)

    def has_data(modality_list):
        for item in modality_list:
            if item is None:
                continue
            if isinstance(item, list) and len(item) == 0:
                continue
            return True
        return False

    processor_kwargs = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "padding_side": "right",
    }

    if has_data(all_audios):
        processor_kwargs["audio"] = all_audios
    if has_data(all_images):
        processor_kwargs["images"] = all_images
    if has_data(all_videos):
        processor_kwargs["videos"] = all_videos

    batch = processor(**processor_kwargs)

    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )

    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor) and value.shape == input_shape:
            batch[key] = value[:, :-1]
    return batch


def default_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
) -> Dict[str, torch.Tensor]:
    """Default collate function for multimodal VLM datasets."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = [example["conversation"] for example in examples]
    batch = processor.apply_chat_template(
        conversations,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    labels = build_labels(
        batch["input_ids"],
        conversations,
        processor,
    )
    batch["labels"] = labels[:, 1:]

    input_shape = batch["input_ids"].shape
    for key in batch:
        if batch[key].shape == input_shape and key != "labels":
            batch[key] = batch[key][:, :-1]
    return batch


def get_rope_index(
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    vision_start_token_id: int = 151652,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone function to calculate 3D position IDs for Qwen3-VL.
    Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids.
    """
    # Since we use timestamps to seperate videos, the video_grid_thw should also be split
    if video_grid_thw is not None:
        # Avoid modifying the input tensor in-place if it's used elsewhere
        video_grid_thw = video_grid_thw.clone()
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

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
        # Ensure attention_mask is on the same device
        attention_mask = attention_mask.to(total_input_ids.device)

        # Iterate over batch
        for i, input_ids_item in enumerate(total_input_ids):
            # Filter valid tokens using attention mask
            valid_input_ids = input_ids_item[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0

            # Find all indices of <vision_start>
            vision_start_indices = torch.argwhere(valid_input_ids == vision_start_token_id).squeeze(1)

            # Check the token immediately following <vision_start> to identify type
            # We need to ensure we don't go out of bounds if vision_start is the last token (unlikely but safe to check)
            if len(vision_start_indices) > 0:
                vision_tokens = valid_input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()

            input_tokens = valid_input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums

            # Iterate through each vision segment in the sequence
            for _ in range(image_nums + video_nums):
                # Find the next image token index
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1

                # Find the next video token index
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                # Determine which comes first
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

                # Calculate grid sizes for LLM (downsampled by spatial_merge_size)
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )

                # Process text before this vision block
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # Process vision block position IDs
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()

                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)

                # Update start pointer
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            # Process remaining text after the last vision block
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            # Fill the position_ids tensor
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))

        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        # Pure text case
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

from wam_diff.utils.scheduler import (
    CondOTScheduler,
    ConvexScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialConvexScheduler,
    Scheduler,
    VPScheduler,
)
from wam_diff.utils.mixture import MixtureDiscreteProbPath


_NOISE_SCHEDULER_REGISTRY = {
    "condot": CondOTScheduler,
    "condotscheduler": CondOTScheduler,
    "polynomial": PolynomialConvexScheduler,
    "polynomialconvex": PolynomialConvexScheduler,
    "polynomialconvexscheduler": PolynomialConvexScheduler,
    "vp": VPScheduler,
    "vpscheduler": VPScheduler,
    "linearvp": LinearVPScheduler,
    "linearvpscheduler": LinearVPScheduler,
    "cosine": CosineScheduler,
    "cosinescheduler": CosineScheduler,
}


def _normalize_scheduler_name(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name).lower()


def _resolve_scheduler_cls(name: str):
    normalized = _normalize_scheduler_name(name)
    scheduler_cls = _NOISE_SCHEDULER_REGISTRY.get(normalized)
    if scheduler_cls is None:
        supported = ", ".join(sorted({"CondOT", "PolynomialConvex", "VP", "LinearVP", "Cosine"}))
        raise ValueError(f"Unsupported noise scheduler '{name}'. Supported schedulers: {supported}")
    return scheduler_cls


def _build_noise_scheduler(noise_scheduler: Any) -> tuple[Optional[Scheduler], Optional[Any]]:
    if noise_scheduler is None:
        noise_scheduler = "CondOT"

    if isinstance(noise_scheduler, dict):
        scheduler_cfg = dict(noise_scheduler)
        scheduler_name = (
            scheduler_cfg.pop("name", None)
            or scheduler_cfg.pop("type", None)
            or scheduler_cfg.pop("scheduler", None)
        )
        target = scheduler_cfg.pop("_target_", None)
        if scheduler_name is None and target is not None:
            scheduler_name = str(target).rsplit(".", maxsplit=1)[-1]
        if scheduler_name is None:
            raise ValueError("Noise scheduler config must provide 'name', 'type', 'scheduler', or '_target_'.")
        scheduler = _resolve_scheduler_cls(str(scheduler_name))(**scheduler_cfg)
    elif isinstance(noise_scheduler, str):
        scheduler = _resolve_scheduler_cls(noise_scheduler)()
    elif isinstance(noise_scheduler, Scheduler):
        scheduler = noise_scheduler
    elif isinstance(noise_scheduler, MixtureDiscreteProbPath):
        return getattr(noise_scheduler, "scheduler", None), noise_scheduler
    elif hasattr(noise_scheduler, "sample"):
        return getattr(noise_scheduler, "scheduler", None), noise_scheduler
    elif hasattr(noise_scheduler, "to_dict"):
        return _build_noise_scheduler(noise_scheduler.to_dict())
    elif hasattr(noise_scheduler, "__dict__"):
        scheduler_cfg = {k: v for k, v in noise_scheduler.__dict__.items() if not k.startswith("_")}
        if scheduler_cfg:
            return _build_noise_scheduler(scheduler_cfg)
        raise TypeError("Received an empty scheduler config object.")
    else:
        raise TypeError(
            "noise_scheduler/path_scheduler must be a scheduler name, scheduler config, scheduler instance, "
            "or path object with a sample() method."
        )

    path = MixtureDiscreteProbPath(scheduler=scheduler) if isinstance(scheduler, ConvexScheduler) else None
    return scheduler, path


def _sample_discrete_noise(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    *,
    scheduler: Optional[Scheduler],
    path: Optional[Any],
) -> torch.Tensor:
    if path is not None:
        return path.sample(t=t, x_0=x_0, x_1=x_1).x_t

    if scheduler is None:
        raise ValueError("A valid noise scheduler or path object is required for Uniform corruption.")

    scheduler_output = scheduler(t)
    sigma_t = getattr(scheduler_output, "sigma_t", None)
    if sigma_t is None:
        raise ValueError(f"Noise scheduler {type(scheduler).__name__} must return sigma_t for discrete corruption.")

    sigma_t = sigma_t.to(device=x_1.device, dtype=torch.float32)
    while sigma_t.ndim < x_1.ndim:
        sigma_t = sigma_t.unsqueeze(-1)
    sigma_t = torch.broadcast_to(sigma_t, x_1.shape)

    source_indices = torch.rand(x_1.shape, device=x_1.device) < sigma_t
    return torch.where(source_indices, x_0, x_1)

class qwen_vl_block_packed_collate_fn:
    def __init__(self, processor, max_len: int = 8192, **kwargs):
        self.processor = processor
        self.max_len = max_len

        self.model_type = kwargs.get("model_type", "qwen3-vl") # qwen3-vl, qwen3-5

        self.image_token_id = processor.tokenizer.encode("<|image_pad|>")[0]
        self.video_token_id = processor.tokenizer.encode("<|video_pad|>")[0]
        self.vision_start_token_id = processor.tokenizer.encode("<|vision_start|>")[0]
        self.mask_token_id = kwargs.get("mask_token_id", 151671) # qwen3-vl
        self.vocab_size = kwargs.get("vocab_size", 151646) # qwen3-vl

        pad_token = kwargs.get("pad_token", "<|im_end|>") # qwen3-vl
        self.pad_token_id = processor.tokenizer.encode(pad_token)[0]
        im_end_token = kwargs.get("im_end_token", "<|im_end|>") # qwen3-vl
        self.im_end_id = processor.tokenizer.encode(im_end_token)[0]

        self.assist_start_ids = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.ignore_index = -100

        noise_scheduler = kwargs.get("noise_scheduler", kwargs.get("path_scheduler", "CondOT"))
        self.noise_scheduler, self.path = _build_noise_scheduler(noise_scheduler)

        self.semantic_top_k = torch.load("qwen3_vl_semantic_noise_k100.pt")["indices"]

    def find_response_span(self, input_ids):
        """
        寻找 Assistant 回复的区间 [start, end)。
        从后往前找(Reverse Search), 因为 Context 中可能包含历史的 Assistant 回复。
        我们需要定位的是最后一个(即当前的)Assistant Header。
        """
        n = len(input_ids)
        m = len(self.assist_start_ids)

        start_idx = -1

        for i in range(n - m, -1, -1):
            # 匹配 <|im_start|>assistant\n
            if input_ids[i : i+m].tolist() == self.assist_start_ids:
                start_idx = i + m # 内容从 header 之后开始
                break

        if start_idx == -1:
            return None, None

        # 在DFM中，我们需要训练所有的response tokens
        # 包括原本的<|im_end|>以及后续随机填充的<|im_end|>(Slack Padding)
        end_idx = -1
        for i in range(start_idx, n):
            if input_ids[i].item() == self.im_end_id:
                end_idx = i + 1 # 包含这个 <|im_end|> 本身
                break

        # 没找到，取到序列末尾进行兜底。
        if end_idx == -1:
            end_idx = n

        return start_idx, end_idx

    def __call__(self, batch):
        batch_labels = []
        batch_input_ids = []
        batch_sample_ids = []
        batch_token_types = []  # 0: prefix, 1: clean, 2: noisy
        batch_block_indices = []
        batch_pos_ids = []
        batch_loss_mask = []
        batch_t = []
        batch_response_mask = []
        batch_response_sample_ids = []

        batch_pixel_values, batch_image_grid_thw = [], []
        batch_pixel_values_videos, batch_video_grid_thw = [], []

        actual_lens = []
        for items in batch:
            # 存储该 packed sequence 内的所有 token 信息
            p_labels, p_ids, p_s_ids, p_t_types, p_b_indices, p_pos, p_lmask, p_rmask, p_rs_ids, p_t = [], [], [], [], [], [], [], [], [], []

            for sample_idx_in_pack, messages in enumerate(items):
                prior_dist = messages[0].pop("prior_dist")
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                # text = re.sub(r"(<\|im_start\|>assistant[\s\S]*?<\|im_end\|>)\n", r"\1", text)
                # 去掉assistant后面多余的\n，不然prompt和response会粘在一起
                text = re.sub(r"(<\|im_start\|>assistant)\n+", r"\1\n", text)
                return_video_metadata = True
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                    return_video_metadata=return_video_metadata,
                    image_patch_size=self.processor.image_processor.patch_size
                )
                video_metadata = None
                if return_video_metadata and video_inputs is not None:
                    video_metadata = [_[1] for _ in video_inputs]
                    video_inputs = [_[0] for _ in video_inputs]

                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=False,
                    return_tensors="pt",
                    video_metadata=video_metadata,
                    **video_kwargs
                )

                raw_ids = inputs.input_ids[0]
                start_idx, end_idx = self.find_response_span(raw_ids)
                if start_idx is None:
                    continue

                # block划分逻辑
                # [prefix (image + prompt) + [clean response blocks] + [noisy response blocks]]
                prefix_ids = raw_ids[:start_idx]
                response_ids = raw_ids[start_idx:]
                l_res_actual = len(response_ids)

                # 计算需要多少个完整的block, 并对齐block size
                num_blocks = (len(response_ids) + self.block_size - 1) // self.block_size
                res_padded = torch.full((num_blocks * self.block_size,), self.pad_token_id) # <|im_end|>
                res_padded[:len(response_ids)] = response_ids

                # 构造noisy部分
                noisy_ids_list = []
                noisy_t_list = []
                # 采样noise level t
                t_low = self.mask_ratio_min if self.mask_ratio_min is not None else (0.0 if prior_dist == "Uniform" else 0.001)
                t_high = self.mask_ratio_max if self.mask_ratio_max is not None else (0.999 if prior_dist == "Uniform" else 1.0)
                if prior_dist == "Uniform":
                    t = torch.rand(num_blocks, device=raw_ids.device).clamp(t_low, t_high)
                elif prior_dist == "Mask":
                    t = torch.rand(num_blocks, device=raw_ids.device).clamp(t_low, t_high)
                else:
                    raise ValueError(f"Unsupported prior_dist '{prior_dist}'")

                antithetic_sampling = True
                if antithetic_sampling:
                    offset = torch.arange(num_blocks, device=raw_ids.device) / num_blocks
                    t = t / num_blocks + offset
                    t_idx = torch.randperm(num_blocks, device=raw_ids.device)
                    t = t[t_idx]

                for i in range(num_blocks):
                    block_clean = res_padded[i * self.block_size : (i + 1) * self.block_size]
                    if prior_dist == "Uniform":    # Uniform Diffusion
                        x_0 = torch.randint_like(block_clean.unsqueeze(0), low=0, high=self.vocab_size)
                        block_noisy = _sample_discrete_noise(
                            t=t[i].unsqueeze(0),
                            x_0=x_0,
                            x_1=block_clean.unsqueeze(0),
                            scheduler=self.noise_scheduler,
                            path=self.path,
                        )[0]
                    elif prior_dist == "Mask":     # MASK Diffusion
                        change_indices = torch.rand(len(block_clean)) <= t[i].repeat(len(block_clean))
                        block_noisy = block_clean.clone()
                        block_noisy = torch.where(change_indices, self.mask_token_id, block_clean)
                    else:
                        raise ValueError(f"Unsupported prior_dist '{prior_dist}'")

                    noisy_ids_list.append(block_noisy)
                    noisy_t_list.append(torch.full((self.block_size,), fill_value=t[i], dtype=torch.float32))

                noisy_ids_flat = torch.cat(noisy_ids_list) # num_blocks * self.block_size
                noisy_t_flat = torch.cat(noisy_t_list)

                # 拼接总序列[prefix] + [clean] + [noisy]
                full_ids = torch.cat([prefix_ids, res_padded, noisy_ids_flat])
                curr_len = len(full_ids)

                l_pre, l_res_padded = len(prefix_ids), len(res_padded)
                if prior_dist == "Uniform":
                    t_pre = torch.ones(l_pre, dtype=torch.float32) * 0.999
                    t_clean = torch.ones(l_res_padded, dtype=torch.float32) * 0.999
                elif prior_dist == "Mask":
                    t_pre = torch.ones(l_pre, dtype=torch.float32) * 0.001
                    t_clean = torch.ones(l_res_padded, dtype=torch.float32) * 0.001
                else:
                    raise ValueError(f"Unsupported prior_dist '{prior_dist}'")
                full_t = torch.cat([t_pre, t_clean, noisy_t_flat])

                # sample ids: 区分packed样本
                s_ids = torch.full((curr_len,), sample_idx_in_pack, dtype=torch.int32)
                rs_ids = torch.full((l_res_padded,), sample_idx_in_pack, dtype=torch.int32)

                # token types: 0=pre, 1=clean, 2=noisy
                t_types = torch.zeros(curr_len, dtype=torch.int8)
                t_types[l_pre : l_pre + l_res_padded] = 1
                t_types[l_pre + l_res_padded:] = 2

                # block indices
                b_indices = torch.zeros(curr_len, dtype=torch.int32)

                # prefix部分的 block id
                num_pre_blocks = (l_pre + self.block_size - 1) // self.block_size
                for b in range(num_pre_blocks):
                    start = b * self.block_size
                    end = min((b + 1) * self.block_size, l_pre) # 不是block size的整数倍时，最后一个block偏小
                    b_indices[start:end] = b
                # num_pre_blocks = 0 # 对于response，prefix tokens完全可见

                for b in range(num_blocks):
                    curr_block_id = num_pre_blocks + b
                    # clean部分的 block id
                    b_indices[l_pre + b * self.block_size: l_pre + (b + 1) * self.block_size] = curr_block_id
                    # noisy部分的 block id
                    b_indices[l_pre + l_res_padded + b * self.block_size: l_pre + l_res_padded + (b + 1) * self.block_size] = curr_block_id

                # position_ids复制逻辑, 先获取 prefix + clean 的 3d position_ids
                pos_pre_clean, _ = get_rope_index(
                    input_ids=torch.cat([prefix_ids, res_padded]).unsqueeze(0),
                    image_grid_thw=inputs.get("image_grid_thw", None),
                    video_grid_thw=inputs.get("video_grid_thw", None),
                    attention_mask=torch.ones(1, l_pre + l_res_padded),
                    image_token_id=self.image_token_id,
                    video_token_id=self.video_token_id,
                    vision_start_token_id=self.vision_start_token_id
                ) # [3, b=1, seq_len]
                pos_noisy = pos_pre_clean[..., l_pre:] # 复制 clean 的位置
                full_pos = torch.cat([pos_pre_clean, pos_noisy], dim=-1)

                response_mask = torch.zeros(curr_len, dtype=bool)
                response_mask[l_pre + l_res_padded:] = True
                loss_mask = torch.zeros(curr_len, dtype=torch.bool)
                if prior_dist == "Uniform":    # 全部response部分计算loss
                    loss_mask[l_pre + l_res_padded:] = True
                elif prior_dist == "Mask":     # 只<MASK>部分计算loss
                    loss_mask[full_ids == self.mask_token_id] = True

                # 收集当前sample的所有tensor
                p_labels.append(torch.cat([prefix_ids, res_padded, res_padded]))
                # p_labels.append(res_padded)
                p_ids.append(full_ids)
                p_s_ids.append(s_ids)
                p_t_types.append(t_types)
                p_b_indices.append(b_indices)
                p_pos.append(full_pos)
                p_lmask.append(loss_mask)
                p_rmask.append(response_mask)
                p_rs_ids.append(rs_ids)
                p_t.append(full_t)
                # p_t.append(noisy_t_flat)

                # 收集视觉特征
                if inputs.get("pixel_values") is not None:
                    batch_pixel_values.append(inputs.pixel_values)
                    batch_image_grid_thw.append(inputs.image_grid_thw)
                if inputs.get("pixel_values_videos") is not None:
                    batch_pixel_values_videos.append(inputs.pixel_values_videos)
                    batch_video_grid_thw.append(inputs.video_grid_thw)

            batch_input_ids.append(torch.cat(p_ids))
            batch_labels.append(torch.cat(p_labels))
            batch_pos_ids.append(torch.cat(p_pos, dim=-1))
            batch_sample_ids.append(torch.cat(p_s_ids))
            batch_token_types.append(torch.cat(p_t_types))
            batch_block_indices.append(torch.cat(p_b_indices))
            batch_loss_mask.append(torch.cat(p_lmask))
            batch_t.append(torch.cat(p_t))
            batch_response_sample_ids.append(torch.cat(p_rs_ids))
            batch_response_mask.append(torch.cat(p_rmask))
            actual_lens.append(batch_input_ids[-1].shape[-1])

        max_len = max(actual_lens)
        pad_lens = max_len - np.array(actual_lens)
        for idx in range(len(batch_input_ids)):
            batch_input_ids[idx] = torch.cat([batch_input_ids[idx], torch.full((pad_lens[idx],), fill_value=self.pad_token_id)])
            batch_labels[idx] = torch.cat([batch_labels[idx], torch.full((pad_lens[idx],), fill_value=self.pad_token_id)])
            batch_pos_ids[idx] = torch.cat([batch_pos_ids[idx], torch.zeros((3, 1, pad_lens[idx]))], dim=-1)
            batch_sample_ids[idx] = torch.cat([batch_sample_ids[idx], torch.full((pad_lens[idx],), fill_value=-1)])
            batch_token_types[idx] = torch.cat([batch_token_types[idx], torch.full((pad_lens[idx],), fill_value=-1)])
            batch_block_indices[idx] = torch.cat([batch_block_indices[idx], torch.full((pad_lens[idx],), fill_value=-1)])
            batch_loss_mask[idx] = torch.cat([batch_loss_mask[idx], torch.full((pad_lens[idx],), fill_value=0)])
            batch_t[idx] = torch.cat([batch_t[idx], torch.full((pad_lens[idx],), fill_value=0.5)]) # trick
            # batch_response_sample_ids[idx] = torch.cat([batch_response_sample_ids[idx], torch.full((pad_lens[idx],), fill_value=-1)])
            batch_response_mask[idx] = torch.cat([batch_response_mask[idx], torch.full((pad_lens[idx],), fill_value=False)])

        # 最终聚合
        results = {
            "input_ids": torch.stack(batch_input_ids),                          # [B, seq_len]
            "labels": torch.stack(batch_labels),                                # [B, seq_len]
            "position_ids": torch.cat(batch_pos_ids, dim=1).to(torch.int64),    # [3, B, seq_len]
            "response_mask": torch.stack(batch_response_mask),                  # [B, seq_len]
            "loss_mask": torch.stack(batch_loss_mask).to(torch.bool),           # [B, seq_len]
            "t": torch.stack(batch_t),                                          # [B, seq_len]
            # "block_metadata": {                                                 # [B, seq_len]
            #     "sample_ids": torch.stack(batch_sample_ids),
            #     "token_types": torch.stack(batch_token_types),
            #     "block_ids": torch.stack(batch_block_indices),
            # },
            "num_samples": torch.stack([torch.tensor(len(items)) for items in batch]),
        }

        if batch_pixel_values:
            results["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            results["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)
        if batch_pixel_values_videos:
            results["pixel_values_videos"] = torch.cat(batch_pixel_values_videos, dim=0)
            results["video_grid_thw"] = torch.cat(batch_video_grid_thw, dim=0)

        return results

class wam_diff2_block_collate_fn:
    def __init__(self, processor, **kwargs):
        # no sequence packing version.
        self.processor = processor
        self.max_len = kwargs.get("max_len", 8192)
        self.model_type = kwargs.get("model_type", "qwen3-vl") # qwen3-vl, qwen3-5

        self.image_token_id = processor.tokenizer.encode("<|image_pad|>")[0]
        self.video_token_id = processor.tokenizer.encode("<|video_pad|>")[0]
        self.vision_start_token_id = processor.tokenizer.encode("<|vision_start|>")[0]
        self.mask_token_id = kwargs.get("mask_token_id", 151671) # qwen3-vl
        self.vocab_size = kwargs.get("vocab_size", 151646) # qwen3-vl

        pad_token = kwargs.get("pad_token", "<|im_end|>") # qwen3-vl
        self.pad_token_id = processor.tokenizer.encode(pad_token)[0]
        im_end_token = kwargs.get("im_end_token", "<|im_end|>") # qwen3-vl
        self.im_end_id = processor.tokenizer.encode(im_end_token)[0]

        self.assist_start_ids = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.ignore_index = -100

        self.block_size = kwargs.get("block_size", 4)
        self.mask_ratio_min = kwargs.get("mask_ratio_min", None)
        self.mask_ratio_max = kwargs.get("mask_ratio_max", None)
        # distil: teacher 的 block_size，用于构造 teacher 专用的 attention_mask
        # teacher 和 student 共享相同的 input_ids / t / loss_mask / position_ids（同一个 x_t），
        # 但 attention 按 teacher 训练时的 block granularity 排列。
        # 例如 student block_size=8, teacher_block_size=4，一个 student block 对应两个 teacher block。
        # 设为 None 时不生成 teacher_attention_mask（sft 训练不受影响）。
        self.teacher_block_size = kwargs.get("teacher_block_size", None)
        noise_scheduler = kwargs.get("noise_scheduler", kwargs.get("path_scheduler", "CondOT"))
        # self.prior_dist = kwargs.get("prior_dist", "Mask")
        # assert self.prior_dist in ["Uniform", "Mask"], f"{self.prior_dist} not support"
        self.noise_scheduler, self.path = _build_noise_scheduler(noise_scheduler)

        semantic_file = "qwen3_vl_semantic_noise_k100.pt"
        if os.path.exists(semantic_file):
            self.semantic_top_k = torch.load("qwen3_vl_semantic_noise_k100.pt")["indices"][:self.vocab_size,]
        else:
            self.semantic_top_k = None

        self._printed_debug_info = False

    def _sample_semantic_or_uniform_token(self, token_id: int) -> int:
        if self.semantic_top_k is not None and 0 <= token_id < self.vocab_size:
            candidates = self.semantic_top_k[token_id]
            sampled_idx = torch.randint(low=0, high=candidates.numel(), size=(1,)).item()
            return int(candidates[sampled_idx].item())
        return int(torch.randint(low=0, high=self.vocab_size, size=(1,)).item())

    def _sample_block_times(self, num_blocks: int, device: torch.device, *, low: float, high: float) -> torch.Tensor:
        t = torch.rand(num_blocks, device=device).clamp(low, high)
        if num_blocks > 0:
            offset = torch.arange(num_blocks, device=device, dtype=t.dtype) / num_blocks
            t = t / num_blocks + offset
            t = t[torch.randperm(num_blocks, device=device)]
        return t

    def _sample_mix_noise(self, response_padded: torch.Tensor, num_res_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = response_padded.device
        if num_res_blocks == 0:
            empty = response_padded.new_empty((0,))
            return empty, torch.empty((0,), dtype=torch.float32, device=device)

        t = self._sample_block_times(num_res_blocks, device=device, low=0.0, high=1.0).repeat_interleave(self.block_size)
        noisy_ids = []
        noisy_t = []

        for idx, token in enumerate(response_padded.tolist()):
            phi = torch.pi / 2 * t[idx]
            probs = torch.stack(
                [
                    torch.clamp(1 - torch.cos(phi), 0.001, 1.0),
                    torch.clamp(torch.sin(phi) + torch.cos(phi) - 1, 0.001, 1.0),
                    torch.clamp(1 - torch.sin(phi), 0.001, 1.0),
                ]
            )
            data_type = torch.multinomial(probs, num_samples=1, replacement=True).item()

            if data_type == 0:
                noisy_ids.append(self.mask_token_id)
                noisy_t.append(1.0)
            elif data_type == 1:
                op_probs = torch.tensor([0.1, 0.1, 0.8], dtype=torch.float32)
                op_type = torch.multinomial(op_probs, num_samples=1, replacement=True).item()
                if op_type == 0:
                    continue
                replacement = self._sample_semantic_or_uniform_token(token)
                if op_type == 1:
                    noisy_ids.append(replacement)
                    noisy_t.append(1.0)
                    noisy_ids.append(token)
                    noisy_t.append(1.0)
                else:
                    noisy_ids.append(replacement if replacement < self.vocab_size else token)
                    noisy_t.append(1.0)
            else:
                noisy_ids.append(token)
                noisy_t.append(1.0)

        target_len = response_padded.numel()
        noisy_ids = noisy_ids[:target_len]
        noisy_t = noisy_t[:target_len]
        if len(noisy_ids) < target_len:
            pad_len = target_len - len(noisy_ids)
            noisy_ids.extend([self.pad_token_id] * pad_len)
            noisy_t.extend([1.0] * pad_len)

        return (
            torch.tensor(noisy_ids, device=device, dtype=response_padded.dtype),
            torch.tensor(noisy_t, device=device, dtype=torch.float32),
        )

    def _sample_edit_noise(self, response_padded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = response_padded.device
        alpha_t = (0.05 + torch.rand(1, device=device) * 0.15).item()
        target_len = response_padded.numel()
        num_edits = round(target_len * alpha_t)
        op_probs = torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32, device=device)
        op_probs = op_probs / op_probs.sum()
        tokens = response_padded.tolist()

        for _ in range(num_edits):
            if not tokens:
                break
            op_type = torch.multinomial(op_probs, num_samples=1, replacement=True).item()
            idx = torch.randint(0, len(tokens), (1,), device=device).item()

            if op_type == 0:
                tokens.pop(idx)
            elif op_type == 1:
                tokens.insert(idx, int(torch.randint(low=0, high=self.vocab_size, size=(1,), device=device).item()))
            else:
                tokens[idx] = int(torch.randint(low=0, high=self.vocab_size, size=(1,), device=device).item())

        tokens = tokens[:target_len]
        if len(tokens) < target_len:
            tokens.extend([self.pad_token_id] * (target_len - len(tokens)))

        return (
            torch.tensor(tokens, device=device, dtype=response_padded.dtype),
            torch.full((target_len,), fill_value=alpha_t, device=device, dtype=torch.float32),
        )

    def find_response_span(self, input_ids):
        n = len(input_ids)
        header_prefix = self.assist_start_ids[:2] #换行？空格？
        m = len(header_prefix)

        start_idx = -1 #<|im_start|>的位置

        # input_ids: action for the next 4 seconds with 8 new waypoints.<|im_end|>\n<|im_start|>assistant\n4.13,-0.03,8.01,-0.08,11.63,-0.10,14.99,-0.12,18.07,-0.14,20.88,-0.16,23.42,-0.17,25.72,-0.15<|im_end|>\n'

        for i in range(n - m, -1, -1): #从 n - m 倒序遍历到 0
            # 匹配 <|im_start|>assistant\n
            if input_ids[i : i+m].tolist() == header_prefix:
                # 找到了 <|im_start|>assistant
                # 我们跳过这个 header 以及紧随其后的“换行/格式”Token
                # 无论它是 \n, \n\n 还是 .\n
                start_idx = i + m + 1 # 内容从 header 之后开始
                break

        if start_idx == -1:
            return None, None

        # 在DFM中，我们需要训练所有的response tokens
        # 包括原本的<|im_end|>以及后续随机填充的<|im_end|>(Slack Padding)
        end_idx = -1
        for i in range(start_idx, n):
            if input_ids[i].item() == self.im_end_id:
                end_idx = i + 1 # 包含这个 <|im_end|> 本身
                break

        # 没找到，取到序列末尾进行兜底。
        if end_idx == -1:
            end_idx = n

        return start_idx, end_idx

    def create_attention_mask(self, prefix_len, res_len, block_size, max_len):
        seq_len = min(prefix_len + 2 * res_len, max_len)
        mask = torch.zeros((max_len, max_len), dtype=torch.bool)
        if seq_len <= 0:
            return mask.unsqueeze(0).unsqueeze(0)

        block_ids = torch.empty((seq_len,), dtype=torch.long)
        token_types = torch.empty((seq_len,), dtype=torch.int8)

        clean_start = min(prefix_len, seq_len)
        clean_end = min(prefix_len + res_len, seq_len)
        noisy_start = clean_end

        block_ids[:clean_start] = torch.arange(clean_start) // block_size
        token_types[:clean_start] = 0

        if clean_end > clean_start:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            clean_len = clean_end - clean_start
            clean_rel_positions = torch.arange(clean_len)
            clean_blocks = num_prefix_blocks + clean_rel_positions // block_size
            block_ids[clean_start:clean_end] = clean_blocks
            token_types[clean_start:clean_end] = 1

        if noisy_start < seq_len:
            num_prefix_blocks = (prefix_len + block_size - 1) // block_size
            noisy_len = seq_len - noisy_start
            noisy_rel_positions = torch.arange(noisy_len)
            noisy_blocks = num_prefix_blocks + noisy_rel_positions // block_size
            block_ids[noisy_start:seq_len] = noisy_blocks
            token_types[noisy_start:seq_len] = 2

        q_blocks = block_ids.view(-1, 1)
        k_blocks = block_ids.view(1, -1)
        q_types = token_types.view(-1, 1)
        k_types = token_types.view(1, -1)

        # block causal for prefix + clean blocks.
        base_mask = q_blocks >= k_blocks
        # attention pattern for noisy blocks.
        noisy_query_mask = q_types == 2
        noisy_visibility = (k_types == 0) | ((k_types == 1) & (k_blocks < q_blocks)) | ((k_types == 2) & (k_blocks == q_blocks))
        local_mask = torch.where(noisy_query_mask, noisy_visibility, base_mask)
        mask[:seq_len, :seq_len] = local_mask

        return mask.unsqueeze(0).unsqueeze(0)

    def __call__(self, batch):
        batch_labels = []
        batch_input_ids = []
        batch_pos_ids = []
        batch_loss_mask = []
        batch_response_mask = []
        batch_t = []
        batch_token_types = []  # 0: prefix, 1: clean, 2: noisy
        batch_block_indices = []
        actual_lens = []
        len_pre_list = []
        len_res_list = []

        batch_pixel_values, batch_image_grid_thw = [], []
        batch_pixel_values_videos, batch_video_grid_thw = [], []

        for messages in batch:
            prior_dist = messages[0].get("prior_dist", "Mask")
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            # 去掉assistant后面多余的\n，不然prompt和response会粘在一起
            text = re.sub(r"(<\|im_start\|>assistant)\n+", r"\1\n", text)
            return_video_metadata = True
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=return_video_metadata,
                image_patch_size=self.processor.image_processor.patch_size
            )
            video_metadata = None
            if return_video_metadata and video_inputs is not None:
                video_metadata = [_[1] for _ in video_inputs]
                video_inputs = [_[0] for _ in video_inputs]

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
                video_metadata=video_metadata,
                **video_kwargs
            )
            if not self._printed_debug_info:
                print(f"image_grid_thw: {inputs.get('image_grid_thw', None)}")
                self._printed_debug_info = True

            raw_ids = inputs.input_ids[0]
            if os.environ.get("DEBUG", "").lower() in ("1", "true"):
                decoded = self.processor.tokenizer.decode(raw_ids, skip_special_tokens=False)
                # print(f"[DEBUG] input_ids len={len(raw_ids)}:\n{decoded}\n{'='*80}")
            start_idx, end_idx = self.find_response_span(raw_ids)
            if start_idx is None:
                continue

            # block划分逻辑
            # [prefix (image + prompt) + [clean response blocks] + [noisy response blocks]]
            prefix_ids = raw_ids[:start_idx] #<|im_start|>user\nHere is a ... waypoints.<|im_end|>\n
            response_ids = raw_ids[start_idx:end_idx] #<|im_start|>assistant\n4.13,-0.03,8.01,-0.08,11.63,-0.10,14.99,-0.12,18.07,-0.14,20.88,-0.16,23.42,-0.17,25.72,-0.15<|im_end|>
            len_pre = len(prefix_ids)

            num_res_blocks = (len(response_ids) + self.block_size - 1) // self.block_size
            response_padded = torch.full(
                (num_res_blocks * self.block_size,),
                self.pad_token_id,
                device=raw_ids.device,
                dtype=raw_ids.dtype,
            )
            response_padded[:len(response_ids)] = response_ids
            len_res_padded = len(response_padded)

            # 构造noisy部分
            if prior_dist == "Mix":
                noisy_response_ids, noisy_t = self._sample_mix_noise(response_padded, num_res_blocks)

            elif prior_dist == "Edit":
                noisy_response_ids, noisy_t = self._sample_edit_noise(response_padded)
            else:
                noisy_ids_list = []
                noisy_t_list = []
                # 采样noise level t
                if prior_dist == "Uniform":
                    t = self._sample_block_times(num_res_blocks, device=raw_ids.device, low=0.0, high=0.999)
                elif prior_dist == "Mask":
                    t = self._sample_block_times(num_res_blocks, device=raw_ids.device, low=0.001, high=1.0)
                else:
                    raise ValueError(f"Unsupported prior_dist '{prior_dist}'")

                for i in range(num_res_blocks):
                    block_clean = response_padded[i * self.block_size: (i + 1) * self.block_size]
                    if prior_dist == "Uniform": # Uniform Diffusion
                        x_0 = torch.randint_like(block_clean.unsqueeze(0), low=0, high=self.vocab_size)
                        block_noisy = _sample_discrete_noise(
                            t=t[i].unsqueeze(0),
                            x_0=x_0,
                            x_1=block_clean.unsqueeze(0),
                            scheduler=self.noise_scheduler,
                            path=self.path,
                        )[0]
                    elif prior_dist == "Mask": # Mask Diffusion
                        change_indices = torch.rand(len(block_clean), device=block_clean.device) <= t[i].repeat(len(block_clean))
                        block_noisy = block_clean.clone()
                        block_noisy = torch.where(change_indices, self.mask_token_id, block_clean) #change_indices部分变成mask
                    else:
                        raise ValueError(f"Unsupported prior_dist '{prior_dist}'")

                    noisy_ids_list.append(block_noisy)
                    noisy_t_list.append(
                        torch.full(
                            (self.block_size,),
                            fill_value=float(t[i].item()),
                            dtype=torch.float32,
                            device=raw_ids.device,
                        )
                    )

                noisy_response_ids = torch.cat(noisy_ids_list)
                noisy_t = torch.cat(noisy_t_list)

            # 拼接输入序列 prefix + clean + noisy
            full_input_ids = torch.cat([prefix_ids, response_padded, noisy_response_ids])
            full_labels = torch.cat([prefix_ids, response_padded, response_padded])

            if prior_dist == "Uniform":
                prefix_t = torch.full((len(prefix_ids),), fill_value=0.999, dtype=torch.float32, device=raw_ids.device)
                clean_t = torch.full((len(response_padded),), fill_value=0.999, dtype=torch.float32, device=raw_ids.device)
            elif prior_dist in ["Mask", "Mix", "Edit"]:
                prefix_t = torch.full((len(prefix_ids),), fill_value=0.001, dtype=torch.float32, device=raw_ids.device)
                clean_t = torch.full((len(response_padded),), fill_value=0.001, dtype=torch.float32, device=raw_ids.device)
            else:
                raise ValueError(f"Unsupported prior_dist '{prior_dist}'")
            full_t = torch.cat([prefix_t, clean_t, noisy_t])

            # token types: 0=prefix, 1=clean, 2=noisy
            t_types = torch.zeros(len(full_input_ids), dtype=torch.int8, device=raw_ids.device)
            t_types[len_pre : len_pre + len_res_padded] = 1 #clean部分的token_types
            t_types[len_pre + len_res_padded:] = 2 #noisy部分的token_types

            block_ids = torch.zeros(len(full_input_ids), dtype=torch.int32, device=raw_ids.device)
            # prefix部分的block_ids
            num_prefix_blocks = (len(prefix_ids) + self.block_size - 1) // self.block_size
            for b in range(num_prefix_blocks):
                start = b * self.block_size
                end = min((b + 1) * self.block_size, len(prefix_ids)) # 不是block size的整数倍时，最后一个block偏小
                block_ids[start: end] = b #给prefix部分的每个token分配一个block_id

            for b in range(num_res_blocks):
                curr_block_id = num_prefix_blocks + b
                # clean部分的block_id
                block_ids[len_pre + b * self.block_size: len_pre + (b + 1) * self.block_size] = curr_block_id
                # noisy部分的block_id
                block_ids[len_pre + len_res_padded + b * self.block_size: len_pre + len_res_padded + (b + 1) * self.block_size] = curr_block_id

            #  position_ids复制逻辑, 先获取 prefix + clean 的 3d position_ids
            pos_pre_clean, _ = get_rope_index(
                    input_ids=torch.cat([prefix_ids, response_padded]).unsqueeze(0),
                    image_grid_thw=inputs.get("image_grid_thw", None),
                    video_grid_thw=inputs.get("video_grid_thw", None),
                    attention_mask=torch.ones(1, len_pre + len_res_padded, device=raw_ids.device),
                    image_token_id=self.image_token_id,
                    video_token_id=self.video_token_id,
                    vision_start_token_id=self.vision_start_token_id
            ) # [3, b=1, seq_len]
            pos_noisy = pos_pre_clean[..., len_pre:] # 复制 clean 的位置
            full_pos_ids = torch.cat([pos_pre_clean, pos_noisy], dim=-1)

            response_mask = torch.zeros(len(full_labels), dtype=torch.bool, device=raw_ids.device)
            response_mask[len_pre + len_res_padded: len_pre + 2 * len_res_padded] = True # [B=1, seq_len]

            loss_mask = torch.zeros(len(full_input_ids), dtype=torch.bool, device=raw_ids.device)
            if prior_dist in ["Uniform", "Edit"]:   # 全部response部分计算loss
                loss_mask[len_pre + len_res_padded:] = True
            elif prior_dist == "Mix":
                loss_mask = torch.where(full_input_ids == full_labels, False, True)
            elif prior_dist == "Mask":              # 只<MASK>部分计算loss
                loss_mask[full_input_ids == self.mask_token_id] = True

                if os.environ.get("DEBUG_TOKENS", "").lower() in ("1", "true") and not getattr(self, "_printed_debug_tokens", False):
                    self._printed_debug_tokens = True
                    tok = self.processor.tokenizer
                    print(f"[DEBUG_TOKENS] === sample {len(batch_input_ids)} ===")
                    print(f"[DEBUG_TOKENS] prefix ({len_pre} tokens):\n  {tok.decode(prefix_ids, skip_special_tokens=False)}")
                    print(f"[DEBUG_TOKENS] clean response ({len_res_padded} tokens):\n  {tok.decode(response_padded, skip_special_tokens=False)}")
                    print(f"[DEBUG_TOKENS] noisy response ({len(noisy_response_ids)} tokens):\n  {tok.decode(noisy_response_ids, skip_special_tokens=False)}")
                    print(f"[DEBUG_TOKENS] labels:\n  {tok.decode(full_labels, skip_special_tokens=False)}")
                    print(f"[DEBUG_TOKENS] loss_mask sum={loss_mask.sum().item()}/{loss_mask.numel()}, t range=[{full_t.min():.3f}, {full_t.max():.3f}]")
                    print(f"[DEBUG_TOKENS] {'='*60}")

            batch_labels.append(full_labels)
            batch_input_ids.append(full_input_ids)
            batch_pos_ids.append(full_pos_ids)
            batch_loss_mask.append(loss_mask)
            batch_t.append(full_t)
            batch_token_types.append(t_types)
            batch_block_indices.append(block_ids)
            batch_response_mask.append(response_mask)
            actual_lens.append(len(full_input_ids))
            len_pre_list.append(len_pre)
            len_res_list.append(len_res_padded)

            if inputs.get("pixel_values") is not None:
                batch_pixel_values.append(inputs.pixel_values)
                batch_image_grid_thw.append(inputs.image_grid_thw)
            if inputs.get("pixel_values_videos") is not None:
                batch_pixel_values_videos.append(inputs.pixel_values_videos)
                batch_video_grid_thw.append(inputs.video_grid_thw)

        if len(actual_lens) == 0:
            raise ValueError("actual_lens is empty")

        max_len = min(max(actual_lens), self.max_len)
        batch_masks = []
        batch_teacher_masks = []

        for idx in range(len(batch_input_ids)):
            diff_len = max_len - actual_lens[idx]

            if diff_len < 0:
                # 截断：当前序列超过了 self.max_len
                batch_input_ids[idx] = batch_input_ids[idx][:max_len]
                batch_labels[idx] = batch_labels[idx][:max_len]
                batch_pos_ids[idx] = batch_pos_ids[idx][..., :max_len]
                batch_response_mask[idx] = batch_response_mask[idx][:max_len]
                batch_loss_mask[idx] = batch_loss_mask[idx][:max_len]
                batch_t[idx] = batch_t[idx][:max_len]
                batch_token_types[idx] = batch_token_types[idx][:max_len]
                batch_block_indices[idx] = batch_block_indices[idx][:max_len]
            elif diff_len > 0:
                # Padding：当前序列短于目标 max_len
                batch_input_ids[idx] = torch.cat([
                    batch_input_ids[idx],
                    torch.full((diff_len,), fill_value=self.pad_token_id, device=batch_input_ids[idx].device, dtype=batch_input_ids[idx].dtype),
                ])
                batch_labels[idx] = torch.cat([
                    batch_labels[idx],
                    torch.full((diff_len,), fill_value=self.pad_token_id, device=batch_labels[idx].device, dtype=batch_labels[idx].dtype),
                ])
                batch_pos_ids[idx] = torch.cat([
                    batch_pos_ids[idx],
                    torch.zeros((3, 1, diff_len), device=batch_pos_ids[idx].device, dtype=batch_pos_ids[idx].dtype),
                ], dim=-1)
                batch_response_mask[idx] = torch.cat([
                    batch_response_mask[idx],
                    torch.zeros((diff_len,), device=batch_response_mask[idx].device, dtype=torch.bool),
                ])
                batch_loss_mask[idx] = torch.cat([
                    batch_loss_mask[idx],
                    torch.zeros((diff_len,), device=batch_loss_mask[idx].device, dtype=torch.bool),
                ])
                batch_t[idx] = torch.cat([
                    batch_t[idx],
                    torch.full((diff_len,), fill_value=0.5, device=batch_t[idx].device, dtype=batch_t[idx].dtype),
                ])
                batch_token_types[idx] = torch.cat([
                    batch_token_types[idx],
                    torch.full((diff_len,), fill_value=-1, device=batch_token_types[idx].device, dtype=batch_token_types[idx].dtype),
                ])
                batch_block_indices[idx] = torch.cat([
                    batch_block_indices[idx],
                    torch.full((diff_len,), fill_value=-1, device=batch_block_indices[idx].device, dtype=batch_block_indices[idx].dtype),
                ])

            # 生成对应的 attention mask
            batch_masks.append(self.create_attention_mask(len_pre_list[idx], len_res_list[idx], self.block_size, max_len))
            batch_teacher_masks.append(self.create_attention_mask(len_pre_list[idx], len_res_list[idx], self.block_size/2, max_len))
        # distil: 生成 teacher 专用的 attention_mask（按 teacher_block_size 的粒度）
        # teacher 和 student 共享相同的 input_ids / t / loss_mask / position_ids，
        # 只有 attention 的 block 划分粒度不同


        results = {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "position_ids": torch.cat(batch_pos_ids, dim=1).to(torch.int64),    # [3, B, max_len]
            "response_mask": torch.stack(batch_response_mask),                  # [B, seq_len]
            "loss_mask": torch.stack(batch_loss_mask).to(torch.bool),
            "attention_mask": torch.cat(batch_masks, dim=0),
            "t": torch.stack(batch_t),                                          # [B, max_len]
            # "block_metadata": {                                               # [B, seq_len]
            #     "token_types": torch.stack(batch_token_types),
            #     "block_ids": torch.stack(batch_block_indices),
            # },
            "block_size": self.block_size,
            "num_samples": torch.tensor(len(batch)),
        }

        # distil: teacher 专用的 attention_mask（只在 teacher_block_size 不为 None 时生成）
        if batch_teacher_masks is not None:
            results["teacher_attention_mask"] = torch.cat(batch_teacher_masks, dim=0)
            results["teacher_block_size"] = self.teacher_block_size

        # === 新增：保留原始 messages，供 on-policy generate 使用 ===
        results["raw_messages"] = list(batch)
        # ===========================================================

        if batch_pixel_values:
            results["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            results["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)  # [B, 3]
        if batch_pixel_values_videos:
            results["pixel_values_videos"] = torch.cat(batch_pixel_values_videos, dim=0)
            results["video_grid_thw"] = torch.cat(batch_video_grid_thw, dim=0)  # [B, 3]

        # DEBUG: 打印 results 中所有 tensor 的 shape
        if os.environ.get("DEBUG", "").lower() in ("1", "true"):
            for k, v in results.items():
                if isinstance(v, torch.Tensor):
                    print(f"[DEBUG collate] {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"[DEBUG collate] {k}: type={type(v).__name__}, value={v}")

        return results


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3OmniMoeProcessor": qwen3_omni_collate_fn,
    "default": default_collate_fn,
}
