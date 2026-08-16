# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import io
import json
import random
import re
import torch
from glob import glob
from PIL import Image
import bisect
import pyarrow.parquet as pq
from multiprocessing import Pool
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

class qwen3_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.ds = json.load(open(path_or_dataset))
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        self.min_pixels = kwargs.get("min_pixels", 384*384)
        self.max_pixels = kwargs.get("max_pixels", 512*512)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]['conversation']

        content = sample[1]['content'][0]
        if content.get("image", None) is not None:
            image_path = os.path.join(self.root_dir, sample[1]['content'][0]['image'])
            assert os.path.exists(image_path), f"{image_path} not exist"
            sample[1]['content'][0]['image'] = Image.open(image_path).convert("RGB") # image字段内容转为PIL.Image
            sample[1]['content'][0]['min_pixels'] = self.min_pixels # 不设定的话，代码中默认值是4*32*32
            sample[1]['content'][0]['max_pixels'] = self.max_pixels # 不设定的话，代码中默认值是16384*32*32

        return sample

class qwen2_5_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.ds = json.load(open(path_or_dataset))
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        self.min_pixels = kwargs.get("min_pixels", 384*384)
        self.max_pixels = kwargs.get("max_pixels", 512*512)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]['conversation']

        content = sample[1]['content'][0]
        if content.get("image", None) is not None:
            image_path = os.path.join(self.root_dir, sample[1]['content'][0]['image'])
            assert os.path.exists(image_path), f"{image_path} not exist"
            sample[1]['content'][0]['image'] = Image.open(image_path) # image字段内容转为PIL.Image
            # sample[1]['content'][0]['image'] = Image.open(image_path).convert("RGB") # image字段内容转为PIL.Image
            sample[1]['content'][0]['min_pixels'] = self.min_pixels # 不设定的话，代码中默认值是4*32*32
            sample[1]['content'][0]['max_pixels'] = self.max_pixels # 不设定的话，代码中默认值是16384*32*32

        return sample

class qwen2_5_packed_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.ds = load_from_disk(path_or_dataset)
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        self.min_pixels = kwargs.get("min_pixels", 384*384)
        self.max_pixels = kwargs.get("max_pixels", 512*512)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        packed_samples = item["packed_samples"]

        for sample in packed_samples:
            content = sample[1]['content'][0]
            if content.get("image", None) is not None:
                image_path = os.path.join(self.root_dir, sample[1]['content'][0]['image'])
                assert os.path.exists(image_path), f"{image_path} not exist"
                sample[1]['content'][0]['image'] = Image.open(image_path) # image字段内容转为PIL.Image
                # sample[1]['content'][0]['image'] = Image.open(image_path).convert("RGB") # image字段内容转为PIL.Image
                sample[1]['content'][0]['min_pixels'] = self.min_pixels # 不设定的话，代码中默认值是4*32*32
                sample[1]['content'][0]['max_pixels'] = self.max_pixels # 不设定的话，代码中默认值是16384*32*32

        # 移除None field.
        packed_samples = [
            [
                {
                    "role": msg["role"],
                    "content": [{k: v for k, v in item.items() if v is not None} for item in msg["content"]]
                }
                for msg in sample
            ]
            for sample in packed_samples
        ]

        return packed_samples

class qwen_vl_packed_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.ds = load_from_disk(path_or_dataset)
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        self.max_len = kwargs.get("max_len", 8192)

        self.min_pixels = kwargs.get("min_pixels", 8*8*32*32)
        self.max_pixels = kwargs.get("max_pixels", 64*64*32*32)
        self.video_min_pixels = 64 * 64     # 4 tokens.
        self.video_max_pixels = 1024 * 1024 # 1024 tokens.
        self.video_total_pixels = self.max_len * 32 * 32 * 0.9
        print(f"Load {len(self.ds)} samples")

    def get_metadata(self):
        return None
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        packed_samples = item["packed_samples"]

        for sample in packed_samples:
            for turn in sample:
                content_list = turn.get('content', [])
                if not isinstance(content_list, list):
                    continue

                for content in content_list:
                    # --- 图像类型 ---
                    if content.get("image") is not None:
                        image_path = content['image']
                        full_path = os.path.join(self.root_dir, image_path)
                        # 移除不需要传给 processor 的元数据
                        content.pop('height', None)
                        content.pop('width', None)
                        content['image'] = Image.open(full_path).convert("RGB")
                        content['min_pixels'] = self.min_pixels
                        content['max_pixels'] = self.max_pixels

                    elif content.get("video") is not None:
                        video_path = content["video"]
                        full_path = os.path.join(self.root_dir, video_path)

                        content.pop('height', None)
                        content.pop('width', None)
                        content.pop('num_frames', None)
                        # 视频传路径，由 processor 内部的 fetch_video 处理抽帧
                        content["video"] = full_path
                        content['min_pixels'] = self.video_min_pixels
                        content['max_pixels'] = self.video_max_pixels
                        # total_pixels控制总预算
                        content['total_pixels'] = self.video_total_pixels

        # 移除None field.
        packed_samples = [
            [
                {
                    "role": msg["role"],
                    "content": [{k: v for k, v in item.items() if v is not None} for item in msg["content"]]
                }
                for msg in sample
            ]
            for sample in packed_samples
        ]

        return packed_samples

class qwen_vl_dataset(torch.utils.data.Dataset):
    def __init__(self, path_or_dataset, **kwargs):
        self.root_dir = kwargs.get("root_dir", None)
        assert self.root_dir is not None and os.path.exists(self.root_dir), f"{self.root_dir}"

        print(f"Loading dataset from {path_or_dataset}...")
        self.dataset = load_from_disk(path_or_dataset)

        column_names = self.dataset.column_names
        self.lengths = self.dataset.data.column("length").to_numpy() if "length" in column_names else None
        self.v_tokens = self.dataset.data.column("v_tokens").to_numpy() if "v_tokens" in column_names else None

        self.max_len = kwargs.get("max_len", 8192)
        self.prior_dist = kwargs.get("prior_dist", "Mask")
        # switch noisy prior dist for curriculum learning
        self.prior_dist_2 = kwargs.get("prior_dist_2", None)
        self.switch_prior_thresh = kwargs.get("switch_prior_thresh", 1.0)

        self.min_pixels = kwargs.get("min_pixels", 8*8*32*32)
        self.max_pixels = kwargs.get("max_pixels", 64*64*32*32)
        self.video_min_pixels = 64 * 64     # 4 tokens.
        self.video_max_pixels = 1024 * 1024 # 1024 tokens.
        self.video_total_pixels = self.max_len * 32 * 32 * 0.9
        print(f"Load {len(self.dataset)} samples")

    def get_metadata(self):
        if self.lengths is not None and self.v_tokens is not None:
            return self.lengths, self.v_tokens
        else:
            return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = idx[0]

        item = self.dataset[idx]
        messages = json.loads(item['messages'])

        # 遍历对话中的每一轮消息 (system, user, assistant)
        for message in messages:
            content_list = message.get('content', [])
            if not isinstance(content_list, list):
                continue

            # 遍历消息中的每一个内容数据块 (text, image, video)
            for content in content_list:
                # --- 图像类型 ---
                if content.get("image") is not None:
                    image_path = content['image']
                    full_path = os.path.join(self.root_dir, image_path)

                    # 移除不需要传给 processor 的元数据
                    content.pop('height', None)
                    content.pop('width', None)

                    content['image'] = Image.open(full_path).convert("RGB")

                    # 注入图像分辨率配置
                    content['min_pixels'] = self.min_pixels
                    content['max_pixels'] = self.max_pixels

                # --- 视频类型 ---
                elif content.get("video") is not None:
                    video_path = content["video"]
                    full_path = os.path.join(self.root_dir, video_path)

                    content.pop('height', None)
                    content.pop('width', None)
                    content.pop('num_frames', None)

                    # 视频通常传路径，由 processor 内部的 fetch_video 处理抽帧
                    content["video"] = full_path

                    # 注入视频分辨率配置
                    content['min_pixels'] = self.video_min_pixels
                    content['max_pixels'] = self.video_max_pixels
                    # total_pixels 字段控制总预算
                    content['total_pixels'] = self.video_total_pixels

        messages[0]["prior_dist"] = self.prior_dist
        return messages


class qwen_vl_nav_dataset(torch.utils.data.Dataset):
    """Dataset for Navtrain JSON format (ShareGPT-style conversations with remote images).

    Converts the following JSON format into Qwen chat template messages:

        {"image": ["obs://..."], "conversations": [{"from": "human", "value": "... <image> ..."}, ...]}

    Output is identical to qwen_vl_dataset.__getitem__, so the same collate_fn works.
    """

    def __init__(self, path_or_dataset, **kwargs):
        from wam_diff.utils.file_ops import image_open as _image_open
        self._image_open = _image_open

        self.root_dir = kwargs.get("root_dir", None)

        print(f"Loading dataset from {path_or_dataset}...")

        if str(path_or_dataset).endswith(".jsonl"):
            with open(path_or_dataset, "r", encoding="utf-8") as f:
                self.ds = [json.loads(line) for line in f if line.strip()]
        else:
            with open(path_or_dataset, "r", encoding="utf-8") as f:
                self.ds = json.load(f)

        self.max_len = kwargs.get("max_len", 8192)
        self.prior_dist = kwargs.get("prior_dist", "Mask")
        self.prior_dist_2 = kwargs.get("prior_dist_2", None)
        self.switch_prior_thresh = kwargs.get("switch_prior_thresh", 1.0)

        self.min_pixels = kwargs.get("min_pixels", 8 * 8 * 32 * 32)
        self.max_pixels = kwargs.get("max_pixels", 64 * 64 * 32 * 32)
        self.video_min_pixels = 64 * 64
        self.video_max_pixels = 1024 * 1024
        self.video_total_pixels = self.max_len * 32 * 32 * 0.9
        print(f"Load {len(self.ds)} samples")

    def get_metadata(self):
        return None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            idx = idx[0]

        item = self.ds[idx]
        images = item.get("image", [])
        conversations = item.get("conversations", [])

        # Build Qwen chat template messages
        messages = []
        for conv in conversations:
            role = {"human": "user", "system": "system"}.get(conv["from"], "assistant")
            # role = {"human": "user", "system": "assistant"}.get(conv["from"], "assistant")
            text = conv["value"]
            content = []

            if role == "user" and images:
                # Split text by <image> placeholders and interleave: text → image → text → image → ...
                parts = text.split("<image>")
                for i, part in enumerate(parts):
                    if part.strip():
                        content.append({"type": "text", "text": part.strip()})
                    if i < len(images):
                        img = self._image_open(images[i]).convert("RGB")
                        content.append({
                            "type": "image",
                            "image": img,
                            "min_pixels": item.get("min_pixels", self.min_pixels),
                            "max_pixels": item.get("max_pixels", self.max_pixels),
                        })
            else:
                content.append({"type": "text", "text": text})

            messages.append({"role": role, "content": content})

        messages[0]["prior_dist"] = self.prior_dist
        if os.environ.get("DEBUG", "").lower() in ("1", "true"):
            print(f"[DEBUG] dataset messages:\n{messages}\n{'='*80}")
        return messages
