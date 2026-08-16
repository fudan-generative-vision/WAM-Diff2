#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import json
import time
import shutil
from collections import defaultdict

import torch
import torch.distributed as dist
from transformers import AutoProcessor
from tqdm import tqdm

try:
    import torch_npu
except ImportError:
    torch_npu = None

from qwen_vl_utils import process_vision_info
from wam_diff.models.wam_diff2 import WAMDiff2ForConditionalGeneration
from wam_diff.utils.file_ops import image_open


# =========================
# OBS 环境变量
# =========================


# =========================
# 工具函数
# =========================
def safe_json_dump(obj, output_path, indent=4):
    """
    原子写 JSON。
    避免写到一半任务中断导致 json 文件损坏。
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tmp_path = output_path + ".tmp"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, output_path)

    except PermissionError as e:
        print(
            f"[WARNING] os.replace failed: {tmp_path} -> {output_path}: {e}", flush=True)
        print("[WARNING] fallback to non-atomic direct/copy write.", flush=True)

        try:
            if os.path.exists(tmp_path):
                shutil.copyfile(tmp_path, output_path)
                os.remove(tmp_path)
                return
        except Exception as copy_e:
            print(f"[WARNING] copy fallback failed: {copy_e}", flush=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())


def sync_device_if_needed(device: str):
    """
    NPU / CUDA 是异步执行的。
    如果不 synchronize，generate 的 time.time() 统计可能不准。
    """
    try:
        if isinstance(device, str) and device.startswith("npu"):
            torch.npu.synchronize()
        elif isinstance(device, str) and device.startswith("cuda"):
            torch.cuda.synchronize()
    except Exception:
        pass


class TimeMeter:
    """
    统计每个阶段的累计耗时和平均耗时。
    """

    def __init__(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def add(self, name: str, cost: float):
        self.total[name] += float(cost)
        self.count[name] += 1

    def avg(self, name: str):
        if self.count[name] == 0:
            return 0.0
        return self.total[name] / self.count[name]

    def sum(self, name: str):
        return self.total[name]

    def summary(self):
        keys = [
            "build_message",
            "image_open",
            "chat_template",
            "process_vision",
            "processor",
            "to_device",
            "generate",
            "decode",
            "save_json",
            "total",
        ]

        parts = []
        for k in keys:
            if self.count[k] > 0:
                parts.append(f"{k}_avg={self.avg(k):.4f}s")

        return ", ".join(parts)


def get_device_mem(device: str):
    """
    打印当前进程设备显存 / HBM 占用。
    """
    try:
        if isinstance(device, str) and device.startswith("npu"):
            allocated = torch.npu.memory_allocated() / 1024**3
            reserved = torch.npu.memory_reserved() / 1024**3
            return f"npu_mem_alloc={allocated:.2f}GB, npu_mem_reserved={reserved:.2f}GB"

        if isinstance(device, str) and device.startswith("cuda"):
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"cuda_mem_alloc={allocated:.2f}GB, cuda_mem_reserved={reserved:.2f}GB"

    except Exception:
        pass

    return "mem=N/A"


def count_visible_npu():
    """
    返回当前进程可见 NPU 数量。
    优先从 ASCEND_RT_VISIBLE_DEVICES 解析。
    """
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()

    if visible_devices:
        return len([x for x in visible_devices.split(",") if x.strip() != ""])

    try:
        return torch.npu.device_count()
    except Exception:
        return 0


def build_messages_from_data(data, args, rank):
    """
    从单条数据构建 Qwen-VL messages。
    同时统计 image_open 耗时。
    """
    messages = []
    image_open_cost = 0.0

    images = data.get("image", [])
    if isinstance(images, str):
        images = [images]

    conversations = data.get("conversations", [])

    for conv in conversations:
        conv_from = conv.get("from", "")
        text = conv.get("value", "")

        if conv_from == "system":
            role = "assistant"
        elif conv_from == "human":
            role = "user"
        elif conv_from == "gpt":
            role = "assistant"
        else:
            print(
                f"[Rank {rank}] Warning: unknown role {conv_from}, skipped.", flush=True)
            continue

        content = []

        if role == "user" and images:
            parts = text.split("<image>")

            for i, part in enumerate(parts):
                if part.strip():
                    content.append({"type": "text", "text": part.strip()})

                if i < len(images):
                    t_img0 = time.time()
                    img = image_open(images[i]).convert("RGB")
                    t_img1 = time.time()

                    image_open_cost += t_img1 - t_img0

                    content.append({
                        "type": "image",
                        "image": img,
                        "min_pixels": args.min_pixels,
                        "max_pixels": args.max_pixels,
                    })
        else:
            content.append({"type": "text", "text": text})

        messages.append({"role": role, "content": content})

        # 只需要第一个 user turn 做推理
        if role == "user":
            break

    return messages, conversations, image_open_cost


def make_result_item(data, conversations, result_text, fallback_id):
    """
    构造输出结果。
    """
    result_item = {
        "id": data.get("id", fallback_id),
        "datasource": data.get("datasource", "navsim"),
        "image": data.get("image", []),
        "question": conversations[0]["value"] if conversations else "",
        "prediction": result_text,
        "ground_truth": "",
    }

    for conv in conversations:
        if conv.get("from") == "gpt":
            result_item["ground_truth"] = conv.get("value", "")
            break

    return result_item


def print_profile(rank, idx, total_len, meter, device, prefix="PROFILE"):
    avg_total = meter.avg("total")
    throughput = 1.0 / avg_total if avg_total > 0 else 0.0

    print(
        f"[{prefix}][Rank {rank}] "
        f"step={idx}/{total_len}, "
        f"avg_total={avg_total:.4f}s, "
        f"throughput={throughput:.4f} samples/s/rank, "
        f"{meter.summary()}, "
        f"{get_device_mem(device)}",
        flush=True,
    )


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 兼容 --model_id 和 --model_path
    parser.add_argument(
        "--model_id",
        "--model_path",
        dest="model_id",
        type=str,
        required=True,
        help="Hugging Face-compatible checkpoint directory or model ID.",
    )

    parser.add_argument("--input_file", type=str, required=True,
                        help="Evaluation JSON file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path for the merged benchmark prediction JSON")

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--block_size",
        type=int,
        default=32,
        help="Response block length used by block diffusion decoding.",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=32,
        help="Maximum denoising steps for each response block.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--remasking_strategy",
        type=str,
        default="low_confidence_dynamic",
        help="Token unmasking strategy inside each response block.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.9,
        help="Dynamic unmasking threshold; tokens with confidence >= threshold are unmasked.",
    )

    parser.add_argument("--min_pixels", type=int, default=12544)
    parser.add_argument("--max_pixels", type=int, default=2073600)

    # Profile 相关参数
    parser.add_argument(
        "--profile_every",
        type=int,
        default=20,
        help="Print average timing every N samples per rank. Set 0 to disable.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save rank intermediate result every N samples. Set 0 to disable.",
    )

    # 如果你想一张 NPU 起多个进程，可以打开这个参数
    # 例如 8 张 NPU，nproc_per_node=16，则 local_rank 会映射到 local_rank % 8
    parser.add_argument(
        "--allow_multi_process_per_device",
        action="store_true",
        help="Allow multiple torchrun local processes to share one visible NPU/GPU.",
    )

    args = parser.parse_args()

    # ========== 分布式 / 设备初始化 ==========
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False

    has_npu = (
        torch_npu is not None
        and hasattr(torch, "npu")
        and torch.npu.is_available()
    )

    if has_npu:
        visible_device_count = count_visible_npu()

        if visible_device_count <= 0:
            raise RuntimeError(
                "No visible NPU found. Please check ASCEND_RT_VISIBLE_DEVICES.")

        if local_rank >= visible_device_count and not args.allow_multi_process_per_device:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} >= visible_npu_count={visible_device_count}. "
                f"Usually this means nproc_per_node is larger than visible NPU count. "
                f"If you intentionally want multiple processes per NPU, add "
                f"--allow_multi_process_per_device."
            )

        device_id = local_rank % visible_device_count
        device = f"npu:{device_id}"
        torch.npu.set_device(device)
        dist_backend = "gloo"

    elif torch.cuda.is_available():
        visible_device_count = torch.cuda.device_count()

        if local_rank >= visible_device_count and not args.allow_multi_process_per_device:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} >= visible_cuda_count={visible_device_count}. "
                f"Usually this means nproc_per_node is larger than visible GPU count. "
                f"If you intentionally want multiple processes per GPU, add "
                f"--allow_multi_process_per_device."
            )

        device_id = local_rank % visible_device_count
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
        dist_backend = "nccl"

    else:
        visible_device_count = 0
        device_id = -1
        device = "cpu"
        dist_backend = "gloo"

    if is_distributed:
        t_pg0 = time.time()
        dist.init_process_group(backend=dist_backend)
        t_pg1 = time.time()
    else:
        t_pg0 = t_pg1 = time.time()

    is_main = rank == 0

    print(
        f"[Dist][Rank {rank}] "
        f"distributed={is_distributed}, "
        f"rank={rank}, world_size={world_size}, "
        f"local_rank={local_rank}, "
        f"visible_device_count={visible_device_count}, "
        f"mapped_device={device}, "
        f"backend={dist_backend}, "
        f"init_pg_time={t_pg1 - t_pg0:.3f}s",
        flush=True,
    )

    print(f"using min pixel{args.min_pixels},max pixel{args.max_pixels}")

    # ========== 加载模型 ==========
    t_load_model0 = time.time()

    if is_main:
        print(f"Loading model from {args.model_id}...", flush=True)

    model = WAMDiff2ForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        _attn_implementation="sdpa",
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    sync_device_if_needed(device)
    t_load_model1 = time.time()

    print(
        f"[PROFILE][Rank {rank}] load_model_and_processor={t_load_model1 - t_load_model0:.3f}s, "
        f"{get_device_mem(device)}",
        flush=True,
    )

    # ========== 加载数据 ==========
    t_load_data0 = time.time()

    if is_main:
        print(f"Loading data from {args.input_file}...", flush=True)
        with open(args.input_file, "r", encoding="utf-8") as f:
            data_list = json.load(f)
        print(f"Total samples: {len(data_list)}", flush=True)
    else:
        data_list = None

    t_load_data1 = time.time()

    if is_main:
        print(
            f"[PROFILE][Rank {rank}] load_json={t_load_data1 - t_load_data0:.3f}s", flush=True)

    # ========== 广播数据给所有 rank ==========
    if is_distributed:
        t_bcast0 = time.time()

        object_list = [None]
        if is_main:
            object_list[0] = json.dumps(data_list, ensure_ascii=False)

        dist.broadcast_object_list(object_list, src=0)

        if not is_main:
            data_list = json.loads(object_list[0])

        t_bcast1 = time.time()

        print(
            f"[PROFILE][Rank {rank}] broadcast_data={t_bcast1 - t_bcast0:.3f}s",
            flush=True,
        )

    # ========== 数据分片 ==========
    num_samples = len(data_list)
    samples_per_rank = num_samples // world_size
    remainder = num_samples % world_size

    if rank < remainder:
        start_idx = rank * (samples_per_rank + 1)
        end_idx = start_idx + samples_per_rank + 1
    else:
        start_idx = rank * samples_per_rank + remainder
        end_idx = start_idx + samples_per_rank

    local_data_list = data_list[start_idx:end_idx]
    print(
        f"[Rank {rank}] Processing samples {start_idx} to {end_idx - 1} "
        f"({len(local_data_list)} samples)",
        flush=True,
    )

    # ========== 推理 ==========
    results = []
    meter = TimeMeter()

    infer_start_time = time.time()

    for idx, data in enumerate(tqdm(local_data_list, desc=f"Rank {rank}")):
        sample_total0 = time.time()

        try:
            # 1. build messages + image_open
            t0 = time.time()
            messages, conversations, image_open_cost = build_messages_from_data(
                data=data,
                args=args,
                rank=rank,
            )
            t1 = time.time()

            meter.add("build_message", t1 - t0)
            meter.add("image_open", image_open_cost)

            # 2. apply_chat_template
            t0 = time.time()
            texts = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            t1 = time.time()
            meter.add("chat_template", t1 - t0)

            # 3. process_vision_info
            t0 = time.time()
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
                image_patch_size=processor.image_processor.patch_size,
            )
            t1 = time.time()
            meter.add("process_vision", t1 - t0)

            video_metadata = None
            if video_inputs is not None:
                video_metadata = [_[1] for _ in video_inputs]
                video_inputs = [_[0] for _ in video_inputs]

            # 4. processor
            t0 = time.time()
            batch = processor(
                text=[texts],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
                video_metadata=video_metadata,
                **video_kwargs,
            )
            t1 = time.time()
            meter.add("processor", t1 - t0)

            # 5. to device
            t0 = time.time()
            batch = batch.to(device)
            sync_device_if_needed(device)
            t1 = time.time()
            meter.add("to_device", t1 - t0)

            # 6. generate
            sync_device_if_needed(device)
            t0 = time.time()

            with torch.no_grad():
                response_ids = model.generate(
                    batch,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    denoising_steps=args.denoising_steps,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    remasking_strategy=args.remasking_strategy,
                    confidence_threshold=args.confidence_threshold,
                )

            sync_device_if_needed(device)
            t1 = time.time()
            meter.add("generate", t1 - t0)

            # 7. decode
            t0 = time.time()
            result_text = processor.tokenizer.batch_decode(
                response_ids,
                skip_special_tokens=True,
            )[0].strip()
            t1 = time.time()
            meter.add("decode", t1 - t0)

            print(
                f"[Rank {rank}] Sample {idx + 1}/{len(local_data_list)}: "
                f"{result_text[:100]}...",
                flush=True,
            )

            result_item = make_result_item(
                data=data,
                conversations=conversations,
                result_text=result_text,
                fallback_id=start_idx + idx,
            )
            results.append(result_item)

        except Exception as e:
            print(
                f"[Rank {rank}] Error processing sample {idx + 1}: {e}", flush=True)
            import traceback
            traceback.print_exc()

            conversations = data.get("conversations", [])

            results.append({
                "id": data.get("id", start_idx + idx),
                "datasource": data.get("datasource", "navsim"),
                "image": data.get("image", []),
                "question": conversations[0]["value"] if conversations else "",
                "prediction": "",
                "ground_truth": "",
                "error": str(e),
            })

        # 8. 中间保存
        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            t0 = time.time()
            rank_output = args.output_file.replace(
                ".json", f"_rank{rank}.json")
            safe_json_dump(results, rank_output, indent=4)
            t1 = time.time()
            meter.add("save_json", t1 - t0)

        sample_total1 = time.time()
        meter.add("total", sample_total1 - sample_total0)

        # 9. 周期性打印 profile
        if args.profile_every > 0 and (idx + 1) % args.profile_every == 0:
            print_profile(
                rank=rank,
                idx=idx + 1,
                total_len=len(local_data_list),
                meter=meter,
                device=device,
                prefix="PROFILE",
            )

    infer_end_time = time.time()

    local_total_time = infer_end_time - infer_start_time
    local_throughput = len(local_data_list) / \
        local_total_time if local_total_time > 0 else 0.0

    print(
        f"[PROFILE][Rank {rank}] inference_done, "
        f"local_samples={len(local_data_list)}, "
        f"local_total_time={local_total_time:.3f}s, "
        f"local_throughput={local_throughput:.4f} samples/s, "
        f"{meter.summary()}, "
        f"{get_device_mem(device)}",
        flush=True,
    )

    # ========== 收集并合并结果 ==========
    rank_output = args.output_file.replace(".json", f"_rank{rank}.json")

    t_save_final_rank0 = time.time()
    safe_json_dump(results, rank_output, indent=4)
    t_save_final_rank1 = time.time()

    print(
        f"[Rank {rank}] saved {len(results)} results to {rank_output}, "
        f"final_rank_save_time={t_save_final_rank1 - t_save_final_rank0:.3f}s",
        flush=True,
    )

    if is_distributed:
        # 第一次 barrier：确保所有 rank 文件写完
        t_barrier0 = time.time()
        dist.barrier()
        t_barrier1 = time.time()

        print(
            f"[PROFILE][Rank {rank}] first_barrier_wait={t_barrier1 - t_barrier0:.3f}s",
            flush=True,
        )

        if is_main:
            t_merge0 = time.time()

            all_results = []

            for r in range(world_size):
                r_file = args.output_file.replace(".json", f"_rank{r}.json")

                if not os.path.exists(r_file):
                    raise FileNotFoundError(
                        f"Missing rank result file: {r_file}")

                t_read0 = time.time()
                with open(r_file, "r", encoding="utf-8") as f:
                    rank_results = json.load(f)
                t_read1 = time.time()

                print(
                    f"[Merge] rank {r}: {len(rank_results)} samples, "
                    f"read_time={t_read1 - t_read0:.3f}s",
                    flush=True,
                )

                all_results.extend(rank_results)

            print(
                f"[Merge] total merged: {len(all_results)} / expected: {num_samples}", flush=True)

            if len(all_results) != num_samples:
                print(
                    f"[WARNING] merged result count mismatch: "
                    f"{len(all_results)} != {num_samples}",
                    flush=True,
                )

            # 按 id 排序
            all_results.sort(key=lambda x: str(x.get("id", "")))

            t_final_save0 = time.time()
            safe_json_dump(all_results, args.output_file, indent=4)
            t_final_save1 = time.time()

            t_merge1 = time.time()

            print(
                f"[PROFILE][Rank {rank}] "
                f"final_save_time={t_final_save1 - t_final_save0:.3f}s, "
                f"merge_and_final_save={t_merge1 - t_merge0:.3f}s",
                flush=True,
            )

            print(f"\nDone! Saved results to {args.output_file}", flush=True)
            print(
                f"Successfully processed: "
                f"{len([r for r in all_results if 'error' not in r])}/{num_samples}",
                flush=True,
            )

        # 第二次 barrier：确保 rank0 合并完成
        t_barrier0 = time.time()
        dist.barrier()
        t_barrier1 = time.time()

        print(
            f"[PROFILE][Rank {rank}] second_barrier_wait={t_barrier1 - t_barrier0:.3f}s",
            flush=True,
        )

    else:
        shutil.copy2(rank_output, args.output_file)

        print(f"\nDone! Saved results to {args.output_file}", flush=True)
        print(
            f"Successfully processed: "
            f"{len([r for r in results if 'error' not in r])}/{len(data_list)}",
            flush=True,
        )

    # ========== 清理分布式进程组 ==========
    if is_distributed:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
