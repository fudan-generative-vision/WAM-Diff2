# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import json
import torch
import torch.distributed as dist
from transformers import AutoProcessor
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from wam_diff.models.wam_diff2 import WAMDiff2ForConditionalGeneration
from wam_diff.utils.file_ops import image_open


def safe_json_dump(obj, output_path, indent=4):
    import os
    import json
    import shutil

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True,
                        help="Hugging Face-compatible checkpoint directory or model ID")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Evaluation JSON file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path for the merged prediction JSON")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32,
                        help="Response block length used by block diffusion decoding.")
    parser.add_argument("--denoising_steps", type=int, default=32,
                        help="Maximum denoising steps for each response block.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence_dynamic",
                        help="Token unmasking strategy inside each response block.")
    parser.add_argument("--confidence_threshold", type=float, default=0.9,
                        help="Dynamic unmasking threshold; tokens with confidence >= threshold are unmasked.")
    parser.add_argument("--min_pixels", type=int, default=12544)
    parser.add_argument("--max_pixels", type=int, default=2073600)
    args = parser.parse_args()

    # ========== 分布式初始化 ==========
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        is_distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        is_distributed = False

    if is_distributed:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    device = f"cuda:{local_rank}" if is_distributed else "cuda"
    is_main = (rank == 0)

    if is_main:
        print(f"[Dist] Running with {world_size} GPUs")

    # ========== 加载模型 ==========
    if is_main:
        print(f"Loading model from {args.model_id}...")
    model = WAMDiff2ForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        _attn_implementation="sdpa",
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    # ========== 加载数据 ==========
    if is_main:
        print(f"Loading data from {args.input_file}...")
        with open(args.input_file, 'r') as f:
            data_list = json.load(f)
        print(f"Total samples: {len(data_list)}")
    else:
        data_list = None

    # 广播数据给所有rank
    if is_distributed:
        object_list = [None]
        if is_main:
            object_list[0] = json.dumps(data_list)
        dist.broadcast_object_list(object_list, src=0)
        data_list = json.loads(object_list[0])

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

    if is_main:
        print(
            f"[Rank {rank}] Processing samples {start_idx} to {end_idx-1} ({len(local_data_list)} samples)")

    # ========== 推理 ==========
    results = []

    for idx, data in enumerate(tqdm(local_data_list, desc=f"Rank {rank}")):
        # Build messages from JSON data (same format as training: qwen_vl_nav_dataset)
        messages = []
        images = data.get("image", [])
        if isinstance(images, str):
            images = [images]
        conversations = data.get("conversations", [])

        for conv in conversations:
            conv_from = conv.get("from", "")
            text = conv.get("value", "")

            if conv_from == "system":
                role = "system"
            elif conv_from == "human":
                role = "user"
            elif conv_from == "gpt":
                role = "assistant"
            else:
                # 遇到未知角色，保险起见跳过
                print(
                    f"[Rank {rank}] Warning: unknown role {conv_from}, skipped.", flush=True)
                continue

            content = []

            if role == "user" and images:
                # Split text by <image> placeholders and interleave with images
                parts = text.split("<image>")
                for i, part in enumerate(parts):
                    if part.strip():
                        content.append({"type": "text", "text": part.strip()})
                    if i < len(images):
                        # Use image_open (same as training) to support obs:// and parquet
                        img = image_open(images[i]).convert("RGB")
                        content.append({
                            "type": "image",
                            "image": img,
                            "min_pixels": args.min_pixels,
                            "max_pixels": args.max_pixels,
                        })
            else:
                content.append({"type": "text", "text": text})

            messages.append({"role": role, "content": content})

            # Only need the first user turn for inference
            if role == "user":
                break

        try:
            # Process inputs
            texts = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                return_video_metadata=True,
                image_patch_size=processor.image_processor.patch_size,
            )

            video_metadata = None
            if video_inputs is not None:
                video_metadata = [_[1] for _ in video_inputs]
                video_inputs = [_[0] for _ in video_inputs]

            batch = processor(
                text=[texts],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                return_tensors="pt",
                video_metadata=video_metadata,
                **video_kwargs,
            ).to(device)

            # Run inference
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

            # Decode result
            result_text = processor.tokenizer.batch_decode(
                response_ids, skip_special_tokens=True)[0].strip()
            print(
                f"[Rank {rank}] Sample {idx+1}/{len(local_data_list)}: {result_text[:100]}...")

            # Store result
            result_item = {
                "id": data.get("id", idx),
                "datasource": data.get("datasource", "navsim"),
                "image": data.get("image", []),
                "question": conversations[0]["value"] if conversations else "",
                "prediction": result_text,
                "ground_truth": "",
            }

            # Add ground truth from gpt conversation
            for conv in conversations:
                if conv["from"] == "gpt":
                    result_item["ground_truth"] = conv["value"]
                    break

            results.append(result_item)

        except Exception as e:
            print(f"[Rank {rank}] Error processing sample {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "id": data.get("id", idx),
                "datasource": data.get("datasource", "navsim"),
                "image": data.get("image", []),
                "question": conversations[0]["value"] if conversations else "",
                "prediction": "",
                "ground_truth": "",
                "error": str(e),
            })

        # Save intermediate results every 50 samples
        # if (idx + 1) % 50 == 0:
        #     rank_output = args.output_file.replace('.json', f'_rank{rank}.json')
        #     with open(rank_output, 'w') as f:
        #         json.dump(results, f, indent=4, ensure_ascii=False)

        # if (idx + 1) % 50 == 0:
        #     rank_output = args.output_file.replace(".json", f"_rank{rank}.json")
        #     tmp_rank_output = rank_output + ".tmp"

        #     with open(tmp_rank_output, "w", encoding="utf-8") as f:
        #         json.dump(results, f, indent=4, ensure_ascii=False)
        #         f.flush()
        #         os.fsync(f.fileno())

        #     os.replace(tmp_rank_output, rank_output)

        if (idx + 1) % 50 == 0:
            rank_output = args.output_file.replace(
                ".json", f"_rank{rank}.json")
            safe_json_dump(results, rank_output, indent=4)

    # # ========== 收集并合并结果 ==========
    # # Each rank writes its own results to a local file, then rank 0 reads and merges.
    # # This avoids all_gather_object which materializes all data on GPU and can OOM.
    # rank_output = args.output_file.replace('.json', f'_rank{rank}.json')
    # with open(rank_output, 'w') as f:
    #     json.dump(results, f, ensure_ascii=False)

    # if is_distributed:
    #     # Signal completion via file-based marker (avoids NCCL barrier timeout
    #     # when ranks finish at very different times due to uneven workloads).
    #     import time
    #     done_dir = args.output_file + ".done"
    #     os.makedirs(done_dir, exist_ok=True)
    #     done_marker = os.path.join(done_dir, f"rank{rank}")
    #     with open(done_marker, 'w') as f:
    #         f.write("done")

    #     if is_main:
    #         # Poll until all ranks have written their marker
    #         all_done = False
    #         while not all_done:
    #             all_done = all(
    #                 os.path.exists(os.path.join(done_dir, f"rank{r}"))
    #                 for r in range(world_size)
    #             )
    #             if not all_done:
    #                 time.sleep(5)

    #         all_results = []
    #         for r in range(world_size):
    #             r_file = args.output_file.replace('.json', f'_rank{r}.json')
    #             with open(r_file, 'r') as f:
    #                 all_results.extend(json.load(f))

    #         # Sort by original id
    #         all_results.sort(key=lambda x: x.get('id', ''))

    #         with open(args.output_file, 'w') as f:
    #             json.dump(all_results, f, indent=4, ensure_ascii=False)

    #         print(f"\nDone! Saved results to {args.output_file}")
    #         print(f"Successfully processed: {len([r for r in all_results if 'error' not in r])}/{num_samples}")

    #         # Clean up markers and per-rank files
    #         import shutil
    #         shutil.rmtree(done_dir, ignore_errors=True)
    #         for r in range(world_size):
    #             r_file = args.output_file.replace('.json', f'_rank{r}.json')
    #             if os.path.exists(r_file):
    #                 os.remove(r_file)
    #     # Non-main ranks exit here; no barrier needed
    # else:
    #     import shutil
    #     shutil.copy2(rank_output, args.output_file)
    #     print(f"\nDone! Saved results to {args.output_file}")
    #     print(f"Successfully processed: {len([r for r in results if 'error' not in r])}/{len(data_list)}")

    # ========== 收集并合并结果 ==========
    import time
    import tempfile
    import shutil

    # 每个 rank 单独写文件
    rank_output = args.output_file.replace(".json", f"_rank{rank}.json")

    safe_json_dump(results, rank_output, indent=4)

    print(
        f"[Rank {rank}] saved {len(results)} results to {rank_output}", flush=True)

    if is_distributed:
        # 用 torch barrier，确保所有 rank 都写完
        dist.barrier()

        if is_main:
            all_results = []

            for r in range(world_size):
                r_file = args.output_file.replace(".json", f"_rank{r}.json")

                if not os.path.exists(r_file):
                    raise FileNotFoundError(
                        f"Missing rank result file: {r_file}")

                with open(r_file, "r", encoding="utf-8") as f:
                    rank_results = json.load(f)

                print(
                    f"[Merge] rank {r}: {len(rank_results)} samples", flush=True)
                all_results.extend(rank_results)

            # 检查数量
            print(
                f"[Merge] total merged: {len(all_results)} / expected: {num_samples}", flush=True)

            if len(all_results) != num_samples:
                print(
                    f"[WARNING] merged result count mismatch: {len(all_results)} != {num_samples}", flush=True)

            # 按 id 排序
            all_results.sort(key=lambda x: str(x.get("id", "")))

            # tmp_output_file = args.output_file + ".tmp"
            # with open(tmp_output_file, "w", encoding="utf-8") as f:
            #     json.dump(all_results, f, indent=4, ensure_ascii=False)
            #     f.flush()
            #     os.fsync(f.fileno())

            # os.replace(tmp_output_file, args.output_file)

            safe_json_dump(all_results, args.output_file, indent=4)

            print(f"\nDone! Saved results to {args.output_file}", flush=True)
            print(
                f"Successfully processed: {len([r for r in all_results if 'error' not in r])}/{num_samples}",
                flush=True,
            )

            # 不建议立刻删除 rank 文件，先保留方便排查
            # for r in range(world_size):
            #     r_file = args.output_file.replace(".json", f"_rank{r}.json")
            #     if os.path.exists(r_file):
            #         os.remove(r_file)

        dist.barrier()

    else:
        shutil.copy2(rank_output, args.output_file)
        print(f"\nDone! Saved results to {args.output_file}")
        print(
            f"Successfully processed: {len([r for r in results if 'error' not in r])}/{len(data_list)}")

    # Clean up (only main rank reaches here in distributed mode after merging)
    if is_distributed and is_main:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
