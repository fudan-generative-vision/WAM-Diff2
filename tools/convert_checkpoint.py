import argparse
import os
import json
import shutil
from pathlib import Path
from collections import OrderedDict

from safetensors import safe_open
from safetensors.torch import save_file
import torch.distributed.checkpoint as dcp
import torch

def load_dcp(ckpt_dir: Path | str) -> tuple[dict, dict]:
    """Loads a DCP checkpoint in a state dictionary from a directory."""
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    fs_reader = dcp.FileSystemReader(ckpt_dir)
    metadata = fs_reader.read_metadata()

    # Load tensor data
    tensor_state_dict = {
        k: torch.empty(tp.size, dtype=tp.properties.dtype)
        for k, tp in metadata.state_dict_metadata.items()
        if type(tp).__name__ == "TensorStorageMetadata"
    }

    if tensor_state_dict:
        dcp.load(tensor_state_dict, storage_reader=fs_reader)

    # Load scheduler data
    sched_keys = [k for k, tp in metadata.state_dict_metadata.items() if "sched" in k]

    sched_state_dict = {}
    if sched_keys:
        sched_state_dict = {k: None for k in sched_keys}
        try:
            dcp.load(sched_state_dict, storage_reader=fs_reader)
        except Exception:
            sched_state_dict = {}

    return tensor_state_dict, sched_state_dict

def load_safetensors(ckpt_dir: Path | str) -> dict[str, torch.Tensor]:
    """
    Loads a safetensors checkpoint in a state dictionary from a directory.
    """
    state_dict = {}
    if not isinstance(ckpt_dir, Path):
        ckpt_dir = Path(ckpt_dir)
    with safe_open(ckpt_dir, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict

def convert_nemo_dcp_to_safetensors(
    model_dict,
    output_dir,
    max_shard_size_gb=5,
    target_dtype=torch.bfloat16
):
    print(f"Step 1: Mapping keys and converting to {target_dtype}...")
    state_dict = OrderedDict()
    for k, v in model_dict.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.to(target_dtype).cpu().contiguous()
        else:
            print(k)
            continue

    print(f"Step 2: Sharding and saving (Max {max_shard_size_gb}GB per shard)...")

    # 首先计算是否需要分片
    total_size_bytes = sum(t.nelement() * t.element_size() for t in state_dict.values())
    needs_sharding = total_size_bytes > max_shard_size_gb * 1024**3

    if not needs_sharding:
        # 情况 A: 模型较小，直接保存为单文件
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        print(f"Conversion Finished! Saved as single file: {save_path}")
        return state_dict

    # 情况 B: 模型较大，执行原有的分片逻辑
    current_shard = {}
    current_size = 0
    shard_count = 0 # 从 0 开始，方便循环内统一处理
    weight_map = {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.nelement() * tensor.element_size()

        if current_size + tensor_size > max_shard_size_gb * 1024**3 and current_shard:
            shard_count += 1
            shard_name = f"model-{shard_count:05d}-of-index.safetensors"
            save_path = os.path.join(output_dir, shard_name)
            save_file(current_shard, save_path)

            for k in current_shard.keys():
                weight_map[k] = shard_name

            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    # 保存最后一个分片
    if current_shard:
        shard_count += 1
        temp_name = f"model-{shard_count:05d}-of-index.safetensors"
        save_file(current_shard, os.path.join(output_dir, temp_name))
        for k in current_shard.keys():
            weight_map[k] = temp_name

    print("Step 3: Generating index.json...")
    final_shard_count = shard_count
    actual_weight_map = {}
    total_str = f"{final_shard_count:05d}"

    # 这里的rename逻辑仅在needs_sharding为True时执行
    for k, v in weight_map.items():
        final_name = v.replace("index", total_str)
        actual_weight_map[k] = final_name

        old_path = os.path.join(output_dir, v)
        new_path = os.path.join(output_dir, final_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)

    index_data = {
        "metadata": {"total_size": total_size_bytes},
        "weight_map": actual_weight_map
    }

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"Conversion Finished! Saved to: {output_dir}")
    return state_dict

def copy_model_metadata(base_model_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer, processor, and model metadata files into the output."""
    for pattern in ("*.json", "*.txt"):
        for source in base_model_dir.glob(pattern):
            shutil.copy2(source, output_dir / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a distributed checkpoint to Hugging Face safetensors")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Distributed checkpoint model directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory")
    parser.add_argument("--base-model-dir", type=Path, help="Directory containing tokenizer/config metadata")
    parser.add_argument("--max-shard-size-gb", type=float, default=5.0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    args = parser.parse_args()

    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.checkpoint_dir}")
    if args.base_model_dir is not None and not args.base_model_dir.is_dir():
        raise FileNotFoundError(f"Base model directory does not exist: {args.base_model_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.base_model_dir is not None:
        copy_model_metadata(args.base_model_dir, args.output_dir)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading DCP from {args.checkpoint_dir}...")
    restored_model_dict, _ = load_dcp(args.checkpoint_dir)
    final_state_dict = convert_nemo_dcp_to_safetensors(
        model_dict=restored_model_dict,
        output_dir=args.output_dir,
        max_shard_size_gb=args.max_shard_size_gb,
        target_dtype=dtype,
    )
    total_params = sum(parameter.numel() for parameter in final_state_dict.values())
    print(f"Total parameters: {total_params / 1e9:.2f}B")


if __name__ == "__main__":
    main()
