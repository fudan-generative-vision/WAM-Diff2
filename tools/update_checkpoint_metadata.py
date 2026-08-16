# SPDX-License-Identifier: Apache-2.0

"""Normalize Hugging Face checkpoint metadata for WAM Diff2."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any


MODEL_TYPE = "wam_diff2"
TEXT_MODEL_TYPE = "wam_diff2_text"
ARCHITECTURE = "WAMDiff2ForConditionalGeneration"


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return checkpoint metadata using the public WAM Diff2 identifiers."""
    normalized = deepcopy(config)
    normalized["model_type"] = MODEL_TYPE
    normalized["architectures"] = [ARCHITECTURE]

    vision_config = normalized.get("vision_config")
    if isinstance(vision_config, dict):
        vision_config["model_type"] = MODEL_TYPE

    text_config = normalized.get("text_config")
    if isinstance(text_config, dict):
        text_config["model_type"] = TEXT_MODEL_TYPE

    # Local WAM Diff2 registration replaces checkpoint-specific remote-code maps.
    normalized.pop("auto_map", None)
    return normalized


def resolve_config_path(path: Path) -> Path:
    """Resolve either a checkpoint directory or its config.json file."""
    config_path = path / "config.json" if path.is_dir() else path
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config does not exist: {config_path}")
    return config_path


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary_path = Path(handle.name)

    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def update_checkpoint_metadata(source: Path, output: Path | None = None) -> Path:
    """Normalize a checkpoint config, preserving a backup for in-place updates."""
    source_config = resolve_config_path(source)
    destination = output or source_config

    with source_config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if output is None:
        backup = source_config.with_name(f"{source_config.name}.pre-wam-diff2.bak")
        if not backup.exists():
            shutil.copy2(source_config, backup)

    write_json_atomic(destination, normalize_config(config))
    return destination


def main() -> None:
    """Run the checkpoint metadata normalizer CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint directory or config.json path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write to a separate JSON file instead of updating in place",
    )
    args = parser.parse_args()

    destination = update_checkpoint_metadata(args.checkpoint, args.output)
    print(f"Updated WAM Diff2 metadata: {destination}")


if __name__ == "__main__":
    main()
