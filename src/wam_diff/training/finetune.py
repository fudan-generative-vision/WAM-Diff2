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

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import pathlib
import time
from omegaconf import OmegaConf
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers import AutoProcessor
from transformers.modeling_utils import no_init_weights
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TRANSFORMERS_CACHE, ContextManagers
from wandb import Settings

from wam_diff._transformers.utils import apply_cache_compatibility_patches
from wam_diff.components._peft.lora import apply_lora_to_linear_modules
from wam_diff.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from wam_diff.components.config._arg_parser import parse_args_and_load_config
from wam_diff.data.collate_fns import COLLATE_FNS
from wam_diff.components.distributed.cp_utils import make_cp_batch_and_ctx
from wam_diff.components.distributed.ddp import DDPManager
from wam_diff.components.distributed.init_utils import (
    get_world_size_safe,
    initialize_distributed,
)
from wam_diff.components.distributed.megatron_fsdp import MegatronFSDPManager
from wam_diff.components.distributed.utils import FirstRankPerNode, get_sync_ctx
from wam_diff.components.loggers.log_utils import setup_logging
from wam_diff.components.loggers.metric_logger import MetricsSample, build_metric_logger
from wam_diff.components.loggers.wandb_utils import suppress_wandb_log_messages
from wam_diff.losses.linear_ce import FusedLinearCrossEntropy
from wam_diff.losses.masked_ce import MaskedCrossEntropy
from wam_diff.losses.mpg_kl import MixturePathGeneralizeKL
from wam_diff.losses.weighted_ce import WeightedCrossEntropy
from wam_diff.data.sampler import BalanceSampler
from wam_diff.components.optim.scheduler import OptimizerParamScheduler
from wam_diff.components.quantization.fp8 import apply_fp8_to_model, build_fp8_config
from wam_diff.components.training.rng import ScopedRNG, StatefulRNG
from wam_diff.components.training.step_scheduler import StepScheduler
from wam_diff.components.training.utils import (
    count_tail_padding,
    scale_grads_and_clip_grad_norm,
)
from wam_diff.components.utils.compile_utils import (
    build_compile_config,
    compile_model,
)
from wam_diff.components.utils.model_utils import (
    _supports_logits_to_keep,
    apply_parameter_freezing,
    init_empty_weights,
    print_trainable_parameters,
)
from wam_diff.training.base_recipe import BaseRecipe

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from wam_diff.components.distributed.init_utils import DistInfo

logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------

def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    else:
        return data

def _get_model_name(cfg_model):
    if cfg_model.get("pretrained_model_name_or_path", None) is not None:
        return cfg_model.pretrained_model_name_or_path
    elif cfg_model.get("config", None) is not None:
        return cfg_model.config.get("pretrained_model_name_or_path", None)
    else:
        return None


def _freeze_model(model: nn.Module, cfg_freeze: Optional[Dict[str, Any]] = None, freeze_embeddings: bool = True):
    """
    Freeze the model.

    Args:
        model: The model to freeze.
        cfg_freeze: The configuration for freezing the model.
        freeze_embeddings: Whether to freeze embeddings.

    Returns:
        nn.Module: The frozen model.
    """
    if cfg_freeze is not None:
        apply_parameter_freezing(model, cfg_freeze)
    elif freeze_embeddings:
        logging.info("Freezing embeddings")
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.requires_grad = False
    return model

# Modified: to support multiple groups
def get_parameter_groups(model, cfg_opt):
    base_lr = cfg_opt.get("lr", 1e-5)
    visual_lr = cfg_opt.get("visual_lr", base_lr)
    language_lr = cfg_opt.get("language_lr", base_lr)
    merger_lr = cfg_opt.get("merger_lr", base_lr)
    weight_decay = cfg_opt.get("weight_decay", 0.0)

    groups = {
        "visual_decay": {
            "params": [],
            "lr": visual_lr,
            "max_lr": visual_lr,
            "min_lr": visual_lr * 0.01,
            "init_lr": visual_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "visual_no_decay": {
            "params": [],
            "lr": visual_lr,
            "max_lr": visual_lr,
            "min_lr": visual_lr * 0.01,
            "init_lr": visual_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "language_decay": {
            "params": [],
            "lr": language_lr,
            "max_lr": language_lr,
            "min_lr": language_lr * 0.01,
            "init_lr": language_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "language_no_decay": {
            "params": [],
            "lr": language_lr,
            "max_lr": language_lr,
            "min_lr": language_lr * 0.01,
            "init_lr": language_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "merger_decay": {
            "params": [],
            "lr": merger_lr,
            "max_lr": merger_lr,
            "min_lr": merger_lr * 0.01,
            "init_lr": merger_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
        "merger_no_decay": {
            "params": [],
            "lr": merger_lr,
            "max_lr": merger_lr,
            "min_lr": merger_lr * 0.01,
            "init_lr": merger_lr * 0.1,
            "weight_decay": 0.0,
            "wd_mult": 0.0,
        },
        "other": {
            "params": [],
            "lr": base_lr,
            "max_lr": base_lr,
            "min_lr": base_lr * 0.01,
            "init_lr": base_lr * 0.1,
            "weight_decay": weight_decay,
            "wd_mult": weight_decay,
        },
    }

    no_decay_keywords = ["norm", "bias", "embed_tokens", "pos_embed"]
    seen_param_ids = set()

    visual_decay_names = []
    visual_no_decay_names = []
    language_decay_names = []
    language_no_decay_names = []
    merger_decay_names = []
    merger_no_decay_names = []
    other_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 权重共享检测：如果这个物理参数已经分过组了，直接跳过
        if id(param) in seen_param_ids:
            logger.info(f"Skipping tied parameter: {name}")
            continue
        seen_param_ids.add(id(param))

        is_no_decay = any(k in name.lower() for k in no_decay_keywords) or (param.ndim <= 1)

        if "visual.merger" in name or "deepstack_merger_list" in name:
            if is_no_decay:
                groups["merger_no_decay"]["params"].append(param)
                merger_no_decay_names.append(name)
            else:
                groups["merger_decay"]["params"].append(param)
                merger_decay_names.append(name)
        elif "visual" in name:
            if is_no_decay:
                groups["visual_no_decay"]["params"].append(param)
                visual_no_decay_names.append(name)
            else:
                groups["visual_decay"]["params"].append(param)
                visual_decay_names.append(name)
        elif "language_model" in name or "lm_head" in name:
            if is_no_decay:
                groups["language_no_decay"]["params"].append(param)
                language_no_decay_names.append(name)
            else:
                groups["language_decay"]["params"].append(param)
                language_decay_names.append(name)
        else:
            groups["other"]["params"].append(param)
            other_names.append(name)

    # print("visual_decay_names:", visual_decay_names)
    # print("visual_no_decay_names:", visual_no_decay_names)
    # print("language_decay_names:", language_decay_names)
    # print("language_no_decay_names:", language_no_decay_names)
    # print("merger_decay_names:", merger_decay_names)
    # print("merger_no_decay_names:", merger_no_decay_names)
    # print("other_names:", other_names)

    param_groups = []
    for group_name, group in groups.items():
        if group["params"]:
            group["name"] = group_name
            param_groups.append(group)

    for k, v in groups.items():
        if len(v["params"]) > 0:
            logger.info(f"Group {k}: {len(v['params'])} params, max_lr={v['max_lr']:.1e}, min_lr={v['min_lr']:.1e}")

    return param_groups


def build_model_and_optimizer(
    device,
    cfg_model,
    cfg_opt,
    cfg_freeze,
    cfg_peft,
    model_wrapper,
    seed,
    checkpointer: Checkpointer,
    tp_size=1,
    cp_size=1,
    freeze_embeddings=True,
    cfg_fp8=None,
    cfg_compile=None,
    loss_fn=None,
    parallelize_fn=None,
    load_base_model=True,
) -> tuple[nn.Module, list[str], "Optimizer"]:  # noqa: F821
    """
    Build and initialize a model for VLM.

    Args:
        device: The target device.
        cfg_model: Configuration for model instantiation.
        cfg_opt: Configuration for optimizer instantiation.
        cfg_freeze: Configuration for freezing parameters.
        cfg_peft: Configuration for PEFT.
        model_wrapper: Optional parallelism wrapper.
        seed: Random seed.
        tp_size: Tensor parallel size.
        freeze_embeddings: Whether to freeze embeddings.
        cfg_fp8: Configuration for FP8.
        cfg_compile: Configuration for torch.compile.
        parallelize_fn: Optional parallelization function.
        load_base_model: Whether to load the base model.

    Returns:
        The instantiated model on the specified device, the state dict keys before any parallelization, and the optimizer.
    """
    is_meta_device = not isinstance(model_wrapper, (MegatronFSDPManager, DDPManager))

    init_ctx = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()
    with ScopedRNG(seed=seed, ranked=True):
        kwargs = {"tp_size": tp_size, "cp_size": cp_size}

        # Instantiate the model in meta device to avoid OOM
        with init_ctx:
            model = cfg_model.instantiate(**kwargs)
            model = _freeze_model(model, cfg_freeze, freeze_embeddings)
            # Optionally apply PEFT (e.g., LoRA/DoRA, etc)
            if cfg_peft is not None:
                if tp_size > 1:
                    logger.info("Disabling Triton with TP ({})".format(tp_size))
                    cfg_peft.use_triton = False
                apply_lora_to_linear_modules(model, cfg_peft)

            if cfg_fp8 is not None:
                fp8_config = build_fp8_config(cfg_fp8)
                model = apply_fp8_to_model(model, config=fp8_config)

        # hold a copy of the model state dict keys before any parallelization
        state_dict_keys = model.state_dict().keys()

        if not _supports_logits_to_keep(model) and not isinstance(loss_fn, MaskedCrossEntropy):
            logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
            loss_fn = MaskedCrossEntropy()

        load_weights = False
        if parallelize_fn is not None and get_world_size_safe() > 1:
            moe_mesh = getattr(model_wrapper, "moe_mesh", None)
            ep_axis_name = "ep" if moe_mesh is not None and "ep" in moe_mesh.mesh_dim_names else None
            ep_shard_axis_names = (
                ("ep_shard",) if moe_mesh is not None and "ep_shard" in moe_mesh.mesh_dim_names else None
            )
            parallelize_fn(
                model,
                world_mesh=model_wrapper.device_mesh,
                moe_mesh=moe_mesh,
                pp_enabled=False,
                dp_axis_names=(
                    ("dp_replicate", "dp_shard_cp")
                    if "dp_replicate" in model_wrapper.device_mesh.mesh_dim_names
                    and "dp_shard_cp" in model_wrapper.device_mesh.mesh_dim_names
                    else ("dp_shard_cp",)
                ),
                cp_axis_name="cp",
                tp_axis_name="tp",
                ep_axis_name=ep_axis_name,
                ep_shard_axis_names=ep_shard_axis_names,
            )
            load_weights = True
        elif callable(getattr(model_wrapper, "parallelize", None)):
            if isinstance(model_wrapper, MegatronFSDPManager):
                trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
                assert len(trainable_params) > 0, "trainable_params cannot be empty"
                if tp_size > 1:
                    cfg_opt.foreach = False
                optimizer = cfg_opt.instantiate(params=trainable_params)
                model, optimizer = model_wrapper.parallelize(model, optimizer)
                return model, state_dict_keys, optimizer
            else:
                load_weights = True
                model = model_wrapper.parallelize(model)

        # Load the weights into the model in parallel.
        if is_meta_device and load_weights:
            checkpointer.load_base_model(
                model,
                device,
                cfg_model.get("cache_dir", TRANSFORMERS_CACHE),
                _get_model_name(cfg_model),
                getattr(cfg_peft, "lora_A_init", None),
                load_base_model=load_base_model,
            )

        print_trainable_parameters(model)
        model = model.to(device)

        # Apply torch.compile if configured
        if cfg_compile is not None:
            compile_config = build_compile_config(cfg_compile)
            model = compile_model(model, compile_config)

        if tp_size > 1:
            # TP does not support foreach
            cfg_opt.foreach = False

        # modified
        # trainable_params = list(filter(lambda x: x.requires_grad, model.parameters()))
        # assert len(trainable_params) > 0, "trainable_params cannot be empty"
        # optimizer = cfg_opt.instantiate(params=trainable_params)

        param_groups = get_parameter_groups(model, cfg_opt)
        assert len(param_groups) > 0, "No trainable parameters found!"

        # remove useless field
        if hasattr(cfg_opt, 'visual_lr'):
            delattr(cfg_opt, 'visual_lr')
        if hasattr(cfg_opt, 'merger_lr'):
            delattr(cfg_opt, 'merger_lr')

        optimizer = cfg_opt.instantiate(params=param_groups)

        return model, state_dict_keys, optimizer


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft) -> CheckpointingConfig:
    """Build a checkpoint configuration.

    Args:
        cfg_ckpt: Configuration for checkpointing.
        cache_dir: Cache directory for the model.
        model_repo_id: Model repository ID.
        is_peft: Whether the model is PEFT.

    Returns:
        The instantiated checkpoint configuration.
    """
    ckpt_kwargs = dict(
        enabled=True,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else TRANSFORMERS_CACHE,
        save_consolidated=True,
        is_peft=is_peft,
    )
    if cfg_ckpt is not None:
        cfg_ckpt = cfg_ckpt.to_dict()
        cfg_ckpt.pop("restore_from", None)
        cfg_ckpt.pop("load_base_model", None)
        ckpt_kwargs |= cfg_ckpt
    if ckpt_kwargs.get("is_peft", False) and ckpt_kwargs.get("model_save_format") == "torch_save":
        raise ValueError(
            "PEFT checkpointing is not supported for torch_save format. Save using `safetensors` format instead."
        )
    checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
    return checkpoint_config


def build_loss_fn(cfg_loss):
    """Build a loss function.

    Args:
        cfg_loss: Loss function configuration.

    Returns:
        The instantiated loss function.
    """
    return cfg_loss.instantiate()


def build_dataloader(
    cfg_ds, cfg_dl, pretrained_model_name_or_path, cfg_processor, device_mesh, seed, local_batch_size
) -> tuple[DataLoader, ProcessorMixin]:
    """Build a DataLoader for the VLM dataset.

    Args:
        cfg_ds: Dataset configuration.
        cfg_dl: DataLoader configuration.
        pretrained_model_name_or_path: Pretrained model name or path for processor loading.
        cfg_processor: Processor configuration or None.
        device_mesh: Device mesh for distributed training.
        seed: Random seed.
        local_batch_size: Local batch size.

    Returns:
        The instantiated DataLoader and processor.
    """
    dist_sampler_kwargs = {
        "shuffle": cfg_dl.get("shuffle", True),
    }
    if device_mesh is not None:
        dist_sampler_kwargs |= {
            "num_replicas": device_mesh["dp"].size(),
            "rank": device_mesh["dp"].get_local_rank(),
        }

    with ScopedRNG(seed=seed, ranked=True):
        processor = None
        processor_kwargs = {}
        if cfg_processor is not None and hasattr(cfg_processor, "instantiate"):
            processor = cfg_processor.instantiate()
        elif cfg_processor is not None:
            processor_kwargs = cfg_processor.to_dict()

        # If no processor was instantiated, try AutoProcessor
        if processor is None:
            try:
                processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
            except Exception as e:
                # Some models do not provide an AutoProcessor
                processor = None
                logging.warning(f"AutoProcessor not available for {pretrained_model_name_or_path} ({e}). ")

        with FirstRankPerNode():
            # ds = cfg_ds.instantiate(path_or_dataset=cfg_ds.path_or_dataset)
            ds_dict = {k: v for k, v in cfg_ds.__dict__.items() if not k.startswith('_')}
            ds = cfg_ds.instantiate(**ds_dict)

        dataset_meta = ds.get_metadata()
        if dataset_meta is not None:
            logging.info("Using BalanceSampler.")
            lengths, v_tokens = dataset_meta[0], dataset_meta[1]
            sampler = BalanceSampler(
                lengths=lengths,
                v_tokens=v_tokens,
                local_batch_size=local_batch_size,
                seed=seed,
                drop_last=cfg_ds.drop_last,
                **dist_sampler_kwargs,
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds,
                **dist_sampler_kwargs,
            )

        collate_cfg = cfg_dl.get("collate_fn", None)
        if collate_cfg:
            collate_fn = collate_cfg.instantiate(processor=processor, max_len=cfg_ds.max_len)
        else:
            processor_type = type(processor).__name__
            if processor_type not in COLLATE_FNS:
                processor_type = "default"
                logging.warning(f"You are using {processor_type} with default collate function.")
            collate_fn = lambda examples: COLLATE_FNS[processor_type](examples, processor)

        return cfg_dl.instantiate(
            dataset=ds, sampler=sampler, collate_fn=collate_fn, batch_size=local_batch_size
        ), processor

def build_distributed(cfg_dist: Dict[str, Any]) -> "DistInfo":  # noqa: F821
    """Build and initialize distributed training resources.

    Args:
        cfg_dist: Configuration for distributed training.

    Returns:
        Distributed training information from initialize_distributed.
    """
    backend = cfg_dist.get("backend", "nccl")
    timeout = cfg_dist.get("timeout_minutes", 1)
    return initialize_distributed(backend=backend, timeout_minutes=timeout)


def build_step_scheduler(cfg, dataloader, dp_group_size, local_batch_size):
    """Build the step scheduler.

    Args:
        cfg: configuration for the StepScheduler class.
        dataloader: the training dataloader, used for extracting the epoch_len (in batches).
        dp_group_size: the size of the data parallel group.
        micro_batch_size: the size of the micro batch.

    Returns:
        StepScheduler: the configured StepScheduler.
    """
    assert "_target_" not in cfg, "_target_ not permitted in step scheduler"
    default_kwargs = dict(
        num_epochs=10,
        global_batch_size=32,
        local_batch_size=local_batch_size,
        dp_size=dp_group_size,
        ckpt_every_steps=100,
        dataloader=dataloader,
    )
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)


def build_lr_scheduler(cfg, optimizer, step_scheduler) -> OptimizerParamScheduler | None:  # noqa: F821
    """Build the learning rate scheduler.

    Args:
        cfg: Configuration for the OptimizerParamScheduler.
        optimizer: The optimizer to be scheduled.
        step_scheduler: The step scheduler to extract training parameters.

    Returns:
        OptimizerParamScheduler: The configured learning rate scheduler, or None if not configured.
    """
    if cfg is None:
        return None

    # Calculate total steps for the training run
    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps

    # Total optimizer steps (accounting for gradient accumulation)
    total_steps = (total_epochs * epoch_len) // grad_acc_steps
    if step_scheduler.max_steps is not None:
        total_steps = min(total_steps, step_scheduler.max_steps)

    # Extract learning rate from optimizer
    # 取第一组的 lr 作为 scheduler 的名义全局 lr, 用于过 init 校验.
    base_lr = optimizer.param_groups[0].get("max_lr", optimizer.param_groups[0]["lr"])

    # Set defaults for scheduler parameters
    default_kwargs = dict(
        optimizer=optimizer,
        init_lr=base_lr * 0.1,  # Start warmup at 10% of base LR
        max_lr=base_lr,
        min_lr=base_lr * 0.01,  # End at 1% of base LR
        lr_warmup_steps=min(1000, total_steps // 10),  # 10% warmup or max 1000 steps
        lr_decay_steps=total_steps,
        lr_decay_style="cosine",
        # start_wd=optimizer.param_groups[0].get("weight_decay", 0.0),
        # end_wd=optimizer.param_groups[0].get("weight_decay", 0.0),
        start_wd=1.0,
        end_wd=1.0,
        wd_incr_steps=total_steps,
        wd_incr_style="constant",
    )

    # Override with user-provided config
    if cfg is not None:
        user_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        default_kwargs.update(user_cfg)

    logger.info(
        f"Building LR scheduler with total_steps={total_steps}, "
        f"warmup_steps={default_kwargs['lr_warmup_steps']}, "
        f"decay_style={default_kwargs['lr_decay_style']}"
    )

    return OptimizerParamScheduler(**default_kwargs)


def build_wandb(cfg) -> wandb.Run:
    """Instantiates wandb and returns the instance. If no name is given, it will use the model name.

    Args:
        cfg: Configuration for wandb.

    Returns:
        The wandb instance.
    """
    assert cfg.get("wandb", None) is not None
    kwargs = cfg.wandb.to_dict()
    if kwargs.get("name", "") == "":
        kwargs["name"] = "_".join(_get_model_name(cfg.model).split("/")[-2:])
    run = wandb.init(
        **kwargs,
        config=cfg.to_dict(),
        settings=Settings(silent=True),
    )
    return run


def calculate_loss(loss_fn, **kwargs) -> torch.Tensor:
    """Calculate the loss.

    Args:
        loss_fn: Loss function.
        **kwargs: Keyword arguments for the loss function.

    Returns:
        The loss.
    """
    loss_fn_kwargs = {}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")

        # find the lm_head in the model
        lm_head = None
        if hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings().weight
        else:
            for n, p in model.named_parameters(remove_duplicate=False):
                if "lm_head" in n and n.endswith(".weight"):
                    lm_head = p
                    break
        if lm_head is None:
            raise ValueError("lm_head.weight not found in model")

        # unshard the possibly sharded lm_head
        lm_head = lm_head.full_tensor() if hasattr(lm_head, "full_tensor") else lm_head
        loss_fn_kwargs.update(
            {
                "hidden_states": kwargs.pop("hidden_states"),
                "labels": labels,
                "lm_weight": lm_head,
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
            }
        )
    elif isinstance(loss_fn, MixturePathGeneralizeKL):
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
                "x_t": kwargs.pop("x_t"),
                "t": kwargs.pop("t"),
                "response_mask": kwargs.pop("response_mask"),
                "loss_mask": kwargs.pop("loss_mask"),
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
                "num_samples": kwargs.pop("num_samples", None),
                "block_size": kwargs.pop("block_size", 0),
            }
        )
    elif isinstance(loss_fn, WeightedCrossEntropy):
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
                "x_t": kwargs.pop("x_t"),
                "t": kwargs.pop("t"),
                "response_mask": kwargs.pop("response_mask"),
                "loss_mask": kwargs.pop("loss_mask"),
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
                "num_samples": kwargs.pop("num_samples", None),
                "block_size": kwargs.pop("block_size", 0),
            }
        )
    else:
        loss_fn_kwargs.update(
            {
                "logits": kwargs.pop("logits"),
                "labels": kwargs.pop("labels"),
                "num_label_tokens": kwargs.pop("num_label_tokens", None),
            }
        )

    return loss_fn(**loss_fn_kwargs)

# ---------------------------------------------------------------------------
#  Trainer class – orchestration only
# ---------------------------------------------------------------------------

class FinetuneRecipeForVLM(BaseRecipe):
    """Recipe for fine-tuning a VLM model."""

    def __init__(self, cfg):
        """Initialize the recipe with configuration.

        Args:
            cfg: Configuration dictionary/object for training.
        """
        self.cfg = cfg

    # ------------------ build phase ------------------
    def setup(self):
        """Builds all components needed for training/validation/logging/checkpointing/etc.

        This is the last place where self.cfg should be referenced.

        Raises:
            NotImplemented: Raises if it tries to restore a checkpoint; will be removed.
        """
        torch.cuda.reset_peak_memory_stats()
        self.dist_env = build_distributed(self.cfg.get("dist_env", {}))
        setup_logging()

        apply_cache_compatibility_patches()

        # Set up the stateful random number generator
        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        self.device_mesh = None
        self.moe_mesh = None
        self.model_wrapper = None
        if "distributed" in self.cfg:
            self.model_wrapper = self.cfg.distributed.instantiate(world_size=self.dist_env.world_size)
            self.device_mesh = getattr(self.model_wrapper, "device_mesh", None)
            self.moe_mesh = getattr(self.model_wrapper, "moe_mesh", None)

        if self.dist_env.is_main and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info("🚀 View run at {}".format(run.url))

        # Log experiment details on main rank
        self._log_experiment_details()
        self._log_library_versions()

        # Build components with VLM-specific functions
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)
        parallelize_fn = getattr(self.cfg.get("parallelizer", None), "instantiate", None)

        # Build checkpoint config
        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            _get_model_name(self.cfg.model),
            True if self.cfg.get("peft", None) else False,
        )

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified in config, using default value of 1.0")
            self.max_grad_norm = 1.0

        # Create Checkpointer instance
        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=self._get_dp_rank(include_cp=True),
            tp_rank=self._get_tp_rank(),
            pp_rank=self._get_pp_rank(),
            moe_mesh=self.moe_mesh,
        )

        self.model, model_state_dict_keys, self.optimizer = build_model_and_optimizer(
            self.dist_env.device,
            self.cfg.model,
            self.cfg.optimizer,
            self.cfg.get("freeze_config", None),
            self.peft_config,
            self.model_wrapper,
            seed=self.cfg.get("seed", 42),
            tp_size=self.cfg.get("distributed.tp_size", 1),
            cp_size=self.cfg.get("distributed.cp_size", 1),
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
            loss_fn=self.loss_fn,
            parallelize_fn=parallelize_fn,
            load_base_model=self.cfg.get("checkpoint.load_base_model", True),
            checkpointer=self.checkpointer,
        )
        self.checkpointer.config.model_state_dict_keys = model_state_dict_keys

        self.dataloader, self.processor = build_dataloader(
            self.cfg.dataset,
            self.cfg.dataloader,
            _get_model_name(self.cfg.model),
            self.cfg.get("processor", None),
            device_mesh=self.device_mesh,
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        # Build validation dataloader if the config provides it
        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset,
                self.cfg.validation_dataloader,
                _get_model_name(self.cfg.model),
                self.cfg.get("processor", None),
                device_mesh=self.device_mesh,
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            )

        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        # Scheduler
        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self._get_dp_group_size(),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        # Build learning rate scheduler
        self.lr_scheduler = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)

        # Log model, parameter counts, norms, optimizer and scheduler
        self._log_model_and_optimizer_details(self.model, self.optimizer, self.lr_scheduler)

        restore_from = self.cfg.get("checkpoint.restore_from", None)

        # Initialize JSONL loggers
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )

        # Optionally resume
        self.load_checkpoint(restore_from)

        # Log step scheduler details
        self._log_step_scheduler_details(self.step_scheduler)

    # ------------------ main loop ------------------
    def run_train_validation_loop(self):
        """Run the training loop over all epochs and batches.

        For each batch, perform a forward pass, compute loss, backpropagate,
        and update model parameters when necessary. Also prints loss every gradient step.
        """
        total_steps = self.step_scheduler.max_steps
        prior_dist_2 = getattr(self.dataloader.dataset, "prior_dist_2", None)
        switch_prior_thresh = self.cfg.dataset.get("switch_prior_thresh", 1.0)
        switch_step = max(0, int(total_steps * switch_prior_thresh))

        self.model.train()
        self.timestamp = time.perf_counter()

        # ---------- Profiler setup ----------
        prof = None
        if os.environ.get("ENABLE_PROFILER", "").lower() in ("1", "true"):
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=5, warmup=2, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.environ.get("PROFILER_LOG_DIR", "./prof_log")
                ),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            prof.start()
            print(f"[Profiler] Enabled, logs → {os.environ.get('PROFILER_LOG_DIR', './prof_log')}")

        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            for batch_idx, batches in enumerate(self.step_scheduler): # batches是个list，长度为acc_steps, 每个item维度是local_batch_size
                cur_step = self.step_scheduler.step
                # switch noisy prior dist for curriculum learning
                if prior_dist_2 is not None and cur_step > switch_step:
                    self.dataloader.dataset.prior_dist = prior_dist_2
                    prior_dist_2 = None

                # print(batches)

                log_data = self._run_train_optim_step(batches, self.max_grad_norm)
                # if self.lr_scheduler is not None:
                #     self.lr_scheduler.step(1)

                # log
                self.log_train_metrics(log_data)

                if prof is not None:
                    prof.step()

                val_loss = {}
                if self.step_scheduler.is_val_step and self.val_dataloader is not None:
                    val_log_data = self._run_validation_epoch(self.val_dataloader)
                    val_loss["val_loss"] = val_log_data.metrics["val_loss"]
                    self.log_val_metrics(val_log_data)
                    self.model.train()

                if self.step_scheduler.is_ckpt_step:
                    self.save_checkpoint(
                        epoch,
                        self.step_scheduler.step,
                        log_data.metrics["loss"],
                        val_loss,
                        best_metric_key=self.best_metric_key,
                    )

        # Close JSONL loggers after training loop completes
        self.metric_logger_train.close()
        self.metric_logger_valid.close()

        if prof is not None:
            prof.stop()
            print("[Profiler] Stopped.")

        self.checkpointer.close()

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step.

        Args:
            batches: List of batches of training data.
            max_grad_norm: Gradient clipping norm. Optional, if None will not clip gradients.
        """
        if 'loss_mask' in batches[0]:
            num_label_tokens = torch.tensor(
                sum((batch["loss_mask"]).sum().item() for batch in batches), dtype=torch.long
            )
        else:
            num_label_tokens = torch.tensor(
                sum((batch["labels"] != -100).sum().item() for batch in batches), dtype=torch.long
            )

        global_batch_size = torch.tensor(len(batches))
        global_batch_size = self._dp_allreduce(global_batch_size).item()

        # 单节点一次iteration的样本数
        num_samples = torch.tensor(
            sum((batch["num_samples"]).sum().item() for batch in batches), dtype=torch.long
        )
        # 全部节点一次iteration时的样本总数
        num_total_samples = self._dp_allreduce(num_samples).item()
        num_processes = dist.get_world_size()

        num_label_tokens = self._dp_allreduce(num_label_tokens).item()
        loss_buffer = []

        # number of tokens in the batch, excluding any tail padding.
        num_tokens_in_batch = torch.tensor(
            sum(batch["labels"].numel() - count_tail_padding(batch["labels"]) for batch in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        num_batches = len(batches)
        # ---------- 逐阶段计时 ----------
        _t_data = 0.0
        _t_fwd = 0.0
        _t_loss = 0.0
        _t_bwd = 0.0
        _t_comm = 0.0

        for i, batch in enumerate(batches): # accumulation_steps维度迭代
            # batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
            _t0 = time.perf_counter()
            batch = to_device(batch, self.dist_env.device)
            labels = batch.pop("labels")
            _t_data += time.perf_counter() - _t0

            batch.pop("raw_messages", None)

            train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels) # local_batch_size维度迭代
            with (
                train_ctx(),
                get_sync_ctx(
                    self.model,
                    i == num_batches - 1,
                    defer_fsdp_grad_sync=getattr(self.model_wrapper, "defer_fsdp_grad_sync", True),
                ),
            ):
                _t0 = time.perf_counter()
                if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                    # use num_logits_to_keep to avoid full logits matrix in memory
                    out = self.model(logits_to_keep=1, **batch)
                    if "hidden_states" not in out:
                        raise ValueError(
                            "FusedLinearCrossEntropy requires the model to output hidden states. Set `model.output_hidden_states=True` in the config."
                        )
                else:
                    out = self.model(**batch)
                _t_fwd += time.perf_counter() - _t0

                # ---------- DEBUG: 打印模型输出 ----------
                if os.environ.get("DEBUG", "").lower() in ("1", "true") and i == len(batches) - 1:
                    with torch.no_grad():
                        logits = getattr(out, "logits", out)
                        pred_ids = logits.argmax(dim=-1)
                        # 只取第一个样本，用 response_mask 定位 response 部分
                        resp_mask = batch.get("response_mask", None)
                        if resp_mask is not None:
                            r_mask = resp_mask[0].bool()
                            # input 中的 noisy response
                            noisy_input = batch["input_ids"][0][r_mask]
                            # 模型预测的 response
                            pred_resp = pred_ids[0][r_mask]
                            # 标签的 response
                            label_resp = labels[0][r_mask]
                            noisy_text = self.processor.tokenizer.decode(noisy_input, skip_special_tokens=False)
                            pred_text = self.processor.tokenizer.decode(pred_resp, skip_special_tokens=True)
                            label_text = self.processor.tokenizer.decode(label_resp, skip_special_tokens=True)
                        else:
                            pred_text = self.processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                            label_text = self.processor.tokenizer.decode(labels[0], skip_special_tokens=True)
                            noisy_text = self.processor.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
                        print(f"\n[DEBUG] === Model Output (step {self.step_scheduler.step}) ===")
                        print(f"[DEBUG] Input (noisy):\n{noisy_text[-300:]}")
                        print(f"[DEBUG] Prediction:\n{pred_text[-100:]}")
                        print(f"[DEBUG] Label (ground truth):\n{label_text[-300:]}")
                        print(f"[DEBUG] {'='*60}")

                _t0 = time.perf_counter()
                ########### origin CrossEntropyLoss #############
                # local_loss = calculate_loss(
                #     self.loss_fn,
                #     logits=getattr(out, "logits", out),
                #     labels=labels,
                #     model=self.model,
                #     hidden_states=out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else None,
                #     num_label_tokens=num_label_tokens,
                # )
                ######## modified for GeneralizedKL loss ########
                local_loss = calculate_loss(
                    self.loss_fn,
                    logits=getattr(out, "logits", out),
                    labels=labels,
                    model=self.model,
                    hidden_states=out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else None,
                    num_label_tokens=num_label_tokens,
                    x_t=batch.get("input_ids", None),
                    t=batch.get("t", None),
                    loss_mask=batch.get("loss_mask", None),
                    response_mask=batch.get("response_mask", None),
                    block_size=batch.get("block_size") if "block_size" in batch else 0, # for block diffusion.
                    num_samples=num_total_samples,
                    # num_samples=num_samples * num_processes,
                )
                _t_loss += time.perf_counter() - _t0

                _t0 = time.perf_counter()
                loss_buffer.append(local_loss.clone().detach())
                local_loss.backward()
                _t_bwd += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm=max_grad_norm,
            model_parts=[self.model],
            norm_type=2.0,
            pp_enabled=False,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name=None,
            foreach=True,
            num_label_tokens=num_label_tokens,
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )
        _t_comm += time.perf_counter() - _t0

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.finish_grad_sync()

        self.checkpointer.maybe_wait_for_staging()
        _t0 = time.perf_counter()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        _t_opt = time.perf_counter() - _t0

        if hasattr(self.model, "update_moe_gate_bias"):
            self.model.update_moe_gate_bias()

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(1)

        # Note(MegatronFSDP): Need to call these functions for MegatronFSDP if not using latest api
        # self.model.install_optimized_model_weights()
        # self.model.zero_grad_buffer()

        # TPS is calculated as follows (assuming grad-accumulation-steps=2):
        # fwd 0 | bwd 0 | fwd 1 | bwd 1 | opt 0 | fwd 2 | bwd 2 | ...
        # ^                                     ^
        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta
        # ---------- 打印逐阶段计时 ----------
        _t_total = _t_data + _t_fwd + _t_loss + _t_bwd + _t_comm + _t_opt
        if _t_total > 0 and os.environ.get("WAM_DIFF_PROFILE", "").lower() in ("1", "true", "yes"):
            print(f"[Timing] data:{_t_data:.3f}s({100*_t_data/_t_total:.0f}%) "
                  f"fwd:{_t_fwd:.3f}s({100*_t_fwd/_t_total:.0f}%) "
                  f"loss:{_t_loss:.3f}s({100*_t_loss/_t_total:.0f}%) "
                  f"bwd:{_t_bwd:.3f}s({100*_t_bwd/_t_total:.0f}%) "
                  f"grad_sync:{_t_comm:.3f}s({100*_t_comm/_t_total:.0f}%) "
                  f"opt:{_t_opt:.3f}s({100*_t_opt/_t_total:.0f}%) "
                  f"total:{_t_total:.3f}s")
        reporting_loss = torch.sum(torch.stack(loss_buffer))
        reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True).item()
        # fix reporting_loss, tps across ranks

        fallback_lr = self.optimizer.param_groups[0]["lr"]

        def group_lr(prefix: str) -> float:
            return next(
                (group["lr"] for group in self.optimizer.param_groups if group.get("name", "").startswith(prefix)),
                fallback_lr,
            )

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "grad_norm": grad_norm,
                "visual_lr": group_lr("visual"),
                "merger_lr": group_lr("merger"),
                "language_lr": group_lr("language"),
                "single_samples": num_samples,
                "total_samples": num_total_samples, # total train samples in each iteration
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / max(self._get_dp_group_size(), 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
            },
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one pass over `self.val_dataloader`."""
        with ScopedRNG(seed=1, ranked=True):
            self.model.eval()

            total_loss = 0.0
            total_tokens = 0
            total_num_label_tokens = 0
            for batch in val_dataloader:
                batch = {k: v.to(self.dist_env.device, non_blocking=True) for k, v in batch.items()}
                labels = batch.pop("labels")
                num_label_tokens = (labels != -100).sum().item()

                if (
                    self.device_mesh
                    and "position_ids" not in batch
                    and (self.device_mesh["cp"].size() > 1 or self.device_mesh["tp"].size() > 1)
                ):
                    batch["position_ids"] = (
                        torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).to(self.model.device)
                    )

                train_ctx, batch = make_cp_batch_and_ctx(self.device_mesh, batch, labels)
                with train_ctx():
                    if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                        out = self.model(logits_to_keep=1, **batch)
                    else:
                        out = self.model(**batch)
                    local_loss = calculate_loss(
                        self.loss_fn,
                        logits=getattr(out, "logits", out),
                        labels=labels,
                        model=self.model,
                        hidden_states=out.hidden_states[-1]
                        if getattr(out, "hidden_states", None) is not None
                        else None,
                        num_label_tokens=num_label_tokens,
                    )
                    total_num_label_tokens += num_label_tokens

                total_loss += local_loss.item() * num_label_tokens
                total_tokens += num_label_tokens

        # Aggregate across ranks if distributed is initialized
        total_loss = self._dp_allreduce(torch.FloatTensor([total_loss]), include_cp=True).item()
        total_tokens = self._dp_allreduce(torch.LongTensor([total_tokens]), include_cp=True).item()
        total_num_label_tokens = self._dp_allreduce(torch.LongTensor([total_num_label_tokens])).item()

        val_loss = total_loss / max(total_tokens, 1e-8)

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "num_label_tokens": total_num_label_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_val_metrics(self, log_data):
        """Log metrics to wandb and other loggers
        Args:
            log_data: MetricsSample object, containing:
                step: int, the current step.
                epoch: int, the current epoch.
                metrics: Dict[str, float], containing:
                    "val_loss": Validation loss.
                    "lr": Learning rate.
                    "num_label_tokens": Number of label tokens.
                    "mem": Memory allocated.
        """

        if not self.dist_env.is_main or log_data is None:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)

        # JSONL validation log
        self.metric_logger_valid.log(log_data)

        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["val_loss"],
                log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
            )
        )

    def log_train_metrics(self, log_data) -> float:
        """Log metrics to wandb.

        Args:
            train_loss: Training loss.
            grad_norm: Grad norm from the training step.
            num_tokens_in_batch: Total number of loss tokens.
            tps: Tokens per second.
        """
        if not self.dist_env.is_main:
            return

        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=self.step_scheduler.step)
        # JSONL training log
        self.metric_logger_train.log(log_data)
        # logging.info(
        #     "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}".format(
        #         log_data.step,
        #         log_data.epoch,
        #         log_data.metrics["loss"],
        #         log_data.metrics["grad_norm"],
        #         log_data.metrics["lr"],
        #         log_data.metrics["mem"],
        #         log_data.metrics["tps"],
        #         log_data.metrics["tps_per_gpu"],
        #         log_data.metrics["num_label_tokens"],
        #     )
        # )

        logging.info(
            "step {} | epoch {} | loss {:.4f} | grad_norm {:.4f} | visual_lr {:.2e} | merger_lr {:.2e} | language_lr {:.2e} | single_samples {:d} | total_samples {:d} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | num_label_tokens {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["loss"],
                log_data.metrics["grad_norm"],
                log_data.metrics["visual_lr"],
                log_data.metrics["merger_lr"],
                log_data.metrics["language_lr"],
                log_data.metrics["single_samples"],
                log_data.metrics["total_samples"],
                log_data.metrics["mem"],
                log_data.metrics["tps"],
                log_data.metrics["tps_per_gpu"],
                log_data.metrics["num_label_tokens"],
            )
        )

        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path=None):
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the trainer, and initiates the training loop.
    """
    if config_path is None:
        config_path = pathlib.Path(__file__).resolve().parents[3] / "configs" / "training" / "block32.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
