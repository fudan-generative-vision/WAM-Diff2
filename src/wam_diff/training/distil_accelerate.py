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

# This file has been modified by the WAM Diff2 project.

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import logging
import pathlib
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Dict, Optional

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from transformers.modeling_utils import no_init_weights
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TRANSFORMERS_CACHE, ContextManagers
from wandb import Settings

from accelerate import Accelerator

from wam_diff._transformers.utils import apply_cache_compatibility_patches
from wam_diff.components._peft.lora import apply_lora_to_linear_modules
from wam_diff.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from wam_diff.components.config._arg_parser import parse_args_and_load_config
from wam_diff.data.collate_fns import COLLATE_FNS
from wam_diff.components.distributed.utils import FirstRankPerNode
from wam_diff.components.loggers.log_utils import setup_logging
from wam_diff.components.loggers.metric_logger import MetricsSample, build_metric_logger
from wam_diff.components.loggers.wandb_utils import suppress_wandb_log_messages
from wam_diff.losses.linear_ce import FusedLinearCrossEntropy
from wam_diff.losses.masked_ce import MaskedCrossEntropy
from wam_diff.losses.mpg_kl import MixturePathGeneralizeKL
from wam_diff.losses.weighted_ce import WeightedCrossEntropy
from wam_diff.data.sampler import BalanceSampler
from wam_diff.components.optim.scheduler import OptimizerParamScheduler
from wam_diff.components.training.rng import ScopedRNG, StatefulRNG
from wam_diff.components.training.step_scheduler import StepScheduler
from wam_diff.components.training.utils import count_tail_padding
from wam_diff.components.utils.compile_utils import build_compile_config, compile_model
from wam_diff.components.utils.model_utils import (
    _supports_logits_to_keep,
    apply_parameter_freezing,
    init_empty_weights,
    print_trainable_parameters,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# ---------------------------
#  Stateless helper functions
# ---------------------------


def _get_model_name(cfg_model):
    if cfg_model.get("pretrained_model_name_or_path", None) is not None:
        return cfg_model.pretrained_model_name_or_path
    elif cfg_model.get("config", None) is not None:
        return cfg_model.config.get("pretrained_model_name_or_path", None)
    else:
        return None


def _freeze_model(model: nn.Module, cfg_freeze: Optional[Dict[str, Any]] = None, freeze_embeddings: bool = True):
    if cfg_freeze is not None:
        apply_parameter_freezing(model, cfg_freeze)
    elif freeze_embeddings:
        logging.info("Freezing embeddings")
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                m.weight.requires_grad = False
    return model


def get_parameter_groups(model, cfg_opt):
    base_lr = cfg_opt.get("lr", 1e-5)
    visual_lr = cfg_opt.get("visual_lr", base_lr)
    language_lr = cfg_opt.get("language_lr", base_lr)
    merger_lr = cfg_opt.get("merger_lr", base_lr)
    weight_decay = cfg_opt.get("weight_decay", 0.0)

    groups = {
        "visual_decay": {
            "params": [], "lr": visual_lr, "max_lr": visual_lr,
            "min_lr": visual_lr * 0.05, "init_lr": visual_lr * 0.1,
            "weight_decay": weight_decay, "wd_mult": weight_decay,
        },
        "visual_no_decay": {
            "params": [], "lr": visual_lr, "max_lr": visual_lr,
            "min_lr": visual_lr * 0.05, "init_lr": visual_lr * 0.1,
            "weight_decay": 0.0, "wd_mult": 0.0,
        },
        "language_decay": {
            "params": [], "lr": language_lr, "max_lr": language_lr,
            "min_lr": language_lr * 0.05, "init_lr": language_lr * 0.1,
            "weight_decay": weight_decay, "wd_mult": weight_decay,
        },
        "language_no_decay": {
            "params": [], "lr": language_lr, "max_lr": language_lr,
            "min_lr": language_lr * 0.05, "init_lr": language_lr * 0.1,
            "weight_decay": 0.0, "wd_mult": 0.0,
        },
        "merger_decay": {
            "params": [], "lr": merger_lr, "max_lr": merger_lr,
            "min_lr": merger_lr * 0.05, "init_lr": merger_lr * 0.1,
            "weight_decay": weight_decay, "wd_mult": weight_decay,
        },
        "merger_no_decay": {
            "params": [], "lr": merger_lr, "max_lr": merger_lr,
            "min_lr": merger_lr * 0.05, "init_lr": merger_lr * 0.1,
            "weight_decay": 0.0, "wd_mult": 0.0,
        },
        "other": {
            "params": [], "lr": base_lr, "max_lr": base_lr,
            "min_lr": base_lr * 0.05, "init_lr": base_lr * 0.1,
            "weight_decay": weight_decay, "wd_mult": weight_decay,
        },
    }

    no_decay_keywords = ["norm", "bias", "embed_tokens", "pos_embed"]
    seen_param_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_param_ids:
            logger.info(f"Skipping tied parameter: {name}")
            continue
        seen_param_ids.add(id(param))

        is_no_decay = any(k in name.lower() for k in no_decay_keywords) or (param.ndim <= 1)

        if "visual.merger" in name or "deepstack_merger_list" in name:
            key = "merger_no_decay" if is_no_decay else "merger_decay"
        elif "visual" in name:
            key = "visual_no_decay" if is_no_decay else "visual_decay"
        elif "language_model" in name or "lm_head" in name:
            key = "language_no_decay" if is_no_decay else "language_decay"
        else:
            key = "other"

        groups[key]["params"].append(param)

    param_groups = [v for k, v in groups.items() if len(v["params"]) > 0]

    for k, v in groups.items():
        if len(v["params"]) > 0:
            logger.info(f"Group {k}: {len(v['params'])} params, max_lr={v['max_lr']:.1e}, min_lr={v['min_lr']:.1e}")

    return param_groups


def _build_teacher_model(cfg_teacher, seed, device):
    """Build and load a frozen teacher model for distillation."""
    assert cfg_teacher is not None, "`teacher_model` section missing from YAML config"
    logger.info("Instantiating teacher model (frozen, no DDP)")

    with ScopedRNG(seed=seed, ranked=True):
        teacher_model = cfg_teacher.instantiate()
        teacher_model = teacher_model.to(device)
        # Keep teacher in train() mode — WAMDiff2's forward uses self.training
        # to select the attention implementation. In train() mode without
        # block_metadata, it uses SDPA which handles custom attention masks.
        # In eval() mode it uses flash_attention_2 which is incompatible with
        # the block causal mask from collate_fn.
        teacher_model.train()

        for m in teacher_model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

        for p in teacher_model.parameters():
            p.requires_grad_(False)

        n_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info(f"Teacher model loaded: {n_params} params (all frozen), device={device}")
        return teacher_model


def build_model_and_optimizer(device, cfg_model, cfg_opt, cfg_freeze, cfg_peft, seed,
                              checkpointer, cfg_fp8=None, cfg_compile=None,
                              loss_fn=None, load_base_model=True):
    """Build model + optimizer without FSDP/DDP. Accelerate handles wrapping later."""

    # Always use meta device init — Accelerate will move to GPU via prepare()
    init_ctx = ContextManagers([no_init_weights(), init_empty_weights()])
    with ScopedRNG(seed=seed, ranked=True):
        with init_ctx:
            model = cfg_model.instantiate()
            model = _freeze_model(model, cfg_freeze)
            if cfg_peft is not None:
                apply_lora_to_linear_modules(model, cfg_peft)
            if cfg_fp8 is not None:
                from wam_diff.components.quantization.fp8 import apply_fp8_to_model, build_fp8_config
                fp8_config = build_fp8_config(cfg_fp8)
                model = apply_fp8_to_model(model, config=fp8_config)

        state_dict_keys = model.state_dict().keys()

        if not _supports_logits_to_keep(model) and not isinstance(loss_fn, MaskedCrossEntropy):
            logger.warning("logits_to_keep not found in model.forward. Using MaskedCrossEntropy instead.")
            loss_fn = MaskedCrossEntropy()

        # Load weights into meta model
        checkpointer.load_base_model(
            model, device,
            cfg_model.get("cache_dir", TRANSFORMERS_CACHE),
            _get_model_name(cfg_model),
            getattr(cfg_peft, "lora_A_init", None),
            load_base_model=load_base_model,
        )
        print_trainable_parameters(model)
        # Do NOT call model.to(device) — Accelerator.prepare handles device placement

        if cfg_compile is not None:
            compile_config = build_compile_config(cfg_compile)
            model = compile_model(model, compile_config)

        param_groups = get_parameter_groups(model, cfg_opt)
        assert len(param_groups) > 0, "No trainable parameters found!"

        # remove useless field
        for attr in ('visual_lr', 'merger_lr'):
            if hasattr(cfg_opt, attr):
                delattr(cfg_opt, attr)

        optimizer = cfg_opt.instantiate(params=param_groups)
        return model, state_dict_keys, optimizer


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft) -> CheckpointingConfig:
    ckpt_kwargs = dict(
        enabled=True, checkpoint_dir="checkpoints/", model_save_format="safetensors",
        model_repo_id=model_repo_id,
        model_cache_dir=cache_dir if cache_dir is not None else TRANSFORMERS_CACHE,
        save_consolidated=True, is_peft=is_peft,
    )
    if cfg_ckpt is not None:
        cfg_ckpt = cfg_ckpt.to_dict()
        cfg_ckpt.pop("restore_from", None)
        cfg_ckpt.pop("load_base_model", None)
        ckpt_kwargs |= cfg_ckpt
    if ckpt_kwargs.get("is_peft", False) and ckpt_kwargs.get("model_save_format") == "torch_save":
        raise ValueError("PEFT checkpointing is not supported for torch_save format.")
    return CheckpointingConfig(**ckpt_kwargs)


def build_loss_fn(cfg_loss):
    return cfg_loss.instantiate()


def build_dataloader(cfg_ds, cfg_dl, pretrained_model_name_or_path, cfg_processor,
                     seed, local_batch_size) -> tuple[DataLoader, ProcessorMixin]:
    """Build DataLoader. No DistributedSampler — Accelerate handles distribution."""
    with ScopedRNG(seed=seed, ranked=True):
        processor = None
        processor_kwargs = {}
        if cfg_processor is not None and hasattr(cfg_processor, "instantiate"):
            processor = cfg_processor.instantiate()
        elif cfg_processor is not None:
            processor_kwargs = cfg_processor.to_dict()

        if processor is None:
            try:
                processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **processor_kwargs)
            except Exception as e:
                processor = None
                logging.warning(f"AutoProcessor not available for {pretrained_model_name_or_path} ({e}). ")

        with FirstRankPerNode():
            ds_dict = {k: v for k, v in cfg_ds.__dict__.items() if not k.startswith('_')}
            ds = cfg_ds.instantiate(**ds_dict)

        # BalanceSampler needs distributed info; provide it from env if available
        dataset_meta = ds.get_metadata()
        if dataset_meta is not None:
            logging.info("Using BalanceSampler.")
            lengths, v_tokens = dataset_meta[0], dataset_meta[1]
            # BalanceSampler with world_size=1 for now; Accelerate will shard
            sampler = BalanceSampler(
                lengths=lengths, v_tokens=v_tokens,
                local_batch_size=local_batch_size, seed=seed,
                drop_last=cfg_ds.drop_last, shuffle=cfg_dl.get("shuffle", True),
            )
        else:
            # Let Accelerate handle the DistributedSampler
            sampler = None  # DataLoader default; Accelerate.prepare will inject DistributedSampler

        collate_cfg = cfg_dl.get("collate_fn", None)
        if collate_cfg:
            collate_fn = collate_cfg.instantiate(processor=processor, max_len=cfg_ds.max_len)
        else:
            processor_type = type(processor).__name__
            if processor_type not in COLLATE_FNS:
                processor_type = "default"
                logging.warning(f"You are using {processor_type} with default collate function.")
            collate_fn = lambda examples: COLLATE_FNS[processor_type](examples, processor)

        dl_kwargs = dict(dataset=ds, collate_fn=collate_fn, batch_size=local_batch_size)
        if sampler is not None:
            dl_kwargs["sampler"] = sampler
        else:
            dl_kwargs["shuffle"] = cfg_dl.get("shuffle", True)

        return cfg_dl.instantiate(**dl_kwargs), processor


def build_step_scheduler(cfg, dataloader, dp_group_size, local_batch_size):
    assert "_target_" not in cfg, "_target_ not permitted in step scheduler"
    default_kwargs = dict(
        num_epochs=10, global_batch_size=32, local_batch_size=local_batch_size,
        dp_size=dp_group_size, ckpt_every_steps=100, dataloader=dataloader,
    )
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)


def build_lr_scheduler(cfg, optimizer, step_scheduler) -> OptimizerParamScheduler | None:
    if cfg is None:
        return None

    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps
    total_steps = (total_epochs * epoch_len) // grad_acc_steps
    if step_scheduler.max_steps is not None:
        total_steps = min(total_steps, step_scheduler.max_steps)

    base_lr = optimizer.param_groups[0].get("max_lr", optimizer.param_groups[0]["lr"])

    default_kwargs = dict(
        optimizer=optimizer, init_lr=base_lr * 0.1, max_lr=base_lr,
        min_lr=base_lr * 0.01, lr_warmup_steps=min(1000, total_steps // 10),
        lr_decay_steps=total_steps, lr_decay_style="cosine",
        start_wd=1.0, end_wd=1.0, wd_incr_steps=total_steps, wd_incr_style="constant",
    )

    user_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    default_kwargs.update(user_cfg)

    logger.info(f"Building LR scheduler with total_steps={total_steps}, "
                f"warmup_steps={default_kwargs['lr_warmup_steps']}, "
                f"decay_style={default_kwargs['lr_decay_style']}")

    return OptimizerParamScheduler(**default_kwargs)


def build_wandb(cfg) -> wandb.Run:
    assert cfg.get("wandb", None) is not None
    kwargs = cfg.wandb.to_dict()
    if kwargs.get("name", "") == "":
        kwargs["name"] = "_".join(_get_model_name(cfg.model).split("/")[-2:])
    return wandb.init(**kwargs, config=cfg.to_dict(), settings=Settings(silent=True))


def calculate_loss(loss_fn, **kwargs) -> torch.Tensor:
    loss_fn_kwargs = {}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        model = kwargs.pop("model")
        labels = kwargs.pop("labels")
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
        lm_head = lm_head.full_tensor() if hasattr(lm_head, "full_tensor") else lm_head
        loss_fn_kwargs.update({
            "hidden_states": kwargs.pop("hidden_states"), "labels": labels,
            "lm_weight": lm_head, "num_label_tokens": kwargs.pop("num_label_tokens", None),
        })
    elif isinstance(loss_fn, (MixturePathGeneralizeKL, WeightedCrossEntropy)):
        loss_fn_kwargs.update({
            "logits": kwargs.pop("logits"), "labels": kwargs.pop("labels"),
            "x_t": kwargs.pop("x_t"), "t": kwargs.pop("t"),
            "response_mask": kwargs.pop("response_mask"),
            "loss_mask": kwargs.pop("loss_mask"),
            "num_label_tokens": kwargs.pop("num_label_tokens", None),
            "num_samples": kwargs.pop("num_samples", None),
            "block_size": kwargs.pop("block_size", 0),
        })
    else:
        loss_fn_kwargs.update({
            "logits": kwargs.pop("logits"), "labels": kwargs.pop("labels"),
            "num_label_tokens": kwargs.pop("num_label_tokens", None),
        })
    return loss_fn(**loss_fn_kwargs)


# ---------------------------------------------------------------------------
#  Trainer class — Accelerate version
# ---------------------------------------------------------------------------

class DistilRecipeForVLM:
    """Recipe for VLM distillation, powered by HuggingFace Accelerate 1.12.0.

    No FSDP2 / MegatronFSDP / DDP — Accelerate manages all distributed logic.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # ----------------------------------------------------------------
    #  Setup
    # ----------------------------------------------------------------
    def setup(self):
        torch.cuda.reset_peak_memory_stats()
        setup_logging()
        apply_cache_compatibility_patches()

        self.rng = StatefulRNG(seed=self.cfg.get("seed", 42), ranked=True)

        # ---- Mixed precision from config ----
        mixed_precision = "no"
        torch_dtype_str = str(self.cfg.get("model.torch_dtype", ""))
        if "bfloat16" in torch_dtype_str.lower() or "bf16" in torch_dtype_str.lower():
            mixed_precision = "bf16"
        elif "float16" in torch_dtype_str.lower() or "fp16" in torch_dtype_str.lower():
            mixed_precision = "fp16"

        # ---- Gradient accumulation steps ----
        # IMPORTANT: Set to 1 because StepScheduler already manages gradient
        # accumulation by grouping micro-batches into a `batches` list.
        # Setting >1 here would cause double accumulation.
        # When using DeepSpeed, the deepspeed_config yaml also sets
        # gradient_accumulation_steps (typically 1 for the same reason).
        _grad_acc = 1

        # ---- Initialize Accelerator ----
        _ckpt_dir = self.cfg.get("checkpoint.checkpoint_dir", "checkpoints/")
        self.accelerator = Accelerator(
            gradient_accumulation_steps=_grad_acc,
            mixed_precision=mixed_precision,
            project_dir=_ckpt_dir,
        )

        logger.info(f"Accelerator initialized: device={self.accelerator.device}, "
                     f"num_processes={self.accelerator.num_processes}, "
                     f"mixed_precision={mixed_precision}, "
                     f"gradient_accumulation_steps={_grad_acc}")

        # ---- wandb (main process only) ----
        if self.accelerator.is_main_process and hasattr(self.cfg, "wandb"):
            suppress_wandb_log_messages()
            run = build_wandb(self.cfg)
            logging.info(f"View run at {run.url}")

        if self.accelerator.is_main_process:
            self._log_experiment_details()

        # ---- Build loss, checkpoint config ----
        self.peft_config = None
        if self.cfg.get("peft", None) is not None:
            self.peft_config = self.cfg.peft.instantiate()
        self.loss_fn = build_loss_fn(self.cfg.loss_fn)

        checkpoint_config = build_checkpoint_config(
            self.cfg.get("checkpoint", None),
            self.cfg.get("model.cache_dir", None),
            _get_model_name(self.cfg.model),
            True if self.cfg.get("peft", None) else False,
        )

        if self.cfg.get("clip_grad_norm.max_norm", None) is not None:
            self.max_grad_norm = float(self.cfg.clip_grad_norm.max_norm)
        else:
            logging.info("No clip_grad_norm.max_norm specified, using default 1.0")
            self.max_grad_norm = 1.0

        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=0,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        )

        # ---- Build student model + optimizer ----
        self.model, model_state_dict_keys, self.optimizer = build_model_and_optimizer(
            device=self.accelerator.device,
            cfg_model=self.cfg.model,
            cfg_opt=self.cfg.optimizer,
            cfg_freeze=self.cfg.get("freeze_config", None),
            cfg_peft=self.peft_config,
            seed=self.cfg.get("seed", 42),
            checkpointer=self.checkpointer,
            cfg_fp8=self.cfg.get("fp8", None),
            cfg_compile=self.cfg.get("compile", None),
            loss_fn=self.loss_fn,
            load_base_model=self.cfg.get("checkpoint.load_base_model", True),
        )
        self.checkpointer.config.model_state_dict_keys = model_state_dict_keys

        # ---- Build teacher model (frozen, evaluation_mode=True) ----
        self._offload_teacher = self.cfg.get("offload_teacher_model", False)
        teacher_device = "cpu" if self._offload_teacher else self.accelerator.device
        self.teacher_model = _build_teacher_model(
            cfg_teacher=self.cfg.get("teacher_model", None),
            seed=self.cfg.get("seed", 42),
            device=teacher_device,
        )
        # Do NOT use accelerator.prepare_model — it wraps the model with autocast
        # which interferes with WAMDiff2's forward dispatch based on self.training.
        # Teacher stays in train() mode (like the original distil.py) so it uses
        # SDPA attention instead of flash_attention_2, which is incompatible with
        # the block causal mask from collate_fn.

        # ---- KD loss ----
        cfg_kd_loss = self.cfg.get("kd_loss_fn", None)
        if cfg_kd_loss is not None:
            self.kd_loss_fn = cfg_kd_loss.instantiate()
        else:
            from wam_diff.losses.kd_loss import DistilKDLoss
            self.kd_loss_fn = DistilKDLoss(temperature=1.0, fp32_upcast=True)

        self.kd_ratio: float = float(self.cfg.get("kd_ratio", 0.5))
        temperature = getattr(self.kd_loss_fn, "temperature", "N/A")
        logger.info(f"Distillation enabled: kd_ratio={self.kd_ratio}, temperature={temperature}")

        # ---- Build dataloaders (no DistributedSampler — Accelerate handles it) ----
        self.dataloader, self.processor = build_dataloader(
            self.cfg.dataset, self.cfg.dataloader,
            _get_model_name(self.cfg.model),
            self.cfg.get("processor", None),
            seed=self.cfg.get("seed", 42),
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        self.val_dataloader = None
        if "validation_dataset" in self.cfg:
            self.val_dataloader, _ = build_dataloader(
                self.cfg.validation_dataset, self.cfg.validation_dataloader,
                _get_model_name(self.cfg.model),
                self.cfg.get("processor", None),
                seed=self.cfg.get("seed", 42),
                local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
            )

        # ---- Prepare model, optimizer, dataloader with Accelerate ----
        # Prepare dataloader first so StepScheduler sees the sharded length
        # (after DistributedSampler is injected by Accelerate).
        self.model, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )

        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # ---- Step scheduler (AFTER prepare, so len(dataloader) reflects sharding) ----
        self.best_metric_key = self.cfg.get("checkpoint.best_metric_key", "default")
        self.step_scheduler = build_step_scheduler(
            self.cfg.get("step_scheduler", None),
            self.dataloader,
            self.accelerator.num_processes,
            local_batch_size=self.cfg.get("step_scheduler.local_batch_size", 1),
        )

        # ---- LR scheduler (AFTER step_scheduler, so total_steps is correct) ----
        self.lr_scheduler = build_lr_scheduler(
            self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler
        )

        # ---- Resume ----
        restore_from = self.cfg.get("checkpoint.restore_from", None)

        # JSONL loggers
        self.metric_logger_train = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "training.jsonl"
        )
        self.metric_logger_valid = build_metric_logger(
            pathlib.Path(self.checkpointer.config.checkpoint_dir) / "validation.jsonl"
        )

        self.load_checkpoint(restore_from)

        # ---- On-policy config ----
        self.collate_fn = self.dataloader.collate_fn
        self.on_policy = self.cfg.get("on_policy", False)
        self.generate_interval = self.cfg.get("generate_interval", 4)
        self.max_new_tokens = self.cfg.get("max_new_tokens", 128)

    # ----------------------------------------------------------------
    #  On-policy helpers
    # ----------------------------------------------------------------
    def _prepare_prompt_messages(self, messages):
        prompt = []
        for msg in messages:
            if msg.get("role") == "assistant":
                break
            prompt.append(msg)
        return prompt

    def _generate_student_response(self, raw_messages: list) -> str:
        """Generate a single student response (kept for backward compatibility)."""
        responses = self._generate_student_responses_batched([raw_messages])
        return responses[0]

    def _generate_student_responses_batched(self, raw_messages_list: list) -> list[str]:
        """Generate student responses for a batch of messages in one forward pass."""
        from qwen_vl_utils import process_vision_info

        device = self.accelerator.device
        prompt_messages_list = [self._prepare_prompt_messages(msg) for msg in raw_messages_list]

        # ---- Build batched inputs (same logic as inference_overfit_batch.py) ----
        texts = []
        images_batch = []
        videos_batch = []
        video_metadata_batch = []
        merged_video_kwargs = {}
        has_any_image = False
        has_any_video = False

        for prompt_messages in prompt_messages_list:
            text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
            )
            texts.append(text)

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                prompt_messages,
                return_video_kwargs=True, return_video_metadata=True,
                image_patch_size=self.processor.image_processor.patch_size,
            )

            video_metadata = None
            if video_inputs is not None:
                video_metadata = [item[1] for item in video_inputs]
                video_inputs = [item[0] for item in video_inputs]

            if image_inputs is not None and len(image_inputs) > 0:
                has_any_image = True
            if video_inputs is not None and len(video_inputs) > 0:
                has_any_video = True

            images_batch.append(image_inputs)
            videos_batch.append(video_inputs)
            video_metadata_batch.append(video_metadata)
            if video_kwargs:
                merged_video_kwargs.update(video_kwargs)

        processor_kwargs = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if has_any_image:
            processor_kwargs["images"] = images_batch
        if has_any_video:
            processor_kwargs["videos"] = videos_batch
            processor_kwargs["video_metadata"] = video_metadata_batch
            processor_kwargs.update(merged_video_kwargs)

        inputs = self.processor(**processor_kwargs).to(device)

        with torch.no_grad():
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.eval()
            original_attn = unwrapped.config._attn_implementation
            unwrapped.config._attn_implementation = "sdpa"
            try:
                response_ids = unwrapped.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    block_size=getattr(self.collate_fn, "block_size", 4),
                    denoising_steps=self.cfg.get("denoising_steps", 4),
                    temperature=self.cfg.get("generate_temperature", 0.0),
                    top_k=self.cfg.get("generate_top_k", 0),
                    top_p=self.cfg.get("generate_top_p", 1.0),
                    remasking_strategy=self.cfg.get("remasking_strategy", "low_confidence_static"),
                    confidence_threshold=self.cfg.get("confidence_threshold", 0.9),
                )
            finally:
                unwrapped.config._attn_implementation = original_attn
                unwrapped.train()
                self.model.train()

        responses = self.processor.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]

    def _rebuild_batch_with_student_response(self, raw_messages_list, student_responses):
        import copy
        new_messages_list = []

        prior_dist_val = "Mask"
        if raw_messages_list and raw_messages_list[0]:
            prior_dist_val = raw_messages_list[0][0].get("prior_dist", "Mask")

        for messages, resp in zip(raw_messages_list, student_responses):
            new_msg = copy.deepcopy(messages)
            replaced = False
            for i in range(len(new_msg) - 1, -1, -1):
                if new_msg[i].get("role") == "assistant":
                    new_msg[i]["content"] = [{"type": "text", "text": resp}]
                    replaced = True
                    break
            if not replaced:
                new_msg.append({"role": "assistant", "content": [{"type": "text", "text": resp}]})
            if len(new_msg) > 0:
                new_msg[0]["prior_dist"] = prior_dist_val
            new_messages_list.append(new_msg)

        return self.collate_fn(new_messages_list)

    # ----------------------------------------------------------------
    #  Training loop
    # ----------------------------------------------------------------
    def run_train_validation_loop(self):
        total_steps = self.step_scheduler.max_steps
        prior_dist_2 = getattr(self.dataloader.dataset, "prior_dist_2", None)
        switch_prior_thresh = self.cfg.dataset.get("switch_prior_thresh", 1.0)
        switch_step = max(0, int(total_steps * switch_prior_thresh))

        self.model.train()
        self.timestamp = time.perf_counter()

        # ---- Profiler ----
        prof = None
        if os.environ.get("ENABLE_PROFILER", "").lower() in ("1", "true"):
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=5, warmup=2, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.environ.get("PROFILER_LOG_DIR", "./prof_log")
                ),
                record_shapes=True, with_stack=True, profile_memory=True,
            )
            prof.start()
            if self.accelerator.is_main_process:
                print(f"[Profiler] Enabled, logs -> {os.environ.get('PROFILER_LOG_DIR', './prof_log')}")

        if self.accelerator.is_main_process:
            print("START TRAINING")

        for epoch in self.step_scheduler.epochs:
            self.step_scheduler.set_epoch(epoch)
            # Set epoch on sampler for proper shuffling
            if hasattr(self.dataloader, "sampler") and hasattr(self.dataloader.sampler, "set_epoch"):
                self.dataloader.sampler.set_epoch(epoch)

            for batch_idx, batches in enumerate(self.step_scheduler):
                cur_step = self.step_scheduler.step
                if prior_dist_2 is not None and cur_step > switch_step:
                    self.dataloader.dataset.prior_dist = prior_dist_2
                    prior_dist_2 = None

                log_data = self._run_train_optim_step(batches, self.max_grad_norm)
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
                        epoch, self.step_scheduler.step,
                        log_data.metrics["loss"], val_loss,
                        best_metric_key=self.best_metric_key,
                    )

        self.metric_logger_train.close()
        self.metric_logger_valid.close()

        if prof is not None:
            prof.stop()
            if self.accelerator.is_main_process:
                print("[Profiler] Stopped.")

        self.checkpointer.close()

    # ----------------------------------------------------------------
    #  Single optimizer step
    # ----------------------------------------------------------------
    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        # ===== ON-POLICY =====
        if getattr(self, "on_policy", False) and (self.step_scheduler.step % self.generate_interval == 0):
            _t0 = time.perf_counter()
            for bi in range(len(batches)):
                raw_messages_list = batches[bi].get("raw_messages")
                if raw_messages_list is None:
                    continue
                student_responses = self._generate_student_responses_batched(raw_messages_list)
                rebuilt_batch = self._rebuild_batch_with_student_response(raw_messages_list, student_responses)
                for k, v in rebuilt_batch.items():
                    if isinstance(v, torch.Tensor):
                        batches[bi][k] = v
                    elif k not in batches[bi]:
                        batches[bi][k] = v
            logger.info(f"[On-policy] generate + re-collate took {time.perf_counter() - _t0:.2f}s")

        # ---- Pre-compute aggregated stats ----
        if 'loss_mask' in batches[0]:
            num_label_tokens = torch.tensor(
                sum(b["loss_mask"].sum().item() for b in batches), dtype=torch.long
            )
        else:
            num_label_tokens = torch.tensor(
                sum((b["labels"] != -100).sum().item() for b in batches), dtype=torch.long
            )

        global_batch_size = self._accel_allreduce(torch.tensor(len(batches), device=self.accelerator.device)).item()

        num_samples = torch.tensor(
            sum(b["num_samples"].sum().item() for b in batches), dtype=torch.long
        )
        num_total_samples = self._accel_allreduce(num_samples.to(self.accelerator.device)).item()
        num_processes = self.accelerator.num_processes

        num_label_tokens = self._accel_allreduce(num_label_tokens.to(self.accelerator.device)).item()

        num_tokens_in_batch = torch.tensor(
            sum(b["labels"].numel() - count_tail_padding(b["labels"]) for b in batches),
            dtype=torch.long,
        )
        num_tokens_in_batch = self._accel_allreduce(num_tokens_in_batch.to(self.accelerator.device)).item()

        loss_buffer = []
        ce_loss_buffer = []
        kd_loss_buffer = []

        _t_data = _t_teacher_fwd = _t_student_fwd = _t_loss = _t_bwd = _t_comm = 0.0

        for i, batch in enumerate(batches):
            # ---- Move to device ----
            _t0 = time.perf_counter()
            batch = self._move_batch_to_device(batch)
            labels = batch.pop("labels")
            batch.pop("raw_messages", None)
            _t_data += time.perf_counter() - _t0

            # ---- Teacher forward (no_grad) ----
            _t0 = time.perf_counter()
            with torch.no_grad():
                if self._offload_teacher:
                    self.teacher_model.to(self.accelerator.device)
                teacher_batch = dict(batch)
                if "teacher_attention_mask" in batch:
                    teacher_batch["attention_mask"] = batch["teacher_attention_mask"]
                    teacher_batch["block_size"] = batch["teacher_block_size"]
                # Call teacher_model directly — not wrapped by Accelerate.
                # Keep teacher in train() mode so WAMDiff2 uses SDPA (not flash_attn).
                teacher_out = self.teacher_model(**teacher_batch)
                teacher_logits = getattr(teacher_out, "logits", teacher_out).detach()
                if self._offload_teacher:
                    self.teacher_model.to("cpu")
                    torch.cuda.empty_cache()
            _t_teacher_fwd += time.perf_counter() - _t0

            # ---- Student forward + backward ----
            # No accelerator.accumulate() — StepScheduler manages gradient
            # accumulation by grouping micro-batches into the `batches` list.
            # With gradient_accumulation_steps=1, gradients are synced every step.
            _t0 = time.perf_counter()
            if isinstance(self.loss_fn, FusedLinearCrossEntropy):
                out = self.model(logits_to_keep=1, **batch)
                if "hidden_states" not in out:
                    raise ValueError(
                        "FusedLinearCrossEntropy requires hidden states. Set `model.output_hidden_states=True`."
                    )
            else:
                out = self.model(**batch)
            student_logits = getattr(out, "logits", out)
            _t_student_fwd += time.perf_counter() - _t0

            # ---- DEBUG ----
            if os.environ.get("DEBUG", "").lower() in ("1", "true") and i == len(batches) - 1:
                self._debug_model_output(out, batch, labels)

            # ---- Loss ----
            _t0 = time.perf_counter()
            ce_loss = calculate_loss(
                self.loss_fn,
                logits=student_logits, labels=labels, model=self.model,
                hidden_states=out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else None,
                num_label_tokens=num_label_tokens,
                x_t=batch.get("input_ids", None),
                t=batch.get("t", None),
                loss_mask=batch.get("loss_mask", None),
                response_mask=batch.get("response_mask", None),
                block_size=batch.get("block_size") if "block_size" in batch else 0,
                num_samples=num_total_samples,
            )

            kd_loss = self.kd_loss_fn(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                loss_mask=batch.get("loss_mask", None),
                response_mask=batch.get("response_mask", None),
                num_samples=num_total_samples,
            )

            local_loss = (1.0 - self.kd_ratio) * ce_loss + self.kd_ratio * kd_loss
            _t_loss += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            loss_buffer.append(local_loss.clone().detach())
            ce_loss_buffer.append(ce_loss.detach().clone())
            kd_loss_buffer.append(kd_loss.detach().clone())
            # accelerator.backward replaces loss.backward()

            loss_for_backward = local_loss
            if self.accelerator.num_processes > 1:
                loss_for_backward = loss_for_backward * self.accelerator.num_processes

            self.accelerator.backward(loss_for_backward)

            # self.accelerator.backward(local_loss)
            _t_bwd += time.perf_counter() - _t0

        # ---- Gradient clipping (only when sync_gradients) ----
        # When using DeepSpeed, gradient clipping is handled by DeepSpeed engine
        # configured via the accelerate config (gradient_clipping field).
        # accelerator.clip_grad_norm_ is a no-op in DeepSpeed mode.
        _t0 = time.perf_counter()
        if self.accelerator.sync_gradients:
            grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
        else:
            grad_norm = 0.0
        _t_comm += time.perf_counter() - _t0

        # ---- Optimizer step + zero_grad ----
        # DeepSpeed: optimizer.step() calls the DeepSpeed engine which handles
        # gradient clipping internally. zero_grad is also handled by DeepSpeed.
        # DDP: standard PyTorch optimizer step + zero_grad.
        self.checkpointer.maybe_wait_for_staging()
        _t0 = time.perf_counter()
        self.optimizer.step()
        if not self._is_deepspeed():
            self.optimizer.zero_grad(set_to_none=True)
        _t_opt = time.perf_counter() - _t0

        # LR scheduler step — always call, including for DeepSpeed.
        # OptimizerParamScheduler directly writes param_groups["lr"], which
        # DeepSpeed respects on the next forward/backward pass.
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(1)

        # ---- TPS ----
        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta

        _t_total = _t_data + _t_teacher_fwd + _t_student_fwd + _t_loss + _t_bwd + _t_comm + _t_opt
        if _t_total > 0 and os.environ.get("DEBUG", "").lower() in ("1", "true") and self.accelerator.is_main_process:
            print(f"[Timing] data:{_t_data:.3f}s({100*_t_data/_t_total:.0f}%) "
                  f"teacher_fwd:{_t_teacher_fwd:.3f}s({100*_t_teacher_fwd/_t_total:.0f}%) "
                  f"student_fwd:{_t_student_fwd:.3f}s({100*_t_student_fwd/_t_total:.0f}%) "
                  f"loss:{_t_loss:.3f}s({100*_t_loss/_t_total:.0f}%) "
                  f"bwd:{_t_bwd:.3f}s({100*_t_bwd/_t_total:.0f}%) "
                  f"grad_sync:{_t_comm:.3f}s({100*_t_comm/_t_total:.0f}%) "
                  f"opt:{_t_opt:.3f}s({100*_t_opt/_t_total:.0f}%) "
                  f"total:{_t_total:.3f}s")

        reporting_loss = self._accel_allreduce(torch.sum(torch.stack(loss_buffer))).item()
        ce_loss_val = self._accel_allreduce(torch.stack(ce_loss_buffer).sum()).item()
        kd_loss_val = self._accel_allreduce(torch.stack(kd_loss_buffer).sum()).item()
        ce_loss_buffer.clear()
        kd_loss_buffer.clear()

        # Safe lr extraction with fallback
        def _safe_lr(idx):
            return self.optimizer.param_groups[idx]["lr"] if idx < len(self.optimizer.param_groups) else self.optimizer.param_groups[0]["lr"]

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": reporting_loss,
                "ce_loss": ce_loss_val,
                "kd_loss": kd_loss_val,
                "grad_norm": grad_norm,
                "visual_lr": _safe_lr(0),
                "merger_lr": _safe_lr(4),
                "language_lr": _safe_lr(2),
                "single_samples": num_samples,
                "total_samples": num_total_samples,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
                "tps": tps,
                "tps_per_gpu": tps / max(self.accelerator.num_processes, 1),
                "num_tokens_per_step": num_tokens_in_batch,
                "num_label_tokens": num_label_tokens,
                "kd_ratio": self.kd_ratio,
            },
        )

    # ----------------------------------------------------------------
    #  Validation
    # ----------------------------------------------------------------
    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        with ScopedRNG(seed=1, ranked=True):
            self.model.eval()

            total_loss = 0.0
            total_tokens = 0
            total_num_label_tokens = 0
            for batch in val_dataloader:
                batch = self._move_batch_to_device(batch)
                labels = batch.pop("labels")
                num_label_tokens = (labels != -100).sum().item()

                out = self.model(**batch)
                local_loss = calculate_loss(
                    self.loss_fn,
                    logits=getattr(out, "logits", out),
                    labels=labels, model=self.model,
                    hidden_states=out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else None,
                    num_label_tokens=num_label_tokens,
                )
                total_num_label_tokens += num_label_tokens
                total_loss += local_loss.item() * num_label_tokens
                total_tokens += num_label_tokens

        total_loss = self._accel_allreduce(torch.tensor([total_loss], device=self.accelerator.device)).item()
        total_tokens = self._accel_allreduce(torch.tensor([total_tokens], dtype=torch.long, device=self.accelerator.device)).item()
        total_num_label_tokens = self._accel_allreduce(
            torch.tensor([total_num_label_tokens], dtype=torch.long, device=self.accelerator.device)
        ).item()

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

    # ----------------------------------------------------------------
    #  Logging
    # ----------------------------------------------------------------
    def log_val_metrics(self, log_data):
        if not self.accelerator.is_main_process or log_data is None:
            return
        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=log_data.step)
        self.metric_logger_valid.log(log_data)
        logging.info(
            "[val] step {} | epoch {} | loss {:.4f} | lr {:.2e} | num_label_tokens {}".format(
                log_data.step, log_data.epoch,
                log_data.metrics["val_loss"], log_data.metrics["lr"],
                log_data.metrics["num_label_tokens"],
            )
        )

    def log_train_metrics(self, log_data):
        if not self.accelerator.is_main_process:
            return
        if wandb.run is not None:
            wandb.log(log_data.to_dict(), step=self.step_scheduler.step)
        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | ce_loss {:.4f} | kd_loss {:.4f} | "
            "grad_norm {:.4f} | visual_lr {:.2e} | merger_lr {:.2e} | language_lr {:.2e} | "
            "single_samples {:d} | total_samples {:d} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | "
            "num_label_tokens {} | kd_ratio {:.2f}".format(
                log_data.step, log_data.epoch,
                log_data.metrics["loss"], log_data.metrics["ce_loss"], log_data.metrics["kd_loss"],
                log_data.metrics["grad_norm"],
                log_data.metrics["visual_lr"], log_data.metrics["merger_lr"], log_data.metrics["language_lr"],
                log_data.metrics["single_samples"], log_data.metrics["total_samples"],
                log_data.metrics["mem"], log_data.metrics["tps"], log_data.metrics["tps_per_gpu"],
                log_data.metrics["num_label_tokens"], log_data.metrics["kd_ratio"],
            )
        )
        torch.cuda.reset_peak_memory_stats()

    # ----------------------------------------------------------------
    #  Checkpoint (Accelerate-based)
    # ----------------------------------------------------------------
    def save_checkpoint(self, epoch, step, train_loss, val_loss=None, best_metric_key="default"):
        if not self.checkpointer.config.enabled:
            return

        self.checkpointer.async_wait()

        path = self.checkpointer.config.checkpoint_dir
        path = os.path.join(path, f"epoch_{epoch}_step_{step}")

        best_val_metric = (
            val_loss[next(iter(val_loss.keys())) if len(val_loss) == 1 else best_metric_key]
            if val_loss else None
        )

        # Save metadata on main process
        if self.accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
            print(f"Saving checkpoint to {path}", flush=True)

            loss_dict = {"train_loss": train_loss, "epoch": epoch, "step": step}
            if val_loss:
                key = next(iter(val_loss.keys()))
                loss_dict["val_loss"] = val_loss.pop(key) if len(val_loss) == 1 else val_loss
            with open(os.path.join(path, "losses.json"), "w") as f:
                try:
                    json.dump(loss_dict, f)
                except Exception:
                    pass

            # Save model weights + tokenizer on main process only
            unwrapped = self.accelerator.unwrap_model(self.model)
            if hasattr(unwrapped, "save_pretrained"):
                unwrapped.save_pretrained(os.path.join(path, "model"), safe_serialization=True)
            if self.processor is not None and hasattr(self.processor, "save_pretrained"):
                self.processor.save_pretrained(os.path.join(path, "model"))

        if dist.is_initialized():
            dist.barrier()

        # Use Accelerate save_state for model/optimizer/scheduler across all ranks
        self.accelerator.save_state(path)

        # Extra training state on main process
        if self.accelerator.is_main_process:
            extra_state = {
                "step_scheduler_step": self.step_scheduler.step,
                "step_scheduler_epoch": self.step_scheduler.epoch,
            }
            torch.save(extra_state, os.path.join(path, "training_state.pt"))

            # Update symlinks
            self._update_latest_symlink(path)
            if best_val_metric is not None:
                self._update_best_symlink(path, float(best_val_metric))

        if dist.is_initialized():
            dist.barrier()

        # OBS upload
        if dist.is_initialized():
            current_rank = dist.get_rank()
            if current_rank % 8 == 0:
                try:
                    import moxing as mox
                except ImportError:
                    mox = None
                if mox is not None:
                    output_url = os.getenv("OBS_SAVE_PATH", "")
                    if output_url:
                        basename = os.path.basename(path.rstrip('/'))
                        dst_path = f"{output_url}/{basename}/rank_{current_rank}/"
                        print(f"Start syncing: {path} -> {dst_path} (rank {current_rank})")
                        try:
                            mox.file.copy_parallel(path, dst_path)
                            print(f"Sync succeeded: {path} -> {dst_path}")
                        except Exception as e:
                            print(f"Sync failed: {e}")

    def load_checkpoint(self, restore_from=None):
        if not self.checkpointer.config.enabled:
            if (not dist.is_initialized() or dist.get_rank() == 0) and restore_from is not None:
                print("Enable checkpointing to resume, skipping...", flush=True)
            return

        if restore_from:
            ckpt_dir = restore_from
        else:
            ckpt_dir = self._find_latest_checkpoint(self.checkpointer.config.checkpoint_dir)
            if ckpt_dir is None:
                return

        if self.accelerator.is_main_process:
            print(f"Loading checkpoint from {ckpt_dir}", flush=True)

        # Use Accelerate load_state for model/optimizer/scheduler
        self.accelerator.load_state(ckpt_dir)

        # Load extra training state
        training_state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            extra_state = torch.load(training_state_path, weights_only=False)
            if "step_scheduler_step" in extra_state:
                logger.info(f"Resuming from step {extra_state['step_scheduler_step']}")

    # ----------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------
    def _move_batch_to_device(self, batch):
        device = self.accelerator.device
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(v) for v in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        else:
            return batch

    def _accel_allreduce(self, tensor, op="sum"):
        """All-reduce across all Accelerate processes."""
        if not dist.is_initialized() or self.accelerator.num_processes == 1:
            return tensor.cpu() if tensor.is_cuda else tensor
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        if op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.cpu()

    def _is_deepspeed(self):
        """Check if the current Accelerate strategy is DeepSpeed."""
        return hasattr(self.accelerator, "deepspeed_engine_wrapped") or \
               str(self.accelerator.distributed_type).endswith("DEEPSPEED")

    def _debug_model_output(self, out, batch, labels):
        if not self.accelerator.is_main_process:
            return
        with torch.no_grad():
            logits = getattr(out, "logits", out)
            pred_ids = logits.argmax(dim=-1)
            resp_mask = batch.get("response_mask", None)
            if resp_mask is not None:
                r_mask = resp_mask[0].bool()
                noisy_text = self.processor.tokenizer.decode(batch["input_ids"][0][r_mask], skip_special_tokens=False)
                pred_text = self.processor.tokenizer.decode(pred_ids[0][r_mask], skip_special_tokens=True)
                label_text = self.processor.tokenizer.decode(labels[0][r_mask], skip_special_tokens=True)
            else:
                pred_text = self.processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
                label_text = self.processor.tokenizer.decode(labels[0], skip_special_tokens=True)
                noisy_text = self.processor.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
            print(f"\n[DEBUG] === Model Output (step {self.step_scheduler.step}) ===")
            print(f"[DEBUG] Input (noisy):\n{noisy_text[-200:]}")
            print(f"[DEBUG] Prediction:\n{pred_text[-100:]}")
            print(f"[DEBUG] Label (ground truth):\n{label_text[-100:]}")
            print(f"[DEBUG] {'='*60}")

    def _log_experiment_details(self):
        import getpass, socket
        from datetime import datetime
        logging.info(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
        logging.info(f"User: {getpass.getuser()}, Host: {socket.gethostname()}")
        logging.info(f"World size: {self.accelerator.num_processes}")
        logging.info(f"Recipe: {self.__class__.__name__}")
        logging.info(f"Model: {_get_model_name(self.cfg.model)}")

    @staticmethod
    def _find_latest_checkpoint(checkpoint_dir):
        import re
        from pathlib import Path
        root = Path(checkpoint_dir)
        if not root.exists():
            return None
        latest_link = os.path.join(os.fspath(root), "LATEST")
        if os.path.islink(latest_link):
            try:
                resolved = os.readlink(latest_link)
                if not os.path.isabs(resolved):
                    resolved = os.path.abspath(os.path.join(os.fspath(root), resolved))
                if os.path.isdir(resolved):
                    return resolved
            except OSError:
                pass
        checkpoint_files = list(root.glob("*step_*"))
        if not checkpoint_files:
            return None
        def _step_num(p):
            m = re.search(r"step_(\d+)$", p.stem)
            return int(m.group(1)) if m else -1
        latest = max(checkpoint_files, key=_step_num)
        return latest if _step_num(latest) != -1 else None

    def _update_latest_symlink(self, target_dir):
        ckpt_root = self.checkpointer.config.checkpoint_dir
        link_path = os.path.join(ckpt_root, "LATEST")
        if os.path.lexists(link_path):
            os.remove(link_path)
        relative = os.path.relpath(os.path.abspath(target_dir), start=os.path.abspath(ckpt_root))
        os.symlink(relative, link_path)

    def _update_best_symlink(self, target_dir, val_loss):
        if not hasattr(self, '_best_val_loss'):
            self._best_val_loss = float("inf")
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            ckpt_root = self.checkpointer.config.checkpoint_dir
            link_path = os.path.join(ckpt_root, "LOWEST_VAL")
            if os.path.lexists(link_path):
                os.remove(link_path)
            relative = os.path.relpath(os.path.abspath(target_dir), start=os.path.abspath(ckpt_root))
            os.symlink(relative, link_path)
            logging.info(f"Updated LOWEST_VAL -> {os.path.basename(target_dir)} (val_loss={val_loss:.4f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config_path=None):
    """Main entry point for the Accelerate-based distillation recipe.

    Launch with:
        accelerate launch -m wam_diff.training.distil_accelerate \\
            --config configs/distillation/<experiment>.yaml

    Or with DeepSpeed Zero-2:
        accelerate launch --config_file configs/distillation/accelerate/deepspeed_zero2.yaml \\
            -m wam_diff.training.distil_accelerate \\
            --config configs/distillation/<experiment>.yaml
    """
    cfg = parse_args_and_load_config(config_path)
    trainer = DistilRecipeForVLM(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
