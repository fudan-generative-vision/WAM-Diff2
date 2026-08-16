# SPDX-License-Identifier: Apache-2.0

"""WAM Diff2 Transformers registration and public model exports."""

from transformers import AutoConfig, AutoModelForImageTextToText

from wam_diff.models.wam_diff2.configuration_wam_diff2 import WAMDiff2Config
from wam_diff.models.wam_diff2.modeling_wam_diff2 import WAMDiff2ForConditionalGeneration

AutoConfig.register(WAMDiff2Config.model_type, WAMDiff2Config, exist_ok=True)
AutoModelForImageTextToText.register(
    WAMDiff2Config,
    WAMDiff2ForConditionalGeneration,
    exist_ok=True,
)

__all__ = ["WAMDiff2Config", "WAMDiff2ForConditionalGeneration"]
