from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS


def register_with_transformers_autodoc(model_type: str, config_cls) -> None:
    """Register custom configs early enough for `@auto_docstring` resolution."""
    config_name = config_cls.__name__
    model_types = {model_type, model_type.replace("_", "-")}

    AutoConfig.register(model_type, config_cls, exist_ok=True)
    for key in model_types:
        CONFIG_MAPPING_NAMES.setdefault(key, config_name)
        HARDCODED_CONFIG_FOR_MODELS.setdefault(key, config_name)
