from typing import List

from .config import MODEL_REGISTRY, ModelConfig
from .schemas import ModelOption


def get_model_config(model_id: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model_id: {model_id}") from exc


def list_model_options() -> List[ModelOption]:
    return [
        ModelOption(
            model_id=model.model_id,
            label=model.label,
            provider_mode=model.provider_mode,
            base_url=model.base_url,
            model_name=model.model_name,
        )
        for model in MODEL_REGISTRY.values()
    ]
