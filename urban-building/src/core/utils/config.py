# src/core/utils/config.py

from omegaconf import OmegaConf


def as_plain_dict(value) -> dict:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        container = OmegaConf.to_container(value, resolve=True)
        return container if isinstance(container, dict) else {}
    if isinstance(value, dict):
        return value
    return {}