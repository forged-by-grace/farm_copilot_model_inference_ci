from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrepareModelConfig:
    params_model_name: str
    params_model_version: int


@dataclass(frozen=True)
class ModelPredictionConfig:
    params_image_size: list
    params_model_classes: list


@dataclass(frozen=True)
class StartUpConfig:
    params_app_title: str
    params_app_version: str
    params_app_debug: bool
    params_app_port: int
    params_app_host: str
    params_app_reload: bool

