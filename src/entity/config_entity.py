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
    params_app_description: str
    params_app_tos: str
    params_app_company_name: str
    params_app_company_url: str
    params_app_lifespan: str
    params_app_entry_point: str
    params_app_company_email: str


@dataclass(frozen=True)
class PromptConfig:
    params_app_prompt_model: str
    params_app_prompt_role: str
    params_app_prompt_llm_temperature: int

