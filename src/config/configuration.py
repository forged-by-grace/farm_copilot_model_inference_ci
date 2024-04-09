import os
from src.constants import PARAMS_FILE_PATH
from src.utils.common import read_yaml
from src.entity.config_entity import PrepareModelConfig, ModelPredictionConfig, StartUpConfig


class ConfigurationManager:
    def __init__(self, params_filepath = PARAMS_FILE_PATH):
        self.params = read_yaml(params_filepath)
    

    def get_prepare_model_config(self) -> PrepareModelConfig:
        prepare_model_config = PrepareModelConfig(
            params_model_name=self.params.MODEL_NAME,
            params_model_version=self.params.MODEL_VERSION
        )
        
        return prepare_model_config


    def get_model_prediction_config(self) -> ModelPredictionConfig:
        model_prediction_config = ModelPredictionConfig(
            params_image_size=tuple(self.params.IMAGE_SIZE),
            params_model_classes=self.params.MODEL_CLASSES
        )

        return model_prediction_config


    def get_start_up_config(self) -> StartUpConfig:
        start_up_config = StartUpConfig(
            params_app_debug=self.params.APP_DEBUG,
            params_app_title=self.params.APP_TITLE,
            params_app_version=self.params.APP_VERSION,
            params_app_host=self.params.APP_HOST,
            params_app_port=self.params.APP_PORT,
            params_app_reload=self.params.APP_RELOAD,
        )

        return start_up_config