from mlflow.keras import load_model
from src.entity.config_entity import PrepareModelConfig
from tensorflow import keras


class PrepareModel:
    def __init__(self, config: PrepareModelConfig) -> None:
        self.config = config

    
    def load_model(self) -> None:
        # Deployed model uri
        model_uri = f"models:/{self.config.params_model_name}/{self.config.params_model_version}"

        self.model = load_model(model_uri=model_uri)

    
    def get_model(self) -> keras.Model:
        return self.model