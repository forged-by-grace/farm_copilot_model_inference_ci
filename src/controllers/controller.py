from typing import Dict
from src.components.model_prediction import ModelPrediction
from src.components.prompt import Prompt
from src.config.configuration import ConfigurationManager
from openai import OpenAI


async def prediction_controller(file, model) -> Dict[str, str | int]:
    # Init configuration 
    config = ConfigurationManager()
    model_prediction_config = config.get_model_prediction_config()

    # Init Model Prediction
    model_prediction = ModelPrediction(config=model_prediction_config, model=model, image_file=file)
    model_prediction.read_image_data()
    model_prediction.preprocess_image()
    prediction = model_prediction.predict()
    return prediction


async def prompt_controller(question: str, llm: OpenAI) -> Dict[str, str]:
    # Init configuration
    config = ConfigurationManager()
    prompt_config = config.get_prompt_config()

    # Init Prompt
    prompt = Prompt(question=question, llm=llm, config=prompt_config)
    await prompt.request_response()
    return prompt.get_response()