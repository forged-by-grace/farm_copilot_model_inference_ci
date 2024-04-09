from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse
from src.components.prepare_model import PrepareModel
from dotenv import load_dotenv
from src.config.configuration import ConfigurationManager
import uvicorn
from src.utils import logger
from src.controllers.controller import prediction_controller


# Load .env variables
load_dotenv()
logger.info('.env files loaded successfully')


# Load start-up configurations
config = ConfigurationManager()
start_up_config = config.get_start_up_config()
logger.info('Start up config files loaded successfully')


# Load prediction model
prepare_model_config = config.get_prepare_model_config()
prepare_model = PrepareModel(config=prepare_model_config)
model = prepare_model.get_model()


# Init FastAPI
app = FastAPI(
    title=start_up_config.params_app_title,
    version=start_up_config.params_app_version,
    debug=start_up_config.params_app_debug
    )


@app.post('/predict')
async def predict(file: UploadFile):
    return await prediction_controller(file=file, model=model)


if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        port=start_up_config.params_app_port,
        reload=start_up_config.params_app_reload,
        host=start_up_config.params_app_host
    )