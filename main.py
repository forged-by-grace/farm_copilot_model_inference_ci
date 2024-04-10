from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import ORJSONResponse
from src.components.prepare_model import PrepareModel
from dotenv import load_dotenv
from src.config.configuration import ConfigurationManager
import uvicorn
from src.utils import logger
from src.controllers.controller import prediction_controller, prompt_controller
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from openai import OpenAI


# Load .env variables
load_dotenv()
logger.info('.env files loaded successfully')


# Load start-up configurations
config = ConfigurationManager()
start_up_config = config.get_start_up_config()
logger.info('Start up config files loaded successfully')


# init model
model = None


# Init llm
llm = None

async def on_startup():
    global model
    global llm

    logger.info('Starting server...')

    # Init LLM
    llm = OpenAI()

    # Load prediction model
    prepare_model_config = config.get_prepare_model_config()
    prepare_model = PrepareModel(config=prepare_model_config)
    prepare_model.load_model()
    model = prepare_model.get_model()
    

async def on_shut_down():
    logger.info('Shutting down inference server')


# init app lifecyle
@asynccontextmanager
async def lifespan(app: FastAPI):
    await on_startup()
    yield
    await on_shut_down()


# Init fastapi
app = FastAPI(
    lifespan=lifespan,
    title=start_up_config.params_app_title,
    version=start_up_config.params_app_version,
    description=start_up_config.params_app_description,
    terms_of_service=start_up_config.params_app_tos,
    contact={
        'name': start_up_config.params_app_company_name,
        'url': start_up_config.params_app_company_url,
        'email': start_up_config.params_app_company_email
    }
)


# Set cors
origins = ['*']


# add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=['*'],
    allow_credentials=True,
    allow_headers=[],
)


@app.post('/predict')
async def predict(file: UploadFile):    
    return await prediction_controller(file=await file.read(), model=model)


@app.post('/query')
async def query(question: str):
    return await prompt_controller(question=question, llm=llm)


@app.get('/history')
async def history():
    pass


if __name__ == '__main__':
    uvicorn.run(
        start_up_config.params_app_entry_point,
        port=start_up_config.params_app_port,
        reload=start_up_config.params_app_reload,
        host=start_up_config.params_app_host,
        lifespan=start_up_config.params_app_lifespan
    )