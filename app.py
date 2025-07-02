import sys, os
import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGO_DB_URL")
print(mongo_db_url)

import pymongo
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
from networkSecurity.pipeline.training_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networkSecurity.utils.main_utils.utils import load_object

client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)
from networkSecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = client[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(CORSMiddleware, allow_origins = origins, allow_credentials=True,allow_methods = ["*"], allow_headers = ["*"])

@app.get("/",tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training Successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)