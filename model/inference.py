from fastapi import FastAPI
from transformers import pipeline

import logging 

logger = logging.getLogger()

app = FastAPI()

model = pipeline("text-generation", model="distilgpt2")


@app.get("/")
def index():
    return {"Hello": "World"}

@app.get("predict")
def predict(inputs):
    logger.info(inputs)
    return {"prediction": model(inputs, max_length=20, num_return_sequences=1)}
