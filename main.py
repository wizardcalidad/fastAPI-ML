import joblib
import sklearn
import numpy as np

# utilities
from utils import clean_text

from pydantic.main import BaseModel

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Response

app = FastAPI()

models = {
    "bernoulli": {
        "count": joblib.load("models/bernoulli_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/bernoulli_naive_bayes_with_tfidf_vectorizer.joblib"),
    },

    "categorical": {
        "count": joblib.load("models/categorical_naive_bayes_with_count_vectorizer.joblib"),
    },
    "complement": {
        "count": joblib.load("models/complement_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/complement_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
    "gaussian": {
        "count": joblib.load("models/gaussian_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/gaussian_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
    "multinomial": {
        "count": joblib.load("models/multinomial_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/multinomial_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
}


class PredictRequest(BaseModel):
    model: str
    vectorizer: str
    text: str


class PredictResponse(BaseModel):
    output: str


class PredictAllRequest(BaseModel):
    text: str


@app.get("/ping")
def ping():
    return Response(content="pong", media_type="text/plain")


@app.post('/predict', response_model=PredictResponse)
def predict(parameters: PredictRequest):
    model = parameters.model
    vectorizer = parameters.vectorizer
    text = parameters.text

    x = [text]
    y = models[model][vectorizer].predict(x)

    response = {"output": "positive" if y else "negative"}
    return response


@app.post('/predict_all')
def predict_all(parameters: PredictAllRequest):
    text = parameters.text

    response = {}

    x = [text]  # the input
    for model in models:
        response[model] = {}

        for vectorizer in models[model]:
            y = models[model][vectorizer].predict(x)  # prediction
            response[model][vectorizer] = "positive" if y else "negative"

    json_compatible_item_data = jsonable_encoder(response)
    return JSONResponse(content=json_compatible_item_data)