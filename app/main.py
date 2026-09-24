from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import CustomerData, PredictionResponse


app = FastAPI(
    title="AI Customer Churn Prediction API",
    description="REST API for predicting telecom customer churn.",
    version="1.0.0"
)


# Locate and load the trained ML pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "model" / "churn_model.pkl"

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "AI Customer Churn Prediction API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):

    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    prediction = model.predict(customer_df)[0]

    churn_probability = model.predict_proba(
        customer_df
    )[0][1]

    return {
        "prediction": "churn" if prediction == 1 else "stay",
        "churn_probability": round(float(churn_probability), 4)
    }