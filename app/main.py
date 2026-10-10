from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
import logging
from fastapi import HTTPException

from app.schemas import CustomerData, PredictionResponse
from app.database import get_db_connection

logger = logging.getLogger(__name__)

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
    customer_df = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(customer_df)[0]
    churn_probability = model.predict_proba(customer_df)[0][1]

    prediction_label = "churn" if prediction == 1 else "stay"
    probability = round(float(churn_probability), 4)

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (prediction, churn_probability)
            VALUES (%s, %s)
            """,
            (prediction_label, probability)
        )

        connection.commit()

    except Exception:
        if connection is not None:
            connection.rollback()

        logger.exception("Failed to save churn prediction")

        raise HTTPException(
            status_code=503,
            detail="Prediction storage is temporarily unavailable"
        )

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()
            
    return {
        "prediction": prediction_label,
        "churn_probability": probability
    }