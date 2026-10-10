
from pathlib import Path

import joblib
import pandas as pd
import pytest


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "ml"
    / "model"
    / "churn_model.pkl"
)

model = joblib.load(MODEL_PATH)


@pytest.fixture
def sample_customer():
    return pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 286.50
    }])


def test_model_prediction(sample_customer):
    prediction = model.predict(sample_customer)[0]

    assert prediction in [0, 1]


def test_model_probability(sample_customer):
    probability = model.predict_proba(sample_customer)[0][1]

    assert 0 <= probability <= 1


def test_model_consistency(sample_customer):
    first_prediction = model.predict(sample_customer)[0]
    second_prediction = model.predict(sample_customer)[0]

    assert first_prediction == second_prediction
