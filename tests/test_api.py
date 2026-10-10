
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import MagicMock, patch

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Customer Churn Prediction API is running"
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_validation():
    response = client.post("/predict", json={})

    assert response.status_code == 422



def test_predict_success():
    customer = {
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
    }

    mock_connection = MagicMock()

    with patch(
        "app.main.get_db_connection",
        return_value=mock_connection
    ):
        response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["churn", "stay"]
    assert 0 <= result["churn_probability"] <= 1

    mock_connection.cursor.return_value.execute.assert_called_once()
    mock_connection.commit.assert_called_once()
    mock_connection.close.assert_called_once()

def test_predict_database_failure():
    customer = {
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
    }

    with patch(
        "app.main.get_db_connection",
        side_effect=Exception("Database unavailable")
    ):
        response = client.post("/predict", json=customer)

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction storage is temporarily unavailable"
    )


