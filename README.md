# AI Customer Churn Prediction Platform

Machine learning application for predicting customer churn based on telecommunications customer data.

The project uses customer information such as contract type, tenure, subscribed services, payment method, and monthly charges to estimate whether a customer is likely to leave the service.

## How It Works

Customer data is processed through a trained machine learning pipeline and exposed through a FastAPI REST API.

The API accepts customer information and returns a churn prediction together with the estimated probability.

Example response:

```json
{
  "prediction": "churn",
  "churn_probability": 0.7551
}
```

## Technologies

Python · Scikit-learn · Pandas · FastAPI · PostgreSQL · Docker · Kubernetes · GitHub Actions

## Current Status

The machine learning pipeline and FastAPI prediction service are currently implemented.

PostgreSQL integration, automated testing, Docker containerization, CI/CD, and Kubernetes deployment will be added in the next development stages.

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```