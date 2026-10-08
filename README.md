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

The machine learning pipeline, FastAPI prediction service, PostgreSQL integration, and Docker containerization are implemented.

Predictions are automatically stored in PostgreSQL, with persistent storage configured through Docker volumes.

Automated testing, CI/CD, and Kubernetes deployment are planned for the next development stages.

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

## Run with Docker

Create a `.env` file in the project root with the following variables:

```env
DB_NAME=churn_db
DB_USER=postgres
DB_PASSWORD=your_password
```

Build and start the application:

```bash
docker compose up -d --build
```

Access the API documentation at:

http://localhost:8000/docs

Stop the application:

```bash
docker compose down
```

PostgreSQL data is preserved in a Docker volume between container restarts.
