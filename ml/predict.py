import joblib
import pandas as pd


# Load trained model
model = joblib.load("ml/model/churn_model.pkl")


# Create a new telecom customer
customer = pd.DataFrame([{
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


# Make prediction
prediction = model.predict(customer)[0]

# Get probability of churn
probability = model.predict_proba(customer)[0][1]


# Display result
print("\n--- CHURN PREDICTION ---")

if prediction == 1:
    print("Prediction: Customer is likely to CHURN")
else:
    print("Prediction: Customer is likely to STAY")

print(f"Churn probability: {probability:.2%}")