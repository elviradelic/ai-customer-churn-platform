import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# 1. Load dataset
df = pd.read_csv("data/telco_churn.csv")


# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])


# 3. Remove customer identifier
df = df.drop(columns=["customerID"])


# 4. Convert target column to 0 and 1
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# 5. Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]


# 6. Detect categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns


print("Categorical columns:")
print(categorical_columns.tolist())

print("\nNumerical columns:")
print(numerical_columns.tolist())


# 7. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)


# 8. Create ML pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# 9. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 10. Train model
print("\nTraining model...")

model.fit(X_train, y_train)


# 11. Make predictions
y_pred = model.predict(X_test)


# 12. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n--- MODEL PERFORMANCE ---")

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 13. Save trained pipeline
model_path = "ml/model/churn_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved successfully: {model_path}")