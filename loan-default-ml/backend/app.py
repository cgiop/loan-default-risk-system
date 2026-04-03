# backend/app.py

from fastapi import FastAPI
import pandas as pd
import joblib

# =========================
# Load trained LightGBM Booster
# =========================
model = joblib.load("../model/loan_model.pkl")

# Load feature means from training
feature_means = joblib.load("../model/feature_means.pkl")

# =========================
# Initialize FastAPI
# =========================
app = FastAPI(title="Loan Default Risk Prediction API")

# =========================
# Root endpoint
# =========================
@app.get("/")
def read_root():
    return {"message": "Loan Default Risk Prediction API is running!"}

# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
def predict(application: dict):
    """
    Accepts a JSON with any subset of features.
    Missing features are automatically filled with the mean values from training data.
    Returns binary prediction and probability.
    """

    # Convert input dict to DataFrame
    input_df = pd.DataFrame([application])

    # Reindex to match model features; missing features appear as NaN
    input_df = input_df.reindex(columns=model.feature_name())

    # Fill missing values with training mean
    for col in input_df.columns:
        input_df[col].fillna(feature_means[col], inplace=True)

    # Predict probability
    prob = model.predict(input_df)[0]

    # Convert to binary prediction (0/1)
    pred = int(prob > 0.5)

    return {"prediction": pred, "probability": float(prob)}