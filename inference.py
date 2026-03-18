import json
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

MODEL_PATH = Path("models/heart_best_model.joblib")
SUMMARY_PATH = Path("models/heart_model_summary.json")


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Saved deployment model not found. Run train.py first.")
    model = joblib.load(MODEL_PATH)
    summary = {}
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
    return model, summary


def make_input_df(payload: Dict[str, Any]) -> pd.DataFrame:
    required = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalch", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    missing = [c for c in required if c not in payload]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))

    row = pd.DataFrame([payload])
    row["age"] = pd.to_numeric(row["age"], errors="coerce")
    row["trestbps"] = pd.to_numeric(row["trestbps"], errors="coerce")
    row["chol"] = pd.to_numeric(row["chol"], errors="coerce")
    row["thalch"] = pd.to_numeric(row["thalch"], errors="coerce")
    row["oldpeak"] = pd.to_numeric(row["oldpeak"], errors="coerce")
    row["ca"] = pd.to_numeric(row["ca"], errors="coerce")
    return row


def top_factors(row: pd.DataFrame):
    r = row.iloc[0]
    factors = []
    if pd.notna(r["cp"]) and str(r["cp"]).lower() == "asymptomatic":
        factors.append("Asymptomatic chest pain pattern")
    if pd.notna(r["oldpeak"]) and float(r["oldpeak"]) >= 2.0:
        factors.append("Elevated ST depression (oldpeak)")
    if pd.notna(r["thalch"]) and float(r["thalch"]) < 120:
        factors.append("Lower maximum heart rate")
    if pd.notna(r["ca"]) and float(r["ca"]) >= 1:
        factors.append("Presence of major vessel involvement")
    if pd.notna(r["exang"]) and str(r["exang"]).lower() == "true":
        factors.append("Exercise induced angina")
    if pd.notna(r["chol"]) and float(r["chol"]) >= 240:
        factors.append("Elevated cholesterol")
    if pd.notna(r["trestbps"]) and float(r["trestbps"]) >= 140:
        factors.append("High resting blood pressure")
    if pd.notna(r["age"]) and float(r["age"]) >= 55:
        factors.append("Older age")
    return factors[:3] if factors else ["Overall feature pattern suggests lower observed risk"]


def predict_from_payload(payload: Dict[str, Any]):
    model, summary = load_artifacts()
    row = make_input_df(payload)
    prob = float(model.predict_proba(row)[0, 1])
    pred = int(prob >= 0.5)
    label = "High Risk of Heart Disease" if pred == 1 else "Lower Risk of Heart Disease"
    recommendation = (
        "Consider physician review and confirmatory cardiovascular assessment."
        if pred == 1 else
        "Continue routine clinical review and consider broader context before action."
    )

    return {
        "deployment_model_name": summary.get("deployment_model_name", "Unknown"),
        "risk_probability": round(prob, 4),
        "prediction": pred,
        "prediction_label": label,
        "recommendation": recommendation,
        "top_factors": top_factors(row),
        "disclaimer": "Educational demonstration only. Not for clinical diagnosis or treatment."
    }
