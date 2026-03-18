import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import load_data, TARGET_COLUMN
from train import split_data, DATA_PATH, MODEL_PATH, SUMMARY_PATH


def evaluate_selected_model():
    if not MODEL_PATH.exists() or not SUMMARY_PATH.exists():
        raise FileNotFoundError("Trained model artifacts not found. Run train.py first.")

    df = load_data(DATA_PATH)
    _, _, X_test, _, _, y_test = split_data(df)

    model = joblib.load(MODEL_PATH)
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Deployment model:", summary["deployment_model_name"])
    print("Selection basis:", summary["deployment_selection_metric"])
    print("Held-out test results:")
    print(json.dumps(summary["selected_model_test_result"], indent=2))

    # ROC curve
    fig = plt.figure()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_value = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{summary['deployment_model_name']} (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Test ROC Curve for Selected Deployment Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confusion matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix, {summary['deployment_model_name']}")
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    evaluate_selected_model()
