import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

from dataset import load_data, get_feature_groups
from model import build_preprocessor, get_model_specs, build_pipeline

RANDOM_STATE = 42
CV_FOLDS = 5
TUNING_SCORING = "roc_auc"
SELECTION_METRIC = "Validation ROC-AUC"

DATA_PATH = Path("data/heart.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "heart_best_model.joblib"
SUMMARY_PATH = MODELS_DIR / "heart_model_summary.json"
VALIDATION_RESULTS_PATH = MODELS_DIR / "validation_results.csv"
TEST_RESULT_PATH = MODELS_DIR / "selected_model_test_result.csv"


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["target"]).copy()
    y = df["target"].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_all_models():
    df = load_data(DATA_PATH)
    numeric_features, categorical_features, _ = get_feature_groups()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    results = []
    fitted_models = {}
    best_params_by_model = {}

    for model_name, spec in get_model_specs().items():
        pipeline = build_pipeline(preprocessor, spec["model"])
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["params"],
            cv=CV_FOLDS,
            scoring=TUNING_SCORING,
            n_jobs=-1,
            refit=True,
            return_train_score=True
        )
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        fitted_models[model_name] = best_model
        best_params_by_model[model_name] = grid.best_params_

        train_pred = best_model.predict(X_train)
        train_prob = best_model.predict_proba(X_train)[:, 1]
        val_pred = best_model.predict(X_val)
        val_prob = best_model.predict_proba(X_val)[:, 1]

        results.append({
            "Model": model_name,
            "Best Params": str(grid.best_params_),
            "Best CV ROC-AUC": grid.best_score_,
            "Train Accuracy": accuracy_score(y_train, train_pred),
            "Train ROC-AUC": roc_auc_score(y_train, train_prob),
            "Validation Accuracy": accuracy_score(y_val, val_pred),
            "Validation Precision": precision_score(y_val, val_pred, zero_division=0),
            "Validation Recall": recall_score(y_val, val_pred, zero_division=0),
            "Validation F1": f1_score(y_val, val_pred, zero_division=0),
            "Validation ROC-AUC": roc_auc_score(y_val, val_prob),
        })

    results_df = pd.DataFrame(results).sort_values(
        by=["Validation ROC-AUC", "Validation Accuracy", "Best CV ROC-AUC"],
        ascending=False
    ).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_model_name]

    # Evaluate the selected deployment model once on the test set.
    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)[:, 1]

    test_result = pd.DataFrame([{
        "Model": best_model_name,
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Test Precision": precision_score(y_test, y_pred_test, zero_division=0),
        "Test Recall": recall_score(y_test, y_pred_test, zero_division=0),
        "Test F1": f1_score(y_test, y_pred_test, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_test, y_prob_test)
    }])

    summary = {
        "deployment_model_name": best_model_name,
        "deployment_model_params": best_params_by_model[best_model_name],
        "deployment_selection_metric": SELECTION_METRIC,
        "deployment_validation_score": float(results_df.iloc[0]["Validation ROC-AUC"]),
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "validation_results": results_df.to_dict(orient="records"),
        "selected_model_test_result": test_result.iloc[0].to_dict()
    }

    joblib.dump(best_model, MODEL_PATH)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    results_df.to_csv(VALIDATION_RESULTS_PATH, index=False)
    test_result.to_csv(TEST_RESULT_PATH, index=False)

    print(f"Deployment model selected from validation set: {best_model_name}")
    print(f"Best parameters: {best_params_by_model[best_model_name]}")
    print(f"Best validation ROC-AUC: {results_df.iloc[0]['Validation ROC-AUC']:.4f}")
    print(f"Saved deployment model to: {MODEL_PATH.resolve()}")
    print(f"Saved summary to: {SUMMARY_PATH.resolve()}")

    return best_model, summary, results_df, test_result


if __name__ == "__main__":
    train_all_models()
