from typing import Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )


def get_model_specs() -> Dict[str, Dict[str, Any]]:
    """
    Candidate models and hyperparameter grids matching the notebook logic.
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=5000, random_state=42),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__solver": ["lbfgs"],
                "model__class_weight": [None, "balanced"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2"]
            }
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "model__C": [0.1, 0.5, 1.0, 5.0, 10.0],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
                "model__class_weight": [None, "balanced"]
            }
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "model__n_estimators": [50, 100, 150, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.8, 1.0]
            }
        }
    }


def build_pipeline(preprocessor, model):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
