from pathlib import Path
from typing import Tuple, List

import pandas as pd

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET_COLUMN = "target"


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and standardize the heart disease dataset used by the notebook.
    Keeps the original dataset feature name `thalch`.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Standardize column names only.
    df.columns = [c.strip().lower() for c in df.columns]

    # Support alternate target names from some dataset versions.
    if TARGET_COLUMN not in df.columns:
        for alt in ["num", "output", "label", "condition"]:
            if alt in df.columns:
                df = df.rename(columns={alt: TARGET_COLUMN})
                break

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_cols)
        )

    df = df[EXPECTED_COLUMNS].drop_duplicates().copy()

    # Standardize target to binary.
    if df[TARGET_COLUMN].dtype == object:
        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .astype(str)
            .str.lower()
            .isin(["1", "true", "yes", "disease", "present"])
        ).astype(int)

    # UCI severity classes may be 0,1,2,3,4. Convert > 0 to disease present.
    df[TARGET_COLUMN] = (
        pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0) > 0
    ).astype(int)

    # Convert only numeric features to numeric.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize categorical text.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    return df

def get_feature_groups() -> Tuple[List[str], List[str], str]:
    return NUMERIC_FEATURES.copy(), CATEGORICAL_FEATURES.copy(), TARGET_COLUMN
