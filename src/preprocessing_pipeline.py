from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = ["Age", "Credit amount", "Duration"]
CATEGORICAL_COLUMNS = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]


def normalize_borrower_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded borrower data to the schema used across the app.
    """
    normalized = df.copy()

    if "Unnamed: 0" in normalized.columns:
        normalized = normalized.drop(columns=["Unnamed: 0"])

    for col in NUMERIC_COLUMNS:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        if col in normalized.columns:
            normalized[col] = normalized[col].fillna("Unknown").astype(str)

    return normalized


def build_preprocessor(
    numeric_columns: Iterable[str] = NUMERIC_COLUMNS,
    categorical_columns: Iterable[str] = CATEGORICAL_COLUMNS,
) -> ColumnTransformer:
    """
    Build a reusable sklearn preprocessing pipeline for raw borrower records.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
    )


def preprocess_uploaded_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Return analytics-friendly transformed features for uploaded raw borrower data.
    """
    normalized = normalize_borrower_frame(df)
    available_numeric = [col for col in NUMERIC_COLUMNS if col in normalized.columns]
    available_categorical = [col for col in CATEGORICAL_COLUMNS if col in normalized.columns]

    preprocessor = build_preprocessor(
        numeric_columns=available_numeric,
        categorical_columns=available_categorical,
    )
    transformed = preprocessor.fit_transform(normalized)
    feature_names = preprocessor.get_feature_names_out()

    transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=normalized.index)
    return {
        "normalized": normalized,
        "transformed": transformed_frame,
        "preprocessor": preprocessor,
    }
