from __future__ import annotations

import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.feature_engineering import create_features
from src.preprocessing_pipeline import normalize_borrower_frame


DEFAULT_MODEL_CANDIDATES = (
    "models/logistic_regression.pkl",
    "models/decision_tree.pkl",
    "models/random_forest.pkl",
)


def load_model(model_path: Optional[str] = None) -> tuple[Any, str]:
    """
    Load the preferred trained model from disk.
    """
    candidate_paths = [model_path] if model_path else list(DEFAULT_MODEL_CANDIDATES)

    for path in candidate_paths:
        if path and os.path.exists(path):
            return joblib.load(path), os.path.basename(path)

    raise FileNotFoundError("Model pipeline not found in 'models/'. Please train a model first.")


def build_input_frame(borrower_profile: Dict[str, Any]) -> pd.DataFrame:
    """
    Normalize the borrower payload into the training schema expected by feature engineering.
    """
    input_frame = pd.DataFrame(
        [
            {
                "Age": borrower_profile.get("age"),
                "Sex": borrower_profile.get("sex"),
                "Job": borrower_profile.get("job"),
                "Housing": borrower_profile.get("housing"),
                "Saving accounts": borrower_profile.get("saving_accounts"),
                "Checking account": borrower_profile.get("checking_account"),
                "Credit amount": borrower_profile.get("credit_amount"),
                "Duration": borrower_profile.get("duration"),
                "Purpose": borrower_profile.get("purpose"),
            }
        ]
    )
    return normalize_borrower_frame(input_frame)


def _align_features(model: Any, features: pd.DataFrame) -> pd.DataFrame:
    actual_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    if hasattr(actual_model, "feature_names_in_"):
        expected_cols = actual_model.feature_names_in_
        for col in expected_cols:
            if col not in features.columns:
                features[col] = 0
        features = features[list(expected_cols)]
        features = features.fillna(features.median(numeric_only=True))

    return features


def summarize_risk_factors(borrower_profile: Dict[str, Any], risk_score: float) -> list[str]:
    """
    Produce human-readable risk signals that can be shown directly or fed into an agent prompt.
    """
    factors: list[str] = []

    if risk_score >= 0.5:
        factors.append("Model classified the application as high risk.")
    else:
        factors.append("Model classified the application as lower risk.")

    credit_amount = borrower_profile.get("credit_amount")
    duration = borrower_profile.get("duration")
    savings = borrower_profile.get("saving_accounts")
    checking = borrower_profile.get("checking_account")
    housing = borrower_profile.get("housing")
    job = borrower_profile.get("job")

    try:
        if credit_amount is not None and float(credit_amount) >= 10000:
            factors.append("Requested credit amount is elevated.")
    except (TypeError, ValueError):
        pass

    try:
        if duration is not None and float(duration) >= 36:
            factors.append("Requested tenor is relatively long.")
    except (TypeError, ValueError):
        pass

    if savings in {"NA", "little"}:
        factors.append("Savings account strength is limited.")
    if checking in {"NA", "little"}:
        factors.append("Checking account history appears weak.")
    if housing == "rent":
        factors.append("Borrower does not own the residence.")
    if job in {0, 1}:
        factors.append("Employment profile maps to lower stability categories.")

    return factors


def predict_risk_score(
    borrower_profile: Dict[str, Any],
    model: Optional[Any] = None,
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score a borrower profile using the existing ML pipeline and return a structured explanation payload.
    """
    resolved_model, resolved_model_name = (model, model_name) if model is not None else load_model(model_path)

    input_frame = build_input_frame(borrower_profile)
    features = create_features(input_frame)
    aligned_features = _align_features(resolved_model, features)

    actual_model = resolved_model.best_estimator_ if hasattr(resolved_model, "best_estimator_") else resolved_model
    risk_score = float(actual_model.predict_proba(aligned_features)[0][1])
    risk_band = "High" if risk_score >= 0.5 else "Low"

    return {
        "risk_score": risk_score,
        "risk_band": risk_band,
        "model_name": resolved_model_name,
        "borrower_profile": borrower_profile,
        "risk_factors": summarize_risk_factors(borrower_profile, risk_score),
    }
