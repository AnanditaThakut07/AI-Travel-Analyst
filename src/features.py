"""
Stage 5 — Feature Engineering

Transforms the cleaned dataset into a fully numeric feature matrix
ready for ML training, and saves two artifacts:
  - data/processed/flights_features.csv   (feature matrix + target)
  - data/processed/feature_columns.json   (ordered list of feature names,
    for consistent use by model.py and recommender.py)

Encoding decisions:
  - One-hot encoding for low-cardinality categoricals (airline, source,
    destination, additional_info). Low-cardinality means ≤15 unique values.
    One-hot avoids implying ordinal relationships between categories.
    Trade-off: adds columns, but Random Forest handles high dimensionality well.
  - Frequency encoding for high-cardinality categoricals (route strings, if
    more than 15 unique values). Frequency encoding replaces each category with
    how often it appears in the training set, preserving information about
    route popularity without the column explosion of one-hot.
    Alternative considered: target encoding. Rejected for Stage 5 because it
    requires careful cross-validation to avoid label leakage; frequency
    encoding is leakage-free.
  - Drop raw string columns after encoding (route_combined, Date_of_Journey)
    to keep the feature matrix fully numeric.

Scaling:
  - StandardScaler applied to numeric columns (duration_minutes, dep_hour,
    days_to_departure) for the Linear Regression baseline only. Tree models
    (Random Forest) are scale-invariant, so we store both a scaled and
    unscaled version. Model.py chooses which to use.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
CLEAN_CSV     = os.path.join(ROOT_DIR, "data", "processed", "flights_clean.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUTS_DIR   = os.path.join(ROOT_DIR, "outputs")


# cardinality threshold for one-hot vs frequency encoding
ONE_HOT_MAX_UNIQUE = 15


def frequency_encode(series: pd.Series, freq_map: dict = None):
    """
    Replace each category value with its frequency (proportion) in the series.
    If freq_map is provided (e.g. from training), use it (to avoid test leakage).
    Returns (encoded_series, freq_map).
    """
    if freq_map is None:
        freq_map = (series.value_counts(normalize=True)).to_dict()
    return series.map(freq_map).fillna(0.0), freq_map


def dep_hour_to_bucket(hour: float) -> int:
    """
    Convert departure hour to a 4-bucket time-of-day category.
    Buckets: 0=night (00-05), 1=morning (06-11), 2=afternoon (12-17), 3=evening (18-23)
    Why bucket instead of raw hour? Bucketing reduces noise from exact minute
    differences and encodes the demand-driven price pattern more cleanly
    (red-eye vs. prime-time) without needing cyclical encoding.
    """
    if pd.isna(hour):
        return -1
    h = int(hour)
    if h < 6:   return 0
    if h < 12:  return 1
    if h < 18:  return 2
    return 3


def engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
      features_df : fully numeric DataFrame (features + Price)
      metadata    : dict with freq_maps and scaler info for reproducibility
    """
    df = df.copy()
    metadata = {"freq_maps": {}, "one_hot_cols": [], "numeric_cols": []}

    # ── 1. dep_hour → time bucket ────────────────────────────────────────
    if "dep_hour" in df.columns:
        df["dep_time_bucket"] = df["dep_hour"].apply(dep_hour_to_bucket)
        df = df.drop(columns=["dep_hour"])

    # ── 2. Identify columns to encode ────────────────────────────────────
    # Drop columns we never want as features
    drop_always = [
        "Date_of_Journey", "Dep_Time", "Arrival_Time", "Route",
        "Additional_Info", "route_combined",  # raw strings
    ]
    # Some of these may not exist; only drop what's actually present
    df = df.drop(columns=[c for c in drop_always if c in df.columns], errors="ignore")

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "Price"]

    # ── 3. Encode categoricals ───────────────────────────────────────────
    ohe_dummies = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= ONE_HOT_MAX_UNIQUE:
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
            ohe_dummies.append(dummies)
            metadata["one_hot_cols"].append(col)
            df = df.drop(columns=[col])
            print(f"[Stage 5] One-hot encoded '{col}' ({n_unique} categories → {len(dummies.columns)} cols)")
        else:
            # Frequency encode
            encoded, freq_map = frequency_encode(df[col])
            df[col] = encoded
            metadata["freq_maps"][col] = freq_map
            print(f"[Stage 5] Frequency encoded '{col}' ({n_unique} categories)")
            num_cols.append(col)

    if ohe_dummies:
        df = pd.concat([df] + ohe_dummies, axis=1)

    metadata["numeric_cols"] = num_cols

    # ── 4. Fill remaining NaNs with column median ─────────────────────────
    feature_cols = [c for c in df.columns if c != "Price"]
    for col in feature_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # ── 5. Reorder: all features, then Price ─────────────────────────────
    feature_cols = [c for c in df.columns if c != "Price"]
    df = df[feature_cols + ["Price"]]

    print(f"[Stage 5] Feature matrix shape: {df.shape} ({len(feature_cols)} features + Price)")
    return df, metadata


def main():
    if not os.path.exists(CLEAN_CSV):
        sys.exit(f"[ERROR] {CLEAN_CSV} not found. Run src/data_prep.py first.")

    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    print(f"[Stage 5] Loaded clean dataset: {df.shape}")

    features_df, metadata = engineer(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feat_path = os.path.join(PROCESSED_DIR, "flights_features.csv")
    features_df.to_csv(feat_path, index=False)
    print(f"[Stage 5] Feature matrix saved to: {feat_path}")

    # Save column list for downstream use
    feature_cols = [c for c in features_df.columns if c != "Price"]
    meta_path = os.path.join(PROCESSED_DIR, "feature_columns.json")
    with open(meta_path, "w") as f:
        json.dump({"feature_cols": feature_cols, "freq_maps": metadata["freq_maps"]}, f, indent=2)
    print(f"[Stage 5] Feature metadata saved to: {meta_path}")

    return features_df


if __name__ == "__main__":
    main()
