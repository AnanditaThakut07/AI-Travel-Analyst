"""
Stage 5 — Feature Engineering

Real schema after cleaning:
  Categorical (low-cardinality ≤15 unique): Airline(39→OHE after grouping),
    Source, Destination, Travel_Class(4), Season(4), Weekday(7),
    Aircraft_Type(8), Booking_Channel(5)
  Categorical (high-cardinality): route_combined (many source×dest combos)
  Numeric: duration_minutes, stops, Distance_km, Days_Before_Departure,
    Passenger_Count, dep_hour, journey_month

Encoding decisions:
  - One-hot: Travel_Class, Season, Weekday, Aircraft_Type, Booking_Channel
    (all ≤10 unique values, no ordinal relationship implied)
  - One-hot: Source, Destination (54 unique each — manageable for RF)
  - Frequency encode: Airline (39 unique), route_combined (many)
    Airline could be one-hot (39 cols) but frequency encoding is more
    parsimonious and still preserves popularity signal.
  - dep_hour → 4-bucket time-of-day (Night/Morning/Afternoon/Evening)
    to reduce noise from exact minute differences
  - All numeric features kept as-is for Random Forest (scale-invariant)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
CLEAN_CSV     = os.path.join(ROOT_DIR, "data", "processed", "flights_clean.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

ONE_HOT_COLS = [
    "Travel_Class", "Season", "Weekday", "Aircraft_Type", "Booking_Channel",
    "Source", "Destination",
]
FREQ_ENC_COLS = ["Airline", "route_combined"]
DROP_COLS     = []  # columns to drop entirely (none needed with this schema)


def dep_hour_to_bucket(hour) -> int:
    """
    0=Night (00–05), 1=Morning (06–11), 2=Afternoon (12–17), 3=Evening (18–23)
    Bucketing captures demand-driven price patterns (red-eye vs prime-time)
    more cleanly than using the raw hour, which adds 24 possible values.
    """
    if pd.isna(hour):
        return -1
    h = int(float(hour))
    if h < 6:   return 0
    if h < 12:  return 1
    if h < 18:  return 2
    return 3


def frequency_encode(series: pd.Series, freq_map: dict = None):
    """Replace each category with its proportion in the series."""
    if freq_map is None:
        freq_map = series.value_counts(normalize=True).to_dict()
    return series.map(freq_map).fillna(0.0), freq_map


def engineer(df: pd.DataFrame) -> tuple:
    df = df.copy()
    freq_maps = {}

    # ── dep_hour → time bucket ────────────────────────────────────────────
    if "dep_hour" in df.columns:
        df["dep_time_bucket"] = df["dep_hour"].apply(dep_hour_to_bucket)
        df = df.drop(columns=["dep_hour"])
        print("[Stage 5] Converted dep_hour → dep_time_bucket (4 buckets)")

    # ── Frequency encode high-cardinality categoricals ────────────────────
    for col in FREQ_ENC_COLS:
        if col in df.columns:
            encoded, fmap = frequency_encode(df[col])
            df[col] = encoded
            freq_maps[col] = fmap
            print(f"[Stage 5] Frequency encoded '{col}'")

    # ── One-hot encode low-cardinality categoricals ───────────────────────
    ohe_parts = []
    for col in ONE_HOT_COLS:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
        ohe_parts.append(dummies)
        df = df.drop(columns=[col])
        print(f"[Stage 5] One-hot encoded '{col}' → {len(dummies.columns)} columns")

    if ohe_parts:
        df = pd.concat([df] + ohe_parts, axis=1)

    # ── Drop any remaining object columns (shouldn't be any) ─────────────
    leftover_obj = df.select_dtypes(include="object").columns.tolist()
    if leftover_obj:
        print(f"[Stage 5] Dropping remaining string columns: {leftover_obj}")
        df = df.drop(columns=leftover_obj)

    # ── Fill any residual NaNs ────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c != "Price"]
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Reorder: features first, Price last ──────────────────────────────
    feature_cols = [c for c in df.columns if c != "Price"]
    df = df[feature_cols + ["Price"]]

    print(f"\n[Stage 5] Final feature matrix: {df.shape[0]:,} rows × {len(feature_cols)} features + Price")
    return df, freq_maps


def main():
    if not os.path.exists(CLEAN_CSV):
        sys.exit(f"[ERROR] {CLEAN_CSV} not found. Run src/data_prep.py first.")

    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    print(f"[Stage 5] Loaded clean dataset: {df.shape}")

    features_df, freq_maps = engineer(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feat_path = os.path.join(PROCESSED_DIR, "flights_features.csv")
    features_df.to_csv(feat_path, index=False)
    print(f"[Stage 5] Feature matrix saved: {feat_path}")

    feature_cols = [c for c in features_df.columns if c != "Price"]
    meta = {"feature_cols": feature_cols, "freq_maps": freq_maps}
    meta_path = os.path.join(PROCESSED_DIR, "feature_columns.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Stage 5] Metadata saved: {meta_path}")

    return features_df


if __name__ == "__main__":
    main()
