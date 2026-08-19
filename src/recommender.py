"""
Stage 8 — Flight Recommendation System

---
Is this really a recommender system?

Common interview question: "Is this ML or just filtering and sorting?"
Honest answer: This is a content-based, rule-based recommender. It is NOT
collaborative filtering (no user history, no user-item matrix). It IS
a legitimate recommender system in the sense that:
  - It takes a user preference signal (cheapest/fastest/fewest_stops/best_value)
  - It scores and ranks candidates from a catalogue (the flight dataset)
  - It uses a trained ML model's output (predicted price) as part of the score,
    which is more principled than sorting purely on listed price
  - It returns a ranked shortlist with auditable score breakdowns

What it cannot do: personalise to a specific user's history or preferences
beyond the stated preference flag, because there is no user interaction data
in this dataset. That limitation is stated explicitly here and in the README.
---

Scoring logic (transparent, documented):

Each candidate flight is scored on three normalised signals (0-1 scale,
lower raw value = better):
  - price_score    : normalised predicted price (0=cheapest candidate, 1=most expensive)
  - duration_score : normalised duration_minutes
  - stops_score    : normalised number of stops

Preference weights:
  cheapest      : price=1.00, duration=0.00, stops=0.00
  fastest       : price=0.20, duration=0.70, stops=0.10
  fewest_stops  : price=0.20, duration=0.10, stops=0.70
  best_value    : price=0.40, duration=0.35, stops=0.25
    (best_value: price matters most, but not at the cost of a very long or
     very indirect journey — weights chosen to reflect typical traveller trade-offs
     and validated against the price-driver rankings from Stage 4)

Final score = weighted sum of normalised signals (lower is better rank).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")
CLEAN_CSV  = os.path.join(ROOT_DIR, "data", "processed", "flights_clean.csv")
META_JSON  = os.path.join(ROOT_DIR, "data", "processed", "feature_columns.json")
MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "model.pkl")

PREFERENCE_WEIGHTS = {
    "cheapest":     {"price": 1.00, "duration": 0.00, "stops": 0.00},
    "fastest":      {"price": 0.20, "duration": 0.70, "stops": 0.10},
    "fewest_stops": {"price": 0.20, "duration": 0.10, "stops": 0.70},
    "best_value":   {"price": 0.40, "duration": 0.35, "stops": 0.25},
}


def load_artifacts():
    missing = [p for p in [CLEAN_CSV, MODEL_PATH, META_JSON] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Missing pipeline artifacts: {missing}\n"
            "Run the pipeline first:\n"
            "  python3 src/features.py\n"
            "  python3 src/model.py"
        )
    df    = pd.read_csv(CLEAN_CSV, low_memory=False)
    model = joblib.load(MODEL_PATH)
    with open(META_JSON) as f:
        meta = json.load(f)
    return df, model, meta


def _normalise_min_max(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]. Returns 0.5 if all values are equal."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def predict_prices(candidates: pd.DataFrame, model, meta: dict) -> np.ndarray:
    """
    Run the trained model to predict prices for each candidate flight.
    Applies the same encoding pipeline as Stage 5 to the candidate rows,
    then aligns to the training feature columns. Missing columns default to 0.
    """
    feature_cols = meta.get("feature_cols", [])
    freq_maps    = meta.get("freq_maps", {})

    feat_df = candidates.copy()

    # dep_hour → bucket (must happen before OHE loop to avoid encoding it)
    if "dep_hour" in feat_df.columns:
        from src.features import dep_hour_to_bucket
        feat_df["dep_time_bucket"] = feat_df["dep_hour"].apply(dep_hour_to_bucket)
        feat_df = feat_df.drop(columns=["dep_hour"], errors="ignore")

    # Apply frequency encoding to high-cardinality columns
    for col, freq_map in freq_maps.items():
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].map(freq_map).fillna(0.0)

    # One-hot: get_dummies on remaining object columns
    cat_cols = feat_df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        dummies = pd.get_dummies(feat_df[col], prefix=col, dtype=float)
        feat_df = pd.concat([feat_df.drop(columns=[col]), dummies], axis=1)

    # Drop non-feature columns
    drop_cols = ["Flight_ID", "Departure_Date", "Departure_Time", "Arrival_Time",
                 "Duration", "Total_Stops", "route_combined", "Price"]
    feat_df = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns], errors="ignore")

    # Align to training feature columns (fills any unseen columns with 0)
    X = feat_df.reindex(columns=feature_cols, fill_value=0).fillna(0).values
    return model.predict(X)


def filter_candidates(
    df: pd.DataFrame,
    source: str,
    destination: str,
    date_str: str = None,
) -> pd.DataFrame:
    """
    Filter the flight catalogue to matching source/destination.
    The real dataset uses 'Source' and 'Destination' column names.
    Date filtering: if date_str is provided, filter within ±30 days of the
    target date (wider window than original estimate due to 2-year date range).
    Falls back to all route matches if window yields fewer than 5 results.
    """
    src_col = "Source" if "Source" in df.columns else next(
        (c for c in df.columns if c.lower() in ("source", "from")), None)
    dst_col = "Destination" if "Destination" in df.columns else next(
        (c for c in df.columns if c.lower() in ("destination", "to")), None)

    if not src_col or not dst_col:
        return df

    mask = (
        df[src_col].str.lower().str.strip() == source.strip().lower()
    ) & (
        df[dst_col].str.lower().str.strip() == destination.strip().lower()
    )
    candidates = df[mask].copy()

    # Date window filter (uses Departure_Date column from real schema)
    date_col = "Departure_Date" if "Departure_Date" in candidates.columns else None
    if date_str and date_col:
        try:
            target_date  = pd.to_datetime(date_str)
            journey_dates = pd.to_datetime(candidates[date_col], errors="coerce")
            date_mask = (journey_dates - target_date).abs() <= pd.Timedelta(days=30)
            if date_mask.sum() >= 5:
                candidates = candidates[date_mask]
        except Exception:
            pass

    return candidates.reset_index(drop=True)


def recommend(
    source: str,
    destination: str,
    date_str: str = None,
    preference: str = "best_value",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Main recommendation function.

    Args:
        source      : departure city (e.g. 'Hyderabad')
        destination : arrival city (e.g. 'Mumbai')
        date_str    : travel date string, optional (e.g. '2025-06-15')
        preference  : one of 'cheapest', 'fastest', 'fewest_stops', 'best_value'
        top_n       : number of results to return

    Returns:
        DataFrame with rank, airline, duration, stops, predicted_price,
        and score breakdown columns.
    """
    df, model, meta = load_artifacts()

    candidates = filter_candidates(df, source, destination, date_str)
    if len(candidates) == 0:
        print(f"[Recommender] No flights found for {source} → {destination}.")
        return pd.DataFrame()

    # ── Predict prices ───────────────────────────────────────────────────
    candidates = candidates.copy()
    candidates["predicted_price"] = predict_prices(candidates, model, meta)

    # ── Normalise three scoring signals ──────────────────────────────────
    price_signal    = candidates["predicted_price"]
    duration_signal = candidates["duration_minutes"] if "duration_minutes" in candidates.columns \
                      else pd.Series(1.0, index=candidates.index)
    stops_signal    = candidates["stops"] if "stops" in candidates.columns \
                      else pd.Series(0, index=candidates.index)

    candidates["price_score"]    = _normalise_min_max(price_signal)
    candidates["duration_score"] = _normalise_min_max(duration_signal)
    candidates["stops_score"]    = _normalise_min_max(stops_signal.astype(float))

    # ── Apply preference weights ─────────────────────────────────────────
    if preference not in PREFERENCE_WEIGHTS:
        preference = "best_value"

    w = PREFERENCE_WEIGHTS[preference]
    candidates["composite_score"] = (
        w["price"]    * candidates["price_score"]
      + w["duration"] * candidates["duration_score"]
      + w["stops"]    * candidates["stops_score"]
    ).round(4)

    # ── Rank and return top N ────────────────────────────────────────────
    results = (
        candidates
        .sort_values("composite_score")
        .head(top_n)
        .reset_index(drop=True)
    )
    results["rank"] = results.index + 1

    # Display columns — use real schema names
    display_cols = ["rank"]
    for candidate_col in ["Airline", "Travel_Class", "duration_minutes", "stops",
                           "Distance_km", "predicted_price",
                           "price_score", "duration_score", "stops_score", "composite_score"]:
        if candidate_col in results.columns:
            display_cols.append(candidate_col)

    return results[[c for c in display_cols if c in results.columns]]


def main():
    """Demo run using the two most common cities from the actual dataset."""
    df, _, _ = load_artifacts()

    src_col = "Source" if "Source" in df.columns else None
    dst_col = "Destination" if "Destination" in df.columns else None

    if src_col and dst_col:
        top_src = df[src_col].value_counts().index[0]
        top_dst = df[dst_col].value_counts().index[0]
        # Make sure src ≠ dst
        if top_src == top_dst:
            top_dst = df[dst_col].value_counts().index[1]
        demo_queries = [
            (top_src, top_dst, None, "best_value"),
            (top_src, top_dst, None, "cheapest"),
        ]
    else:
        demo_queries = [("Hyderabad", "Mumbai", None, "best_value")]

    for src, dst, date, pref in demo_queries:
        print(f"\n{'='*60}")
        print(f"Query: {src} → {dst} | preference={pref}")
        print("=" * 60)
        results = recommend(src, dst, date, pref)
        if not results.empty:
            print(results.to_string(index=False))
        else:
            print("No results.")


if __name__ == "__main__":
    main()
