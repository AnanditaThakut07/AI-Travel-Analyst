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
    for p in [CLEAN_CSV, MODEL_PATH, META_JSON]:
        if not os.path.exists(p):
            sys.exit(f"[ERROR] {p} not found. Run earlier stages first.")

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
    We use the same feature engineering logic as Stage 5, applied to the
    cleaned dataset rows. Features not present default to 0.
    """
    feature_cols = meta.get("feature_cols", [])
    freq_maps    = meta.get("freq_maps", {})

    feat_df = candidates.copy()

    # Apply frequency encoding to high-cardinality columns
    for col, freq_map in freq_maps.items():
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].map(freq_map).fillna(0.0)

    # One-hot: get_dummies on remaining object columns, then align to model cols
    cat_cols = feat_df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        dummies = pd.get_dummies(feat_df[col], prefix=col, dtype=float)
        feat_df = pd.concat([feat_df.drop(columns=[col]), dummies], axis=1)

    # Dep_hour → bucket
    if "dep_hour" in feat_df.columns:
        from src.features import dep_hour_to_bucket
        feat_df["dep_time_bucket"] = feat_df["dep_hour"].apply(dep_hour_to_bucket)
        feat_df = feat_df.drop(columns=["dep_hour"], errors="ignore")

    # Drop non-feature columns
    drop_cols = ["Date_of_Journey","Dep_Time","Arrival_Time","Route",
                 "Additional_Info","route_combined","Price"]
    feat_df = feat_df.drop(columns=[c for c in drop_cols if c in feat_df.columns], errors="ignore")

    # Align to training feature columns
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
    Date filtering: if date_str is provided and a Date_of_Journey column exists,
    we filter to flights within ±7 days of the target date to handle sparse data.
    If no date is given, all route matches are returned (user can narrow further).
    """
    src_col = next((c for c in df.columns if c.lower() in ("source", "from")), None)
    dst_col = next((c for c in df.columns if c.lower() in ("destination", "to")), None)

    if not src_col or not dst_col:
        return df  # cannot filter — return all

    mask = (
        df[src_col].str.lower().str.strip() == source.strip().lower()
    ) & (
        df[dst_col].str.lower().str.strip() == destination.strip().lower()
    )
    candidates = df[mask].copy()

    # Date window filter
    if date_str and "Date_of_Journey" in candidates.columns:
        try:
            target_date = pd.to_datetime(date_str, dayfirst=True)
            journey_dates = pd.to_datetime(candidates["Date_of_Journey"], errors="coerce")
            date_mask = (journey_dates - target_date).abs() <= pd.Timedelta(days=7)
            if date_mask.sum() >= 3:
                candidates = candidates[date_mask]
            # If fewer than 3 flights in the window, fall back to all route matches
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
        source      : departure city (e.g. 'Delhi')
        destination : arrival city (e.g. 'Mumbai')
        date_str    : travel date string, optional (e.g. '15/03/2019')
        preference  : one of 'cheapest', 'fastest', 'fewest_stops', 'best_value'
        top_n       : number of results to return

    Returns:
        DataFrame with columns: airline, duration_minutes, stops,
        predicted_price, price_score, duration_score, stops_score,
        composite_score, rank
    """
    df, model, meta = load_artifacts()

    candidates = filter_candidates(df, source, destination, date_str)
    if len(candidates) == 0:
        print(f"[Recommender] No flights found for {source} → {destination}.")
        return pd.DataFrame()

    # ── Predict prices for all candidates ───────────────────────────────
    candidates = candidates.copy()
    candidates["predicted_price"] = predict_prices(candidates, model, meta)

    # ── Normalise the three scoring signals ─────────────────────────────
    # Use actual listed price if available, else predicted price
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
        print(f"[Recommender] Unknown preference '{preference}'; defaulting to 'best_value'.")
        preference = "best_value"

    w = PREFERENCE_WEIGHTS[preference]
    candidates["composite_score"] = (
        w["price"]    * candidates["price_score"]
      + w["duration"] * candidates["duration_score"]
      + w["stops"]    * candidates["stops_score"]
    )

    # ── Rank and return top N ────────────────────────────────────────────
    results = (
        candidates
        .sort_values("composite_score")
        .head(top_n)
        .reset_index(drop=True)
    )
    results["rank"] = results.index + 1

    # Select display columns (use whatever is available in the dataset)
    display_cols = ["rank"]
    airline_col  = next((c for c in results.columns if "airline" in c.lower()), None)
    if airline_col: display_cols.append(airline_col)
    if "duration_minutes" in results.columns: display_cols.append("duration_minutes")
    if "stops" in results.columns:            display_cols.append("stops")
    display_cols += ["predicted_price", "price_score", "duration_score",
                     "stops_score", "composite_score"]

    return results[[c for c in display_cols if c in results.columns]]


def main():
    """Demo run with sample query."""
    # Try a common route from the dataset
    demo_queries = [
        ("Delhi", "Cochin",  None, "best_value"),
        ("Mumbai", "Delhi",   None, "cheapest"),
        ("Kolkata", "Bangalore", None, "fastest"),
    ]

    df, _, _ = load_artifacts()
    src_col = next((c for c in df.columns if c.lower() in ("source", "from")), None)
    dst_col = next((c for c in df.columns if c.lower() in ("destination", "to")), None)

    if src_col and dst_col:
        # Use actual values from the dataset for the demo
        actual_src  = df[src_col].value_counts().index[0]
        actual_dst  = df[dst_col].value_counts().index[0]
        demo_queries = [(actual_src, actual_dst, None, "best_value")]

    for src, dst, date, pref in demo_queries:
        print(f"\n{'='*60}")
        print(f"Query: {src} → {dst} | date={date or 'any'} | preference={pref}")
        print("=" * 60)
        results = recommend(src, dst, date, pref)
        if not results.empty:
            print(results.to_string(index=False))
        else:
            print("No results.")


if __name__ == "__main__":
    main()
