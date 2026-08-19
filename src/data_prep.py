"""
Stage 2 — Data Cleaning & Preprocessing

Reads the raw dataset, fixes all structural and quality issues, and writes
a clean CSV to data/processed/flights_clean.csv.

What "cleaning" means here:
  - Standardising inconsistent string formats (dates, duration, times)
  - Removing or imputing missing values with a documented justification for each
  - Dropping exact duplicates that would bias model training
  - Capping extreme price outliers that are almost certainly data entry errors
  - Deriving a days_to_departure feature (proxy for booking lead time)

Every fix is justified inline with an alternative that was considered but
rejected, so the decisions are defensible in an interview.
"""

import os
import re
import sys
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
RAW_XLSX      = os.path.join(ROOT_DIR, "data", "raw", "flights.xlsx")
RAW_CSV       = os.path.join(ROOT_DIR, "data", "raw", "flights.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUT_PATH      = os.path.join(PROCESSED_DIR, "flights_clean.csv")


# ── helpers ────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    if os.path.exists(RAW_XLSX):
        return pd.read_excel(RAW_XLSX)
    if os.path.exists(RAW_CSV):
        return pd.read_csv(RAW_CSV)
    sys.exit("[ERROR] Raw dataset not found. Run src/inspect_data.py first.")


def parse_duration(duration_str) -> float:
    """
    Convert duration strings like '2h 30m', '1h', '45m', '2h 0m' to minutes.
    Returns NaN for unparseable values.

    Why minutes, not hours? Minutes is integer-friendly and avoids float
    precision issues when encoding fractional hours.
    """
    if pd.isna(duration_str):
        return np.nan
    s = str(duration_str).strip().lower()
    hours   = re.search(r"(\d+)\s*h", s)
    minutes = re.search(r"(\d+)\s*m", s)
    total   = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    return float(total) if (hours or minutes) else np.nan


def parse_stops(stops_val) -> int:
    """
    Normalise the 'Total_Stops' column to an integer.
    'non-stop' → 0, '1 stop' → 1, '2 stops' → 2, etc.
    """
    if pd.isna(stops_val):
        return np.nan
    s = str(stops_val).strip().lower()
    if "non" in s or s == "0":
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan


def parse_dep_hour(time_str) -> float:
    """
    Extract departure hour (0-23) from strings like '06:45', '18:30 PM'.
    Returns NaN on failure.
    """
    if pd.isna(time_str):
        return np.nan
    s = str(time_str).strip()
    m = re.match(r"(\d{1,2}):(\d{2})", s)
    if m:
        return float(m.group(1))
    return np.nan


def parse_date(date_str) -> pd.Timestamp:
    """Parse date strings with pandas' flexible parser; return NaT on failure."""
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except Exception:
        return pd.NaT


def days_to_departure_proxy(df: pd.DataFrame) -> pd.Series:
    """
    The dataset does not have an explicit 'booking date' column, so we cannot
    compute true days-to-departure (DTD) from actual purchase timestamps.

    Proxy approach: treat the journey date relative to the earliest journey
    date in the dataset as a relative ordering feature. This won't give
    absolute DTD numbers, but it preserves the temporal ordering signal
    (earlier journeys relative to the dataset window tend to have been booked
    further in advance on average), which still carries predictive information.

    Alternative considered: drop DTD entirely. Rejected because booking lead
    time is a well-documented price driver and losing it weakens the model.
    Alternative considered: use a fixed reference date (e.g. dataset pull date).
    Rejected because we don't know when the dataset was pulled.
    """
    if "Date_of_Journey" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    min_date = df["Date_of_Journey"].min()
    return (df["Date_of_Journey"] - min_date).dt.days.astype(float)


# ── main cleaning pipeline ─────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[Stage 2] Raw shape: {df.shape}")

    # ── 1. column name normalisation ─────────────────────────────────────
    # Strip whitespace from column names (common in Excel exports)
    df.columns = df.columns.str.strip()

    # ── 2. drop exact duplicates ─────────────────────────────────────────
    # Exact duplicates are almost certainly copy-paste errors in the source
    # spreadsheet. Keeping them would inflate the weight of those records
    # in training. Alternative: keep first occurrence — same net effect.
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Stage 2] Dropped {before - len(df)} exact duplicate rows.")

    # ── 3. parse Date_of_Journey ─────────────────────────────────────────
    if "Date_of_Journey" in df.columns:
        df["Date_of_Journey"] = df["Date_of_Journey"].apply(parse_date)
        n_bad = df["Date_of_Journey"].isna().sum()
        if n_bad > 0:
            print(f"[Stage 2] Dropping {n_bad} rows with unparseable journey dates.")
            df = df.dropna(subset=["Date_of_Journey"])
        df["journey_month"] = df["Date_of_Journey"].dt.month
        df["journey_dow"]   = df["Date_of_Journey"].dt.dayofweek  # 0=Mon

    # ── 4. derive days_to_departure proxy ───────────────────────────────
    df["days_to_departure"] = days_to_departure_proxy(df)

    # ── 5. parse Duration → duration_minutes ────────────────────────────
    dur_col = next((c for c in df.columns if "duration" in c.lower()), None)
    if dur_col:
        df["duration_minutes"] = df[dur_col].apply(parse_duration)
        n_bad = df["duration_minutes"].isna().sum()
        if n_bad > 0:
            # Impute with median rather than drop: losing rows over a parseable
            # issue is wasteful; median is robust to the outliers still present.
            median_dur = df["duration_minutes"].median()
            print(f"[Stage 2] Imputing {n_bad} unparseable durations with median ({median_dur:.0f} min).")
            df["duration_minutes"] = df["duration_minutes"].fillna(median_dur)

    # ── 6. parse Total_Stops → stops (integer) ───────────────────────────
    stop_col = next((c for c in df.columns if "stop" in c.lower()), None)
    if stop_col:
        df["stops"] = df[stop_col].apply(parse_stops)
        n_bad = df["stops"].isna().sum()
        if n_bad > 0:
            # Impute with mode (most common number of stops); dropping would
            # lose valid price/flight data over a single column issue.
            mode_stops = int(df["stops"].mode()[0])
            print(f"[Stage 2] Imputing {n_bad} unknown stop counts with mode ({mode_stops}).")
            df["stops"] = df["stops"].fillna(mode_stops).astype(int)
        else:
            df["stops"] = df["stops"].astype(int)

    # ── 7. parse Dep_Time → dep_hour ────────────────────────────────────
    dep_col = next((c for c in df.columns if "dep" in c.lower() and "time" in c.lower()), None)
    if dep_col:
        df["dep_hour"] = df[dep_col].apply(parse_dep_hour)
        n_bad = df["dep_hour"].isna().sum()
        if n_bad > 0:
            median_h = df["dep_hour"].median()
            print(f"[Stage 2] Imputing {n_bad} unparseable dep_hours with median ({median_h:.0f}h).")
            df["dep_hour"] = df["dep_hour"].fillna(median_h)

    # ── 8. clean the Price column ─────────────────────────────────────────
    price_col = next((c for c in df.columns if "price" in c.lower()), None)
    if price_col:
        # Coerce to numeric; non-numeric entries become NaN
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        n_bad = df[price_col].isna().sum()
        if n_bad > 0:
            print(f"[Stage 2] Dropping {n_bad} rows where Price is non-numeric.")
            df = df.dropna(subset=[price_col])

        # Outlier treatment: prices below ₹500 or above the 99.5th percentile
        # are almost certainly data errors (test records, miskeys).
        # Cap rather than drop, to keep the row count high.
        low_clip  = 500
        high_clip = df[price_col].quantile(0.995)
        n_low  = (df[price_col] < low_clip).sum()
        n_high = (df[price_col] > high_clip).sum()
        if n_low > 0:
            print(f"[Stage 2] Dropping {n_low} rows with Price < ₹{low_clip} (likely errors).")
            df = df[df[price_col] >= low_clip]
        if n_high > 0:
            print(f"[Stage 2] Capping {n_high} rows at 99.5th-percentile Price = ₹{high_clip:.0f}.")
            df[price_col] = df[price_col].clip(upper=high_clip)

        # Rename to a standard column name used by all downstream scripts
        df = df.rename(columns={price_col: "Price"})

    # ── 9. strip leading/trailing whitespace from all string columns ──────
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # ── 10. build route feature ───────────────────────────────────────────
    src_col  = next((c for c in df.columns if c.lower() in ("source", "from")), None)
    dst_col  = next((c for c in df.columns if c.lower() in ("destination", "to")), None)
    if src_col and dst_col:
        df["route_combined"] = df[src_col].str.lower() + "_to_" + df[dst_col].str.lower()

    print(f"[Stage 2] Clean shape: {df.shape}")
    return df


def main():
    df = load_raw()
    df = clean(df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[Stage 2] Clean dataset saved to: {OUT_PATH}")
    return df


if __name__ == "__main__":
    main()
