"""
Stage 2 — Data Cleaning & Preprocessing

Actual schema confirmed in Stage 1:
  Flight_ID, Airline, Source, Destination, Departure_Date, Departure_Time,
  Arrival_Time, Duration, Total_Stops, Distance_km, Travel_Class,
  Days_Before_Departure, Season, Weekday, Aircraft_Type, Booking_Channel,
  Passenger_Count, Price

Key cleaning challenges discovered:
  1. All columns loaded as object dtype — numeric columns need coercion
  2. Duration is a mix of float strings ("1.67") and "Xh Ym" strings
  3. Total_Stops is a mix of "0"/"1"/"2" integers and "non-stop"/"1 stop"/"2 stops" strings
  4. Departure_Time is a mix of "HH:MM" and "HH:MM AM/PM" formats
  5. ~5% null rate in every column
  6. Price has right-skewed distribution with some extreme international fares

Every fix is justified inline with the alternative considered.
"""

import os
import re
import sys
import numpy as np
import pandas as pd

ROOT_DIR      = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV       = os.path.join(ROOT_DIR, "data", "raw", "flights.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUT_PATH      = os.path.join(PROCESSED_DIR, "flights_clean.csv")


def load_raw() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        sys.exit("[ERROR] Raw dataset not found. Place flights.csv in data/raw/")
    return pd.read_csv(RAW_CSV, low_memory=False)


def parse_duration_to_minutes(val) -> float:
    """
    Normalise the heterogeneous Duration column to float minutes.
    Observed formats:
      - "1.67"  → treat as hours (multiply by 60)
      - "0h 45m" → parse hours + minutes
      - "14.80"  → treat as hours
    Assumption: bare floats are in hours (consistent with short domestic
    flights being <2.0 and long-hauls being >10.0).
    Alternative: treat bare floats as minutes. Rejected because "0.75"
    would imply 0.75 minutes (45 seconds), which makes no sense for a flight.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Pattern: "Xh Ym" or "Xh" or "Ym"
    h_match = re.search(r"(\d+)\s*h", s)
    m_match = re.search(r"(\d+)\s*m", s)
    if h_match or m_match:
        hours   = int(h_match.group(1)) if h_match else 0
        minutes = int(m_match.group(1)) if m_match else 0
        return float(hours * 60 + minutes)
    # Bare numeric — treat as hours
    try:
        return float(s) * 60.0
    except ValueError:
        return np.nan


def parse_stops(val) -> int:
    """
    Normalise Total_Stops to integer.
    Observed: "0", "1", "2", "3", "4", "non-stop", "1 stop", "2 stops"
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ("non-stop", "nonstop", "0"):
        return 0
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan


def parse_dep_hour(val) -> float:
    """
    Extract departure hour (0–23) from mixed time formats.
    "07:05" → 7.0
    "8:10 PM" → 20.0
    "10:40 PM" → 22.0
    "12:10" → 12.0 (noon)
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", s, re.IGNORECASE)
    if not m:
        return np.nan
    hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if ampm:
        ampm = ampm.upper()
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
    return float(hour)


def main():
    df = load_raw()
    print(f"[Stage 2] Raw shape: {df.shape}")

    # ── 1. Strip whitespace from column names ────────────────────────────
    df.columns = df.columns.str.strip()

    # ── 2. Drop exact duplicates ─────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Stage 2] Dropped {before - len(df)} duplicate rows.")

    # ── 3. Drop Flight_ID — not a feature, just an identifier ────────────
    df = df.drop(columns=["Flight_ID"], errors="ignore")

    # ── 4. Coerce numeric columns ─────────────────────────────────────────
    # Distance_km, Days_Before_Departure, Passenger_Count, Price all stored
    # as strings due to mixed types. Coerce with errors='coerce' → NaN.
    for col in ["Distance_km", "Days_Before_Departure", "Passenger_Count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 5. Parse Price ────────────────────────────────────────────────────
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    n_bad_price = df["Price"].isna().sum()
    if n_bad_price:
        print(f"[Stage 2] Dropping {n_bad_price} rows with non-numeric Price.")
        df = df.dropna(subset=["Price"])

    # Remove prices ≤ 0 (data errors)
    n_zero = (df["Price"] <= 0).sum()
    if n_zero:
        print(f"[Stage 2] Dropping {n_zero} rows with Price ≤ 0.")
        df = df[df["Price"] > 0]

    # Cap at 99.5th percentile to reduce influence of extreme international fares
    # Alternative: log-transform. Not done here to keep Price interpretable in ₹.
    high_clip = df["Price"].quantile(0.995)
    n_high = (df["Price"] > high_clip).sum()
    if n_high:
        print(f"[Stage 2] Capping {n_high} extreme prices at ₹{high_clip:,.0f} (99.5th pct).")
        df["Price"] = df["Price"].clip(upper=high_clip)

    # ── 6. Parse Duration → duration_minutes ────────────────────────────
    if "Duration" in df.columns:
        df["duration_minutes"] = df["Duration"].apply(parse_duration_to_minutes)
        n_bad = df["duration_minutes"].isna().sum()
        med_dur = df["duration_minutes"].median()
        if n_bad:
            print(f"[Stage 2] Imputing {n_bad} unparseable durations with median ({med_dur:.0f} min).")
            df["duration_minutes"] = df["duration_minutes"].fillna(med_dur)
        # Sanity: flights < 20 min or > 24 h are implausible
        n_implausible = ((df["duration_minutes"] < 20) | (df["duration_minutes"] > 1440)).sum()
        if n_implausible:
            print(f"[Stage 2] Capping {n_implausible} implausible durations to [20, 1440] min.")
            df["duration_minutes"] = df["duration_minutes"].clip(lower=20, upper=1440)
        df = df.drop(columns=["Duration"])

    # ── 7. Parse Total_Stops → stops (int) ──────────────────────────────
    if "Total_Stops" in df.columns:
        df["stops"] = df["Total_Stops"].apply(parse_stops)
        n_bad = df["stops"].isna().sum()
        if n_bad:
            mode_stops = int(df["stops"].mode()[0])
            print(f"[Stage 2] Imputing {n_bad} unknown stop counts with mode ({mode_stops}).")
            df["stops"] = df["stops"].fillna(mode_stops)
        df["stops"] = df["stops"].astype(int)
        df = df.drop(columns=["Total_Stops"])

    # ── 8. Parse Departure_Time → dep_hour ──────────────────────────────
    if "Departure_Time" in df.columns:
        df["dep_hour"] = df["Departure_Time"].apply(parse_dep_hour)
        n_bad = df["dep_hour"].isna().sum()
        if n_bad:
            med_h = df["dep_hour"].median()
            print(f"[Stage 2] Imputing {n_bad} dep_hours with median ({med_h:.0f}h).")
            df["dep_hour"] = df["dep_hour"].fillna(med_h)
        df = df.drop(columns=["Departure_Time"])

    # ── 9. Parse Departure_Date ──────────────────────────────────────────
    if "Departure_Date" in df.columns:
        df["Departure_Date"] = pd.to_datetime(df["Departure_Date"], errors="coerce")
        df["journey_month"]  = df["Departure_Date"].dt.month.fillna(-1).astype(int)
        df = df.drop(columns=["Departure_Date"])

    # ── 10. Drop Arrival_Time (not useful as raw string; duration is better) ─
    df = df.drop(columns=["Arrival_Time"], errors="ignore")

    # ── 11. Handle remaining nulls in categorical columns ────────────────
    # Fill with "Unknown" — preserves rows while signalling missingness.
    # Alternative: drop rows with any null. Rejected because ~5% nulls per
    # column means we'd lose nearly half the dataset.
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        n_null = df[col].isna().sum()
        if n_null:
            df[col] = df[col].fillna("Unknown")

    # ── 12. Handle remaining numeric nulls with column median ────────────
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        n_null = df[col].isna().sum()
        if n_null and col != "Price":
            df[col] = df[col].fillna(df[col].median())

    # ── 13. Build route_combined feature ────────────────────────────────
    if "Source" in df.columns and "Destination" in df.columns:
        df["route_combined"] = (
            df["Source"].str.strip().str.lower()
            + "_to_"
            + df["Destination"].str.strip().str.lower()
        )

    print(f"[Stage 2] Clean shape: {df.shape}")
    print(f"[Stage 2] Columns: {df.columns.tolist()}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[Stage 2] Saved clean dataset: {OUT_PATH}")
    return df


if __name__ == "__main__":
    main()
