"""
Stage 1 — Data Acquisition & Inspection

Loads the flight dataset from data/raw/flights.csv, prints a comprehensive
schema summary, and saves the report to outputs/schema_summary.txt.

Why this comes first: before any cleaning or modelling we need to know
exactly what columns, data types, null patterns, and value distributions
are in the file. Every later stage adapts to what we find here.

Actual schema confirmed (100,000 rows × 18 columns):
  Flight_ID, Airline, Source, Destination, Departure_Date, Departure_Time,
  Arrival_Time, Duration, Total_Stops, Distance_km, Travel_Class,
  Days_Before_Departure, Season, Weekday, Aircraft_Type, Booking_Channel,
  Passenger_Count, Price
"""

import os
import sys
import pandas as pd

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")
RAW_CSV  = os.path.join(ROOT_DIR, "data", "raw", "flights.csv")
OUT_DIR  = os.path.join(ROOT_DIR, "outputs")


def load_dataset(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def inspect(df: pd.DataFrame) -> str:
    lines = []

    def h(text):
        lines.append("\n" + "=" * 60)
        lines.append(text)
        lines.append("=" * 60)

    h("SHAPE")
    lines.append(f"Rows: {df.shape[0]:,}    Columns: {df.shape[1]}")

    h("COLUMN NAMES")
    for i, col in enumerate(df.columns):
        lines.append(f"  [{i:02d}] {col}")

    h("DATA TYPES")
    lines.append(df.dtypes.to_string())

    h("NULL COUNTS (per column)")
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    null_df     = pd.DataFrame({"nulls": null_counts, "pct_%": null_pct})
    lines.append(null_df.to_string())

    h("DUPLICATE ROWS")
    n_dup = df.duplicated().sum()
    lines.append(f"{n_dup:,} duplicate rows ({n_dup/len(df)*100:.2f}%)")

    h("NUMERIC SUMMARY")
    # Coerce to numeric for describe
    num_df = df.apply(pd.to_numeric, errors="coerce")
    num_cols = num_df.dropna(axis=1, how="all").columns.tolist()
    if num_cols:
        lines.append(num_df[num_cols].describe().round(2).to_string())

    h("CATEGORICAL COLUMNS — unique value counts & top-5")
    for col in df.columns:
        n_unique = df[col].nunique()
        top_vals = df[col].value_counts().head(5).to_dict()
        lines.append(f"\n  {col!r}  ({n_unique} unique values)")
        lines.append(f"    top-5: {top_vals}")

    h("SAMPLE — first 5 rows")
    lines.append(df.head(5).to_string())

    return "\n".join(lines)


def main():
    if not os.path.exists(RAW_CSV):
        sys.exit(
            f"[ERROR] Dataset not found at {RAW_CSV}\n"
            "Please place flight_pricing_dataset.csv in data/raw/ as flights.csv"
        )

    print(f"[Stage 1] Loading: {RAW_CSV}")
    df = load_dataset(RAW_CSV)

    report = inspect(df)
    print(report)

    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "schema_summary.txt")
    with open(summary_path, "w") as f:
        f.write(report)
    print(f"\n[Stage 1] Schema summary saved to: {summary_path}")
    return df


if __name__ == "__main__":
    main()
