"""
Stage 1 — Data Acquisition & Inspection

Downloads the raw flight dataset from Google Drive (if not already present),
loads it, and prints a comprehensive schema summary to stdout and to
outputs/schema_summary.txt.

Why this comes first: before any cleaning or modelling, we need to know
exactly what columns, data types, null patterns, and value distributions
are actually in the file. Every later stage adapts to what we find here.
"""

import os
import sys
import textwrap
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
RAW_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")
GDRIVE_ID = "1tNUDxjXHzbRXe8CQdIoyJWh8OweGW0rR"

# Try both possible extensions — Drive exports are often xlsx
RAW_XLSX  = os.path.join(RAW_DIR, "flights.xlsx")
RAW_CSV   = os.path.join(RAW_DIR, "flights.csv")


def download_if_needed() -> str:
    """Download dataset from Google Drive using gdown; return local path."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_XLSX):
        print(f"[INFO] Dataset already present: {RAW_XLSX}")
        return RAW_XLSX
    if os.path.exists(RAW_CSV):
        print(f"[INFO] Dataset already present: {RAW_CSV}")
        return RAW_CSV

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        dest = RAW_XLSX
        print(f"[INFO] Downloading dataset from Google Drive …")
        gdown.download(url, dest, quiet=False, fuzzy=True)
        if not os.path.exists(dest):
            raise FileNotFoundError("gdown did not create the output file.")
        return dest
    except Exception as e:
        print(f"[ERROR] Auto-download failed: {e}")
        print(
            "[ACTION REQUIRED] Please download the dataset manually from:\n"
            "  https://drive.google.com/file/d/1tNUDxjXHzbRXe8CQdIoyJWh8OweGW0rR/view\n"
            "and place it at:  data/raw/flights.xlsx  (or flights.csv)\n"
            "Then re-run this script."
        )
        sys.exit(1)


def load_dataset(path: str) -> pd.DataFrame:
    """Load xlsx or csv into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> str:
    """Build a detailed schema/inspection report as a string."""
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
    null_df     = pd.DataFrame({"nulls": null_counts, "pct": null_pct})
    lines.append(null_df.to_string())

    h("DUPLICATE ROWS")
    n_dup = df.duplicated().sum()
    lines.append(f"{n_dup:,} duplicate rows ({n_dup/len(df)*100:.2f}%)")

    h("NUMERIC SUMMARY (describe)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        lines.append(df[num_cols].describe().round(2).to_string())
    else:
        lines.append("No numeric columns detected.")

    h("CATEGORICAL COLUMNS — unique value counts")
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_vals = df[col].value_counts().head(5).to_dict()
        lines.append(f"\n  {col!r}  ({n_unique} unique values)")
        lines.append(f"    top-5: {top_vals}")

    h("SAMPLE — first 5 rows")
    lines.append(df.head(5).to_string())

    return "\n".join(lines)


def main():
    path = download_if_needed()
    print(f"[INFO] Loading: {path}")
    df   = load_dataset(path)

    report = inspect(df)
    print(report)

    os.makedirs(OUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUT_DIR, "schema_summary.txt")
    with open(summary_path, "w") as f:
        f.write(report)
    print(f"\n[INFO] Schema summary saved to: {summary_path}")

    # Expose df shape as a sanity check for downstream scripts
    return df


if __name__ == "__main__":
    main()
