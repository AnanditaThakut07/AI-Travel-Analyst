"""
Stage 6 — Model Building & Evaluation

Trains two models on the engineered feature matrix:
  1. Linear Regression (baseline) — chosen because it's the simplest
     interpretable model and sets the floor for what a "good" model must beat.
  2. Random Forest Regressor (primary) — chosen over XGBoost because:
       a. RF is less sensitive to hyperparameter tuning; it works well
          out-of-the-box, which matters for a first submission.
       b. RF handles mixed feature types (OHE + continuous) cleanly without
          requiring careful learning-rate tuning.
       c. XGBoost would be the next step in a production pipeline (better
          on tabular data at scale with proper tuning) — this is noted below.
     Alternative to RF: XGBoost. It typically achieves lower RMSE on
     structured tabular data, but requires tuning n_estimators, max_depth,
     learning_rate, and subsample to avoid overfitting. Given the dataset
     size and the goal of a defensible first model, RF is preferred.

Evaluation metrics (regression only):
  - RMSE (Root Mean Squared Error): penalises large errors quadratically.
    In flight-price terms, an RMSE of ₹X means on average our predictions
    are off by roughly ₹X, but large outliers are penalised more.
  - MAE (Mean Absolute Error): average absolute error. Easier to interpret:
    "on average, our price prediction is off by ₹MAE."
  - R² (coefficient of determination): proportion of price variance explained
    by the model. R²=1.0 is perfect; R²=0.0 means the model is no better than
    predicting the mean price for every flight.

All metrics and the trained model are saved to outputs/.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

warnings.filterwarnings("ignore")

ROOT_DIR     = os.path.join(os.path.dirname(__file__), "..")
FEAT_CSV     = os.path.join(ROOT_DIR, "data", "processed", "flights_features.csv")
META_JSON    = os.path.join(ROOT_DIR, "data", "processed", "feature_columns.json")
OUTPUTS_DIR  = os.path.join(ROOT_DIR, "outputs")
MODEL_PATH   = os.path.join(OUTPUTS_DIR, "model.pkl")
SCALER_PATH  = os.path.join(OUTPUTS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")

TEST_SIZE   = 0.2
RANDOM_SEED = 42


def compute_metrics(y_true, y_pred, label: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n  [{label}]")
    print(f"    RMSE : ₹{rmse:,.2f}")
    print(f"    MAE  : ₹{mae:,.2f}")
    print(f"    R²   : {r2:.4f}")
    return {"model": label, "RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}


def main():
    if not os.path.exists(FEAT_CSV):
        sys.exit(f"[ERROR] {FEAT_CSV} not found. Run src/features.py first.")

    df = pd.read_csv(FEAT_CSV, low_memory=False)
    print(f"[Stage 6] Loaded feature matrix: {df.shape}")

    feature_cols = [c for c in df.columns if c != "Price"]
    X = df[feature_cols].values
    y = df["Price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"[Stage 6] Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    all_metrics = []

    # ── Baseline: Linear Regression (requires scaling) ───────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    all_metrics.append(compute_metrics(y_test, lr_preds, "Linear Regression (baseline)"))

    # ── Primary: Random Forest ────────────────────────────────────────────
    # n_estimators=300: enough trees to stabilise variance without being slow.
    # max_depth=None: let trees grow fully; RF's bagging controls overfitting.
    # min_samples_leaf=5: prevents individual trees from memorising single rows.
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rf.fit(X_train, y_train)  # RF does not need scaling
    rf_preds = rf.predict(X_test)
    all_metrics.append(compute_metrics(y_test, rf_preds, "Random Forest"))

    # ── Improvement summary ───────────────────────────────────────────────
    lr_rmse = all_metrics[0]["RMSE"]
    rf_rmse = all_metrics[1]["RMSE"]
    print(f"\n[Stage 6] RF improves RMSE by {(lr_rmse - rf_rmse)/lr_rmse*100:.1f}% over baseline.")

    # ── Save artifacts ────────────────────────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Stage 6] Model saved:  {MODEL_PATH}")
    print(f"[Stage 6] Scaler saved: {SCALER_PATH}")

    # Save feature column list with the model for inference
    with open(META_JSON) as f:
        meta = json.load(f)
    meta["feature_cols"] = feature_cols
    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump({"metrics": all_metrics, "test_size": TEST_SIZE, "random_seed": RANDOM_SEED}, f, indent=2)
    print(f"[Stage 6] Metrics saved: {METRICS_PATH}")

    return rf, feature_cols


if __name__ == "__main__":
    main()
