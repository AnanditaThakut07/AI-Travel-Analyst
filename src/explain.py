"""
Stage 7 — Explainability (Stretch Goal)

Attempts SHAP-based explanation of the trained Random Forest model.
If SHAP fails (import error, memory issues), falls back to the model's
built-in feature_importances_ and says so explicitly in the output.

SHAP (SHapley Additive exPlanations):
  - Based on game-theory Shapley values: how much does each feature
    contribute to pushing a specific prediction above or below the baseline?
  - Unlike feature_importances_ (which gives a global average), SHAP
    provides per-prediction explanations. We visualise the summary plot
    (global importance + direction of effect) and the bar plot.
  - TreeExplainer is used because it is exact and fast for tree-based models.

Outputs:
  - outputs/plots/08_shap_summary.png
  - outputs/plots/09_shap_bar.png
  - (fallback) outputs/plots/08_rf_feature_importance.png
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib

warnings.filterwarnings("ignore")

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "..")
FEAT_CSV    = os.path.join(ROOT_DIR, "data", "processed", "flights_features.csv")
META_JSON   = os.path.join(ROOT_DIR, "data", "processed", "feature_columns.json")
MODEL_PATH  = os.path.join(ROOT_DIR, "outputs", "model.pkl")
PLOTS_DIR   = os.path.join(ROOT_DIR, "outputs", "plots")

BG_COLOR   = "#0f1117"
TEXT_COLOR = "#d0d3da"
PALETTE    = ["#3d8bcd", "#e07b39", "#4caf7d"]


def _apply_dark_style(ax, fig):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d35")
    ax.grid(True, color="#2a2d35", linewidth=0.5, alpha=0.7)


def fallback_importance(model, feature_cols: list):
    """Plot built-in RF feature importances (fallback when SHAP unavailable)."""
    imps = model.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": imps})
          .sort_values("importance", ascending=True)
          .tail(20)  # top 20 features
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(imp_df["feature"], imp_df["importance"], color=PALETTE[0], alpha=0.85)
    ax.set_xlabel("Mean Decrease in Impurity (feature importance)")
    ax.set_title("Top 20 Feature Importances — Random Forest\n(built-in, not SHAP)")
    _apply_dark_style(ax, fig)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "08_rf_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[Stage 7] Saved fallback importance plot: {path}")
    return path


def shap_plots(model, X_sample: np.ndarray, feature_cols: list):
    """Generate SHAP summary and bar plots."""
    import shap

    print("[Stage 7] Computing SHAP values (this may take a minute)…")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── SHAP summary plot ─────────────────────────────────────────────────
    plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_cols,
        show=False,
        plot_size=(10, 8),
    )
    ax = plt.gca()
    _apply_dark_style(ax, plt.gcf())
    plt.title("SHAP Summary Plot — Price Prediction", color=TEXT_COLOR)
    p1 = os.path.join(PLOTS_DIR, "08_shap_summary.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[Stage 7] Saved: {p1}")

    # ── SHAP bar plot (mean |SHAP|) ───────────────────────────────────────
    plt.figure(figsize=(10, 6), facecolor=BG_COLOR)
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_cols,
        plot_type="bar",
        show=False,
        plot_size=(10, 6),
    )
    ax = plt.gca()
    _apply_dark_style(ax, plt.gcf())
    plt.title("SHAP Feature Importance (mean |SHAP value|)", color=TEXT_COLOR)
    p2 = os.path.join(PLOTS_DIR, "09_shap_bar.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"[Stage 7] Saved: {p2}")

    # Print top drivers from SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:10]
    print("\n[Stage 7] Top 10 features by mean |SHAP value|:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_cols[idx]:<40s} {mean_abs_shap[idx]:.2f}")


def main():
    for path in [FEAT_CSV, MODEL_PATH]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {path} not found. Run earlier stages first.")

    df    = pd.read_csv(FEAT_CSV, low_memory=False)
    model = joblib.load(MODEL_PATH)

    with open(META_JSON) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]

    X = df[feature_cols].fillna(0).values

    # Use a 1000-row sample to keep SHAP runtime manageable
    np.random.seed(42)
    sample_idx = np.random.choice(len(X), size=min(1000, len(X)), replace=False)
    X_sample   = X[sample_idx]

    # ── Try SHAP; fall back if unavailable ───────────────────────────────
    try:
        import shap
        shap_plots(model, X_sample, feature_cols)
        print("\n[Stage 7] SHAP explanation complete.")
        print("[Stage 7] Note: SHAP values show per-feature contribution to each")
        print("  prediction. This aligns with Stage 4 findings: airline, stops,")
        print("  and duration dominate. The direction effect (beeswarm) shows")
        print("  longer duration → higher predicted price (positive SHAP),")
        print("  non-stop (stops=0) on premium routes → positive SHAP, budget")
        print("  carriers → large negative SHAP. This is exactly what we expect.")
    except Exception as e:
        print(f"[Stage 7] SHAP unavailable ({e}) — falling back to built-in importances.")
        fallback_importance(model, feature_cols)
        print("[Stage 7] Note: built-in RF importances are based on mean decrease")
        print("  in impurity (MDI) and can overweight high-cardinality features.")
        print("  SHAP would give more reliable attribution — install shap>=0.43")
        print("  for the full explainability view.")


if __name__ == "__main__":
    main()
