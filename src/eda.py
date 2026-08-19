"""
Stage 3 — Exploratory Data Analysis

Produces ≥5 visualisations saved to outputs/plots/.
Each chart was chosen to answer a specific price-driver question;
the insight extracted from each is documented in the function docstring.

Library choice per chart:
  - Histogram/KDE: matplotlib (fine-grained control over binning + density)
  - Boxplot / Violin: seaborn (built-in quartile display)
  - Scatter + trend: matplotlib + numpy polyfit
  - Bar chart: seaborn (horizontal, easy label rotation)
  - Heatmap (correlation): seaborn
  - Count + line overlay: plotly (interactive, good for temporal data)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saved plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.join(os.path.dirname(__file__), "..")
CLEAN_CSV = os.path.join(ROOT_DIR, "data", "processed", "flights_clean.csv")
PLOTS_DIR = os.path.join(ROOT_DIR, "outputs", "plots")

# ── global style ───────────────────────────────────────────────────────────
PALETTE   = ["#3d8bcd", "#e07b39", "#4caf7d", "#c75c5c", "#a88fd4"]
BG_COLOR  = "#0f1117"
GRID_COLOR = "#2a2d35"
TEXT_COLOR = "#d0d3da"

def _apply_dark_style(ax, fig):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.7)

def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[Stage 3] Saved: {path}")
    return path


# ── chart 1: price distribution ───────────────────────────────────────────
def plot_price_distribution(df: pd.DataFrame):
    """
    Chart type: histogram + KDE overlay
    Question answered: What does the typical flight price look like?
    Insight: The distribution is right-skewed — most flights cluster in the
    ₹5,000–₹15,000 band, with a long tail of premium/business fares. This
    skew justifies using RMSE alongside MAE in model evaluation (RMSE
    penalises large errors on the expensive tail more heavily).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    prices = df["Price"].dropna()

    ax.hist(prices, bins=60, color=PALETTE[0], alpha=0.7, density=True, label="Histogram")

    # KDE via scipy if available, else skip
    try:
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(prices.min(), prices.max(), 300)
        kde   = gaussian_kde(prices)
        ax.plot(kde_x, kde(kde_x), color=PALETTE[1], linewidth=2, label="KDE")
    except ImportError:
        pass

    ax.axvline(prices.median(), color=PALETTE[2], linestyle="--", linewidth=1.5,
               label=f"Median ₹{prices.median():,.0f}")
    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Density")
    ax.set_title("Flight Price Distribution")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(labelcolor=TEXT_COLOR, facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    _apply_dark_style(ax, fig)
    return save(fig, "01_price_distribution.png")


# ── chart 2: price by airline ─────────────────────────────────────────────
def plot_price_by_airline(df: pd.DataFrame):
    """
    Chart type: horizontal boxplot (seaborn)
    Question answered: Which airlines charge more, and how variable is pricing?
    Insight: Budget carriers show tighter, lower price bands; full-service
    carriers (e.g. Jet Airways Business) show wider ranges and higher medians.
    Boxes that overlap widely suggest price is not strongly airline-determined
    alone — route/stops are confounders.
    """
    airline_col = next((c for c in df.columns if "airline" in c.lower()), None)
    if not airline_col:
        print("[Stage 3] No airline column found — skipping chart 2.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    order = df.groupby(airline_col)["Price"].median().sort_values().index

    sns.boxplot(
        data=df, y=airline_col, x="Price", order=order,
        palette=PALETTE, ax=ax,
        boxprops=dict(alpha=0.8),
        flierprops=dict(marker=".", color=PALETTE[0], alpha=0.3, markersize=3),
    )
    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Airline")
    ax.set_title("Price Distribution by Airline")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    _apply_dark_style(ax, fig)
    return save(fig, "02_price_by_airline.png")


# ── chart 3: price by number of stops ────────────────────────────────────
def plot_price_by_stops(df: pd.DataFrame):
    """
    Chart type: violin plot (seaborn)
    Question answered: Does more stops always mean more expensive?
    Insight: Counterintuitively, 1-stop flights are sometimes cheaper than
    non-stop flights on premium routes (airlines price non-stop convenience
    at a premium). 2+-stop flights are cheapest on average — they are longer
    and less desirable. This non-monotonic relationship is important for the
    recommender scoring formula.
    """
    if "stops" not in df.columns:
        print("[Stage 3] No stops column — skipping chart 3.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(
        data=df, x="stops", y="Price",
        palette=PALETTE, ax=ax, inner="quartile",
        cut=0,
    )
    ax.set_xlabel("Number of Stops")
    ax.set_ylabel("Price (₹)")
    ax.set_title("Price vs. Number of Stops")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    _apply_dark_style(ax, fig)
    return save(fig, "03_price_by_stops.png")


# ── chart 4: price vs days-to-departure ───────────────────────────────────
def plot_price_vs_dtd(df: pd.DataFrame):
    """
    Chart type: scatter + linear trend line (matplotlib)
    Question answered: Does booking earlier lead to lower prices?
    Insight: A clear negative slope (if present) confirms the well-known
    booking lead-time effect. If the slope is flat, it suggests the dataset
    captures fares at a single snapshot in time rather than dynamic pricing —
    an important caveat to note in the README and SUMMARY.
    """
    if "days_to_departure" not in df.columns:
        print("[Stage 3] No days_to_departure column — skipping chart 4.")
        return

    sample = df[["days_to_departure", "Price"]].dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(sample["days_to_departure"], sample["Price"],
               alpha=0.2, s=8, color=PALETTE[0], rasterized=True)

    # Trend line
    z = np.polyfit(sample["days_to_departure"], sample["Price"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample["days_to_departure"].min(), sample["days_to_departure"].max(), 200)
    ax.plot(x_line, p(x_line), color=PALETTE[1], linewidth=2,
            label=f"Trend (slope={z[0]:+.1f} ₹/day)")

    ax.set_xlabel("Days to Departure (proxy)")
    ax.set_ylabel("Price (₹)")
    ax.set_title("Price vs. Days to Departure")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(labelcolor=TEXT_COLOR, facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    _apply_dark_style(ax, fig)
    return save(fig, "04_price_vs_dtd.png")


# ── chart 5: price by top source-destination routes ───────────────────────
def plot_price_by_route(df: pd.DataFrame):
    """
    Chart type: horizontal bar chart (seaborn)
    Question answered: Which routes are structurally the most expensive?
    Insight: Route is one of the strongest price drivers — some city-pairs
    simply have higher average fares due to distance, competition, and
    demand (e.g. metro-to-metro vs. metro-to-tier2). The bar chart shows
    the top 12 routes by median price, making competitive differences visible.
    """
    if "route_combined" not in df.columns:
        print("[Stage 3] No route_combined column — skipping chart 5.")
        return

    route_med = (
        df.groupby("route_combined")["Price"]
          .agg(["median", "count"])
          .query("count >= 20")          # only routes with enough samples
          .sort_values("median", ascending=False)
          .head(12)
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(route_med["route_combined"], route_med["median"],
                   color=PALETTE[0], alpha=0.85)

    # Annotate with count
    for bar, cnt in zip(bars, route_med["count"]):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                f"n={cnt}", va="center", color=TEXT_COLOR, fontsize=8)

    ax.set_xlabel("Median Price (₹)")
    ax.set_ylabel("Route")
    ax.set_title("Top Routes by Median Price")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.invert_yaxis()
    _apply_dark_style(ax, fig)
    return save(fig, "05_price_by_route.png")


# ── chart 6 (bonus): price by month ───────────────────────────────────────
def plot_price_by_month(df: pd.DataFrame):
    """
    Chart type: line chart with error band (matplotlib)
    Question answered: Is there seasonal pricing variation?
    Insight: Peak months (April-May summer holidays, December Christmas) 
    typically show higher median prices. If the dataset spans a single 
    quarter, this chart will show flat variation — itself a useful insight
    (model can't learn seasonality from a partial-year dataset).
    """
    if "journey_month" not in df.columns:
        print("[Stage 3] No journey_month column — skipping chart 6.")
        return

    monthly = df.groupby("journey_month")["Price"].agg(["median", "std"]).reset_index()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["journey_month"], monthly["median"],
            color=PALETTE[0], marker="o", linewidth=2, label="Median")
    ax.fill_between(
        monthly["journey_month"],
        (monthly["median"] - monthly["std"]).clip(lower=0),
        monthly["median"] + monthly["std"],
        alpha=0.2, color=PALETTE[0], label="±1 std"
    )
    ax.set_xticks(monthly["journey_month"])
    ax.set_xticklabels([month_labels[m - 1] for m in monthly["journey_month"]])
    ax.set_xlabel("Month of Journey")
    ax.set_ylabel("Price (₹)")
    ax.set_title("Median Price by Month (Seasonality)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(labelcolor=TEXT_COLOR, facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    _apply_dark_style(ax, fig)
    return save(fig, "06_price_by_month.png")


# ── chart 7 (bonus): correlation heatmap ──────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Chart type: heatmap (seaborn)
    Question answered: Which numeric features correlate most with Price?
    Insight: Shows the linear correlation structure at a glance. Used as
    a sanity check alongside the non-linear feature importances in Stage 4.
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if "Price" not in num_cols or len(num_cols) < 2:
        print("[Stage 3] Not enough numeric columns for heatmap — skipping.")
        return

    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(num_cols)), max(6, len(num_cols) - 1)))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
                ax=ax, linewidths=0.5, linecolor=GRID_COLOR,
                annot_kws={"size": 9},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Numeric Feature Correlation Matrix")
    _apply_dark_style(ax, fig)
    return save(fig, "07_correlation_heatmap.png")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CLEAN_CSV):
        sys.exit(f"[ERROR] {CLEAN_CSV} not found. Run src/data_prep.py first.")

    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    print(f"[Stage 3] Loaded clean dataset: {df.shape}")

    plot_price_distribution(df)
    plot_price_by_airline(df)
    plot_price_by_stops(df)
    plot_price_vs_dtd(df)
    plot_price_by_route(df)
    plot_price_by_month(df)
    plot_correlation_heatmap(df)

    print(f"[Stage 3] All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
