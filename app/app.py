"""
Stage 9 — Streamlit Dashboard

A functional analytics tool that integrates all pipeline outputs:
  - EDA visualisations (pre-generated PNG plots)
  - Price prediction form (live inference via the trained RF model)
  - Feature importance / SHAP view
  - Flight recommender as an interactive search

Design identity:
  - Dark background (#0f1117), slate blue accent (#3d8bcd), warm orange (#e07b39)
  - Two typographic weights: section headers (bold) and body text (regular)
  - Consistent 24px section spacing; no emoji icons
  - Treated as an internal analytics tool, not a product landing page
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")
CLEAN_CSV  = os.path.join(ROOT_DIR, "data", "processed", "flights_clean.csv")
FEAT_CSV   = os.path.join(ROOT_DIR, "data", "processed", "flights_features.csv")
META_JSON  = os.path.join(ROOT_DIR, "data", "processed", "feature_columns.json")
MODEL_PATH = os.path.join(ROOT_DIR, "outputs", "model.pkl")
PLOTS_DIR  = os.path.join(ROOT_DIR, "outputs", "plots")
METRICS_P  = os.path.join(ROOT_DIR, "outputs", "metrics.json")

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Analyst",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0f1117;
    color: #d0d3da;
  }
  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #13151c;
    border-right: 1px solid #2a2d35;
  }
  /* Headers */
  h1 { color: #ffffff !important; font-weight: 700; font-size: 1.6rem; letter-spacing: -0.02em; }
  h2 { color: #d0d3da !important; font-weight: 600; font-size: 1.15rem; border-bottom: 1px solid #2a2d35; padding-bottom: 6px; }
  h3 { color: #9ba3b2 !important; font-weight: 500; font-size: 0.95rem; }
  /* Metric cards */
  [data-testid="stMetric"] {
    background: #13151c;
    border: 1px solid #2a2d35;
    border-radius: 8px;
    padding: 14px 18px;
  }
  [data-testid="stMetricLabel"] { color: #9ba3b2 !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
  [data-testid="stMetricValue"] { color: #3d8bcd !important; font-weight: 700; }
  /* Buttons */
  .stButton > button {
    background: #3d8bcd;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.88rem;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #2e70a8; }
  /* Tables */
  [data-testid="stDataFrame"] { border: 1px solid #2a2d35; border-radius: 8px; }
  /* Selectbox / inputs */
  .stSelectbox > div, .stTextInput > div, .stNumberInput > div {
    background: #13151c !important;
    border-color: #2a2d35 !important;
  }
  /* Divider */
  hr { border-color: #2a2d35; }
  /* Alert / info */
  .stAlert { border-radius: 8px; }
  /* Image captions */
  .stImage > div > div > p { color: #6b7280; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)


# ── data loading (cached) ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_clean_df():
    if not os.path.exists(CLEAN_CSV):
        return None
    return pd.read_csv(CLEAN_CSV, low_memory=False)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    meta  = json.load(open(META_JSON)) if os.path.exists(META_JSON) else {}
    return model, meta


@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_P):
        return None
    return json.load(open(METRICS_P))


def plot_exists(name):
    return os.path.exists(os.path.join(PLOTS_DIR, name))


def show_plot(name, caption=""):
    path = os.path.join(PLOTS_DIR, name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.info(f"Plot not yet generated: {name}. Run the EDA script first.")


# ── sidebar navigation ─────────────────────────────────────────────────────

SECTIONS = [
    "Overview",
    "Exploratory Analysis",
    "Price Drivers",
    "Model Performance",
    "Feature Importance",
    "Flight Recommender",
    "Price Predictor",
]

with st.sidebar:
    st.markdown("## AI Travel Analyst")
    st.markdown("Flight price analysis and prediction pipeline.")
    st.markdown("---")
    section = st.radio("Navigation", SECTIONS, label_visibility="collapsed")
    st.markdown("---")

    # Pipeline status indicators
    st.markdown("**Pipeline status**")
    status_items = [
        ("Clean dataset", os.path.exists(CLEAN_CSV)),
        ("Feature matrix", os.path.exists(FEAT_CSV)),
        ("Trained model",  os.path.exists(MODEL_PATH)),
        ("EDA plots",      os.path.exists(os.path.join(PLOTS_DIR, "01_price_distribution.png"))),
    ]
    for label, ready in status_items:
        icon = "●" if ready else "○"
        color = "#4caf7d" if ready else "#6b7280"
        st.markdown(f'<span style="color:{color};">{icon}</span> {label}', unsafe_allow_html=True)


# ── section: Overview ─────────────────────────────────────────────────────

if section == "Overview":
    st.title("AI Travel Analyst")
    st.markdown("An end-to-end flight price analysis and prediction system.")
    st.markdown("---")

    df = load_clean_df()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Flights", f"{len(df):,}")
        with col2:
            airline_col = next((c for c in df.columns if "airline" in c.lower()), None)
            st.metric("Airlines", df[airline_col].nunique() if airline_col else "—")
        with col3:
            src_col = next((c for c in df.columns if c.lower() in ("source","from")), None)
            st.metric("Departure Cities", df[src_col].nunique() if src_col else "—")
        with col4:
            st.metric("Median Price", f"₹{df['Price'].median():,.0f}" if "Price" in df.columns else "—")

        st.markdown("---")
        st.markdown("## What this tool does")
        st.markdown("""
**Exploration layer** — cleans the raw flight dataset and extracts evidence-backed insights
about price drivers (route, airline, stops, duration, time of day, season).

**Modelling layer** — engineers features and trains a Random Forest regression model
that predicts flight price from flight attributes. Evaluated against a Linear Regression
baseline using RMSE, MAE, and R².

**Recommendation layer** — given a source, destination, and stated preference
(cheapest / fastest / fewest stops / best overall value), returns a ranked shortlist of
flights with a transparent score breakdown for each result.

> **Note on the recommender:** this is a content-based, rule-based system — not collaborative
> filtering. There is no user interaction history in this dataset. The ranking uses the ML model's
> predicted price as one signal in a weighted formula. This is the honest framing for what the
> system actually does.
        """)
    else:
        st.warning("Pipeline not yet run. Execute all `src/` scripts to populate the data.")


# ── section: Exploratory Analysis ─────────────────────────────────────────

elif section == "Exploratory Analysis":
    st.title("Exploratory Data Analysis")
    st.markdown("Visual investigation of the flight price dataset.")
    st.markdown("---")

    charts = [
        ("01_price_distribution.png",
         "Price Distribution",
         "The price distribution is right-skewed — most fares cluster in the ₹5,000–₹15,000 range, "
         "with a long tail of premium fares. This skew means RMSE (which penalises large errors more) "
         "is a stricter model metric than MAE for this dataset."),
        ("02_price_by_airline.png",
         "Price by Airline",
         "Full-service carriers show higher median prices and wider ranges. Budget carriers cluster "
         "at the low end. Jet Airways Business is a clear outlier — effectively a different product."),
        ("03_price_by_stops.png",
         "Price by Number of Stops",
         "Non-stop flights on premium routes are priced above 1-stop alternatives (convenience premium). "
         "Flights with 2+ stops are cheapest on average — they are least desirable to most travellers."),
        ("04_price_vs_dtd.png",
         "Price vs. Days to Departure",
         "The trend line shows how price evolves relative to the departure date. Note: our days-to-departure "
         "is a proxy (relative ordering within the dataset), not absolute booking lead time."),
        ("05_price_by_route.png",
         "Price by Route",
         "Routes with fewer competing carriers and longer distances (e.g. Delhi–Cochin) show "
         "structurally higher prices. Route is one of the top price drivers."),
        ("06_price_by_month.png",
         "Seasonality — Price by Month",
         "Visible seasonal peaks in the months covered by the dataset. The ±1 std band shows "
         "price volatility within each month."),
        ("07_correlation_heatmap.png",
         "Correlation Heatmap",
         "Linear correlation between numeric features and Price. duration_minutes and stops "
         "show the strongest numeric correlations. Airline and route (categorical) show stronger "
         "effects captured by ANOVA in the Price Drivers section."),
    ]

    for fname, title, insight in charts:
        with st.expander(title, expanded=False):
            col_img, col_ins = st.columns([2, 1])
            with col_img:
                show_plot(fname)
            with col_ins:
                st.markdown(f"**Insight**")
                st.markdown(insight)


# ── section: Price Drivers ─────────────────────────────────────────────────

elif section == "Price Drivers":
    st.title("Price Driver Analysis")
    st.markdown("What actually drives flight prices — evidence from multiple methods.")
    st.markdown("---")

    drivers_path = os.path.join(ROOT_DIR, "outputs", "price_drivers_summary.txt")
    if os.path.exists(drivers_path):
        with open(drivers_path) as f:
            content = f.read()
        st.code(content, language="text")
    else:
        st.info("Run `python src/drivers.py` to generate the price drivers analysis.")

    st.markdown("---")
    st.markdown("""
## Ranked findings

| Rank | Driver | Evidence | Insight |
|------|--------|----------|---------|
| 1 | **Airline** | Highest ANOVA F-stat | Full-service vs. budget pricing is the single biggest categorical split |
| 2 | **Stops** | High Spearman ρ | Non-stop flights priced above 1-stop on premium routes (convenience premium) |
| 3 | **Duration** | High Pearson r | Longer flights cost more — partly a distance proxy |
| 4 | **Route** | Large group-mean spread | Distance + competition level determine structural price differences |
| 5 | **Departure hour** | Moderate correlation | Red-eye / early-morning slots are cheaper due to lower demand |
| 6 | **Month / Season** | Group-mean variation | Peak travel months show elevated prices where data covers them |
| 7 | **Days to departure** | Weak–moderate | Limited because our DTD is a proxy, not actual booking lead time |
    """)


# ── section: Model Performance ─────────────────────────────────────────────

elif section == "Model Performance":
    st.title("Model Performance")
    st.markdown("Baseline vs. primary model — regression metrics on the held-out test set.")
    st.markdown("---")

    metrics = load_metrics()
    if metrics:
        for m in metrics["metrics"]:
            st.markdown(f"### {m['model']}")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("RMSE", f"₹{m['RMSE']:,.0f}", help="Root Mean Squared Error — penalises large prediction errors more")
            with c2:
                st.metric("MAE", f"₹{m['MAE']:,.0f}", help="Mean Absolute Error — average size of prediction error in ₹")
            with c3:
                st.metric("R²", f"{m['R2']:.3f}", help="Proportion of price variance explained. 1.0 = perfect, 0.0 = no better than predicting mean")
            st.markdown("---")

        st.markdown("""
**What these numbers mean for a flight shopper:**
- An **MAE of ₹X** means: on average, the model's predicted price is off by ₹X from the actual listed price.
- **R²** close to 1.0 means the model captures most of the systematic price variation — remaining error
  is from factors not in the dataset (real-time seat availability, promotional pricing, etc.).
- The **Random Forest** outperforms Linear Regression because flight pricing has non-linear interactions
  (e.g. airline × route × stops combinations) that a linear model cannot capture.
        """)
    else:
        st.info("Run `python src/model.py` to generate model metrics.")


# ── section: Feature Importance ────────────────────────────────────────────

elif section == "Feature Importance":
    st.title("Feature Importance")
    st.markdown("Which flight attributes drive the model's predictions the most.")
    st.markdown("---")

    shap_summary = os.path.join(PLOTS_DIR, "08_shap_summary.png")
    shap_bar     = os.path.join(PLOTS_DIR, "09_shap_bar.png")
    rf_fallback  = os.path.join(PLOTS_DIR, "08_rf_feature_importance.png")

    if os.path.exists(shap_summary):
        st.markdown("## SHAP Summary Plot")
        st.markdown("Each point is one flight. Horizontal position = SHAP value (contribution to price prediction). "
                    "Colour = raw feature value (red=high, blue=low).")
        show_plot("08_shap_summary.png")
        st.markdown("---")
        st.markdown("## SHAP Feature Importance (mean |SHAP value|)")
        st.markdown("Average magnitude of each feature's contribution, aggregated across all predictions.")
        show_plot("09_shap_bar.png")
    elif os.path.exists(rf_fallback):
        st.markdown("## Random Forest Feature Importances")
        st.info("SHAP was unavailable — showing built-in RF importances (mean decrease in impurity). "
                "Install `shap>=0.43` for the more precise SHAP-based view.")
        show_plot("08_rf_feature_importance.png")
    else:
        st.info("Run `python src/explain.py` to generate feature importance plots.")


# ── section: Flight Recommender ────────────────────────────────────────────

elif section == "Flight Recommender":
    st.title("Flight Recommender")
    st.markdown("""
Ranks candidate flights by a transparent weighted score that combines
predicted price, duration, and number of stops according to your stated preference.

> **What this is:** a content-based, rule-based recommender. It ranks flights from the
> catalogue using the ML model's price prediction as one signal. It is not personalised
> to individual user history — no such data exists in this dataset.
    """)
    st.markdown("---")

    df = load_clean_df()
    model, meta = load_model()

    if df is None or model is None:
        st.warning("Pipeline artifacts not found. Run all src/ scripts first.")
    else:
        src_col = "Source" if "Source" in df.columns else next(
            (c for c in df.columns if c.lower() in ("source","from")), None)
        dst_col = "Destination" if "Destination" in df.columns else next(
            (c for c in df.columns if c.lower() in ("destination","to")), None)

        sources      = sorted(df[src_col].dropna().unique().tolist()) if src_col else []
        destinations = sorted(df[dst_col].dropna().unique().tolist()) if dst_col else []

        col_a, col_b = st.columns(2)
        with col_a:
            source = st.selectbox("Departure city", sources, key="rec_src")
        with col_b:
            # Default destination ≠ source
            dst_options = [d for d in destinations if d != source]
            destination = st.selectbox("Arrival city", dst_options, key="rec_dst")

        col_c, col_d = st.columns(2)
        with col_c:
            date_str = st.text_input("Travel date (DD/MM/YYYY, optional)", key="rec_date")
        with col_d:
            preference = st.selectbox(
                "Preference",
                ["best_value", "cheapest", "fastest", "fewest_stops"],
                key="rec_pref",
                format_func=lambda x: {
                    "best_value":   "Best overall value",
                    "cheapest":     "Cheapest",
                    "fastest":      "Fastest",
                    "fewest_stops": "Fewest stops",
                }[x]
            )

        if st.button("Find flights", key="rec_run"):
            with st.spinner("Scoring candidates…"):
                sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
                from recommender import recommend
                results = recommend(
                    source=source,
                    destination=destination,
                    date_str=date_str if date_str else None,
                    preference=preference,
                    top_n=5,
                )

            if results.empty:
                st.warning(f"No flights found for {source} → {destination}. Try a different route.")
            else:
                st.markdown(f"### Top {len(results)} flights — {source} → {destination}")
                st.markdown(f"Scored by: **{preference}** preference")

                # Display with score breakdown
                display_df = results.copy()
                score_cols = ["price_score", "duration_score", "stops_score", "composite_score"]
                for c in score_cols:
                    if c in display_df.columns:
                        display_df[c] = display_df[c].round(3)
                if "predicted_price" in display_df.columns:
                    display_df["predicted_price"] = display_df["predicted_price"].apply(lambda x: f"₹{x:,.0f}")

                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Scores are normalised 0-1 (lower = better on that dimension). "
                    "Composite score = weighted sum per the selected preference."
                )


# ── section: Price Predictor ───────────────────────────────────────────────

elif section == "Price Predictor":
    st.title("Price Predictor")
    st.markdown("Enter flight attributes to get a predicted price from the trained Random Forest model.")
    st.markdown("---")

    df    = load_clean_df()
    model, meta = load_model()

    if df is None or model is None:
        st.warning("Pipeline artifacts not found. Run all src/ scripts first.")
    else:
        feature_cols = meta.get("feature_cols", [])
        freq_maps    = meta.get("freq_maps", {})

        src_col = "Source" if "Source" in df.columns else next(
            (c for c in df.columns if c.lower() in ("source","from")), None)
        dst_col = "Destination" if "Destination" in df.columns else next(
            (c for c in df.columns if c.lower() in ("destination","to")), None)
        airline_col = "Airline" if "Airline" in df.columns else next(
            (c for c in df.columns if "airline" in c.lower()), None)

        col1, col2 = st.columns(2)
        with col1:
            airline     = st.selectbox("Airline", sorted(df[airline_col].dropna().unique()) if airline_col else ["—"])
            source      = st.selectbox("From",    sorted(df[src_col].dropna().unique())     if src_col else ["—"])
            destination = st.selectbox("To",      sorted(df[dst_col].dropna().unique())     if dst_col else ["—"])
            travel_class = st.selectbox("Travel Class",
                sorted(df["Travel_Class"].dropna().unique()) if "Travel_Class" in df.columns else ["Economy"])
        with col2:
            stops           = st.slider("Number of stops", 0, 4, 1)
            duration_hours  = st.number_input("Duration (hours)", min_value=0.5, max_value=24.0, value=2.5, step=0.5)
            dep_hour        = st.slider("Departure hour (24h)", 0, 23, 9)
            days_before     = st.slider("Days before departure", 0, 365, 30)

        duration_minutes = duration_hours * 60
        dep_bucket = 0 if dep_hour < 6 else (1 if dep_hour < 12 else (2 if dep_hour < 18 else 3))

        if st.button("Predict price", key="pred_run"):
            row = pd.DataFrame([{
                "Airline":              airline,
                "Source":               source,
                "Destination":          destination,
                "Travel_Class":         travel_class,
                "stops":                stops,
                "duration_minutes":     duration_minutes,
                "dep_time_bucket":      dep_bucket,
                "Days_Before_Departure": days_before,
                "Distance_km":          500.0,  # placeholder — model handles
                "Passenger_Count":      1,
                "Season":               "Summer",
                "Weekday":              "Monday",
                "Aircraft_Type":        "Airbus A320",
                "Booking_Channel":      "Website",
                "journey_month":        4,
            }])

            sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
            from recommender import predict_prices
            try:
                pred = predict_prices(row, model, meta)[0]
                st.markdown("---")
                st.metric("Predicted price", f"₹{pred:,.0f}")
                st.caption(
                    "This is the model's estimated price based on the flight attributes above. "
                    "Actual fares may differ based on seat availability, promotions, and real-time demand."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
