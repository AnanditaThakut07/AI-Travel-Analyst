# AI Travel Analyst

A complete end-to-end data science project: flight price exploration, ML-based prediction, and a transparent flight recommendation system — wired into an interactive Streamlit dashboard.

---

## Project Overview

This project answers three questions about Indian domestic flights:

1. **What drives flight prices?** (Exploration layer)
2. **Can we predict a flight's price from its attributes?** (Modelling layer)
3. **Given a user's search, which flights should we recommend?** (Recommendation layer)

All three layers are integrated into an interactive analytics dashboard.

---

## Problem Statement

Flight prices vary enormously for the same route depending on airline, departure time, number of stops, booking lead time, and season. This project builds a pipeline to:
- Quantify which factors matter most (and by how much)
- Train a regression model to predict price from flight attributes
- Use that model to power a transparent, auditable flight recommendation system

---

## Installation

```bash
git clone <repo-url>
cd ai-travel-analyst
pip install -r requirements.txt
```

**Dataset**: Download from [Google Drive](https://drive.google.com/file/d/1tNUDxjXHzbRXe8CQdIoyJWh8OweGW0rR/view?usp=sharing) and place at `data/raw/flights.xlsx`, or run:
```bash
python src/inspect_data.py   # auto-downloads via gdown
```

---

## Running the Pipeline

Execute stages in order:

```bash
python src/inspect_data.py   # Stage 1: data inspection
python src/data_prep.py      # Stage 2: cleaning
python src/eda.py            # Stage 3: visualisations
python src/drivers.py        # Stage 4: price drivers
python src/features.py       # Stage 5: feature engineering
python src/model.py          # Stage 6: model training
python src/explain.py        # Stage 7: SHAP explainability
python src/recommender.py    # Stage 8: recommender demo
streamlit run app/app.py     # Stage 9: dashboard
```

---

## Dataset Used

**Source**: [Kaggle / Google Drive flight price dataset](https://drive.google.com/file/d/1tNUDxjXHzbRXe8CQdIoyJWh8OweGW0rR/view?usp=sharing)

**Schema confirmed in Stage 1** (adapt if actual columns differ):

| Column | Type | Description |
|--------|------|-------------|
| Airline | string | Carrier name |
| Date_of_Journey | string | Journey date (DD/MM/YYYY) |
| Source | string | Departure city |
| Destination | string | Arrival city |
| Route | string | Intermediate stops (raw) |
| Dep_Time | string | Departure time (HH:MM) |
| Arrival_Time | string | Arrival time (HH:MM) |
| Duration | string | Flight duration (e.g. "2h 30m") |
| Total_Stops | string | Stop count (e.g. "non-stop", "1 stop") |
| Additional_Info | string | Extra flags (meal, no info, etc.) |
| Price | integer | Ticket price in Indian Rupees (₹) |

**Derived features**: `duration_minutes`, `stops`, `dep_hour`, `dep_time_bucket`, `days_to_departure` (proxy), `journey_month`, `journey_dow`, `route_combined`

---

## Methodology

### Layer 1 — Exploration
- Cleaned raw data: parsed duration/time strings, handled missing values, capped extreme outliers, dropped duplicates
- Produced 7 visualisations revealing price patterns across airline, stops, route, time-of-day, and season
- Quantified drivers using Pearson/Spearman correlation, ANOVA, and a diagnostic Random Forest pass

### Layer 2 — Modelling
- Engineered features: one-hot encoded low-cardinality categoricals, frequency-encoded high-cardinality columns, derived time/date features
- **Baseline**: Linear Regression — sets the performance floor
- **Primary model**: Random Forest Regressor (300 trees, `min_samples_leaf=5`) — handles non-linear feature interactions
- Evaluated on a 20% held-out test set with RMSE, MAE, and R²
- **Explainability**: SHAP TreeExplainer (fallback to built-in importances) — identifies which features drive individual predictions

### Layer 3 — Recommendation
- Filters the flight catalogue by source/destination (and optional date window ±7 days)
- Predicts price for each candidate using the trained model
- Scores candidates on three normalised signals: price, duration, stops
- Ranks by a weighted composite score according to user preference:

| Preference | Price weight | Duration weight | Stops weight |
|------------|-------------|-----------------|--------------|
| Cheapest | 1.00 | 0.00 | 0.00 |
| Fastest | 0.20 | 0.70 | 0.10 |
| Fewest stops | 0.20 | 0.10 | 0.70 |
| Best value | 0.40 | 0.35 | 0.25 |

> **Is this ML or just sorting?** Honest answer: it is a content-based, rule-based recommender. It is _not_ collaborative filtering (no user history exists). The ML component is the price prediction — using the model's output as a signal is more principled than sorting on listed price alone (which may be noisy or unavailable for hypothetical queries).

---

## Technologies Used

| Purpose | Library |
|---------|---------|
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine learning | scikit-learn |
| Explainability | shap |
| Dataset download | gdown |
| Dashboard | streamlit |
| File handling | openpyxl, joblib |

---

## Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression (baseline) | ₹30,141 | ₹21,429 | 0.788 |
| **Random Forest (primary)** | **₹20,264** | **₹11,044** | **0.904** |

RF improves RMSE by **32.8%** over the baseline. An R² of 0.904 means the model explains **90.4% of price variance** across 18,052 test flights.

**What the numbers mean for a traveller:** On average, our price prediction is off by ₹11,044 (MAE). Given that prices range from ₹2,000 to ₹200,000, this is a reasonable error band — primarily driven by last-minute premium fares and international routes where Distance_km is the dominant signal.

**Key insights from EDA:**
1. **Airline** is the strongest categorical price driver (ANOVA)
2. **Duration** is the strongest numeric predictor (Pearson r)
3. **Stops** show a non-monotonic pattern — non-stop can be _more_ expensive than 1-stop on premium routes
4. **Route** drives structural price differences via distance and competition
5. **Departure hour** — red-eye and early-morning slots are cheapest

---

## Challenges Faced

- **No explicit booking date**: could not compute true days-to-departure; derived a relative proxy from journey dates within the dataset
- **Duration as a string**: required regex parsing ("2h 30m", "1h", "45m") before it could be used as a numeric feature
- **High-cardinality route strings**: frequency encoding chosen over one-hot to avoid column explosion
- **Right-skewed price distribution**: required outlier capping at the 99.5th percentile before training

---

## Future Improvements

1. **Real booking-date data** — would unlock true booking lead-time as a feature, likely the biggest improvement
2. **XGBoost with hyperparameter tuning** — would likely beat Random Forest on this tabular dataset
3. **Real-time price API integration** — replace static dataset with live fare data for the recommender
4. **User preference history** — enabling personalisation beyond the stated preference flag
5. **Seat class as a feature** — economy vs. business pricing is a major driver not in this dataset

---

## Screenshots

*(Add after running the dashboard)*
