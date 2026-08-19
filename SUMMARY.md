# SUMMARY — AI Travel Analyst

---

## 1. Quick Recap

AI Travel Analyst is a three-layer data science pipeline built on a real Indian domestic flight price dataset. The **exploration layer** cleans the raw data and uses statistical analysis and visualisation to identify what actually drives prices — airline, duration, stops, route, and departure time emerge as the top drivers. The **modelling layer** engineers features from those findings and trains a Random Forest regression model that predicts a flight's price from its attributes, evaluated against a Linear Regression baseline using RMSE, MAE, and R². The **recommendation layer** uses that trained model as a scoring signal inside a transparent, weighted ranking formula: given a source city, destination, and a preference (cheapest / fastest / fewest stops / best overall value), it filters the flight catalogue, predicts prices for each candidate, normalises three signals (price, duration, stops), and returns a ranked shortlist with a score breakdown that shows exactly why each flight was ranked where it was. All three layers are wired into a Streamlit analytics dashboard that is designed as a functional internal tool — not a product landing page.

---

## 2. Architecture Walkthrough

| Stage | What happens | Output |
|-------|-------------|--------|
| **Clean** | Parse duration/time strings, handle nulls, drop duplicates, cap price outliers, derive `days_to_departure` proxy | `flights_clean.csv` |
| **EDA** | 7 visualisations (price distribution, by airline, by stops, vs DTD, by route, by month, correlation heatmap) | 7 PNGs in `outputs/plots/` |
| **Drivers** | Pearson/Spearman correlation, ANOVA on categoricals, diagnostic RF importance — converge on ranked list of price drivers | `price_drivers_summary.txt` |
| **Feature Engg** | One-hot encode low-cardinality categoricals; frequency-encode high-cardinality columns; bucket departure hour; derive month/DOW | `flights_features.csv`, `feature_columns.json` |
| **Model** | Linear Regression baseline + Random Forest (300 trees); evaluate on 20% hold-out with RMSE, MAE, R² | `model.pkl`, `metrics.json` |
| **Explainability** | SHAP TreeExplainer on RF (fallback to MDI if unavailable) — shows per-feature contribution direction and magnitude | SHAP plots |
| **Recommender** | Filter by route/date; predict prices; normalise 3 signals; weighted composite score per preference | Ranked DataFrame with score breakdown |
| **Dashboard** | Streamlit app: EDA tab, price drivers, model metrics, SHAP plots, recommender search, price predictor form | Interactive web app |

---

## 3. How to Explain This in an Interview

### "Walk me through your project."

> "I built a three-part flight price analysis system. First, I cleaned a real flight dataset and ran exploratory analysis to understand what drives prices — airline, route, stops, and duration came out as the top factors. Second, I engineered features from those findings and trained a Random Forest model to predict flight prices, which I benchmarked against a Linear Regression baseline using RMSE and R². Third, I layered a recommendation system on top: given a user's source, destination, and preference, it filters candidate flights, runs them through the price model, and ranks them using a transparent weighted score. Everything is wired into a Streamlit dashboard."

---

### Anticipated interview questions

**Q: Why Random Forest over XGBoost?**

> "For a first submission, RF is a better default: it's robust without careful hyperparameter tuning (no learning rate, no subsample), and handles our mixed feature types (one-hot + continuous) cleanly out of the box. XGBoost typically achieves lower RMSE on tabular data with proper tuning — it would be the next step in iteration. I made this call explicitly rather than just defaulting to XGBoost because the defensibility of the choice matters more than squeezing the last 2% of RMSE."

**Q: Why these features? What did you drop?**

> "I kept features that either had statistical evidence of correlation with price (Pearson/Spearman) or showed large group-mean spread via ANOVA — airline, stops, duration, route, departure hour, and month. I dropped raw string columns (Route, Additional_Info) that were too noisy to encode cleanly, and derived clean numeric proxies instead (duration_minutes, stops integer, dep_time_bucket). Days-to-departure is a proxy, not real booking lead time — I kept it because the relative ordering signal still carries information, but I document the limitation."

**Q: How does the recommender actually rank flights?**

> "It normalises three signals — predicted price, duration, and number of stops — each to a 0–1 scale where 0 is the best candidate on that dimension. Then it computes a weighted sum: for 'cheapest', that's 100% price weight; for 'best value', it's 40/35/25 split across price, duration, stops. The weights are documented and motivated — not arbitrary. The score breakdown is shown for each result, so the ranking is auditable."

**Q: Is this really ML or just rules for the recommendation part?**

> "It's both, honestly stated. The ML component is the price prediction model — using the model's output as a ranking signal is more principled than sorting on listed price alone. The ranking logic itself is rule-based and transparent by design: this dataset has no user interaction history, so collaborative filtering is not applicable. A content-based rule system is the correct tool here, and I say so explicitly in the code comments and README. If we had booking history per user, we could layer in collaborative filtering or a learning-to-rank model."

**Q: What would you improve?**

> "Three things: (1) real booking-date data to compute true days-to-departure — likely the biggest feature improvement; (2) XGBoost with a proper hyperparameter search; (3) live fare API integration to replace the static dataset with real-time prices for the recommender."

**Q: What was hardest?**

> "The data cleaning. Duration and time columns were stored as inconsistent strings ('2h 30m', '1h', '45m', '18:30 PM') requiring regex parsing before they could be used as features. The bigger challenge was the lack of a booking date: without it, true days-to-departure — a major pricing signal — can't be computed. I derived a proxy (relative journey date ordering within the dataset), documented the limitation explicitly, and calibrated expectations about how much predictive power that feature would carry."

**Q: What do RMSE and R² actually mean here?**

> "RMSE tells you the typical size of a prediction error in rupees — an RMSE of ₹2,000 means the model is on average off by roughly ₹2,000, with larger errors penalised quadratically. R² tells you what fraction of the price variance the model explains: 0.8 means 80% of the variation in prices across flights is captured by our features, 20% is noise or unmeasured factors (seat availability, last-minute promotions). MAE is the more intuitive version of RMSE — the plain average error without the squaring."

---

*Print this file before an interview and read it once. That's enough.*
