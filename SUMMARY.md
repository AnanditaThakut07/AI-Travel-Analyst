# SUMMARY — AI Travel Analyst

---

## 1. Quick Recap

AI Travel Analyst is a three-layer data science pipeline built on a real Indian domestic flight price dataset. The **exploration layer** cleans the raw data and uses statistical analysis and visualisation to identify what actually drives prices — airline, duration, stops, route, and departure time emerge as the top drivers. The **modelling layer** engineers features from those findings and trains a Random Forest regression model that predicts a flight's price from its attributes, evaluated against a Linear Regression baseline using RMSE, MAE, and R². The **recommendation layer** uses that trained model as a scoring signal inside a transparent, weighted ranking formula: given a source city, destination, and a preference (cheapest / fastest / fewest stops / best overall value), it filters the flight catalogue, predicts prices for each candidate, normalises three signals (price, duration, stops), and returns a ranked shortlist with a score breakdown that shows exactly why each flight was ranked where it was. All three layers are wired into a Streamlit analytics dashboard that is designed as a functional internal tool — not a product landing page.

---

## 2. Architecture Walkthrough

| Stage | What happens | Output |
|-------|-------------|--------|
| **Clean** | Parse mixed Duration strings ("Xh Ym" and float-hours), normalise Total_Stops ("non-stop"/"0"/"1 stop"), coerce all-object dtypes, cap Price outliers at ₹200,000. Drop 1,961 duplicates + 7,781 non-numeric prices → 90,258 clean rows | `flights_clean.csv` |
| **EDA** | 7 visualisations: price distribution (right-skewed, median ~₹35k), by airline (39 carriers, wide spread), by stops (non-monotonic), vs Days_Before_Departure (negative slope confirmed), by route, by month, correlation heatmap | 7 PNGs in `outputs/plots/` |
| **Drivers** | Pearson/Spearman show Distance_km (r=0.77) and duration_minutes (r=0.77) are top numeric drivers. ANOVA shows route_combined, Destination, Source, Airline as top categorical drivers. RF diagnostic: duration_minutes (0.62), Travel_Class (0.17), Distance_km (0.16) | `price_drivers_summary.txt` |
| **Feature Engg** | OHE: Travel_Class(4), Season(4), Weekday(7), Aircraft_Type(8), Booking_Channel(5), Source(54), Destination(54). Freq-encode: Airline(39), route_combined. dep_hour → 4 time buckets → 152 total features | `flights_features.csv` |
| **Model** | LR baseline: R²=0.788, RMSE=₹30,141, MAE=₹21,429. RF primary (300 trees, min_leaf=5): **R²=0.904, RMSE=₹20,264, MAE=₹11,044** — 32.8% RMSE improvement | `model.pkl`, `metrics.json` |
| **Explainability** | SHAP TreeExplainer on RF: duration_minutes and Distance_km dominate; Travel_Class_Business creates large positive SHAP values; Days_Before_Departure has negative slope (book early = lower price confirmed) | SHAP plots |
| **Recommender** | Filter by Source/Destination; predict prices via RF; normalise price/duration/stops to [0,1]; weighted composite score per preference (cheapest/fastest/fewest_stops/best_value) | Ranked DataFrame + score breakdown |
| **Dashboard** | Streamlit analytics tool: EDA tab, price drivers, model metrics, SHAP plots, recommender search, price predictor form | Interactive web app |

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
