# Last-mile Optimization (Streamlit) — Demo App

This small Streamlit app lets you upload:
- a deliveries CSV (`Delivery_ID, Distance_km, Delivery_Time_Minutes, Delivery_Cost_INR` recommended)
- a `joblib` model artifact (the file `last_mile_model_artifacts.joblib`)

The app will load the model, predict per-delivery ETA (minutes), show aggregate Before/After totals (time, cost, fuel saving heuristics), and let you download results.

## How to deploy (Streamlit Cloud)

1. Create a GitHub repository and push the files from this repo.
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud) with your GitHub account.
3. Click **New app**, select the repository and branch, and deploy.
4. Upload your `gati_extracted_from_xlsx.csv` and `last_mile_model_artifacts.joblib` through the app UI.

## How to run locally

1. Install Python 3.8+
2. Create a virtualenv and install requirements:
```
pip install -r requirements.txt
```
3. Run the app:
```
streamlit run app.py
```

## Notes & next steps
- The app assumes the model artifact is a `joblib` dump with either the model object directly or a dict containing keys `model`, `features`, and `categorical_mappings` (this is what I produced earlier).
- For production, hook the app to an `/api/optimize` endpoint which computes route-level optimized sequences using OR-Tools or another solver.