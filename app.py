import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO, BytesIO
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Last-mile Optimization (Before/After)", layout="wide")

st.title("Last-mile Optimization — Before / After (Streamlit)")
st.markdown("""
Upload:
1. A deliveries CSV (columns: Delivery_ID, Distance_km, Delivery_Time_Minutes, Delivery_Cost_INR are recommended).  
2. The trained model artifact (`last_mile_model_artifacts.joblib`) produced earlier.
""")

with st.expander("Sample CSV format"):
    st.code("Delivery_ID,Distance_km,Delivery_Time_Minutes,Delivery_Cost_INR\nDEL1,10.2,45,120\nDEL2,2.5,18,40\n...")

col1, col2 = st.columns(2)

with col1:
    uploaded_csv = st.file_uploader("Upload deliveries CSV", type=["csv"])
with col2:
    uploaded_model = st.file_uploader("Upload joblib model artifact", type=["joblib","pkl"])

if uploaded_csv is not None:
    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error("Failed to read CSV: " + str(e))
        st.stop()
    st.success(f"Loaded CSV with {len(df)} rows")
    st.dataframe(df.head(10))

    # Basic checks and fill defaults
    if 'Delivery_Time_Minutes' not in df.columns:
        st.warning("Column 'Delivery_Time_Minutes' not found — app will still run but baseline ETA will be assumed from a 'Baseline_ETA' column or 0.")
    if 'Distance_km' not in df.columns:
        st.warning("Column 'Distance_km' not found — distance-based estimations will be limited.")

else:
    st.info("Please upload a CSV to proceed. You can use the sample format above.")

if uploaded_csv is not None and uploaded_model is not None:
    # Load model artifact
    try:
        model_art = joblib.load(uploaded_model)
    except Exception as e:
        st.error("Failed to load model artifact: " + str(e))
        st.stop()

    st.write("Model artifact loaded. Keys (if dict):", getattr(model_art, 'keys', lambda: [])())

    # Prepare features similar to training: we attempt to use available columns
    df_proc = df.copy()

    # derive hour/day/month if Date present
    if 'Date' in df_proc.columns:
        try:
            df_proc['Date'] = pd.to_datetime(df_proc['Date'], errors='coerce')
            df_proc['hour'] = df_proc['Date'].dt.hour.fillna(12).astype(int)
            df_proc['dayofweek'] = df_proc['Date'].dt.dayofweek
            df_proc['month'] = df_proc['Date'].dt.month
        except:
            pass

    # load features list from artifact if present
    features = None
    if isinstance(model_art, dict) and 'features' in model_art:
        features = model_art['features']

    # Basic heuristic: if features missing, use distance/hour/day/month
    if not features:
        candidate = []
        if 'Distance_km' in df_proc.columns: candidate.append('Distance_km')
        if 'Package_Weight_kg' in df_proc.columns: candidate.append('Package_Weight_kg')
        if 'Num_Packages_in_Route' in df_proc.columns: candidate.append('Num_Packages_in_Route')
        for f in ['hour','dayofweek','month']:
            if f in df_proc.columns: candidate.append(f)
            else:
                df_proc[f] = 0; candidate.append(f)
        features = candidate

    st.write("Using features:", features)

    # Prepare X for model scoring: try to select columns, fillna
    X = df_proc.reindex(columns=features, fill_value=0).copy()
    # If categorical mappings in artifact, try to map
    if isinstance(model_art, dict) and 'categorical_mappings' in model_art:
        for col, info in model_art['categorical_mappings'].items():
            if col in X.columns:
                mapping = info.get('mapping', {})
                global_mean = info.get('global_mean', 0)
                X[col] = X[col].map(mapping).fillna(global_mean)

    # Ensure numeric
    for c in X.columns:
        try:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
        except:
            pass

    # Baseline totals
    baseline_time = 0.0
    if 'Delivery_Time_Minutes' in df_proc.columns:
        baseline_time = df_proc['Delivery_Time_Minutes'].astype(float).sum()
    baseline_cost = df_proc['Delivery_Cost_INR'].astype(float).sum() if 'Delivery_Cost_INR' in df_proc.columns else 0.0
    baseline_distance = df_proc['Distance_km'].astype(float).sum() if 'Distance_km' in df_proc.columns else 0.0

    # Predict using model -> we expect model_art to contain 'model' or be the model object itself
    model_obj = None
    if isinstance(model_art, dict) and 'model' in model_art:
        model_obj = model_art['model']
    else:
        model_obj = model_art

    try:
        preds = model_obj.predict(X)
    except Exception as e:
        st.error("Model prediction failed: " + str(e))
        st.stop()

    df_proc['predicted_eta_min'] = preds
    predicted_total_time = float(df_proc['predicted_eta_min'].sum())
    # crude predicted cost/distance reduction simulation (you can replace with route optimizer later)
    predicted_total_cost = baseline_cost * 0.9 if baseline_cost > 0 else 0.0
    predicted_total_distance = baseline_distance * 0.98 if baseline_distance > 0 else 0.0
    fuel_saved_liters = (baseline_distance - predicted_total_distance) * 0.12

    # Show metrics
    st.header("Before vs After (aggregate)")
    c1, c2, c3 = st.columns(3)
    #c1.metric("Total time (before) min", f\"{baseline_time:.1f}\", delta=None)
    #c2.metric("Total time (after) min", f\"{predicted_total_time:.1f}\", delta=f\"{baseline_time - predicted_total_time:.1f}\")
    #c3.metric("Time saved (min)", f\"{baseline_time - predicted_total_time:.1f}\")

    c1.metric("Total time (before) min", f"{baseline_time:.1f}", delta=None)
    c2.metric("Total time (after) min", f"{predicted_total_time:.1f}", delta=f"{baseline_time - predicted_total_time:.1f}")
    c3.metric("Time saved (min)", f"{baseline_time - predicted_total_time:.1f}")

    c4, c5, c6 = st.columns(3)
    #c4.metric("Total cost (before) INR", f\"{baseline_cost:.2f}\")
    #c5.metric("Total cost (after) INR", f\"{predicted_total_cost:.2f}\", delta=f\"{baseline_cost - predicted_total_cost:.2f}\")
    #c6.metric("Fuel saved (liters)", f\"{fuel_saved_liters:.2f}\")

    c4.metric("Total cost (before) INR", f"{baseline_cost:.2f}")
    c5.metric("Total cost (after) INR", f"{predicted_total_cost:.2f}", delta=f"{baseline_cost - predicted_total_cost:.2f}")
    c6.metric("Fuel saved (liters)", f"{fuel_saved_liters:.2f}")

    st.subheader("Per-delivery sample")
    st.dataframe(df_proc[['Delivery_ID','Distance_km','Delivery_Time_Minutes','predicted_eta_min']].head(50))

    # Show simple evaluation if baseline present
    if 'Delivery_Time_Minutes' in df_proc.columns:
        mae = mean_absolute_error(df_proc['Delivery_Time_Minutes'], df_proc['predicted_eta_min'])
        rmse = mean_squared_error(df_proc['Delivery_Time_Minutes'], df_proc['predicted_eta_min'], squared=False)
        st.write(f\"Model MAE: {mae:.2f} min, RMSE: {rmse:.2f} min\")

    # Allow download of results
    def convert_df_to_csv(df_out):
        return df_out.to_csv(index=False).encode('utf-8')

    csv_out = convert_df_to_csv(df_proc)
    st.download_button(\"Download results CSV\", csv_out, file_name=\"predicted_results.csv\", mime='text/csv')

    st.success(\"Analysis complete — use Optimize endpoint later to compute route-level savings.\")
else:
    st.info(\"Upload both CSV and model artifact to run analysis.\")
