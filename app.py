import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ---- 0. Page Config ---- #
st.set_page_config(page_title="Ship Power & Weight Prediction", layout="wide")

# ---- 1. Load Models Safely ---- #
@st.cache_resource
def load_models():
    # Ensure the directory exists to avoid errors
    model_path = "saved_models"
    p_path = os.path.join(model_path, "power_pipeline.pkl")
    w_path = os.path.join(model_path, "weight_pipeline.pkl")

    if not os.path.exists(p_path) or not os.path.exists(w_path):
        st.error(f"❌ Models NOT FOUND in {model_path}. Please check your file structure.")
        st.stop()
        
    p_model = joblib.load(p_path)
    w_model = joblib.load(w_path)
    return p_model, w_model

# ---- 2. Presets ---- #
PRESETS = {
    "Tug": {"loa": 32.0, "breadth": 11.0, "depth": 5.0, "draft": 4.0, "speed": 12.0},
    "Container": {"loa": 300.0, "breadth": 48.0, "depth": 24.0, "draft": 13.0, "speed": 22.0},
    "Bulk Carrier": {"loa": 225.0, "breadth": 32.2, "depth": 19.0, "draft": 14.2, "speed": 14.5},
    "OSV": {"loa": 85.0, "breadth": 19.0, "depth": 8.0, "draft": 6.5, "speed": 13.0}
}

# ---- 3. Sanity Checks ---- #
def run_sanity_checks(l, b, d, dr, s, t):
    errors, warnings = [], []
    if dr > d:
        errors.append("Draft cannot be deeper than Depth.")
    if l > 450:
        errors.append("LOA exceeds 450m.")
    if t == "Tug" and l > 100:
        errors.append("Tugs rarely exceed 100m.")
    if t == "Bulk Carrier" and s > 18:
        warnings.append("Speed high for Bulk Carrier.")
    if b > 0:
        lb = l/b
        if lb < 3 or lb > 12:
            warnings.append(f"Unusual L/B ratio: {lb:.2f}")
    return errors, warnings

# ---- 4. Feature Engineering ---- #
def engineer_features(df):
    # Ensure we don't modify the original dataframe in-place in a way that breaks Streamlit
    df = df.copy()
    df["L_B"] = df["loa_m"] / df["breadth_m"].replace(0, 1)
    df["B_D"] = df["breadth_m"] / df["depth_m"].replace(0, 1)
    df["L_D"] = df["loa_m"] / df["depth_m"].replace(0, 1)
    df["speed_length_ratio"] = df["service_speed_kn"] / (df["loa_m"] ** 0.5).replace(0, 1)
    
    df["year_bucket"] = pd.cut(
        df["year_built"], 
        bins=[0, 1999, 2009, 2019, 2035], 
        labels=["old", "mid", "modern", "latest"]
    )

    # Required columns for the pipeline
    default_cols = {
        "block_coefficient": 0.7, "fresh_water_capacity_m3": 100, "dp_class": "DP1",
        "deck_area_m2": 500, "dataset_source": "manual", "number_of_engines": 2,
        "bollard_pull_t": 50, "deck_load_t": 200, "strata": "medium",
        "number_of_thrusters": 2, "payload_t": 1000, "level1_group": "support",
        "cargo_pump_capacity_m3hr": 0, "engine_type": "diesel", "dwt_t": 5000,
        "size_band": "medium", "teu_capacity": 0, "displacement_t": 8000,
        "propeller_diameter_m": 4, "lpp_m": df["loa_m"] * 0.95,
        "propulsion_type": "conventional", "fuel_capacity_m3": 300
    }
    
    for col, val in default_cols.items():
        if col not in df.columns:
            df[col] = val
    return df

# ---- UI Logic ---- #
power_model, weight_model = load_models()

with st.sidebar:
    st.header("Engineering Panel")
    validation_mode = st.toggle("Validation Mode")

st.title("Ship Power & Weight Prediction")
selected_type = st.selectbox(
    "Ship Type", ["Select...", "OSV", "Tug", "Bulk Carrier", "Container"]
)

# Load defaults based on selection
defaults = PRESETS.get(selected_type, {"loa": 0.0, "breadth": 0.0, "depth": 0.0, "draft": 0.0, "speed": 0.0})

col1, col2 = st.columns(2)
with col1:
    loa = st.number_input("LOA (m)", value=float(defaults["loa"]))
    breadth = st.number_input("Breadth (m)", value=float(defaults["breadth"]))
    depth = st.number_input("Depth (m)", value=float(defaults["depth"]))
    draft = st.number_input("Draft (m)", value=float(defaults["draft"]))
with col2:
    speed = st.number_input("Speed (kn)", value=float(defaults["speed"]))
    year = st.number_input("Year Built", value=2024)

if st.button("Predict"):
    errs, warns = run_sanity_checks(loa, breadth, depth, draft, speed, selected_type)
    
    if errs:
        for e in errs: st.error(e)
        st.stop()
    for w in warns: 
        st.warning(w)

    # Prepare data
    input_df = pd.DataFrame([{
        "loa_m": loa, "breadth_m": breadth, "depth_m": depth,
        "draft_m": draft, "service_speed_kn": speed, 
        "year_built": year, "ship_type": selected_type
    }])
    
    processed_df = engineer_features(input_df)

    # ---- Predictions ---- #
    p_pred = power_model.predict(processed_df)[0]
    w_pred = weight_model.predict(processed_df)[0]

    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Predicted Power", f"{p_pred:,.0f} kW")
    res_col2.metric("Estimated Weight", f"{w_pred:,.0f} t")

    # ---- SHAP Analysis ---- #
    st.markdown("---")
    st.markdown("### 🔍 Feature Contribution (Power Model)")
    try:
        # 1. Transform the raw data through the pipeline's preprocessing steps
        # This assumes your pipeline is structured as [preprocessor, model]
        preprocessor = power_model[:-1]
        model_step = power_model[-1]
        
        X_transformed = preprocessor.transform(processed_df)
        
        # 2. Extract feature names if possible
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = None

        explainer = shap.TreeExplainer(model_step)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.info("Detailed SHAP visualization is unavailable for this specific pipeline configuration.")

# ---- Batch Processing ---- #
st.markdown("---")
st.subheader("Batch Prediction")
file = st.file_uploader("Upload CSV for bulk analysis", type=["csv"])

if file:
    batch_data = pd.read_csv(file)
    try:
        batch_processed = engineer_features(batch_data)
        batch_data["Predicted Power (kW)"] = power_model.predict(batch_processed)
        batch_data["Predicted Weight (t)"] = weight_model.predict(batch_processed)
        
        st.dataframe(batch_data.head(10))
        
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "ship_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")
