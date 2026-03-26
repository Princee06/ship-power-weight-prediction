import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---- 0. Page Config ---- #
st.set_page_config(page_title="Ship Power & Weight Prediction", layout="wide")

# ---- 1. Load Models Safely ---- #
@st.cache_resource
def load_models():
    p_model = joblib.load("saved_models/power_pipeline.pkl")
    w_model = joblib.load("saved_models/weight_pipeline.pkl")
    return p_model, w_model

power_model, weight_model = None, None

try:
    power_model, weight_model = load_models()
except Exception as e:
    st.error("❌ Model loading failed")
    st.exception(e)

# STOP if models not loaded
if power_model is None or weight_model is None:
    st.stop()

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
        lb = l / b
        if lb < 3 or lb > 12:
            warnings.append(f"Unusual L/B ratio: {lb:.2f}")

    return errors, warnings

# ---- 4. Feature Engineering ---- #
def engineer_features(df):
    df["L_B"] = df["loa_m"] / df["breadth_m"]
    df["B_D"] = df["breadth_m"] / df["depth_m"]
    df["L_D"] = df["loa_m"] / df["depth_m"]
    df["speed_length_ratio"] = df["service_speed_kn"] / (df["loa_m"] ** 0.5)

    df["year_bucket"] = pd.cut(
        df["year_built"],
        bins=[0, 1999, 2009, 2019, 2035],
        labels=["old", "mid", "modern", "latest"]
    )

    default_cols = {
        "block_coefficient": 0.7,
        "fresh_water_capacity_m3": 100,
        "dp_class": "DP1",
        "deck_area_m2": 500,
        "dataset_source": "manual",
        "number_of_engines": 2,
        "bollard_pull_t": 50,
        "deck_load_t": 200,
        "strata": "medium",
        "number_of_thrusters": 2,
        "payload_t": 1000,
        "level1_group": "support",
        "cargo_pump_capacity_m3hr": 0,
        "engine_type": "diesel",
        "dwt_t": 5000,
        "size_band": "medium",
        "teu_capacity": 0,
        "displacement_t": 8000,
        "propeller_diameter_m": 4,
        "lpp_m": df["loa_m"] * 0.95,
        "propulsion_type": "conventional",
        "fuel_capacity_m3": 300
    }

    for col, val in default_cols.items():
        if col not in df.columns:
            df[col] = val

    return df

# ---- Sidebar ---- #
with st.sidebar:
    st.header("Engineering Panel")
    validation_mode = st.toggle("Validation Mode")

# ---- Main UI ---- #
st.title("Ship Power & Weight Prediction")

selected_type = st.selectbox(
    "Ship Type",
    ["Select...", "OSV", "Tug", "Bulk Carrier", "Container"]
)

defaults = PRESETS.get(selected_type, {"loa": 0, "breadth": 0, "depth": 0, "draft": 0, "speed": 0})

col1, col2 = st.columns(2)

with col1:
    loa = st.number_input("LOA", value=defaults["loa"])
    breadth = st.number_input("Breadth", value=defaults["breadth"])
    depth = st.number_input("Depth", value=defaults["depth"])
    draft = st.number_input("Draft", value=defaults["draft"])

with col2:
    speed = st.number_input("Speed", value=defaults["speed"])
    year = st.number_input("Year", value=2024)

# ---- Prediction ---- #
if st.button("Predict"):

    errs, warns = run_sanity_checks(loa, breadth, depth, draft, speed, selected_type)

    if selected_type == "Select...":
        st.error("Please select ship type.")
        st.stop()

    if errs:
        for e in errs:
            st.error(e)
        st.stop()

    for w in warns:
        st.warning(w)

    raw_df = pd.DataFrame([{
        "loa_m": loa,
        "breadth_m": breadth,
        "depth_m": depth,
        "draft_m": draft,
        "service_speed_kn": speed,
        "year_built": year,
        "ship_type": selected_type
    }])

    input_data = engineer_features(raw_df)

    # ---- Predictions ---- #
    p_val = power_model.predict(input_data)[0]
    w_val = weight_model.predict(input_data)[0]

    st.metric("Power (kW)", f"{p_val:,.0f}")
    st.metric("Weight (t)", f"{w_val:,.0f}")

    # ---- SHAP ---- #
    st.markdown("### 🔍 Feature Contribution")

    try:
        X_transformed = power_model[:-1].transform(input_data)
        model = power_model.named_steps["model"]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=power_model[:-1].get_feature_names_out(),
            show=False
        )
        st.pyplot(fig)

    except Exception:
        st.warning("SHAP visualization not supported.")

# ---- Batch Prediction ---- #
st.markdown("---")
file = st.file_uploader("Upload CSV")

if file:
    data = pd.read_csv(file)

    processed = engineer_features(data.copy())

    data["power"] = power_model.predict(processed)
    data["weight"] = weight_model.predict(processed)

    st.dataframe(data.head())

    st.download_button(
        "Download",
        data.to_csv(index=False),
        "predictions.csv"
    )
