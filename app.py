import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import traceback

# ---- 0. Page Config ---- #
st.set_page_config(page_title="Ship Power & Weight Prediction", layout="wide")

# ---- 1. Load Models Safely ---- #
@st.cache_resource
def load_models():
    try:
        if not os.path.exists("saved_models/power_pipeline.pkl"):
            st.error("❌ power_pipeline.pkl NOT FOUND")
            st.stop()

        if not os.path.exists("saved_models/weight_pipeline.pkl"):
            st.error("❌ weight_pipeline.pkl NOT FOUND")
            st.stop()

        p_model = joblib.load("saved_models/power_pipeline.pkl")
        w_model = joblib.load("saved_models/weight_pipeline.pkl")
        return p_model, w_model

    except Exception as e:
        st.error(f"Full error:\n{traceback.format_exc()}")
        st.stop()

# ✅ FIX: Call the function to assign the models
power_model, weight_model = load_models()

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

# ---- 4. Fuzzy Column Mapper ---- #
def auto_map_columns(df):
    mappings = {
        "loa_m": ["loa", "length", "length_m", "loa_m", "len", "overall_length", "ship_length", "l"],
        "breadth_m": ["breadth", "beam", "width", "breadth_m", "beam_m", "width_m", "b"],
        "depth_m": ["depth", "depth_m", "moulded_depth", "hull_depth", "d"],
        "draft_m": ["draft", "draught", "draft_m", "draught_m", "t"],
        "service_speed_kn": ["speed", "service_speed", "speed_kn", "service_speed_kn",
                              "service_speed_knots", "design_speed", "knots"],
        "year_built": ["year", "year_built", "built", "build_year", "construction_year"],
        "ship_type": ["type", "ship_type", "vessel_type", "category", "ship_class"],
        "dwt_t": ["dwt", "dwt_t", "dwt_tonnes", "deadweight", "dead_weight"],
        "payload_t": ["payload", "payload_t", "payload_tonnes", "cargo"],
        "displacement_t": ["displacement", "displacement_t", "lightship",
                            "lightship_weight", "lightship_weight_tonnes"],
    }

    rename_dict = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for standard, variants in mappings.items():
        for variant in variants:
            if variant.lower() in cols_lower:
                original = cols_lower[variant.lower()]
                if original not in rename_dict.values():
                    rename_dict[original] = standard
                break

    return df.rename(columns=rename_dict)

# ---- 5. Feature Engineering ---- #
def engineer_features(df):
    df["L_B"] = df["loa_m"] / df["breadth_m"]
    df["B_D"] = df["breadth_m"] / df["depth_m"]
    df["L_D"] = df["loa_m"] / df["depth_m"]
    df["speed_length_ratio"] = df["service_speed_kn"] / (df["loa_m"] ** 0.5)
    df["year_bucket"] = pd.cut(
        df["year_built"], bins=[0, 1999, 2009, 2019, 2035],
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
    "Ship Type", ["Select...", "OSV", "Tug", "Bulk Carrier", "Container"]
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

if st.button("Predict"):
    errs, warns = run_sanity_checks(loa, breadth, depth, draft, speed, selected_type)
    if errs:
        for e in errs: st.error(e)
        st.stop()
    for w in warns: st.warning(w)

    df = pd.DataFrame([{
        "loa_m": loa, "breadth_m": breadth, "depth_m": depth, "draft_m": draft,
        "service_speed_kn": speed, "year_built": year, "ship_type": selected_type
    }])
    df = engineer_features(df)

    # ---- Predictions ---- #
    p = power_model.predict(df)[0]
    w = weight_model.predict(df)[0]
    st.metric("Power (kW)", f"{p:,.0f}")
    st.metric("Weight (t)", f"{w:,.0f}")

    # ---- SHAP ---- #
    st.markdown("### 🔍 Feature Contribution")
    try:
        X_transformed = power_model[:-1].transform(df)
        model = power_model.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_transformed, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning("SHAP visualization not supported for this model.")

# ---- Batch ---- #
st.markdown("---")
st.subheader("Batch Prediction")
st.caption("Upload any CSV — column names are auto-detected (e.g. length, Length_m, loa, LOA all work)")
file = st.file_uploader("Upload CSV")
if file:
    data = pd.read_csv(file)

    # Show detected columns
    st.write("📋 Detected columns:", list(data.columns))

    # Auto map columns
    data = auto_map_columns(data)

    # Fill missing required columns with defaults
    if "depth_m" not in data.columns:
        data["depth_m"] = data["draft_m"] * 1.3 if "draft_m" in data.columns else 8.0
    if "year_built" not in data.columns:
        data["year_built"] = 2015
    if "ship_type" not in data.columns:
        data["ship_type"] = "OSV"
    if "lpp_m" not in data.columns:
        data["lpp_m"] = data["loa_m"] * 0.95 if "loa_m" in data.columns else 80.0

    data = engineer_features(data)
    data["predicted_power_kW"] = power_model.predict(data)
    data["predicted_weight_t"] = weight_model.predict(data)

    display_cols = [c for c in ["loa_m", "breadth_m", "draft_m", "service_speed_kn",
                                 "predicted_power_kW", "predicted_weight_t"] if c in data.columns]
    st.dataframe(data[display_cols])
    st.download_button("⬇️ Download Results", data.to_csv(index=False), file_name="predictions.csv")
