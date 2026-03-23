import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -------------------------------
# 0. Page Config
# -------------------------------
st.set_page_config(page_title="Ship Power & Weight Prediction", layout="wide")

# -------------------------------
# 1. Load Models & Pipelines
# -------------------------------
@st.cache_resource
def load_models():
    path = "saved_models"
    # Pipelines
    power_pipeline = joblib.load(f"{path}/power_pipeline.pkl")
    weight_pipeline = joblib.load(f"{path}/weight_pipeline.pkl")
    # Prediction Models
    power_model = joblib.load(f"{path}/power_prediction_model.pkl")
    weight_model = joblib.load(f"{path}/weight_prediction_model.pkl")
    # Feature columns
    feature_cols = joblib.load(f"{path}/feature_columns.pkl")
    return feature_cols, power_pipeline, weight_pipeline, power_model, weight_model

feature_columns, power_pipeline, weight_pipeline, power_model, weight_model = load_models()

# -------------------------------
# 2. Presets & Sanity Checks
# -------------------------------
PRESETS = {
    "Tug": {"loa": 32.0, "breadth": 11.0, "depth": 5.0, "draft": 4.0, "speed": 12.0},
    "Container": {"loa": 300.0, "breadth": 48.0, "depth": 24.0, "draft": 13.0, "speed": 22.0},
    "Bulk Carrier": {"loa": 225.0, "breadth": 32.2, "depth": 19.0, "draft": 14.2, "speed": 14.5},
    "OSV": {"loa": 85.0, "breadth": 19.0, "depth": 8.0, "draft": 6.5, "speed": 13.0}
}

def run_sanity_checks(l, b, d, dr, s, t):
    errors, warnings = [], []
    if dr > d: errors.append("Draft cannot be deeper than Depth.")
    if l > 450: errors.append("LOA exceeds maximum known ship limits (450m).")
    if t == "Tug" and l > 100: errors.append("Standard Tugs rarely exceed 100m. Check LOA.")
    if t == "Bulk Carrier" and s > 18: warnings.append("Speed is high for a Bulk Carrier (typically <16 kts).")
    lb_ratio = l / b if b > 0 else 0
    if lb_ratio != 0 and (lb_ratio < 3.0 or lb_ratio > 12): warnings.append(f"Unusual L/B Ratio ({lb_ratio:.2f}) for this ship type.")
    return errors, warnings

# -------------------------------
# 3. Feature Engineering
# -------------------------------
def engineer_features(df):
    df["L_B"] = df["loa_m"] / df["breadth_m"]
    df["B_D"] = df["breadth_m"] / df["depth_m"]
    df["L_D"] = df["loa_m"] / df["depth_m"]
    df["speed_length_ratio"] = df["service_speed_kn"] / (df["loa_m"] ** 0.5)
    df["year_bucket"] = pd.cut(df["year_built"], bins=[0, 1999, 2009, 2019, 2035], labels=["old","mid","modern","latest"])

    # Default columns needed by model
    defaults = {
        "block_coefficient": 0.7, "fresh_water_capacity_m3": 100, "dp_class": "DP1",
        "deck_area_m2": 500, "dataset_source": "manual", "number_of_engines": 2,
        "bollard_pull_t": 50, "deck_load_t": 200, "strata": "medium",
        "number_of_thrusters": 2, "payload_t": 1000, "level1_group": "support",
        "cargo_pump_capacity_m3hr": 0, "engine_type": "diesel", "dwt_t": 5000,
        "size_band": "medium", "teu_capacity": 0, "displacement_t": 8000,
        "propeller_diameter_m": 4, "lpp_m": df["loa_m"] * 0.95,
        "propulsion_type": "conventional", "fuel_capacity_m3": 300
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df

# -------------------------------
# 4. Sidebar
# -------------------------------
with st.sidebar:
    st.header("Engineering Panel")
    st.info("AI Model v2.0 | TAI Engineers")

    with st.expander("🛠 Assumptions & Limitations", expanded=True):
        st.markdown("""
        **Reliability Zone:**
        - LOA: 25m - 400m
        - Speed: 10 - 25 kts
        **Assumptions:**
        - Standard Steel Hull.
        - Installed Power = Total MCR.
        **Limitations:**
        - No environmental factors.
        - Accuracy: ±8-12%.
        """)

    st.markdown("---")
    validation_mode = st.checkbox("Enable Engineer Validation Mode")

# -------------------------------
# 5. Single Ship Prediction
# -------------------------------
st.title("Ship Power & Weight Prediction")
st.subheader("Interactive Design Estimator")

selected_type = st.selectbox("Select Ship Type", ["Select..."] + list(PRESETS.keys()))
defaults = PRESETS.get(selected_type, {"loa":0,"breadth":0,"depth":0,"draft":0,"speed":0})

col1, col2 = st.columns(2)
with col1:
    loa_m = st.number_input("LOA [m]", value=defaults["loa"])
    breadth_m = st.number_input("Breadth [m]", value=defaults["breadth"])
    depth_m = st.number_input("Depth [m]", value=defaults["depth"])
    draft_m = st.number_input("Draft [m]", value=defaults["draft"])
with col2:
    speed = st.number_input("Service Speed [kn]", value=defaults["speed"])
    year_built = st.number_input("Year Built", 1980, 2035, 2024)
    manual_p = st.number_input("Manual Power Estimate (kW)", value=0.0) if validation_mode else 0.0

if st.button("Calculate Results"):
    errs, warns = run_sanity_checks(loa_m, breadth_m, depth_m, draft_m, speed, selected_type)

    if selected_type == "Select...":
        st.error("Please select a Ship Type.")
    elif errs:
        for e in errs: st.error(f"❌ {e}")
    else:
        for w in warns: st.warning(f"⚠️ {w}")

        raw_data = pd.DataFrame([{
            "loa_m": loa_m, "breadth_m": breadth_m, "depth_m": depth_m,
            "draft_m": draft_m, "service_speed_kn": speed,
            "year_built": year_built, "ship_type": selected_type
        }])
        input_data = engineer_features(raw_data.copy())

        # Transform & Predict
        X_power = power_pipeline.transform(input_data)
        X_weight = weight_pipeline.transform(input_data)
        p_val = power_model.predict(X_power)[0]
        w_val = weight_model.predict(X_weight)[0]

        # Display
        st.markdown("### Estimation Results")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Predicted Power (kW)", f"{p_val:,.2f}")
        m_col2.metric("Steel Weight (t)", f"{w_val:,.2f}")
        if validation_mode and manual_p > 0:
            var = abs(p_val - manual_p)/manual_p*100
            m_col3.metric("Variance", f"{var:.1f}%", delta=f"{p_val-manual_p:,.0f} kW", delta_color="inverse")

        st.write(f"**Derived Ratios:** L/B: {input_data['L_B'][0]:.2f} | B/D: {input_data['B_D'][0]:.2f} | Speed-Len: {input_data['speed_length_ratio'][0]:.2f}")

        # SHAP for single prediction
        st.markdown("---")
        st.write("**Feature Contribution (SHAP)**")
        try:
            explainer = shap.TreeExplainer(power_model)
            shap_vals = explainer.shap_values(X_power)
            fig, ax = plt.subplots(figsize=(8,3.5))
            shap.summary_plot(shap_vals, X_power, plot_type="bar", show=False)
            st.pyplot(fig, use_container_width=False)
        except:
            st.info("SHAP visualization not available.")

# -------------------------------
# 6. Batch Prediction
# -------------------------------
st.markdown("---")
st.subheader("Batch Estimator (CSV Upload)")
up_file = st.file_uploader("Upload CSV", type=["csv"])
if up_file:
    b_df = pd.read_csv(up_file)
    essential = ["loa_m","breadth_m","depth_m","draft_m","service_speed_kn","year_built","ship_type"]
    if all(c in b_df.columns for c in essential):
        b_proc = engineer_features(b_df.copy())
        b_df["pred_power_kW"] = power_model.predict(power_pipeline.transform(b_proc))
        b_df["pred_weight_t"] = weight_model.predict(weight_pipeline.transform(b_proc))
        st.dataframe(b_df.head(10))

        if st.checkbox("Show SHAP Influence (Beeswarm)"):
            with st.spinner("Analyzing trends..."):
                sample = b_proc.head(100)
                X_sample = power_pipeline.transform(sample)
                explainer_b = shap.TreeExplainer(power_model)
                shap_vals_b = explainer_b.shap_values(X_sample)
                fig_b, ax_b = plt.subplots(figsize=(10,5))
                shap.summary_plot(shap_vals_b, X_sample, feature_names=feature_columns, show=False)
                st.pyplot(fig_b, use_container_width=True)

        st.download_button("Download Results CSV", b_df.to_csv(index=False), "ship_predictions.csv")
    else:
        st.error(f"Missing essential columns: {essential}")
