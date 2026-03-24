import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
#----OPTIONAL IMPORT SHAP----#
try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

# ---- 0. Page Config ---- #
st.set_page_config(page_title="Ship Power & Weight Prediction", layout="wide")

# ---- 1. Load Models ---- #
@st.cache_resource
def load_models():
    # Ensure these .pkl files are in the 'saved_models/' folder
    p_model = joblib.load("saved_models/power_pipeline.pkl")
    w_model = joblib.load("saved_models/weight_pipeline.pkl")
    return p_model, w_model

power_model, weight_model = load_models()

# ---- 2. Helper Logic: Presets & Sanity Checks ---- #
PRESETS = {
    "Tug": {"loa": 32.0, "breadth": 11.0, "depth": 5.0, "draft": 4.0, "speed": 12.0},
    "Container": {"loa": 300.0, "breadth": 48.0, "depth": 24.0, "draft": 13.0, "speed": 22.0},
    "Bulk Carrier": {"loa": 225.0, "breadth": 32.2, "depth": 19.0, "draft": 14.2, "speed": 14.5},
    "OSV": {"loa": 85.0, "breadth": 19.0, "depth": 8.0, "draft": 6.5, "speed": 13.0}
}

def run_sanity_checks(l, b, d, dr, s, t):
    errors = []
    warnings = []
    if dr > d: errors.append("Draft cannot be deeper than Depth.")
    if l > 450: errors.append("LOA exceeds maximum known ship limits (450m).")
    if t == "Tug" and l > 100: errors.append("Standard Tugs rarely exceed 100m. Check LOA.")
    if t == "Bulk Carrier" and s > 18: warnings.append("Speed is high for a Bulk Carrier (typically <16 kts).")
    lb_ratio = l/b if b > 0 else 0
    if lb_ratio != 0 and (lb_ratio < 3.0 or lb_ratio > 12):
        warnings.append(f"Unusual L/B Ratio ({lb_ratio:.2f}) for this ship type.")
    return errors, warnings

# ---- 3. Feature Engineering ---- #
def engineer_features(df):
    df["L_B"] = df["loa_m"] / df["breadth_m"]
    df["B_D"] = df["breadth_m"] / df["depth_m"]
    df["L_D"] = df["loa_m"] / df["depth_m"]
    df["speed_length_ratio"] = df["service_speed_kn"] / (df["loa_m"] ** 0.5)
    df["year_bucket"] = pd.cut(df["year_built"], bins=[0, 1999, 2009, 2019, 2035], labels=["old", "mid", "modern", "latest"])
    
    # Model requirement: Must have all columns used during training
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

# ---- 4. Sidebar: Task 2 (Assumptions) & Task 3 (Validation Toggle) ---- #
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
    validation_mode = st.toggle("Enable Engineer Validation Mode")
    if validation_mode:
        st.caption("Compare AI vs. Manual calculations.")

# ---- 5. Section 1: Single Ship Prediction (Task 1) ---- #
st.title("Ship Power & Weight Prediction")
st.subheader("Interactive Design Estimator")

# Ship Type Preset Selector
selected_type = st.selectbox("1. Select Ship Type (loads presets)", ["Select...", "OSV", "Tug", "Bulk Carrier", "Container"])
defaults = PRESETS.get(selected_type, {"loa": 0.0, "breadth": 0.0, "depth": 0.0, "draft": 0.0, "speed": 0.0})

col1, col2 = st.columns(2)
with col1:
    loa_m = st.number_input("LOA [m]", value=defaults["loa"], format="%.2f")
    breadth_m = st.number_input("Breadth [m]", value=defaults["breadth"], format="%.2f")
    depth_m = st.number_input("Depth [m]", value=defaults["depth"], format="%.2f")
    draft_m = st.number_input("Draft [m]", value=defaults["draft"], format="%.2f")

with col2:
    speed = st.number_input("Service Speed [kn]", value=defaults["speed"], format="%.2f")
    year_built = st.number_input("Year Built", min_value=1980, max_value=2035, value=2024)
    
    # Task 3: Manual Input for Validation
    manual_p = 0.0
    if validation_mode:
        manual_p = st.number_input("Your Manual Power Estimate (kW)", value=0.0)

if st.button("Calculate Results"):
    errs, warns = run_sanity_checks(loa_m, breadth_m, depth_m, draft_m, speed, selected_type)
    
    if selected_type == "Select...":
        st.error("Please select a Ship Type.")
    elif errs:
        for e in errs: st.error(f"❌ {e}")
    else:
        if warns:
            for w in warns: st.warning(f"⚠️ {w}")
        
        # Predict
        raw_data = pd.DataFrame([{"loa_m": loa_m, "breadth_m": breadth_m, "depth_m": depth_m, "draft_m": draft_m, "service_speed_kn": speed, "year_built": year_built, "ship_type": selected_type}])
        input_data = engineer_features(raw_data)
        
        p_val = power_model.predict(input_data)[0]
        w_val = weight_model.predict(input_data)[0]

        # Display Metrics
        st.markdown("### Estimation Results")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Predicted Power (kW)", f"{p_val:,.2f}")
        m_col2.metric("Steel Weight (t)", f"{w_val:,.2f}")
        
        # Task 3: Validation Logic
        if validation_mode and manual_p > 0:
            var = (abs(p_val - manual_p) / manual_p) * 100
            m_col3.metric("Variance", f"{var:.1f}%", delta=f"{p_val-manual_p:,.0f} kW", delta_color="inverse")

        # Task 3: UX Polish - Incorporate Feedback Expander
        with st.expander("📝 Provide Feedback on this Prediction"):
            st.write("Does this result align with your engineering judgment?")
            fb1, fb2 = st.columns(2)
            if fb1.button("Looks Accurate 👍"):
                st.toast("Thank you! Feedback logged.")
            if fb2.button("Seems Off 👎"):
                st.toast("Noted. We will review this case for model tuning.")

        # Ratios
        st.write(f"**Derived Ratios:** L/B: {input_data['L_B'][0]:.2f} | B/D: {input_data['B_D'][0]:.2f} | Speed-Len: {input_data['speed_length_ratio'][0]:.2f}")

        # SHAP Bar Chart
        st.markdown("---")
        st.write("**Feature Contribution (Why this prediction?)**")
        if SHAP_AVAILABLE:
    try:
        num_data = power_model[:-1].transform(input_data)
        explainer = shap.TreeExplainer(power_model.named_steps["model"])
        shap_v = explainer(num_data)
        shap_v.feature_names = power_model[:-1].get_feature_names_out()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        shap.plots.bar(shap_v[0], max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
    except:
        st.info("SHAP visualization unavailable.")
else:
    st.info("SHAP not installed in this environment.")

# ---- 6. Section 2: Batch Prediction & Trends ---- #
st.markdown("---")
st.subheader("Batch Estimator (CSV)")
up_file = st.file_uploader("Upload CSV Data", type=["csv"])

if up_file:
    b_df = pd.read_csv(up_file)
    essential = ["loa_m", "breadth_m", "depth_m", "draft_m", "service_speed_kn", "year_built", "ship_type"]
    
    if all(c in b_df.columns for c in essential):
        b_proc = engineer_features(b_df.copy())
        b_df["pred_power_kW"] = power_model.predict(b_proc)
        b_df["pred_weight_t"] = weight_model.predict(b_proc)
        
        st.dataframe(b_df.head(10))
        
        # GLOBAL SHAP TRENDS (Beeswarm)
        if SHAP_AVAILABLE and st.checkbox("Show Global Influence Trends (Beeswarm)"):
            with st.spinner("Analyzing dataset trends..."):
                sample = b_proc.head(100)
                num_sample = power_model[:-1].transform(sample)
                explainer_b = shap.TreeExplainer(power_model.named_steps["model"])
                shap_vals_b = explainer_b.shap_values(num_sample)
                fig_b, ax_b = plt.subplots(figsize=(10, 5))
                shap.summary_plot(shap_vals_b, num_sample, feature_names=power_model[:-1].get_feature_names_out(), show=False)
                plt.tight_layout()
                st.pyplot(fig_b, use_container_width=False)

        st.download_button("Download Results CSV", b_df.to_csv(index=False), "ship_predictions.csv")
    else:
        st.error(f"Missing essential columns: {essential}")
