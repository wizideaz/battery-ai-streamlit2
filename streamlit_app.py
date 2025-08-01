import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Battery AI Simulator", page_icon="🔋")
st.title("🔋 Battery AI Health Predictor")
st.markdown("This smart predictor uses a trained AI model to estimate your battery's State of Health (SoH) based on real-world parameters.")

# Input fields
voltage = st.number_input("🔌 Voltage Measured (V)", min_value=0.0, max_value=5.0, step=0.1)
current = st.number_input("⚡ Current Measured (A)", min_value=0.0, max_value=5.0, step=0.1)
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=100.0)
current_load = st.number_input("🔋 Current Load (A)", min_value=0.0, max_value=5.0)
voltage_load = st.number_input("🔌 Voltage Load (V)", min_value=0.0, max_value=5.0)
time = st.number_input("⏱️ Time (s)", min_value=0)

if st.button("📊 Predict SoH"):
    try:
        model = joblib.load("soh_predictor_6features.joblib")
        features = np.array([[voltage, current, temperature, current_load, voltage_load, time]])
        soh = model.predict(features)[0]
        st.success(f"🔋 Predicted State of Health (SoH): {soh:.2f}%")

        if soh > 80:
            st.info("✅ Battery health is good.")
        elif soh > 50:
            st.warning("⚠️ Battery health is moderate. Consider maintenance.")
        else:
            st.error("❌ Battery health is poor!")
    except Exception as e:
        st.error(f"Model prediction failed: {e}")