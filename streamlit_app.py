import streamlit as st
import numpy as np
import joblib
import json
import requests
from streamlit_lottie import st_lottie

# Page config
st.set_page_config(page_title="Battery SoH Simulator", page_icon="🔋", layout="centered")

# Load Lottie Animation
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

battery_anim = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_qp1q7mct.json")
robot_anim = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_tll0j4bb.json")

# Load model
try:
   model = joblib.load("model/soh_predictor_6features.joblib")
    input_data = np.array([[voltage, current, temperature, load_current, load_voltage, time]])
    soh = model.predict(input_data)[0]
    st.success(f"📊 Predicted State of Health: **{soh:.2f}%**")

# --- Header ---
st_lottie(robot_anim, height=180, key="robot")
st.markdown("<h1 style='text-align: center; color: #33cccc;'>🔋 Battery SoH Simulator</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #888;'>by <b>WIZIDEAZ</b></h4>", unsafe_allow_html=True)

# --- Input Form ---
with st.form("battery_form"):
    st.subheader("Enter Battery Parameters:")
    col1, col2 = st.columns(2)

    with col1:
        voltage_measured = st.number_input("Voltage Measured (V)", min_value=0.0, step=0.1)
        current_measured = st.number_input("Current Measured (A)", min_value=0.0, step=0.1)
        temperature = st.number_input("Temperature (°C)", min_value=-40.0, max_value=100.0, step=0.1)

    with col2:
        current_load = st.number_input("Current Load (A)", min_value=0.0, step=0.1)
        voltage_load = st.number_input("Voltage Load (V)", min_value=0.0, step=0.1)
        time_sec = st.number_input("Time (s)", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("🔍 Simulate SoH")

# --- Prediction ---
if submitted:
    try:
        features = np.array([[voltage_measured, current_measured, temperature,
                              current_load, voltage_load, time_sec]])
        soh = model.predict(features)[0]
        soh_percent = round(float(soh) * 100, 2)

        st.success(f"✅ Predicted State of Health: **{soh_percent}%**")

        if soh_percent > 80:
            st.info("Battery health is good! 👍")
        elif soh_percent > 60:
            st.warning("Battery health is moderate. ⚠️ Consider checkup.")
        else:
            st.error("Battery health is poor. ❌ Maintenance advised!")

        if temperature > 60:
            st.warning("🌡️ High temperature! May affect battery life.")
        if voltage_measured < 3.0:
            st.warning("⚡ Low voltage detected. Recharge recommended.")

        st_lottie(battery_anim, height=250, key="battery")

    except Exception as e:
        st.error(f"❌ Model prediction failed: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 13px;'>Built with ❤️ by <b>WIZIDEAZ</b> | 2025</p>", unsafe_allow_html=True)
