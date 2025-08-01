import streamlit as st
import joblib
import numpy as np
from streamlit_lottie import st_lottie
import json

st.set_page_config(page_title="Battery SoH Predictor", page_icon="🔋", layout="centered")

# === Load Animation JSON ===
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# === Sidebar Branding ===
st.sidebar.image("wizideaz_logo.png", use_column_width=True)
st.sidebar.title("⚡ WIZIDEAZ")
st.sidebar.markdown("Innovating Battery Intelligence")

# === Header with Animation ===
lottie_battery = load_lottiefile("battery.json")  # Place animation JSON in same repo
st_lottie(lottie_battery, speed=1, loop=True, quality="high", height=200)
st.title("🔋 Battery SoH Prediction")

# === User Input ===
st.subheader("Enter Battery Parameters")
voltage = st.number_input("Voltage Measured (V)", min_value=0.0, step=0.1)
current = st.number_input("Current Measured (A)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=-20.0, step=1.0)
load_current = st.number_input("Current Load (A)", min_value=0.0, step=0.1)
load_voltage = st.number_input("Voltage Load (V)", min_value=0.0, step=0.1)
time = st.number_input("Time (s)", min_value=0.0, step=10.0)

# === Predict ===
if st.button("🔍 Predict SoH"):
    model = joblib.load("model/soh_predictor_6features.joblib")
    input_data = np.array([[voltage, current, temperature, load_current, load_voltage, time]])
    soh = model.predict(input_data)[0]
    st.success(f"📊 Predicted State of Health: **{soh:.2f}%**")

    if soh > 80:
        st.balloons()
        st.info("✅ Battery Health is Good")
    elif soh > 60:
        st.warning("⚠️ Battery Health is Fair")
    else:
        st.error("❌ Battery Health is Poor")

