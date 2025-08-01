import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# Load AI model
try:
    model = joblib.load("soh_predictor_6features.joblib")
    model_status = "✅ Model loaded successfully."
except Exception as e:
    model = None
    model_status = f"❌ Failed to load model: {e}"

# App config
st.set_page_config(page_title="Battery SoH Simulator", layout="centered", page_icon="🔋")

# Wizideaz branding animation
st.markdown(
    """
    <style>
    .header-title {
        font-size: 50px;
        font-weight: 800;
        color: #27AE60;
        animation: glow 2s infinite alternate;
        text-align: center;
    }

    @keyframes glow {
        from {
            text-shadow: 0 0 5px #27AE60;
        }
        to {
            text-shadow: 0 0 20px #2ECC71;
        }
    }
    </style>
    <div class="header-title">🔋 Wizideaz Battery AI Simulator 🔋</div>
    """,
    unsafe_allow_html=True
)

st.write("")
st.success(model_status)

st.subheader("📊 Enter battery parameters:")

# User inputs
voltage_measured = st.number_input("Voltage Measured (V)", min_value=0.0, step=0.1)
current_measured = st.number_input("Current Measured (A)", min_value=0.0, step=0.1)
temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.5)
current_load = st.number_input("Current Load (A)", min_value=0.0, step=0.1)
voltage_load = st.number_input("Voltage Load (V)", min_value=0.0, step=0.1)
time_sec = st.number_input("Time (s)", min_value=0.0, step=1.0)

if st.button("🔍 Simulate SoH"):
    if model is None:
        st.error("AI model is not loaded. Please check deployment.")
    else:
        try:
            input_features = np.array([[voltage_measured, current_measured, temperature,
                                        current_load, voltage_load, time_sec]])
            soh = model.predict(input_features)[0]

            # Assume model predicts in 0-1 range; scale to 0-100%
            if soh <= 1.0:
                soh *= 100

            st.metric("🔋 Predicted State of Health (SoH)", f"{soh:.2f}%")

            # Health interpretation
            if soh > 85:
                st.success("✅ Battery is in **excellent** health!")
            elif 60 <= soh <= 85:
                st.warning("⚠️ Battery is in **fair** condition. Monitor usage.")
            elif 40 <= soh < 60:
                st.error("❗ Battery is in **poor** condition. Consider servicing.")
            else:
                st.error("🚨 Battery is in **critical** condition! Replace soon.")

            # Extra warnings
            if temperature > 50:
                st.warning("🔥 **High temperature detected!** Reduce load to prevent thermal stress.")
            if voltage_measured < 3.0:
                st.warning("⚡ **Low voltage warning!** Recharge recommended.")

        except Exception as e:
            st.error(f"Model prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("🔧 Built with ❤️ by **Wizideaz** | Battery AI Powered | 2025")
