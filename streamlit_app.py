import streamlit as st
import joblib
import numpy as np
import time

# === CONFIG ===
st.set_page_config(
    page_title="WIZIDEAZ Battery AI",
    page_icon="🔋",
    layout="centered"
)

# === BRAND COLORS ===
WIZI_YELLOW = "#f1c40f"

# === SPLASH SCREEN ===
if "splash_shown" not in st.session_state:
    st.markdown(
        f"""
        <div style='
            height:100vh;
            display:flex;
            align-items:center;
            justify-content:center;
            background-color:{WIZI_YELLOW};
            font-size:48px;
            font-weight:bold;
            color:black;
        '>WIZIDEAZ</div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(2)
    st.session_state.splash_shown = True
    st.rerun()

# === LOAD MODEL ===
try:
    model = joblib.load("soh_predictor_6features.joblib")
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model Load Failed: {e}")

# === APP HEADER ===
st.markdown(
    f"""
    <h2 style='text-align: center; color: {WIZI_YELLOW};'>
        Battery SoH Simulator
    </h2>
    <hr style="border-top: 2px solid {WIZI_YELLOW};" />
    """,
    unsafe_allow_html=True
)

# === INPUT FORM ===
st.subheader("Battery Parameters")
with st.form("soh_form"):
    voltage_measured = st.number_input("Voltage Measured (V)", min_value=0.0, step=0.1)
    current_measured = st.number_input("Current Measured (A)", min_value=0.0, step=0.1)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, step=0.5)
    current_load = st.number_input("Current Load (A)", min_value=0.0, step=0.1)
    voltage_load = st.number_input("Voltage Load (V)", min_value=0.0, step=0.1)
    time_sec = st.number_input("Time (s)", min_value=0.0, step=1.0)
    submitted = st.form_submit_button("Simulate SoH")

# === SIMULATE ===
if submitted:
    if not model_loaded:
        st.error("Model not loaded properly. Try again later.")
    else:
        with st.spinner("Running battery simulation..."):
            try:
                input_features = np.array([[voltage_measured, current_measured, temperature,
                                            current_load, voltage_load, time_sec]])
                soh = model.predict(input_features)[0]

                # Convert to %
                if soh <= 1.0:
                    soh *= 100

                # Determine status
                if soh > 85:
                    status = "Excellent health"
                    color = "success"
                elif 60 <= soh <= 85:
                    status = "Fair condition"
                    color = "warning"
                elif 40 <= soh < 60:
                    status = "Poor condition"
                    color = "danger"
                else:
                    status = "Critical! Replace battery"
                    color = "error"

                # Custom alert
                st.toast(f"Simulation Complete: {soh:.2f}% - {status}", icon="🔋")

                # Show result in modal-like block
                st.markdown(
                    f"""
                    <div style='
                        padding:20px;
                        margin-top:20px;
                        border:2px solid {WIZI_YELLOW};
                        border-radius:10px;
                        background-color:#fffbea;
                        text-align:center;
                    '>
                        <h3 style='color:{WIZI_YELLOW};'>Estimated SoH</h3>
                        <h1 style='font-size: 40px;'>{soh:.2f}%</h1>
                        <p>Status: <strong>{status}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if temperature > 50:
                    st.warning("⚠️ High temperature detected! May reduce battery life.")
                if voltage_measured < 3.0:
                    st.warning("⚡ Low voltage warning! Recharge recommended.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# === FOOTER ===
st.markdown("---")
st.markdown(
    f"<center style='color: gray;'>© 2025 WIZIDEAZ | Battery Intelligence</center>",
    unsafe_allow_html=True
)
