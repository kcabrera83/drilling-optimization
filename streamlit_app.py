import streamlit as st
import pickle
import joblib
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Drilling Optimization", layout="wide")
st.title("Drilling Optimization")
st.markdown("Optimize drilling parameters (WOB, RPM) to maximize ROP and minimize vibrations.")

@st.cache_resource
def load_models():
    d = Path(__file__).parent / "outputs" / "models"
    return {k: joblib.load(d / v) for k, v in [("rop", "rop_predictor.pkl"), ("torque", "torque_predictor.pkl"), ("vibration", "vibration_analyzer.pkl")]}

models = load_models()

st.sidebar.header("Input Parameters")
depth_m = st.sidebar.slider("Depth M", 500, 5000, 2750)
wob_klbf = st.sidebar.slider("Wob Klbf", 10, 80, 45)
rpm = st.sidebar.slider("Rpm", 30, 200, 115)
flow_rate_gpm = st.sidebar.slider("Flow Rate Gpm", 200, 1500, 850)
mud_weight_ppg = st.sidebar.slider("Mud Weight Ppg", 8, 20, 14)
formation = st.sidebar.selectbox("Formation", ['sandstone', 'shale', 'limestone', 'dolomite'])
bit_type = st.sidebar.selectbox("Bit Type", ['roller_cone', 'pdc', 'diamond'])
bit_diameter_in = st.sidebar.slider("Bit Diameter In", 6, 26, 16)

if st.sidebar.button("Run Prediction"):
    try:
        features = np.array([[depth_m, wob_klbf, rpm, flow_rate_gpm, mud_weight_ppg, formation, bit_type, bit_diameter_in]])
        m = models["rop"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Rop", result if isinstance(result, str) else f"{result:.4f}")
        m = models["torque"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Torque", result if isinstance(result, str) else f"{result:.4f}")
        m = models["vibration"]
        if isinstance(m, dict):
            X = m.get("scaler").transform(features) if m.get("scaler") else features
            pred = m["model"].predict(X)
            if "label_encoder" in m:
                result = m["label_encoder"].inverse_transform(pred)[0]
            else:
                result = pred[0]
        else:
            result = m.predict(features)[0]
        st.metric("Vibration", result if isinstance(result, str) else f"{result:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")
