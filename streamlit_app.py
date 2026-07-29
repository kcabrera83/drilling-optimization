import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Drilling Optimization", layout="wide")
st.title("Drilling Optimization")
st.markdown("Optimize drilling parameters to maximize ROP and minimize vibrations.")

import joblib, numpy as np
d = Path(__file__).parent / 'outputs' / 'models'
models = {'rop': joblib.load(d / 'rop_predictor.pkl'), 'torque': joblib.load(d / 'torque_predictor.pkl'), 'vibration': joblib.load(d / 'vibration_analyzer.pkl')}

st.sidebar.header("Input Parameters")
depth_m = st.sidebar.slider('Depth M', 500, 5000, 2750)
wob_klbf = st.sidebar.slider('Wob Klbf', 10, 80, 45)
rpm = st.sidebar.slider('Rpm', 30, 200, 115)
flow_rate_gpm = st.sidebar.slider('Flow Rate Gpm', 200, 1500, 850)
mud_weight_ppg = st.sidebar.slider('Mud Weight Ppg', 8, 20, 14)
formation = st.sidebar.selectbox('Formation', ['sandstone','shale','limestone','dolomite'])
bit_type = st.sidebar.selectbox('Bit Type', ['roller_cone','pdc','diamond'])
bit_diameter_in = st.sidebar.slider('Bit Diameter In', 6, 26, 16)

if st.sidebar.button("Run"):
    try:
        x = np.array([[depth_m, wob_klbf, rpm, flow_rate_gpm, mud_weight_ppg, formation, bit_type, bit_diameter_in]])
        cols = st.columns(3)
        for i, (k, m) in enumerate(models.items()):
            X = m['scaler'].transform(x)
            p = m['model'].predict(X)
            if 'label_encoder' in m:
                val = m['label_encoder'].inverse_transform(p)[0]
            else:
                val = f'{p[0]:.2f}'
            cols[i].metric(k.title(), val)
    except Exception as e:
        st.error(str(e))