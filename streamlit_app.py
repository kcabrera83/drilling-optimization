import streamlit as st
import joblib, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="Drilling Optimization", layout="wide")
st.title("Drilling Optimization")
st.markdown("Maximize ROP while minimizing torque and vibration")

@st.cache_resource
def load_models():
    base = Path(__file__).parent / 'outputs' / 'models'
    return {'rop': joblib.load(base / 'rop_predictor.pkl'), 'torque': joblib.load(base / 'torque_predictor.pkl'), 'vibration': joblib.load(base / 'vibration_analyzer.pkl')}

models = load_models()

def predict(name, x):
    m = models[name]
    if isinstance(m, dict):
        X = m['scaler'].transform(x)
        p = m['model'].predict(X)
        if 'label_encoder' in m:
            return m['label_encoder'].inverse_transform(p)[0]
        return float(p[0])
    return float(m.predict(x)[0])

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader('Parameters')
    depth = st.slider('Depth', 500, 5000, 2750)
    wob = st.slider('Wob', 10, 80, 45)
    rpm = st.slider('Rpm', 30, 200, 115)
    flow = st.slider('Flow', 200, 1500, 850)
    mud = st.slider('Mud', 8, 20, 14)
    formation = st.selectbox('Formation', ['sandstone','shale','limestone','dolomite'])
    bit = st.selectbox('Bit', ['roller_cone','pdc','diamond'])
    diam = st.slider('Diam', 6, 26, 16)
    run = st.button('Run Prediction', use_container_width=True)

with col2:
    if run:
        x = np.array([[depth, wob, rpm, flow, mud, formation, bit, diam]])
        results = {}
        results['rop'] = predict('rop', x)
        results['torque'] = predict('torque', x)
        results['vibration'] = predict('vibration', x)
        st.subheader('Results')
        rcols = st.columns(len(results))
        for i, (k, v) in enumerate(results.items()):
            label = k.replace('_', ' ').title()
            if isinstance(v, str):
                rcols[i].metric(label, v)
            else:
                rcols[i].metric(label, f'{v:.2f}')
        # Plot
        fig, ax = plt.subplots()
        names = [k.replace('_',' ').title() for k in results]
        vals = [float(v) if isinstance(v, (int,float,str)) and str(v).replace('.','').replace('-','').isdigit() else 0 for v in results.values()]
        if any(v != 0 for v in vals):
            ax.bar(names, vals, color=['#0077B6','#00B4D8','#90E0EF'])
            ax.set_ylabel('Value')
            st.pyplot(fig)