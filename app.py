"""Servidor web Flask para optimizacion de perforacion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

_rop_model = None
_torque_model = None
_vib_model = None
_preprocessor = None


def get_models():
    global _rop_model, _torque_model, _vib_model, _preprocessor
    if _rop_model is None:
        from drilling_optimization.models.rop_predictor import ROPPredictor
        from drilling_optimization.models.torque_predictor import TorquePredictor
        from drilling_optimization.models.vibration_analyzer import VibrationAnalyzer
        _rop_model = ROPPredictor.load("outputs/models/rop_predictor.pkl")
        _torque_model = TorquePredictor.load("outputs/models/torque_predictor.pkl")
        _vib_model = VibrationAnalyzer.load("outputs/models/vibration_analyzer.pkl")
        with open("outputs/models/preprocessor.pkl", "rb") as f:
            _preprocessor = pickle.load(f)
    return _rop_model, _torque_model, _vib_model, _preprocessor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json
        import pandas as pd
        df = pd.DataFrame([{
            "depth_m": float(data.get("depth_m", 2000)),
            "wob_klbf": float(data.get("wob_klbf", 20)),
            "rpm": int(data.get("rpm", 120)),
            "flow_rate_gpm": float(data.get("flow_rate_gpm", 500)),
            "mud_weight_ppg": float(data.get("mud_weight_ppg", 10.5)),
            "formation": data.get("formation", "arena"),
            "bit_type": data.get("bit_type", "polycrystalline"),
            "bit_diameter_in": float(data.get("bit_diameter_in", 12.25)),
            "mud_viscosity_cp": float(data.get("mud_viscosity_cp", 50)),
            "hyd_pressure_psi": float(data.get("hyd_pressure_psi", 3000)),
        }])

        rop_model, torque_model, vib_model, preprocessor = get_models()
        X = preprocessor.transform(df)

        rop_pred = float(rop_model.predict(X)[0])
        torque_pred = float(torque_model.predict(X)[0])
        vib_result = vib_model.predict(X)

        return jsonify({
            "status": "ok",
            "rop_ft_hr": round(rop_pred, 2),
            "torque_klft": round(torque_pred, 2),
            "vibration_g": round(float(vib_result["vibration_g"][0]), 3),
            "vibration_severity": vib_result["severity_label"][0],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    try:
        data = request.json
        import pandas as pd

        rop_model, torque_model, vib_model, preprocessor = get_models()

        base = {
            "depth_m": float(data.get("depth_m", 2000)),
            "flow_rate_gpm": float(data.get("flow_rate_gpm", 500)),
            "mud_weight_ppg": float(data.get("mud_weight_ppg", 10.5)),
            "formation": data.get("formation", "arena"),
            "bit_type": data.get("bit_type", "polycrystalline"),
            "bit_diameter_in": float(data.get("bit_diameter_in", 12.25)),
            "mud_viscosity_cp": float(data.get("mud_viscosity_cp", 50)),
            "hyd_pressure_psi": float(data.get("hyd_pressure_psi", 3000)),
        }

        best_rop = 0
        best_params = {}
        best_torque = 0

        for wob in np.arange(10, 35, 5):
            for rpm in np.arange(80, 160, 20):
                row = {**base, "wob_klbf": wob, "rpm": rpm}
                df = pd.DataFrame([row])
                X = preprocessor.transform(df)
                rop = float(rop_model.predict(X)[0])
                torque = float(torque_model.predict(X)[0])
                vib = float(vib_model.predict(X)["vibration_g"][0])

                if vib < 2.5 and rop > best_rop:
                    best_rop = rop
                    best_torque = torque
                    best_params = {"wob_klbf": float(wob), "rpm": int(rpm)}

        return jsonify({
            "status": "ok",
            "optimal_wob": best_params.get("wob_klbf"),
            "optimal_rpm": best_params.get("rpm"),
            "predicted_rop": round(best_rop, 2),
            "predicted_torque": round(best_torque, 2),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/models")
def api_models():
    rop_model, torque_model, vib_model, _ = get_models()
    return jsonify({
        "status": "ok",
        "rop_model": rop_model.best_name,
        "torque_model": torque_model.best_name,
        "vibration_status": "trained" if vib_model.trained else "not_trained",
    })


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "service": "drilling-optimization"})


if __name__ == "__main__":
    print("=" * 60)
    print("  Servidor Web - Optimizacion de Perforacion")
    print("=" * 60)
    print("  Cargando modelos...")
    get_models()
    print("  Servidor iniciando en http://127.0.0.1:5004")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5004, debug=True)
