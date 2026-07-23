"""FastAPI web server for drilling optimization."""

import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

app = FastAPI(
    title="Drilling Optimization",
    description="ROP prediction, torque prediction, vibration analysis, and drilling parameter optimization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

models: dict[str, Any] = {}


@app.on_event("startup")
async def load_models():
    from drilling_optimization.models.rop_predictor import ROPPredictor
    from drilling_optimization.models.torque_predictor import TorquePredictor
    from drilling_optimization.models.vibration_analyzer import VibrationAnalyzer
    try:
        models["rop"] = ROPPredictor.load("outputs/models/rop_predictor.pkl")
        models["torque"] = TorquePredictor.load("outputs/models/torque_predictor.pkl")
        models["vibration"] = VibrationAnalyzer.load("outputs/models/vibration_analyzer.pkl")
        with open("outputs/models/preprocessor.pkl", "rb") as f:
            models["preprocessor"] = pickle.load(f)
    except Exception as e:
        print(f"  Error loading models: {e}")


class DrillPredictRequest(BaseModel):
    depth_m: float = 2000.0
    wob_klbf: float = 20.0
    rpm: int = 120
    flow_rate_gpm: float = 500.0
    mud_weight_ppg: float = 10.5
    formation: str = "arena"
    bit_type: str = "polycrystalline"
    bit_diameter_in: float = 12.25
    mud_viscosity_cp: float = 50.0
    hyd_pressure_psi: float = 3000.0


class DrillOptimizeRequest(BaseModel):
    depth_m: float = 2000.0
    flow_rate_gpm: float = 500.0
    mud_weight_ppg: float = 10.5
    formation: str = "arena"
    bit_type: str = "polycrystalline"
    bit_diameter_in: float = 12.25
    mud_viscosity_cp: float = 50.0
    hyd_pressure_psi: float = 3000.0


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "drilling-optimization"}


@app.get("/api/models")
async def api_models():
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "ok",
        "rop_model": models["rop"].best_name,
        "torque_model": models["torque"].best_name,
        "vibration_status": "trained" if models["vibration"].trained else "not_trained",
    }


@app.post("/api/predict")
async def api_predict(request: DrillPredictRequest):
    if not all(k in models for k in ("rop", "torque", "vibration", "preprocessor")):
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        df = pd.DataFrame([{
            "depth_m": request.depth_m,
            "wob_klbf": request.wob_klbf,
            "rpm": request.rpm,
            "flow_rate_gpm": request.flow_rate_gpm,
            "mud_weight_ppg": request.mud_weight_ppg,
            "formation": request.formation,
            "bit_type": request.bit_type,
            "bit_diameter_in": request.bit_diameter_in,
            "mud_viscosity_cp": request.mud_viscosity_cp,
            "hyd_pressure_psi": request.hyd_pressure_psi,
        }])
        X = models["preprocessor"].transform(df)
        rop_pred = float(models["rop"].predict(X)[0])
        torque_pred = float(models["torque"].predict(X)[0])
        vib_result = models["vibration"].predict(X)
        return {
            "status": "ok",
            "rop_ft_hr": round(rop_pred, 2),
            "torque_klft": round(torque_pred, 2),
            "vibration_g": round(float(vib_result["vibration_g"][0]), 3),
            "vibration_severity": vib_result["severity_label"][0],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/optimize")
async def api_optimize(request: DrillOptimizeRequest):
    if not all(k in models for k in ("rop", "torque", "vibration", "preprocessor")):
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        base = {
            "depth_m": request.depth_m,
            "flow_rate_gpm": request.flow_rate_gpm,
            "mud_weight_ppg": request.mud_weight_ppg,
            "formation": request.formation,
            "bit_type": request.bit_type,
            "bit_diameter_in": request.bit_diameter_in,
            "mud_viscosity_cp": request.mud_viscosity_cp,
            "hyd_pressure_psi": request.hyd_pressure_psi,
        }
        best_rop = 0
        best_params = {}
        best_torque = 0
        best_vib = float("inf")
        all_combos = []
        for wob in np.arange(10, 35, 5):
            for rpm in np.arange(80, 160, 20):
                row = {**base, "wob_klbf": wob, "rpm": rpm}
                df = pd.DataFrame([row])
                X = models["preprocessor"].transform(df)
                rop = float(models["rop"].predict(X)[0])
                torque = float(models["torque"].predict(X)[0])
                vib = float(models["vibration"].predict(X)["vibration_g"][0])
                all_combos.append({"wob_klbf": float(wob), "rpm": int(rpm), "rop": rop, "torque": torque, "vibration_g": vib})
                if vib < 2.5 and rop > best_rop:
                    best_rop = rop
                    best_torque = torque
                    best_vib = vib
                    best_params = {"wob_klbf": float(wob), "rpm": int(rpm)}

        if not best_params:
            fallback = min(all_combos, key=lambda c: c["vibration_g"])
            return {
                "status": "ok",
                "optimal_wob": fallback["wob_klbf"],
                "optimal_rpm": fallback["rpm"],
                "predicted_rop": round(fallback["rop"], 2),
                "predicted_torque": round(fallback["torque"], 2),
                "warning": "No WOB/RPM combination met the vibration threshold (< 2.5g). Returning lowest-vibration option.",
            }

        return {
            "status": "ok",
            "optimal_wob": best_params.get("wob_klbf"),
            "optimal_rpm": best_params.get("rpm"),
            "predicted_rop": round(best_rop, 2),
            "predicted_torque": round(best_torque, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5004)

