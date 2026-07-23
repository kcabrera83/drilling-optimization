# Drilling Optimization

Machine learning system for oil drilling parameter optimization using LightGBM, Optuna, and SHAP for explainable predictions.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | **LightGBM** - gradient boosting for ROP prediction |
| Hyperparameter Tuning | **Optuna** - Bayesian optimization |
| Model Explainability | **SHAP** - SHapley Additive exPlanations |
| Web Server | **FastAPI** + uvicorn |
| Data Processing | pandas, numpy, joblib |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |

### Key Libraries
- LightGBM - Fast gradient boosting framework
- Optuna - Automatic hyperparameter optimization
- SHAP - Model interpretation and feature importance
- FastAPI - Modern async web framework

## ML Models

- **ROP Predictor**: LightGBM for Rate of Penetration prediction (ft/hr)
- **Torque Predictor**: Drilling torque prediction (klft)
- **Vibration Analyzer**: Vibration severity classification + continuous prediction (g)

## Requirements

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

Generates synthetic data (5000 records), trains the 3 models, and saves to `outputs/models/`.

## Dashboard

```bash
python app.py
```

Available at http://127.0.0.1:5004

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Web dashboard |
| POST | `/api/predict` | Predict ROP, torque, vibration |
| POST | `/api/optimize` | Automatically optimize WOB and RPM |
| GET | `/api/models` | Trained model info |
| GET | `/api/health` | Health check |

## API Predict

```json
POST /api/predict
{
    "depth_m": 2000,
    "wob_klbf": 20,
    "rpm": 120,
    "flow_rate_gpm": 500,
    "mud_weight_ppg": 10.5,
    "formation": "arena",
    "bit_type": "polycrystalline",
    "bit_diameter_in": 12.25
}
```

## API Optimize

```json
POST /api/optimize
{
    "depth_m": 2000,
    "flow_rate_gpm": 500,
    "mud_weight_ppg": 10.5,
    "formation": "arena"
}
```

Returns optimal WOB and RPM for maximum ROP with safe vibration levels.

## Tests

```bash
python test_api.py
```

## Structure

```
drilling-optimization/
├── drilling_optimization/
│   ├── data_generator.py
│   ├── utils/preprocessor.py
│   └── models/
│       ├── rop_predictor.py
│       ├── torque_predictor.py
│       └── vibration_analyzer.py
├── outputs/models/
├── templates/
├── app.py
├── train.py
├── test_api.py
└── requirements.txt
```

---

Elaborado por Ing. Kelvin Cabrera
