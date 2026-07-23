# Drilling Optimization

Sistema de machine learning para optimizacion de parametros de perforacion petrolera.

## Modelos ML

- **ROP Predictor**: Gradient Boosting / Random Forest para prediccion de Rate of Penetration (ft/hr)
- **Torque Predictor**: Prediccion de torque de perforacion (klft)
- **Vibration Analyzer**: Clasificacion de severidad de vibracion + prediccion continua (g)

## Requisitos

```bash
pip install -r requirements.txt
```

## Entrenamiento

```bash
python train.py
```

Genera datos sinteticos (5000 registros), entrena los 3 modelos y guarda en `outputs/models/`.

## Dashboard

```bash
python app.py
```

Disponible en http://127.0.0.1:5004

### Endpoints

| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| GET | `/` | Dashboard web |
| POST | `/api/predict` | Predecir ROP, torque, vibracion |
| POST | `/api/optimize` | Optimizar WOB y RPM automaticamente |
| GET | `/api/models` | Info de modelos entrenados |
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

## Estructura

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

Elaborado por Ing. Kelvin Cabrera
