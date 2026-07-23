# User Guide - Drilling Optimization

## Overview
Machine Learning system for oil drilling parameter optimization. Predicts Rate of Penetration (ROP), drilling torque, and vibration severity. Automatically optimizes WOB and RPM for maximum drilling efficiency with safe vibration levels.

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/kcabrera83/drilling-optimization.git
cd drilling-optimization
pip install -r requirements.txt
```

### Training Models
```bash
python train.py
```
Generates 5000 synthetic drilling records, trains 3 models, saves to `outputs/models/`.

### Starting the Server
```bash
python app.py
```
Open http://localhost:5004 in your browser.

## Dashboard Features
- **Drilling Prediction**: Input drilling parameters and get ROP, torque, and vibration predictions
- **Parameter Optimization**: Automatically find optimal WOB and RPM for maximum ROP
- **Model Information**: View trained model details and performance metrics
- **Interactive Forms**: Real-time prediction with validation

## Drilling Parameters

### Input Parameters
| Parameter | Unit | Description |
|-----------|------|-------------|
| depth_m | m | Well depth |
| wob_klbf | klbf | Weight on bit |
| rpm | RPM | Rotation speed |
| flow_rate_gpm | GPM | Mud flow rate |
| mud_weight_ppg | ppg | Mud density |
| formation | - | Rock type (arena, caliza, lutita, dolomita) |
| bit_type | - | Drill bit type (polycrystalline, roller_cone, pdc) |
| bit_diameter_in | in | Bit diameter |
| mud_viscosity_cp | cP | Mud viscosity |
| hyd_pressure_psi | psi | Hydraulic pressure |

### Output Parameters
| Parameter | Unit | Description |
|-----------|------|-------------|
| rop_ft_hr | ft/hr | Rate of penetration |
| torque_klft | klft | Drilling torque |
| vibration_g | g | Vibration acceleration |
| vibration_severity | - | low / medium / high / critical |

## API Usage

### Using curl
```bash
# Predict drilling parameters
curl -X POST http://localhost:5004/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "depth_m": 2000,
    "wob_klbf": 20,
    "rpm": 120,
    "flow_rate_gpm": 500,
    "mud_weight_ppg": 10.5,
    "formation": "arena",
    "bit_type": "polycrystalline",
    "bit_diameter_in": 12.25
  }'

# Optimize drilling parameters
curl -X POST http://localhost:5004/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "depth_m": 2000,
    "flow_rate_gpm": 500,
    "mud_weight_ppg": 10.5,
    "formation": "arena"
  }'

# Get model info
curl http://localhost:5004/api/models
```

### Using Python
```python
import requests

# Predict drilling parameters
response = requests.post("http://localhost:5004/api/predict", json={
    "depth_m": 2000, "wob_klbf": 20, "rpm": 120,
    "flow_rate_gpm": 500, "mud_weight_ppg": 10.5,
    "formation": "arena", "bit_type": "polycrystalline"
})
result = response.json()
print(f"ROP: {result['rop_ft_hr']} ft/hr")
print(f"Torque: {result['torque_klft']} klft")
print(f"Vibration: {result['vibration_g']} g ({result['vibration_severity']})")

# Optimize parameters
response = requests.post("http://localhost:5004/api/optimize", json={
    "depth_m": 2000, "flow_rate_gpm": 500,
    "mud_weight_ppg": 10.5, "formation": "arena"
})
opt = response.json()
print(f"Optimal WOB: {opt['optimal_wob']} klbf, RPM: {opt['optimal_rpm']}")
```

## Training Results
- **Dataset**: 5000 synthetic drilling records
- **ROP Predictor**: GradientBoosting (R2 > 0.95)
- **Torque Predictor**: GradientBoosting (R2 > 0.95)
- **Vibration Analyzer**: Classification + regression (Accuracy > 0.90)
