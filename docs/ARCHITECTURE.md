# Architecture - Drilling Optimization

## System Overview
```
                    +-------------------+
                    |   Flask Server    |
                    |   (app.py)        |
                    |   Port 5004       |
                    +--------+----------+
                             |
          +------------------+------------------+
          |                  |                  |
+---------v-------+  +------v--------+  +------v--------+
| ROP             |  | Torque        |  | Vibration     |
| Predictor       |  | Predictor     |  | Analyzer      |
| (GB Regressor)  |  | (GB Regressor)|  | (CLS + REG)   |
+---------+-------+  +------+--------+  +------+--------+
          |                  |                  |
+---------v------------------v------------------v-------+
|              DrillingPreprocessor                      |
|       (Encoding + Scaling + Feature Engineering)       |
+------------------------+------------------------------+
                         |
               +---------v-----------+
               |  Synthetic Dataset  |
               |  (5000 records)     |
               +--------------------+
```

## Components

### Data Layer
- **Data Source**: Synthetic drilling data generator (`DrillingDataGenerator`) producing 5000 records
- **Formations**: arena (sand), caliza (limestone), lutita (shale), dolomita (dolomite)
- **Bit Types**: polycrystalline, roller_cone, pdc
- **Preprocessing**: One-hot encoding for categorical features, standard scaling

### Model Layer

#### ROP Predictor
- **Algorithm**: GradientBoosting (best) among Random Forest, ExtraTrees, SVR, MLP
- **Target**: Rate of Penetration (ft/hr)
- **Features**: depth, WOB, RPM, flow rate, mud weight, formation, bit type, bit diameter, mud viscosity, hydraulic pressure
- **Metrics**: R2, MAE, MAPE

#### Torque Predictor
- **Algorithm**: GradientBoosting (best)
- **Target**: Drilling torque (klft)
- **Same features as ROP**
- **Metrics**: R2, MAE, MAPE

#### Vibration Analyzer
- **Algorithm**: Dual-purpose model
  - **Classification**: Vibration severity level (low/medium/high/critical)
  - **Regression**: Continuous vibration value (g)
- **Features**: Same drilling parameters
- **Metrics**: Classification accuracy, Regression R2

### Optimization Engine
- **Method**: Grid search over WOB (10-30 klbf) and RPM (80-140)
- **Objective**: Maximize predicted ROP
- **Constraint**: Vibration < 2.5 g (safe operating limit)
- **Output**: Optimal WOB, RPM, predicted ROP and torque

### API Layer
- **Framework**: Flask
- **Endpoints**: 5 REST endpoints (health, models, predict, optimize, docs)
- **Model Loading**: Lazy loading with pickle deserialization

### Dashboard Layer
- **Frontend**: Flask + HTML/CSS/JS
- **Charts**: Drilling parameter visualizations, optimization results

## Data Flow

1. **Input**: Drilling parameters (depth, WOB, RPM, mud properties, formation, bit)
2. **Preprocessing**: `DrillingPreprocessor` encodes categoricals and scales numerics
3. **Prediction**: Three models predict ROP, torque, and vibration simultaneously
4. **Optimization**: Grid search finds optimal WOB/RPM combinations
5. **Response**: Predicted values and optimal parameters returned as JSON

## Training Pipeline
1. Generate synthetic drilling data (5000 records)
2. Preprocess with encoding and scaling
3. Split 80/20 train/test
4. Train ROP, Torque, and Vibration models
5. Evaluate each model (R2, MAE, MAPE, Accuracy)
6. Save models and preprocessor to `outputs/models/`

## File Structure
```
drilling-optimization/
├── drilling_optimization/
│   ├── data_generator.py       # Synthetic drilling data
│   ├── models/
│   │   ├── rop_predictor.py    # ROP prediction
│   │   ├── torque_predictor.py # Torque prediction
│   │   └── vibration_analyzer.py # Vibration analysis
│   └── utils/
│       └── preprocessor.py     # Data preprocessing
├── outputs/models/             # Trained models
├── templates/index.html        # Dashboard
├── app.py                      # Flask server
├── train.py                    # Training pipeline
└── test_api.py                 # API tests
```
