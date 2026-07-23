# API Documentation - Drilling Optimization

## Base URL
```
http://localhost:5004
```

## Endpoints

### GET /
Main dashboard with interactive web interface.

**Response:** HTML page with drilling optimization panels.

---

### GET /api/health
Service health check.

**Response (200):**
```json
{"status": "ok", "service": "drilling-optimization"}
```

---

### GET /api/models
Information about trained models.

**Response (200):**
```json
{
  "status": "ok",
  "rop_model": "GradientBoosting",
  "torque_model": "GradientBoosting",
  "vibration_status": "trained"
}
```

---

### POST /api/predict
Predict Rate of Penetration (ROP), torque, and vibration severity.

**Request:**
```json
{
  "depth_m": 2000,
  "wob_klbf": 20,
  "rpm": 120,
  "flow_rate_gpm": 500,
  "mud_weight_ppg": 10.5,
  "formation": "arena",
  "bit_type": "polycrystalline",
  "bit_diameter_in": 12.25,
  "mud_viscosity_cp": 50,
  "hyd_pressure_psi": 3000
}
```

**Response (200):**
```json
{
  "status": "ok",
  "rop_ft_hr": 45.30,
  "torque_klft": 12.50,
  "vibration_g": 1.200,
  "vibration_severity": "low"
}
```

**Error Response (400):**
```json
{"status": "error", "message": "Error description"}
```

---

### POST /api/optimize
Automatically optimize WOB (Weight on Bit) and RPM for maximum ROP with safe vibration levels (< 2.5 g).

**Request:**
```json
{
  "depth_m": 2000,
  "flow_rate_gpm": 500,
  "mud_weight_ppg": 10.5,
  "formation": "arena",
  "bit_type": "polycrystalline",
  "bit_diameter_in": 12.25,
  "mud_viscosity_cp": 50,
  "hyd_pressure_psi": 3000
}
```

**Response (200):**
```json
{
  "status": "ok",
  "optimal_wob": 25.0,
  "optimal_rpm": 140,
  "predicted_rop": 52.80,
  "predicted_torque": 14.20
}
```

**Optimization Grid:**
- WOB: 10 to 30 klbf (step 5)
- RPM: 80 to 140 (step 20)
- Constraint: vibration < 2.5 g

**Error Response (400):**
```json
{"status": "error", "message": "Error description"}
```

---

### GET /api/docs
OpenAPI 3.0 self-documentation.

**Response (200):**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "Drilling Optimization", "version": "1.0.0"},
  "paths": { ... }
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - invalid input or processing error |
| 500 | Internal server error |
