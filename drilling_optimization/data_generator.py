"""Generador de datos sinteticos para optimizacion de perforacion."""

import numpy as np
import pandas as pd


class DrillingDataGenerator:
    """Genera datos sinteticos realistas de operaciones de perforacion."""

    FORMATIONS = ["arcilla", "arena", "caliza", "dolomita", "anhidrita", "shale"]
    BIT_TYPES = ["polycrystalline", "insert", "steel_tooth"]

    def __init__(self, n_samples=5000, random_state=42):
        self.n_samples = n_samples
        self.rng = np.random.RandomState(random_state)

    def generate(self):
        n = self.n_samples
        rng = self.rng

        depth = rng.uniform(500, 4000, n)
        wob = rng.uniform(5, 40, n)
        rpm = rng.uniform(60, 180, n)
        flow_rate = rng.uniform(200, 800, n)
        mud_weight = rng.uniform(8.5, 14.0, n)
        torque = rng.uniform(5, 45, n)
        formation = rng.choice(self.FORMATIONS, n)
        bit_type = rng.choice(self.BIT_TYPES, n)
        bit_diameter = rng.choice([8.5, 12.25, 17.5], n)
        mud_viscosity = rng.uniform(30, 80, n)
        hyd_pressure = rng.uniform(1000, 6000, n)

        formation_hardness = {
            "arcilla": 0.3, "arena": 0.5, "caliza": 0.7,
            "dolomita": 0.8, "anhidrita": 0.75, "shale": 0.4,
        }
        hardness = np.array([formation_hardness[f] for f in formation])

        rop_base = (
            50
            * (wob / 20) ** 0.6
            * (rpm / 120) ** 0.4
            * (flow_rate / 500) ** 0.3
            / (hardness + 0.1)
            / (mud_weight / 10) ** 0.5
            * (bit_diameter / 12.25) ** (-0.3)
        )
        rop = rop_base + rng.normal(0, rop_base * 0.1, n)
        rop = np.clip(rop, 1, 200)

        torque_pred = (
            10
            * (wob / 20) ** 0.8
            * (rpm / 120) ** 0.3
            * (hardness + 0.2)
            * (bit_diameter / 12.25) ** 0.5
            + rng.normal(0, 2, n)
        )
        torque_final = torque_pred + rng.uniform(-3, 3, n)
        torque_final = np.clip(torque_final, 2, 60)

        vibration = (
            0.5
            * (wob / 20) ** 1.2
            * (rpm / 120) ** 1.5
            * (1 + hardness)
            + rng.normal(0, 0.3, n)
        )
        vibration = np.clip(vibration, 0.1, 5.0)

        MSE = (
            (wob * 1000 / (np.pi * (bit_diameter / 2) ** 2))
            * (1 + 0.1 * torque / wob)
            * (1 + vibration)
        )
        MSE = MSE + rng.normal(0, MSE * 0.05, n)

        specific_energy = MSE / (rop + 0.1) + rng.normal(0, 0.5, n)
        specific_energy = np.clip(specific_energy, 10, 500)

        df = pd.DataFrame({
            "depth_m": np.round(depth, 1),
            "wob_klbf": np.round(wob, 2),
            "rpm": np.round(rpm, 1),
            "flow_rate_gpm": np.round(flow_rate, 1),
            "mud_weight_ppg": np.round(mud_weight, 2),
            "torque_klft": np.round(torque_final, 2),
            "formation": formation,
            "bit_type": bit_type,
            "bit_diameter_in": bit_diameter,
            "mud_viscosity_cp": np.round(mud_viscosity, 1),
            "hyd_pressure_psi": np.round(hyd_pressure, 1),
            "rop_ft_hr": np.round(rop, 2),
            "vibration_g": np.round(vibration, 3),
            "mse_psi": np.round(MSE, 1),
            "specific_energy_mse": np.round(specific_energy, 2),
        })
        return df

    def save(self, path="outputs/data/drilling_data.csv"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.generate()
        df.to_csv(path, index=False)
        return df
