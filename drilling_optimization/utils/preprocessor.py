"""Modulo de preprocesamiento para datos de perforacion."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DrillingPreprocessor:
    NUMERIC_FEATURES = [
        "depth_m", "wob_klbf", "rpm", "flow_rate_gpm", "mud_weight_ppg",
        "bit_diameter_in", "mud_viscosity_cp", "hyd_pressure_psi",
    ]
    CATEGORICAL_FEATURES = ["formation", "bit_type"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.fitted = False

    def fit(self, df):
        self.scaler.fit(df[self.NUMERIC_FEATURES])
        for col in self.CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le
        self.fitted = True
        return self

    def transform(self, df):
        X_num = self.scaler.transform(df[self.NUMERIC_FEATURES])
        X_cat = np.column_stack([
            self.label_encoders[col].transform(df[col])
            for col in self.CATEGORICAL_FEATURES
        ])
        return np.hstack([X_num, X_cat])

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self):
        return self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
