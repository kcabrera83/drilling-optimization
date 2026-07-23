"""Modelo de analisis de vibracion."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import pickle


class VibrationAnalyzer:
    VIBRATION_THRESHOLDS = [1.0, 2.0, 3.0]

    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        self.severity_regressor = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.trained = False

    def _label_severity(self, vibration):
        labels = np.zeros(len(vibration), dtype=int)
        labels[vibration >= 1.0] = 1
        labels[vibration >= 2.0] = 2
        labels[vibration >= 3.0] = 3
        return labels

    def train(self, X, y):
        labels = self._label_severity(y)
        self.classifier.fit(X, labels)
        self.severity_regressor.fit(X, y)
        self.trained = True

        pred_labels = self.classifier.predict(X)
        pred_vib = self.severity_regressor.predict(X)
        return {
            "classification_accuracy": accuracy_score(labels, pred_labels),
            "regression_r2": r2_score(y, pred_vib),
            "regression_mae": mean_absolute_error(y, pred_vib),
        }

    def predict(self, X):
        labels = self.classifier.predict(X)
        severity = self.severity_regressor.predict(X)
        severity_names = ["normal", "moderate", "high", "critical"]
        return {
            "severity_label": [severity_names[l] for l in labels],
            "vibration_g": severity,
        }

    def evaluate(self, X, y):
        labels = self._label_severity(y)
        pred_labels = self.classifier.predict(X)
        pred_vib = self.severity_regressor.predict(X)
        return {
            "classification_accuracy": accuracy_score(labels, pred_labels),
            "regression_r2": r2_score(y, pred_vib),
            "regression_mae": mean_absolute_error(y, pred_vib),
        }

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
