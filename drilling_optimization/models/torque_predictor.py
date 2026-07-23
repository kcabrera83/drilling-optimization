"""Modelo de prediccion de torque."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


class TorquePredictor:
    def __init__(self):
        self.models = {
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
        }
        self.best_model = None
        self.best_name = None
        self.results = {}

    def train(self, X, y):
        for name, model in self.models.items():
            model.fit(X, y)
            pred = model.predict(X)
            r2 = r2_score(y, pred)
            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            self.results[name] = {"r2": r2, "mae": mae, "rmse": rmse}

        best_name = max(self.results, key=lambda k: self.results[k]["r2"])
        self.best_model = self.models[best_name]
        self.best_name = best_name
        return self.results

    def predict(self, X):
        return self.best_model.predict(X)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "r2": r2_score(y, pred),
            "mae": mean_absolute_error(y, pred),
            "rmse": np.sqrt(mean_squared_error(y, pred)),
            "mape": np.mean(np.abs((y - pred) / (y + 1e-8))) * 100,
        }

    def feature_importance(self):
        if hasattr(self.best_model, "feature_importances_"):
            return self.best_model.feature_importances_
        return None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
