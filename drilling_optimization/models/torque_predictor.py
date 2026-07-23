import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle


class TorquePredictor:
    def __init__(self):
        self.model = None
        self.best_name = "lightgbm"
        self.results = {}

    def train(self, X, y):
        X_train, X_val = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
        y_train, y_val = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'verbose': -1,
                'random_state': 42,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            pred = model.predict(X_val)
            return mean_squared_error(y_val, pred, squared=False)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)

        self.model = lgb.LGBMRegressor(**study.best_params, verbose=-1, random_state=42)
        self.model.fit(X, y)

        pred = self.model.predict(X)
        self.results["lightgbm"] = {
            "r2": r2_score(y, pred),
            "mae": mean_absolute_error(y, pred),
            "rmse": np.sqrt(mean_squared_error(y, pred)),
        }
        return self.results

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return {
            "r2": r2_score(y, pred),
            "mae": mean_absolute_error(y, pred),
            "rmse": np.sqrt(mean_squared_error(y, pred)),
            "mape": np.mean(np.abs((y - pred) / (y + 1e-8))) * 100,
        }

    def feature_importance(self):
        if self.model is not None:
            return self.model.feature_importances_
        return None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
