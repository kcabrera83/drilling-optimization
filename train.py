import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drilling_optimization.data_generator import DrillingDataGenerator
from drilling_optimization.utils.preprocessor import DrillingPreprocessor
from drilling_optimization.models.rop_predictor import ROPPredictor
from drilling_optimization.models.torque_predictor import TorquePredictor
from drilling_optimization.models.vibration_analyzer import VibrationAnalyzer


def main():
    print("=" * 60)
    print("  Drilling Optimization - LightGBM + SHAP + Optuna")
    print("=" * 60)

    print("\n[1/6] Generating synthetic data...")
    gen = DrillingDataGenerator(n_samples=5000, random_state=42)
    df = gen.save("outputs/data/drilling_data.csv")
    print(f"  Dataset: {len(df)} records, {len(df.columns)} columns")

    print("\n[2/6] Preprocessing data...")
    preprocessor = DrillingPreprocessor()
    X = preprocessor.fit_transform(df)

    y_rop = df["rop_ft_hr"].values
    y_torque = df["torque_klft"].values
    y_vibration = df["vibration_g"].values

    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_rop_train, y_rop_test = y_rop[:split], y_rop[split:]
    y_torque_train, y_torque_test = y_torque[:split], y_torque[split:]
    y_vib_train, y_vib_test = y_vibration[:split], y_vibration[split:]

    print(f"  Train: {split} | Test: {len(df) - split}")

    print("\n[3/6] Training ROP model (LightGBM + Optuna tuning)...")
    rop_model = ROPPredictor()
    rop_results = rop_model.train(X_train, y_rop_train)
    rop_eval = rop_model.evaluate(X_test, y_rop_test)
    print(f"  Model: {rop_model.best_name}")
    print(f"  Train R2: {rop_results[rop_model.best_name]['r2']:.4f}")
    print(f"  Test  R2: {rop_eval['r2']:.4f} | MAE: {rop_eval['mae']:.2f} ft/hr | MAPE: {rop_eval['mape']:.1f}%")

    print("\n[4/6] Training Torque model (LightGBM + Optuna tuning)...")
    torque_model = TorquePredictor()
    torque_results = torque_model.train(X_train, y_torque_train)
    torque_eval = torque_model.evaluate(X_test, y_torque_test)
    print(f"  Model: {torque_model.best_name}")
    print(f"  Train R2: {torque_results[torque_model.best_name]['r2']:.4f}")
    print(f"  Test  R2: {torque_eval['r2']:.4f} | MAE: {torque_eval['mae']:.2f} klft | MAPE: {torque_eval['mape']:.1f}%")

    print("\n[5/6] Training Vibration model (LightGBM)...")
    vib_model = VibrationAnalyzer()
    vib_results = vib_model.train(X_train, y_vib_train)
    vib_eval = vib_model.evaluate(X_test, y_vib_test)
    print(f"  Train Accuracy: {vib_results['classification_accuracy']:.4f}")
    print(f"  Test  Accuracy: {vib_eval['classification_accuracy']:.4f} | R2: {vib_eval['regression_r2']:.4f}")

    print("\n  SHAP explainability (top 5 features for ROP)...")
    shap_values = rop_model.explain(X_test[:50])
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_names = preprocessor.get_feature_names()
    top_idx = np.argsort(mean_abs_shap)[::-1][:5]
    for idx in top_idx:
        print(f"    {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

    print("\n[6/6] Saving models...")
    os.makedirs("outputs/models", exist_ok=True)
    rop_model.save("outputs/models/rop_predictor.pkl")
    torque_model.save("outputs/models/torque_predictor.pkl")
    vib_model.save("outputs/models/vibration_analyzer.pkl")

    import pickle
    with open("outputs/models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("  Models saved to outputs/models/")

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  ROP Predictor:      R2={rop_eval['r2']:.4f} | MAE={rop_eval['mae']:.2f} ft/hr | MAPE={rop_eval['mape']:.1f}%")
    print(f"  Torque Predictor:   R2={torque_eval['r2']:.4f} | MAE={torque_eval['mae']:.2f} klft | MAPE={torque_eval['mape']:.1f}%")
    print(f"  Vibration Analyzer: Acc={vib_eval['classification_accuracy']:.4f} | R2={vib_eval['regression_r2']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
