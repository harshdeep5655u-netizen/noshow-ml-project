import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

MODEL_DIR = "../models"

def load_latest_model():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not files:
        raise RuntimeError("No model found in models/")
    latest = sorted(files)[-1]
    bundle = joblib.load(os.path.join(MODEL_DIR, latest))
    return bundle["model"], bundle["features"]

def check_feature_alignment(features):
    print("🔍 Checking feature alignment...")
    print(f"Model expects {len(features)} features")
    print("Feature list:")
    for f in features:
        print(" -", f)

def check_prediction_stability(model, features):
    print("\n🔍 Checking prediction stability...")
    x = np.zeros(len(features)).reshape(1, -1)
    pred = model.predict(x)[0]
    print(f"Prediction on zero-vector: {pred}")

def check_data_drift(df, features):
    print("\n🔍 Checking data drift...")
    missing = [f for f in features if f not in df.columns]
    if missing:
        print("⚠ Missing features in dataset:", missing)
    else:
        print("No missing features detected.")

def run_monitoring():
    print("🚀 Running monitoring checks...\n")
    model, features = load_latest_model()

    check_feature_alignment(features)
    check_prediction_stability(model, features)

    # Optional drift check if dataset exists
    data_path = "../data/appointments.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        check_data_drift(df, features)
    else:
        print("\n⚠ No dataset found for drift check.")

    print("\n✅ Monitoring complete.")

if __name__ == "__main__":
    run_monitoring()

