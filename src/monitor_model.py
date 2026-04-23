import os
import glob
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

DATA_PATH = "data/appointments.csv"
MODEL_DIR = "models/"
FIG_DIR = "figures/"
REPORT_DIR = "reports/"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def load_latest_model():
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
    if not model_files:
        raise RuntimeError("No model found in models/ directory")
    latest = max(model_files, key=os.path.getctime)
    return joblib.load(latest), latest

def generate_report(text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"monitoring_report_{timestamp}.txt")
    with open(report_path, "w") as f:
        f.write(text)
    print(f"📄 Monitoring report saved to {report_path}")

def plot_feature_distribution(df, feature):
    plt.figure(figsize=(6,4))
    df[feature].hist(bins=30)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(FIG_DIR, f"{feature}_distribution_{timestamp}.png")
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved: {path}")

def main():
    print("🚀 Starting monitoring pipeline...")

    model, model_path = load_latest_model()
    print(f"✅ Loaded model: {model_path}")

    df = pd.read_csv(DATA_PATH)
    print(f"📥 Loaded dataset: {df.shape[0]} rows")

    report = []
    report.append(f"Model used: {model_path}")
    report.append(f"Dataset rows: {df.shape[0]}")
    report.append("")

    # Missing values
    missing = df.isnull().sum()
    report.append("Missing Values:")
    report.append(str(missing))
    report.append("")

    # Valid numeric features from your dataset
    numeric_features = [
        "Age",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received"
    ]

    # Feature distributions
    for feature in numeric_features:
        plot_feature_distribution(df, feature)

    # Simple drift check (mean comparison)
    report.append("Feature Drift Check:")
    for feature in numeric_features:
        mean_val = df[feature].mean()
        report.append(f"{feature} mean: {mean_val:.3f}")
    report.append("")

    generate_report("\n".join(report))
    print("🎯 Monitoring complete.")

if __name__ == "__main__":
    main()

