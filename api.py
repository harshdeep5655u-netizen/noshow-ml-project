from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

MODEL_DIR = "models"

def load_latest_model():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not files:
        raise RuntimeError("No model found in models/ directory")

    latest = sorted(files)[-1]
    bundle = joblib.load(os.path.join(MODEL_DIR, latest))
    return bundle["model"], bundle["features"]

model, feature_names = load_latest_model()

@app.get("/")
def root():
    return {"message": "No-Show Prediction API is running"}

@app.post("/predict")
def predict(payload: dict):
    # Build feature vector in correct order
    x = []
    for f in feature_names:
        x.append(payload.get(f, 0))

    x = np.array(x).reshape(1, -1)

    pred = model.predict(x)[0]
    return {"prediction": int(pred)}
