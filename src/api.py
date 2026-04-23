from fastapi import FastAPI
import joblib
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI(title="No-Show Prediction API")

# Paths
MODEL_DIR = "models"
MODEL_NAME = "model.pkl"
PREPROCESSOR_NAME = "preprocessor.pkl"

# Load model + preprocessor
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
preprocessor_path = os.path.join(MODEL_DIR, PREPROCESSOR_NAME)

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

@app.get("/")
def root():
    return {"message": "No-Show Prediction API is running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)}

