import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocessing import preprocess
from config import DATA_PATH, MODEL_DIR, MODEL_BASENAME, FIGURE_DIR

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

def load_data():
    return pd.read_csv(DATA_PATH)

def save_model(model, features):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"{MODEL_BASENAME}_{timestamp}.pkl")
    joblib.dump({"model": model, "features": features}, model_path)
    print(f"✅ Model saved to {model_path}")

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    fig_path = os.path.join(FIGURE_DIR, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 Confusion matrix saved to {fig_path}")

def train():
    print("🚀 Starting training pipeline...")
    df = load_data()
    X_train, X_test, y_train, y_test, features = preprocess(df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    plot_confusion(y_test, preds)
    save_model(model, features)

    print("🎯 Training complete.")

if __name__ == "__main__":
    ensure_dirs()
    train()

