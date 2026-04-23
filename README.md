# No-Show Medical Appointment Prediction

This project predicts whether a patient will attend their medical appointment using machine learning.  
It includes a full ML pipeline, API, monitoring, and deployment-ready structure.

---

noshow_clean/
│
├── api.py
├── main.py
├── requirements.txt
├── Dockerfile.api
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── monitor.py
│
├── models/
├── data/
├── figures/

---

## 📌 How to Train the Model

Run:python3 src/train_model.py

This will:

- preprocess the dataset  
- train a Logistic Regression model  
- save a timestamped model in `models/`  
- generate a confusion matrix in `figures/`  

---

## 📌 How to Run the API Locally

Start the APIpython3 main.py

Open your browser:


http://localhost:8000/docs


You can test the `/predict` endpoint using the interactive Swagger UI.

---

## 📌 API Endpoints

### **GET /**  
Returns a simple health message.

### **POST /predict**  
Accepts a JSON payload matching the model's feature names.  
Returns:

{
"prediction": 0 or 1
}
[200~python3 src/monitor.py

This script checks:

- feature alignment  
- prediction stability  
- optional data drift  
- model loading health  

---

## 📌 Deployment (Render)

1. Push this project to GitHub  
2. Create a new Web Service on Render  
3. Build command:
pip install -r requirements.txt
4. Start command:

uvicorn main:app --host 0.0.0.0 --port 10000

5. Open:
https://<your-app>.onrender.com/docs

---

## 📌 Requirements

fastapi
uvicorn
pandas
numpy
scikit-learn
matplotlib
joblib

---

## 📌 Evidence Checklist (Screenshots)

- Training output  
- Confusion matrix image  
- Saved model file  
- API docs page  
- Successful prediction  
- Monitoring script output  
- Render deployment page  
- Public API URL  

---

## 📌 Notes

This project is fully reproducible and deployment-ready.  
All code is modular, clean, and aligned with ML engineering best practices.

