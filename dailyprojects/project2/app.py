from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd



# Load model
model = joblib.load("wait_time_model.pkl")
w = np.array(model['weights'])
b = model['bias']
columns = model['columns']

app = FastAPI()



# Input schema
class PatientInput(BaseModel):
    Department: str
    Urgency_level: float
    Queue_length: float
    Doctor_Availability: float
    Patient_Arrival_hour: float

@app.post("/predict")
def predict_wait_time(data: PatientInput):
    # Convert input to dataframe
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    # One-hot encode
    # inside predict_wait_time
    df = pd.get_dummies(df)
    for col in columns:
        if col not in df:
            df[col] = 0
    df = df[columns]

    # Ensure it's 2D
    X_input = df.values.astype(float).reshape(1, -1)
    row_norm = np.linalg.norm(X_input, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1
    X_input = X_input / row_norm

    y_hat = np.dot(X_input, w) + b
    return {"predicted_wait_time": y_hat[0]}