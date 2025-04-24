
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load trained model artifacts
model = joblib.load("model_weights.pkl")
bias = joblib.load("model_bias.pkl")

class InputFeatures(BaseModel):
    Delivery_person_Age: float
    Delivery_person_Ratings: float
    distance: float
    Type_of_vehicle_encoded: list  # one-hot encoded list

@app.post("/predict")
def predict_delay(features: InputFeatures):
    # Combine input features into array
    input_array = np.array([
        features.Delivery_person_Age,
        features.Delivery_person_Ratings,
        features.distance,
        *features.Type_of_vehicle_encoded  # unpack one-hot list
    ]).reshape(1, -1)

    # Normalize using L2
    norm = np.linalg.norm(input_array, axis=1, keepdims=True)
    norm[norm == 0] = 1
    input_array = input_array / norm

    # Predict
    prediction = np.dot(input_array, model) + bias
    return {"predicted_time_taken": prediction[0]}
