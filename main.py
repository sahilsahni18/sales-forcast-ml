from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import xgboost as xgb
import os

app = FastAPI()

model_path = "xgb_sales_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Train it using train_model.py")

model = joblib.load(model_path)

class SalesFeatures(BaseModel):
    price: float
    units_sold: float

@app.post("/predict")
def predict_sales(features: SalesFeatures):
    input_data = np.array([[features.price, features.units_sold]])
    prediction = model.predict(input_data)[0]
    return {"predicted_sales": float(prediction)}

