# Sales Forecasting ML Pipeline + FastAPI Deployment

A complete project demonstrating sales prediction using XGBoost and exposing it as a REST API via FastAPI.

## Features
- Trains an XGBoost regression model on sales data
- Saves the model with joblib
- Deploys model with FastAPI
- Dockerfile provided for easy containerization

## Usage
### 1. Train the Model
```bash
pip install -r requirements.txt
python train_model.py
```

### 2. Start the API
```bash
uvicorn main:app --reload
```

### 3. API Endpoint
- **POST** `/predict`
```json
{
  "price": 100.0,
  "units_sold": 3.0
}
```

### 4. Run with Docker
```bash
docker build -t sales-api .
docker run -p 8000:8000 sales-api
```

## Sample Response
```json
{
  "predicted_sales": 300.45
}
```
"""
