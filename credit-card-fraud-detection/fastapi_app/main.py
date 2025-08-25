from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any
from .model_handler import model_instance # Import the model instance

app = FastAPI(title="Credit Card Fraud Detection API", description="API for predicting credit card fraud using XGBoost (SMOTE).")

# Load the model when the application starts
@app.on_event("startup")
async def load_model_on_startup():
    try:
        model_instance.load_model()
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        # Depending on requirements, you might want to stop the app or log critically
        # raise e # Uncomment to potentially stop the app if model fails to load

# --- Define Pydantic model for input validation ---
# Based on the typical dataset structure (Time, V1-V28, Amount)
# You should ideally load feature names dynamically or ensure this matches feature_names.csv exactly
class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    # Add other features if present in your specific dataset/model

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Card Fraud Detection API", "status": "API is running"}

@app.post("/predict/", response_model=Dict[str, Any]) # Specify response model
def predict_fraud(transaction: TransactionData):
    """
    Predicts if a transaction is fraudulent.
    """
    data_dict = transaction.dict()

    try:
        result = model_instance.predict(data_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Optional: Health check endpoint
@app.get("/health")
def health_check():
    if model_instance.is_loaded:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False, "details": "Model not loaded or failed to load"}

# To run this app locally:
# Navigate to the `credit_fraud_project` directory in your terminal.
# Run: `uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000`
# Access docs at: http://localhost:8000/docs
