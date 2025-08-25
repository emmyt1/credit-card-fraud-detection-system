import joblib
import pandas as pd
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = "best_xgb_model_smote.pkl"
SCALER_PATH = "scaler_smote.pkl"
FEATURES_PATH = "feature_names.csv"
# --- End Configuration ---

class FraudDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False

    def load_model(self):
        """Loads the trained model, scaler, and feature names."""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {os.path.abspath(MODEL_PATH)}")
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler file not found at {os.path.abspath(SCALER_PATH)}")
            if not os.path.exists(FEATURES_PATH):
                raise FileNotFoundError(f"Features file not found at {os.path.abspath(FEATURES_PATH)}")

            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_names = pd.read_csv(FEATURES_PATH)['feature_name'].tolist()

            # Basic validation
            if self.model is None:
                raise ValueError("Failed to load model object.")
            if self.scaler is None:
                raise ValueError("Failed to load scaler object.")
            if not self.feature_names:
                 raise ValueError("Failed to load feature names.")

            self.is_loaded = True
            print("Model, scaler, and features loaded successfully.")
            print(f"Expected features: {self.feature_names}")
        except Exception as e:
            print(f"Error loading model components: {e}")
            raise

    def predict(self, data: dict) -> dict:
        """
        Makes a fraud prediction for a single transaction.

        Args:
            data: A dictionary containing transaction features.

        Returns:
            A dictionary with 'prediction' (0 or 1) and 'fraud_probability'.
        """
        if not self.is_loaded:
             raise RuntimeError("Model components not loaded. Call load_model() first.")

        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([data])

            # --- Feature Validation ---
            # Ensure all expected features are present
            missing_features = set(self.feature_names) - set(input_df.columns)
            extra_features = set(input_df.columns) - set(self.feature_names)
            if missing_features:
                raise ValueError(f"Missing features in input data: {missing_features}")
            if extra_features:
                print(f"Warning: Extra features provided, ignoring: {extra_features}")
                # Optionally drop extra features
                input_df = input_df[self.feature_names]

            # Reorder columns to match training data (important!)
            input_df = input_df[self.feature_names]

            # --- Handle Feature Scaling ---
            # Check if 'Amount' is in the features used by the model and needs scaling
            # The scaler was fitted on the training 'Amount' feature.
            if 'Amount' in self.feature_names and hasattr(self.scaler, 'scale_'):
                # Transform the 'Amount' feature using the loaded scaler
                # Ensure it's a 2D array for transform
                input_df[['Amount']] = self.scaler.transform(input_df[['Amount']])

            # --- Prediction ---
            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            fraud_probability = probabilities[1] # Probability of class 1 (Fraud)

            return {
                "prediction": int(prediction),
                "fraud_probability": float(fraud_probability)
            }

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise # Re-raise for the API endpoint to handle

# Create a global instance to be used by the API
model_instance = FraudDetectionModel()

# Load the model when this module is imported (optional, can be done in main.py startup)
# try:
#     model_instance.load_model()
# except Exception as e:
#     print(f"Warning: Failed to load model automatically in model_handler: {e}")
