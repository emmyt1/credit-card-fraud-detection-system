import streamlit as st
import requests
import json

# --- Configuration ---
# URL of the FastAPI backend
# Adjust this URL if your FastAPI app runs on a different host/port
FASTAPI_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{FASTAPI_URL}/predict/"
HEALTH_ENDPOINT = f"{FASTAPI_URL}/health"
# --- End Configuration ---

# check_backend_health() function
def check_backend_health():
    """Checks if the FastAPI backend is reachable and healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        response.raise_for_status()

        # Check if the response is JSON before parsing
        content_type = response.headers.get('content-type', '')
        if 'application/json' not in content_type:
            st.warning(f"Health check endpoint returned non-JSON content: {content_type}")
            return False

        data = response.json()
        # Add checks to ensure keys exist
        status_ok = data.get("status") == "healthy"
        model_loaded = data.get("model_loaded", False) # Default to False if key missing
        return status_ok and model_loaded

    except requests.exceptions.ConnectionError:
        # More specific handling for connection issues
        st.warning(f"Could not connect to backend at {HEALTH_ENDPOINT}. Is it running?")
        return False
    except requests.exceptions.Timeout:
         st.warning(f"Health check timed out contacting {HEALTH_ENDPOINT}.")
         return False
    except requests.exceptions.RequestException as e:
        st.error(f"HTTP error connecting to backend ({HEALTH_ENDPOINT}): {e}")
        return False
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from health check ({HEALTH_ENDPOINT}): {e}. Response text: {response.text[:200]}") # Show snippet
        return False
    except Exception as e: # Catch any other unexpected errors
         st.error(f"Unexpected error during health check: {e}")
         return False

def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳")
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("Enter transaction details to check for potential fraud using the deployed XGBoost model.")

    # Check backend health on app load
    if not check_backend_health():
        st.warning("⚠️ Backend service might be unavailable or not ready. Predictions might fail.")

    st.subheader("Transaction Details")
    st.caption("Please enter the values for all features. V1-V28 are PCA components.")
    # Create input fields for features
    # Grouping for better UI
    col1, col2, col3 = st.columns(3)

    with col1:
        time = st.number_input("Time", value=0.0, format="%.2f", step=1.0)
        v1 = st.number_input("V1", value=0.0, format="%.6f")
        v2 = st.number_input("V2", value=0.0, format="%.6f")
        v3 = st.number_input("V3", value=0.0, format="%.6f")
        v4 = st.number_input("V4", value=0.0, format="%.6f")
        v5 = st.number_input("V5", value=0.0, format="%.6f")
        v6 = st.number_input("V6", value=0.0, format="%.6f")
        v7 = st.number_input("V7", value=0.0, format="%.6f")
        v8 = st.number_input("V8", value=0.0, format="%.6f")
        v9 = st.number_input("V9", value=0.0, format="%.6f")

    with col2:
        v10 = st.number_input("V10", value=0.0, format="%.6f")
        v11 = st.number_input("V11", value=0.0, format="%.6f")
        v12 = st.number_input("V12", value=0.0, format="%.6f")
        v13 = st.number_input("V13", value=0.0, format="%.6f")
        v14 = st.number_input("V14", value=0.0, format="%.6f")
        v15 = st.number_input("V15", value=0.0, format="%.6f")
        v16 = st.number_input("V16", value=0.0, format="%.6f")
        v17 = st.number_input("V17", value=0.0, format="%.6f")
        v18 = st.number_input("V18", value=0.0, format="%.6f")
        v19 = st.number_input("V19", value=0.0, format="%.6f")

    with col3:
        v20 = st.number_input("V20", value=0.0, format="%.6f")
        v21 = st.number_input("V21", value=0.0, format="%.6f")
        v22 = st.number_input("V22", value=0.0, format="%.6f")
        v23 = st.number_input("V23", value=0.0, format="%.6f")
        v24 = st.number_input("V24", value=0.0, format="%.6f")
        v25 = st.number_input("V25", value=0.0, format="%.6f")
        v26 = st.number_input("V26", value=0.0, format="%.6f")
        v27 = st.number_input("V27", value=0.0, format="%.6f")
        v28 = st.number_input("V28", value=0.0, format="%.6f")
        amount = st.number_input("Amount", value=0.0, format="%.2f", min_value=0.0, step=0.01)

    # Prepare data for the API request
    transaction_data = {
        "Time": time,
        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
        "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
        "V11": v11, "V12": v12, "V13": v13, "V14": v14,
        "V15": v15, "V16": v16, "V17": v17, "V18": v18, "V19": v19,
        "V20": v20, "V21": v21, "V22": v22, "V23": v23, "V24": v24,
        "V25": v25, "V26": v26, "V27": v27, "V28": v28,
        "Amount": amount
    }

    st.divider()
    # Button to trigger prediction
    if st.button("🔍 Predict Fraud", type="primary"):
        try:
            with st.spinner("Analyzing transaction..."):
                # Send POST request to FastAPI backend
                response = requests.post(PREDICT_ENDPOINT, json=transaction_data, timeout=10)

            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                probability = result.get("fraud_probability")

                # Display results
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error(f"🚨 **Fraudulent Transaction Detected!** \n\n (Fraud Probability: {probability:.2%})")
                else:
                    st.success(f"✅ **Transaction Appears Legitimate** \n\n (Fraud Probability: {probability:.2%})")
                st.caption(f"*Raw Probability Score: {probability:.4f}*")

            else:
                # Handle non-200 responses
                try:
                    error_detail = response.json().get("detail", "Unknown error")
                except json.JSONDecodeError:
                    error_detail = response.text
                st.error(f"Backend Error (Status {response.status_code}): {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("🚫 Failed to connect to the prediction service. Is the FastAPI backend running?")
            st.info(f"Please ensure the FastAPI app is running at `{FASTAPI_URL}`.")
        except requests.exceptions.Timeout:
            st.error("⏰ Request to the prediction service timed out.")
        except requests.exceptions.RequestException as e:
            st.error(f"❗ An error occurred during the request: {e}")
        except json.JSONDecodeError:
            st.error("❌ Failed to decode JSON response from backend.")
        except Exception as e:
            st.error(f"💥 An unexpected error occurred: {e}")

    st.divider()
    st.markdown("**How to Run:**")
    st.code("""
# Terminal 1: Start FastAPI
cd path/to/credit_fraud_project
uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit (ensure FASTAPI_URL in app.py is correct)
cd path/to/credit_fraud_project
streamlit run streamlit_app/app.py
        """, language='bash') # <--- This closing part is crucial
        

if __name__ == "__main__":
    main()

