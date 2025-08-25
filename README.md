# Credit Card Fraud Detection

This project aims to develop and evaluate machine learning models for predicting fraudulent credit card transactions. The goal is to build a robust model that can effectively identify fraudulent activities while minimizing false positives and negatives.

## 🌬️ Project Overview

This repository contains the code and resources for a machine learning project focused on detecting fraudulent credit card transactions. Due to the highly imbalanced nature of the dataset, various techniques like SMOTE were explored to handle the class imbalance. XGBoost was chosen as the final model based on its performance metrics.

## 🛠️ Technologies Used

*   Python
*   Pandas
*   NumPy
*   Scikit-learn
*   XGBoost
*   Imbalanced-learn
*   Matplotlib / Seaborn
*   FastAPI (for model deployment API)
*   Streamlit (for user interface)
*   Google Colab (for development)

## ⚙️ Project Structure
```
credit-card-fraud-detection/
│
├── best_xgb_model_smote.pkl # Saved trained model
├── scaler_smote.pkl # Saved feature scaler
├── feature_names.csv # List of features used by the model
│
├── fastapi_app/ # FastAPI backend code
│ ├── main.py # Main FastAPI application
│ └── model_handler.py # Model loading and prediction logic
│
└── streamlit_app/ # Streamlit frontend code
└── app.py # Main Streamlit application
data/
└── Credit_Card_Fraud_Detection_ML.ipynb      # Main Jupyter Notebook
README.md                                 # This file
```

## Dataset

The dataset used in this project is the "Credit Card Fraud Detection" dataset. Due to its size, it is not included in this repository.

You can download the dataset from Kaggle:
[https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

Please download the `creditcard.csv` file and place it in the root directory of this project folder before running the notebook or training scripts.

## 🚀 How to Run (Locally)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/emmyt1/credit-card-fraud-detection-system.git
    cd credit-card-fraud-detection-system
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but good practice)
    python -m venv myenv

    # Activate the virtual environment
    # On Windows:
    myenv\Scripts\activate
    # On macOS/Linux:
    source myenv/bin/activate

    # Install required packages
    pip install fastapi uvicorn pydantic scikit-learn pandas numpy joblib xgboost streamlit requests
    ```

3.  **Run the FastAPI Backend:**
    Open a new terminal window/tab (make sure your virtual environment is activated if you used one).
    ```bash
    cd path/to/credit-card-fraud-detection # Navigate to your project folder
    uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API should now be running at `http://localhost:8000`. You can access the interactive API docs at `http://localhost:8000/docs`.

4.  **Run the Streamlit Frontend:**
    Open another terminal window/tab (make sure your virtual environment is activated).
    ```bash
    cd path/to/credit-card-fraud-detection # Navigate to your project folder
    streamlit run streamlit_app/app.py
    ```
    Streamlit will provide a URL (usually `http://localhost:8501`) in the terminal. Open this URL in your web browser to access the user interface.

## 📊 Results

The XGBoost model trained on the SMOTE-balanced dataset achieved the best performance on the test set:

*   **Accuracy:** ~99.25%
*   **Sensitivity (Recall):** ~86.73%
*   **Specificity:** ~99.28%
*   **ROC-AUC Score:** ~0.965

## 👨‍💻 Author

**Oluwaseun E. Olubunmi**
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/ooluwaseun/)
- [![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/emmyt1)
