import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import zipfile
from sklearn.preprocessing import FunctionTransformer

# ------------------- Preprocessing function -------------------
def totalcharges_preprocess(X):
    X = X.copy()
    for col in ['TotalCharges', 'MonthlyCharges', 'tenure', 'SeniorCitizen']:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].mean())
    return X

zip_path = "churn_model.zip"
extract_dir = "model"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

pipeline_path = os.path.join(extract_dir, "churn_model.joblib")
pipeline = joblib.load(pipeline_path)

# ------------------- Streamlit App -------------------
st.title("Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload customer CSV", type="csv")
if uploaded_file:
    df_new = pd.read_csv(uploaded_file)
    
    # Keep customer IDs separate
    customer_ids = df_new['customerID'] if 'customerID' in df_new.columns else range(len(df_new))
    df_new_model = df_new.drop('customerID', axis=1, errors='ignore')
    
    # Numeric preprocessing
    numeric_cols = ['TotalCharges', 'MonthlyCharges', 'tenure', 'SeniorCitizen']
    for col in numeric_cols:
        if col in df_new_model.columns:
            df_new_model[col] = pd.to_numeric(df_new_model[col], errors='coerce')
            df_new_model[col] = df_new_model[col].fillna(df_new_model[col].mean())
    
    # Predict probabilities
    best_thresh = 0.54
    try:
        y_probs = pipeline.predict_proba(df_new_model)[:,1]
        y_pred = (y_probs >= best_thresh).astype(int)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()
    
    # Show results
    results = pd.DataFrame({
        'customerID': customer_ids,
        'Churn_Probability': y_probs,
        'Churn_Prediction': y_pred
    })
    
    st.subheader("Prediction Results")
    st.dataframe(results)
