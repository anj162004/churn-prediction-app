import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

# ------------------- Extract the zipped model -------------------
zip_path = "churn_model.zip"  # your uploaded zip file
extract_dir = "model"

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    # Automatically find the .pkl file inside the zip
    pkl_files = [f for f in zip_ref.namelist() if f.endswith(".pkl")]
    if not pkl_files:
        st.error("No .pkl file found inside the zip!")
        st.stop()
    pipeline_path = os.path.join(extract_dir, pkl_files[0])

# Load the pickle pipeline
with open(pipeline_path, "rb") as f:
    pipeline = pickle.load(f)

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
    
    # Threshold slider
    best_thresh = st.slider("Select threshold for churn prediction", 0.0, 1.0, 0.54, 0.01)
    
    # Predict probabilities
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
    
    # Optional: allow downloading results
    csv = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='churn_predictions.csv',
        mime='text/csv'
    )
