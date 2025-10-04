# app.py
import streamlit as st
import pandas as pd
import pickle

def totalcharges_preprocess(X):
    X = X.copy()
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].mean())
    return X

# ------------------- Load Model -------------------
with open("customer_churn.pkl", "rb") as f:
    pipeline = pickle.load(f)

# ------------------- App Title -------------------
st.title("Telco Customer Churn Prediction")
st.write("Upload your CSV file with customer data to predict churn probability and outcome.")

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df_new = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df_new.head())

    # ------------------- Prediction Function -------------------
    def predict_new(df_new, pipeline, threshold=0.5):
        # Keep customerID if exists
        customer_ids = df_new['customerID'] if 'customerID' in df_new.columns else range(len(df_new))
        
        # Drop customerID for prediction
        df_new_model = df_new.drop('customerID', axis=1, errors='ignore')
        
        # Ensure numeric columns are numeric
        numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        for col in numeric_cols:
            if col in df_new_model.columns:
                df_new_model[col] = pd.to_numeric(df_new_model[col], errors='coerce')
                df_new_model[col] = df_new_model[col].fillna(df_new_model[col].mean())
        
        # Predict
        try:
            y_probs = pipeline.predict_proba(df_new_model)[:,1]
        except ValueError:
            # Add missing dummy columns if needed
            missing_cols = list(set(pipeline.named_steps['preprocess'].get_feature_names_out()) - set(df_new_model.columns))
            for col in missing_cols:
                df_new_model[col] = 0
            y_probs = pipeline.predict_proba(df_new_model)[:,1]
        
        y_pred = (y_probs >= threshold).astype(int)
        
        results = pd.DataFrame({
            'customerID': customer_ids,
            'Churn_Probability': y_probs,
            'Churn_Prediction': y_pred
        })
        return results

    # ------------------- Make Predictions -------------------
    best_thresh = 0.5  # you can adjust if needed
    predictions = predict_new(df_new, pipeline, best_thresh)
    st.write("Predictions:")
    st.dataframe(predictions)

    # ------------------- Download Predictions -------------------
    csv = predictions.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )
