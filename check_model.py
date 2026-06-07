import pandas as pd
import pickle

def totalcharges_preprocess(X):
    X = X.copy()
    for col in ['TotalCharges', 'MonthlyCharges', 'tenure', 'SeniorCitizen']:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].mean())
    return X

with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))
print(model)