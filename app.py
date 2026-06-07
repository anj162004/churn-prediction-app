from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from datetime import datetime
import shap
import matplotlib.pyplot as plt

app = Flask(__name__)

# Required because your pipeline contains this function
def totalcharges_preprocess(X):
    X = X.copy()
    for col in ['TotalCharges', 'MonthlyCharges', 'tenure', 'SeniorCitizen']:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].mean())
    return X

# Load trained pipeline
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)
rf_model = model.named_steps['rf']
explainer = shap.TreeExplainer(rf_model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'gender': [request.form['gender']],
        'SeniorCitizen': [int(request.form['SeniorCitizen'])],
        'Partner': [request.form['Partner']],
        'Dependents': [request.form['Dependents']],
        'tenure': [float(request.form['tenure'])],
        'PhoneService': [request.form['PhoneService']],
        'MultipleLines': [request.form['MultipleLines']],
        'InternetService': [request.form['InternetService']],
        'OnlineSecurity': [request.form['OnlineSecurity']],
        'OnlineBackup': [request.form['OnlineBackup']],
        'DeviceProtection': [request.form['DeviceProtection']],
        'TechSupport': [request.form['TechSupport']],
        'StreamingTV': [request.form['StreamingTV']],
        'StreamingMovies': [request.form['StreamingMovies']],
        'Contract': [request.form['Contract']],
        'PaperlessBilling': [request.form['PaperlessBilling']],
        'PaymentMethod': [request.form['PaymentMethod']],
        'MonthlyCharges': [float(request.form['MonthlyCharges'])],
        'TotalCharges': [float(request.form['TotalCharges'])]
    }

    df = pd.DataFrame(data)

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    
    # Apply preprocessing
    # Apply totalcharges transformer first
    df_processed = model.named_steps['totalcharges'].transform(df)

    # Apply column transformer
    processed_data = model.named_steps['preprocess'].transform(df_processed)

    # Convert sparse matrix to dense matrix
    if hasattr(processed_data, "toarray"):
        processed_data = processed_data.toarray()

    # Convert to float
    processed_data = processed_data.astype(float)

    # SHAP values
    shap_values = explainer.shap_values(
        processed_data,
        check_additivity=False
    )
    
    feature_names = model.named_steps['preprocess'].get_feature_names_out()

    # SHAP values for churn class (class 1)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "value": shap_values[0, :, 1]
    })

    importance_df["feature"] = importance_df["feature"].str.replace("cat__", "")
    importance_df["feature"] = importance_df["feature"].apply(
        lambda x: "TotalCharges" if x.startswith("TotalCharges_") else x
    )

    importance_df = (
        importance_df
        .groupby("feature")["value"]
        .sum()
        .reset_index()
    )

    importance = list(
        zip(
            importance_df["feature"],
            importance_df["value"]
        )
    )

    importance = sorted(
        importance,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_factors = []

    for feat, value in importance[:10]:
        feat = feat.replace("cat__", "")
        feat = feat.replace("remainder__", "")

        if feat.startswith("Contract"):
            feat = "Contract Type"
        elif feat.startswith("PaymentMethod"):
            feat = "Payment Method"
        elif feat.startswith("InternetService"):
            feat = "Internet Service"
        elif feat.startswith("TechSupport"):
            feat = "Tech Support"
        elif feat.startswith("DeviceProtection"):
            feat = "Device Protection"
        elif feat.startswith("OnlineSecurity"):
            feat = "Online Security"
        elif feat.startswith("MultipleLines"):
            feat = "Multiple Lines"
        elif feat.startswith("PhoneService"):
            feat = "Phone Service"
        elif feat.startswith("StreamingTV"):
            feat = "Streaming TV"
        elif feat.startswith("StreamingMovies"):
            feat = "Streaming Movies"
        elif feat.startswith("Partner"):
            feat = "Partner Status"
        elif feat.startswith("Dependents"):
            feat = "Dependents"
        elif feat.startswith("gender"):
            feat = "Gender"
        elif feat == "tenure":
            feat = "Customer Tenure"
        elif feat == "MonthlyCharges":
            feat = "Monthly Charges"
        elif feat.startswith("TotalCharges"):
            feat = "Total Charges"

        top_factors.append(feat)

    # Remove duplicates
    allowed_features = [
    "Gender",
    "Partner Status",
    "Dependents",
    "Customer Tenure",
    "Total Charges",
    "Monthly Charges",
    "Phone Service",
    "Internet Service",
    "Contract Type",
    "Online Security",
    "Tech Support",
    "Payment Method"
]

    top_factors = [f for f in top_factors if f in allowed_features]
    top_factors = list(dict.fromkeys(top_factors))

    top_factors = top_factors[:5]

    result = "Customer Will Churn" if pred == 1 else "Customer Will Not Churn"

    # Risk Meter
    if prob >= 0.75:
        risk = "🔴 High Risk Customer"
    elif prob >= 0.50:
        risk = "🟡 Medium Risk Customer"
    else:
        risk = "🟢 Low Risk Customer"

    # Retention Recommendations
    if pred == 1:
        recommendations = [
            "Offer Loyalty Discount",
            "Offer Annual Contract Plan",
            "Provide Premium Customer Support",
            "Offer Personalized Retention Campaign"
        ]
    else:
        recommendations = [
            "Customer Likely To Stay",
            "Maintain Current Service Quality",
            "Continue Engagement Programs"
        ]

    log_data = pd.DataFrame({
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Prediction": [result],
        "Probability": [round(prob * 100, 2)],
        "Risk": [risk]
    })

    if os.path.exists("prediction_log.csv"):
        log_data.to_csv("prediction_log.csv", mode="a", header=False, index=False)
    else:
        log_data.to_csv("prediction_log.csv", index=False)
   
    return render_template(
        "index.html",
        prediction=result,
        probability=round(prob * 100, 2),
        risk=risk,
        recommendations=recommendations,
        top_factors=top_factors
    )

@app.route('/history')
def history():
    if os.path.exists("prediction_log.csv"):
        history_df = pd.read_csv("prediction_log.csv")
        history_data = history_df.tail(10).to_dict(orient="records")
    else:
        history_data = []
    return render_template("history.html", history=history_data)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return jsonify({"prediction": pred, "probability": round(prob, 4)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)