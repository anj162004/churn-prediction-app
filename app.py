from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
from datetime import datetime

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
    print("RISK =", risk)
    print("RECOMMENDATIONS =", recommendations)
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
        recommendations=recommendations
    )

# REST API Endpoint
@app.route('/predict_api', methods=['POST'])
@app.route('/history')
def history():

    if os.path.exists("prediction_log.csv"):
        history_df = pd.read_csv("prediction_log.csv")
        history_data = history_df.tail(10).to_dict(orient="records")
    else:
        history_data = []

    return render_template(
        "history.html",
        history=history_data
    )
def predict_api():
    data = request.get_json()
    df = pd.DataFrame([data])
    
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])

    return jsonify({
        "prediction": pred,
        "probability": round(prob, 4)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)