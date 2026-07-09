import os
import pickle
import numpy as np
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Load the pickle file (contains model, scaler, feature list, and mappings)
MODEL_PATH = "rdf.pkl"
model_data = None

def load_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model_data = pickle.load(f)
            print("Model loaded successfully from:", MODEL_PATH)
        except Exception as e:
            print("Error loading model:", str(e))
    else:
        print(f"Model file '{MODEL_PATH}' not found. Ensure train.py has run successfully.")

# Initial model load
load_model()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    # Reload model if not loaded yet
    global model_data
    if model_data is None:
        load_model()
        if model_data is None:
            return "Model not trained. Run 'train.py' first.", 500
            
    # Retrieve form data
    raw_inputs = {
        'Gender': request.form.get('Gender', 'Male'),
        'Married': request.form.get('Married', 'No'),
        'Dependents': request.form.get('Dependents', '0'),
        'Education': request.form.get('Education', 'Graduate'),
        'Self_Employed': request.form.get('Self_Employed', 'No'),
        'ApplicantIncome': request.form.get('ApplicantIncome', '0'),
        'CoapplicantIncome': request.form.get('CoapplicantIncome', '0'),
        'LoanAmount': request.form.get('LoanAmount', '0'),
        'Loan_Amount_Term': request.form.get('Loan_Amount_Term', '360'),
        'Credit_History': request.form.get('Credit_History', '1.0'),
        'Property_Area': request.form.get('Property_Area', 'Urban')
    }

    try:
        # Resolve mappings from model metadata
        mappings = model_data.get('mappings', {})
        
        # Preprocess and map inputs
        gender = mappings['Gender'].get(raw_inputs['Gender'], 1)
        married = mappings['Married'].get(raw_inputs['Married'], 0)
        dependents = mappings['Dependents'].get(raw_inputs['Dependents'], 0)
        education = mappings['Education'].get(raw_inputs['Education'], 1)
        self_employed = mappings['Self_Employed'].get(raw_inputs['Self_Employed'], 0)
        
        # Convert continuous fields
        app_income = float(raw_inputs['ApplicantIncome'])
        coapp_income = float(raw_inputs['CoapplicantIncome'])
        loan_amt = float(raw_inputs['LoanAmount'])
        term = float(raw_inputs['Loan_Amount_Term'])
        
        # Credit history: convert string representation of float (e.g. "1.0") to float
        credit_history = float(raw_inputs['Credit_History'])
        
        property_area = mappings['Property_Area'].get(raw_inputs['Property_Area'], 2)
        
        # Build features array in exact order:
        # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        #  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        #  'Credit_History', 'Property_Area']
        import pandas as pd
        features = pd.DataFrame([[
            gender, married, dependents, education, self_employed,
            app_income, coapp_income, loan_amt, term, credit_history, property_area
        ]], columns=model_data['features'])
        
        # Apply standard scaling
        scaler = model_data['scaler']
        scaled_features = scaler.transform(features)
        
        # Convert back to DataFrame to preserve feature names for model inference
        scaled_features_df = pd.DataFrame(scaled_features, columns=model_data['features'])
        
        # Make prediction using the model (XGBoost)
        model = model_data['model']
        prediction = model.predict(scaled_features_df)
        
        pred_val = int(prediction[0])
        print(f"Prediction: {pred_val} (Raw input values: {raw_inputs})")
        
        # Calculate Risk Score (0-100) & Risk Level based on credit history, income, and debt-to-income ratio
        monthly_emi = (loan_amt * 1000) / term if term > 0 else 0
        total_income = app_income + coapp_income
        dti = monthly_emi / total_income if total_income > 0 else 0
        
        risk_score = 0
        # 1. Credit History Factor (max 40 pts)
        if credit_history == 0.0:
            risk_score += 40
        # 2. DTI Ratio Factor (max 30 pts)
        if dti > 0.45:
            risk_score += 30
        elif dti > 0.35:
            risk_score += 15
        # 3. Income Factor (max 15 pts)
        if total_income < 2500:
            risk_score += 15
        elif total_income < 5000:
            risk_score += 5
        # 4. Loan Size Factor (max 15 pts)
        if loan_amt > 300:
            risk_score += 15
        elif loan_amt > 150:
            risk_score += 5
            
        # Determine risk category
        if risk_score >= 60:
            risk_level = "High Risk"
            risk_color = "danger"
        elif risk_score >= 30:
            risk_level = "Medium Risk"
            risk_color = "warning"
        else:
            risk_level = "Low Risk"
            risk_color = "success"
            
        return render_template('submit.html', prediction=pred_val, inputs=raw_inputs, risk_level=risk_level, risk_color=risk_color, risk_score=risk_score)
        
    except Exception as e:
        print("Error during prediction:", str(e))
        return f"An error occurred during evaluation: {str(e)}", 400

@app.route('/risk')
def risk():
    return redirect(url_for('predict', tab='risk'))

@app.route('/settings')
def settings():
    return redirect(url_for('predict', tab='settings'))

if __name__ == '__main__':
    # Start Flask web server
    app.run(host='0.0.0.0', port=5000, debug=True)
