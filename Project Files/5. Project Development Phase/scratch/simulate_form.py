import os
import pickle
import pandas as pd
import numpy as np

def run():
    with open("rdf.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    features_list = data["features"]
    mappings = data["mappings"]

    # Form inputs that the user might submit in the web app
    # Applicant 1: Poor Credit History, moderate income, high loan
    raw_inputs = {
        'Gender': 'Male',
        'Married': 'No',
        'Dependents': '3+',
        'Education': 'Not Graduate',
        'Self_Employed': 'Yes',
        'ApplicantIncome': '3000',
        'CoapplicantIncome': '0',
        'LoanAmount': '350',
        'Loan_Amount_Term': '360',
        'Credit_History': '0.0',
        'Property_Area': 'Rural'
    }

    gender = mappings['Gender'].get(raw_inputs['Gender'], 1)
    married = mappings['Married'].get(raw_inputs['Married'], 0)
    dependents = mappings['Dependents'].get(raw_inputs['Dependents'], 0)
    education = mappings['Education'].get(raw_inputs['Education'], 1)
    self_employed = mappings['Self_Employed'].get(raw_inputs['Self_Employed'], 0)
    
    app_income = float(raw_inputs['ApplicantIncome'])
    coapp_income = float(raw_inputs['CoapplicantIncome'])
    loan_amt = float(raw_inputs['LoanAmount'])
    term = float(raw_inputs['Loan_Amount_Term'])
    credit_history = float(raw_inputs['Credit_History'])
    property_area = mappings['Property_Area'].get(raw_inputs['Property_Area'], 2)

    features = pd.DataFrame([[
        gender, married, dependents, education, self_employed,
        app_income, coapp_income, loan_amt, term, credit_history, property_area
    ]], columns=features_list)

    scaled_features = scaler.transform(features)
    scaled_features_df = pd.DataFrame(scaled_features, columns=features_list)
    
    pred = model.predict(scaled_features_df)[0]
    proba = model.predict_proba(scaled_features_df)[0][1]

    print("Prediction:", pred)
    print("Probability:", proba)

if __name__ == "__main__":
    run()
