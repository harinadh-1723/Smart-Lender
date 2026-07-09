import os
import pickle
import pandas as pd
import numpy as np

def run_suite():
    # Load model and scaler
    model_path = "rdf.pkl"
    if not os.path.exists(model_path):
        print(f"Error: model file {model_path} not found.")
        return
        
    with open(model_path, "rb") as f:
        data = pickle.load(f)
        
    model = data["model"]
    scaler = data["scaler"]
    features_list = data["features"]
    mappings = data["mappings"]

    # Define 20 test cases
    test_cases = [
        # Approved Cases (Low Risk, Good credit history, stable income)
        {"Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 8000, "CoapplicantIncome": 2000, "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Semiurban", "Expected": 1},
        {"Gender": "Female", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 12000, "CoapplicantIncome": 0, "LoanAmount": 200, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Urban", "Expected": 1},
        {"Gender": "Male", "Married": "Yes", "Dependents": "2", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 6000, "CoapplicantIncome": 1500, "LoanAmount": 100, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Semiurban", "Expected": 1},
        {"Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 9500, "CoapplicantIncome": 3000, "LoanAmount": 250, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Urban", "Expected": 1},
        {"Gender": "Female", "Married": "No", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 7000, "CoapplicantIncome": 2000, "LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Semiurban", "Expected": 1},
        
        # Rejected Cases (Poor Credit History = 0.0)
        {"Gender": "Male", "Married": "No", "Dependents": "3+", "Education": "Not Graduate", "Self_Employed": "Yes", "ApplicantIncome": 5000, "CoapplicantIncome": 0, "LoanAmount": 150, "Loan_Amount_Term": 360, "Credit_History": "0.0", "Property_Area": "Rural", "Expected": 0},
        {"Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 8000, "CoapplicantIncome": 2000, "LoanAmount": 250, "Loan_Amount_Term": 360, "Credit_History": "0.0", "Property_Area": "Urban", "Expected": 0},
        {"Gender": "Female", "Married": "No", "Dependents": "0", "Education": "Not Graduate", "Self_Employed": "No", "ApplicantIncome": 3500, "CoapplicantIncome": 0, "LoanAmount": 100, "Loan_Amount_Term": 360, "Credit_History": "0.0", "Property_Area": "Rural", "Expected": 0},
        {"Gender": "Male", "Married": "Yes", "Dependents": "2", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 12000, "CoapplicantIncome": 0, "LoanAmount": 300, "Loan_Amount_Term": 360, "Credit_History": "0.0", "Property_Area": "Urban", "Expected": 0},
        {"Gender": "Female", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 6000, "CoapplicantIncome": 1000, "LoanAmount": 120, "Loan_Amount_Term": 360, "Credit_History": "0.0", "Property_Area": "Semiurban", "Expected": 0},
        
        # Rejected Cases (High Debt-to-Income / DTI > 45%)
        {"Gender": "Male", "Married": "No", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 3000, "CoapplicantIncome": 0, "LoanAmount": 500, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Rural", "Expected": 0},
        {"Gender": "Female", "Married": "No", "Dependents": "1", "Education": "Not Graduate", "Self_Employed": "Yes", "ApplicantIncome": 2000, "CoapplicantIncome": 0, "LoanAmount": 350, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Rural", "Expected": 0},
        {"Gender": "Male", "Married": "Yes", "Dependents": "2", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 4000, "CoapplicantIncome": 0, "LoanAmount": 600, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Urban", "Expected": 0},
        {"Gender": "Male", "Married": "No", "Dependents": "3+", "Education": "Not Graduate", "Self_Employed": "No", "ApplicantIncome": 2500, "CoapplicantIncome": 0, "LoanAmount": 450, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Rural", "Expected": 0},
        {"Gender": "Female", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 3500, "CoapplicantIncome": 0, "LoanAmount": 550, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Semiurban", "Expected": 0},
        
        # Mixed Cases (Medium Risk, Moderate parameters)
        {"Gender": "Male", "Married": "Yes", "Dependents": "1", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 4500, "CoapplicantIncome": 1000, "LoanAmount": 180, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Semiurban", "Expected": 1},
        {"Gender": "Female", "Married": "No", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 3500, "CoapplicantIncome": 0, "LoanAmount": 110, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Rural", "Expected": 1},
        {"Gender": "Male", "Married": "Yes", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 3000, "CoapplicantIncome": 1200, "LoanAmount": 130, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Semiurban", "Expected": 1},
        {"Gender": "Male", "Married": "No", "Dependents": "0", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 5000, "CoapplicantIncome": 0, "LoanAmount": 140, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Urban", "Expected": 1},
        {"Gender": "Male", "Married": "Yes", "Dependents": "2", "Education": "Graduate", "Self_Employed": "No", "ApplicantIncome": 2800, "CoapplicantIncome": 0, "LoanAmount": 90, "Loan_Amount_Term": 360, "Credit_History": "1.0", "Property_Area": "Semiurban", "Expected": 1}
    ]

    results = []
    for idx, case in enumerate(test_cases):
        # Mappings
        gender = mappings['Gender'].get(case['Gender'], 1)
        married = mappings['Married'].get(case['Married'], 0)
        dependents = mappings['Dependents'].get(case['Dependents'], 0)
        education = mappings['Education'].get(case['Education'], 1)
        self_employed = mappings['Self_Employed'].get(case['Self_Employed'], 0)
        property_area = mappings['Property_Area'].get(case['Property_Area'], 2)
        credit_history = float(case['Credit_History'])
        
        app_income = float(case['ApplicantIncome'])
        coapp_income = float(case['CoapplicantIncome'])
        loan_amt = float(case['LoanAmount'])
        term = float(case['Loan_Amount_Term'])
        
        # Build features df
        features = pd.DataFrame([[
            gender, married, dependents, education, self_employed,
            app_income, coapp_income, loan_amt, term, credit_history, property_area
        ]], columns=features_list)
        
        # Transform and Predict
        scaled_features = scaler.transform(features)
        scaled_features_df = pd.DataFrame(scaled_features, columns=features_list)
        pred = model.predict(scaled_features_df)[0]
        
        # Compute Risk Level
        monthly_emi = (loan_amt * 1000) / term
        total_income = app_income + coapp_income
        dti = monthly_emi / total_income
        
        if credit_history == 0.0 or dti > 0.45 or (total_income < 2500 and loan_amt > 150):
            risk_level = "High Risk"
        elif credit_history == 1.0 and dti <= 0.30 and total_income >= 4000:
            risk_level = "Low Risk"
        else:
            risk_level = "Medium Risk"
            
        results.append({
            "No": idx + 1,
            "Credit History": case["Credit_History"],
            "Income": f"${case['ApplicantIncome']}",
            "Loan": f"${case['LoanAmount']}k",
            "DTI": f"{dti * 100:.1f}%",
            "Expected": "Approved" if case["Expected"] == 1 else "Rejected",
            "Actual": "Approved" if pred == 1 else "Rejected",
            "Risk Level": risk_level
        })
        
    os.makedirs("docs", exist_ok=True)
    report_path = "docs/Test_Suite_Results.md"
    
    with open(report_path, "w") as f:
        f.write("# Automated Test Suite Results\n\n")
        f.write("This document summarizes the validation of 20 realistic test cases against the trained **Smart Lender** XGBoost classifier model.\n\n")
        f.write("## Summary Table\n\n")
        
        # Write manual markdown table
        f.write("| No | Credit History | Income | Loan | DTI | Expected | Actual | Risk Level |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for row in results:
            f.write(f"| {row['No']} | {row['Credit History']} | {row['Income']} | {row['Loan']} | {row['DTI']} | {row['Expected']} | {row['Actual']} | {row['Risk Level']} |\n")
            
        f.write("\n\n")
        f.write("## Risk Analysis & Logic Definitions\n")
        f.write("- **High Risk (Red):** poor credit history OR high loan-to-income ratio (DTI > 45%) OR extremely low income combined with high loan size.\n")
        f.write("- **Medium Risk (Orange):** moderate debt ratios and average incomes with basic credit standings.\n")
        f.write("- **Low Risk (Green):** stable credit standing bureau histories, comfortable DTI ratios (DTI <= 30%), and high incomes.\n")
        
    print(f"Test suite report generated successfully at: {report_path}")

if __name__ == "__main__":
    run_suite()
