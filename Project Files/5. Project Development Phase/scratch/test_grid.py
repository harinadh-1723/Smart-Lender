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

    # Generate 100 test cases
    results = []
    for credit_hist in [1.0, 0.0]:
        for income in [1500, 3000, 5000, 10000]:
            for loan in [15, 30, 50, 100, 150]:
                case = pd.DataFrame([[
                    mappings['Gender'].get('Male', 1),
                    mappings['Married'].get('Yes', 1),
                    mappings['Dependents'].get('0', 0),
                    mappings['Education'].get('Graduate', 1),
                    mappings['Self_Employed'].get('No', 0),
                    income,
                    0.0,
                    loan,
                    360.0,
                    credit_hist,
                    mappings['Property_Area'].get('Semiurban', 1)
                ]], columns=features_list)
                
                scaled = pd.DataFrame(scaler.transform(case), columns=features_list)
                pred = model.predict(scaled)[0]
                proba = model.predict_proba(scaled)[0][1]
                
                results.append({
                    "Credit_History": credit_hist,
                    "Income": income,
                    "Loan": loan,
                    "Prediction": pred,
                    "Probability": round(proba, 4)
                })
                
    df = pd.DataFrame(results)
    print("All predictions:")
    print(df.to_string())
    print("\nTotal Approved:", sum(df["Prediction"] == 1))
    print("Total Rejected:", sum(df["Prediction"] == 0))

if __name__ == "__main__":
    run()
