import os
import pickle
import pandas as pd

def check():
    with open("rdf.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    features_list = data["features"]
    mappings = data["mappings"]

    # Test cases:
    # 1. High risk (Credit History = 0)
    case1 = pd.DataFrame([[
        mappings['Gender'].get('Male', 1),
        mappings['Married'].get('No', 0),
        mappings['Dependents'].get('3+', 3),
        mappings['Education'].get('Not Graduate', 0),
        mappings['Self_Employed'].get('Yes', 1),
        1000.0, # Low income
        0.0,
        500.0, # High loan
        360.0,
        0.0, # Bad credit history
        mappings['Property_Area'].get('Rural', 0)
    ]], columns=features_list)

    # 2. Low risk (Credit History = 1)
    case2 = pd.DataFrame([[
        mappings['Gender'].get('Male', 1),
        mappings['Married'].get('Yes', 1),
        mappings['Dependents'].get('0', 0),
        mappings['Education'].get('Graduate', 1),
        mappings['Self_Employed'].get('No', 0),
        8000.0,
        2000.0,
        100.0,
        360.0,
        1.0, # Good credit history
        mappings['Property_Area'].get('Semiurban', 1)
    ]], columns=features_list)

    # Transform and Predict
    scaled1 = pd.DataFrame(scaler.transform(case1), columns=features_list)
    scaled2 = pd.DataFrame(scaler.transform(case2), columns=features_list)

    pred1 = model.predict(scaled1)
    pred2 = model.predict(scaled2)

    print("Case 1 Prediction (High Risk):", pred1[0])
    print("Case 2 Prediction (Low Risk):", pred2[0])

if __name__ == "__main__":
    check()
