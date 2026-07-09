import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_loan_data(num_records=800):
    loan_ids = [f"LP{1000 + i:06d}" for i in range(num_records)]
    
    # Probabilities for categorical features based on Kaggle distributions
    genders = np.random.choice(["Male", "Female"], size=num_records, p=[0.80, 0.20])
    married = np.random.choice(["Yes", "No"], size=num_records, p=[0.65, 0.35])
    dependents = np.random.choice(["0", "1", "2", "3+"], size=num_records, p=[0.55, 0.18, 0.17, 0.10])
    education = np.random.choice(["Graduate", "Not Graduate"], size=num_records, p=[0.75, 0.25])
    self_employed = np.random.choice(["Yes", "No"], size=num_records, p=[0.15, 0.85])
    
    # Continuous features (Realistic ranges)
    # Monthly Applicant Income: $1,500 to $25,000 (mean around $6,000)
    applicant_income = np.random.exponential(scale=4500, size=num_records) + 1500
    applicant_income = np.clip(applicant_income, 1500, 25000).astype(int)
    
    # Monthly Coapplicant Income: $0 to $12,000
    has_coapplicant = np.random.choice([True, False], size=num_records, p=[0.50, 0.50])
    coapplicant_income = np.zeros(num_records)
    coapplicant_income[has_coapplicant] = np.random.exponential(scale=3500, size=sum(has_coapplicant)) + 1000
    coapplicant_income = np.clip(coapplicant_income, 0, 12000).astype(int)
    
    total_income = applicant_income + coapplicant_income
    
    # Loan Amount in thousands: $10k to $700k (sensible mapping based on income)
    # Average loan is 2.5x to 4x of total annual income, but here we generate it as a factor of monthly total income
    loan_amount = (total_income * np.random.uniform(1.5, 4.0, size=num_records)).astype(int)
    # Scale loan amount to thousands (e.g. 150 represents 150,000)
    loan_amount = np.clip(loan_amount // 100, 15, 650)
    
    # Loan term (mostly 360 months, i.e. 30 years)
    loan_amount_term = np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360, 480], size=num_records, p=[0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.01, 0.02, 0.84, 0.02])
    
    # Credit History: 80% good, 20% bad
    credit_history = np.random.choice([1.0, 0.0], size=num_records, p=[0.80, 0.20])
    
    # Property area
    property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=num_records, p=[0.33, 0.38, 0.29])
    
    # Inject missing values (around 2-3% to preserve demonstration of cleaning)
    def inject_nan(arr, p=0.03):
        arr = arr.astype(object)
        mask = np.random.random(len(arr)) < p
        arr[mask] = np.nan
        return arr

    genders = inject_nan(genders, 0.02)
    married = inject_nan(married, 0.01)
    dependents = inject_nan(dependents, 0.02)
    self_employed = inject_nan(self_employed, 0.03)
    loan_amount = inject_nan(loan_amount, 0.02)
    loan_amount_term = inject_nan(loan_amount_term, 0.02)
    credit_history = inject_nan(credit_history, 0.05)
    
    # Determine Loan Status with STRICT realistic underwriting rules
    loan_status = []
    for i in range(num_records):
        ch = credit_history[i]
        inc = total_income[i]
        amt = loan_amount[i] if not pd.isna(loan_amount[i]) else 150
        term = loan_amount_term[i] if not pd.isna(loan_amount_term[i]) else 360
        edu = education[i]
        
        # Calculate monthly EMI
        monthly_emi = (amt * 1000) / term
        debt_to_income = monthly_emi / inc
        
        # Strict underwriting rules
        prob = 0.85 # Base high probability for ideal cases
        
        # 1. Credit History is the primary constraint
        if ch == 0.0:
            prob = 0.05 # Rejection rate is 95% if credit history is bad
        # 2. Debt to income constraint
        elif debt_to_income > 0.45:
            prob = 0.10 # Rejection rate is 90% if debt-to-income is too high
        elif debt_to_income > 0.35:
            prob = 0.45 # Moderate chance of approval
            
        # 3. Income check
        if inc < 2500 and ch != 0.0:
            prob = min(prob, 0.30) # Low income restricts approval
            
        # 4. Small modifiers for other positive indicators
        if ch == 1.0 and debt_to_income <= 0.35:
            if edu == "Graduate":
                prob = min(prob + 0.05, 0.98)
            else:
                prob = max(prob - 0.10, 0.50)
            if married[i] == "Yes":
                prob = min(prob + 0.05, 0.98)
            if property_area[i] == "Semiurban":
                prob = min(prob + 0.05, 0.98)
                
        # Draw status based on score
        status = "Y" if np.random.random() < prob else "N"
        loan_status.append(status)
        
    df = pd.DataFrame({
        "Loan_ID": loan_ids,
        "Gender": genders,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_loan_data()
    df.to_csv("data/loan_data.csv", index=False)
    print("Dataset generated and saved successfully to data/loan_data.csv")
    print(df.info())
    print("\nTarget Class Distribution:")
    print(df["Loan_Status"].value_counts())
