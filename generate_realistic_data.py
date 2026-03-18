import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

NUM_SAMPLES = 5000

def generate_data():
    data = []
    
    for i in range(NUM_SAMPLES):
        # 1. Basic Demographics
        loan_id = f"LPtest{i:04d}"
        gender = random.choice(["Male", "Female"])
        married = random.choice(["Yes", "No"])
        dependents = random.choices(["0", "1", "2", "3+"], weights=[0.5, 0.2, 0.2, 0.1])[0]
        education = random.choices(["Graduate", "Not Graduate"], weights=[0.8, 0.2])[0]
        self_employed = random.choices(["No", "Yes"], weights=[0.85, 0.15])[0]
        
        # 2. Income (in MYR) - realistic for Malaysia
        # Base income between rm 2000 to rm 25000
        applicant_income = int(np.random.lognormal(mean=8.5, sigma=0.6))
        applicant_income = max(1500, min(applicant_income, 50000)) # clamp between 1.5k and 50k
        
        # Coapplicant Income (often 0 if single, otherwise some value)
        if married == "Yes" and random.random() > 0.4:
            coapplicant_income = int(np.random.lognormal(mean=8.2, sigma=0.6))
            coapplicant_income = max(1500, min(coapplicant_income, 40000))
        else:
            coapplicant_income = 0
            
        total_income_monthly = applicant_income + coapplicant_income
        
        # 3. Property & Loan Amount
        property_area = random.choice(["Urban", "Semiurban", "Rural"])
        
        # Realistic Malaysian Property Prices (in thousands, e.g., 300 = RM 300,000)
        if property_area == "Urban":
            loan_amount = random.randint(350, 1500) # 350k to 1.5m
        elif property_area == "Semiurban":
            loan_amount = random.randint(250, 800)  # 250k to 800k
        else:
            loan_amount = random.randint(100, 400)  # 100k to 400k
            
        # 4. Loan Term (Months)
        loan_term = random.choices([120, 180, 240, 300, 360, 420], weights=[0.05, 0.05, 0.1, 0.2, 0.5, 0.1])[0]
        
        # 5. Credit Score (300 to 850)
        # Higher income tends to have slightly better credit scores, but still random
        base_score = np.random.normal(loc=650, scale=80)
        if applicant_income > 8000:
            base_score += 30
        credit_score = int(max(300, min(base_score, 850)))
        
        # For backwards compatibility with existing model/form (0.0 or 1.0)
        # In reality CTOS Good is > 650 usually.
        credit_history = 1.0 if credit_score >= 650 else 0.0
        
        # 6. Loan Approval Logic (DSR - Debt Service Ratio)
        # Approximate monthly installment (assume ~4.5% interest rate p.a.)
        # Formula: M = P [ i(1 + i)^n ] / [ (1 + i)^n - 1 ]
        P = loan_amount * 1000 # convert to actual RM
        r = 0.045 / 12 # monthly rate
        n = loan_term
        monthly_installment = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
        
        # DSR = (Total Commitments / Net Income) * 100
        # Let's assume other commitments take up 15-30% of income already
        other_commitments_ratio = random.uniform(0.15, 0.35)
        dsr = ((monthly_installment + (total_income_monthly * other_commitments_ratio)) / total_income_monthly) * 100
        
        # Approval Rules:
        # 1. CTOS Score must be >= 650 (Good) to easily pass, <600 usually rejected immediately for big loans
        # 2. DSR must be <= 70% (Standard bank threshold)
        # 3. Minimum income requirement (e.g. RM 2000)
        
        loan_status = 'N'
        if total_income_monthly >= 2000:
            if credit_score >= 650 and dsr <= 70:
                loan_status = 'Y'
            elif credit_score >= 600 and dsr <= 50: # Marginal credit score but very strong DSR
                loan_status = 'Y'
            elif credit_score >= 750 and dsr <= 85: # Excellent credit, bank might allow higher DSR
                loan_status = 'Y'
                
        # Inject some random noise (banks have exceptions, 5% of the time override rules)
        if random.random() < 0.05:
            loan_status = 'Y' if loan_status == 'N' else 'N'

        data.append([
            loan_id, gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount, loan_term,
            credit_history, property_area, credit_score, loan_status
        ])
        
    return pd.DataFrame(data, columns=[
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'Credit_Score', 'Loan_Status'
    ])

df = generate_data()

# Check what the original columns look like (e.g., if there's no Credit_Score, we append it)
# We overwrite it directly here
df.to_csv("c:/Users/Safwan Rahimi/Desktop/BIT3054-HousingLoanApprovalPrediction-main/ml_model/loan_data.csv", index=False)
print("Successfully generated 5000 rows of realistic Malaysian Housing Loan Data with Credit Score!")
print("Here's a sample:")
print(df.head())
print("\nApproval Rate:")
print(df['Loan_Status'].value_counts(normalize=True))
