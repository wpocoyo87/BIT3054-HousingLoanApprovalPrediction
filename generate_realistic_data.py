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

        # 2. Malaysian Geographical Data

        state = random.choice([
            "Selangor", "WP Kuala Lumpur", "Johor", "Penang", "Perak", 
            "Kedah", "Melaka", "Negeri Sembilan", "Pahang", "Perlis", 
            "Kelantan", "Terengganu", "Sabah", "Sarawak", "WP Putrajaya"
        ])
        bumi_status = random.choices(["Yes", "No"], weights=[0.65, 0.35])[0]
        
        # 3. Employment & Income (in MYR)
        employment_sector = random.choice(["Private", "Government", "Statutory", "Professional", "Self-Employed"])
        lppsa_eligible = "Yes" if employment_sector in ["Government", "Statutory"] and random.random() > 0.1 else "No"
        
        applicant_income = int(np.random.lognormal(mean=8.5, sigma=0.6))
        applicant_income = max(1800, min(applicant_income, 60000)) # MYR
        
        if married == "Yes" and random.random() > 0.4:
            coapplicant_income = int(np.random.lognormal(mean=8.2, sigma=0.6))
            coapplicant_income = max(1500, min(coapplicant_income, 45000))
        else:
            coapplicant_income = 0
            
        total_income_monthly = applicant_income + coapplicant_income
        
        # 4. Property & Loan Amount
        property_area = random.choice(["Urban", "Semiurban", "Rural"])
        property_type = random.choice(["Terrace", "Condominium", "Bungalow", "Apartment", "Townhouse"])
        financing_type = random.choice(["Islamic", "Conventional"])
        
        # Realistic Malaysian Property Prices
        if property_area == "Urban":
            property_value = random.randint(400, 2000) # 400k to 2m
        elif property_area == "Semiurban":
            property_value = random.randint(300, 900)  # 300k to 900k
        else:
            property_value = random.randint(150, 500)  # 150k to 500k
            
        margin = random.choices([70, 80, 90, 100], weights=[0.1, 0.2, 0.6, 0.1])[0]
        loan_amount = (property_value * margin) / 100.0
        
        # 5. Loan Term (Months)
        loan_term = random.choices([180, 240, 300, 360, 420], weights=[0.05, 0.1, 0.2, 0.6, 0.05])[0]
        
        # 6. Credit Score (300 to 850)
        base_score = np.random.normal(loc=660, scale=90)
        if applicant_income > 7000: base_score += 25
        if education == "Graduate": base_score += 15
        credit_score = int(max(300, min(base_score, 850)))
        credit_history = 1.0 if credit_score >= 640 else 0.0
        
        # 7. DSR & NDI Logic
        P = loan_amount * 1000
        r = 0.043 / 12 # 4.3% p.a. approx
        n = loan_term
        monthly_installment = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
        
        # Estimate other commitments
        other_commitments = total_income_monthly * random.uniform(0.1, 0.4)
        dsr = ((monthly_installment + other_commitments) / total_income_monthly) * 100
        
        # NDI = Total Income - Monthly Installment - Other Commitments
        ndi = total_income_monthly - monthly_installment - other_commitments
        
        # 8. Approval Logic (Enhanced with NDI)
        loan_status = 'N'
        # Rule-based foundation for simulation
        # Minimum NDI threshold for Malaysia (usually RM 1500 to RM 3000 depending on location)
        min_ndi = 2500 if property_area == "Urban" else 1800
        
        if lppsa_eligible == "Yes":
            if credit_score >= 580 and dsr <= 85 and ndi >= (min_ndi * 0.8): loan_status = 'Y'
        else:
            if credit_score >= 660 and dsr <= 70 and ndi >= min_ndi: loan_status = 'Y'
            elif credit_score >= 620 and dsr <= 55 and ndi >= (min_ndi + 500): loan_status = 'Y'
            elif credit_score >= 750 and dsr <= 80 and ndi >= (min_ndi * 0.9): loan_status = 'Y'
        
        # Min income check
        if total_income_monthly < 2500 and loan_status == 'Y':
            loan_status = 'N'

        # Random exceptions
        if random.random() < 0.04:
            loan_status = 'Y' if loan_status == 'N' else 'N'

        data.append([
            loan_id, gender, married, dependents, education, employment_sector,
            applicant_income, coapplicant_income, loan_amount, loan_term,
            credit_history, property_area, credit_score, state, bumi_status,
            lppsa_eligible, property_type, margin, financing_type, dsr, ndi, loan_status
        ])
        
    return pd.DataFrame(data, columns=[
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Employment_Sector',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'Credit_Score', 'State', 'Bumi_Status',
        'LPPSA_Eligible', 'Property_Type', 'Margin', 'Financing_Type', 'DSR', 'NDI', 'Loan_Status'
    ])


df = generate_data()
df.to_csv("ml_model/loan_data.csv", index=False)
print("Successfully generated 5000 rows of enhanced Malaysian Financing Data!")
print(f"Approval Rate: {df['Loan_Status'].value_counts(normalize=True)['Y']:.1%}")

