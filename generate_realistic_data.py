import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

NUM_SAMPLES = 5000

def generate_data():
    data = []
    
    # Malaysian Demographic Lists
    states = ["Selangor", "Kuala Lumpur", "Johor", "Penang", "Perak", "Kedah", "Negeri Sembilan", "Melaka", "Pahang"]
    state_weights = [0.25, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01] # Approx population distribution
    
    prop_types = ["Terrace", "Condominium", "Semi-D", "Bungalow", "Apartment", "Townhouse"]
    prop_type_weights = [0.45, 0.25, 0.10, 0.05, 0.10, 0.05]
    
    for i in range(NUM_SAMPLES):
        loan_id = f"LPtest{i:04d}"
        gender = random.choice(["Male", "Female"])
        married = random.choice(["Yes", "No"])
        education = random.choices(["Graduate", "Not Graduate"], weights=[0.8, 0.2])[0]
        
        # Realistic Income Distribution (RM) for 2026
        income = int(np.random.lognormal(mean=8.8, sigma=0.5) + np.random.randint(-500, 500))
        income = max(2500, min(income, 65000))

        # Dependents (0 to 4+)
        dependents = "0"
        if married == "Yes":
            dependents = random.choices(["0", "1", "2", "3+"], weights=[0.2, 0.3, 0.3, 0.2])[0]
        else:
            dependents = random.choices(["0", "1", "2"], weights=[0.8, 0.15, 0.05])[0]

        # Coapplicant Income (often spouse)
        coapplicant_income = 0
        if married == "Yes" and random.random() > 0.3: # 70% of married couples apply together
            coapplicant_income = int(np.random.normal(income * 0.8, income * 0.3))
            coapplicant_income = max(2000, coapplicant_income)
            
        total_income = income + coapplicant_income
        
        # Property Value (RM)
        annual_total_income = total_income * 12
        base_value = annual_total_income * random.uniform(3, 7)
        
        # Clamp to realistic 2026 Malaysian ranges: RM 300k to RM 4.5M
        prop_val = int(max(300000, min(base_value, 4500000)))
        
        # Margin of Finance (MOF)
        mof = random.choices([0.90, 0.85, 0.80], weights=[0.75, 0.15, 0.10])[0]
        loan_amt = int(prop_val * mof)
        term = 360 # 30 years standard
        
        # Dynamic Interest Rates
        actual_ir = round(random.uniform(4.2, 5.0), 2)
        stress_ir = 6.5 # Standard Stress Rate
        
        # Monthly Installment Formulas
        r_actual = (actual_ir / 100) / 12
        r_stress = (stress_ir / 100) / 12
        
        # p is loan amount in RM
        p = loan_amt
        
        # actual installment for NDI
        actual_inst = (p * r_actual * (1 + r_actual)**term) / ((1 + r_actual)**term - 1)
        # stress installment for DSR
        stress_inst = (p * r_stress * (1 + r_stress)**term) / ((1 + r_stress)**term - 1)
        
        # Credit Score (300nd to 850)
        score = int(max(300, min(np.random.normal(680, 100), 850)))
        
        # commitments (PTPTN, Car, Credit Cards)
        commitments = total_income * random.uniform(0.1, 0.45)
        
        # DSR (using Total Income)
        dsr = ((stress_inst + commitments) / total_income) * 100
        # NDI (using Actual Installment)
        ndi = total_income - actual_inst - commitments
        
        # 🚨 REALISTIC LABELING LOGIC (Bank Criteria 2026)
        is_eligible = True
        if dsr > 75: is_eligible = False # Strict Cutoff
        
        # Joint/Family Floor 2026
        ndi_floor = 1800 if dependents == "0" else 2500
        if ndi < ndi_floor: is_eligible = False 
        
        if score < 600: is_eligible = False # Credit floor
        
        # Generate Status based on eligibility + higher noise (15%) for realism
        loan_status = 'Y' if is_eligible else 'N'
        if random.random() < 0.15: # Increased noise to match Colab's balanced importance
            loan_status = 'Y' if loan_status == 'N' else 'N'
            
        data.append([
            loan_id, gender, married, dependents, education, "Private",
            income, coapplicant_income, loan_amt, term,
            1.0 if score >= 650 else 0.0, 
            random.choice(["Urban", "Semiurban", "Rural"]), 
            score, 
            random.choices(states, weights=state_weights[:len(states)])[0], 
            "Yes" if random.random() > 0.3 else "No",
            "No", 
            random.choices(prop_types, weights=prop_type_weights)[0], 
            int(mof * 100), 
            random.choice(["Islamic", "Conventional"]), 
            actual_ir,
            round(dsr, 2), 
            round(ndi, 2), 
            loan_status
        ])
    
    cols = [
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Employment_Sector',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'Credit_Score', 'State', 'Bumi_Status',
        'LPPSA_Eligible', 'Property_Type', 'Margin', 'Financing_Type', 'Interest_Rate',
        'DSR', 'NDI', 'Loan_Status'
    ]
    df = pd.DataFrame(data, columns=cols)

    # Add missing values for "Data Quality" presentation (1.0% gaps)
    for col in ['Gender', 'Married', 'LoanAmount', 'Credit_History']:
        mask = df.sample(frac=0.01).index
        df.loc[mask, col] = np.nan

    return df

df = generate_data()
df.to_csv("ml_model/loan_data.csv", index=False)
print(f"Realistic Malaysian Dataset created with categories: Financing, States, Property Types.")
