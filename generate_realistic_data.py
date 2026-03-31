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
    state_weights = [0.25, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01] 
    
    prop_types = ["Terrace", "Condominium", "Semi-D", "Bungalow", "Apartment", "Townhouse"]
    prop_type_weights = [0.45, 0.25, 0.10, 0.05, 0.10, 0.05]
    
    for i in range(NUM_SAMPLES):
        loan_id = f"LPtest{i:04d}"
        gender = random.choice(["Male", "Female"])
        married = random.choice(["Yes", "No"])
        education = random.choices(["Graduate", "Not Graduate"], weights=[0.8, 0.2])[0]
        
        # Position & Experience (New Feature)
        years_employed = random.randint(1, 15)
        
        # Realistic Income Distribution (RM) for 2026
        base_income = 2500 + (years_employed * 300) if education == "Graduate" else 2000 + (years_employed * 150)
        income = int(np.random.lognormal(mean=np.log(base_income), sigma=0.3))
        income = max(2000, min(income, 65000))

        # Dependents (0 to 4+)
        dep_val = 0
        if married == "Yes":
            dep_val = random.choices([0, 1, 2, 3], weights=[0.2, 0.3, 0.3, 0.2])[0]
        else:
            dep_val = random.choices([0, 1, 2], weights=[0.8, 0.15, 0.05])[0]
        dependents = str(dep_val) if dep_val < 3 else "3+"

        # Coapplicant Income
        coapplicant_income = 0
        if married == "Yes" and random.random() > 0.4: 
            coapplicant_income = int(np.random.normal(income * 0.7, income * 0.2))
            coapplicant_income = max(1500, coapplicant_income)
            
        total_income = income + coapplicant_income
        
        # Property Details
        area = random.choice(["Urban", "Semiurban", "Rural"])
        prop_val = int(total_income * 12 * random.uniform(3, 8))
        prop_val = max(250000, min(prop_val, 5000000))
        
        mof = random.choices([0.90, 0.85, 0.80], weights=[0.75, 0.15, 0.10])[0]
        loan_amt = int(prop_val * mof)
        term = 360 # 30 years
        
        actual_ir = round(random.uniform(4.1, 4.8), 2)
        stress_ir = 6.5 
        
        r_actual = (actual_ir / 100) / 12
        r_stress = (stress_ir / 100) / 12
        p = loan_amt
        
        actual_inst = (p * r_actual * (1 + r_actual)**term) / ((1 + r_actual)**term - 1)
        stress_inst = (p * r_stress * (1 + r_stress)**term) / ((1 + r_stress)**term - 1)
        
        score = int(max(300, min(np.random.normal(700, 80), 850)))
        commitments = total_income * random.uniform(0.1, 0.3) if score > 750 else total_income * random.uniform(0.2, 0.5)
        
        dsr = ((stress_inst + commitments) / total_income) * 100
        ndi = total_income - actual_inst - commitments
        
        # 🚨 BANK OFFICER LOGIC (Relaxed Experience)
        if area == 'Urban': ndi_floor = 1500 + (250 * dep_val)
        elif area == 'Semiurban': ndi_floor = 1200 + (150 * dep_val)
        else: ndi_floor = 1000 + (100 * dep_val)
        
        # Standard limit at 75% as long as employed >= 1 year
        dsr_limit = 75 if years_employed >= 1 else 65
        
        is_eligible = (dsr <= dsr_limit) and (ndi >= ndi_floor) and (score >= 650)
        
        # Credit History mapping
        credit_history = 1.0 if score >= 650 else 0.0
        if random.random() < 0.05: credit_history = 0.0 # Random credit issues
        
        if credit_history == 0.0: is_eligible = False

        loan_status = 'Y' if is_eligible else 'N'
        # Small noise for ML realism (5% only to keep logic strong)
        if random.random() < 0.05: 
            loan_status = 'Y' if loan_status == 'N' else 'N'
            
        data.append([
            loan_id, gender, married, dependents, education, "Private",
            income, coapplicant_income, loan_amt, term,
            credit_history, area, score, 
            random.choices(states, weights=state_weights[:len(states)])[0], 
            "Yes" if random.random() > 0.3 else "No",
            "No", random.choice(prop_types), int(mof * 100), 
            random.choice(["Islamic", "Conventional"]), actual_ir,
            round(dsr, 2), round(ndi, 2), years_employed, loan_status
        ])
    
    cols = [
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Employment_Sector',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Property_Area', 'Credit_Score', 'State', 'Bumi_Status',
        'LPPSA_Eligible', 'Property_Type', 'Margin', 'Financing_Type', 'Interest_Rate',
        'DSR', 'NDI', 'Years_Employed', 'Loan_Status'
    ]
    df = pd.DataFrame(data, columns=cols)

    # Add minor missing values for realism
    for col in ['Gender', 'Married', 'LoanAmount', 'Credit_History']:
        mask = df.sample(frac=0.01).index
        df.loc[mask, col] = np.nan

    return df

# Get absolute path to the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "ml_model", "loan_data.csv")

df = generate_data()
df.to_csv(csv_path, index=False)
print(f"SUCCESS: Realistic Dataset (v3) created at {csv_path}")
