import pandas as pd
import numpy as np

def preprocess_training_data(df):
    """
    Cleans and prepares the dataset for the ML models.
    Matches professional banking standards: Area-aware NDI and Experience-aware Scoring.
    """
    df = df.copy()
    
    # 1. Matching Colab Imputers
    df['Gender'] = df['Gender'].fillna('Female')
    df['Dependents'] = df['Dependents'].fillna(0).astype(str).str.extract('(\d+)').fillna(0).astype(int)
    
    # Critical Fix: Ensure high score = Clean Credit History for ML
    df['Credit_History'] = df['Credit_History'].fillna(1.0)
    df.loc[df['Credit_Score'] >= 650, 'Credit_History'] = 1.0
    
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Years_Employed'] = df['Years_Employed'].fillna(1).astype(float)
    
    # Handle different education naming conventions
    df['Education'] = df['Education'].astype(str).str.title()
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0, '1': 1, '0': 0}).fillna(0)
    
    # 2. Dynamic NDI Buffer calculation (Area-Aware Logic)
    def get_ndi_threshold(row):
        area = str(row.get('Property_Area', 'Urban'))
        if area == 'Urban': return 1500
        if area == 'Semiurban': return 1200
        return 1000

    # 3. Feature Engineering
    stress_ir = 6.5 / 100 / 12
    actual_ir = (df['Interest_Rate'].fillna(4.5) / 100 / 12)
    n = df['Loan_Amount_Term'].fillna(360) 
    p = df['LoanAmount']
    
    stress_inst = (p * stress_ir * (1 + stress_ir)**n) / ((1 + stress_ir)**n - 1)
    actual_inst = (p * actual_ir * (1 + actual_ir)**n) / ((1 + actual_ir)**n - 1)
    
    total_income = df['ApplicantIncome'] + df['CoapplicantIncome'].fillna(0)
    
    df['Monthly_Installment'] = actual_inst 
    df['DSR'] = (stress_inst / total_income.replace(0, 1)) * 100
    df['NDI'] = total_income - actual_inst - df.get('Monthly_Commitments', 0).fillna(0)
    
    # 4. Professional Labeling Logic (Experience & Area Aware)
    df['NDI_Threshold'] = df.apply(get_ndi_threshold, axis=1)
    dsr_limit = np.where(df['Years_Employed'] >= 1, 75, 65)
    
    df['Loan_Status'] = np.where(
        (df['DSR'] <= dsr_limit) & 
        (df['NDI'] >= df['NDI_Threshold']) &
        (df['Credit_Score'].fillna(600) >= 650) &
        (df['Credit_History'] > 0.5),
        'Y', 'N'
    )

    # 5. Final Feature Selection (Expanded)
    features = ['ApplicantIncome', 'Credit_Score', 'DSR', 'NDI', 'Monthly_Installment', 'Years_Employed', 'Education', 'Dependents']
    X = df[features].copy()
    y = df['Loan_Status'].map({'Y': 1, 'N': 0}).fillna(0).astype(int)
    
    X = X.fillna(X.mean())

    return X, y

def preprocess_input(df):
    X, _ = preprocess_training_data(df)
    return X
