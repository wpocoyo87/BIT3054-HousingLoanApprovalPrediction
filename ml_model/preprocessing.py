import pandas as pd
import numpy as np

def preprocess_training_data(df):
    """
    Cleans and prepares the dataset for the ML models.
    Matches the logic used in professional banking assessments (Stress Test vs Actual).
    """
    # 1. Feature Engineering (Professional Banking Logic)
    
    # Use 6.5% Stress Test Rate for DSR (standard risk assessment)
    stress_ir = 6.5 / 100 / 12
    # Use Interest_Rate from data if available, else fallback to 4.5%
    actual_ir_val = df['Interest_Rate'] if 'Interest_Rate' in df.columns else 4.5
    actual_ir = (actual_ir_val / 100) / 12
    
    n = df['Loan_Amount_Term']
    p = df['LoanAmount']
    
    # Calculate Two Types of Installments
    # A. Stress Installment (for DSR)
    stress_inst = (p * stress_ir * (1 + stress_ir)**n) / ((1 + stress_ir)**n - 1)
    
    # B. Actual Installment (for NDI)
    actual_inst = (p * actual_ir * (1 + actual_ir)**n) / ((1 + actual_ir)**n - 1)
    
    # Update Dataframe with more realistic numbers
    # Total Income for better accuracy
    co_inc = df['CoapplicantIncome'].fillna(0) if 'CoapplicantIncome' in df.columns else 0
    total_income = df['ApplicantIncome'] + co_inc
    
    df['Monthly_Installment'] = actual_inst 
    df['NDI'] = total_income - actual_inst
    df['DSR'] = (stress_inst / total_income) * 100

    # 2. Select Features (Matching your X list)
    features = ['ApplicantIncome', 'Credit_Score', 'DSR', 'NDI', 'Monthly_Installment']
    X = df[features].copy()
    
    # 3. Handle Target Label
    if 'Loan_Status' in df.columns:
        y = df['Loan_Status'].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0).astype(int)
    else:
        # Fallback to rules if Loan_Status is missing (e.g. for prediction)
        y = ((df['DSR'] < 70) & (df['NDI'] > 1500) & (df['Credit_Score'] > 600)).astype(int)

    # 4. Final Cleanup
    X = X.fillna(0)

    return X, y

def preprocess_input(df):
    """
    Preprocesses a single application (or batch) for prediction.
    Only returns the features (X).
    """
    X, _ = preprocess_training_data(df)
    return X
