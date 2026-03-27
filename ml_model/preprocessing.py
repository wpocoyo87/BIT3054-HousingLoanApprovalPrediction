import pandas as pd
import numpy as np

def preprocess_training_data(df):
    """
    Cleans and prepares the dataset for the ML models.
    Matches the logic used in professional banking assessments (Stress Test vs Actual).
    """
    # 1. Matching Colab Imputers (Step 5 in Colab)
    df = df.copy()
    
    # Filling missing Gender with a default (e.g., 'Female')
    df['Gender'] = df['Gender'].fillna('Female')
    # Filling missing Dependents with 0
    df['Dependents'] = df['Dependents'].fillna(0)
    # Filling missing Credit_History with the mean
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mean())
    # Filling missing LoanAmount with the average
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    
    # 2. Feature Engineering (Professional Banking Logic)
    stress_ir = 6.5 / 100 / 12
    # Fallback to 4.5% if missing
    actual_ir = (df['Interest_Rate'].fillna(4.5) / 100 / 12)
    
    n = df['Loan_Amount_Term'].fillna(360) 
    p = df['LoanAmount']
    
    # Calculate Two Types of Installments
    stress_inst = (p * stress_ir * (1 + stress_ir)**n) / ((1 + stress_ir)**n - 1)
    actual_inst = (p * actual_ir * (1 + actual_ir)**n) / ((1 + actual_ir)**n - 1)
    
    co_inc = df['CoapplicantIncome'].fillna(0)
    total_income = df['ApplicantIncome'] + co_inc
    
    df['Monthly_Installment'] = actual_inst 
    df['NDI'] = total_income - actual_inst
    df['DSR'] = (stress_inst / total_income.replace(0, 1)) * 100

    # 3. Label Logic (Matching Colab mapping)
    if 'Loan_Status' not in df.columns or df['Loan_Status'].isnull().any():
        # Create rule-based labels if missing (Matching Step 8 in Colab)
        df['Loan_Status'] = np.where(
            (df['DSR'] < 70) &
            (df['Credit_Score'].fillna(600) > 600) &
            (df['NDI'] > 1500) &
            (df['Credit_History'] > 0.5),
            'Y', 'N'
        )

    # 4. Final Selection
    features = ['ApplicantIncome', 'Credit_Score', 'DSR', 'NDI', 'Monthly_Installment']
    X = df[features].copy()
    y = df['Loan_Status'].map({'Y': 1, 'N': 0, 1: 1, 0: 0}).fillna(0).astype(int)
    
    # Ensure no NaNs remain in features (Matching Colab cleanup)
    X = X.fillna(X.mean())

    return X, y

def preprocess_input(df):
    """
    Preprocesses a single application (or batch) for prediction.
    Only returns the features (X).
    """
    X, _ = preprocess_training_data(df)
    return X
