import pandas as pd
import numpy as np

def preprocess_input(df):
    """
    Preprocess user input from the web form matching training data.
    """
    df = df.copy()
    
    education_mapping = {'Graduate': 1, 'Not Graduate': 0}
    married_mapping = {'Yes': 1, 'No': 0}
    lppsa_mapping = {'Yes': 1, 'No': 0}
    
    df['Education'] = df['Education'].map(education_mapping).fillna(1)
    df['Married'] = df['Married'].map(married_mapping).fillna(0)
    df['LPPSA_Eligible_Binary'] = df['LPPSA_Eligible'].map(lppsa_mapping).fillna(0)
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)
    
    # Handle Property Area dummies
    df['Property_Area_Rural'] = (df['Property_Area'] == 'Rural').astype(int)
    df['Property_Area_Semiurban'] = (df['Property_Area'] == 'Semiurban').astype(int)
    df['Property_Area_Urban'] = (df['Property_Area'] == 'Urban').astype(int)
    
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_Score', 'Education', 'Married', 'Dependents', 'DSR', 'NDI', 'LPPSA_Eligible_Binary',
            'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']
            
    for col in cols:
        if col not in df.columns:
            df[col] = 0
            
    return df[cols]

def preprocess_training_data(df):
    """
    Preprocess the raw training dataset.
    """
    df = df.copy()
    
    # Fix LPPSA Mapping
    df['LPPSA_Eligible_Binary'] = df['LPPSA_Eligible'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # Map target
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0}).fillna(1)
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0}).fillna(0)
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)
    
    # One-hot encoding for Property_Area
    if 'Property_Area' in df.columns:
        for area in ['Rural', 'Semiurban', 'Urban']:
            df[f'Property_Area_{area}'] = (df['Property_Area'] == area).astype(int)
            
    if 'Loan_Status' in df.columns:
        X = df.drop(columns=['Loan_Status'])
        y = df['Loan_Status']
    else:
        X = df
        y = None
        
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_Score', 'Education', 'Married', 'Dependents', 'DSR', 'NDI', 'LPPSA_Eligible_Binary',
            'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']

            
    for col in cols:
        if col not in X.columns:
            X[col] = 0
            
    X = X[cols]
    return X, y

