import pandas as pd
import numpy as np

def preprocess_input(df):
    """
    Preprocess user input from the web form matching training data.
    """
    df = df.copy()
    
    education_mapping = {'Graduate': 1, 'Not Graduate': 0}
    married_mapping = {'Yes': 1, 'No': 0}
    
    df['Education'] = df['Education'].map(education_mapping)
    df['Married'] = df['Married'].map(married_mapping)
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)
    
    df['Property_Area_Rural'] = (df['Property_Area'] == 'Rural').astype(int)
    df['Property_Area_Semiurban'] = (df['Property_Area'] == 'Semiurban').astype(int)
    df['Property_Area_Urban'] = (df['Property_Area'] == 'Urban').astype(int)
    
    # We drop Property_Area safely if it exists
    if 'Property_Area' in df.columns:
        df = df.drop(columns=['Property_Area'])
    
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_Score', 'Education', 'Married', 'Dependents', 
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
    
    # Fill missing values
    df['Gender'] = df.get('Gender', pd.Series(['Male']*len(df))).fillna('Male')
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df.get('Self_Employed', pd.Series(['No']*len(df))).fillna('No')
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    
    if 'Credit_Score' in df.columns:
        df['Credit_Score'] = df['Credit_Score'].fillna(df['Credit_Score'].median())
    
    # Map target
    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)
    
    df = pd.get_dummies(df, columns=['Property_Area'])
    
    cols_to_drop = ['Loan_ID', 'Gender', 'Self_Employed', 'Credit_History']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    if 'Loan_Status' in df.columns:
        X = df.drop(columns=['Loan_Status'])
        y = df['Loan_Status']
    else:
        X = df
        y = None
        
    cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_Score', 'Education', 'Married', 'Dependents', 
            'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban']
            
    for col in cols:
        if col not in X.columns:
            X[col] = 0
            
    X = X[cols]
            
    return X, y
