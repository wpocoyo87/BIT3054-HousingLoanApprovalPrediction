import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from ml_model.preprocessing import preprocess_training_data

# Create plots directory
os.makedirs('static/plots', exist_ok=True)

# 1. Load Data
df = pd.read_csv('ml_model/loan_data.csv')

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
# Select only numeric for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap (Malaysian Loan Data)')
plt.savefig('static/plots/correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 3. DSR vs. NDI Distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='DSR', y='NDI', hue='Loan_Status', alpha=0.5)
plt.axvline(x=70, color='red', linestyle='--', label='DSR Limit (70%)')
plt.axhline(y=1500, color='blue', linestyle='--', label='Urban NDI Limit (RM 1,500)')
plt.title('DSR vs NDI - Separation of Loan Status')
plt.legend()
plt.savefig('static/plots/dsr_ndi_distribution.png')
plt.close()

# 4. Feature Importance (from trained model)
model_path = 'ml_model/model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    X_train, y_train = preprocess_training_data(df)
    
    importances = model.feature_importances_
    features = X_train.columns
    feat_importances = pd.Series(importances, index=features).sort_values(ascending=True)

    
    plt.figure(figsize=(10, 8))
    feat_importances.plot(kind='barh', color='teal')
    plt.title('Feature Importance (Random Forest Model)')
    plt.xlabel('Importance')
    plt.savefig('static/plots/feature_importance.png', bbox_inches='tight')
    plt.close()

print("Successfully generated all project visuals in static/plots/")
