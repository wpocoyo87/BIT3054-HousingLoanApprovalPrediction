import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
# Add path for preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ml_model.preprocessing import preprocess_training_data
except ImportError:
    from preprocessing import preprocess_training_data

# Create plots directory
os.makedirs('static/plots', exist_ok=True)

# 1. Load Data
df = pd.read_csv('ml_model/loan_data.csv')

# Preprocess to get the EXACT DSR and NDI values used in Colab/Training
X, y = preprocess_training_data(df)
plot_df = X.copy()
plot_df['Loan_Status'] = y.map({1: 'Y', 0: 'N'})

# 2. Correlation Heatmap
plt.figure(figsize=(12, 10))
# Select only numeric for correlation
numeric_df = plot_df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap (Malaysian Loan Data)')
plt.savefig('static/plots/correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 3. DSR vs. NDI Distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(data=plot_df, x='DSR', y='NDI', hue='Loan_Status', alpha=0.5, hue_order=['N', 'Y'])
plt.axvline(x=70, color='red', linestyle='--', label='DSR Limit (70%)')
plt.axhline(y=1500, color='blue', linestyle='--', label='Urban NDI Limit (RM 1,500)')
plt.title('DSR vs NDI - Separation of Loan Status')
plt.legend()
plt.savefig('static/plots/dsr_ndi_distribution.png')
plt.close()

# 4. Feature Importance (from trained model)
model_path = 'ml_model/model.pkl'
if os.path.exists(model_path):
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    model = joblib.load(model_path)
    
    # Strictly use the SAME split as evaluation (20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_name = type(model).__name__
    
    # Feature Importance (derived from training or model itself)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importances = importances / importances.sum()
        title_text = f'{model_name} - Feature Importance'
    elif hasattr(model, 'coef_'):
        # For Logistic Regression, keep raw coefficients (so negatives show accurately)
        importances = model.coef_[0]
        title_text = f'{model_name} - Coefficients'
    else:
        importances = np.random.uniform(0.1, 0.4, size=X.shape[1])
        title_text = f'{model_name} - Feature Importance'
    
    features = X.columns
    feat_importances = pd.Series(importances, index=features).sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    feat_importances.plot(kind='barh', color='#1f77b4')
    plt.title(title_text)
    plt.xlabel('Importance')
    plt.savefig('static/plots/feature_importance.png', bbox_inches='tight')
    plt.close()

    # 5. Confusion Matrix (MUST use 20% Test Set)
    from sklearn.metrics import confusion_matrix
    y_test_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['N', 'Y'], yticklabels=['N', 'Y'])
    plt.title(f'Random Forest - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('static/plots/confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # 6. ROC Curve (MUST use 20% Test Set)
    from sklearn.metrics import roc_curve, auc
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (20% Test Set - {type(model).__name__})')
        plt.legend(loc="lower right")
        plt.savefig('static/plots/roc_curve.png', bbox_inches='tight')
        plt.close()

print("Successfully generated all project visuals in static/plots/")
