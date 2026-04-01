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

# Preprocess to get the EXACT DSR and NDI values
X, y = preprocess_training_data(df)
plot_df = X.copy()
# Note: we use 1 and 0 based on preprocessing label encoding mapping 
plot_df['Loan_Status'] = y.map({1: 1, 0: 0}) 

# ====================== 1. DSR vs NDI Plot (on top, separate) ======================
plt.figure(figsize=(12, 7))
sns.scatterplot(data=plot_df, x='NDI', y='DSR', 
                hue='Loan_Status', 
                palette={1: '#2ecc71', 0: '#e74c3c'}, 
                alpha=0.6)

plt.title('Decision Boundary: DSR vs NDI', fontsize=16, fontweight='bold', pad=20)
plt.axhline(y=75, color='gray', linestyle='--', linewidth=2, label='Max DSR Limit')
plt.axvline(x=1500, color='gray', linestyle=':', linewidth=2, label='Urban Min NDI')
plt.xlim(0, max(plot_df['NDI'].quantile(0.95), 5000))
plt.xlabel('NDI')
plt.ylabel('DSR')
plt.legend(title="Loan Status (1=Yes, 0=No)", loc='upper right')
plt.tight_layout()
plt.savefig('static/plots/dsr_ndi_distribution.png', bbox_inches='tight')
plt.close()


# ====================== 2. Main Dashboard - 1x3 Horizontal ======================
model_path = 'ml_model/model.pkl'
if os.path.exists(model_path):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    
    model = joblib.load(model_path)
    
    # Strictly use the SAME split as evaluation (20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_name_map = {
        'RandomForestClassifier': 'Random Forest',
        'LogisticRegression': 'Logistic Regression',
        'XGBClassifier': 'XGBoost',
        'HistGradientBoostingClassifier': 'XGBoost (HistGradientBoosting)'
    }
    raw_name = type(model).__name__
    model_name = model_name_map.get(raw_name, raw_name)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))   # Wide horizontal layout
    fig.suptitle(f'Housing Loan Approval - {model_name} Dashboard', 
                 fontsize=20, fontweight='bold', y=1.02)

    # 1. Feature Importance
    ax1 = axes[0]
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        importances = importances.sort_values(ascending=False)
        sns.barplot(x=importances.values, y=importances.index, palette='viridis', ax=ax1)
        ax1.set_title('1. Feature Importance Analysis', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Relative Importance Weight')
        
    elif hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_[0], index=X_test.columns)
        coefs = coefs.sort_values(ascending=False)
        sns.barplot(x=coefs.values, y=coefs.index, palette='viridis', ax=ax1)
        ax1.set_title('1. Feature Importance (Coefficients)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Coefficient Value')

    # 2. Confusion Matrix
    ax2 = axes[1]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Rejected (0)', 'Approved (1)'], 
                yticklabels=['Rejected (0)', 'Approved (1)'],
                annot_kws={"size": 14}, ax=ax2)
    
    ax2.set_title('2. Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Actual Status (Ground Truth)')
    ax2.set_xlabel('AI Predicted Status')

    # 3. ROC Curve
    ax3 = axes[2]
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Fallback if probability is not supported
        y_prob = model.predict(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    ax3.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.fill_between(fpr, tpr, color='darkorange', alpha=0.1)

    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_title('3. ROC Curve Analysis', fontsize=14, fontweight='bold')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc="lower right", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('static/plots/model_dashboard.png', bbox_inches='tight')
    plt.close()

print("Successfully generated DSR distribution and dynamic 1x3 model dashboard in static/plots/")
