import pandas as pd
import numpy as np
import joblib
import os
import sys

# Ensure the ML model directory is in the path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    from preprocessing import preprocess_training_data
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.getcwd())
    from ml_model.preprocessing import preprocess_training_data

def train_and_save(model_type='random_forest'):
    data_path = os.path.join(BASE_DIR, 'loan_data.csv')
    save_path = os.path.join(BASE_DIR, 'model.pkl')

    if not os.path.exists(data_path):
        print(f"FAILED: Data file not found at {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X, y = preprocess_training_data(df)

    # 80/20 Split for Realism
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = model_type.lower().strip()
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'xgboost':
        model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=15, random_state=42)

    print(f"Training {model_type} on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    joblib.dump(model, save_path)
    print(f"SUCCESS: Model saved to {save_path}")
    return save_path

if __name__ == '__main__':
    mt = sys.argv[1] if len(sys.argv) > 1 else 'random_forest'
    train_and_save(model_type=mt)
