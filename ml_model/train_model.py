import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Absolute path to project root to allow 'app' imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Ensure current directory is also in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

try:
    from preprocessing import preprocess_training_data
except ImportError:
    # Fallback for different execution contexts
    sys.path.append(os.getcwd())
    from ml_model.preprocessing import preprocess_training_data

def train_and_save(model_type='random_forest'):
    data_path = os.path.join(BASE_DIR, 'loan_data.csv')
    save_path = os.path.join(BASE_DIR, 'model.pkl')
    metadata_path = os.path.join(BASE_DIR, 'model_metadata.json')

    if not os.path.exists(data_path):
        print(f"FAILED: Data file not found at {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X, y = preprocess_training_data(df)

    # 80/20 Split (Same as Colab/Evaluation script for consistency)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = model_type.lower().strip()
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
        model_display = "Logistic Regression"
    elif model_type == 'xgboost':
        # Try native XGBoost first, fallback to Sklearn's version if library is missing
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
            model_display = "XGBoost (XGBClassifier)"
        except (ImportError, Exception):
            print("WARNING: Native XGBoost not available, falling back to HistGradientBoostingClassifier")
            model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
            model_display = "XGBoost (HistGradientBoosting)"
    elif model_type == 'random_forest':
        # Matching Colab's default RandomForest with seed 42
        model = RandomForestClassifier(random_state=42)
        model_display = "Random Forest Classifier"
    else:
        # Default to Random Forest if an unknown type is provided
        model = RandomForestClassifier(random_state=42) # Reverted to standard 42
        model_display = "Random Forest Classifier"


    print(f"Training {model_display} on {len(X_train)} samples...")
    model.fit(X_train, y_train)

    # --- Save Model ---
    joblib.dump(model, save_path)
    print(f"SUCCESS: Model saved to {save_path}")

    # --- Calculate Metrics for Metadata ---
    print("Calculating evaluation metrics...")
    y_pred = model.predict(X_test)
    
    # Probabilities for ROC-AUC
    roc_auc = 0.5
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4)

    cm = confusion_matrix(y_test, y_pred)
    
    # Feature Importance (top 5)
    feature_importance = []
    if hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
        cols = X.columns
        for i in imps.argsort()[::-1][:5]:
            feature_importance.append({'name': cols[i].replace('_', ' ').title(), 'value': round(float(imps[i]), 4)})

    metadata = {
        'success': True,
        'model_name': model_display,
        'type': model_type,
        'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'accuracy': round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'roc_auc': roc_auc,
        'confusion_matrix': {
            'tn': int(cm[0][0]), 'fp': int(cm[0][1]),
            'fn': int(cm[1][0]), 'tp': int(cm[1][1])
        },
        'feature_importance': feature_importance,
        'sample_size': len(y_test)
    }

    # Save metadata to JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"SUCCESS: Metadata saved to {metadata_path}")

    # --- Save to Database History ---
    try:
        from mongoengine import connect, get_connection
        try:
            get_connection()
        except:
            # Initialize connection if running as standalone script
            mongo_uri = os.environ.get('MONGO_URI')
            if mongo_uri:
                connect(host=mongo_uri)
            else:
                try:
                    import mongomock
                    connect('housing_loan_db', host='mongodb://localhost', mongo_client_class=mongomock.MongoClient)
                except ImportError:
                    connect('housing_loan_db') # Fallback to real local mongo if mock not found

        from app.models import ModelResult
        db_res = ModelResult(
            model_name=metadata['model_name'],
            model_type=metadata['type'],
            accuracy=float(metadata['accuracy']),
            precision=float(metadata['precision']),
            recall=float(metadata['recall']),
            f1=float(metadata['f1']),
            roc_auc=float(metadata['roc_auc']),
            trained_at=datetime.utcnow()
        )
        db_res.save()
        print("SUCCESS: History record saved to MongoDB.")
    except Exception as e:
        print(f"WARNING: Database save failed: {e}")

    return save_path

if __name__ == '__main__':
    mt = sys.argv[1] if len(sys.argv) > 1 else 'random_forest'
    train_and_save(model_type=mt)
