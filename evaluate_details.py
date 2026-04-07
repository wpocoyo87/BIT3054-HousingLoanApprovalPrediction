import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys

# Ensure correct path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

BASE_DIR = 'ml_model'
df = pd.read_csv(os.path.join(BASE_DIR, 'loan_data.csv'))

from ml_model.preprocessing import preprocess_training_data
X, y = preprocess_training_data(df)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
y_pred = model.predict(X_test)

print("--- Accuracy ---")
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

print("--- Classification Report ---")
print(classification_report(y_test, y_pred))
