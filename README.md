# 🏠 AI-Powered Malaysian Housing Loan Approval Prediction System

**Course:** BIT3054 - Data Science  
**Prediction Type:** Binary Classification — `Approved (1)` / `Rejected (0)`

---

## 📌 Project Goal

A full-stack intelligent web system for **Housing Loan Analysts and Property Agents** to instantly predict client loan approval probability using machine learning. The system also provides AI-powered financial insights, maximum property price recommendations, and bank-matching for rejected applicants.

The Machine Learning model actively learns and mimics authentic Malaysian Banking Rules, including:
- **Maximum Debt Service Ratio (DSR):** 75% for experienced professionals.
- **Minimum Net Disposable Income (NDI):** RM 1,500 for Urban applicants.
- **CCRIS / Credit Score:** Strict adherence to a 650 minimum threshold.

---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| **Backend** | Python 3, Flask |
| **Machine Learning** | Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn |
| **Database** | MongoDB (NoSQL) via MongoEngine *(Note: PostgreSQL has been completely removed in favor of MongoDB)* |
| **Frontend UI** | HTML5, Bootstrap 5, Jinja2 |
| **Deployment** | Render.com (connected to GitHub for CI/CD) |

---

## 🚀 Setup & Run Locally

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Train the ML model (generates model.pkl)
python ml_model/train_model.py

# 4. Generate the visualization dashboard plots
python ml_model/generate_plots.py

# 5. Run the Flask development server
python run.py
```

> **Note:** The app uses **MongoDB Atlas** in production. For local development, it automatically falls back to `mongomock` (no local MongoDB installation required).

---

## 🤖 Machine Learning Models

- **Baseline:** Logistic Regression (`max_iter=1000`)
- **Improved:** Random Forest Classifier (`n_estimators=100, random_state=42`)
- **Advanced:** XGBoost Classifier
- **Train/Test Split:** 80:20 on 5,000 simulated, realistic Malaysian bank records (with 5% dynamic real-world noise)
- **Key Metrics (Random Forest):** Accuracy: **81.0%** | F1 Score: **0.75** | AUC-ROC: **0.867**

---

## 👥 Role-Based Access

| Role | Capabilities |
|:---|:---|
| **Admin (Manager)** | Manage bank staff/users, upload dataset, manually retrain ML model, access full Evaluation Plots / Diagnostic Dashboard, export all system records. |
| **User (Loan Analyst / Agent)** | Submit single loan assessment forms, receive AI predictions & DSR/NDI feedback, export personal Monthly Reports. |

---

## 🌟 Key Features
- **Dynamic Dashboarding:** System auto-generates 1x3 Visualization Dashboards (Feature Importance, Confusion Matrix, ROC-AUC) whenever the Admin retrains the model.
- **Decision Boundary Plotting:** Provides visual proof of DSR vs NDI thresholds for each applicant using Matplotlib.
- **Export Monthly Report:** Feature allowing branch managers to download CSV data of all tested loan inputs.
- **Explainable AI:** Uses Feature Importances to explain *why* an applicant was approved or rejected (Income vs Commitments).

---

## 📁 Project Structure

```
├── app/                        # Flask routes, auth, models (MongoEngine)
├── ml_model/
│   ├── train_model.py          # ML training script
│   ├── evaluate_model.py       # Evaluation metrics
│   ├── generate_plots.py       # Generates DSR vs NDI and 1x3 Diagnostic Dashboard
│   ├── preprocessing.py        # Feature engineering & strict bank mappings
│   └── loan_data.csv           # Final dataset with 5000+ realistic records
├── static/                     # CSS, AI-generated plots
├── templates/                  # Jinja2 HTML pages (Dashboard, Docs, Eval)
├── generate_realistic_data.py  # Python script to synthesize raw applicant data
├── config.py                   # MongoDB Atlas & Application configuration
└── run.py                      # Server Launch Script
```
