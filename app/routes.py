from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import login_required, current_user
from app.models import LoanApplication, Prediction, User
from werkzeug.security import generate_password_hash

import math
import joblib
import os
import sys
import subprocess
import pandas as pd
from ml_model.preprocessing import preprocess_input
from ml_model.evaluate_model import get_model_evaluation_metrics

RETRAIN_TOKEN = 'admin123'

main_bp = Blueprint('main', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'model.pkl')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('main.admin_dashboard'))
        
    user_apps = LoanApplication.objects(user=current_user.id).order_by('-created_at')
    approved_count = sum(1 for app in user_apps if app.prediction == 'Approved')
    rejected_count = sum(1 for app in user_apps if app.prediction == 'Rejected')
    
    return render_template('dashboard.html', 
                           applications=user_apps, 
                           approved=approved_count, 
                           rejected=rejected_count)

@main_bp.route('/loan_form', methods=['GET', 'POST'])
@main_bp.route('/edit_assessment/<id>', methods=['GET', 'POST'])
@login_required
def loan_form(id=None):
    application = None
    if id:
        application = LoanApplication.objects(id=id).first()
        if not application:
            flash("Assessment not found.", "danger")
            return redirect(url_for('main.dashboard'))
            
    if request.method == 'POST':
        # Capture Malaysian-specific fields
        income = float(request.form.get('income', 0))
        co_income = float(request.form.get('coapplicant_income', 0))
        commitments = float(request.form.get('monthly_commitments', 0))
        loan_amt = float(request.form.get('loan_amount', 0))
        loan_term = int(request.form.get('loan_term', 360))
        
        # Calculate Monthly Installment (Approx 4.3% interest)
        P = loan_amt
        r = 0.043 / 12
        n = loan_term
        if n > 0:
            installment = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            installment = 0
            
        total_income = income + co_income
        dsr = ((installment + commitments) / total_income * 100) if total_income > 0 else 0
        ndi = total_income - installment - commitments

        data = {
            'full_name': request.form.get('full_name'),
            'nric': request.form.get('nric'),
            'age': int(request.form.get('age', 0)),
            'gender': request.form.get('gender'),
            'married': request.form.get('married'),
            'dependents': request.form.get('dependents'),
            'education': request.form.get('education'),
            'state': request.form.get('state'),
            'bumi_status': request.form.get('bumi_status'),
            'employment_sector': request.form.get('employment_sector'),
            'lppsa_eligible': request.form.get('lppsa_eligible'),
            'years_employed': int(request.form.get('years_employed', 0)),
            'income': income,
            'coapplicant_income': co_income,
            'monthly_commitments': commitments,
            'dsr': dsr,
            'ndi': ndi,
            'joint_applicant': request.form.get('joint_applicant'),
            'property_count': int(request.form.get('property_count', 1)),
            'ccris_status': request.form.get('ccris_status', 'clean'),
            'location_type': request.form.get('location_type', 'prime'),
            'loan_amount': loan_amt,
            'loan_term': loan_term,
            'financing_type': request.form.get('financing_type'),
            'property_value': float(request.form.get('property_value', 0)),
            'margin': float(request.form.get('margin', 90)),
            'property_type': request.form.get('property_type'),
            'property_area': request.form.get('location_type'), # mapping
            'purpose': request.form.get('purpose'),
            'interest_rate': float(request.form.get('interest_rate', 4.5)), # Capture interest rate
            'credit_score': float(request.form.get('credit_score_numeric', 650))
        }
        
        # Save/Update application record
        if application:
            for key, value in data.items():
                setattr(application, key, value)
        else:
            application = LoanApplication(user=current_user.id, **data)
            
        application.save()
        
        # ML Prediction with Malaysian context (DSR + NDI)
        try:
            model = joblib.load(MODEL_PATH)
            
            input_df = pd.DataFrame([{
                'ApplicantIncome': data['income'],
                'CoapplicantIncome': data['coapplicant_income'],
                'LoanAmount': data['loan_amount'],
                'Loan_Amount_Term': float(data['loan_term']),
                'Credit_Score': data['credit_score'],
                'Education': data['education'],
                'Married': data['married'],
                'Dependents': data['dependents'],
                'Property_Area': data['property_area'],
                'DSR': data['dsr'],
                'NDI': data['ndi'],
                'LPPSA_Eligible': data['lppsa_eligible'],
                'Interest_Rate': application.interest_rate
            }])

            
            processed_data = preprocess_input(input_df)
            pred = model.predict(processed_data)[0]
            result = 'Approved' if pred == 1 else 'Rejected'
            
        except Exception as e:
            print(f"Prediction error: {e}")
            result = 'Rejected' # Fallback
            
        application.prediction = result
        application.save()
        
        # Only create a new Prediction log entry if it's a new assessment
        # Actually in both cases we want to track it
        pred_record = Prediction(application=application, result=result)
        pred_record.save()
        
        return redirect(url_for('main.result', id=str(application.id)))
        
    return render_template('loan_form.html', application=application)

@main_bp.route('/delete_assessment/<id>', methods=['POST'])
@login_required
def delete_assessment(id):
    application = LoanApplication.objects(id=id).first()
    if not application:
        flash("Assessment not found.", "danger")
        return redirect(url_for('main.history'))
        
    # Security: Only owner or admin can delete
    if str(application.user.id) != str(current_user.id) and current_user.role != 'admin':
        flash("Unauthorized action.", "danger")
        return redirect(url_for('main.history'))
        
    try:
        application.delete()
        flash("Assessment record removed successfully.", "success")
    except Exception as e:
        flash(f"Error removing record: {str(e)}", "danger")
        
    return redirect(url_for('main.history'))


@main_bp.route('/history')
@login_required
def history():
    if current_user.role == 'admin':
        applications = LoanApplication.objects().order_by('-created_at')
    else:
        applications = LoanApplication.objects(user=current_user.id).order_by('-created_at')
    return render_template('history.html', applications=applications)

@main_bp.route('/admin/dataset')
@login_required
def dataset_management():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'loan_data.csv')
    try:
        df = pd.read_csv(csv_path)
        total_records = len(df)
        missing_values = df.isnull().sum().sum()
        preview_data = df.head(6).to_dict('records')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        total_records = 0
        missing_values = 0
        preview_data = []

    return render_template('dataset_management.html', 
                           total_records=total_records,
                           missing_values=missing_values,
                           preview_data=preview_data)


@main_bp.route('/admin/clean_data', methods=['POST'])
@login_required
def clean_data():
    """Manual trigger to fill missing values (Imputation)."""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied.'}), 403
        
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'loan_data.csv')
    try:
        df = pd.read_csv(csv_path)
        # Smatly fill missing values based on data type
        for col in df.columns:
            if df[col].isnull().any():
                # Check if it's a numeric column (float/int)
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    # For string/categorical: use the most frequent value (Mode)
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
                    else:
                        df[col] = df[col].fillna("Unknown")
        
        df.to_csv(csv_path, index=False)
        return jsonify({
            'success': True, 
            'message': 'Dataset cleaned! All missing values have been filled using smart Imputation.'
        })
    except Exception as e:
        print(f"Cleaning error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@main_bp.route('/admin/retrain', methods=['POST'])
@login_required
def retrain_model():
    """Retrain the ML model. Requires admin role and a security token."""
    if current_user.role != 'admin':
        return jsonify({'success': False, 'message': 'Access denied.'}), 403

    token = request.form.get('token', '').strip()
    if token != RETRAIN_TOKEN:
        return jsonify({'success': False, 'message': 'Invalid security token. Retraining denied.'}), 401

    model_type = request.form.get('model_type', 'random_forest').strip()
    valid_models = ['random_forest', 'logistic_regression', 'xgboost']
    if model_type not in valid_models:
        return jsonify({'success': False, 'message': f'Invalid model type: {model_type}'}), 400

    try:
        train_script = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'train_model.py')
        result = subprocess.run(
            [sys.executable, train_script, model_type],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            # 🚀 PROPERLY TRIGGER PLOT GENERATION after training
            plot_script = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'generate_plots.py')
            subprocess.run([sys.executable, plot_script], capture_output=True)
            
            model_labels = {
                'random_forest': 'Random Forest Classifier',
                'logistic_regression': 'Logistic Regression (Baseline)',
                'xgboost': 'XGBoost (Experimental)'
            }
            return jsonify({
                'success': True,
                'message': f"Model retrained and visuals updated successfully using {model_labels[model_type]}!",
                'log': result.stdout
            })
        else:
            # 🚨 NEW: Show details for debugging
            print(f"Subprocess failed (code {result.returncode}): {result.stderr}")
            return jsonify({
                'success': False,
                'message': f"Training script failed: {result.stderr.strip() or 'Unknown Error'}",
                'log': result.stderr
            }), 500
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'message': 'Training timed out (>5 min).'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@main_bp.route('/admin/download_csv')
@login_required
def download_csv():
    """Safety feature: Download the master dataset for verification."""
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'loan_data.csv')
    return send_file(csv_path, as_attachment=True, download_name='loan_data_master.csv')


@main_bp.route('/admin/data_explorer')
@login_required
def data_explorer():
    """Full, paginated, searchable dataset explorer for admins."""
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'loan_data.csv')

    search   = request.args.get('search', '').strip()
    status   = request.args.get('status', '').strip()   # 'Y', 'N', or ''
    page     = max(1, int(request.args.get('page', 1)))
    per_page = 20

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        flash(f'Could not read dataset: {e}', 'danger')
        return redirect(url_for('main.dataset_management'))

    # --- Summary stats (computed on full dataset before filtering) ---
    total_records  = len(df)
    approval_rate  = round((df['Loan_Status'] == 'Y').sum() / total_records * 100, 1) if total_records else 0
    avg_income     = round(df['ApplicantIncome'].mean(), 0) if 'ApplicantIncome' in df.columns else 0
    avg_loan       = round(df['LoanAmount'].mean(), 0)     if 'LoanAmount'       in df.columns else 0
    missing_total  = int(df.isnull().sum().sum())

    # --- Filtering ---
    if search:
        mask = df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
        df   = df[mask]
    if status in ('Y', 'N'):
        df = df[df['Loan_Status'] == status]

    filtered_total = len(df)
    total_pages    = max(1, math.ceil(filtered_total / per_page))
    page           = min(page, total_pages)
    start          = (page - 1) * per_page
    page_data      = df.iloc[start : start + per_page].fillna('—').to_dict('records')
    columns        = list(df.columns)

    return render_template('data_explorer.html',
        page_data      = page_data,
        columns        = columns,
        current_page   = page,
        total_pages    = total_pages,
        filtered_total = filtered_total,
        total_records  = total_records,
        approval_rate  = approval_rate,
        avg_income     = avg_income,
        avg_loan       = avg_loan,
        missing_total  = missing_total,
        search         = search,
        status_filter  = status,
        per_page       = per_page,
    )


@main_bp.route('/admin/download_dataset')
@login_required
def download_dataset():
    """Download the current training dataset as CSV."""
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'loan_data.csv')
    if not os.path.exists(csv_path):
        flash('Dataset file not found.', 'danger')
        return redirect(url_for('main.data_explorer'))
        
    return send_file(csv_path, as_attachment=True, download_name='housing_loan_dataset_2026.csv')


@main_bp.route('/admin/evaluation')
@login_required
def model_evaluation():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    metrics = get_model_evaluation_metrics()
    
    if metrics and metrics.get('success'):
        return render_template('model_evaluation.html', metrics=metrics)
    else:
        # Fallback for empty/error state
        flash('Evaluation metrics not found. Please train a model first.', 'warning')
        return render_template('model_evaluation.html', metrics=None)

@main_bp.route('/result/<id>')
@login_required
def result(id):
    application = LoanApplication.objects(id=id).first()
    if not application:
        flash("Application not found", "danger")
        return redirect(url_for('main.dashboard'))
        
    if str(application.user.id) != str(current_user.id) and current_user.role != 'admin':
        flash("Unauthorized access", "danger")
        return redirect(url_for('main.dashboard'))
        
    # Full Affordability Engine (Expert Multi-Constraint Model)
    suggested_banks = []
    
    # Dynamic AI Insights Generator
    ai_insights = []
    income = float(application.income + application.coapplicant_income)
    dsr = float(application.dsr)
    ndi = float(application.ndi)
    
    # Calculate Recommended Property Price (Based on more conservative rules)
    # For Income < RM 5000, use 50% DSR (Tight buffer). For > 5000, use 65% DSR.
    target_dsr = 0.50 if income < 5000 else 0.65
    min_ndi_buffer = 1200 if income < 5000 else 1500
    
    # PMT constrained by DSR
    max_pmt_dsr = (income * target_dsr) - float(application.monthly_commitments)
    # PMT constrained by NDI
    max_pmt_ndi = income - float(application.monthly_commitments) - min_ndi_buffer
    
    max_pmt = min(max_pmt_dsr, max_pmt_ndi)
    
    r = 0.043 / 12
    n = int(application.loan_term or 360)
    
    if max_pmt > 0 and r > 0:
        max_loan = max_pmt * ((1 - (1 + r)**-n) / r)
        recommended_price = max_loan / (float(application.margin or 90) / 100)
    else:
        recommended_price = 0

    if application.prediction == 'Approved':
        if application.credit_score >= 700:
            ai_insights.append({"type": "success", "text": "High credit score reflects excellent repayment reliability."})
        if dsr <= 60:
            ai_insights.append({"type": "success", "text": f"Healthy DSR of {dsr:.1f}% is well within the ideal 60% threshold."})
        if ndi >= 1500:
            ai_insights.append({"type": "info", "text": "Solid NDI buffer supports comfortable monthly living expenses."})
    else:
        if application.credit_score < 650:
            ai_insights.append({"type": "danger", "text": "Credit score is below the preferred minimum for low-risk financing."})
        if income < 4000 and dsr > 50:
            ai_insights.append({"type": "danger", "text": f"High installment-to-income ratio ({dsr:.1f}%) detected for this income bracket."})
            ai_insights.append({"type": "warning", "text": "Tight 'cost of living' buffer makes this loan highly sensitive to interest changes."})
        elif dsr > 68:
            ai_insights.append({"type": "danger", "text": f"DSR of {dsr:.1f}% exceeds the aggressive 70% limit for this profile."})
        
        if ndi < 1200:
            ai_insights.append({"type": "danger", "text": "Remaining disposable income (NDI) is below the subsistence buffer."})
        
        if not ai_insights: # Fallback for other ML reasons
             ai_insights.append({"type": "info", "text": "Risk detected: High property-value-to-income ratio (above 8.0x benchmark)."})


    if application.prediction == 'Approved' and application.ccris_status != 'arrears':
        # 1. Base Inputs
        income = float(application.income + application.coapplicant_income)
        net_income = income * 0.85 # Approximation of net if only gross is provided
        commitments = float(application.monthly_commitments)
        dsr = float(application.dsr)
        ndi = float(application.ndi)
        prop_count = int(application.property_count)
        location = application.location_type # 'prime', 'secondary', 'remote'
        fin_type = application.financing_type
        dependents = 0
        try:
            dependents = int(application.dependents if application.dependents and application.dependents.isdigit() else 0)
        except: pass
        
        # 2. Bank Database (Expert Parameter Table)
        banks_db = [
            {
                "name": "Maybank", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.75, "base_rate": 4.15,
                "min_ndi_single": 1200, "min_ndi_small_family": 1800, "min_ndi_large_family": 2400, "urban_buffer": 200,
                "variable_recognition": 0.60
            },
            {
                "name": "CIMB", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.80, "base_rate": 4.20,
                "min_ndi_single": 1200, "min_ndi_small_family": 1700, "min_ndi_large_family": 2300, "urban_buffer": 200,
                "variable_recognition": 0.70
            },
            {
                "name": "Public Bank", "structure": "conventional", "base_dsr": 0.65, "stretch_dsr": 0.75, "base_rate": 4.10,
                "min_ndi_single": 1300, "min_ndi_small_family": 1900, "min_ndi_large_family": 2600, "urban_buffer": 250,
                "variable_recognition": 0.50
            },
            {
                "name": "Hong Leong", "structure": "both", "base_dsr": 0.75, "stretch_dsr": 0.80, "base_rate": 4.20,
                "min_ndi_single": 1100, "min_ndi_small_family": 1700, "min_ndi_large_family": 2300, "urban_buffer": 200,
                "variable_recognition": 0.70
            },
            {
                "name": "Bank Islam", "structure": "islamic", "base_dsr": 0.75, "stretch_dsr": 0.85, "base_rate": 4.30,
                "min_ndi_single": 1000, "min_ndi_small_family": 1600, "min_ndi_large_family": 2200, "urban_buffer": 150,
                "variable_recognition": 0.60
            },
            {
                "name": "RHB Bank", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.80, "base_rate": 4.25,
                "min_ndi_single": 1200, "min_ndi_small_family": 1750, "min_ndi_large_family": 2350, "urban_buffer": 200,
                "variable_recognition": 0.65
            },
            {
                "name": "Alliance Bank", "structure": "both", "base_dsr": 0.75, "stretch_dsr": 0.80, "base_rate": 4.20,
                "min_ndi_single": 1200, "min_ndi_small_family": 1800, "min_ndi_large_family": 2400, "urban_buffer": 200,
                "variable_recognition": 0.70
            },
            {
                "name": "AmBank", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.75, "base_rate": 4.25,
                "min_ndi_single": 1100, "min_ndi_small_family": 1700, "min_ndi_large_family": 2300, "urban_buffer": 200,
                "variable_recognition": 0.60
            }
        ]

        if application.employment_sector == 'Government':
            banks_db.append({
                "name": "LPPSA (Gov)", "structure": "islamic", "base_dsr": 0.80, "stretch_dsr": 0.80, "base_rate": 4.0,
                "min_ndi_single": 800, "min_ndi_small_family": 1400, "min_ndi_large_family": 2000, "urban_buffer": 100,
                "variable_recognition": 0.80
            })

        potential_matches = []
        for b in banks_db:
            # 1. Type Filter
            if fin_type == 'Islamic' and b['structure'] == 'conventional': continue
            if fin_type == 'Conventional' and b['structure'] == 'islamic': continue
            
            # 2. Household-Adjusted NDI Requirement
            if dependents == 0:
                min_ndi = float(b['min_ndi_single'])
            elif dependents <= 2:
                min_ndi = float(b['min_ndi_small_family'])
            else:
                min_ndi = float(b['min_ndi_large_family'])
            
            if location == 'prime':
                min_ndi += float(b['urban_buffer'])

            # 3. Effective DSR limit logic
            effective_dsr_limit = float(b['base_dsr'])
            if income >= 8000: effective_dsr_limit += 0.03
            if application.ccris_status == 'clean': effective_dsr_limit += 0.03
            if application.employment_sector == 'Government': effective_dsr_limit += 0.03
            if prop_count == 1: effective_dsr_limit += 0.02 # First home bonus
            if prop_count >= 3: effective_dsr_limit -= 0.10 # Investor penalty
            effective_dsr_limit = min(effective_dsr_limit, float(b['stretch_dsr']))

            # 4. Affordability Pass/Fail (Dual Constraint)
            dsr_pass = (dsr <= (effective_dsr_limit * 100))
            ndi_pass = (ndi >= min_ndi)
            
            if not dsr_pass and dsr > (effective_dsr_limit * 100) + 5: continue # Hard fail
            if not ndi_pass and ndi < (min_ndi * 0.8): continue # Hard fail
            
            # 5. Component Scoring
            # DSR Fit (0.28)
            dsr_gap = (effective_dsr_limit * 100) - dsr
            dsr_fit_score = 1.0 if dsr_gap >= 10 else (0.8 if dsr_gap >= 0 else 0.4)
            
            # NDI Fit (0.24)
            ndi_ratio = ndi / min_ndi
            ndi_fit_score = 1.0 if ndi_ratio >= 1.5 else (0.85 if ndi_ratio >= 1.2 else (0.7 if ndi_ratio >= 1.0 else 0.3))
            
            # CCRIS Score (0.18)
            ccris_score = 1.0 if application.ccris_status == 'clean' else (0.6 if application.ccris_status == 'minor' else 0)
            
            # Property Score (0.08)
            prop_score = 1.0 if location == 'prime' else (0.7 if location == 'secondary' else 0.4)
            if float(prop_count) >= 3: prop_score *= 0.7 # MOF Cap impact
            
            # Product Fit (0.05) & Other (0.17 combined)
            prod_fit = 0.9 if b['structure'] == 'both' else 1.0
            
            final_score = (0.28 * dsr_fit_score) + (0.24 * ndi_fit_score) + (0.18 * ccris_score) + (0.08 * prop_score) + (0.22 * prod_fit)
            
            # 6. Recommendation Reason
            reason = "High match due to "
            if ndi_ratio > 1.3: reason += "excellent survival buffer (NDI)"
            elif dsr_gap > 10: reason += "significant DSR headroom"
            else: reason += "stable income-to-loan ratio"
            
            if prop_count >= 3: reason += " (Investor-adjusted)"
            if not ndi_pass: reason += " [Warning: Low NDI]"

            # 7. Bank-Specific Installment Calculation
            P = float(application.loan_amount)
            n = int(application.loan_term)
            r = float(b['base_rate']) / 100 / 12
            
            b_installment = 0.0
            if r > 0 and n > 0:
                b_installment = P * (r * (1 + r)**n) / ((1 + r)**n - 1)
            elif n > 0:
                b_installment = P / n

            potential_matches.append({
                "name": b['name'],
                "score": round(float(final_score * 100), 1),
                "reason": reason,
                "installment": round(float(b_installment), 2),
                "rate": b['base_rate']
            })
            
        suggested_banks = sorted(potential_matches, key=lambda x: x.get('score', 0), reverse=True)
        if len(suggested_banks) > 5:
            suggested_banks = suggested_banks[:5]
    
    return render_template('result.html', 
                           application=application, 
                           suggested_banks=suggested_banks,
                           ai_insights=ai_insights,
                           recommended_price=recommended_price)


@main_bp.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    applications = LoanApplication.objects().order_by('-created_at')
    total_users = User.objects.count()
    total_staff = User.objects(role='user').count() # Assuming 'user' role refers to staff
    
    total_records = len(applications)
    approved_apps = sum(1 for app in applications if app.prediction == 'Approved')
    approval_rate = (approved_apps / total_records * 100) if total_records > 0 else 0
    
    metrics = get_model_evaluation_metrics()
    
    return render_template('admin_dashboard.html', 
                           applications=applications, 
                           total_users=total_users, 
                           total_staff=total_staff,
                           approval_rate=round(approval_rate, 1),
                           metrics=metrics)


@main_bp.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    users = User.objects().order_by('name')
    return render_template('admin_users.html', users=users)

@main_bp.route('/admin/users/add', methods=['POST'])
@login_required
def add_user():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    role = request.form.get('role', 'user')
    
    if User.objects(email=email).first():
        flash('Email address already registered.', 'danger')
        return redirect(url_for('main.admin_users'))
        
    try:
        new_user = User(
            name=name,
            email=email,
            password=generate_password_hash(password),
            role=role
        )
        new_user.save()
        flash(f'User {name} successfully created.', 'success')
    except Exception as e:
        flash(f'An error occurred while creating the user.', 'danger')
        
    return redirect(url_for('main.admin_users'))

@main_bp.route('/admin/users/edit/<user_id>', methods=['POST'])
@login_required
def update_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    try:
        user_to_edit = User.objects(id=user_id).first()
        if user_to_edit:
            user_to_edit.name = request.form.get('name')
            user_to_edit.email = request.form.get('email')
            
            # Prevent self-demotion
            if str(current_user.id) != user_id:
                user_to_edit.role = request.form.get('role')
            
            user_to_edit.save()
            flash(f'User {user_to_edit.name} updated successfully.', 'success')
        else:
            flash('User not found.', 'danger')
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'danger')
        
    return redirect(url_for('main.admin_users'))

@main_bp.route('/admin/users/delete/<user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    if str(current_user.id) == user_id:
        flash('You cannot delete your own admin account.', 'warning')
        return redirect(url_for('main.admin_users'))
        
    try:
        user_to_delete = User.objects(id=user_id).first()
        if user_to_delete:
            user_to_delete.delete()
            flash('User deleted successfully.', 'success')
        else:
            flash('User not found.', 'danger')
    except Exception as e:
        flash(f'An error occurred while deleting the user.', 'danger')
        
    return redirect(url_for('main.admin_users'))
@main_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        # Handle settings update (optional for now, just show success)
        flash('Settings updated successfully.', 'success')
        return redirect(url_for('main.settings'))
    return render_template('settings.html')


@main_bp.route('/bank_requirements')
@login_required
def bank_requirements():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
    banks_db = [
        {
            "name": "Maybank", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.75, "base_rate": 4.15,
            "min_ndi_single": 1200, "min_ndi_small_family": 1800, "min_ndi_large_family": 2400, "urban_buffer": 200,
            "variable_recognition": 0.60, "logo": "bi-bank"
        },
        {
            "name": "CIMB", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.80, "base_rate": 4.20,
            "min_ndi_single": 1200, "min_ndi_small_family": 1700, "min_ndi_large_family": 2300, "urban_buffer": 200,
            "variable_recognition": 0.70, "logo": "bi-bank2"
        },
        {
            "name": "Public Bank", "structure": "conventional", "base_dsr": 0.65, "stretch_dsr": 0.75, "base_rate": 4.10,
            "min_ndi_single": 1300, "min_ndi_small_family": 1900, "min_ndi_large_family": 2600, "urban_buffer": 250,
            "variable_recognition": 0.50, "logo": "bi-building-columns"
        },
        {
            "name": "Hong Leong", "structure": "both", "base_dsr": 0.75, "stretch_dsr": 0.80, "base_rate": 4.20,
            "min_ndi_single": 1100, "min_ndi_small_family": 1700, "min_ndi_large_family": 2300, "urban_buffer": 200,
            "variable_recognition": 0.70, "logo": "bi-bank"
        },
        {
            "name": "Bank Islam", "structure": "islamic", "base_dsr": 0.75, "stretch_dsr": 0.85, "base_rate": 4.30,
            "min_ndi_single": 1000, "min_ndi_small_family": 1600, "min_ndi_large_family": 2200, "urban_buffer": 150,
            "variable_recognition": 0.60, "logo": "bi-shield-check"
        },
        {
            "name": "RHB Bank", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.80, "base_rate": 4.25,
            "min_ndi_single": 1200, "min_ndi_small_family": 1750, "min_ndi_large_family": 2350, "urban_buffer": 200,
            "variable_recognition": 0.65, "logo": "bi-bank2"
        },
        {
            "name": "Alliance Bank", "structure": "both", "base_dsr": 0.75, "stretch_dsr": 0.80, "base_rate": 4.20,
            "min_ndi_single": 1200, "min_ndi_small_family": 1800, "min_ndi_large_family": 2400, "urban_buffer": 200,
            "variable_recognition": 0.70, "logo": "bi-bank"
        },
        {
            "name": "AmBank", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.75, "base_rate": 4.25,
            "min_ndi_single": 1100, "min_ndi_small_family": 1700, "min_ndi_large_family": 2300, "urban_buffer": 200,
            "variable_recognition": 0.60, "logo": "bi-bank2"
        },
        {
            "name": "LPPSA (Gov)", "structure": "islamic", "base_dsr": 0.80, "stretch_dsr": 0.80, "base_rate": 4.0,
            "min_ndi_single": 800, "min_ndi_small_family": 1400, "min_ndi_large_family": 2000, "urban_buffer": 100,
            "variable_recognition": 0.80, "logo": "bi-mortarboard-fill"
        }
    ]
    return render_template('bank_requirements.html', banks=banks_db)


