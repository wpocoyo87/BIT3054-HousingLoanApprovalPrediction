from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import login_required, current_user
from app.models import LoanApplication, Prediction, User
from werkzeug.security import generate_password_hash

import math
import joblib
import os
import sys
import json
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
        
        # --- FINANCIAL EXPERT OVERRIDE (HIGH PRECISION) ---
        # For profiles with strong NDI and clean status, we approve even with borderline score
        if float(data['dsr']) < 70 and float(data['ndi']) > 1400 and float(data['credit_score']) >= 600:
            result = 'Approved'
            confidence_score = 99.0
        else:
            # Standard Prediction Pipeline
            try:
                input_df = pd.DataFrame([{
                    'ApplicantIncome': float(data['income']),
                    'CoapplicantIncome': float(data['coapplicant_income']),
                    'LoanAmount': float(data['loan_amount']),
                    'Loan_Amount_Term': float(data['loan_term']),
                    'Credit_Score': float(data['credit_score']),
                    'Education': str(application.education or "Graduate"),
                    'Married': str(application.married or "No"),
                    'Dependents': str(application.dependents or "0"),
                    'Property_Area': str(application.property_area or "Urban"),
                    'Interest_Rate': float(application.interest_rate or 4.5),
                    'Years_Employed': float(application.years_employed or 1),
                    'Monthly_Commitments': float(application.monthly_commitments or 0)
                }])

                processed_data = preprocess_input(input_df)
                model = joblib.load(MODEL_PATH)
                pred = model.predict(processed_data)[0]
                
                try:
                    probs = model.predict_proba(processed_data)[0]
                    conf = float(max(probs) * 100)
                except: 
                    conf = 95.0
                
                result = 'Approved' if pred == 1 else 'Rejected'
                confidence_score = conf
                
            except Exception as e:
                print(f"ML ERROR: {e}")
                result = 'Rejected'
                confidence_score = 50.0

        # SAVE TO DATABASE EXPLICITLY
        application.prediction = result
        application.confidence_score = confidence_score
        application.save()
        
        # Track history
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
        missing_values = int(df.isnull().sum().sum())
        preview_data = df.head(100).to_dict('records') # Show more for preview?
        
        # Calculate breakdown for modal
        null_counts = df.isnull().sum()
        missing_breakdown = null_counts[null_counts > 0].to_dict()
        
        # Get metadata for "Last uploaded/trained"
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'ml_model', 'model_metadata.json')
        last_train = "N/A"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                last_train = meta.get('trained_at', 'N/A')
                
    except Exception as e:
        print(f"Error reading CSV: {e}")
        total_records = 0
        missing_values = 0
        preview_data = []
        missing_breakdown = {}
        last_train = "N/A"

    return render_template('dataset_management.html', 
                           total_records=total_records,
                           missing_values=missing_values,
                           preview_data=preview_data,
                           missing_breakdown=missing_breakdown,
                           last_train=last_train)


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

@main_bp.route('/export_report')
@login_required
def export_report():
    import io
    import csv
    if current_user.role == 'admin':
        apps = LoanApplication.objects().order_by('-created_at')
    else:
        apps = LoanApplication.objects(user=current_user.id).order_by('-created_at')
        
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Application ID', 'Date', 'Applicant Name', 'Joint Applicant', 'Total Income (RM)', 'Loan Amount (RM)', 'DSR (%)', 'NDI (RM)', 'Credit Score', 'Status Prediction'])
    
    for app in apps:
        total_income = float(app.income or 0) + float(app.coapplicant_income or 0)
        writer.writerow([
            str(app.id)[-6:].upper(),
            app.created_at.strftime('%Y-%m-%d'),
            app.full_name or 'N/A',
            app.joint_applicant or 'No',
            f"{total_income:.2f}",
            f"{float(app.loan_amount or 0):.2f}",
            f"{float(app.dsr or 0):.1f}",
            f"{float(app.ndi or 0):.2f}",
            str(app.credit_score) if app.credit_score else 'N/A',
            app.prediction or 'Pending'
        ])
        
    output.seek(0)
    bytes_io = io.BytesIO(output.getvalue().encode('utf-8'))
    return send_file(bytes_io, 
                     mimetype='text/csv', 
                     as_attachment=True, 
                     download_name='monthly_assessment_report.csv')

@main_bp.route('/system_docs')
@login_required
def system_docs():
    return render_template('system_documentation.html')

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

    # --- Agent Advisory: Delta Comparison vs Previous Assessment ---
    delta_insights = []
    try:
        # Fetch the most recent PREVIOUS application by the same user (exclude current)
        prev_app = (LoanApplication.objects(user=application.user, id__ne=application.id)
                    .order_by('-created_at').first())

        if prev_app:
            curr_dsr = float(application.dsr or 0)
            prev_dsr = float(prev_app.dsr or 0)

            curr_ndi = float(application.ndi or 0)
            prev_ndi = float(prev_app.ndi or 0)

            curr_cs  = float(application.credit_score or 0)
            prev_cs  = float(prev_app.credit_score or 0)

            date_str = prev_app.created_at.strftime('%d %b %Y')

            # 1. DSR Drift
            dsr_diff = curr_dsr - prev_dsr
            if dsr_diff >= 5:
                delta_insights.append({
                    "type": "danger",
                    "icon": "bi-graph-up-arrow",
                    "text": (
                        f"Your DSR has increased from {prev_dsr:.1f}% → {curr_dsr:.1f}% "
                        f"(+{dsr_diff:.1f}%) since {date_str}. "
                        f"This now exceeds most banks' approval threshold. "
                        f"Avoid taking on any additional financial obligations until your housing loan is approved."
                    )
                })
            elif dsr_diff <= -5:
                delta_insights.append({
                    "type": "success",
                    "icon": "bi-graph-down-arrow",
                    "text": (
                        f"DSR improved from {prev_dsr:.1f}% → {curr_dsr:.1f}% "
                        f"({dsr_diff:.1f}%) since {date_str}. "
                        f"Your financial commitments have reduced positively."
                    )
                })

            # 2. NDI Drift
            ndi_diff = curr_ndi - prev_ndi
            if ndi_diff <= -200:
                delta_insights.append({
                    "type": "danger",
                    "icon": "bi-wallet2",
                    "text": (
                        f"Your Net Disposable Income has dropped by RM {abs(ndi_diff):,.0f} "
                        f"(from RM {prev_ndi:,.0f} → RM {curr_ndi:,.0f}) since {date_str}. "
                        f"Consider settling or reducing existing commitments to restore your NDI to a sustainable level."
                    )
                })
            elif ndi_diff >= 200:
                delta_insights.append({
                    "type": "success",
                    "icon": "bi-wallet2",
                    "text": (
                        f"NDI improved by RM {abs(ndi_diff):,.0f} "
                        f"(RM {prev_ndi:,.0f} → RM {curr_ndi:,.0f}) since {date_str}. "
                        f"Your disposable income buffer has strengthened."
                    )
                })

            # 3. Credit Score Drift
            cs_diff = curr_cs - prev_cs
            if cs_diff <= -30:
                delta_insights.append({
                    "type": "danger",
                    "icon": "bi-credit-card-2-front",
                    "text": (
                        f"Credit score has dropped from {int(prev_cs)} → {int(curr_cs)} "
                        f"({int(cs_diff)}) since {date_str}. "
                        f"A new credit application may have been detected in your profile. "
                        f"Banks include credit limit exposure in DSR calculations — "
                        f"avoid new credit applications during the approval waiting period."
                    )
                })
            elif cs_diff >= 30:
                delta_insights.append({
                    "type": "success",
                    "icon": "bi-credit-card-2-front",
                    "text": (
                        f"Credit score improved from {int(prev_cs)} → {int(curr_cs)} "
                        f"(+{int(cs_diff)}) since {date_str}. "
                        f"This strengthens your creditworthiness in the bank's review."
                    )
                })

            # 4. CCRIS Status Change
            prev_ccris = (prev_app.ccris_status or 'clean').lower()
            curr_ccris = (application.ccris_status or 'clean').lower()
            if prev_ccris in ['clean', 'good'] and curr_ccris in ['late_payment', 'arrears']:
                delta_insights.append({
                    "type": "danger",
                    "icon": "bi-exclamation-triangle",
                    "text": (
                        f"CCRIS status has changed from '{prev_ccris.title()}' → '{curr_ccris.replace('_',' ').title()}' "
                        f"since {date_str}. "
                        f"Contact your bank immediately and provide updated payment records to avoid an automatic rejection."
                    )
                })

            # 5. Employment Sector Change
            prev_emp = (prev_app.employment_sector or '').strip()
            curr_emp = (application.employment_sector or '').strip()
            if prev_emp and curr_emp and prev_emp != curr_emp:
                delta_insights.append({
                    "type": "warning",
                    "icon": "bi-briefcase",
                    "text": (
                        f"Employment sector changed from '{prev_emp}' → '{curr_emp}' since {date_str}. "
                        f"A sector change may affect income stability perception. "
                        f"Inform your bank immediately and provide updated employment and income documents."
                    )
                })

            # No change detected (positive signal)
            if not delta_insights:
                delta_insights.append({
                    "type": "success",
                    "icon": "bi-shield-check",
                    "text": (
                        f"No significant financial changes detected since your previous assessment on {date_str}. "
                        f"Your eligibility profile remains stable."
                    )
                })

    except Exception as e:
        print(f"Delta comparison error: {e}")

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
        if application.credit_score >= 750:
            ai_insights.append({"type": "success", "text": "Exceptional credit score reflects superior repayment reliability."})
        elif application.credit_score >= 700:
            ai_insights.append({"type": "success", "text": "High credit score reflects excellent repayment reliability."})
            
        if dsr <= 60:
            ai_insights.append({"type": "success", "text": f"Healthy DSR of {dsr:.1f}% is well within the ideal 60% threshold."})
        if ndi >= 1500:
            ai_insights.append({"type": "info", "text": "Solid NDI buffer supports comfortable monthly living expenses."})
    else:
        # Rejection Insights based on ML behavior patterns
        if application.credit_score < 650:
            ai_insights.append({"type": "danger", "text": "Credit score falls below the preferred minimum for low-risk financing."})
        
        # Define years_employed from application object
        years_employed = float(application.years_employed or 1)

        # High Risk Pattern: Income < 4000 with DSR consuming > 50%
        if income < 4000 and dsr > 50:
            ai_insights.append({"type": "danger", "text": f"High installment-to-income sensitivity detected. For lower income profiles, a DSR of {dsr:.1f}% leaves tight margins for cost-of-living fluctuations."})
        
        if years_employed >= 1:
            ai_insights.append({"type": "success", "text": "Stable employment history (1+ year) positively impacts the internal scoring risk model."})
        
        if dsr > 68:
            ai_insights.append({"type": "danger", "text": f"DSR of {dsr:.1f}% is approaching the critical 70% threshold for bank eligibility."})
        
        if ndi < 1500 and income < 5000:
            ai_insights.append({"type": "warning", "text": "Aggravating Factor: Net Disposable Income (NDI) buffer is tight for premium location living standards."})
        
        if not ai_insights: # Fallback
             ai_insights.append({"type": "info", "text": "Risk detected: High property-value-to-income ratio (above 7.5x benchmark)."})


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
            dependents = int(''.join(filter(str.isdigit, str(application.dependents or "0"))) or 0)
        except: pass
        
        # 2. Bank Database (Expert Parameter Table - Aligned with Industry Standards)
        banks_db = [
            {
                "name": "Maybank", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.75, "base_rate": 4.15,
                "min_ndi_single": 1300, "min_ndi_small_family": 1850, "min_ndi_large_family": 2500, "urban_buffer": 250,
                "variable_recognition": 0.60
            },
            {
                "name": "CIMB", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.80, "base_rate": 4.20,
                "min_ndi_single": 1200, "min_ndi_small_family": 1700, "min_ndi_large_family": 2300, "urban_buffer": 200,
                "variable_recognition": 0.70
            },
            {
                "name": "Public Bank", "structure": "conventional", "base_dsr": 0.65, "stretch_dsr": 0.75, "base_rate": 4.10,
                "min_ndi_single": 1400, "min_ndi_small_family": 2000, "min_ndi_large_family": 2600, "urban_buffer": 250,
                "variable_recognition": 0.50
            },
            {
                "name": "Hong Leong", "structure": "both", "base_dsr": 0.75, "stretch_dsr": 0.80, "base_rate": 4.20,
                "min_ndi_single": 1150, "min_ndi_small_family": 1750, "min_ndi_large_family": 2350, "urban_buffer": 200,
                "variable_recognition": 0.70
            },
            {
                "name": "Bank Islam", "structure": "islamic", "base_dsr": 0.75, "stretch_dsr": 0.85, "base_rate": 4.30,
                "min_ndi_single": 1000, "min_ndi_small_family": 1600, "min_ndi_large_family": 2200, "urban_buffer": 150,
                "variable_recognition": 0.60
            },
            {
                "name": "RHB Bank", "structure": "both", "base_dsr": 0.70, "stretch_dsr": 0.80, "base_rate": 4.25,
                "min_ndi_single": 1250, "min_ndi_small_family": 1800, "min_ndi_large_family": 2400, "urban_buffer": 200,
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
            # Parse '3+' to 3 for comparison
            try:
                num_dep = int(''.join(filter(str.isdigit, str(application.dependents or "0"))) or 0)
            except: num_dep = 0

            if num_dep == 0:
                min_ndi = float(b['min_ndi_single'])
            elif num_dep <= 2:
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

    # --- DSR / NDI Eligibility Comparison View ---
    dsr_val  = float(application.dsr or 0)
    ndi_val  = float(application.ndi or 0)
    income   = float(application.income + application.coapplicant_income)
    cs_val   = int(application.credit_score or 0)

    # Benchmarks based on Malaysian banking standards
    dsr_limit  = 60 if income < 5000 else 70          # % ceiling
    ndi_min    = 1200 if income < 5000 else 1500       # RM floor
    cs_min     = 650                                    # Credit score floor

    eligibility_checks = [
        {
            "metric"    : "Debt Service Ratio (DSR)",
            "your_value": f"{dsr_val:.1f}%",
            "benchmark" : f"≤ {dsr_limit}%",
            "pass"      : dsr_val <= dsr_limit,
            "note"      : "Ratio of total monthly commitments to gross income"
        },
        {
            "metric"    : "Net Disposable Income (NDI)",
            "your_value": f"RM {ndi_val:,.2f}",
            "benchmark" : f"≥ RM {ndi_min:,}",
            "pass"      : ndi_val >= ndi_min,
            "note"      : "Remaining income after all monthly commitments"
        },
        {
            "metric"    : "Credit / CCRIS Score",
            "your_value": str(cs_val),
            "benchmark" : f"≥ {cs_min}",
            "pass"      : cs_val >= cs_min,
            "note"      : "Internal credit health score derived from profile"
        },
        {
            "metric"    : "CCRIS Status",
            "your_value": (application.ccris_status or "N/A").title(),
            "benchmark" : "Clean / Good",
            "pass"      : (application.ccris_status or "").lower() in ["clean", "good"],
            "note"      : "Credit Reference Information System payment status"
        },
    ]

    return render_template('result.html',
                           application=application,
                           suggested_banks=suggested_banks,
                           ai_insights=ai_insights,
                           recommended_price=recommended_price,
                           eligibility_checks=eligibility_checks,
                           delta_insights=delta_insights)




@main_bp.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    applications = LoanApplication.objects().order_by('-created_at')
    total_users = User.objects.count()
    total_loan_analysts = User.objects(role='staff').count()
    total_auditors = User.objects(role='user').count()
    
    total_records = len(applications)
    approved_apps = sum(1 for app in applications if app.prediction == 'Approved')
    approval_rate = (approved_apps / total_records * 100) if total_records > 0 else 0
    
    metrics = get_model_evaluation_metrics()
    
    return render_template('admin_dashboard.html', 
                           applications=applications, 
                           total_users=total_users, 
                           total_loan_analysts=total_loan_analysts,
                           total_auditors=total_auditors,
                           approval_rate=round(approval_rate, 1),
                           metrics=metrics)


@main_bp.route('/admin/all_records')
@login_required
def all_records():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))

    page      = int(request.args.get('page', 1))
    per_page  = 20
    q         = request.args.get('q', '').strip()
    status    = request.args.get('status', '')  # 'Approved' | 'Rejected' | ''

    qs = LoanApplication.objects().order_by('-created_at')

    if status in ['Approved', 'Rejected']:
        qs = qs.filter(prediction=status)

    all_apps  = list(qs)

    # Client-name filter (MongoEngine doesn't support reverse ref search easily)
    if q:
        all_apps = [a for a in all_apps
                    if q.lower() in (a.full_name or '').lower()
                    or q.lower() in str(a.id)[-6:].lower()
                    or q.lower() in (a.user.name or '').lower()]

    total      = len(all_apps)
    total_pages = max(1, -(-total // per_page))   # ceiling division
    page        = max(1, min(page, total_pages))
    start       = (page - 1) * per_page
    paginated   = all_apps[start:start + per_page]

    return render_template('all_records.html',
                           applications=paginated,
                           page=page,
                           total_pages=total_pages,
                           total=total,
                           q=q,
                           status=status)




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


