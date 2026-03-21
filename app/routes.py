from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from app.models import LoanApplication, Prediction, User
from werkzeug.security import generate_password_hash

import joblib
import os
import pandas as pd
from ml_model.preprocessing import preprocess_input

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
@login_required
def loan_form():
    if request.method == 'POST':
        # Capture Malaysian-specific fields
        income = float(request.form.get('income', 0))
        co_income = float(request.form.get('coapplicant_income', 0))
        commitments = float(request.form.get('monthly_commitments', 0))
        loan_amt = float(request.form.get('loan_amount', 0))
        loan_term = int(request.form.get('loan_term', 360))
        
        # Calculate Monthly Installment (Approx 4.3% interest)
        P = loan_amt * 1000
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
            'loan_amount': loan_amt,
            'loan_term': loan_term,
            'financing_type': request.form.get('financing_type'),
            'property_value': float(request.form.get('property_value', 0)),
            'margin': float(request.form.get('margin', 90)),
            'property_type': request.form.get('property_type'),
            'property_area': request.form.get('property_area'),
            'purpose': request.form.get('purpose'),
            'credit_score': float(request.form.get('credit_score_numeric', 650))
        }
        
        # Save application record
        application = LoanApplication(user=current_user.id, **data)
        application.save()
        
        # ML Prediction with Malaysian context (DSR + NDI)
        try:
            model = joblib.load(MODEL_PATH)
            
            input_df = pd.DataFrame([{
                'ApplicantIncome': data['income'],
                'CoapplicantIncome': data['coapplicant_income'],
                'LoanAmount': data['loan_amount'] / 1000.0,
                'Loan_Amount_Term': float(data['loan_term']),
                'Credit_Score': data['credit_score'],
                'Education': data['education'],
                'Married': data['married'],
                'Dependents': data['dependents'],
                'Property_Area': data['property_area'],
                'DSR': data['dsr'],
                'NDI': data['ndi'],
                'LPPSA_Eligible': data['lppsa_eligible']
            }])

            
            processed_data = preprocess_input(input_df)
            pred = model.predict(processed_data)[0]
            result = 'Approved' if pred == 1 else 'Rejected'
            
        except Exception as e:
            print(f"Prediction error: {e}")
            result = 'Rejected' # Fallback
            
        application.prediction = result
        application.save()
        
        pred_record = Prediction(application=application, result=result)
        pred_record.save()
        
        return redirect(url_for('main.result', id=str(application.id)))
        
    return render_template('loan_form.html')


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
    return render_template('dataset_management.html')

@main_bp.route('/admin/evaluation')
@login_required
def model_evaluation():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
    return render_template('model_evaluation.html')

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
        
    return render_template('result.html', application=application)

@main_bp.route('/admin_dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    applications = LoanApplication.objects().order_by('-created_at')
    total_users = User.objects.count()
    total_staff = User.objects(role='user').count() # Assuming 'user' role refers to staff
    
    total_apps = len(applications)
    approved_apps = sum(1 for app in applications if app.prediction == 'Approved')
    approval_rate = (approved_apps / total_apps * 100) if total_apps > 0 else 0
    
    return render_template('admin_dashboard.html', 
                           applications=applications, 
                           total_users=total_users, 
                           total_staff=total_staff,
                           approval_rate=round(approval_rate, 1))


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


