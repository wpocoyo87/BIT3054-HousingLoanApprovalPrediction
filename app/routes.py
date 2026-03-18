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
        # Retrieve form data
        income = float(request.form.get('income'))
        coapplicant_income = float(request.form.get('coapplicant_income'))
        loan_amount = float(request.form.get('loan_amount'))
        loan_term = float(request.form.get('loan_term'))
        credit_score = float(request.form.get('credit_history')) # We kept the HTML input name 'credit_history' previously, let's just parse it as credit_score
        education = request.form.get('education')
        married = request.form.get('married')
        dependents = request.form.get('dependents')
        property_area = request.form.get('property_area')
        
        # Save application
        application = LoanApplication(
            user=current_user.id,
            income=income,
            coapplicant_income=coapplicant_income,
            loan_amount=loan_amount,
            loan_term=loan_term,
            credit_score=credit_score,
            education=education,
            married=married,
            dependents=dependents,
            property_area=property_area
        )
        application.save()
        
        # ML Prediction
        try:
            model = joblib.load(MODEL_PATH)
            
            input_data = pd.DataFrame([{
                'ApplicantIncome': income,
                'CoapplicantIncome': coapplicant_income,
                'LoanAmount': loan_amount / 1000.0, # The model was trained on loan amounts in thousands (e.g. 185 = 185k)
                'Loan_Amount_Term': loan_term,
                'Credit_Score': credit_score,
                'Education': education,
                'Married': married,
                'Dependents': dependents,
                'Property_Area': property_area
            }])
            
            processed_data = preprocess_input(input_data)
            print("--- DEBUG: PROCESSED DATA ---")
            print(processed_data)
            print("--- DEBUG: PROCESSED DATA TYPES ---")
            print(processed_data.dtypes)
            pred = model.predict(processed_data)[0]
            print(f"--- DEBUG: PREDICTION RESULT: {pred} ---")
            
            result = 'Approved' if pred == 1 else 'Rejected'
            
        except Exception as e:
            result = 'Error'
            
        application.prediction = result
        application.save()
        
        pred_record = Prediction(application=application, result=result)
        pred_record.save()
        
        return redirect(url_for('main.result', id=str(application.id)))
        
    return render_template('loan_form.html')

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
    return render_template('admin_dashboard.html', applications=applications)

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
    
    # Check if user already exists
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
        print(f"Error adding user: {e}")
        
    return redirect(url_for('main.admin_users'))

@main_bp.route('/admin/users/delete/<user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('main.dashboard'))
        
    # Prevent admin from deleting themselves
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
        print(f"Error deleting user: {e}")
        
    return redirect(url_for('main.admin_users'))

