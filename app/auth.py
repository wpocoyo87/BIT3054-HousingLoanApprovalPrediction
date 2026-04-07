from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app.models import User

from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/', methods=['GET', 'POST'])
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('main.admin_dashboard'))
        return redirect(url_for('main.dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.objects(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('main.admin_dashboard'))
            else:
                return redirect(url_for('main.dashboard'))
        else:
            flash('Invalid email or password', 'danger')
            
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

@auth_bp.route('/register_dummy', methods=['GET'])
def register_dummy():
    if not User.objects(email='user@test.com').first():
        user = User(name='Test User', email='user@test.com', password=generate_password_hash('password'), role='user')
        user.save()
    if not User.objects(email='admin@test.com').first():
        admin = User(name='Test Admin', email='admin@test.com', password=generate_password_hash('admin123'), role='admin')
        admin.save()
    if not User.objects(email='staff@test.com').first():
        staff = User(name='Test Staff', email='staff@test.com', password=generate_password_hash('staff123'), role='staff')
        staff.save()
    return "Dummy users created! staff@test.com:staff123 | admin@test.com:admin123", 200

