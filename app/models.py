from flask_login import UserMixin
from datetime import datetime
import mongoengine as me

class User(UserMixin, me.Document):
    meta = {'collection': 'users'}
    name = me.StringField(required=True, max_length=100)
    email = me.StringField(required=True, unique=True, max_length=120)
    password = me.StringField(required=True, max_length=200)
    role = me.StringField(required=True, default='user', max_length=20)
    
    def get_id(self):
        return str(self.id)

class LoanApplication(me.Document):
    meta = {'collection': 'loan_applications', 'strict': False}
    user = me.ReferenceField(User, required=True, reverse_delete_rule=me.CASCADE)
    
    # Personal info
    full_name = me.StringField(max_length=150)
    nric = me.StringField(max_length=20)
    age = me.IntField()
    gender = me.StringField(max_length=20)
    married = me.StringField(max_length=20)
    dependents = me.StringField(max_length=20)
    education = me.StringField(max_length=50)
    state = me.StringField(max_length=50)
    bumi_status = me.StringField(max_length=10)
    property_count = me.IntField(default=1) # 1, 2, 3+ 
    ccris_status = me.StringField(default='clean', max_length=50) # 'clean', 'late_payment', 'arrears'
    location_type = me.StringField(max_length=50, default='City Center / Major Hub')
    
    # Financial info
    employment_sector = me.StringField(max_length=50)
    lppsa_eligible = me.StringField(max_length=10)
    years_employed = me.IntField()

    income = me.FloatField()
    coapplicant_income = me.FloatField(default=0.0)
    monthly_commitments = me.FloatField(default=0.0)
    dsr = me.FloatField(default=0.0)
    ndi = me.FloatField(default=0.0)
    joint_applicant = me.StringField(max_length=10)

    
    # Financing info
    loan_amount = me.FloatField()
    loan_term = me.IntField()
    financing_type = me.StringField(max_length=20)
    property_value = me.FloatField()
    margin = me.FloatField()
    property_type = me.StringField(max_length=50)
    property_area = me.StringField(max_length=50)
    interest_rate = me.FloatField(default=4.5)
    purpose = me.StringField(max_length=50)
    
    # Prediction
    credit_score = me.FloatField()
    prediction = me.StringField(max_length=50, null=True)
    created_at = me.DateTimeField(default=datetime.utcnow)


class Prediction(me.Document):
    meta = {'collection': 'predictions'}
    application = me.ReferenceField(LoanApplication, required=True, reverse_delete_rule=me.CASCADE)
    result = me.StringField(required=True, max_length=50)
    created_at = me.DateTimeField(default=datetime.utcnow)
