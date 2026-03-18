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
    meta = {'collection': 'loan_applications'}
    user = me.ReferenceField(User, required=True, reverse_delete_rule=me.CASCADE)
    income = me.FloatField(required=True)
    coapplicant_income = me.FloatField(required=True)
    loan_amount = me.FloatField(required=True)
    loan_term = me.FloatField(required=True)
    credit_score = me.FloatField(required=True)
    education = me.StringField(required=True, max_length=50)
    married = me.StringField(required=True, max_length=10)
    dependents = me.StringField(required=True, max_length=10)
    property_area = me.StringField(required=True, max_length=50)
    prediction = me.StringField(max_length=50, null=True)
    created_at = me.DateTimeField(default=datetime.utcnow)

class Prediction(me.Document):
    meta = {'collection': 'predictions'}
    application = me.ReferenceField(LoanApplication, required=True, reverse_delete_rule=me.CASCADE)
    result = me.StringField(required=True, max_length=50)
    created_at = me.DateTimeField(default=datetime.utcnow)
