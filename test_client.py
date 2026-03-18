from app import create_app
from app.models import User, LoanApplication
import mongoengine as me

app = create_app()

with app.test_client() as client:
    with app.app_context():
        # Clear existing to avoid unique constraint issues
        User.objects.delete()
        user = User(name='Test', email='test@test.com', password='pwd', role='user')
        user.save()
        
        # Login
        with client.session_transaction() as sess:
            sess['_user_id'] = str(user.id)
            sess['_fresh'] = True
            
        print("Sending POST request to /loan_form")
        response = client.post('/loan_form', data={
            'income': '3300',
            'coapplicant_income': '0',
            'loan_amount': '185',
            'loan_term': '360',
            'credit_history': '750', # we grab this as credit_score in routes.py
            'education': 'Graduate',
            'married': 'Yes',
            'dependents': '0',
            'property_area': 'Urban'
        }, follow_redirects=True)
        
        print("Response URL:", response.request.path)
        
        # Check the prediction from the DB
        app_record = LoanApplication.objects.first()
        if app_record:
             print("Database logged prediction:", app_record.prediction)
        else:
             print("No loan application was saved to DB.")
