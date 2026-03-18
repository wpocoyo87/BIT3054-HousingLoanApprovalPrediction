import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_very_secret_key'
    # Fallback to a free cloud MongoDB Atlas cluster so the app works without local installation
    MONGODB_SETTINGS = {
        'host': os.environ.get('MONGO_URI') or 'mongodb+srv://demo_user:demo_password123@cluster0.zox2a.mongodb.net/housing_loan_db?retryWrites=true&w=majority'
    }
