from flask import Flask
from config import Config
from flask_login import LoginManager
from mongoengine import connect
import mongomock
import os

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

def create_app(config_class=Config):
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config.from_object(config_class)
    
    # Initialize MongoEngine using mongomock for testing without a real DB
    connect('housing_loan_db', host='mongodb://localhost', mongo_client_class=mongomock.MongoClient)
    
    login_manager.init_app(app)

    # Register models here to ensure they are known to SQLAlchemy
    from app.models import User
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.objects(id=user_id).first()

    from app.auth import auth_bp
    app.register_blueprint(auth_bp)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
