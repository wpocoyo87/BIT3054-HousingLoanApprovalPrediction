from app import create_app
from app.models import User, LoanApplication, Prediction

app = create_app()

@app.shell_context_processor
def make_shell_context():
    return {'User': User, 'LoanApplication': LoanApplication, 'Prediction': Prediction}

if __name__ == '__main__':
    app.run(debug=True, port=5001)
