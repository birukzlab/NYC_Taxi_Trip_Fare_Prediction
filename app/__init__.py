from flask import Flask
from dotenv import load_dotenv
import os

def create_app():
    """
    Create and configure the Flask app.
    """
    load_dotenv()

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default_secret_key')

    from app.routes import main
    app.register_blueprint(main)

    return app

