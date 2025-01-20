# toy_project/app/__init__.py
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # 업로드 설정
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
    app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
    
    # 블루프린트 등록
    from app.routes import main
    app.register_blueprint(main)
    
    return app