# toy_project/app/routes.py
from flask import Blueprint, request, jsonify, render_template, current_app
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from test import JellyfishDetector
import cv2

main = Blueprint('main', __name__)

# 모델 초기화
base_dir = Path(os.getcwd())
weights_path = base_dir / 'runs' / 'train' / 'exp' / 'best_model_20250117_221328.pth'
detector = JellyfishDetector(weights_path)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 이미지 처리 및 탐지
            result_img, detections = detector.detect(filepath)
            
            # 결과 이미지 저장
            result_filename = f'result_{filename}'
            result_path = os.path.join(current_app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_img)
            
            # 결과 반환
            results = {
                'detections': detections,
                'result_image': f'/static/uploads/{result_filename}'
            }
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        finally:
            # 원본 파일 삭제
            if os.path.exists(filepath):
                os.remove(filepath)
    
    return jsonify({'error': 'Invalid file type'}), 400