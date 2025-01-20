# toy_project/run.py
from app import create_app
import os

app = create_app()

if __name__ == '__main__':
    # uploads 폴더가 없다면 생성
    os.makedirs('app/static/uploads', exist_ok=True)
    app.run(debug=True)