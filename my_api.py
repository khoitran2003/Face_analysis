from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import os
import cv2

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

@app.route('/')
def hello():
    return render_template('index.html')

# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8325', debug=True)