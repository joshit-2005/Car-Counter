from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO("yolov8l.pt")

# COCO class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = model(img, conf=0.3)
    
    # Count objects per class
    class_counts = {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            class_counts[currentClass] = class_counts.get(currentClass, 0) + 1
            
            # Draw bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'{currentClass} {math.ceil((box.conf[0] * 100)) / 100:.2f}', 
                              (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=8)
    
    total_count = len(results[0].boxes)
    output_path = image_path.replace('uploads/', 'uploads/detected_')
    
    cv2.imwrite(output_path, img)
    return total_count, class_counts, output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        total_count, class_counts, detected_path = detect_objects(filepath)
        
        return jsonify({
            'total_objects': total_count,
            'class_counts': class_counts,
            'image_url': detected_path.replace('static/', '/static/')
        })

if __name__ == '__main__':
    app.run(debug=True)
