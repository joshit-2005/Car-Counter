import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math

# Load YOLO model (will auto-use cache)
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

# Load your image
img_path = "BikesHelmets0.png"
img = cv2.imread(img_path)

# Run YOLO detection
results = model(img, conf=0.3)

# Process detections
for r in results:
    boxes = r.boxes
    for box in boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Calculate width and height
        w, h = x2 - x1, y2 - y1
        
        # Get confidence and class
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
        currentClass = classNames[cls]
        
        # Draw bounding box and label with THINNER lines
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=1, colorR=(255, 0, 255))  # rt=1 (was 2)
        cvzone.putTextRect(img, f'{currentClass} {conf:.2f}', 
                          (max(0, x1), max(35, y1)),
                          scale=0.6, thickness=1, offset=8)  # Smaller scale & thinner text

# Display count of detected objects
print(f"Total objects detected: {len(results[0].boxes)}")

# Show the image - FULL SIZE (no resizing)
cv2.namedWindow("YOLO Object Detection", cv2.WINDOW_NORMAL)  # Allows resizing
cv2.imshow("YOLO Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite("detected_objects.jpg", img)
print("Result saved as 'detected_objects.jpg'")
