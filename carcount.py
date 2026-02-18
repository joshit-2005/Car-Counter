import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import Sort

cap = cv2.VideoCapture("cars.mp4")
model = YOLO("yolov8l.pt")

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

# ✅ FIXED: Safe mask loading
mask = cv2.imread("mask.png")
height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"Video size: {width}x{height}")

if mask is None:
    print("⚠️ mask.png not found - using FULL FRAME")
    mask = None
elif mask.shape[:2] != (height, width):
    print(f"⚠️ Mask size {mask.shape[:2]} != video {height}x{width} - using FULL FRAME")
    mask = None
else:
    print("✅ Mask loaded correctly!")

# ✅ FIXED: Safe graphics loading  
try:
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
except:
    print("⚠️ graphics.png not found - skipping overlay")
    imgGraphics = None

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [150, 297, 950, 297]  # LEFT=150, RIGHT=950 (much wider!)

totalCount = []

while True:
    success, img = cap.read()
    if not success:
        break
    
    # ✅ FIXED: Safe masking
    if mask is not None:
        imgRegion = cv2.bitwise_and(img, img, mask=mask)
    else:
        imgRegion = img
    
    # Safe graphics overlay
    if imgGraphics is not None:
        try:
            img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
        except:
            pass  # Skip if error
    
    # ✅ FIXED: NO stream=True + verbose debug
    results = model(imgRegion, conf=0.25, verbose=False)
    
    detections = np.empty((0, 5))
    
    # Process detections
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                
                # ✅ FIXED: Proper vehicle logic with DEBUG
                if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.25:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
                    print(f"✅ Vehicle: {currentClass} {conf:.2f}")  # DEBUG
    
    print(f"Trackers: {len(resultsTracker := tracker.update(detections))}")  # DEBUG
    
    # Draw counting line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (x1, y1-10), scale=1.5, thickness=2, offset=5)
        
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Count vehicles crossing line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in totalCount:
                totalCount.append(id)
                print(f"🎉 COUNTED #{len(totalCount)} - ID {int(id)}")
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display count
    cv2.putText(img, f'Vehicles: {len(totalCount)}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.imshow("Car Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"🎊 FINAL TOTAL: {len(totalCount)} vehicles counted!")
