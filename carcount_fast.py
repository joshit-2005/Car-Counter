import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import Sort

# ⚡ ULTRA-FAST MODEL (nano instead of large)
model = YOLO("yolov8n.pt")  # 10x faster than yolov8l.pt!

cap = cv2.VideoCapture("cars.mp4")

# FASTER VEHICLE CLASSES ONLY (no full COCO list)
VEHICLES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.4)  # Faster params
limits = [150, 297, 950, 297]  # Wide line
totalCount = []

# Skip frames for speed
FRAME_SKIP = 2  # Process every 2nd frame
frame_count = 0

print("🚀 ULTRA-FAST MODE ACTIVATED!")
print("Model: yolov8n.pt | Frame skip: 2 | Max FPS boost: 5x")

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # ⚡ SKIP FRAMES (major speedup)
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Car Counter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # ⚡ RESIZE FRAME (2x speedup)
    h, w = img.shape[:2]
    img_small = cv2.resize(img, (640, 480))  # Standard YOLO input
    imgRegion = img_small  # No mask for max speed
    
    # ⚡ ULTRA-FAST PREDICTION
    results = model(imgRegion, conf=0.3, verbose=False, imgsz=640)
    
    detections = np.empty((0, 5))
    
    # ⚡ FAST PROCESSING (no math.ceil, direct indexing)
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # ⚡ VEHICLE CHECK BY CLASS ID (faster than string compare)
                if cls in VEHICLES and conf > 0.3:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
    
    # ⚡ FAST TRACKING
    resultsTracker = tracker.update(detections)
    
    # Scale back to original size
    scale_x = w / 640
    scale_y = h / 480
    
    # Draw on original image
    cv2.line(img, 
             (int(limits[0] * scale_x), int(limits[1] * scale_y)), 
             (int(limits[2] * scale_x), int(limits[3] * scale_y)), 
             (0, 0, 255), 3)
    
    for result in resultsTracker:
        x1, y1, x2, y2, track_id = result
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        w_rect, h_rect = x2 - x1, y2 - y1
        
        # ⚡ FAST DRAWING
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img, str(int(track_id)), (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        cx, cy = x1 + w_rect // 2, y1 + h_rect // 2
        cv2.circle(img, (cx, cy), 4, (255, 0, 255), -1)

        # Count crossing
        line_y = int(limits[1] * scale_y)
        if (int(limits[0] * scale_x) < cx < int(limits[2] * scale_x) and 
            line_y - 20 < cy < line_y + 20):
            if track_id not in totalCount:
                totalCount.append(track_id)
                cv2.line(img, 
                        (int(limits[0] * scale_x), line_y), 
                        (int(limits[2] * scale_x), line_y), (0, 255, 0), 5)

    # ⚡ FAST TEXT
    cv2.putText(img, f'Vehicles: {len(totalCount)}', (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    cv2.imshow("🚀 ULTRA-FAST Car Counter", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"🎉 FINAL COUNT: {len(totalCount)} vehicles!")
