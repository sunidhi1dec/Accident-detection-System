from flask import Flask, request, jsonify, render_template
import cv2
import math
import cvzone
from ultralytics import YOLO
import numpy as np
import base64

app = Flask(__name__)

# Initialize YOLO model and video capture
model = YOLO('../yolo-weights/yolov8n.pt')
classNames = ["persons", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "trucks"]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def detect_objects(image):
    results = model(image, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)
            # class names
            cls = int(box.cls[0])
            if 0 <= cls < len(classNames):
                # Access the class name and confidence score using valid indices
                label = f'{classNames[cls]} {conf}'
                cvzone.putTextRect(image, label, (max(0, x1), max(35, y1)))
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_with_objects = detect_objects(frame)
    _, buffer = cv2.imencode('.jpg', frame_with_objects)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': encoded_image})

if __name__ == '__main__':
    app.run(debug=True)
