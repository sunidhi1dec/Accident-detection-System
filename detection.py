import math
import os
import cv2
import cvzone
from ultralytics import YOLO
from PIL import Image


cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
model = YOLO('../yolo-weights/yolov8n.pt')
classNames = ["persons", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "trucks"]
save_dir = r'C:\Users\DELL\OneDrive\Desktop\overlapping images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_image=False
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # if classNames=="cars" or classNames=="persons":
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # w, h = x1-x2, y2-y1
            # cvzone.cornerRect(img, (w, h, x1, y1))
            # confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)
            # class names
            cls = int(box.cls[0])
            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1.), max(35, y1)))
            if 0 <= cls < len(classNames):
                # Access the class name and confidence score using valid indices
                label = f'{classNames[cls]} {conf}'
                cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)))


    def calculate_overlapping_bbox(bbox, overlapping_objects):
        x, y, w, h = bbox
        for obj in overlapping_objects:
            x = min(x, obj.bbox[0])
            y = min(y, obj.bbox[1])
            w = max(obj.bbox[0] + obj.bbox[2], x + w) - x
            h = max(obj.bbox[1] + obj.bbox[3], y + h) - y
        return x, y, w, h


    def rectangles_overlap(bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        # Check if the rectangles overlap
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

        # Perform object detection (you need to load a trained model)


    detected_objects = results

    # Initialize a list to store overlapping rectangles
    overlapping_rectangles = []

    # Iterate through detected objects
    for obj1 in detected_objects:
        for obj2 in detected_objects:
            if obj1.name == obj2.name and obj1 != obj2:
                if rectangles_overlap(obj1.bbox, obj2.bbox):
                    overlapping_rectangles.append(obj1)
    # Now, capture the image of the overlapping region
    #for obj in overlapping_rectangles:
     #   x, y, w, h = calculate_overlapping_bbox(obj.bbox, overlapping_rectangles)

      #  overlapping_image = img[y:y + h, x:x + w]
       # if save_image:
        #    filename = os.path.join(save_dir, f"overlapping_region_{str(obj.id)}.jpg")
         #   cv2.imwrite("overlapping_region.jpg", overlapping_image)
          #  save_image = False

        #cv2.imshow("Image", img)
        #save_image.save(r'C:\Users\DELL\OneDrive\Desktop\overlapping images')

    #key = cv2.waitKey(1)
    # Check if the "s" key is pressed to save an image
    #if key == ord('s'):
     #   save_image = True

cv2.namedWindow("Python Webcam Screenshot App")
img_counter = 0
while True:
    ret,frame= cam.read()

    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("test",frame)

    k=cv2.waitKey(0)

    if k%256 == 27:
        print("Escape hit,closing the app")
        break
    elif k%256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name,frame)
        print("Screenshot taken")
        img_counter+=1
cv2.destroyAllWindows()
cam.release()