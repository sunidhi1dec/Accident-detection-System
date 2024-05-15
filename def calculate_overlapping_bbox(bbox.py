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
    detected_objects = results(model)

    # Initialize a list to store overlapping rectangles
    overlapping_rectangles = []

    # Iterate through detected objects
    for obj1 in detected_objects:
        for obj2 in detected_objects:
            if obj1.name == obj2.name and obj1 != obj2:
                if rectangles_overlap(obj1.bbox, obj2.bbox):
                    overlapping_rectangles.append(obj1)

    # Now, capture the image of the overlapping region
    for obj in overlapping_rectangles:
        x, y, w, h = calculate_overlapping_bbox(obj.bbox, overlapping_rectangles)
        overlapping_image = image[y:y + h, x:x + w]
        cv2.imwrite("overlapping_region.jpg", overlapping_image)