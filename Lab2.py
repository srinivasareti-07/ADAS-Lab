import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load parking image
image = cv2.imread("pedistrian.jpg")

# Run object detection
results = model(image)

# Copy image
output = image.copy()

for result in results:

    boxes = result.boxes

    for box in boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Confidence
        confidence = float(box.conf[0])

        # Class ID
        class_id = int(box.cls[0])

        # Class name
        label = model.names[class_id]

        # Width and Height
        width = x2 - x1
        height = y2 - y1

        # Bounding box area
        area = width * height

        # Estimate distance
        if area > 120000:
            warning = "STOP!"
            color = (0,0,255)

        elif area > 60000:
            warning = "BRAKE NOW"
            color = (0,140,255)

        elif area > 25000:
            warning = "SLOW DOWN"
            color = (0,255,255)

        else:
            warning = "SAFE"
            color = (0,255,0)

        # Draw rectangle
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)

        # Object name
        cv2.putText(output,
                    f"{label} {confidence:.2f}",
                    (x1,y1-35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2)

        # Warning message
        cv2.putText(output,
                    warning,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

cv2.imshow("Parking Assistance System",output)

cv2.waitKey(0)

cv2.destroyAllWindows()

