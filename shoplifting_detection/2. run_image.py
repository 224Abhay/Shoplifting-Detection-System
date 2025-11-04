import cv2
from ultralytics import YOLO

model = YOLO(r"shoplifting_detection\runs\detect\train\weights\best.pt")

image_path = r"test.jpeg"
image = cv2.imread(image_path)

results = model(image, conf=0.1)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])

        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("YOLOv11 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
