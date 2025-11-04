import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)

yolo_model = YOLO(r"model\yolov8n.pt")

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x

class ShopliftingDetector(nn.Module):
    def __init__(self, num_classes=2, hidden_size=64):
        super(ShopliftingDetector, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.cnn(x).unsqueeze(1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShopliftingDetector().to(device)
model.load_state_dict(torch.load(r"model\shoplifting_detector.pth", map_location=device))
model.eval()

frame_width, frame_height = 224, 224

def detect_humans(image):
    results = yolo_model(image)
    human_crops = []
    bboxes = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if int(cls) == 0:
                human_crop = image[int(y1):int(y2), int(x1):int(x2)]
                if human_crop.size > 0:
                    human_crop = cv2.resize(human_crop, (frame_width, frame_height))
                    human_crop = human_crop / 255.0
                    human_crops.append(human_crop)
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    return human_crops, bboxes

def predict_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image")
        return
    
    human_crops, bboxes = detect_humans(image)
    
    for i, human in enumerate(human_crops):
        human_tensor = torch.tensor(human, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(human_tensor)
            prediction = torch.argmax(output, dim=1).item()
            label = "Shoplifting" if prediction == 1 else "Normal"
        
        x1, y1, x2, y2 = bboxes[i]
        color = (0, 0, 255) if label == "Shoplifting" else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Shoplifting Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

predict_on_image(r"C:\Users\abhay\Downloads\image-225.jpeg")
