import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import collections
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
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)
        return x

class ShopliftingDetector(nn.Module):
    def __init__(self, num_classes=2, hidden_size=64):
        super(ShopliftingDetector, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ShopliftingDetector().to(device)
model.load_state_dict(torch.load(r"model\shoplifting_detector.pth"))
model.eval()

sequence_length = 16
frame_width, frame_height = 224, 224
frame_buffer = collections.deque(maxlen=sequence_length)

def detect_humans(frame):
    results = yolo_model(frame)
    human_crops = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()
            if int(cls) == 0:
                human_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if human_crop.size > 0:
                    human_crop = cv2.resize(human_crop, (frame_width, frame_height))
                    human_crop = human_crop / 255.0 
                    human_crops.append(human_crop)
    return human_crops

def predict_live():
    cap = cv2.VideoCapture(r"data\Shoplifting_raw\Shoplifting (59).mp4")  # Open video file

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        human_crops = detect_humans(frame)
        bboxes = []
        for result in yolo_model(frame):
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                if int(cls) == 0:  # Class 0 is 'person'
                    bboxes.append((int(x1), int(y1), int(x2), int(y2)))

        for i, human in enumerate(human_crops):
            frame_buffer.append(human)

            if len(frame_buffer) == sequence_length:
                seq = np.array(frame_buffer)
                seq = torch.tensor(seq, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)
                with torch.no_grad():
                    output = model(seq)
                    prediction = torch.argmax(output, dim=1).item()
                    label = "Shoplifting" if prediction == 1 else "Normal"

                # Draw bounding box and label
                x1, y1, x2, y2 = bboxes[i]
                color = (0, 0, 255) if label == "Shoplifting" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Live Shoplifting Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

predict_live()