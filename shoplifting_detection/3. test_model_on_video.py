import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import collections

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=None)  # Use ResNet50 as the backbone
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove the fully connected layer

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape  # Correct the input shape order

        x = x.view(batch_size * seq_len, c, h, w)  # Flatten the sequence dimension
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape back to (batch_size, seq_len, features)

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

def preprocess_frame(frame):
    frame = cv2.resize(frame, (frame_width, frame_height))
    frame = frame / 255.0 
    return frame

def predict_live():
    cap = cv2.VideoCapture(r"data\Shoplifting_raw\Shoplifting (59).mp4")  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        frame_buffer.append(processed_frame)

        if len(frame_buffer) == sequence_length:
            seq = np.array(frame_buffer)
            seq = torch.tensor(seq, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)  # (1, T, C, H, W)
            with torch.no_grad():
                output = model(seq)
                probabilities = torch.softmax(output, dim=1).squeeze()  # Get probabilities
                prediction = torch.argmax(probabilities).item()

                # Apply threshold for "Shoplifting"
                shoplifting_prob = probabilities[1].item()
                if shoplifting_prob > 0.25:  # Threshold for "Shoplifting"
                    label = "Shoplifting"
                else:
                    label = "Normal"

            # Display prediction
            color = (0, 0, 255) if label == "Shoplifting" else (0, 255, 0)
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Live Shoplifting Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run Live Prediction
predict_live()