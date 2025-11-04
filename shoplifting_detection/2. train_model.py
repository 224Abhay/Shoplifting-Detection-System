import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

class ShopliftingDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.labels_df = pd.read_csv(os.path.join(data_folder, "labels.csv"))
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        sequence = np.load(os.path.join(self.data_folder, row["filename"]))
        sequence = torch.tensor(sequence, dtype=torch.float32).permute(3, 0, 1, 2)
        label = torch.tensor(row["label"], dtype=torch.long)
        return sequence, label

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        batch_size, c, seq_len, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * seq_len, c, h, w)
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = ShopliftingDataset(data_folder="processed_data")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "model\shoplifting_detector.pth")
print("Model training complete and saved as model\shoplifting_detector.pth")
