import torch
from ultralytics import YOLO

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    model = YOLO("yolo11n.pt")

    results = model.train(data=r"data/data.yaml", epochs=20, imgsz=640, device=device)
