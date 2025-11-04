import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_frames(video_path, frame_rate=5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)
    frames = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    return frames

def process_videos(input_folder, output_folder, sequence_length=16):
    os.makedirs(output_folder, exist_ok=True)
    label_map = {"normal": 0, "shoplifting": 1}
    label_list = []
    
    seq_count = 0
    for label in ["normal", "shoplifting"]:
        class_folder = os.path.join(input_folder, label)
        for video in tqdm(os.listdir(class_folder)):
            video_path = os.path.join(class_folder, video)
            frames = extract_frames(video_path)
            
            if len(frames) >= sequence_length:
                for i in range(0, len(frames) - sequence_length + 1, sequence_length):
                    sequence = np.array(frames[i:i + sequence_length])
                    seq_filename = f"seq_{seq_count:04d}.npy"
                    np.save(os.path.join(output_folder, seq_filename), sequence)
                    label_list.append([seq_filename, label_map[label]])
                    seq_count += 1
    
    label_df = pd.DataFrame(label_list, columns=["filename", "label"])
    label_df.to_csv(os.path.join(output_folder, "labels.csv"), index=False)
    print(f"Processed {seq_count} sequences and saved to {output_folder}")

process_videos(input_folder=r"train_for_shoplifting\Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad", output_folder="processed_data")