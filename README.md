# Shoplifting Detection System (v2)

## Overview
The Shoplifting Detection System is designed to identify shoplifting activities using advanced machine learning techniques. This project leverages computer vision and temporal memory to detect suspicious activities in real-time.


## What's New in v2
This version (v2) introduces significant improvements over the initial version (v1):
- **v1**: Utilized YOLO for object detection and image data for shoplifting detection.
- **v2**: 
  - Introduced a **CNN** (ResNet50) for feature extraction.
  - Added an **LSTM** for temporal memory to analyze sequences of video frames.
  - Transitioned from image data to **video data**, which is preprocessed and converted into NumPy arrays for training and inference.

## Features
- Real-time detection of shoplifting activities.
- Pre-trained weights for quick deployment.
- Configurable training and inference scripts.
- Improved accuracy with temporal memory and video-based analysis.


## Output Example

Below is an example of the system's output, showcasing real-time shoplifting detection:

![Shoplifting Detection Output](sample/output_video.gif)

## Project Structure
```
shoplifting_detection/
├── preprocess_data.py  # Script for preprocessing video data into NumPy arrays
├── train_model.py      # Script for training the model
├── test_model_on_video.py  # Script for running inference on video data
model/
├── shoplifting_detector.pth
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required libraries (install via `requirements.txt` if available)

### Training the Model
To train the model, run the following command:
```bash
python train_model.py
```

### Running Inference
To run inference on a video, use:
```bash
python test_model_on_video.py
```

## Data
https://data.mendeley.com/datasets/r3yjf35hzr/1

The project uses the "Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad" which contains:
- **Normal**: Contains video data of normal activities.
- **Shoplifting**: Contains video data of shoplifting activities. This data was has been refined for better training. Original was "Shoplifting Raw".

### Data Preprocessing
The video data is preprocessed and converted into NumPy arrays for efficient training and inference. The preprocessing steps include:
1. Resizing frames to a fixed resolution.
2. Normalizing pixel values.
3. Converting video sequences into NumPy arrays.

The `preprocess_data.py` script handles this preprocessing and organizes the data for training and testing.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Open-source libraries and frameworks used in this project.
- Contributors and collaborators.