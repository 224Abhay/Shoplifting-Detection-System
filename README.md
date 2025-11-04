# Shoplifting Detection System

## Overview
The Shoplifting Detection System is designed to identify shoplifting activities using advanced machine learning techniques. This project leverages computer vision to detect suspicious activities in real-time.

## Features
- Real-time detection of shoplifting activities.
- Pre-trained weights for quick deployment.
- Configurable training and inference scripts.

## Project Structure
```
shoplifting_detection/
├── train.py          # Script for training the model
├── run_image.py      # Script for running inference on images
runs/
├── detect/
│   ├── train/
│   │   ├── args.yaml       # Training arguments
│   │   ├── results.csv     # Training results
│   │   ├── weights/
│   │   │   ├── best.pt     # Best model weights
│   │   │   ├── last.pt     # Last model weights
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required libraries (install via `requirements.txt` if available)

### Training the Model
To train the model, run the following command:
```bash
python train.py
```

### Running Inference
To run inference on an image, use:
```bash
python run_image.py --image_path <path_to_image>
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Open-source libraries and frameworks used in this project.
- Contributors and collaborators.