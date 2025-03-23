# Emotion Detection Model

This project implements an emotion detection model using PyTorch. The model is trained on the FER2013 dataset to classify facial expressions into seven different emotions: angry, disgust, fear, happy, sad, surprise, and neutral.

## Dataset

The model uses the FER2013 dataset, which can be downloaded from Kaggle:
[FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

## Project Structure

```
emotion_detection/
├── data/               # Dataset directory
├── models/            # Saved model checkpoints
├── src/
│   ├── dataset.py     # Dataset loading and preprocessing
│   ├── model.py       # Model architecture
│   ├── train.py       # Training script
│   └── utils.py       # Utility functions
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the FER2013 dataset and place it in the `data` directory.

## Usage

To train the model:
```bash
python src/train.py
```

## Model Architecture

The model uses a CNN architecture with the following layers:
- Input: 48x48 grayscale images
- Conv2D layers with ReLU activation and BatchNorm
- MaxPooling layers
- Fully connected layers
- Output: 7 emotion classes

## Results

The model achieves approximately 58-61% accuracy on the test set. 
