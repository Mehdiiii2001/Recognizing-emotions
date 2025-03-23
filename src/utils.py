import torch
import numpy as np
import matplotlib.pyplot as plt
from model import EmotionNet

def predict_emotion(model, image, device):
    """
    Predict emotion from a single image
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

def get_emotion_label(emotion_idx):
    """
    Convert emotion index to label
    """
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[emotion_idx]

def visualize_prediction(image, prediction, true_label=None):
    """
    Visualize the image and prediction
    """
    plt.figure(figsize=(4, 4))
    plt.imshow(image.squeeze(), cmap='gray')
    title = f'Predicted: {get_emotion_label(prediction)}'
    if true_label is not None:
        title += f'\nTrue: {get_emotion_label(true_label)}'
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_model(model_path, device):
    """
    Load a trained model
    """
    model = EmotionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model 