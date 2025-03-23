import torch
from PIL import Image
import torchvision.transforms as transforms
from utils import load_model, predict_emotion, get_emotion_label, visualize_prediction

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction
    """
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load and convert to grayscale
    image = Image.open(image_path).convert('L')
    # Apply transforms
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    return image

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load the trained model
    model = load_model('models/best_model.pth', device)
    
    # Get image path from user
    image_path = input('Enter the path to your image: ')
    
    try:
        # Preprocess the image
        image = preprocess_image(image_path)
        
        # Make prediction
        prediction = predict_emotion(model, image, device)
        emotion = get_emotion_label(prediction)
        
        print(f'\nPredicted Emotion: {emotion}')
        
        # Visualize the result
        visualize_prediction(image.squeeze(), prediction)
        
    except Exception as e:
        print(f'Error: {str(e)}')

if __name__ == '__main__':
    main() 