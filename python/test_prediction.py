import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from python.hybrid_model import HybridBrainTumorModel
import matplotlib.pyplot as plt

def load_model(model_path):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HybridBrainTumorModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Preprocess image for model input"""
    # Load and resize image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((256, 256))  # Resize to match model input size
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def predict_image(model, image_tensor, device):
    """Get model prediction"""
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence, probabilities[0].cpu().numpy()

def visualize_prediction(image, prediction, confidence, probabilities, save_path=None):
    """Visualize prediction results"""
    plt.figure(figsize=(10, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Prediction: {"Tumor" if prediction == 1 else "No Tumor"}\nConfidence: {confidence:.2%}')
    plt.axis('off')
    
    # Plot probabilities
    plt.subplot(1, 2, 2)
    plt.bar(['No Tumor', 'Tumor'], probabilities)
    plt.title('Class Probabilities')
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Create results directory
    os.makedirs('prediction_results', exist_ok=True)
    
    # Load the model
    model_path = 'hybrid_model.pth'
    print(f"Loading model from: {model_path}")
    
    model, device = load_model(model_path)
    
    # Process images from pred directory
    pred_dir = 'pred'
    for img_name in os.listdir(pred_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            print(f"\nProcessing {img_name}...")
            img_path = os.path.join(pred_dir, img_name)
            
            # Preprocess image
            image_tensor, original_image = preprocess_image(img_path)
            
            # Get prediction
            pred_class, confidence, probs = predict_image(model, image_tensor, device)
            
            # Visualize results
            save_path = os.path.join('prediction_results', f'result_{img_name}')
            visualize_prediction(original_image, pred_class, confidence, probs, save_path)
            print(f"Prediction: {'Tumor' if pred_class == 1 else 'No Tumor'} (Confidence: {confidence:.2%})")
            print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    main() 