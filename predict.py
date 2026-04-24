import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import glob
import os

from model_architecture import build_model

def load_image(path_or_url):
   
   # We load an image from the drive or a website.
    if path_or_url.startswith('http'):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(path_or_url, headers=headers)
        response.raise_for_status() 
        return Image.open(BytesIO(response.content)).convert('RGB')
    else:
        return Image.open(path_or_url).convert('RGB')

def analyze_image(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find the best checkpoint
    checkpoints = glob.glob("checkpoints/*.pth")
    if not checkpoints:
        print("ERROR: No checkpoints found in the folder!")
        return
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    
    # Rebuild the blank architecture and inject the weights
    model, _, _ = build_model(device)
    
    # Unpack the time capsule
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict']) 
    
    model.eval() # Lock the brain into evaluation mode
    
    # Preprocess the image exactly as we did during training and without destroying the aspect ratio
    transform = transforms.Compose([
        transforms.Resize(256),         # 1. Scale the shortest edge to 256, preserving the true shape
        transforms.CenterCrop(224),     # 2. Punch a perfect 224x224 square out of the center
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    # Load and format the image
    img = load_image(image_path)
    input_tensor = transform(img).unsqueeze(0).to(device) 
    
    # Run the inference
    with torch.no_grad():

        # AMP should make this prediction happen in milliseconds
        with torch.amp.autocast('cuda'):
            raw_output = model(input_tensor)
            probabilities = F.softmax(raw_output[0], dim=0)
        
    classes = ['FAKE', 'REAL']
    predicted_index = torch.argmax(probabilities).item()
    confidence = torch.max(probabilities).item() * 100
    
    print("\n" + "="*60)
    print(f" TARGET: {image_path}")
    print("="*60)
    print(f" Prediction : {classes[predicted_index]}")
    print(f" Confidence : {confidence:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    fake1 = "fake.png"
    analyze_image(fake)

    real1 = "real.jpg"
    analyze_image(real)
