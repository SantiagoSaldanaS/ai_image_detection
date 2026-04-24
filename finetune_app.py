import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import glob
import os
from model_architecture import build_model

def train_app_ready_model():
    
    dataset_path = r"D:\AI_Data\App_Dataset\train"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the best checkpoint.
    checkpoints = glob.glob("checkpoints/*.pth")
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"\nLoading base knowledge from: {latest_checkpoint}")
    
    model, _, _ = build_model(device)
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Freeze the Backbones
    print("Freezing ResNet and ViT backbones...")
    for param in model.resnet.parameters(): param.requires_grad = False
    for param in model.vit.parameters(): param.requires_grad = False
    for param in model.classifier.parameters(): param.requires_grad = True

    # This intentionally degrades the images so the AI learns to ignore bad compression
    app_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5), # Fakes bad camera focus
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Fakes weird screenshot colors
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5), # Fakes digital phone sharpening
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    train_data = datasets.ImageFolder(root=dataset_path, transform=app_transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # Low learning rate for 2 quick Epochs
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    print("\nFinetuning started...")
    model.train()
    
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}/2 (Simulating Internet Artifacts)")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 25 == 0:
                _, predicted = torch.max(outputs.data, 1)
                accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
                print(f"Step [{i+1}], Loss: {loss.item():.4f}, Batch Accuracy: {accuracy:.2f}%")

    torch.save({'model_state_dict': model.state_dict()}, "checkpoints/step_APP_READY.pth")
    print("\n[SUCCESS] App-Ready weights saved to step_APP_READY.pth.")

if __name__ == "__main__":
    train_app_ready_model()