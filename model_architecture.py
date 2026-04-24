import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights

class HybridDeepfakeDetector(nn.Module):
    def __init__(self):
        super(HybridDeepfakeDetector, self).__init__()
        
        # Load ResNet-50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Freeze ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Chop off the final layer using nn.Identity(). ResNet outputs 2048 features.
        self.resnet.fc = nn.Identity()
        
        # Load the Vision Transformer
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # Freeze ViT layers
        for param in self.vit.parameters():
            param.requires_grad = False

        # Chop off the final layer. ViT outputs 768 features.
        self.vit.heads = nn.Identity()
        
        # Build Fusion Head
        # Combine ResNet (2048) + ViT (768) = 2816 total features
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2) # Final output: FAKE (0) or REAL (1)
        )

    def forward(self, x):

        # Pass the image through both models simultaneously
        cnn_features = self.resnet(x)
        vit_features = self.vit(x)
        
        # Join the mathematical arrays together side by side
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        
        # Pass the fused data into our custom decision maker
        out = self.classifier(combined_features)
        return out

def build_model(device):

    print("Building Hybrid Model...")
    model = HybridDeepfakeDetector()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # We only want the optimizer to train our brand new fusion head
    # The backbones remain locked to save VRAM and prevent catastrophic forgetting
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) 
    
    return model, criterion, optimizer