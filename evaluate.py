import torch
import glob
import os
from dataset_import import setup_real_data
from model_architecture import build_model

def run_final_exam(model, test_loader, device):
    print("\n" + "="*50)
    print("Testing the model with 100,000 unseen)")
    print("="*50)
    
    model.eval()  # Lock the brain so it cannot learn/cheat
    correct = 0
    total = 0
    
    # torch.no_grad() disables gradient tracking to save VRAM
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Using AMP (16-bit math) evaluate faster on compatible GPUs.
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print an update every 100 batches
            if (i + 1) % 100 == 0:
                print(f"Graded [{total}/100000] images... Current Score: {100 * correct / total:.2f}%")
                
    accuracy = 100 * correct / total
    print("\n" + "="*50)
    print(f"True Generalized Accuracy: {accuracy:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    
    # Load the data (It will automatically grab the SafeImageFolder for the test set)
    _, test_loader, classes = setup_real_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build the blank architecture
    model, _, _ = build_model(device)
    
    # Find the best checkpoint
    checkpoints = glob.glob("checkpoints/*.pth")
    if not checkpoints:
        print("ERROR: No checkpoints found in the folder.")
    else:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"\nLoading weights from: {latest_checkpoint}")
        
        # Inject the weights
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Start the test
        run_final_exam(model, test_loader, device)