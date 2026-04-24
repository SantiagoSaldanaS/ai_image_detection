import torch
import os
import glob
from dataset_import import setup_real_data
from model_architecture import build_model

def manage_checkpoints(folder, prefix, max_keep):
    
    # We delete older checkpoints.
    files = glob.glob(os.path.join(folder, f"{prefix}_*.pth"))
    
    # Sort by modification time (oldest first)
    files.sort(key=os.path.getmtime)
    
    while len(files) > max_keep:
        file_to_delete = files.pop(0)
        os.remove(file_to_delete)
        print(f"Removed old checkpoint: {file_to_delete}")

def train_model(model, train_loader, criterion, optimizer, device, epochs=3, start_epoch=0, global_step=0):
    print(f"\n-Starting Training (Epoch {start_epoch+1}/{epochs}) ---")
    model.train() 
    
    os.makedirs("checkpoints", exist_ok=True)
    scaler = torch.amp.GradScaler('cuda') # Activates the AMP Speed Boost
    
    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Use 16-bit math for the forward pass to double the speed
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scale the loss backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            global_step += 1
            
            if global_step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{global_step}], "
                      f"Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")
            
            # Checkpoint every 15k steps
            if global_step % 15000 == 0:
                step_path = f"checkpoints/step_{global_step}.pth"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, step_path)
                print(f"\n[SAVE] Step Checkpoint created: {step_path}")
                manage_checkpoints("checkpoints", "step", max_keep=5)
                
        # Epoch Checkpoint
        epoch_path = f"checkpoints/epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, epoch_path)
        print(f"\n[SAVE] Epoch {epoch+1} Checkpoint created: {epoch_path}\n")
        manage_checkpoints("checkpoints", "epoch", max_keep=2)

    return model

def resume_training(model, optimizer):
    
    # We load the most recent checkpoint.
    checkpoints = glob.glob("checkpoints/*.pth")
    if not checkpoints:
        return model, optimizer, 0, 0
        
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Found checkpoint: {latest_checkpoint}. Resuming...")
    
    checkpoint = torch.load(latest_checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['global_step']

if __name__ == "__main__":
    train_loader, test_loader, classes = setup_real_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Compute device set to: {device}")
    
    model, criterion, optimizer = build_model(device)
    
    # Check for crashes and resume automatically
    model, optimizer, start_epoch, global_step = resume_training(model, optimizer)
    
    # Train for 3 epochs total
    if start_epoch < 3:
        trained_model = train_model(model, train_loader, criterion, optimizer, device, epochs=3, start_epoch=start_epoch, global_step=global_step)
        
        # We will evaluate after training completes fully
        from dataset_import import evaluate_model
        evaluate_model(trained_model, test_loader, device)
        torch.save(trained_model.state_dict(), "Hybrid_Final_Master.pth")
    else:
        print("Model has already finished all 3 epochs of training!")