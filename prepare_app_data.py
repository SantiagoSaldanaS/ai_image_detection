from datasets import load_dataset
import os

def build_app_dataset():
    base_dir = r"D:\AI_Data\App_Dataset\train"
    real_dir = os.path.join(base_dir, "REAL")
    fake_dir = os.path.join(base_dir, "FAKE")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print("Loading AI-vs-Real dataset from cache...")
    
    dataset = load_dataset("Parveshiiii/AI-vs-Real", split="train", verification_mode="no_checks")
    
    print(f"Dataset loaded! Unpacking {len(dataset)} images to disk...")
    
    for i, item in enumerate(dataset):
        try:
            # Convert to RGB to prevent crashes from weird grayscale PNGs
            img = item['image'].convert('RGB') 
        
            label = item['binary_label'] 
            
            if label == 1:
                img.save(os.path.join(real_dir, f"real_{i}.jpg"))
            else:
                img.save(os.path.join(fake_dir, f"fake_{i}.jpg"))
                
            if (i + 1) % 2000 == 0: 
                print(f"Saved {i + 1} / {len(dataset)} images...")
                
        except Exception as e:
            print(f"Skipped image {i} due to error: {e}")
            continue

    print("\nApp Dataset assembled.")

if __name__ == "__main__":
    build_app_dataset()