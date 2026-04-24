import os
import shutil
import subprocess
import glob

def prepare_massive_dataset():
    source_root = r"D:\AI_Data\GenImage_Raw\genimage"
    final_dir = r"D:\AI_Data\GenImage_Ready"
    seven_zip_exe = r"C:\Program Files\7-Zip\7z.exe"

    if not os.path.exists(seven_zip_exe):
        print("ERROR: 7-Zip not found at default location. Please install 7-Zip.")
        return

    categories = {
        "train_real": os.path.join(final_dir, "train", "REAL"),
        "train_fake": os.path.join(final_dir, "train", "FAKE"),
        "test_real": os.path.join(final_dir, "test", "REAL"),
        "test_fake": os.path.join(final_dir, "test", "FAKE")
    }

    for folder in categories.values():
        os.makedirs(folder, exist_ok=True)

    generators = ["Midjourney", "glide", "stable_diffusion_v_1_4", 
                  "stable_diffusion_v_1_5", "wukong", "VQDM", "BigGAN", "ADM"]

    for gen in generators:
        gen_dir = os.path.join(source_root, gen)
        if not os.path.exists(gen_dir):
            continue
            
        print(f"\n{'='*40}\nProcessing {gen}...\n{'='*40}")
        
        # Extract the Master Zip
        master_zip = glob.glob(os.path.join(gen_dir, "*.zip")) + glob.glob(os.path.join(gen_dir, "*.z01"))
        if master_zip:
            print(f"Extracting {master_zip[0]}...")
            subprocess.run([seven_zip_exe, "x", master_zip[0], f"-o{gen_dir}", "-y"], check=True)
            
            # Delete raw archive chunks to refund space
            print("Extraction complete. Deleting raw archives...")
            
            # Use 'set' to remove duplicates, and try/except for safety
            archives = set(glob.glob(os.path.join(gen_dir, "*.z*")) + glob.glob(os.path.join(gen_dir, "*.zip")))
            for archive_file in archives:
                try:
                    os.remove(archive_file)
                except FileNotFoundError:
                    pass
        
        # Move and Rename the extracted files safely
        print("Organizing images into GenImage_Ready...")
        for root, _, files in os.walk(gen_dir):
            normalized_root = root.replace('\\', '/')
            dest = None
            if '/train/nature' in normalized_root: dest = categories["train_real"]
            elif '/train/ai' in normalized_root: dest = categories["train_fake"]
            elif '/val/nature' in normalized_root: dest = categories["test_real"]
            elif '/val/ai' in normalized_root: dest = categories["test_fake"]

            if dest and files:
                for item in files:
                    src_path = os.path.join(root, item)
                    dest_path = os.path.join(dest, f"{gen}_{item}")
                    try:
                        shutil.move(src_path, dest_path)
                    except Exception:
                        pass # Skip if already moved

        # Delete the empty generator folder completely
        shutil.rmtree(gen_dir, ignore_errors=True)
        print(f"{gen} successfully processed and cleaned up!")

if __name__ == "__main__":
    prepare_massive_dataset()