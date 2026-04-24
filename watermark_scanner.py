from PIL import Image, ExifTags

def scan_for_ai_metadata(image_path):

    # Scans the hidden EXIF data and PNG text chunks of an image for known AI signatures. Returns: (is_fake: bool, reason: str)
    
    ai_signatures = [
        "midjourney", "dall-e", "openai", "stable diffusion", 
        "google ai", "google", "synthid", "novelai", "stablediffusion",
        "invokeai", "comfyui", "bing image creator"
    ]
    
    try:
        with Image.open(image_path) as img:
    
            # Check PNG Text Chunks (Where Midjourney and Stable Diffusion hide their prompts)
            if img.info:
                for key, value in img.info.items():
                    text_data = str(value).lower()
                    for sig in ai_signatures:
                        if sig in text_data:
                            return True, f"Found AI signature '{sig}' in PNG metadata ({key})"

            # Check traditional EXIF Data (Where DALL-E and standard cameras hide data)
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                    text_data = str(value).lower()
                    
                    # Usually hides in 'Software', 'Make', or 'ImageDescription'
                    for sig in ai_signatures:
                        if sig in text_data:
                            return True, f"Found AI signature '{sig}' in EXIF metadata ({tag_name})"
                            
    except Exception as e:
        return False, f"Could not read metadata: {e}"
        
    return False, "No known AI metadata found."

# Quick test if run directly.
if __name__ == "__main__":

    test_image = "fake1.png"
    is_fake, reason = scan_for_ai_metadata(test_image)
    if is_fake:
        print(f"[METADATA FOUND] Image is FAKE. Reason: {reason}")
    else:
        print("[CLEAR] No AI metadata found. Handing off to PyTorch...")
