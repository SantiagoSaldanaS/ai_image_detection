# ai_image_detection
Hybrid Deepfake Detector combining a ResNet-50 CNN and a Vision Transformer (ViT-B/16) to identify AI-generated images. It combines two layers to increase accuracy: a Layer 1 basic EXIF/PNG metadata scanner to catch raw generations, and a Layer 2 PyTorch vision model for deep structural analysis.


## Installation & Usage

1. Clone this repository.
2. Install the required dependencies: `pip install torch torchvision Pillow requests huggingface_hub`
3. Run the prediction script: `python predict.py`

**Note:** The model weights (~423 MB) are securely hosted on Hugging Face. The first time you run `predict.py`, the script will automatically download the `step_APP_READY.pth` checkpoint to your local machine. You do not need to download it manually, though here is the repo: https://huggingface.co/Santiago64/ai-image-detector-hybrid
