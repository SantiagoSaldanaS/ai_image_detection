# ai_image_detection
Hybrid Deepfake Detector combining a ResNet-50 CNN and a Vision Transformer (ViT-B/16) to identify AI-generated images. It combines two layers to increase accuracy: a Layer 1 basic EXIF/PNG metadata scanner to catch raw generations, and a Layer 2 PyTorch vision model for deep structural analysis.
