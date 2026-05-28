# Remote Sensing AI Demo

AI-powered scene classification and image analysis, built with PyTorch + ResNet50.

## Overview

This project demonstrates how deep learning models can be used for scene understanding and image classification — a core technique in remote sensing and geospatial analysis. The same deep learning approach is used for:

- Land cover / land use classification from satellite imagery
- Urban change detection
- Vegetation health monitoring
- Natural disaster assessment

## How It Works

1. **Input**: Captures a photo via webcam (or falls back to screenshot)
2. **Model**: Uses pre-trained ResNet50 (ImageNet) to classify the scene
3. **Output**: Generates a chart showing top predictions and scene categories

## Results

The script outputs visual results in the `results/` folder:

| File | Description |
|---|---|
| `webcam_photo.jpg` | Captured input image |
| `classification_result.png` | Side-by-side view: input + top predictions chart |
| `segmentation_map.png` | (if using DeepLabV3) Pixel-wise classification |
| `overlay.png` | (if using DeepLabV3) Input blended with segmentation |

## Usage

```bash
pip install -r requirements.txt
python scene_classifier.py
```

## Techniques Demonstrated

- **Transfer Learning**: Using pre-trained models (no training from scratch)
- **Convolutional Neural Networks (CNNs)**: The backbone of modern image AI
- **Scene Classification**: Categorizing an image into semantic classes
- **Semantic Segmentation**: (DeepLabV3 mode) Classifying every pixel

## Tech Stack

- Python 3.9+
- PyTorch / torchvision
- ResNet50 (pre-trained on ImageNet)
- OpenCV (webcam capture)
- Matplotlib (visualization)

## Author

桂林理工大学 遥感科学与技术

## License

MIT
