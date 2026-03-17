# Image Forgery Detection Web App

A web application for detecting image forgery using deep learning and Error Level Analysis (ELA).

## Features

- **Image Upload**: Drag and drop or select images to analyze
- **Real-time Analysis**: Uses a trained ResNet model with ELA preprocessing
- **Confidence Meter**: Visual confidence score display
- **Tampered Region Heatmap**: Grad-CAM visualization of suspicious regions
- **Interactive Charts**: Probability distribution and confidence metrics

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (if not already trained):
   ```bash
   python train_casia2.py --mode train --dataset_root CASIA2.0_revised --out_dir runs/smoke
   ```

3. **Run the Web App**:
   ```bash
   python app.py
   ```

4. **Open in Browser**:
   Navigate to `http://localhost:5000`

## Model Architecture

- **Base Model**: ResNet18/ResNet50 with 6 input channels (RGB + ELA RGB)
- **Preprocessing**: Error Level Analysis (ELA) for artifact detection
- **Output**: Binary classification (Authentic vs Tampered)

## Files

- `app.py`: Flask web application
- `train_casia2.py`: Training and inference script
- `templates/index.html`: Web interface
- `requirements.txt`: Python dependencies
- `CASIA2.0_revised/`: Dataset directory (not included)

## Usage

1. Upload an image by dragging it to the upload area or clicking "Choose File"
2. The app will analyze the image and display:
   - Prediction (Authentic/Tampered)
   - Confidence score with visual meter
   - Original image
   - Tampered region heatmap overlay
   - Probability charts

## Technical Details

- **ELA Processing**: Images are recompressed at 90% JPEG quality and the difference is enhanced
- **Grad-CAM**: Uses gradient-weighted class activation mapping for heatmap generation
- **Preprocessing**: Images are resized to 224x224, normalized with ImageNet statistics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Flask
- OpenCV
- Pillow
- NumPy