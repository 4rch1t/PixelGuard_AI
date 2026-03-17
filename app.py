import io
import os
from pathlib import Path
from datetime import datetime
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from torchvision import models, transforms
import cv2
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.units import inch

app = Flask(__name__)
app.secret_key = 'pixelguard_secret_key'

# Store analysis results temporarily
last_analysis = {}

# Model class (same as in train_casia2.py)
class ResNet6Ch(nn.Module):
    def __init__(self, base: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if base == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif base == "resnet50":
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError("base must be resnet18 or resnet50")

        # Replace first conv to accept 6 channels (RGB + ELA RGB).
        old = m.conv1
        new = nn.Conv2d(6, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
        with torch.no_grad():
            # Initialize: copy ImageNet weights for first 3 channels; duplicate for ELA channels.
            new.weight[:, :3, :, :] = old.weight
            new.weight[:, 3:, :, :] = old.weight
        m.conv1 = new

        # Replace classifier head for binary logit.
        m.fc = nn.Linear(feat_dim, 1)
        self.backbone = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # logits shape (B,1)

# GradCAM class for heatmap generation
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._acts = None
        self._grads = None
        self._hooks = []

        def fwd_hook(_m, _inp, out):
            self._acts = out

        def bwd_hook(_m, _gin, gout):
            self._grads = gout[0]

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits.squeeze()
        score.backward(retain_graph=False)

        acts = self._acts  # (B,C,H,W)
        grads = self._grads  # (B,C,H,W)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam  # (B,1,H,W) in [0,1]

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing functions
def compute_ela_rgb(img_rgb: Image.Image, quality: int = 90, enhance: int = 15) -> Image.Image:
    from PIL import ImageChops, ImageEnhance
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    jpeg = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img_rgb, jpeg)
    diff = ImageEnhance.Brightness(diff).enhance(enhance)
    return diff

def preprocess_image(image):
    image_size = 224
    ela_quality = 90
    ela_enhance = 15

    img = image.convert("RGB")

    ela = compute_ela_rgb(img, quality=ela_quality, enhance=ela_enhance)

    rgb_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    rgb_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    ela_norm = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    rgb_t = rgb_tf(img)
    ela_t = rgb_tf(ela)

    rgb_t = rgb_norm(rgb_t)
    ela_t = ela_norm(ela_t)

    x = torch.cat([rgb_t, ela_t], dim=0).unsqueeze(0)  # Add batch dimension
    return x

def extract_metadata(image: Image.Image, filename: str = None) -> dict:
    """Extract metadata and forensic information from image."""
    metadata = {}
    
    # Basic file info
    metadata['filename'] = filename or 'unknown'
    metadata['format'] = image.format or 'Unknown'
    metadata['mode'] = image.mode
    metadata['size'] = f"{image.width} x {image.height} px"
    
    # EXIF data
    try:
        exif_data = image._getexif()
        if exif_data:
            from PIL.ExifTags import TAGS
            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                try:
                    if isinstance(value, bytes):
                        value = value.decode('utf-8', errors='ignore')
                    exif_dict[tag_name] = str(value)[:100]  # Limit length
                except:
                    pass
            # Extract key EXIF fields
            if 'DateTime' in exif_dict:
                metadata['capture_date'] = exif_dict['DateTime']
            if 'Model' in exif_dict:
                metadata['camera_model'] = exif_dict['Model']
            if 'Make' in exif_dict:
                metadata['camera_make'] = exif_dict['Make']
            if 'Software' in exif_dict:
                metadata['software'] = exif_dict['Software']
            metadata['exif_count'] = len(exif_dict)
        else:
            metadata['exif_data'] = 'No EXIF data found'
    except:
        metadata['exif_data'] = 'Could not extract EXIF data'
    
    return metadata


def analyze_compression_artifacts(image: Image.Image) -> dict:
    """Analyze compression artifacts at multiple JPEG quality levels."""
    artifacts = {}
    
    # Convert to RGB if needed
    img_rgb = image.convert('RGB')
    img_array = np.array(img_rgb, dtype=np.float32)
    
    # Analyze at different quality levels
    quality_levels = [95, 85, 75, 50]
    quality_artifacts = []
    
    for quality in quality_levels:
        buf = io.BytesIO()
        img_rgb.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert('RGB')
        recomp_array = np.array(recompressed, dtype=np.float32)
        
        # Calculate difference
        diff = np.abs(img_array - recomp_array).mean()
        quality_artifacts.append({
            'quality': quality,
            'artifact_level': float(diff)
        })
    
    artifacts['quality_analysis'] = quality_artifacts
    
    # Calculate overall artifact score (how different from original)
    buf = io.BytesIO()
    img_rgb.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    recompressed = Image.open(buf).convert('RGB')
    recomp_array = np.array(recompressed, dtype=np.float32)
    avg_diff = np.abs(img_array - recomp_array).mean()
    artifacts['avg_artifact_score'] = float(avg_diff)
    
    return artifacts


def load_model():
    global model
    model_path = Path("runs/smoke/best_model.pt")
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model = ResNet6Ch(base=checkpoint.get("base_model", "resnet18"), pretrained=False)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        print("Model loaded successfully")
    else:
        print("Model not found. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if model is None:
        return jsonify({'error': 'Model not loaded'})

    try:
        # Read image
        image = Image.open(io.BytesIO(file.read()))

        # Preprocess
        input_tensor = preprocess_image(image).to(device)

        # Get prediction
        with torch.no_grad():
            logits = model(input_tensor)
            prob = torch.sigmoid(logits).item()
            prediction = "Tampered" if prob >= 0.5 else "Authentic"
            confidence = prob if prob >= 0.5 else 1 - prob

        # Generate heatmap
        cam = GradCAM(model, model.backbone.layer4)
        input_tensor.requires_grad_(True)
        cam_map = cam(input_tensor)
        cam.close()

        # Convert CAM to numpy and resize
        heatmap = cam_map.squeeze().detach().cpu().numpy()
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))

        # Create heatmap overlay
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_img = Image.fromarray(heatmap_colored)

        # Overlay on original image
        base = image.convert("RGBA")
        heat = heatmap_img.convert("RGBA")
        heat.putalpha(128)  # Semi-transparent
        overlay = Image.alpha_composite(base, heat)

        heatmap_buf = io.BytesIO()
        overlay.save(heatmap_buf, format='PNG')
        heatmap_b64 = f"data:image/png;base64,{__import__('base64').b64encode(heatmap_buf.getvalue()).decode()}"

        # Convert original image to base64 for display
        img_buf = io.BytesIO()
        image.save(img_buf, format='PNG')
        img_b64 = f"data:image/png;base64,{base64.b64encode(img_buf.getvalue()).decode()}"
        
        # Extract metadata
        metadata = extract_metadata(image, file.filename)
        
        # Analyze compression artifacts
        artifacts = analyze_compression_artifacts(image)
        
        # Generate compression comparison images (JPEG at different qualities)
        compression_images = {}
        for quality in [95, 75, 50]:
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            comp_img = Image.open(buf).convert('RGB')
            comp_array = np.array(comp_img)
            # Highlight artifacts with edge detection
            edges = cv2.Canny(cv2.cvtColor(comp_array, cv2.COLOR_RGB2GRAY), 50, 150)
            comp_img_pil = Image.fromarray(comp_array)
            comp_buf = io.BytesIO()
            comp_img_pil.save(comp_buf, format='PNG')
            comp_b64 = f"data:image/png;base64,{base64.b64encode(comp_buf.getvalue()).decode()}"
            compression_images[f"q{quality}"] = comp_b64

        # Store analysis for report generation
        analysis_id = str(uuid.uuid4())
        last_analysis[analysis_id] = {
            'prediction': prediction,
            'confidence': confidence,
            'probability': prob,
            'image': img_b64,
            'heatmap': heatmap_b64,
            'metadata': metadata,
            'compression_artifacts': artifacts,
            'compression_images': compression_images,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': file.filename
        }
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probability': prob,
            'image': img_b64,
            'heatmap': heatmap_b64,
            'metadata': metadata,
            'compression_artifacts': artifacts,
            'compression_images': compression_images,
            'analysis_id': analysis_id
        })

    except Exception as e:
        return jsonify({'error': str(e)})


def generate_forensics_report(analysis_id):
    """Generate a comprehensive forensics report PDF."""
    analysis = last_analysis.get(analysis_id)
    if not analysis:
        return None
    
    # Create PDF
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#5c4033'),
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#5c4033'),
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#8b7355'),
        spaceAfter=6
    )
    
    # Header
    story.append(Paragraph('PixelGuard', title_style))
    story.append(Paragraph('Image Forensics & Authenticity Analysis Report', styles['Heading3']))
    story.append(Spacer(1, 0.2*inch))
    
    # Report Info
    info_data = [
        ['Generated:', analysis['timestamp']],
        ['File Name:', analysis['filename']],
        ['Report ID:', analysis_id[:8]]
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f3f0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#5c4033')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e8e4e0'))
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Prediction Section
    story.append(Paragraph('Authenticity Assessment', heading_style))
    pred_color = colors.HexColor('#2d5016') if analysis['prediction'] == 'Authentic' else colors.HexColor('#8b3a3a')
    story.append(Paragraph(f'<b>Prediction: <font color="{pred_color.hexval()}">{analysis["prediction"]}</font></b>', normal_style))
    story.append(Paragraph(f'Confidence: {analysis["confidence"]*100:.1f}%', normal_style))
    story.append(Paragraph(f'Model Probability: {analysis["probability"]*100:.1f}%', normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata Section
    story.append(Paragraph('Image Metadata', heading_style))
    metadata = analysis['metadata']
    metadata_items = []
    for key, value in metadata.items():
        if key not in ['exif_data']:
            key_display = key.replace('_', ' ').title()
            metadata_items.append(Paragraph(f'<b>{key_display}:</b> {value}', normal_style))
    
    for item in metadata_items[:6]:
        story.append(item)
    story.append(Spacer(1, 0.2*inch))
    
    # Compression Analysis
    story.append(Paragraph('Compression Artifacts Analysis', heading_style))
    artifacts = analysis['compression_artifacts']
    avg_score = artifacts.get('avg_artifact_score', 0)
    story.append(Paragraph(f'Average Artifact Score: {avg_score:.4f}', normal_style))
    story.append(Paragraph('Quality-based Analysis:', normal_style))
    
    for item in artifacts['quality_analysis']:
        story.append(Paragraph(f'  • JPEG Quality {item["quality"]}: {item["artifact_level"]:.4f}', normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Analysis Details
    story.append(Paragraph('Analysis Summary', heading_style))
    if analysis['prediction'] == 'Authentic':
        summary = 'The image appears to be authentic with no significant signs of manipulation detected. The compression artifact analysis shows consistent patterns across quality levels.'
    else:
        summary = 'The image exhibits signs of potential manipulation or editing. Anomalies were detected in the forensic heatmap, and compression artifact patterns suggest possible modifications.'
    story.append(Paragraph(summary, normal_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Footer
    story.append(Paragraph(
        '<i>This report was generated using PixelGuard\'s AI-powered forensic analysis system. '
        'Results should be considered alongside other forensic techniques for comprehensive authentication.</i>',
        normal_style
    ))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer


@app.route('/generate-report/<analysis_id>', methods=['GET'])
def download_report(analysis_id):
    """Generate and download forensics report PDF."""
    pdf_buffer = generate_forensics_report(analysis_id)
    
    if pdf_buffer is None:
        return jsonify({'error': 'Analysis not found'}), 404
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'PixelGuard_Report_{timestamp}.pdf'
    
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)