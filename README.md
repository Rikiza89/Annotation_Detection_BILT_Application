# BILT Detection & Annotation System

A complete CPU-optimized object detection and dataset annotation platform using BILT (Because I Like Twice) - a PyTorch-based detection library designed for efficient training and inference on CPU devices including Windows PCs and Raspberry Pi 4.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Applications](#applications)
- [BILT Framework](#bilt-framework)
- [Workflow](#workflow)
- [Project Structure](#project-structure)

---

## Overview

This system provides three main capabilities:

1. **Dataset Creation & Annotation** - Capture images from camera and annotate them with bounding boxes
2. **Model Training** - Train custom BILT detection models on your annotated datasets
3. **Real-time Detection** - Run trained models for live object detection with counting and chain detection modes

### Key Features

- **CPU-Optimized**: Designed for efficient training and inference without GPU requirements
- **Platform Support**: Windows and Raspberry Pi 4
- **Camera Integration**: Multi-camera support with resolution control
- **Auto-labeling**: Use trained models to automatically label new images
- **Chain Detection**: Multi-step detection sequences for quality control workflows
- **Object Counting**: Track and count detected objects

---

## Installation

### Prerequisites

- **Windows**: Python 3.9-3.11
- **Raspberry Pi 4**: Python 3.9+ (comes pre-installed on Raspberry Pi OS)
- **Minimum RAM**: 4GB (8GB recommended)

### Step 1: Create Virtual Environment

Open Command Prompt (Windows) or Terminal (Raspberry Pi) in the project directory:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Raspberry Pi)
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: On Raspberry Pi 4, PyTorch installation may take 30-60 minutes as it builds from source.

### Step 3: Verify Installation

```bash
python -c "import torch; import cv2; print('Installation successful!')"
```

---

## Applications

The system consists of three main applications, each with dedicated `.bat` launchers (Windows) or can be run directly with Python (Raspberry Pi).

### 1. Annotation App (Port 5000)

**Purpose**: Create datasets by capturing images and annotating them with bounding boxes.

**Windows Launch**:
```bash
annotation\Start_BILT_Annotation.bat
```

**Raspberry Pi Launch**:
```bash
cd annotation
source ../venv/bin/activate
python bilt_service.py &  # Start service in background
python annotation_app.py
```

**Features**:
- Camera integration with live preview
- Bounding box annotation tool
- Class management
- Train/Val dataset split
- Model training interface
- Auto-labeling with trained models
- Re-labeling existing datasets

**Access**: http://127.0.0.1:5000

**Architecture**:
```
annotation_app.py (Flask GUI, Port 5000)
    ↓ communicates via HTTP
bilt_service.py (BILT operations, Port 5001)
```

The annotation app uses a client-server architecture:
- `annotation_app.py` provides the web interface
- `bilt_service.py` handles all BILT model operations (training, prediction, auto-labeling)
- `bilt_client.py` connects the two components

### 2. Detection App (Port 5003)

**Purpose**: Run real-time object detection with trained models.

**Windows Launch**:
```bash
detection\Start_BILT_Detection.bat
```

**Raspberry Pi Launch**:
```bash
cd detection
source ../venv/bin/activate
python bilt_service.py &  # Start service in background
python detection_app.py
```

**Features**:
- Real-time detection from camera feed
- Adjustable confidence/IoU thresholds
- Object counting mode
- Chain detection (multi-step detection sequences)
- Image capture during detection
- Detection statistics

**Access**: http://127.0.0.1:5003

**Architecture**:
```
detection_app.py (Flask GUI, Port 5003)
    ↓ communicates via HTTP
bilt_service.py (BILT operations, Port 5002)
```

### Using the .bat Files (Windows)

The `.bat` files provide a menu-driven interface to manage services:

```
[1] Start Complete System (Client + GUI)
[2] Start Client Service Only
[3] Start GUI Only
[4] Stop Client Service
[5] Stop GUI Server
[6] Stop All Services
[7] Toggle Python Window (Show/Hide)
[8] Open Browser
[9] View System Status
[0] Exit (Auto-stops all services)
```

**Recommended**: Use option `[1]` to start both components together.

---

## BILT Framework

BILT (Because I Like Twice) is a lightweight object detection framework built on PyTorch's SSDLite320 MobileNetV3 architecture, optimized for CPU inference.

### Why BILT?

- **CPU-Optimized**: Designed from the ground up for efficient CPU training and inference
- **Lightweight**: Uses MobileNetV3 backbone for speed
- **Simple API**: Similar to YOLOv8 interface but fully open-source
- **No GPU Required**: Trains effectively on CPU, ideal for edge devices
- **Platform Agnostic**: Runs on Windows, Linux, Raspberry Pi

### BILT Architecture

```python
# Model structure
SSDLite320_MobileNetV3
├── Backbone: MobileNetV3-Large (feature extraction)
├── SSD Head: Multi-scale detection
└── Output: Bounding boxes + class predictions
```

### Detection Format

BILT uses YOLO-style normalized coordinates:

```
Format: class_id x_center y_center width height
Example: 0 0.5 0.5 0.3 0.4
```

All coordinates are normalized (0-1) relative to image dimensions.

### Performance Characteristics

**Raspberry Pi 4 (4GB)**:
- Training: ~2-5 minutes per epoch (small datasets)
- Inference: ~5-10 FPS at 640x640
- Memory: ~1.5GB during training

**Windows PC (8GB RAM)**:
- Training: ~30-60 seconds per epoch
- Inference: ~15-30 FPS at 640x640
- Memory: ~2GB during training

---

## Workflow

### Complete Training Workflow

#### 1. Prepare Your Dataset

**Option A: Camera Capture**
1. Open Annotation App
2. Select camera and connect
3. Click "Capture Image" to save images to train folder
4. Repeat until you have 50-100 images minimum

**Option B: Import Images**
1. Create project: `projects/your_project_name/`
2. Add images to `train/images/` folder
3. Create `classes.txt` with one class name per line

#### 2. Annotate Images

1. Load project in Annotation App
2. Define classes in "Classes" tab
3. Select image from gallery
4. Draw bounding boxes by click-drag
5. Assign class to each box
6. Press Save (or Ctrl+S)
7. Repeat for all images

**Tip**: Aim for at least 50-100 annotated images per class for good results.

#### 3. Train Model

**In Annotation App > Training Configuration**:

```
Model: scratch (for new model)
Epochs: 50-100 (start small, increase if needed)
Batch Size: 4-8 (lower on Raspberry Pi)
Image Size: 640x640
Learning Rate: 0.0005 (BILT default)
```

Click "Start Training" - monitor console output for progress.

**Training Output**:
```
projects/your_project/training_run_TIMESTAMP/
├── weights/
│   └── best.pth (your trained model)
├── results.png (training metrics)
└── training_log.txt
```

#### 4. Use Auto-labeling (Optional)

Once you have a trained model:

1. Add more unlabeled images to `train/images/`
2. Go to "Auto Training" or "Relabel" section
3. Select your `best.pth` model
4. Set confidence threshold (0.25-0.50)
5. Click "Start Relabeling" for labeled images or "Start Auto Training"
6. Review and correct auto-generated labels
7. Re-train with expanded dataset

### Detection Workflow

#### 1. Deploy Model

1. Copy `best.pth` to `models/` folder
2. Rename to descriptive name (e.g., `cpu_detector.pth`)

#### 2. Run Detection

1. Open Detection App
2. Select camera
3. Load your model (`cpu_detector.pth`)
4. Adjust settings:
   - Confidence: 0.6-0.8 (higher = fewer false positives)
   - IoU: 0.1-0.4 (lower = less overlap filtering)
5. Click "Start Detection"

#### 3. Counting Mode

Enable "Counter Mode" to count unique objects:
- Each class gets a counter
- Counter increments when object first appears
- Reset counters between batches

#### 4. Chain Detection

For multi-step quality control:

**Example: CPU Assembly Verification**
```
Step 1: Detect CPU package
Step 2: Detect opened package  
Step 3: Detect CPU installed
```

**Setup**:
1. Click "Add Step"
2. Name step (e.g., "CPU Package")
3. Select required classes and counts
4. Repeat for all steps
5. Set timeout per step (seconds)
6. Enable "Auto-advance on timeout" for automatic flow

**Operation**:
- System progresses through steps sequentially
- Alert if object from future step detected (skip detection)
- Pause between cycles for part changeover
- Save/load chain configurations

---

## Project Structure

```
project_root/
├── annotation/                      # Annotation application
│   ├── Start_BILT_Annotation.bat  # Windows launcher
│   ├── annotation_app.py           # Flask web interface
│   ├── bilt_service.py             # BILT operations service
│   ├── bilt_client.py              # Service communication
│   └── templates/                  # HTML templates
│
├── detection/                       # Detection application
│   ├── Start_BILT_Detection.bat   # Windows launcher
│   ├── detection_app.py            # Flask web interface
│   ├── bilt_service.py             # BILT operations service
│   ├── bilt_client.py              # Service communication
│   ├── bilt_managers.py            # Detection managers
│   ├── config.py                   # Application config
│   └── templates/                  # HTML templates
│
├── bilt/                            # BILT framework
│   ├── __init__.py
│   ├── model.py                    # Main BILT interface
│   ├── core.py                     # Model architecture
│   ├── trainer.py                  # Training engine
│   ├── inferencer.py               # Inference engine
│   ├── dataset.py                  # Dataset handling
│   ├── evaluator.py                # Model evaluation
│   ├── utils.py                    # Utility functions
│   └── config.py                   # BILT configuration
│
├── projects/                        # Your datasets
│   └── project_name/
│       ├── train/
│       │   ├── images/             # Training images
│       │   └── labels/             # YOLO format labels
│       ├── val/
│       │   ├── images/             # Validation images
│       │   └── labels/             # Validation labels
│       ├── classes.txt             # Class names
│       └── data.yaml               # Dataset config
│
├── models/                          # Trained models (.pth)
├── venv/                           # Virtual environment
└── requirements.txt                # Dependencies
```

### Data Format

**classes.txt**:
```
person
car
bicycle
```

**Label file (train/labels/image1.txt)**:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2
```

**data.yaml** (auto-generated):
```yaml
train: /absolute/path/to/train/images
val: /absolute/path/to/val/images
nc: 3
names: ['person', 'car', 'bicycle']
```

---

## Tips & Best Practices

### Dataset Creation
- **Quality over quantity**: 50 well-annotated images > 200 poor annotations
- **Variety**: Include different angles, lighting, backgrounds
- **Tight boxes**: Draw boxes close to object boundaries
- **Consistent labeling**: Use same criteria across all images
- **Validation split**: Keep 20% of data for validation

### Training
- **Start small**: Begin with 30-50 epochs, increase if underfitting
- **Monitor loss**: Should decrease steadily; if plateaus, may need more data
- **Batch size**: Use 2-4 on Raspberry Pi, 8-16 on desktop
- **Save often**: BILT saves best model automatically
- **Iterative improvement**: Train → test → add data → retrain

### Detection
- **Confidence tuning**: Start at 0.6, adjust based on false positives/negatives
- **Lighting matters**: Match detection lighting to training conditions
- **Camera positioning**: Keep consistent with training images
- **Resolution**: Higher resolution = better accuracy but slower FPS

### Performance Optimization
- **Raspberry Pi**: Use 320x320 or 480x480 image size
- **Windows**: Can handle 640x640 or higher
- **Reduce batch size** if running out of memory
- **Close other applications** during training

---

## Troubleshooting

### Common Issues

**"No cameras found"**
- Check camera is connected and not in use by other applications
- Windows: Try different camera indices (0, 1, 2)
- Raspberry Pi: Ensure camera is enabled in `raspi-config`

**Training very slow**
- Reduce batch size to 2-4
- Reduce image size to 480x480 or 320x320
- Reduce epochs for initial testing
- Ensure no other heavy processes running

**Poor detection accuracy**
- Add more training images (aim for 100+ per class)
- Improve annotation quality
- Match detection conditions to training data
- Adjust confidence threshold
- Train for more epochs

**Out of memory errors**
- Reduce batch size to 2
- Reduce image size to 320x320
- Close other applications
- Use smaller dataset for testing

**Service won't start**
- Check ports 5000-5003 are not in use
- Verify virtual environment is activated
- Check Python version (3.9-3.11 required)
- Review error messages in console

---

## License

This project uses BILT framework which is licensed under AGPL-3.0. The annotation and detection applications are provided as-is for educational and research purposes.

## Support

For issues, questions, or contributions, refer to the project repository or documentation.

---

**Happy Detecting! 🎯**
